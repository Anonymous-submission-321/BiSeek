#!/usr/bin/env python3
"""Uniform cost accounting for the controlled timing subset.

Every method in the subset batch -- six baselines under their own pinned
interpreters, and ours -- is measured by this one module, so the numbers differ
only because the methods differ. It is inert unless ``BISEEK_COST_METER=1``:
importing it from the shared protocol modules costs nothing in ordinary runs.

What it separates, and how:

* **decode** -- time inside decord (``VideoReader`` open/seek/``get_batch``/
  ``__getitem__`` and ``NDArray.asnumpy``). Every adapter and our own
  ``VideoSampler`` reach frames through decord, so wrapping it once covers all
  of them. Reactive cost was ~88% ingest on StreamingBench, and a method that
  decodes a whole window up front must not look slower at *modelling* for it.
* **model** -- wall time minus decode. Includes preprocessing, which is part of
  what each method chooses to do with a frame.
* **large-model calls** -- outermost calls into a ``PreTrainedModel`` with at
  least ``LARGE_PARAMS`` parameters. One ``generate`` counts once, however many
  tokens it decodes; a bare forward (a per-frame trigger, an 8B router) counts
  once per forward. Smaller models (2B gates, embedding detectors) are counted
  separately so the gate's own calls are visible, not hidden.
* **persistent state** -- ``memory_allocated`` minus the resident weights,
  sampled at the entry of every large call, i.e. before its activations exist.
  That is what the method carries between frames: KV cache, token banks, frame
  buffers on the GPU. Process RSS is sampled alongside for CPU-side memory.

Stdlib at import time, like the protocol modules that import it; torch,
transformers and decord are imported only once the meter is switched on.
"""

from __future__ import annotations

import atexit
import json
import os
import resource
import time
from pathlib import Path
from typing import Any

LARGE_PARAMS = 4e9
# Adapters that chdir into a baseline's repo (Dispider) would otherwise write a
# relative BISEEK_COST_OUT into the wrong tree; pin it to the launch directory.
if os.environ.get("BISEEK_COST_OUT") and not os.path.isabs(os.environ["BISEEK_COST_OUT"]):
    os.environ["BISEEK_COST_OUT"] = os.path.abspath(os.environ["BISEEK_COST_OUT"])
ENABLED = os.environ.get("BISEEK_COST_METER", "") == "1"


def apply_subset(questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep only the uids listed in ``BISEEK_SUBSET`` (a JSON list), if set."""
    path = os.environ.get("BISEEK_SUBSET", "")
    if not path:
        return questions
    keep = set(json.loads(Path(path).read_text()))
    out = [q for q in questions if f"{q['video_id']}::{q['qid']}" in keep]
    print(f"[cost-meter] subset {Path(path).name}: {len(out)}/{len(questions)} "
          f"questions kept", flush=True)
    return out


def _rss_bytes() -> int:
    try:
        with open("/proc/self/statm") as f:
            return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except Exception:
        return 0


class _Meter:
    def __init__(self) -> None:
        self.installed = False
        self.torch = None
        self.weights_bytes: int | None = None
        self.run_started: float | None = None
        self.mark: float | None = None       # end of the previous question
        self._in_decode = 0
        self._in_model = 0
        self._numel: dict[int, int] = {}
        self.run = self._zero()
        self.q = self._zero()
        self.questions = 0

    @staticmethod
    def _zero() -> dict[str, float]:
        return {"decode_seconds": 0.0, "frames_decoded": 0, "large_calls": 0,
                "small_calls": 0, "state_bytes_max": 0, "rss_bytes_max": 0}

    # -- lifecycle -------------------------------------------------------
    def _start(self) -> None:
        """First sight of work: models are loaded, nothing else is resident."""
        if self.run_started is not None:
            return
        self.run_started = self.mark = time.perf_counter()
        if self.torch is not None and self.torch.cuda.is_available():
            self.torch.cuda.synchronize()
            self.weights_bytes = int(self.torch.cuda.memory_allocated())

    def begin_question(self) -> None:
        if not self.installed:
            return
        self._start()
        if self.torch is not None and self.torch.cuda.is_available():
            self.torch.cuda.reset_peak_memory_stats()

    def end_question(self) -> dict[str, Any]:
        """Cost since the previous question ended; includes the decode that a
        sampler did before ``begin_question`` (adapters open it first)."""
        if not self.installed:
            return {}
        self._start()
        torch = self.torch
        if torch is not None and torch.cuda.is_available():
            torch.cuda.synchronize()
        now = time.perf_counter()
        wall = now - (self.mark or now)
        self.mark = now
        out = dict(self.q)
        out["wall_seconds"] = wall
        out["model_seconds"] = max(0.0, wall - out["decode_seconds"])
        if torch is not None and torch.cuda.is_available() and self.weights_bytes is not None:
            out["state_bytes_end"] = int(torch.cuda.memory_allocated()) - self.weights_bytes
            out["peak_bytes_over_weights"] = (int(torch.cuda.max_memory_allocated())
                                              - self.weights_bytes)
            out["weights_bytes"] = self.weights_bytes
        for k in ("decode_seconds", "frames_decoded", "large_calls", "small_calls"):
            self.run[k] += self.q[k]
        for k in ("state_bytes_max", "rss_bytes_max"):
            self.run[k] = max(self.run[k], self.q[k])
        self.questions += 1
        self.q = self._zero()
        self._flush()
        return {"cost": out}

    def _flush(self) -> None:
        path = os.environ.get("BISEEK_COST_OUT", "")
        if not path or self.run_started is None:
            return
        # Work not yet closed by end_question -- all of it, for engines that
        # never call it (our reactive engine runs per video, not per question).
        run = dict(self.run)
        for k in ("decode_seconds", "frames_decoded", "large_calls", "small_calls"):
            run[k] += self.q[k]
        for k in ("state_bytes_max", "rss_bytes_max"):
            run[k] = max(run[k], self.q[k])
        run["wall_seconds"] = time.perf_counter() - self.run_started
        run["model_seconds"] = max(0.0, run["wall_seconds"] - run["decode_seconds"])
        run.update({"questions": self.questions, "weights_bytes": self.weights_bytes,
                    "gpu": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                    "large_params_threshold": LARGE_PARAMS})
        tmp = Path(path + ".tmp")
        tmp.write_text(json.dumps(run, indent=2) + "\n")
        tmp.replace(path)

    # -- sampling --------------------------------------------------------
    def _sample_state(self) -> None:
        torch = self.torch
        if torch is not None and torch.cuda.is_available() and self.weights_bytes is not None:
            state = int(torch.cuda.memory_allocated()) - self.weights_bytes
            self.q["state_bytes_max"] = max(self.q["state_bytes_max"], state)
        self.q["rss_bytes_max"] = max(self.q["rss_bytes_max"], _rss_bytes())

    def _is_large(self, module: Any) -> bool:
        key = id(module)
        if key not in self._numel:
            try:
                self._numel[key] = sum(p.numel() for p in module.parameters())
            except Exception:
                self._numel[key] = 0
        return self._numel[key] >= LARGE_PARAMS

    # -- installation ----------------------------------------------------
    def install(self) -> None:
        if self.installed:
            return
        self.installed = True
        try:
            import torch
            self.torch = torch
        except Exception:
            self.torch = None
        self._wrap_decord()
        self._wrap_transformers()
        atexit.register(self._flush)
        print("[cost-meter] on: decode / large-call / state accounting "
              f"-> {os.environ.get('BISEEK_COST_OUT', '(per-record only)')}", flush=True)

    def _timed_decode(self, fn, count):
        meter = self

        def wrapper(*args, **kwargs):
            if meter._in_decode:
                return fn(*args, **kwargs)
            if fn.__name__ == "__init__":
                meter._start()
            meter._in_decode += 1
            t0 = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                meter._in_decode -= 1
                meter.q["decode_seconds"] += time.perf_counter() - t0
                if count is not None:
                    try:
                        meter.q["frames_decoded"] += int(count(*args, **kwargs))
                    except Exception:
                        pass

        wrapper.__name__ = fn.__name__
        return wrapper

    def _wrap_decord(self) -> None:
        try:
            import decord
        except Exception:
            return
        VR = decord.VideoReader

        def n_getitem(self_, idx, *a, **k):
            if isinstance(idx, slice):
                return len(range(*idx.indices(len(self_))))
            return 1

        specs = {"__init__": None, "get_batch": lambda self_, idx, *a, **k: len(idx),
                 "__getitem__": n_getitem, "next": lambda *a, **k: 1,
                 "seek": None, "seek_accurate": None, "skip_frames": None}
        for name, count in specs.items():
            fn = getattr(VR, name, None)
            if fn is not None:
                setattr(VR, name, self._timed_decode(fn, count))
        try:
            from decord.ndarray import NDArray
            NDArray.asnumpy = self._timed_decode(NDArray.asnumpy, None)
        except Exception:
            pass

    def _wrap_transformers(self) -> None:
        try:
            from transformers import PreTrainedModel
        except Exception:
            return
        meter = self
        base_call = PreTrainedModel.__call__
        base_generate = getattr(PreTrainedModel, "generate", None)

        def enter(module) -> bool:
            if meter._in_model:
                return False
            large = meter._is_large(module)
            if large:
                meter._sample_state()
            meter.q["large_calls" if large else "small_calls"] += 1
            return True

        def call(self_, *args, **kwargs):
            enter(self_)
            meter._in_model += 1
            try:
                return base_call(self_, *args, **kwargs)
            finally:
                meter._in_model -= 1

        PreTrainedModel.__call__ = call

        def wrap_generate(base):
            def generate(self_, *args, **kwargs):
                enter(self_)
                meter._in_model += 1
                try:
                    return base(self_, *args, **kwargs)
                finally:
                    meter._in_model -= 1
            generate.__doc__ = base.__doc__
            return generate

        # Since transformers 4.50 `generate` lives on GenerationMixin, which the
        # *ForConditionalGeneration classes mix in beside PreTrainedModel -- so
        # patching PreTrainedModel alone missed it, and every decoded token was
        # counted as a large call (one smoke run: 252 "calls" in 42 steps).
        try:
            from transformers.generation.utils import GenerationMixin
            GenerationMixin.generate = wrap_generate(GenerationMixin.generate)
        except Exception:
            pass
        if "generate" in vars(PreTrainedModel) and base_generate is not None:
            PreTrainedModel.generate = wrap_generate(base_generate)


METER = _Meter()
if ENABLED:
    METER.install()
