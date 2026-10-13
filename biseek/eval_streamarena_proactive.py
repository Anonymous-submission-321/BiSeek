#!/usr/bin/env python3
"""BiSeek proactive triggering on StreamArena `Pro`, in the benchmark's own protocol.

Why this benchmark. StreamingBench PO anchors its answer a median of 6 s after
the question, so a policy that waits a fixed delay and never looks at the video
scores 52.4% there -- statistically tied with the full system. OVO Forward is
Yes/No, so a blind fixed-interval schedule beats the gate outright. StreamArena
Pro closes both holes: the waiting horizon has a median of 56 s and a p90 of
266 s, the answer is open-ended and LLM-judged, and `judge.py` reads only
`e_proactive_msgs[0]` -- the FIRST utterance -- so repeated firing cannot help.
Measured on the annotations alone, under the official acceptance window, a
blind fixed delay reaches 5.2% (2.9-4.8% held out) and blind periodic 1.8-3.1%.

Protocol, matched to `streamarena/judge.py`:

* the acceptance window is ``start_vid_ts - ref_sec`` in [-0.5, +2.0] s, taken
  verbatim from TIMING_LOWER_SEC / TIMING_UPPER_SEC; it is asymmetric because
  speaking early is a spoiler;
* one utterance per question -- the controller's state profile is absorbing, so
  the run stops at the first trigger, which is also the only one judged;
* records are written in the runner's schema so the repo's judge can score them
  unchanged.

The monitoring cap is a declared protocol parameter and is deliberately
independent of ``ref_sec``: using the reference time to decide how long to watch
would leak the answer. At 600 s it covers 98.1% of Pro questions; the remainder
are recorded as misses rather than silently dropped.
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from biseek.verifier_margin_defaults import default_verifier_accept_margin
from biseek.cost_meter import METER, apply_subset
from biseek.eval_ovo_streaming import (
    AnswerModel,
    SemanticMemoryEncoder,
    VideoSampler,
    distributed_context,
)
from biseek.proactive_cascade import build_proactive_controller

DATA_ROOT = Path("data/StreamArena")
TIMING_LOWER_SEC = -0.5   # streamarena/judge.py
TIMING_UPPER_SEC = 2.0

UTTERANCE_PROMPT = (
    "You are a live assistant watching a video with the user. Earlier the user "
    "asked you to speak up when something specific happened, and you have just "
    "decided that it is happening now.\n\n"
    "The user's request: {question}\n\n"
    "Say the one short sentence you would say out loud at this instant. State "
    "what you now see that answers the request. Do not narrate your reasoning, "
    "do not mention frames or timestamps, and do not ask a question."
)


def _rss_gb() -> float:
    try:
        with open("/proc/self/status") as handle:
            for line in handle:
                if line.startswith("RssAnon:"):
                    return int(line.split()[1]) / 1048576.0
    except OSError:
        pass
    return 0.0


_LIBC = None


def release_memory() -> None:
    """Free per-question state and hand the pages back to the OS.

    One question watches up to 600 s of video, which churns gigabytes through
    large short-lived allocations: decoded frames, window tensors, and an
    incremental KV session that is never reset inside a question
    (`incremental_kv_reset_every_n_frames = 0`). Python frees them, but glibc
    keeps the arenas, so resident memory only ever grows: two ranks of this
    script reached 270 GB and 265 GB of anonymous memory after 61 questions,
    filled all 7 GB of swap, drove load average past 50, and an earlier full
    run was OOM-killed outright. `malloc_trim` is what actually returns the
    freed arenas; `gc.collect()` alone did not, because the pages were already
    free from Python's point of view.
    """
    global _LIBC
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    try:
        if _LIBC is None:
            _LIBC = ctypes.CDLL("libc.so.6")
        _LIBC.malloc_trim(0)
    except Exception:
        pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/streamarena_pro.json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--monitor-cap-seconds",
        type=float,
        default=600.0,
        help=(
            "how long to keep watching after the question. Fixed in advance and "
            "never derived from ref_sec, which would leak the answer. 600 s "
            "covers 98.1% of Pro questions."
        ),
    )
    parser.add_argument("--poll-interval-seconds", type=float, default=0.5)
    parser.add_argument("--sample-fps", type=float, default=1.0)
    parser.add_argument("--recent-window-frames", type=int, default=3)
    parser.add_argument("--proposal-match-window-frames", type=int, default=2)
    parser.add_argument("--proposal-surge-threshold", type=float, default=0.03)
    parser.add_argument("--proposal-count", type=int, default=4)
    parser.add_argument("--proposal-match-encoding", default="vit_feature_joint_decoder")
    # The proposal compiler prompt. Every StreamArena Pro run so far used
    # target_anchored_v2; the engine also implements target_anchor_plus3_v7
    # and target_anchor_plusn_v8, neither of which has ever been run on this
    # benchmark. It was hard-coded in the defaults below rather than exposed,
    # so the ablation could not be expressed without editing the file.
    # Default None, not the v2 string: the merge below is `cfg.setdefault`, so
    # the config FILE wins over anything this parser puts in `defaults`. A flag
    # with a real default is therefore silently ignored whenever the config
    # already carries the key -- which is exactly what happened: two runs were
    # launched with target_anchor_plus3_v7 and both executed v2, twenty GPU-hours
    # producing a duplicate of the arm they were meant to ablate against.
    # `inference_config.json` recorded `target_anchored_v2` throughout and was
    # not read. None means "not passed", and an explicit value overrides.
    parser.add_argument(
        "--proposal-prompt-style",
        choices=("target_anchored_v2", "target_anchor_plus3_v7",
                 "target_anchor_plusn_v8"),
        default=None)
    parser.add_argument(
        "--schedule-mode", choices=("gate", "periodic", "dense"), default="gate"
    )
    parser.add_argument("--schedule-period-seconds", type=float, default=8.0)
    parser.add_argument(
        "--gate-warmup-seconds",
        type=float,
        default=0.0,
        help=(
            "suppress triggering for this long after the question. The surge is "
            "a delta against a running history; with two polls on record that "
            "history is not a baseline yet, and the first spurious surge is "
            "taken. Speaking 0.5 s early already fails the official window, so "
            "an unstable early estimate is the most expensive kind of error."
        ),
    )
    parser.add_argument(
        "--video-split",
        choices=("all", "dev", "test"),
        default="all",
        help="dev is a fixed 20%% of videos by id hash; thresholds are chosen there and reported on test",
    )
    parser.add_argument(
        "--proposal-manifest-path",
        default="",
        help=(
            "replay compiled proposals from this manifest instead of compiling "
            "them. This is how the one-variable proposal ablations are run: the "
            "shuffled variant hands each question another question's proposals, "
            "so anything the gate still gets right was not coming from the "
            "proposals."
        ),
    )
    parser.add_argument(
        "--proposal-manifest-texts-only",
        action="store_true",
        help="replay the texts but re-embed them, isolating the compiler from the encoder",
    )
    # store_true would give a real default (False) and lose to the config
    # file under cfg.setdefault, which is exactly how --proposal-prompt-style
    # was silently ignored for twenty GPU-hours. None means "not passed".
    parser.add_argument("--gate-only", action="store_const", const=True,
                        default=None,
                        help=("stop at the 2B gate and never call the 8B "
                              "verifier: what the cheap stage is worth alone"))
    parser.add_argument("--max-questions", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    return parser.parse_args()


def load_pro_questions() -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in (DATA_ROOT / "question.en.jsonl").read_text().splitlines()
        if line.strip()
    ]
    out = []
    for r in rows:
        if r.get("qtype") != "Pro":
            continue
        if r.get("ref_sec") is None:
            continue  # judge scores these as missing_answer_time; nothing to run
        path = DATA_ROOT / "videos" / r["video_id"] / f"{r['video_id']}.mp4"
        if not path.is_file():
            continue
        out.append({**r, "video_path": str(path)})
    return apply_subset(out)


def main() -> None:
    args = parse_args()
    cfg = json.loads((REPO_ROOT / args.config).read_text())
    cfg.update(
        {
            "benchmark": "StreamArena",
            "track": "pro",
            "decision_mode": "proposal_match",
            "sample_fps": float(args.sample_fps),
            "recent_window_frames": int(args.recent_window_frames),
            "proposal_match_window_frames": int(args.proposal_match_window_frames),
            "proposal_surge_threshold": float(args.proposal_surge_threshold),
            "proposal_count": int(args.proposal_count),
            "proposal_match_encoding": str(args.proposal_match_encoding),
            "proposal_schedule_mode": str(args.schedule_mode),
            "proposal_schedule_period_seconds": float(args.schedule_period_seconds),
            "proposal_manifest_path": str(args.proposal_manifest_path),
            "proposal_manifest_texts_only": bool(args.proposal_manifest_texts_only),
            "token_reduction_enabled": False,
            "monitor_cap_seconds": float(args.monitor_cap_seconds),
            "poll_interval_seconds": float(args.poll_interval_seconds),
            "gate_warmup_seconds": float(args.gate_warmup_seconds),
            "video_split": str(args.video_split),
        }
    )

    # The shipped config omits keys the shared model/controller constructors
    # require; the other evals fill them in their own main, so mirror that here
    # rather than editing a config other benchmarks also read.
    defaults = {
        "memory_chunk_frames": 4,
        "min_sample_frames": 1,
        "max_sample_frames": 4096,
        "frame_factor": 1,
        "dtype": "bfloat16",
        "attn_implementation": "flash_attention_2",
        "max_new_tokens": 128,
        "incremental_kv_reset_every_n_frames": 0,
        "verifier_accept_margin": default_verifier_accept_margin(
            cfg.get("answer_model_family"), cfg.get("answer_model_path")
        ),
        "proposal_prompt_style": "target_anchored_v2",
        "proposal_generation_window_frames": 5,
        "proposal_context_retrieval_enabled": False,
        "proposal_context_retrieval_top_k": 1,
        "proposal_context_retrieval_chunk_frames": 4,
        "proposal_context_retrieval_max_vectors": 128,
        "proposal_context_retrieval_cache_size": 4096,
        "proposal_delta_lookback_seconds": 1.0,
        "proposal_require_above_history_mean": True,
        "proposal_duplicate_similarity_threshold": 0.8,
        "proposal_duplicate_cooldown_seconds": 5.0,
        "proposal_initial_verify": False,
        "proposal_audit_only": False,
        "proposal_manifest_texts_only": False,
        "proposal_max_new_tokens": 256,
        "proposal_window_cache_size": 64,
        "proposal_frame_cache_size": 256,
        "proposal_vit_cache_size": 8,
        "proposal_candidate_episode_seconds": 2.0,
        "proposal_candidate_retry_interval_seconds": 1.0,
        "proposal_batch_new_frames_per_poll": False,
        "token_reduction_layer": 12,
        "token_reduction_selector": "semantic",
        "token_reduction_stc_interval": 4,
        "token_reduction_budget": 0.1,
        "token_reduction_parallel": True,
        "embedding_backend": "qwen3vl",
        "verifier_prompt_mode": "task_specific",
    }
    for key, value in defaults.items():
        cfg.setdefault(key, value)
    # Explicit CLI overrides, applied AFTER the defaults merge so they beat the
    # config file rather than losing to it. Only keys the user actually passed.
    if args.proposal_prompt_style is not None:
        cfg["proposal_prompt_style"] = str(args.proposal_prompt_style)
    if args.gate_only is not None:
        cfg["gate_only"] = bool(args.gate_only)
        # proposal_match reads proposal_audit_only, not gate_only: its decide()
        # overrides the parent's, and the parent is the only reader of
        # gate_only. Without this the two "gate-only" StreamArena runs were
        # the full cascade under another name (fps1: all 774 triggers and
        # answers identical to the cascade; fps2: ~27 verifier calls/question).
        cfg["proposal_audit_only"] = bool(args.gate_only)

    rank, world_size, _, device = distributed_context()
    if world_size == 1 and int(args.world_size) > 1:
        rank, world_size = int(args.rank), int(args.world_size)
    args.rank, args.world_size = rank, world_size

    questions = load_pro_questions()
    # Shard by video so each rank opens every video reader exactly once and the
    # 2-7 questions on that video reuse it.
    by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for q in questions:
        by_video[q["video_id"]].append(q)
    videos = sorted(by_video)
    if args.video_split != "all":
        import hashlib

        want_dev = args.video_split == "dev"
        videos = [
            v for v in videos
            if (int(hashlib.md5(v.encode()).hexdigest(), 16) % 5 == 0) == want_dev
        ]
    # Bin-pack by estimated cost instead of round-robin. A video carries 1-7
    # questions whose horizons span 10 s to 3470 s, so `i % world_size` left the
    # slowest rank with 6.2x the polls of the fastest and wall time is set by the
    # slowest -- 42% of the run was one rank finishing alone. Cost is the polls a
    # video will need if nothing triggers early, which is the only bound known
    # before the run; longest-first into the least-loaded rank.
    def video_cost(v: str) -> float:
        return sum(
            min(float(q.get("horizon_sec") or 0.0), float(args.monitor_cap_seconds))
            / float(args.poll_interval_seconds)
            for q in by_video[v]
        )

    loads = [0.0] * args.world_size
    buckets: list[list[str]] = [[] for _ in range(args.world_size)]
    for v in sorted(videos, key=video_cost, reverse=True):
        k = min(range(args.world_size), key=lambda i: loads[i])
        buckets[k].append(v)
        loads[k] += video_cost(v)
    mine = sorted(buckets[args.rank])
    work = [q for v in mine for q in sorted(by_video[v], key=lambda r: r["ask_sec"])]
    if args.rank == 0:
        spread = max(loads) / max(1e-9, min(loads))
        print(f"bin-packed {len(videos)} videos over {args.world_size} ranks; "
              f"estimated poll spread max/min = {spread:.2f}x", flush=True)
    if args.max_questions:
        work = work[: args.max_questions]

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rank_path = out_dir / f"rank_{args.rank:02d}.jsonl"

    # Resume. Records are appended, but the work list was rebuilt from scratch on
    # every launch, so a run killed at question 61 of 774 redid all 61. Sharding
    # is deterministic (bin-packed from the annotations, which do not change), so
    # a uid already written by THIS rank is done; anything else is still owed.
    done_uids: set[str] = set()
    if rank_path.exists():
        for line in rank_path.read_text(errors="ignore").splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
                # Only a successful record retires a question; see the same
                # guard in every adapter. A transient failure otherwise drops
                # that question for good and the run still reports 774/774.
                if not str(r.get("error", "")):
                    done_uids.add(r["uid"])
            except Exception:
                continue
    if done_uids:
        before = len(work)
        work = [q for q in work
                if f"{q['video_id']}::{q['qid']}" not in done_uids]
        print(f"[rank {args.rank}] resuming: {before - len(work)} already done, "
              f"{len(work)} to go", flush=True)
    if args.rank == 0:
        (out_dir / "inference_config.json").write_text(json.dumps(cfg, indent=2) + "\n")

    # AnswerModel indexes the device (device.index), so it needs a real
    # torch.device, not the string "cuda".
    answer_model = AnswerModel(cfg, device)
    encoder = SemanticMemoryEncoder(cfg, "full_embedding", device)
    controller = build_proactive_controller(
        encoder=encoder,
        answer_model=answer_model,
        cfg=cfg,
        task="STREAMARENA_PRO",
        decision_mode="proposal_match",
    )

    # Compiled proposals are written as they are produced, not at the end: the
    # other runners emit `proposal_manifest.json` on completion, and a run that
    # is killed at question 600 of 774 -- which has happened repeatedly here --
    # would leave nothing to build an ablation from.
    # NOT rank_XX_manifest.jsonl: every tool here globs `rank_*.jsonl` for
    # records, and that pattern matches it. The watchdog duly counted each
    # manifest line as a record with a missing "error" field, i.e. as a failure,
    # and the results table would have averaged over them too.
    manifest_path = out_dir / f"manifest_rank_{args.rank:02d}.jsonl"
    seen_manifest_keys: set[str] = set()
    if manifest_path.exists():
        for line in manifest_path.read_text(errors="ignore").splitlines():
            if line.strip():
                try:
                    seen_manifest_keys.add(json.loads(line)["proposal_manifest_key"])
                except Exception:
                    continue

    sampler: VideoSampler | None = None
    sampler_video: str | None = None
    started = time.time()
    with rank_path.open("a", buffering=1) as handle, \
            manifest_path.open("a", buffering=1) as manifest_handle:
        for index, q in enumerate(work, 1):
            if sampler_video != q["video_id"]:
                sampler = VideoSampler(q["video_path"], cfg, causal_fixed_fps=True)
                sampler_video = q["video_id"]
            ask = float(q["ask_sec"])
            ref = float(q["ref_sec"])
            record: dict[str, Any] = {
                "uid": f"{q['video_id']}::{q['qid']}",
                "video_id": q["video_id"],
                "qid": q["qid"],
                "qtype_code": "e",
                "qtype_label": "E Proactive Interaction",
                "qtype_public": "Pro",
                "ask_time_sec": ask,
                "ask_time_str": q.get("ask_time", ""),
                "question": q["question"],
                "reference_answer": q["answer"],
                "answer_time_sec": ref,
                "horizon_sec": q.get("horizon_sec"),
                "horizon_layer": q.get("horizon_layer"),
                "model": "BiSeek-proposal-gate",
                "mode": f"{args.schedule_mode}-fps{args.sample_fps:g}",
                "error": "",
            }
            METER.begin_question()
            try:
                state = controller.register(
                    query_id=f"{q['video_id']}:{q['qid']}",
                    target_question=q["question"],
                    registration_timestamp=ask,
                )
                controller.prime(state=state, sampler=sampler, timestamp=ask)
                compiled = state.registration_metrics.get("proposal_compilation")
                if compiled and compiled.get("proposal_manifest_key") not in seen_manifest_keys:
                    seen_manifest_keys.add(compiled["proposal_manifest_key"])
                    manifest_handle.write(json.dumps({
                        **compiled,
                        "benchmark": "StreamArena",
                        "task": "STREAMARENA_PRO",
                        "question_id": f"{q['video_id']}::{q['qid']}",
                        "target_question": q["question"],
                    }, ensure_ascii=False) + "\n")
                stop = ask + float(args.monitor_cap_seconds)
                step = float(args.poll_interval_seconds)
                trigger: float | None = None
                frames: list[Any] = []
                polls = 0
                timestamp = ask + step
                while timestamp <= stop:
                    decision = controller.decide(
                        state=state, sampler=sampler, timestamp=float(timestamp)
                    )
                    polls += 1
                    warm = (timestamp - ask) >= float(args.gate_warmup_seconds)
                    if decision.positive and warm:
                        trigger = float(timestamp)
                        frames = list(decision.frames)
                        break
                    timestamp += step
                stats = controller.state_statistics(state)

                if trigger is None:
                    record.update(
                        {
                            "model_answer": "",
                            "e_missed": True,
                            "e_proactive_msgs": [],
                        }
                    )
                else:
                    utterance, _ = answer_model.answer(
                        video=frames,
                        prompt=UTTERANCE_PROMPT.format(question=q["question"]),
                        fps=float(cfg["sample_fps"]),
                        max_frames=len(frames),
                        max_new_tokens_override=int(args.max_new_tokens),
                    )
                    record.update(
                        {
                            "model_answer": str(utterance).strip(),
                            "e_missed": False,
                            # start_vid_ts is what the judge compares to
                            # answer_time_sec; it is the instant we began
                            # speaking, i.e. the poll that triggered.
                            "e_proactive_msgs": [
                                {"start_vid_ts": trigger, "vid_ts": trigger}
                            ],
                        }
                    )
                delta = None if trigger is None else trigger - ref
                record.update(
                    {
                        "trigger_vid_ts": trigger,
                        "timing_delta_sec": delta,
                        "timing_in_window": (
                            delta is not None
                            and TIMING_LOWER_SEC <= delta <= TIMING_UPPER_SEC
                        ),
                        "polls": polls,
                        "state_statistics": stats,
                    }
                )
            except Exception as error:  # a failed row must not kill the shard
                record.update(
                    {
                        "model_answer": "",
                        "e_missed": True,
                        "e_proactive_msgs": [],
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
            record.update(METER.end_question())
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            # The question is over: its controller state, KV sessions and decoded
            # frames are dead weight, and holding them is what filled memory.
            state = None
            decision = None
            frames = []
            release_memory()
            if index % 5 == 0 or index == len(work):
                hit = "-" if record.get("timing_delta_sec") is None else (
                    f"{record['timing_delta_sec']:+.1f}s"
                )
                print(
                    f"[rank {args.rank}] {index}/{len(work)} {q['video_id']}:{q['qid']} "
                    f"delta={hit} in_window={record.get('timing_in_window')} "
                    f"rss={_rss_gb():.1f}G elapsed={time.time() - started:.0f}s",
                    flush=True,
                )
    (out_dir / f"rank_{args.rank:02d}.done").write_text("done\n")
    print(f"[rank {args.rank}] wrote {rank_path}", flush=True)


if __name__ == "__main__":
    main()
