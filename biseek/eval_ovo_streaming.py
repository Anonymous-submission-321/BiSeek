#!/usr/bin/env python3
"""Evaluate full-history and online vector-memory streaming on OVO-Bench.

The official pre-chunked OVO videos already stop at the timestamp at which a
    question is asked.  This runner therefore never sees future frames.  It
    supports three protocols:

* full_history: Qwen3-VL sees the complete causal prefix sampled at 1 fps.
* full_embedding: q_final retrieves fixed-duration chunks via official d_final.
* decoder12: q_final retrieves chunks via the last-position readout from
  decoder block 12 and genuinely exits before blocks 13--28.

Full-embedding and decoder-12 retain 2048-D vectors in an online memory.
Positive ``max_memory_vectors`` values impose a FIFO bound; ``-1`` retains all
causally observed vectors without eviction.  The 2B encoder and 8B answer model
remain resident together.  Each source is replayed chronologically: newly
completed chunks are encoded once, and a query is retrieved and answered
immediately when its timestamp is reached.  No full-source index is
materialized on disk.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from PIL import Image


BACKWARD_TASKS = ["EPM", "ASI", "HLD"]
REALTIME_TASKS = ["OCR", "ACR", "ATR", "STU", "FPD", "OJR"]
CATEGORIES = {"backward": BACKWARD_TASKS, "realtime": REALTIME_TASKS}

BR_PROMPT_TEMPLATE = """\
Question: {}
Options:
{}

Respond only with the letter corresponding to your chosen option (e.g., A, B, C).
Do not include any additional text or explanation in your response.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--answer-model-path", default=None)
    parser.add_argument("--answer-model-family", default=None,
                        choices=["qwen3vl", "qwen2_5vl", "qwen2vl"])
    parser.add_argument(
        "--mode",
        choices=["full_history", "full_embedding", "decoder12"],
        required=True,
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--tasks", default="")
    parser.add_argument("--max-samples", type=int, default=None, help="Smoke-test limit before sharding.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--sample-fps", type=float, default=None)
    parser.add_argument("--memory-chunk-frames", type=int, default=None)
    parser.add_argument("--max-memory-vectors", type=int, default=None)
    parser.add_argument("--recent-pool-frames", type=int, default=None)
    parser.add_argument("--history-pool-frames", type=int, default=None)
    parser.add_argument("--history-temporal-nms", type=parse_bool, default=None)
    parser.add_argument("--answer-frame-budget", type=int, default=None)
    parser.add_argument(
        "--reactive-policy",
        choices=[
            "retrieve_first",
            "recent_first",
        ],
        default=None,
        help=(
            "Use question-vector retrieve-first, or first gate on the latest "
            "four frames and fall back to question-vector retrieval."
        ),
    )
    parser.add_argument(
        "--reactive-retrieval-query",
        choices=["question"],
        default=None,
        help="Fallback retrieval query used only by the recent-first branch.",
    )
    parser.add_argument("--recent-first-frames", type=int, default=None)
    parser.add_argument("--reactive-proposal-count", type=int, default=None)
    parser.add_argument(
        "--reactive-proposal-aggregation",
        choices=["max", "mean", "logsumexp", "question_anchor_rrf"],
        default=None,
    )
    parser.add_argument("--reactive-rrf-k", type=float, default=None)
    parser.add_argument("--reactive-question-weight", type=float, default=None)
    parser.add_argument("--reactive-proposal-weight", type=float, default=None)
    parser.add_argument("--token-reduction", type=parse_bool, default=None)
    parser.add_argument("--token-reduction-layer", type=int, default=None)
    parser.add_argument("--token-reduction-budget", type=float, default=None)
    parser.add_argument("--token-reduction-parallel", type=parse_bool, default=None)
    parser.add_argument(
        "--token-reduction-selector",
        choices=["semantic", "stc_dynamic", "ctr", "random", "uniform", "norm"],
        default=None,
        help="Which token score drives selection at a fixed keep budget. "
             "'semantic' is the query-conditioned default; the rest are ablation arms.",
    )
    parser.add_argument("--full-history-visual-token-budget", type=int, default=None)
    return parser.parse_args()


def load_json(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(
        f"Expected a boolean value (true/false, 1/0, yes/no), got {value!r}"
    )


def validate_memory_capacity(value: int) -> int:
    capacity = int(value)
    if capacity != -1 and capacity <= 0:
        raise ValueError(
            "max_memory_vectors must be -1 (unbounded) or a positive integer"
        )
    return capacity


def memory_capacity_policy(value: int) -> str:
    return "unbounded" if int(value) == -1 else "fifo_bounded"


def memory_execution_protocol(value: int) -> str:
    return f"chronological_online_{memory_capacity_policy(value)}_vector_memory"


def nominal_retained_frame_capacity(
    max_memory_vectors: int, memory_chunk_frames: int
) -> int | None:
    capacity = int(max_memory_vectors)
    return None if capacity == -1 else capacity * int(memory_chunk_frames)


def dtype_from_name(value: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
    }[value.lower()]


def distributed_context() -> tuple[int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not torch.cuda.is_available():
        raise RuntimeError("OVO evaluation requires CUDA GPUs")
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank, torch.device("cuda", local_rank)


def build_prompt(item: dict[str, Any]) -> str:
    options = "; ".join(f"{chr(65 + index)}. {option}" for index, option in enumerate(item["options"])) + ";"
    return BR_PROMPT_TEMPLATE.format(item["question"], options)


def _plain_options(item: dict[str, Any]) -> str:
    values = item.get("choices", item.get("options", []))
    return "\n".join(
        f"{chr(65 + index)}. {str(value).strip()}"
        for index, value in enumerate(values)
    )


def abstention_option_index(item: dict[str, Any]) -> int | None:
    """Find an explicit answerability/null option without using the task id."""

    labels = {
        "unable to answer",
        "cannot answer",
        "can't answer",
        "insufficient information",
        "not enough information",
        "none of the above can be determined",
    }
    values = item.get("choices", item.get("options", []))
    for index, value in enumerate(values):
        normalized = re.sub(r"\s+", " ", str(value)).strip(" .").casefold()
        if normalized in labels:
            return index
    return None


def build_option_conditioned_queries(item: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Build deterministic evidence hypotheses for every concrete option."""

    abstention_index = abstention_option_index(item)
    if abstention_index is None:
        raise ValueError("Option-conditioned abstention retrieval needs a null option")
    values = item.get("choices", item.get("options", []))
    queries: list[str] = []
    labels: list[str] = []
    for index, option in enumerate(values):
        if index == abstention_index:
            continue
        letter = chr(65 + index)
        queries.append(
            f"Question: {item['question']} Candidate answer: {str(option).strip()}. "
            "Find direct visible evidence that specifically supports this answer."
        )
        labels.append(f"option_{letter}")
    return queries, labels


def build_abstention_aware_answer_prompt(item: dict[str, Any]) -> str:
    abstention_index = abstention_option_index(item)
    if abstention_index is None:
        return build_prompt(item)
    abstention_letter = chr(65 + abstention_index)
    return f"""\
Question: {item['question']}
Options:
{_plain_options(item)}

The supplied images are retrieval candidates, not guaranteed evidence.
Choose a concrete option only when the images directly show evidence that
specifically supports it. Do not infer an answer from plausibility, scene
context, or mere semantic similarity. If none of the concrete options is
directly supported, choose {abstention_letter}, the explicit inability-to-answer option.

Respond only with the chosen option letter. Do not explain."""


def build_recent_first_gate_prompt(item: dict[str, Any]) -> str:
    """Strict, benchmark-agnostic recent-evidence gate.

    This branch is intentionally simple.  It asks the frozen 8B model to
    either answer from the recent images or request historical retrieval.  A
    malformed response is treated conservatively as RETRIEVE by the parser.
    """

    return f"""\
Look only at the recent images above.

Question: {item['question']}
Options:
{_plain_options(item)}

If the images contain enough visual evidence, output exactly `ANSWER: X`,
where X is the option letter. Otherwise output exactly `RETRIEVE`.
Do not explain."""


def build_reactive_proposal_prompt(
    item: dict[str, Any], proposal_count: int, *, query_context: str | None = None
) -> str:
    """Ask for concrete *past-evidence* images, not a temporal state machine."""

    return f"""\
Question context: {query_context or item['question']}
Options:
{_plain_options(item)}

Predict {proposal_count} short descriptions of earlier video frames that
would visibly contain the evidence needed to answer this question.
Describe only what the image should show. Do not answer the question.
Output one description per line and nothing else."""


def build_recent_first_plan_prompt(
    item: dict[str, Any], proposal_count: int, *, query_context: str | None = None
) -> str:
    """Fuse recent answerability routing and missing-evidence planning."""

    return f"""\
Look only at the recent images above.

Question context: {query_context or item['question']}
Options:
{_plain_options(item)}

If the recent images contain enough visual evidence, output exactly `ANSWER: X`,
where X is the option letter.

Otherwise output exactly `RETRIEVE: ["cue", ...]` with one to
{proposal_count} short descriptions of missing earlier visual evidence. Each cue
must describe one concrete frame or short event that would help answer the exact
question. Preserve required people, objects, numbers, attributes, relations, and
temporal order. Do not guess the answer, invent an option-specific fact, predict a
story, or copy incidental details from the recent images. Use JSON strings and do
not explain."""


def parse_recent_first_gate(
    response: str, *, num_options: int | None = None
) -> tuple[str, str | None]:
    """Return (ANSWER|RETRIEVE, option_letter).

    Only an explicit ``ANSWER: X`` is accepted.  This prevents an incidental
    option letter in free text from bypassing retrieval.
    """

    value = str(response or "").strip()
    matched = re.fullmatch(r"ANSWER\s*:\s*([A-E])", value, flags=re.IGNORECASE)
    if matched:
        letter = matched.group(1).upper()
        if num_options is None or ord(letter) - ord("A") < int(num_options):
            return "ANSWER", letter
    return "RETRIEVE", None


def parse_recent_first_plan(
    response: str,
    *,
    num_options: int | None = None,
    proposal_limit: int = 3,
) -> tuple[str, str | None, list[str]]:
    """Parse strict ANSWER or fused RETRIEVE-with-evidence output.

    Malformed retrieval plans conservatively become question-only retrieval.
    """

    decision, answer = parse_recent_first_gate(
        response, num_options=num_options
    )
    if decision == "ANSWER":
        return decision, answer, []
    value = str(response or "").strip()
    matched = re.fullmatch(
        r"RETRIEVE\s*:\s*(\[.*\])", value, flags=re.IGNORECASE | re.DOTALL
    )
    if matched is None:
        return "RETRIEVE", None, []
    try:
        parsed = json.loads(matched.group(1))
    except json.JSONDecodeError:
        return "RETRIEVE", None, []
    if not isinstance(parsed, list):
        return "RETRIEVE", None, []
    proposals = [str(item).strip() for item in parsed if str(item).strip()]
    proposals = list(dict.fromkeys(proposals))[: int(proposal_limit)]
    return "RETRIEVE", None, proposals


def parse_visual_proposals(response: str, limit: int) -> list[str]:
    value = str(response or "").strip()
    if not value:
        return []
    proposals: list[str] = []
    # Accept a JSON list when the model happens to emit one, but keep the
    # requested prompt itself line based and simple.
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, list):
        proposals = [str(item).strip() for item in parsed if str(item).strip()]
    else:
        for line in value.splitlines():
            line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip()
            line = line.strip('"').strip()
            if line:
                proposals.append(line)
    return list(dict.fromkeys(proposals))[: int(limit)]


def category_for_task(task: str) -> str:
    if task in BACKWARD_TASKS:
        return "backward"
    if task in REALTIME_TASKS:
        return "realtime"
    raise ValueError(f"Unsupported task in the Table-1 protocol: {task}")


def sample_frame_indices(
    total_frames: int,
    raw_fps: float,
    target_fps: float,
    min_frames: int,
    max_frames: int,
    factor: int,
) -> list[int]:
    """Mirror Qwen's duration-derived, factor-aligned linspace sampling."""
    if total_frames <= 0:
        raise ValueError("Video contains no frames")
    raw_fps = max(float(raw_fps), 1e-6)
    lower = min(max(1, int(min_frames)), total_frames)
    upper = min(max(lower, int(max_frames)), total_frames)
    wanted = total_frames / raw_fps * float(target_fps)
    wanted = min(max(wanted, lower), upper)
    nframes = int(math.floor(wanted / max(1, factor)) * max(1, factor))
    if nframes < lower:
        nframes = min(upper, int(math.ceil(lower / max(1, factor)) * max(1, factor)))
    nframes = min(max(1, nframes), total_frames)
    return np.linspace(0, total_frames - 1, nframes).round().astype(np.int64).tolist()


def fixed_fps_frame_indices(total_frames: int, raw_fps: float, target_fps: float) -> list[int]:
    """Causal fixed-rate sampling whose past indices never depend on future duration."""
    if total_frames <= 0:
        raise ValueError("Video contains no frames")
    raw_fps = max(float(raw_fps), 1e-6)
    target_fps = max(float(target_fps), 1e-6)
    duration = (total_frames - 1) / raw_fps
    timestamps = np.arange(0.0, duration + 1e-7, 1.0 / target_fps)
    indices = np.rint(timestamps * raw_fps).astype(np.int64)
    indices = np.clip(indices, 0, total_frames - 1)
    return np.unique(indices).tolist()


class VideoSampler:
    def __init__(self, path: str, cfg: dict[str, Any], *, causal_fixed_fps: bool = False):
        from decord import VideoReader, cpu

        self.path = path
        # Decode threads. StreamArena's reactive tracks spend 98% of their
        # time in stream ingest (642 s of a 656 s question), and decode is
        # thread-limited: measured on an HR video at the real 8-frame chunk
        # size, 2 threads give 35 fps, 8 give 72, 16 give 90. The default stays
        # 2 so no other benchmark's timing changes; StreamArena's jobs opt in
        # through the environment. Frame output is identical either way -- this
        # buys wall time, not different results.
        self.reader = VideoReader(
            path, ctx=cpu(0),
            num_threads=int(os.environ.get("BISEEK_DECORD_THREADS", "2")))
        self.total_frames = len(self.reader)
        self.raw_fps = float(self.reader.get_avg_fps())
        if causal_fixed_fps:
            self.indices = fixed_fps_frame_indices(
                self.total_frames, self.raw_fps, float(cfg["sample_fps"])
            )
        else:
            self.indices = sample_frame_indices(
                self.total_frames,
                self.raw_fps,
                float(cfg["sample_fps"]),
                int(cfg["min_sample_frames"]),
                int(cfg["max_sample_frames"]),
                int(cfg["frame_factor"]),
            )
        self.sample_timestamps = [index / max(self.raw_fps, 1e-6) for index in self.indices]

    def positions_available_at(self, timestamp: float) -> list[int]:
        limit = float(timestamp) + 1e-6
        return [position for position, value in enumerate(self.sample_timestamps) if value <= limit]

    def decode_positions(self, positions: list[int]) -> list[Image.Image]:
        if not positions:
            return []
        raw_indices = [self.indices[position] for position in positions]
        try:
            batch = self.reader.get_batch(raw_indices).asnumpy()
            return [Image.fromarray(frame) for frame in batch]
        except Exception as error:
            # A few official OVO clips expose a nominal final frame that sits
            # in a damaged/slow final GOP. Fall back to OpenCV and, only when
            # necessary, walk backwards to the nearest decodable frame.
            print(
                f"Decord batch decode failed for {self.path}; using OpenCV fallback: {error}",
                file=sys.stderr,
                flush=True,
            )
            return self._decode_positions_opencv(raw_indices)

    def _decode_positions_opencv(self, raw_indices: list[int]) -> list[Image.Image]:
        import cv2

        capture = cv2.VideoCapture(self.path)
        if not capture.isOpened():
            raise RuntimeError(f"OpenCV could not open fallback video: {self.path}")
        decoded: list[Image.Image] = []
        try:
            for requested in raw_indices:
                frame = None
                used = requested
                for backoff in (0, 1, 2, 4, 8, 16, 32, 64, 128):
                    used = max(0, requested - backoff)
                    capture.set(cv2.CAP_PROP_POS_FRAMES, used)
                    ok, value = capture.read()
                    if ok and value is not None:
                        frame = cv2.cvtColor(value, cv2.COLOR_BGR2RGB)
                        break
                if frame is None:
                    raise RuntimeError(
                        f"Neither Decord nor OpenCV could decode frame {requested} from {self.path}"
                    )
                if used != requested:
                    print(
                        f"Substituted undecodable frame {requested} with frame {used} in {self.path}",
                        file=sys.stderr,
                        flush=True,
                    )
                decoded.append(Image.fromarray(frame))
        finally:
            capture.release()
        return decoded

    def timestamps(self, positions: list[int]) -> list[float]:
        return [self.sample_timestamps[position] for position in positions]


def align_in_memory_frame_timestamps(
    frame_timestamps: list[float] | None,
    processed_frame_count: int,
) -> tuple[list[float] | None, int]:
    """Align timestamps with qwen-vl-utils' even-frame padding.

    ``fetch_video`` pads an in-memory odd-length frame list by repeating its
    final image so that the temporal frame count is divisible by
    ``FRAME_FACTOR=2``.  The repeated image must carry the same original
    timestamp; inventing a future timestamp would be causally incorrect.
    """
    if frame_timestamps is None:
        return None, 0
    aligned = [float(value) for value in frame_timestamps]
    if len(aligned) == processed_frame_count:
        return aligned, 0
    if not aligned:
        raise ValueError(
            f"Got no timestamps for {processed_frame_count} processed in-memory frames"
        )
    padding = int(processed_frame_count) - len(aligned)
    # The installed qwen-vl-utils uses FRAME_FACTOR=2, hence an in-memory
    # frame list can only acquire one repeated final frame.  Fail loudly for
    # any other transformation instead of silently corrupting MRoPE timing.
    if padding != 1 or len(aligned) % 2 != 1 or processed_frame_count % 2 != 0:
        raise ValueError(
            f"Got {len(aligned)} timestamps for {processed_frame_count} frames; "
            "only one repeated final frame from even-frame padding is supported"
        )
    aligned.append(aligned[-1])
    return aligned, padding


def answer_model_class(name: str):
    """Resolve the answer model's HF class by family name.

    The backbone was hard-wired to Qwen3-VL, which makes the published
    comparison against Qwen2-VL-based streaming methods confounded: our win
    could be the architecture or could be the newer backbone. Selecting the
    class from config lets the same controller run on either family so the
    two can be separated. The bounded-KV session only touches
    `model.model.get_rope_index` and `model.model.visual.spatial_merge_size`,
    which both Qwen2-VL and Qwen3-VL expose.
    """

    from transformers import (
        Qwen2VLForConditionalGeneration,
        Qwen2_5_VLForConditionalGeneration,
        Qwen3VLForConditionalGeneration,
    )

    families = {
        "qwen3vl": Qwen3VLForConditionalGeneration,
        "qwen2_5vl": Qwen2_5_VLForConditionalGeneration,
        "qwen2vl": Qwen2VLForConditionalGeneration,
    }
    if name not in families:
        raise ValueError(
            f"answer_model_family must be one of {sorted(families)}, got {name!r}"
        )
    return families[name]


class AnswerModel:
    def __init__(self, cfg: dict[str, Any], device: torch.device):
        from transformers import AutoProcessor

        dtype = dtype_from_name(cfg["dtype"])
        self.device = device
        self.answer_model_family = str(cfg.get("answer_model_family", "qwen3vl"))
        self.model = answer_model_class(self.answer_model_family).from_pretrained(
            cfg["answer_model_path"],
            torch_dtype=dtype,
            attn_implementation=cfg["attn_implementation"],
            device_map={"": device.index},
        )
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(cfg["answer_model_path"])
        self.max_new_tokens = int(cfg["max_new_tokens"])
        self.token_reduction_enabled = bool(cfg.get("token_reduction_enabled", False))
        self.token_reduction_layer = int(cfg.get("token_reduction_layer", 12))
        # "semantic" is BiSeek's query-conditioned 2B decoder-token score;
        # "stc_dynamic" swaps in STC's temporal-redundancy criterion for a matched-budget ablation.
        self.token_reduction_selector = str(cfg.get("token_reduction_selector", "semantic"))
        self.token_reduction_stc_interval = int(cfg.get("token_reduction_stc_interval", 4))
        self.token_reduction_budget = float(cfg.get("token_reduction_budget", 0.1))
        self.token_reduction_parallel = bool(
            cfg.get("token_reduction_parallel", True)
        )
        if self.token_reduction_enabled and self.token_reduction_selector == "ctr":
            # CTR ranks dynamic tokens by ViT attention, which the forward pass does
            # not expose, so a pre-hook on the last vision block recomputes it.  The
            # hook is installed only for this arm so the other selectors pay nothing.
            from transformers.models.qwen3_vl.ctr_compression import (
                install_ctr_attention_hook,
            )
            visual = self.model.model.visual
            install_ctr_attention_hook(visual)
            visual.ctr_capture_enabled = True
        self._token_reduction_executor: ThreadPoolExecutor | None = None
        self._token_reduction_stream: torch.cuda.Stream | None = None
        if self.token_reduction_enabled and self.token_reduction_parallel:
            self._token_reduction_executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix=f"biseek-token-reduction-cuda{device.index}",
            )
            self._token_reduction_stream = torch.cuda.Stream(device=device)

    def _token_reduction_trace_config(self) -> dict[str, Any]:
        return {
            "enable_trace_vit": True,
            "trace_apply_to_images": True,
            "trace_total_budget": self.token_reduction_budget,
            "trace_budget_scope": "batch_total",
            "trace_component_temporal_spine": True,
            "trace_spine_tokens_per_frame": 1,
            "trace_component_semantic_detail": True,
            "trace_detail_budget_mode": "query_adaptive",
            "trace_semantic_video_source": "embedding_2b_decoder_tokens",
            "trace_semantic_score_mode": "decoder_token",
            "trace_decoder_token_layer": self.token_reduction_layer,
            "trace_frequency_residual_mode": False,
            "trace_record_mask": False,
            "trace_visualize": False,
            "trace_token_selector": getattr(self, "token_reduction_selector", "semantic"),
            "trace_stc_cache_interval": int(getattr(self, "token_reduction_stc_interval", 4)),
            "trace_ctr_similarity_threshold": float(
                getattr(self, "token_reduction_ctr_tau", 0.9)
            ),
            "trace_ctr_k": int(getattr(self, "token_reduction_ctr_k", 7)),
            "trace_ctr_beta": float(getattr(self, "token_reduction_ctr_beta", 0.6)),
        }

    def _prepare(
        self,
        video: str | list[Image.Image],
        prompt: str,
        fps: float,
        max_frames: int | None,
        visual_token_budget: int | None,
        frame_timestamps: list[float] | None,
        frame_context_labels: list[str] | None,
    ) -> tuple[Any, dict[str, Any]]:
        from qwen_vl_utils import process_vision_info

        if isinstance(video, list):
            if not video:
                messages = [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": prompt}],
                    }
                ]
                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = self.processor(
                    text=[text], padding=True, return_tensors="pt"
                )
                return inputs.to(self.device), {
                    "answer_input_modality": "text_only_no_causal_oracle_evidence",
                    "answer_requested_frames": 0,
                    "answer_processed_frames": 0,
                    "answer_frame_padding": 0,
                    "answer_frame_context_labels": False,
                }
            # Match SimpleStream's Qwen3-VL protocol: sparse evidence frames
            # are separate images in one conversation.  Treating them as a
            # video would invoke temporal patching and create an artificial
            # equally-spaced timeline between non-contiguous history frames.
            if (
                frame_context_labels is not None
                and len(frame_context_labels) != len(video)
            ):
                raise ValueError(
                    "frame_context_labels must contain exactly one label per "
                    f"independent image: labels={len(frame_context_labels)}, "
                    f"images={len(video)}"
                )
            content: list[dict[str, Any]] = []
            for index, frame in enumerate(video):
                if frame_context_labels is not None:
                    content.append(
                        {
                            "type": "text",
                            "text": str(frame_context_labels[index]),
                        }
                    )
                content.append({"type": "image", "image": frame})
            content.append({"type": "text", "text": prompt})
            messages = [{"role": "user", "content": content}]
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            images, videos = process_vision_info(messages, image_patch_size=16)
            if videos is not None:
                raise RuntimeError("Independent-image answer input unexpectedly produced video data")
            inputs = self.processor(
                text=[text],
                images=images,
                videos=None,
                padding=True,
                return_tensors="pt",
            )
            processed_images = int(inputs["image_grid_thw"].shape[0])
            if processed_images != len(video):
                raise ValueError(
                    f"Requested {len(video)} independent images but processor produced "
                    f"{processed_images} image grids"
                )
            return inputs.to(self.device), {
                "answer_input_modality": "independent_images",
                "answer_requested_frames": len(video),
                "answer_processed_frames": processed_images,
                "answer_frame_padding": 0,
                "answer_frame_context_labels": bool(frame_context_labels),
            }

        video_item: dict[str, Any] = {"type": "video", "video": video, "fps": float(fps)}
        if max_frames is not None:
            video_item["max_frames"] = int(max_frames)
        if visual_token_budget is not None:
            # Qwen3-VL: 16-pixel patch x 2-way spatial merger = 32x32
            # input pixels per language-model visual token.
            video_item["total_pixels"] = int(visual_token_budget) * 32 * 32
        messages = [{"role": "user", "content": [video_item, {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos, video_kwargs = process_vision_info(
            messages,
            image_patch_size=16,
            return_video_kwargs=True,
            return_video_metadata=True,
        )
        metadata = None
        requested_frame_count = len(video) if isinstance(video, list) else None
        processed_frame_counts: list[int] = []
        timestamp_padding = 0
        if videos is not None:
            videos, metadata = zip(*videos)
            videos, metadata = list(videos), list(metadata)
            # qwen-vl-utils assigns its generic 2-fps metadata to an in-memory
            # frame list.  These frames were sampled at cfg.sample_fps, so keep
            # the pixels but correct the timestamps seen by the processor/MRoPE.
            if isinstance(video, list):
                for item, tensor in zip(metadata, videos):
                    if isinstance(item, dict):
                        frame_count = int(tensor.shape[0])
                        processed_frame_counts.append(frame_count)
                        item["fps"] = float(fps)
                        if frame_timestamps is not None:
                            aligned_timestamps, padded = align_in_memory_frame_timestamps(
                                frame_timestamps, frame_count
                            )
                            timestamp_padding += padded
                            frame_indices = [
                                int(round(float(value) * float(fps)))
                                for value in aligned_timestamps
                            ]
                        else:
                            frame_indices = list(range(frame_count))
                        item["frames_indices"] = frame_indices
                        item["total_num_frames"] = float(max(frame_indices, default=-1) + 1)
        inputs = self.processor(
            text=[text],
            images=images,
            videos=videos,
            video_metadata=metadata,
            do_resize=False,
            padding=True,
            return_tensors="pt",
            **(video_kwargs or {}),
        )
        prepare_metrics = {"answer_frame_padding": int(timestamp_padding)}
        if requested_frame_count is not None:
            prepare_metrics["answer_requested_frames"] = int(requested_frame_count)
        if processed_frame_counts:
            prepare_metrics["answer_processed_frames"] = int(sum(processed_frame_counts))
        prepare_metrics["answer_input_modality"] = "video"
        return inputs.to(self.device), prepare_metrics

    @torch.inference_mode()
    def binary_choice(
        self,
        video: list[Image.Image],
        prompt: str,
        *,
        fps: float,
        frame_timestamps: list[float],
        frame_context_labels: list[str] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Choose ``A`` or ``B`` from the next-token logits.

        Proactive verification is classification, not free-form answering.  A
        single forward pass is both cheaper and less parser-sensitive than
        autoregressive ``generate``.  Sparse observations retain the same
        independent-image input protocol as :meth:`answer`.
        """

        if not video:
            raise ValueError("binary_choice requires at least one image")
        torch.cuda.reset_peak_memory_stats(self.device)
        started = time.perf_counter()
        inputs, prepare_metrics = self._prepare(
            video,
            prompt,
            fps,
            len(video),
            None,
            frame_timestamps,
            frame_context_labels,
        )
        preprocess_done = time.perf_counter()
        tokenizer = self.processor.tokenizer
        token_ids: dict[str, int] = {}
        for label in ("A", "B"):
            encoded = tokenizer.encode(label, add_special_tokens=False)
            if len(encoded) != 1:
                raise RuntimeError(
                    f"Expected {label!r} to be one tokenizer token, got {encoded}"
                )
            token_ids[label] = int(encoded[0])

        core_model = self.model.model
        core_model.trace_config = None
        core_model.trace_measure_image_vit = False
        outputs = self.model(
            **inputs,
            use_cache=False,
            return_dict=True,
            logits_to_keep=1,
        )
        torch.cuda.synchronize(self.device)
        forward_done = time.perf_counter()
        logits = outputs.logits[0, -1].float()
        logit_a = float(logits[token_ids["A"]].item())
        logit_b = float(logits[token_ids["B"]].item())
        response = "B" if logit_b > logit_a else "A"
        input_ids = inputs["input_ids"]
        video_token_id = int(getattr(self.model.config, "video_token_id", 151656))
        image_token_id = int(getattr(self.model.config, "image_token_id", 151655))
        visual_tokens = int(
            ((input_ids == video_token_id) | (input_ids == image_token_id))
            .sum()
            .item()
        )
        return response, {
            **prepare_metrics,
            "binary_choice_backend": "qwen3_vl_next_token_ab",
            "binary_choice_logit_a": logit_a,
            "binary_choice_logit_b": logit_b,
            "binary_choice_margin_b_minus_a": logit_b - logit_a,
            "answer_preprocess_seconds": preprocess_done - started,
            "answer_forward_seconds": forward_done - preprocess_done,
            "answer_total_seconds": forward_done - started,
            "answer_input_tokens": int(input_ids.shape[1]),
            "answer_visual_tokens": visual_tokens,
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(self.device)),
        }

    @torch.inference_mode()
    def answer(
        self,
        video: str | list[Image.Image],
        prompt: str,
        fps: float,
        max_frames: int | None,
        visual_token_budget: int | None = None,
        frame_timestamps: list[float] | None = None,
        frame_context_labels: list[str] | None = None,
        token_reduction_encoder: Any | None = None,
        token_reduction_query: torch.Tensor | None = None,
        max_new_tokens_override: int | None = None,
        token_reduction_enabled_override: bool | None = None,
    ) -> tuple[str, dict[str, Any]]:
        token_reduction_active = (
            self.token_reduction_enabled
            if token_reduction_enabled_override is None
            else bool(token_reduction_enabled_override)
        )
        torch.cuda.reset_peak_memory_stats(self.device)
        started = time.perf_counter()
        inputs, prepare_metrics = self._prepare(
            video,
            prompt,
            fps,
            max_frames,
            visual_token_budget,
            frame_timestamps,
            frame_context_labels,
        )
        preprocess_done = time.perf_counter()
        input_ids = inputs["input_ids"]
        video_token_id = int(getattr(self.model.config, "video_token_id", 151656))
        image_token_id = int(getattr(self.model.config, "image_token_id", 151655))
        visual_tokens = int(
            ((input_ids == video_token_id) | (input_ids == image_token_id)).sum().item()
        )
        token_reduction_metrics: dict[str, Any] = {
            "token_reduction_enabled": token_reduction_active,
            "token_reduction_layer": self.token_reduction_layer,
            "token_reduction_selector": self.token_reduction_selector,
            "token_reduction_budget": self.token_reduction_budget,
            "token_reduction_parallel": (
                token_reduction_active and self.token_reduction_parallel
            ),
            "token_reduction_seconds": 0.0,
            "token_reduction_future_wait_seconds": 0.0,
            "token_reduction_cuda_overlap_with_8b_vit_seconds": 0.0,
        }
        core_model = self.model.model
        core_model.trace_config = None
        core_model.trace_measure_image_vit = True
        core_model.trace_last_token_counts = None
        core_model.trace_last_parallel_timing = None
        core_model.trace_last_image_vit_events = None
        token_reduction_future: Future | None = None
        token_reduction_common_start: torch.cuda.Event | None = None
        for attribute in (
            "trace_semantic_image_token_scores",
            "trace_semantic_image_token_scores_future",
        ):
            if hasattr(core_model, attribute):
                delattr(core_model, attribute)

        if token_reduction_active:
            if not isinstance(video, list):
                raise ValueError(
                    "Streaming token reduction currently requires the selected "
                    "independent-image evidence path."
                )
            if self.token_reduction_selector == "ctr":
                pass
            elif token_reduction_encoder is None or token_reduction_query is None:
                raise ValueError(
                    "Token reduction is enabled but its 2B encoder/query vector is missing."
                )
            image_grid_thw = inputs.get("image_grid_thw")
            if image_grid_thw is None:
                raise RuntimeError("8B processor did not produce image_grid_thw.")
            merge_size = int(core_model.visual.spatial_merge_size)
            target_grids = [
                (
                    int(grid[1].item()) // merge_size,
                    int(grid[2].item()) // merge_size,
                )
                for grid in image_grid_thw
            ]
            core_model.trace_config = self._token_reduction_trace_config()
            if self.token_reduction_parallel:
                if (
                    self._token_reduction_executor is None
                    or self._token_reduction_stream is None
                ):
                    raise RuntimeError("Parallel token-reduction worker was not initialized.")
                token_reduction_common_start = torch.cuda.Event(enable_timing=True)
                token_reduction_common_start.record(torch.cuda.current_stream(self.device))
                token_reduction_future = self._token_reduction_executor.submit(
                    token_reduction_encoder.score_selected_frame_tokens,
                    video,
                    token_reduction_query,
                    layer=self.token_reduction_layer,
                    target_grid_hw=target_grids,
                    cuda_stream=self._token_reduction_stream,
                    common_start_event=token_reduction_common_start,
                )
                core_model.trace_semantic_image_token_scores_future = (
                    token_reduction_future
                )
            else:
                token_scores, scorer_metrics = (
                    token_reduction_encoder.score_selected_frame_tokens(
                        video,
                        token_reduction_query,
                        layer=self.token_reduction_layer,
                        target_grid_hw=target_grids,
                    )
                )
                token_reduction_metrics.update(scorer_metrics)
                core_model.trace_semantic_image_token_scores = token_scores

        generate_started = time.perf_counter()
        try:
            generated = self.model.generate(
                **inputs,
                max_new_tokens=(
                    self.max_new_tokens
                    if max_new_tokens_override is None
                    else int(max_new_tokens_override)
                ),
                do_sample=False,
                use_cache=True,
            )
        finally:
            core_model.trace_config = None
            core_model.trace_measure_image_vit = False
            for attribute in (
                "trace_semantic_image_token_scores",
                "trace_semantic_image_token_scores_future",
            ):
                if hasattr(core_model, attribute):
                    delattr(core_model, attribute)
        torch.cuda.synchronize(self.device)
        generate_done = time.perf_counter()
        if token_reduction_future is not None:
            _, scorer_metrics = token_reduction_future.result()
            token_reduction_metrics.update(scorer_metrics)
        scorer_start_event = token_reduction_metrics.pop(
            "_token_reduction_cuda_start_event", None
        )
        scorer_end_event = token_reduction_metrics.pop(
            "_token_reduction_cuda_end_event", None
        )
        if scorer_start_event is not None and scorer_end_event is not None:
            token_reduction_metrics["token_reduction_seconds"] = (
                float(scorer_start_event.elapsed_time(scorer_end_event)) / 1000.0
            )
        vit_events = getattr(core_model, "trace_last_image_vit_events", None)
        answer_8b_vit_seconds = 0.0
        if vit_events is not None:
            vit_start_event, vit_end_event = vit_events
            answer_8b_vit_seconds = (
                float(vit_start_event.elapsed_time(vit_end_event)) / 1000.0
            )
        parallel_timing = getattr(core_model, "trace_last_parallel_timing", None) or {}
        token_reduction_metrics.update(parallel_timing)
        parallel_join_event = token_reduction_metrics.pop(
            "_token_reduction_parallel_join_event", None
        )
        if token_reduction_common_start is not None and parallel_join_event is not None:
            token_reduction_metrics[
                "token_reduction_parallel_visual_critical_path_seconds"
            ] = (
                float(
                    token_reduction_common_start.elapsed_time(parallel_join_event)
                )
                / 1000.0
            )
        if (
            token_reduction_common_start is not None
            and scorer_start_event is not None
            and scorer_end_event is not None
            and vit_events is not None
        ):
            vit_start_event, vit_end_event = vit_events
            score_start_ms = float(
                token_reduction_common_start.elapsed_time(scorer_start_event)
            )
            score_end_ms = float(
                token_reduction_common_start.elapsed_time(scorer_end_event)
            )
            vit_start_ms = float(
                token_reduction_common_start.elapsed_time(vit_start_event)
            )
            vit_end_ms = float(
                token_reduction_common_start.elapsed_time(vit_end_event)
            )
            overlap_ms = max(
                0.0,
                min(score_end_ms, vit_end_ms) - max(score_start_ms, vit_start_ms),
            )
            token_reduction_metrics[
                "token_reduction_cuda_overlap_with_8b_vit_seconds"
            ] = overlap_ms / 1000.0
        trace_counts = getattr(core_model, "trace_last_token_counts", None)
        kept_visual_tokens = (
            int(trace_counts["kept_visual_tokens"])
            if token_reduction_active and trace_counts is not None
            else visual_tokens
        )
        if token_reduction_active and trace_counts is None:
            raise RuntimeError(
                "Token reduction was enabled, but the 8B model did not report a BiSeek image mask."
            )
        trimmed = [output[len(source) :] for source, output in zip(input_ids, generated)]
        response = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()
        metrics = {
            **prepare_metrics,
            **token_reduction_metrics,
            "answer_preprocess_seconds": preprocess_done - started,
            "answer_8b_vit_seconds": answer_8b_vit_seconds,
            "answer_generate_seconds": generate_done - generate_started,
            "answer_full_generate_seconds": generate_done - generate_started,
            "answer_generate_scope": (
                "full_model_generate_including_8b_vit_llm_prefill_and_all_output_tokens"
            ),
            "answer_total_seconds": generate_done - started,
            "answer_input_tokens": int(input_ids.shape[1]),
            "answer_input_tokens_after_reduction": (
                int(input_ids.shape[1]) - visual_tokens + kept_visual_tokens
            ),
            "answer_visual_tokens_before_reduction": visual_tokens,
            "answer_visual_tokens": kept_visual_tokens,
            "answer_visual_token_keep_ratio": kept_visual_tokens / max(1, visual_tokens),
            "answer_generated_tokens": int(trimmed[0].numel()),
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(self.device)),
        }
        return response, metrics


class _DecoderReadoutEarlyExit(RuntimeError):
    """Internal control flow used to stop the 2B decoder after a chosen block."""

    def __init__(self, hidden: torch.Tensor):
        super().__init__("decoder readout captured")
        self.hidden = hidden


class SemanticMemoryEncoder:
    def __init__(self, cfg: dict[str, Any], mode: str, device: torch.device):
        # Ablation hook: an alternative encoder answers query_vectors/encode_frame
        # and nothing else, so it can only drive the frame-mean matching path.
        backend = str(cfg.get("embedding_backend", "qwen3vl") or "qwen3vl").lower()
        if backend not in ("qwen3vl", "qwen3_vl", "default"):
            encoding = str(cfg.get("proposal_match_encoding", "joint_window"))
            # Two-tower encoders have no decoder and no KV, so frame-mean is the
            # only path open to them.  The Qwen2-VL-based embedders do have both,
            # and cache B's session builds raw Qwen prompt tokens that their
            # processors share, so incremental_kv is open to them as well.
            TWO_TOWER = {"clip", "siglip", "jina_clip"}
            allowed = {"cached_frame_mean"}
            if backend not in TWO_TOWER:
                allowed.add("incremental_kv")
                allowed.add("vit_feature_joint_decoder")   # cache A, see embedding_backends
            if encoding not in allowed:
                raise ValueError(
                    f"embedding_backend={backend!r} supports "
                    f"proposal_match_encoding in {sorted(allowed)}, got {encoding!r}. "
                    "Cache A needs encode_joint_images_cached_vit, which the "
                    "alternative backends do not implement; two-tower encoders "
                    "additionally have no decoder or KV at all."
                )
            from embedding_backends import build_embedding_backend

            self.cfg = cfg
            self.mode = mode
            self.device = device
            self.dtype = dtype_from_name(cfg["dtype"])
            self.backend_name = backend
            self._backend = build_embedding_backend(
                backend, cfg["embedding_model_path"], device, self.dtype
            )
            self.embedder = self._backend
            self.model = self._backend.model
            return
        self.backend_name = "qwen3vl"
        self._backend = None
        helper_dir = Path(cfg["embedding_helper_dir"]).resolve()
        if not (helper_dir / "qwen3_vl_embedding.py").is_file():
            raise FileNotFoundError(f"Embedding helper not found under {helper_dir}")
        if str(helper_dir) not in sys.path:
            sys.path.insert(0, str(helper_dir))
        from qwen3_vl_embedding import Qwen3VLEmbedder

        self.cfg = cfg
        self.mode = mode
        self.device = device
        self.dtype = dtype_from_name(cfg["dtype"])
        self.embedder = Qwen3VLEmbedder(
            model_name_or_path=cfg["embedding_model_path"],
            num_frames=int(cfg["memory_chunk_frames"]),
            max_frames=int(cfg["memory_chunk_frames"]),
            torch_dtype=self.dtype,
            attn_implementation=cfg["attn_implementation"],
        )
        self.model = self.embedder.model
        self.model.eval()

    @property
    def num_decoder_layers(self) -> int:
        return len(self.model.model.language_model.layers)

    @property
    def readout_layer(self) -> int:
        if self.mode == "decoder12":
            return int(self.cfg["decoder_layer"])
        return self.num_decoder_layers

    @property
    def readout_name(self) -> str:
        if self.mode == "full_embedding":
            return "official_final_last_valid_post_final_norm"
        return f"decoder_block_{self.readout_layer}_last_valid_early_exit_pre_final_norm"

    @property
    def representation_kind(self) -> str:
        return "single_vector"

    def query_vector(self, question: str) -> tuple[torch.Tensor, float]:
        vectors, elapsed = self.query_vectors([question])
        return vectors[0], elapsed

    def query_vectors(
        self,
        texts: list[str],
        *,
        instruction: str | None = None,
    ) -> tuple[list[torch.Tensor], float]:
        if self._backend is not None:
            return self._backend.query_vectors(
                texts, instruction=instruction or self.cfg["query_instruction"]
            )
        if not texts:
            raise ValueError("query_vectors requires at least one text")
        started = time.perf_counter()
        value = self.embedder.process(
            [
                {
                    "text": str(text),
                    "instruction": str(
                        instruction or self.cfg["query_instruction"]
                    ),
                }
                for text in texts
            ],
            normalize=True,
        )
        torch.cuda.synchronize(self.device)
        vectors = [
            item.detach().float().cpu().reshape(-1) for item in value
        ]
        return vectors, time.perf_counter() - started

    def _inputs_for_independent_images(
        self,
        frames: list[Image.Image],
        instruction: str,
    ) -> dict[str, torch.Tensor]:
        conversations = [
            self.embedder.format_model_input(image=frame, instruction=instruction)
            for frame in frames
        ]
        inputs = self.embedder._preprocess_inputs(conversations)
        return {
            key: value.to(self.model.device) if torch.is_tensor(value) else value
            for key, value in inputs.items()
        }

    def _inputs_for_joint_images(
        self,
        frames: list[Image.Image],
        instruction: str,
    ) -> dict[str, torch.Tensor]:
        """Put ordered frames in one multimodal sequence as independent images.

        Unlike ``_inputs_for_independent_images`` this is one batch row, so the
        language decoder jointly contextualizes all cached visual features in
        chronological order.  The image path is intentional: a sliding video
        tensor uses temporal tubelets whose pairing changes after a one-frame
        shift and therefore cannot reuse an exact per-frame ViT output.
        """

        content = [
            {
                "type": "image",
                "image": frame,
                "min_pixels": self.embedder.min_pixels,
                "max_pixels": self.embedder.max_pixels,
            }
            for frame in frames
        ]
        conversation = [
            {"role": "system", "content": [{"type": "text", "text": instruction}]},
            {"role": "user", "content": content},
        ]
        inputs = self.embedder._preprocess_inputs([conversation])
        return {
            key: value.to(self.model.device) if torch.is_tensor(value) else value
            for key, value in inputs.items()
        }

    @torch.inference_mode()
    def encode_joint_images_cached_vit(
        self,
        frames: list[Image.Image],
        frame_keys: list[tuple[str, int]],
        feature_cache: Any,
        cache_size: int,
        _allow_replay: bool = True,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Reuse exact independent-image ViT outputs, rerun the joint decoder.

        Cached entries include the final visual tokens and every DeepStack
        feature required by Qwen3-VL.  They stay on the model device and are
        detached BF16 tensors.  Only the decoder is recomputed for the ordered
        multi-image sequence at every sliding-window step.
        """

        if getattr(self, "_backend", None) is not None:
            return self._backend.encode_joint_images_cached_vit(
                frames, frame_keys, feature_cache, cache_size)
        if not frames or len(frames) != len(frame_keys):
            raise ValueError("cached-ViT encoding needs one key per non-empty frame")
        if int(cache_size) <= 0:
            raise ValueError("cached-ViT cache_size must be positive")

        # Every frame in one video has the same resolution, so the tokenised
        # sequence (input_ids / attention_mask / grids) depends only on the
        # window size and that shared grid -- it is identical at every poll.
        # The image processor, by contrast, re-ran on all frames each poll and
        # threw most of its output away, because the ViT cache is only
        # consulted afterwards.  Probe the cache first, replay the cached token
        # sequence, and send only the frames that actually miss through the
        # processor -- a one-step slide misses exactly the newest frame.
        probe = [feature_cache.get(key) for key in frame_keys]
        hit_grid = next(
            (
                tuple(int(value) for value in item["grid"].tolist())
                for item in probe
                if item is not None
            ),
            None,
        )
        tokens = getattr(self, "_joint_token_cache", None)
        if tokens is None:
            tokens = self._joint_token_cache = {}
        replay = (
            tokens.get((len(frames), hit_grid))
            if hit_grid is not None and _allow_replay
            else None
        )
        # Replay assumes every cached frame shares hit_grid; otherwise a cached
        # frame of another grid would be treated as uncached with no pixels to
        # recompute it, so take the full path instead.
        if replay is not None and any(
            item is not None
            and tuple(int(value) for value in item["grid"].tolist()) != hit_grid
            for item in probe
        ):
            replay = None

        inputs: Any = None
        image_grid_thw: Any = None
        pixel_chunks: Any = []
        if replay is not None:
            missing = [index for index, item in enumerate(probe) if item is None]
            chunks: list[Any] = [None] * len(frames)
            usable = True
            if missing:
                partial = self._inputs_for_joint_images(
                    [frames[index] for index in missing],
                    str(self.cfg["document_instruction"]),
                )
                partial_grid = partial.get("image_grid_thw")
                partial_pixels = partial.get("pixel_values")
                if partial_grid is None or partial_pixels is None:
                    raise RuntimeError("Joint-image processor omitted image tensors")
                # A frame whose resolution differs from the cached ones would
                # need a different token sequence, so fall back to the full path.
                usable = all(
                    tuple(int(value) for value in row.tolist()) == hit_grid
                    for row in partial_grid
                )
                if usable:
                    split = [int(value) for value in partial_grid.prod(-1).tolist()]
                    for slot, chunk in zip(
                        missing, torch.split(partial_pixels, split, dim=0)
                    ):
                        chunks[slot] = chunk
            if usable:
                inputs = replay
                image_grid_thw = replay["image_grid_thw"]
                pixel_chunks = chunks

        if inputs is None:
            inputs = self._inputs_for_joint_images(
                frames, str(self.cfg["document_instruction"])
            )
            image_grid_thw = inputs.get("image_grid_thw")
            pixel_values = inputs.get("pixel_values")
            if image_grid_thw is None or pixel_values is None:
                raise RuntimeError("Joint-image processor omitted image tensors")
            if int(image_grid_thw.shape[0]) != len(frames):
                raise RuntimeError("Joint-image processor did not emit one grid per frame")
            raw_split_sizes = [int(value) for value in image_grid_thw.prod(-1).tolist()]
            pixel_chunks = torch.split(pixel_values, raw_split_sizes, dim=0)
            grids = {tuple(int(v) for v in g.tolist()) for g in image_grid_thw}
            if len(grids) == 1:
                tokens[(len(frames), next(iter(grids)))] = {
                    "input_ids": inputs["input_ids"],
                    "attention_mask": inputs["attention_mask"],
                    "mm_token_type_ids": inputs.get("mm_token_type_ids"),
                    "image_grid_thw": image_grid_thw,
                }
                if len(tokens) > 8:
                    tokens.pop(next(iter(tokens)))

        entries: list[dict[str, Any]] = []
        hits = 0
        vit_calls = 0
        vit_seconds = 0.0
        for key, grid, pixels in zip(frame_keys, image_grid_thw, pixel_chunks):
            cached = feature_cache.pop(key, None)
            grid_cpu = grid.detach().cpu()
            if cached is not None and torch.equal(cached["grid"], grid_cpu):
                feature_cache[key] = cached
                entries.append(cached)
                hits += 1
                continue
            if pixels is None:
                if _allow_replay:
                    # The replayed sequence cannot cover this frame; redo the
                    # window on the full path instead of failing the question.
                    return self.encode_joint_images_cached_vit(
                        frames, frame_keys, feature_cache, cache_size,
                        _allow_replay=False,
                    )
                raise RuntimeError("cached-token replay reached an uncached frame")
            started = time.perf_counter()
            output = self.model.model.get_image_features(
                pixels,
                grid.unsqueeze(0),
                return_dict=True,
            )
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            vit_seconds += time.perf_counter() - started
            cached = {
                "grid": grid_cpu,
                "pooled": output.pooler_output[0].detach(),
                "deepstack": tuple(value.detach() for value in output.deepstack_features),
            }
            feature_cache[key] = cached
            while len(feature_cache) > int(cache_size):
                feature_cache.popitem(last=False)
            entries.append(cached)
            vit_calls += 1

        core = self.model.model
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        inputs_embeds = core.get_input_embeddings()(input_ids)
        image_embeds = torch.cat([value["pooled"] for value in entries], dim=0).to(
            inputs_embeds.device, inputs_embeds.dtype
        )
        image_mask, _ = core.get_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
        )
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
        visual_pos_masks = image_mask[..., 0]
        deepstack_visual_embeds = [
            torch.cat([entry["deepstack"][index] for entry in entries], dim=0)
            for index in range(len(entries[0]["deepstack"]))
        ]
        position_ids = core.compute_3d_position_ids(
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=None,
            mm_token_type_ids=inputs.get("mm_token_type_ids"),
        )
        decoder_started = time.perf_counter()
        output = core.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            use_cache=False,
            return_dict=True,
        )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        decoder_seconds = time.perf_counter() - decoder_started
        vector = self.embedder._pooling_last(
            output.last_hidden_state, attention_mask
        )[0]
        cache_bytes = sum(
            entry["pooled"].numel() * entry["pooled"].element_size()
            + sum(item.numel() * item.element_size() for item in entry["deepstack"])
            for entry in feature_cache.values()
        )
        return F.normalize(vector.float(), dim=-1).cpu(), {
            "proposal_vit_feature_cache_hits_this_window": hits,
            "proposal_vit_feature_encode_calls_this_window": vit_calls,
            "proposal_vit_feature_seconds_this_window": vit_seconds,
            "proposal_joint_decoder_seconds_this_window": decoder_seconds,
            "proposal_vit_feature_cache_bytes": int(cache_bytes),
        }

    @torch.inference_mode()
    def _decoder_hidden_for_inputs(
        self,
        inputs: dict[str, torch.Tensor],
        layer: int,
    ) -> tuple[torch.Tensor, int]:
        layers = self.model.model.language_model.layers
        layer = int(layer)
        if layer == -1:
            outputs = self.embedder.forward(inputs)
            return outputs["last_hidden_state"], len(layers)
        if layer < 1 or layer > len(layers):
            raise ValueError(
                f"token_reduction_layer must be -1 or in [1, {len(layers)}], got {layer}"
            )

        def stop_after_layer(module, args, output):
            hidden = output[0] if isinstance(output, (tuple, list)) else output
            raise _DecoderReadoutEarlyExit(hidden)

        hook = layers[layer - 1].register_forward_hook(stop_after_layer)
        try:
            try:
                self.embedder.forward(inputs)
            except _DecoderReadoutEarlyExit as early_exit:
                hidden = early_exit.hidden
            else:
                raise RuntimeError(f"Decoder layer {layer} hook did not trigger")
        finally:
            hook.remove()
        return hidden, layer

    @torch.inference_mode()
    def score_selected_frame_tokens(
        self,
        frames: list[Image.Image],
        query_vector: torch.Tensor,
        *,
        layer: int,
        target_grid_hw: list[tuple[int, int]],
        cuda_stream: torch.cuda.Stream | None = None,
        common_start_event: torch.cuda.Event | None = None,
    ) -> tuple[tuple[torch.Tensor, ...], dict[str, Any]]:
        """Score temporary 2B image-position tokens for 8B BiSeek pruning.

        The returned score maps are immediately consumed by the answer model;
        decoder hidden states are never inserted into the persistent streaming
        memory or written to disk.
        """
        if not frames:
            raise ValueError("Token reduction needs at least one selected frame.")
        if len(frames) != len(target_grid_hw):
            raise ValueError(
                f"Got {len(frames)} selected frames but {len(target_grid_hw)} target grids."
            )

        started = time.perf_counter()
        torch.cuda.set_device(self.device)
        active_stream = cuda_stream or torch.cuda.current_stream(self.device)
        scorer_start_event = torch.cuda.Event(enable_timing=True)
        scorer_end_event = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(active_stream):
            if common_start_event is not None:
                active_stream.wait_event(common_start_event)
            scorer_start_event.record(active_stream)
            instruction = str(
                self.cfg.get(
                    "token_reduction_document_instruction",
                    "Represent this frame for token-level retrieval.",
                )
            )
            inputs = self._inputs_for_independent_images(frames, instruction)
            hidden, layers_executed = self._decoder_hidden_for_inputs(inputs, int(layer))
            input_ids = inputs["input_ids"]
            image_grid_thw = inputs.get("image_grid_thw")
            if image_grid_thw is None or int(image_grid_thw.shape[0]) != len(frames):
                raise RuntimeError(
                    "Embedding-2B processor did not produce one image grid per selected frame."
                )

            image_token_id = int(getattr(self.model.config, "image_token_id", 151655))
            merge_size = int(self.model.model.visual.spatial_merge_size)
            query = F.normalize(query_vector.to(hidden.device).float().reshape(-1), dim=-1)
            score_maps: list[torch.Tensor] = []
            scored_tokens = 0
            for row_index, target_grid in enumerate(target_grid_hw):
                token_hidden = hidden[row_index][input_ids[row_index] == image_token_id]
                grid = image_grid_thw[row_index]
                temporal = int(grid[0].item())
                source_h = int(grid[1].item()) // merge_size
                source_w = int(grid[2].item()) // merge_size
                expected = temporal * source_h * source_w
                if temporal != 1:
                    raise RuntimeError(
                        f"Independent image {row_index} has temporal grid {temporal}, expected 1."
                    )
                if int(token_hidden.shape[0]) != expected:
                    raise RuntimeError(
                        "Embedding decoder image token count mismatch: "
                        f"expected {expected}, got {token_hidden.shape[0]}."
                    )
                token_hidden = F.normalize(token_hidden.float(), dim=-1)
                scores = torch.matmul(token_hidden, query.view(-1, 1)).squeeze(-1)
                target_h, target_w = (int(target_grid[0]), int(target_grid[1]))
                if target_h <= 0 or target_w <= 0:
                    raise ValueError(f"Invalid 8B target grid {target_grid}.")
                scores = scores.view(1, 1, source_h, source_w)
                if (source_h, source_w) != (target_h, target_w):
                    scores = F.interpolate(
                        scores,
                        size=(target_h, target_w),
                        mode="bilinear",
                        align_corners=False,
                    )
                # Keep the temporary score maps on GPU.  The 8B default stream
                # waits on scorer_end_event before consuming them, so no
                # device-to-host/device round trip is needed.
                score_maps.append(scores.reshape(-1).detach().float())
                scored_tokens += expected
            scorer_end_event.record(active_stream)

        worker_enqueue_seconds = time.perf_counter() - started
        return tuple(score_maps), {
            "token_reduction_worker_enqueue_seconds": worker_enqueue_seconds,
            "token_reduction_decoder_layers_executed": layers_executed,
            "token_reduction_frames": len(frames),
            "token_reduction_2b_visual_tokens_scored": scored_tokens,
            "token_reduction_peak_gpu_memory_bytes": int(
                torch.cuda.max_memory_allocated(self.device)
            ),
            "token_reduction_peak_gpu_memory_scope": (
                "shared_gpu_parallel_region"
                if cuda_stream is not None
                else "shared_gpu_sequential_region"
            ),
            "token_reduction_persistent_decoder_token_bytes": 0,
            "_token_reduction_cuda_start_event": scorer_start_event,
            "_token_reduction_cuda_end_event": scorer_end_event,
        }

    def _inputs_for_frames(self, frames: list[Image.Image]):
        from qwen_vl_utils import process_vision_info

        conversation = self.embedder.format_model_input(
            video=frames,
            instruction=self.cfg["document_instruction"],
        )
        text = self.embedder.processor.apply_chat_template(
            [conversation], add_generation_prompt=True, tokenize=False
        )
        images, videos, video_kwargs = process_vision_info(
            [conversation],
            image_patch_size=16,
            return_video_kwargs=True,
            return_video_metadata=True,
        )
        metadata = None
        if videos is not None:
            videos, metadata = zip(*videos)
            videos, metadata = list(videos), list(metadata)
            for item, tensor in zip(metadata, videos):
                if isinstance(item, dict):
                    frame_count = int(tensor.shape[0])
                    item["fps"] = float(self.cfg["sample_fps"])
                    item["frames_indices"] = list(range(frame_count))
                    item["total_num_frames"] = float(frame_count)
        inputs = self.embedder.processor(
            text=text,
            images=images,
            videos=videos,
            video_metadata=metadata,
            truncation=True,
            max_length=self.embedder.max_length,
            padding=True,
            do_resize=False,
            return_tensors="pt",
            **(video_kwargs or {}),
        )
        return {key: value.to(self.model.device) if torch.is_tensor(value) else value for key, value in inputs.items()}

    @torch.inference_mode()
    def _official_final_vector(self, frames: list[Image.Image]) -> torch.Tensor:
        inputs = self._inputs_for_frames(frames)
        outputs = self.embedder.forward(inputs)
        value = self.embedder._pooling_last(outputs["last_hidden_state"], outputs["attention_mask"])[0]
        return F.normalize(value.float(), dim=-1).cpu()

    @torch.inference_mode()
    def _decoder_block_readout(self, frames: list[Image.Image], layer: int) -> torch.Tensor:
        """Pool the last valid sequence position and stop after ``layer``.

        Decoder block outputs are full ``[B, sequence_length, hidden_size]``
        tensors.  They are not already one chunk vector.  We use the same last
        valid sequence position as the official embedder, but before the final
        model RMSNorm.  Raising from the hook is intentional: for layer 12 it
        prevents blocks 13--28 from running.
        """
        inputs = self._inputs_for_frames(frames)
        layers = self.model.model.language_model.layers
        if layer < 1 or layer > len(layers):
            raise ValueError(f"Decoder readout layer must be in [1, {len(layers)}], got {layer}")

        def stop_after_layer(module, args, output):
            hidden = output[0] if isinstance(output, (tuple, list)) else output
            raise _DecoderReadoutEarlyExit(hidden)

        hook = layers[layer - 1].register_forward_hook(stop_after_layer)
        try:
            try:
                self.embedder.forward(inputs)
            except _DecoderReadoutEarlyExit as early_exit:
                hidden = early_exit.hidden
            else:
                raise RuntimeError(f"Decoder layer {layer} hook did not trigger")
        finally:
            hook.remove()
        value = self.embedder._pooling_last(hidden, inputs["attention_mask"])[0]
        return F.normalize(value.float(), dim=-1).cpu()

    def encode_chunk(self, frames: list[Image.Image]) -> torch.Tensor:
        if self.mode == "full_embedding":
            return self._official_final_vector(frames)
        return self._decoder_block_readout(frames, self.readout_layer)

    @torch.inference_mode()
    def encode_frame(self, frame: Image.Image) -> torch.Tensor:
        # Some lightweight tests and downstream adapters construct this object
        # with ``__new__`` and populate only the legacy attributes.  Treat a
        # missing backend exactly like the normal local-embedding path.
        backend = getattr(self, "_backend", None)
        if backend is not None:
            return backend.encode_frame(frame)
        """Encode exactly one independent image into one normalized vector.

        This deliberately uses the embedding model's image input path.  Calling
        ``encode_chunk([frame])`` is not equivalent: the upstream helper samples
        every in-memory video list to ``num_frames`` and would therefore repeat
        a one-frame list to the configured chunk length.
        """

        inputs = self._inputs_for_independent_images(
            [frame], str(self.cfg["document_instruction"])
        )
        if self.mode == "full_embedding":
            outputs = self.embedder.forward(inputs)
            value = self.embedder._pooling_last(
                outputs["last_hidden_state"], outputs["attention_mask"]
            )[0]
        else:
            hidden, _ = self._decoder_hidden_for_inputs(inputs, self.readout_layer)
            value = self.embedder._pooling_last(hidden, inputs["attention_mask"])[0]
        return F.normalize(value.float(), dim=-1).cpu()


class OnlineVectorMemory:
    """Online vector memory updated only as causal chunks arrive.

    ``capacity=-1`` is unbounded.  Any positive value enables FIFO eviction.
    """

    def __init__(self, encoder: SemanticMemoryEncoder, cfg: dict[str, Any]):
        self.mode = encoder.mode
        self.readout_name = encoder.readout_name
        self.readout_layer = encoder.readout_layer
        self.hidden_size = int(encoder.model.config.text_config.hidden_size)
        self.dtype = dtype_from_name(cfg["memory_vector_dtype"])
        self.capacity = validate_memory_capacity(cfg["max_memory_vectors"])
        self.chunk_frames = int(cfg["memory_chunk_frames"])
        self.chunks: list[dict[str, Any]] = []
        self.vectors: list[torch.Tensor] = []
        self.total_vectors_generated = 0
        self.total_vectors_evicted = 0
        self.total_frames_ingested = 0
        self.total_ingest_seconds = 0.0
        self.peak_gpu_memory_bytes = 0

    @property
    def is_unbounded(self) -> bool:
        return self.capacity == -1

    @property
    def capacity_policy(self) -> str:
        return "unbounded" if self.is_unbounded else "fifo_bounded"

    @property
    def capacity_display(self) -> str:
        return "unbounded" if self.is_unbounded else str(self.capacity)

    def append(
        self,
        positions: list[int],
        timestamps: list[float],
        vector: torch.Tensor,
    ) -> None:
        value = vector.detach().reshape(-1).to(device="cpu", dtype=self.dtype)
        if int(value.numel()) != self.hidden_size:
            raise ValueError(
                f"Expected a {self.hidden_size}-D memory vector, got {tuple(value.shape)}"
            )
        record = {
            "stream_chunk_id": self.total_vectors_generated,
            "sample_positions": list(positions),
            "start_timestamp_seconds": float(timestamps[0]),
            "end_timestamp_seconds": float(timestamps[-1]),
        }
        self.total_vectors_generated += 1
        self.total_frames_ingested += len(positions)
        self.chunks.append(record)
        self.vectors.append(value)
        if not self.is_unbounded and len(self.vectors) > self.capacity:
            self.vectors.pop(0)
            self.chunks.pop(0)
            self.total_vectors_evicted += 1

    @property
    def persistent_bytes(self) -> int:
        if not self.vectors:
            return 0
        return int(len(self.vectors) * self.hidden_size * self.vectors[0].element_size())

    @property
    def retained_frame_count(self) -> int:
        return sum(len(item["sample_positions"]) for item in self.chunks)

    @property
    def peak_persistent_bytes(self) -> int:
        if self.is_unbounded:
            peak_vectors = self.total_vectors_generated
        else:
            peak_vectors = min(self.total_vectors_generated, self.capacity)
        return int(
            peak_vectors
            * self.hidden_size
            * torch.empty((), dtype=self.dtype).element_size()
        )

    def as_index(self) -> dict[str, Any]:
        vectors = (
            torch.stack(self.vectors)
            if self.vectors
            else torch.empty((0, self.hidden_size), dtype=self.dtype)
        )
        return {
            "mode": self.mode,
            "readout_name": self.readout_name,
            "representation_kind": "single_vector",
            "vectors": vectors,
            "chunks": list(self.chunks),
            "decoder_layers_executed_per_chunk": self.readout_layer,
            "post_final_norm": self.mode == "full_embedding",
            "max_memory_vectors": self.capacity,
            "memory_capacity_policy": self.capacity_policy,
            "memory_is_unbounded": self.is_unbounded,
            "memory_vectors_retained": len(self.vectors),
            "memory_vectors_generated": self.total_vectors_generated,
            "memory_vectors_evicted": self.total_vectors_evicted,
            "memory_persistent_bytes": self.persistent_bytes,
            "memory_retained_source_frames": self.retained_frame_count,
        }


def ingest_complete_chunks_until(
    memory: OnlineVectorMemory,
    encoder: SemanticMemoryEncoder,
    sampler: VideoSampler,
    next_position: int,
    timestamp: float,
) -> tuple[int, dict[str, Any]]:
    """Encode each newly completed non-overlapping chunk exactly once."""
    available = sampler.positions_available_at(timestamp)
    available_count = available[-1] + 1 if available else 0
    chunk_size = memory.chunk_frames
    chunks_added = 0
    frames_added = 0
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(encoder.device)
    while next_position + chunk_size <= available_count:
        positions = list(range(next_position, next_position + chunk_size))
        frames = sampler.decode_positions(positions)
        vector = encoder.encode_chunk(frames)
        memory.append(positions, sampler.timestamps(positions), vector)
        next_position += chunk_size
        chunks_added += 1
        frames_added += chunk_size
    if chunks_added:
        torch.cuda.synchronize(encoder.device)
    elapsed = time.perf_counter() - started
    memory.total_ingest_seconds += elapsed
    memory.peak_gpu_memory_bytes = max(
        memory.peak_gpu_memory_bytes,
        int(torch.cuda.max_memory_allocated(encoder.device)),
    )
    return next_position, {
        "stream_ingest_seconds_since_previous_query": elapsed,
        "stream_ingest_chunks_since_previous_query": chunks_added,
        "stream_ingest_frames_since_previous_query": frames_added,
        "stream_pending_unencoded_frames": max(0, available_count - next_position),
    }


def uniform_subset(values: list[int], count: int) -> list[int]:
    if count <= 0 or not values:
        return []
    if count >= len(values):
        return list(values)
    indices = np.linspace(0, len(values) - 1, count).round().astype(np.int64)
    return [values[index] for index in np.unique(indices)]


def robust_history_confidence(scores: np.ndarray, cfg: dict[str, Any]) -> dict[str, Any]:
    if scores.size == 0:
        return {"tier": "weak", "top_score": None, "top_z": None, "median": None, "mad": None}
    median = float(np.median(scores))
    mad = float(np.median(np.abs(scores - median)))
    scale = 1.4826 * mad
    top_score = float(scores.max())
    top_z = 0.0 if scores.size < 3 or scale < 1e-6 else (top_score - median) / scale
    medium = float(cfg["history_confidence_z_medium"])
    strong = float(cfg["history_confidence_z_strong"])
    tier = "strong" if top_z >= strong else "medium" if top_z >= medium else "weak"
    return {
        "tier": tier,
        "top_score": top_score,
        "top_z": float(top_z),
        "median": median,
        "mad": mad,
    }


def rank_chunks_with_temporal_nms(
    chunk_ids: list[int], scores: np.ndarray, max_chunks: int, nms_gap: int
) -> list[int]:
    ranked_local = np.argsort(-scores, kind="stable").tolist()
    selected: list[int] = []
    for local_index in ranked_local:
        chunk_id = chunk_ids[local_index]
        if all(abs(chunk_id - previous) > nms_gap for previous in selected):
            selected.append(chunk_id)
            if len(selected) >= max_chunks:
                return selected
    for local_index in ranked_local:
        chunk_id = chunk_ids[local_index]
        if chunk_id not in selected:
            selected.append(chunk_id)
            if len(selected) >= max_chunks:
                break
    return selected


def rank_chunks_balanced_across_queries(
    chunk_ids: list[int],
    similarities: np.ndarray,
    max_chunks: int,
    nms_gap: int,
) -> tuple[list[int], list[int | None]]:
    """Allocate one distinct historical chunk per query before global refill."""

    values = np.asarray(similarities, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] != len(chunk_ids):
        raise ValueError(
            "similarities must have shape [num_chunks, num_queries], got "
            f"{values.shape} for {len(chunk_ids)} chunks"
        )
    limit = min(max(0, int(max_chunks)), len(chunk_ids))
    selected: list[int] = []
    selected_for_query: list[int | None] = []

    def choose_for_query(query_index: int, *, enforce_nms: bool) -> int | None:
        order = np.argsort(-values[:, query_index], kind="stable").tolist()
        for local_index in order:
            chunk_id = chunk_ids[local_index]
            if chunk_id in selected:
                continue
            if enforce_nms and any(
                abs(chunk_id - previous) <= int(nms_gap) for previous in selected
            ):
                continue
            return chunk_id
        return None

    for query_index in range(values.shape[1]):
        if len(selected) >= limit:
            selected_for_query.append(None)
            continue
        chosen = choose_for_query(query_index, enforce_nms=True)
        if chosen is None:
            chosen = choose_for_query(query_index, enforce_nms=False)
        selected_for_query.append(chosen)
        if chosen is not None:
            selected.append(chosen)

    aggregate = values.max(axis=1)
    for enforce_nms in (True, False):
        for local_index in np.argsort(-aggregate, kind="stable").tolist():
            if len(selected) >= limit:
                break
            chunk_id = chunk_ids[local_index]
            if chunk_id in selected:
                continue
            if enforce_nms and any(
                abs(chunk_id - previous) <= int(nms_gap) for previous in selected
            ):
                continue
            selected.append(chunk_id)
    return selected, selected_for_query


def history_pool_from_chunks(
    index: dict[str, Any], selected_chunk_ids: list[int], pool_size: int
) -> list[int]:
    if not selected_chunk_ids or pool_size <= 0:
        return []
    base, remainder = divmod(pool_size, len(selected_chunk_ids))
    positions: list[int] = []
    for order, chunk_id in enumerate(selected_chunk_ids):
        count = base + int(order < remainder)
        chunk_positions = list(index["chunks"][chunk_id]["sample_positions"])
        positions.extend(uniform_subset(chunk_positions, count))
    return sorted(set(positions))


def score_memory_chunks(
    index: dict[str, Any],
    chunk_ids: list[int],
    query: torch.Tensor,
    cfg: dict[str, Any],
    device: torch.device | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    score_device = device or query.device
    normalized_query = F.normalize(query.float(), dim=-1).to(score_device)
    if index["representation_kind"] == "single_vector":
        vectors = F.normalize(index["vectors"][chunk_ids].float(), dim=-1).to(score_device)
        if normalized_query.ndim == 1:
            return (vectors @ normalized_query).float().cpu().numpy(), {
                "type": "single_vector_cosine",
                "stored_vectors_scored": len(chunk_ids),
                "retrieval_query_count": 1,
                "retrieval_query_aggregation": "single",
            }
        if normalized_query.ndim != 2:
            raise ValueError(
                "Retrieval query must have shape [D] or [num_queries, D], got "
                f"{tuple(normalized_query.shape)}"
            )
        similarities = vectors @ normalized_query.transpose(0, 1)
        aggregation = str(cfg.get("reactive_proposal_aggregation", "max"))
        if aggregation == "max":
            scores, winners = similarities.max(dim=1)
        elif aggregation == "mean":
            scores = similarities.mean(dim=1)
            winners = similarities.argmax(dim=1)
        elif aggregation == "logsumexp":
            # Smooth OR over independently useful visual proposals.  Subtract
            # log(P) so its scale stays comparable as proposal count changes.
            scores = torch.logsumexp(similarities, dim=1) - math.log(
                max(1, int(similarities.shape[1]))
            )
            winners = similarities.argmax(dim=1)
        elif aggregation == "question_anchor_rrf":
            # Query 0 is always the original question. Remaining queries are
            # 8B-requested missing evidence. Rank fusion avoids comparing raw
            # cosine scales across generated queries while the higher question
            # weight prevents one hallucinated cue from taking over retrieval.
            similarities_np = similarities.float().cpu().numpy()
            ranks = np.empty_like(similarities_np, dtype=np.float32)
            for query_index in range(similarities_np.shape[1]):
                order = np.argsort(
                    -similarities_np[:, query_index], kind="stable"
                )
                ranks[order, query_index] = np.arange(
                    1, len(order) + 1, dtype=np.float32
                )
            rrf_k = float(cfg.get("reactive_rrf_k", 60.0))
            question_weight = float(
                cfg.get("reactive_question_weight", 1.0)
            )
            proposal_weight = float(
                cfg.get("reactive_proposal_weight", 0.5)
            )
            weights = np.full(
                similarities_np.shape[1], proposal_weight, dtype=np.float32
            )
            weights[0] = question_weight
            fused = (weights.reshape(1, -1) / (rrf_k + ranks)).sum(axis=1)
            scores = torch.from_numpy(fused)
            winners = similarities.argmax(dim=1)
        else:
            raise ValueError(f"Unknown proposal aggregation: {aggregation}")
        return scores.float().cpu().numpy(), {
            "type": "multi_query_single_vector_cosine",
            "stored_vectors_scored": len(chunk_ids),
            "retrieval_query_count": int(normalized_query.shape[0]),
            "retrieval_query_aggregation": aggregation,
            "winning_query_index_per_chunk": winners.cpu().tolist(),
            "rrf_k": (
                float(cfg.get("reactive_rrf_k", 60.0))
                if aggregation == "question_anchor_rrf"
                else None
            ),
            "question_query_weight": (
                float(cfg.get("reactive_question_weight", 1.0))
                if aggregation == "question_anchor_rrf"
                else None
            ),
            "proposal_query_weight": (
                float(cfg.get("reactive_proposal_weight", 0.5))
                if aggregation == "question_anchor_rrf"
                else None
            ),
        }
    raise ValueError(
        f"Chunk scoring requires single vectors, got {index['representation_kind']}"
    )


def memory_bytes_for_chunks(index: dict[str, Any], chunk_ids: list[int]) -> int:
    if not chunk_ids:
        return 0
    if index["representation_kind"] != "single_vector":
        raise ValueError("Chunk byte accounting requires a single_vector index")
    return int(len(chunk_ids) * index["vectors"].shape[1] * index["vectors"].element_size())

def select_semantic_memory_frames(
    index: dict[str, Any],
    query: torch.Tensor,
    sampler: VideoSampler,
    query_timestamp: float,
    cfg: dict[str, Any],
    score_device: torch.device | None = None,
    balanced_query_labels: list[str] | None = None,
    oracle_evidence_ts: list[float] | None = None,
) -> tuple[list[int], dict[str, Any]]:
    started = time.perf_counter()
    oracle_boosted: list[int] = []
    available_positions = sampler.positions_available_at(query_timestamp)
    if not available_positions:
        raise RuntimeError(f"No causal frames are available at timestamp {query_timestamp}")
    recent_pool_size = int(cfg["recent_pool_frames"])
    history_pool_size = int(cfg["history_pool_frames"])
    recent_pool = available_positions[-recent_pool_size:]
    recent_start = recent_pool[0]
    budget = int(cfg["answer_frame_budget"])
    requested_history = min(history_pool_size, budget)
    requested_recent = min(recent_pool_size, max(0, budget - requested_history))
    candidate_chunk_ids: list[int] = []
    selected_chunk_ids: list[int] = []
    if index["representation_kind"] != "single_vector":
        raise ValueError(
            f"Bounded streaming memory requires single vectors, got "
            f"{index['representation_kind']}"
        )
    candidate_chunk_ids = [
        chunk_id
        for chunk_id, chunk in enumerate(index["chunks"])
        if int(chunk["sample_positions"][-1]) < recent_start
        and float(chunk["end_timestamp_seconds"]) <= float(query_timestamp) + 1e-6
    ]
    if candidate_chunk_ids:
        if balanced_query_labels is not None:
            normalized_query = F.normalize(query.float(), dim=-1)
            if normalized_query.ndim != 2:
                raise ValueError("Balanced query allocation requires a query matrix")
            if len(balanced_query_labels) != int(normalized_query.shape[0]):
                raise ValueError("Balanced query labels do not match query count")
            device = score_device or normalized_query.device
            vectors = F.normalize(
                index["vectors"][candidate_chunk_ids].float(), dim=-1
            ).to(device)
            similarities = (
                vectors @ normalized_query.to(device).transpose(0, 1)
            ).float().cpu().numpy()
            scores = similarities.max(axis=1)
            selected_chunk_ids, per_query_chunks = rank_chunks_balanced_across_queries(
                candidate_chunk_ids,
                similarities,
                requested_history,
                (
                    int(cfg["history_chunk_nms_gap"])
                    if bool(cfg["history_temporal_nms"])
                    else -1
                ),
            )
            score_details = {
                "type": "balanced_option_conditioned_cosine",
                "stored_vectors_scored": len(candidate_chunk_ids),
                "retrieval_query_count": int(normalized_query.shape[0]),
                "retrieval_query_aggregation": "one_chunk_per_query_then_max_refill",
                "balanced_query_labels": list(balanced_query_labels),
                "balanced_selected_chunk_id_per_query": per_query_chunks,
            }
            history_ranking_policy = "balanced_one_per_query_with_temporal_nms"
        else:
            scores, score_details = score_memory_chunks(
                index, candidate_chunk_ids, query, cfg, score_device
            )
            selected_chunk_ids = []
        # Oracle rerank (analysis only, off by default): an upper bound for a
        # reranker that reorders the retriever's own top-k. Within the top-k by
        # the original score, chunks within a tolerance of an annotated evidence
        # timestamp jump to the front; nothing outside the top-k is touched, so
        # questions whose evidence the retriever ranked lower stay as they were.
        # The logged scores remain the original ones.
        rank_scores = scores
        oracle_k = int(cfg.get("oracle_rerank_top_k", 0) or 0)
        if oracle_k > 0 and oracle_evidence_ts and balanced_query_labels is None:
            tol = float(cfg.get("oracle_rerank_tolerance_seconds", 4.0))
            evidence = [float(t) for t in oracle_evidence_ts if float(t) <= float(query_timestamp)]
            rank_scores = np.array(scores, dtype=np.float64, copy=True)
            for local_index in np.argsort(-scores, kind="stable")[:oracle_k].tolist():
                chunk = index["chunks"][candidate_chunk_ids[local_index]]
                lo = float(chunk["start_timestamp_seconds"]) - tol
                hi = float(chunk["end_timestamp_seconds"]) + tol
                if any(lo <= t <= hi for t in evidence):
                    rank_scores[local_index] += 10.0
                    oracle_boosted.append(int(candidate_chunk_ids[local_index]))
        if balanced_query_labels is not None:
            pass
        elif bool(cfg["history_temporal_nms"]):
            selected_chunk_ids = rank_chunks_with_temporal_nms(
                candidate_chunk_ids,
                rank_scores,
                int(cfg["history_max_chunks"]),
                int(cfg["history_chunk_nms_gap"]),
            )
            history_ranking_policy = "cosine_topk_with_temporal_nms"
        else:
            ranked_local = np.argsort(-rank_scores, kind="stable").tolist()
            selected_chunk_ids = [
                candidate_chunk_ids[local_index]
                for local_index in ranked_local[: int(cfg["history_max_chunks"])]
            ]
            history_ranking_policy = "cosine_topk_without_temporal_nms"
        history_pool = history_pool_from_chunks(
            index, selected_chunk_ids, history_pool_size
        )
        memory_bytes_available = memory_bytes_for_chunks(index, candidate_chunk_ids)
    else:
        scores = np.asarray([], dtype=np.float32)
        score_details = {"type": "no_historical_chunk_candidates"}
        history_ranking_policy = (
            "cosine_topk_with_temporal_nms"
            if bool(cfg["history_temporal_nms"])
            else "cosine_topk_without_temporal_nms"
        )
        history_pool = []
        memory_bytes_available = 0

    confidence = robust_history_confidence(scores, cfg)
    history_frames = history_pool[:requested_history]
    recent_frames = recent_pool[-requested_recent:]
    retrieved_history_set = set(history_frames)
    requested_recent_set = set(recent_frames)
    selected = list(dict.fromkeys(history_frames + recent_frames))

    # In FIFO mode, never refill from evicted history: MAX_MEMORY_VECTORS is a
    # real information bound.  In unbounded mode no vectors are evicted.  In
    # both cases we may only use positions represented by the current memory
    # plus a raw recent ring buffer of ``budget`` frames.
    retained_positions = sorted(
        {
            int(position)
            for chunk in index["chunks"]
            for position in chunk["sample_positions"]
            if int(position) < recent_start
        }
    )
    raw_recent_buffer = available_positions[-budget:]
    refill_order = history_pool + list(reversed(retained_positions)) + list(
        reversed(raw_recent_buffer)
    )
    for position in refill_order:
        if len(selected) >= min(budget, len(available_positions)):
            break
        if position not in selected:
            selected.append(position)
    selected = sorted(selected)
    metrics = {
        "router_seconds": time.perf_counter() - started,
        "memory_num_chunks_total": len(index["chunks"]),
        "memory_num_chunks_available": len(candidate_chunk_ids),
        "memory_num_frames_available": len(retained_positions),
        "memory_index_bytes_available": memory_bytes_available,
        "memory_persistent_bytes": int(index.get("memory_persistent_bytes", 0)),
        "memory_vectors_capacity": int(index.get("max_memory_vectors", 0)),
        "memory_capacity_policy": str(
            index.get("memory_capacity_policy", "fifo_bounded")
        ),
        "memory_is_unbounded": bool(index.get("memory_is_unbounded", False)),
        "memory_vectors_retained": int(index.get("memory_vectors_retained", 0)),
        "memory_vectors_generated": int(index.get("memory_vectors_generated", 0)),
        "memory_vectors_evicted": int(index.get("memory_vectors_evicted", 0)),
        "memory_retained_source_frames": int(
            index.get("memory_retained_source_frames", 0)
        ),
        "memory_candidate_chunk_ids": candidate_chunk_ids,
        "memory_candidate_chunk_stream_ids": [
            int(index["chunks"][value]["stream_chunk_id"])
            for value in candidate_chunk_ids
        ],
        "memory_candidate_chunk_scores": scores.tolist(),
        "memory_score_details": score_details,
        "history_temporal_nms": bool(cfg["history_temporal_nms"]),
        "history_temporal_nms_gap": int(cfg["history_chunk_nms_gap"]),
        "history_ranking_policy": history_ranking_policy,
        "memory_selected_chunk_ids": selected_chunk_ids,
        "oracle_rerank_top_k": int(cfg.get("oracle_rerank_top_k", 0) or 0),
        "oracle_rerank_boosted_chunk_ids": oracle_boosted,
        "memory_selected_chunk_stream_ids": [
            int(index["chunks"][value]["stream_chunk_id"])
            for value in selected_chunk_ids
        ],
        "memory_confidence": confidence,
        "frame_allocation_policy": (
            f"fixed_history{requested_history}_recent{requested_recent}"
        ),
        "history_pool_positions": history_pool,
        "recent_pool_positions": recent_pool,
        "requested_history_frames": requested_history,
        "requested_recent_frames": requested_recent,
        "answer_history_frames": len([value for value in selected if value < recent_start]),
        "answer_recent_frames": len([value for value in selected if value >= recent_start]),
        "answer_retrieved_history_frames": len(retrieved_history_set.intersection(selected)),
        "answer_requested_recent_frames": len(requested_recent_set.intersection(selected)),
        "answer_fallback_frames": len(
            set(selected).difference(retrieved_history_set).difference(requested_recent_set)
        ),
    }
    return selected, metrics


def _prefixed_metrics(prefix: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}{key}": value for key, value in metrics.items()}


def _recent_only_router_metrics(
    index: dict[str, Any],
    selected: list[int],
    available: list[int],
) -> dict[str, Any]:
    return {
        "router_seconds": 0.0,
        "memory_num_chunks_total": len(index.get("chunks", [])),
        "memory_num_chunks_available": 0,
        "memory_num_frames_available": 0,
        "memory_index_bytes_available": 0,
        "memory_persistent_bytes": int(index.get("memory_persistent_bytes", 0)),
        "memory_vectors_capacity": int(index.get("max_memory_vectors", 0)),
        "memory_capacity_policy": str(index.get("memory_capacity_policy", "fifo_bounded")),
        "memory_is_unbounded": bool(index.get("memory_is_unbounded", False)),
        "memory_vectors_retained": int(index.get("memory_vectors_retained", 0)),
        "memory_vectors_generated": int(index.get("memory_vectors_generated", 0)),
        "memory_vectors_evicted": int(index.get("memory_vectors_evicted", 0)),
        "memory_retained_source_frames": int(index.get("memory_retained_source_frames", 0)),
        "memory_candidate_chunk_ids": [],
        "memory_candidate_chunk_stream_ids": [],
        "memory_candidate_chunk_scores": [],
        "memory_score_details": {"type": "recent_first_direct_no_retrieval"},
        "history_ranking_policy": "not_called",
        "memory_selected_chunk_ids": [],
        "memory_selected_chunk_stream_ids": [],
        "frame_allocation_policy": f"recent_first_direct_{len(selected)}",
        "history_pool_positions": [],
        "recent_pool_positions": list(selected),
        "requested_history_frames": 0,
        "requested_recent_frames": len(selected),
        "answer_history_frames": 0,
        "answer_recent_frames": len(selected),
        "answer_retrieved_history_frames": 0,
        "answer_requested_recent_frames": len(selected),
        "answer_fallback_frames": 0,
        "raw_available_frames_at_query": len(available),
    }


def select_oracle_interval_frames(
    item: dict[str, Any],
    sampler: VideoSampler,
    query_timestamp: float,
    cfg: dict[str, Any],
    *,
    index: dict[str, Any] | None = None,
) -> tuple[list[int], dict[str, Any]]:
    """Select causal recent frames plus frames from a GT evidence interval.

    This is an explicitly non-deployable accuracy-ceiling baseline.  The
    interval comes from benchmark annotation metadata, never from the answer
    option.  Intervals are clipped at the query timestamp; future pixels are
    never exposed even when an annotation range extends past the question.
    """

    reference = item.get("time_reference") or []
    if len(reference) != 2:
        raise ValueError(
            "oracle_interval requires one annotated [start, end] evidence interval"
        )
    start, end = sorted(float(value) for value in reference)
    available = sampler.positions_available_at(query_timestamp)
    if not available:
        raise RuntimeError(
            f"No causal frames are available at timestamp {query_timestamp}"
        )
    recent_count = min(int(cfg.get("oracle_recent_frames", 4)), len(available))
    recent_positions = available[-recent_count:] if recent_count > 0 else []
    causal_end = min(end, float(query_timestamp))
    tolerance = 0.5 / max(float(cfg["sample_fps"]), 1e-6) + 1e-6
    oracle_pool = [
        position
        for position in available
        if start - tolerance
        <= float(sampler.sample_timestamps[position])
        <= causal_end + tolerance
    ]
    requested_oracle = int(cfg.get("oracle_evidence_frames", 4))
    if oracle_pool:
        local_indices = np.linspace(
            0,
            len(oracle_pool) - 1,
            min(requested_oracle, len(oracle_pool)),
        ).round().astype(np.int64)
        oracle_positions = [oracle_pool[int(value)] for value in np.unique(local_indices)]
    else:
        oracle_positions = []
    selected = sorted(dict.fromkeys([*oracle_positions, *recent_positions]))
    recent_set = set(recent_positions)
    oracle_set = set(oracle_positions)
    router_metrics = _recent_only_router_metrics(index or {}, selected, available)
    router_metrics.update(
        {
            "memory_score_details": {
                "type": "oracle_ground_truth_evidence_interval_no_retrieval"
            },
            "frame_allocation_policy": (
                f"oracle_interval{requested_oracle}_recent{recent_count}"
            ),
            "history_pool_positions": oracle_positions,
            "recent_pool_positions": recent_positions,
            "requested_history_frames": requested_oracle,
            "requested_recent_frames": recent_count,
            "answer_history_frames": len(oracle_set.difference(recent_set)),
            "answer_recent_frames": len(recent_set.intersection(selected)),
            "answer_retrieved_history_frames": 0,
            "answer_requested_recent_frames": len(recent_set.intersection(selected)),
            "answer_fallback_frames": 0,
            "oracle_ground_truth_metadata_used": True,
            "oracle_reference_interval_seconds": [start, end],
            "oracle_causal_interval_seconds": (
                [start, causal_end] if causal_end + tolerance >= start else []
            ),
            "oracle_interval_has_causal_overlap": bool(oracle_pool),
            "oracle_interval_extended_past_query": end > query_timestamp + 1e-6,
            "oracle_requested_evidence_frames": requested_oracle,
            "oracle_selected_evidence_frames": len(oracle_positions),
            "oracle_requested_recent_frames": int(
                cfg.get("oracle_recent_frames", 4)
            ),
            "oracle_selected_recent_frames": len(recent_positions),
            "oracle_selected_positions": oracle_positions,
        }
    )
    return selected, router_metrics


def execute_reactive_query(
    *,
    item: dict[str, Any],
    cfg: dict[str, Any],
    encoder: SemanticMemoryEncoder,
    answer_model: AnswerModel,
    index: dict[str, Any],
    sampler: VideoSampler,
    query_timestamp: float,
    answer_prompt: str,
    recent_gate_prompt: str,
    planner_prompt: str,
    proposal_prompt: str,
    router_query: str,
    score_device: torch.device,
) -> tuple[str, list[int], dict[str, Any]]:
    """Run the frozen retrieve-first baseline or the isolated recent-first branch.

    ``retrieve_first`` is byte-for-byte the old decision order: encode the
    question, retrieve history, then call 8B once.  ``recent_first`` first asks
    8B whether the recent window is sufficient.  Only an explicit
    ``ANSWER: X`` bypasses retrieval.  Otherwise 2B retrieves with either the
    original question or a small set of 8B-predicted visible evidence frames.
    ``recent_first_plan`` is separate: its first 8B call either answers or
    emits the missing-evidence queries, eliminating the extra compiler call.
    ``abstention_aware`` keeps ordinary recent-first behavior for normal
    questions, but bypasses that unsafe early exit when a null-answer option
    exists and retrieves one historical candidate per concrete option.
    """

    policy = str(cfg.get("reactive_policy", "retrieve_first"))
    retrieval_query_mode = str(cfg.get("reactive_retrieval_query", "question"))
    abstention_index = abstention_option_index(item)
    abstention_active = bool(
        policy == "abstention_aware" and abstention_index is not None
    )
    if abstention_active:
        retrieval_query_mode = "option_hypotheses_balanced"
    if policy == "recent_first_plan":
        retrieval_query_mode = "question_plus_missing_evidence"
    available = sampler.positions_available_at(query_timestamp)
    if not available:
        raise RuntimeError(f"No causal frames are available at timestamp {query_timestamp}")

    recent_gate_decode_seconds = 0.0
    recent_gate_metrics: dict[str, Any] = {}
    proposal_compiler_metrics: dict[str, Any] = {}
    proposal_texts: list[str] = []
    proposal_parse_fallback = False
    recent_decision = "not_used"
    recent_response = ""
    recent_positions: list[int] = []
    recent_images: list[Image.Image] = []
    balanced_query_labels: list[str] | None = None
    option_conditioned_queries: list[str] = []

    if policy == "oracle_interval":
        selected_positions, router_metrics = select_oracle_interval_frames(
            item,
            sampler,
            query_timestamp,
            cfg,
            index=index,
        )
        timestamps = sampler.timestamps(selected_positions)
        evidence_started = time.perf_counter()
        answer_input = sampler.decode_positions(selected_positions)
        evidence_decode_seconds = time.perf_counter() - evidence_started
        response, answer_metrics = answer_model.answer(
            answer_input,
            answer_prompt,
            fps=float(cfg["sample_fps"]),
            max_frames=len(selected_positions),
            visual_token_budget=None,
            frame_timestamps=timestamps,
            token_reduction_enabled_override=False,
        )
        query_pipeline = (
            evidence_decode_seconds
            + float(router_metrics["router_seconds"])
            + float(answer_metrics["answer_total_seconds"])
        )
        metrics = {
            **router_metrics,
            **answer_metrics,
            "reactive_policy": policy,
            "reactive_retrieval_query": "ground_truth_evidence_interval",
            "recent_first_decision": "oracle_interval",
            "recent_first_raw_response": "",
            "recent_first_frames": int(
                router_metrics["oracle_selected_recent_frames"]
            ),
            "recent_gate_decode_seconds": 0.0,
            "reactive_retrieval_called": False,
            "reactive_proposal_compiler_called": False,
            "reactive_proposal_texts": [],
            "reactive_proposal_count": 0,
            "reactive_proposal_parse_fallback": False,
            "embedding_2b_query_calls": 0,
            "answer_8b_calls_total": 1,
            "query_encode_seconds": 0.0,
            "query_encoder_peak_gpu_memory_bytes": 0,
            "router_peak_gpu_memory_bytes": 0,
            "evidence_decode_seconds": evidence_decode_seconds,
            "query_pipeline_seconds": query_pipeline,
        }
        return response, selected_positions, metrics

    uses_recent_gate = bool(
        policy in {"recent_first", "recent_first_plan"}
        or (policy == "abstention_aware" and not abstention_active)
    )
    if abstention_active:
        recent_count = min(int(cfg.get("recent_first_frames", 4)), len(available))
        recent_positions = available[-recent_count:]
        recent_decision = "abstention_aware_retrieve"
    elif uses_recent_gate:
        recent_count = min(int(cfg.get("recent_first_frames", 4)), len(available))
        recent_positions = available[-recent_count:]
        recent_timestamps = sampler.timestamps(recent_positions)
        decode_started = time.perf_counter()
        recent_images = sampler.decode_positions(recent_positions)
        recent_gate_decode_seconds = time.perf_counter() - decode_started
        recent_response, raw_gate_metrics = answer_model.answer(
            recent_images,
            planner_prompt if policy == "recent_first_plan" else recent_gate_prompt,
            fps=float(cfg["sample_fps"]),
            max_frames=len(recent_positions),
            visual_token_budget=None,
            frame_timestamps=recent_timestamps,
            max_new_tokens_override=int(
                cfg.get(
                    "recent_first_plan_max_new_tokens",
                    64,
                )
                if policy == "recent_first_plan"
                else cfg.get("recent_first_gate_max_new_tokens", 8)
            ),
            token_reduction_enabled_override=False,
        )
        if policy == "recent_first_plan":
            recent_decision, direct_answer, proposal_texts = (
                parse_recent_first_plan(
                    recent_response,
                    num_options=len(item.get("choices", item.get("options", []))),
                    proposal_limit=int(cfg.get("reactive_proposal_count", 3)),
                )
            )
            proposal_parse_fallback = recent_decision == "RETRIEVE" and not proposal_texts
        else:
            recent_decision, direct_answer = parse_recent_first_gate(
                recent_response,
                num_options=len(item.get("choices", item.get("options", []))),
            )
        recent_gate_metrics = _prefixed_metrics("recent_gate_", raw_gate_metrics)
        if recent_decision == "ANSWER" and direct_answer is not None:
            router_metrics = _recent_only_router_metrics(index, recent_positions, available)
            metrics = {
                **raw_gate_metrics,
                **recent_gate_metrics,
                **router_metrics,
                "reactive_policy": policy,
                "reactive_retrieval_query": retrieval_query_mode,
                "recent_first_decision": "direct_answer",
                "recent_first_raw_response": recent_response,
                "recent_first_frames": len(recent_positions),
                "recent_gate_decode_seconds": recent_gate_decode_seconds,
                "reactive_retrieval_called": False,
                "reactive_proposal_compiler_called": False,
                "reactive_proposal_texts": [],
                "reactive_proposal_parse_fallback": False,
                "embedding_2b_query_calls": 0,
                "answer_8b_calls_total": 1,
                "query_encode_seconds": 0.0,
                "query_encoder_peak_gpu_memory_bytes": 0,
                "router_peak_gpu_memory_bytes": 0,
                "evidence_decode_seconds": recent_gate_decode_seconds,
                "query_pipeline_seconds": (
                    recent_gate_decode_seconds + float(raw_gate_metrics["answer_total_seconds"])
                ),
            }
            return direct_answer, recent_positions, metrics
    elif policy not in {"retrieve_first", "abstention_aware"}:
        raise ValueError(f"Unknown reactive_policy: {policy}")

    query_encode_seconds = 0.0
    embedding_query_calls = 0
    token_reduction_query: torch.Tensor | None = None
    torch.cuda.reset_peak_memory_stats(score_device)
    if abstention_active:
        option_conditioned_queries, option_labels = build_option_conditioned_queries(
            item
        )
        option_vectors, elapsed = encoder.query_vectors(
            option_conditioned_queries,
            instruction=(
                "Represent a candidate-answer hypothesis for finding direct visual "
                "evidence in an earlier video segment."
            ),
        )
        query_encode_seconds += float(elapsed)
        embedding_query_calls += 1
        question_vector, elapsed = encoder.query_vector(router_query)
        query_encode_seconds += float(elapsed)
        embedding_query_calls += 1
        retrieval_query = torch.stack([*option_vectors, question_vector], dim=0)
        balanced_query_labels = [*option_labels, "original_question"]
        token_reduction_query = question_vector
    elif policy == "recent_first_plan":
        question_vector, elapsed = encoder.query_vector(router_query)
        query_encode_seconds += float(elapsed)
        embedding_query_calls += 1
        token_reduction_query = question_vector
        if proposal_texts:
            vectors, elapsed = encoder.query_vectors(
                proposal_texts,
                instruction=str(
                    cfg.get(
                        "reactive_proposal_instruction",
                        "Find the past video segment that visually matches this evidence description.",
                    )
                ),
            )
            query_encode_seconds += float(elapsed)
            embedding_query_calls += 1
            retrieval_query = torch.stack([question_vector, *vectors], dim=0)
        else:
            retrieval_query = question_vector
    elif policy == "recent_first" and retrieval_query_mode == "proposal":
        proposal_response, raw_proposal_metrics = answer_model.answer(
            recent_images,
            proposal_prompt,
            fps=float(cfg["sample_fps"]),
            max_frames=len(recent_positions),
            visual_token_budget=None,
            frame_timestamps=sampler.timestamps(recent_positions),
            max_new_tokens_override=int(
                cfg.get("reactive_proposal_max_new_tokens", 64)
            ),
            token_reduction_enabled_override=False,
        )
        proposal_compiler_metrics = _prefixed_metrics(
            "reactive_proposal_compiler_", raw_proposal_metrics
        )
        proposal_texts = parse_visual_proposals(
            proposal_response, int(cfg.get("reactive_proposal_count", 4))
        )
        if len(proposal_texts) == int(cfg.get("reactive_proposal_count", 4)):
            vectors, elapsed = encoder.query_vectors(
                proposal_texts,
                instruction=str(
                    cfg.get(
                        "reactive_proposal_instruction",
                        "Find the past video segment that visually matches this evidence description.",
                    )
                ),
            )
            retrieval_query = torch.stack(vectors, dim=0)
            query_encode_seconds += float(elapsed)
            embedding_query_calls += 1
        else:
            proposal_parse_fallback = True
            proposal_texts = []
            retrieval_query, elapsed = encoder.query_vector(router_query)
            query_encode_seconds += float(elapsed)
            embedding_query_calls += 1
    else:
        retrieval_query, elapsed = encoder.query_vector(router_query)
        query_encode_seconds += float(elapsed)
        embedding_query_calls += 1

    if bool(cfg.get("token_reduction_enabled", False)) and token_reduction_query is None:
        if retrieval_query.ndim == 1 and (
            retrieval_query_mode == "question" or policy == "retrieve_first"
        ):
            token_reduction_query = retrieval_query
        else:
            token_reduction_query, elapsed = encoder.query_vector(router_query)
            query_encode_seconds += float(elapsed)
            embedding_query_calls += 1

    query_peak = int(torch.cuda.max_memory_allocated(score_device))
    torch.cuda.reset_peak_memory_stats(score_device)
    selected_positions, router_metrics = select_semantic_memory_frames(
        index,
        retrieval_query,
        sampler,
        query_timestamp,
        cfg,
        score_device=score_device,
        balanced_query_labels=balanced_query_labels,
        oracle_evidence_ts=(item.get("evidence_ts_sec")
                            if int(cfg.get("oracle_rerank_top_k", 0) or 0) > 0 else None),
    )
    router_metrics["router_peak_gpu_memory_bytes"] = int(
        torch.cuda.max_memory_allocated(score_device)
    )
    timestamps = sampler.timestamps(selected_positions)
    if any(value > query_timestamp + 1e-6 for value in timestamps):
        raise RuntimeError(
            f"Causal assertion failed: query={query_timestamp}, selected={timestamps}"
        )
    evidence_started = time.perf_counter()
    answer_input = sampler.decode_positions(selected_positions)
    evidence_decode_seconds = time.perf_counter() - evidence_started
    final_answer_prompt = (
        build_abstention_aware_answer_prompt(item)
        if abstention_active
        else answer_prompt
    )
    response, answer_metrics = answer_model.answer(
        answer_input,
        final_answer_prompt,
        fps=float(cfg["sample_fps"]),
        max_frames=len(selected_positions),
        visual_token_budget=None,
        frame_timestamps=timestamps,
        token_reduction_encoder=encoder,
        token_reduction_query=token_reduction_query,
    )
    compiler_calls = int(bool(proposal_compiler_metrics))
    query_pipeline = (
        recent_gate_decode_seconds
        + float(recent_gate_metrics.get("recent_gate_answer_total_seconds", 0.0))
        + float(
            proposal_compiler_metrics.get(
                "reactive_proposal_compiler_answer_total_seconds", 0.0
            )
        )
        + query_encode_seconds
        + float(router_metrics["router_seconds"])
        + evidence_decode_seconds
        + float(answer_metrics["answer_total_seconds"])
    )
    metrics = {
        **router_metrics,
        **answer_metrics,
        **recent_gate_metrics,
        **proposal_compiler_metrics,
        "reactive_policy": policy,
        "reactive_retrieval_query": retrieval_query_mode,
        "recent_first_decision": (
            recent_decision
            if abstention_active
            else "retrieve"
            if uses_recent_gate
            else "not_used"
        ),
        "recent_first_raw_response": recent_response,
        "recent_first_frames": len(recent_positions),
        "recent_gate_decode_seconds": recent_gate_decode_seconds,
        "reactive_retrieval_called": True,
        "reactive_proposal_compiler_called": bool(compiler_calls),
        "reactive_proposal_texts": proposal_texts,
        "reactive_proposal_count": len(proposal_texts),
        "reactive_proposal_aggregation": str(
            cfg.get("reactive_proposal_aggregation", "max")
        ),
        "reactive_proposal_parse_fallback": proposal_parse_fallback,
        "abstention_aware_active": abstention_active,
        "abstention_option_index": abstention_index,
        "abstention_option_letter": (
            chr(65 + abstention_index) if abstention_index is not None else None
        ),
        "option_conditioned_queries": option_conditioned_queries,
        "option_conditioned_query_count": len(option_conditioned_queries),
        "balanced_query_labels": balanced_query_labels,
        "embedding_2b_query_calls": embedding_query_calls,
        "answer_8b_calls_total": (
            1
            + int(uses_recent_gate)
            + compiler_calls
        ),
        "query_encode_seconds": query_encode_seconds,
        "query_encoder_peak_gpu_memory_bytes": query_peak,
        "evidence_decode_seconds": evidence_decode_seconds,
        "query_pipeline_seconds": query_pipeline,
    }
    return response, selected_positions, metrics


def group_annotations_by_source(
    annotations: list[dict[str, Any]], world_size: int
) -> list[list[list[dict[str, Any]]]]:
    """Keep all questions for one source video on the same rank.

    OVO ships cumulative prefix clips rather than source videos in this local
    installation.  For a source group, the longest prefix is a valid causal
    source for every earlier question as long as all index/filter operations
    are bounded by each item's ``realtime`` timestamp.
    """
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in annotations:
        grouped[str(item["video"])].append(item)
    groups = [sorted(items, key=lambda value: (float(value["realtime"]), int(value["id"]))) for items in grouped.values()]

    # Greedy deterministic balancing accounts for one history ingest per
    # source plus one 8B answer pass per question.
    bins: list[list[list[dict[str, Any]]]] = [[] for _ in range(world_size)]
    loads = [0.0 for _ in range(world_size)]
    groups.sort(
        key=lambda items: (
            -(max(float(item["realtime"]) for item in items) + 16.0 * len(items)),
            str(items[0]["video"]),
        )
    )
    for items in groups:
        rank = int(np.argmin(np.asarray(loads)))
        bins[rank].append(items)
        loads[rank] += max(float(item["realtime"]) for item in items) + 16.0 * len(items)
    return bins


def source_prefix_path(items: list[dict[str, Any]], video_root: str | Path) -> Path:
    longest = max(items, key=lambda value: (float(value["realtime"]), int(value["id"])))
    path = Path(video_root) / f"{longest['id']}.mp4"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def load_done(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                done.add(json.loads(line)["_key"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def prepare_resumable_jsonl(path: Path) -> None:
    """Make append-resume safe after an interrupted final JSONL write.

    A process can be killed after writing only part of the last JSON object.
    Appending directly would concatenate the next record onto that fragment.
    Only a malformed *trailing* fragment is discarded; malformed completed
    lines remain a hard error because silently skipping them could mix or lose
    experiment records.
    """

    if not path.exists() or path.stat().st_size == 0:
        return
    payload = path.read_bytes()
    lines = payload.splitlines(keepends=True)
    valid_end = 0
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            valid_end += len(line)
            continue
        try:
            value = json.loads(stripped)
            if "_key" not in value:
                raise KeyError("_key")
        except (json.JSONDecodeError, KeyError) as error:
            if index != len(lines) - 1:
                raise RuntimeError(
                    f"Malformed non-trailing JSONL record in {path} at "
                    f"line {index + 1}; refuse unsafe resume."
                ) from error
            path.write_bytes(payload[:valid_end])
            return
        valid_end += len(line)
    if payload and not payload.endswith(b"\n"):
        path.write_bytes(payload + b"\n")


def score_results(records: list[dict[str, Any]]) -> dict[str, Any]:
    values: dict[str, list[int]] = defaultdict(list)
    for record in records:
        response = str(record.get("response") or "").upper()
        values[record["task"]].append(int(record["ground_truth"] in response))
    task_scores = {
        task: {
            "accuracy": sum(items) / len(items),
            "paper_table_value": 100.0 * sum(items) / len(items),
            "num_evaluations": len(items),
        }
        for task, items in sorted(values.items())
    }
    category_scores = {}
    for category, tasks in CATEGORIES.items():
        available = [task_scores[task]["accuracy"] for task in tasks if task in task_scores]
        category_scores[category] = {
            "accuracy": sum(available) / len(available) if available else None,
            "paper_table_value": 100.0 * sum(available) / len(available) if available else None,
            "num_tasks": len(available),
        }
    category_values = [
        category_scores[name]["accuracy"]
        for name in ("realtime", "backward")
        if category_scores[name]["accuracy"] is not None
    ]
    overall_accuracy = sum(category_values) / len(category_values) if category_values else None
    recent_breakdown: dict[str, Any] = {}
    for decision in ("direct_answer", "retrieve"):
        subset = [
            record for record in records
            if record.get("metrics", {}).get("recent_first_decision") == decision
        ]
        if subset:
            correct = [
                int(str(record["ground_truth"]).upper() in str(record.get("response") or "").upper())
                for record in subset
            ]
            recent_breakdown[decision] = {
                "accuracy": float(np.mean(correct)),
                "paper_table_value": float(100.0 * np.mean(correct)),
                "num_evaluations": len(correct),
            }
    abstention_subset = [
        record
        for record in records
        if record.get("metrics", {}).get("recent_first_decision")
        == "abstention_aware_retrieve"
    ]
    if abstention_subset:
        correct = [
            int(str(record["ground_truth"]).upper() in str(record.get("response") or "").upper())
            for record in abstention_subset
        ]
        recent_breakdown["abstention_aware_retrieve"] = {
            "accuracy": float(np.mean(correct)),
            "paper_table_value": float(100.0 * np.mean(correct)),
            "num_evaluations": len(correct),
        }
    return {
        "overall": {
            "accuracy": overall_accuracy,
            "paper_table_value": (
                100.0 * overall_accuracy if overall_accuracy is not None else None
            ),
            "definition": "macro_average_of_realtime_and_backward",
            "num_categories": len(category_values),
        },
        "task_scores": task_scores,
        "category_scores": category_scores,
        "table1_average": {
            "accuracy": overall_accuracy,
            "paper_table_value": (
                100.0 * overall_accuracy if overall_accuracy is not None else None
            ),
            "alias_of": "overall",
        },
        "recent_first_breakdown": recent_breakdown,
    }


def efficiency_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    keys = [
        "raw_sampled_frames",
        "answer_frames",
        "answer_requested_frames",
        "answer_processed_frames",
        "answer_frame_padding",
        "frame_keep_ratio",
        "stream_ingest_seconds_since_previous_query",
        "stream_ingest_chunks_since_previous_query",
        "stream_ingest_frames_since_previous_query",
        "stream_pending_unencoded_frames",
        "memory_index_bytes_available",
        "memory_persistent_bytes",
        "memory_vectors_capacity",
        "memory_vectors_retained",
        "memory_vectors_generated",
        "memory_vectors_evicted",
        "memory_retained_source_frames",
        "query_encode_seconds",
        "embedding_2b_query_calls",
        "router_seconds",
        "router_peak_gpu_memory_bytes",
        "evidence_decode_seconds",
        "query_encoder_peak_gpu_memory_bytes",
        "memory_build_peak_gpu_memory_bytes",
        "answer_preprocess_seconds",
        "answer_8b_vit_seconds",
        "answer_generate_seconds",
        "answer_full_generate_seconds",
        "answer_total_seconds",
        "answer_8b_calls_total",
        "recent_first_frames",
        "recent_gate_decode_seconds",
        "recent_gate_answer_total_seconds",
        "reactive_proposal_compiler_answer_total_seconds",
        "reactive_proposal_count",
        "query_pipeline_seconds",
        "online_event_total_seconds",
        "sample_total_seconds",
        "answer_input_tokens",
        "answer_input_tokens_after_reduction",
        "answer_visual_tokens_before_reduction",
        "answer_visual_tokens",
        "answer_visual_token_keep_ratio",
        "token_reduction_seconds",
        "token_reduction_worker_enqueue_seconds",
        "token_reduction_future_wait_seconds",
        "token_reduction_parallel_visual_critical_path_seconds",
        "token_reduction_cuda_overlap_with_8b_vit_seconds",
        "token_reduction_decoder_layers_executed",
        "token_reduction_frames",
        "token_reduction_2b_visual_tokens_scored",
        "token_reduction_peak_gpu_memory_bytes",
        "token_reduction_persistent_decoder_token_bytes",
        "peak_gpu_memory_bytes",
        "resident_models_allocated_bytes",
        "resident_models_reserved_bytes",
    ]
    result: dict[str, Any] = {"num_samples": len(records)}
    for key in keys:
        values = [float(item["metrics"][key]) for item in records if key in item.get("metrics", {})]
        if values:
            result[key] = {
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "p95": float(np.percentile(values, 95)),
                "total": float(np.sum(values)) if "seconds" in key else None,
            }
    recent_records = [
        item for item in records
        if item.get("metrics", {}).get("reactive_policy")
        in {"recent_first", "recent_first_plan", "abstention_aware"}
    ]
    if recent_records:
        direct = sum(
            item["metrics"].get("recent_first_decision") == "direct_answer"
            for item in recent_records
        )
        proposal_calls = sum(
            bool(item["metrics"].get("reactive_proposal_compiler_called", False))
            for item in recent_records
        )
        result["recent_first_routing"] = {
            "reactive_policy": str(
                recent_records[0]["metrics"].get("reactive_policy")
            ),
            "num_samples": len(recent_records),
            "direct_answer_count": direct,
            "direct_answer_rate": direct / len(recent_records),
            "retrieval_fallback_count": len(recent_records) - direct,
            "retrieval_fallback_rate": (len(recent_records) - direct) / len(recent_records),
            "proposal_compiler_call_count": proposal_calls,
            "proposal_compiler_call_rate": proposal_calls / len(recent_records),
            "proposal_parse_fallback_count": sum(
                bool(item["metrics"].get("reactive_proposal_parse_fallback", False))
                for item in recent_records
            ),
        }
    oracle_records = [
        item
        for item in records
        if item.get("metrics", {}).get("reactive_policy") == "oracle_interval"
    ]
    if oracle_records:
        causal_overlap = sum(
            bool(item["metrics"].get("oracle_interval_has_causal_overlap"))
            for item in oracle_records
        )
        result["oracle_interval_ceiling"] = {
            "non_deployable_ground_truth_metadata": True,
            "num_samples": len(oracle_records),
            "causal_interval_overlap_count": causal_overlap,
            "causal_interval_overlap_rate": causal_overlap / len(oracle_records),
            "recent_only_fallback_count": len(oracle_records) - causal_overlap,
            "interval_extended_past_query_count": sum(
                bool(item["metrics"].get("oracle_interval_extended_past_query"))
                for item in oracle_records
            ),
            "mean_selected_oracle_frames": float(
                np.mean(
                    [
                        int(item["metrics"].get("oracle_selected_evidence_frames", 0))
                        for item in oracle_records
                    ]
                )
            ),
        }
    return result


def load_source_metrics(output_dir: Path, world_size: int) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for rank in range(world_size):
        path = output_dir / f"rank_{rank:02d}_memory.json"
        if not path.is_file():
            continue
        for item in load_json(path):
            merged[item["source_key"]] = item
    return list(merged.values())


def workload_summary(
    records: list[dict[str, Any]], source_metrics: list[dict[str, Any]]
) -> dict[str, Any]:
    ingest_seconds = sum(float(item.get("memory_build_seconds", 0.0)) for item in source_metrics)
    sampled_frames = sum(int(item.get("sampled_frames_ingested", 0)) for item in source_metrics)
    represented_seconds = sum(
        max(
            (
                float(record["metrics"].get("query_timestamp_seconds", 0.0))
                for record in records
                if record.get("video") == item.get("source_key")
            ),
            default=0.0,
        )
        for item in source_metrics
    )
    query_seconds = sum(
        float(item["metrics"].get("query_pipeline_seconds", 0.0)) for item in records
    )
    total_seconds = ingest_seconds + query_seconds
    peak_vector_bytes = max(
        (int(item.get("peak_persistent_tensor_bytes", 0)) for item in source_metrics),
        default=0,
    )
    capacities = {int(item.get("max_memory_vectors", 0)) for item in source_metrics}
    chunk_sizes = {int(item.get("memory_chunk_frames", 0)) for item in source_metrics}
    capacity_value = next(iter(capacities)) if len(capacities) == 1 else None
    chunk_value = next(iter(chunk_sizes)) if len(chunk_sizes) == 1 else None
    unbounded = capacity_value == -1
    return {
        "num_unique_sources": len(source_metrics),
        "source_sampled_frames_ingested_total": sampled_frames,
        "represented_video_seconds_total": represented_seconds,
        "memory_ingest_gpu_seconds_total": ingest_seconds,
        "memory_ingest_ms_per_sampled_frame": (
            1000.0 * ingest_seconds / sampled_frames if sampled_frames else None
        ),
        "memory_ingest_frames_per_second": (
            sampled_frames / ingest_seconds if ingest_seconds > 0 else None
        ),
        "memory_ingest_rtf": (
            ingest_seconds / represented_seconds if represented_seconds > 0 else None
        ),
        "query_pipeline_gpu_seconds_total": query_seconds,
        "workload_gpu_seconds_total_excluding_model_load": total_seconds,
        "workload_gpu_seconds_per_source_video_minute": (
            total_seconds / (represented_seconds / 60.0)
            if represented_seconds > 0
            else None
        ),
        "vector_memory_peak_bytes_per_source": peak_vector_bytes,
        "bounded_vector_memory_peak_bytes_per_source": (
            None if unbounded else peak_vector_bytes
        ),
        "memory_capacity_policy": "unbounded" if unbounded else "fifo_bounded",
        "max_memory_vectors": capacity_value if capacity_value is not None else sorted(capacities),
        "memory_chunk_frames": chunk_value if chunk_value is not None else sorted(chunk_sizes),
        "nominal_retained_sampled_frame_capacity": (
            capacity_value * chunk_value
            if capacity_value is not None
            and capacity_value > 0
            and chunk_value is not None
            else None
        ),
        "models_simultaneously_resident": True,
        "model_weight_load_time_included": False,
    }


def merge_rank_outputs(output_dir: Path, world_size: int, cfg: dict[str, Any]) -> None:
    merged: dict[str, dict[str, Any]] = {}
    for rank in range(world_size):
        path = output_dir / f"rank_{rank:02d}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Missing rank result file: {path}")
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    item = json.loads(line)
                    merged[item["_key"]] = item
    records = sorted(merged.values(), key=lambda item: item["_index"])
    grouped = {"backward": [], "realtime": [], "forward": []}
    for item in records:
        grouped[item["category"]].append(
            {
                "id": item["id"],
                "video": item["video"],
                "task": item["task"],
                "question": item["question"],
                "response": item["response"],
                "ground_truth": item["ground_truth"],
                "metrics": item["metrics"],
            }
        )
    (output_dir / "results.json").write_text(json.dumps(grouped, indent=2, ensure_ascii=False) + "\n")
    (output_dir / "summary.json").write_text(json.dumps(score_results(records), indent=2) + "\n")
    (output_dir / "efficiency_summary.json").write_text(
        json.dumps(efficiency_summary(records), indent=2) + "\n"
    )
    source_metrics = load_source_metrics(output_dir, world_size)
    if source_metrics:
        (output_dir / "workload_summary.json").write_text(
            json.dumps(workload_summary(records, source_metrics), indent=2) + "\n"
        )


def build_run_manifest(
    mode: str,
    cfg: dict[str, Any],
    configured_tasks: list[str],
    annotations: list[dict[str, Any]],
    world_size: int,
) -> dict[str, Any]:
    scoped_cfg = {
        key: value
        for key, value in cfg.items()
        if key != "output_root"
    }
    if str(cfg.get("reactive_policy", "retrieve_first")) == "retrieve_first":
        # Preserve the fingerprint of completed pre-branch baselines so they
        # remain resumable after this experimental branch was added.
        for key in (
            "reactive_policy",
            "reactive_retrieval_query",
            "recent_first_frames",
            "reactive_proposal_count",
            "reactive_proposal_aggregation",
            "reactive_rrf_k",
            "reactive_question_weight",
            "reactive_proposal_weight",
        ):
            scoped_cfg.pop(key, None)
    if str(cfg.get("reactive_policy", "retrieve_first")) != "recent_first_plan":
        # RRF knobs belong only to the new fused planner. Do not perturb the
        # fingerprints of completed recent_first question/proposal runs.
        for key in (
            "reactive_rrf_k",
            "reactive_question_weight",
            "reactive_proposal_weight",
        ):
            scoped_cfg.pop(key, None)
    annotation_scope = [
        {
            "id": item["id"],
            "task": item["task"],
            "video": item["video"],
            "realtime": item.get("realtime"),
            "question": item["question"],
            "options": item["options"],
            "gt": item["gt"],
        }
        for item in annotations
    ]
    annotation_digest = hashlib.sha256(
        json.dumps(annotation_scope, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    scope = {
        "mode": mode,
        "config": scoped_cfg,
        "tasks": configured_tasks,
        "annotation_digest": annotation_digest,
        "num_annotations": len(annotations),
        "world_size": int(world_size),
    }
    fingerprint = hashlib.sha256(
        json.dumps(scope, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    return {"run_scope_version": 1, "fingerprint": fingerprint, **scope}


def main() -> None:
    args = parse_args()
    rank, world_size, local_rank, device = distributed_context()
    cfg = load_json(args.config)
    if args.answer_model_path is not None:
        cfg["answer_model_path"] = str(args.answer_model_path)
    if args.answer_model_family is not None:
        cfg["answer_model_family"] = str(args.answer_model_family)
    if args.sample_fps is not None:
        cfg["sample_fps"] = float(args.sample_fps)
    if float(cfg["sample_fps"]) <= 0:
        raise ValueError(f"sample_fps must be positive, got {cfg['sample_fps']}")
    if args.memory_chunk_frames is not None:
        cfg["memory_chunk_frames"] = int(args.memory_chunk_frames)
    if args.max_memory_vectors is not None:
        cfg["max_memory_vectors"] = int(args.max_memory_vectors)
    if args.recent_pool_frames is not None:
        cfg["recent_pool_frames"] = int(args.recent_pool_frames)
    if args.history_pool_frames is not None:
        cfg["history_pool_frames"] = int(args.history_pool_frames)
    if args.history_temporal_nms is not None:
        cfg["history_temporal_nms"] = bool(args.history_temporal_nms)
    if args.answer_frame_budget is not None:
        cfg["answer_frame_budget"] = int(args.answer_frame_budget)
    if args.reactive_policy is not None:
        cfg["reactive_policy"] = str(args.reactive_policy)
    if args.reactive_retrieval_query is not None:
        cfg["reactive_retrieval_query"] = str(args.reactive_retrieval_query)
    if args.recent_first_frames is not None:
        cfg["recent_first_frames"] = int(args.recent_first_frames)
    if args.reactive_proposal_count is not None:
        cfg["reactive_proposal_count"] = int(args.reactive_proposal_count)
    if args.reactive_proposal_aggregation is not None:
        cfg["reactive_proposal_aggregation"] = str(
            args.reactive_proposal_aggregation
        )
    if args.reactive_rrf_k is not None:
        cfg["reactive_rrf_k"] = float(args.reactive_rrf_k)
    if args.reactive_question_weight is not None:
        cfg["reactive_question_weight"] = float(args.reactive_question_weight)
    if args.reactive_proposal_weight is not None:
        cfg["reactive_proposal_weight"] = float(args.reactive_proposal_weight)
    if args.token_reduction is not None:
        cfg["token_reduction_enabled"] = bool(args.token_reduction)
    if args.token_reduction_layer is not None:
        cfg["token_reduction_layer"] = int(args.token_reduction_layer)
    if args.token_reduction_budget is not None:
        cfg["token_reduction_budget"] = float(args.token_reduction_budget)
    if args.token_reduction_parallel is not None:
        cfg["token_reduction_parallel"] = bool(args.token_reduction_parallel)
    if args.token_reduction_selector is not None:
        cfg["token_reduction_selector"] = str(args.token_reduction_selector)
    if args.full_history_visual_token_budget is not None:
        cfg["full_history_visual_token_budget"] = int(args.full_history_visual_token_budget)

    cfg["token_reduction_enabled"] = bool(
        cfg.get("token_reduction_enabled", False)
    )
    cfg["token_reduction_layer"] = int(cfg.get("token_reduction_layer", 12))
    cfg["token_reduction_budget"] = float(cfg.get("token_reduction_budget", 0.1))
    cfg["token_reduction_parallel"] = bool(
        cfg.get("token_reduction_parallel", True)
    )
    cfg["reactive_policy"] = str(cfg.get("reactive_policy", "retrieve_first"))
    cfg["reactive_retrieval_query"] = str(
        cfg.get("reactive_retrieval_query", "question")
    )
    cfg["recent_first_frames"] = int(cfg.get("recent_first_frames", 4))
    cfg["reactive_proposal_count"] = int(cfg.get("reactive_proposal_count", 4))
    cfg["reactive_proposal_aggregation"] = str(
        cfg.get("reactive_proposal_aggregation", "max")
    )
    cfg["reactive_rrf_k"] = float(cfg.get("reactive_rrf_k", 60.0))
    cfg["reactive_question_weight"] = float(
        cfg.get("reactive_question_weight", 1.0)
    )
    cfg["reactive_proposal_weight"] = float(
        cfg.get("reactive_proposal_weight", 0.5)
    )
    if cfg["reactive_policy"] not in {
        "retrieve_first",
        "recent_first",
    }:
        raise ValueError(f"Unknown reactive_policy: {cfg['reactive_policy']}")
    if cfg["reactive_retrieval_query"] != "question":
        raise ValueError(
            f"Unknown reactive_retrieval_query: {cfg['reactive_retrieval_query']}"
        )
    if cfg["recent_first_frames"] <= 0 or cfg["reactive_proposal_count"] <= 0:
        raise ValueError("recent_first_frames and reactive_proposal_count must be positive")
    if cfg["reactive_proposal_aggregation"] not in {
        "max",
        "mean",
        "logsumexp",
        "question_anchor_rrf",
    }:
        raise ValueError(
            "reactive_proposal_aggregation must be max, mean, logsumexp, "
            "or question_anchor_rrf"
        )
    if cfg["reactive_policy"] == "recent_first_plan":
        cfg["reactive_proposal_aggregation"] = "question_anchor_rrf"
    if (
        cfg["reactive_rrf_k"] <= 0
        or cfg["reactive_question_weight"] <= 0
        or cfg["reactive_proposal_weight"] <= 0
    ):
        raise ValueError("RRF k and query weights must be positive")
    if cfg["token_reduction_layer"] == 0 or cfg["token_reduction_layer"] < -1:
        raise ValueError("token_reduction_layer must be -1 or a positive 1-based layer id")
    if cfg["token_reduction_budget"] <= 0:
        raise ValueError("token_reduction_budget must be positive")
    if args.mode == "full_history" and cfg["token_reduction_enabled"]:
        raise ValueError(
            "Streaming token reduction applies to selected independent images, "
            "not the full_history video baseline."
        )

    if args.mode != "full_history":
        chunk_frames = int(cfg["memory_chunk_frames"])
        if not 1 <= chunk_frames <= 16:
            raise ValueError(
                "Streaming MEMORY_CHUNK_FRAMES must be an integer in [1, 16], "
                f"got {chunk_frames}"
            )
        cfg["max_memory_vectors"] = validate_memory_capacity(
            cfg["max_memory_vectors"]
        )
        if int(cfg["answer_frame_budget"]) <= 0:
            raise ValueError("answer_frame_budget must be positive")
        if int(cfg["recent_pool_frames"]) < 0 or int(cfg["history_pool_frames"]) < 0:
            raise ValueError("recent_pool_frames and history_pool_frames must be non-negative")

    configured_tasks = list(cfg["tasks"])
    if args.tasks:
        requested = [item.strip() for item in args.tasks.split(",") if item.strip()]
        configured_tasks = [task for task in configured_tasks if task in requested]
    annotations = [item for item in load_json(cfg["annotation_path"]) if item["task"] in configured_tasks]
    if args.max_samples is not None:
        annotations = annotations[: int(args.max_samples)]
    annotations = [{**item, "_index": index} for index, item in enumerate(annotations)]

    recent_branch_suffix = ""
    if cfg["reactive_policy"] == "recent_first":
        recent_branch_suffix = (
            f"-rf{cfg['recent_first_frames']}"
            f"-{cfg['reactive_retrieval_query']}"
        )
        if cfg["reactive_retrieval_query"] == "proposal":
            recent_branch_suffix += (
                f"{cfg['reactive_proposal_count']}"
                f"-{cfg['reactive_proposal_aggregation']}"
            )
    elif cfg["reactive_policy"] == "recent_first_plan":
        recent_branch_suffix = (
            f"-rfplan{cfg['recent_first_frames']}"
            f"-p{cfg['reactive_proposal_count']}"
            f"-rrf{cfg['reactive_rrf_k']:g}"
            f"-qw{cfg['reactive_question_weight']:g}"
            f"-pw{cfg['reactive_proposal_weight']:g}"
        )
    elif cfg["reactive_policy"] == "abstention_aware":
        recent_branch_suffix = f"-abstainaware-rf{cfg['recent_first_frames']}"
    suffix = (
        f"fps{cfg['sample_fps']:g}-vtokens{cfg['full_history_visual_token_budget']}"
        if args.mode == "full_history"
        else (
            f"online-resident-fps{cfg['sample_fps']:g}"
            f"-chunk{cfg['memory_chunk_frames']}"
            f"-maxvec{cfg['max_memory_vectors']}"
            f"-nms{int(bool(cfg['history_temporal_nms']))}"
            f"-history{cfg['history_pool_frames']}"
            f"-recent{cfg['recent_pool_frames']}"
            f"-budget{cfg['answer_frame_budget']}"
            f"-tr{int(cfg['token_reduction_enabled'])}"
            f"-trlayer{cfg['token_reduction_layer']}"
            f"-trbudget{cfg['token_reduction_budget']:g}"
            f"-trparallel{int(cfg['token_reduction_parallel'])}"
            f"{recent_branch_suffix}"
        )
    )
    output_dir = Path(args.output_dir or Path(cfg["output_root"]) / f"{args.mode}-{suffix}")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "run_manifest.json"
    run_manifest = build_run_manifest(args.mode, cfg, configured_tasks, annotations, world_size)
    existing_manifest = None
    if manifest_path.is_file():
        try:
            existing_manifest = load_json(manifest_path)
        except (json.JSONDecodeError, OSError) as error:
            if not args.overwrite:
                raise RuntimeError(
                    f"Could not validate existing run manifest {manifest_path}; use --overwrite "
                    "or choose a new OUTPUT_DIR."
                ) from error
    elif any(path.stat().st_size > 0 for path in output_dir.glob("rank_*.jsonl")):
        if not args.overwrite:
            raise RuntimeError(
                f"{output_dir} contains rank results without a run manifest. Use --overwrite "
                "or choose a new OUTPUT_DIR; blindly resuming could mix experiment scopes."
            )
    if (
        existing_manifest is not None
        and existing_manifest.get("fingerprint") != run_manifest["fingerprint"]
        and not args.overwrite
    ):
        raise RuntimeError(
            f"Run scope mismatch in {output_dir}. Existing fingerprint "
            f"{existing_manifest.get('fingerprint')} != current {run_manifest['fingerprint']}. "
            "Use --overwrite or a distinct OUTPUT_DIR (recommended for smoke tests)."
        )

    rank_path = output_dir / f"rank_{rank:02d}.jsonl"
    done_path = output_dir / f"rank_{rank:02d}.done"
    run_id = os.environ.get("TORCHELASTIC_RUN_ID", "single-process")
    if args.overwrite and rank_path.exists():
        rank_path.unlink()
    rank_path.touch(exist_ok=True)
    if done_path.exists():
        done_path.unlink()
    done = load_done(rank_path)

    if rank == 0:
        temporary_manifest = manifest_path.with_suffix(f".tmp-{os.getpid()}.json")
        temporary_manifest.write_text(json.dumps(run_manifest, indent=2) + "\n")
        os.replace(temporary_manifest, manifest_path)
        snapshot = dict(cfg)
        snapshot.update(
            {
                "mode": args.mode,
                "world_size": world_size,
                "tasks_run": configured_tasks,
                "num_annotations": len(annotations),
                "protocol_note": (
                    "True chronological online replay. Each completed 1--16-frame chunk is encoded "
                    "once by Embedding-2B and appended to a bounded FIFO vector memory. At each query "
                    "timestamp, retrieval and 8B answering happen immediately using only retained "
                    "vectors plus the raw recent-frame buffer. Embedding-2B and Qwen3-VL-8B remain "
                    "simultaneously resident. No full-source index or final visual-token cache is built."
                ),
                    "latency_scope": (
                        "One-time model construction/load is reported separately. Stream-ingest time is "
                        "measured when each chunk arrives; per-query timings include query embedding, "
                        "retrieval, evidence decode, optional temporary 2B decoder-token scoring, "
                        "8B preprocessing and generation."
                    ),
            }
        )
        (output_dir / "inference_config.json").write_text(json.dumps(snapshot, indent=2) + "\n")

    if args.mode == "full_history":
        answer_model = AnswerModel(cfg, device)
        assigned = [item for item in annotations if int(item["_index"]) % world_size == rank]
        with rank_path.open("a", encoding="utf-8", buffering=1) as handle:
            for local_index, item in enumerate(assigned, start=1):
                category = category_for_task(item["task"])
                key = f"{category}:{item['id']}"
                if key in done:
                    continue
                video_path = str(Path(cfg["video_root"]) / f"{item['id']}.mp4")
                if not Path(video_path).is_file():
                    raise FileNotFoundError(video_path)
                sampler = VideoSampler(video_path, cfg)
                sample_started = time.perf_counter()
                response, answer_metrics = answer_model.answer(
                    video_path,
                    build_prompt(item),
                    fps=float(cfg["sample_fps"]),
                    max_frames=int(cfg["max_sample_frames"]),
                    visual_token_budget=int(cfg["full_history_visual_token_budget"]),
                )
                selected_positions = list(range(len(sampler.indices)))
                metrics = {
                    **answer_metrics,
                    "router_seconds": 0.0,
                    "sample_total_seconds": time.perf_counter() - sample_started,
                    "raw_video_frames": sampler.total_frames,
                    "raw_video_fps": sampler.raw_fps,
                    "raw_sampled_frames": len(sampler.indices),
                    "answer_frames": len(selected_positions),
                    "frame_keep_ratio": 1.0,
                    "selected_sample_positions": selected_positions,
                    "selected_raw_frame_indices": sampler.indices,
                    "selected_timestamps_seconds": sampler.timestamps(selected_positions),
                }
                record = {
                    "_key": key,
                    "_index": item["_index"],
                    "rank": rank,
                    "id": item["id"],
                    "video": item["video"],
                    "category": category,
                    "task": item["task"],
                    "question": item["question"],
                    "response": response,
                    "ground_truth": chr(65 + int(item["gt"])),
                    "metrics": metrics,
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                print(
                    f"[rank {rank}] {local_index}/{len(assigned)} {item['task']} id={item['id']} "
                    f"full-history response={response!r}",
                    flush=True,
                )
    else:
        rank_groups = group_annotations_by_source(annotations, world_size)[rank]
        pending_groups = [
            items
            for items in rank_groups
            if any(f"{category_for_task(item['task'])}:{item['id']}" not in done for item in items)
        ]
        source_metrics: dict[str, dict[str, Any]] = {}

        if pending_groups:
            embedding_load_started = time.perf_counter()
            encoder = SemanticMemoryEncoder(cfg, args.mode, device)
            torch.cuda.synchronize(device)
            embedding_model_load_seconds = time.perf_counter() - embedding_load_started
            answer_load_started = time.perf_counter()
            answer_model = AnswerModel(cfg, device)
            torch.cuda.synchronize(device)
            answer_model_load_seconds = time.perf_counter() - answer_load_started
            resident_allocated_bytes = int(torch.cuda.memory_allocated(device))
            resident_reserved_bytes = int(torch.cuda.memory_reserved(device))
            (output_dir / f"rank_{rank:02d}_runtime.json").write_text(
                json.dumps(
                    {
                        "embedding_model_load_seconds": embedding_model_load_seconds,
                        "answer_model_load_seconds": answer_model_load_seconds,
                        "models_simultaneously_resident": True,
                        "resident_models_allocated_bytes": resident_allocated_bytes,
                        "resident_models_reserved_bytes": resident_reserved_bytes,
                        "execution_protocol": memory_execution_protocol(
                            cfg["max_memory_vectors"]
                        ),
                        "latency_scope": "warm_resident_model_load_excluded_from_per_query_latency",
                    },
                    indent=2,
                )
                + "\n"
            )
            pending_total = sum(
                f"{category_for_task(item['task'])}:{item['id']}" not in done
                for items in pending_groups
                for item in items
            )
            completed = 0
            with rank_path.open("a", encoding="utf-8", buffering=1) as handle:
                for group_index, items in enumerate(pending_groups, start=1):
                    try:
                        source_key = str(items[0]["video"])
                        source_path = source_prefix_path(items, cfg["video_root"])
                        sampler = VideoSampler(str(source_path), cfg, causal_fixed_fps=True)
                        memory = OnlineVectorMemory(encoder, cfg)
                        next_position = 0
                        source_started = time.perf_counter()
                        for item in items:
                            category = category_for_task(item["task"])
                            key = f"{category}:{item['id']}"
                            query_timestamp = float(item["realtime"])
                            next_position, ingest_metrics = ingest_complete_chunks_until(
                                memory,
                                encoder,
                                sampler,
                                next_position,
                                query_timestamp,
                            )
                            if key in done:
                                continue
                            index = memory.as_index()
                            response, selected_positions, query_metrics = execute_reactive_query(
                                item=item,
                                cfg=cfg,
                                encoder=encoder,
                                answer_model=answer_model,
                                index=index,
                                sampler=sampler,
                                query_timestamp=query_timestamp,
                                answer_prompt=build_prompt(item),
                                recent_gate_prompt=build_recent_first_gate_prompt(item),
                                planner_prompt=build_recent_first_plan_prompt(
                                    item,
                                    int(cfg["reactive_proposal_count"]),
                                    query_context=item.get("router_query", item["question"]),
                                ),
                                proposal_prompt=build_reactive_proposal_prompt(
                                    item,
                                    int(cfg["reactive_proposal_count"]),
                                    query_context=item.get("router_query", item["question"]),
                                ),
                                router_query=item["question"],
                                score_device=device,
                            )
                            timestamps = sampler.timestamps(selected_positions)
                            if any(value > query_timestamp + 1e-6 for value in timestamps):
                                raise AssertionError(
                                    f"Causal violation for {key}: query={query_timestamp}, "
                                    f"selected={timestamps}"
                                )
                            available_positions = sampler.positions_available_at(query_timestamp)
                            metrics = {
                                **query_metrics,
                                **ingest_metrics,
                                "execution_protocol": memory_execution_protocol(
                                    cfg["max_memory_vectors"]
                                ),
                                "models_simultaneously_resident": True,
                                "resident_models_allocated_bytes": resident_allocated_bytes,
                                "resident_models_reserved_bytes": resident_reserved_bytes,
                                "memory_readout_name": memory.readout_name,
                                "memory_representation_kind": "single_vector",
                                "memory_decoder_layers_executed_per_chunk": memory.readout_layer,
                                "memory_post_final_norm": args.mode == "full_embedding",
                                "memory_build_peak_gpu_memory_bytes": memory.peak_gpu_memory_bytes,
                                "online_event_total_seconds": float(
                                    ingest_metrics["stream_ingest_seconds_since_previous_query"]
                                )
                                + float(query_metrics["query_pipeline_seconds"]),
                                "sample_total_seconds": float(
                                    ingest_metrics["stream_ingest_seconds_since_previous_query"]
                                )
                                + float(query_metrics["query_pipeline_seconds"]),
                                "query_timestamp_seconds": query_timestamp,
                                "raw_video_frames": sampler.total_frames,
                                "raw_video_fps": sampler.raw_fps,
                                "raw_sampled_frames": len(available_positions),
                                "answer_frames": len(selected_positions),
                                "frame_keep_ratio": len(selected_positions) / max(1, len(available_positions)),
                                "selected_sample_positions": selected_positions,
                                "selected_raw_frame_indices": [
                                    sampler.indices[position] for position in selected_positions
                                ],
                                "selected_timestamps_seconds": timestamps,
                            }
                            record = {
                                "_key": key,
                                "_index": item["_index"],
                                "rank": rank,
                                "id": item["id"],
                                "video": item["video"],
                                "category": category,
                                "task": item["task"],
                                "question": item["question"],
                                "response": response,
                                "ground_truth": chr(65 + int(item["gt"])),
                                "metrics": metrics,
                            }
                            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                            completed += 1
                            print(
                                f"[rank {rank}] online {completed}/{pending_total} "
                                f"{item['task']} id={item['id']} "
                                f"vectors={len(memory.vectors)}/{memory.capacity_display} "
                                f"H/R={query_metrics['answer_history_frames']}/"
                                f"{query_metrics['answer_recent_frames']} response={response!r}",
                                flush=True,
                            )

                        source_metrics[source_key] = {
                            "source_key": source_key,
                            "source_prefix_clip": str(source_path),
                            "num_questions": len(items),
                            "execution_protocol": (
                                f"chronological_online_{memory.capacity_policy}_vector_memory"
                            ),
                            "memory_build_seconds": memory.total_ingest_seconds,
                            "source_online_wall_seconds": time.perf_counter() - source_started,
                            "memory_build_peak_gpu_memory_bytes": memory.peak_gpu_memory_bytes,
                            "cache_file_bytes": 0,
                            "num_chunks_generated": memory.total_vectors_generated,
                            "num_vectors_retained": len(memory.vectors),
                            "num_vectors_evicted": memory.total_vectors_evicted,
                            "max_memory_vectors": memory.capacity,
                            "memory_capacity_policy": memory.capacity_policy,
                            "memory_chunk_frames": memory.chunk_frames,
                            "sampled_frames_ingested": memory.total_frames_ingested,
                            "readout_name": memory.readout_name,
                            "representation_kind": "single_vector",
                            "post_final_norm": args.mode == "full_embedding",
                            "stored_representation_count": len(memory.vectors),
                            "stored_visual_token_count": 0,
                            "stored_tensor_bytes": memory.persistent_bytes,
                            "peak_persistent_tensor_bytes": memory.peak_persistent_bytes,
                            "decoder_layers_executed_per_chunk": memory.readout_layer,
                        }
                        (output_dir / f"rank_{rank:02d}_memory.json").write_text(
                            json.dumps(list(source_metrics.values()), indent=2) + "\n"
                        )
                        print(
                            f"[rank {rank}] completed online source "
                            f"{group_index}/{len(pending_groups)} vectors_generated="
                            f"{memory.total_vectors_generated} retained={len(memory.vectors)} "
                            f"evicted={memory.total_vectors_evicted} {source_key}",
                            flush=True,
                        )
                    except Exception as source_exc:
                        # One corrupt/undecodable video (e.g. sample_332_real.mp4,
                        # frame 7522 unreadable by both decord and opencv) used to
                        # take the whole distributed rank down mid-run, losing every
                        # already-completed question in the process -- this crashed
                        # a 1213/2500 run. Skip the rest of this one source's
                        # questions (they stay out of `done` and can be retried in a
                        # later run) and continue with the next video. Read the
                        # source name off `items` directly rather than the
                        # loop-local `source_key`, which may not exist yet if the
                        # failure happened before it was assigned.
                        print(
                            f"[rank {rank}] SKIPPING source {group_index}/{len(pending_groups)} "
                            f"{items[0].get('video', '?')}: "
                            f"{type(source_exc).__name__}: {source_exc}",
                            flush=True,
                        )
                        continue

    done_path.write_text(
        json.dumps({"run_id": run_id, "rank": rank, "records": len(load_done(rank_path))}) + "\n"
    )
    if rank == 0:
        marker_paths = [output_dir / f"rank_{value:02d}.done" for value in range(world_size)]

        def marker_is_current(path: Path) -> bool:
            if not path.exists():
                return False
            try:
                return load_json(path).get("run_id") == run_id
            except (json.JSONDecodeError, OSError):
                return False

        while not all(marker_is_current(path) for path in marker_paths):
            missing = [path.stem for path in marker_paths if not marker_is_current(path)]
            print(f"Rank 0 waiting for workers to finish: {missing}", flush=True)
            time.sleep(30)
        merge_rank_outputs(output_dir, world_size, cfg)
        print(f"Completed {args.mode}; results: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
