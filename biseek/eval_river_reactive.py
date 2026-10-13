#!/usr/bin/env python3
"""Evaluate online vector-memory BiSeek streaming on RIVER reactive tasks.

RIVER provides complete source videos.  Unlike OVO-Bench's pre-truncated
prefix clips, this runner must enforce the question timestamp explicitly.
Every source is replayed chronologically on a causal fixed-FPS grid.  Completed
chunks enter an online vector memory exactly once; each question is retrieved
and answered immediately at ``question_time`` while both the 2B encoder and 8B
answer model remain resident.  Positive capacities use FIFO eviction and
``max_memory_vectors=-1`` retains all causally observed vectors.

The supported representations intentionally match the current OVO runner:

* ``full_embedding``: one official 2048-D final embedding per video chunk;
* ``decoder12``: one layer-12 last-valid-position vector per video chunk with
  genuine early exit after decoder block 12.

All modes combine four retrieved historical frames with four recent frames by
default.  The selected raw frames are then reread and passed to Qwen3-VL-8B as
independent images.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from eval_ovo_streaming import (
    AnswerModel,
    OnlineVectorMemory,
    SemanticMemoryEncoder,
    VideoSampler,
    efficiency_summary,
    execute_reactive_query,
    ingest_complete_chunks_until,
    load_done,
    load_json,
    memory_capacity_policy,
    memory_execution_protocol,
    nominal_retained_frame_capacity,
    parse_bool,
    select_semantic_memory_frames,
    validate_memory_capacity,
)


REACTIVE_TASKS = [
    "Live-Perception",
    "Retro-Memory@short",
    "Retro-Memory@medium",
    "Retro-Memory@long",
    "Retro-Memory@very_long",
]
RETRO_TASKS = [task for task in REACTIVE_TASKS if task.startswith("Retro-Memory@")]

RIVER_PROMPT_TEMPLATE = """\
Question: {question}
Options:
{options}

Respond only with the letter corresponding to your chosen option (for example, A, B, C, or D).
Do not include any additional text or explanation in your response.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--mode",
        choices=["full_embedding", "decoder12"],
        required=True,
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--tasks", default="")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--sample-fps", type=float, default=None)
    parser.add_argument("--answer-model-path", default=None)
    parser.add_argument(
        "--answer-model-family",
        choices=("qwen3vl", "qwen2_5vl", "qwen2vl"),
        default=None,
    )
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
    )
    parser.add_argument(
        "--reactive-retrieval-query",
        choices=["question"],
        default=None,
    )
    parser.add_argument("--recent-first-frames", type=int, default=None)
    parser.add_argument("--oracle-evidence-frames", type=int, default=None)
    parser.add_argument("--oracle-recent-frames", type=int, default=None)
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
        "--verify-only",
        action="store_true",
        help="Validate annotations, videos, timestamps and sharding without CUDA/model loading.",
    )
    parser.add_argument(
        "--verify-decode-samples",
        type=int,
        default=0,
        help="During --verify-only, decode three causal frames from this many source videos.",
    )
    return parser.parse_args()


def apply_overrides(cfg: dict[str, Any], args: argparse.Namespace) -> None:
    mapping = {
        "sample_fps": args.sample_fps,
        "answer_model_path": args.answer_model_path,
        "answer_model_family": args.answer_model_family,
        "memory_chunk_frames": args.memory_chunk_frames,
        "max_memory_vectors": args.max_memory_vectors,
        "recent_pool_frames": args.recent_pool_frames,
        "history_pool_frames": args.history_pool_frames,
        "history_temporal_nms": args.history_temporal_nms,
        "answer_frame_budget": args.answer_frame_budget,
        "reactive_policy": args.reactive_policy,
        "reactive_retrieval_query": args.reactive_retrieval_query,
        "recent_first_frames": args.recent_first_frames,
        "oracle_evidence_frames": args.oracle_evidence_frames,
        "oracle_recent_frames": args.oracle_recent_frames,
        "reactive_proposal_count": args.reactive_proposal_count,
        "reactive_proposal_aggregation": args.reactive_proposal_aggregation,
        "reactive_rrf_k": args.reactive_rrf_k,
        "reactive_question_weight": args.reactive_question_weight,
        "reactive_proposal_weight": args.reactive_proposal_weight,
        "token_reduction_enabled": args.token_reduction,
        "token_reduction_layer": args.token_reduction_layer,
        "token_reduction_budget": args.token_reduction_budget,
        "token_reduction_parallel": args.token_reduction_parallel,
    }
    for key, value in mapping.items():
        if value is not None:
            cfg[key] = value
    if float(cfg["sample_fps"]) <= 0:
        raise ValueError("sample_fps must be positive")
    cfg["memory_chunk_frames"] = int(cfg["memory_chunk_frames"])
    if not 1 <= cfg["memory_chunk_frames"] <= 16:
        raise ValueError(
            "Streaming MEMORY_CHUNK_FRAMES must be an integer in [1, 16], "
            f"got {cfg['memory_chunk_frames']}"
        )
    cfg["max_memory_vectors"] = validate_memory_capacity(
        cfg["max_memory_vectors"]
    )
    for key in (
        "recent_pool_frames",
        "history_pool_frames",
        "answer_frame_budget",
    ):
        cfg[key] = int(cfg[key])
        if cfg[key] <= 0:
            raise ValueError(f"{key} must be positive, got {cfg[key]}")
    if cfg["history_pool_frames"] + cfg["recent_pool_frames"] < cfg["answer_frame_budget"]:
        raise ValueError(
            "history_pool_frames + recent_pool_frames must cover answer_frame_budget"
        )
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
    cfg["oracle_evidence_frames"] = int(cfg.get("oracle_evidence_frames", 4))
    cfg["oracle_recent_frames"] = int(cfg.get("oracle_recent_frames", 4))
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
    if cfg["oracle_evidence_frames"] <= 0 or cfg["oracle_recent_frames"] < 0:
        raise ValueError(
            "oracle evidence frames must be positive and oracle recent frames non-negative"
        )
    if (
        cfg["reactive_policy"] == "oracle_interval"
        and cfg["oracle_evidence_frames"] + cfg["oracle_recent_frames"]
        > cfg["answer_frame_budget"]
    ):
        raise ValueError(
            "oracle evidence + recent frames cannot exceed answer_frame_budget"
        )
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


def category_for_task(task: str) -> str:
    if task == "Live-Perception":
        return "live_perception"
    if task in RETRO_TASKS:
        return "retro_memory"
    raise ValueError(f"Unsupported RIVER reactive task: {task}")


def build_prompt(item: dict[str, Any]) -> str:
    return RIVER_PROMPT_TEMPLATE.format(
        question=item["question"], options="\n".join(item["choices"])
    )


def _reactive_options(item: dict[str, Any]) -> str:
    return "\n".join(
        str(value)
        if re.match(r"^[A-E][.):]\s*", str(value), flags=re.IGNORECASE)
        else f"{chr(65 + index)}. {str(value).strip()}"
        for index, value in enumerate(item["choices"])
    )


def build_recent_first_gate_prompt(item: dict[str, Any]) -> str:
    history = ""
    if item.get("conversation_history"):
        history = (
            "Corrected prior interactions for reference:\n"
            + "\n".join(
                f"Q: {entry['question']} A: {entry['answer']}"
                for entry in item["conversation_history"]
            )
            + "\n\n"
        )
    return f"""\
Look only at the recent images above.

{history}Question: {item['question']}
Options:
{_reactive_options(item)}

If the images contain enough visual evidence, output exactly `ANSWER: X`,
where X is the option letter. Otherwise output exactly `RETRIEVE`.
Do not explain."""


def build_reactive_proposal_prompt(
    item: dict[str, Any], proposal_count: int, *, query_context: str | None = None
) -> str:
    return f"""\
Question context: {query_context or item['question']}
Options:
{_reactive_options(item)}

Predict {proposal_count} short descriptions of earlier video frames that
would visibly contain the evidence needed to answer this question.
Describe only what the image should show. Do not answer the question.
Output one description per line and nothing else."""


def build_recent_first_plan_prompt(
    item: dict[str, Any], proposal_count: int, *, query_context: str | None = None
) -> str:
    history = ""
    if item.get("conversation_history"):
        history = (
            "Corrected prior interactions for reference:\n"
            + "\n".join(
                f"Q: {entry['question']} A: {entry['answer']}"
                for entry in item["conversation_history"]
            )
            + "\n\n"
        )
    return f"""\
Look only at the recent images above.

{history}Question context: {query_context or item['question']}
Options:
{_reactive_options(item)}

If the recent images contain enough visual evidence, output exactly `ANSWER: X`,
where X is the option letter.

Otherwise output exactly `RETRIEVE: ["cue", ...]` with one to
{proposal_count} short descriptions of missing earlier visual evidence. Each cue
must describe one concrete frame or short event that would help answer the exact
question. Preserve required people, objects, numbers, attributes, relations, and
temporal order. Do not guess the answer, invent an option-specific fact, predict a
story, or copy incidental details from the recent images. Use JSON strings and do
not explain."""


def extract_answer_letter(response: str | None) -> str | None:
    """Extract one standalone multiple-choice letter without substring scoring."""
    value = str(response or "").strip().upper()
    if not value:
        return None
    leading = re.match(r"^[\s\(\[]*([A-E])(?:[\s\)\].,:;]|$)", value)
    if leading:
        return leading.group(1)
    labelled = re.search(r"(?:ANSWER|OPTION|CHOICE)\s*(?:IS|:)?\s*[\(\[]?([A-E])\b", value)
    if labelled:
        return labelled.group(1)
    standalone = re.search(r"(?<![A-Z])([A-E])(?![A-Z])", value)
    return standalone.group(1) if standalone else None


def source_path_for_record(raw: dict[str, Any], cfg: dict[str, Any]) -> Path:
    source = str(raw["video_source"])
    if source not in cfg["video_roots"]:
        raise KeyError(f"No video root configured for RIVER source {source!r}")
    return Path(cfg["video_roots"][source]) / str(raw["video_path"])


def normalize_annotations(
    cfg: dict[str, Any], configured_tasks: list[str]
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for annotation_name, annotation_path in cfg["annotation_paths"].items():
        for raw in load_json(annotation_path):
            task = str(raw["question_type"])
            if task not in configured_tasks:
                continue
            gt = str(raw["correct_answer"]).strip().upper()
            choices = [str(value).strip() for value in raw["choices"]]
            if gt not in {chr(65 + index) for index in range(len(choices))}:
                raise ValueError(
                    f"Invalid answer {gt!r} for {len(choices)} choices: {raw['question_id']}"
                )
            source_key = f"{raw['video_source']}:{raw['video_id']}"
            item_key = f"{task}:{raw['question_id']}"
            reference = [float(value) for value in raw.get("time_reference", [])]
            records.append(
                {
                    "_key": item_key,
                    "id": item_key,
                    "question_id": str(raw["question_id"]),
                    "video": source_key,
                    "video_source": str(raw["video_source"]),
                    "video_id": str(raw["video_id"]),
                    "video_path": str(source_path_for_record(raw, cfg)),
                    "duration_sec": float(raw["duration_sec"]),
                    "annotation_fps": float(raw["fps"]),
                    "category": category_for_task(task),
                    "task": task,
                    "question": str(raw["question"]),
                    "choices": choices,
                    "ground_truth": gt,
                    "realtime": float(raw["question_time"]),
                    "time_reference": reference,
                    "annotation_file": str(annotation_name),
                }
            )
    keys = [item["_key"] for item in records]
    duplicates = [key for key, count in Counter(keys).items() if count > 1]
    if duplicates:
        raise ValueError(f"Duplicate normalized RIVER record keys: {duplicates[:5]}")
    return records


SUBSET_BY_HASH = os.environ.get("SUBSET_BY_HASH", "").lower() in {"1", "true", "yes"}


def limit_annotations_balanced(
    annotations: list[dict[str, Any]], max_samples: int | None
) -> list[dict[str, Any]]:
    if max_samples is None or max_samples >= len(annotations):
        return annotations
    if max_samples <= 0:
        return []
    per_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in annotations:
        per_task[item["task"]].append(item)
    for values in per_task.values():
        if SUBSET_BY_HASH:
            # Ordering by timestamp takes the earliest questions of each task,
            # which is a prefix of the benchmark in disguise: the Dispider traces
            # were selected that way and the resulting subset turned out to be
            # measurably easier than the whole set. Hashing the key spreads the
            # choice across the task instead, and is still deterministic.
            values.sort(key=lambda item: hashlib.sha1(str(item["_key"]).encode()).hexdigest())
        else:
            values.sort(key=lambda item: (float(item["realtime"]), item["_key"]))
    selected: list[dict[str, Any]] = []
    offset = 0
    while len(selected) < max_samples:
        progressed = False
        for task in REACTIVE_TASKS:
            values = per_task.get(task, [])
            if offset < len(values):
                selected.append(values[offset])
                progressed = True
                if len(selected) >= max_samples:
                    break
        if not progressed:
            break
        offset += 1
    return selected


def group_annotations_by_source(
    annotations: list[dict[str, Any]], world_size: int
) -> list[list[list[dict[str, Any]]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in annotations:
        grouped[item["video"]].append(item)
    groups = [
        sorted(values, key=lambda item: (float(item["realtime"]), item["_key"]))
        for values in grouped.values()
    ]
    bins: list[list[list[dict[str, Any]]]] = [[] for _ in range(world_size)]
    loads = [0.0 for _ in range(world_size)]
    groups.sort(
        key=lambda values: (
            -(max(float(item["realtime"]) for item in values) + 16.0 * len(values)),
            values[0]["video"],
        )
    )
    for values in groups:
        rank = int(np.argmin(np.asarray(loads)))
        bins[rank].append(values)
        loads[rank] += max(float(item["realtime"]) for item in values) + 16.0 * len(values)
    return bins


def trim_sampler_to_timestamp(sampler: VideoSampler, timestamp: float) -> VideoSampler:
    positions = sampler.positions_available_at(timestamp)
    if not positions:
        raise RuntimeError(
            f"No fixed-FPS frame from {sampler.path} is available at t={timestamp:.3f}s"
        )
    count = positions[-1] + 1
    sampler.indices = sampler.indices[:count]
    sampler.sample_timestamps = sampler.sample_timestamps[:count]
    return sampler


def sampler_for_group(items: list[dict[str, Any]], cfg: dict[str, Any]) -> VideoSampler:
    paths = {item["video_path"] for item in items}
    if len(paths) != 1:
        raise ValueError(f"One source key resolves to multiple paths: {sorted(paths)}")
    path = next(iter(paths))
    if not Path(path).is_file():
        raise FileNotFoundError(path)
    sampler = VideoSampler(path, cfg, causal_fixed_fps=True)
    required_timestamp = max(float(item["realtime"]) for item in items)
    return trim_sampler_to_timestamp(sampler, required_timestamp)


def reference_metrics(item: dict[str, Any], selected_timestamps: list[float], fps: float) -> dict[str, Any]:
    reference = item.get("time_reference") or []
    if len(reference) != 2:
        return {
            "reference_interval_seconds": reference,
            "reference_interval_valid": False,
            "selected_reference_hit": None,
            "selected_reference_frame_count": None,
        }
    start, end = sorted(float(value) for value in reference)
    valid = start >= 0.0 and end <= float(item["realtime"]) + 1e-6
    if not valid:
        return {
            "reference_interval_seconds": [start, end],
            "reference_interval_valid": False,
            "selected_reference_hit": None,
            "selected_reference_frame_count": None,
        }
    tolerance = 0.5 / max(float(fps), 1e-6) + 1e-6
    hits = [value for value in selected_timestamps if start - tolerance <= value <= end + tolerance]
    return {
        "reference_interval_seconds": [start, end],
        "reference_interval_valid": True,
        "selected_reference_hit": bool(hits),
        "selected_reference_frame_count": len(hits),
    }


def verify_dataset(
    annotations: list[dict[str, Any]], cfg: dict[str, Any], decode_samples: int
) -> dict[str, Any]:
    from decord import VideoReader, cpu

    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in annotations:
        by_source[item["video"]].append(item)
    missing: list[str] = []
    empty_videos: list[str] = []
    question_after_video: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    total_sampled_frames = 0
    sorted_sources = sorted(by_source.items())
    for source_index, (source_key, items) in enumerate(sorted_sources, start=1):
        path = Path(items[0]["video_path"])
        if not path.is_file():
            missing.append(str(path))
            continue
        reader = VideoReader(str(path), ctx=cpu(0), num_threads=1)
        raw_frames = len(reader)
        raw_fps = float(reader.get_avg_fps())
        if raw_frames <= 0:
            empty_videos.append(str(path))
            continue
        # Container duration conventionally includes the display interval of
        # the final frame, whereas its timestamp is ``(N-1)/fps``.  RIVER
        # question times are rounded to centiseconds and frequently point at
        # that inclusive endpoint, so verify against ``N/fps`` with half a
        # sampling interval of tolerance.  Selection remains strictly bounded
        # by real decoded-frame timestamps later in the runner.
        actual_duration = raw_frames / max(raw_fps, 1e-6)
        required = max(float(item["realtime"]) for item in items)
        endpoint_tolerance = 0.5 / max(float(cfg["sample_fps"]), 1e-6) + 1e-3
        if required > actual_duration + endpoint_tolerance:
            question_after_video.append(
                {
                    "source": source_key,
                    "required_timestamp": required,
                    "actual_duration": actual_duration,
                }
            )
        sampled = min(raw_frames, int(np.floor(required * float(cfg["sample_fps"]))) + 1)
        total_sampled_frames += sampled
        source_rows.append(
            {
                "source": source_key,
                "path": str(path),
                "num_questions": len(items),
                "raw_frames": raw_frames,
                "raw_fps": raw_fps,
                "actual_duration_seconds": actual_duration,
                "max_reactive_question_seconds": required,
                "estimated_fixed_fps_frames": sampled,
            }
        )
        if source_index % 10 == 0 or source_index == len(sorted_sources):
            print(
                f"Verified RIVER video metadata {source_index}/{len(sorted_sources)}",
                flush=True,
            )

    decoded_sources: list[str] = []
    if decode_samples > 0 and source_rows:
        selected = np.linspace(
            0, len(source_rows) - 1, min(decode_samples, len(source_rows))
        ).round().astype(np.int64)
        for row_index in np.unique(selected):
            row = source_rows[int(row_index)]
            reader = VideoReader(row["path"], ctx=cpu(0), num_threads=1)
            last_allowed = min(
                len(reader) - 1,
                int(round(row["max_reactive_question_seconds"] * row["raw_fps"])),
            )
            indices = np.linspace(0, last_allowed, 3).round().astype(np.int64).tolist()
            reader.get_batch(indices).asnumpy()
            decoded_sources.append(row["source"])

    invalid_references = sum(
        1
        for item in annotations
        if len(item["time_reference"]) != 2
        or max(float(value) for value in item["time_reference"])
        > float(item["realtime"]) + 1e-6
    )
    report = {
        "num_reactive_records": len(annotations),
        "num_unique_source_videos": len(by_source),
        "task_counts": dict(Counter(item["task"] for item in annotations)),
        "source_counts": dict(Counter(item["video_source"] for item in annotations)),
        "sample_fps": float(cfg["sample_fps"]),
        "memory_chunk_frames": int(cfg["memory_chunk_frames"]),
        "max_memory_vectors": int(cfg["max_memory_vectors"]),
        "history_temporal_nms": bool(cfg["history_temporal_nms"]),
        "memory_capacity_policy": memory_capacity_policy(
            cfg["max_memory_vectors"]
        ),
        "nominal_retained_sampled_frame_capacity": nominal_retained_frame_capacity(
            cfg["max_memory_vectors"], cfg["memory_chunk_frames"]
        ),
        "estimated_fixed_fps_frames_to_latest_queries": total_sampled_frames,
        "missing_videos": missing,
        "empty_videos": empty_videos,
        "question_after_video": question_after_video,
        "invalid_or_noncausal_reference_intervals": invalid_references,
        "decoded_source_samples": decoded_sources,
        "sources": source_rows,
    }
    if missing or empty_videos or question_after_video:
        raise RuntimeError(json.dumps(report, indent=2))
    return report


def distributed_context() -> tuple[int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not torch.cuda.is_available():
        raise RuntimeError("Semantic-memory benchmark evaluation requires CUDA GPUs")
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank, torch.device("cuda", local_rank)


def build_run_manifest(
    mode: str,
    cfg: dict[str, Any],
    tasks: list[str],
    annotations: list[dict[str, Any]],
    world_size: int,
) -> dict[str, Any]:
    scoped_cfg = {
        key: value
        for key, value in cfg.items()
        if key != "output_root"
    }
    if str(cfg.get("reactive_policy", "retrieve_first")) == "retrieve_first":
        # Keep existing retrieve-first run manifests resumable.
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
        # The new RRF fields must not invalidate old recent_first manifests.
        for key in (
            "reactive_rrf_k",
            "reactive_question_weight",
            "reactive_proposal_weight",
        ):
            scoped_cfg.pop(key, None)
    if str(cfg.get("reactive_policy", "retrieve_first")) != "oracle_interval":
        for key in ("oracle_evidence_frames", "oracle_recent_frames"):
            scoped_cfg.pop(key, None)
    annotation_scope = [
        {
            "key": item["_key"],
            "video": item["video"],
            "question_time": item["realtime"],
            "question": item["question"],
            "choices": item["choices"],
            "ground_truth": item["ground_truth"],
        }
        for item in annotations
    ]
    if str(cfg.get("reactive_policy", "retrieve_first")) == "oracle_interval":
        for scope_item, annotation in zip(annotation_scope, annotations):
            scope_item["time_reference"] = annotation.get("time_reference", [])
    annotation_digest = hashlib.sha256(
        json.dumps(annotation_scope, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    scope = {
        "benchmark": str(cfg.get("benchmark", "RIVER")),
        "track": str(cfg.get("track", "reactive")),
        "mode": mode,
        "config": scoped_cfg,
        "tasks": tasks,
        "annotation_digest": annotation_digest,
        "num_annotations": len(annotations),
        "world_size": world_size,
    }
    fingerprint = hashlib.sha256(
        json.dumps(scope, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    return {"run_scope_version": 1, "fingerprint": fingerprint, **scope}


def score_results(records: list[dict[str, Any]]) -> dict[str, Any]:
    values: dict[str, list[int]] = defaultdict(list)
    unresolved = 0
    for record in records:
        prediction = extract_answer_letter(record.get("response"))
        if prediction is None:
            unresolved += 1
        values[record["task"]].append(int(prediction == record["ground_truth"]))
    task_scores = {
        task: {
            "accuracy": float(sum(items) / len(items)),
            "percentage": float(100.0 * sum(items) / len(items)),
            "num_evaluations": len(items),
        }
        for task, items in sorted(values.items())
    }
    retro_values = [task_scores[task]["accuracy"] for task in RETRO_TASKS if task in task_scores]
    live_value = (
        task_scores["Live-Perception"]["accuracy"]
        if "Live-Perception" in task_scores
        else None
    )
    retro_macro = float(np.mean(retro_values)) if retro_values else None
    reactive_values = [value for value in (live_value, retro_macro) if value is not None]
    reactive_overall = float(np.mean(reactive_values)) if reactive_values else None
    all_correct = [value for items in values.values() for value in items]
    recent_breakdown: dict[str, Any] = {}
    for decision in ("direct_answer", "retrieve"):
        subset = [
            record for record in records
            if record.get("metrics", {}).get("recent_first_decision") == decision
        ]
        if subset:
            correct = [
                int(extract_answer_letter(record.get("response")) == record["ground_truth"])
                for record in subset
            ]
            recent_breakdown[decision] = {
                "accuracy": float(np.mean(correct)),
                "percentage": float(100.0 * np.mean(correct)),
                "num_evaluations": len(correct),
            }
    return {
        "overall": {
            "accuracy": reactive_overall,
            "percentage": None if reactive_overall is None else 100.0 * reactive_overall,
            "definition": "macro_average_of_retro_memory_and_live_perception",
            "num_categories": len(reactive_values),
            "scope": "RIVER reactive track only; Pro-Response is not included",
        },
        "task_scores": task_scores,
        "category_scores": {
            "live_perception": {
                "accuracy": live_value,
                "percentage": None if live_value is None else 100.0 * live_value,
            },
            "retro_memory_macro_over_four_intervals": {
                "accuracy": retro_macro,
                "percentage": None if retro_macro is None else 100.0 * retro_macro,
                "num_intervals": len(retro_values),
            },
        },
        "reactive_macro_live_and_retro": {
            "accuracy": reactive_overall,
            "percentage": None if reactive_overall is None else 100.0 * reactive_overall,
            "alias_of": "overall",
        },
        "five_task_macro": {
            "accuracy": float(np.mean([value["accuracy"] for value in task_scores.values()])),
            "percentage": float(
                100.0 * np.mean([value["accuracy"] for value in task_scores.values()])
            ),
        },
        "micro_accuracy": {
            "accuracy": float(np.mean(all_correct)) if all_correct else None,
            "percentage": float(100.0 * np.mean(all_correct)) if all_correct else None,
            "num_evaluations": len(all_correct),
        },
        "unparsed_responses": unresolved,
        "recent_first_breakdown": recent_breakdown,
    }


def causal_audit(records: list[dict[str, Any]]) -> dict[str, Any]:
    violations: list[dict[str, Any]] = []
    margins: list[float] = []
    for record in records:
        query_time = float(record["metrics"]["query_timestamp_seconds"])
        selected = [float(value) for value in record["metrics"]["selected_timestamps_seconds"]]
        max_selected = max(selected, default=float("-inf"))
        margin = query_time - max_selected if selected else float("inf")
        if np.isfinite(margin):
            margins.append(margin)
        if any(value > query_time + 1e-6 for value in selected):
            violations.append(
                {
                    "key": record["_key"],
                    "query_timestamp": query_time,
                    "selected_timestamps": selected,
                }
            )
    return {
        "num_records": len(records),
        "future_frame_violations": len(violations),
        "max_future_overrun_seconds": max(
            [
                max(item["selected_timestamps"]) - item["query_timestamp"]
                for item in violations
            ],
            default=0.0,
        ),
        "minimum_causal_margin_seconds": min(margins, default=None),
        "violations": violations,
    }


def evidence_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_task: dict[str, list[int]] = defaultdict(list)
    invalid = 0
    for record in records:
        value = record["metrics"].get("selected_reference_hit")
        if value is None:
            invalid += 1
        else:
            by_task[record["task"]].append(int(bool(value)))
    return {
        "task_reference_hit_rate": {
            task: {
                "hit_rate": float(np.mean(values)),
                "percentage": float(100.0 * np.mean(values)),
                "num_valid_references": len(values),
            }
            for task, values in sorted(by_task.items())
        },
        "invalid_or_noncausal_reference_intervals_skipped": invalid,
    }


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
    ingest_seconds = sum(float(item["memory_build_seconds"]) for item in source_metrics)
    sampled_frames = sum(int(item["source_sampled_frames"]) for item in source_metrics)
    represented_seconds = sum(
        float(item["covered_timestamp_seconds"]) for item in source_metrics
    )
    query_seconds = sum(
        float(item["metrics"].get("query_pipeline_seconds", 0.0)) for item in records
    )
    total_gpu_seconds = ingest_seconds + query_seconds
    peak_vector_bytes = max(
        (int(item.get("peak_persistent_tensor_bytes", 0)) for item in source_metrics),
        default=0,
    )
    capacities = {
        int(item.get("max_memory_vectors", 0)) for item in source_metrics
    }
    chunk_sizes = {
        int(item.get("memory_chunk_frames", 0)) for item in source_metrics
    }
    capacity_value = next(iter(capacities)) if len(capacities) == 1 else None
    chunk_value = next(iter(chunk_sizes)) if len(chunk_sizes) == 1 else None
    unbounded = capacity_value == -1
    return {
        "num_unique_sources": len(source_metrics),
        "source_sampled_frames_total": sampled_frames,
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
        "workload_gpu_seconds_total_excluding_model_load": total_gpu_seconds,
        "workload_gpu_seconds_per_source_video_minute": (
            total_gpu_seconds / (represented_seconds / 60.0)
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


def merge_rank_outputs(output_dir: Path, world_size: int) -> None:
    merged: dict[str, dict[str, Any]] = {}
    for rank in range(world_size):
        path = output_dir / f"rank_{rank:02d}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"Missing rank result file: {path}")
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    item = json.loads(line)
                    merged[item["_key"]] = item
    records = sorted(merged.values(), key=lambda item: int(item["_index"]))
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in records:
        grouped[item["category"]].append(
            {
                "question_id": item["question_id"],
                "video": item["video"],
                "task": item["task"],
                "question": item["question"],
                "choices": item["choices"],
                "response": item["response"],
                "predicted_answer": extract_answer_letter(item["response"]),
                "ground_truth": item["ground_truth"],
                "requires_audio": bool(item.get("requires_audio", False)),
                "temporal_clue_type": item.get("temporal_clue_type", ""),
                "frames_required": item.get("frames_required", ""),
                "metrics": item["metrics"],
            }
        )
    (output_dir / "results.json").write_text(
        json.dumps(grouped, indent=2, ensure_ascii=False) + "\n"
    )
    (output_dir / "summary.json").write_text(
        json.dumps(score_results(records), indent=2) + "\n"
    )
    (output_dir / "efficiency_summary.json").write_text(
        json.dumps(efficiency_summary(records), indent=2) + "\n"
    )
    audit = causal_audit(records)
    (output_dir / "causal_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    if audit["future_frame_violations"]:
        raise RuntimeError(
            f"Causal audit found {audit['future_frame_violations']} future-frame violations"
        )
    (output_dir / "evidence_retrieval_summary.json").write_text(
        json.dumps(evidence_summary(records), indent=2) + "\n"
    )
    source_metrics = load_source_metrics(output_dir, world_size)
    (output_dir / "workload_summary.json").write_text(
        json.dumps(workload_summary(records, source_metrics), indent=2) + "\n"
    )


def write_source_metrics(
    path: Path, new_values: dict[str, dict[str, Any]]
) -> None:
    merged: dict[str, dict[str, Any]] = {}
    if path.is_file():
        try:
            merged.update({item["source_key"]: item for item in load_json(path)})
        except (json.JSONDecodeError, OSError, KeyError):
            pass
    merged.update(new_values)
    path.write_text(json.dumps(list(merged.values()), indent=2) + "\n")


def main() -> None:
    args = parse_args()
    cfg = load_json(args.config)
    apply_overrides(cfg, args)
    configured_tasks = list(cfg["tasks"])
    if args.tasks:
        requested = [value.strip() for value in args.tasks.split(",") if value.strip()]
        unknown = [value for value in requested if value not in configured_tasks]
        if unknown:
            raise ValueError(f"Unknown tasks requested: {unknown}")
        configured_tasks = requested
    annotations = normalize_annotations(cfg, configured_tasks)
    annotations = limit_annotations_balanced(annotations, args.max_samples)
    annotations = [{**item, "_index": index} for index, item in enumerate(annotations)]

    if args.verify_only:
        report = verify_dataset(annotations, cfg, args.verify_decode_samples)
        output_dir = Path(
            args.output_dir or Path(cfg["output_root"]) / "verification"
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / str(
            cfg.get("verification_report_name", "river_reactive_setup_report.json")
        )
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({**report, "sources": f"{len(report['sources'])} rows"}, indent=2))
        print(f"Verification report: {report_path}")
        return

    rank, world_size, local_rank, device = distributed_context()
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
    elif cfg["reactive_policy"] == "oracle_interval":
        recent_branch_suffix = (
            f"-oracleint{cfg['oracle_evidence_frames']}"
            f"-recent{cfg['oracle_recent_frames']}"
        )
    suffix = (
        f"online-resident-fps{cfg['sample_fps']:g}-chunk{cfg['memory_chunk_frames']}"
        f"-maxvec{cfg['max_memory_vectors']}"
        f"-nms{int(bool(cfg['history_temporal_nms']))}"
        f"-history{cfg['history_pool_frames']}-recent{cfg['recent_pool_frames']}"
        f"-budget{cfg['answer_frame_budget']}"
        f"-tr{int(cfg['token_reduction_enabled'])}"
        f"-trlayer{cfg['token_reduction_layer']}"
        f"-trbudget{cfg['token_reduction_budget']:g}"
        f"-trparallel{int(cfg['token_reduction_parallel'])}"
        f"{recent_branch_suffix}"
    )
    output_dir = Path(args.output_dir or Path(cfg["output_root"]) / f"{args.mode}-{suffix}")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "run_manifest.json"
    manifest = build_run_manifest(
        args.mode, cfg, configured_tasks, annotations, world_size
    )
    existing_manifest = None
    if manifest_path.is_file():
        try:
            existing_manifest = load_json(manifest_path)
        except (json.JSONDecodeError, OSError) as error:
            if not args.overwrite:
                raise RuntimeError(
                    f"Could not validate {manifest_path}; use --overwrite or a new OUTPUT_DIR"
                ) from error
    elif any(path.stat().st_size > 0 for path in output_dir.glob("rank_*.jsonl")):
        if not args.overwrite:
            raise RuntimeError(
                f"{output_dir} has rank outputs without a manifest; use a new OUTPUT_DIR"
            )
    if (
        existing_manifest is not None
        and existing_manifest.get("fingerprint") != manifest["fingerprint"]
        and not args.overwrite
    ):
        raise RuntimeError(
            f"Run scope mismatch in {output_dir}; use --overwrite or a distinct OUTPUT_DIR"
        )

    rank_path = output_dir / f"rank_{rank:02d}.jsonl"
    done_path = output_dir / f"rank_{rank:02d}.done"
    if args.overwrite and rank_path.exists():
        rank_path.unlink()
    rank_path.touch(exist_ok=True)
    if done_path.exists():
        done_path.unlink()
    done = load_done(rank_path)
    run_id = os.environ.get("TORCHELASTIC_RUN_ID", "single-process")

    if rank == 0:
        temporary = manifest_path.with_suffix(f".tmp-{os.getpid()}.json")
        temporary.write_text(json.dumps(manifest, indent=2) + "\n")
        os.replace(temporary, manifest_path)
        snapshot = {
            **cfg,
            "benchmark": str(cfg.get("benchmark", "RIVER")),
            "track": str(cfg.get("track", "reactive")),
            "mode": args.mode,
            "world_size": world_size,
            "tasks_run": configured_tasks,
            "num_annotations": len(annotations),
            "num_unique_sources": len({item["video"] for item in annotations}),
            "protocol_note": (
                "Non-deployable 8B-only oracle baseline: use the annotated evidence "
                "interval clipped at question time plus recent causal frames; do not "
                "load 2B, build vector memory, or retrieve."
                if cfg["reactive_policy"] == "oracle_interval"
                else str(
                    "Complete RIVER videos are replayed chronologically. Each completed chunk "
                    f"is encoded once into a {memory_capacity_policy(cfg['max_memory_vectors'])} "
                    "vector memory; at each question time, retrieval and 8B answering happen "
                    "immediately using only retained vectors plus recent raw frames. Optional "
                    "query-time 2B decoder-token scores prune the selected images after the 8B "
                    "ViT and before its LLM. Both models stay resident. No proactive tasks are "
                    "included."
                )
            ),
        }
        (output_dir / "inference_config.json").write_text(
            json.dumps(snapshot, indent=2) + "\n"
        )

    rank_groups = group_annotations_by_source(annotations, world_size)[rank]
    pending_groups = [
        values for values in rank_groups if any(item["_key"] not in done for item in values)
    ]
    source_metrics: dict[str, dict[str, Any]] = {}

    if pending_groups:
        oracle_interval = cfg["reactive_policy"] == "oracle_interval"
        run_execution_protocol = (
            "oracle_interval_8b_only_no_embedding_or_retrieval"
            if oracle_interval
            else memory_execution_protocol(cfg["max_memory_vectors"])
        )
        if oracle_interval:
            encoder = None
            embedding_model_load_seconds = 0.0
        else:
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
                    "models_simultaneously_resident": not oracle_interval,
                    "oracle_8b_only_execution": oracle_interval,
                    "resident_models_allocated_bytes": resident_allocated_bytes,
                    "resident_models_reserved_bytes": resident_reserved_bytes,
                    "execution_protocol": run_execution_protocol,
                    "latency_scope": "warm_resident_model_load_excluded_from_per_query_latency",
                },
                indent=2,
            )
            + "\n"
        )

        pending_total = sum(
            item["_key"] not in done for values in pending_groups for item in values
        )
        completed = 0
        with rank_path.open("a", encoding="utf-8", buffering=1) as handle:
            for group_index, items in enumerate(pending_groups, start=1):
                try:
                    source_key = items[0]["video"]
                    sampler = sampler_for_group(items, cfg)
                    if oracle_interval:
                        memory = SimpleNamespace(
                            vectors=[],
                            capacity_display="not-used",
                            readout_name="not_used_oracle_interval",
                            readout_layer=0,
                            peak_gpu_memory_bytes=0,
                            total_ingest_seconds=0.0,
                            total_frames_ingested=0,
                            total_vectors_generated=0,
                            total_vectors_evicted=0,
                            capacity=0,
                            capacity_policy="not_used",
                            chunk_frames=0,
                            persistent_bytes=0,
                            peak_persistent_bytes=0,
                            as_index=lambda: {},
                        )
                    else:
                        memory = OnlineVectorMemory(encoder, cfg)
                    next_position = 0
                    source_started = time.perf_counter()
                    for item in items:
                        key = item["_key"]
                        query_timestamp = float(item["realtime"])
                        if oracle_interval:
                            ingest_metrics = {
                                "stream_ingest_seconds_since_previous_query": 0.0,
                                "stream_ingest_chunks_since_previous_query": 0,
                                "stream_ingest_frames_since_previous_query": 0,
                                "stream_pending_unencoded_frames": len(
                                    sampler.positions_available_at(query_timestamp)
                                ),
                            }
                        else:
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
                            router_query=item.get("router_query", item["question"]),
                            score_device=device,
                        )
                        timestamps = sampler.timestamps(selected_positions)
                        if any(value > query_timestamp + 1e-6 for value in timestamps):
                            raise RuntimeError(f"Causal assertion failed immediately before 8B: {key}")
                        available = sampler.positions_available_at(query_timestamp)
                        metrics = {
                            **query_metrics,
                            **ingest_metrics,
                            **reference_metrics(item, timestamps, float(cfg["sample_fps"])),
                            "execution_protocol": run_execution_protocol,
                            "models_simultaneously_resident": not oracle_interval,
                            "oracle_8b_only_execution": oracle_interval,
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
                            "causal_max_selected_timestamp_seconds": (
                                max(timestamps) if timestamps else None
                            ),
                            "causal_margin_seconds": (
                                query_timestamp - max(timestamps) if timestamps else None
                            ),
                            "raw_video_frames": sampler.total_frames,
                            "raw_video_fps": sampler.raw_fps,
                            "raw_sampled_frames": len(available),
                            "answer_frames": len(selected_positions),
                            "frame_keep_ratio": len(selected_positions) / max(1, len(available)),
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
                            "question_id": item["question_id"],
                            "video": item["video"],
                            "category": item["category"],
                            "task": item["task"],
                            "question": item["question"],
                            "choices": item["choices"],
                            "response": response,
                            "ground_truth": item["ground_truth"],
                            "requires_audio": bool(item.get("requires_audio", False)),
                            "temporal_clue_type": item.get("temporal_clue_type", ""),
                            "frames_required": item.get("frames_required", ""),
                            "metrics": metrics,
                        }
                        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                        completed += 1
                        print(
                            f"[rank {rank}] online {completed}/{pending_total} {item['task']} "
                            f"qid={item['question_id']} H/R="
                            f"{query_metrics['answer_history_frames']}/"
                            f"{query_metrics['answer_recent_frames']} "
                            f"vectors={len(memory.vectors)}/{memory.capacity_display} "
                            f"response={response!r}",
                            flush=True,
                        )

                    source_metrics[source_key] = {
                        "source_key": source_key,
                        "source_video_path": sampler.path,
                        "num_questions": len(items),
                        "execution_protocol": run_execution_protocol,
                        "memory_build_seconds": memory.total_ingest_seconds,
                        "source_online_wall_seconds": time.perf_counter() - source_started,
                        "covered_timestamp_seconds": max(
                            float(item["realtime"]) for item in items
                        ),
                        "source_sampled_frames": memory.total_frames_ingested,
                        "memory_build_peak_gpu_memory_bytes": memory.peak_gpu_memory_bytes,
                        "cache_file_bytes": 0,
                        "num_chunks_generated": memory.total_vectors_generated,
                        "num_vectors_retained": len(memory.vectors),
                        "num_vectors_evicted": memory.total_vectors_evicted,
                        "max_memory_vectors": memory.capacity,
                        "memory_capacity_policy": memory.capacity_policy,
                        "memory_chunk_frames": memory.chunk_frames,
                        "readout_name": memory.readout_name,
                        "representation_kind": "single_vector",
                        "post_final_norm": args.mode == "full_embedding",
                        "stored_representation_count": len(memory.vectors),
                        "stored_visual_token_count": 0,
                        "stored_tensor_bytes": memory.persistent_bytes,
                        "peak_persistent_tensor_bytes": memory.peak_persistent_bytes,
                        "decoder_layers_executed_per_chunk": memory.readout_layer,
                    }
                    write_source_metrics(
                        output_dir / f"rank_{rank:02d}_memory.json", source_metrics
                    )
                    print(
                        f"[rank {rank}] completed online {cfg.get('benchmark', 'RIVER')} "
                        f"source {group_index}/{len(pending_groups)} generated="
                        f"{memory.total_vectors_generated} retained={len(memory.vectors)} "
                        f"evicted={memory.total_vectors_evicted} {source_key}",
                        flush=True,
                    )
                except Exception as source_exc:
                    # This is the loop StreamingBench reactive actually runs through
                    # (eval_streamingbench_reactive.py -> eval_river_reactive.py:main),
                    # a separate file from eval_ovo_streaming.py's execute_reactive_query
                    # loop, which got this same fix first and did NOT cover this path --
                    # a corrupt video (sample_332_real.mp4, frame 7522, unreadable by both
                    # decord and opencv) crashed this loop twice at question 1213/2500
                    # after the other file's patch was already live. Skip the rest of this
                    # one source's questions and continue with the next video.
                    print(
                        f"[rank {rank}] SKIPPING source {group_index}/{len(pending_groups)} "
                        f"{source_key}: {type(source_exc).__name__}: {source_exc}",
                        flush=True,
                    )
                    continue

    done_path.write_text(
        json.dumps({"run_id": run_id, "rank": rank, "records": len(load_done(rank_path))})
        + "\n"
    )
    if rank == 0:
        marker_paths = [output_dir / f"rank_{value:02d}.done" for value in range(world_size)]

        def current_marker(path: Path) -> bool:
            if not path.is_file():
                return False
            try:
                return load_json(path).get("run_id") == run_id
            except (json.JSONDecodeError, OSError):
                return False

        while not all(current_marker(path) for path in marker_paths):
            missing = [path.stem for path in marker_paths if not current_marker(path)]
            print(
                f"Rank 0 waiting for {cfg.get('benchmark', 'RIVER')} workers: {missing}",
                flush=True,
            )
            time.sleep(30)
        merge_rank_outputs(output_dir, world_size)
        print(
            f"Completed {cfg.get('benchmark', 'RIVER')} "
            f"{cfg.get('track', 'reactive')} {args.mode}: {output_dir}",
            flush=True,
        )


if __name__ == "__main__":
    main()
