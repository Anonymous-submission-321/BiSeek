#!/usr/bin/env python3
"""Evaluate bounded-vector BiSeek streaming on StreamingBench immediate QA.

This adapter deliberately reuses the exercised OVO/RIVER semantic-memory
engine.  It supplies StreamingBench-specific normalization, prompts, dataset
verification, and scoring while retaining the same causal sampler, bounded
FIFO vector memory, four-rank resume behavior, and two representation modes.

Scope: 4,245 immediate multiple-choice questions from the three official
``*_stream.json`` files across Real-Time Visual
Understanding, Omni-Source Understanding, ACU/MCU, and SQA.  The 250 Proactive
Output questions require a separate wait/respond protocol and are not silently
converted into immediate QA here.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

import eval_river_reactive as engine
from eval_ovo_streaming import load_json


TASKS = [
    "Object Perception",
    "Causal Reasoning",
    "Clips Summarize",
    "Attribute Perception",
    "Event Understanding",
    "Text-Rich Understanding",
    "Prospective Reasoning",
    "Spatial Understanding",
    "Action Perception",
    "Counting",
    "Emotion Recognition",
    "Scene Understanding",
    "Source Discrimination",
    "Multimodal Alignment",
    "Anomaly Context Understanding",
    "Misleading Context Recognition",
    "Sequential Question Answering",
]

CATEGORIES = [
    "real_time_visual_understanding",
    "omni_source_understanding",
    "contextual_understanding",
]

PROMPT = """\
You are an advanced video question-answering AI assistant. You have been
provided with frames from a video stream up to the current query time and a
multiple-choice question. Carefully analyze the available evidence and choose
the best answer. Respond only with the letter A, B, C, or D.

{history}Question: {question}

Options:
{options}

The best option is:"""


REQUIRED_INTERVAL_PATTERN = re.compile(
    r"^\s*[\[【]?\s*(\d+):(\d+):(\d+)\s*-\s*"
    r"(\d+):(\d+):(\d+)\s*[\]】]?\s*$"
)


def parse_frames_required_interval(value: str) -> list[float]:
    """Parse the official StreamingBench required-frame time range."""

    match = REQUIRED_INTERVAL_PATTERN.fullmatch(str(value))
    if match is None:
        return []
    parts = [int(component) for component in match.groups()]
    if any(component >= 60 for component in (parts[1], parts[2], parts[4], parts[5])):
        return []
    start = float(parts[0] * 3600 + parts[1] * 60 + parts[2])
    end = float(parts[3] * 3600 + parts[4] * 60 + parts[5])
    return sorted([start, end])


def option_lines(options: list[str]) -> list[str]:
    values = []
    for index, option in enumerate(options):
        text = str(option).strip()
        if not re.match(r"^[A-D][\.\):]\s*", text, flags=re.IGNORECASE):
            text = f"{chr(65 + index)}. {text}"
        values.append(text)
    return values


def history_text(history: list[dict[str, Any]]) -> str:
    if not history:
        return ""
    lines = [
        "Here is the corrected interaction history for this same video stream. "
        "Use it only to resolve references in the current question:"
    ]
    for item in history:
        choices = option_lines(item["options"])
        answer_index = ord(str(item["answer"]).strip().upper()) - ord("A")
        answer_text = (
            choices[answer_index]
            if 0 <= answer_index < len(choices)
            else str(item["answer"])
        )
        lines.append(
            f"At {item['time_stamp']}, Question: {item['question']} "
            f"Correct answer: {answer_text}"
        )
    return "\n".join(lines) + "\n\n"


def build_prompt(item: dict[str, Any]) -> str:
    return PROMPT.format(
        history=history_text(item.get("conversation_history", [])),
        question=item["question"],
        options="\n".join(option_lines(item["choices"])),
    )


def normalize_annotations(
    cfg: dict[str, Any], configured_tasks: list[str]
) -> list[dict[str, Any]]:
    source = load_json(cfg["annotation_path"])
    records: list[dict[str, Any]] = []
    for raw in source:
        task = str(raw["task_type"])
        if task not in configured_tasks:
            continue
        choices = [str(value).strip() for value in raw["options"]]
        ground_truth = str(raw["answer"]).strip().upper()
        if len(choices) != 4 or ground_truth not in {"A", "B", "C", "D"}:
            raise ValueError(
                f"Invalid MCQ annotation {raw['question_id']}: "
                f"{len(choices)} options, answer={ground_truth!r}"
            )
        history = list(raw.get("conversation_history", []))
        router_query = str(raw["question"])
        if history:
            router_query = (
                "Corrected prior interactions:\n"
                + "\n".join(
                    f"Q: {item['question']} A: {item['answer']}"
                    for item in history
                )
                + f"\nCurrent question: {raw['question']}"
            )
        records.append(
            {
                "_key": str(raw["_key"]),
                "id": str(raw["_key"]),
                "question_id": str(raw["question_id"]),
                "video": str(raw["video"]),
                "video_source": "StreamingBench",
                "video_id": str(raw["video"]),
                "video_path": str(raw["video_path"]),
                "category": str(raw["category"]),
                "task": task,
                "question": str(raw["question"]),
                "router_query": router_query,
                "conversation_history": history,
                "choices": choices,
                "ground_truth": ground_truth,
                "realtime": float(raw["question_time_seconds"]),
                "time_reference": parse_frames_required_interval(
                    str(raw.get("frames_required", ""))
                ),
                "requires_audio": bool(raw.get("requires_audio", False)),
                "temporal_clue_type": str(raw.get("temporal_clue_type", "")),
                "frames_required": str(raw.get("frames_required", "")),
                "annotation_file": str(raw.get("annotation_csv", "")),
            }
        )
    keys = [item["_key"] for item in records]
    duplicates = [key for key, count in Counter(keys).items() if count > 1]
    if duplicates:
        raise ValueError(f"Duplicate StreamingBench keys: {duplicates[:5]}")
    return records


def score_results(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_task: dict[str, list[int]] = defaultdict(list)
    by_category: dict[str, list[int]] = defaultdict(list)
    audio_required: list[int] = []
    vision_only: list[int] = []
    unresolved = 0
    for record in records:
        prediction = engine.extract_answer_letter(record.get("response"))
        if prediction is None:
            unresolved += 1
        correct = int(prediction == record["ground_truth"])
        by_task[record["task"]].append(correct)
        by_category[record["category"]].append(correct)
        if record.get("requires_audio", False):
            audio_required.append(correct)
        else:
            vision_only.append(correct)

    def row(values: list[int]) -> dict[str, Any]:
        accuracy = float(np.mean(values)) if values else None
        return {
            "accuracy": accuracy,
            "percentage": None if accuracy is None else 100.0 * accuracy,
            "num_evaluations": len(values),
        }

    all_values = [value for values in by_task.values() for value in values]
    task_scores = {task: row(by_task[task]) for task in sorted(by_task)}
    category_scores = {
        category: row(by_category[category]) for category in CATEGORIES if category in by_category
    }
    contextual = category_scores.get("contextual_understanding")
    if contextual is not None:
        contextual_tasks = [
            task
            for task in (
                "Anomaly Context Understanding",
                "Misleading Context Recognition",
                "Sequential Question Answering",
            )
            if task in task_scores
        ]
        contextual.update(
            {
                "scope": "reactive_subset_only",
                "included_tasks": contextual_tasks,
                "missing_official_task": "Proactive Output",
                "official_contextual_all": False,
                "reactive_subset_complete": len(contextual_tasks) == 3,
                "definition": (
                    "micro_accuracy_over_ACU_MCU_SQA_only; use "
                    "tools/merge_streamingbench_contextual.py with a Proactive "
                    "Output summary to obtain the official four-task macro average"
                ),
            }
        )
    task_macro_values = [value["accuracy"] for value in task_scores.values()]
    category_macro_values = [value["accuracy"] for value in category_scores.values()]
    task_macro = float(np.mean(task_macro_values)) if task_macro_values else None
    category_macro = (
        float(np.mean(category_macro_values)) if category_macro_values else None
    )
    overall = row(all_values)
    overall.update(
        {
            "definition": "micro_accuracy_over_selected_non_proactive_questions",
            "full_reactive_scope_questions": 4245,
            "scope": (
                "StreamingBench immediate/reactive subset only. The official Table-2 "
                "Contextual All additionally uses the released full 250-question "
                "Proactive Output evaluation."
            ),
            "official_table_overall": False,
        }
    )
    recent_breakdown: dict[str, Any] = {}
    for decision in ("direct_answer", "retrieve"):
        subset = [
            record for record in records
            if record.get("metrics", {}).get("recent_first_decision") == decision
        ]
        if subset:
            correct = [
                int(engine.extract_answer_letter(record.get("response")) == record["ground_truth"])
                for record in subset
            ]
            recent_breakdown[decision] = {
                "accuracy": float(np.mean(correct)),
                "percentage": float(100.0 * np.mean(correct)),
                "num_evaluations": len(correct),
            }
    oracle_breakdown: dict[str, Any] = {}
    for name, expected_overlap in (("causal_overlap", True), ("future_only_fallback", False)):
        subset = [
            record
            for record in records
            if record.get("metrics", {}).get("reactive_policy") == "oracle_interval"
            and bool(
                record.get("metrics", {}).get("oracle_interval_has_causal_overlap")
            )
            == expected_overlap
        ]
        if subset:
            correct = [
                int(engine.extract_answer_letter(record.get("response")) == record["ground_truth"])
                for record in subset
            ]
            oracle_breakdown[name] = row(correct)
    return {
        "benchmark": "StreamingBench",
        "track": "reactive",
        "annotation_scope": "official_stream_4245",
        "overall": overall,
        "task_scores": task_scores,
        "category_scores": category_scores,
        "task_macro": {
            "accuracy": task_macro,
            "percentage": None if task_macro is None else 100.0 * task_macro,
            "num_tasks": len(task_macro_values),
        },
        "category_macro": {
            "accuracy": category_macro,
            "percentage": None if category_macro is None else 100.0 * category_macro,
            "num_categories": len(category_macro_values),
        },
        "audio_required_subset": row(audio_required),
        "vision_only_subset": row(vision_only),
        "unparsed_responses": unresolved,
        "recent_first_breakdown": recent_breakdown,
        "oracle_interval_breakdown": oracle_breakdown,
    }


def verify_dataset(
    annotations: list[dict[str, Any]], cfg: dict[str, Any], decode_samples: int
) -> dict[str, Any]:
    from decord import VideoReader, cpu

    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in annotations:
        by_source[item["video"]].append(item)
    missing: list[str] = []
    empty: list[str] = []
    question_after_video: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for index, (source_key, items) in enumerate(sorted(by_source.items()), start=1):
        path = Path(items[0]["video_path"])
        if not path.is_file():
            missing.append(str(path))
            continue
        reader = VideoReader(str(path), ctx=cpu(0), num_threads=1)
        raw_frames = len(reader)
        raw_fps = float(reader.get_avg_fps())
        if raw_frames <= 0:
            empty.append(str(path))
            continue
        duration = raw_frames / max(raw_fps, 1e-6)
        required = max(float(item["realtime"]) for item in items)
        tolerance = 0.5 / max(float(cfg["sample_fps"]), 1e-6) + 1e-3
        if required > duration + tolerance:
            question_after_video.append(
                {
                    "source": source_key,
                    "required_timestamp": required,
                    "actual_duration": duration,
                }
            )
        estimated = min(
            raw_frames,
            int(np.floor(required * float(cfg["sample_fps"]))) + 1,
        )
        sources.append(
            {
                "source": source_key,
                "path": str(path),
                "num_questions": len(items),
                "raw_frames": raw_frames,
                "raw_fps": raw_fps,
                "actual_duration_seconds": duration,
                "latest_question_seconds": required,
                "estimated_fixed_fps_frames": estimated,
            }
        )
        if index % 50 == 0 or index == len(by_source):
            print(f"Verified StreamingBench metadata {index}/{len(by_source)}", flush=True)

    decoded_sources: list[str] = []
    if decode_samples > 0 and sources:
        selected = np.linspace(
            0, len(sources) - 1, min(decode_samples, len(sources))
        ).round().astype(np.int64)
        for row_index in np.unique(selected):
            row = sources[int(row_index)]
            reader = VideoReader(row["path"], ctx=cpu(0), num_threads=1)
            last_allowed = min(
                len(reader) - 1,
                int(round(row["latest_question_seconds"] * row["raw_fps"])),
            )
            indices = np.linspace(0, last_allowed, 3).round().astype(np.int64).tolist()
            reader.get_batch(indices).asnumpy()
            decoded_sources.append(row["source"])

    report = {
        "num_reactive_records": len(annotations),
        "num_unique_source_videos": len(by_source),
        "task_counts": dict(Counter(item["task"] for item in annotations)),
        "category_counts": dict(Counter(item["category"] for item in annotations)),
        "audio_required_questions": sum(item["requires_audio"] for item in annotations),
        "sqa_questions_with_history": sum(
            bool(item["conversation_history"]) for item in annotations
        ),
        "sample_fps": float(cfg["sample_fps"]),
        "memory_chunk_frames": int(cfg["memory_chunk_frames"]),
        "max_memory_vectors": int(cfg["max_memory_vectors"]),
        "memory_capacity_policy": engine.memory_capacity_policy(
            cfg["max_memory_vectors"]
        ),
        "history_temporal_nms": bool(cfg["history_temporal_nms"]),
        "nominal_retained_sampled_frame_capacity": (
            engine.nominal_retained_frame_capacity(
                cfg["max_memory_vectors"], cfg["memory_chunk_frames"]
            )
        ),
        "estimated_fixed_fps_frames_to_latest_queries": sum(
            row["estimated_fixed_fps_frames"] for row in sources
        ),
        "missing_videos": missing,
        "empty_videos": empty,
        "question_after_video": question_after_video,
        "required_interval_parseable_questions": sum(
            len(item.get("time_reference", [])) == 2 for item in annotations
        ),
        "required_interval_causal_overlap_questions": sum(
            len(item.get("time_reference", [])) == 2
            and min(item["time_reference"]) <= float(item["realtime"]) + 1e-6
            for item in annotations
        ),
        "required_interval_starts_after_question_questions": sum(
            len(item.get("time_reference", [])) == 2
            and min(item["time_reference"]) > float(item["realtime"]) + 1e-6
            for item in annotations
        ),
        "required_interval_extends_past_question_questions": sum(
            len(item.get("time_reference", [])) == 2
            and max(item["time_reference"]) > float(item["realtime"]) + 1e-6
            for item in annotations
        ),
        "required_interval_width_counts": dict(
            Counter(
                str(int(max(item["time_reference"]) - min(item["time_reference"])))
                for item in annotations
                if len(item.get("time_reference", [])) == 2
            )
        ),
        "decoded_source_samples": decoded_sources,
        "sources": sources,
        "protocol_note": (
            "All selected timestamps must be <= the question timestamp. SQA prompts use "
            "ground-truth answers from earlier interactions, matching the official protocol. "
            "The current Qwen3-VL path is vision+text only, so the 1000 omni-source questions "
            "are scored without audio and reported explicitly as an audio-required subset."
        ),
    }
    if missing or empty or question_after_video:
        raise RuntimeError(json.dumps(report, indent=2))
    return report


def main() -> None:
    engine.REACTIVE_TASKS = TASKS
    engine.RETRO_TASKS = []
    engine.normalize_annotations = normalize_annotations
    engine.build_prompt = build_prompt
    engine.score_results = score_results
    engine.verify_dataset = verify_dataset
    engine.main()


if __name__ == "__main__":
    main()
