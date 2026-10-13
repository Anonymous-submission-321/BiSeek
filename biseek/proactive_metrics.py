"""Post-hoc metrics for proactive streaming experiments.

The functions in this module are deliberately model-free: completed result
files can be rescored without rerunning either Qwen model.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def _records_from_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        if isinstance(payload.get("forward"), list):
            return payload["forward"]
        if isinstance(payload.get("results"), list):
            return payload["results"]
    raise ValueError("Expected a result list or a JSON object containing forward/results")


def load_records(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open() as handle:
        return _records_from_payload(json.load(handle))


def ovo_ground_truth_trigger_times(record: dict[str, Any]) -> list[float]:
    """Return the exact event timestamps used by Em-Garde's OVO online eval."""

    task = str(record["task"])
    if task == "CRR":
        return [float(record["clue_time"])]
    if task == "REC":
        return sorted(float(value) + 2.0 for value in record.get("start_times", []))
    if task == "SSR":
        values = record.get("start_time", [])
        if not isinstance(values, list):
            values = [values]
        return sorted(float(value) + 2.0 for value in values)
    raise ValueError(f"Unsupported OVO online-trigger task: {task}")


def _expanded_decisions(record: dict[str, Any]) -> Iterable[tuple[float, dict[str, Any]]]:
    for episode in record.get("cascade_episodes", []):
        for decision in episode.get("task_state_decisions", []):
            outer_timestamp = float(decision.get("timestamp_seconds", 0.0))
            metrics = decision.get("metrics", {})
            steps = metrics.get("incremental_step_metrics")
            if not isinstance(steps, list):
                yield outer_timestamp, metrics
                continue
            for step in steps:
                selected = step.get("selected_timestamps_seconds") or []
                timestamp = float(selected[-1]) if selected else outer_timestamp
                yield timestamp, step


def _legacy_rec_count_trigger_times(record: dict[str, Any]) -> list[float]:
    predicted: list[float] = []
    previous = 0
    tests = sorted(
        record.get("test_info", []), key=lambda value: float(value["realtime"])
    )
    for test in tests:
        digits = "".join(value for value in str(test.get("response", "")) if value.isdigit())
        current = int(digits) if digits else previous
        # A jump by k represents k newly counted occurrences. Repeating the
        # timestamp lets the one-to-one matcher credit at most k GT events.
        predicted.extend([float(test["realtime"])] * max(0, current - previous))
        previous = max(previous, current)
    return predicted


def _rec_count_trigger_times(record: dict[str, Any], *, stage: str) -> list[float]:
    """Extract REC events without discarding unified-controller timing.

    The legacy interval counter only emits cumulative values at official test
    points, so its event time can only be approximated by the checkpoint.  The
    unified proposal controller, however, records every matcher candidate and
    every verified integer delta in its continuous decision trace.  Prefer
    those timestamps when present; otherwise preserve the legacy behavior.
    """

    decisions = list(_expanded_decisions(record))
    has_unified_trace = any(
        "rec_verified_interval_delta" in metrics for _, metrics in decisions
    )
    if not has_unified_trace:
        return _legacy_rec_count_trigger_times(record)

    predicted: list[float] = []
    for timestamp, metrics in decisions:
        if stage == "proposal_candidate":
            if metrics.get("proposal_new_candidate") is True:
                predicted.append(float(timestamp))
            continue
        delta = metrics.get("rec_verified_interval_delta")
        if (
            bool(metrics.get("verifier_8b_called"))
            and metrics.get("verifier_positive") is True
            and isinstance(delta, (int, float))
            and int(delta) > 0
        ):
            predicted.extend([float(timestamp)] * int(delta))
    return sorted(predicted)


def ovo_predicted_trigger_times(
    record: dict[str, Any], *, stage: str
) -> list[float]:
    """Extract causal detector events from a BiSeek result trace.

    ``proposal_candidate`` audits the Embedding-2B matcher before the 8B
    verifier. ``verified_positive_call`` counts every positive 8B verdict.
    ``verified_state_onset`` is the deployed response event: the public state
    changes from negative to positive. REC is task-native and uses positive
    count increments for every stage.
    """

    if str(record["task"]) == "REC":
        return _rec_count_trigger_times(record, stage=stage)
    if stage not in {
        "proposal_candidate",
        "verified_positive_call",
        "verified_state_onset",
    }:
        raise ValueError(f"Unknown online trigger stage: {stage}")
    values: list[float] = []
    for timestamp, metrics in _expanded_decisions(record):
        if stage == "proposal_candidate":
            keep = metrics.get("proposal_new_candidate") is True
        elif stage == "verified_positive_call":
            keep = (
                bool(metrics.get("verifier_8b_called"))
                and metrics.get("verifier_positive") is True
            )
        else:
            keep = (
                metrics.get("proposal_public_state_before") is False
                and metrics.get("proposal_public_state_after") is True
            )
        if keep:
            values.append(float(timestamp))
    return sorted(values)


def match_trigger_times(
    ground_truth: list[float], predictions: list[float], tolerance_seconds: float
) -> dict[str, Any]:
    """Reproduce Em-Garde's one-to-one, GT-ordered @2s event matching."""

    remaining = list(sorted(float(value) for value in predictions))
    matches: list[dict[str, float]] = []
    for ground_time in sorted(float(value) for value in ground_truth):
        for index, predicted_time in enumerate(remaining):
            if abs(predicted_time - ground_time) <= tolerance_seconds:
                matches.append(
                    {
                        "ground_truth_seconds": ground_time,
                        "prediction_seconds": predicted_time,
                        "error_seconds": predicted_time - ground_time,
                    }
                )
                remaining.pop(index)
                break
    correct = len(matches)
    recall = correct / len(ground_truth) if ground_truth else 0.0
    precision = correct / len(predictions) if predictions else 0.0
    return {
        "ground_truth_count": len(ground_truth),
        "prediction_count": len(predictions),
        "correct_count": correct,
        "precision": precision,
        "recall": recall,
        "matches": matches,
        "false_positive_times_seconds": remaining,
    }


def score_ovo_event_timing_proxy(
    records: list[dict[str, Any]], *, tolerance_seconds: float = 2.0
) -> dict[str, Any]:
    """Score BiSeek events with Em-Garde's event-matching mathematics.

    This is a diagnostic proxy, not an official online benchmark result. BiSeek
    REC emits checkpoint count increments, while its SSR queries are registered
    from the OVO test-point schedule. Neither is a continuous t=0 detector trace.
    """

    stages: dict[str, Any] = {}
    for stage in (
        "proposal_candidate",
        "verified_positive_call",
        "verified_state_onset",
    ):
        per_task_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
        details: list[dict[str, Any]] = []
        for record in records:
            task = str(record.get("task"))
            if task not in {"REC", "SSR", "CRR"}:
                continue
            ground_truth = ovo_ground_truth_trigger_times(record)
            predictions = ovo_predicted_trigger_times(record, stage=stage)
            matched = match_trigger_times(
                ground_truth, predictions, float(tolerance_seconds)
            )
            row = {
                "id": record.get("id"),
                "task": task,
                "ground_truth_times_seconds": ground_truth,
                "prediction_times_seconds": predictions,
                **matched,
            }
            per_task_rows[task].append(row)
            details.append(row)

        task_scores: dict[str, Any] = {}
        for task in ("REC", "SSR", "CRR"):
            rows = per_task_rows.get(task, [])
            if not rows:
                continue
            macro_precision = sum(value["precision"] for value in rows) / len(rows)
            macro_recall = sum(value["recall"] for value in rows) / len(rows)
            f1 = (
                2.0 * macro_precision * macro_recall / (macro_precision + macro_recall)
                if macro_precision + macro_recall
                else 0.0
            )
            correct = sum(value["correct_count"] for value in rows)
            predicted = sum(value["prediction_count"] for value in rows)
            ground = sum(value["ground_truth_count"] for value in rows)
            micro_precision = correct / predicted if predicted else 0.0
            micro_recall = correct / ground if ground else 0.0
            micro_f1 = (
                2.0 * micro_precision * micro_recall / (micro_precision + micro_recall)
                if micro_precision + micro_recall
                else 0.0
            )
            task_scores[task] = {
                "num_annotations": len(rows),
                "macro_precision": macro_precision,
                "macro_recall": macro_recall,
                "macro_f1": f1,
                "f1_percent": 100.0 * f1,
                "event_counts": {
                    "ground_truth": ground,
                    "predicted": predicted,
                    "matched": correct,
                },
                "micro_precision": micro_precision,
                "micro_recall": micro_recall,
                "micro_f1": micro_f1,
            }
        stages[stage] = {
            "task_scores": task_scores,
            "macro_task_f1": (
                sum(value["macro_f1"] for value in task_scores.values())
                / len(task_scores)
                if task_scores
                else None
            ),
            "details": details,
        }
    return {
        "metric_math": (
            "Em-Garde event-matching mathematics: GT is CRR clue_time, REC "
            "start_times+2s, SSR start_time+2s; predictions are greedily "
            "one-to-one matched within ± tolerance; per-task precision/recall "
            "are annotation-macro averages and F1 is their harmonic mean."
        ),
        "comparability": "diagnostic_proxy_not_official_online_benchmark",
        "limitations": {
            "REC": (
                "Unified-controller predictions use continuous matcher and "
                "verified integer-delta timestamps. Legacy interval-counter "
                "records fall back to positive checkpoint count increments."
            ),
            "SSR": (
                "Standing queries are registered from annotation test points, "
                "not once at t=0 with a tutorial-level online query."
            ),
            "CRR": (
                "Uses BiSeek public-state onsets, but is not an official evaluator "
                "submission and is grouped with non-comparable REC/SSR proxies."
            ),
        },
        "tolerance_seconds": float(tolerance_seconds),
        "primary_stage": "verified_state_onset",
        "stages": stages,
    }


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _observed_duration(record: dict[str, Any]) -> float:
    if isinstance(record.get("decision_history"), list):
        start = float(record.get("query_time_seconds", 0.0))
        ends = [
            float(value.get("timestamp_seconds", start))
            for value in record["decision_history"]
        ]
        return max(0.0, max(ends, default=start) - start)
    starts: list[float] = []
    ends: list[float] = []
    for episode in record.get("cascade_episodes", []):
        starts.append(float(episode.get("registration_timestamp", 0.0)))
        for decision in episode.get("task_state_decisions", []):
            ends.append(float(decision.get("timestamp_seconds", starts[-1])))
    if not ends:
        return 0.0
    return max(0.0, max(ends) - min(starts or [0.0]))


def aggregate_token_reduction_metrics(
    records: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Reconstruct query-time reduction totals from saved decision traces.

    Older workload logs only counted the task-native REC answer path and did
    not include sparse 8B verifier calls.  StreamingBench therefore reported
    zero reductions, while OVO reported REC only.  The result trace is the
    authoritative source because every actual 8B call records its before/after
    visual-token counts.

    Incremental controllers copy a summary onto the outer decision and retain
    the per-frame calls under ``incremental_step_metrics``.  Consume the steps
    when present to avoid double counting the copied outer summary.
    """

    events: list[dict[str, Any]] = []

    def add_decision(metrics: Any) -> None:
        if not isinstance(metrics, dict):
            return
        steps = metrics.get("incremental_step_metrics")
        candidates = steps if isinstance(steps, list) else [metrics]
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            before = int(candidate.get("answer_visual_tokens_before_reduction", 0))
            after = int(candidate.get("answer_visual_tokens", 0))
            if bool(candidate.get("token_reduction_enabled")) and before > 0:
                events.append(
                    {
                        "before": before,
                        "after": after,
                        "seconds": float(candidate.get("token_reduction_seconds", 0.0)),
                        "critical_path_seconds": float(
                            candidate.get(
                                "token_reduction_parallel_visual_critical_path_seconds",
                                0.0,
                            )
                        ),
                    }
                )

    for record in records:
        events_before_record = len(events)
        for decision in record.get("decision_history", []):
            add_decision(decision.get("metrics", {}))
        for episode in record.get("cascade_episodes", []):
            for decision in episode.get("task_state_decisions", []):
                add_decision(decision.get("metrics", {}))
        # REC invokes the task-native interval counter at official test points
        # instead of the sparse verifier.  Only count checkpoints that really
        # called the answer model; non-call entries can still carry defaults.
        # New REC results mirror those calls into ``cascade_episodes`` for a
        # unified trace.  Legacy REC results have ``test_info`` only.  Fall
        # back to test checkpoints only when this record contributed no trace
        # event, otherwise the same counter call would be counted twice.
        if len(events) == events_before_record:
            for test in record.get("test_info", []):
                metrics = test.get("metrics", {})
                if bool(metrics.get("answer_model_called_at_this_test")):
                    add_decision(metrics)

    before = sum(value["before"] for value in events)
    after = sum(value["after"] for value in events)
    return {
        "calls": len(events),
        "seconds": sum(value["seconds"] for value in events),
        "parallel_visual_critical_path_seconds": sum(
            value["critical_path_seconds"] for value in events
        ),
        "visual_tokens_before": before,
        "visual_tokens_after": after,
        "keep_ratio": after / before if before else None,
        "source": "saved_decision_traces",
    }


def summarize_proactive_efficiency(result_dir: str | Path) -> dict[str, Any]:
    result_dir = Path(result_dir)
    workload_paths = sorted(result_dir.glob("rank_*_workload.json"))
    if not workload_paths:
        raise FileNotFoundError(f"No rank_*_workload.json under {result_dir}")
    ranks: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for path in workload_paths:
        with path.open() as handle:
            values = json.load(handle)
        wall = sum(
            float(value.get("source_online_wall_seconds", 0.0))
            + float(value.get("source_registration_wall_seconds", 0.0))
            for value in values
        )
        ranks.append({"rank_file": path.name, "sources": len(values), "wall_sum_seconds": wall})
        sources.extend(values)
    source_walls = [
        float(value.get("source_online_wall_seconds", 0.0))
        + float(value.get("source_registration_wall_seconds", 0.0))
        for value in sources
    ]
    results_path = result_dir / "results.json"
    records = load_records(results_path) if results_path.is_file() else []
    observed_seconds = sum(_observed_duration(value) for value in records)
    rank_sum = sum(value["wall_sum_seconds"] for value in ranks)
    rank_makespan = max((value["wall_sum_seconds"] for value in ranks), default=0.0)
    totals = {
        name: sum(float(value.get(name, 0.0)) for value in sources)
        for name in (
            "gate_2b_seconds",
            "verifier_8b_seconds",
            "proposal_compiler_8b_seconds",
            "proposal_context_retrieval_seconds",
            "rec_counter_8b_seconds",
        )
    }
    calls = {
        name: sum(int(value.get(name, 0)) for value in sources)
        for name in (
            "gate_2b_calls",
            "verifier_8b_calls",
            "proposal_compiler_8b_calls",
            "proposal_context_retrieval_2b_calls",
            "rec_counter_8b_calls",
            "2b_model_forward_calls",
            "8b_model_forward_calls",
        )
    }
    workload_token_reduction = {
        "calls": sum(int(value.get("8b_token_reduction_calls", 0)) for value in sources),
        "seconds": sum(
            float(value.get("8b_token_reduction_seconds", 0.0)) for value in sources
        ),
        "visual_tokens_before": sum(
            int(value.get("8b_visual_tokens_before_reduction", 0))
            for value in sources
        ),
        "visual_tokens_after": sum(
            int(value.get("8b_visual_tokens_after_reduction", 0))
            for value in sources
        ),
    }
    workload_token_reduction["keep_ratio"] = (
        workload_token_reduction["visual_tokens_after"]
        / workload_token_reduction["visual_tokens_before"]
        if workload_token_reduction["visual_tokens_before"]
        else None
    )
    workload_token_reduction["source"] = "rank_workload_logs"
    trace_token_reduction = aggregate_token_reduction_metrics(records)
    token_reduction = (
        trace_token_reduction
        if trace_token_reduction["calls"] > 0
        else workload_token_reduction
    )
    config = {}
    config_path = result_dir / "inference_config.json"
    if config_path.is_file():
        with config_path.open() as handle:
            config = json.load(handle)
    sample_fps = float(config.get("sample_fps", 0.0) or 0.0)
    official_poll_interval = float(
        config.get("official_poll_interval_seconds", 0.0) or 0.0
    )
    batch_new_frames = bool(config.get("proposal_batch_new_frames_per_poll", False))
    matcher_target_fps = (
        1.0 / official_poll_interval
        if batch_new_frames and official_poll_interval > 0
        else sample_fps
    )
    gate_seconds = totals["gate_2b_seconds"]
    return {
        "scope": (
            "warm-model replay measurements from saved per-source workload logs; "
            "rank makespan is an approximation that excludes common startup/merge"
        ),
        "result_dir": str(result_dir.resolve()),
        "world_size": len(ranks),
        "num_sources": len(sources),
        "source_wall_seconds": {
            "sum": sum(source_walls),
            "mean": sum(source_walls) / len(source_walls) if source_walls else None,
            "median": _percentile(source_walls, 0.5),
            "p95": _percentile(source_walls, 0.95),
            "max": max(source_walls) if source_walls else None,
        },
        "rank_work": ranks,
        "rank_sum_seconds": rank_sum,
        "approx_parallel_makespan_seconds": rank_makespan,
        "observed_causal_video_seconds": observed_seconds,
        "single_gpu_equivalent_replay_rtf": (
            rank_sum / observed_seconds if observed_seconds else None
        ),
        "parallel_job_replay_rtf": (
            rank_makespan / observed_seconds if observed_seconds else None
        ),
        "component_seconds": totals,
        "component_share_of_rank_sum": {
            key: value / rank_sum if rank_sum else None for key, value in totals.items()
        },
        "model_calls": calls,
        "token_reduction": token_reduction,
        "matcher_polls_per_gpu_second": (
            calls["gate_2b_calls"] / gate_seconds if gate_seconds else None
        ),
        "observation_sample_fps": sample_fps or None,
        "matcher_target_poll_fps": matcher_target_fps or None,
        "matcher_poll_capacity_headroom": (
            calls["gate_2b_calls"] / gate_seconds / matcher_target_fps
            if gate_seconds and matcher_target_fps
            else None
        ),
        "max_memory_bytes": {
            name: max((int(value.get(name, 0)) for value in sources), default=0)
            for name in (
                "max_2b_kv_cache_bytes",
                "max_8b_kv_cache_bytes",
                "persistent_proposal_vector_bytes",
                "max_persistent_proposal_vector_bytes",
                "persistent_shared_window_vector_cache_bytes",
                "max_persistent_shared_window_vector_cache_bytes",
                "persistent_shared_frame_vector_cache_bytes",
                "max_persistent_shared_frame_vector_cache_bytes",
                "persistent_shared_vit_feature_cache_bytes",
                "max_persistent_shared_vit_feature_cache_bytes",
            )
        },
        "config_identity": {
            key: config.get(key)
            for key in (
                "benchmark",
                "track",
                "controller",
                "decision_mode",
                "sample_fps",
                "official_poll_interval_seconds",
                "proposal_batch_new_frames_per_poll",
                "proposal_match_window_frames",
                "proposal_surge_threshold",
                "proposal_match_encoding",
                "proposal_frame_cache_size",
                "proposal_vit_cache_size",
                "token_reduction_enabled",
                "token_reduction_layer",
                "token_reduction_budget",
                "token_reduction_parallel",
            )
            if key in config
        },
    }
