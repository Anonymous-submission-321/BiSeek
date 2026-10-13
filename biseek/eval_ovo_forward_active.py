#!/usr/bin/env python3
"""OVO-Bench Forward Active with cached baselines and a unified update gate."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

_TOOLS_DIR = str(Path(__file__).resolve().parent)
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)

from eval_ovo_streaming import (
    AnswerModel,
    SemanticMemoryEncoder,
    VideoSampler,
    load_done,
    load_json,
    parse_bool,
    prepare_resumable_jsonl,
)
from proactive_cascade import (
    default_verifier_accept_margin,
    CONTROLLER_VERSION,
    DECISION_MODES,
    PROPOSAL_PROMPT_STYLES,
    CachedProactiveController,
    ProactiveState,
    build_proactive_controller,
    controller_name,
    inspect_proposal_manifest,
    VERIFIER_PROMPT_MODES,
    set_verifier_prompt_mode,
)
from proactive_metrics import (
    score_ovo_event_timing_proxy,
    summarize_proactive_efficiency,
)
from video_probe import probe_video_duration_seconds


FORWARD_TASKS = ("REC", "SSR", "CRR")


def unified_rec_enabled(cfg: dict[str, Any]) -> bool:
    return bool(
        cfg.get("decision_mode") == "proposal_match"
        and cfg.get("rec_controller") == "proposal_match"
    )


def atomic_write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--tasks", default="")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--sample-fps", type=float)
    parser.add_argument("--recent-window-frames", type=int)
    parser.add_argument("--rec-interval-max-frames", type=int)
    parser.add_argument("--rec-sample-fps", type=float)
    parser.add_argument("--incremental-kv-reset-every-n-frames", type=int)
    parser.add_argument("--verifier-accept-margin", type=float)
    parser.add_argument("--answer-model-path")
    parser.add_argument(
        "--answer-model-family",
        choices=("qwen3vl", "qwen2_5vl", "qwen2vl"),
    )
    parser.add_argument(
        "--rec-prompt-style",
        choices=("integer_only", "enumerate_then_count", "conservative"),
    )
    parser.add_argument("--rec-max-new-tokens", type=int)
    parser.add_argument("--decision-mode", choices=DECISION_MODES)
    parser.add_argument("--proposal-count", type=int)
    parser.add_argument(
        "--proposal-prompt-style",
        choices=PROPOSAL_PROMPT_STYLES,
    )
    parser.add_argument("--proposal-generation-window-frames", type=int)
    parser.add_argument("--proposal-context-retrieval-enabled", type=parse_bool)
    parser.add_argument("--proposal-context-retrieval-top-k", type=int)
    parser.add_argument("--proposal-context-retrieval-chunk-frames", type=int)
    parser.add_argument("--proposal-context-retrieval-max-vectors", type=int)
    parser.add_argument("--proposal-context-retrieval-cache-size", type=int)
    parser.add_argument("--proposal-match-window-frames", type=int)
    parser.add_argument(
        "--proposal-schedule-mode",
        choices=("gate", "periodic", "dense"),
        help=(
            "how the 8B verifier is scheduled. `gate` is the method; `periodic` "
            "and `dense` observe identically but ignore the matcher scores, so "
            "an accuracy/cost comparison isolates the trigger rule."
        ),
    )
    parser.add_argument("--proposal-schedule-period-seconds", type=float)
    parser.add_argument("--proposal-surge-threshold", type=float)
    parser.add_argument("--proposal-delta-lookback-seconds", type=float)
    parser.add_argument("--proposal-duplicate-similarity-threshold", type=float)
    parser.add_argument("--proposal-duplicate-cooldown-seconds", type=float)
    parser.add_argument("--proposal-initial-verify", type=parse_bool)
    parser.add_argument("--proposal-audit-only", type=parse_bool)
    parser.add_argument("--proposal-window-cache-size", type=int)
    parser.add_argument(
        "--proposal-match-encoding",
        choices=(
            "joint_window",
            "cached_frame_mean",
            "vit_feature_joint_decoder",
            "incremental_kv",
        ),
    )
    parser.add_argument("--proposal-frame-cache-size", type=int)
    parser.add_argument("--proposal-vit-cache-size", type=int)
    parser.add_argument("--proposal-max-new-tokens", type=int)
    parser.add_argument("--proposal-candidate-episode-seconds", type=float)
    parser.add_argument("--proposal-candidate-retry-interval-seconds", type=float)
    parser.add_argument("--proposal-manifest-path")
    parser.add_argument(
        "--proposal-manifest-texts-only", type=parse_bool, default=None,
        help="replay only the manifest's compiled proposal texts and re-encode "
             "them with the encoder under test, instead of reusing the stored "
             "vectors.  Required for an encoder ablation on a frozen manifest: "
             "reusing the vectors would match every arm against whichever "
             "encoder wrote the manifest.",
    )
    parser.add_argument(
        "--embedding-backend",
        default=None,
        help="which encoder hosts the gate: qwen3vl (default), clip, siglip, "
             "jina_clip, gme_qwen2vl.  Non-default backends require "
             "--proposal-match-encoding cached_frame_mean.",
    )
    parser.add_argument(
        "--verifier-prompt-mode", choices=list(VERIFIER_PROMPT_MODES), default=None,
        help="task_specific (default) phrases the 8B verifier per task; agnostic "
             "uses one standing-instruction wording for every task, to test "
             "whether the score depends on task-shaped phrasing.",
    )
    parser.add_argument("--embedding-model-path", default=None)
    parser.add_argument("--token-reduction", type=parse_bool)
    parser.add_argument("--token-reduction-layer", type=int)
    parser.add_argument("--token-reduction-budget", type=float)
    parser.add_argument("--token-reduction-parallel", type=parse_bool)
    parser.add_argument(
        "--token-reduction-selector",
        choices=("semantic", "stc_dynamic", "ctr", "random", "uniform", "norm"),
        help="Which token score drives selection at a fixed keep budget. "
             "'semantic' is the query-conditioned default; the rest are ablation arms.",
    )
    parser.add_argument(
        "--rec-controller",
        choices=("proposal_match", "interval_delta_8b"),
    )
    parser.add_argument("--proactive-head-checkpoint")
    parser.add_argument(
        "--proactive-head-threshold",
        type=float,
        default=None,
        help=(
            "override the gate threshold stored in the checkpoint; omit to use "
            "the checkpoint's own, which was selected on the training domain"
        ),
    )
    return parser.parse_args()


def distributed_context(verify_only: bool) -> tuple[int, int, int, torch.device | None]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if verify_only:
        return rank, world_size, local_rank, None
    if not torch.cuda.is_available():
        raise RuntimeError("OVO Forward evaluation requires CUDA")
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank, torch.device("cuda", local_rank)


def apply_overrides(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    values = {
        "sample_fps": args.sample_fps,
        "recent_window_frames": args.recent_window_frames,
        "rec_interval_max_frames": args.rec_interval_max_frames,
        "rec_sample_fps": args.rec_sample_fps,
        "decision_mode": args.decision_mode,
        "proposal_count": args.proposal_count,
        "proposal_prompt_style": args.proposal_prompt_style,
        "proposal_generation_window_frames": args.proposal_generation_window_frames,
        "proposal_context_retrieval_enabled": args.proposal_context_retrieval_enabled,
        "proposal_context_retrieval_top_k": args.proposal_context_retrieval_top_k,
        "proposal_context_retrieval_chunk_frames": (
            args.proposal_context_retrieval_chunk_frames
        ),
        "proposal_context_retrieval_max_vectors": (
            args.proposal_context_retrieval_max_vectors
        ),
        "proposal_context_retrieval_cache_size": (
            args.proposal_context_retrieval_cache_size
        ),
        "proposal_match_window_frames": args.proposal_match_window_frames,
        "proposal_schedule_mode": args.proposal_schedule_mode,
        "proposal_schedule_period_seconds": args.proposal_schedule_period_seconds,
        "proposal_surge_threshold": args.proposal_surge_threshold,
        "proposal_delta_lookback_seconds": args.proposal_delta_lookback_seconds,
        "proposal_duplicate_similarity_threshold": args.proposal_duplicate_similarity_threshold,
        "proposal_duplicate_cooldown_seconds": args.proposal_duplicate_cooldown_seconds,
        "proposal_initial_verify": args.proposal_initial_verify,
        "proposal_audit_only": args.proposal_audit_only,
        "proposal_window_cache_size": args.proposal_window_cache_size,
        "proposal_match_encoding": args.proposal_match_encoding,
        "proposal_frame_cache_size": args.proposal_frame_cache_size,
        "proposal_vit_cache_size": args.proposal_vit_cache_size,
        "proposal_max_new_tokens": args.proposal_max_new_tokens,
        "proposal_candidate_episode_seconds": args.proposal_candidate_episode_seconds,
        "proposal_candidate_retry_interval_seconds": (
            args.proposal_candidate_retry_interval_seconds
        ),
        "proposal_manifest_path": args.proposal_manifest_path,
        "proposal_manifest_texts_only": args.proposal_manifest_texts_only,
        "rec_controller": args.rec_controller,
        "incremental_kv_reset_every_n_frames": args.incremental_kv_reset_every_n_frames,
        "verifier_accept_margin": args.verifier_accept_margin,
        "answer_model_path": args.answer_model_path,
        "answer_model_family": args.answer_model_family,
        "rec_prompt_style": args.rec_prompt_style,
        "rec_max_new_tokens": args.rec_max_new_tokens,
        "proactive_head_checkpoint": args.proactive_head_checkpoint,
        "proactive_head_threshold": args.proactive_head_threshold,
        "token_reduction_enabled": args.token_reduction,
        "token_reduction_layer": args.token_reduction_layer,
        "token_reduction_budget": args.token_reduction_budget,
        "token_reduction_parallel": args.token_reduction_parallel,
        "token_reduction_selector": args.token_reduction_selector,
        "embedding_backend": args.embedding_backend,
        "verifier_prompt_mode": args.verifier_prompt_mode,
        "embedding_model_path": args.embedding_model_path,
    }
    for key, value in values.items():
        if value is not None:
            cfg[key] = value
    cfg.setdefault("sample_fps", 1.0)
    cfg.setdefault("recent_window_frames", 4)
    cfg.setdefault("rec_interval_max_frames", 16)
    cfg.setdefault("rec_sample_fps", 1.0)
    cfg.setdefault("decision_mode", "proposal_match")
    cfg.setdefault("proposal_count", 4)
    cfg.setdefault("proposal_prompt_style", "target_anchored_v2")
    cfg.setdefault("proposal_generation_window_frames", 5)
    cfg.setdefault("proposal_context_retrieval_enabled", False)
    cfg.setdefault("proposal_context_retrieval_top_k", 1)
    cfg.setdefault("proposal_context_retrieval_chunk_frames", 2)
    cfg.setdefault("proposal_context_retrieval_max_vectors", 128)
    cfg.setdefault("proposal_context_retrieval_cache_size", 4096)
    cfg.setdefault("proposal_match_window_frames", 2)
    cfg.setdefault("proposal_schedule_mode", "gate")
    cfg.setdefault("proposal_schedule_period_seconds", 4.0)
    cfg.setdefault("proposal_surge_threshold", 0.035)
    cfg.setdefault("proposal_delta_lookback_seconds", 1.0)
    cfg.setdefault("proposal_require_above_history_mean", True)
    cfg.setdefault("proposal_duplicate_similarity_threshold", 0.8)
    cfg.setdefault("proposal_duplicate_cooldown_seconds", 5.0)
    cfg.setdefault("proposal_initial_verify", True)
    cfg.setdefault("proposal_audit_only", False)
    cfg.setdefault("proposal_manifest_texts_only", False)
    cfg.setdefault("proposal_max_new_tokens", 256)
    cfg.setdefault("proposal_window_cache_size", 64)
    cfg.setdefault("proposal_match_encoding", "incremental_kv")
    cfg.setdefault("proposal_frame_cache_size", 256)
    cfg.setdefault("proposal_vit_cache_size", 8)
    cfg.setdefault("proposal_candidate_episode_seconds", 2.0)
    cfg.setdefault("proposal_candidate_retry_interval_seconds", 1.0)
    cfg.setdefault("proposal_manifest_path", "")
    if str(cfg["proposal_manifest_path"]).strip():
        manifest_metadata = inspect_proposal_manifest(cfg["proposal_manifest_path"])
        cfg.update(manifest_metadata)
        cfg["proposal_count"] = int(
            manifest_metadata["proposal_manifest_max_proposal_count"]
        )
    cfg.setdefault("rec_controller", "interval_delta_8b")
    cfg.setdefault("incremental_kv_reset_every_n_frames", 0)
    # The verifier accept margin defaults per backbone (see verifier_margin_defaults);
    # an explicit command-line value takes precedence.
    if cfg.get("verifier_accept_margin") is None:
        cfg["verifier_accept_margin"] = default_verifier_accept_margin(
            cfg.get("answer_model_family"), cfg.get("answer_model_path")
        )
    cfg.setdefault("answer_model_family", "qwen3vl")
    cfg.setdefault("rec_prompt_style", "integer_only")
    cfg.setdefault("rec_max_new_tokens", 320)
    cfg.setdefault("token_reduction_enabled", False)
    cfg.setdefault("token_reduction_layer", 12)
    cfg.setdefault("token_reduction_budget", 0.1)
    cfg.setdefault("token_reduction_parallel", True)
    cfg.setdefault("token_reduction_selector", "semantic")
    cfg.setdefault("embedding_backend", "qwen3vl")
    cfg.setdefault("verifier_prompt_mode", "task_specific")
    set_verifier_prompt_mode(str(cfg["verifier_prompt_mode"]))
    # SemanticMemoryEncoder requires the field, although the gate does not
    # construct a persistent temporal memory.
    cfg["memory_chunk_frames"] = int(
        cfg["proposal_match_window_frames"]
        if cfg["decision_mode"] == "proposal_match"
        else cfg["recent_window_frames"]
    )
    if float(cfg["sample_fps"]) <= 0:
        raise ValueError("sample_fps must be positive")
    if int(cfg["recent_window_frames"]) <= 0:
        raise ValueError("recent_window_frames must be positive")
    if int(cfg["rec_interval_max_frames"]) <= 0:
        raise ValueError("rec_interval_max_frames must be positive")
    if float(cfg["rec_sample_fps"]) <= 0:
        raise ValueError("rec_sample_fps must be positive")
    if int(cfg["proposal_match_window_frames"]) <= 0:
        raise ValueError("proposal_match_window_frames must be positive")
    for name in (
        "proposal_context_retrieval_top_k",
        "proposal_context_retrieval_chunk_frames",
        "proposal_context_retrieval_max_vectors",
        "proposal_context_retrieval_cache_size",
    ):
        if int(cfg[name]) <= 0:
            raise ValueError(f"{name} must be positive")
    if bool(cfg["proposal_context_retrieval_enabled"]) and int(
        cfg["proposal_context_retrieval_chunk_frames"]
    ) != int(cfg["proposal_match_window_frames"]):
        raise ValueError(
            "proposal_context_retrieval_chunk_frames must equal "
            "proposal_match_window_frames for the shared 2B encoder"
        )
    if bool(cfg["proposal_context_retrieval_enabled"]) and str(
        cfg["decision_mode"]
    ) != "proposal_match":
        raise ValueError(
            "proposal_context_retrieval_enabled requires decision_mode=proposal_match"
        )
    if bool(cfg["proposal_context_retrieval_enabled"]) and str(
        cfg.get("proposal_manifest_path", "")
    ).strip():
        raise ValueError(
            "proposal_context_retrieval_enabled cannot be combined with "
            "proposal_manifest_path because manifest replay skips compilation"
        )
    if int(cfg["proposal_window_cache_size"]) <= 0:
        raise ValueError("proposal_window_cache_size must be positive")
    if int(cfg["proposal_frame_cache_size"]) <= 0:
        raise ValueError("proposal_frame_cache_size must be positive")
    if int(cfg["proposal_vit_cache_size"]) <= 0:
        raise ValueError("proposal_vit_cache_size must be positive")
    if float(cfg["proposal_candidate_episode_seconds"]) < 0:
        raise ValueError("proposal_candidate_episode_seconds cannot be negative")
    if float(cfg["proposal_candidate_retry_interval_seconds"]) <= 0:
        raise ValueError(
            "proposal_candidate_retry_interval_seconds must be positive"
        )
    if int(cfg["token_reduction_layer"]) == 0 or int(
        cfg["token_reduction_layer"]
    ) < -1:
        raise ValueError(
            "token_reduction_layer must be -1 or a positive 1-based layer id"
        )
    if float(cfg["token_reduction_budget"]) <= 0:
        raise ValueError("token_reduction_budget must be positive")
    if bool(cfg["token_reduction_enabled"]) and cfg["decision_mode"] != "proposal_match":
        raise ValueError(
            "Proactive token reduction is defined for the proposal_match sparse "
            "8B verifier; use decision_mode=proposal_match."
        )
    return cfg


def output_config_snapshot(cfg: dict[str, Any], tasks: list[str]) -> dict[str, Any]:
    keep = {
        "answer_model_path",
        "embedding_model_path",
        "embedding_helper_dir",
        "annotation_path",
        "video_root",
        "sample_fps",
        "recent_window_frames",
        "rec_interval_max_frames",
        "rec_sample_fps",
        "decision_mode",
        "proactive_head_checkpoint",
        "proactive_head_threshold",
        "proposal_count",
        "proposal_prompt_style",
        "proposal_generation_window_frames",
        "proposal_context_retrieval_enabled",
        "proposal_context_retrieval_top_k",
        "proposal_context_retrieval_chunk_frames",
        "proposal_context_retrieval_max_vectors",
        "proposal_context_retrieval_cache_size",
        "proposal_match_window_frames",
        "proposal_schedule_mode",
        "proposal_schedule_period_seconds",
        "proposal_surge_threshold",
        "proposal_delta_lookback_seconds",
        "proposal_require_above_history_mean",
        "proposal_duplicate_similarity_threshold",
        "proposal_duplicate_cooldown_seconds",
        "proposal_initial_verify",
        "proposal_audit_only",
        "proposal_max_new_tokens",
        "proposal_window_cache_size",
        "proposal_match_encoding",
        "proposal_frame_cache_size",
        "proposal_vit_cache_size",
        "proposal_candidate_episode_seconds",
        "proposal_candidate_retry_interval_seconds",
        "proposal_manifest_path",
        "proposal_manifest_sha256",
        "proposal_manifest_entry_count",
        "proposal_manifest_min_proposal_count",
        "proposal_manifest_max_proposal_count",
        "proposal_manifest_proposal_counts",
        "proposal_manifest_prompt_versions",
        "rec_controller",
        "rec_prompt_style",
        "rec_max_new_tokens",
        "attn_implementation",
        "dtype",
        "max_new_tokens",
        "token_reduction_enabled",
        "token_reduction_layer",
        "token_reduction_selector",
        "embedding_backend",
        "verifier_prompt_mode",
        "token_reduction_stc_interval",
        "token_reduction_budget",
        "token_reduction_parallel",
    }
    snapshot = {
        **{key: value for key, value in cfg.items() if key in keep},
        "tasks": tasks,
        "controller": experiment_controller_name(cfg, tasks),
        "controller_version": CONTROLLER_VERSION,
        "shared_update_gate_across_tasks": (
            cfg["decision_mode"] == "update_head" and any(
                task in tasks for task in ("SSR", "CRR")
            )
        ),
        "task_protocols": {
            "REC": (
                "shared_proposal_match_verified_integer_delta_accumulator"
                if unified_rec_enabled(cfg)
                else "interval_delta_count_8b_plus_programmatic_accumulator"
            ),
            "SSR": (
                "proposal_match_verified_state_change_with_hold"
                if cfg["decision_mode"] == "proposal_match"
                else f"{cfg['decision_mode']}_current_state"
            ),
            "CRR": (
                "proposal_match_bounded_retry_absorbing_readiness"
                if cfg["decision_mode"] == "proposal_match"
                else f"{cfg['decision_mode']}_absorbing_readiness"
            ),
        },
    }
    if not bool(cfg.get("proposal_context_retrieval_enabled", False)):
        for retrieval_only in (
            "proposal_context_retrieval_enabled",
            "proposal_context_retrieval_top_k",
            "proposal_context_retrieval_chunk_frames",
            "proposal_context_retrieval_max_vectors",
            "proposal_context_retrieval_cache_size",
        ):
            snapshot.pop(retrieval_only, None)
    if cfg.get("proposal_manifest_path"):
        for compiler_only in (
            "proposal_prompt_style",
            "proposal_generation_window_frames",
            "proposal_max_new_tokens",
        ):
            snapshot.pop(compiler_only, None)
    checkpoint = cfg.get("proactive_head_checkpoint")
    if (
        cfg["decision_mode"] == "update_head"
        and any(task in tasks for task in ("SSR", "CRR"))
        and checkpoint
        and Path(checkpoint).is_file()
    ):
        snapshot["proactive_head_checkpoint_sha256"] = hashlib.sha256(
            Path(checkpoint).read_bytes()
        ).hexdigest()
        head_checkpoint = torch.load(checkpoint, map_location="cpu", weights_only=True)
        snapshot["proactive_head_training_objective"] = head_checkpoint.get(
            "training_objective"
        )
        snapshot["proactive_head_training_objective_version"] = head_checkpoint.get(
            "training_objective_version"
        )
        snapshot["proactive_head_loss_config"] = head_checkpoint.get("loss_config")
        snapshot["proactive_head_operating_point"] = head_checkpoint.get(
            "validation", {}
        ).get("candidate_operating_point")
    return snapshot


def experiment_controller_name(cfg: dict[str, Any], tasks: list[str]) -> str:
    binary_tasks = [task for task in tasks if task in {"SSR", "CRR"}]
    if "REC" in tasks and not binary_tasks:
        return (
            controller_name(cfg["decision_mode"]) + "-rec-integer-delta"
            if unified_rec_enabled(cfg)
            else "interval-delta-count-8b"
        )
    if "REC" in tasks:
        if unified_rec_enabled(cfg):
            return controller_name(cfg["decision_mode"]) + "-unified-forward-state"
        return (
            "rec-interval-delta-count-8b-plus-"
            + controller_name(cfg["decision_mode"])
        )
    return controller_name(cfg["decision_mode"])


def default_output_suffix(cfg: dict[str, Any], tasks: list[str]) -> str:
    suffix = "forward"
    has_binary_task = any(task in tasks for task in ("SSR", "CRR"))
    has_proposal_task = bool(
        has_binary_task or ("REC" in tasks and unified_rec_enabled(cfg))
    )
    if "REC" in tasks:
        suffix += (
            "-recunified"
            if unified_rec_enabled(cfg)
            else "-recdelta8b"
        )
    if has_proposal_task:
        if cfg["decision_mode"] == "proposal_match":
            method = (
                "proposalmatchrearm"
                f"-p{int(cfg['proposal_count'])}"
                f"-win{int(cfg['proposal_match_window_frames'])}"
                f"-d{float(cfg['proposal_surge_threshold']):g}"
                + (
                    ""
                    if cfg.get("proposal_schedule_mode", "gate") == "gate"
                    else f"-{cfg['proposal_schedule_mode']}"
                    + (
                        f"{float(cfg['proposal_schedule_period_seconds']):g}s"
                        if cfg["proposal_schedule_mode"] == "periodic"
                        else ""
                    )
                )
            )
            if cfg.get("proposal_manifest_path"):
                versions = cfg.get("proposal_manifest_prompt_versions", ["unknown"])
                version_tag = re.sub(r"[^A-Za-z0-9]+", "_", "+".join(versions)).strip("_")
                digest = str(cfg.get("proposal_manifest_sha256", "unknown"))[:10]
                method += f"-pm{version_tag}-{digest}"
            else:
                prompt_tags = {
                    "target_anchored_v2": "v2",
                    "target_anchor_plus3_v7": "v7",
                }
                prompt_style = cfg.get("proposal_prompt_style")
                if prompt_style is not None:
                    method += f"-pc{prompt_tags[str(prompt_style)]}"
            if float(cfg.get("proposal_delta_lookback_seconds", 1.0)) != 1.0:
                method += f"-lb{float(cfg['proposal_delta_lookback_seconds']):g}"
            if float(cfg.get("proposal_duplicate_similarity_threshold", 0.8)) != 0.8:
                method += (
                    f"-dupsim{float(cfg['proposal_duplicate_similarity_threshold']):g}"
                )
            if float(cfg.get("proposal_duplicate_cooldown_seconds", 5.0)) != 5.0:
                method += (
                    f"-dupcd{float(cfg['proposal_duplicate_cooldown_seconds']):g}"
                )
            if (
                cfg.get("proposal_match_encoding", "joint_window") == "joint_window"
                and int(cfg.get("proposal_window_cache_size", 64)) != 64
            ):
                method += f"-wcache{int(cfg['proposal_window_cache_size'])}"
            encoding = cfg.get("proposal_match_encoding", "joint_window")
            if encoding == "cached_frame_mean":
                method += f"-encframe_mean-fcache{int(cfg['proposal_frame_cache_size'])}"
            elif encoding == "vit_feature_joint_decoder":
                method += f"-encvit_joint-vcache{int(cfg['proposal_vit_cache_size'])}"
            elif encoding == "incremental_kv":
                method += f"-enckv-kvwin{int(cfg['proposal_match_window_frames'])}"
            if not cfg.get("proposal_manifest_path") and int(cfg.get("proposal_generation_window_frames", 5)) != 5:
                method += f"-gctx{int(cfg['proposal_generation_window_frames'])}"
            if bool(cfg.get("proposal_context_retrieval_enabled", False)):
                method += (
                    "-rctx"
                    f"-k{int(cfg['proposal_context_retrieval_top_k'])}"
                    f"-c{int(cfg['proposal_context_retrieval_chunk_frames'])}"
                    f"-m{int(cfg['proposal_context_retrieval_max_vectors'])}"
                )
            if not cfg.get("proposal_manifest_path") and int(cfg.get("proposal_max_new_tokens", 256)) != 256:
                method += f"-gnew{int(cfg['proposal_max_new_tokens'])}"
            if cfg.get("proposal_audit_only"):
                method += "-audit"
            else:
                method += (
                    f"-ep{float(cfg['proposal_candidate_episode_seconds']):g}"
                    f"-ri{float(cfg['proposal_candidate_retry_interval_seconds']):g}"
                )
        else:
            method = (
                "embcalv9"
                if cfg["decision_mode"] == "update_head"
                else cfg["decision_mode"].replace("_", "")
            )
        suffix += f"-{method}"
        if (
            cfg["decision_mode"] == "proposal_match"
            and cfg.get("proposal_initial_verify")
        ):
            suffix += "-init"
    if has_proposal_task:
        suffix += f"-fps{float(cfg['sample_fps']):g}"
    if "REC" in tasks and not unified_rec_enabled(cfg):
        rec_fps = float(cfg.get("rec_sample_fps", cfg["sample_fps"]))
        if not has_binary_task or rec_fps != float(cfg["sample_fps"]):
            suffix += f"-recfps{rec_fps:g}"
        suffix += f"-recmax{int(cfg['rec_interval_max_frames'])}"
        if str(cfg.get("rec_prompt_style", "integer_only")) != "integer_only":
            suffix += f"-recprompt{cfg['rec_prompt_style']}"
            if int(cfg.get("rec_max_new_tokens", 320)) != 320:
                suffix += f"-rectok{int(cfg['rec_max_new_tokens'])}"
    if has_proposal_task and not (
        cfg["decision_mode"] == "proposal_match"
        and cfg.get("proposal_audit_only")
    ):
        suffix += f"-recent{int(cfg['recent_window_frames'])}"
    if tuple(tasks) != FORWARD_TASKS:
        suffix += "-tasks" + "_".join(tasks)
    if bool(cfg.get("token_reduction_enabled", False)):
        suffix += (
            f"-tr-trlayer{int(cfg['token_reduction_layer'])}"
            f"-trbudget{float(cfg['token_reduction_budget']):g}"
            + (
                "-trparallel"
                if bool(cfg.get("token_reduction_parallel", True))
                else "-trsequential"
            )
        )
    return suffix


def rec_interval_positions(
    sampler: VideoSampler,
    *,
    start_exclusive: float,
    end_inclusive: float,
    max_frames: int,
) -> tuple[list[int], int]:
    """Select only newly observed REC samples, uniformly capping long intervals."""

    positions = [
        index
        for index, timestamp in enumerate(sampler.sample_timestamps)
        if float(timestamp) > float(start_exclusive) + 1e-6
        and float(timestamp) <= float(end_inclusive) + 1e-6
    ]
    before_cap = len(positions)
    if before_cap > int(max_frames):
        subset = np.linspace(0, before_cap - 1, int(max_frames)).round().astype(np.int64)
        positions = [positions[int(index)] for index in subset]
    return positions, before_cap


def rec_delta_prompt(
    *,
    activity: str,
    start_exclusive: float,
    end_inclusive: float,
    prompt_style: str = "integer_only",
) -> str:
    if prompt_style == "integer_only":
        return (
            "The independent images above are timestamped causal observations from "
            "only the NEW interval "
            f"({start_exclusive:.3f}, {end_inclusive:.3f}] seconds, in chronological order.\n"
            f"Target repeated activity: {activity}\n"
            "Count each newly COMPLETED occurrence of the target activity in this "
            "new interval exactly once. Do not count setup, an incomplete occurrence, "
            "a pose that merely resembles the activity, or repeated views of the same "
            "occurrence. If no occurrence is completed in this interval, answer 0.\n"
            "Answer with exactly the number of NEW completions in this interval as a "
            "single non-negative integer and no other text."
        )
    if prompt_style == "conservative":
        # Qwen2-VL-7B over-counts on 63.8% of REC checkpoints against Qwen3's
        # 44.3%, with a long tail (+3 on 70 checkpoints, +4 on 42, +5 on 33),
        # and every response parses, so this is not a format problem -- the
        # model counts one occurrence several times. The existing wording lists
        # what not to count but never says which way to resolve a genuine tie,
        # so an uncertain model drifts upward. This adds only that tie-break,
        # leaking no dataset statistics into the prompt.
        return (
            "The independent images above are timestamped causal observations from "
            "only the NEW interval "
            f"({start_exclusive:.3f}, {end_inclusive:.3f}] seconds, in chronological order.\n"
            f"Target repeated activity: {activity}\n"
            "Count each newly COMPLETED occurrence of the target activity in this "
            "new interval exactly once. Do not count setup, an incomplete occurrence, "
            "a pose that merely resembles the activity, or repeated views of the same "
            "occurrence. Several images may show one single occurrence at different "
            "moments; that is still one occurrence.\n"
            "If you are unsure whether something counts, do NOT count it. When two "
            "readings are equally plausible, answer the smaller number. If no "
            "occurrence is completed in this interval, answer 0.\n"
            "Answer with exactly the number of NEW completions in this interval as a "
            "single non-negative integer and no other text."
        )
    if prompt_style == "enumerate_then_count":
        # Error-case analysis on the interval_delta_8b counter (2026-09-05):
        # wrong checkpoints are dominated by over-counting (311 over vs 113
        # under; delta=+1 is the single most common error), even on short,
        # uncapped intervals -- so it is not a frame-sampling problem. Forcing
        # a bare integer with zero reasoning space seems to invite the model
        # to report a plausible-looking count rather than one it can justify.
        # This style asks for an explicit, falsifiable list first.
        return (
            "The independent images above are timestamped causal observations from "
            "only the NEW interval "
            f"({start_exclusive:.3f}, {end_inclusive:.3f}] seconds, in chronological order.\n"
            f"Target repeated activity: {activity}\n"
            "List each newly COMPLETED occurrence of the target activity visible in "
            "this new interval, one per line, as:\n"
            "- <approximate timestamp> seconds: <one short phrase citing the visual "
            "evidence that this occurrence actually completed>\n"
            "Only list an occurrence if it is clearly COMPLETED inside this interval. "
            "Do not list setup, an occurrence still in progress, a pose that merely "
            "resembles the activity, or the same occurrence more than once. If no "
            "occurrence completes in this interval, write \"None\".\n"
            "After the list, on its own final line, write exactly:\n"
            "Final count: <the number of items you listed above, as a single "
            "non-negative integer>"
        )
    raise ValueError(f"Unknown rec_prompt_style: {prompt_style!r}")


def normalize_rec_delta(
    response: str, *, prompt_style: str = "integer_only"
) -> tuple[int, bool]:
    if prompt_style == "enumerate_then_count":
        # Prefer the explicit final tag so intervening timestamps in the
        # enumerated list can never be mistaken for the count itself. Search
        # from the end so a stray earlier mention of the words doesn't win.
        matches = list(
            re.finditer(r"final\s*count\s*[:\-]?\s*(\d+)", str(response), re.IGNORECASE)
        )
        if matches:
            return int(matches[-1].group(1)), True
        # No compliant tag: count listed bullet lines as a fallback signal
        # rather than silently defaulting to 0, but flag it as unparsed so
        # this failure mode stays visible in the metrics.
        bullets = re.findall(r"^\s*-\s", str(response), re.MULTILINE)
        return len(bullets), False
    match = re.search(r"\d+", str(response))
    if match is None:
        return 0, False
    return int(match.group(0)), True


def query_specs(item: dict[str, Any]) -> list[dict[str, Any]]:
    if item["task"] == "REC":
        return [{
            "query_id": "rec",
            "registration_timestamp": 0.0,
            "end_timestamp": max(float(value["realtime"]) for value in item["test_info"]),
            "query_text": str(item["activity"]),
            "test_indices": list(range(len(item["test_info"]))),
        }]
    if item["task"] == "CRR":
        return [{
            "query_id": "crr",
            "registration_timestamp": float(item["ask_time"]),
            "end_timestamp": max(float(value["realtime"]) for value in item["test_info"]),
            "query_text": str(item["question"]),
            "test_indices": list(range(len(item["test_info"]))),
        }]
    grouped: dict[str, list[int]] = {}
    for index, test in enumerate(item["test_info"]):
        grouped.setdefault(str(test["step"]), []).append(index)
    return [
        {
            "query_id": f"step-{order}",
            "registration_timestamp": min(float(item["test_info"][index]["realtime"]) for index in indices),
            "end_timestamp": max(float(item["test_info"][index]["realtime"]) for index in indices),
            "query_text": step,
            "test_indices": indices,
        }
        for order, (step, indices) in enumerate(grouped.items())
    ]


def longest_prefix_candidate(item: dict[str, Any], root: str | Path) -> Path:
    index = max(range(len(item["test_info"])), key=lambda value: float(item["test_info"][value]["realtime"]))
    return Path(root) / f"{item['id']}_{index}.mp4"


def forward_verification_report(annotations: list[dict[str, Any]], cfg: dict[str, Any]) -> dict[str, Any]:
    fatal = []
    active_tasks = [
        task for task in FORWARD_TASKS if any(value["task"] == task for value in annotations)
    ]
    for item in annotations:
        path = longest_prefix_candidate(item, cfg["video_root"])
        required = max(float(value["realtime"]) for value in item["test_info"])
        problems = []
        duration = None
        if not path.is_file():
            problems.append("missing_longest_prefix")
        else:
            try:
                duration, _ = probe_video_duration_seconds(path)
                if duration + 1.0 < required:
                    problems.append("longest_prefix_too_short")
            except Exception as error:
                problems.append(f"video_probe_failed:{type(error).__name__}:{error}")
        if problems:
            fatal.append({"id": item["id"], "task": item["task"], "path": str(path), "required": required, "duration": duration, "problems": problems})
    report = {
        "num_meta_annotations": len(annotations),
        "num_test_points": sum(len(value["test_info"]) for value in annotations),
        "task_test_point_counts": {task: sum(len(value["test_info"]) for value in annotations if value["task"] == task) for task in FORWARD_TASKS},
        "fatal_prefix_problems": fatal,
        "controller": experiment_controller_name(cfg, active_tasks),
        "ground_truth_visible_to_models": False,
        "state_semantics": {"REC": "cumulative count", "SSR": "reversible current state", "CRR": "absorbing readiness"},
        "protocol": (
            "REC shares the frozen 8B proposal compiler, bounded incremental "
            "Embedding-2B proposal matcher and sparse cached 8B verifier with "
            "SSR/CRR; verified REC A/B/C/D deltas update a programmatic integer "
            "accumulator."
            if unified_rec_enabled(cfg)
            and "REC" in active_tasks
            else (
            "REC uses an 8B disjoint-interval delta counter with a programmatic "
            "cumulative accumulator."
            if active_tasks == ["REC"]
            else (
                "REC uses an 8B disjoint-interval delta counter with a programmatic "
                "cumulative accumulator. SSR/CRR share one cached Embedding-2B "
                "readout with native Yes/No cosines. An event-coverage and "
                "call-sparsity trained temporal head proposes candidates; raw cosine "
                "cannot bypass it. A bounded-KV frozen 8B verifier makes every final "
                "state change."
                if cfg["decision_mode"] == "update_head"
                else (
                    "REC uses an 8B disjoint-interval delta counter. At query "
                    "registration, frozen Qwen3-VL-8B compiles observable proposals "
                    "for SSR/CRR. Frozen Embedding-2B matches causal sliding windows; "
                    "a one-second proposal-score surge opens a bounded retry episode "
                    "for the cached 8B verifier. Unresolved expiry re-arms duplicate "
                    "detection instead of leaving a cooldown dead zone. SSR holds "
                    "its previous state between verified changes. REC remains an "
                    "independent disjoint-interval "
                    f"counter sampled at {float(cfg['rec_sample_fps']):g} fps."
                    if cfg["decision_mode"] == "proposal_match"
                    else "REC uses an 8B disjoint-interval delta counter. " + (
                "SSR/CRR initialize from latest recent independent images, then "
                "append only each newly arrived image with bounded multimodal KV; "
                f"using the selected {cfg['decision_mode']} controller."
                )
                )
            )
            )
        )
        + (
            " In the opt-in retrieval-context branch, Embedding-2B retrieves "
            "completed historical chunks using only the registered SSR/CRR "
            "target; retrieved frames augment recent frames only for one-time "
            "proposal compilation. REC interval counting is unchanged."
            if bool(cfg.get("proposal_context_retrieval_enabled", False))
            else ""
        ),
    }
    if bool(cfg.get("proposal_context_retrieval_enabled", False)):
        report["proposal_context_retrieval"] = {
            "enabled": True,
            "affected_tasks": [
                task for task in active_tasks if task in {"SSR", "CRR"}
            ],
            "query": "registered task target only",
            "top_k_chunks": int(cfg["proposal_context_retrieval_top_k"]),
            "chunk_frames": int(cfg["proposal_context_retrieval_chunk_frames"]),
            "max_memory_vectors": int(
                cfg["proposal_context_retrieval_max_vectors"]
            ),
            "causality": (
                "completed pre-registration chunks only; retrieved chunks "
                "cannot overlap the recent proposal context"
            ),
        }
    return report


def run_rec_annotation(
    *,
    item: dict[str, Any],
    answer_model: AnswerModel,
    encoder: SemanticMemoryEncoder | None,
    cfg: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Count disjoint-interval completion deltas and accumulate them in code.

    REC is a count-state task, not an answer/wait task.  The model is called
    only at released REC checkpoints and sees only samples not processed at a
    previous checkpoint. Its integer output is only that interval's increment,
    so a wrong cumulative value cannot recursively contaminate later prompts.
    """

    if item["task"] != "REC":
        raise ValueError("run_rec_annotation only accepts REC annotations")
    path = longest_prefix_candidate(item, cfg["video_root"])
    rec_sample_fps = float(cfg.get("rec_sample_fps", cfg["sample_fps"]))
    rec_cfg = {**cfg, "sample_fps": rec_sample_fps}
    sampler = VideoSampler(str(path), rec_cfg, causal_fixed_fps=True)
    output = copy.deepcopy(item["test_info"])
    previous_boundary = -1.0
    cumulative_count = 0
    decisions: list[dict[str, Any]] = []
    answer_calls: list[dict[str, Any]] = []
    answer_seconds = 0.0
    started = time.perf_counter()

    ordered_indices = sorted(
        range(len(output)), key=lambda index: (float(output[index]["realtime"]), index)
    )
    for test_index in ordered_indices:
        timestamp = float(output[test_index]["realtime"])
        positions, before_cap = rec_interval_positions(
            sampler,
            start_exclusive=previous_boundary,
            end_inclusive=timestamp,
            max_frames=int(cfg["rec_interval_max_frames"]),
        )
        metrics: dict[str, Any] = {
            "query_id": "rec",
            "query_timestamp_seconds": timestamp,
            "task_native_state": "REC_interval_delta_plus_cumulative_accumulator",
            "shared_wait_state": False,
            "gate_2b_called": False,
            "verifier_8b_called": False,
            "rec_counter_8b_called": False,
            "rec_interval_start_exclusive_seconds": previous_boundary,
            "rec_interval_end_inclusive_seconds": timestamp,
            "rec_interval_frames_before_cap": before_cap,
            "rec_interval_frames_after_cap": len(positions),
            "rec_interval_max_frames": int(cfg["rec_interval_max_frames"]),
            "rec_sample_fps": rec_sample_fps,
            "rec_interval_frame_selection": "uniform_if_over_configured_cap",
            "rec_prompt_style": str(cfg.get("rec_prompt_style", "integer_only")),
        }
        raw_response = ""
        parsed = False
        if positions:
            decode_started = time.perf_counter()
            frames = sampler.decode_positions(positions)
            timestamps = sampler.timestamps(positions)
            decode_seconds = time.perf_counter() - decode_started
            rec_prompt_style = str(cfg.get("rec_prompt_style", "integer_only"))
            prompt = rec_delta_prompt(
                activity=str(item["activity"]),
                start_exclusive=previous_boundary,
                end_inclusive=timestamp,
                prompt_style=rec_prompt_style,
            )
            token_reduction_query = None
            token_reduction_query_seconds = 0.0
            if bool(cfg.get("token_reduction_enabled", False)):
                if encoder is None:
                    raise RuntimeError(
                        "REC token reduction requires the active Embedding-2B encoder"
                    )
                token_reduction_query, token_reduction_query_seconds = (
                    encoder.query_vector(prompt)
                )
            raw_response, answer_metrics = answer_model.answer(
                video=frames,
                prompt=prompt,
                fps=rec_sample_fps,
                max_frames=len(frames),
                frame_timestamps=timestamps,
                frame_context_labels=[
                    f"New-interval observation at {value:.3f} seconds:"
                    for value in timestamps
                ],
                token_reduction_encoder=encoder,
                token_reduction_query=token_reduction_query,
                max_new_tokens_override=(
                    int(cfg.get("rec_max_new_tokens", 320))
                    if rec_prompt_style == "enumerate_then_count"
                    else None
                ),
            )
            interval_delta, parsed = normalize_rec_delta(
                raw_response, prompt_style=rec_prompt_style
            )
            cumulative_count += interval_delta
            answer_seconds += float(answer_metrics.get("answer_total_seconds", 0.0))
            metrics.update(
                {
                    **answer_metrics,
                    "evidence_decode_seconds": decode_seconds,
                    "token_reduction_query_seconds": (
                        token_reduction_query_seconds
                    ),
                    "token_reduction_scope": (
                        "rec_new_interval_frames_before_8b_llm"
                        if bool(cfg.get("token_reduction_enabled", False))
                        else "disabled"
                    ),
                    "rec_counter_8b_called": True,
                    "selected_sample_positions": positions,
                    "selected_raw_frame_indices": [sampler.indices[value] for value in positions],
                    "selected_timestamps_seconds": timestamps,
                    "answer_frames": len(frames),
                    "answer_raw_response": raw_response,
                    "answer_delta_parse_succeeded": parsed,
                    "answer_normalized_interval_delta": interval_delta,
                }
            )
            answer_calls.append(
                {
                    "timestamp_seconds": timestamp,
                    "response": str(cumulative_count),
                    "metrics": copy.deepcopy(metrics),
                }
            )
        else:
            metrics["decision_reason"] = "no_new_causal_frame_carry_count"

        metrics["answer_normalized_cumulative_count"] = str(cumulative_count)
        metrics["answer_model_called_at_this_test"] = bool(positions)
        output[test_index]["response"] = str(cumulative_count)
        output[test_index]["metrics"] = metrics
        decisions.append(
            {
                "timestamp_seconds": timestamp,
                "metrics": copy.deepcopy(metrics),
            }
        )
        previous_boundary = timestamp

    rec_calls = sum(
        bool(value["metrics"].get("rec_counter_8b_called")) for value in decisions
    )
    rec_token_reduction_calls = sum(
        bool(value["metrics"].get("token_reduction_enabled"))
        and bool(value["metrics"].get("rec_counter_8b_called"))
        for value in decisions
    )
    rec_token_reduction_seconds = sum(
        float(value["metrics"].get("token_reduction_seconds", 0.0))
        for value in decisions
    )
    rec_token_reduction_query_seconds = sum(
        float(value["metrics"].get("token_reduction_query_seconds", 0.0))
        for value in decisions
    )
    rec_visual_tokens_before = sum(
        int(value["metrics"].get("answer_visual_tokens_before_reduction", 0))
        for value in decisions
    )
    rec_visual_tokens_after = sum(
        int(value["metrics"].get("answer_visual_tokens", 0))
        for value in decisions
        if value["metrics"].get("token_reduction_enabled")
    )
    episode = {
        "query_id": "rec",
        "query_text": f"How many times was '{item['activity']}' completed?",
        "registration_timestamp": 0.0,
        "registration_metrics": {
            "task_profile": "interval_delta_count_8b_plus_programmatic_accumulator",
            "rec_interval_max_frames": int(cfg["rec_interval_max_frames"]),
            "rec_sample_fps": rec_sample_fps,
        },
        "task_state_decisions": decisions,
        "gate_2b_calls": 0,
        "gate_positive_count": 0,
        "gate_2b_seconds": 0.0,
        "verifier_8b_calls": 0,
        "verifier_positive_count": 0,
        "verifier_8b_seconds": 0.0,
        "rec_counter_8b_calls": rec_calls,
        "rec_counter_8b_seconds": answer_seconds,
        "token_reduction_calls": rec_token_reduction_calls,
        "token_reduction_seconds": rec_token_reduction_seconds,
        "token_reduction_query_seconds": rec_token_reduction_query_seconds,
        "answer_calls": answer_calls,
    }
    result = {
        **{key: copy.deepcopy(value) for key, value in item.items() if not key.startswith("_")},
        "test_info": output,
        "proactive_controller": "interval-delta-count-8b",
        "cascade_episodes": [episode],
        "gate_2b_call_count": 0,
        "gate_positive_count": 0,
        "verifier_8b_call_count": 0,
        "verifier_positive_count": 0,
        "rec_counter_8b_call_count": rec_calls,
        "total_8b_call_count": rec_calls,
    }
    source = {
        "source_key": f"REC:{item['id']}",
        "task": "REC",
        "source_online_wall_seconds": time.perf_counter() - started,
        "gate_2b_calls": 0,
        "gate_positive_count": 0,
        "gate_2b_seconds": 0.0,
        "verifier_8b_calls": 0,
        "verifier_positive_count": 0,
        "verifier_8b_seconds": 0.0,
        "rec_counter_8b_calls": rec_calls,
        "rec_counter_8b_seconds": answer_seconds,
        "total_8b_calls": rec_calls,
        "persistent_gate_document_vector_bytes": 0,
        # Each reduced REC interval uses one query-vector forward and one
        # selected-frame decoder-token scorer forward.
        "2b_model_forward_calls": 2 * rec_token_reduction_calls,
        "8b_model_forward_calls": rec_calls,
        "8b_token_reduction_calls": rec_token_reduction_calls,
        "8b_token_reduction_seconds": rec_token_reduction_seconds,
        "8b_visual_tokens_before_reduction": rec_visual_tokens_before,
        "8b_visual_tokens_after_reduction": rec_visual_tokens_after,
        "token_reduction_query_seconds": rec_token_reduction_query_seconds,
        "max_2b_kv_cache_bytes": 0,
        "max_8b_kv_cache_bytes": 0,
    }
    return result, source


def run_annotation(
    *,
    item: dict[str, Any],
    controller: CachedProactiveController,
    cfg: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = longest_prefix_candidate(item, cfg["video_root"])
    sampler = VideoSampler(str(path), cfg, causal_fixed_fps=True)
    specs = query_specs(item)
    runtime = {
        spec["query_id"]: {
            "spec": spec,
            "state": None,
            "decisions": [],
            "gate_calls": 0,
            "gate_positive_count": 0,
            "gate_seconds": 0.0,
            "verifier_calls": 0,
            "verifier_positive_count": 0,
            "verifier_seconds": 0.0,
            "final_cache_statistics": None,
        }
        for spec in specs
    }
    test_to_query = {
        index: spec["query_id"]
        for spec in specs
        for index in spec["test_indices"]
    }
    output = copy.deepcopy(item["test_info"])
    registrations: dict[float, list[str]] = defaultdict(list)
    tests: dict[float, list[int]] = defaultdict(list)
    samples: dict[float, bool] = {}
    for spec in specs:
        registrations[float(spec["registration_timestamp"])].append(spec["query_id"])
    for index, value in enumerate(item["test_info"]):
        tests[float(value["realtime"])].append(index)
    max_test = max(tests)
    for timestamp in sampler.sample_timestamps:
        if float(timestamp) <= max_test + 1e-6:
            samples[float(timestamp)] = True
    timeline = sorted(set(registrations) | set(tests) | set(samples))
    started = time.perf_counter()

    def accumulate_decision_metrics(value: dict[str, Any], metrics: dict[str, Any]) -> None:
        gate_called = bool(metrics.get("gate_2b_called", False))
        verifier_called = bool(metrics.get("verifier_8b_called", False))
        value["gate_calls"] += int(
            metrics.get("gate_2b_calls_this_decision", gate_called)
        )
        value["gate_positive_count"] += int(
            metrics.get(
                "gate_positive_count_this_decision",
                gate_called and metrics.get("gate_positive") is True,
            )
        )
        value["gate_seconds"] += float(metrics.get("gate_2b_seconds", 0.0))
        value["verifier_calls"] += int(
            metrics.get("verifier_8b_calls_this_decision", verifier_called)
        )
        value["verifier_positive_count"] += int(
            metrics.get(
                "verifier_positive_count_this_decision",
                verifier_called and metrics.get("verifier_positive") is True,
            )
        )
        value["verifier_seconds"] += float(
            metrics.get("answer_total_seconds", 0.0)
        )

    def register(query_id: str, timestamp: float) -> None:
        value = runtime[query_id]
        if value["state"] is not None:
            return
        spec = value["spec"]
        value["state"] = controller.register(
            query_id=query_id,
            target_question=spec["query_text"],
            registration_timestamp=float(spec["registration_timestamp"]),
            proposal_manifest_key=f"OVO:{item['task']}:{item['id']}:{query_id}",
        )
        # Evaluate the latest causal window immediately at registration.  This
        # handles targets already active when the first official checkpoint is
        # released without exposing any future frame.
        decision = controller.decide(state=value["state"], sampler=sampler, timestamp=timestamp)
        value["decisions"].append({"timestamp_seconds": timestamp, "metrics": decision.metrics})
        accumulate_decision_metrics(value, decision.metrics)

    for timestamp in timeline:
        for query_id in registrations.get(timestamp, []):
            register(query_id, timestamp)
        if timestamp in samples:
            for query_id, value in runtime.items():
                state: ProactiveState | None = value["state"]
                if state is None or abs(float(timestamp) - float(state.registration_timestamp)) <= 1e-6:
                    continue
                if float(timestamp) > float(value["spec"]["end_timestamp"]) + 1e-6:
                    continue
                decision = controller.decide(state=state, sampler=sampler, timestamp=timestamp)
                value["decisions"].append({"timestamp_seconds": timestamp, "metrics": decision.metrics})
                accumulate_decision_metrics(value, decision.metrics)
        for test_index in tests.get(timestamp, []):
            query_id = test_to_query[test_index]
            register(query_id, timestamp)
            value = runtime[query_id]
            state: ProactiveState = value["state"]
            if item["task"] == "REC":
                response = state.cached_answer
            elif item["task"] == "CRR":
                response = "Yes" if state.positive_latched else "No"
            else:
                response = "Yes" if state.current_positive else "No"
            output[test_index]["response"] = response
            output[test_index]["metrics"] = {
                "query_id": query_id,
                "query_timestamp_seconds": timestamp,
                "controller": controller.name,
                "gate_2b_calls_so_far": value["gate_calls"],
                "gate_positive_count_so_far": value["gate_positive_count"],
                "verifier_8b_calls_so_far": value["verifier_calls"],
                "verifier_positive_count_so_far": value["verifier_positive_count"],
                "current_positive": state.current_positive,
                "positive_latched": state.positive_latched,
                "latest_gate_scores_yes_no": state.latest_gate_scores,
                "cached_answer": state.cached_answer,
            }
        # A step query has no consumers after its final official test point.
        # Preserve counters, then release its potentially large per-layer KV
        # immediately instead of retaining every finished step until the end
        # of the source video.
        for value in runtime.values():
            state = value["state"]
            if (
                state is not None
                and value["final_cache_statistics"] is None
                and float(timestamp) >= float(value["spec"]["end_timestamp"]) - 1e-6
            ):
                value["final_cache_statistics"] = controller.state_statistics(state)
                state.gate_session = None
                state.verifier_session = None
    def final_statistics(value: dict[str, Any]) -> dict[str, Any]:
        return value["final_cache_statistics"] or controller.state_statistics(
            value["state"]
        )

    episodes = [
        {
            "query_id": query_id,
            "query_text": value["spec"]["query_text"],
            "registration_timestamp": value["spec"]["registration_timestamp"],
            "registration_metrics": value["state"].registration_metrics,
            "task_state_decisions": value["decisions"],
            "gate_2b_calls": value["gate_calls"],
            "gate_positive_count": value["gate_positive_count"],
            "gate_2b_seconds": value["gate_seconds"],
            "verifier_8b_calls": value["verifier_calls"],
            "verifier_positive_count": value["verifier_positive_count"],
            "verifier_8b_seconds": value["verifier_seconds"],
            "proposal_compiler_8b_calls": final_statistics(value).get(
                "proposal_compiler_8b_calls", 0
            ),
            "proposal_compiler_8b_seconds": final_statistics(value).get(
                "proposal_compiler_8b_seconds", 0.0
            ),
            "proposal_context_retrieval_2b_calls": final_statistics(value).get(
                "proposal_context_retrieval_2b_calls", 0
            ),
            "proposal_context_retrieval_seconds": final_statistics(value).get(
                "proposal_context_retrieval_seconds", 0.0
            ),
            "proposal_raw_candidate_count": final_statistics(value).get(
                "proposal_raw_candidate_count", 0
            ),
            "proposal_suppressed_candidate_count": final_statistics(value).get(
                "proposal_suppressed_candidate_count", 0
            ),
            "proposal_candidate_count": final_statistics(value).get(
                "proposal_candidate_count", 0
            ),
            "proposal_candidate_episode_count": final_statistics(value).get(
                "proposal_candidate_episode_count", 0
            ),
            "proposal_retry_candidate_count": final_statistics(value).get(
                "proposal_retry_candidate_count", 0
            ),
            "proposal_retry_positive_count": final_statistics(value).get(
                "proposal_retry_positive_count", 0
            ),
            "proposal_retry_state_change_count": final_statistics(value).get(
                "proposal_retry_state_change_count", 0
            ),
            "proposal_episode_expired_count": final_statistics(value).get(
                "proposal_episode_expired_count", 0
            ),
            "proposal_wait_expiry_rearm_count": final_statistics(value).get(
                "proposal_wait_expiry_rearm_count", 0
            ),
            "proposal_state_hold_count": final_statistics(value).get(
                "proposal_state_hold_count", 0
            ),
            "cache_statistics": final_statistics(value),
        }
        for query_id, value in runtime.items()
    ]
    gate_calls = sum(value["gate_calls"] for value in runtime.values())
    gate_positive_count = sum(value["gate_positive_count"] for value in runtime.values())
    verifier_calls = sum(value["verifier_calls"] for value in runtime.values())
    verifier_positive_count = sum(value["verifier_positive_count"] for value in runtime.values())
    gate_seconds = sum(value["gate_seconds"] for value in runtime.values())
    verifier_seconds = sum(value["verifier_seconds"] for value in runtime.values())
    proposal_compiler_calls = sum(
        int(final_statistics(value).get("proposal_compiler_8b_calls", 0))
        for value in runtime.values()
    )
    proposal_compiler_seconds = sum(
        float(final_statistics(value).get("proposal_compiler_8b_seconds", 0.0))
        for value in runtime.values()
    )
    total_8b_calls = verifier_calls + proposal_compiler_calls
    result = {
        **{key: copy.deepcopy(value) for key, value in item.items() if not key.startswith("_")},
        "test_info": output,
        "proactive_controller": controller.name,
        "cascade_episodes": episodes,
        "gate_2b_call_count": gate_calls,
        "gate_positive_count": gate_positive_count,
        "verifier_8b_call_count": verifier_calls,
        "verifier_positive_count": verifier_positive_count,
        "proposal_compiler_8b_call_count": proposal_compiler_calls,
        "rec_counter_8b_call_count": 0,
        "total_8b_call_count": total_8b_calls,
    }
    source = {
        "source_key": f"{item['task']}:{item['id']}",
        "task": item["task"],
        "source_online_wall_seconds": time.perf_counter() - started,
        "gate_2b_calls": gate_calls,
        "gate_positive_count": gate_positive_count,
        "gate_2b_seconds": gate_seconds,
        "verifier_8b_calls": verifier_calls,
        "verifier_positive_count": verifier_positive_count,
        "verifier_8b_seconds": verifier_seconds,
        "proposal_compiler_8b_calls": proposal_compiler_calls,
        "proposal_compiler_8b_seconds": proposal_compiler_seconds,
        "proposal_context_retrieval_2b_calls": sum(
            int(
                final_statistics(value).get(
                    "proposal_context_retrieval_2b_calls", 0
                )
            )
            for value in runtime.values()
        ),
        "proposal_context_retrieval_seconds": sum(
            float(
                final_statistics(value).get(
                    "proposal_context_retrieval_seconds", 0.0
                )
            )
            for value in runtime.values()
        ),
        "max_proposal_context_retrieval_cache_bytes": max(
            (
                int(
                    final_statistics(value).get(
                        "proposal_context_retrieval_peak_cache_bytes", 0
                    )
                )
                for value in runtime.values()
            ),
            default=0,
        ),
        "rec_counter_8b_calls": 0,
        "rec_counter_8b_seconds": 0.0,
        "total_8b_calls": total_8b_calls,
        "persistent_gate_document_vector_bytes": (
            controller.persistent_document_vector_bytes
        ),
        "max_2b_kv_cache_bytes": max(
            (
                final_statistics(value).get("2b_peak_cache_bytes", 0)
                for value in runtime.values()
            ),
            default=0,
        ),
        "max_8b_kv_cache_bytes": max(
            (
                final_statistics(value).get("8b_peak_cache_bytes", 0)
                for value in runtime.values()
            ),
            default=0,
        ),
        "2b_model_forward_calls": sum(
            final_statistics(value).get("2b_model_forward_calls", 0)
            for value in runtime.values()
        ),
        "8b_model_forward_calls": sum(
            final_statistics(value).get("8b_model_forward_calls", 0)
            for value in runtime.values()
        ),
        "max_persistent_proposal_vector_bytes": max(
            (
                final_statistics(value).get("persistent_proposal_vector_bytes", 0)
                for value in runtime.values()
            ),
            default=0,
        ),
        "max_persistent_shared_window_vector_cache_bytes": max(
            (
                final_statistics(value).get(
                    "persistent_shared_window_vector_cache_bytes", 0
                )
                for value in runtime.values()
            ),
            default=0,
        ),
        "max_persistent_shared_frame_vector_cache_bytes": max(
            (
                final_statistics(value).get(
                    "persistent_shared_frame_vector_cache_bytes", 0
                )
                for value in runtime.values()
            ),
            default=0,
        ),
        "max_persistent_shared_vit_feature_cache_bytes": max(
            (
                final_statistics(value).get(
                    "persistent_shared_vit_feature_cache_bytes", 0
                )
                for value in runtime.values()
            ),
            default=0,
        ),
        "proposal_raw_candidate_count": sum(
            int(final_statistics(value).get("proposal_raw_candidate_count", 0))
            for value in runtime.values()
        ),
        "proposal_suppressed_candidate_count": sum(
            int(
                final_statistics(value).get(
                    "proposal_suppressed_candidate_count", 0
                )
            )
            for value in runtime.values()
        ),
        "proposal_candidate_count": sum(
            int(final_statistics(value).get("proposal_candidate_count", 0))
            for value in runtime.values()
        ),
        "proposal_candidate_episode_count": sum(
            int(
                final_statistics(value).get(
                    "proposal_candidate_episode_count", 0
                )
            )
            for value in runtime.values()
        ),
        "proposal_retry_candidate_count": sum(
            int(
                final_statistics(value).get(
                    "proposal_retry_candidate_count", 0
                )
            )
            for value in runtime.values()
        ),
        "proposal_retry_positive_count": sum(
            int(
                final_statistics(value).get(
                    "proposal_retry_positive_count", 0
                )
            )
            for value in runtime.values()
        ),
        "proposal_retry_state_change_count": sum(
            int(
                final_statistics(value).get(
                    "proposal_retry_state_change_count", 0
                )
            )
            for value in runtime.values()
        ),
        "proposal_episode_expired_count": sum(
            int(
                final_statistics(value).get(
                    "proposal_episode_expired_count", 0
                )
            )
            for value in runtime.values()
        ),
        "proposal_wait_expiry_rearm_count": sum(
            int(
                final_statistics(value).get(
                    "proposal_wait_expiry_rearm_count", 0
                )
            )
            for value in runtime.values()
        ),
        "proposal_state_hold_count": sum(
            int(final_statistics(value).get("proposal_state_hold_count", 0))
            for value in runtime.values()
        ),
        "proposal_window_encode_2b_calls": sum(
            int(final_statistics(value).get("proposal_window_encode_2b_calls", 0))
            for value in runtime.values()
        ),
        "proposal_window_cache_hits": sum(
            int(final_statistics(value).get("proposal_window_cache_hits", 0))
            for value in runtime.values()
        ),
        "proposal_frame_encode_2b_calls": sum(
            int(final_statistics(value).get("proposal_frame_encode_2b_calls", 0))
            for value in runtime.values()
        ),
        "proposal_frame_cache_hits": sum(
            int(final_statistics(value).get("proposal_frame_cache_hits", 0))
            for value in runtime.values()
        ),
        "proposal_vit_feature_encode_2b_calls": sum(
            int(final_statistics(value).get("proposal_vit_feature_encode_2b_calls", 0))
            for value in runtime.values()
        ),
        "proposal_vit_feature_cache_hits": sum(
            int(final_statistics(value).get("proposal_vit_feature_cache_hits", 0))
            for value in runtime.values()
        ),
        "proposal_vit_feature_seconds": sum(
            float(final_statistics(value).get("proposal_vit_feature_seconds", 0.0))
            for value in runtime.values()
        ),
        "proposal_joint_decoder_2b_calls": sum(
            int(final_statistics(value).get("proposal_joint_decoder_2b_calls", 0))
            for value in runtime.values()
        ),
        "proposal_joint_decoder_seconds": sum(
            float(final_statistics(value).get("proposal_joint_decoder_seconds", 0.0))
            for value in runtime.values()
        ),
    }
    return result, source


def official_yes_no_correct(response: Any, expected_type: int) -> bool:
    if response is None:
        return False
    value = str(response)
    if (value == "N" and int(expected_type) == 0) or (value == "Y" and int(expected_type) == 1):
        return True
    return ("No" if int(expected_type) == 0 else "Yes") in value


def score_forward(records: list[dict[str, Any]]) -> dict[str, Any]:
    per_task: dict[str, list[int]] = defaultdict(list)
    for item in records:
        for test in item["test_info"]:
            if item["task"] == "REC":
                digits = "".join(re.findall(r"\d+", str(test.get("response", ""))))
                correct = digits == str(test["count"])
            else:
                correct = official_yes_no_correct(test.get("response"), int(test["type"]))
            per_task[item["task"]].append(int(correct))
    task_scores = {
        task: {"accuracy": float(np.mean(values)), "paper_table_value": 100.0 * float(np.mean(values)), "correct": sum(values), "total": len(values)}
        for task, values in sorted(per_task.items())
    }
    macro = float(np.mean([value["accuracy"] for value in task_scores.values()])) if task_scores else None
    return {"protocol": "OVO-Bench Forward Active official test-point accuracy", "task_scores": task_scores, "forward_macro": {"accuracy": macro, "paper_table_value": None if macro is None else 100.0 * macro}}


def proposal_manifest_from_ovo_records(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Export compiler outputs before scoring so later ablations replay them."""

    entries: dict[str, Any] = {}
    for record in records:
        for episode in record.get("cascade_episodes", []):
            compilation = episode.get("registration_metrics", {}).get(
                "proposal_compilation", {}
            )
            key = compilation.get("proposal_manifest_key")
            texts = compilation.get("proposal_texts")
            vectors = compilation.get("proposal_vectors")
            if not key or not texts or vectors is None:
                continue
            entries[str(key)] = {
                "benchmark": "OVO-Bench",
                "task": str(record["task"]),
                "annotation_id": str(record["id"]),
                "query_id": str(episode["query_id"]),
                "target_question": str(episode["query_text"]),
                "proposal_prompt_version": compilation.get(
                    "proposal_prompt_version", "target-anchored-v2"
                ),
                "compiler_context_frames": compilation.get("compiler_context_frames"),
                "compiler_sample_fps": compilation.get("compiler_sample_fps"),
                "proposal_texts": texts,
                "proposal_vectors": vectors,
                "source_manifest_replay": bool(
                    compilation.get("manifest_replay", False)
                ),
            }
    canonical = json.dumps(entries, sort_keys=True, ensure_ascii=False)
    return {
        "format": "biseek-proposal-manifest-v2",
        "selection_policy": (
            "fixed_before_test_scoring; never select by test accuracy"
        ),
        "entries_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
        "num_entries": len(entries),
        "entries": entries,
    }


def assign_annotations(values: list[dict[str, Any]], world_size: int) -> list[list[dict[str, Any]]]:
    bins = [[] for _ in range(world_size)]
    for index, value in enumerate(values):
        bins[index % world_size].append(value)
    return bins


def build_manifest(cfg: dict[str, Any], tasks: list[str], values: list[dict[str, Any]], world_size: int) -> dict[str, Any]:
    scope = {"run_scope_version": 13, "controller": experiment_controller_name(cfg, tasks), "config": output_config_snapshot(cfg, tasks), "annotation_ids": [f"{value['task']}:{value['id']}" for value in values], "world_size": world_size}
    return {"fingerprint": hashlib.sha256(json.dumps(scope, sort_keys=True).encode()).hexdigest(), **scope}


def merge_outputs(
    output_dir: Path,
    world_size: int,
    cfg: dict[str, Any],
    active_tasks: list[str],
) -> None:
    merged: dict[str, dict[str, Any]] = {}
    sources: list[dict[str, Any]] = []
    for rank in range(world_size):
        with (output_dir / f"rank_{rank:02d}.jsonl").open() as handle:
            for line in handle:
                if line.strip():
                    value = json.loads(line)
                    merged[value["_key"]] = value
        path = output_dir / f"rank_{rank:02d}_workload.json"
        if path.is_file():
            sources.extend(load_json(path))
    ordered = sorted(merged.values(), key=lambda value: value["_index"])
    records = [{key: value for key, value in item.items() if not key.startswith("_")} for item in ordered]

    # Normalize counters from the decision trace.  This also repairs completed
    # v6.1 rank files where an absorbing CRR state was historically counted as
    # a new positive verdict despite making no model call.
    for record in records:
        for episode in record["cascade_episodes"]:
            metrics = [
                value["metrics"] for value in episode["task_state_decisions"]
            ]
            episode["gate_2b_calls"] = sum(
                bool(value.get("gate_2b_called")) for value in metrics
            )
            episode["gate_positive_count"] = sum(
                bool(value.get("gate_2b_called"))
                and value.get("gate_positive") is True
                for value in metrics
            )
            episode["verifier_8b_calls"] = sum(
                bool(value.get("verifier_8b_called")) for value in metrics
            )
            episode["verifier_positive_count"] = sum(
                bool(value.get("verifier_8b_called"))
                and value.get("verifier_positive") is True
                for value in metrics
            )
            episode["rec_counter_8b_calls"] = sum(
                bool(value.get("rec_counter_8b_called")) for value in metrics
            )
            episode["rec_counter_8b_seconds"] = sum(
                float(value.get("answer_total_seconds", 0.0))
                for value in metrics
                if value.get("rec_counter_8b_called")
            )
        record["gate_2b_call_count"] = sum(
            value["gate_2b_calls"] for value in record["cascade_episodes"]
        )
        record["gate_positive_count"] = sum(
            value["gate_positive_count"] for value in record["cascade_episodes"]
        )
        record["verifier_8b_call_count"] = sum(
            value["verifier_8b_calls"] for value in record["cascade_episodes"]
        )
        record["verifier_positive_count"] = sum(
            value["verifier_positive_count"] for value in record["cascade_episodes"]
        )
        record["rec_counter_8b_call_count"] = sum(
            value["rec_counter_8b_calls"] for value in record["cascade_episodes"]
        )
        record["proposal_compiler_8b_call_count"] = sum(
            int(value.get("proposal_compiler_8b_calls", 0))
            for value in record["cascade_episodes"]
        )
        record["total_8b_call_count"] = (
            record["verifier_8b_call_count"]
            + record["rec_counter_8b_call_count"]
            + record["proposal_compiler_8b_call_count"]
        )

    atomic_write_json(output_dir / "results.json", {"backward": [], "realtime": [], "forward": records})
    atomic_write_json(
        output_dir / "proposal_manifest.json",
        proposal_manifest_from_ovo_records(records),
    )
    def decision_metrics(values: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            decision["metrics"]
            for record in values
            for episode in record["cascade_episodes"]
            for decision in episode["task_state_decisions"]
        ]

    decisions = decision_metrics(records)
    gate_calls = sum(
        int(value.get("gate_2b_calls_this_decision", bool(value.get("gate_2b_called"))))
        for value in decisions
    )
    gate_positive = sum(
        int(
            value.get(
                "gate_positive_count_this_decision",
                bool(value.get("gate_2b_called"))
                and value.get("gate_positive") is True,
            )
        )
        for value in decisions
    )
    verifier_calls = sum(
        int(value.get("verifier_8b_calls_this_decision", bool(value.get("verifier_8b_called"))))
        for value in decisions
    )
    verifier_positive = sum(
        int(
            value.get(
                "verifier_positive_count_this_decision",
                bool(value.get("verifier_8b_called"))
                and value.get("verifier_positive") is True,
            )
        )
        for value in decisions
    )
    calibrated_candidates = sum(
        int(value.get("calibrated_candidate_count_this_decision", value.get("gate_calibrated_candidate") is True))
        for value in decisions
    )
    raw_cosine_positive = sum(
        int(value.get("raw_cosine_positive_count_this_decision", value.get("gate_raw_cosine_positive") is True))
        for value in decisions
    )
    raw_cosine_suppressed = sum(
        int(value.get("raw_cosine_suppressed_count_this_decision", value.get("gate_raw_cosine_suppressed") is True))
        for value in decisions
    )
    if cfg["decision_mode"] == "update_head" and not (
        gate_positive == calibrated_candidates == verifier_calls
    ):
        raise RuntimeError(
            "Calibrated gate invariant failed: each positive candidate must invoke "
            "exactly one 8B verification"
        )
    rec_counter_calls = sum(
        bool(value.get("rec_counter_8b_called")) for value in decisions
    )
    gate_seconds = sum(float(value.get("gate_2b_seconds", 0.0)) for value in decisions)
    verifier_seconds = sum(
        float(value.get("answer_total_seconds", 0.0))
        for value in decisions
        if value.get("verifier_8b_called")
    )
    rec_counter_seconds = sum(
        float(value.get("answer_total_seconds", 0.0))
        for value in decisions
        if value.get("rec_counter_8b_called")
    )
    proposal_compiler_calls = sum(
        int(value.get("proposal_compiler_8b_calls", 0)) for value in sources
    )
    proposal_compiler_seconds = sum(
        float(value.get("proposal_compiler_8b_seconds", 0.0))
        for value in sources
    )
    proposal_context_retrieval_calls = sum(
        int(value.get("proposal_context_retrieval_2b_calls", 0))
        for value in sources
    )
    proposal_context_retrieval_seconds = sum(
        float(value.get("proposal_context_retrieval_seconds", 0.0))
        for value in sources
    )
    by_task: dict[str, dict[str, Any]] = {}
    for task in FORWARD_TASKS:
        task_records = [value for value in records if value.get("task") == task]
        task_sources = [value for value in sources if value.get("task") == task]
        task_decisions = decision_metrics(task_records)
        task_gate_calls = sum(
            int(value.get("gate_2b_calls_this_decision", bool(value.get("gate_2b_called"))))
            for value in task_decisions
        )
        task_gate_positive = sum(
            int(
                value.get(
                    "gate_positive_count_this_decision",
                    bool(value.get("gate_2b_called"))
                    and value.get("gate_positive") is True,
                )
            )
            for value in task_decisions
        )
        task_calibrated_candidates = sum(
            int(value.get("calibrated_candidate_count_this_decision", value.get("gate_calibrated_candidate") is True))
            for value in task_decisions
        )
        task_raw_cosine_positive = sum(
            int(value.get("raw_cosine_positive_count_this_decision", value.get("gate_raw_cosine_positive") is True))
            for value in task_decisions
        )
        task_raw_cosine_suppressed = sum(
            int(value.get("raw_cosine_suppressed_count_this_decision", value.get("gate_raw_cosine_suppressed") is True))
            for value in task_decisions
        )
        task_verifier_calls = sum(
            int(value.get("verifier_8b_calls_this_decision", bool(value.get("verifier_8b_called"))))
            for value in task_decisions
        )
        task_verifier_positive = sum(
            int(
                value.get(
                    "verifier_positive_count_this_decision",
                    bool(value.get("verifier_8b_called"))
                    and value.get("verifier_positive") is True,
                )
            )
            for value in task_decisions
        )
        task_rec_counter_calls = sum(
            bool(value.get("rec_counter_8b_called")) for value in task_decisions
        )
        task_proposal_compiler_calls = sum(
            int(value.get("proposal_compiler_8b_calls", 0))
            for value in sources
            if value.get("task") == task
        )
        task_total_8b_calls = (
            task_verifier_calls
            + task_rec_counter_calls
            + task_proposal_compiler_calls
        )
        task_online_8b_calls = task_verifier_calls + task_rec_counter_calls
        by_task[task] = {
            "num_meta_annotations": len(task_records),
            "num_decision_steps": len(task_decisions),
            "gate_2b_calls": task_gate_calls,
            "gate_positive_rate": (
                task_gate_positive / task_gate_calls if task_gate_calls else None
            ),
            "calibrated_candidate_count": task_calibrated_candidates,
            "raw_cosine_positive_count": task_raw_cosine_positive,
            "raw_cosine_suppressed_count": task_raw_cosine_suppressed,
            "verifier_8b_calls": task_verifier_calls,
            "verifier_accept_rate": (
                task_verifier_positive / task_verifier_calls
                if task_verifier_calls else None
            ),
            "rec_counter_8b_calls": task_rec_counter_calls,
            "proposal_compiler_8b_calls": task_proposal_compiler_calls,
            "proposal_context_retrieval_2b_calls": sum(
                int(value.get("proposal_context_retrieval_2b_calls", 0))
                for value in task_sources
            ),
            "proposal_context_retrieval_seconds": sum(
                float(value.get("proposal_context_retrieval_seconds", 0.0))
                for value in task_sources
            ),
            "proposal_candidate_count": sum(
                int(value.get("proposal_candidate_count", 0))
                for value in task_sources
            ),
            "proposal_candidate_episode_count": sum(
                int(value.get("proposal_candidate_episode_count", 0))
                for value in task_sources
            ),
            "proposal_retry_candidate_count": sum(
                int(value.get("proposal_retry_candidate_count", 0))
                for value in task_sources
            ),
            "proposal_retry_positive_count": sum(
                int(value.get("proposal_retry_positive_count", 0))
                for value in task_sources
            ),
            "proposal_retry_state_change_count": sum(
                int(value.get("proposal_retry_state_change_count", 0))
                for value in task_sources
            ),
            "proposal_episode_expired_count": sum(
                int(value.get("proposal_episode_expired_count", 0))
                for value in task_sources
            ),
            "proposal_wait_expiry_rearm_count": sum(
                int(value.get("proposal_wait_expiry_rearm_count", 0))
                for value in task_sources
            ),
            "proposal_state_hold_count": sum(
                int(value.get("proposal_state_hold_count", 0))
                for value in task_sources
            ),
            "total_8b_calls": task_total_8b_calls,
            "8b_call_fraction_of_decision_steps": (
                task_online_8b_calls / len(task_decisions)
                if task_decisions else None
            ),
            "total_8b_calls_per_decision_step": (
                task_total_8b_calls / len(task_decisions)
                if task_decisions else None
            ),
            "8b_call_fraction_of_2b_polls": (
                task_verifier_calls / task_gate_calls if task_gate_calls else None
            ),
        }
    total_8b_calls = verifier_calls + rec_counter_calls + proposal_compiler_calls
    efficiency = {
        "num_meta_annotations": len(records),
        "num_test_points": sum(len(value["test_info"]) for value in records),
        "gate_2b_calls": gate_calls,
        "gate_positive_count": gate_positive,
        "gate_positive_rate": gate_positive / max(1, gate_calls),
        "calibrated_candidate_count": calibrated_candidates,
        "raw_cosine_positive_count": raw_cosine_positive,
        "raw_cosine_suppressed_count": raw_cosine_suppressed,
        "gate_2b_seconds": gate_seconds,
        "mean_gate_2b_seconds_per_call": gate_seconds / max(1, gate_calls),
        "verifier_8b_calls": verifier_calls,
        "verifier_positive_count": verifier_positive,
        "verifier_accept_rate": verifier_positive / max(1, verifier_calls),
        "verifier_8b_seconds": verifier_seconds,
        "mean_verifier_8b_seconds_per_call": (
            verifier_seconds / max(1, verifier_calls)
        ),
        "rec_counter_8b_calls": rec_counter_calls,
        "rec_counter_8b_seconds": rec_counter_seconds,
        "proposal_compiler_8b_calls": proposal_compiler_calls,
        "proposal_compiler_8b_seconds": proposal_compiler_seconds,
        "proposal_context_retrieval_2b_calls": (
            proposal_context_retrieval_calls
        ),
        "proposal_context_retrieval_seconds": (
            proposal_context_retrieval_seconds
        ),
        "max_proposal_context_retrieval_cache_bytes": max(
            (
                int(
                    value.get("max_proposal_context_retrieval_cache_bytes", 0)
                )
                for value in sources
            ),
            default=0,
        ),
        "mean_rec_counter_8b_seconds_per_call": (
            rec_counter_seconds / max(1, rec_counter_calls)
        ),
        "proactive_head_operating_point": output_config_snapshot(
            cfg, active_tasks
        ).get("proactive_head_operating_point"),
        "total_8b_calls": total_8b_calls,
        "8b_call_fraction_of_decision_steps": (
            verifier_calls + rec_counter_calls
        ) / max(1, len(decisions)),
        "total_8b_calls_per_decision_step": total_8b_calls / max(1, len(decisions)),
        "8b_call_fraction_of_2b_polls": (
            verifier_calls / gate_calls if gate_calls else None
        ),
        "8b_calls_avoided_by_gate": (
            gate_calls - verifier_calls if gate_calls else None
        ),
        "mean_verifier_8b_calls_per_annotation": verifier_calls / max(1, len(records)),
        "mean_total_8b_calls_per_annotation": total_8b_calls / max(1, len(records)),
        "by_task": by_task,
        "max_persistent_gate_document_vector_bytes": max(
            (value["persistent_gate_document_vector_bytes"] for value in sources),
            default=0,
        ),
        "max_2b_kv_cache_bytes": max(
            (value.get("max_2b_kv_cache_bytes", 0) for value in sources),
            default=0,
        ),
        "max_8b_kv_cache_bytes": max(
            (value.get("max_8b_kv_cache_bytes", 0) for value in sources),
            default=0,
        ),
        "max_persistent_proposal_vector_bytes": max(
            (value.get("max_persistent_proposal_vector_bytes", 0) for value in sources),
            default=0,
        ),
        "max_persistent_shared_window_vector_cache_bytes": max(
            (
                value.get(
                    "max_persistent_shared_window_vector_cache_bytes", 0
                )
                for value in sources
            ),
            default=0,
        ),
        "max_persistent_shared_frame_vector_cache_bytes": max(
            (
                value.get("max_persistent_shared_frame_vector_cache_bytes", 0)
                for value in sources
            ),
            default=0,
        ),
        "max_persistent_shared_vit_feature_cache_bytes": max(
            (
                value.get("max_persistent_shared_vit_feature_cache_bytes", 0)
                for value in sources
            ),
            default=0,
        ),
        "proposal_raw_candidate_count": sum(
            int(value.get("proposal_raw_candidate_count", 0)) for value in sources
        ),
        "proposal_suppressed_candidate_count": sum(
            int(value.get("proposal_suppressed_candidate_count", 0))
            for value in sources
        ),
        "proposal_candidate_count": sum(
            int(value.get("proposal_candidate_count", 0)) for value in sources
        ),
        "proposal_candidate_episode_count": sum(
            int(value.get("proposal_candidate_episode_count", 0))
            for value in sources
        ),
        "proposal_retry_candidate_count": sum(
            int(value.get("proposal_retry_candidate_count", 0))
            for value in sources
        ),
        "proposal_retry_positive_count": sum(
            int(value.get("proposal_retry_positive_count", 0))
            for value in sources
        ),
        "proposal_retry_state_change_count": sum(
            int(value.get("proposal_retry_state_change_count", 0))
            for value in sources
        ),
        "proposal_episode_expired_count": sum(
            int(value.get("proposal_episode_expired_count", 0))
            for value in sources
        ),
        "proposal_wait_expiry_rearm_count": sum(
            int(value.get("proposal_wait_expiry_rearm_count", 0))
            for value in sources
        ),
        "proposal_state_hold_count": sum(
            int(value.get("proposal_state_hold_count", 0))
            for value in sources
        ),
        "proposal_window_encode_2b_calls": sum(
            int(value.get("proposal_window_encode_2b_calls", 0))
            for value in sources
        ),
        "proposal_window_cache_hits": sum(
            int(value.get("proposal_window_cache_hits", 0))
            for value in sources
        ),
        "proposal_frame_encode_2b_calls": sum(
            int(value.get("proposal_frame_encode_2b_calls", 0))
            for value in sources
        ),
        "proposal_frame_cache_hits": sum(
            int(value.get("proposal_frame_cache_hits", 0)) for value in sources
        ),
        "proposal_vit_feature_encode_2b_calls": sum(
            int(value.get("proposal_vit_feature_encode_2b_calls", 0))
            for value in sources
        ),
        "proposal_vit_feature_cache_hits": sum(
            int(value.get("proposal_vit_feature_cache_hits", 0))
            for value in sources
        ),
        "proposal_vit_feature_seconds": sum(
            float(value.get("proposal_vit_feature_seconds", 0.0))
            for value in sources
        ),
        "proposal_joint_decoder_2b_calls": sum(
            int(value.get("proposal_joint_decoder_2b_calls", 0))
            for value in sources
        ),
        "proposal_joint_decoder_seconds": sum(
            float(value.get("proposal_joint_decoder_seconds", 0.0))
            for value in sources
        ),
        "2b_model_forward_calls": sum(
            value.get("2b_model_forward_calls", 0) for value in sources
        ),
        "8b_model_forward_calls": sum(
            value.get("8b_model_forward_calls", 0) for value in sources
        ),
        "2b_gate_poll_calls": gate_calls,
        "2b_low_level_forward_calls": sum(
            value.get("2b_model_forward_calls", 0) for value in sources
        ),
        "8b_high_level_calls": total_8b_calls,
        "8b_low_level_forward_calls": sum(
            value.get("8b_model_forward_calls", 0) for value in sources
        ),
        "call_metric_definitions": {
            "2b_gate_poll_calls": (
                "Number of causal gate decisions; use this as the 2B poll count."
            ),
            "2b_low_level_forward_calls": (
                "Underlying 2B model.forward invocations, including separate "
                "ViT feature encoding and decoder readout work when applicable."
            ),
            "8b_high_level_calls": (
                "Proposal compiler, verifier, and task-native REC calls at the "
                "service/API level."
            ),
            "8b_low_level_forward_calls": (
                "Underlying 8B model.forward invocations inside cached sessions."
            ),
        },
    }
    event_timing_proxy = score_ovo_event_timing_proxy(
        records, tolerance_seconds=2.0
    )
    event_timing_proxy_compact = {
        **{
            key: value
            for key, value in event_timing_proxy.items()
            if key != "stages"
        },
        "stages": {
            stage: {key: value for key, value in data.items() if key != "details"}
            for stage, data in event_timing_proxy["stages"].items()
        },
    }
    summary = score_forward(records)
    summary["event_timing_proxy"] = event_timing_proxy_compact
    summary["efficiency"] = efficiency
    atomic_write_json(output_dir / "summary.json", summary)
    atomic_write_json(output_dir / "efficiency_summary.json", efficiency)
    atomic_write_json(
        output_dir / "event_timing_proxy_summary.json", event_timing_proxy
    )
    atomic_write_json(
        output_dir / "serving_efficiency_summary.json",
        summarize_proactive_efficiency(output_dir),
    )


def main() -> None:
    args = parse_args()
    cfg = apply_overrides(load_json(args.config), args)
    rank, world_size, _, device = distributed_context(args.verify_only)
    tasks = list(cfg.get("tasks", FORWARD_TASKS))
    if args.tasks:
        requested = {value.strip() for value in args.tasks.split(",") if value.strip()}
        tasks = [value for value in tasks if value in requested]
    invalid = sorted(set(tasks).difference(FORWARD_TASKS))
    if invalid:
        raise ValueError(f"Unsupported tasks: {invalid}")
    if cfg["decision_mode"] == "update_head" and any(
        task in tasks for task in ("SSR", "CRR")
    ):
        checkpoint = cfg.get("proactive_head_checkpoint")
        if not checkpoint or not Path(checkpoint).is_file():
            raise FileNotFoundError(
                f"Missing proactive_head_checkpoint: {checkpoint!r}"
            )
    annotations = [value for value in load_json(cfg["annotation_path"]) if value["task"] in tasks]
    if args.max_samples is not None:
        annotations = annotations[: args.max_samples]
    annotations = [{**value, "_index": index} for index, value in enumerate(annotations)]
    output_dir = Path(args.output_dir or Path(cfg["output_root"]) / default_output_suffix(cfg, tasks))
    output_dir.mkdir(parents=True, exist_ok=True)
    verify = forward_verification_report(annotations, cfg)
    if rank == 0:
        atomic_write_json(output_dir / "verification_report.json", verify)
    if args.verify_only:
        if rank == 0:
            print(json.dumps(verify, indent=2))
        return
    if verify["fatal_prefix_problems"]:
        raise RuntimeError(json.dumps(verify["fatal_prefix_problems"][:5], indent=2))
    assert device is not None
    manifest = build_manifest(cfg, tasks, annotations, world_size)
    manifest_path = output_dir / "run_manifest.json"
    if manifest_path.is_file() and not args.overwrite and load_json(manifest_path).get("fingerprint") != manifest["fingerprint"]:
        raise RuntimeError("Run scope mismatch; use a distinct OUTPUT_DIR or --overwrite")
    if rank == 0:
        atomic_write_json(manifest_path, manifest)
        atomic_write_json(output_dir / "inference_config.json", output_config_snapshot(cfg, tasks))
    rank_path = output_dir / f"rank_{rank:02d}.jsonl"
    metrics_path = output_dir / f"rank_{rank:02d}_workload.json"
    done_path = output_dir / f"rank_{rank:02d}.done"
    if args.overwrite:
        for path in (rank_path, metrics_path):
            if path.exists():
                path.unlink()
    rank_path.touch(exist_ok=True)
    prepare_resumable_jsonl(rank_path)
    done = load_done(rank_path)
    pending = [value for value in assign_annotations(annotations, world_size)[rank] if f"forward:{value['task']}:{value['id']}" not in done]
    metrics_by_key = {
        value["source_key"]: value
        for value in (load_json(metrics_path) if metrics_path.is_file() else [])
    }
    if pending:
        gated_tasks = {
            value["task"]
            for value in pending
            if value["task"] != "REC"
            or (
                unified_rec_enabled(cfg)
            )
        }
        needs_token_reduction_encoder = bool(
            cfg.get("token_reduction_enabled", False)
        ) and any(value["task"] == "REC" for value in pending)
        answer_model = (
            AnswerModel(cfg, device)
            if cfg["decision_mode"] == "update_head"
            or "REC" in tasks
            or cfg["decision_mode"] in {
                "8b_cache",
                "2b8b_cache",
                "proposal_match",
            }
            else None
        )
        encoder = (
            SemanticMemoryEncoder(cfg, "full_embedding", device)
            if (gated_tasks or needs_token_reduction_encoder)
            and cfg["decision_mode"] in {
                "2b8b_cache",
                "update_head",
                "proposal_match",
            }
            else None
        )
        controllers = {
            task: build_proactive_controller(
                encoder=encoder,
                answer_model=answer_model,
                cfg=cfg,
                task=task,
                decision_mode=cfg["decision_mode"],
            )
            for task in gated_tasks
        }
        with rank_path.open("a", buffering=1) as handle:
            for local_index, item in enumerate(pending, 1):
                if (
                    item["task"] == "REC"
                    and not unified_rec_enabled(cfg)
                ):
                    if answer_model is None:
                        raise RuntimeError("REC requires the 8B interval-delta counter")
                    result, source = run_rec_annotation(
                        item=item,
                        answer_model=answer_model,
                        encoder=encoder,
                        cfg=cfg,
                    )
                else:
                    result, source = run_annotation(
                        item=item, controller=controllers[item["task"]], cfg=cfg
                    )
                key = f"forward:{item['task']}:{item['id']}"
                result.update({"_key": key, "_index": item["_index"], "rank": rank})
                metrics_by_key[source["source_key"]] = source
                atomic_write_json(metrics_path, list(metrics_by_key.values()))
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                print(f"[rank {rank}] {local_index}/{len(pending)} {key} tests={len(item['test_info'])} 8B_calls={source['total_8b_calls']}", flush=True)
    run_id = os.environ.get("TORCHELASTIC_RUN_ID", "single-process")
    done_path.write_text(json.dumps({"run_id": run_id, "rank": rank}) + "\n")
    if rank == 0:
        markers = [output_dir / f"rank_{value:02d}.done" for value in range(world_size)]
        while not all(path.is_file() and load_json(path).get("run_id") == run_id for path in markers):
            time.sleep(10)
        merge_outputs(output_dir, world_size, cfg, tasks)
        print(f"Completed OVO Forward Active: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
