#!/usr/bin/env python3
"""StreamingBench Proactive Output with a frozen-2B/frozen-8B cascade."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

_TOOLS_DIR = str(Path(__file__).resolve().parent)
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)

import cost_meter  # noqa: E402,F401  -- inert unless BISEEK_COST_METER=1
from eval_ovo_forward_active import atomic_write_json, distributed_context
from proactive_cascade import VERIFIER_PROMPT_MODES, set_verifier_prompt_mode, default_verifier_accept_margin
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
    CONTROLLER_VERSION,
    DECISION_MODES,
    PROPOSAL_PROMPT_STYLES,
    CachedProactiveController,
    build_proactive_controller,
    controller_name,
    inspect_proposal_manifest,
)
from proactive_metrics import summarize_proactive_efficiency
from video_probe import probe_video_duration_seconds


OUTPUT_COMMAND_PATTERN = re.compile(
    r"""(?is)^(?P<condition>.*?)\s*,?\s*\boutput\s+
        (?:"(?P<double>.*?)"|“(?P<curly>.*?)”|'(?P<single>.*?)')
        \s*\.?\s*$""",
    re.VERBOSE,
)


def parse_public_trigger_spec(question: str) -> tuple[str, str]:
    match = OUTPUT_COMMAND_PATTERN.match(str(question).strip())
    if match is None:
        raise ValueError(f"Unparseable proactive output instruction: {question!r}")
    output = next(
        value
        for value in (
            match.group("double"),
            match.group("curly"),
            match.group("single"),
        )
        if value is not None
    ).strip()
    condition = match.group("condition").strip().rstrip(",")
    if condition.lower().startswith("when "):
        condition = condition[5:].strip()
    if not condition or not output:
        raise ValueError(f"Invalid proactive output instruction: {question!r}")
    return condition, output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--annotation-scope", choices=("official50", "full250"))
    parser.add_argument("--output-dir")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--sample-fps", type=float)
    parser.add_argument("--recent-window-frames", type=int)
    parser.add_argument("--official-poll-interval-seconds", type=float)
    parser.add_argument("--official-max-after-ground-truth-seconds", type=float)
    parser.add_argument("--official-tolerance-seconds", type=float)
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
    parser.add_argument("--incremental-kv-reset-every-n-frames", type=int)
    parser.add_argument("--verifier-accept-margin", type=float)
    parser.add_argument("--answer-model-path")
    parser.add_argument(
        "--answer-model-family",
        choices=("qwen3vl", "qwen2_5vl", "qwen2vl"),
    )
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
    parser.add_argument("--proposal-batch-new-frames-per-poll", type=parse_bool)
    parser.add_argument("--proposal-manifest-path")
    parser.add_argument(
        "--proposal-manifest-texts-only", type=parse_bool, default=None,
        help="replay only the manifest's compiled proposal texts and re-encode "
             "them with the encoder under test, instead of reusing the stored "
             "vectors.  Required for an encoder ablation on a frozen manifest: "
             "reusing the vectors would match every arm against whichever "
             "encoder wrote the manifest.",
    )
    parser.add_argument("--token-reduction", type=parse_bool)
    parser.add_argument("--token-reduction-layer", type=int)
    parser.add_argument(
        "--verifier-prompt-mode", choices=list(VERIFIER_PROMPT_MODES), default=None,
        help="task_specific (default) phrases the 8B verifier per task; agnostic "
             "uses one standing-instruction wording for every task, to test "
             "whether the score depends on task-shaped phrasing.",
    )
    parser.add_argument(
        "--embedding-backend", default=None,
        help="which encoder hosts the gate: qwen3vl (default), gme_qwen2vl, opsmm, "
             "vlm2vec, clip, siglip.  Non-default backends require "
             "--proposal-match-encoding cached_frame_mean.",
    )
    parser.add_argument("--embedding-model-path", default=None)
    parser.add_argument("--token-reduction-selector", type=str,
                        choices=["semantic", "stc_dynamic", "ctr", "random", "uniform", "norm"],
                        help="semantic = BiSeek query-conditioned 2B score; stc_dynamic = STC temporal-redundancy criterion")
    parser.add_argument("--token-reduction-budget", type=float)
    parser.add_argument("--token-reduction-parallel", type=parse_bool)
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


def apply_overrides(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    values = {
        "annotation_scope": args.annotation_scope,
        "sample_fps": args.sample_fps,
        "recent_window_frames": args.recent_window_frames,
        "official_poll_interval_seconds": args.official_poll_interval_seconds,
        "official_max_after_ground_truth_seconds": args.official_max_after_ground_truth_seconds,
        "official_tolerance_seconds": args.official_tolerance_seconds,
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
        "incremental_kv_reset_every_n_frames": args.incremental_kv_reset_every_n_frames,
        "verifier_accept_margin": args.verifier_accept_margin,
        "answer_model_path": args.answer_model_path,
        "answer_model_family": args.answer_model_family,
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
        "proposal_batch_new_frames_per_poll": args.proposal_batch_new_frames_per_poll,
        "proposal_manifest_path": args.proposal_manifest_path,
        "proposal_manifest_texts_only": args.proposal_manifest_texts_only,
        "proactive_head_checkpoint": args.proactive_head_checkpoint,
        "proactive_head_threshold": args.proactive_head_threshold,
        "token_reduction_enabled": args.token_reduction,
        "token_reduction_layer": args.token_reduction_layer,
        "token_reduction_selector": args.token_reduction_selector,
        "embedding_backend": args.embedding_backend,
        "verifier_prompt_mode": args.verifier_prompt_mode,
        "embedding_model_path": args.embedding_model_path,
        "token_reduction_budget": args.token_reduction_budget,
        "token_reduction_parallel": args.token_reduction_parallel,
    }
    for key, value in values.items():
        if value is not None:
            cfg[key] = value
    cfg.setdefault("annotation_scope", "full250")
    cfg.setdefault("sample_fps", 1.0)
    cfg.setdefault("recent_window_frames", 4)
    cfg.setdefault("incremental_kv_reset_every_n_frames", 0)
    # The verifier accept margin defaults per backbone (see verifier_margin_defaults);
    # an explicit command-line value takes precedence.
    if cfg.get("verifier_accept_margin") is None:
        cfg["verifier_accept_margin"] = default_verifier_accept_margin(
            cfg.get("answer_model_family"), cfg.get("answer_model_path")
        )
    cfg.setdefault("answer_model_family", "qwen3vl")
    cfg.setdefault("official_poll_interval_seconds", 1.0)
    cfg.setdefault("official_max_after_ground_truth_seconds", 4.0)
    cfg.setdefault("official_tolerance_seconds", 2.0)
    cfg.setdefault("decision_mode", "proposal_match")
    cfg.setdefault("proposal_count", 4)
    cfg.setdefault("proposal_prompt_style", "target_anchored_v2")
    cfg.setdefault("proposal_generation_window_frames", 5)
    cfg.setdefault("proposal_context_retrieval_enabled", False)
    cfg.setdefault("proposal_context_retrieval_top_k", 1)
    cfg.setdefault("proposal_context_retrieval_chunk_frames", 4)
    cfg.setdefault("proposal_context_retrieval_max_vectors", 128)
    cfg.setdefault("proposal_context_retrieval_cache_size", 4096)
    cfg.setdefault("proposal_match_window_frames", 2)
    cfg.setdefault("proposal_surge_threshold", 0.035)
    cfg.setdefault("proposal_delta_lookback_seconds", 1.0)
    cfg.setdefault("proposal_require_above_history_mean", True)
    cfg.setdefault("proposal_duplicate_similarity_threshold", 0.8)
    cfg.setdefault("proposal_duplicate_cooldown_seconds", 5.0)
    cfg.setdefault("proposal_initial_verify", False)
    cfg.setdefault("proposal_audit_only", False)
    cfg.setdefault("proposal_manifest_texts_only", False)
    cfg.setdefault("proposal_max_new_tokens", 256)
    cfg.setdefault("proposal_window_cache_size", 64)
    cfg.setdefault("proposal_match_encoding", "incremental_kv")
    cfg.setdefault("proposal_frame_cache_size", 256)
    cfg.setdefault("proposal_vit_cache_size", 8)
    cfg.setdefault("proposal_candidate_episode_seconds", 2.0)
    cfg.setdefault("proposal_candidate_retry_interval_seconds", 1.0)
    cfg.setdefault("proposal_batch_new_frames_per_poll", False)
    cfg.setdefault("proposal_manifest_path", "")
    if str(cfg["proposal_manifest_path"]).strip():
        manifest_metadata = inspect_proposal_manifest(cfg["proposal_manifest_path"])
        cfg.update(manifest_metadata)
        cfg["proposal_count"] = int(
            manifest_metadata["proposal_manifest_max_proposal_count"]
        )
    cfg.setdefault("token_reduction_enabled", False)
    cfg.setdefault("token_reduction_layer", 12)
    cfg.setdefault("token_reduction_selector", "semantic")
    cfg.setdefault("embedding_backend", "qwen3vl")
    cfg.setdefault("verifier_prompt_mode", "task_specific")
    set_verifier_prompt_mode(str(cfg["verifier_prompt_mode"]))
    cfg.setdefault("token_reduction_stc_interval", 4)
    cfg.setdefault("token_reduction_budget", 0.1)
    cfg.setdefault("token_reduction_parallel", True)
    if cfg["decision_mode"] == "update_head":
        checkpoint = cfg.get("proactive_head_checkpoint")
        if not checkpoint or not Path(checkpoint).is_file():
            raise FileNotFoundError(
                f"Missing proactive_head_checkpoint: {checkpoint!r}"
            )
    # SemanticMemoryEncoder expects this field although proactive matching does
    # not build a persistent chunk memory.
    cfg["memory_chunk_frames"] = int(
        cfg["proposal_match_window_frames"]
        if cfg["decision_mode"] == "proposal_match"
        else cfg["recent_window_frames"]
    )
    if cfg["annotation_scope"] == "full250":
        cfg["annotation_path"] = cfg["extended_annotation_path"]
    if float(cfg["sample_fps"]) <= 0:
        raise ValueError("sample_fps must be positive")
    if int(cfg["recent_window_frames"]) <= 0:
        raise ValueError("recent_window_frames must be positive")
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


def output_config_snapshot(cfg: dict[str, Any]) -> dict[str, Any]:
    keep = {
        "benchmark",
        "track",
        "annotation_scope",
        "annotation_path",
        "extended_annotation_path",
        "video_root",
        "embedding_model_path",
        "embedding_helper_dir",
        "answer_model_path",
        "sample_fps",
        "recent_window_frames",
        "official_poll_interval_seconds",
        "official_max_after_ground_truth_seconds",
        "official_tolerance_seconds",
        "attn_implementation",
        "dtype",
        "max_new_tokens",
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
        "proposal_batch_new_frames_per_poll",
        "proposal_manifest_path",
        "proposal_manifest_sha256",
        "proposal_manifest_entry_count",
        "proposal_manifest_min_proposal_count",
        "proposal_manifest_max_proposal_count",
        "proposal_manifest_proposal_counts",
        "proposal_manifest_prompt_versions",
        "token_reduction_enabled",
        "token_reduction_layer",
        "token_reduction_budget",
        "token_reduction_parallel",
        "token_reduction_selector",
        "embedding_backend",
        "verifier_prompt_mode",
        "token_reduction_stc_interval",
    }
    snapshot = {
        **{key: value for key, value in cfg.items() if key in keep},
        "controller": controller_name(cfg["decision_mode"]),
        "controller_version": CONTROLLER_VERSION,
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


def default_output_suffix(cfg: dict[str, Any]) -> str:
    if cfg["decision_mode"] == "proposal_match":
        method = (
            "proposalmatchrearm"
            f"-p{int(cfg['proposal_count'])}"
            f"-win{int(cfg['proposal_match_window_frames'])}"
            f"-d{float(cfg['proposal_surge_threshold']):g}"
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
            method += f"-dupcd{float(cfg['proposal_duplicate_cooldown_seconds']):g}"
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
        if bool(cfg.get("proposal_batch_new_frames_per_poll", False)):
            method += "-gatebatch"
        if cfg.get("proposal_audit_only"):
            method += "-audit"
        else:
            method += (
                f"-ep{float(cfg['proposal_candidate_episode_seconds']):g}"
                f"-ri{float(cfg['proposal_candidate_retry_interval_seconds']):g}"
            )
    else:
        method = "embcalv9" if cfg["decision_mode"] == "update_head" else cfg["decision_mode"].replace("_", "")
    suffix = (
        f"{method}-{cfg['annotation_scope']}"
        f"-fps{float(cfg['sample_fps']):g}"
    )
    if (
        cfg["decision_mode"] == "proposal_match"
        and cfg.get("proposal_initial_verify")
    ):
        suffix += "-init"
    if not (
        cfg["decision_mode"] == "proposal_match"
        and cfg.get("proposal_audit_only")
    ):
        suffix += f"-recent{int(cfg['recent_window_frames'])}"
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


def verification_report(annotations: list[dict[str, Any]], cfg: dict[str, Any]) -> dict[str, Any]:
    fatal: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    duration_cache: dict[str, tuple[float, str]] = {}
    for item in annotations:
        problems: list[str] = []
        try:
            _, output = parse_public_trigger_spec(item["question"])
            if output != str(item["ground_truth_output"]):
                problems.append("public_output_mismatch")
        except (KeyError, ValueError):
            problems.append("invalid_public_instruction")
        path = str(item.get("video_path", ""))
        if not Path(path).is_file():
            problems.append("missing_video")
        else:
            if path not in duration_cache:
                duration_cache[path] = probe_video_duration_seconds(path)
            required = float(item["ground_truth_time_seconds"]) + float(
                cfg["official_max_after_ground_truth_seconds"]
            )
            if duration_cache[path][0] + 1.0 < required:
                problems.append("video_too_short_for_evaluation_horizon")
        if any(value in {"missing_video", "video_too_short_for_evaluation_horizon"} for value in problems):
            fatal.append({"question_id": item.get("question_id"), "problems": problems})
        elif problems:
            warnings.append(
                {"question_id": item.get("question_id"), "problems": problems}
            )
    return {
        "annotation_scope": cfg["annotation_scope"],
        "num_questions": len(annotations),
        "num_unique_videos": len({item["video"] for item in annotations}),
        "fatal_annotation_problems": fatal,
        "annotation_warnings": warnings,
        "controller": controller_name(cfg["decision_mode"]),
        "ground_truth_visible_to_models": False,
        "proposal_context_retrieval": (
            {
                "enabled": True,
                "query": "standing target question only",
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
            if bool(cfg.get("proposal_context_retrieval_enabled", False))
            else {"enabled": False}
        ),
        "protocol": (
            "At query registration, frozen Qwen3-VL-8B compiles a small set of "
            "observable proposals from causal context. Frozen Embedding-2B matches "
            "each causal sliding window to the cached proposal vectors. A one-second "
            "score surge above historical mean creates a duplicate-suppressed "
            "candidate episode. If the first bounded-KV 8B verdict is WAIT, the "
            "episode retries at the configured interval for a bounded duration. "
            "An unresolved expiry re-arms duplicate detection so newer evidence "
            "is not blocked by a stale cooldown; ordinary non-candidate frames "
            "preserve the existing public state."
            + (
                " In the opt-in retrieval-context branch, Embedding-2B first "
                "retrieves completed historical chunks using only the standing "
                "target; the selected frames augment the recent frames only for "
                "this one-time proposal compilation."
                if bool(cfg.get("proposal_context_retrieval_enabled", False))
                else ""
            )
            if cfg["decision_mode"] == "proposal_match"
            else
            "At the first decision the frozen Embedding-2B caches the latest recent "
            "independent images with the fixed question prefix; later polls append "
            "only each new image. One cached readout yields z, delta-z, native "
            "Yes/No cosines, margin and margin-delta. An event-coverage and "
            "call-sparsity trained temporal head proposes candidates; raw cosine "
            "cannot bypass it. A sparse bounded-KV frozen 8B verifier alone emits "
            "WAIT/TRIGGER."
            if cfg["decision_mode"] == "update_head"
            else (
                f"{cfg['decision_mode']} initializes from the latest recent window, "
                "then appends only each newly arrived independent image while "
                "reusing bounded multimodal KV. In 2b8b mode only a 2B Yes "
                "candidate invokes the sparse cached 8B verifier."
            )
        ),
    }


def poll_times(item: dict[str, Any], cfg: dict[str, Any]) -> list[float]:
    start = float(item["query_time_seconds"])
    stop = float(item["ground_truth_time_seconds"]) + float(
        cfg["official_max_after_ground_truth_seconds"]
    )
    interval = float(cfg["official_poll_interval_seconds"])
    count = max(0, int(np.floor((stop - start) / interval + 1e-9)))
    return [start + interval * index for index in range(1, count + 1)]


def run_record(
    *,
    item: dict[str, Any],
    controller: CachedProactiveController,
    cfg: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    condition, requested_output = parse_public_trigger_spec(item["question"])
    sampler = VideoSampler(item["video_path"], cfg, causal_fixed_fps=True)
    query_time = float(item["query_time_seconds"])
    registration_started = time.perf_counter()
    state = controller.register(
        query_id=str(item["question_id"]),
        target_question=condition,
        registration_timestamp=query_time,
        proposal_manifest_key=f"StreamingBench:PO:{item['question_id']}",
    )
    controller.prime(state=state, sampler=sampler, timestamp=query_time)
    registration_wall_seconds = time.perf_counter() - registration_started
    started = time.perf_counter()
    history: list[dict[str, Any]] = []
    answered = False
    trigger_time: float | None = None
    answer_metrics: dict[str, Any] | None = None
    gate_calls = 0
    gate_positive_count = 0
    gate_2b_seconds = 0.0
    verifier_calls = 0
    verifier_positive_count = 0
    verifier_8b_seconds = 0.0
    for timestamp in poll_times(item, cfg):
        decision = controller.decide(
            state=state, sampler=sampler, timestamp=float(timestamp)
        )
        gate_called = bool(decision.metrics.get("gate_2b_called", False))
        gate_calls += int(
            decision.metrics.get("gate_2b_calls_this_decision", gate_called)
        )
        gate_positive_count += int(
            decision.metrics.get(
                "gate_positive_count_this_decision",
                gate_called and decision.metrics.get("gate_positive") is True,
            )
        )
        gate_2b_seconds += float(decision.metrics.get("gate_2b_seconds", 0.0))
        verifier_called = bool(decision.metrics.get("verifier_8b_called", False))
        verifier_calls += int(
            decision.metrics.get("verifier_8b_calls_this_decision", verifier_called)
        )
        verifier_positive_count += int(
            decision.metrics.get(
                "verifier_positive_count_this_decision",
                verifier_called and decision.metrics.get("verifier_positive") is True,
            )
        )
        verifier_8b_seconds += float(
            decision.metrics.get("answer_total_seconds", 0.0)
        )
        history.append(
            {
                "timestamp_seconds": float(timestamp),
                "decision": "TRIGGER" if decision.positive else "WAIT",
                "positive": bool(decision.positive),
                "metrics": decision.metrics,
            }
        )
        if decision.positive:
            answered = True
            trigger_time = float(timestamp)
            answer_metrics = {
                "output_emission_mode": "public_instruction_deterministic",
                "selected_sample_positions": decision.positions,
                "selected_timestamps_seconds": decision.timestamps,
                "answer_frames": len(decision.positions),
            }
            break
    state_statistics = controller.state_statistics(state)
    proposal_compiler_calls = int(
        state_statistics.get("proposal_compiler_8b_calls", 0)
    )
    proposal_compiler_seconds = float(
        state_statistics.get("proposal_compiler_8b_seconds", 0.0)
    )
    proposal_context_retrieval = state.registration_metrics.get(
        "proposal_context_retrieval", {}
    )
    total_8b_calls = verifier_calls + proposal_compiler_calls
    result = {
        **item,
        "controller_name": controller.name,
        "trigger_condition": condition,
        "public_requested_output": requested_output,
        "answered": answered,
        "trigger_time_seconds": trigger_time,
        "output_response": requested_output if answered else "",
        "registration_metrics": state.registration_metrics,
        "decision_history": history,
        "metrics": {
            "gate_2b_calls": gate_calls,
            "gate_positive_count": gate_positive_count,
            "gate_2b_seconds": gate_2b_seconds,
            "verifier_8b_calls": verifier_calls,
            "verifier_positive_count": verifier_positive_count,
            "verifier_8b_seconds": verifier_8b_seconds,
            "proposal_compiler_8b_calls": proposal_compiler_calls,
            "proposal_compiler_8b_seconds": proposal_compiler_seconds,
            "proposal_context_retrieval_2b_calls": int(
                proposal_context_retrieval.get("retrieval_2b_calls", 0)
            ),
            "proposal_context_retrieval_seconds": float(
                proposal_context_retrieval.get("retrieval_total_seconds", 0.0)
            ),
            "registration_wall_seconds": registration_wall_seconds,
            "total_8b_calls": total_8b_calls,
            "cache_statistics": state_statistics,
            "answer_metrics": answer_metrics,
        },
    }
    source_metrics = {
        "source_key": item["question_id"],
        "video": item["video"],
        "source_online_wall_seconds": time.perf_counter() - started,
        "source_registration_wall_seconds": registration_wall_seconds,
        "gate_2b_calls": gate_calls,
        "gate_positive_count": gate_positive_count,
        "gate_2b_seconds": gate_2b_seconds,
        "verifier_8b_calls": verifier_calls,
        "verifier_positive_count": verifier_positive_count,
        "verifier_8b_seconds": verifier_8b_seconds,
        "proposal_compiler_8b_calls": proposal_compiler_calls,
        "proposal_compiler_8b_seconds": proposal_compiler_seconds,
        "proposal_context_retrieval_2b_calls": int(
            proposal_context_retrieval.get("retrieval_2b_calls", 0)
        ),
        "proposal_context_retrieval_chunk_encoder_2b_calls": int(
            proposal_context_retrieval.get("chunk_encoder_2b_calls", 0)
        ),
        "proposal_context_retrieval_cache_hits": int(
            proposal_context_retrieval.get("chunk_vector_cache_hits", 0)
        ),
        "proposal_context_retrieval_seconds": float(
            proposal_context_retrieval.get("retrieval_total_seconds", 0.0)
        ),
        "proposal_context_retrieved_frames": len(
            proposal_context_retrieval.get("retrieved_positions", [])
        ),
        "proposal_context_retrieval_cache_bytes": int(
            proposal_context_retrieval.get(
                "persistent_chunk_vector_cache_bytes", 0
            )
        ),
        "proposal_context_retrieval_peak_cache_bytes": int(
            proposal_context_retrieval.get("peak_chunk_vector_cache_bytes", 0)
        ),
        "total_8b_calls": total_8b_calls,
        "persistent_gate_document_vector_bytes": (
            controller.persistent_document_vector_bytes
        ),
        "max_2b_kv_cache_bytes": state_statistics.get(
            "2b_peak_cache_bytes", 0
        ),
        "max_8b_kv_cache_bytes": state_statistics.get(
            "8b_peak_cache_bytes", 0
        ),
        "persistent_proposal_vector_bytes": state_statistics.get(
            "persistent_proposal_vector_bytes", 0
        ),
        "persistent_shared_window_vector_cache_bytes": state_statistics.get(
            "persistent_shared_window_vector_cache_bytes", 0
        ),
        "persistent_shared_frame_vector_cache_bytes": state_statistics.get(
            "persistent_shared_frame_vector_cache_bytes", 0
        ),
        "persistent_shared_vit_feature_cache_bytes": state_statistics.get(
            "persistent_shared_vit_feature_cache_bytes", 0
        ),
        "proposal_raw_candidate_count": state_statistics.get(
            "proposal_raw_candidate_count", 0
        ),
        "proposal_suppressed_candidate_count": state_statistics.get(
            "proposal_suppressed_candidate_count", 0
        ),
        "proposal_candidate_count": state_statistics.get(
            "proposal_candidate_count", 0
        ),
        "proposal_candidate_episode_count": state_statistics.get(
            "proposal_candidate_episode_count", 0
        ),
        "proposal_retry_candidate_count": state_statistics.get(
            "proposal_retry_candidate_count", 0
        ),
        "proposal_retry_positive_count": state_statistics.get(
            "proposal_retry_positive_count", 0
        ),
        "proposal_retry_state_change_count": state_statistics.get(
            "proposal_retry_state_change_count", 0
        ),
        "proposal_episode_expired_count": state_statistics.get(
            "proposal_episode_expired_count", 0
        ),
        "proposal_wait_expiry_rearm_count": state_statistics.get(
            "proposal_wait_expiry_rearm_count", 0
        ),
        "proposal_state_hold_count": state_statistics.get(
            "proposal_state_hold_count", 0
        ),
        "proposal_window_encode_2b_calls": state_statistics.get(
            "proposal_window_encode_2b_calls", 0
        ),
        "proposal_window_cache_hits": state_statistics.get(
            "proposal_window_cache_hits", 0
        ),
        "proposal_frame_encode_2b_calls": state_statistics.get(
            "proposal_frame_encode_2b_calls", 0
        ),
        "proposal_frame_cache_hits": state_statistics.get(
            "proposal_frame_cache_hits", 0
        ),
        "proposal_vit_feature_encode_2b_calls": state_statistics.get(
            "proposal_vit_feature_encode_2b_calls", 0
        ),
        "proposal_vit_feature_cache_hits": state_statistics.get(
            "proposal_vit_feature_cache_hits", 0
        ),
        "proposal_vit_feature_seconds": state_statistics.get(
            "proposal_vit_feature_seconds", 0.0
        ),
        "proposal_joint_decoder_2b_calls": state_statistics.get(
            "proposal_joint_decoder_2b_calls", 0
        ),
        "proposal_joint_decoder_seconds": state_statistics.get(
            "proposal_joint_decoder_seconds", 0.0
        ),
        "2b_model_forward_calls": state_statistics.get(
            "2b_model_forward_calls", 0
        ),
        "8b_model_forward_calls": state_statistics.get(
            "8b_model_forward_calls", 0
        ),
    }
    return result, source_metrics


def score_record(record: dict[str, Any], tolerance: float) -> dict[str, Any]:
    trigger = record.get("trigger_time_seconds")
    delay = None if trigger is None else float(trigger) - float(record["ground_truth_time_seconds"])
    timing = delay is not None and abs(delay) <= tolerance
    content = bool(record.get("answered")) and str(record["ground_truth_output"]) in str(record.get("output_response", ""))
    return {
        "answered": bool(record.get("answered")),
        "delay_seconds": delay,
        "timing_correct": bool(timing),
        "answer_correct": bool(timing and content),
        "premature": delay is not None and delay < -tolerance,
        "late": delay is not None and delay > tolerance,
        "missed": trigger is None,
    }


def score_results(records: list[dict[str, Any]], tolerance: float) -> dict[str, Any]:
    scored = [score_record(value, tolerance) for value in records]
    return {
        "num_questions": len(records),
        "time_accuracy": float(np.mean([value["timing_correct"] for value in scored])) if scored else None,
        "answer_accuracy": float(np.mean([value["answer_correct"] for value in scored])) if scored else None,
        "answered": sum(value["answered"] for value in scored),
        "premature": sum(value["premature"] for value in scored),
        "late": sum(value["late"] for value in scored),
        "missed": sum(value["missed"] for value in scored),
    }


def assign_records(records: list[dict[str, Any]], world_size: int) -> list[list[dict[str, Any]]]:
    bins = [[] for _ in range(world_size)]
    for index, item in enumerate(records):
        bins[index % world_size].append(item)
    return bins


def build_manifest(cfg: dict[str, Any], records: list[dict[str, Any]], world_size: int) -> dict[str, Any]:
    scope = {
        "run_scope_version": 10,
        "controller": controller_name(cfg["decision_mode"]),
        "config": output_config_snapshot(cfg),
        "question_ids": [value["question_id"] for value in records],
        "world_size": world_size,
    }
    return {
        "fingerprint": hashlib.sha256(
            json.dumps(scope, sort_keys=True).encode()
        ).hexdigest(),
        **scope,
    }


def proposal_manifest_from_streamingbench_records(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    entries: dict[str, Any] = {}
    for record in records:
        compilation = record.get("registration_metrics", {}).get(
            "proposal_compilation", {}
        )
        key = compilation.get("proposal_manifest_key")
        texts = compilation.get("proposal_texts")
        vectors = compilation.get("proposal_vectors")
        if not key or not texts or vectors is None:
            continue
        entries[str(key)] = {
            "benchmark": "StreamingBench",
            "task": "STREAMINGBENCH_PO",
            "question_id": str(record["question_id"]),
            "target_question": str(record["trigger_condition"]),
            "proposal_prompt_version": compilation.get(
                "proposal_prompt_version", "target-anchored-v2"
            ),
            "compiler_context_frames": compilation.get("compiler_context_frames"),
            "compiler_sample_fps": compilation.get("compiler_sample_fps"),
            "proposal_texts": texts,
            "proposal_vectors": vectors,
            "source_manifest_replay": bool(compilation.get("manifest_replay", False)),
        }
    canonical = json.dumps(entries, sort_keys=True, ensure_ascii=False)
    return {
        "format": "biseek-proposal-manifest-v2",
        "selection_policy": "fixed before test scoring; never select by test accuracy",
        "entries_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
        "num_entries": len(entries),
        "entries": entries,
    }
def merge_outputs(output_dir: Path, world_size: int, cfg: dict[str, Any]) -> None:
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
    atomic_write_json(output_dir / "results.json", records)
    atomic_write_json(
        output_dir / "proposal_manifest.json",
        proposal_manifest_from_streamingbench_records(records),
    )
    decisions = [
        decision["metrics"]
        for record in records
        for decision in record["decision_history"]
    ]
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
        int(
            value.get(
                "calibrated_candidate_count_this_decision",
                value.get("gate_calibrated_candidate") is True,
            )
        )
        for value in decisions
    )
    raw_cosine_positive = sum(
        int(
            value.get(
                "raw_cosine_positive_count_this_decision",
                value.get("gate_raw_cosine_positive") is True,
            )
        )
        for value in decisions
    )
    raw_cosine_suppressed = sum(
        int(
            value.get(
                "raw_cosine_suppressed_count_this_decision",
                value.get("gate_raw_cosine_suppressed") is True,
            )
        )
        for value in decisions
    )
    if cfg["decision_mode"] == "update_head" and not (
        gate_positive == calibrated_candidates == verifier_calls
    ):
        raise RuntimeError(
            "Calibrated gate invariant failed: each positive candidate must invoke "
            "exactly one 8B verification"
        )
    gate_seconds = sum(float(value.get("gate_2b_seconds", 0.0)) for value in decisions)
    verifier_seconds = sum(
        float(value.get("answer_total_seconds", 0.0)) for value in decisions
    )
    proposal_compiler_calls = sum(
        int(value.get("proposal_compiler_8b_calls", 0)) for value in sources
    )
    proposal_compiler_seconds = sum(
        float(value.get("proposal_compiler_8b_seconds", 0.0))
        for value in sources
    )
    total_8b_calls = sum(int(value.get("total_8b_calls", 0)) for value in sources)
    proposal_raw_candidates = sum(
        int(value.get("proposal_raw_candidate_count", 0)) for value in sources
    )
    proposal_suppressed_candidates = sum(
        int(value.get("proposal_suppressed_candidate_count", 0))
        for value in sources
    )
    proposal_candidates = sum(
        int(value.get("proposal_candidate_count", 0)) for value in sources
    )
    proposal_episodes = sum(
        int(value.get("proposal_candidate_episode_count", 0))
        for value in sources
    )
    proposal_retries = sum(
        int(value.get("proposal_retry_candidate_count", 0))
        for value in sources
    )
    proposal_retry_positives = sum(
        int(value.get("proposal_retry_positive_count", 0))
        for value in sources
    )
    proposal_retry_state_changes = sum(
        int(value.get("proposal_retry_state_change_count", 0))
        for value in sources
    )
    efficiency = {
            "num_questions": len(records),
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
            "proposal_compiler_8b_calls": proposal_compiler_calls,
            "proposal_compiler_8b_seconds": proposal_compiler_seconds,
            "proposal_context_retrieval_2b_calls": sum(
                int(value.get("proposal_context_retrieval_2b_calls", 0))
                for value in sources
            ),
            "proposal_context_retrieval_chunk_encoder_2b_calls": sum(
                int(
                    value.get(
                        "proposal_context_retrieval_chunk_encoder_2b_calls", 0
                    )
                )
                for value in sources
            ),
            "proposal_context_retrieval_cache_hits": sum(
                int(value.get("proposal_context_retrieval_cache_hits", 0))
                for value in sources
            ),
            "proposal_context_retrieval_seconds": sum(
                float(value.get("proposal_context_retrieval_seconds", 0.0))
                for value in sources
            ),
            "proposal_context_retrieved_frames": sum(
                int(value.get("proposal_context_retrieved_frames", 0))
                for value in sources
            ),
            "max_proposal_context_retrieval_cache_bytes": max(
                (
                    int(
                        value.get(
                            "proposal_context_retrieval_peak_cache_bytes", 0
                        )
                    )
                    for value in sources
                ),
                default=0,
            ),
            "proposal_raw_candidate_count": proposal_raw_candidates,
            "proposal_suppressed_candidate_count": proposal_suppressed_candidates,
            "proposal_candidate_count": proposal_candidates,
            "proposal_candidate_episode_count": proposal_episodes,
            "proposal_retry_candidate_count": proposal_retries,
            "proposal_retry_positive_count": proposal_retry_positives,
            "proposal_retry_state_change_count": proposal_retry_state_changes,
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
                int(value.get("proposal_frame_cache_hits", 0))
                for value in sources
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
            "mean_verifier_8b_seconds_per_call": (
                verifier_seconds / max(1, verifier_calls)
            ),
            "total_8b_calls": total_8b_calls,
            "8b_call_fraction_of_2b_polls": (
                verifier_calls / gate_calls if gate_calls else None
            ),
            "8b_calls_avoided_by_gate": (
                gate_calls - verifier_calls if gate_calls else None
            ),
            "mean_verifier_8b_calls_per_question": verifier_calls / max(1, len(records)),
            "mean_total_8b_calls_per_question": total_8b_calls / max(1, len(records)),
            "proactive_head_operating_point": output_config_snapshot(cfg).get(
                "proactive_head_operating_point"
            ),
            "max_persistent_gate_document_vector_bytes": max(
                (value.get("persistent_gate_document_vector_bytes", 0) for value in sources),
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
                (value.get("persistent_proposal_vector_bytes", 0) for value in sources),
                default=0,
            ),
            "max_persistent_shared_window_vector_cache_bytes": max(
                (
                    value.get(
                        "persistent_shared_window_vector_cache_bytes", 0
                    )
                    for value in sources
                ),
                default=0,
            ),
            "max_persistent_shared_frame_vector_cache_bytes": max(
                (
                    value.get("persistent_shared_frame_vector_cache_bytes", 0)
                    for value in sources
                ),
                default=0,
            ),
            "max_persistent_shared_vit_feature_cache_bytes": max(
                (
                    value.get("persistent_shared_vit_feature_cache_bytes", 0)
                    for value in sources
                ),
                default=0,
            ),
            "2b_model_forward_calls": sum(
                value.get("2b_model_forward_calls", 0) for value in sources
            ),
            "8b_model_forward_calls": sum(
                value.get("8b_model_forward_calls", 0) for value in sources
            ),
            # Explicit aliases: gate/verifier/compiler counts are semantic
            # service calls, whereas *_model_forward_calls count low-level
            # model.forward invocations.  Cache A can execute one ViT forward
            # per newly encoded frame plus one joint decoder forward per gate,
            # so low-level 2B forwards are deliberately larger than polls.
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
                    "ViT feature encoding, proposal-context retrieval, and decoder "
                    "readout work when applicable."
                ),
                "8b_high_level_calls": (
                    "Proposal compiler plus verifier calls at the service/API level."
                ),
                "8b_low_level_forward_calls": (
                    "Underlying 8B model.forward invocations used to append frames "
                    "and produce cached-session readouts."
                ),
            },
    }
    summary = score_results(records, float(cfg["official_tolerance_seconds"]))
    summary.update(
        {
            "benchmark": "StreamingBench",
            "track": "proactive_output",
            "annotation_scope": str(cfg["annotation_scope"]),
            "annotation_path": str(cfg["annotation_path"]),
            "official_contextual_table_split": (
                str(cfg["annotation_scope"]) == "full250" and len(records) == 250
            ),
        }
    )
    summary["efficiency"] = efficiency
    atomic_write_json(output_dir / "summary.json", summary)
    atomic_write_json(output_dir / "efficiency_summary.json", efficiency)
    atomic_write_json(
        output_dir / "serving_efficiency_summary.json",
        summarize_proactive_efficiency(output_dir),
    )


def main() -> None:
    args = parse_args()
    cfg = apply_overrides(load_json(args.config), args)
    rank, world_size, _, device = distributed_context(args.verify_only)
    annotations = load_json(cfg["annotation_path"])
    if args.max_samples is not None:
        annotations = annotations[: args.max_samples]
    annotations = [{**item, "_index": index} for index, item in enumerate(annotations)]
    output_dir = Path(args.output_dir or Path(cfg["output_root"]) / default_output_suffix(cfg))
    output_dir.mkdir(parents=True, exist_ok=True)
    verify = verification_report(annotations, cfg)
    if rank == 0:
        atomic_write_json(output_dir / "verification_report.json", verify)
    if args.verify_only:
        if rank == 0:
            print(json.dumps(verify, indent=2))
        return
    if verify["fatal_annotation_problems"]:
        raise RuntimeError(json.dumps(verify["fatal_annotation_problems"][:5], indent=2))
    assert device is not None
    manifest = build_manifest(cfg, annotations, world_size)
    manifest_path = output_dir / "run_manifest.json"
    if manifest_path.is_file() and not args.overwrite:
        existing_manifest = load_json(manifest_path)
        if not isinstance(existing_manifest, dict):
            raise RuntimeError(
                f"Invalid run manifest in {manifest_path}; use --overwrite once "
                "or choose a distinct OUTPUT_DIR."
            )
        if existing_manifest.get("fingerprint") != manifest["fingerprint"]:
            raise RuntimeError(
                "Run scope mismatch; use a distinct OUTPUT_DIR or --overwrite"
            )
    if rank == 0:
        atomic_write_json(manifest_path, manifest)
        atomic_write_json(output_dir / "inference_config.json", output_config_snapshot(cfg))
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
    pending = [item for item in assign_records(annotations, world_size)[rank] if item["question_id"] not in done]
    metrics_by_key = {
        value["source_key"]: value
        for value in (load_json(metrics_path) if metrics_path.is_file() else [])
    }
    if pending:
        answer_model = AnswerModel(cfg, device)
        encoder = (
            SemanticMemoryEncoder(cfg, "full_embedding", device)
            if cfg["decision_mode"] in {
                "2b8b_cache",
                "update_head",
                "proposal_match",
            }
            else None
        )
        controller = build_proactive_controller(
            encoder=encoder,
            answer_model=answer_model,
            cfg=cfg,
            task="STREAMINGBENCH_PO",
            decision_mode=cfg["decision_mode"],
        )
        with rank_path.open("a", buffering=1) as handle:
            for local_index, item in enumerate(pending, 1):
                result, source = run_record(item=item, controller=controller, cfg=cfg)
                result.update({"_key": item["question_id"], "_index": item["_index"], "rank": rank})
                metrics_by_key[source["source_key"]] = source
                atomic_write_json(metrics_path, list(metrics_by_key.values()))
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                print(f"[rank {rank}] {local_index}/{len(pending)} {item['question_id']} answered={result['answered']} 8B_calls={source['total_8b_calls']}", flush=True)
    run_id = os.environ.get("TORCHELASTIC_RUN_ID", "single-process")
    done_path.write_text(json.dumps({"run_id": run_id, "rank": rank}) + "\n")
    if rank == 0:
        markers = [output_dir / f"rank_{value:02d}.done" for value in range(world_size)]
        while not all(path.is_file() and load_json(path).get("run_id") == run_id for path in markers):
            time.sleep(10)
        merge_outputs(output_dir, world_size, cfg)
        print(f"Completed StreamingBench Proactive Output: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
