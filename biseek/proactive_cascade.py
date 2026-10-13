#!/usr/bin/env python3
"""Frozen proactive controllers with bounded multimodal KV.

The frozen baselines and the trained update-head controller share the same
causal image protocol:

``8b_cache``
    Qwen3-VL-8B decides at every observation.  The first decision sees the
    configured recent window; later decisions append only the newly arrived
    independent image and reuse a bounded question+frame KV cache.

``2b8b_cache``
    Qwen3-VL-Embedding-2B makes the same incremental Yes/No cosine decision.
    Only a positive 2B candidate invokes the 8B verifier.  The 2B cache keeps
    every recent observation; the 8B cache is intentionally sparse and keeps
    only candidate observations that the verifier actually processed.

``update_head``
    The same cached 2B readout and its native Yes/No document cosines are
    calibrated by a small temporal head trained with an event-coverage and
    explicit call-sparsity objective.  This single probability proposes
    candidates; raw cosine is diagnostic only and cannot bypass the head.  The
    bounded cached 8B verifier makes the final decision.

The cache stores decoder keys/values, not the project's compact 2048-D vector
memory.  It is bounded by frames: the fixed question prefix is retained while
the oldest frame block is physically removed from every decoder layer before
a new frame is appended.  Explicit offset 3-D position IDs are supplied for
every newly appended image; the stock generation shortcut assumes that cached
continuations are text-only and is therefore not correct for this use case.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from biseek.proactive_head import (
    CACHED_ANSWER_PRESENT,
    NO_CACHED_ANSWER,
    PROACTIVE_TASK_IDS,
    PROACTIVE_SCORE_FEATURE_NAMES,
    load_temporal_update_head,
    update_time_feature,
)


DECISION_MODES = ("8b_cache", "2b8b_cache", "update_head", "proposal_match")
PROACTIVE_TASKS = ("STREAMINGBENCH_PO", "REC", "SSR", "CRR", "STREAMARENA_PRO")
CONTROLLER_VERSION = "11.13"
PROPOSAL_PROMPT_STYLES = (
    "target_anchored_v2",
    "target_anchor_plus3_v7",
    "target_anchor_plusn_v8",
)
PROPOSAL_PROMPT_VERSIONS = {
    "target_anchored_v2": "target-anchored-v2",
    "target_anchor_plus3_v7": "target-anchor-plus-three-v7",
    "target_anchor_plusn_v8": "target-anchor-plus-n-v8",
}
DEFAULT_PROPOSAL_PROMPT_STYLE = "target_anchored_v2"

from verifier_margin_defaults import (  # noqa: F401  (re-exported for the eval scripts)
    BACKBONE_VERIFIER_ACCEPT_MARGIN,
    default_verifier_accept_margin,
)

VERIFIER_PROMPT_MODES = (
    "task_specific",
    "agnostic",
    "perceptual",
    "strict_now",
    "yes_no",
)
_VERIFIER_PROMPT_MODE = "task_specific"


def set_verifier_prompt_mode(mode: str) -> None:
    global _VERIFIER_PROMPT_MODE
    if mode not in VERIFIER_PROMPT_MODES:
        raise ValueError(
            f"verifier_prompt_mode must be one of {VERIFIER_PROMPT_MODES}, got {mode!r}"
        )
    _VERIFIER_PROMPT_MODE = mode


PROPOSAL_PROMPT_VERSION = PROPOSAL_PROMPT_VERSIONS[DEFAULT_PROPOSAL_PROMPT_STYLE]
GATE_PROMPT_VERSION = "evidence-readiness-v2"
CACHED_UPDATE_FEATURE_PROTOCOL = "bounded_multimodal_kv_embedding_calibration_v2"
UPDATE_READOUT_PROMPT_VERSION = "abstract-cached-answer-update-v3"
GATE_DOCUMENTS = (
    "Yes. All visual evidence required to resolve the standing target has "
    "already been observed.",
    "No. Some visual evidence required to resolve the standing target has "
    "not been observed yet.",
)
GATE_DOCUMENT_INSTRUCTION = (
    "Represent this evidence-readiness statement for matching to a causal "
    "visual question state."
)


def inspect_proposal_manifest(path: str | Path) -> dict[str, Any]:
    """Return replay-critical metadata for fixed or variable-count manifests.

    Legacy manifests may contain fewer proposals for samples whose format
    repair was incomplete.  Their exact SHA plus the full count/version sets
    identify that replay protocol, so report the mixture explicitly rather
    than silently pretending it is fixed-count.
    """
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing proposal manifest: {manifest_path}")
    digest = hashlib.sha256()
    with manifest_path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    loaded = json.loads(manifest_path.read_text())
    entries = loaded.get("entries", loaded)
    if not isinstance(entries, dict) or not entries:
        raise ValueError("proposal manifest entries must be a non-empty JSON object")
    counts = {
        len(entry.get("proposal_texts", []))
        for entry in entries.values()
        if isinstance(entry, dict)
    }
    if not counts or min(counts) <= 0:
        raise ValueError(f"proposal manifest has invalid cue counts: {counts}")
    versions = sorted(
        {
            str(entry.get("proposal_prompt_version", "unknown"))
            for entry in entries.values()
            if isinstance(entry, dict)
        }
    )
    return {
        "proposal_manifest_sha256": digest.hexdigest(),
        "proposal_manifest_entry_count": len(entries),
        "proposal_manifest_min_proposal_count": min(counts),
        "proposal_manifest_max_proposal_count": max(counts),
        "proposal_manifest_proposal_counts": sorted(counts),
        "proposal_manifest_prompt_versions": versions,
    }


@dataclass(frozen=True)
class TaskProfile:
    name: str
    positive_state_absorbing: bool


TASK_PROFILES = {
    "STREAMINGBENCH_PO": TaskProfile("streamingbench_proactive_output", True),
    "REC": TaskProfile("ovo_repetition_count_state", False),
    "SSR": TaskProfile("ovo_current_step_state", False),
    "CRR": TaskProfile("ovo_clue_completion_state", True),
    # StreamArena Pro: the user asks now and the answer becomes available later.
    # Absorbing like StreamingBench PO -- the judge reads only the FIRST
    # proactive utterance, so speaking is a one-shot commitment.
    "STREAMARENA_PRO": TaskProfile("streamarena_proactive_interaction", True),
}


def controller_name(decision_mode: str) -> str:
    value = str(decision_mode)
    if value == "8b_cache":
        return "8b-only-bounded-multimodal-kv"
    if value == "2b8b_cache":
        return "embedding-2b-gated-8b-bounded-multimodal-kv"
    if value == "update_head":
        return "cached-embedding-2b-event-sparse-gated-8b-kv"
    if value == "proposal_match":
        return "8b-proposal-compiled-embedding-2b-surge-gated-8b-kv"
    raise ValueError(value)


def parse_visual_proposals(response: str, *, limit: int) -> list[str]:
    """Parse a deterministic proposal list without trusting one output format."""

    text = str(response).strip()
    values: list[str] = []
    # Prefer a complete JSON value. The compiler can hit its generation limit
    # after emitting several complete strings but before the closing bracket.
    # The old line fallback then treated JSON punctuation and `"...,` fragments
    # as proposals, silently poisoning the matcher.
    candidates = [text, *re.findall(r"\[[\s\S]*?\]", text)]
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, list):
            values = [str(value).strip() for value in parsed]
            break
    if not values and text.lstrip().startswith("["):
        # Salvage only fully closed JSON strings from a truncated array. Never
        # index the opening bracket or the unfinished last string as a cue.
        for encoded in re.findall(r'"((?:\\.|[^"\\])*)"', text):
            try:
                values.append(json.loads(f'"{encoded}"').strip())
            except (json.JSONDecodeError, AttributeError):
                continue
    if not values:
        for line in text.splitlines():
            value = re.sub(
                r"^\s*(?:[-*]|\d+[.)]|proposal\s*\d+\s*:)\s*",
                "",
                line,
                flags=re.IGNORECASE,
            ).strip()
            value = value.strip('"\'` ')
            if value and value not in {"[", "]", "[]"}:
                values.append(value)
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = re.sub(r"\s+", " ", value).strip(" .,[]")
        if normalized.endswith('"'):
            normalized = normalized[:-1].rstrip()
        key = normalized.casefold()
        if not normalized or key in seen:
            continue
        seen.add(key)
        output.append(normalized)
        if len(output) >= int(limit):
            break
    return output


def proposal_surge_candidate(
    *,
    current_scores: torch.Tensor,
    history: list[tuple[float, torch.Tensor]],
    timestamp: float,
    lookback_seconds: float,
    surge_threshold: float,
    require_above_history_mean: bool,
) -> dict[str, Any]:
    """Em-Garde-style proposal surge test with an explicit one-second lag."""

    current = current_scores.detach().float().cpu().reshape(-1)
    reference: tuple[float, torch.Tensor] | None = None
    cutoff = float(timestamp) - float(lookback_seconds)
    for item in history:
        if float(item[0]) <= cutoff + 1e-6:
            reference = item
        else:
            break
    if reference is None:
        return {
            "candidate": False,
            "reason": "insufficient_lookback_history",
            "winner_index": int(torch.argmax(current).item()),
            "winner_score": float(torch.max(current).item()),
            "winner_delta": None,
            "above_history_mean": False,
            "reference_timestamp_seconds": None,
        }
    delta = current - reference[1]
    winner = int(torch.argmax(delta).item())
    history_mean = torch.stack([value for _, value in history], dim=0).mean(dim=0)
    above_mean = bool(current[winner] > history_mean[winner])
    surge = float(delta[winner].item())
    candidate = bool(
        surge > float(surge_threshold)
        and (above_mean or not bool(require_above_history_mean))
    )
    return {
        "candidate": candidate,
        "reason": (
            "proposal_score_surge"
            if candidate
            else "below_surge_threshold_or_history_mean"
        ),
        "winner_index": winner,
        "winner_score": float(current[winner].item()),
        "winner_delta": surge,
        "above_history_mean": above_mean,
        "reference_timestamp_seconds": float(reference[0]),
        "history_mean_at_winner": float(history_mean[winner].item()),
    }


def parse_binary_verdict(response: str, task: str = "STREAMINGBENCH_PO") -> bool:
    """True when the verifier emitted its positive label.

    The positive label is "B" for every letter-based wording and "Yes" under
    the yes_no wording, so this has to ask which pair is in force rather than
    hard-coding "B" -- otherwise yes_no silently reads every Yes as a refusal.
    """

    _, positive_label = verifier_label_pair(task)
    return str(response).strip().upper() == positive_label.upper()


def recent_positions(sampler: Any, timestamp: float, window_frames: int) -> list[int]:
    available = [
        index
        for index, value in enumerate(sampler.sample_timestamps)
        if float(value) <= float(timestamp) + 1e-6
    ]
    return available[-int(window_frames) :]


def _cache_bytes(cache: Any) -> int:
    if cache is None:
        return 0
    total = 0
    for layer in getattr(cache, "layers", []):
        for name in ("keys", "values"):
            value = getattr(layer, name, None)
            if torch.is_tensor(value):
                total += int(value.numel() * value.element_size())
    return total


def _remove_cache_token_range(cache: Any, start: int, end: int) -> None:
    """Remove a physical token interval from every DynamicCache layer."""

    if cache is None or end <= start:
        return
    for layer in getattr(cache, "layers", []):
        for name in ("keys", "values"):
            value = getattr(layer, name, None)
            if not torch.is_tensor(value):
                continue
            if value.shape[-2] < end:
                raise RuntimeError(
                    f"Cannot prune cache range [{start}, {end}) from {name} "
                    f"with sequence length {value.shape[-2]}"
                )
            setattr(
                layer,
                name,
                torch.cat((value[..., :start, :], value[..., end:, :]), dim=-2),
            )


def _concat_processor_batches(values: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    if not values:
        raise ValueError("At least one processor batch is required")
    output: dict[str, torch.Tensor] = {}
    sequence_keys = {"input_ids", "attention_mask", "mm_token_type_ids", "token_type_ids"}
    row_keys = {"pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw"}
    all_keys = set().union(*(value.keys() for value in values))
    for key in all_keys:
        present = [value[key] for value in values if key in value]
        if len(present) != len(values) and key in sequence_keys:
            raise RuntimeError(f"Processor sequence key {key!r} missing from one frame block")
        if not present:
            continue
        if key in sequence_keys:
            output[key] = torch.cat(present, dim=1)
        elif key in row_keys:
            output[key] = torch.cat(present, dim=0)
        elif len(present) == 1:
            output[key] = present[0]
        else:
            raise RuntimeError(f"Unsupported repeated processor key: {key}")
    return output


def _task_system_prompt(task: str) -> str:
    if task in {"STREAMINGBENCH_PO", "REC", "SSR", "CRR", "STREAMARENA_PRO"}:
        return (
            "You monitor a causal video stream. Use only observations available "
            "so far to decide whether the standing target requires publishing a "
            "new or revised answer now."
        )
    raise ValueError(task)


def _agnostic_verifier_question(task: str, question: str, timestamp: float) -> str:
    """One wording for every task, for the mode-agnostic ablation.

    The default verifier prompt is phrased per task, which is what the OVO
    harness itself does -- its constant.py ships four templates and ViSpeak and
    Dispider both consume whichever one the harness hands them.  StreamingBench
    PO has no such excuse: it is a single task type, so any PO-specific phrasing
    is ours rather than the protocol's.  This variant strips every task-shaped
    hint down to the generic standing-instruction test, so the run answers
    whether the score depends on that phrasing.

    REC keeps its four-letter answer set because it reports a count rather than a
    yes/no; removing that would change what is being asked, not how it is worded.
    """

    if task == "REC":
        body = (
            f"Standing instruction: {question}\n"
            "How many new occurrences of the described event have completed in "
            "the newly appended observations since the previous decision?"
        )
        labels = "Choose A for 0, B for 1, C for 2, or D for 3 or more."
        letters = "A, B, C, or D"
    else:
        body = (
            f"Standing instruction: {question}\n"
            "If the condition in the standing instruction is satisfied now, "
            "produce the required output; otherwise WAIT."
        )
        labels = "Choose B to produce the output or A to WAIT."
        letters = "A or B"
    return (
        f"Current stream time: {timestamp:.3f} seconds.\n{body}\n{labels}\n"
        f"Answer with exactly one letter: {letters}."
    )


def _perceptual_verifier_question(task: str, question: str, timestamp: float) -> str:
    """Strip the decision framing and ask only what is on screen.

    Both existing wordings wrap the judgement in output/WAIT semantics, so the
    model has to reason about whether to emit rather than about what it sees.
    On StreamingBench PO a Qwen2-VL-7B verifier returned near-tied A/B logits
    under the agnostic wording (accept margin +0.80 against Qwen3's +12.20),
    i.e. no usable signal; the same model was fine on OVO under task_specific
    wording, which points at the framing rather than the model. This variant
    asks a bare perceptual question and nothing else.
    """

    if task == "REC":
        body = (
            f"Activity: {question}\n"
            "In the newest observations only, how many times has this activity "
            "been completed since the previous check?"
        )
        labels = "Choose A for 0, B for 1, C for 2, or D for 3 or more."
        letters = "A, B, C, or D"
    else:
        body = (
            f"Look at the most recent observation.\n"
            f"Is this visible right now: {question}"
        )
        labels = "Choose B for yes or A for no."
        letters = "A or B"
    return (
        f"Current stream time: {timestamp:.3f} seconds.\n{body}\n{labels}\n"
        f"Answer with exactly one letter: {letters}."
    )


def _strict_now_verifier_question(task: str, question: str, timestamp: float) -> str:
    """task_specific plus an explicit anti-anticipation clause.

    Premature firing is the dominant error on StreamingBench PO (66 of 170
    triggers for the Qwen2-VL verifier, 19 of 173 for Qwen3), so this variant
    keeps the per-task phrasing but forbids answering on something that is
    about to happen rather than already visible.
    """

    if task == "REC":
        return _verifier_question_task_specific(task, question, timestamp)
    base = _verifier_question_task_specific(task, question, timestamp)
    clause = (
        "Answer B only if the condition is already visible in the newest "
        "observation. If it merely looks likely to happen soon, answer A."
    )
    head, _, tail = base.rpartition("Answer with exactly one letter:")
    return f"{head}{clause}\nAnswer with exactly one letter:{tail}"


def verifier_label_pair(task: str) -> tuple[str, str]:
    """(negative, positive) next-token labels the verdict reads logits for.

    Every mode but yes_no uses the arbitrary letters A/B, which force the model
    to bind "A means wait, B means trigger" before it can answer. Em-Garde's
    own verifier instead asks for 'Yes'/'No' (see its
    `Does one of the following descriptions match ...` prompt), tokens that
    carry their own meaning. A Qwen2-VL-7B verifier returned near-tied A/B
    logits here (+0.80 accept, -0.59 reject, against Qwen3's +12.20/-9.94),
    which is what a failed label binding looks like, so the yes_no mode tests
    that explanation directly. REC reports a count and keeps its letters.
    """

    if task == "REC" or _VERIFIER_PROMPT_MODE != "yes_no":
        return ("A", "B")
    return ("No", "Yes")


def _yes_no_verifier_question(task: str, question: str, timestamp: float) -> str:
    """Perceptual question answered with semantically loaded Yes/No tokens."""

    if task == "REC":
        return _verifier_question_task_specific(task, question, timestamp)
    return (
        f"Current stream time: {timestamp:.3f} seconds.\n"
        f"Does what is happening in the most recent observation match this "
        f"description: {question}\n"
        "If it matches, answer 'Yes'. Otherwise, answer 'No'.\n"
        "Only answer in 'Yes' or 'No'."
    )


def _verifier_question(task: str, question: str, timestamp: float) -> str:
    if _VERIFIER_PROMPT_MODE == "agnostic":
        return _agnostic_verifier_question(task, question, timestamp)
    if _VERIFIER_PROMPT_MODE == "perceptual":
        return _perceptual_verifier_question(task, question, timestamp)
    if _VERIFIER_PROMPT_MODE == "strict_now":
        return _strict_now_verifier_question(task, question, timestamp)
    if _VERIFIER_PROMPT_MODE == "yes_no":
        return _yes_no_verifier_question(task, question, timestamp)
    return _verifier_question_task_specific(task, question, timestamp)


def _verifier_question_task_specific(task: str, question: str, timestamp: float) -> str:
    if task in {"STREAMINGBENCH_PO", "STREAMARENA_PRO"}:
        body = (
            f"Trigger condition: {question}\n"
            "Is this exact visual condition satisfied now? Do not add extra "
            "requirements such as centering, persistence, or lack of occlusion."
        )
        labels = "Choose B to TRIGGER or A to WAIT."
    elif task == "SSR":
        body = (
            f"Target tutorial step: {question}\n"
            "Is the person CURRENTLY performing this exact step? The newest "
            "observation determines the current state; cached observations are context."
        )
        labels = "Choose B for Yes or A for No."
    elif task == "CRR":
        body = (
            f"Target question: {question}\n"
            "Is the observed evidence now sufficient to answer this question "
            "without guessing? Do not predict an unseen future answer."
        )
        labels = "Choose B for Yes or A for No."
    elif task == "REC":
        body = (
            f"Repeated activity: {question}\n"
            "Count only NEW, fully completed repetitions visible in the newly "
            "appended observations since the preceding verifier decision. Cached "
            "older observations are context and must not be counted again."
        )
        labels = (
            "Choose A for 0 new completions, B for 1, C for 2, or D for 3 or more."
        )
    else:
        raise ValueError(task)
    output_letters = "A, B, C, or D" if task == "REC" else "A or B"
    return (
        f"Current stream time: {timestamp:.3f} seconds.\n{body}\n{labels}\n"
        f"Answer with exactly one letter: {output_letters}."
    )


def _gate_question(timestamp: float) -> str:
    return (
        f"Current stream time: {timestamp:.3f} seconds.\n"
        "Based only on the causal visual observations available up to the "
        "current time, determine whether all visual evidence required to "
        "resolve the standing target has already been observed. Resolving the "
        "target means answering its question or satisfying its requested visual "
        "condition. Retrieve the matching evidence-readiness statement."
    )


def _update_readout_question(timestamp: float, cached_answer: str) -> str:
    """Readout prompt shared by cached feature extraction and online serving."""

    answer_state = (
        NO_CACHED_ANSWER
        if not cached_answer or cached_answer == NO_CACHED_ANSWER
        else CACHED_ANSWER_PRESENT
    )
    return (
        f"Current stream time: {timestamp:.3f} seconds.\n"
        f"Published-answer state: {answer_state}\n"
        "Represent whether the newly observed causal evidence requires "
        "publishing a new or revised answer now."
    )


@dataclass
class KVSessionStats:
    model_forward_calls: int = 0
    decision_calls: int = 0
    total_forward_seconds: float = 0.0
    peak_cache_tokens: int = 0
    peak_cache_bytes: int = 0
    evicted_frame_blocks: int = 0
    evicted_tokens: int = 0
    visual_frames_encoded: int = 0
    token_reduction_calls: int = 0
    visual_tokens_before_reduction: int = 0
    visual_tokens_after_reduction: int = 0
    token_reduction_seconds: float = 0.0


class IncrementalMultimodalKVSession:
    """One query-specific bounded multimodal cache for either frozen model."""

    def __init__(
        self,
        *,
        backend: str,
        owner: Any,
        target_question: str,
        task: str,
        frame_window: int,
        reset_every_n_frames: int = 0,
    ) -> None:
        if backend not in {"8b", "2b"}:
            raise ValueError(backend)
        self.backend = backend
        self.owner = owner
        self.target_question = str(target_question)
        self.task = task
        self.frame_window = int(frame_window)
        if self.frame_window <= 0:
            raise ValueError("frame_window must be positive")
        # A sliding window only deletes the evicted frames' OWN cached K/V.
        # Every frame still in the window had its K/V computed by causal
        # self-attention *while the now-evicted frames were still present*,
        # so their hidden states -- and therefore their K/V at every layer --
        # keep carrying influence from content that eviction was supposed to
        # remove. Over a long poll sequence this residue accumulates and is
        # the leading suspect for why the incremental-KV gate (cache B) reads
        # worse standalone / lower-precision than the joint-decoder path
        # (cache A), which re-encodes its window fresh every poll. Hard-reset
        # every N *ingested* frames trades some of that cheapness back for a
        # periodic clean slate. 0 disables it (unchanged prior behavior).
        if reset_every_n_frames < 0:
            raise ValueError("reset_every_n_frames must be >= 0")
        self.reset_every_n_frames = int(reset_every_n_frames)
        self.total_frames_appended = 0
        self.hard_resets = 0
        if backend == "8b":
            self.wrapper = owner.model
            self.core = owner.model.model
            self.processor = owner.processor
            self.device = owner.device
        else:
            self.wrapper = owner.embedder.model
            self.core = owner.embedder.model.model
            self.processor = owner.embedder.processor
            self.device = owner.device
        self.cache: Any = None
        self.prefix_tokens = 0
        self.frame_block_tokens: list[int] = []
        self.next_mrope_position = 0
        self.initialized = False
        self.assistant_open = False
        self.stats = KVSessionStats()

    def _hard_reset(self) -> None:
        # Same as a brand-new session, minus re-touching the counters that
        # track a full history of this session's use (hard_resets itself,
        # total_frames_appended, and cumulative KVSessionStats).
        self.cache = None
        self.prefix_tokens = 0
        self.frame_block_tokens = []
        self.next_mrope_position = 0
        self.initialized = False
        self.assistant_open = False
        self.hard_resets += 1

    def _processor_batch(self, text: str, image: Image.Image | None = None) -> dict[str, torch.Tensor]:
        kwargs: dict[str, Any] = {
            "text": [text],
            "padding": False,
            "return_tensors": "pt",
        }
        if image is not None:
            kwargs["images"] = [image]
        value = self.processor(**kwargs)
        return {
            key: tensor.to(self.device) if torch.is_tensor(tensor) else tensor
            for key, tensor in value.items()
        }

    def _prefix_text(self) -> str:
        if self.backend == "2b":
            system = (
                "Represent the readiness of the causal visual evidence for the "
                "standing target."
            )
        else:
            system = _task_system_prompt(self.task)
        return (
            f"<|im_start|>system\n{system}<|im_end|>\n"
            "<|im_start|>user\n"
            f"Standing target: {self.target_question}\n"
            "Causal observations follow in chronological order."
        )

    def _frame_text(
        self,
        timestamp: float,
        *,
        decision: bool,
        continuation: bool,
        decision_prompt: str | None = None,
    ) -> str:
        prefix = ""
        if continuation:
            prefix = "<|im_end|>\n<|im_start|>user\n"
        text = (
            f"{prefix}\nObservation at {timestamp:.3f} seconds:\n"
            "<|vision_start|><|image_pad|><|vision_end|>\n"
        )
        if decision:
            prompt = (
                str(decision_prompt)
                if decision_prompt is not None
                else _gate_question(timestamp)
                if self.backend == "2b"
                else _verifier_question(self.task, self.target_question, timestamp)
            )
            text += f"{prompt}<|im_end|>\n<|im_start|>assistant\n"
        return text

    def _local_position_ids(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        input_ids = inputs["input_ids"]
        current_mask = inputs.get("attention_mask")
        has_multimodal = inputs.get("image_grid_thw") is not None or inputs.get(
            "video_grid_thw"
        ) is not None
        if has_multimodal:
            mm_types = inputs.get("mm_token_type_ids")
            if mm_types is None:
                raise RuntimeError("Processor omitted mm_token_type_ids for an image block")
            positions, _ = self.core.get_rope_index(
                input_ids,
                mm_token_type_ids=mm_types,
                image_grid_thw=inputs.get("image_grid_thw"),
                video_grid_thw=inputs.get("video_grid_thw"),
                attention_mask=current_mask,
            )
        else:
            positions = torch.arange(input_ids.shape[1], device=self.device)
            positions = positions.view(1, 1, -1).expand(3, input_ids.shape[0], -1)
        positions = positions + int(self.next_mrope_position)
        self.next_mrope_position = int(positions.max().item()) + 1
        return positions

    @torch.inference_mode()
    def _forward(self, inputs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, Any]]:
        # This session is inference-only.  Merely putting the models in eval()
        # does not disable autograd: without this boundary each newly returned
        # KV tensor retains the graph that created the previous KV.  A sliding
        # cache can then keep only four frames at the token level while still
        # retaining every historical forward activation, eventually exhausting
        # a 48-GB GPU after only a handful of polling steps.
        current_tokens = int(inputs["input_ids"].shape[1])
        past_tokens = 0 if self.cache is None else int(self.cache.get_seq_length())
        position_ids = self._local_position_ids(inputs)
        attention_mask = torch.ones(
            (1, past_tokens + current_tokens),
            dtype=inputs["attention_mask"].dtype,
            device=self.device,
        )
        cache_position = torch.arange(
            past_tokens,
            past_tokens + current_tokens,
            dtype=torch.long,
            device=self.device,
        )
        call = {
            key: value
            for key, value in inputs.items()
            if key not in {"attention_mask"}
        }
        started = time.perf_counter()
        if self.backend == "8b":
            outputs = self.wrapper(
                **call,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=self.cache,
                cache_position=cache_position,
                use_cache=True,
                return_dict=True,
                logits_to_keep=1,
            )
            readout = outputs.logits[0, -1].detach().float()
        else:
            outputs = self.core(
                **call,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=self.cache,
                cache_position=cache_position,
                use_cache=True,
                return_dict=True,
            )
            readout = outputs.last_hidden_state[0, -1].detach().float()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - started
        self.cache = outputs.past_key_values
        self.stats.model_forward_calls += 1
        self.stats.total_forward_seconds += elapsed
        cache_tokens = int(self.cache.get_seq_length())
        cache_bytes = _cache_bytes(self.cache)
        self.stats.peak_cache_tokens = max(self.stats.peak_cache_tokens, cache_tokens)
        self.stats.peak_cache_bytes = max(self.stats.peak_cache_bytes, cache_bytes)
        visual_id = int(getattr(self.core.config, "image_token_id", 151655))
        visual_tokens = int((inputs["input_ids"] == visual_id).sum().item())
        return readout, {
            f"{self.backend}_forward_seconds": elapsed,
            f"{self.backend}_current_input_tokens": current_tokens,
            f"{self.backend}_current_visual_tokens": visual_tokens,
            f"{self.backend}_cache_tokens": cache_tokens,
            f"{self.backend}_cache_bytes": cache_bytes,
            f"{self.backend}_cached_frame_blocks": len(self.frame_block_tokens),
        }

    def initialize_prefix(self) -> None:
        if self.initialized:
            return
        inputs = self._processor_batch(self._prefix_text())
        self._forward(inputs)
        self.prefix_tokens = int(self.cache.get_seq_length())
        self.initialized = True

    def _make_room(self, incoming_frames: int) -> tuple[int, int]:
        evicted_frames = 0
        evicted_tokens = 0
        while self.frame_block_tokens and (
            len(self.frame_block_tokens) + int(incoming_frames) > self.frame_window
        ):
            length = int(self.frame_block_tokens.pop(0))
            _remove_cache_token_range(
                self.cache,
                self.prefix_tokens,
                self.prefix_tokens + length,
            )
            evicted_frames += 1
            evicted_tokens += length
        self.stats.evicted_frame_blocks += evicted_frames
        self.stats.evicted_tokens += evicted_tokens
        return evicted_frames, evicted_tokens

    def append_frames(
        self,
        frames: list[Image.Image],
        timestamps: list[float],
        *,
        decide: bool,
        decision_prompt: str | None = None,
        token_reduction_encoder: Any | None = None,
        token_reduction_query: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        if len(frames) != len(timestamps) or not frames:
            raise ValueError("append_frames needs equal non-empty frame/timestamp lists")
        if (
            self.reset_every_n_frames > 0
            and self.total_frames_appended > 0
            and self.total_frames_appended % self.reset_every_n_frames == 0
        ):
            self._hard_reset()
        self.total_frames_appended += len(frames)
        self.initialize_prefix()
        evicted_frames, evicted_tokens = self._make_room(len(frames))
        batches: list[dict[str, torch.Tensor]] = []
        block_lengths: list[int] = []
        continuation = bool(self.assistant_open)
        for index, (frame, timestamp) in enumerate(zip(frames, timestamps)):
            block = self._processor_batch(
                self._frame_text(
                    float(timestamp),
                    decision=bool(decide and index == len(frames) - 1),
                    continuation=bool(continuation and index == 0),
                    decision_prompt=(
                        decision_prompt if index == len(frames) - 1 else None
                    ),
                ),
                image=frame,
            )
            batches.append(block)
            block_lengths.append(int(block["input_ids"].shape[1]))
        combined = _concat_processor_batches(batches)
        token_reduction_metrics: dict[str, Any] = {
            "token_reduction_enabled": False,
        }
        token_reduction_future: Any | None = None
        token_reduction_common_start: torch.cuda.Event | None = None
        if self.backend == "8b" and bool(
            getattr(self.owner, "token_reduction_enabled", False)
        ):
            ctr_arm = (
                str(getattr(self.owner, "token_reduction_selector", "semantic")) == "ctr"
            )
            if not ctr_arm and (
                token_reduction_encoder is None or token_reduction_query is None
            ):
                raise ValueError(
                    "Proactive 8B token reduction requires the active 2B encoder "
                    "and a proposal query vector."
                )
            image_grid_thw = combined.get("image_grid_thw")
            if image_grid_thw is None or int(image_grid_thw.shape[0]) != len(frames):
                raise RuntimeError(
                    "Incremental proactive verifier did not produce one image grid "
                    "per newly appended frame."
                )
            merge_size = int(self.core.visual.spatial_merge_size)
            target_grids = [
                (
                    int(grid[1].item()) // merge_size,
                    int(grid[2].item()) // merge_size,
                )
                for grid in image_grid_thw
            ]
            self.core.trace_config = self.owner._token_reduction_trace_config()
            self.core.trace_last_token_counts = None
            self.core.trace_last_parallel_timing = None
            self.core.trace_last_image_vit_events = None
            for attribute in (
                "trace_semantic_image_token_scores",
                "trace_semantic_image_token_scores_future",
            ):
                if hasattr(self.core, attribute):
                    delattr(self.core, attribute)
            if ctr_arm:
                # No 2B pass at all on this arm - that absence is the point of the
                # comparison, so it must show up in the measured cost.
                pass
            elif bool(getattr(self.owner, "token_reduction_parallel", True)):
                executor = getattr(self.owner, "_token_reduction_executor", None)
                stream = getattr(self.owner, "_token_reduction_stream", None)
                if executor is None or stream is None:
                    raise RuntimeError(
                        "Parallel proactive token-reduction worker was not initialized."
                    )
                token_reduction_common_start = torch.cuda.Event(enable_timing=True)
                token_reduction_common_start.record(
                    torch.cuda.current_stream(self.device)
                )
                token_reduction_future = executor.submit(
                    token_reduction_encoder.score_selected_frame_tokens,
                    frames,
                    token_reduction_query,
                    layer=int(self.owner.token_reduction_layer),
                    target_grid_hw=target_grids,
                    cuda_stream=stream,
                    common_start_event=token_reduction_common_start,
                )
                self.core.trace_semantic_image_token_scores_future = (
                    token_reduction_future
                )
            else:
                scores, scorer_metrics = (
                    token_reduction_encoder.score_selected_frame_tokens(
                        frames,
                        token_reduction_query,
                        layer=int(self.owner.token_reduction_layer),
                        target_grid_hw=target_grids,
                    )
                )
                self.core.trace_semantic_image_token_scores = scores
                token_reduction_metrics.update(scorer_metrics)
            token_reduction_metrics.update(
                {
                    "token_reduction_enabled": True,
                    "token_reduction_layer": int(self.owner.token_reduction_layer),
                    "token_reduction_budget": float(self.owner.token_reduction_budget),
                    "token_reduction_parallel": bool(
                        self.owner.token_reduction_parallel
                    ),
                    "token_reduction_scope": (
                        "sparse_proactive_8b_verifier_new_frames_before_kv_write"
                    ),
                    "token_reduction_selector": str(
                        getattr(self.owner, "token_reduction_selector", "semantic")
                    ),
                }
            )
        else:
            self.core.trace_config = None

        try:
            readout, metrics = self._forward(combined)
        finally:
            if self.backend == "8b":
                self.core.trace_config = None
                for attribute in (
                    "trace_semantic_image_token_scores",
                    "trace_semantic_image_token_scores_future",
                ):
                    if hasattr(self.core, attribute):
                        delattr(self.core, attribute)
        if token_reduction_future is not None:
            _, scorer_metrics = token_reduction_future.result()
            token_reduction_metrics.update(scorer_metrics)
        scorer_start = token_reduction_metrics.pop(
            "_token_reduction_cuda_start_event", None
        )
        scorer_end = token_reduction_metrics.pop(
            "_token_reduction_cuda_end_event", None
        )
        if scorer_start is not None and scorer_end is not None:
            token_reduction_metrics["token_reduction_seconds"] = (
                float(scorer_start.elapsed_time(scorer_end)) / 1000.0
            )
        parallel_timing = getattr(
            self.core, "trace_last_parallel_timing", None
        ) or {}
        token_reduction_metrics.update(parallel_timing)
        join_event = token_reduction_metrics.pop(
            "_token_reduction_parallel_join_event", None
        )
        if token_reduction_common_start is not None and join_event is not None:
            token_reduction_metrics[
                "token_reduction_parallel_visual_critical_path_seconds"
            ] = (
                float(token_reduction_common_start.elapsed_time(join_event)) / 1000.0
            )
        trace_counts = getattr(self.core, "trace_last_token_counts", None)
        if token_reduction_metrics["token_reduction_enabled"]:
            if trace_counts is None:
                raise RuntimeError(
                    "Proactive verifier token reduction produced no BiSeek token counts."
                )
            original_per_item = [
                int(value)
                for value in trace_counts["original_visual_tokens_per_item"]
            ]
            kept_per_item = [
                int(value) for value in trace_counts["kept_visual_tokens_per_item"]
            ]
            if len(kept_per_item) != len(block_lengths):
                raise RuntimeError(
                    "BiSeek per-frame token counts do not match verifier frame blocks."
                )
            block_lengths = [
                int(length - original + kept)
                for length, original, kept in zip(
                    block_lengths, original_per_item, kept_per_item
                )
            ]
            token_reduction_metrics.update(
                {
                    "answer_visual_tokens_before_reduction": int(
                        trace_counts["original_visual_tokens"]
                    ),
                    "answer_visual_tokens": int(
                        trace_counts["kept_visual_tokens"]
                    ),
                    "answer_visual_token_keep_ratio": float(
                        trace_counts["keep_ratio"]
                    ),
                    "token_reduction_persistent_decoder_token_bytes": 0,
                }
            )
            metrics["8b_current_visual_tokens_after_reduction"] = int(
                trace_counts["kept_visual_tokens"]
            )
            self.stats.token_reduction_calls += 1
            self.stats.visual_tokens_before_reduction += int(
                trace_counts["original_visual_tokens"]
            )
            self.stats.visual_tokens_after_reduction += int(
                trace_counts["kept_visual_tokens"]
            )
            self.stats.token_reduction_seconds += float(
                token_reduction_metrics.get("token_reduction_seconds", 0.0)
            )
        self.frame_block_tokens.extend(block_lengths)
        self.assistant_open = bool(decide)
        self.stats.visual_frames_encoded += len(frames)
        metrics.update(
            {
                **token_reduction_metrics,
                f"{self.backend}_new_frames": len(frames),
                f"{self.backend}_evicted_frame_blocks": evicted_frames,
                f"{self.backend}_evicted_tokens": evicted_tokens,
                f"{self.backend}_cached_frame_blocks": len(self.frame_block_tokens),
                f"{self.backend}_cache_policy": "fixed_question_prefix_plus_bounded_frame_kv",
                f"{self.backend}_cache_frame_window": self.frame_window,
            }
        )
        if decide:
            self.stats.decision_calls += 1
        return readout, metrics

    def statistics(self) -> dict[str, Any]:
        prefix = self.backend
        return {
            f"{prefix}_model_forward_calls": self.stats.model_forward_calls,
            f"{prefix}_decision_calls": self.stats.decision_calls,
            f"{prefix}_total_forward_seconds": self.stats.total_forward_seconds,
            f"{prefix}_peak_cache_tokens": self.stats.peak_cache_tokens,
            f"{prefix}_peak_cache_bytes": self.stats.peak_cache_bytes,
            f"{prefix}_evicted_frame_blocks": self.stats.evicted_frame_blocks,
            f"{prefix}_evicted_tokens": self.stats.evicted_tokens,
            f"{prefix}_visual_frames_encoded": self.stats.visual_frames_encoded,
            f"{prefix}_token_reduction_calls": self.stats.token_reduction_calls,
            f"{prefix}_visual_tokens_before_reduction": (
                self.stats.visual_tokens_before_reduction
            ),
            f"{prefix}_visual_tokens_after_reduction": (
                self.stats.visual_tokens_after_reduction
            ),
            f"{prefix}_token_reduction_seconds": self.stats.token_reduction_seconds,
            f"{prefix}_visual_token_keep_ratio": (
                self.stats.visual_tokens_after_reduction
                / self.stats.visual_tokens_before_reduction
                if self.stats.visual_tokens_before_reduction
                else 1.0
            ),
        }


@dataclass
class ProactiveState:
    query_id: str
    target_question: str
    registration_timestamp: float
    registration_metrics: dict[str, Any]
    current_positive: bool = False
    positive_latched: bool = False
    decision_count: int = 0
    gate_call_count: int = 0
    gate_positive_count: int = 0
    verifier_call_count: int = 0
    verifier_positive_count: int = 0
    last_observation_position: int | None = None
    last_positive_timestamp: float | None = None
    latest_gate_scores: list[float] = field(default_factory=list)
    gate_session: IncrementalMultimodalKVSession | None = None
    verifier_session: IncrementalMultimodalKVSession | None = None
    last_verifier_position: int | None = None
    temporal_hidden: torch.Tensor | None = None
    previous_joint_feature: torch.Tensor | None = None
    previous_decision_timestamp: float | None = None
    previous_gate_margin: float | None = None
    update_head_call_count: int = 0
    update_head_seconds: float = 0.0
    cached_answer: str = NO_CACHED_ANSWER
    answer_update_count: int = 0
    proposal_texts: list[str] = field(default_factory=list)
    proposal_vectors: torch.Tensor | None = None
    proposal_score_history: list[tuple[float, torch.Tensor]] = field(default_factory=list)
    proposal_compiler_calls: int = 0
    proposal_compiler_seconds: float = 0.0
    # Accumulated verifier time, alongside proposal_match_seconds on the gate side,
    # so per-frame cost can be computed from the records.
    verifier_seconds: float = 0.0
    proposal_embedding_seconds: float = 0.0
    proposal_match_calls: int = 0
    proposal_window_encode_calls: int = 0
    proposal_window_cache_hits: int = 0
    proposal_frame_encode_calls: int = 0
    proposal_frame_cache_hits: int = 0
    proposal_vit_feature_encode_calls: int = 0
    proposal_vit_feature_cache_hits: int = 0
    proposal_vit_feature_seconds: float = 0.0
    proposal_joint_decoder_seconds: float = 0.0
    proposal_kv_last_position: int | None = None
    proposal_match_seconds: float = 0.0
    proposal_raw_candidate_count: int = 0
    proposal_suppressed_candidate_count: int = 0
    proposal_candidate_count: int = 0
    proposal_candidate_episode_count: int = 0
    proposal_retry_candidate_count: int = 0
    proposal_retry_positive_count: int = 0
    proposal_retry_state_change_count: int = 0
    proposal_episode_expired_count: int = 0
    proposal_wait_expiry_rearm_count: int = 0
    proposal_state_hold_count: int = 0
    last_candidate_vector: torch.Tensor | None = None
    last_candidate_timestamp: float | None = None
    proposal_episode_start_timestamp: float | None = None
    proposal_episode_until_timestamp: float | None = None
    proposal_last_verifier_timestamp: float | None = None
    proposal_manifest_key: str | None = None
    proposal_manifest_replayed: bool = False
    rec_cumulative_count: int = 0


@dataclass
class ProactiveDecision:
    positive: bool
    positions: list[int]
    timestamps: list[float]
    frames: list[Image.Image]
    metrics: dict[str, Any]
    answer: str | None = None


class CachedProactiveController:
    def __init__(
        self,
        *,
        encoder: Any | None,
        answer_model: Any,
        cfg: dict[str, Any],
        task: str,
        decision_mode: str,
    ) -> None:
        if task not in PROACTIVE_TASKS:
            raise ValueError(task)
        if decision_mode not in DECISION_MODES:
            raise ValueError(decision_mode)
        if decision_mode == "2b8b_cache" and encoder is None:
            raise ValueError("2b8b_cache requires an embedding encoder")
        self.encoder = encoder
        self.answer_model = answer_model
        self.cfg = cfg
        self.task = task
        self.decision_mode = decision_mode
        self.gate_only = bool(cfg.get("gate_only", False))
        self.name = controller_name(decision_mode)
        self.profile = TASK_PROFILES[task]
        self.frame_window = int(cfg["recent_window_frames"])
        self.document_embedding_seconds = 0.0
        self.yes_vector: torch.Tensor | None = None
        self.no_vector: torch.Tensor | None = None
        if decision_mode == "2b8b_cache":
            assert encoder is not None
            cache_attribute = "_proactive_evidence_readiness_v2_document_cache"
            cached = getattr(encoder, cache_attribute, None)
            if cached is None:
                vectors, elapsed = encoder.query_vectors(
                    list(GATE_DOCUMENTS),
                    instruction=GATE_DOCUMENT_INSTRUCTION,
                )
                cached = (vectors, float(elapsed))
                setattr(encoder, cache_attribute, cached)
            else:
                vectors, _ = cached
                elapsed = 0.0
            self.yes_vector = F.normalize(
                vectors[0].to(answer_model.device).float().reshape(-1), dim=0
            )
            self.no_vector = F.normalize(
                vectors[1].to(answer_model.device).float().reshape(-1), dim=0
            )
            self.document_embedding_seconds = float(elapsed)

    @property
    def persistent_document_vector_bytes(self) -> int:
        if self.yes_vector is None or self.no_vector is None:
            return 0
        return int(
            self.yes_vector.numel() * self.yes_vector.element_size()
            + self.no_vector.numel() * self.no_vector.element_size()
        )

    def register(
        self,
        *,
        query_id: str,
        target_question: str,
        registration_timestamp: float,
        **_: Any,
    ) -> ProactiveState:
        return ProactiveState(
            query_id=str(query_id),
            target_question=str(target_question),
            registration_timestamp=float(registration_timestamp),
            registration_metrics={
                "controller": self.name,
                "controller_version": CONTROLLER_VERSION,
                "decision_mode": self.decision_mode,
                "task_profile": self.profile.name,
                "cache_protocol": "fixed_question_prefix_plus_bounded_incremental_multimodal_kv",
                "cache_frame_window": self.frame_window,
                "initial_input": f"latest_{self.frame_window}_causal_independent_images",
                "subsequent_input": "only_newly_arrived_independent_image",
                "gate_backend": (
                    "none"
                    if self.decision_mode == "8b_cache"
                    else "frozen_embedding_2b_incremental_joint_vqa_cosine"
                ),
                "gate_prompt_version": (
                    None
                    if self.decision_mode == "8b_cache"
                    else GATE_PROMPT_VERSION
                ),
                "gate_documents": (
                    None
                    if self.decision_mode == "8b_cache"
                    else list(GATE_DOCUMENTS)
                ),
                "verifier_backend": "frozen_qwen3_vl_8b_incremental_next_token_ab",
                "8b_cache_visibility": (
                    "all_polled_frames"
                    if self.decision_mode == "8b_cache"
                    else "candidate_frames_only_sparse_cache"
                ),
                "document_embedding_seconds_shared": self.document_embedding_seconds,
            },
        )

    def _new_positions(self, state: ProactiveState, sampler: Any, timestamp: float) -> list[int]:
        available = sampler.positions_available_at(float(timestamp))
        if state.last_observation_position is None:
            return available[-self.frame_window :]
        return [value for value in available if value > state.last_observation_position]

    def _new_session(self, backend: str, state: ProactiveState) -> IncrementalMultimodalKVSession:
        owner = self.answer_model if backend == "8b" else self.encoder
        if owner is None:
            raise RuntimeError(f"Missing owner for {backend} cache")
        return IncrementalMultimodalKVSession(
            backend=backend,
            owner=owner,
            target_question=state.target_question,
            task=self.task,
            frame_window=self.frame_window,
        )

    def prime(
        self,
        *,
        state: ProactiveState,
        sampler: Any,
        timestamp: float,
    ) -> dict[str, Any]:
        """Cache the registration-time recent window without making a decision."""

        positions = recent_positions(sampler, float(timestamp), self.frame_window)
        if not positions:
            return {"prime_called": False, "prime_reason": "no_causal_frame"}
        frames = sampler.decode_positions(positions)
        timestamps = sampler.timestamps(positions)
        backend = "8b" if self.decision_mode == "8b_cache" else "2b"
        if backend == "8b":
            if state.verifier_session is None:
                state.verifier_session = self._new_session("8b", state)
            session = state.verifier_session
        else:
            if state.gate_session is None:
                state.gate_session = self._new_session("2b", state)
            session = state.gate_session
        _, metrics = session.append_frames(frames, timestamps, decide=False)
        state.last_observation_position = int(positions[-1])
        value = {
            **metrics,
            "prime_called": True,
            "prime_backend": backend,
            "prime_positions": positions,
            "prime_timestamps_seconds": timestamps,
            "prime_frames": len(positions),
            "prime_makes_answer_wait_decision": False,
        }
        state.registration_metrics["prime"] = value
        return value

    def _8b_verdict(
        self,
        session: IncrementalMultimodalKVSession,
        frames: list[Image.Image],
        timestamps: list[float],
        *,
        token_reduction_encoder: Any | None = None,
        token_reduction_query: torch.Tensor | None = None,
    ) -> tuple[bool, str, dict[str, Any]]:
        append_kwargs: dict[str, Any] = {}
        if bool(getattr(self.answer_model, "token_reduction_enabled", False)):
            append_kwargs.update(
                {
                    "token_reduction_encoder": token_reduction_encoder,
                    "token_reduction_query": token_reduction_query,
                }
            )
        logits, metrics = session.append_frames(
            frames,
            timestamps,
            decide=True,
            **append_kwargs,
        )
        tokenizer = self.answer_model.processor.tokenizer
        negative_label, positive_label = verifier_label_pair(self.task)
        token_ids: dict[str, int] = {}
        for label in (negative_label, positive_label):
            values = tokenizer.encode(label, add_special_tokens=False)
            if len(values) != 1:
                raise RuntimeError(f"Expected one token for {label}, got {values}")
            token_ids[label] = int(values[0])
        logit_a = float(logits[token_ids[negative_label]].item())
        logit_b = float(logits[token_ids[positive_label]].item())
        response = positive_label if logit_b > logit_a else negative_label
        metrics.update(
            {
                "binary_choice_backend": "incremental_qwen3_vl_next_token_ab",
                "binary_choice_logit_a": logit_a,
                "binary_choice_logit_b": logit_b,
                "binary_choice_margin_b_minus_a": logit_b - logit_a,
                "answer_total_seconds": float(metrics["8b_forward_seconds"]),
            }
        )
        return parse_binary_verdict(response, self.task), response, metrics

    def _gate_verdict(
        self,
        session: IncrementalMultimodalKVSession,
        frames: list[Image.Image],
        timestamps: list[float],
    ) -> tuple[bool, float, float, dict[str, Any]]:
        if self.yes_vector is None or self.no_vector is None:
            raise RuntimeError("2B gate documents are not initialized")
        hidden, metrics = session.append_frames(frames, timestamps, decide=True)
        query = F.normalize(hidden.reshape(-1), dim=0)
        yes = float(torch.dot(query, self.yes_vector).item())
        no = float(torch.dot(query, self.no_vector).item())
        metrics.update(
            {
                "gate_input_modality": "incremental_independent_image_plus_cached_question_and_recent_frames",
                "gate_score_yes": yes,
                "gate_score_no": no,
                "gate_margin_yes_minus_no": yes - no,
                "gate_2b_seconds": float(metrics["2b_forward_seconds"]),
            }
        )
        return bool(yes > no), yes, no, metrics

    def _transition(self, state: ProactiveState, verdict: bool, timestamp: float) -> None:
        if self.task == "SSR":
            state.current_positive = bool(verdict)
            if verdict:
                state.last_positive_timestamp = float(timestamp)
            return
        state.positive_latched = state.positive_latched or bool(verdict)
        state.current_positive = state.positive_latched
        if verdict and state.last_positive_timestamp is None:
            state.last_positive_timestamp = float(timestamp)

    def decide(
        self,
        *,
        state: ProactiveState,
        sampler: Any,
        timestamp: float,
    ) -> ProactiveDecision:
        if self.profile.positive_state_absorbing and state.positive_latched:
            state.decision_count += 1
            return ProactiveDecision(
                positive=True,
                positions=[],
                timestamps=[],
                frames=[],
                metrics={
                    "decision_positive": True,
                    "decision_reason": "absorbing_positive_state",
                    "gate_2b_called": False,
                    "verifier_8b_called": False,
                    "decision_index": state.decision_count - 1,
                },
            )

        positions = self._new_positions(state, sampler, float(timestamp))
        if not positions:
            return ProactiveDecision(
                positive=state.positive_latched or state.current_positive,
                positions=[],
                timestamps=[],
                frames=[],
                metrics={
                    "decision_positive": bool(state.positive_latched or state.current_positive),
                    "decision_reason": "no_new_causal_frame",
                    "gate_2b_called": False,
                    "verifier_8b_called": False,
                },
            )

        # The initial call is one batched recent-window decision.  Later calls
        # normally contain one position; if a poll skips time, process each new
        # frame causally rather than collapsing it into a pseudo-video.
        batches = [positions] if state.last_observation_position is None else [[value] for value in positions]
        all_frames: list[Image.Image] = []
        all_timestamps: list[float] = []
        step_metrics: list[dict[str, Any]] = []
        final_verdict = state.positive_latched or state.current_positive
        for block_positions in batches:
            block_timestamps = sampler.timestamps(block_positions)
            block_frames = sampler.decode_positions(block_positions)
            all_frames.extend(block_frames)
            all_timestamps.extend(block_timestamps)
            current_position = int(block_positions[-1])
            gate_positive: bool | None = None
            verifier_positive: bool | None = None
            raw_response = ""
            metrics: dict[str, Any] = {
                "selected_sample_positions": block_positions,
                "selected_timestamps_seconds": block_timestamps,
                "new_frame_count": len(block_positions),
                "gate_2b_called": False,
                "verifier_8b_called": False,
            }

            if self.decision_mode == "8b_cache":
                if state.verifier_session is None:
                    state.verifier_session = self._new_session("8b", state)
                verifier_positive, raw_response, verifier_metrics = self._8b_verdict(
                    state.verifier_session, block_frames, block_timestamps
                )
                state.verifier_call_count += 1
                state.verifier_positive_count += int(verifier_positive)
                metrics.update(verifier_metrics)
                metrics.update(
                    {
                        "verifier_8b_called": True,
                        "verifier_positive": verifier_positive,
                        "verifier_raw_response": raw_response,
                    }
                )
                final_verdict = bool(verifier_positive)
            else:
                if state.gate_session is None:
                    state.gate_session = self._new_session("2b", state)
                gate_positive, yes, no, gate_metrics = self._gate_verdict(
                    state.gate_session, block_frames, block_timestamps
                )
                state.gate_call_count += 1
                state.gate_positive_count += int(gate_positive)
                state.latest_gate_scores = [yes, no]
                metrics.update(gate_metrics)
                metrics.update(
                    {
                        "gate_2b_called": True,
                        "gate_positive": gate_positive,
                    }
                )
                # gate_only: stop at the 2B gate and never consult the 8B
                # verifier. This is the ablation that says what the cheap
                # stage is worth on its own -- the cascade's whole claim is
                # that the gate finds candidates and the verifier rejects the
                # false ones, and that claim is only testable with the second
                # stage removed.
                if gate_positive and not self.gate_only:
                    if state.verifier_session is None:
                        state.verifier_session = self._new_session("8b", state)
                        verifier_positions = recent_positions(
                            sampler, block_timestamps[-1], self.frame_window
                        )
                        verifier_frames = sampler.decode_positions(verifier_positions)
                        verifier_timestamps = sampler.timestamps(verifier_positions)
                        metrics["8b_cache_initialization"] = "latest_recent_window_at_first_candidate"
                    else:
                        verifier_positions = block_positions
                        verifier_frames = block_frames
                        verifier_timestamps = block_timestamps
                        metrics["8b_cache_initialization"] = "existing_sparse_candidate_cache"
                    verifier_positive, raw_response, verifier_metrics = self._8b_verdict(
                        state.verifier_session,
                        verifier_frames,
                        verifier_timestamps,
                    )
                    state.verifier_call_count += 1
                    state.verifier_positive_count += int(verifier_positive)
                    metrics.update(verifier_metrics)
                    metrics.update(
                        {
                            "verifier_8b_called": True,
                            "verifier_positive": verifier_positive,
                            "verifier_raw_response": raw_response,
                            "verifier_selected_sample_positions": verifier_positions,
                            "verifier_selected_timestamps_seconds": verifier_timestamps,
                        }
                    )
                final_verdict = bool(gate_positive) if self.gate_only \
                    else bool(gate_positive and verifier_positive)

            self._transition(state, final_verdict, block_timestamps[-1])
            state.last_observation_position = current_position
            state.decision_count += 1
            positive_state = state.positive_latched or state.current_positive
            metrics.update(
                {
                    "decision_positive": bool(positive_state),
                    "decision_reason": (
                        "8b_verified_positive"
                        if verifier_positive
                        else "8b_wait"
                        if metrics.get("verifier_8b_called")
                        else "2b_gate_no"
                    ),
                    "decision_index": state.decision_count - 1,
                }
            )
            step_metrics.append(metrics)
            if self.profile.positive_state_absorbing and positive_state:
                break

        merged = dict(step_metrics[-1])
        merged["processed_incremental_steps"] = len(step_metrics)
        merged["gate_2b_calls_this_decision"] = sum(
            bool(value.get("gate_2b_called")) for value in step_metrics
        )
        merged["gate_positive_count_this_decision"] = sum(
            bool(value.get("gate_2b_called"))
            and value.get("gate_positive") is True
            for value in step_metrics
        )
        merged["verifier_8b_calls_this_decision"] = sum(
            bool(value.get("verifier_8b_called")) for value in step_metrics
        )
        merged["verifier_positive_count_this_decision"] = sum(
            bool(value.get("verifier_8b_called"))
            and value.get("verifier_positive") is True
            for value in step_metrics
        )
        merged["gate_2b_seconds"] = sum(
            float(value.get("gate_2b_seconds", 0.0)) for value in step_metrics
        )
        merged["answer_total_seconds"] = sum(
            float(value.get("answer_total_seconds", 0.0)) for value in step_metrics
        )
        if len(step_metrics) > 1:
            merged["incremental_step_metrics"] = step_metrics
        return ProactiveDecision(
            positive=bool(state.positive_latched or state.current_positive),
            positions=positions,
            timestamps=all_timestamps,
            frames=all_frames,
            metrics=merged,
        )

    def state_statistics(self, state: ProactiveState) -> dict[str, Any]:
        value: dict[str, Any] = {
            "gate_2b_calls": state.gate_call_count,
            "gate_positive_count": state.gate_positive_count,
            "verifier_8b_calls": state.verifier_call_count,
            "verifier_positive_count": state.verifier_positive_count,
        }
        if state.gate_session is not None:
            value.update(state.gate_session.statistics())
        if state.verifier_session is not None:
            value.update(state.verifier_session.statistics())
        return value


class ProposalMatchProactiveController(CachedProactiveController):
    """Em-Garde-inspired proposal matching with a frozen Qwen3 stack.

    The 8B model compiles a small set of observable state-change proposals once
    when the standing target is registered.  The frozen Embedding-2B model then
    embeds every causal visual window and compares it with the cached proposal
    vectors.  A proposal-score surge creates a sparse candidate; only candidates
    invoke the bounded-KV 8B verifier.  No benchmark label or ground-truth time
    is visible to any model.
    """

    def __init__(
        self,
        *,
        encoder: Any,
        answer_model: Any,
        cfg: dict[str, Any],
        task: str,
    ) -> None:
        super().__init__(
            encoder=encoder,
            answer_model=answer_model,
            cfg=cfg,
            task=task,
            decision_mode="proposal_match",
        )
        self.proposal_count = int(cfg.get("proposal_count", 4))
        self.generation_window_frames = int(
            cfg.get("proposal_generation_window_frames", 5)
        )
        self.match_window_frames = int(cfg.get("proposal_match_window_frames", 2))
        # How the expensive verifier is scheduled. `gate` is the method under
        # test; `periodic` and `dense` are compute-configuration controls that
        # observe identically but ignore the matcher scores entirely, so the
        # accuracy/cost comparison isolates the trigger rule.
        self.schedule_mode = str(cfg.get("proposal_schedule_mode", "gate"))
        if self.schedule_mode not in ("gate", "periodic", "dense"):
            raise ValueError(f"unknown proposal_schedule_mode {self.schedule_mode!r}")
        self.schedule_period_seconds = float(
            cfg.get("proposal_schedule_period_seconds", 4.0)
        )
        if self.schedule_mode == "periodic" and self.schedule_period_seconds <= 0:
            raise ValueError("proposal_schedule_period_seconds must be positive")
        self.surge_threshold = float(cfg.get("proposal_surge_threshold", 0.035))
        self.lookback_seconds = float(
            cfg.get("proposal_delta_lookback_seconds", 1.0)
        )
        self.require_above_history_mean = bool(
            cfg.get("proposal_require_above_history_mean", True)
        )
        self.duplicate_similarity_threshold = float(
            cfg.get("proposal_duplicate_similarity_threshold", 0.8)
        )
        self.duplicate_cooldown_seconds = float(
            cfg.get("proposal_duplicate_cooldown_seconds", 5.0)
        )
        self.initial_verify = bool(cfg.get("proposal_initial_verify", False))
        # gate_only is the parent's switch; honour it here too, where decide()
        # is overridden and would otherwise ignore it silently.
        self.audit_only = bool(cfg.get("proposal_audit_only", False)) or self.gate_only
        # Opt-in, default off: existing manifest replays keep reusing their
        # stored vectors and stay bitwise reproducible.
        self.manifest_texts_only = bool(
            cfg.get("proposal_manifest_texts_only", False)
        )
        self.max_proposal_tokens = int(cfg.get("proposal_max_new_tokens", 256))
        manifest_path = str(cfg.get("proposal_manifest_path", "")).strip()
        self.proposal_manifest_path = Path(manifest_path) if manifest_path else None
        if self.proposal_manifest_path is None:
            self.proposal_prompt_style = str(
                cfg.get("proposal_prompt_style", DEFAULT_PROPOSAL_PROMPT_STYLE)
            )
            if self.proposal_prompt_style not in PROPOSAL_PROMPT_STYLES:
                raise ValueError(
                    "proposal_prompt_style must be one of "
                    f"{PROPOSAL_PROMPT_STYLES}, got {self.proposal_prompt_style!r}"
                )
            self.proposal_prompt_version = PROPOSAL_PROMPT_VERSIONS[
                self.proposal_prompt_style
            ]
        else:
            self.proposal_prompt_style = "manifest_replay"
            versions = cfg.get("proposal_manifest_prompt_versions", ["unknown"])
            self.proposal_prompt_version = "+".join(str(value) for value in versions)
        self.window_cache_size = int(cfg.get("proposal_window_cache_size", 64))
        self.match_encoding = str(cfg.get("proposal_match_encoding", "joint_window"))
        # incremental_kv only: force a clean cache every N ingested frames
        # instead of running one growing/evicting session for a question's
        # whole polling lifetime. 0 (default) is the original behavior.
        self.incremental_kv_reset_every_n_frames = int(
            cfg.get("incremental_kv_reset_every_n_frames", 0)
        )
        # Error-case analysis (2026-09-05): premature Proactive Output triggers
        # cluster at a much weaker binary_choice_margin_b_minus_a (mean 4.82,
        # median 3.5) than on-time triggers (mean 14.13, median 14.5) on cache
        # B's main operating point. The raw verdict accepts on any margin > 0,
        # so a razor-thin logit edge counts the same as a confident one. This
        # raises the bar: a candidate only counts as verifier-positive once the
        # B-over-A margin clears this threshold, independent of the 2B gate's
        # own decision. 0.0 (default) reproduces the original behavior exactly.
        self.verifier_accept_margin = float(cfg.get("verifier_accept_margin", 0.0))
        self.frame_cache_size = int(cfg.get("proposal_frame_cache_size", 256))
        self.vit_cache_size = int(cfg.get("proposal_vit_cache_size", 8))
        self.candidate_episode_seconds = float(
            cfg.get("proposal_candidate_episode_seconds", 2.0)
        )
        self.candidate_retry_interval_seconds = float(
            cfg.get("proposal_candidate_retry_interval_seconds", 1.0)
        )
        self.batch_new_frames_per_poll = bool(
            cfg.get("proposal_batch_new_frames_per_poll", False)
        )
        self.context_retrieval_enabled = bool(
            cfg.get("proposal_context_retrieval_enabled", False)
        )
        self.context_retriever = None
        if self.context_retrieval_enabled:
            if self.proposal_manifest_path is not None:
                raise ValueError(
                    "proposal_context_retrieval_enabled cannot be combined with "
                    "proposal_manifest_path because manifest replay skips compilation"
                )
            from biseek.proactive_retrieval_context import (
                QuestionRetrievalProposalContext,
            )

            self.context_retriever = QuestionRetrievalProposalContext(
                encoder=self.encoder,
                cfg=cfg,
            )
        self.proposal_manifest: dict[str, Any] = {}
        if self.proposal_manifest_path is not None:
            if not self.proposal_manifest_path.is_file():
                raise FileNotFoundError(
                    f"Missing proposal manifest: {self.proposal_manifest_path}"
                )
            loaded = json.loads(self.proposal_manifest_path.read_text())
            entries = loaded.get("entries", loaded)
            if not isinstance(entries, dict):
                raise ValueError("proposal manifest entries must be a JSON object")
            self.proposal_manifest = entries
        if self.window_cache_size <= 0:
            raise ValueError("proposal_window_cache_size must be positive")
        if self.match_encoding not in {
            "joint_window",
            "cached_frame_mean",
            "vit_feature_joint_decoder",
            "incremental_kv",
        }:
            raise ValueError(
                "proposal_match_encoding must be joint_window, cached_frame_mean, "
                "vit_feature_joint_decoder, or incremental_kv"
            )
        if self.frame_cache_size <= 0:
            raise ValueError("proposal_frame_cache_size must be positive")
        if self.vit_cache_size <= 0:
            raise ValueError("proposal_vit_cache_size must be positive")
        self._window_vector_cache: OrderedDict[
            tuple[str, tuple[int, ...]], torch.Tensor
        ] = OrderedDict()
        self._frame_vector_cache: OrderedDict[tuple[str, int], torch.Tensor] = (
            OrderedDict()
        )
        self._vit_feature_cache: OrderedDict[tuple[str, int], dict[str, Any]] = (
            OrderedDict()
        )
        for name, value in (
            ("proposal_count", self.proposal_count),
            ("proposal_generation_window_frames", self.generation_window_frames),
            ("proposal_match_window_frames", self.match_window_frames),
            ("proposal_max_new_tokens", self.max_proposal_tokens),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if (
            self.proposal_prompt_style == "target_anchor_plus3_v7"
            and self.proposal_count != 4
        ):
            raise ValueError(
                "target_anchor_plus3_v7 requires proposal_count=4 "
                "(one deterministic target anchor plus three generated cues)"
            )
        if (
            self.proposal_prompt_style == "target_anchor_plusn_v8"
            and self.proposal_count < 2
        ):
            raise ValueError(
                "target_anchor_plusn_v8 requires proposal_count>=2 "
                "(one deterministic target anchor plus proposal_count-1 cues)"
            )
        if self.lookback_seconds <= 0:
            raise ValueError("proposal_delta_lookback_seconds must be positive")
        if self.duplicate_cooldown_seconds < 0:
            raise ValueError("proposal_duplicate_cooldown_seconds cannot be negative")
        if self.candidate_episode_seconds < 0:
            raise ValueError("proposal_candidate_episode_seconds cannot be negative")
        if self.candidate_retry_interval_seconds <= 0:
            raise ValueError(
                "proposal_candidate_retry_interval_seconds must be positive"
            )

    @property
    def persistent_document_vector_bytes(self) -> int:
        # Proposal vectors are query-specific and reported by state_statistics;
        # there is no controller-global Yes/No document cache.
        return 0

    def register(
        self,
        *,
        query_id: str,
        target_question: str,
        registration_timestamp: float,
        proposal_manifest_key: str | None = None,
        **_: Any,
    ) -> ProactiveState:
        state = ProactiveState(
            query_id=str(query_id),
            target_question=str(target_question),
            registration_timestamp=float(registration_timestamp),
            proposal_manifest_key=(
                str(proposal_manifest_key)
                if proposal_manifest_key is not None
                else f"{self.task}:{query_id}"
            ),
            registration_metrics={
                "controller": self.name,
                "controller_version": CONTROLLER_VERSION,
                "decision_mode": self.decision_mode,
                "task_profile": self.profile.name,
                "proposal_compiler_backend": "frozen_qwen3_vl_8b_once_per_query",
                "proposal_matcher_backend": "frozen_qwen3_vl_embedding_2b_cosine",
                "proposal_count": self.proposal_count,
                "proposal_generation_window_frames": self.generation_window_frames,
                "proposal_match_window_frames": self.match_window_frames,
                "proposal_schedule_mode": self.schedule_mode,
                "proposal_schedule_period_seconds": (
                    self.schedule_period_seconds
                    if self.schedule_mode == "periodic"
                    else None
                ),
                "proposal_surge_threshold": self.surge_threshold,
                "proposal_delta_lookback_seconds": self.lookback_seconds,
                "proposal_require_above_history_mean": self.require_above_history_mean,
                "proposal_duplicate_similarity_threshold": self.duplicate_similarity_threshold,
                "proposal_duplicate_cooldown_seconds": self.duplicate_cooldown_seconds,
                "proposal_initial_verify": self.initial_verify,
                "proposal_audit_only": self.audit_only,
                "proposal_manifest_texts_only": self.manifest_texts_only,
                "proposal_window_cache_size": self.window_cache_size,
                "proposal_match_encoding": self.match_encoding,
                "proposal_manifest_path": (
                    None
                    if self.proposal_manifest_path is None
                    else str(self.proposal_manifest_path)
                ),
                "proposal_frame_cache_size": self.frame_cache_size,
                "proposal_vit_cache_size": self.vit_cache_size,
                "proposal_prompt_style": self.proposal_prompt_style,
                "proposal_prompt_version": self.proposal_prompt_version,
                "proposal_candidate_episode_seconds": self.candidate_episode_seconds,
                "proposal_candidate_retry_interval_seconds": (
                    self.candidate_retry_interval_seconds
                ),
                "proposal_batch_new_frames_per_poll": self.batch_new_frames_per_poll,
                "proposal_context_retrieval_enabled": self.context_retrieval_enabled,
                "proposal_retry_policy": (
                    "retry_while_episode_active_after_unchanged_8b_verdict"
                ),
                "proposal_duplicate_rearm_policy": (
                    "clear_duplicate_reference_after_unresolved_episode_expiry"
                ),
                "proposal_state_policy": (
                    "hold_without_candidate_update_only_on_verified_change"
                    if self.task == "SSR"
                    else "verified_integer_delta_plus_cumulative_accumulator"
                    if self.task == "REC"
                    else "absorbing_positive_hold_negative_until_verified"
                ),
                "verifier_backend": (
                    "none_score_only_audit"
                    if self.audit_only
                    else "frozen_qwen3_vl_8b_incremental_next_token_abcd_delta"
                    if self.task == "REC"
                    else "frozen_qwen3_vl_8b_incremental_next_token_ab"
                ),
                "8b_cache_visibility": (
                    "none_score_only_audit"
                    if self.audit_only
                    else "latest_unseen_recent_window_at_sparse_candidate_times"
                ),
                "ground_truth_visible_to_models": False,
                "proposal_match_encoding_note": (
                    "official joint multimodal window embedding"
                    if self.match_encoding == "joint_window"
                    else (
                        "ordered independent-image ViT features are cached; the "
                        "full 2B language decoder is rerun jointly over the window"
                    )
                    if self.match_encoding == "vit_feature_joint_decoder"
                    else (
                        "one new independent image is appended to a bounded "
                        "question-plus-frame Embedding-2B decoder KV cache"
                    )
                    if self.match_encoding == "incremental_kv"
                    else (
                        "ablation: cached per-frame frozen 2B embeddings are "
                        "mean-pooled and normalized; this is not representation-"
                        "equivalent to a joint-window decoder embedding"
                    )
                ),
            },
        )
        state.cached_answer = "0" if self.task == "REC" else NO_CACHED_ANSWER
        return state

    def _proposal_prompt(self, state: ProactiveState) -> str:
        if self.task in {"STREAMINGBENCH_PO", "STREAMARENA_PRO"}:
            objective = (
                "The standing target is a visual condition. Proposals should mark "
                "observable moments at which that exact condition becomes true."
            )
        elif self.task == "SSR":
            objective = (
                "The standing target is a tutorial-step state. Include observable "
                "changes for both starting and stopping that exact step, because "
                "either change can require revising the current answer."
            )
        elif self.task == "CRR":
            objective = (
                "The standing target is a question whose answer may require future "
                "evidence. Proposals should mark observable moments that complete "
                "the evidence needed to answer without guessing."
            )
        elif self.task == "REC":
            objective = (
                "The standing target is a repeated activity. Proposals should mark "
                "the observable completion boundary of one full repetition. Do not "
                "propose preparation, partial progress, or merely starting it."
            )
        else:
            raise ValueError(self.task)
        if self.proposal_prompt_style == "target_anchored_v2":
            return (
                "You are compiling visual proposals for a causal streaming-video "
                "detector.\n"
                f"Standing target: {state.target_question}\n"
                f"{objective}\n"
                "The causal frames are background context only. Do not replace any "
                "number, color, object, relation, action, or completion condition in "
                "the standing target with what happens to be visible now. Preserve "
                "every exact constraint from the target. Each cue must describe "
                "evidence sufficient for the target, not a preparatory action. If "
                "the future answer is unknown, describe the observation that would "
                "reveal the answer without guessing that answer.\n"
                f"Return exactly {self.proposal_count} short, concrete, visually "
                "observable cues. Each cue must describe what could appear in a "
                "small video window; do not answer the target, invent a timestamp, "
                "or mention WAIT/TRIGGER. Return only a JSON array of strings."
            )
        if self.proposal_prompt_style in {
            "target_anchor_plus3_v7",
            "target_anchor_plusn_v8",
        }:
            cue_count = self._generated_cue_count()
            if self.task in {"STREAMINGBENCH_PO", "STREAMARENA_PRO"}:
                retrieval_goal = (
                    "Cover complementary future views, transitions, or visible "
                    "results of the exact target condition."
                )
            elif self.task == "SSR":
                retrieval_goal = (
                    "Cover complementary visible changes that identify the step's "
                    "start, completion or result, and stopping state."
                )
            elif self.task == "CRR":
                retrieval_goal = (
                    "Cover distinct plausible future events or visible outcomes "
                    "that could reveal the unknown answer."
                )
            elif self.task == "REC":
                retrieval_goal = (
                    "Cover complementary visible views of one full repetition "
                    "reaching its completion boundary."
                )
            else:
                raise ValueError(self.task)
            return (
                f"Generate exactly {cue_count} complementary future visual "
                "retrieval queries.\n"
                f"Target: {state.target_question}\n"
                f"Task: {retrieval_goal}\n"
                "A separate deterministic query already preserves the target, so "
                "do not merely repeat it. "
                "Use the supplied frames only to identify "
                "referenced people or objects. Describe future evidence, not an "
                "action or state already visible in those frames. Optimize coverage: "
                f"the {cue_count} queries should retrieve different useful moments, and a "
                "later verifier will reject false matches. Preserve target-specific "
                "people, objects, numbers, colors, relations, order, and completion "
                "conditions whenever they distinguish the event. Do not invent a "
                f"timestamp. Return exactly {cue_count} JSON strings, normally 8-18 "
                "words each."
            )
        raise ValueError(f"Unsupported proposal prompt style: {self.proposal_prompt_style}")

    def _proposal_repair_prompt(self, state: ProactiveState, response: str) -> str:
        output_schema = json.dumps(
            [f"<cue {index}>" for index in range(1, self.proposal_count + 1)]
        )
        if self.proposal_prompt_style in {
            "target_anchor_plus3_v7",
            "target_anchor_plusn_v8",
        }:
            cue_count = self._generated_cue_count()
            return (
                "Repair and complete these future visual retrieval queries.\n"
                f"Target: {state.target_question}\n"
                f"Previous response: {response}\n"
                f"Return exactly {cue_count} distinct JSON strings describing "
                "complementary "
                "future visible evidence. Do not repeat the target verbatim, copy "
                "the current registration state, or invent a timestamp. Preserve "
                "target-specific constraints. Return the JSON array only."
            )
        return (
            "Repair only the format and completeness of a visual proposal list.\n"
            f"Standing target: {state.target_question}\n"
            f"Previous incomplete response: {response}\n"
            f"Return exactly {self.proposal_count} distinct, atomic, visually "
            "observable cues, normally 5-12 words each. Preserve the target's "
            "identity, count, ordinal, attribute, relation, order, and completion "
            "constraints. Do not guess the future answer and do not copy incidental "
            "registration-frame details. Return one line only with this exact JSON "
            f"shape: {output_schema}"
        )

    def _ensure_proposals(
        self,
        *,
        state: ProactiveState,
        sampler: Any,
        timestamp: float,
    ) -> None:
        if state.proposal_vectors is not None:
            return
        manifest_key = state.proposal_manifest_key or f"{self.task}:{state.query_id}"
        if self.proposal_manifest_path is not None:
            if manifest_key not in self.proposal_manifest:
                raise KeyError(
                    f"Proposal manifest {self.proposal_manifest_path} has no entry "
                    f"for {manifest_key!r}"
                )
            entry = self.proposal_manifest[manifest_key]
            if str(entry.get("task")) != self.task:
                raise ValueError(
                    f"Proposal manifest task mismatch for {manifest_key}: "
                    f"{entry.get('task')!r} != {self.task!r}"
                )
            if str(entry.get("target_question")) != state.target_question:
                raise ValueError(
                    f"Proposal manifest target mismatch for {manifest_key}"
                )
            proposals = [str(value) for value in entry["proposal_texts"]]
            if self.manifest_texts_only:
                # Replaying the stored vectors would defeat an encoder ablation:
                # every arm would then match against the vectors of whichever
                # encoder wrote the manifest, and the swap would silently not
                # happen. This mode reuses only the compiled texts and re-encodes
                # them with the encoder actually under test, which is what
                # "same proposals, different encoder" has to mean.
                vectors, _ = self.encoder.query_vectors(
                    proposals,
                    instruction=(
                        "Represent this short observable visual proposal for matching "
                        "against a causal video window."
                    ),
                )
                # Normalize exactly as the compile path does. Without this the
                # branch produced raw encoder output and tripped the very
                # normalization assertion five lines below, so texts_only had
                # never actually executed against this encoder.
                vectors = torch.stack(
                    [F.normalize(v.float().reshape(-1), dim=0) for v in vectors],
                    dim=0,
                ).cpu()
            else:
                vectors = torch.tensor(entry["proposal_vectors"], dtype=torch.float32)
            if vectors.ndim != 2 or vectors.shape[0] != len(proposals):
                raise ValueError(
                    f"Invalid proposal vector matrix for {manifest_key}: "
                    f"{tuple(vectors.shape)} vs {len(proposals)} texts"
                )
            if not torch.isfinite(vectors).all():
                raise ValueError(f"Non-finite proposal vectors for {manifest_key}")
            norms = torch.linalg.vector_norm(vectors, dim=1)
            if not torch.allclose(norms, torch.ones_like(norms), atol=1e-5, rtol=1e-5):
                raise ValueError(
                    f"Proposal vectors are not normalized for {manifest_key} "
                    f"(texts_only={self.manifest_texts_only})"
                )
            state.proposal_texts = proposals
            # JSON float round-trips preserve float32 values; do not normalize
            # again here or a replay would no longer be bitwise-identical.
            state.proposal_vectors = vectors.cpu()
            state.proposal_manifest_replayed = True
            state.registration_metrics["proposal_compilation"] = {
                "proposal_manifest_key": manifest_key,
                "proposal_prompt_version": str(
                    entry.get("proposal_prompt_version", "unknown")
                ),
                "proposal_texts": proposals,
                "proposal_vectors": state.proposal_vectors.tolist(),
                "manifest_replay": True,
                "compiler_context_frames": entry.get("compiler_context_frames"),
                "compiler_sample_fps": entry.get("compiler_sample_fps"),
                "compiler_8b_called": False,
                "proposal_embedding_2b_called": False,
                "persistent_proposal_vector_bytes": int(
                    state.proposal_vectors.numel()
                    * state.proposal_vectors.element_size()
                ),
            }
            return
        positions = recent_positions(
            sampler, float(timestamp), self.generation_window_frames
        )
        if not positions:
            raise RuntimeError(
                f"No causal frame is available to compile proposals for {state.query_id}"
            )
        context_labels = [
            self._proposal_frame_label(value)
            for value in sampler.timestamps(positions)
        ]
        if self.context_retriever is not None:
            context = self.context_retriever.build(
                sampler=sampler,
                target_question=state.target_question,
                registration_timestamp=float(timestamp),
                recent_positions=positions,
            )
            positions = context["positions"]
            context_labels = context["labels"]
            state.registration_metrics["proposal_context_retrieval"] = context[
                "metrics"
            ]
        frames = sampler.decode_positions(positions)
        timestamps = sampler.timestamps(positions)
        response, generation_metrics = self.answer_model.answer(
            video=frames,
            prompt=self._proposal_prompt(state),
            fps=float(self.cfg["sample_fps"]),
            max_frames=len(frames),
            frame_timestamps=timestamps,
            frame_context_labels=context_labels,
            max_new_tokens_override=self.max_proposal_tokens,
            # Proposal compilation creates the textual queries that the 2B
            # scorer will subsequently use.  No token-reduction query exists
            # yet at this stage, so BiSeek applies only to sparse verifier calls.
            token_reduction_enabled_override=False,
        )
        anchor_plus3_v7 = self.proposal_prompt_style in {
            "target_anchor_plus3_v7",
            "target_anchor_plusn_v8",
        }
        fixed_future_three = anchor_plus3_v7
        generated_proposal_count = (
            self._generated_cue_count() if fixed_future_three else self.proposal_count
        )

        def filter_generated(values: list[str]) -> tuple[list[str], list[str]]:
            if not fixed_future_three:
                return values, []
            accepted: list[str] = []
            rejected: list[str] = []
            normalize = lambda text: " ".join(  # noqa: E731
                re.findall(r"[a-z0-9]+", str(text).lower())
            )
            seen = {normalize(state.target_question)}
            for value in values:
                normalized = normalize(value)
                invalid_timestamp = bool(
                    re.search(
                        r"\b\d+(?:\.\d+)?\s*seconds?\b",
                        value,
                        re.IGNORECASE,
                    )
                )
                if not normalized or normalized in seen or invalid_timestamp:
                    rejected.append(value)
                    continue
                accepted.append(value.strip())
                seen.add(normalized)
            return accepted[:generated_proposal_count], rejected

        proposals, rejected_proposals = filter_generated(
            parse_visual_proposals(response, limit=self.proposal_count)
        )
        generation_attempts = [generation_metrics]
        raw_responses = [response]
        repair_called = False
        proposal_count_valid = bool(proposals) and len(proposals) == generated_proposal_count
        if not proposal_count_valid:
            repair_called = True
            repaired_response, repaired_metrics = self.answer_model.answer(
                video=frames,
                prompt=self._proposal_repair_prompt(state, response),
                fps=float(self.cfg["sample_fps"]),
                max_frames=len(frames),
                frame_timestamps=timestamps,
                frame_context_labels=[
                    self._proposal_frame_label(value) for value in timestamps
                ],
                max_new_tokens_override=self.max_proposal_tokens,
                token_reduction_enabled_override=False,
            )
            repaired, repair_rejected = filter_generated(
                parse_visual_proposals(
                    repaired_response,
                    limit=self.proposal_count,
                )
            )
            rejected_proposals.extend(repair_rejected)
            generation_attempts.append(repaired_metrics)
            raw_responses.append(repaired_response)
            if len(repaired) > len(proposals):
                proposals = repaired
                response = repaired_response
                generation_metrics = repaired_metrics
        parse_fallback = False
        generated_padding_count = 0
        generated_actual_count_before_padding = len(proposals)
        if fixed_future_three:
            target_anchor = state.target_question.strip()
            fallback_prefixes = {
                "STREAMINGBENCH_PO": (
                    "Observable frame where this target condition becomes true:",
                    "Visible completion state for this target condition:",
                    "Alternate visual view of this target condition:",
                ),
                # StreamArena Pro is the same "speak when the target condition
                # becomes true" protocol, so it shares PO's padding phrasing.
                # Without this key the v7 prompt raises KeyError the first time
                # the model under-produces cues on this benchmark.
                "STREAMARENA_PRO": (
                    "Observable frame where this target condition becomes true:",
                    "Visible completion state for this target condition:",
                    "Alternate visual view of this target condition:",
                ),
                "SSR": (
                    "Visible start or completion evidence for this tutorial step:",
                    "Observable result of this tutorial step:",
                    "Visible stopping-state evidence for this tutorial step:",
                ),
                "CRR": (
                    "Future frame revealing enough evidence to answer:",
                    "Observable next event that resolves this question:",
                    "Visible result that makes the answer knowable:",
                ),
                "REC": (
                    "Visible completion boundary for one full repetition of:",
                    "Observable result after one complete repetition of:",
                    "Alternate view confirming one completed repetition of:",
                ),
                }[self.task]
            while len(proposals) < generated_proposal_count:
                proposals.append(
                    f"{fallback_prefixes[len(proposals) % len(fallback_prefixes)]}"
                    f" {target_anchor}"
                )
                generated_padding_count += 1
            proposals = proposals[:generated_proposal_count]
            if anchor_plus3_v7:
                proposals = [target_anchor, *proposals]
            parse_fallback = generated_actual_count_before_padding == 0
        elif not proposals:
            proposals = [
                "Visual evidence directly matching the standing target: "
                + state.target_question
            ]
            parse_fallback = True
        vectors, embedding_seconds = self.encoder.query_vectors(
            proposals,
            instruction=(
                "Represent this short observable visual proposal for matching "
                "against a causal video window."
            ),
        )
        state.proposal_texts = proposals
        state.proposal_vectors = torch.stack(
            [F.normalize(value.float().reshape(-1), dim=0) for value in vectors],
            dim=0,
        ).cpu()
        state.proposal_compiler_calls += len(generation_attempts)
        state.proposal_compiler_seconds += sum(
            float(value.get("answer_total_seconds", 0.0))
            for value in generation_attempts
        )
        state.proposal_embedding_seconds += float(embedding_seconds)
        state.registration_metrics["proposal_compilation"] = {
            "proposal_manifest_key": manifest_key,
            "proposal_prompt_version": self.proposal_prompt_version,
            "proposal_texts": proposals,
            "proposal_vectors": state.proposal_vectors.tolist(),
            "raw_response": response,
            "raw_responses": raw_responses,
            "parse_fallback": parse_fallback,
            "format_repair_called": repair_called,
            "requested_proposal_count": self.proposal_count,
            "actual_proposal_count": len(proposals),
            "proposal_count_complete": len(proposals) == self.proposal_count,
            "proposal_count_policy": (
                "deterministic_anchor_plus_exact_generated"
                if anchor_plus3_v7
                else "exact"
            ),
            "proposal_count_valid": bool(proposals) and len(proposals) == self.proposal_count,
            "deterministic_target_anchor": (
                state.target_question.strip()
                if anchor_plus3_v7
                else None
            ),
            "proposal_roles": None,
            "generated_future_only": False,
            "generated_requested_count": generated_proposal_count,
            "generated_actual_count_before_padding": (
                generated_actual_count_before_padding
            ),
            "generated_padding_count": generated_padding_count,
            "generated_rejected_proposals": rejected_proposals,
            "context_positions": positions,
            "context_timestamps_seconds": timestamps,
            "context_frames": len(frames),
            "compiler_context_frames": len(frames),
            "compiler_sample_fps": float(self.cfg["sample_fps"]),
            "8b_generation_metrics": generation_metrics,
            "8b_generation_attempt_metrics": generation_attempts,
            "proposal_embedding_seconds": float(embedding_seconds),
            "persistent_proposal_vector_bytes": int(
                state.proposal_vectors.numel()
                * state.proposal_vectors.element_size()
            ),
        }

    def _generated_cue_count(self) -> int:
        """Model-written cues, excluding the deterministic target anchor."""
        if self.proposal_prompt_style == "target_anchor_plus3_v7":
            return 3
        return max(1, self.proposal_count - 1)

    def _proposal_frame_label(self, timestamp: float) -> str:
        if self.proposal_prompt_style == "target_anchored_v2":
            return f"Causal context at {timestamp:.3f} seconds:"
        if self.proposal_prompt_style in {
            "target_anchor_plus3_v7",
            "target_anchor_plusn_v8",
        }:
            return (
                "Referent-identification context only; do not describe the current "
                "frame as future evidence:"
            )
        return (
            "Referent-grounding context only at "
            f"{timestamp:.3f} seconds; do not copy its incidental action as "
            "future evidence:"
        )

    def _8b_rec_delta(
        self,
        session: IncrementalMultimodalKVSession,
        frames: list[Image.Image],
        timestamps: list[float],
        *,
        token_reduction_query: torch.Tensor | None = None,
    ) -> tuple[int, str, dict[str, Any]]:
        append_kwargs: dict[str, Any] = {}
        if bool(getattr(self.answer_model, "token_reduction_enabled", False)):
            append_kwargs.update(
                {
                    "token_reduction_encoder": self.encoder,
                    "token_reduction_query": token_reduction_query,
                }
            )
        logits, metrics = session.append_frames(
            frames,
            timestamps,
            decide=True,
            **append_kwargs,
        )
        tokenizer = self.answer_model.processor.tokenizer
        token_ids: dict[str, int] = {}
        for label in ("A", "B", "C", "D"):
            values = tokenizer.encode(label, add_special_tokens=False)
            if len(values) != 1:
                raise RuntimeError(f"Expected one token for {label}, got {values}")
            token_ids[label] = int(values[0])
        label_logits = {
            label: float(logits[token_id].item())
            for label, token_id in token_ids.items()
        }
        response = max(label_logits, key=label_logits.get)
        delta = {"A": 0, "B": 1, "C": 2, "D": 3}[response]
        metrics.update(
            {
                "rec_delta_choice_backend": "incremental_qwen3_vl_next_token_abcd",
                "rec_delta_choice_logits": label_logits,
                "rec_delta_choice": response,
                "rec_verified_interval_delta": delta,
                "answer_total_seconds": float(metrics["8b_forward_seconds"]),
            }
        )
        return delta, response, metrics

    def prime(
        self,
        *,
        state: ProactiveState,
        sampler: Any,
        timestamp: float,
    ) -> dict[str, Any]:
        self._ensure_proposals(state=state, sampler=sampler, timestamp=timestamp)
        if self.initial_verify:
            value = {
                "prime_called": True,
                "prime_backend": "8b_proposal_compiler_only",
                "prime_makes_answer_wait_decision": False,
                "prime_note": "first post-registration poll performs initial verification",
            }
            state.registration_metrics["prime"] = value
            return value
        available = sampler.positions_available_at(float(timestamp))
        match_positions = available[-self.match_window_frames :]
        if not match_positions:
            raise RuntimeError("Proposal matcher prime has no causal frame")
        frames = sampler.decode_positions(match_positions)
        timestamps = sampler.timestamps(match_positions)
        _, _, baseline_metrics = self._proposal_match(
            state=state,
            frames=frames,
            timestamp=float(timestamps[-1]),
            cache_key=(str(sampler.path), tuple(match_positions)),
            frame_timestamps=timestamps,
        )
        state.last_observation_position = int(match_positions[-1])
        value = {
            "prime_called": True,
            "prime_backend": "8b_proposal_compiler_plus_2b_score_baseline",
            "prime_makes_answer_wait_decision": False,
            "prime_positions": match_positions,
            "prime_timestamps_seconds": timestamps,
            "prime_match_metrics": baseline_metrics,
        }
        state.registration_metrics["prime"] = value
        return value

    def _new_match_positions(
        self, state: ProactiveState, sampler: Any, timestamp: float
    ) -> list[int]:
        available = sampler.positions_available_at(float(timestamp))
        if not available:
            return []
        if state.last_observation_position is None:
            return [available[-1]]
        return [value for value in available if value > state.last_observation_position]

    def _proposal_match(
        self,
        *,
        state: ProactiveState,
        frames: list[Image.Image],
        timestamp: float,
        cache_key: tuple[str, tuple[int, ...]],
        frame_timestamps: list[float] | None = None,
    ) -> tuple[bool, torch.Tensor, dict[str, Any]]:
        if state.proposal_vectors is None:
            raise RuntimeError("Proposal vectors were not initialized")
        cache_hit = False
        frame_cache_hits = 0
        frame_encode_calls = 0
        vit_metrics: dict[str, Any] = {}
        kv_metrics: dict[str, Any] = {}
        started = time.perf_counter()
        if self.match_encoding == "incremental_kv":
            _video_path, positions = cache_key
            if len(positions) != len(frames):
                raise RuntimeError("Incremental-KV positions do not match decoded frames")
            if frame_timestamps is None or len(frame_timestamps) != len(frames):
                raise RuntimeError("Incremental-KV matching requires frame timestamps")
            if state.gate_session is None:
                state.gate_session = IncrementalMultimodalKVSession(
                    backend="2b",
                    owner=self.encoder,
                    target_question=state.target_question,
                    task=self.task,
                    frame_window=self.match_window_frames,
                    reset_every_n_frames=self.incremental_kv_reset_every_n_frames,
                )
            new_indices = [
                index
                for index, position in enumerate(positions)
                if state.proposal_kv_last_position is None
                or int(position) > int(state.proposal_kv_last_position)
            ]
            if not new_indices:
                raise RuntimeError("Incremental-KV proposal match received no new frame")
            readout, kv_metrics = state.gate_session.append_frames(
                [frames[index] for index in new_indices],
                [float(frame_timestamps[index]) for index in new_indices],
                decide=True,
                decision_prompt=(
                    "Represent the current causal visual state for matching "
                    "the compiled observable proposals."
                ),
            )
            state.proposal_kv_last_position = int(positions[-1])
            window_vector = F.normalize(readout.detach().float().cpu().reshape(-1), dim=0)
        elif self.match_encoding == "vit_feature_joint_decoder":
            video_path, positions = cache_key
            window_vector, vit_metrics = self.encoder.encode_joint_images_cached_vit(
                frames,
                [(video_path, int(position)) for position in positions],
                self._vit_feature_cache,
                self.vit_cache_size,
            )
            state.proposal_vit_feature_encode_calls += int(
                vit_metrics.get("proposal_vit_feature_encode_calls_this_window", 0)
            )
            state.proposal_vit_feature_cache_hits += int(
                vit_metrics.get("proposal_vit_feature_cache_hits_this_window", 0)
            )
            state.proposal_vit_feature_seconds += float(
                vit_metrics.get("proposal_vit_feature_seconds_this_window", 0.0)
            )
            state.proposal_joint_decoder_seconds += float(
                vit_metrics.get("proposal_joint_decoder_seconds_this_window", 0.0)
            )
        elif self.match_encoding == "cached_frame_mean":
            video_path, positions = cache_key
            if len(positions) != len(frames):
                raise RuntimeError("Frame-cache positions do not match decoded frames")
            vectors: list[torch.Tensor] = []
            for position, frame in zip(positions, frames):
                frame_key = (video_path, int(position))
                if frame_key in self._frame_vector_cache:
                    value = self._frame_vector_cache.pop(frame_key)
                    self._frame_vector_cache[frame_key] = value
                    frame_cache_hits += 1
                else:
                    value = F.normalize(
                        self.encoder.encode_frame(frame)
                        .detach()
                        .float()
                        .cpu()
                        .reshape(-1),
                        dim=0,
                    )
                    self._frame_vector_cache[frame_key] = value.detach().clone()
                    while len(self._frame_vector_cache) > self.frame_cache_size:
                        self._frame_vector_cache.popitem(last=False)
                    frame_encode_calls += 1
                vectors.append(value)
            window_vector = F.normalize(torch.stack(vectors).mean(dim=0), dim=0)
            state.proposal_frame_cache_hits += frame_cache_hits
            state.proposal_frame_encode_calls += frame_encode_calls
        else:
            cache_hit = cache_key in self._window_vector_cache
            if cache_hit:
                window_vector = self._window_vector_cache.pop(cache_key)
                self._window_vector_cache[cache_key] = window_vector
                state.proposal_window_cache_hits += 1
            else:
                window_vector = F.normalize(
                    self.encoder.encode_chunk(frames)
                    .detach()
                    .float()
                    .cpu()
                    .reshape(-1),
                    dim=0,
                )
                self._window_vector_cache[cache_key] = window_vector.detach().clone()
                while len(self._window_vector_cache) > self.window_cache_size:
                    self._window_vector_cache.popitem(last=False)
                state.proposal_window_encode_calls += 1
        raw_scores = torch.mv(state.proposal_vectors, window_vector)
        scores = raw_scores
        if self.encoder.device.type == "cuda":
            torch.cuda.synchronize(self.encoder.device)
        elapsed = time.perf_counter() - started
        surge = proposal_surge_candidate(
            current_scores=scores,
            history=state.proposal_score_history,
            timestamp=float(timestamp),
            lookback_seconds=self.lookback_seconds,
            surge_threshold=self.surge_threshold,
            require_above_history_mean=self.require_above_history_mean,
        )
        initial_candidate = bool(
            not state.proposal_score_history and self.initial_verify
        )
        if self.schedule_mode == "gate":
            raw_candidate = bool(surge["candidate"] or initial_candidate)
        elif self.schedule_mode == "dense":
            raw_candidate = True
        else:  # periodic
            since_last = (
                float("inf")
                if state.last_candidate_timestamp is None
                else float(timestamp) - float(state.last_candidate_timestamp)
            )
            raw_candidate = bool(
                initial_candidate or since_last >= self.schedule_period_seconds
            )
        duplicate_similarity = None
        duplicate_suppressed = False
        # Duplicate suppression compares window vectors, so it is a content
        # signal. Leaving it on would make the blind schedules secretly
        # content-aware and the comparison would no longer be blind.
        if self.schedule_mode == "gate" and raw_candidate and state.last_candidate_vector is not None:
            duplicate_similarity = float(
                torch.dot(window_vector, state.last_candidate_vector).item()
            )
            since = float(timestamp) - float(state.last_candidate_timestamp)
            duplicate_suppressed = bool(
                duplicate_similarity >= self.duplicate_similarity_threshold
                and since < self.duplicate_cooldown_seconds
            )
        candidate = bool(raw_candidate and not duplicate_suppressed)
        if raw_candidate:
            state.proposal_raw_candidate_count += 1
        # The duplicate reference is the most recent candidate that was
        # actually admitted.  Updating it for a suppressed raw surge would
        # continuously extend the cooldown and could suppress an entire
        # sustained event forever.
        if candidate:
            state.last_candidate_vector = window_vector.detach().clone()
            state.last_candidate_timestamp = float(timestamp)
            state.proposal_candidate_count += 1
        if duplicate_suppressed:
            state.proposal_suppressed_candidate_count += 1
        state.proposal_score_history.append((float(timestamp), scores.detach().clone()))
        state.proposal_match_calls += 1
        state.proposal_match_seconds += float(elapsed)
        winner = int(surge["winner_index"])
        metrics = {
            **vit_metrics,
            **kv_metrics,
            "gate_input_modality": "causal_sliding_video_window",
            "gate_2b_called": True,
            "gate_2b_seconds": float(elapsed),
            "gate_positive": candidate,
            "proposal_new_candidate": candidate,
            "proposal_scores": [float(value) for value in scores.tolist()],
            "proposal_raw_scores": None,
            "proposal_scoring_policy": "independent_max_delta",
            "proposal_hard_negative_weight": None,
            "proposal_hard_negative_max_score": None,
            "proposal_contrastive_margin": None,
            "proposal_winner_index": winner,
            "proposal_winner_text": state.proposal_texts[winner],
            "proposal_winner_score": float(surge["winner_score"]),
            "proposal_winner_delta": surge["winner_delta"],
            "proposal_reference_timestamp_seconds": surge[
                "reference_timestamp_seconds"
            ],
            "proposal_above_history_mean": surge["above_history_mean"],
            "proposal_history_mean_at_winner": surge.get(
                "history_mean_at_winner"
            ),
            "proposal_raw_candidate": raw_candidate,
            "proposal_initial_candidate": initial_candidate,
            "proposal_duplicate_similarity": duplicate_similarity,
            "proposal_duplicate_suppressed": duplicate_suppressed,
            "proposal_candidate_reason": (
                "initial_verify"
                if initial_candidate
                else "duplicate_suppressed"
                if duplicate_suppressed
                else surge["reason"]
            ),
            "proposal_match_window_frames": len(frames),
            "proposal_window_cache_hit": cache_hit,
            "proposal_window_encoder_2b_called": (
                not cache_hit
                if self.match_encoding == "joint_window"
                else frame_encode_calls > 0
            ),
            "proposal_match_encoding": self.match_encoding,
            "proposal_frame_cache_hits_this_window": frame_cache_hits,
            "proposal_frame_encode_2b_calls_this_window": frame_encode_calls,
        }
        return candidate, window_vector, metrics

    @staticmethod
    def _close_candidate_episode(state: ProactiveState) -> None:
        state.proposal_episode_start_timestamp = None
        state.proposal_episode_until_timestamp = None
        state.proposal_last_verifier_timestamp = None

    @staticmethod
    def _rearm_after_wait_expiry(state: ProactiveState) -> bool:
        """Remove a stale duplicate lock after an unresolved episode expires.

        Duplicate suppression is useful while a candidate episode is active:
        the bounded retry loop already owns that local event.  Once every
        verifier call in the episode has returned WAIT / unchanged, retaining
        the longer duplicate cooldown creates a dead zone in which genuinely
        newer evidence cannot open another episode.  Only unresolved expiry
        calls this helper; verified positives and SSR state changes retain the
        normal cooldown.
        """

        rearmed = bool(
            state.last_candidate_vector is not None
            or state.last_candidate_timestamp is not None
        )
        state.last_candidate_vector = None
        state.last_candidate_timestamp = None
        state.proposal_wait_expiry_rearm_count += int(rearmed)
        return rearmed

    def _candidate_episode_active(
        self, state: ProactiveState, timestamp: float
    ) -> bool:
        until = state.proposal_episode_until_timestamp
        return until is not None and float(timestamp) <= float(until) + 1e-6

    def decide(
        self,
        *,
        state: ProactiveState,
        sampler: Any,
        timestamp: float,
    ) -> ProactiveDecision:
        if self.profile.positive_state_absorbing and state.positive_latched:
            state.decision_count += 1
            return ProactiveDecision(
                positive=True,
                positions=[],
                timestamps=[],
                frames=[],
                metrics={
                    "decision_positive": True,
                    "decision_reason": "absorbing_positive_state",
                    "gate_2b_called": False,
                    "verifier_8b_called": False,
                    "decision_index": state.decision_count - 1,
                },
            )
        self._ensure_proposals(state=state, sampler=sampler, timestamp=timestamp)
        positions = self._new_match_positions(state, sampler, float(timestamp))
        if not positions:
            return ProactiveDecision(
                positive=bool(state.positive_latched or state.current_positive),
                positions=[],
                timestamps=[],
                frames=[],
                metrics={
                    "decision_positive": bool(
                        state.positive_latched or state.current_positive
                    ),
                    "decision_reason": "no_new_causal_frame",
                    "gate_2b_called": False,
                    "verifier_8b_called": False,
                },
            )

        step_metrics: list[dict[str, Any]] = []
        all_frames: list[Image.Image] = []
        all_timestamps: list[float] = []
        # StreamingBench exposes one official decision boundary per second,
        # while a 2-fps observation stream contributes two new images.  The
        # optional batched mode retains both images in the cached 4-image
        # window but computes one joint gate/readout at the poll endpoint.
        # Per-frame mode remains available as the historical control.
        decision_positions = (
            [positions[-1]]
            if self.batch_new_frames_per_poll and len(positions) > 1
            else positions
        )
        for current_position in decision_positions:
            realized_timestamp = float(sampler.sample_timestamps[current_position])
            expired_before_step = bool(
                state.proposal_episode_until_timestamp is not None
                and realized_timestamp
                > float(state.proposal_episode_until_timestamp) + 1e-6
            )
            duplicate_rearmed_before_step = False
            if expired_before_step:
                state.proposal_episode_expired_count += 1
                self._close_candidate_episode(state)
                duplicate_rearmed_before_step = self._rearm_after_wait_expiry(
                    state
                )
            available = sampler.positions_available_at(realized_timestamp)
            match_positions = available[-self.match_window_frames :]
            match_frames = sampler.decode_positions(match_positions)
            match_timestamps = sampler.timestamps(match_positions)
            all_frames.extend(match_frames)
            all_timestamps.extend(match_timestamps)
            new_candidate, _, gate_metrics = self._proposal_match(
                state=state,
                frames=match_frames,
                timestamp=realized_timestamp,
                cache_key=(str(sampler.path), tuple(match_positions)),
                frame_timestamps=match_timestamps,
            )
            episode_active_before = self._candidate_episode_active(
                state, realized_timestamp
            )
            episode_started = False
            if (
                new_candidate
                and not episode_active_before
                and self.candidate_episode_seconds > 0
                and not self.audit_only
            ):
                state.proposal_episode_start_timestamp = realized_timestamp
                state.proposal_episode_until_timestamp = (
                    realized_timestamp + self.candidate_episode_seconds
                )
                state.proposal_last_verifier_timestamp = None
                state.proposal_candidate_episode_count += 1
                episode_started = True
            episode_active = self._candidate_episode_active(
                state, realized_timestamp
            )
            retry_due = bool(
                not self.audit_only
                and episode_active
                and not new_candidate
                and state.proposal_last_verifier_timestamp is not None
                and realized_timestamp
                - float(state.proposal_last_verifier_timestamp)
                >= self.candidate_retry_interval_seconds - 1e-6
            )
            verifier_candidate = bool(new_candidate or retry_due)
            if retry_due:
                state.proposal_retry_candidate_count += 1
            state.gate_call_count += 1
            state.gate_positive_count += int(verifier_candidate)

            verifier_called = False
            verifier_positive: bool | None = None
            rec_delta: int | None = None
            raw_response: str | None = None
            verifier_metrics: dict[str, Any] = {}
            verifier_positions: list[int] = []
            verifier_timestamps: list[float] = []
            verifier_metrics: dict[str, Any] = {}
            if verifier_candidate and not self.audit_only:
                verifier_token_query: torch.Tensor | None = None
                if bool(
                    getattr(self.answer_model, "token_reduction_enabled", False)
                ):
                    winner_index = int(gate_metrics["proposal_winner_index"])
                    if state.proposal_vectors is None:
                        raise RuntimeError("Proposal verifier has no proposal vectors")
                    verifier_token_query = state.proposal_vectors[winner_index]
                if state.verifier_session is None:
                    state.verifier_session = self._new_session("8b", state)
                    verifier_positions = recent_positions(
                        sampler, realized_timestamp, self.frame_window
                    )
                    cache_initialization = "latest_recent_window_at_first_candidate"
                else:
                    verifier_positions = [
                        value
                        for value in available
                        if state.last_verifier_position is None
                        or value > state.last_verifier_position
                    ][-self.frame_window :]
                    cache_initialization = "catch_up_latest_unseen_frames"
                if not verifier_positions:
                    raise RuntimeError("Proposal candidate has no unseen verifier frame")
                verifier_frames = sampler.decode_positions(verifier_positions)
                verifier_timestamps = sampler.timestamps(verifier_positions)
                if self.task == "REC":
                    rec_kwargs: dict[str, Any] = {}
                    if bool(
                        getattr(self.answer_model, "token_reduction_enabled", False)
                    ):
                        rec_kwargs["token_reduction_query"] = verifier_token_query
                    rec_delta, raw_response, verifier_metrics = self._8b_rec_delta(
                        state.verifier_session,
                        verifier_frames,
                        verifier_timestamps,
                        **rec_kwargs,
                    )
                    verifier_positive = rec_delta > 0
                else:
                    verifier_positive, raw_response, verifier_metrics = self._8b_verdict(
                        state.verifier_session,
                        verifier_frames,
                        verifier_timestamps,
                        token_reduction_encoder=self.encoder,
                        token_reduction_query=verifier_token_query,
                    )
                    # A negative threshold has to work too.  The guard used to
                    # be `> 0.0`, which silently ignored negative values, so the
                    # knob could only ever *reject* more -- fine for a backbone
                    # that accepts too readily, useless for one that is too
                    # conservative.  `!= 0.0` keeps 0.0 as exactly the original
                    # behavior (verdict = sign(margin)) and lets the threshold
                    # move in either direction.
                    if self.verifier_accept_margin != 0.0:
                        margin = float(
                            verifier_metrics.get("binary_choice_margin_b_minus_a", 0.0)
                        )
                        verifier_positive = margin > self.verifier_accept_margin
                state.last_verifier_position = int(verifier_positions[-1])
                state.verifier_call_count += 1
                state.verifier_seconds += float(
                    verifier_metrics.get("answer_total_seconds", 0.0) or 0.0
                )
                state.verifier_positive_count += int(verifier_positive)
                state.proposal_last_verifier_timestamp = realized_timestamp
                verifier_called = True

            state_before: int | bool = (
                state.rec_cumulative_count
                if self.task == "REC"
                else bool(state.current_positive or state.positive_latched)
            )
            state_changed = False
            episode_closed_reason: str | None = None
            if self.audit_only and new_candidate:
                if self.task != "REC":
                    self._transition(state, True, realized_timestamp)
                    state_changed = not bool(state_before)
                episode_closed_reason = "score_only_audit_candidate"
            elif verifier_called:
                if self.task == "REC":
                    if rec_delta is None:
                        raise RuntimeError("REC verifier omitted interval delta")
                    if rec_delta > 0:
                        state.rec_cumulative_count += int(rec_delta)
                        state.cached_answer = str(state.rec_cumulative_count)
                        state.last_positive_timestamp = realized_timestamp
                        state_changed = True
                        episode_closed_reason = "verified_rec_integer_delta"
                        self._close_candidate_episode(state)
                        # REC must admit the next completed repetition even if
                        # it is visually similar and occurs inside the generic
                        # duplicate cooldown. A verified count update defines
                        # the end of the current event identity.
                        state.last_candidate_vector = None
                        state.last_candidate_timestamp = None
                    else:
                        state.proposal_state_hold_count += 1
                elif self.task == "SSR":
                    state_changed = bool(verifier_positive) != bool(
                        state.current_positive
                    )
                    if state_changed:
                        self._transition(
                            state, bool(verifier_positive), realized_timestamp
                        )
                        episode_closed_reason = "verified_ssr_state_change"
                        self._close_candidate_episode(state)
                    else:
                        # A same-state verdict at the surge boundary may simply
                        # be early.  Keep the public state and retry briefly.
                        state.proposal_state_hold_count += 1
                elif verifier_positive:
                    self._transition(state, True, realized_timestamp)
                    state_changed = not state_before
                    episode_closed_reason = "verified_absorbing_positive"
                    self._close_candidate_episode(state)
                else:
                    # WAIT does not negate accumulated causal state.  The
                    # episode remains open for a bounded number of retries.
                    state.proposal_state_hold_count += 1
            else:
                # No candidate means no new state evidence.  In particular,
                # SSR must hold its previous Yes/No state instead of reverting
                # to No on every ordinary frame.
                state.proposal_state_hold_count += 1

            if retry_due and verifier_called:
                state.proposal_retry_positive_count += int(
                    verifier_positive is True
                )
                state.proposal_retry_state_change_count += int(state_changed)

            episode_active_after_verdict = self._candidate_episode_active(
                state, realized_timestamp
            )
            duplicate_rearmed_after_verdict = False
            if (
                episode_active_after_verdict
                and state.proposal_episode_until_timestamp is not None
                and realized_timestamp
                >= float(state.proposal_episode_until_timestamp) - 1e-6
            ):
                state.proposal_episode_expired_count += 1
                episode_closed_reason = episode_closed_reason or "retry_window_expired"
                self._close_candidate_episode(state)
                duplicate_rearmed_after_verdict = self._rearm_after_wait_expiry(
                    state
                )
            state.last_observation_position = int(current_position)
            state.decision_count += 1
            positive = (
                bool(rec_delta and rec_delta > 0)
                if self.task == "REC"
                else bool(state.positive_latched or state.current_positive)
            )
            public_state_after: int | bool = (
                state.rec_cumulative_count
                if self.task == "REC"
                else bool(state.current_positive or state.positive_latched)
            )
            metrics = {
                **gate_metrics,
                **verifier_metrics,
                "decision_positive": positive,
                "decision_reason": (
                    "proposal_audit_candidate"
                    if self.audit_only and new_candidate
                    else "8b_verified_state_change"
                    if state_changed
                    else "8b_verified_positive"
                    if verifier_positive
                    else "8b_retry_wait"
                    if verifier_called and retry_due
                    else "8b_initial_wait"
                    if verifier_called
                    else "proposal_candidate_episode_waiting"
                    if episode_active_after_verdict
                    else "proposal_match_no_candidate"
                ),
                "decision_index": state.decision_count - 1,
                "selected_sample_positions": match_positions,
                "selected_timestamps_seconds": match_timestamps,
                "gate_2b_called": True,
                "verifier_8b_called": verifier_called,
                "verifier_positive": verifier_positive,
                "rec_verified_interval_delta": rec_delta,
                "rec_cumulative_count": state.rec_cumulative_count,
                "verifier_raw_response": raw_response,
                # Logged so one run yields the whole accept-threshold curve
                # offline: lowering the threshold can only make an *earlier*
                # call fire, and every call before that one is unchanged, so
                # replaying the recorded margins simulates a lower threshold
                # exactly.  Purely diagnostic -- nothing reads it at runtime.
                "verifier_margin_b_minus_a": (
                    float(verifier_metrics.get("binary_choice_margin_b_minus_a"))
                    if verifier_called
                    and verifier_metrics.get("binary_choice_margin_b_minus_a") is not None
                    else None
                ),
                "verifier_selected_sample_positions": verifier_positions,
                "verifier_selected_timestamps_seconds": verifier_timestamps,
                "8b_cache_initialization": (
                    cache_initialization if verifier_called else None
                ),
                "gate_positive": verifier_candidate,
                "proposal_new_candidate": new_candidate,
                "proposal_candidate_episode_started": episode_started,
                "proposal_candidate_episode_active_before": episode_active_before,
                "proposal_candidate_episode_retry": retry_due,
                "proposal_candidate_episode_until_seconds": (
                    state.proposal_episode_until_timestamp
                ),
                "proposal_public_state_before": state_before,
                "proposal_public_state_after": public_state_after,
                "proposal_public_state_changed": state_changed,
                "proposal_state_held": not state_changed,
                "proposal_candidate_episode_closed_reason": episode_closed_reason,
                "proposal_episode_expired_before_step": expired_before_step,
                "proposal_duplicate_rearmed_after_wait_expiry": bool(
                    duplicate_rearmed_before_step
                    or duplicate_rearmed_after_verdict
                ),
            }
            step_metrics.append(metrics)
            if self.profile.positive_state_absorbing and positive:
                break

        merged = dict(step_metrics[-1])
        merged["processed_incremental_steps"] = len(step_metrics)
        merged["new_observation_frames_this_poll"] = len(positions)
        merged["proposal_batch_new_frames_per_poll"] = self.batch_new_frames_per_poll
        merged["gate_2b_calls_this_decision"] = len(step_metrics)
        merged["gate_positive_count_this_decision"] = sum(
            value.get("gate_positive") is True for value in step_metrics
        )
        merged["proposal_raw_candidate_count_this_decision"] = sum(
            value.get("proposal_raw_candidate") is True for value in step_metrics
        )
        merged["proposal_suppressed_candidate_count_this_decision"] = sum(
            value.get("proposal_duplicate_suppressed") is True
            for value in step_metrics
        )
        merged["proposal_candidate_count_this_decision"] = sum(
            value.get("proposal_new_candidate") is True for value in step_metrics
        )
        merged["proposal_retry_candidate_count_this_decision"] = sum(
            value.get("proposal_candidate_episode_retry") is True
            for value in step_metrics
        )
        merged["proposal_candidate_episode_count_this_decision"] = sum(
            value.get("proposal_candidate_episode_started") is True
            for value in step_metrics
        )
        merged["proposal_wait_expiry_rearm_count_this_decision"] = sum(
            value.get("proposal_duplicate_rearmed_after_wait_expiry") is True
            for value in step_metrics
        )
        merged["verifier_8b_calls_this_decision"] = sum(
            bool(value.get("verifier_8b_called")) for value in step_metrics
        )
        merged["verifier_positive_count_this_decision"] = sum(
            bool(value.get("verifier_8b_called"))
            and value.get("verifier_positive") is True
            for value in step_metrics
        )
        merged["gate_2b_seconds"] = sum(
            float(value.get("gate_2b_seconds", 0.0)) for value in step_metrics
        )
        merged["answer_total_seconds"] = sum(
            float(value.get("answer_total_seconds", 0.0)) for value in step_metrics
        )
        if len(step_metrics) > 1:
            merged["incremental_step_metrics"] = step_metrics
        return ProactiveDecision(
            positive=(
                bool(merged.get("rec_verified_interval_delta", 0))
                if self.task == "REC"
                else bool(state.positive_latched or state.current_positive)
            ),
            positions=positions,
            timestamps=all_timestamps,
            frames=all_frames,
            metrics=merged,
        )

    def state_statistics(self, state: ProactiveState) -> dict[str, Any]:
        retrieval = state.registration_metrics.get(
            "proposal_context_retrieval", {}
        )
        retrieval_2b_calls = int(retrieval.get("retrieval_2b_calls", 0))
        proposal_bytes = (
            0
            if state.proposal_vectors is None
            else int(
                state.proposal_vectors.numel()
                * state.proposal_vectors.element_size()
            )
        )
        value: dict[str, Any] = {
            "gate_2b_calls": state.gate_call_count,
            "gate_positive_count": state.gate_positive_count,
            "verifier_8b_calls": state.verifier_call_count,
            "verifier_positive_count": state.verifier_positive_count,
            "proposal_compiler_8b_calls": state.proposal_compiler_calls,
            "proposal_compiler_8b_seconds": state.proposal_compiler_seconds,
            # Lets the records report per-frame cost and seconds per correct answer.
            "gate_2b_seconds": state.proposal_match_seconds,
            "verifier_8b_seconds": state.verifier_seconds,
            "proposal_vit_feature_seconds": state.proposal_vit_feature_seconds,
            "proposal_joint_decoder_seconds": state.proposal_joint_decoder_seconds,
            "proposal_context_retrieval_2b_calls": retrieval_2b_calls,
            "proposal_context_retrieval_seconds": float(
                retrieval.get("retrieval_total_seconds", 0.0)
            ),
            "proposal_context_retrieval_cache_bytes": int(
                retrieval.get("persistent_chunk_vector_cache_bytes", 0)
            ),
            "proposal_context_retrieval_peak_cache_bytes": int(
                retrieval.get("peak_chunk_vector_cache_bytes", 0)
            ),
            "proposal_embedding_2b_calls": int(
                state.proposal_vectors is not None
                and not state.proposal_manifest_replayed
            ),
            "proposal_manifest_replayed": state.proposal_manifest_replayed,
            "proposal_embedding_seconds": state.proposal_embedding_seconds,
            "proposal_match_2b_calls": state.proposal_match_calls,
            "proposal_window_encode_2b_calls": state.proposal_window_encode_calls,
            "proposal_window_cache_hits": state.proposal_window_cache_hits,
            "proposal_frame_encode_2b_calls": state.proposal_frame_encode_calls,
            "proposal_frame_cache_hits": state.proposal_frame_cache_hits,
            "proposal_vit_feature_encode_2b_calls": (
                state.proposal_vit_feature_encode_calls
            ),
            "proposal_vit_feature_cache_hits": state.proposal_vit_feature_cache_hits,
            "proposal_vit_feature_seconds": state.proposal_vit_feature_seconds,
            "proposal_joint_decoder_2b_calls": (
                state.proposal_match_calls
                if self.match_encoding == "vit_feature_joint_decoder"
                else 0
            ),
            "proposal_joint_decoder_seconds": state.proposal_joint_decoder_seconds,
            "proposal_match_encoding": self.match_encoding,
            "proposal_match_2b_seconds": state.proposal_match_seconds,
            "proposal_raw_candidate_count": state.proposal_raw_candidate_count,
            "proposal_suppressed_candidate_count": (
                state.proposal_suppressed_candidate_count
            ),
            "proposal_candidate_count": state.proposal_candidate_count,
            "proposal_candidate_episode_count": (
                state.proposal_candidate_episode_count
            ),
            "proposal_retry_candidate_count": (
                state.proposal_retry_candidate_count
            ),
            "proposal_retry_positive_count": (
                state.proposal_retry_positive_count
            ),
            "proposal_retry_state_change_count": (
                state.proposal_retry_state_change_count
            ),
            "proposal_episode_expired_count": (
                state.proposal_episode_expired_count
            ),
            "proposal_wait_expiry_rearm_count": (
                state.proposal_wait_expiry_rearm_count
            ),
            "proposal_state_hold_count": state.proposal_state_hold_count,
            "persistent_proposal_vector_bytes": proposal_bytes,
            "2b_model_forward_calls": 0,
            "persistent_shared_window_vector_cache_bytes": int(
                sum(
                    value.numel() * value.element_size()
                    for value in self._window_vector_cache.values()
                )
            ),
            "persistent_shared_frame_vector_cache_bytes": int(
                sum(
                    item.numel() * item.element_size()
                    for item in self._frame_vector_cache.values()
                )
            ),
            "persistent_shared_vit_feature_cache_bytes": int(
                sum(
                    item["pooled"].numel() * item["pooled"].element_size()
                    + sum(
                        value.numel() * value.element_size()
                        for value in item.get("deepstack", ())   # Qwen2-VL embedders have none
                    )
                    for item in self._vit_feature_cache.values()
                )
            ),
            "8b_model_forward_calls": (
                state.proposal_compiler_calls
                + (
                    0
                    if state.verifier_session is None
                    else state.verifier_session.stats.model_forward_calls
                )
            ),
        }
        if self.match_encoding == "incremental_kv":
            if state.gate_session is not None:
                value.update(state.gate_session.statistics())
                value["2b_model_forward_calls"] = (
                    int(
                        state.proposal_vectors is not None
                        and not state.proposal_manifest_replayed
                    )
                    + state.gate_session.stats.model_forward_calls
                )
            else:
                value["2b_model_forward_calls"] = int(
                    state.proposal_vectors is not None
                    and not state.proposal_manifest_replayed
                )
        elif self.match_encoding == "vit_feature_joint_decoder":
            value["2b_model_forward_calls"] = (
                int(state.proposal_vectors is not None)
                * int(not state.proposal_manifest_replayed)
                + state.proposal_vit_feature_encode_calls
                + state.proposal_match_calls
            )
        else:
            value["2b_model_forward_calls"] = (
                state.proposal_window_encode_calls
                + state.proposal_frame_encode_calls
                + int(
                    state.proposal_vectors is not None
                    and not state.proposal_manifest_replayed
                )
            )
        if state.verifier_session is not None:
            value.update(state.verifier_session.statistics())
            value["8b_model_forward_calls"] = (
                state.proposal_compiler_calls
                + state.verifier_session.stats.model_forward_calls
            )
        value["2b_model_forward_calls"] += retrieval_2b_calls
        return value


class TemporalUpdateProactiveController:
    """Frozen Embedding-2B event-sparse gate plus cached 8B verifier.

    A single cached 2B multimodal forward supplies the joint readout and native
    Yes/No cosine geometry.  The small temporal head calibrates those signals
    into the sole candidate probability.  Raw cosine never bypasses this gate;
    only a calibrated candidate may invoke the frozen 8B verifier and change
    the public proactive state.
    """

    def __init__(
        self,
        *,
        encoder: Any,
        answer_model: Any,
        cfg: dict[str, Any],
        task: str,
    ) -> None:
        if task not in {"STREAMINGBENCH_PO", "SSR", "CRR"}:
            raise ValueError(
                "update_head is only the shared readiness controller for PO/SSR/CRR; "
                "REC uses disjoint interval-delta counting"
            )
        self.encoder = encoder
        self.answer_model = answer_model
        self.cfg = cfg
        self.task = task
        self.name = controller_name("update_head")
        self.profile = TASK_PROFILES[task]
        self.frame_window = int(cfg["recent_window_frames"])
        checkpoint_path = Path(cfg["proactive_head_checkpoint"]).resolve()
        cache_key = str(checkpoint_path)
        cache = getattr(encoder, "_temporal_update_head_cache", {})
        if cache_key not in cache:
            cache[cache_key] = load_temporal_update_head(
                checkpoint_path,
                encoder.device,
            )
            setattr(encoder, "_temporal_update_head_cache", cache)
        self.head, self.checkpoint = cache[cache_key]
        if str(self.checkpoint.get("training_objective")) != (
            "embedding_calibrated_event_sparse_v9"
        ):
            raise RuntimeError(
                "update_head requires a v9 event-sparse checkpoint; "
                f"got {self.checkpoint.get('training_objective')!r}"
            )
        protocol = self.checkpoint.get("feature_protocol") or {}
        expected_window = int(protocol.get("recent_window_frames", -1))
        expected_fps = float(protocol.get("sample_fps", -1.0))
        if self.frame_window != expected_window:
            raise RuntimeError(
                f"Checkpoint was trained with recent{expected_window}, "
                f"but inference requested recent{self.frame_window}"
            )
        if abs(float(cfg["sample_fps"]) - expected_fps) > 1e-6:
            raise RuntimeError(
                f"Checkpoint was trained at {expected_fps:g} fps, "
                f"but inference requested {float(cfg['sample_fps']):g} fps"
            )
        feature_protocol = str(protocol.get("feature_protocol", ""))
        if feature_protocol != CACHED_UPDATE_FEATURE_PROTOCOL:
            raise RuntimeError(
                "update_head checkpoint was not trained from the exact bounded-KV "
                f"readout protocol: expected {CACHED_UPDATE_FEATURE_PROTOCOL!r}, "
                f"got {feature_protocol!r}"
            )
        if tuple(protocol.get("score_feature_names", ())) != PROACTIVE_SCORE_FEATURE_NAMES:
            raise RuntimeError("update_head checkpoint has incompatible score features")
        if int(protocol.get("feature_format_version", -1)) != 4:
            raise RuntimeError("update_head checkpoint is not feature format-v4")
        if tuple(protocol.get("gate_documents", ())) != GATE_DOCUMENTS:
            raise RuntimeError("update_head checkpoint has incompatible gate documents")
        if str(protocol.get("gate_document_instruction", "")) != GATE_DOCUMENT_INSTRUCTION:
            raise RuntimeError(
                "update_head checkpoint has incompatible gate document instruction"
            )
        if str(protocol.get("readout_prompt_version", "")) != UPDATE_READOUT_PROMPT_VERSION:
            raise RuntimeError("update_head checkpoint has incompatible readout prompt")
        self.update_threshold = float(self.checkpoint["update_threshold"])
        # The checkpoint's threshold is selected on the training domain.  Candidate
        # rates drift heavily across domains (67/58/33/12% in the handoff), so a
        # single inherited threshold can make a benchmark run fail for the wrong
        # reason; this sweeps it without retraining.  Absent, behaviour is
        # unchanged.
        self.checkpoint_threshold = self.update_threshold
        override = cfg.get("proactive_head_threshold")
        if override is not None:
            value = float(override)
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"proactive_head_threshold must lie in [0, 1], got {value}"
                )
            self.update_threshold = value
        self.operating_point = dict(
            self.checkpoint.get("validation", {}).get(
                "candidate_operating_point", {}
            )
        )
        self.checkpoint_path = checkpoint_path
        cache_attribute = "_proactive_evidence_readiness_v2_document_cache"
        cached = getattr(encoder, cache_attribute, None)
        if cached is None:
            vectors, elapsed = encoder.query_vectors(
                list(GATE_DOCUMENTS),
                instruction=GATE_DOCUMENT_INSTRUCTION,
            )
            cached = (vectors, float(elapsed))
            setattr(encoder, cache_attribute, cached)
        else:
            vectors, _ = cached
            elapsed = 0.0
        self.yes_vector = F.normalize(
            vectors[0].to(encoder.device).float().reshape(-1), dim=0
        )
        self.no_vector = F.normalize(
            vectors[1].to(encoder.device).float().reshape(-1), dim=0
        )
        self.document_embedding_seconds = float(elapsed)

    @property
    def persistent_document_vector_bytes(self) -> int:
        return int(
            self.yes_vector.numel() * self.yes_vector.element_size()
            + self.no_vector.numel() * self.no_vector.element_size()
        )

    def register(
        self,
        *,
        query_id: str,
        target_question: str,
        registration_timestamp: float,
        **_: Any,
    ) -> ProactiveState:
        return ProactiveState(
            query_id=str(query_id),
            target_question=str(target_question),
            registration_timestamp=float(registration_timestamp),
            registration_metrics={
                "controller": self.name,
                "controller_version": CONTROLLER_VERSION,
                "decision_mode": "update_head",
                "task_profile": self.profile.name,
                "cache_protocol": "fixed_question_prefix_plus_bounded_incremental_multimodal_kv",
                "cache_frame_window": self.frame_window,
                "feature_protocol": CACHED_UPDATE_FEATURE_PROTOCOL,
                "gate_backend": (
                    "frozen_embedding_2b_joint_readout_plus_yes_no_cosines_"
                    "event_sparse_temporal_gru"
                ),
                "checkpoint": str(self.checkpoint_path),
                "training_objective": self.checkpoint["training_objective"],
                "update_threshold": self.update_threshold,
                "checkpoint_update_threshold": self.checkpoint_threshold,
                "threshold_selection": (
                    "minimum_8b_candidates_subject_to_validation_event_recall"
                ),
                "head_loss_config": self.checkpoint.get("loss_config"),
                "validation_candidate_operating_point": self.operating_point,
                "head_trigger_policy": "calibrated_probability_level_threshold",
                "candidate_policy": "calibrated_head_only_raw_cosine_cannot_bypass",
                "initial_input": f"latest_{self.frame_window}_causal_independent_images",
                "subsequent_input": "only_newly_arrived_independent_image",
                "persistent_state": (
                    "cached_answer_plus_previous_2048d_vector_plus_previous_margin_"
                    "plus_gru_hidden"
                ),
                "answer_executor": "frozen_qwen3_vl_8b_cached_verifier_on_candidates_only",
                "8b_cache_visibility": (
                    "latest_unseen_recent_window_appended_only_at_candidate_times"
                ),
                "document_embedding_seconds_shared": self.document_embedding_seconds,
            },
            # Keep the head's initial cached-answer text identical to MMDuet
            # feature extraction.  SSR/CRR public state is still No because
            # current_positive/positive_latched both start false.
            cached_answer=NO_CACHED_ANSWER,
        )

    def prime(
        self,
        *,
        state: ProactiveState,
        sampler: Any,
        timestamp: float,
    ) -> dict[str, Any]:
        value = {
            "prime_called": False,
            "prime_reason": (
                "first_decision_initializes_latest_recent_window_to_match_"
                "cached_feature_training"
            ),
        }
        state.registration_metrics["prime"] = value
        return value

    def _new_session(
        self, backend: str, state: ProactiveState
    ) -> IncrementalMultimodalKVSession:
        owner = self.answer_model if backend == "8b" else self.encoder
        return IncrementalMultimodalKVSession(
            backend=backend,
            owner=owner,
            target_question=state.target_question,
            task=self.task,
            frame_window=self.frame_window,
        )

    def _new_positions(
        self, state: ProactiveState, sampler: Any, timestamp: float
    ) -> list[int]:
        available = sampler.positions_available_at(float(timestamp))
        if state.last_observation_position is None:
            return available[-self.frame_window :]
        return [value for value in available if value > state.last_observation_position]

    def _2b_readout(
        self,
        session: IncrementalMultimodalKVSession,
        frames: list[Image.Image],
        timestamps: list[float],
        cached_answer: str,
    ) -> tuple[torch.Tensor, bool, float, float, dict[str, Any]]:
        hidden, metrics = session.append_frames(
            frames,
            timestamps,
            decide=True,
            decision_prompt=_update_readout_question(timestamps[-1], cached_answer),
        )
        feature = F.normalize(hidden.reshape(-1), dim=0)
        yes = float(torch.dot(feature, self.yes_vector).item())
        no = float(torch.dot(feature, self.no_vector).item())
        metrics.update(
            {
                "gate_input_modality": (
                    "new_independent_image_plus_cached_question_recent_frame_kv_"
                    "plus_cached_answer_readout"
                ),
                "gate_score_yes": yes,
                "gate_score_no": no,
                "gate_margin_yes_minus_no": yes - no,
                "gate_2b_seconds": float(metrics["2b_forward_seconds"]),
            }
        )
        return feature, bool(yes > no), yes, no, metrics

    def _8b_verdict(
        self,
        session: IncrementalMultimodalKVSession,
        frames: list[Image.Image],
        timestamps: list[float],
    ) -> tuple[bool, str, dict[str, Any]]:
        logits, metrics = session.append_frames(frames, timestamps, decide=True)
        tokenizer = self.answer_model.processor.tokenizer
        negative_label, positive_label = verifier_label_pair(self.task)
        token_ids: dict[str, int] = {}
        for label in (negative_label, positive_label):
            values = tokenizer.encode(label, add_special_tokens=False)
            if len(values) != 1:
                raise RuntimeError(f"Expected one token for {label}, got {values}")
            token_ids[label] = int(values[0])
        logit_a = float(logits[token_ids[negative_label]].item())
        logit_b = float(logits[token_ids[positive_label]].item())
        response = positive_label if logit_b > logit_a else negative_label
        metrics.update(
            {
                "binary_choice_backend": "incremental_qwen3_vl_next_token_ab",
                "binary_choice_logit_a": logit_a,
                "binary_choice_logit_b": logit_b,
                "binary_choice_margin_b_minus_a": logit_b - logit_a,
                "answer_total_seconds": float(metrics["8b_forward_seconds"]),
            }
        )
        return parse_binary_verdict(response, self.task), response, metrics

    def _apply_verified_state(
        self, state: ProactiveState, verdict: bool, timestamp: float
    ) -> bool:
        previous = state.cached_answer
        if self.task == "SSR":
            if verdict:
                state.cached_answer = "Yes"
            elif previous != NO_CACHED_ANSWER:
                state.cached_answer = "No"
            state.current_positive = bool(verdict)
        else:
            if verdict:
                state.cached_answer = (
                    "TRIGGER" if self.task == "STREAMINGBENCH_PO" else "Yes"
                )
                state.positive_latched = True
            state.current_positive = state.positive_latched
        changed = state.cached_answer != previous
        if changed:
            state.answer_update_count += 1
        if verdict and state.last_positive_timestamp is None:
            state.last_positive_timestamp = float(timestamp)
        return changed

    def decide(
        self,
        *,
        state: ProactiveState,
        sampler: Any,
        timestamp: float,
    ) -> ProactiveDecision:
        if self.profile.positive_state_absorbing and state.positive_latched:
            state.decision_count += 1
            return ProactiveDecision(
                positive=True,
                positions=[],
                timestamps=[],
                frames=[],
                metrics={
                    "decision_positive": True,
                    "decision_reason": "absorbing_positive_state",
                    "gate_2b_called": False,
                    "verifier_8b_called": False,
                    "decision_index": state.decision_count - 1,
                },
            )
        positions = self._new_positions(state, sampler, float(timestamp))
        if not positions:
            return ProactiveDecision(
                positive=bool(state.positive_latched or state.current_positive),
                positions=[],
                timestamps=[],
                frames=[],
                metrics={
                    "decision_positive": bool(state.positive_latched or state.current_positive),
                    "decision_reason": "no_causal_frame",
                    "gate_2b_called": False,
                    "verifier_8b_called": False,
                },
            )
        batches = (
            [positions]
            if state.last_observation_position is None
            else [[value] for value in positions]
        )
        all_frames: list[Image.Image] = []
        all_timestamps: list[float] = []
        step_metrics: list[dict[str, Any]] = []
        for block_positions in batches:
            block_frames = sampler.decode_positions(block_positions)
            block_timestamps = sampler.timestamps(block_positions)
            all_frames.extend(block_frames)
            all_timestamps.extend(block_timestamps)
            realized_timestamp = float(block_timestamps[-1])
            if state.gate_session is None:
                state.gate_session = self._new_session("2b", state)
            feature, cosine_positive, yes, no, gate_metrics = self._2b_readout(
                state.gate_session,
                block_frames,
                block_timestamps,
                state.cached_answer,
            )
            time_value = update_time_feature(
                realized_timestamp,
                state.registration_timestamp,
                state.previous_decision_timestamp,
            ).to(self.encoder.device)
            started = time.perf_counter()
            value = self.head.step(
                feature,
                state.previous_joint_feature,
                PROACTIVE_TASK_IDS[self.task],
                time_value,
                torch.tensor(
                    [
                        yes,
                        no,
                        yes - no,
                        0.0
                        if state.previous_gate_margin is None
                        else (yes - no) - state.previous_gate_margin,
                    ],
                    device=self.encoder.device,
                    dtype=torch.float32,
                ),
                state.temporal_hidden,
            )
            if self.encoder.device.type == "cuda":
                torch.cuda.synchronize(self.encoder.device)
            head_seconds = time.perf_counter() - started
            update_probability = float(value["update_probability"].item())
            temporal_logit = float(value["temporal_logit"].item())
            candidate = bool(update_probability >= self.update_threshold)
            margin_delta = (
                0.0
                if state.previous_gate_margin is None
                else (yes - no) - state.previous_gate_margin
            )
            state.temporal_hidden = value["hidden"].detach()
            state.previous_joint_feature = feature.detach()
            state.previous_decision_timestamp = realized_timestamp
            state.previous_gate_margin = yes - no
            state.last_observation_position = int(block_positions[-1])
            state.update_head_call_count += 1
            state.update_head_seconds += float(head_seconds)
            state.gate_call_count += 1
            state.gate_positive_count += int(candidate)
            state.latest_gate_scores = [
                update_probability,
                yes,
                no,
                yes - no,
                margin_delta,
            ]

            verifier_called = False
            verifier_positive: bool | None = None
            raw_response = ""
            answer_changed = False
            verifier_metrics: dict[str, Any] = {}
            verifier_positions: list[int] = []
            verifier_timestamps: list[float] = []
            if candidate:
                if state.verifier_session is None:
                    state.verifier_session = self._new_session("8b", state)
                    verifier_positions = recent_positions(
                        sampler, realized_timestamp, self.frame_window
                    )
                    cache_initialization = "latest_recent_window_at_first_candidate"
                else:
                    available = sampler.positions_available_at(realized_timestamp)
                    verifier_positions = [
                        position
                        for position in available
                        if state.last_verifier_position is None
                        or position > state.last_verifier_position
                    ][-self.frame_window :]
                    cache_initialization = (
                        "catch_up_latest_unseen_frames_into_bounded_cache"
                    )
                if not verifier_positions:
                    raise RuntimeError("Candidate has no unseen frame for 8B verifier")
                verifier_frames = sampler.decode_positions(verifier_positions)
                verifier_timestamps = sampler.timestamps(verifier_positions)
                verifier_positive, raw_response, verifier_metrics = self._8b_verdict(
                    state.verifier_session, verifier_frames, verifier_timestamps
                )
                state.last_verifier_position = int(verifier_positions[-1])
                verifier_called = True
                state.verifier_call_count += 1
                state.verifier_positive_count += int(verifier_positive)
                answer_changed = self._apply_verified_state(
                    state, verifier_positive, realized_timestamp
                )
            state.decision_count += 1
            positive = bool(state.positive_latched or state.current_positive)
            metrics = {
                **gate_metrics,
                **verifier_metrics,
                "decision_positive": positive,
                "decision_reason": (
                    "8b_verified_positive"
                    if verifier_positive
                    else "8b_wait"
                    if verifier_called
                    else "calibrated_2b_gate_no"
                ),
                "decision_index": state.decision_count - 1,
                "selected_sample_positions": block_positions,
                "selected_timestamps_seconds": block_timestamps,
                "gate_2b_called": True,
                "gate_positive": candidate,
                "gate_candidate_policy": (
                    "calibrated_head_only_raw_cosine_cannot_bypass"
                ),
                "gate_calibrated_candidate": candidate,
                "gate_raw_cosine_positive": cosine_positive,
                "gate_raw_cosine_suppressed": bool(
                    cosine_positive and not candidate
                ),
                "gate_score_update": update_probability,
                "gate_temporal_logit": temporal_logit,
                "gate_threshold": self.update_threshold,
                "gate_score_yes": yes,
                "gate_score_no": no,
                "gate_margin_yes_minus_no": yes - no,
                "gate_margin_delta": margin_delta,
                "update_head_seconds": float(head_seconds),
                "verifier_8b_called": verifier_called,
                "verifier_positive": verifier_positive,
                "verifier_raw_response": raw_response if verifier_called else None,
                "verifier_selected_sample_positions": verifier_positions,
                "verifier_selected_timestamps_seconds": verifier_timestamps,
                "8b_cache_initialization": (
                    cache_initialization if verifier_called else None
                ),
                "cached_answer": state.cached_answer,
                "answer_changed": answer_changed,
            }
            step_metrics.append(metrics)
            if self.profile.positive_state_absorbing and positive:
                break

        merged = dict(step_metrics[-1])
        merged["processed_incremental_steps"] = len(step_metrics)
        merged["gate_2b_calls_this_decision"] = len(step_metrics)
        merged["gate_positive_count_this_decision"] = sum(
            value.get("gate_positive") is True for value in step_metrics
        )
        merged["calibrated_candidate_count_this_decision"] = sum(
            value.get("gate_calibrated_candidate") is True for value in step_metrics
        )
        merged["raw_cosine_positive_count_this_decision"] = sum(
            value.get("gate_raw_cosine_positive") is True for value in step_metrics
        )
        merged["raw_cosine_suppressed_count_this_decision"] = sum(
            value.get("gate_raw_cosine_suppressed") is True for value in step_metrics
        )
        merged["calibrated_with_raw_cosine_count_this_decision"] = sum(
            value.get("gate_calibrated_candidate") is True
            and value.get("gate_raw_cosine_positive") is True
            for value in step_metrics
        )
        merged["verifier_8b_calls_this_decision"] = sum(
            bool(value.get("verifier_8b_called")) for value in step_metrics
        )
        merged["verifier_positive_count_this_decision"] = sum(
            bool(value.get("verifier_8b_called"))
            and value.get("verifier_positive") is True
            for value in step_metrics
        )
        merged["gate_2b_seconds"] = sum(
            float(value.get("gate_2b_seconds", 0.0)) for value in step_metrics
        )
        merged["answer_total_seconds"] = sum(
            float(value.get("answer_total_seconds", 0.0)) for value in step_metrics
        )
        if len(step_metrics) > 1:
            merged["incremental_step_metrics"] = step_metrics
        return ProactiveDecision(
            positive=bool(state.positive_latched or state.current_positive),
            positions=positions,
            timestamps=all_timestamps,
            frames=all_frames,
            metrics=merged,
            answer=state.cached_answer,
        )

    def state_statistics(self, state: ProactiveState) -> dict[str, Any]:
        previous_bytes = (
            int(
                state.previous_joint_feature.numel()
                * state.previous_joint_feature.element_size()
            )
            if state.previous_joint_feature is not None
            else 0
        )
        hidden_bytes = (
            int(state.temporal_hidden.numel() * state.temporal_hidden.element_size())
            if state.temporal_hidden is not None
            else 0
        )
        value: dict[str, Any] = {
            "gate_2b_calls": state.gate_call_count,
            "gate_positive_count": state.gate_positive_count,
            "verifier_8b_calls": state.verifier_call_count,
            "verifier_positive_count": state.verifier_positive_count,
            "update_head_calls": state.update_head_call_count,
            "update_head_total_seconds": state.update_head_seconds,
            "answer_update_count": state.answer_update_count,
            "cached_answer": state.cached_answer,
            "persistent_previous_vector_bytes": previous_bytes,
            "persistent_gru_hidden_bytes": hidden_bytes,
            "persistent_temporal_state_bytes": previous_bytes + hidden_bytes,
        }
        if state.gate_session is not None:
            value.update(state.gate_session.statistics())
        if state.verifier_session is not None:
            value.update(state.verifier_session.statistics())
        return value


def build_proactive_controller(
    *,
    encoder: Any | None,
    answer_model: Any | None,
    cfg: dict[str, Any],
    task: str,
    decision_mode: str,
) -> (
    CachedProactiveController
    | ProposalMatchProactiveController
    | TemporalUpdateProactiveController
):
    if decision_mode == "proposal_match":
        if encoder is None:
            raise ValueError("proposal_match requires an embedding encoder")
        if answer_model is None:
            raise ValueError(
                "proposal_match requires the 8B proposal compiler/verifier"
            )
        return ProposalMatchProactiveController(
            encoder=encoder,
            answer_model=answer_model,
            cfg=cfg,
            task=task,
        )
    if decision_mode == "update_head":
        if encoder is None:
            raise ValueError("update_head requires an embedding encoder")
        if answer_model is None:
            raise ValueError("update_head requires the cached 8B verifier")
        return TemporalUpdateProactiveController(
            encoder=encoder,
            answer_model=answer_model,
            cfg=cfg,
            task=task,
        )
    if answer_model is None:
        raise ValueError(f"{decision_mode} requires an answer model")
    return CachedProactiveController(
        encoder=encoder,
        answer_model=answer_model,
        cfg=cfg,
        task=task,
        decision_mode=decision_mode,
    )
