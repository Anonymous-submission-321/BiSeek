#!/usr/bin/env python3
"""Temporal calibrator over frozen cached Qwen3-VL-Embedding readouts.

The serving/extraction path keeps a fixed question prefix and bounded recent
frame KV, appends only each new independent image, and conditions the readout
prompt on the answer currently exposed to the user.  This module trains one
causal GRU head to propose answer-update candidates.  The head consumes both
the joint multimodal readout and the embedding model's native Yes/No document
cosines.  It never makes the final answer/wait decision; only a positive,
calibrated candidate invokes the frozen 8B verifier.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


PROACTIVE_TASK_IDS = {
    "MAGQA": 0,
    "STREAMINGBENCH_PO": 1,
    "REC": 2,
    "SSR": 3,
    "CRR": 4,
}
NO_CACHED_ANSWER = "[NO PREVIOUS ANSWER]"
CACHED_ANSWER_PRESENT = "[ANSWER PREVIOUSLY PUBLISHED]"
PROACTIVE_SCORE_FEATURE_NAMES = (
    "yes_cosine",
    "no_cosine",
    "yes_minus_no_margin",
    "margin_delta",
)
MARGIN_SCORE_FEATURE_INDEX = PROACTIVE_SCORE_FEATURE_NAMES.index("yes_minus_no_margin")
# v10 derives these two extra inputs from the frozen margin already stored in
# format-v4 features.  They are deliberately *not* computed from the head's own
# output: feeding a calibrated logit back into the head would make the module
# autoregressive and destroy the single-forward-pass property that keeps
# trajectory sampling cheap.
PROACTIVE_SURGE_FEATURE_NAMES = (
    "margin_surge_z",
    "margin_surge_z_delta",
)
EVENT_SPARSE_TRAINING_OBJECTIVE = (
    "embedding_calibrated_event_sparse_v9"
)
TARGET_BAND_GRPO_TRAINING_OBJECTIVE = (
    "embedding_calibrated_target_band_grpo_v10"
)
# Objective name -> objective version.  The loader accepts exactly these and
# fails closed on anything older, so a v8 point-wise-BCE checkpoint can never be
# silently served as a calibrated candidate gate.
SUPPORTED_TRAINING_OBJECTIVES = {
    EVENT_SPARSE_TRAINING_OBJECTIVE: 9,
    TARGET_BAND_GRPO_TRAINING_OBJECTIVE: 10,
}


def update_time_feature(
    timestamp: float,
    registration_timestamp: float,
    previous_timestamp: float | None,
) -> torch.Tensor:
    elapsed = max(0.0, float(timestamp) - float(registration_timestamp))
    delta = elapsed if previous_timestamp is None else max(
        0.0, float(timestamp) - float(previous_timestamp)
    )
    return torch.tensor(
        [
            math.log1p(elapsed) / math.log1p(300.0),
            math.log1p(delta) / math.log1p(10.0),
        ],
        dtype=torch.float32,
    )


DEFAULT_SURGE_WINDOW = 64
DEFAULT_SURGE_MIN_HISTORY = 4
DEFAULT_SURGE_CLIP = 8.0
_SURGE_EPSILON = 1e-6
_MAD_TO_SIGMA = 1.4826


def _robust_surge_z(
    window: torch.Tensor,
    *,
    min_history: int,
    clip: float,
) -> float:
    """Robust z-score of ``window[-1]`` against the rest of the causal window.

    One shared implementation keeps the online serving tracker and the batched
    training derivation bit-identical.  ``torch.median`` returns the lower middle
    element for even counts rather than the average of the two middles; routing
    both paths through this helper means that convention can never drift between
    training and serving.
    """

    values = window.reshape(-1).float()
    if int(values.numel()) < int(min_history):
        return 0.0
    center = torch.median(values)
    deviation = torch.median((values - center).abs())
    scale = _MAD_TO_SIGMA * float(deviation) + _SURGE_EPSILON
    value = (float(values[-1]) - float(center)) / scale
    # A constant window drives MAD to zero, so an unclipped ratio can reach
    # thousands and destabilize the unnormalized score-feature path.
    return float(max(-float(clip), min(float(clip), value)))


class MarginSurgeTracker:
    """Causal, per-query self-normalized surge over the frozen 2B margin.

    This is *not* the retired RCSC controller.  RCSC fed the same robust z-score
    into a hand-written enter/exit/cooldown state machine that made the final
    publish decision, and failed (StreamingBench 8.0%).  Here the z-score is
    only one more input feature for a head whose decision rule is still learned,
    which is what lets the gate normalize against each query's own score history
    instead of against one global threshold shared across domains.
    """

    def __init__(
        self,
        *,
        window: int = DEFAULT_SURGE_WINDOW,
        min_history: int = DEFAULT_SURGE_MIN_HISTORY,
        clip: float = DEFAULT_SURGE_CLIP,
    ) -> None:
        if int(window) < 2:
            raise ValueError("surge window must keep at least two observations")
        if int(min_history) < 2:
            raise ValueError("surge min_history must be at least two")
        if int(min_history) > int(window):
            raise ValueError("surge min_history cannot exceed the window")
        if float(clip) <= 0.0:
            raise ValueError("surge clip must be positive")
        self.window = int(window)
        self.min_history = int(min_history)
        self.clip = float(clip)
        self._history: deque[float] = deque(maxlen=self.window)
        self._previous_z: float | None = None

    def update(self, margin: float) -> tuple[float, float]:
        """Append one new margin and return ``(surge_z, surge_z_delta)``."""

        self._history.append(float(margin))
        value = _robust_surge_z(
            torch.tensor(list(self._history), dtype=torch.float32),
            min_history=self.min_history,
            clip=self.clip,
        )
        delta = 0.0 if self._previous_z is None else value - self._previous_z
        self._previous_z = value
        return value, delta

    def features(self, margin: float, device: torch.device | None = None) -> torch.Tensor:
        value, delta = self.update(margin)
        return torch.tensor([value, delta], dtype=torch.float32, device=device)


def margin_surge_features(
    margins: torch.Tensor,
    *,
    window: int = DEFAULT_SURGE_WINDOW,
    min_history: int = DEFAULT_SURGE_MIN_HISTORY,
    clip: float = DEFAULT_SURGE_CLIP,
) -> torch.Tensor:
    """Batched causal surge features for one episode's margin sequence.

    Equivalent by construction to feeding the same margins through
    :class:`MarginSurgeTracker` one step at a time; ``tests`` assert that
    equivalence so a training/serving mismatch fails loudly.
    """

    values = margins.reshape(-1).float()
    length = int(values.numel())
    output = torch.zeros(length, len(PROACTIVE_SURGE_FEATURE_NAMES), dtype=torch.float32)
    previous: float | None = None
    for index in range(length):
        start = max(0, index + 1 - int(window))
        value = _robust_surge_z(
            values[start : index + 1],
            min_history=min_history,
            clip=clip,
        )
        output[index, 0] = value
        output[index, 1] = 0.0 if previous is None else value - previous
        previous = value
    return output


@dataclass(frozen=True)
class ProactiveHeadConfig:
    input_dim: int = 2048
    hidden_dim: int = 256
    task_embedding_dim: int = 0
    num_tasks: int = 5
    time_feature_dim: int = 2
    score_feature_dim: int = len(PROACTIVE_SCORE_FEATURE_NAMES)
    # 0 keeps the exact v8/v9 input contract; v10 sets this to the number of
    # derived surge features.
    surge_feature_dim: int = 0
    dropout: float = 0.1


class TemporalUpdateHead(nn.Module):
    """One shared causal update gate for every proactive task."""

    def __init__(self, config: ProactiveHeadConfig) -> None:
        super().__init__()
        self.config = config
        self.input_norm = nn.LayerNorm(config.input_dim)
        self.task_embedding = (
            nn.Embedding(config.num_tasks, config.task_embedding_dim)
            if config.task_embedding_dim > 0
            else None
        )
        projected_input = (
            config.input_dim * 2
            + config.task_embedding_dim
            + config.time_feature_dim
            + config.score_feature_dim
            + config.surge_feature_dim
        )
        self.input_projection = nn.Sequential(
            nn.Linear(projected_input, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.gru = nn.GRU(config.hidden_dim, config.hidden_dim, batch_first=True)
        self.update_head = nn.Linear(config.hidden_dim, 1)
        # Keep the frozen embedding geometry visible at the output instead of
        # asking the temporal head to rediscover it from z_t.  This scalar is
        # trained jointly with the small head; the 2B backbone stays frozen.
        self.margin_scale = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        features: torch.Tensor,
        task_ids: torch.Tensor,
        time_features: torch.Tensor,
        score_features: torch.Tensor,
        hidden: torch.Tensor | None = None,
        surge_features: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if features.ndim != 3 or features.shape[-1] != self.config.input_dim:
            raise ValueError(
                f"features must be [B,T,{self.config.input_dim}], got {features.shape}"
            )
        if time_features.shape[:2] != features.shape[:2]:
            raise ValueError("time_features must align with features [B,T]")
        expected_score_shape = (*features.shape[:2], self.config.score_feature_dim)
        if tuple(score_features.shape) != expected_score_shape:
            raise ValueError(
                f"score_features must be {expected_score_shape}, got "
                f"{tuple(score_features.shape)}"
            )
        if self.config.surge_feature_dim > 0:
            expected_surge_shape = (*features.shape[:2], self.config.surge_feature_dim)
            if surge_features is None:
                raise ValueError(
                    "surge_features are required when surge_feature_dim > 0"
                )
            if tuple(surge_features.shape) != expected_surge_shape:
                raise ValueError(
                    f"surge_features must be {expected_surge_shape}, got "
                    f"{tuple(surge_features.shape)}"
                )
        elif surge_features is not None:
            raise ValueError(
                "surge_features were supplied to a head configured without them"
            )
        normalized = self.input_norm(features.float())
        delta = torch.cat(
            (torch.zeros_like(normalized[:, :1]), normalized[:, 1:] - normalized[:, :-1]),
            dim=1,
        )
        components = [
            normalized,
            delta,
            score_features.float(),
            time_features.float(),
        ]
        if surge_features is not None:
            components.append(surge_features.float())
        if self.task_embedding is not None:
            components.append(
                self.task_embedding(task_ids.long()).unsqueeze(1).expand(
                    -1, features.shape[1], -1
                )
            )
        projected = self.input_projection(torch.cat(components, dim=-1))
        output, next_hidden = self.gru(projected, hidden)
        temporal_logits = self.update_head(output).squeeze(-1)
        margin = score_features[..., 2].float()
        return {
            "update_logits": temporal_logits + self.margin_scale * margin,
            "temporal_logits": temporal_logits,
            "hidden": next_hidden,
        }

    @torch.inference_mode()
    def step(
        self,
        feature: torch.Tensor,
        previous_feature: torch.Tensor | None,
        task_id: int,
        time_features: torch.Tensor,
        score_features: torch.Tensor,
        hidden: torch.Tensor | None,
        surge_features: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if feature.ndim != 1:
            raise ValueError("feature must be one [D] vector")
        if tuple(score_features.shape) != (self.config.score_feature_dim,):
            raise ValueError(
                f"score_features must be [{self.config.score_feature_dim}], got "
                f"{tuple(score_features.shape)}"
            )
        if self.config.surge_feature_dim > 0:
            if surge_features is None:
                raise ValueError(
                    "surge_features are required when surge_feature_dim > 0"
                )
            if tuple(surge_features.shape) != (self.config.surge_feature_dim,):
                raise ValueError(
                    f"surge_features must be [{self.config.surge_feature_dim}], got "
                    f"{tuple(surge_features.shape)}"
                )
        elif surge_features is not None:
            raise ValueError(
                "surge_features were supplied to a head configured without them"
            )
        current = self.input_norm(feature.float())
        delta = torch.zeros_like(current)
        if previous_feature is not None:
            delta = current - self.input_norm(previous_feature.float())
        components = [current, delta, score_features.float(), time_features.float()]
        if surge_features is not None:
            components.append(surge_features.float())
        if self.task_embedding is not None:
            components.append(
                self.task_embedding(
                    torch.tensor([task_id], device=feature.device, dtype=torch.long)
                )[0]
            )
        projected = self.input_projection(torch.cat(components).view(1, -1)).view(1, 1, -1)
        output, next_hidden = self.gru(projected, hidden)
        temporal_logit = self.update_head(output[:, -1]).squeeze()
        logit = temporal_logit + self.margin_scale * score_features[2].float()
        return {
            "update_logit": logit,
            "temporal_logit": temporal_logit,
            "update_probability": torch.sigmoid(logit),
            "hidden": next_hidden,
        }

    def checkpoint(self) -> dict[str, Any]:
        return {"config": asdict(self.config), "state_dict": self.state_dict()}


def load_temporal_update_head(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[TemporalUpdateHead, dict[str, Any]]:
    path = Path(checkpoint_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Temporal update checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    objective = checkpoint.get("training_objective")
    expected_version = SUPPORTED_TRAINING_OBJECTIVES.get(str(objective))
    if (
        expected_version is None
        or int(checkpoint.get("training_objective_version", -1)) != expected_version
    ):
        raise RuntimeError(
            "Legacy proactive-head checkpoint is not supported; event-sparse v9 "
            "or target-band GRPO v10 retraining from calibrated format-v4 "
            f"features is required, got {objective!r}"
        )
    if int(checkpoint.get("config", {}).get("score_feature_dim", -1)) != len(
        PROACTIVE_SCORE_FEATURE_NAMES
    ):
        raise RuntimeError("Checkpoint has incompatible proactive score features")
    surge_dim = int(checkpoint.get("config", {}).get("surge_feature_dim", 0))
    if surge_dim not in {0, len(PROACTIVE_SURGE_FEATURE_NAMES)}:
        raise RuntimeError(
            f"Checkpoint declares {surge_dim} surge features; only 0 or "
            f"{len(PROACTIVE_SURGE_FEATURE_NAMES)} are implemented"
        )
    model = TemporalUpdateHead(ProactiveHeadConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device).eval()
    return model, checkpoint


def adapt_state_dict_for_surge(
    state_dict: dict[str, Any],
    config: ProactiveHeadConfig,
) -> tuple[dict[str, Any], list[str]]:
    """Widen a surge-free ``input_projection`` so a v9 checkpoint can warm-start v10.

    The new surge columns are zeroed rather than randomly initialized, so the
    adapted head reproduces the source checkpoint's logits exactly for *any*
    surge input.  Reinforcement learning then starts from the v9 operating point
    instead of a perturbed one, and the KL reference anchors a policy that is
    genuinely v9 rather than v9 plus noise.
    """

    name = "input_projection.0.weight"
    if config.surge_feature_dim <= 0 or name not in state_dict:
        return dict(state_dict), []
    weight = state_dict[name]
    expected = (
        2 * config.input_dim
        + config.score_feature_dim
        + config.time_feature_dim
        + config.surge_feature_dim
        + config.task_embedding_dim
    )
    if int(weight.shape[1]) == expected:
        return dict(state_dict), []
    if int(weight.shape[1]) != expected - config.surge_feature_dim:
        raise RuntimeError(
            f"Cannot adapt {name} with {weight.shape[1]} input columns to a head "
            f"expecting {expected}"
        )
    # Surge inputs are appended after the time features and before any task
    # embedding, so the copy has to split there rather than at the end.
    split = (
        2 * config.input_dim + config.score_feature_dim + config.time_feature_dim
    )
    widened = weight.new_zeros((int(weight.shape[0]), expected))
    widened[:, :split] = weight[:, :split]
    widened[:, split + config.surge_feature_dim :] = weight[:, split:]
    adapted = dict(state_dict)
    adapted[name] = widened
    return adapted, [name]


def _negative_log_event_probability(logits: torch.Tensor) -> torch.Tensor:
    """Return ``-log(1 - prod(1 - sigmoid(logits)))`` stably.

    The expression is the negative log likelihood that at least one Bernoulli
    candidate fires in an event window.  Expressing the no-candidate
    probability through ``softplus`` avoids explicitly multiplying values near
    one, while the two branches are stable on either side of ``log(2)``.
    """

    if logits.numel() == 0:
        raise ValueError("An event window must contain at least one valid step")
    values = logits.float()
    # For very negative logits, ``softplus(x)`` is approximately ``exp(x)``;
    # materializing it first can underflow even though the desired NLL remains
    # finite (approximately ``-logsumexp(x)``).  Work in log-hazard space and
    # use that asymptote before exponentiating.
    safe_values = values.clamp_min(-50.0)
    log_softplus = torch.where(
        values < -50.0,
        values,
        torch.log(F.softplus(safe_values)),
    )
    log_hazard = torch.logsumexp(log_softplus, dim=0)
    if float(log_hazard.detach()) < -10.0:
        return -log_hazard
    hazard = torch.exp(log_hazard)
    cutoff = math.log(2.0)
    log_probability = torch.where(
        hazard <= cutoff,
        torch.log(-torch.expm1(-hazard)),
        torch.log1p(-torch.exp(-hazard)),
    )
    return -log_probability


def event_sparse_update_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    *,
    event_tolerance_steps: int,
    event_coverage_weight: float,
    outside_window_weight: float,
    duplicate_weight: float,
    call_cost_weight: float,
) -> dict[str, torch.Tensor]:
    """Event-level sparse candidate objective over padded causal episodes.

    Each exact update label defines a causal window ``[event, event+tolerance]``.
    A window is clipped before the next event so one candidate cannot satisfy
    two updates.  The loss asks for at least one candidate in every window,
    treats every valid step outside all windows as negative, penalizes expected
    duplicate calls beyond one per window, and adds a global expected-call
    cost.  Padded steps never participate.
    """

    if logits.ndim != 2 or targets.shape != logits.shape or mask.shape != logits.shape:
        raise ValueError("logits, targets, and mask must have identical [B,T] shapes")
    if int(event_tolerance_steps) < 0:
        raise ValueError("event_tolerance_steps must be non-negative")
    weights = {
        "event_coverage_weight": event_coverage_weight,
        "outside_window_weight": outside_window_weight,
        "duplicate_weight": duplicate_weight,
        "call_cost_weight": call_cost_weight,
    }
    if any(float(value) < 0.0 for value in weights.values()):
        raise ValueError(f"Event-sparse loss weights must be non-negative: {weights}")

    valid_mask = mask.bool()
    event_losses: list[torch.Tensor] = []
    duplicate_losses: list[torch.Tensor] = []
    outside_masks = valid_mask.clone()
    probabilities = torch.sigmoid(logits.float())
    event_count = 0

    for row in range(int(logits.shape[0])):
        valid_length = int(valid_mask[row].sum().item())
        if valid_length == 0:
            continue
        event_indices = torch.nonzero(
            targets[row, :valid_length] > 0.5,
            as_tuple=False,
        ).flatten().tolist()
        for event_offset, event_index in enumerate(event_indices):
            next_event = (
                int(event_indices[event_offset + 1])
                if event_offset + 1 < len(event_indices)
                else valid_length
            )
            stop = min(
                valid_length,
                int(event_index) + int(event_tolerance_steps) + 1,
                next_event,
            )
            if stop <= int(event_index):
                raise RuntimeError("Event window is empty after causal clipping")
            window_logits = logits[row, int(event_index) : stop]
            window_probabilities = probabilities[row, int(event_index) : stop]
            event_losses.append(_negative_log_event_probability(window_logits))
            duplicate_losses.append(
                torch.relu(window_probabilities.sum() - 1.0).square()
            )
            outside_masks[row, int(event_index) : stop] = False
            event_count += 1

    differentiable_zero = logits.float().sum() * 0.0
    coverage = (
        torch.stack(event_losses).mean() if event_losses else differentiable_zero
    )
    duplicates = (
        torch.stack(duplicate_losses).mean()
        if duplicate_losses
        else differentiable_zero
    )
    outside = (
        F.softplus(logits.float()[outside_masks]).mean()
        if bool(outside_masks.any())
        else differentiable_zero
    )
    call_cost = (
        probabilities[valid_mask].mean()
        if bool(valid_mask.any())
        else differentiable_zero
    )
    total = (
        float(event_coverage_weight) * coverage
        + float(outside_window_weight) * outside
        + float(duplicate_weight) * duplicates
        + float(call_cost_weight) * call_cost
    )
    return {
        "loss": total,
        "event_coverage_loss": coverage,
        "outside_window_loss": outside,
        "duplicate_loss": duplicates,
        "call_cost_loss": call_cost,
        "event_count": torch.tensor(
            event_count,
            device=logits.device,
            dtype=torch.long,
        ),
        "valid_step_count": valid_mask.sum(),
        "outside_step_count": outside_masks.sum(),
    }


# --------------------------------------------------------------------------
# v10: target-band GRPO over sampled candidate trajectories
# --------------------------------------------------------------------------


def causal_event_windows(
    targets: list[int], tolerance: int, valid_length: int | None = None
) -> list[tuple[int, int]]:
    """Inclusive ``[event, stop]`` windows, clipped before the next event.

    Identical window arithmetic to :func:`event_sparse_update_loss` and
    :func:`match_events`, so the v9 metric, the v9 loss, and the v10 reward all
    credit exactly the same opportunities.
    """

    events = sorted(int(value) for value in targets)
    windows: list[tuple[int, int]] = []
    for index, event in enumerate(events):
        stop = event + int(tolerance)
        if index + 1 < len(events):
            stop = min(stop, events[index + 1] - 1)
        if valid_length is not None:
            stop = min(stop, int(valid_length) - 1)
        windows.append((event, max(event, stop)))
    return windows


@dataclass(frozen=True)
class EventMatching:
    matched: int
    delays: tuple[int, ...]
    matched_fires: tuple[int, ...]
    in_window_fires: tuple[int, ...]
    windows: tuple[tuple[int, int], ...]


def match_events_detailed(
    predicted: list[int], targets: list[int], tolerance: int
) -> EventMatching:
    """Maximum monotone causal matching, retaining which fires were used.

    A candidate strictly before an event never counts: the 8B verifier cannot
    confirm evidence that has not arrived yet.  One candidate can satisfy at
    most one event because each window is clipped before the next event.
    """

    predictions = sorted(int(value) for value in predicted)
    events = sorted(int(value) for value in targets)
    windows = causal_event_windows(events, tolerance)
    matched = 0
    delays: list[int] = []
    matched_fires: list[int] = []
    prediction_index = 0
    for event, stop in windows:
        while (
            prediction_index < len(predictions)
            and predictions[prediction_index] < event
        ):
            prediction_index += 1
        if prediction_index >= len(predictions):
            break
        prediction = predictions[prediction_index]
        if prediction <= stop:
            matched += 1
            delays.append(prediction - event)
            matched_fires.append(prediction)
            prediction_index += 1
    in_window = [
        value
        for value in predictions
        if any(start <= value <= stop for start, stop in windows)
    ]
    return EventMatching(
        matched=matched,
        delays=tuple(delays),
        matched_fires=tuple(matched_fires),
        in_window_fires=tuple(in_window),
        windows=tuple(windows),
    )


def match_events(
    predicted: list[int], targets: list[int], tolerance: int
) -> tuple[int, list[int]]:
    """Maximum monotone causal matching in non-overlapping event windows."""

    value = match_events_detailed(predicted, targets, tolerance)
    return value.matched, list(value.delays)


@dataclass(frozen=True)
class TrajectoryRewardConfig:
    """Deployment-shaped reward for one sampled candidate trajectory.

    Every term maps to a metric the handoff already requires reporting: timely
    event coverage, premature/late/missed splits, duplicate verifier calls, and
    total 8B call volume.  ``target_escalation_ratio`` closes the loop that v8
    and v9 both left open: v8 had no constraint on the call rate and collapsed
    to an ~86-92% pass-through, while v9's fixed-weight call cost overshot and
    dropped SSR recall.  The band penalizes *both* directions instead.
    """

    event_tolerance_steps: int = 2
    hit_reward: float = 2.0
    late_discount: float = 0.5
    miss_penalty: float = 1.5
    duplicate_penalty: float = 0.25
    false_positive_penalty: float = 1.0
    call_cost_weight: float = 0.05
    target_escalation_ratio: float = 0.25
    target_band_halfwidth: float = 0.10
    target_band_weight: float = 1.0

    def __post_init__(self) -> None:
        if int(self.event_tolerance_steps) < 0:
            raise ValueError("event_tolerance_steps must be non-negative")
        if not 0.0 <= float(self.late_discount) <= 1.0:
            raise ValueError("late_discount must lie in [0, 1]")
        if not 0.0 <= float(self.target_escalation_ratio) <= 1.0:
            raise ValueError("target_escalation_ratio must lie in [0, 1]")
        if float(self.target_band_halfwidth) < 0.0:
            raise ValueError("target_band_halfwidth must be non-negative")
        negatives = {
            "hit_reward": self.hit_reward,
            "miss_penalty": self.miss_penalty,
            "duplicate_penalty": self.duplicate_penalty,
            "false_positive_penalty": self.false_positive_penalty,
            "call_cost_weight": self.call_cost_weight,
            "target_band_weight": self.target_band_weight,
        }
        offenders = {name: value for name, value in negatives.items() if float(value) < 0.0}
        if offenders:
            raise ValueError(f"Reward magnitudes must be non-negative: {offenders}")


def target_band_deviation(ratio: float, center: float, halfwidth: float) -> float:
    """Signed-free distance from ``ratio`` to the target escalation band.

    Zero inside ``[center-halfwidth, center+halfwidth]``, growing linearly
    outside it in either direction.  Suppressing only over-escalation reproduces
    v9's failure mode, where the gate is free to collapse toward silence.
    """

    lower = float(center) - float(halfwidth)
    upper = float(center) + float(halfwidth)
    if ratio > upper:
        return float(ratio) - upper
    if ratio < lower:
        return lower - float(ratio)
    return 0.0


def _volume_terms(
    fired: int,
    false_positives: int,
    valid_steps: int,
    config: TrajectoryRewardConfig,
) -> float:
    """Reward terms that depend only on call counts, not on the matching.

    Isolated so a counterfactual flip at a step outside every event window can be
    scored in closed form instead of re-running the matcher.
    """

    ratio = fired / valid_steps
    deviation = target_band_deviation(
        ratio,
        config.target_escalation_ratio,
        config.target_band_halfwidth,
    )
    return (
        -float(config.false_positive_penalty) * (false_positives / valid_steps)
        - float(config.call_cost_weight) * ratio
        # Charged per *excess call* rather than per unit of rate deviation.
        # ``deviation * valid_steps`` is the number of calls beyond the band edge,
        # so the marginal cost of one more out-of-band call is exactly
        # ``target_band_weight`` no matter how long the episode is.  Penalizing the
        # rate instead makes that marginal cost ``weight / valid_steps``, which on
        # a 12-step episode is under 0.1 and loses outright to a coverage term
        # worth ``hit_reward + miss_penalty`` -- the band then cannot hold the call
        # rate at all, which is the whole reason it exists.
        - float(config.target_band_weight) * deviation * valid_steps
    )


def trajectory_reward(
    actions: list[int],
    targets: list[int],
    config: TrajectoryRewardConfig,
) -> dict[str, float]:
    """Deterministic reward for one 0/1 candidate trajectory over valid steps."""

    if len(actions) != len(targets):
        raise ValueError("actions and targets must cover the same valid steps")
    valid_steps = len(actions)
    if valid_steps == 0:
        raise ValueError("A trajectory must contain at least one valid step")
    fires = [index for index, value in enumerate(actions) if value]
    events = [index for index, value in enumerate(targets) if value]
    matching = match_events_detailed(fires, events, config.event_tolerance_steps)
    tolerance = max(1, int(config.event_tolerance_steps))
    coverage = 0.0
    for delay in matching.delays:
        timeliness = 1.0 - float(config.late_discount) * (float(delay) / tolerance)
        coverage += float(config.hit_reward) * timeliness
    missed = len(events) - matching.matched
    miss = -float(config.miss_penalty) * missed
    duplicates = len(matching.in_window_fires) - matching.matched
    duplicate = -float(config.duplicate_penalty) * duplicates
    false_positives = len(fires) - len(matching.in_window_fires)
    # Linear in the spurious-call *rate*, deliberately not Em-Garde's saturating
    # ``1 - 2^{-n_fp/n}``.  Saturation was designed for short proposal windows; on
    # these long sparse episodes it flattens after a handful of false calls, so
    # every further spurious step becomes almost free and the per-step gradient
    # for suppressing it vanishes.  Dividing by the valid step count keeps the
    # term scale-free across episodes of very different lengths.
    false_positive = -float(config.false_positive_penalty) * (
        false_positives / valid_steps
    )
    ratio = len(fires) / valid_steps
    call_cost = -float(config.call_cost_weight) * ratio
    deviation = target_band_deviation(
        ratio,
        config.target_escalation_ratio,
        config.target_band_halfwidth,
    )
    band = -float(config.target_band_weight) * deviation * valid_steps
    return {
        "reward": coverage + miss + duplicate + false_positive + call_cost + band,
        "coverage_reward": coverage,
        "miss_penalty": miss,
        "duplicate_penalty": duplicate,
        "false_positive_penalty": false_positive,
        "call_cost": call_cost,
        "target_band_penalty": band,
        "target_band_deviation": deviation,
        "escalation_ratio": ratio,
        "matched_events": float(matching.matched),
        "target_events": float(len(events)),
        "fired_steps": float(len(fires)),
        "valid_steps": float(valid_steps),
    }


def per_step_counterfactual_advantages(
    actions: list[int],
    targets: list[int],
    config: TrajectoryRewardConfig,
) -> list[float]:
    """Exact ``R(a) - R(flip_t(a))`` for every step of one trajectory.

    A single scalar trajectory advantage cannot teach this gate what it exists to
    learn.  One spurious step moves the reward by roughly
    ``false_positive_penalty / valid_steps`` while the coverage term swings it by
    ``hit_reward + miss_penalty``; that shared coverage noise does not cancel in
    any estimator built from whole-trajectory returns, so resolving the spurious
    step's contribution would need thousands of rollouts.  Differencing against
    the same trajectory with only step ``t`` flipped cancels every term the flip
    cannot affect, leaving the step's own contribution exactly.

    Flipping a step outside every causal event window cannot change the matching,
    so those steps use the closed-form volume terms and only in-window steps pay
    for a re-match.  ``tests`` assert the shortcut equals the full recomputation.
    """

    base = trajectory_reward(actions, targets, config)
    valid_steps = len(actions)
    events = [index for index, value in enumerate(targets) if value]
    windows = causal_event_windows(events, config.event_tolerance_steps, valid_steps)
    covered = {
        step
        for start, stop in windows
        for step in range(start, min(stop, valid_steps - 1) + 1)
    }
    fired = int(base["fired_steps"])
    false_positives = fired - len(
        match_events_detailed(
            [index for index, value in enumerate(actions) if value],
            events,
            config.event_tolerance_steps,
        ).in_window_fires
    )
    base_volume = _volume_terms(fired, false_positives, valid_steps, config)
    advantages: list[float] = []
    for step in range(valid_steps):
        if step in covered:
            flipped = list(actions)
            flipped[step] = 0 if actions[step] else 1
            alternative = trajectory_reward(flipped, targets, config)["reward"]
            advantages.append(base["reward"] - alternative)
            continue
        change = -1 if actions[step] else 1
        advantages.append(
            base_volume
            - _volume_terms(
                fired + change,
                false_positives + change,
                valid_steps,
                config,
            )
        )
    return advantages


ADVANTAGE_MODES = ("counterfactual", "trajectory")


def target_band_grpo_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    *,
    reward_config: TrajectoryRewardConfig,
    rollouts: int,
    sampling_temperature: float = 1.0,
    entropy_weight: float = 0.0,
    kl_weight: float = 0.0,
    reference_logits: torch.Tensor | None = None,
    advantage_epsilon: float = 1e-6,
    advantage_mode: str = "counterfactual",
    generator: torch.Generator | None = None,
) -> dict[str, torch.Tensor]:
    """Group-relative policy gradient on sampled candidate trajectories.

    The head is *not* autoregressive: each step's logit depends only on frozen
    2B features, never on the head's own past actions.  One forward pass
    therefore yields the whole causal logit sequence, and ``rollouts``
    trajectories can be sampled from it and scored by a deterministic CPU
    reward.  That is what makes this objective cost roughly the same as the
    supervised v9 objective instead of requiring per-step rollouts through the
    backbone.

    ``advantage_mode`` matters more than it looks.  ``trajectory`` is textbook
    GRPO: one group-standardized scalar per rollout, shared by every step.  It is
    kept only as an ablation, because on these episodes it cannot teach the
    behaviour this objective exists to learn -- see
    :func:`per_step_counterfactual_advantages` for why.  ``counterfactual`` is
    the default and scores each step against the same trajectory with only that
    step flipped.
    """

    if logits.ndim != 2 or targets.shape != logits.shape or mask.shape != logits.shape:
        raise ValueError("logits, targets, and mask must have identical [B,T] shapes")
    if int(rollouts) < 2:
        raise ValueError("Group-relative advantages need at least two rollouts")
    if advantage_mode not in ADVANTAGE_MODES:
        raise ValueError(
            f"advantage_mode must be one of {ADVANTAGE_MODES}, got {advantage_mode!r}"
        )
    if float(sampling_temperature) <= 0.0:
        raise ValueError("sampling_temperature must be positive")
    if kl_weight and reference_logits is None:
        raise ValueError("kl_weight requires reference_logits")
    if reference_logits is not None and reference_logits.shape != logits.shape:
        raise ValueError("reference_logits must match the logits shape")

    valid_mask = mask.bool()
    batch, steps = logits.shape
    scaled = logits.float() / float(sampling_temperature)
    log_probability_one = F.logsigmoid(scaled)
    log_probability_zero = F.logsigmoid(-scaled)
    probabilities = torch.sigmoid(scaled)

    expanded = probabilities.detach().unsqueeze(1).expand(batch, int(rollouts), steps)
    uniform = torch.rand(
        expanded.shape,
        device=logits.device,
        dtype=expanded.dtype,
        generator=generator,
    )
    actions = (uniform < expanded).to(probabilities.dtype)
    keep = valid_mask.unsqueeze(1).expand_as(actions)
    actions = actions * keep

    step_log_probability = (
        actions * log_probability_one.unsqueeze(1)
        + (1.0 - actions) * log_probability_zero.unsqueeze(1)
    ) * keep
    valid_counts = valid_mask.sum(dim=1).clamp_min(1)
    # Mean rather than sum over steps keeps the gradient scale comparable across
    # episodes whose lengths differ by an order of magnitude.
    log_probability = step_log_probability.sum(dim=2) / valid_counts.unsqueeze(1)

    action_rows = actions.detach().to("cpu").to(torch.int64).tolist()
    target_rows = targets.detach().to("cpu").to(torch.int64).tolist()
    length_rows = valid_mask.sum(dim=1).detach().to("cpu").tolist()
    rewards = torch.zeros(batch, int(rollouts), dtype=torch.float32)
    step_advantages = torch.zeros(batch, int(rollouts), steps, dtype=torch.float32)
    component_totals: dict[str, float] = {}
    scored = 0
    for row in range(batch):
        length = int(length_rows[row])
        if length == 0:
            continue
        episode_targets = target_rows[row][:length]
        for rollout in range(int(rollouts)):
            trajectory = action_rows[row][rollout][:length]
            value = trajectory_reward(trajectory, episode_targets, reward_config)
            rewards[row, rollout] = value["reward"]
            if advantage_mode == "counterfactual":
                step_advantages[row, rollout, :length] = torch.tensor(
                    per_step_counterfactual_advantages(
                        trajectory, episode_targets, reward_config
                    ),
                    dtype=torch.float32,
                )
            for name, component in value.items():
                component_totals[name] = component_totals.get(name, 0.0) + component
            scored += 1
    rewards = rewards.to(logits.device)
    step_advantages = step_advantages.to(logits.device)

    episode_mask = valid_mask.any(dim=1)
    centered = rewards - rewards.mean(dim=1, keepdim=True)
    spread = rewards.std(dim=1, keepdim=True, unbiased=False)
    # A group whose rollouts all earn the same reward carries no learning signal;
    # tracking how often that happens is the primary diagnostic for a policy that
    # has already saturated and stopped exploring.
    degenerate = (spread.squeeze(1) <= advantage_epsilon) & episode_mask
    scale = spread + advantage_epsilon

    episode_count = episode_mask.sum().clamp_min(1)
    if advantage_mode == "trajectory":
        advantages = (centered / scale) * episode_mask.unsqueeze(1)
        policy_gradient = -(
            (advantages.detach() * log_probability).sum()
            / (episode_count * int(rollouts))
        )
    else:
        # Already in absolute reward units and already centred by construction, so
        # it is deliberately *not* divided by the group reward spread: that would
        # blow up to 1/advantage_epsilon exactly when the rollouts agree.  It also
        # means a saturated policy still receives a correct gradient, whereas the
        # trajectory mode stalls once every rollout earns the same return.  The
        # sign is intrinsic, so it multiplies log P(taken action) directly.
        advantages = step_advantages * keep
        policy_gradient = -(
            (advantages.detach() * step_log_probability).sum()
            / (valid_mask.sum().clamp_min(1) * int(rollouts))
        )

    entropy_per_step = -(
        probabilities * log_probability_one + (1.0 - probabilities) * log_probability_zero
    )
    entropy = (entropy_per_step * valid_mask).sum() / valid_mask.sum().clamp_min(1)

    if reference_logits is None:
        kl = logits.float().sum() * 0.0
    else:
        reference = reference_logits.float() / float(sampling_temperature)
        kl_per_step = probabilities * (
            log_probability_one - F.logsigmoid(reference)
        ) + (1.0 - probabilities) * (
            log_probability_zero - F.logsigmoid(-reference)
        )
        kl = (kl_per_step * valid_mask).sum() / valid_mask.sum().clamp_min(1)

    total = policy_gradient + float(kl_weight) * kl - float(entropy_weight) * entropy
    scored_count = max(1, scored)
    output: dict[str, torch.Tensor] = {
        "loss": total,
        "policy_gradient_loss": policy_gradient,
        "entropy": entropy,
        "kl_to_reference": kl,
        "mean_reward": rewards[episode_mask].mean()
        if bool(episode_mask.any())
        else logits.float().sum() * 0.0,
        "degenerate_group_count": degenerate.sum(),
        "episode_count": episode_mask.sum(),
        "rollout_count": torch.tensor(
            scored, device=logits.device, dtype=torch.long
        ),
        "valid_step_count": valid_mask.sum(),
    }
    for name, component in component_totals.items():
        output[f"reward_{name}"] = torch.tensor(
            component / scored_count, device=logits.device, dtype=torch.float32
        )
    return output
