#!/usr/bin/env python3
"""Per-backbone verifier accept threshold, with no heavy imports.

Kept separate from ``proactive_cascade`` so the shell runners can ask for the
default without paying for a torch import -- one source of truth for both the
Python side and the run-name suffix the runners build.

The A/B verdict is ``logit_B > logit_A + margin``, so 0.0 means "any win
counts".  That is the right operating point for Qwen3-VL and Qwen2-VL.
Qwen2.5-VL is different: measured on StreamingBench PO it answers WAIT on 82%
of verifier calls, leaving 138 of 250 questions with no utterance at all.  Its
discrimination is not the problem -- in-window recall 30.9% against a 5.6%
false-positive rate is a separation of 25.3, better than Qwen2-VL's 17.9 -- the
decision point simply sits far toward WAIT.

-1.0 was selected on **OVO Forward** by replaying that run's recorded margins
(a different benchmark, different videos, different task), then transferred
unchanged.  On StreamingBench PO it moves 38.0 -> 44.8 at fps1 and 40.0 -> 50.4
at fps2 while spending *fewer* 8B calls (1,115 -> 996), because latching
earlier ends the polling sooner.
"""
from __future__ import annotations

# No backbone uses a non-zero margin by default. -1.0 helped Qwen2.5-VL on OVO Forward
# and StreamingBench PO but cost 4.2 points on StreamArena Pro, whose questions are
# monitored for far longer, so it is kept as an ablation rather than a default.
BACKBONE_VERIFIER_ACCEPT_MARGIN: dict[str, float] = {}


def default_verifier_accept_margin(
    answer_model_family: str | None = None,
    answer_model_path: str | None = None,
) -> float:
    """The accept threshold to use when the caller did not name one."""
    family = str(answer_model_family or "").strip().lower()
    if family in BACKBONE_VERIFIER_ACCEPT_MARGIN:
        return BACKBONE_VERIFIER_ACCEPT_MARGIN[family]
    # The family is sometimes left unset and only the path identifies the model.
    path = str(answer_model_path or "").lower().replace("-", "").replace("_", "")
    for key, value in BACKBONE_VERIFIER_ACCEPT_MARGIN.items():
        tag = key.replace("_", "").replace("vl", "")
        if tag and (f"qwen{tag}vl" in path or f"qwen{tag}" in path):
            return value
    return 0.0


if __name__ == "__main__":  # called by the shell runners
    import sys
    a = sys.argv[1] if len(sys.argv) > 1 else ""
    b = sys.argv[2] if len(sys.argv) > 2 else ""
    print(default_verifier_accept_margin(a, b))
