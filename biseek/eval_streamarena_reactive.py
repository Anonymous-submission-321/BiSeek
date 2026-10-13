#!/usr/bin/env python3
"""BiSeek reactive on StreamArena's HR and RTP tracks.

A thin adapter over the same `eval_river_reactive` engine that StreamingBench
and OVO use, so the memory machinery being measured here -- the causal sampler,
the bounded FIFO vector memory, the recent/history pools, both retrieval
policies -- is the exercised one and not a re-implementation. What this file
supplies is StreamArena: how its rows become items, how an open-ended question
is asked, and how an answer is scored.

**The tracks.**

| | n | horizon | the question | frames that matter |
|---|---|---|---|---|
| HR | 877 | set, up to 7076 s | "in the past five minutes, how many shots..." | anywhere back to the start |
| RTP | 263 | always null | "is the woman speaking **now** the lead actress?" | the present |

**Open-ended, not multiple choice.** StreamingBench and OVO give four options and
the gate can be told to emit `ANSWER: X` for an option letter. StreamArena's
reactive rows have a free-text reference -- "6 times", "Pegasus 3", "No" -- so
that prompt cannot be reused. The recent-evidence gate here asks for the answer
itself or the literal token `RETRIEVE`, and scoring is left to the benchmark's
own LLM judge rather than a string match, because "6 times" and "6" are the same
answer and an exact-match scorer would call one of them wrong.

**Both policies are run, and they differ in where the 8B is spent:**

* `recent_first` -- show the model only the most recent frames and let it decide
  whether that is enough. It answers, or it emits `RETRIEVE` and only then is
  the memory searched. A malformed reply is read as RETRIEVE, which is the
  conservative direction. One extra 8B call per question buys skipping retrieval
  on the questions that do not need it.
* `retrieve_first` -- search memory with the question every time and answer from
  the retrieved history plus the recent pool. No routing call, constant cost,
  and no chance of the gate wrongly deciding it already knows.

Memory settings are inherited from the runs already exercised on OVO and
StreamingBench, so a number here is comparable with those rather than tuned
fresh for this benchmark.
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
TOOLS = REPO_ROOT / "biseek"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import biseek.cost_meter  # noqa: E402,F401  -- inert unless BISEEK_COST_METER=1
import eval_river_reactive as engine  # noqa: E402

DATA_ROOT = Path("data/StreamArena")
TRACKS = ("HR", "RTP")
QTYPE_CODE = {"HR": "h", "RTP": "r"}

# Which track this process is evaluating. Set from the config before
# engine.main() runs, because the engine calls our hooks without arguments.
TRACK = "HR"

ANSWER_PROMPT = """\
The images above are what you have seen of this video stream.

Question: {question}

Answer in English, so it can be compared with the English reference. If the
answer is text you read on screen in another language, give the English
translation and put the original in parentheses after it.

Give the answer itself and nothing else -- no preamble, no description of the
images, no reasoning. If the answer is a count or a name, give just that."""

RECENT_GATE_PROMPT = """\
Look only at the recent images above.

Question: {question}

Answer in English, so it can be compared with the English reference. If the
answer is text you read on screen in another language, give the English
translation and put the original in parentheses after it.

If those images contain enough visual evidence to answer, output the answer
itself on one line and nothing else. If they do not, output exactly `RETRIEVE`.
Do not explain and do not start with a preamble such as "From the images"."""


def build_prompt(item: dict[str, Any]) -> str:
    return ANSWER_PROMPT.format(question=item["question"])


def build_recent_gate_prompt(item: dict[str, Any]) -> str:
    """Open-ended replacement for the engine's option-letter gate.

    The engine's own gate asks for `ANSWER: X` where X is an option letter;
    StreamArena has no options, so a letter can never be produced and every
    question would route to RETRIEVE regardless of what the model saw -- which
    would silently turn `recent_first` into `retrieve_first` and make the
    comparison between them meaningless.
    """
    return RECENT_GATE_PROMPT.format(question=item["question"])


def parse_recent_gate(response: str, *,
                      num_options: int | None = None) -> tuple[str, str | None]:
    """Return ("ANSWER", text) or ("RETRIEVE", None), the engine's own contract.

    The signature and the two literals are copied from
    `eval_ovo_streaming.parse_recent_first_gate` because the caller unpacks
    exactly that; the second element carries an option letter there and carries
    the free-text answer here.

    Their parser accepts only a strict `ANSWER: X` for an option letter. On an
    open-ended track no answer can ever match it, so every question would route
    to RETRIEVE no matter what the model saw -- `recent_first` would silently
    become `retrieve_first` and the comparison between the two policies would be
    between a policy and itself. That is the bug this replacement exists to
    avoid.

    Unparseable stays RETRIEVE, the conservative direction: an unnecessary
    retrieval costs compute, answering without the evidence costs a point.
    """
    text = str(response or "").strip()
    if not text:
        return "RETRIEVE", None
    head = text.splitlines()[0].strip()
    if head.upper().startswith("RETRIEVE"):
        return "RETRIEVE", None
    head = re.sub(r"^ANSWER\s*:\s*", "", head, flags=re.IGNORECASE).strip()
    # A gate reply that begins by narrating the images is not an answer, it is
    # the model ignoring the format. With a small generation budget it also
    # arrives truncated -- `From the images, the "Target"` was scored as the
    # model's answer, which is a wrong answer AND a skipped retrieval from one
    # malformed reply. Route it to RETRIEVE, the conservative direction.
    if re.match(r"^(from|based on|according to|in|the)\s+(the\s+)?"
                r"(image|images|frame|frames|video|recent)\b",
                head, flags=re.IGNORECASE):
        return "RETRIEVE", None
    return ("ANSWER", head) if head else ("RETRIEVE", None)


def normalize_annotations(cfg: dict[str, Any],
                          configured_tasks: list[str]) -> list[dict[str, Any]]:
    rows = [json.loads(line)
            for line in (DATA_ROOT / "question.en.jsonl").read_text().splitlines()
            if line.strip()]
    records: list[dict[str, Any]] = []
    for raw in rows:
        if raw.get("qtype") != TRACK:
            continue
        video_path = DATA_ROOT / "videos" / raw["video_id"] / f"{raw['video_id']}.mp4"
        if not video_path.is_file():
            continue
        key = f"{raw['video_id']}::{raw['qid']}"
        records.append({
            "_key": key,
            "id": key,
            "question_id": str(raw["qid"]),
            "video": str(raw["video_id"]),
            "video_source": "StreamArena",
            "video_id": str(raw["video_id"]),
            "video_path": str(video_path),
            "category": TRACK,
            "task": TRACK,
            "question": str(raw["question"]),
            "router_query": str(raw["question"]),
            "conversation_history": [],
            # No options: scoring goes through the benchmark's judge, and
            # `choices` is left empty so any option-letter path fails loudly
            # rather than quietly scoring letters that do not exist.
            "choices": [],
            "ground_truth": str(raw["answer"]),
            "realtime": float(raw["ask_sec"]),
            "time_reference": None,
            "requires_audio": False,
            "horizon_sec": raw.get("horizon_sec"),
            "horizon_layer": raw.get("horizon_layer"),
            "evidence_ts_sec": raw.get("evidence_ts_sec"),
            "qtype_code": QTYPE_CODE[TRACK],
            "annotation_file": "question.en.jsonl",
        })
    dupes = [k for k, c in Counter(r["_key"] for r in records).items() if c > 1]
    if dupes:
        raise ValueError(f"Duplicate StreamArena keys: {dupes[:5]}")
    if not records:
        raise ValueError(f"No {TRACK} rows with a video on disk")
    subset = os.environ.get("BISEEK_SUBSET", "")
    if subset:
        keep = set(json.loads(Path(subset).read_text()))
        records = [r for r in records if r["_key"] in keep]
        print(f"[cost-meter] subset {Path(subset).name}: {len(records)} kept", flush=True)
    return records


def score_results(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Coverage and routing only -- correctness is the judge's job.

    An exact-match scorer would mark "6" wrong against "6 times". The records
    are written in the schema `judge_streamarena_local.py` reads, and the
    accuracy number comes from there, scored by the same judge as every other
    method on this benchmark.
    """
    by_layer: dict[str, int] = defaultdict(int)
    answered = 0
    routing = Counter()
    for r in records:
        if str(r.get("response", "")).strip():
            answered += 1
        by_layer[str(r.get("horizon_layer"))] += 1
        decision = (r.get("metrics") or {}).get("recent_first_decision")
        if decision:
            routing[str(decision)] += 1
    return {
        "track": TRACK,
        "records": len(records),
        "answered": answered,
        "answered_pct": round(100.0 * answered / max(1, len(records)), 2),
        "by_horizon_layer": dict(by_layer),
        "recent_first_routing": dict(routing),
        "scoring_note": ("open-ended; accuracy comes from "
                         "tools/judge_streamarena_local.py, not from this file"),
    }


def verify_dataset(cfg: dict[str, Any], configured_tasks: list[str],
                   *args: Any, **kwargs: Any) -> dict[str, Any]:
    records = normalize_annotations(cfg, configured_tasks)
    videos = {r["video_id"] for r in records}
    return {
        "benchmark": "StreamArena",
        "track": TRACK,
        "questions": len(records),
        "videos": len(videos),
        "open_ended": True,
        "has_options": False,
    }


def main() -> None:
    global TRACK
    for i, arg in enumerate(sys.argv):
        if arg == "--track" and i + 1 < len(sys.argv):
            TRACK = sys.argv[i + 1]
            del sys.argv[i : i + 2]
            break
    if TRACK not in TRACKS:
        raise SystemExit(f"--track must be one of {TRACKS}, got {TRACK!r}")

    engine.REACTIVE_TASKS = [TRACK]
    engine.RETRO_TASKS = []
    engine.normalize_annotations = normalize_annotations
    engine.build_prompt = build_prompt
    engine.score_results = score_results
    engine.verify_dataset = verify_dataset
    # Open-ended replacements for the option-letter gate. The prompt is built in
    # the engine and the reply is parsed in eval_ovo_streaming, so both modules
    # have to be patched -- patching only one leaves a strict `ANSWER: X` parser
    # reading a free-text answer and routing every question to RETRIEVE.
    import eval_ovo_streaming as ovo

    patched = []
    for mod in (engine, ovo):
        if hasattr(mod, "build_recent_first_gate_prompt"):
            mod.build_recent_first_gate_prompt = build_recent_gate_prompt
            patched.append(f"{mod.__name__}.build_recent_first_gate_prompt")
        if hasattr(mod, "parse_recent_first_gate"):
            mod.parse_recent_first_gate = parse_recent_gate
            patched.append(f"{mod.__name__}.parse_recent_first_gate")
    if not any("parse_recent_first_gate" in p for p in patched):
        raise SystemExit(
            "could not patch parse_recent_first_gate; the option-letter parser "
            "would route every open-ended answer to RETRIEVE")
    print("[streamarena-reactive] patched: " + ", ".join(patched), flush=True)
    engine.main()


if __name__ == "__main__":
    main()
