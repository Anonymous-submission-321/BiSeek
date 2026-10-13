#!/usr/bin/env python3
"""StreamArena's LLM judge, run against a local model instead of the Gemini API.

The prompts and the acceptance constants are copied verbatim from
``streamarena/judge.py`` so the scoring rule is the benchmark's, not ours. Only
the transport changes: their ``llm_as_judge`` posts to a Gemini endpoint, this
loads a local causal LM. We have no API key, and every baseline will be rerun
here anyway, so what matters is that one judge scores every method identically
rather than that it agrees with the paper's absolute numbers.

The Pro timing gate runs before the model is consulted, exactly as upstream:
a first utterance outside [-0.5, +2.0] s of the reference is zero regardless of
content, so those rows never reach the judge.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

# --- verbatim from streamarena/judge.py -----------------------------------
ARENA_ROOT = Path("outputs/StreamArena")
POOL_PRO = ARENA_ROOT / "pro"
POOL_HR = ARENA_ROOT / "hr"
POOL_RTP = ARENA_ROOT / "rtp"

TTFT_BIAS_SEC = 2.0
TIMING_LOWER_SEC = -0.5
TIMING_UPPER_SEC = +2.0

JUDGE_PROMPT_TMPL = """You are an impartial judge.
Decide whether the model's prediction correctly answers the question, based on the ground-truth answer.

Be lenient on:
- Wording / phrasing / language (Chinese vs English)
- Extra explanation around the right answer
- Equivalent expressions
- Numeric formats that mean the same thing

Be strict on:
- Wrong facts, wrong names, wrong numbers, wrong yes/no.
- If the model says it cannot answer / refuses / hallucinates, mark Incorrect.

Question: {question}
Ground truth answer: {gt}
Model prediction: {pred}

Answer only with a single word: "Correct" or "Incorrect".
"""

JUDGE_PROMPT_TMPL_E = """You are an impartial judge for a **proactive event-detection** task.

Task setup:
- The user asked the model to notify them when a SPECIFIC event happens.
- ``Question`` describes when the model should speak up.
- ``Ground truth answer`` is the SHORT trigger phrase the user wants to hear.
  Treat it as a signal that the target event was correctly detected -- NOT as
  a literal wording requirement.
- ``Model prediction`` is the sentence the model produced when it decided the
  event happened. Descriptive/caption-style wording is fine as long as it
  refers to the SAME event.

Mark **Correct** when the prediction:
- describes the same event / scene / action / moment the ground truth points to;
- captures the correct key subject / object / action of that event;
- adds extra factual context around the correct event.

Mark **Incorrect** when the prediction refers to a different event, names the
wrong subject or object, or reports something the ground truth does not.

Question: {question}
Ground truth answer: {gt}
Model prediction: {pred}

Answer only with a single word: "Correct" or "Incorrect".
"""
# --------------------------------------------------------------------------


def timing_ok(record: dict[str, Any]) -> tuple[bool, str]:
    """Upstream ``_e_timing_check``: gate Pro rows on the first utterance."""
    reference = record.get("answer_time_sec")
    if reference is None:
        return False, "missing_answer_time"
    messages = record.get("e_proactive_msgs") or []
    if not messages:
        return False, "empty_model_answer"
    start = messages[0].get("start_vid_ts")
    if start is None:
        commit = messages[0].get("vid_ts")
        if commit is None:
            return False, "missing_vid_ts"
        start = commit - TTFT_BIAS_SEC
    delta = float(start) - float(reference)
    if TIMING_LOWER_SEC <= delta <= TIMING_UPPER_SEC:
        return True, "judged"
    return False, "timing_violation"


def parse_verdict(raw: str) -> float:
    """Read a correct/incorrect verdict out of the judge's reply.

    Two traps. "incorrect" contains "correct", so a substring test scores every
    rejection as a pass; and a reasoning model puts its verdict at the END,
    after an analysis channel, so a prefix test reads the reasoning instead of
    the answer. Take the LAST standalone occurrence of either word.
    """
    text = raw.lower()
    last, verdict = -1, 0.0
    for word, value in (("incorrect", 0.0), ("correct", 1.0)):
        start = 0
        while True:
            i = text.find(word, start)
            if i < 0:
                break
            # "incorrect" also matches at the "correct" inside it; a hit whose
            # preceding characters spell "in" belongs to the longer word.
            if not (word == "correct" and text[max(0, i - 2):i] == "in"):
                if i > last:
                    last, verdict = i, value
            start = i + 1
    return verdict


def normalise(r: dict[str, Any]) -> dict[str, Any]:
    """Map a reactive (HR/RTP) record onto the proactive field names.

    Left alone, a reactive row has no `model_answer`, so the emptiness check
    scores it 0 before the judge ever sees it. `qtype_code` is set from the
    track because the timing gate must run for Pro rows ("e") and must NOT run
    for reactive rows, which have no trigger to be early or late for.
    """
    if "model_answer" in r or "trigger_vid_ts" in r:
        return r                                   # already proactive-shaped
    track = str(r.get("task") or r.get("category") or "").upper()
    return {
        **r,
        "uid": r.get("_key") or f"{r.get('video')}::{r.get('question_id')}",
        "video_id": r.get("video_id", r.get("video")),
        "qid": r.get("qid", r.get("question_id")),
        "model_answer": r.get("response", ""),
        "reference_answer": r.get("ground_truth", ""),
        "qtype_code": {"HR": "h", "RTP": "r"}.get(track, track.lower()[:1]),
        "qtype_public": track or r.get("qtype_public", ""),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", nargs="+", required=True,
                        help="rank_*.jsonl globs from one or more runs")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-memory-per-gpu", default="42GiB",
                        help="per-device ceiling handed to device_map='auto'; "
                             "empty string disables the cap")
    parser.add_argument("--model-path",
                        default="models/gpt-oss-120b",
                        help=("the one judge every method is scored by. gpt-oss-120b "
                              "is the strongest thing on this machine (122 GB, "
                              "complete) and stands in for the paper's Gemini; "
                              "pass Qwen2.5-7B-Instruct to cross-check agreement"))
    parser.add_argument("--max-new-tokens", type=int, default=192,
                        help=("gpt-oss emits a reasoning channel before its answer, "
                              "so four tokens returned nothing usable"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--verdict-prefix", default="judge",
                        help=("filename stem for per-run verdicts, so a second "
                              "judge writes judge7b-<run>.jsonl instead of "
                              "overwriting -- and, more importantly, does not "
                              "get resumed from by a different judge, which "
                              "would silently mix two models' verdicts in one "
                              "column"))
    parser.add_argument("--no-resume", action="store_true",
                        help=("re-judge rows that already have a verdict in "
                              "judge-<run>.jsonl instead of reusing it"))
    parser.add_argument("--judge-incomplete", action="store_true",
                        help=("judge runs that have not written one rank_*.done "
                              "per rank file. Off by default: an in-flight run's "
                              "error rows get frozen at 0 and are never re-judged "
                              "when the adapter retries them"))
    parser.add_argument("--dry-run", action="store_true",
                        help=("resolve the timing gate and report how many rows "
                              "would reach the judge, without loading it. The "
                              "judge only sees rows that already passed timing, "
                              "so this is what sizes the job"))
    args = parser.parse_args()

    # Tag every row with the run directory it came from. Each Pro arm answers
    # the SAME 774 questions, so uid is not unique across runs: keyed by uid
    # alone, one flat judged file silently keeps whichever arm was written last
    # and the report would read the ablation's content scores as the main arm's.
    # `ours-gate` and `ours-gate-poll2` even share (model, mode) exactly.
    seen: dict[tuple[str, str], dict[str, Any]] = {}
    run_pool: dict[str, Path] = {}
    incomplete: set[str] = set()
    for pattern in args.records:
        for path in sorted(glob.glob(pattern)):
            # Qualify with the pool. Run names are unique within a pool but
            # not across them -- `base-timechat` existed under pro/, hr/ and
            # rtp/ at once -- so the bare directory name would merge three
            # different runs' verdicts under one key.
            _p = Path(path).resolve().parent
            run = f"{_p.parent.name}/{_p.name}"
            # Skip runs that have not finished. The docstring above promises
            # this script can be run at any time and will "pick up the runs
            # that have finished since"; it could not, because two rules
            # combine badly on an UNfinished run. Resume keeps the last row
            # written per (run, uid), so a retried question's success does
            # overwrite its earlier `error` row -- but a verdict already in
            # judged_all.jsonl is reused unconditionally further down. Judge an
            # in-flight run and its error rows are frozen at
            # `<empty_model_answer>` 0, so the answers produced when the
            # adapter retries them are never scored. Nothing raises; the run
            # just reads low forever. Flash-VStream RTP had exactly three such
            # rows waiting when this was found.
            #
            # `.done` is written by the rank itself as its last act, so one per
            # rank file is the run's own statement that it finished.
            if not args.judge_incomplete:
                ranks = len(glob.glob(str(_p / "rank_[0-9][0-9].jsonl")))
                if len(glob.glob(str(_p / "rank_*.done"))) < ranks:
                    incomplete.add(run)
                    continue

            for line in open(path, errors="ignore"):
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                # Two record schemas reach this judge. The proactive runners
                # write `uid` / `model_answer` / `reference_answer`; the
                # reactive engine writes `_key` / `response` / `ground_truth`
                # and none of the others. Judged as-is, every reactive row has
                # an empty `model_answer` and is scored 0 as
                # `<empty_model_answer>` -- the whole HR/RTP track would come
                # back at 0% correctness with no error raised. Normalise here
                # rather than in the runners, so records already on disk work.
                r = normalise(r)
                uid = r.get("uid") or f"{r.get('video_id')}:{r.get('qid')}"
                # A resumed rank re-appends a uid; keep the last one written.
                seen[(run, uid)] = {**r, "_run": run}
                run_pool[run] = Path(path).resolve().parent.parent
    rows = list(seen.values())
    if not rows:
        raise SystemExit("no records")
    runs = sorted({r["_run"] for r in rows})
    print(f"loaded {len(rows)} records from {len(runs)} run(s): {', '.join(runs)}")
    if incomplete:
        # Say it out loud. Silently judging fewer runs than asked for is how a
        # summary table ends up missing an arm that everyone assumes is there.
        print(f"SKIPPED {len(incomplete)} run(s) still in flight (no "
              f"rank_*.done): {', '.join(sorted(incomplete))}")
        print("  re-run this once they finish; --judge-incomplete overrides, "
              "resolved rows are re-judged only for runs that finished")

    # Runs that exist on disk and were never matched by any --records pattern.
    #
    # The skip message above only speaks for runs a glob DID reach. A run the
    # patterns never touched is invisible to it, and that is the failure that
    # actually happened: the Pro pool was enumerated by prefix
    # (`pro/ours-*`, `pro/base-*`, `pro/blind-*`) while hr/ and rtp/ used `*`,
    # so four complete Qwen2 Pro runs sat unjudged with nothing reporting it.
    # An allow-list of prefixes silently excludes every prefix invented later.
    on_disk: set[str] = set()
    for pool in (POOL_PRO, POOL_HR, POOL_RTP):
        for d in sorted(pool.glob("*")):
            if d.is_dir() and d.name != "state" and any(
                    d.glob("rank_[0-9][0-9].jsonl")):
                on_disk.add(f"{pool.name}/{d.name}")
    unmatched = sorted(on_disk - set(runs) - incomplete)
    if unmatched:
        print(f"NOT MATCHED by any --records pattern: {', '.join(unmatched)}")
        print("  these runs exist and are not being judged. Widen the globs.")

    # Resume: rows already judged in a previous window keep their verdict, so a
    # short window can judge what is finished and a later one only the rest.
    prior: dict[tuple[str, str], dict[str, Any]] = {}
    if not args.no_resume:
        for run in runs:
            f = run_pool[run] / f"{args.verdict_prefix}-{run.split('/')[-1]}.jsonl"
            if not f.is_file():
                continue
            for line in open(f, errors="ignore"):
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                uid = r.get("uid") or f"{r.get('video_id')}:{r.get('qid')}"
                if "judge_score" in r:
                    prior[(run, uid)] = r
        if prior:
            print(f"resume: {len(prior)} rows already judged")

    # Timing first: those rows are decided without the model, as upstream does.
    pending, decided = [], []
    for r in rows:
        was = prior.get((r["_run"], r.get("uid") or f"{r.get('video_id')}:{r.get('qid')}"))
        if was is not None:
            # Re-tag on the way through: a verdict file written before this
            # script tagged rows (or by hand) has no `_run`, and without it the
            # resumed rows all collapse into one "unknown" run at output time --
            # which is the very merge the tagging exists to prevent.
            decided.append({**was, "_run": r["_run"]})
            continue
        if str(r.get("qtype_code")) == "e":
            ok, kind = timing_ok(r)
            if not ok:
                decided.append({**r, "judge_score": 0.0, "judge_raw": f"<{kind}>"})
                continue
        if not str(r.get("model_answer", "")).strip():
            decided.append({**r, "judge_score": 0.0, "judge_raw": "<empty_model_answer>"})
            continue
        pending.append(r)
    print(f"timing/emptiness resolved {len(decided)}; {len(pending)} go to the judge")

    if args.dry_run:
        by_run: dict[str, list[int]] = {}
        for r in rows:
            key = str(r.get("_run", "?"))
            slot = by_run.setdefault(key, [0, 0])
            slot[0] += 1
            # Same rule the real path uses, not a proactive-only one: a
            # reactive row has no `timing_in_window`, so testing it reported 0
            # rows to judge for HR and RTP while 133 of them were in fact
            # queued -- a sizing table that disagrees with the job it sizes.
            gated = str(r.get("qtype_code")) != "e" or r.get("timing_in_window")
            if gated and str(r.get("model_answer", "")).strip():
                slot[1] += 1
        print(f"{'run':26s} {'rows':>7s} {'to judge':>9s}")
        for key, (n, k) in sorted(by_run.items()):
            print(f"{key:26s} {n:7d} {k:9d}")
        print(f"total rows {len(rows)}, to judge {len(pending)} "
              f"({100 * len(pending) / max(1, len(rows)):.1f}%)")
        return

    if pending:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        snap = args.model_path
        if os.path.isdir(os.path.join(snap, "snapshots")):
            inner = sorted(glob.glob(os.path.join(snap, "snapshots", "*")))
            snap = inner[-1] if inner else snap
        tok = AutoTokenizer.from_pretrained(snap)

        # gpt-oss-120b ships MXFP4. Keeping it quantized would fit three cards,
        # but transformers' MXFP4 path imports
        # `triton.tools.tensor_descriptor`, which needs triton >= 3.3; this
        # environment has 3.2.0 and upgrading it would touch the torch install
        # that every running experiment depends on. So it is dequantized to
        # bf16 instead: ~240 GB across six cards rather than ~122 GB across
        # three. The judge is a short job -- only rows that already passed the
        # timing gate reach it, about 5% -- so holding the machine briefly at
        # the end costs less than a mid-run dependency change.
        load_kwargs: dict[str, Any] = {"dtype": torch.bfloat16, "device_map": "auto"}
        # Cap each card explicitly. `device_map="auto"` fills greedily from
        # GPU 0 and the first attempt died there with 45.30 of 47.40 GiB in
        # use -- 11.97 GiB of that merely RESERVED by the caching allocator,
        # not live tensors. The aggregate was never the problem: 240 GB of
        # dequantised weights across six 47.4 GB cards is 284 GB of capacity.
        # The distribution was. A per-device ceiling forces the balance and
        # leaves headroom for activations, and expandable_segments (set in
        # the job script) stops the reserved-but-unused fragmentation that
        # made a card look full while holding 12 GB of nothing.
        if torch.cuda.is_available() and args.max_memory_per_gpu:
            n = torch.cuda.device_count()
            load_kwargs["max_memory"] = {i: args.max_memory_per_gpu for i in range(n)}
            print(f"max_memory: {args.max_memory_per_gpu} x {n} cards", flush=True)
        try:
            cfg = json.loads(Path(snap, "config.json").read_text())
            if (cfg.get("quantization_config") or {}).get("quant_method") == "mxfp4":
                from transformers import Mxfp4Config

                load_kwargs["quantization_config"] = Mxfp4Config(dequantize=True)
                print("mxfp4 checkpoint: loading dequantized to bf16")
        except Exception:
            pass
        model = AutoModelForCausalLM.from_pretrained(snap, **load_kwargs).eval()
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "left"

        for start in range(0, len(pending), args.batch_size):
            chunk = pending[start : start + args.batch_size]
            prompts = []
            for r in chunk:
                tmpl = (JUDGE_PROMPT_TMPL_E if str(r.get("qtype_code")) == "e"
                        else JUDGE_PROMPT_TMPL)
                text = tmpl.format(question=r.get("question", ""),
                                   gt=r.get("reference_answer", ""),
                                   pred=r.get("model_answer", ""))
                prompts.append(tok.apply_chat_template(
                    [{"role": "user", "content": text}],
                    tokenize=False, add_generation_prompt=True))
            # 234 GB of weights across six 48 GB cards leaves about 9 GB a card
            # for activations and KV. This job gets one six-card window at the
            # end of the schedule, so an OOM must not end it: fall back to
            # smaller sub-batches, down to one prompt at a time, and only give
            # up on a row if even that fails.
            def _generate(items: list[dict[str, Any]], texts: list[str]):
                size = len(items)
                while size >= 1:
                    try:
                        done = []
                        for at in range(0, len(items), size):
                            sub = tok(texts[at : at + size], return_tensors="pt",
                                      padding=True).to(model.device)
                            with torch.inference_mode():
                                seqs = model.generate(
                                    **sub, max_new_tokens=int(args.max_new_tokens),
                                    do_sample=False, pad_token_id=tok.pad_token_id)
                            width = sub["input_ids"].shape[1]
                            done += [tok.decode(q[width:], skip_special_tokens=True).strip()
                                     for q in seqs]
                        return done
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        size //= 2
                        print(f"  OOM at batch {size * 2}; retrying at {size}",
                              flush=True)
                return None

            raws = _generate(chunk, prompts)
            if raws is None:
                for r in chunk:
                    decided.append({**r, "judge_score": 0.0, "judge_raw": "<oom>"})
                print("  giving up on this batch after OOM at batch size 1",
                      flush=True)
                continue
            for r, raw in zip(chunk, raws):
                score = parse_verdict(raw)
                decided.append({**r, "judge_score": score, "judge_raw": raw[-120:]})
            if (start // args.batch_size) % 10 == 0:
                print(f"  judged {min(start + args.batch_size, len(pending))}/{len(pending)}",
                      flush=True)

    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as handle:
        for r in decided:
            handle.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Per-run files, the layout summarise_streamarena.py already reads. The
    # combined file above keeps `_run` so it can still be split later; these are
    # what the reports consume.
    per_run: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in decided:
        per_run[str(r.get("_run", "unknown"))].append(r)
    for run, group in sorted(per_run.items()):
        f = run_pool.get(run, out_path.parent) / f"{args.verdict_prefix}-{run.split('/')[-1]}.jsonl"
        with f.open("w") as handle:
            for r in group:
                handle.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  wrote {f.name}  ({len(group)} rows)")

    print(f"\n===== per run (failures count as 0) =====")
    print(f"  {'run':26s} {'n':>6s} {'timing+content':>15s}")
    for run, group in sorted(per_run.items()):
        t = sum(r["judge_score"] for r in group)
        print(f"  {run:26s} {len(group):6d} {t / max(1, len(group)) * 100:14.1f}%")

    total = sum(r["judge_score"] for r in decided)
    print(f"\n===== pooled across all arms (not a result; sanity only) =====")
    print(f"  acc = {total / len(decided):.4f}  ({int(total)}/{len(decided)})")
    kinds: dict[str, int] = defaultdict(int)
    for r in decided:
        raw = str(r.get("judge_raw", ""))
        kinds[raw[1:-1] if raw.startswith("<") else "judged"] += 1
    print("\n  breakdown:")
    for k, v in sorted(kinds.items(), key=lambda kv: -kv[1]):
        print(f"    {k:22s} {v:5d}  ({v / len(decided) * 100:5.1f}%)")
    by_layer: dict[str, list[float]] = defaultdict(list)
    for r in decided:
        by_layer[str(r.get("horizon_layer"))].append(r["judge_score"])
    print("\n  by horizon layer:")
    for k in sorted(by_layer):
        v = by_layer[k]
        print(f"    {k:6s} acc={sum(v)/len(v):.4f}  ({int(sum(v))}/{len(v)})")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
