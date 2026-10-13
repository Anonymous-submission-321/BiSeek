#!/usr/bin/env python3
"""Score OVO-Bench result files with the current official metric semantics.

The upstream scorer assumes every model response is a string, while several
result files shipped in the official repository store ``batch_decode`` output
as a one-element list.  It also rejects model names not hard-coded upstream.
This small adapter keeps the same task-level correctness rules and macro
averaging, but accepts arbitrary model names and normalizes response shapes.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


CATEGORIES = {
    "backward": ["EPM", "ASI", "HLD"],
    "realtime": ["OCR", "ACR", "ATR", "STU", "FPD", "OJR"],
    "forward": ["REC", "SSR", "CRR"],
}


def response_text(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def load_results(result_dir: Path) -> dict[str, list]:
    merged = {category: [] for category in CATEGORIES}
    paths = sorted(path for path in result_dir.glob("*.json") if not path.name.endswith("_score.json"))
    if not paths:
        raise FileNotFoundError(f"No OVO-Bench result JSON files under {result_dir}")
    for path in paths:
        value = json.loads(path.read_text())
        if not all(category in value for category in CATEGORIES):
            continue
        for category in CATEGORIES:
            merged[category].extend(value[category])
    return merged


def score_choice(response, ground_truth: str) -> int:
    text = response_text(response)
    return int(text is not None and ground_truth in text)


def score_forward(task: str, response, info: dict) -> int:
    text = response_text(response)
    if text is None:
        return 0
    if task == "REC":
        digits = "".join(re.findall(r"\d+", text))
        return int(digits == str(info["count"]))

    expected_type = int(info["type"])
    if (text == "N" and expected_type == 0) or (text == "Y" and expected_type == 1):
        return 1
    expected_text = "No" if expected_type == 0 else "Yes"
    return int(expected_text in text)


def calculate(results: dict[str, list]) -> dict:
    scores: dict[str, list[int]] = defaultdict(list)
    for category in ("backward", "realtime"):
        for item in results[category]:
            scores[item["task"]].append(score_choice(item.get("response"), item["ground_truth"]))
    for item in results["forward"]:
        for info in item["test_info"]:
            scores[item["task"]].append(score_forward(item["task"], info.get("response"), info))

    task_scores = {
        task: {
            "accuracy": sum(values) / len(values),
            "paper_table_value": 100.0 * sum(values) / len(values),
            "num_evaluations": len(values),
        }
        for task, values in sorted(scores.items())
        if values
    }
    category_scores = {}
    for category, tasks in CATEGORIES.items():
        available = [task_scores[task]["accuracy"] for task in tasks if task in task_scores]
        if available:
            category_scores[category] = {
                "accuracy": sum(available) / len(available),
                "paper_table_value": 100.0 * sum(available) / len(available),
                "num_tasks": len(available),
            }
    overall_values = [category_scores[name]["accuracy"] for name in CATEGORIES if name in category_scores]
    overall = None
    if overall_values:
        overall = {
            "accuracy": sum(overall_values) / len(overall_values),
            "paper_table_value": 100.0 * sum(overall_values) / len(overall_values),
            "num_categories": len(overall_values),
        }
    return {"task_scores": task_scores, "category_scores": category_scores, "overall": overall}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    summary = calculate(load_results(args.result_dir))
    print(json.dumps(summary, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
