#!/usr/bin/env python3
"""Prepare and verify the official OVO-Bench pre-chunked video release.

The Hugging Face dataset stores one uncompressed tar archive split into
``chunked_videos.tar.partaa`` ... ``partao``.  This script streams the parts
directly into ``tar`` (without creating another 143 GB combined archive),
checks the filenames required by the current official annotation file, and
links the extracted clips into the vendored official evaluator.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import string
import subprocess
from collections import Counter
from pathlib import Path


DEFAULT_ROOT = Path("data/OVO-Bench")
PART_PREFIX = "chunked_videos.tar.part"
FORWARD_TASKS = {"REC", "SSR", "CRR"}


def archive_parts(archive_dir: Path) -> list[Path]:
    parts = sorted(archive_dir.glob(f"{PART_PREFIX}??"))
    if not parts:
        raise FileNotFoundError(f"No {PART_PREFIX}?? files found under {archive_dir}")

    # Python 3.8 compatibility: str.removeprefix was added in Python 3.9.
    suffixes = [path.name[len(PART_PREFIX) :] for path in parts]
    expected = ["a" + letter for letter in string.ascii_lowercase[: len(parts)]]
    if suffixes != expected:
        raise RuntimeError(f"Split archive is not contiguous: got {suffixes}, expected {expected}")
    if suffixes[-1] != "ao":
        raise RuntimeError(f"Expected the official 15 parts aa..ao, found aa..{suffixes[-1]}")

    empty = [path.name for path in parts if path.stat().st_size == 0]
    if empty:
        raise RuntimeError(f"Empty archive parts: {empty}")
    return parts


def extract_split_tar(parts: list[Path], output: Path) -> None:
    marker = output / ".ovo_bench_extraction_complete"
    if marker.is_file():
        print(f"Extraction marker exists, reusing {output}", flush=True)
        return

    output.mkdir(parents=True, exist_ok=True)
    print(
        f"Streaming {len(parts)} split parts ({sum(p.stat().st_size for p in parts) / 2**30:.2f} GiB) "
        f"into {output.parent}",
        flush=True,
    )
    cat = subprocess.Popen(["cat", *map(str, parts)], stdout=subprocess.PIPE)
    try:
        assert cat.stdout is not None
        tar = subprocess.run(
            ["tar", "-xf", "-", "--no-same-owner", "-C", str(output.parent)],
            stdin=cat.stdout,
            check=False,
        )
    finally:
        if cat.stdout is not None:
            cat.stdout.close()
    cat_code = cat.wait()
    if cat_code != 0 or tar.returncode != 0:
        raise RuntimeError(f"OVO-Bench extraction failed: cat={cat_code}, tar={tar.returncode}")
    marker.write_text("official split archive aa..ao extracted successfully\n")


def expected_clips(annotations: list[dict]) -> list[str]:
    paths: list[str] = []
    for item in annotations:
        if item["task"] in FORWARD_TASKS:
            paths.extend(f"{item['id']}_{index}.mp4" for index in range(len(item["test_info"])))
        else:
            paths.append(f"{item['id']}.mp4")
    return paths


def replace_symlink(link: Path, target: Path) -> None:
    if link.is_symlink():
        if link.resolve() == target.resolve():
            return
        link.unlink()
    elif link.exists():
        raise FileExistsError(f"Refusing to replace non-symlink evaluator path: {link}")
    link.symlink_to(target, target_is_directory=True)
    print(f"Linked {link} -> {target}", flush=True)


def probe_video(path: Path) -> dict:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,width,height:format=duration",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return {"ok": False, "error": result.stderr.strip()}
    info = json.loads(result.stdout)
    return {"ok": bool(info.get("streams")), "info": info}


def verify(root: Path, repo: Path, clip_root: Path, probe_count: int) -> Path:
    annotation_path = repo / "data" / "ovo_bench_new.json"
    if not annotation_path.is_file():
        raise FileNotFoundError(f"Official annotation file is missing: {annotation_path}")
    annotations = json.loads(annotation_path.read_text())
    clips = expected_clips(annotations)
    duplicate_names = [name for name, count in Counter(clips).items() if count > 1]
    missing = [name for name in clips if not (clip_root / name).is_file()]
    empty = [name for name in clips if (clip_root / name).is_file() and (clip_root / name).stat().st_size == 0]

    sample_count = min(max(probe_count, 0), len(clips))
    if sample_count:
        indices = sorted({round(index * (len(clips) - 1) / max(sample_count - 1, 1)) for index in range(sample_count)})
        probes = {clips[index]: probe_video(clip_root / clips[index]) for index in indices if clips[index] not in missing}
    else:
        probes = {}

    mode_tasks = {
        "backward": {"EPM", "ASI", "HLD"},
        "realtime": {"STU", "OJR", "ATR", "ACR", "OCR", "FPD"},
        "forward": FORWARD_TASKS,
    }
    mode_counts = {}
    for mode, tasks in mode_tasks.items():
        rows = [item for item in annotations if item["task"] in tasks]
        mode_counts[mode] = {
            "annotations": len(rows),
            "evaluations": sum(len(item.get("test_info", [None])) for item in rows),
        }

    report = {
        "dataset_root": str(root),
        "official_repo": str(repo),
        "annotation_file": str(annotation_path),
        "annotation_count": len(annotations),
        "source_video_count": len({item["video"] for item in annotations}),
        "expected_clip_count": len(clips),
        "unique_expected_clip_count": len(set(clips)),
        "mode_counts": mode_counts,
        "task_counts": dict(sorted(Counter(item["task"] for item in annotations).items())),
        "duplicate_expected_names": duplicate_names,
        "missing_clip_count": len(missing),
        "missing_clips": missing,
        "empty_clip_count": len(empty),
        "empty_clips": empty,
        "ffprobe_samples": probes,
    }
    report_path = root / "data" / "ovo_bench_setup_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")

    bad_probes = [name for name, value in probes.items() if not value["ok"]]
    if duplicate_names or missing or empty or bad_probes:
        raise RuntimeError(
            "OVO-Bench verification failed: "
            f"duplicates={len(duplicate_names)}, missing={len(missing)}, empty={len(empty)}, "
            f"bad_ffprobe={len(bad_probes)}. See {report_path}"
        )

    evaluator_clip_path = repo / "data" / "chunked_videos"
    if evaluator_clip_path.absolute() != clip_root.absolute():
        replace_symlink(evaluator_clip_path, clip_root)
    print(
        f"Verified {len(clips)} evaluation clips for {len(annotations)} annotations; report: {report_path}",
        flush=True,
    )
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--repo", type=Path, default=None)
    parser.add_argument("--archive-dir", type=Path, default=None)
    parser.add_argument("--clip-dir", type=Path, default=None)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument(
        "--probe-count",
        type=int,
        default=24,
        help="Number of evenly spaced clips to decode-check with ffprobe after filename verification.",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    repo = (args.repo or root).resolve()
    archive_dir = (args.archive_dir or root / "data" / "huggingface_snapshot").resolve()
    clip_root = (args.clip_dir or root / "data" / "chunked_videos").resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    if not repo.is_dir():
        raise FileNotFoundError(repo)

    parts = archive_parts(archive_dir)
    if not args.verify_only:
        extract_split_tar(parts, clip_root)
    verify(root, repo, clip_root, args.probe_count)


if __name__ == "__main__":
    main()
