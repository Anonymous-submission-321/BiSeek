#!/usr/bin/env python3
"""Install, extract, normalize, and verify the official StreamingBench data.

The Hugging Face snapshot is distributed as 21 ZIP archives plus CSV files.
The official GitHub preprocessing script destructively moves videos out of the
extracted tree.  This setup keeps the archives immutable, extracts each video
exactly once into ``data/videos``, and exposes that directory to the official
repository through ``src/data/videos``.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import re
import shutil
import subprocess
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ARCHIVES: dict[str, tuple[str, str, int]] = {
    "Anomaly Context Understanding.zip": ("omni", "Anomaly_Context_Understanding", 50),
    "Emotion Recognition.zip": ("omni", "Emotion_Recognition", 50),
    "Misleading Context Understanding.zip": ("omni", "Misleading_Context_Understanding", 50),
    "Multimodal Alignment.zip": ("omni", "Multimodal_Alignment", 50),
    "Proactive Output_1-25.zip": ("proactive", "proactive", 25),
    "Proactive Output_26-50.zip": ("proactive", "proactive", 25),
    "Real-Time Visual Understanding_1-50.zip": ("real", "real", 50),
    "Real-Time Visual Understanding_51-100.zip": ("real", "real", 50),
    "Real-Time Visual Understanding_101-150.zip": ("real", "real", 50),
    "Real-Time Visual Understanding_151-200.zip": ("real", "real", 50),
    "Real-Time Visual Understanding_201-250.zip": ("real", "real", 50),
    "Real-Time Visual Understanding_251-300.zip": ("real", "real", 50),
    "Real-Time Visual Understanding_301-350.zip": ("real", "real", 50),
    "Real-Time Visual Understanding_351-400.zip": ("real", "real", 50),
    "Real-Time Visual Understanding_401-450.zip": ("real", "real", 50),
    "Real-Time Visual Understanding_451-500.zip": ("real", "real", 50),
    "Scene Understanding_1-25.zip": ("omni", "Scene_Understanding_1-25", 25),
    "Scene Understanding_26-50.zip": ("omni", "Scene_Understanding_26-50", 25),
    "Sequential Question Answering_1-25.zip": ("sqa", "sqa", 25),
    "Sequential Question Answering_26-50.zip": ("sqa", "sqa", 25),
    "Source Discrimination.zip": ("omni", "Source_Discrimination", 50),
}

CSV_SPECS = {
    "Real_Time_Visual_Understanding.csv": "real_time_visual_understanding",
    "Omni_Source_Understanding.csv": "omni_source_understanding",
    "Contextual_Understanding.csv": "contextual_understanding",
    "Sequential_Question_Answering.csv": "contextual_understanding",
}

OFFICIAL_STREAM_SPECS = {
    "questions_real_stream.json": "real_time_visual_understanding",
    "questions_omni_stream.json": "task_dependent",
    "questions_sqa_stream.json": "contextual_understanding",
}

OFFICIAL_TASK_NAMES = {
    "Object Recognition": "Object Perception",
    "Attribute Recognition": "Attribute Perception",
    "Action Recognition": "Action Perception",
    "Misleading Context Understanding": "Misleading Context Recognition",
}

# The released CSV contains three unambiguous timestamp formatting mistakes.
# The corrected values agree with the time windows in the official JSON files.
REACTIVE_TIMESTAMP_CORRECTIONS: dict[str, tuple[str, str]] = {
    "Real-Time Visual Understanding_sample_242_1": (
        "00:00:24",
        "released CSV says 00:24:00; official JSON says 00:24 and the event window is 0:00-0:30",
    ),
    "Anomaly Context Understanding_sample_32_1": (
        "00:00:21",
        "released CSV says 00:21:00; official event window is 0:00-1:00",
    ),
    "Anomaly Context Understanding_sample_32_5": (
        "00:05:11",
        "released CSV says 00:11:00; official event window is 5:00-6:10",
    ),
}

# The executable stream JSON fixes the CSV typo for sample 242, but retains
# three other unambiguous formatting errors.  These values are cross-checked
# against the released CSV and the JSON event windows.
OFFICIAL_STREAM_TIMESTAMP_CORRECTIONS: dict[str, tuple[str, str]] = {
    "Real-Time Visual Understanding_sample_132_5": (
        "00:12:30",
        "official stream JSON says 00:12450; released CSV says 00:12:30 and the event window is 0:12-0:13",
    ),
    "Anomaly Context Understanding_sample_32_1": (
        "00:00:21",
        "official stream JSON says 00:21:00; the event window is 0:00-0:01",
    ),
    "Anomaly Context Understanding_sample_32_5": (
        "00:05:11",
        "official stream JSON says 05:11:00; the event window is 0:05-0:06:10",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root", default="data/StreamingBench"
    )
    parser.add_argument(
        "--official-source",
        default=None,
        help="A checked-out THUNLP-MT/StreamingBench repository to install at dataset-root.",
    )
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--verify-crc", action="store_true")
    parser.add_argument("--overwrite-videos", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument(
        "--ffprobe-samples",
        type=int,
        default=0,
        help="0 probes all videos; a positive number probes evenly spaced samples.",
    )
    return parser.parse_args()


def install_official_repo(root: Path, official_source: Path) -> Path:
    if not (official_source / ".git").is_dir():
        raise FileNotFoundError(f"Not a Git checkout: {official_source}")
    snapshot = root / "data" / "huggingface_snapshot"
    snapshot.mkdir(parents=True, exist_ok=True)
    if not (root / ".git").is_dir():
        for entry in list(root.iterdir()):
            if entry.name == "data":
                continue
            destination = snapshot / entry.name
            if destination.exists():
                raise FileExistsError(destination)
            shutil.move(str(entry), str(destination))
        for entry in official_source.iterdir():
            destination = root / entry.name
            if entry.is_dir():
                shutil.copytree(entry, destination, symlinks=True)
            else:
                shutil.copy2(entry, destination)
    return snapshot


def locate_snapshot(root: Path) -> Path:
    organized = root / "data" / "huggingface_snapshot"
    if organized.is_dir():
        return organized
    return root


def archive_video_members(path: Path) -> list[zipfile.ZipInfo]:
    with zipfile.ZipFile(path) as archive:
        return [
            item
            for item in archive.infolist()
            if not item.is_dir()
            and item.filename.lower().endswith(".mp4")
            and "__macosx" not in item.filename.lower()
            and "/._" not in item.filename
        ]


def sample_number(member_name: str) -> int:
    match = re.search(r"(?:^|/)sample_(\d+)(?:/|$)", member_name)
    if not match:
        raise ValueError(f"Cannot identify sample number in {member_name!r}")
    return int(match.group(1))


def validate_archives(snapshot: Path, verify_crc: bool) -> dict[str, Any]:
    missing: list[str] = []
    bad_crc: list[dict[str, str]] = []
    rows: list[dict[str, Any]] = []
    for name, (group, suffix, expected_videos) in ARCHIVES.items():
        path = snapshot / name
        if not path.is_file():
            missing.append(name)
            continue
        members = archive_video_members(path)
        if len(members) != expected_videos:
            raise RuntimeError(
                f"{name}: expected {expected_videos} videos, found {len(members)}"
            )
        crc_result = "not_requested"
        if verify_crc:
            with zipfile.ZipFile(path) as archive:
                bad_member = archive.testzip()
            crc_result = "passed" if bad_member is None else f"failed:{bad_member}"
            if bad_member is not None:
                bad_crc.append({"archive": name, "member": bad_member})
        rows.append(
            {
                "archive": name,
                "bytes": path.stat().st_size,
                "group": group,
                "suffix": suffix,
                "video_members": len(members),
                "crc": crc_result,
            }
        )
    if missing or bad_crc:
        raise RuntimeError(json.dumps({"missing": missing, "bad_crc": bad_crc}, indent=2))
    return {"archives": rows, "archive_count": len(rows), "bad_crc": bad_crc}


def extract_videos(
    snapshot: Path, videos_dir: Path, overwrite: bool
) -> dict[str, Any]:
    videos_dir.mkdir(parents=True, exist_ok=True)
    extracted = 0
    reused = 0
    destinations: dict[str, str] = {}
    for archive_name, (_, suffix, _) in ARCHIVES.items():
        path = snapshot / archive_name
        with zipfile.ZipFile(path) as archive:
            members = [
                item
                for item in archive.infolist()
                if not item.is_dir()
                and item.filename.lower().endswith(".mp4")
                and "__macosx" not in item.filename.lower()
                and "/._" not in item.filename
            ]
            for member in members:
                number = sample_number(member.filename)
                filename = f"sample_{number}_{suffix}.mp4"
                destination = videos_dir / filename
                previous = destinations.get(filename)
                if previous is not None:
                    raise RuntimeError(
                        f"Duplicate normalized video {filename}: {previous}, {archive_name}"
                    )
                destinations[filename] = archive_name
                if (
                    destination.is_file()
                    and destination.stat().st_size == member.file_size
                    and not overwrite
                ):
                    reused += 1
                    continue
                temporary = destination.with_suffix(
                    destination.suffix + f".partial-{os.getpid()}"
                )
                try:
                    with archive.open(member) as source, temporary.open("wb") as target:
                        shutil.copyfileobj(source, target, length=16 * 1024 * 1024)
                    if temporary.stat().st_size != member.file_size:
                        raise IOError(
                            f"Size mismatch extracting {archive_name}:{member.filename}"
                        )
                    os.replace(temporary, destination)
                    extracted += 1
                finally:
                    if temporary.exists():
                        temporary.unlink()
        print(f"Prepared {archive_name}", flush=True)
    if len(destinations) != 900:
        raise RuntimeError(f"Expected 900 normalized videos, got {len(destinations)}")
    return {"expected_videos": len(destinations), "extracted": extracted, "reused": reused}


def ensure_official_video_link(root: Path, videos_dir: Path) -> None:
    link = root / "src" / "data" / "videos"
    link.parent.mkdir(parents=True, exist_ok=True)
    expected = os.path.relpath(videos_dir, link.parent)
    if link.is_symlink():
        if os.readlink(link) != expected:
            raise RuntimeError(f"Unexpected symlink target for {link}: {os.readlink(link)}")
        return
    if link.exists():
        raise FileExistsError(f"Refusing to replace existing {link}")
    link.symlink_to(expected, target_is_directory=True)


def timestamp_seconds(value: str) -> float:
    parts = [float(part) for part in str(value).strip().split(":")]
    if len(parts) == 2:
        return parts[0] * 60.0 + parts[1]
    if len(parts) != 3:
        raise ValueError(f"Invalid timestamp: {value!r}")
    return parts[0] * 3600.0 + parts[1] * 60.0 + parts[2]


def parse_options(value: str) -> list[str]:
    parsed = ast.literal_eval(value)
    if not isinstance(parsed, (list, tuple)) or len(parsed) < 4:
        raise ValueError(f"Expected at least four options, got: {value!r}")
    # Eleven released Scene Understanding rows contain 5--7 entries.  The
    # official four-placeholder prompt silently consumes only the first four,
    # so preserve that behavior and expose the original count separately.
    return [str(item).strip() for item in parsed[:4]]


def parse_question_id(question_id: str) -> tuple[int, int]:
    match = re.search(r"_sample_(\d+)_(\d+)$", question_id)
    if not match:
        raise ValueError(f"Unrecognized question_id: {question_id!r}")
    return int(match.group(1)), int(match.group(2))


def video_name_for_row(csv_name: str, row: dict[str, str]) -> str:
    if csv_name == "Proactive_Output_50.csv":
        raise ValueError(
            "Proactive_Output_50.csv row IDs are selection indices, not video "
            "sample IDs. Use the row-aligned questions_proactive_50.json map."
        )
    else:
        sample, _ = parse_question_id(row["question_id"])
    if csv_name == "Real_Time_Visual_Understanding.csv":
        suffix = "real"
    elif csv_name == "Sequential_Question_Answering.csv":
        suffix = "sqa"
    elif csv_name == "Contextual_Understanding.csv":
        if row["task_type"] == "Misleading Context Recognition":
            suffix = "Misleading_Context_Understanding"
        elif row["task_type"] == "Anomaly Context Understanding":
            suffix = "Anomaly_Context_Understanding"
        else:
            raise ValueError(f"Unknown contextual task: {row['task_type']}")
    elif csv_name == "Omni_Source_Understanding.csv":
        task = row["task_type"]
        if task == "Scene Understanding":
            suffix = (
                "Scene_Understanding_1-25" if sample <= 25 else "Scene_Understanding_26-50"
            )
        else:
            suffix = task.replace(" ", "_")
    elif csv_name.startswith("Proactive_Output"):
        suffix = "proactive"
    else:
        raise ValueError(csv_name)
    return f"sample_{sample}_{suffix}.mp4"


def normalize_reactive_annotations(
    csv_dir: Path, videos_dir: Path
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for csv_name, category in CSV_SPECS.items():
        path = csv_dir / csv_name
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open(newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            sample, sequence = parse_question_id(row["question_id"])
            video_name = video_name_for_row(csv_name, row)
            source_options = ast.literal_eval(row["options"])
            original_time_stamp = row["time_stamp"].strip()
            corrected = REACTIVE_TIMESTAMP_CORRECTIONS.get(row["question_id"])
            effective_time_stamp = corrected[0] if corrected else original_time_stamp
            records.append(
                {
                    "_key": row["question_id"],
                    "question_id": row["question_id"],
                    "category": category,
                    "task_type": row["task_type"],
                    "sample_id": sample,
                    "sequence_index": sequence,
                    "video": video_name,
                    "video_path": str((videos_dir / video_name).resolve()),
                    "question": row["question"].strip(),
                    "time_stamp": effective_time_stamp,
                    "original_time_stamp": original_time_stamp,
                    "question_time_seconds": timestamp_seconds(effective_time_stamp),
                    "timestamp_correction": None if corrected is None else corrected[1],
                    "answer": row["answer"].strip().upper(),
                    "options": parse_options(row["options"]),
                    "source_option_count": len(source_options),
                    "frames_required": row.get("frames_required", "").strip(),
                    "temporal_clue_type": row.get("temporal_clue_type", "").strip(),
                    "requires_audio": category == "omni_source_understanding",
                    "conversation_history": [],
                    "annotation_csv": csv_name,
                }
            )
    histories: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in sorted(
        records,
        key=lambda value: (
            value["video"],
            value["question_time_seconds"],
            value["sequence_index"],
        ),
    ):
        if item["task_type"] == "Sequential Question Answering":
            item["conversation_history"] = list(histories[item["video"]])
            histories[item["video"]].append(
                {
                    "time_stamp": item["time_stamp"],
                    "question": item["question"],
                    "options": item["options"],
                    "answer": item["answer"],
                }
            )
    records.sort(key=lambda value: value["_key"])
    if len(records) != 4250:
        raise RuntimeError(f"Expected 4250 reactive questions, got {len(records)}")
    keys = [item["_key"] for item in records]
    if len(keys) != len(set(keys)):
        raise RuntimeError("Duplicate reactive question IDs")
    return records


def official_category(json_name: str, task_type: str) -> str:
    category = OFFICIAL_STREAM_SPECS[json_name]
    if category != "task_dependent":
        return category
    if task_type in {
        "Anomaly Context Understanding",
        "Misleading Context Understanding",
    }:
        return "contextual_understanding"
    return "omni_source_understanding"


def official_question_prefix(json_name: str, task_type: str) -> str:
    if json_name == "questions_real_stream.json":
        return "Real-Time Visual Understanding"
    if json_name == "questions_sqa_stream.json":
        return "Sequential Question Answering"
    return task_type


def normalize_official_stream_annotations(
    official_data_dir: Path, videos_dir: Path
) -> list[dict[str, Any]]:
    """Normalize the exact question split released for streaming inference.

    The CSV release contains two real-time sources that the official streaming
    JSON intentionally omits.  In particular, sample 332 has a truncated H.264
    payload in the released ZIP.  Keep the CSV normalization as an audit
    artifact, but make these JSON files the default executable benchmark split.
    """

    records: list[dict[str, Any]] = []
    for json_name in OFFICIAL_STREAM_SPECS:
        path = official_data_dir / json_name
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise TypeError(f"Expected a list in {path}")
        for video_record in payload:
            video_name = Path(str(video_record["video_path"])).name
            sample_match = re.search(r"^sample_(\d+)_", video_name)
            if not sample_match:
                raise ValueError(f"Unrecognized official video name: {video_name}")
            sample = int(sample_match.group(1))
            history: list[dict[str, Any]] = []
            for sequence, question in enumerate(video_record["questions"], start=1):
                raw_task = str(question["task_type"]).strip()
                task = OFFICIAL_TASK_NAMES.get(raw_task, raw_task)
                prefix = official_question_prefix(json_name, raw_task)
                question_id = f"{prefix}_sample_{sample}_{sequence}"
                source_options = [str(value).strip() for value in question["options"]]
                # The released Scene Understanding JSON contains rows with
                # 5--7 choices, while the official prompt exposes only A--D.
                # Match that executable behavior and retain the source count
                # so the truncation remains auditable.
                options = source_options[:4]
                answer = str(question["answer"]).strip().upper()
                if len(options) != 4 or answer not in {"A", "B", "C", "D"}:
                    raise ValueError(
                        f"Invalid official MCQ {question_id}: "
                        f"{len(options)} options, answer={answer!r}"
                    )
                original_time_stamp = str(question["time_stamp"]).strip()
                corrected = OFFICIAL_STREAM_TIMESTAMP_CORRECTIONS.get(question_id)
                time_stamp = corrected[0] if corrected else original_time_stamp
                conversation_history = list(history) if json_name == "questions_sqa_stream.json" else []
                category = official_category(json_name, raw_task)
                records.append(
                    {
                        "_key": question_id,
                        "question_id": question_id,
                        "category": category,
                        "task_type": task,
                        "official_task_type": raw_task,
                        "sample_id": sample,
                        "sequence_index": sequence,
                        "video": video_name,
                        "video_path": str((videos_dir / video_name).resolve()),
                        "question": str(question["question"]).strip(),
                        "time_stamp": time_stamp,
                        "original_time_stamp": original_time_stamp,
                        "question_time_seconds": timestamp_seconds(time_stamp),
                        "timestamp_correction": None if corrected is None else corrected[1],
                        "answer": answer,
                        "options": options,
                        "source_option_count": len(source_options),
                        "frames_required": str(question.get("time", "")).strip(),
                        "temporal_clue_type": str(
                            question.get("required_ability", "")
                        ).strip(),
                        "requires_audio": category == "omni_source_understanding",
                        "conversation_history": conversation_history,
                        "annotation_csv": json_name,
                        "annotation_source": "official_stream_json",
                    }
                )
                if json_name == "questions_sqa_stream.json":
                    history.append(
                        {
                            "time_stamp": time_stamp,
                            "question": str(question["question"]).strip(),
                            "options": options,
                            "answer": answer,
                        }
                    )

    records.sort(key=lambda value: value["_key"])
    keys = [item["_key"] for item in records]
    videos = {item["video"] for item in records}
    if len(records) != 4245 or len(videos) != 848:
        raise RuntimeError(
            f"Expected official stream split 4245 questions/848 videos, got "
            f"{len(records)} questions/{len(videos)} videos"
        )
    if len(keys) != len(set(keys)):
        raise RuntimeError("Duplicate official stream question IDs")
    forbidden = {"sample_316_real.mp4", "sample_332_real.mp4"} & videos
    if forbidden:
        raise RuntimeError(f"Official omitted videos unexpectedly present: {forbidden}")
    return records


def normalize_proactive_annotations(
    csv_dir: Path,
    videos_dir: Path,
    csv_name: str = "Proactive_Output.csv",
    expected_count: int = 250,
    official_data_dir: Path | None = None,
) -> list[dict[str, Any]]:
    path = csv_dir / csv_name
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    official_video_names: list[str] | None = None
    if csv_name == "Proactive_Output_50.csv":
        if official_data_dir is None:
            raise ValueError(
                "official_data_dir is required for Proactive_Output_50.csv"
            )
        official_path = official_data_dir / "questions_proactive_50.json"
        source_entries = json.loads(official_path.read_text(encoding="utf-8"))
        if len(source_entries) != len(rows):
            raise RuntimeError(
                f"{official_path}: expected {len(rows)} row-aligned entries, "
                f"found {len(source_entries)}"
            )
        official_video_names = []
        for index, (row, entry) in enumerate(
            zip(rows, source_entries), start=1
        ):
            questions = entry.get("questions", [])
            if not questions or (
                str(questions[0].get("question", "")).strip()
                != row["question"].strip()
            ):
                raise RuntimeError(
                    "Proactive-50 CSV/JSON row alignment failed at row "
                    f"{index}: {row['question_id']}"
                )
            video_name = Path(entry["video_path"]).name
            if not re.fullmatch(r"sample_\d+_proactive\.mp4", video_name):
                raise RuntimeError(
                    f"Unexpected proactive video path: {entry['video_path']}"
                )
            official_video_names.append(video_name)

    records = []
    for index, row in enumerate(rows):
        video_name = (
            official_video_names[index]
            if official_video_names is not None
            else video_name_for_row(path.name, row)
        )
        records.append(
            {
                "_key": row["question_id"],
                "question_id": row["question_id"],
                "category": "proactive_output",
                "task_type": row["task_type"],
                "video": video_name,
                "video_path": str((videos_dir / video_name).resolve()),
                "question": row["question"].strip(),
                "query_time_seconds": timestamp_seconds(row["time_stamp"]),
                "ground_truth_time_seconds": timestamp_seconds(
                    row["ground_truth_time_stamp"]
                ),
                "ground_truth_output": row["ground_truth_output"].strip(),
                "frames_required": row.get("frames_required", "").strip(),
                "temporal_clue_type": row.get("temporal_clue_type", "").strip(),
                "annotation_csv": path.name,
            }
        )
    if len(records) != expected_count:
        raise RuntimeError(
            f"Expected {expected_count} proactive questions in {csv_name}, got {len(records)}"
        )
    return records


def ffprobe(path: Path) -> dict[str, Any]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration:stream=codec_type,width,height,r_frame_rate",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(command, check=True, text=True, capture_output=True)
    payload = json.loads(result.stdout)
    return {
        "duration_seconds": float(payload["format"]["duration"]),
        "streams": payload.get("streams", []),
    }


def verify_videos(
    videos_dir: Path,
    reactive: list[dict[str, Any]],
    proactive: list[dict[str, Any]],
    ffprobe_samples: int,
) -> dict[str, Any]:
    video_paths = sorted(videos_dir.glob("*.mp4"))
    expected_names = {item["video"] for item in reactive + proactive}
    actual_names = {path.name for path in video_paths}
    missing = sorted(expected_names - actual_names)
    extras = sorted(actual_names - expected_names)
    empty = [path.name for path in video_paths if path.stat().st_size == 0]
    if missing or empty or len(video_paths) != 900:
        raise RuntimeError(
            json.dumps(
                {
                    "num_videos": len(video_paths),
                    "missing": missing,
                    "extras": extras,
                    "empty": empty,
                },
                indent=2,
            )
        )

    reactive_by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    proactive_by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in reactive:
        reactive_by_video[item["video"]].append(item)
    for item in proactive:
        proactive_by_video[item["video"]].append(item)

    if ffprobe_samples > 0 and ffprobe_samples < len(video_paths):
        count = ffprobe_samples
        indices = sorted(
            {
                round(index * (len(video_paths) - 1) / max(1, count - 1))
                for index in range(count)
            }
        )
        probe_paths = [video_paths[index] for index in indices]
    else:
        probe_paths = video_paths

    failures: list[dict[str, str]] = []
    too_short: list[dict[str, Any]] = []
    released_reactive_timestamp_clamps: list[dict[str, Any]] = []
    audio_streams = 0
    videos_without_audio: list[str] = []
    for index, path in enumerate(probe_paths, start=1):
        try:
            metadata = ffprobe(path)
            if any(stream.get("codec_type") == "audio" for stream in metadata["streams"]):
                audio_streams += 1
            else:
                videos_without_audio.append(path.name)
            duration = metadata["duration_seconds"]
            for item in reactive_by_video.get(path.name, []):
                required = float(item["question_time_seconds"])
                if duration + 1e-3 < required:
                    effective = max(0.0, duration - 0.1)
                    released_reactive_timestamp_clamps.append(
                        {
                            "question_id": item["question_id"],
                            "video": path.name,
                            "released_question_time_seconds": required,
                            "video_duration_seconds": duration,
                            "effective_question_time_seconds": effective,
                            "reason": "released annotation is later than the end of the released video",
                        }
                    )
                    item["released_question_time_seconds"] = required
                    item["question_time_seconds"] = effective
                    item["timestamp_clamped_to_video_end"] = True
            for item in proactive_by_video.get(path.name, []):
                item["video_duration_seconds"] = duration
                required = float(item["ground_truth_time_seconds"]) + 4.0
                if duration + 1.0 < required:
                    too_short.append(
                        {
                            "question_id": item["question_id"],
                            "video": path.name,
                            "duration_seconds": duration,
                            "required_seconds": required,
                        }
                    )
        except (subprocess.CalledProcessError, KeyError, ValueError, json.JSONDecodeError) as error:
            failures.append({"video": path.name, "error": str(error)})
        if index % 100 == 0 or index == len(probe_paths):
            print(f"ffprobe {index}/{len(probe_paths)}", flush=True)
    if failures or too_short:
        raise RuntimeError(
            json.dumps({"ffprobe_failures": failures, "too_short": too_short}, indent=2)
        )
    return {
        "num_videos": len(video_paths),
        "unused_extracted_videos": extras,
        "total_video_bytes": sum(path.stat().st_size for path in video_paths),
        "ffprobe_checked": len(probe_paths),
        "ffprobe_failures": failures,
        "videos_with_audio_among_checked": audio_streams,
        "videos_without_audio_among_checked": videos_without_audio,
        "question_or_poll_after_video": too_short,
        "released_reactive_timestamp_clamps": released_reactive_timestamp_clamps,
    }


def git_commit(root: Path) -> str | None:
    if not (root / ".git").is_dir():
        return None
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip()


def main() -> None:
    args = parse_args()
    root = Path(args.dataset_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.official_source:
        snapshot = install_official_repo(root, Path(args.official_source).resolve())
    else:
        snapshot = locate_snapshot(root)
    archive_report = validate_archives(snapshot, args.verify_crc)

    videos_dir = root / "data" / "videos"
    extraction_report: dict[str, Any] = {"requested": False}
    if args.extract:
        extraction_report = {
            "requested": True,
            **extract_videos(snapshot, videos_dir, args.overwrite_videos),
        }
    if not videos_dir.is_dir():
        raise FileNotFoundError(
            f"{videos_dir} does not exist; run with --extract before --verify-only"
        )
    ensure_official_video_link(root, videos_dir)

    csv_dir = snapshot / "StreamingBench"
    annotations_dir = root / "data" / "annotations"
    annotations_dir.mkdir(parents=True, exist_ok=True)
    reactive_csv = normalize_reactive_annotations(csv_dir, videos_dir)
    reactive_official = normalize_official_stream_annotations(root / "src" / "data", videos_dir)
    proactive = normalize_proactive_annotations(csv_dir, videos_dir)
    proactive_official_50 = normalize_proactive_annotations(
        csv_dir,
        videos_dir,
        "Proactive_Output_50.csv",
        50,
        official_data_dir=root / "src" / "data",
    )
    video_report = verify_videos(
        videos_dir,
        reactive_official,
        proactive + proactive_official_50,
        int(args.ffprobe_samples),
    )
    # Write annotations after verification because the verifier records and
    # causally clamps any released query timestamp that exceeds its
    # corresponding released video duration.
    (annotations_dir / "streamingbench_reactive_csv_full.json").write_text(
        json.dumps(reactive_csv, indent=2, ensure_ascii=False) + "\n"
    )
    (annotations_dir / "streamingbench_reactive_official_stream.json").write_text(
        json.dumps(reactive_official, indent=2, ensure_ascii=False) + "\n"
    )
    (annotations_dir / "streamingbench_proactive.json").write_text(
        json.dumps(proactive, indent=2, ensure_ascii=False) + "\n"
    )
    (annotations_dir / "streamingbench_proactive_official_50.json").write_text(
        json.dumps(proactive_official_50, indent=2, ensure_ascii=False) + "\n"
    )

    report = {
        "created_at_unix": time.time(),
        "dataset_root": str(root),
        "official_commit": git_commit(root),
        "huggingface_snapshot": str(snapshot),
        **archive_report,
        "extraction": extraction_report,
        "annotations": {
            "reactive_questions": len(reactive_official),
            "reactive_questions_official_stream": len(reactive_official),
            "reactive_questions_csv_full": len(reactive_csv),
            "proactive_questions": len(proactive),
            "proactive_questions_full": len(proactive),
            "proactive_questions_legacy_50_subset": len(proactive_official_50),
            "total_questions": len(reactive_official) + len(proactive),
            "reactive_unique_videos": len({item["video"] for item in reactive_official}),
            "reactive_unique_videos_csv_full": len({item["video"] for item in reactive_csv}),
            "all_unique_videos": len({item["video"] for item in reactive_official + proactive}),
            "category_counts": dict(Counter(item["category"] for item in reactive_official)),
            "task_counts": dict(Counter(item["task_type"] for item in reactive_official)),
            "audio_required_questions": sum(item["requires_audio"] for item in reactive_official),
            "option_rows_truncated_to_official_four": sum(
                item["source_option_count"] != 4 for item in reactive_official
            ),
            "known_csv_timestamp_corrections": sum(
                item["timestamp_correction"] is not None for item in reactive_csv
            ),
            "official_stream_timestamp_corrections": sum(
                item["timestamp_correction"] is not None for item in reactive_official
            ),
            "released_video_end_clamps": len(
                video_report["released_reactive_timestamp_clamps"]
            ),
        },
        "videos": video_report,
        "protocol": {
            "default_runner_scope": "4245 immediate/reactive questions from the three official *_stream.json files",
            "proactive_scope": "The full 250-question Proactive_Output.csv is the paper-comparable PO evaluation used by later StreamingBench results such as ViSpeak 50.8. The separately released Proactive_Output_50.csv is retained as a legacy 50-question subset. Wait/respond is not scored by the reactive runner.",
            "normalization_source": "official GitHub questions_real_stream.json, questions_omni_stream.json, and questions_sqa_stream.json",
            "default_annotation": "data/annotations/streamingbench_reactive_official_stream.json",
            "csv_audit_annotation": "data/annotations/streamingbench_reactive_csv_full.json",
            "official_json_note": "The default split follows the executable official stream JSON exactly. The CSV audit split retains all 500 real-time videos, including archive samples 316 and 332; sample 332 is not fully decodable in the released ZIP.",
        },
    }
    report_path = root / "data" / "streamingbench_setup_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Setup report: {report_path}")


if __name__ == "__main__":
    main()
