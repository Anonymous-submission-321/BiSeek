"""Robust video-duration probing for benchmark preflight checks.

The command-line FFmpeg tools are not installed on every GPU server used by
this project.  Prefer ``ffprobe`` when available, then fall back to Python
video backends that are already runtime dependencies.  Every backend also
decodes the nominal final frame so a truncated copy cannot pass based on stale
container metadata alone.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np


def probe_video_duration_seconds(video_path: str | Path) -> tuple[float, str]:
    """Return ``(duration_seconds, backend_name)`` for a decodable video."""

    path = str(video_path)
    errors: list[str] = []
    ffprobe = shutil.which("ffprobe")
    if ffprobe is not None:
        try:
            result = subprocess.run(
                [
                    ffprobe,
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    path,
                ],
                check=True,
                text=True,
                capture_output=True,
            )
            duration = float(result.stdout.strip())
            if np.isfinite(duration) and duration > 0.0:
                return duration, "ffprobe"
            errors.append(f"ffprobe returned invalid duration {duration}")
        except Exception as error:
            errors.append(f"ffprobe: {type(error).__name__}: {error}")

    try:
        from decord import VideoReader, cpu

        reader = VideoReader(path, ctx=cpu(0), num_threads=1)
        frame_count = len(reader)
        fps = float(reader.get_avg_fps())
        if frame_count <= 0 or not np.isfinite(fps) or fps <= 0.0:
            raise RuntimeError(
                f"invalid frame_count/fps: {frame_count}/{fps}"
            )
        reader.get_batch([frame_count - 1]).asnumpy()
        return frame_count / fps, "decord"
    except Exception as error:
        errors.append(f"decord: {type(error).__name__}: {error}")

    try:
        import cv2

        capture = cv2.VideoCapture(path)
        if not capture.isOpened():
            raise RuntimeError("VideoCapture could not open the file")
        try:
            frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            if frame_count <= 0 or not np.isfinite(fps) or fps <= 0.0:
                raise RuntimeError(
                    f"invalid frame_count/fps: {frame_count}/{fps}"
                )
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError(
                    f"nominal final frame {frame_count - 1} is not decodable"
                )
            return frame_count / fps, "opencv"
        finally:
            capture.release()
    except Exception as error:
        errors.append(f"opencv: {type(error).__name__}: {error}")

    raise RuntimeError(
        f"Could not determine and validate video duration for {path}. "
        "Tried ffprobe (when installed), Decord, and OpenCV. "
        + " | ".join(errors)
    )
