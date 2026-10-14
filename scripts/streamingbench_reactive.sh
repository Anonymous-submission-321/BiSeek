#!/usr/bin/env bash
# StreamingBench reactive questions. Main configuration: retrieve-first, 1 fps.
source "$(dirname "$0")/common.sh"
CONFIG="${CONFIG:-configs/streamingbench_reactive.json}"
OUT="${OUT:-outputs/StreamingBench/reactive/main}"
launch biseek/eval_streamingbench_reactive.py --config "$CONFIG" --mode full_embedding --output-dir "$OUT" "$@"
