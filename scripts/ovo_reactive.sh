#!/usr/bin/env bash
# OVO-Bench Real-Time + Backward. Main configuration: retrieve-first, 1 fps,
# 1-frame chunks, 128-vector memory.
source "$(dirname "$0")/common.sh"
CONFIG="${CONFIG:-configs/ovo_reactive.json}"
OUT="${OUT:-outputs/OVO-Bench/reactive/main}"
launch biseek/eval_ovo_streaming.py --config "$CONFIG" --mode full_embedding --output-dir "$OUT" "$@"
"$PYTHON" biseek/score_ovo_bench.py --result-dir "$OUT"
