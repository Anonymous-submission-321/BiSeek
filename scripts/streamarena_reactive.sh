#!/usr/bin/env bash
# StreamArena HR or RTP. Usage: TRACK=HR scripts/streamarena_reactive.sh (or TRACK=RTP).
# Main configuration: retrieve-first, 1 fps; HR keeps an unbounded vector memory,
# RTP a 16-vector one.
source "$(dirname "$0")/common.sh"
TRACK="${TRACK:-HR}"
track="$(echo "$TRACK" | tr '[:upper:]' '[:lower:]')"
CONFIG="${CONFIG:-configs/streamarena_${track}.json}"
OUT="${OUT:-outputs/StreamArena/${track}/main}"
launch biseek/eval_streamarena_reactive.py --track "$TRACK" --mode full_embedding --config "$CONFIG" --output-dir "$OUT" "$@"
