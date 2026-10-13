#!/usr/bin/env bash
# StreamArena Pro (774 questions). Main configuration: cache A, 1 fps, v7 proposals.
# One process per GPU; each takes a cost-balanced shard of the videos.
source "$(dirname "$0")/common.sh"
OUT="${OUT:-outputs/StreamArena/pro/main}"
pids=()
for rank in "${!GPU_LIST[@]}"; do
  CUDA_VISIBLE_DEVICES="${GPU_LIST[$rank]}" "$PYTHON" biseek/eval_streamarena_proactive.py \
    --config configs/streamarena_pro.json --output-dir "$OUT" \
    --rank "$rank" --world-size "$NGPU" "$@" &
  pids+=($!)
done
for pid in "${pids[@]}"; do wait "$pid"; done
