#!/usr/bin/env bash
# Score StreamArena runs with a local LLM judge (gpt-oss-120b by default), using the
# benchmark's own judge prompts; Pro rows outside the [-0.5, +2.0] s window score 0
# before the judge is consulted. Dequantized to bf16 the judge needs ~240 GB of GPU
# memory, e.g. 6 x 48 GB. Usage: scripts/judge_streamarena.sh outputs/StreamArena/pro/main ...
source "$(dirname "$0")/common.sh"
records=()
for run in "$@"; do records+=("$run/rank_*.jsonl"); done
"$PYTHON" biseek/judge_streamarena_local.py --records "${records[@]}" \
  --output outputs/StreamArena/judged.jsonl --batch-size 4
