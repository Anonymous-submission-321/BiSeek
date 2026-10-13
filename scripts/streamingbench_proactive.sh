#!/usr/bin/env bash
# StreamingBench Proactive Output (250 questions). Main configuration:
# cache A (vit_feature_joint_decoder), 1 fps, surge threshold 0.03, v7 proposals.
source "$(dirname "$0")/common.sh"
CONFIG="${CONFIG:-configs/streamingbench_proactive.json}"
OUT="${OUT:-outputs/StreamingBench/proactive/main}"
launch biseek/eval_streamingbench_proactive.py --config "$CONFIG" --output-dir "$OUT" "$@"
