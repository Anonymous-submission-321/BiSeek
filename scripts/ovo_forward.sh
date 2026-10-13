#!/usr/bin/env bash
# OVO-Bench Forward Active Responding. SSR and CRR run the proposal-match cascade;
# REC runs the interval-delta counter. The two runs together give the Forward score.
source "$(dirname "$0")/common.sh"
OUT="${OUT:-outputs/OVO-Bench/forward}"
launch biseek/eval_ovo_forward_active.py --config configs/ovo_forward_ssr_crr.json --output-dir "$OUT/ssr_crr" "$@"
launch biseek/eval_ovo_forward_active.py --config configs/ovo_forward_rec.json --output-dir "$OUT/rec" "$@"
