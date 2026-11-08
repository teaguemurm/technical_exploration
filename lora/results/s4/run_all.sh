#!/bin/bash
set -x
cd /Users/teaguemm/Documents/Programming/IdeaProjects/technical-exploration/lora
for c in 400 600 800; do python3.11 results/s4/run_variant.py --name s4_ckpt${c}_pen --adapter results/s4/ad_ckpt$c --penalty on || exit 1; done
python3.11 results/s4/run_variant.py --name s4_ckpt1000_final_pen --adapter adapters_s4_lr1e-4 --penalty on || exit 1
echo SUBSET_DONE
python3.11 scripts/run_comparison.py --backend mlx --model Qwen/Qwen2.5-1.5B-Instruct --adapter adapters_s4_lr1e-4 --layout app --comparison data/comparison/comparison.jsonl --output results/s4_mlx_app_comparison.csv > results/s4_mlx_app_comparison.log 2>&1
echo FULL_DONE $?
