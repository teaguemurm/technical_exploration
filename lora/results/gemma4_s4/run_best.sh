#!/bin/bash
# Full 910-row runs for the selected Gemma checkpoints (800 = best on the subset, 600 = same iter as the chosen Qwen checkpoint).
# Waits for run_all.sh (pid 23255) so the two do not share the GPU. Reuses the cached Gemma base pass.
set -x
while kill -0 23255 2>/dev/null; do sleep 20; done
cd /Users/teaguemm/Documents/Programming/IdeaProjects/technical-exploration/lora
G=results/gemma4_s4
for c in 800 600; do python3.11 $G/run_full_gemma.py --base-cache $G/gemma4_base_full_outputs.jsonl --adapter /tmp/g4ad/ckpt$c --output $G/gemma4_s4_ckpt${c}_mlx_app_comparison.csv > $G/gemma4_s4_ckpt${c}_mlx_app_comparison.log 2>&1; echo FULL_${c}_DONE $?; done
