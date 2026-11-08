#!/bin/bash
# Gemma 4 E2B s4 evaluation, mirrors results/s4/run_all.sh. Checkpoint dirs /tmp/g4ad/ckptN and /tmp/q4ad/ckptN
# = symlink to the NNNNNNN_adapters.safetensors + copy of the run's adapter_config.json (deleted afterwards).
set -x
cd /Users/teaguemm/Documents/Programming/IdeaProjects/technical-exploration/lora
G=results/gemma4_s4
python3.11 $G/run_variant.py 2>/dev/null
python3.11 $G/run_variant_gemma.py --name gemma4_base_pen || exit 1
for c in 100 200 300 400 500 600 700 800 900 1000; do python3.11 $G/run_variant_gemma.py --name gemma4_s4_ckpt${c}_pen --adapter /tmp/g4ad/ckpt$c || exit 1; done
for c in 100 200 300 500 600 700 900; do python3.11 $G/run_variant_gemma.py --model Qwen/Qwen2.5-1.5B-Instruct --name qwen_s4_ckpt${c}_pen --adapter /tmp/q4ad/ckpt$c || exit 1; done
echo SUBSET_DONE
python3.11 $G/val_loss.py google/gemma-4-E2B-it adapters_gemma4_s4_lr1e-4 gemma4_s4_val_loss.json || exit 1
python3.11 $G/val_loss.py Qwen/Qwen2.5-1.5B-Instruct adapters_s4_lr1e-4 qwen_s4_val_loss_recomputed.json || exit 1
echo VAL_DONE
python3.11 $G/run_full_gemma.py --base-cache $G/gemma4_base_full_outputs.jsonl --adapter adapters_gemma4_s4_lr1e-4 --output $G/gemma4_s4_mlx_app_comparison.csv > $G/gemma4_s4_mlx_app_comparison.log 2>&1
echo FULL_FINAL_DONE $?
