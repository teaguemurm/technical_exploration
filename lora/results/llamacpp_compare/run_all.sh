#!/bin/bash
# Driver for the llama.cpp b10481 laptop comparison. Runs every step in sequence,
# one llama-server at a time, skips steps whose output is complete, resumes partial
# ones (run_llamacpp.py keeps finished rows). Safe to re-run at any time:
#   cd lora/results/llamacpp_compare && nohup bash run_all.sh >> driver.log 2>&1 &
# Progress: tail -f driver.log ; per-step server logs in raw/ and smoke/.
cd "$(dirname "$0")" || exit 1
D=$(pwd)
LOCK="$D/.driver.lock"
if ! mkdir "$LOCK" 2>/dev/null; then
  if [ -f "$LOCK/pid" ] && kill -0 "$(cat "$LOCK/pid")" 2>/dev/null; then
    echo "$(date '+%F %T') another driver is running (pid $(cat "$LOCK/pid")), exiting"; exit 0
  fi
  echo "$(date '+%F %T') stale lock, taking over"; rm -rf "$LOCK"; mkdir "$LOCK"
fi
echo $$ > "$LOCK/pid"
trap 'rm -rf "$LOCK"' EXIT
L=/Users/teaguemm/Documents/Programming/IdeaProjects/technical-exploration/lora
QG=$HOME/models/app_gguf/qwen2.5-1.5b-instruct-q4_k_m.gguf
GG=$HOME/models/app_gguf/gemma-4-E2B-it-Q3_K_M.gguf
QL=$L/out/qwen25-1.5b-constraint-s4-lora-ckpt600-f16.gguf
GF=$L/out/gemma4-e2b-constraint-s4-lora-f16.gguf
G8=$L/out/gemma4-e2b-constraint-s4-lora-ckpt800-f16.gguf
PY=python3.11
# name family gguf lora(-=none) cond limit(0=all) outdir
STEPS="
smoke_qwen_base_B qwen $QG - B 5 smoke
smoke_qwen_lora_B qwen $QG $QL B 5 smoke
smoke_gemma_base_B gemma $GG - B 5 smoke
smoke_gemma_final_B gemma $GG $GF B 5 smoke
smoke_gemma_800_B gemma $GG $G8 B 5 smoke
qwen_base_C qwen $QG - C 0 raw
qwen_lora_B qwen $QG $QL B 0 raw
gemma_base_C gemma $GG - C 0 raw
gemma_final_B gemma $GG $GF B 0 raw
qwen_lora_C qwen $QG $QL C 0 raw
qwen_lora_A qwen $QG $QL A 0 raw
gemma_final_C gemma $GG $GF C 0 raw
gemma_final_A gemma $GG $GF A 0 raw
gemma_800_B gemma $GG $G8 B 0 raw
gemma_800_C gemma $GG $G8 C 0 raw
gemma_800_A gemma $GG $G8 A 0 raw
"
echo "$(date '+%F %T') driver start pid $$"
echo "$STEPS" | while read -r name fam gguf lora cond limit outdir; do
  [ -z "$name" ] && continue
  want=$([ "$limit" = 0 ] && echo 910 || echo "$limit")
  out="$D/$outdir/out_$name.jsonl"; meta="$D/$outdir/out_$name.meta.json"
  if [ -f "$meta" ] && [ -f "$out" ] && [ "$(wc -l < "$out" | tr -d ' ')" = "$want" ]; then
    echo "$(date '+%F %T') SKIP $name (complete, $want rows)"; continue
  fi
  # never two model jobs: leftover server on our port from a killed run?
  pkill -f "llama-server .*--port 8791" 2>/dev/null && sleep 2
  echo "$(date '+%F %T') START $name"
  args=(--family "$fam" --gguf "$gguf" --cond "$cond" --name "$name" --outdir "$D/$outdir")
  [ "$lora" != "-" ] && args+=(--lora "$lora")
  [ "$limit" != 0 ] && args+=(--limit "$limit")
  t0=$(date +%s)
  if $PY run_llamacpp.py "${args[@]}" < /dev/null; then
    echo "$(date '+%F %T') DONE $name in $(( $(date +%s) - t0 ))s"
  else
    echo "$(date '+%F %T') FAIL $name (exit $?) - stopping driver"; exit 1
  fi
done || exit 1
echo "$(date '+%F %T') scoring"
$PY score.py && $PY make_summary.py
echo "$(date '+%F %T') ALL_DONE"
