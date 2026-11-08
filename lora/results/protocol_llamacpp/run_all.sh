#!/bin/bash
# Resumable driver: nohup bash run_all.sh >> driver.log 2>&1 &
cd "$(dirname "$0")" || exit 1
D=$(pwd); LOCK="$D/.driver.lock"
if ! mkdir "$LOCK" 2>/dev/null; then
  if [ -f "$LOCK/pid" ] && kill -0 "$(cat "$LOCK/pid")" 2>/dev/null; then echo "$(date '+%F %T') driver already running"; exit 0; fi
  rm -rf "$LOCK"; mkdir "$LOCK"
fi
echo $$ > "$LOCK/pid"; trap 'rm -rf "$LOCK"' EXIT
L=/Users/teaguemm/Documents/Programming/IdeaProjects/technical-exploration/lora
QG=$HOME/models/app_gguf/qwen2.5-1.5b-instruct-q4_k_m.gguf
GG=$HOME/models/app_gguf/gemma-4-E2B-it-Q3_K_M.gguf
QL=$L/out/qwen25-1.5b-constraint-s4-lora-ckpt600-f16.gguf
GF=$L/out/gemma4-e2b-constraint-s4-lora-f16.gguf
STEPS="
qwen_base qwen $QG -
gemma_base gemma $GG -
qwen_lora qwen $QG $QL
gemma_lora gemma $GG $GF
"
echo "$(date '+%F %T') driver start pid $$"
echo "$STEPS" | while read -r name fam gguf lora; do
  [ -z "$name" ] && continue
  out="$D/raw/out_$name.jsonl"
  if [ -f "$D/raw/out_$name.meta.json" ] && [ "$(wc -l < "$out" | tr -d ' ')" = 50 ]; then echo "$(date '+%F %T') SKIP $name"; continue; fi
  pkill -f "llama-server .*--port 8792" 2>/dev/null && sleep 2
  echo "$(date '+%F %T') START $name"
  args=(--family "$fam" --gguf "$gguf" --name "$name")
  [ "$lora" != "-" ] && args+=(--lora "$lora")
  if python3.11 run_protocol.py "${args[@]}" < /dev/null; then echo "$(date '+%F %T') DONE $name"; else echo "$(date '+%F %T') FAIL $name"; fi
done
echo "$(date '+%F %T') driver end"
