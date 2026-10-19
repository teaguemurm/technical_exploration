#!/usr/bin/env bash
# Wrap llama.cpp convert_lora_to_gguf.py.
# The converter is not vendored here; pass its path. It needs the PEFT
# directory from scripts/mlx_to_peft.py and a base directory that contains
# config.json and tokenizer.json (weights are not required).
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: convert_adapter_gguf.sh --converter PATH --peft DIR --base DIR --output FILE

  --converter   path to llama.cpp convert_lora_to_gguf.py
  --peft        PEFT adapter directory (adapter_config.json + adapter_model.safetensors)
  --base        Hugging Face config+tokenizer directory for Qwen/Qwen2.5-1.5B-Instruct
  --output      destination .gguf path
  --outtype     f16 (default), f32, bf16, or q8_0

Example:
  ./scripts/convert_adapter_gguf.sh \
    --converter ~/src/llama.cpp/convert_lora_to_gguf.py \
    --peft peft_adapter \
    --base base_hf \
    --output wisent-constraint-lora-f16.gguf
EOF
}

CONVERTER=""
PEFT=""
BASE=""
OUTPUT=""
OUTTYPE="f16"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --converter) CONVERTER="${2:-}"; shift 2 ;;
    --peft) PEFT="${2:-}"; shift 2 ;;
    --base) BASE="${2:-}"; shift 2 ;;
    --output) OUTPUT="${2:-}"; shift 2 ;;
    --outtype) OUTTYPE="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "$CONVERTER" || -z "$PEFT" || -z "$BASE" || -z "$OUTPUT" ]]; then
  usage
  exit 2
fi
if [[ ! -f "$CONVERTER" ]]; then
  echo "converter not found: $CONVERTER" >&2
  exit 2
fi
if [[ ! -f "$PEFT/adapter_config.json" || ! -f "$PEFT/adapter_model.safetensors" ]]; then
  echo "PEFT dir must contain adapter_config.json and adapter_model.safetensors: $PEFT" >&2
  exit 2
fi
if [[ ! -f "$BASE/config.json" ]]; then
  echo "base dir must contain config.json (tokenizer.json is also expected): $BASE" >&2
  exit 2
fi

python3 "$CONVERTER" "$PEFT" \
  --base "$BASE" \
  --outtype "$OUTTYPE" \
  --outfile "$OUTPUT"
