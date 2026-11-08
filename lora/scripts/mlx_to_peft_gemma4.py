#!/usr/bin/env python3
"""Convert an mlx-lm Gemma 4 LoRA adapter into a PEFT adapter directory.

This sits next to scripts/mlx_to_peft.py, which only accepts Qwen-style names.
mlx-lm loads Gemma 4 as Model.language_model (mlx_lm/models/gemma4.py), so its
LoRA keys look like

    language_model.model.layers.N.self_attn.q_proj.lora_a

Hugging Face's Gemma4ForConditionalGeneration names the same weight
`model.language_model.layers.N.self_attn.q_proj.weight`. This script writes
`base_model.model.model.language_model.layers.N....lora_A.weight`. llama.cpp's
convert_lora_to_gguf.py strips `language_model.` (ModelBase.filter_tensors) and
maps the rest to blk.N.attn_q and so on.

The factor layout and lora_alpha rule are the same as wisent_lora.peft_convert:
mlx lora_a (in, rank) -> PEFT lora_A (rank, in); mlx lora_b (rank, out) ->
PEFT lora_B (out, rank); lora_alpha = scale * rank.

Requires numpy and safetensors (see requirements-mac.txt).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wisent_lora.peft_convert import peft_config, read_mlx_lora_config  # noqa: E402

MLX_KEY = re.compile(r"^language_model\.model\.layers\.(\d+)\.(.+)\.lora_([ab])$")
FORBIDDEN = ("lm_head", "embed", "per_layer")


def mlx_key_to_peft(key: str) -> tuple[str, str]:
    match = MLX_KEY.match(key)
    if match is None:
        raise ValueError(
            f"unrecognised Gemma 4 mlx LoRA key: {key} "
            "(expected language_model.model.layers.N.<module>.lora_a/lora_b)"
        )
    layer, module, side = match.groups()
    if any(part in module for part in FORBIDDEN):
        raise ValueError(f"{key} targets an embedding, per-layer input or lm_head; leave it out")
    which = side.upper()
    return f"base_model.model.model.language_model.layers.{layer}.{module}.lora_{which}.weight", which


def convert(mlx_adapter: Path, output: Path, checkpoint: Path | None, base_model: str) -> dict:
    import numpy as np
    from safetensors.numpy import load_file, save_file

    rank, scale, dropout = read_mlx_lora_config(mlx_adapter)
    if scale == 0:
        raise ValueError("mlx scale is 0; the adapter would be a no-op")
    weights_path = checkpoint or (mlx_adapter / "adapters.safetensors")
    if not weights_path.is_file():
        raise FileNotFoundError(weights_path)
    raw = load_file(str(weights_path))
    out: dict[str, np.ndarray] = {}
    sides: dict[str, set[str]] = {}
    for key, value in raw.items():
        peft_key, which = mlx_key_to_peft(key)
        array = np.asarray(value, dtype=np.float32)
        if array.ndim != 2:
            raise ValueError(f"{key} is not 2-D: {array.shape}")
        if which == "A" and array.shape[1] != rank:
            raise ValueError(f"{key} has shape {array.shape}; expected (in_features, {rank})")
        if which == "B" and array.shape[0] != rank:
            raise ValueError(f"{key} has shape {array.shape}; expected ({rank}, out_features)")
        out[peft_key] = np.ascontiguousarray(array.T)
        sides.setdefault(peft_key.rsplit(".lora_", 1)[0], set()).add(which)
    missing = [stem for stem, seen in sides.items() if seen != {"A", "B"}]
    if missing:
        raise ValueError(f"LoRA factors are missing a side: {missing[:5]}")
    if not out:
        raise ValueError("adapter has no LoRA factors")
    output.mkdir(parents=True, exist_ok=True)
    save_file(out, str(output / "adapter_model.safetensors"))
    config = peft_config(
        rank=rank, scale=scale, peft_keys=out.keys(), base_model=base_model, dropout=dropout
    )
    (output / "adapter_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mlx-adapter", type=Path, required=True,
                        help="Directory with adapter_config.json and adapters.safetensors")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Optional NNNNNNN_adapters.safetensors to convert instead of the final file")
    parser.add_argument("--output", type=Path, required=True,
                        help="PEFT directory to write (adapter_config.json, adapter_model.safetensors)")
    parser.add_argument("--base-model", default="google/gemma-4-E2B-it",
                        help="base_model_name_or_path written into the PEFT config")
    args = parser.parse_args()
    config = convert(args.mlx_adapter, args.output, args.checkpoint, args.base_model)
    print(f"wrote {args.output} rank={config['r']} lora_alpha={config['lora_alpha']} "
          f"modules={config['target_modules']}")


if __name__ == "__main__":
    main()
