"""Convert an mlx-lm LoRA adapter into a PEFT directory.

mlx-lm (current main) stores LoRA factors as `lora_a` of shape
(in_features, rank) and `lora_b` of shape (rank, out_features), and a Python
float `scale` rather than PEFT's `lora_alpha`. The forward pass is

    y + scale * (x @ lora_a @ lora_b)

PEFT stores `lora_A.weight` (rank, in_features) and `lora_B.weight`
(out_features, rank) and applies `(lora_alpha / r) * B(A(x))`. Transposing
both factors and setting `lora_alpha = scale * rank` makes those the same
update. llama.cpp then uses `adapter_scale * lora_alpha / rank`, so an app
loraScale of 1.0 reproduces the mlx scale.

Qwen2.5 ties token embeddings to `lm_head`. llama.cpp's converter exits if
either tensor is in the adapter, and a tied `lm_head` adapter is dropped or
rejected. This conversion refuses those keys instead of silently skipping them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

FORBIDDEN_KEY_PARTS = ("lm_head", "embed_tokens", "embedding")


def transpose_2d(matrix: Sequence[Sequence[float]]) -> list[list[float]]:
    rows = [list(row) for row in matrix]
    if not rows or not rows[0]:
        raise ValueError("cannot transpose an empty matrix")
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError("matrix is ragged")
    return [[rows[r][c] for r in range(len(rows))] for c in range(width)]


def mlx_key_to_peft(key: str) -> tuple[str, str]:
    """Return (peft_key, which) where which is 'A' or 'B'.

    Accepts mlx names such as `model.layers.0.self_attn.q_proj.lora_a`.
    """
    if key.endswith(".lora_a"):
        which = "A"
        stem = key[: -len(".lora_a")]
    elif key.endswith(".lora_b"):
        which = "B"
        stem = key[: -len(".lora_b")]
    else:
        raise ValueError(f"not an mlx LoRA factor: {key}")
    lowered = stem.lower()
    for part in FORBIDDEN_KEY_PARTS:
        if part in lowered:
            raise ValueError(
                f"{key} targets {part}. Qwen2.5 ties embeddings and lm_head; "
                "leave both out of the adapter."
            )
    if ".layers." not in stem and not stem.startswith("layers."):
        raise ValueError(f"LoRA key is not on a transformer block: {key}")
    if stem.startswith("layers."):
        prefixed = f"model.{stem}"
    elif stem.startswith("model."):
        prefixed = stem
    else:
        raise ValueError(f"unrecognised mlx LoRA key: {key}")
    peft_key = f"base_model.model.{prefixed}.lora_{which}.weight"
    return peft_key, which


def lora_alpha(scale: float, rank: int) -> float:
    if rank < 1:
        raise ValueError(f"rank must be positive, got {rank}")
    return float(scale) * int(rank)


def convert_factors(
    tensors: dict[str, Sequence[Sequence[float]]],
    *,
    rank: int,
    scale: float,
) -> dict[str, list[list[float]]]:
    """Transpose mlx factors into PEFT layout. Shapes must match `rank`."""
    converted: dict[str, list[list[float]]] = {}
    seen_stems: dict[str, set[str]] = {}
    for key, matrix in tensors.items():
        peft_key, which = mlx_key_to_peft(key)
        transposed = transpose_2d(matrix)
        if which == "A":
            # mlx (in, rank) -> PEFT (rank, in)
            if len(matrix[0]) != rank:
                raise ValueError(
                    f"{key} has shape ({len(matrix)}, {len(matrix[0])}); "
                    f"expected in_features x rank {rank}"
                )
        else:
            # mlx (rank, out) -> PEFT (out, rank)
            if len(matrix) != rank:
                raise ValueError(
                    f"{key} has shape ({len(matrix)}, {len(matrix[0])}); "
                    f"expected rank {rank} x out_features"
                )
        converted[peft_key] = transposed
        stem = peft_key.rsplit(".lora_", 1)[0]
        seen_stems.setdefault(stem, set()).add(which)
    missing = [stem for stem, which in seen_stems.items() if which != {"A", "B"}]
    if missing:
        raise ValueError(f"LoRA factors are missing a side: {missing[:5]}")
    if not converted:
        raise ValueError("adapter has no LoRA factors")
    # Touch scale so a zero scale is a deliberate, visible choice.
    if scale == 0:
        raise ValueError("mlx scale is 0; the adapter would be a no-op")
    return converted


def target_modules(peft_keys: Iterable[str]) -> list[str]:
    modules = []
    for key in peft_keys:
        # ...q_proj.lora_A.weight -> q_proj
        stem = key.split(".lora_", 1)[0]
        name = stem.rsplit(".", 1)[-1]
        if name not in modules:
            modules.append(name)
    return modules


def peft_config(
    *,
    rank: int,
    scale: float,
    peft_keys: Iterable[str],
    base_model: str,
    dropout: float = 0.0,
) -> dict:
    return {
        "peft_type": "LORA",
        "base_model_name_or_path": base_model,
        "task_type": "CAUSAL_LM",
        "r": rank,
        "lora_alpha": lora_alpha(scale, rank),
        "lora_dropout": dropout,
        "bias": "none",
        "target_modules": target_modules(peft_keys),
        "inference_mode": True,
        "fan_in_fan_out": False,
        "modules_to_save": None,
    }


def read_mlx_lora_config(adapter_dir: Path) -> tuple[int, float, float]:
    path = adapter_dir / "adapter_config.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    params = config.get("lora_parameters") or {}
    if "rank" not in params or "scale" not in params:
        raise ValueError(
            f"{path} has no lora_parameters.rank and lora_parameters.scale. "
            "mlx-lm writes `scale`, not `alpha`."
        )
    rank = int(params["rank"])
    scale = float(params["scale"])
    dropout = float(params.get("dropout", 0.0))
    return rank, scale, dropout


def _matrix_from_array(array) -> list[list[float]]:
    if array.ndim != 2:
        raise ValueError(f"expected a 2-D LoRA factor, got shape {tuple(array.shape)}")
    return array.astype("float32").tolist()


def convert_adapter_directory(
    mlx_adapter: Path,
    output: Path,
    *,
    checkpoint: Path | None = None,
    base_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
) -> dict:
    """Read mlx safetensors and write a PEFT adapter directory.

    Imports numpy and safetensors only when called, so the pure conversion
    functions can be unit-tested without those packages.
    """
    import numpy as np
    from safetensors.numpy import load_file, save_file

    rank, scale, dropout = read_mlx_lora_config(mlx_adapter)
    weights_path = checkpoint or (mlx_adapter / "adapters.safetensors")
    if not weights_path.is_file():
        raise FileNotFoundError(weights_path)
    raw = load_file(str(weights_path))
    tensors = {key: _matrix_from_array(np.asarray(value)) for key, value in raw.items()}
    converted = convert_factors(tensors, rank=rank, scale=scale)
    arrays = {key: np.asarray(value, dtype=np.float32) for key, value in converted.items()}
    output.mkdir(parents=True, exist_ok=True)
    save_file(arrays, str(output / "adapter_model.safetensors"))
    config = peft_config(
        rank=rank,
        scale=scale,
        peft_keys=converted.keys(),
        base_model=base_model,
        dropout=dropout,
    )
    (output / "adapter_config.json").write_text(
        json.dumps(config, indent=2) + "\n",
        encoding="utf-8",
    )
    return config
