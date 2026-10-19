#!/usr/bin/env python3
"""Convert an mlx-lm adapter directory into a PEFT adapter directory.

Requires numpy and safetensors (see requirements-mac.txt). The reshape rules
themselves are covered by lora/tests/test_peft.py without those packages.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wisent_lora.peft_convert import convert_adapter_directory  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mlx-adapter",
        type=Path,
        required=True,
        help="Directory with adapter_config.json and adapters.safetensors",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional NNNNNNN_adapters.safetensors chosen by validation loss",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="PEFT directory to write (adapter_config.json, adapter_model.safetensors)",
    )
    parser.add_argument(
        "--base-model",
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="base_model_name_or_path written into the PEFT config",
    )
    args = parser.parse_args()
    config = convert_adapter_directory(
        args.mlx_adapter,
        args.output,
        checkpoint=args.checkpoint,
        base_model=args.base_model,
    )
    print(
        f"wrote {args.output} rank={config['r']} "
        f"lora_alpha={config['lora_alpha']} modules={config['target_modules']}"
    )


if __name__ == "__main__":
    main()
