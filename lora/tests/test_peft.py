"""MLX LoRA names and shapes become PEFT names and shapes."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wisent_lora.peft_convert import (  # noqa: E402
    convert_factors,
    lora_alpha,
    mlx_key_to_peft,
    peft_config,
)


class PeftTests(unittest.TestCase):
    def test_transpose_and_alpha(self) -> None:
        tensors = {
            "model.layers.0.self_attn.q_proj.lora_a": [
                [1, 2],
                [3, 4],
                [5, 6],
                [7, 8],
            ],
            "model.layers.0.self_attn.q_proj.lora_b": [
                [9, 10, 11],
                [12, 13, 14],
            ],
        }
        converted = convert_factors(tensors, rank=2, scale=20.0)
        self.assertEqual(
            converted["base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"],
            [[1, 3, 5, 7], [2, 4, 6, 8]],
        )
        self.assertEqual(
            converted["base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight"],
            [[9, 12], [10, 13], [11, 14]],
        )
        self.assertEqual(lora_alpha(20.0, 16), 320.0)
        config = peft_config(
            rank=2,
            scale=20.0,
            peft_keys=converted,
            base_model="Qwen/Qwen2.5-1.5B-Instruct",
        )
        self.assertEqual(config["lora_alpha"], 40.0)
        self.assertEqual(config["target_modules"], ["q_proj"])
        self.assertEqual(config["peft_type"], "LORA")

    def test_layers_prefix_without_model(self) -> None:
        key, which = mlx_key_to_peft("layers.3.mlp.down_proj.lora_b")
        self.assertEqual(which, "B")
        self.assertEqual(
            key,
            "base_model.model.model.layers.3.mlp.down_proj.lora_B.weight",
        )

    def test_tied_embeddings_are_rejected(self) -> None:
        for name in ("model.lm_head.lora_a", "model.embed_tokens.lora_b"):
            with self.assertRaises(ValueError):
                mlx_key_to_peft(name)

    def test_rank_mismatch_and_missing_side(self) -> None:
        bad = {
            "model.layers.0.mlp.up_proj.lora_a": [[1, 2, 3], [4, 5, 6]],
            "model.layers.0.mlp.up_proj.lora_b": [[1, 2], [3, 4]],
        }
        with self.assertRaises(ValueError):
            convert_factors(bad, rank=2, scale=20.0)
        with self.assertRaises(ValueError):
            convert_factors(
                {"model.layers.1.self_attn.v_proj.lora_a": [[1, 2], [3, 4]]},
                rank=2,
                scale=20.0,
            )
        with self.assertRaises(ValueError):
            lora_alpha(20.0, 0)


if __name__ == "__main__":
    unittest.main()
