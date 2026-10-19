"""Rendered prompts follow the app roles and the training roles."""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wisent_lora.protocol import load_persona, load_real_notes, load_refusal, load_prompts, repo_root  # noqa: E402
from wisent_lora.render import messages_for, render_protocol  # noqa: E402


class RenderTests(unittest.TestCase):
    def test_app_b_has_notes_as_a_user_turn(self) -> None:
        text = render_protocol(condition="B", prompt_id="what-animal", layout="app")
        self.assertEqual(text.count("<|im_start|>user\n"), 2)
        self.assertNotIn("STRICT GROUNDING", text)
        self.assertNotIn(load_refusal(), text)
        self.assertIn("The Kraansvlak herd is a herd of wisents", text)
        self.assertTrue(text.endswith("<|im_start|>assistant\n"))

    def test_train_layout_puts_notes_in_the_system_message(self) -> None:
        text = render_protocol(condition="B", prompt_id="what-animal", layout="train")
        self.assertEqual(text.count("<|im_start|>user\n"), 1)
        system, _, rest = text.partition("<|im_end|>\n")
        self.assertIn("Local notes", system)
        user = rest.split("<|im_start|>user\n", 1)[1].split("<|im_end|>", 1)[0]
        self.assertNotIn("Local notes", user)
        self.assertIn("What animal is this?", user)

    def test_condition_c_carries_the_constraint_suffix(self) -> None:
        text = render_protocol(condition="C", prompt_id="morning-meal", layout="app")
        self.assertIn("STRICT GROUNDING & FORM CONSTRAINTS:", text)
        self.assertIn(load_refusal(), text)
        self.assertIn("Do NOT preach or moralise at the visitor.", text)

    def test_condition_a_has_no_notes(self) -> None:
        text = render_protocol(condition="A", prompt_id="speak-as", layout="app")
        self.assertNotIn("Local notes", text)
        self.assertEqual(text.count("<|im_start|>user\n"), 1)
        self.assertIn(load_persona(), text)

    def test_sim_rag_without_a_mapping_injects_the_empty_note(self) -> None:
        root = repo_root()
        messages = messages_for(
            condition="C-sim",
            prompt={"id": "not-in-the-map", "text": "Hello"},
            notes=load_real_notes(root),
            persona=load_persona(root),
            refusal=load_refusal(root),
            sim_rag={"k": 2, "retrieval": {}},
            layout="app",
        )
        self.assertIn("(No specific local notes match this query.)", messages[1][1])
        self.assertEqual(len(load_prompts(root)), 10)

    def test_converter_shell_parses(self) -> None:
        script = Path(__file__).resolve().parents[1] / "scripts" / "convert_adapter_gguf.sh"
        subprocess.run(["bash", "-n", str(script)], check=True)


if __name__ == "__main__":
    unittest.main()
