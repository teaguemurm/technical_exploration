"""Protocol strings stay aligned with the Flutter assets."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wisent_lora.protocol import (  # noqa: E402
    CONSTRAINT_MARKERS,
    condition_b_app_messages,
    condition_b_system,
    load_local_notes_intro,
    load_local_notes_title,
    load_persona,
    load_prompts,
    load_protocol_decoding,
    load_real_notes,
    load_refusal,
    local_notes_message,
    trim_single_trailing_newline,
)


class ProtocolTests(unittest.TestCase):
    def test_trim_matches_dart_rules(self) -> None:
        self.assertEqual(trim_single_trailing_newline("ab\n"), "ab")
        self.assertEqual(trim_single_trailing_newline("ab\r\n"), "ab")
        self.assertEqual(trim_single_trailing_newline("ab\n\n"), "ab\n")
        self.assertEqual(trim_single_trailing_newline("ab"), "ab")

    def test_refusal_is_the_dart_constant(self) -> None:
        self.assertEqual(
            load_refusal(),
            "I do not have information about that in the approved notes.",
        )

    def test_persona_has_no_constraint_text(self) -> None:
        persona = load_persona()
        self.assertFalse(persona.endswith("\n"))
        self.assertIn("living lab", persona)
        for marker in CONSTRAINT_MARKERS:
            self.assertNotIn(marker, persona)

    def test_notes_block_matches_the_app(self) -> None:
        block = local_notes_message(["Note one."])
        self.assertEqual(
            block,
            "Local notes\n"
            "\n"
            "Local notes follow. They are the only place-specific notes supplied for this run.\n"
            "Note one.",
        )
        system = condition_b_system(load_persona(), ["Note one."])
        self.assertTrue(system.startswith(load_persona() + "\n\nLocal notes\n"))
        for marker in CONSTRAINT_MARKERS:
            self.assertNotIn(marker, system)
        self.assertEqual(condition_b_system(load_persona(), []), load_persona())

    def test_decoding_settings_come_from_the_app(self) -> None:
        decoding = load_protocol_decoding()
        self.assertEqual(decoding["temperature"], 0.0)
        self.assertEqual(decoding["seed"], 42)
        self.assertEqual(decoding["top_k"], 40)
        self.assertEqual(decoding["top_p"], 0.9)
        self.assertEqual(decoding["repeat_penalty"], 1.1)
        self.assertEqual(decoding["max_tokens"], 256)
        self.assertEqual(decoding["context_size"], 2048)
        self.assertEqual(decoding["repeat_last_n"], 64)
        sampler = (
            Path(__file__).resolve().parents[2]
            / "packages"
            / "llm_llamacpp"
            / "lib"
            / "src"
            / "inference_sampler_config.dart"
        ).read_text(encoding="utf-8")
        self.assertNotIn("llama_sampler_init_min_p", sampler)

    def test_app_condition_b_messages_follow_the_dart_shape(self) -> None:
        persona = load_persona()
        messages = condition_b_app_messages(persona, ["Note one."], "Where do they drink?")
        self.assertEqual([message["role"] for message in messages], ["system", "user", "user"])
        self.assertEqual(messages[0]["content"], persona)
        self.assertEqual(messages[1]["content"], local_notes_message(["Note one."]))
        self.assertEqual(messages[2]["content"], "Where do they drink?")
        self.assertEqual(load_local_notes_title(), "Local notes")
        self.assertTrue(load_local_notes_intro().startswith("Local notes follow."))
        empty = condition_b_app_messages(persona, [], "Where do they drink?")
        self.assertEqual([message["role"] for message in empty], ["system", "user"])

    def test_locked_catalog_sizes(self) -> None:
        self.assertEqual(len(load_prompts()), 10)
        self.assertEqual(len(load_real_notes()), 10)


if __name__ == "__main__":
    unittest.main()
