"""Protocol strings stay aligned with the Flutter assets."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wisent_lora.protocol import (  # noqa: E402
    CONSTRAINT_MARKERS,
    condition_b_system,
    load_persona,
    load_prompts,
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

    def test_locked_catalog_sizes(self) -> None:
        self.assertEqual(len(load_prompts()), 10)
        self.assertEqual(len(load_real_notes()), 10)


if __name__ == "__main__":
    unittest.main()
