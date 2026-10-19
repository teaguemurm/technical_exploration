"""The leakage checker catches prompts and misses ordinary questions."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wisent_lora.leakage import check_examples, leak_reason  # noqa: E402

PROMPT = "I'm at Kraansvlak and I can see a herd. What animal is this?"
PARAPHRASE = "Pretend to be one of the Kraansvlak wisents and say what you feel."


def _example(question: str, system: str = "persona") -> dict:
    return {
        "metadata": {"id": "probe"},
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": question},
            {"role": "assistant", "content": "unused"},
        ],
    }


class LeakageTests(unittest.TestCase):
    def test_exact_prompt_is_a_hit(self) -> None:
        hits = check_examples([_example(PROMPT)], [PROMPT])
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["reason"], "containment")

    def test_shared_4gram_is_a_hit(self) -> None:
        question = "Please tell me what animal is this exactly."
        self.assertIsNotNone(leak_reason(question, PROMPT))

    def test_ordinary_question_is_clean(self) -> None:
        question = "Do the bands use reed beds for shelter?"
        self.assertIsNone(leak_reason(question, PROMPT))
        self.assertIsNone(leak_reason(question, PARAPHRASE))

    def test_4gram_inside_the_system_message_is_a_hit(self) -> None:
        system = "Notes. One of the Kraansvlak herd is coming onto the path towards me today."
        on_path = "One of the Kraansvlak herd is coming onto the path towards me. What should I do?"
        hits = check_examples([_example("Where do the deer drink?", system)], [on_path])
        self.assertTrue(hits)
        self.assertTrue(hits[0]["reason"].startswith("4-gram:"))


if __name__ == "__main__":
    unittest.main()
