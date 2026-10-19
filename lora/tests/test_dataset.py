"""The generated chat set matches the builder and the mix targets."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wisent_lora.build import assign_splits, build_examples  # noqa: E402
from wisent_lora.leakage import check_dataset_dir  # noqa: E402
from wisent_lora.protocol import load_refusal, repo_root  # noqa: E402
from wisent_lora.validate import validate_examples, validate_files  # noqa: E402


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = repo_root()
        cls.examples, cls.note_text = build_examples(cls.root)
        cls.train, cls.valid = assign_splits(cls.examples)
        cls.refusal = load_refusal(cls.root)

    def test_builder_is_deterministic(self) -> None:
        again, _ = build_examples(self.root)
        train, valid = assign_splits(again)
        self.assertEqual(self.train, train)
        self.assertEqual(self.valid, valid)

    def test_disk_matches_builder(self) -> None:
        data = self.root / "lora" / "data"
        self.assertEqual(_read_jsonl(data / "train.jsonl"), self.train)
        self.assertEqual(_read_jsonl(data / "valid.jsonl"), self.valid)
        report = validate_files(data, refusal=self.refusal, note_texts=self.note_text)
        self.assertTrue(report.ok, report.errors[:12])
        self.assertEqual(check_dataset_dir(data, self.root), [])

    def test_mix_and_pairs(self) -> None:
        report = validate_examples(
            self.train + self.valid,
            refusal=self.refusal,
            note_texts=self.note_text,
        )
        self.assertTrue(report.ok, report.errors[:12])
        self.assertGreaterEqual(report.n_examples, 300)
        self.assertLessEqual(report.n_examples, 600)
        self.assertGreater(report.n_pairs, 0)
        self.assertEqual(report.counts["answerable"], 191)
        self.assertEqual(report.counts["refusal"], 124)
        self.assertEqual(report.counts["speak_as"], 60)
        self.assertEqual(report.counts["preaching"], 40)

    def test_metadata_is_ignored_by_the_mlx_chat_loader(self) -> None:
        """ChatDataset.process reads messages and tools. Other keys are ignored.

        mlx-lm LORA.md: "keys in each line not expected by the loader will be ignored."
        """
        sample = self.train[0]
        self.assertIn("metadata", sample)
        self.assertNotIn("prompt", sample)
        self.assertNotIn("completion", sample)
        messages = sample["messages"]
        tools = sample.get("tools")
        self.assertEqual([message["role"] for message in messages], ["system", "user", "assistant"])
        self.assertIsNone(tools)
        self.assertNotIn("STRICT GROUNDING", messages[0]["content"])

    def test_contrastive_pair_flips(self) -> None:
        by_id = {row["metadata"]["id"]: row for row in self.train + self.valid}
        with_note = by_id["wisent-feed-people-with"]
        without = by_id["wisent-feed-people-without"]
        self.assertEqual(with_note["messages"][1]["content"], without["messages"][1]["content"])
        self.assertEqual(with_note["messages"][2]["content"], "The wisents are not fed by people.")
        self.assertEqual(without["messages"][2]["content"], self.refusal)
        self.assertNotIn("not fed by people", without["messages"][0]["content"])
        self.assertEqual(with_note["metadata"]["split"], without["metadata"]["split"])
        self.assertEqual(with_note["metadata"]["notes_origin"], "real")
        self.assertFalse(with_note["metadata"]["synthetic"])


if __name__ == "__main__":
    unittest.main()
