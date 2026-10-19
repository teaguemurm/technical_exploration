"""The comparison set copies the training rows and adds new questions."""

from __future__ import annotations

import hashlib
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wisent_lora.comparison import (  # noqa: E402
    COMPARISON_SIZE_BOUNDS,
    build_comparison,
    write_comparison,
)
from wisent_lora.leakage import check_examples, default_probes  # noqa: E402
from wisent_lora.protocol import load_refusal, repo_root  # noqa: E402
from wisent_lora.validate import validate_examples  # noqa: E402

# Recorded before the comparison files were added. The training set must stay
# byte-identical.
ORIGINAL_SHA256 = {
    "train.jsonl": "849790bd3d294932040e165b6fe2fd0078a77228e3cdc4b9993062a54a17ac91",
    "valid.jsonl": "b0d7e09340835761dc63d9f45290c6a9c51d4129db827ed0199cdb4b00acf159",
    "train.meta.jsonl": "8d5fb2ba25047a2b32f70663ba4229748a89a6fb7f323cc40e8b1d0e845540da",
    "valid.meta.jsonl": "aa0d3bd76c3793fd9c03b017dbebf77939221e7430d3e47001c8d2b7e0d47ed8",
    "stats.json": "344a0df77ec8ac1eae45b77c6ce323fa56acc74ac59566c18950a696bf09f82e",
    "synthetic_notes.json": "e8e50f8df41e7e13fa70fd80481a43a38524c536bb35cd982e8c14f037d801ce",
}


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class ComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = repo_root()
        cls.rows, cls.note_text = build_comparison(cls.root)
        cls.refusal = load_refusal(cls.root)

    def test_original_files_are_byte_identical(self) -> None:
        data = self.root / "lora" / "data"
        for name, expected in ORIGINAL_SHA256.items():
            digest = hashlib.sha256((data / name).read_bytes()).hexdigest()
            self.assertEqual(digest, expected, name)

    def test_disk_matches_builder(self) -> None:
        folder = self.root / "lora" / "data" / "comparison"
        self.assertEqual(_read_jsonl(folder / "comparison.jsonl"), self.rows)
        sidecar = _read_jsonl(folder / "comparison.meta.jsonl")
        self.assertEqual(len(sidecar), len(self.rows))
        for row, meta in zip(self.rows, sidecar):
            self.assertNotIn("messages", meta)
            self.assertEqual(meta["metadata"], row["metadata"])

    def test_copied_messages_match_the_training_files(self) -> None:
        data = self.root / "lora" / "data"
        source = _read_jsonl(data / "train.jsonl") + _read_jsonl(data / "valid.jsonl")
        copied = [row for row in self.rows if row["metadata"]["question_in_original"]]
        self.assertEqual(len(copied), len(source))
        for row, original in zip(copied, source):
            self.assertEqual(row["messages"], original["messages"])
            self.assertEqual(row["metadata"]["source_split"], original["metadata"]["split"])

    def test_mix_counts_and_flags(self) -> None:
        report = validate_examples(
            self.rows,
            refusal=self.refusal,
            note_texts=self.note_text,
            size_bounds=COMPARISON_SIZE_BOUNDS,
            check_split_ratio=False,
        )
        self.assertTrue(report.ok, report.errors[:12])
        self.assertEqual(report.n_examples, 910)
        self.assertEqual(report.counts["answerable"], 421)
        self.assertEqual(report.counts["refusal"], 269)
        self.assertEqual(report.counts["speak_as"], 132)
        self.assertEqual(report.counts["preaching"], 88)
        flags = [row["metadata"]["question_in_original"] for row in self.rows]
        self.assertEqual(flags.count(True), 415)
        self.assertEqual(flags.count(False), 495)
        self.assertEqual(check_examples(self.rows, default_probes(self.root)), [])

    def test_training_size_gate_still_rejects_this_file(self) -> None:
        report = validate_examples(
            self.rows,
            refusal=self.refusal,
            note_texts=self.note_text,
        )
        self.assertFalse(report.ok)
        self.assertTrue(any("outside 300–600" in error for error in report.errors))

    def test_new_prompts_omit_the_leaking_wisent_sentence(self) -> None:
        herd = "The Kraansvlak herd is a herd of wisents"
        fresh = [row for row in self.rows if not row["metadata"]["question_in_original"]]
        self.assertTrue(fresh)
        for row in fresh:
            self.assertNotIn(herd, row["messages"][0]["content"])
            self.assertNotIn("STRICT GROUNDING", row["messages"][0]["content"])

    def test_new_contrastive_pair_flips(self) -> None:
        by_id = {row["metadata"]["id"]: row for row in self.rows}
        with_note = by_id["cmp-pwn-initials"]
        without = by_id["cmp-pwn-initials-without"]
        self.assertEqual(with_note["messages"][1]["content"], without["messages"][1]["content"])
        self.assertEqual(with_note["messages"][2]["content"], "PWN manages the area.")
        self.assertEqual(without["messages"][2]["content"], self.refusal)
        self.assertFalse(with_note["metadata"]["question_in_original"])
        self.assertEqual(with_note["metadata"]["reference_answer"], "PWN manages the area.")

    def test_write_refuses_the_training_directory(self) -> None:
        with self.assertRaises(ValueError):
            write_comparison(self.root / "lora" / "data", self.rows)


class ComparisonRunnerTests(unittest.TestCase):
    def test_csv_and_llama_prompt(self) -> None:
        import importlib.util

        from wisent_lora.protocol import repo_root as root_of

        root = root_of()
        script = root / "lora" / "scripts" / "run_comparison.py"
        spec = importlib.util.spec_from_file_location("run_comparison", script)
        assert spec is not None and spec.loader is not None
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        rows, _ = build_comparison(root)
        sample = next(row for row in rows if row["metadata"]["id"] == "cmp-rd-unk-spread")
        prompt = runner.llama_prompt(sample)
        self.assertIn("Local notes", prompt)
        self.assertIn("<|im_start|>user\n", prompt)
        self.assertNotIn(sample["messages"][2]["content"], prompt)
        self.assertNotIn("STRICT GROUNDING", prompt)
        record = runner.csv_record(sample, "base text", "lora text")
        self.assertEqual(list(record), runner.FIELDNAMES)
        self.assertEqual(record["output_without_adapter"], "base text")
        self.assertEqual(record["output_with_adapter"], "lora text")
        self.assertEqual(record["question_in_original"], "false")
        self.assertEqual(record["category"], "refusal")
        self.assertEqual(record["grounded_note_ids"], "none")
