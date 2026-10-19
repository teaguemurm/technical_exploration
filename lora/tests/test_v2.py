"""v2 matches condition B and refuses preaching and nature-decode traps."""

from __future__ import annotations

import hashlib
import json
import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wisent_lora.leakage import check_examples, default_probes, normalize  # noqa: E402
from wisent_lora.protocol import load_refusal, repo_root  # noqa: E402
from wisent_lora.v2 import build_v2_examples  # noqa: E402
from wisent_lora.validate import validate_examples  # noqa: E402

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


def _question(row: dict) -> str:
    return [message["content"] for message in row["messages"] if message["role"] == "user"][-1]


def _trim_one_newline(raw: str) -> str:
    if raw.endswith("\r\n"):
        return raw[:-2]
    if raw.endswith("\n"):
        return raw[:-1]
    return raw


def _dart_const(text: str, name: str) -> str:
    match = re.search(rf"const\s+{name}\s*=\s*(?:\n\s*)?'([^']*)'", text)
    if match is None:
        raise AssertionError(f"{name} missing from lib/messages.dart")
    return match.group(1)


def _app_condition_b(root: Path, note_texts: list[str], question: str) -> list[dict]:
    """Assemble condition B from the Dart source and the system-prompt asset."""
    dart = (root / "lib" / "messages.dart").read_text(encoding="utf-8")
    persona = _trim_one_newline((root / "assets" / "system_prompt.txt").read_text(encoding="utf-8"))
    messages = [{"role": "system", "content": persona}]
    if note_texts:
        block = "\n".join(
            [
                _dart_const(dart, "localNotesTitle"),
                "",
                _dart_const(dart, "localNotesIntro"),
                *note_texts,
            ]
        )
        messages.append({"role": "user", "content": block})
    messages.append({"role": "user", "content": question})
    return messages


def _note_map(root: Path) -> dict[str, str]:
    notes = {}
    for item in json.loads((root / "assets" / "dataset.json").read_text(encoding="utf-8")):
        notes[item["id"]] = item["text"]
    document = json.loads((root / "lora" / "data" / "synthetic_notes.json").read_text(encoding="utf-8"))
    for pack in document["packs"]:
        for note in pack["notes"]:
            notes[note["id"]] = note["text"]
    return notes


class V2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = repo_root()
        cls.train, cls.valid, cls.note_text = build_v2_examples(cls.root)
        cls.rows = cls.train + cls.valid
        cls.refusal = load_refusal(cls.root)
        cls.by_id = {row["metadata"]["id"]: row for row in cls.rows}

    def test_original_checksums_are_unchanged(self) -> None:
        data = self.root / "lora" / "data"
        for name, expected in ORIGINAL_SHA256.items():
            digest = hashlib.sha256((data / name).read_bytes()).hexdigest()
            self.assertEqual(digest, expected, name)

    def test_disk_matches_the_builder(self) -> None:
        folder = self.root / "lora" / "data" / "v2"
        self.assertEqual(_read_jsonl(folder / "train.jsonl"), self.train)
        self.assertEqual(_read_jsonl(folder / "valid.jsonl"), self.valid)

    def test_preaching_rows_use_the_refusal_string(self) -> None:
        preaching = [row for row in self.rows if row["metadata"]["category"] == "preaching"]
        self.assertEqual(len(preaching), 40)
        for row in preaching:
            self.assertEqual(row["messages"][-1]["content"], self.refusal)
            self.assertIsInstance(row["metadata"]["grounded_note_ids"], list)

    def test_nature_decode_rows_refuse_and_stay_off_the_held_out_prompts(self) -> None:
        nature = [row for row in self.rows if row["metadata"]["category"] == "nature_decode"]
        self.assertEqual(len(nature), 40)
        kinds = [row["metadata"]["trap_kind"] for row in nature]
        self.assertEqual(kinds.count("speak_as_nature"), 20)
        self.assertEqual(kinds.count("decode_behaviour"), 20)
        self.assertEqual(sum(1 for row in nature if row["metadata"]["prompt_note_ids"]), 20)
        self.assertEqual(sum(1 for row in nature if not row["metadata"]["prompt_note_ids"]), 20)
        self.assertEqual(sum(1 for row in nature if row["metadata"]["split"] == "valid"), 4)
        self.assertEqual(sum(1 for row in nature if row["metadata"]["split"] == "train"), 36)
        for row in nature:
            self.assertEqual(row["messages"][-1]["content"], self.refusal)
            self.assertEqual(row["metadata"]["grounded_note_ids"], "none")
        self.assertEqual(check_examples(nature, default_probes(self.root)), [])
        self.assertEqual(check_examples(self.rows, default_probes(self.root)), [])

    def test_comparison_only_questions_do_not_leak_into_v2(self) -> None:
        comparison = _read_jsonl(self.root / "lora" / "data" / "comparison" / "comparison.jsonl")
        held_out = {
            normalize(_question(row))
            for row in comparison
            if not row["metadata"]["question_in_original"]
        }
        v2_questions = {normalize(_question(row)) for row in self.rows}
        self.assertTrue(held_out)
        self.assertFalse(held_out & v2_questions)

    def test_rendered_example_matches_the_app_condition_b_prompt(self) -> None:
        notes = _note_map(self.root)
        with_notes = next(
            row
            for row in self.rows
            if row["metadata"]["prompt_note_ids"] and row["metadata"]["category"] == "answerable"
        )
        texts = [notes[note_id] for note_id in with_notes["metadata"]["prompt_note_ids"]]
        expected = _app_condition_b(self.root, texts, _question(with_notes))
        self.assertEqual(with_notes["messages"][:-1], expected)
        self.assertNotEqual(with_notes["messages"][-1]["role"], "user")

        empty = next(row for row in self.rows if not row["metadata"]["prompt_note_ids"])
        expected_empty = _app_condition_b(self.root, [], _question(empty))
        self.assertEqual(empty["messages"][:-1], expected_empty)
        self.assertEqual([message["role"] for message in expected_empty], ["system", "user"])

    def test_copied_rows_keep_answers_notes_and_splits(self) -> None:
        data = self.root / "lora" / "data"
        original_train = _read_jsonl(data / "train.jsonl")
        original_valid = _read_jsonl(data / "valid.jsonl")
        original = original_train + original_valid
        copied_train = [row for row in self.train if row["metadata"]["category"] != "nature_decode"]
        copied_valid = [row for row in self.valid if row["metadata"]["category"] != "nature_decode"]
        self.assertEqual(
            [row["metadata"]["id"] for row in copied_train],
            [row["metadata"]["id"] for row in original_train],
        )
        self.assertEqual(
            [row["metadata"]["id"] for row in copied_valid],
            [row["metadata"]["id"] for row in original_valid],
        )
        for old in original:
            new = self.by_id[old["metadata"]["id"]]
            self.assertEqual(new["metadata"]["split"], old["metadata"]["split"])
            self.assertEqual(new["metadata"]["category"], old["metadata"]["category"])
            self.assertEqual(new["metadata"]["prompt_note_ids"], old["metadata"]["prompt_note_ids"])
            self.assertEqual(new["metadata"]["grounded_note_ids"], old["metadata"]["grounded_note_ids"])
            self.assertEqual(new["metadata"]["must_include"], old["metadata"]["must_include"])
            self.assertEqual(_question(new), old["messages"][1]["content"])
            if old["metadata"]["category"] == "preaching":
                self.assertEqual(new["messages"][-1]["content"], self.refusal)
                self.assertNotEqual(old["messages"][2]["content"], self.refusal)
            else:
                self.assertEqual(new["messages"][-1]["content"], old["messages"][2]["content"])

    def test_train_layout_still_rejects_a_preaching_refusal(self) -> None:
        data = self.root / "lora" / "data"
        original = next(
            row for row in _read_jsonl(data / "train.jsonl") if row["metadata"]["category"] == "preaching"
        )
        mutated = json.loads(json.dumps(original))
        mutated["messages"][2]["content"] = self.refusal
        report = validate_examples(
            [mutated],
            refusal=self.refusal,
            note_texts=self.note_text,
            layout="train",
            size_bounds=None,
            check_split_ratio=False,
        )
        self.assertFalse(report.ok)
        self.assertTrue(any("may not refuse" in error or "refusal string" in error for error in report.errors))


if __name__ == "__main__":
    unittest.main()
