"""App-layout training set.

Copies the original examples, moves notes into the condition-B user turn, and
turns preaching answers into the refusal string. Adds nature_decode rows for
the Nature-voice and behaviour-decoding bans. Writes only under lora/data/v2.
"""

from __future__ import annotations

import copy
import json
import random
from collections import Counter
from pathlib import Path

from wisent_lora.build import SPLIT_SEED, VALID_FRACTION, assign_splits, build_examples
from wisent_lora.corpus import PACKS
from wisent_lora.leakage import check_examples, default_probes
from wisent_lora.protocol import (
    condition_b_app_messages,
    load_persona,
    load_refusal,
    repo_root,
)
from wisent_lora.traps import NATURE_DECODE
from wisent_lora.validate import CATEGORIES, format_report, validate_examples

COPIED_COUNTS = {
    "answerable": 191,
    "refusal": 124,
    "speak_as": 60,
    "preaching": 40,
}
NATURE_RULES = (
    "Condition C rule 4, Nature clause: Do NOT speak as Nature.",
    "Condition C rule 5: Do NOT claim to decode or translate the animal's behaviour.",
)


def _copied_row(row: dict, persona: str, refusal: str, note_text: dict[str, str]) -> dict:
    row = copy.deepcopy(row)
    metadata = row["metadata"]
    question = row["messages"][1]["content"]
    answer = row["messages"][2]["content"]
    if metadata["category"] == "preaching":
        answer = refusal
    texts = [note_text[note_id] for note_id in metadata["prompt_note_ids"]]
    row["messages"] = condition_b_app_messages(persona, texts, question) + [
        {"role": "assistant", "content": answer}
    ]
    return row


def _check_nature_bank() -> None:
    pack_ids = [pack["id"] for pack in PACKS]
    if set(NATURE_DECODE) != set(pack_ids):
        raise ValueError("nature_decode bank does not match the synthetic packs")
    for pack_id, items in NATURE_DECODE.items():
        counts = Counter((kind, with_notes) for _, with_notes, kind in items)
        for kind in ("speak_as_nature", "decode_behaviour"):
            for with_notes in (True, False):
                if counts[(kind, with_notes)] != 2:
                    raise ValueError(f"{pack_id} needs two {kind} rows with notes={with_notes}")


def nature_decode_rows(persona: str, refusal: str, note_text: dict[str, str]) -> list[dict]:
    _check_nature_bank()
    rows = []
    for pack in PACKS:
        all_ids = [note["id"] for note in pack["notes"]]
        indexes = Counter()
        for question, with_notes, kind in NATURE_DECODE[pack["id"]]:
            indexes[kind] += 1
            suffix = "nature" if kind == "speak_as_nature" else "decode"
            prompt_ids = list(all_ids) if with_notes else []
            origin = "synthetic" if with_notes else "none"
            rows.append(
                {
                    "messages": condition_b_app_messages(
                        persona,
                        [note_text[note_id] for note_id in prompt_ids],
                        question,
                    )
                    + [{"role": "assistant", "content": refusal}],
                    "metadata": {
                        "id": f"{pack['id']}-{suffix}-{indexes[kind]:02d}",
                        "category": "nature_decode",
                        "grounded_note_ids": "none",
                        "prompt_note_ids": prompt_ids,
                        "notes_origin": origin,
                        "synthetic": origin == "synthetic",
                        "species_pack": pack["id"],
                        "pair_id": None,
                        "pair_role": "single",
                        "must_include": [],
                        "trap_kind": kind,
                    },
                }
            )
    return rows


def split_nature_rows(rows: list[dict], seed: int = SPLIT_SEED) -> tuple[list[dict], list[dict]]:
    """Same 10% rule as assign_splits, on its own random stream."""
    rng = random.Random(seed)
    units = list(rows)
    rng.shuffle(units)
    n_valid = int(round(len(units) * VALID_FRACTION))
    if len(units) >= 10:
        n_valid = max(1, n_valid)
    valid: list[dict] = []
    train: list[dict] = []
    for row in units[:n_valid]:
        row["metadata"]["split"] = "valid"
        valid.append(row)
    for row in units[n_valid:]:
        row["metadata"]["split"] = "train"
        train.append(row)
    return train, valid


def build_v2_examples(root: Path | None = None) -> tuple[list[dict], list[dict], dict[str, str]]:
    root = root or repo_root()
    persona = load_persona(root)
    refusal = load_refusal(root)
    examples, note_text = build_examples(root)
    train, valid = assign_splits(examples)
    train = [_copied_row(row, persona, refusal, note_text) for row in train]
    valid = [_copied_row(row, persona, refusal, note_text) for row in valid]
    extra_train, extra_valid = split_nature_rows(nature_decode_rows(persona, refusal, note_text))
    train.extend(extra_train)
    valid.extend(extra_valid)
    return train, valid, note_text


def _stats(train: list[dict], valid: list[dict], refusal: str) -> dict:
    def tally(rows: list[dict]) -> dict[str, int]:
        counts: Counter[str] = Counter(row["metadata"]["category"] for row in rows)
        return {category: counts[category] for category in CATEGORIES}

    all_rows = train + valid
    copied = [row for row in all_rows if row["metadata"]["category"] != "nature_decode"]
    nature = [row for row in all_rows if row["metadata"]["category"] == "nature_decode"]
    origins: Counter[str] = Counter(row["metadata"]["notes_origin"] for row in all_rows)
    pairs = len({row["metadata"]["pair_id"] for row in all_rows if row["metadata"]["pair_id"]})
    subtypes: Counter[str] = Counter(row["metadata"]["trap_kind"] for row in nature)
    return {
        "layout": "app_condition_b",
        "preaching_answer": "refusal",
        "refusal_string": refusal,
        "nature_decode_targets": list(NATURE_RULES),
        "copied_from_original": {
            "total": len(copied),
            "by_category": {category: tally(copied)[category] for category in COPIED_COUNTS},
        },
        "train": {"n": len(train), "by_category": tally(train)},
        "valid": {"n": len(valid), "by_category": tally(valid)},
        "total": len(all_rows),
        "by_category": tally(all_rows),
        "nature_decode": {
            "n": len(nature),
            "by_subtype": {
                "speak_as_nature": subtypes["speak_as_nature"],
                "decode_behaviour": subtypes["decode_behaviour"],
            },
            "with_notes": sum(1 for row in nature if row["metadata"]["prompt_note_ids"]),
            "no_notes": sum(1 for row in nature if not row["metadata"]["prompt_note_ids"]),
            "train": sum(1 for row in nature if row["metadata"]["split"] == "train"),
            "valid": sum(1 for row in nature if row["metadata"]["split"] == "valid"),
        },
        "notes_origin": dict(origins),
        "contrastive_pairs": pairs,
    }


def _assert_counts(stats: dict) -> None:
    copied = stats["copied_from_original"]
    if copied["by_category"] != COPIED_COUNTS:
        raise SystemExit(f"copied category counts changed: {copied['by_category']}")
    if copied["total"] != 415:
        raise SystemExit(f"copied row count is {copied['total']}, expected 415")
    nature = stats["nature_decode"]
    if nature["n"] != 40 or nature["by_subtype"] != {"speak_as_nature": 20, "decode_behaviour": 20}:
        raise SystemExit(f"nature_decode mix is wrong: {nature}")
    if nature["with_notes"] != 20 or nature["no_notes"] != 20:
        raise SystemExit(f"nature_decode notes mix is wrong: {nature}")
    if nature["train"] != 36 or nature["valid"] != 4:
        raise SystemExit(f"nature_decode split is wrong: {nature}")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    path.write_text(text, encoding="utf-8")


def write_v2(data_dir: Path, train: list[dict], valid: list[dict], refusal: str) -> dict:
    data_dir.mkdir(parents=True, exist_ok=True)
    stats = _stats(train, valid, refusal)
    _assert_counts(stats)
    for split, rows in (("train", train), ("valid", valid)):
        _write_jsonl(data_dir / f"{split}.jsonl", rows)
        _write_jsonl(data_dir / f"{split}.meta.jsonl", [{"metadata": row["metadata"]} for row in rows])
    (data_dir / "stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    return stats


def build_and_check_v2(data_dir: Path | None = None, root: Path | None = None) -> dict:
    root = root or repo_root()
    data_dir = data_dir or (root / "lora" / "data" / "v2")
    if data_dir.resolve() == (root / "lora" / "data").resolve():
        raise SystemExit("v2 must not be written over lora/data")
    train, valid, note_text = build_v2_examples(root)
    refusal = load_refusal(root)
    rows = train + valid
    report = validate_examples(rows, refusal=refusal, note_texts=note_text, layout="app")
    if not report.ok:
        raise SystemExit("v2 validation failed:\n" + format_report(report))
    hits = check_examples(rows, default_probes(root))
    if hits:
        preview = "\n".join(f"{hit['id']}: {hit['reason']}: {hit['probe']}" for hit in hits[:30])
        raise SystemExit(f"v2 leakage check failed ({len(hits)} hits):\n{preview}")
    stats = write_v2(data_dir, train, valid, refusal)
    stats["validation_errors"] = 0
    stats["leakage_hits"] = 0
    return stats
