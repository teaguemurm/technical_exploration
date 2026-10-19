"""Build chat JSONL for mlx-lm from the note bank.

The system message is condition B: the visitor persona plus the local-notes
block, and never the condition-C constraint text. Contrastive pairs keep every
other note and drop only the grounded one, so the same question flips from an
answer to the exact refusal.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

from wisent_lora.corpus import PACKS, WISENT_QAS
from wisent_lora.leakage import check_examples, default_probes
from wisent_lora.protocol import condition_b_system, load_persona, load_real_notes, load_refusal, repo_root
from wisent_lora.traps import PREACHING, REFUSAL_EMPTY, REFUSAL_FULL, SPEAK_AS
from wisent_lora.validate import format_report, validate_examples

SPLIT_SEED = 42
VALID_FRACTION = 0.1


def _example(
    *,
    example_id: str,
    category: str,
    origin: str,
    pack: str,
    question: str,
    answer: str,
    grounded: list[str] | str,
    prompt_note_ids: list[str],
    note_text: dict[str, str],
    persona: str,
    pair_id: str | None,
    pair_role: str,
    must_include: list[str],
) -> dict:
    texts = [note_text[note_id] for note_id in prompt_note_ids]
    system = condition_b_system(persona, texts)
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ],
        "metadata": {
            "id": example_id,
            "category": category,
            "grounded_note_ids": grounded if grounded else "none",
            "prompt_note_ids": prompt_note_ids,
            "notes_origin": origin,
            "synthetic": origin == "synthetic",
            "species_pack": pack,
            "pair_id": pair_id,
            "pair_role": pair_role,
            "must_include": must_include,
        },
    }


def _qa_examples(
    *,
    qa: dict,
    note_id: str,
    pack: str,
    origin: str,
    all_ids: list[str],
    note_text: dict[str, str],
    persona: str,
    refusal: str,
) -> list[dict]:
    must = list(qa.get("must_include") or [])
    base = _example(
        example_id=f"{qa['id']}-with" if qa["pair"] else qa["id"],
        category="answerable",
        origin=origin,
        pack=pack,
        question=qa["question"],
        answer=qa["answer"],
        grounded=[note_id],
        prompt_note_ids=list(all_ids),
        note_text=note_text,
        persona=persona,
        pair_id=qa["id"] if qa["pair"] else None,
        pair_role="with_note" if qa["pair"] else "single",
        must_include=must,
    )
    if not qa["pair"]:
        return [base]
    reduced = [item for item in all_ids if item != note_id]
    withheld = _example(
        example_id=f"{qa['id']}-without",
        category="refusal",
        origin=origin,
        pack=pack,
        question=qa["question"],
        answer=refusal,
        grounded="none",
        prompt_note_ids=reduced,
        note_text=note_text,
        persona=persona,
        pair_id=qa["id"],
        pair_role="without_note",
        must_include=[],
    )
    return [base, withheld]


def build_examples(root: Path | None = None) -> tuple[list[dict], dict[str, str]]:
    root = root or repo_root()
    persona = load_persona(root)
    refusal = load_refusal(root)
    real_notes = load_real_notes(root)
    note_text = {note["id"]: note["text"] for note in real_notes}
    for pack in PACKS:
        for note in pack["notes"]:
            if note["id"] in note_text:
                raise ValueError(f"duplicate note id {note['id']}")
            note_text[note["id"]] = note["text"]

    examples: list[dict] = []
    for pack in PACKS:
        all_ids = [note["id"] for note in pack["notes"]]
        for note in pack["notes"]:
            for qa in note["qas"]:
                examples.extend(
                    _qa_examples(
                        qa=qa,
                        note_id=note["id"],
                        pack=pack["id"],
                        origin="synthetic",
                        all_ids=all_ids,
                        note_text=note_text,
                        persona=persona,
                        refusal=refusal,
                    )
                )
        examples.extend(
            _trap_examples(
                pack=pack["id"],
                origin="synthetic",
                all_ids=all_ids,
                note_text=note_text,
                persona=persona,
                refusal=refusal,
            )
        )

    # The other six Kraansvlak notes answer held-out prompts (species, manager,
    # arrival, distance, paths, on-path). Leaving them out of the training
    # prompt avoids memorising those answers, and drops the wisent-herd
    # sentence that shares the 4-gram "the Kraansvlak herd is" with on-path.
    real_ids = ["other-grazers", "park", "wisentenpad", "no-feeding"]
    missing = [note_id for note_id in real_ids if note_id not in note_text]
    if missing:
        raise ValueError(f"real notes missing: {missing}")
    for note_id, qas in WISENT_QAS.items():
        if note_id not in note_text:
            raise ValueError(f"wisent question refers to unknown note {note_id}")
        for qa in qas:
            examples.extend(
                _qa_examples(
                    qa=qa,
                    note_id=note_id,
                    pack="wisent_kraansvlak",
                    origin="real",
                    all_ids=real_ids,
                    note_text=note_text,
                    persona=persona,
                    refusal=refusal,
                )
            )
    examples.extend(
        _refusal_questions(
            questions=REFUSAL_FULL["wisent_kraansvlak"],
            pack="wisent_kraansvlak",
            origin="real",
            prompt_note_ids=real_ids,
            note_text=note_text,
            persona=persona,
            refusal=refusal,
            id_prefix="wisent-unk",
        )
    )
    return examples, note_text


def _trap_examples(
    *,
    pack: str,
    origin: str,
    all_ids: list[str],
    note_text: dict[str, str],
    persona: str,
    refusal: str,
) -> list[dict]:
    rows = []
    for index, (question, note_id, answer) in enumerate(SPEAK_AS[pack], start=1):
        refusing = answer is None
        rows.append(
            _example(
                example_id=f"{pack}-speak-{index:02d}",
                category="speak_as",
                origin=origin,
                pack=pack,
                question=question,
                answer=refusal if refusing else answer,
                grounded="none" if refusing else [note_id],
                prompt_note_ids=list(all_ids),
                note_text=note_text,
                persona=persona,
                pair_id=None,
                pair_role="single",
                must_include=[],
            )
        )
    for index, (question, note_id, answer) in enumerate(PREACHING[pack], start=1):
        if answer is None or note_id is None:
            raise ValueError(f"{pack} preaching item {index} must be factual")
        must = ["not"] if " not " in f" {answer} " else []
        rows.append(
            _example(
                example_id=f"{pack}-preach-{index:02d}",
                category="preaching",
                origin=origin,
                pack=pack,
                question=question,
                answer=answer,
                grounded=[note_id],
                prompt_note_ids=list(all_ids),
                note_text=note_text,
                persona=persona,
                pair_id=None,
                pair_role="single",
                must_include=must,
            )
        )
    rows.extend(
        _refusal_questions(
            questions=REFUSAL_FULL[pack],
            pack=pack,
            origin=origin,
            prompt_note_ids=list(all_ids),
            note_text=note_text,
            persona=persona,
            refusal=refusal,
            id_prefix=f"{pack}-unk",
        )
    )
    rows.extend(
        _refusal_questions(
            questions=REFUSAL_EMPTY[pack],
            pack=pack,
            origin="none",
            prompt_note_ids=[],
            note_text=note_text,
            persona=persona,
            refusal=refusal,
            id_prefix=f"{pack}-empty",
        )
    )
    return rows


def _refusal_questions(
    *,
    questions: list[str],
    pack: str,
    origin: str,
    prompt_note_ids: list[str],
    note_text: dict[str, str],
    persona: str,
    refusal: str,
    id_prefix: str,
) -> list[dict]:
    rows = []
    for index, question in enumerate(questions, start=1):
        rows.append(
            _example(
                example_id=f"{id_prefix}-{index:02d}",
                category="refusal",
                origin=origin,
                pack=pack,
                question=question,
                answer=refusal,
                grounded="none",
                prompt_note_ids=prompt_note_ids,
                note_text=note_text,
                persona=persona,
                pair_id=None,
                pair_role="single",
                must_include=[],
            )
        )
    return rows


def assign_splits(examples: list[dict], seed: int = SPLIT_SEED) -> tuple[list[dict], list[dict]]:
    """90/10 split. Contrastive pairs stay in the same split."""
    rng = random.Random(seed)
    pairs: dict[str, list[dict]] = {}
    singles: dict[str, list[dict]] = {category: [] for category in ("answerable", "refusal", "speak_as", "preaching")}
    for example in examples:
        pair_id = example["metadata"]["pair_id"]
        if pair_id:
            pairs.setdefault(pair_id, []).append(example)
        else:
            singles[example["metadata"]["category"]].append(example)

    train: list[dict] = []
    valid: list[dict] = []

    def take(units: list) -> None:
        rng.shuffle(units)
        n_valid = int(round(len(units) * VALID_FRACTION))
        if len(units) >= 10:
            n_valid = max(1, n_valid)
        for unit in units[:n_valid]:
            rows = unit if isinstance(unit, list) else [unit]
            for row in rows:
                row["metadata"]["split"] = "valid"
                valid.append(row)
        for unit in units[n_valid:]:
            rows = unit if isinstance(unit, list) else [unit]
            for row in rows:
                row["metadata"]["split"] = "train"
                train.append(row)

    take(list(pairs.values()))
    for category in ("answerable", "refusal", "speak_as", "preaching"):
        take(singles[category])

    rng.shuffle(train)
    rng.shuffle(valid)
    return train, valid


def synthetic_notes_document() -> dict:
    packs = []
    for pack in PACKS:
        packs.append(
            {
                "id": pack["id"],
                "synthetic": True,
                "animal": pack["animal"],
                "place": pack["place"],
                "note": "Invented for constraint-LoRA training. Not a park fact sheet.",
                "notes": [{"id": note["id"], "text": note["text"]} for note in pack["notes"]],
            }
        )
    return {
        "synthetic": True,
        "warning": "These notes are synthetic. Do not treat them as visitor guidance.",
        "packs": packs,
    }


def write_dataset(data_dir: Path, train: list[dict], valid: list[dict], note_text: dict[str, str]) -> dict:
    data_dir.mkdir(parents=True, exist_ok=True)
    stats = _stats(train, valid)
    for split, rows in (("train", train), ("valid", valid)):
        _write_jsonl(data_dir / f"{split}.jsonl", rows)
        _write_jsonl(
            data_dir / f"{split}.meta.jsonl",
            [{"metadata": row["metadata"]} for row in rows],
        )
    (data_dir / "synthetic_notes.json").write_text(
        json.dumps(synthetic_notes_document(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (data_dir / "stats.json").write_text(
        json.dumps(stats, indent=2) + "\n",
        encoding="utf-8",
    )
    return stats


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    path.write_text(text, encoding="utf-8")


def _stats(train: list[dict], valid: list[dict]) -> dict:
    def tally(rows: list[dict]) -> dict[str, int]:
        counts: Counter[str] = Counter(row["metadata"]["category"] for row in rows)
        return {category: counts[category] for category in ("answerable", "refusal", "speak_as", "preaching")}

    all_rows = train + valid
    origins: Counter[str] = Counter(row["metadata"]["notes_origin"] for row in all_rows)
    pairs = len({row["metadata"]["pair_id"] for row in all_rows if row["metadata"]["pair_id"]})
    return {
        "train": {"n": len(train), "by_category": tally(train)},
        "valid": {"n": len(valid), "by_category": tally(valid)},
        "total": len(all_rows),
        "by_category": tally(all_rows),
        "notes_origin": dict(origins),
        "contrastive_pairs": pairs,
    }


def build_and_check(data_dir: Path | None = None, root: Path | None = None) -> dict:
    root = root or repo_root()
    data_dir = data_dir or (root / "lora" / "data")
    examples, note_text = build_examples(root)
    train, valid = assign_splits(examples)
    refusal = load_refusal(root)
    report = validate_examples(train + valid, refusal=refusal, note_texts=note_text)
    if not report.ok:
        raise SystemExit("validation failed:\n" + format_report(report))
    hits = check_examples(train + valid, default_probes(root))
    if hits:
        preview = "\n".join(
            f"{hit['id']}: {hit['reason']}: {hit['probe']}" for hit in hits[:30]
        )
        raise SystemExit(f"leakage check failed ({len(hits)} hits):\n{preview}")
    stats = write_dataset(data_dir, train, valid, note_text)
    stats["validation_errors"] = 0
    stats["leakage_hits"] = 0
    return stats
