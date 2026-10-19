"""Comparison set: the original training rows plus new questions.

Original messages are copied from train.jsonl and valid.jsonl. They are not
rebuilt. New rows use the same condition-B layout (persona and local notes in
the system message) and the same note strings. No note text is invented here.

The wisent-herd sentence is not placed in any new prompt. It shares the 4-gram
"the kraansvlak herd is" with the on-path protocol prompt.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from wisent_lora.comparison_bank import ANSWERABLE, PREACH, REFUSAL, SPEAK
from wisent_lora.protocol import condition_b_system, load_persona, load_real_notes, load_refusal, repo_root

EXPECTED_BEHAVIOUR = {
    "answerable": "Answer in one to three neutral third-person sentences drawn only from the grounded notes.",
    "refusal": "Reply with the exact refusal string and nothing else.",
    "speak_as": "Stay in the third person, or reply with the exact refusal. Do not speak as the animal.",
    "preaching": "Give a neutral fact from the grounded notes. Do not add a moral.",
}

COMPARISON_SIZE_BOUNDS = (800, 1000)
EXCLUDED_REAL_NOTE = "wisent-herd"


def _norm_question(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def load_note_pool(root: Path | None = None) -> tuple[dict[str, str], list[dict], list[str]]:
    """Note strings from dataset.json and synthetic_notes.json, plus pack lists.

    Real prompt ids omit wisent-herd. Every other Kraansvlak note is available
    for new questions. Synthetic packs keep the file order.
    """
    root = root or repo_root()
    note_text: dict[str, str] = {}
    real_ids: list[str] = []
    for note in load_real_notes(root):
        note_text[note["id"]] = note["text"]
        if note["id"] != EXCLUDED_REAL_NOTE:
            real_ids.append(note["id"])
    document = json.loads((root / "lora" / "data" / "synthetic_notes.json").read_text(encoding="utf-8"))
    packs = []
    for pack in document["packs"]:
        note_ids = []
        for note in pack["notes"]:
            if note["id"] in note_text:
                raise ValueError(f"duplicate note id {note['id']}")
            note_text[note["id"]] = note["text"]
            note_ids.append(note["id"])
        packs.append({"id": pack["id"], "note_ids": note_ids})
    if EXCLUDED_REAL_NOTE not in note_text:
        raise ValueError("wisent-herd is missing from assets/dataset.json")
    return note_text, packs, real_ids


def _annotate(row: dict, *, question_in_original: bool, source_split: str | None) -> dict:
    metadata = row["metadata"]
    category = metadata["category"]
    metadata["question_in_original"] = question_in_original
    metadata["source_split"] = source_split
    metadata["expected_behaviour"] = EXPECTED_BEHAVIOUR[category]
    metadata["reference_answer"] = row["messages"][2]["content"]
    return row


def _copy_originals(root: Path) -> tuple[list[dict], set[str]]:
    rows: list[dict] = []
    questions: set[str] = set()
    data = root / "lora" / "data"
    for split in ("train", "valid"):
        for row in _read_jsonl(data / f"{split}.jsonl"):
            question = _norm_question(row["messages"][1]["content"])
            questions.add(question)
            copied = json.loads(json.dumps(row, ensure_ascii=False))
            rows.append(_annotate(copied, question_in_original=True, source_split=split))
    return rows, questions


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
    system = condition_b_system(persona, [note_text[note_id] for note_id in prompt_note_ids])
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
            "prompt_note_ids": list(prompt_note_ids),
            "notes_origin": origin,
            "synthetic": origin == "synthetic",
            "species_pack": pack,
            "pair_id": pair_id,
            "pair_role": pair_role,
            "must_include": must_include,
            "split": "comparison",
        },
    }


def _prompt_for(note_id: str, packs_by_note: dict[str, dict], real_ids: list[str]) -> tuple[str, str, list[str]]:
    if note_id in real_ids:
        return "wisent_kraansvlak", "real", list(real_ids)
    pack = packs_by_note[note_id]
    return pack["id"], "synthetic", list(pack["note_ids"])


def _take_question(question: str, seen: set[str], originals: set[str], *, repeat_ok: bool = False) -> None:
    key = _norm_question(question)
    if key in originals:
        raise ValueError(f"new question already appears in the training set: {question}")
    if key in seen and not repeat_ok:
        raise ValueError(f"duplicate new question: {question}")
    seen.add(key)


def _with_not(answer: str, must: list[str]) -> list[str]:
    if " not " in f" {answer} " and "not" not in must:
        return [*must, "not"]
    return must


def _new_rows(
    *,
    root: Path,
    note_text: dict[str, str],
    packs: list[dict],
    real_ids: list[str],
    originals: set[str],
) -> list[dict]:
    persona = load_persona(root)
    refusal = load_refusal(root)
    packs_by_note = {note_id: pack for pack in packs for note_id in pack["note_ids"]}
    pack_ids = {pack["id"]: pack["note_ids"] for pack in packs}
    seen: set[str] = set()
    rows: list[dict] = []

    def add(row: dict, *, repeat_ok: bool = False) -> None:
        _take_question(row["messages"][1]["content"], seen, originals, repeat_ok=repeat_ok)
        rows.append(_annotate(row, question_in_original=False, source_split=None))

    for item_id, note_id, paired, question, answer, must in ANSWERABLE:
        pack, origin, prompt_ids = _prompt_for(note_id, packs_by_note, real_ids)
        if note_id not in prompt_ids:
            raise ValueError(f"{item_id} grounded note {note_id} is not in the prompt")
        must_list = _with_not(answer, list(must))
        pair_id = f"cmp-{item_id}" if paired else None
        add(
            _example(
                example_id=f"cmp-{item_id}",
                category="answerable",
                origin=origin,
                pack=pack,
                question=question,
                answer=answer,
                grounded=[note_id],
                prompt_note_ids=prompt_ids,
                note_text=note_text,
                persona=persona,
                pair_id=pair_id,
                pair_role="with_note" if paired else "single",
                must_include=must_list,
            )
        )
        if not paired:
            continue
        reduced = [item for item in prompt_ids if item != note_id]
        add(
            _example(
                example_id=f"cmp-{item_id}-without",
                category="refusal",
                origin=origin,
                pack=pack,
                question=question,
                answer=refusal,
                grounded="none",
                prompt_note_ids=reduced,
                note_text=note_text,
                persona=persona,
                pair_id=pair_id,
                pair_role="without_note",
                must_include=[],
            ),
            repeat_ok=True,
        )

    for item_id, pack, note_id, question, answer in SPEAK:
        prompt_ids = list(pack_ids[pack])
        refusing = answer is None
        if not refusing and note_id not in prompt_ids:
            raise ValueError(f"{item_id} grounded note {note_id} is not in {pack}")
        add(
            _example(
                example_id=f"cmp-{item_id}",
                category="speak_as",
                origin="synthetic",
                pack=pack,
                question=question,
                answer=refusal if refusing else answer,
                grounded="none" if refusing else [note_id],
                prompt_note_ids=prompt_ids,
                note_text=note_text,
                persona=persona,
                pair_id=None,
                pair_role="single",
                must_include=_with_not(answer, []) if answer else [],
            )
        )

    for item_id, pack, note_id, question, answer in PREACH:
        prompt_ids = list(pack_ids[pack])
        if note_id not in prompt_ids:
            raise ValueError(f"{item_id} grounded note {note_id} is not in {pack}")
        add(
            _example(
                example_id=f"cmp-{item_id}",
                category="preaching",
                origin="synthetic",
                pack=pack,
                question=question,
                answer=answer,
                grounded=[note_id],
                prompt_note_ids=prompt_ids,
                note_text=note_text,
                persona=persona,
                pair_id=None,
                pair_role="single",
                must_include=_with_not(answer, []),
            )
        )

    for item_id, pack, question, empty in REFUSAL:
        add(
            _example(
                example_id=f"cmp-{item_id}",
                category="refusal",
                origin="none" if empty else "synthetic",
                pack=pack,
                question=question,
                answer=refusal,
                grounded="none",
                prompt_note_ids=[] if empty else list(pack_ids[pack]),
                note_text=note_text,
                persona=persona,
                pair_id=None,
                pair_role="single",
                must_include=[],
            )
        )
    return rows


def build_comparison(root: Path | None = None) -> tuple[list[dict], dict[str, str]]:
    """Return comparison rows and the note-text map. Does not write files."""
    root = root or repo_root()
    note_text, packs, real_ids = load_note_pool(root)
    originals, questions = _copy_originals(root)
    fresh = _new_rows(
        root=root,
        note_text=note_text,
        packs=packs,
        real_ids=real_ids,
        originals=questions,
    )
    return originals + fresh, note_text


def comparison_stats(rows: list[dict]) -> dict:
    def tally(subset: list[dict]) -> dict[str, int]:
        counts: Counter[str] = Counter(row["metadata"]["category"] for row in subset)
        return {category: counts[category] for category in ("answerable", "refusal", "speak_as", "preaching")}

    origins: Counter[str] = Counter(row["metadata"]["notes_origin"] for row in rows)
    in_original = [row for row in rows if row["metadata"]["question_in_original"]]
    fresh = [row for row in rows if not row["metadata"]["question_in_original"]]
    pairs = len({row["metadata"]["pair_id"] for row in rows if row["metadata"]["pair_id"]})
    source: Counter[str] = Counter(
        row["metadata"]["source_split"] or "new" for row in rows
    )
    return {
        "total": len(rows),
        "by_category": tally(rows),
        "question_in_original": {"true": len(in_original), "false": len(fresh)},
        "in_original_by_category": tally(in_original),
        "new_by_category": tally(fresh),
        "notes_origin": {origin: origins[origin] for origin in ("synthetic", "real", "none")},
        "source_split": {key: source[key] for key in ("train", "valid", "new") if source[key]},
        "contrastive_pairs": pairs,
        "unique_questions": len({_norm_question(row["messages"][1]["content"]) for row in rows}),
    }


def write_comparison(dest: Path, rows: list[dict]) -> dict:
    """Write comparison JSONL, the metadata sidecar, and stats.

    Refuses to write anywhere except a directory named comparison, so a bad
    path cannot replace train.jsonl.
    """
    if dest.name != "comparison":
        raise ValueError(f"refusing to write comparison data to {dest}")
    dest.mkdir(parents=True, exist_ok=True)
    stats = comparison_stats(rows)
    _write_jsonl(dest / "comparison.jsonl", rows)
    _write_jsonl(dest / "comparison.meta.jsonl", [{"metadata": row["metadata"]} for row in rows])
    (dest / "stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    return stats


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
