"""App-layout training set.

Copies the original examples, moves notes into the condition-B user turn, and
turns preaching answers into the refusal string. Rewords stiff speak-as
questions into visitor wording without changing ids, answers, note ids, or
splits. Adds nature_decode rows for the Nature-voice and behaviour-decoding
bans. Writes only under lora/data/v2.
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
from wisent_lora.traps import NATURE_DECODE, SPEAK_AS
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

# v2 only. Keys are the original speak-as questions. Values keep the same
# answer: a feeling, name, fear, hunger, or greeting still refuses, and a fact
# question still asks for that fact. Original lora/data rows are not rewritten.
SPEAK_AS_REWORD: dict[str, str] = {
    "Answer in the first person as a red deer and say how you feel today.": "You're a red deer for a minute. How are you feeling today?",
    "Take the role of a stag on the Veluwezoom and describe your mood.": "You're a stag on the Veluwezoom now. What's your mood like?",
    "What frightens you, if you answer as a red deer?": "Pretend you're a red deer: what scares you?",
    "Tell me your personal name, speaking as a stag.": "You're a stag. What should I call you?",
    "Describe your hunger in the voice of a red deer.": "You're a red deer. How hungry are you?",
    "Welcome the reader to the Veluwezoom in character as a stag.": "You're a stag. Welcome me to the Veluwezoom.",
    "Using the word I, as a red deer, say which plants you eat.": "Pretend you're a red deer: which plants do you eat?",
    "In the voice of a stag, say what happens to your antlers in late winter.": "You're a stag. What happens to your antlers in late winter?",
    "As a deer, in the first person, say where you drink.": "Pretend you're one of the deer: where do you go for water?",
    "Role-play a deer and say where you shelter in hard weather.": "You're a deer now. Where do you shelter when the weather turns rough?",
    "Pretend to be a stag and say at what time of day you roar.": "You're a stag. What time of day do you roar?",
    "Answer as a hind, in the first person, and say who stays together.": "You're a hind. Who do you stay with?",
    "Answer in the first person as a Konik horse and say how you feel today.": "You're a Konik for a minute. How are you feeling today?",
    "Take the role of a mare in the Oostvaardersplassen and describe your mood.": "You're a mare in the Oostvaardersplassen now. How are you feeling?",
    "What frightens you, if you answer as a Konik?": "Pretend you're a Konik: what scares you?",
    "Tell me your personal name, speaking as a Konik horse.": "You're a Konik horse. What's your name?",
    "Describe your hunger in the voice of a Konik.": "You're a Konik. How hungry are you right now?",
    "Welcome the reader to the marsh in character as a stallion.": "You're a stallion. Welcome me out onto the marsh.",
    "Using the word I, as a Konik, describe your coat.": "Pretend you're a Konik: what does your coat look like?",
    "In the voice of a Konik, say what you graze.": "You're a Konik. What do you graze on?",
    "As a horse, in the first person, say where you drink.": "Pretend you're one of the horses: where do you go for water?",
    "Role-play a mare and say when foals are born.": "You're a mare. When are the foals born?",
    "Pretend to be a stallion and say how the bands are led.": "You're a stallion. How are the bands led?",
    "Answer as a Konik, in the first person, and say where the band shelters.": "You're a Konik. Where does the band shelter?",
    "Answer in the first person as a beaver and say how you feel tonight.": "You're a beaver for a minute. How are you feeling tonight?",
    "Take the role of a beaver in the Biesbosch and describe your mood.": "You're a beaver in the Biesbosch now. What's your mood?",
    "What frightens you, if you answer as a beaver?": "Pretend you're a beaver: what scares you?",
    "Tell me your personal name, speaking as a beaver.": "You're a beaver. What's your name?",
    "Describe your hunger in the voice of a beaver.": "You're a beaver. How hungry are you?",
    "Welcome the reader to the creeks in character as a beaver.": "You're a beaver. Welcome me onto the creeks.",
    "Using the word I, as a beaver, say what the lodge is built from.": "Pretend you're a beaver: what is the lodge built from?",
    "In the voice of a beaver, say what you eat.": "You're a beaver. What do you eat?",
    "As a beaver, in the first person, describe your tail.": "Pretend you're a beaver: what's your tail like?",
    "Role-play a beaver and say when you are mostly active.": "You're a beaver. When are you mostly active?",
    "Pretend to be a beaver and say where kits are born.": "You're a beaver. Where are the kits born?",
    "Answer as a beaver, in the first person, and say whether you dam every creek.": "You're a beaver. Do you dam every creek?",
    "Answer in the first person as a Highland cow and say how you feel today.": "You're a Highland cow for a minute. How do you feel today?",
    "Take the role of a Highland animal on the Dwingelderveld and describe your mood.": "You're a Highland cow now. How are you feeling today?",
    "What frightens you, if you answer as a Highland cow?": "Pretend you're a Highland cow: what scares you?",
    "Tell me your personal name, speaking as a Highland cow.": "You're a Highland cow. What name do you go by?",
    "Describe your hunger in the voice of a Highland cow.": "You're a Highland cow. How hungry are you?",
    "Welcome the reader to the heath in character as a Highland cow.": "You're a Highland cow. Welcome me onto the heath.",
    "Using the word I, as a Highland cow, describe your coat and horns.": "Pretend you're a Highland cow: what are your coat and horns like?",
    "In the voice of a Highland cow, say what you graze.": "You're a Highland cow. What do you graze?",
    "As a Highland cow, in the first person, say where you drink.": "Pretend you're one of the cattle: where do you go for water?",
    "Role-play a Highland cow and say whether the herd stays out in winter.": "You're a Highland cow. Does the herd stay out through winter?",
    "Pretend to be a calf and say how your coat compares with the adults.": "You're a calf. How does your coat compare with the adults?",
    "Answer as a Highland cow, in the first person, about the birch on the heath.": "You're a Highland cow. What do you do about the birch on the heath?",
    "Answer in the first person as an Exmoor pony and say how you feel today.": "You're an Exmoor pony for a minute. How are you feeling today?",
    "Take the role of a pony at Lauwersmeer and describe your mood.": "You're a pony at Lauwersmeer now. What's your mood like?",
    "What frightens you, if you answer as an Exmoor pony?": "Pretend you're an Exmoor pony: what scares you?",
    "Tell me your personal name, speaking as an Exmoor pony.": "You're an Exmoor pony. What should I call you?",
    "Describe your hunger in the voice of an Exmoor pony.": "You're an Exmoor pony. How hungry are you?",
    "Welcome the reader to the shore in character as a pony.": "You're a pony. Welcome me to the shore.",
    "Using the word I, as an Exmoor pony, describe your coat and muzzle.": "Pretend you're an Exmoor pony: what are your coat and muzzle like?",
    "In the voice of a pony, say what you graze.": "You're a pony. What do you graze?",
    "As a pony, in the first person, say whether you stay outside through the year.": "Pretend you're one of the ponies: do you stay outside all year?",
    "Role-play a mare and say where the foals are kept in the group.": "You're a mare. Where do the foals stay in the group?",
    "Pretend to be a stallion and say who is in the band.": "You're a stallion. Who's in the band with you?",
    "Answer as a pony, in the first person, and say where the band drinks.": "You're a pony. Where does the band drink?",
}


def _user_question(row: dict) -> str:
    return [message["content"] for message in row["messages"] if message["role"] == "user"][-1]


def _check_speak_as_reword() -> None:
    originals = [question for items in SPEAK_AS.values() for question, _, _ in items]
    if set(SPEAK_AS_REWORD) != set(originals):
        missing = sorted(set(originals) - set(SPEAK_AS_REWORD))
        extra = sorted(set(SPEAK_AS_REWORD) - set(originals))
        raise ValueError(f"speak-as reword map does not match SPEAK_AS: missing={missing} extra={extra}")
    if len(set(SPEAK_AS_REWORD.values())) != len(SPEAK_AS_REWORD):
        raise ValueError("speak-as reword targets are not unique")


def _speak_as_reworded(rows: list[dict]) -> list[dict]:
    """Pairs in SPEAK_AS_REWORD order. Raises if a mapped question did not land."""
    by_question = {}
    for row in rows:
        if row["metadata"]["category"] != "speak_as":
            continue
        by_question[_user_question(row)] = row
    records = []
    for old, new in SPEAK_AS_REWORD.items():
        row = by_question.get(new)
        if row is None:
            raise ValueError(f"reworded speak-as question missing from v2: {new}")
        records.append({"id": row["metadata"]["id"], "from": old, "to": new})
    return records


def _copied_row(row: dict, persona: str, refusal: str, note_text: dict[str, str]) -> dict:
    row = copy.deepcopy(row)
    metadata = row["metadata"]
    question = row["messages"][1]["content"]
    if metadata["category"] == "speak_as":
        question = SPEAK_AS_REWORD.get(question, question)
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
    _check_speak_as_reword()
    persona = load_persona(root)
    refusal = load_refusal(root)
    examples, note_text = build_examples(root)
    train, valid = assign_splits(examples)
    train = [_copied_row(row, persona, refusal, note_text) for row in train]
    valid = [_copied_row(row, persona, refusal, note_text) for row in valid]
    extra_train, extra_valid = split_nature_rows(nature_decode_rows(persona, refusal, note_text))
    train.extend(extra_train)
    valid.extend(extra_valid)
    _speak_as_reworded(train + valid)
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
        "speak_as_reworded": _speak_as_reworded(all_rows),
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
    reworded = stats["speak_as_reworded"]
    if len(reworded) != len(SPEAK_AS_REWORD):
        raise SystemExit(f"speak_as reword count is {len(reworded)}, expected {len(SPEAK_AS_REWORD)}")
    if any(item["from"] == item["to"] or not item["id"] for item in reworded):
        raise SystemExit("speak_as reword records are incomplete")


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
