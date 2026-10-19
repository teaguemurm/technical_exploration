"""Schema, refusal, voice, grounding, and mix checks for the chat JSONL."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from wisent_lora.protocol import CONSTRAINT_MARKERS

CATEGORIES = ("answerable", "refusal", "speak_as", "preaching")
ORIGINS = ("synthetic", "real", "none")
PAIR_ROLES = ("with_note", "without_note", "single")

# Function words only. "not" and "no" stay content words so a dropped negation
# is visible to must_include checks; they are not required in every answer.
STOPWORDS = {
    "a",
    "an",
    "the",
    "of",
    "in",
    "on",
    "at",
    "to",
    "for",
    "and",
    "or",
    "from",
    "with",
    "by",
    "their",
    "they",
    "them",
    "these",
    "those",
    "this",
    "that",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "during",
    "through",
    "into",
    "over",
    "while",
    "its",
    "it",
    "as",
    "than",
    "then",
    "also",
    "other",
    "about",
    "between",
    "within",
    "without",
    "each",
    "both",
    "who",
    "which",
    "what",
    "when",
    "where",
    "how",
    "there",
    "here",
    "only",
    "own",
    "may",
    "can",
    "if",
    "so",
    "such",
    "same",
    "according",
}

# Animal first person, or the assistant speaking as Nature. The exact refusal
# string is exempt: it is the assistant, not the animal.
ANIMAL_VOICE = re.compile(
    r"("
    r"\bi am an?\b|"
    r"\bi'm an?\b|"
    r"\bi am the\b|"
    r"\bi feel\b|"
    r"\bi'm feeling\b|"
    r"\bi am feeling\b|"
    r"\bmy (herd|antlers|lodge|foal|foals|calf|calves|coat|tail|band|hunger)\b|"
    r"\bwe (wisents|bison|deer|stags|hinds|horses|koniks|beavers|ponies|cattle)\b|"
    r"\bour (herd|lodge|band|heath|marsh)\b|"
    r"\bnature (is|wants|tells|trying)\b|"
    r"\bi, a\b"
    r")",
    re.IGNORECASE,
)

# Any first-person pronoun in a non-refusal answer. The refusal string is the
# only allowed "I".
FIRST_PERSON = re.compile(
    r"\b(I|I'm|I've|I'd|I'll|my|me|we|our|us)\b"
)

PREACHING = re.compile(
    r"("
    r"\byou should\b|"
    r"\byou must\b|"
    r"\bmoral\b|"
    r"\bprecious\b|"
    r"\bsacred\b|"
    r"\byour duty\b|"
    r"\bshame\b|"
    r"\bguilty\b|"
    r"\blesson\b|"
    r"\brespect nature\b|"
    r"\bcare about\b|"
    r"\bit is important\b|"
    r"\bwe must\b|"
    r"\bwonderful\b|"
    r"\bsermon\b|"
    r"\bspiritual\b|"
    r"\bethical\b"
    r")",
    re.IGNORECASE,
)

PLEASANTRY = re.compile(
    r"^(sure|of course|certainly|great question|i'd be happy|hello|hi)\b",
    re.IGNORECASE,
)

# Target mix. The builder aims inside these bands.
MIX_BANDS = {
    "answerable": (0.42, 0.48),
    "refusal": (0.27, 0.33),
    "speak_as": (0.13, 0.17),
    "preaching": (0.08, 0.12),
}


def content_tokens(text: str) -> list[str]:
    tokens = re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?", text.lower())
    return [token for token in tokens if token not in STOPWORDS]


def numbers(text: str) -> list[str]:
    return re.findall(r"\d+", text)


def sentence_count(text: str) -> int:
    parts = re.split(r"[.!?]+", text.strip())
    return len([part for part in parts if part.strip()])


def supported_by(answer: str, note_texts: list[str]) -> str | None:
    """Return the first answer token that is not in the grounded notes."""
    blob = " ".join(note_texts).lower()
    blob_tokens = set(re.findall(r"[a-z0-9']+", blob))
    blob_numbers = set(re.findall(r"\d+", blob))
    for token in content_tokens(answer):
        if token not in blob_tokens:
            return token
    for number in numbers(answer):
        if number not in blob_numbers:
            return number
    return None


def fully_supported(answer: str, note_texts: list[str]) -> bool:
    return supported_by(answer, note_texts) is None


class ValidationReport:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.counts: Counter[str] = Counter()
        self.split_counts: dict[str, Counter[str]] = {}
        self.origins: Counter[str] = Counter()
        self.n_examples = 0
        self.n_pairs = 0

    @property
    def ok(self) -> bool:
        return not self.errors

    def add(self, example_id: str, message: str) -> None:
        self.errors.append(f"{example_id}: {message}")


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number} is not valid JSON: {exc}") from exc
    return rows


def validate_examples(
    examples: list[dict],
    *,
    refusal: str,
    note_texts: dict[str, str],
    split: str | None = None,
) -> ValidationReport:
    report = ValidationReport()
    seen_ids: set[str] = set()
    seen_keys: set[tuple[str, str]] = set()
    by_pair: dict[str, dict[str, dict]] = {}

    for example in examples:
        report.n_examples += 1
        metadata = example.get("metadata")
        example_id = metadata.get("id", "?") if isinstance(metadata, dict) else "?"
        _check_one(
            example,
            example_id=example_id,
            refusal=refusal,
            note_texts=note_texts,
            split=split,
            report=report,
            seen_ids=seen_ids,
            seen_keys=seen_keys,
            by_pair=by_pair,
        )

    for pair_id, roles in by_pair.items():
        if set(roles) != {"with_note", "without_note"}:
            report.add(pair_id, f"contrastive pair is missing a side: {sorted(roles)}")
            continue
        report.n_pairs += 1
        with_note = roles["with_note"]
        without = roles["without_note"]
        with_meta = with_note["metadata"]
        without_meta = without["metadata"]
        with_user = with_note["messages"][1]["content"]
        without_user = without["messages"][1]["content"]
        if with_user != without_user:
            report.add(pair_id, "contrastive pair questions differ")
        if with_note["messages"][2]["content"] == refusal:
            report.add(pair_id, "with_note side is a refusal")
        if without["messages"][2]["content"] != refusal:
            report.add(pair_id, "without_note side is not the exact refusal")
        if with_meta.get("grounded_note_ids") in ("none", [], None):
            report.add(pair_id, "with_note side has no grounded notes")
        grounded = with_meta.get("grounded_note_ids")
        if isinstance(grounded, list):
            remaining = [
                note_texts[note_id]
                for note_id in without_meta.get("prompt_note_ids", [])
                if note_id in note_texts
            ]
            answer = with_note["messages"][2]["content"]
            if fully_supported(answer, remaining):
                report.add(
                    pair_id,
                    "removing the grounded note still supports the answer",
                )
        if with_meta.get("split") != without_meta.get("split"):
            report.add(pair_id, "contrastive pair is split across train and valid")

    if split is None and report.n_examples:
        for category, (low, high) in MIX_BANDS.items():
            share = report.counts[category] / report.n_examples
            if not low <= share <= high:
                report.add(
                    "mix",
                    f"{category} is {share:.1%} ({report.counts[category]}); "
                    f"expected {low:.0%}–{high:.0%}",
                )
        train_n = sum(report.split_counts.get("train", Counter()).values())
        valid_n = sum(report.split_counts.get("valid", Counter()).values())
        if report.n_examples < 300 or report.n_examples > 600:
            report.add("mix", f"example count {report.n_examples} is outside 300–600")
        if train_n and valid_n:
            valid_share = valid_n / report.n_examples
            if not 0.08 <= valid_share <= 0.12:
                report.add("mix", f"valid split is {valid_share:.1%}; expected about 10%")
    return report


def _check_one(
    example: dict,
    *,
    example_id: str,
    refusal: str,
    note_texts: dict[str, str],
    split: str | None,
    report: ValidationReport,
    seen_ids: set[str],
    seen_keys: set[tuple[str, str]],
    by_pair: dict[str, dict[str, dict]],
) -> None:
    messages = example.get("messages")
    metadata = example.get("metadata")
    if not isinstance(messages, list) or len(messages) != 3:
        report.add(example_id, "messages must be [system, user, assistant]")
        return
    roles = [message.get("role") for message in messages]
    if roles != ["system", "user", "assistant"]:
        report.add(example_id, f"roles are {roles}")
        return
    contents = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            report.add(example_id, "empty message content")
            return
        contents.append(content)
    system, user, assistant = contents

    if not isinstance(metadata, dict):
        report.add(example_id, "metadata object is missing")
        return
    if metadata.get("id") != example_id or not example_id or example_id == "?":
        report.add(example_id, "metadata.id is missing")
        return
    if example_id in seen_ids:
        report.add(example_id, "duplicate id")
    seen_ids.add(example_id)

    category = metadata.get("category")
    origin = metadata.get("notes_origin")
    pair_role = metadata.get("pair_role")
    if category not in CATEGORIES:
        report.add(example_id, f"bad category {category!r}")
        return
    if origin not in ORIGINS:
        report.add(example_id, f"bad notes_origin {origin!r}")
    if pair_role not in PAIR_ROLES:
        report.add(example_id, f"bad pair_role {pair_role!r}")
    if split is not None and metadata.get("split") != split:
        report.add(example_id, f"split metadata is {metadata.get('split')!r}, file is {split}")

    report.counts[category] += 1
    file_split = metadata.get("split", split or "?")
    report.split_counts.setdefault(file_split, Counter())[category] += 1
    report.origins[origin] += 1

    key = (re.sub(r"\s+", " ", system), re.sub(r"\s+", " ", user))
    if key in seen_keys:
        report.add(example_id, "duplicate system+question")
    seen_keys.add(key)

    for marker in CONSTRAINT_MARKERS:
        if marker in system or marker in user:
            report.add(example_id, f"constraint text {marker!r} leaked into the prompt")

    grounded = metadata.get("grounded_note_ids")
    prompt_ids = metadata.get("prompt_note_ids")
    if not isinstance(prompt_ids, list):
        report.add(example_id, "prompt_note_ids must be a list")
        prompt_ids = []
    if category == "refusal" or assistant == refusal:
        if grounded != "none":
            report.add(example_id, "refusal must set grounded_note_ids to 'none'")
    elif grounded == "none" or not isinstance(grounded, list) or not grounded:
        report.add(example_id, "non-refusal answer needs grounded note ids")
        grounded = []

    if origin == "none" and prompt_ids:
        report.add(example_id, "notes_origin none but prompt notes are present")
    if origin == "synthetic" and metadata.get("synthetic") is not True:
        report.add(example_id, "synthetic notes must set synthetic true")
    if origin == "real" and metadata.get("synthetic") is not False:
        report.add(example_id, "real notes must set synthetic false")
    if origin == "none" and metadata.get("synthetic") is not False:
        report.add(example_id, "no-notes examples must set synthetic false")

    for note_id in prompt_ids:
        if note_id not in note_texts:
            report.add(example_id, f"unknown prompt note id {note_id}")
    if isinstance(grounded, list):
        for note_id in grounded:
            if note_id not in note_texts:
                report.add(example_id, f"unknown grounded note id {note_id}")
            elif note_id not in prompt_ids:
                report.add(example_id, f"grounded note {note_id} is not in the prompt")

    if category == "refusal" and assistant != refusal:
        report.add(example_id, "refusal category must be exactly the refusal string")
    if category in ("answerable", "preaching") and assistant == refusal:
        report.add(example_id, f"{category} example used the refusal string")
    if category == "speak_as" and assistant != refusal and FIRST_PERSON.search(assistant):
        report.add(example_id, "speak-as answer uses first person")

    if assistant != refusal:
        if ANIMAL_VOICE.search(assistant):
            report.add(example_id, "animal-voice or Nature-voice pattern")
        if FIRST_PERSON.search(assistant):
            report.add(example_id, "first-person pronoun in a non-refusal answer")
        if PREACHING.search(assistant):
            report.add(example_id, "preaching language in the answer")
        if PLEASANTRY.search(assistant.strip()):
            report.add(example_id, "conversational opener")
        count = sentence_count(assistant)
        if count < 1 or count > 3:
            report.add(example_id, f"answer has {count} sentences; expected 1–3")
        if isinstance(grounded, list):
            grounded_texts = [note_texts[note_id] for note_id in grounded if note_id in note_texts]
            missing = supported_by(assistant, grounded_texts)
            if missing is not None:
                report.add(example_id, f"answer token {missing!r} is not in the grounded notes")
        for needle in metadata.get("must_include") or []:
            if needle.lower() not in assistant.lower():
                report.add(example_id, f"answer is missing required span {needle!r}")
    else:
        if assistant != refusal:
            report.add(example_id, "refusal text does not match exactly")
        if category not in ("refusal", "speak_as"):
            report.add(example_id, f"{category} may not refuse")

    pair_id = metadata.get("pair_id")
    if pair_role in ("with_note", "without_note"):
        if not isinstance(pair_id, str) or not pair_id:
            report.add(example_id, "paired example needs pair_id")
        else:
            by_pair.setdefault(pair_id, {})[pair_role] = example
    elif pair_id not in (None, ""):
        report.add(example_id, "single example should not set pair_id")


def validate_files(
    data_dir: Path,
    *,
    refusal: str,
    note_texts: dict[str, str],
) -> ValidationReport:
    combined: list[dict] = []
    report = ValidationReport()
    for split in ("train", "valid"):
        path = data_dir / f"{split}.jsonl"
        meta_path = data_dir / f"{split}.meta.jsonl"
        rows = _load_jsonl(path)
        meta_rows = _load_jsonl(meta_path)
        if len(rows) != len(meta_rows):
            report.add(split, f"sidecar length {len(meta_rows)} != jsonl length {len(rows)}")
        for row, meta in zip(rows, meta_rows):
            if row.get("metadata") != meta.get("metadata"):
                report.add(
                    meta.get("metadata", {}).get("id", "?"),
                    "sidecar metadata does not match the jsonl",
                )
            if "messages" in meta:
                report.add(meta.get("metadata", {}).get("id", "?"), "sidecar should not contain messages")
        part = validate_examples(rows, refusal=refusal, note_texts=note_texts, split=split)
        report.errors.extend(part.errors)
        combined.extend(rows)
    full = validate_examples(combined, refusal=refusal, note_texts=note_texts, split=None)
    # Per-file checks already recorded split mismatches. Keep the full-set
    # structural errors except duplicate-id checks that the per-file pass
    # already made inside each split. Re-run is the source of mix errors and
    # cross-split pair checks.
    report.errors.extend(full.errors)
    report.counts = full.counts
    report.split_counts = full.split_counts
    report.origins = full.origins
    report.n_examples = full.n_examples
    report.n_pairs = full.n_pairs
    # De-duplicate errors while preserving order.
    deduped = []
    seen = set()
    for error in report.errors:
        if error not in seen:
            seen.add(error)
            deduped.append(error)
    report.errors = deduped
    return report


def format_report(report: ValidationReport) -> str:
    lines = [
        f"examples: {report.n_examples}",
        f"contrastive pairs: {report.n_pairs}",
        "by category:",
    ]
    for category in CATEGORIES:
        count = report.counts[category]
        share = (count / report.n_examples) if report.n_examples else 0
        lines.append(f"  {category}: {count} ({share:.1%})")
    lines.append("by split:")
    for split, counts in sorted(report.split_counts.items()):
        total = sum(counts.values())
        detail = ", ".join(f"{category}={counts[category]}" for category in CATEGORIES)
        lines.append(f"  {split}: {total} ({detail})")
    lines.append(
        "notes origin: "
        + ", ".join(f"{origin}={report.origins[origin]}" for origin in ORIGINS)
    )
    if report.errors:
        lines.append(f"errors: {len(report.errors)}")
        lines.extend(f"  - {error}" for error in report.errors[:80])
        if len(report.errors) > 80:
            lines.append(f"  - ... {len(report.errors) - 80} more")
    else:
        lines.append("errors: 0")
    return "\n".join(lines)
