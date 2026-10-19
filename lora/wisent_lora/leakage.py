"""Leakage check against the locked prompts and a paraphrase list.

A training question leaks when it copies a protocol prompt, shares a word
4-gram with one, or is a close fuzzy match (full string or a same-length
window). The check looks at every message, so a prompt pasted into a note
fails too. Fuzzy matching is limited to the user question: the system message
is long and would otherwise dilute the ratio.
"""

from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from pathlib import Path

from wisent_lora.protocol import load_prompts, repo_root

NGRAM_N = 4
FUZZY_RATIO = 0.72
WINDOW_RATIO = 0.80


def normalize(text: str) -> str:
    text = text.lower().replace("\u2019", "'").replace("\u2018", "'")
    text = re.sub(r"[^a-z0-9'\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def word_ngrams(text: str, n: int = NGRAM_N) -> set[tuple[str, ...]]:
    words = normalize(text).split()
    if len(words) < n:
        return set()
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def _window_ratio(question: str, probe: str) -> float:
    q_words = question.split()
    p_words = probe.split()
    if len(p_words) < NGRAM_N or len(q_words) < len(p_words):
        return 0.0
    width = len(p_words)
    best = 0.0
    for start in range(len(q_words) - width + 1):
        window = " ".join(q_words[start : start + width])
        best = max(best, SequenceMatcher(None, window, probe).ratio())
        if best >= WINDOW_RATIO:
            return best
    return best


def leak_reason(question: str, probe: str, *, full_text_ngrams: set[tuple[str, ...]] | None = None) -> str | None:
    """Return a short reason if `question` (and optional extra n-grams) leaks `probe`."""
    q_norm = normalize(question)
    p_norm = normalize(probe)
    if not p_norm:
        return None
    if q_norm == p_norm or p_norm in q_norm or (q_norm and q_norm in p_norm and len(q_norm.split()) >= NGRAM_N):
        return "containment"
    probe_ngrams = word_ngrams(p_norm)
    question_ngrams = word_ngrams(q_norm)
    overlap = probe_ngrams & question_ngrams
    if overlap:
        joined = " ".join(next(iter(overlap)))
        return f"4-gram:{joined}"
    if full_text_ngrams is not None:
        extra = probe_ngrams & full_text_ngrams
        if extra:
            joined = " ".join(next(iter(extra)))
            return f"4-gram:{joined}"
    if q_norm and p_norm:
        ratio = SequenceMatcher(None, q_norm, p_norm).ratio()
        if ratio >= FUZZY_RATIO:
            return f"fuzzy:{ratio:.2f}"
        window = _window_ratio(q_norm, p_norm)
        if window >= WINDOW_RATIO:
            return f"window:{window:.2f}"
    return None


def load_paraphrases(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    lines = data.get("paraphrases", data if isinstance(data, list) else None)
    if not isinstance(lines, list) or not lines:
        raise ValueError(f"{path} must contain a non-empty paraphrases list")
    texts = []
    for item in lines:
        if isinstance(item, str):
            texts.append(item)
        elif isinstance(item, dict) and isinstance(item.get("text"), str):
            texts.append(item["text"])
        else:
            raise ValueError(f"Bad paraphrase entry in {path}: {item!r}")
    return texts


def check_examples(
    examples: list[dict],
    probes: list[str],
) -> list[dict]:
    """Return one record per leak. Empty means the set is clean."""
    hits = []
    for example in examples:
        messages = example.get("messages", [])
        user = ""
        blobs = []
        for message in messages:
            content = message.get("content", "")
            blobs.append(content)
            if message.get("role") == "user":
                user = content
        full_ngrams: set[tuple[str, ...]] = set()
        for blob in blobs:
            full_ngrams |= word_ngrams(blob)
        example_id = example.get("metadata", {}).get("id", "?")
        for probe in probes:
            reason = leak_reason(user, probe, full_text_ngrams=full_ngrams)
            if reason:
                hits.append(
                    {
                        "id": example_id,
                        "reason": reason,
                        "probe": probe,
                        "question": user,
                    }
                )
    return hits


def default_probes(root: Path | None = None) -> list[str]:
    root = root or repo_root()
    prompts = [item["text"] for item in load_prompts(root)]
    paraphrase_path = root / "lora" / "data" / "held_out_paraphrases.json"
    paraphrases = load_paraphrases(paraphrase_path)
    return prompts + paraphrases


def check_dataset_dir(data_dir: Path, root: Path | None = None) -> list[dict]:
    probes = default_probes(root)
    hits = []
    for name in ("train.jsonl", "valid.jsonl"):
        path = data_dir / name
        examples = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        for hit in check_examples(examples, probes):
            hit["split"] = name.replace(".jsonl", "")
            hits.append(hit)
    return hits
