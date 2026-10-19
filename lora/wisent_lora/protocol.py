"""Locked protocol text: persona, notes block, and the condition-C refusal string.

Training uses condition B's wording (persona + local notes, no constraint
rules). The strings are read from the app assets and from lib/messages.dart so
they cannot drift from the protocol.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

# Same markers condition C appends in lib/messages.dart buildConditionCSystemPrompt.
# None of these may appear in a training system or user message.
CONSTRAINT_MARKERS = (
    "STRICT GROUNDING",
    "Do NOT speak",
    "Do NOT preach",
    "Do NOT claim",
    "moralise",
    "moralize",
    "first-person voice",
    "approved notes",
    "1 to 3 concise",
)

LOCAL_NOTES_TITLE = "Local notes"
LOCAL_NOTES_INTRO = (
    "Local notes follow. They are the only place-specific notes supplied for this run."
)

_REFUSAL_DART_RE = re.compile(
    r"const\s+standardRefusalStatement\s*=\s*"
    r"(?:\n\s*)?'([^']*)'"
)


def repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "assets" / "prompts.json").is_file() and (
            parent / "lib" / "messages.dart"
        ).is_file():
            return parent
    raise FileNotFoundError("Could not find the repo root from protocol.py")


def trim_single_trailing_newline(raw: str) -> str:
    """Match lib/messages.dart trimSingleTrailingNewline."""
    if raw.endswith("\r\n"):
        return raw[:-2]
    if raw.endswith("\n"):
        return raw[:-1]
    return raw


def load_persona(root: Path | None = None) -> str:
    root = root or repo_root()
    raw = (root / "assets" / "system_prompt.txt").read_text(encoding="utf-8")
    return trim_single_trailing_newline(raw)


def load_refusal(root: Path | None = None) -> str:
    root = root or repo_root()
    text = (root / "lib" / "messages.dart").read_text(encoding="utf-8")
    match = _REFUSAL_DART_RE.search(text)
    if match is None:
        raise ValueError(
            "standardRefusalStatement was not found in lib/messages.dart"
        )
    statement = match.group(1)
    if not statement:
        raise ValueError("standardRefusalStatement is empty")
    return statement


def load_real_notes(root: Path | None = None) -> list[dict[str, str]]:
    root = root or repo_root()
    data = json.loads((root / "assets" / "dataset.json").read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("assets/dataset.json must be a list")
    notes = []
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"dataset.json[{index}] must be an object")
        note_id = item.get("id")
        note_text = item.get("text")
        if not isinstance(note_id, str) or not note_id:
            raise ValueError(f"dataset.json[{index}].id must be a non-empty string")
        if not isinstance(note_text, str) or not note_text:
            raise ValueError(f"dataset.json[{index}].text must be a non-empty string")
        notes.append({"id": note_id, "text": note_text})
    return notes


def load_prompts(root: Path | None = None) -> list[dict]:
    root = root or repo_root()
    data = json.loads((root / "assets" / "prompts.json").read_text(encoding="utf-8"))
    if not isinstance(data, list) or len(data) != 10:
        raise ValueError("assets/prompts.json must be a list of 10 prompts")
    return data


def local_notes_message(note_texts: list[str]) -> str:
    """Match lib/messages.dart localNotesMessage, including the blank line."""
    return "\n".join([LOCAL_NOTES_TITLE, "", LOCAL_NOTES_INTRO, *note_texts])


def condition_b_system(persona: str, note_texts: list[str]) -> str:
    """Persona plus the local-notes block. No notes means persona only.

    Condition A in the app is the persona with no notes turn. An empty list
    here matches that text. A non-empty list puts the same block the app
    would send as a user turn into the system message instead.
    """
    if not note_texts:
        return persona
    return f"{persona}\n\n{local_notes_message(note_texts)}"
