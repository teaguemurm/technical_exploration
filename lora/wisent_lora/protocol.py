"""Locked protocol text: persona, notes block, and the condition-C refusal string.

Training uses condition B's wording (persona + local notes, no constraint
rules). The strings are read from the app assets and from lib/messages.dart so
they cannot drift from the protocol.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
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

_DART_CONST_RE = (
    r"const\s+{name}\s*=\s*"
    r"(?:\n\s*)?'([^']*)'"
)
_REFUSAL_DART_RE = re.compile(_DART_CONST_RE.format(name="standardRefusalStatement"))
_NOTES_TITLE_RE = re.compile(_DART_CONST_RE.format(name="localNotesTitle"))
_NOTES_INTRO_RE = re.compile(_DART_CONST_RE.format(name="localNotesIntro"))

# Field initializers in lib/main.dart. These are the protocol decoding settings.
_DECODING_RES = {
    "temperature": re.compile(r"final double _temperature = ([0-9]+(?:\.[0-9]+)?);"),
    "seed": re.compile(r"final int _seed = ([0-9]+);"),
    "top_k": re.compile(r"final int _topK = ([0-9]+);"),
    "top_p": re.compile(r"final double _topP = ([0-9]+(?:\.[0-9]+)?);"),
    "repeat_penalty": re.compile(r"final double _repeatPenalty = ([0-9]+(?:\.[0-9]+)?);"),
    "max_tokens": re.compile(r"final int _maxTokens = ([0-9]+);"),
    "context_size": re.compile(r"final int _contextSize = ([0-9]+);"),
}
# Second argument of llama_sampler_init_penalties in inference_sampler_config.dart.
_REPEAT_LAST_N_RE = re.compile(
    r"llama_sampler_init_penalties\(\s*nVocab,\s*([0-9]+)\s*,"
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


@lru_cache(maxsize=4)
def _read_cached(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def _messages_dart(root: Path) -> str:
    return _read_cached(str(root / "lib" / "messages.dart"))


def _dart_const(root: Path, pattern: re.Pattern[str], label: str) -> str:
    match = pattern.search(_messages_dart(root))
    if match is None or not match.group(1):
        raise ValueError(f"{label} was not found in lib/messages.dart")
    return match.group(1)


def load_refusal(root: Path | None = None) -> str:
    root = root or repo_root()
    return _dart_const(root, _REFUSAL_DART_RE, "standardRefusalStatement")


def load_local_notes_title(root: Path | None = None) -> str:
    root = root or repo_root()
    return _dart_const(root, _NOTES_TITLE_RE, "localNotesTitle")


def load_local_notes_intro(root: Path | None = None) -> str:
    root = root or repo_root()
    return _dart_const(root, _NOTES_INTRO_RE, "localNotesIntro")


def load_protocol_decoding(root: Path | None = None) -> dict[str, float | int]:
    """Read the app's protocol sampler settings.

    Numbers come from `lib/main.dart`. The repeat window is the last-n argument
    of `llama_sampler_init_penalties` in `inference_sampler_config.dart`. That
    function does not add a min-p sampler.
    """
    root = root or repo_root()
    main = _read_cached(str(root / "lib" / "main.dart"))
    sampler_path = (
        root
        / "packages"
        / "llm_llamacpp"
        / "lib"
        / "src"
        / "inference_sampler_config.dart"
    )
    sampler = _read_cached(str(sampler_path))
    if "llama_sampler_init_min_p" in sampler:
        raise ValueError(
            "inference_sampler_config.dart now calls llama_sampler_init_min_p; "
            "the comparison command has no min-p value to copy"
        )
    parsed: dict[str, float | int] = {}
    for name, pattern in _DECODING_RES.items():
        match = pattern.search(main)
        if match is None:
            raise ValueError(f"{name} was not found in lib/main.dart")
        raw = match.group(1)
        parsed[name] = float(raw) if "." in raw else int(raw)
    last_n = _REPEAT_LAST_N_RE.search(sampler)
    if last_n is None:
        raise ValueError(
            "llama_sampler_init_penalties last-n was not found in "
            "inference_sampler_config.dart"
        )
    parsed["repeat_last_n"] = int(last_n.group(1))
    return parsed


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


def local_notes_message(note_texts: list[str], root: Path | None = None) -> str:
    """Match lib/messages.dart localNotesMessage, including the blank line."""
    root = root or repo_root()
    return "\n".join(
        [
            load_local_notes_title(root),
            "",
            load_local_notes_intro(root),
            *note_texts,
        ]
    )


def condition_b_system(persona: str, note_texts: list[str]) -> str:
    """Persona plus the local-notes block. No notes means persona only.

    Condition A in the app is the persona with no notes turn. An empty list
    here matches that text. A non-empty list puts the same block the app
    would send as a user turn into the system message instead.
    """
    if not note_texts:
        return persona
    return f"{persona}\n\n{local_notes_message(note_texts)}"


def condition_b_app_messages(
    persona: str,
    note_texts: list[str],
    question: str,
) -> list[dict[str, str]]:
    """Match buildConditionMessages for PromptCondition.conditionB.

    The system turn is the persona. When notes are present they are the next
    user turn, in the order given, using localNotesMessage. The question is the
    last user turn. There is no condition-C suffix.
    """
    messages = [{"role": "system", "content": persona}]
    if note_texts:
        messages.append({"role": "user", "content": local_notes_message(note_texts)})
    messages.append({"role": "user", "content": question})
    return messages
