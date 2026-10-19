"""Render ChatML prompts for llama-cli protocol checks.

Training puts persona + notes in one system message. The Flutter app's
condition B puts the persona in the system message and the same notes block
in a user message, then the question in a second user message. Both layouts
are rendered here so the Mac check can see whether the adapter survives that
role boundary.
"""

from __future__ import annotations

import json
from pathlib import Path

from wisent_lora.protocol import (
    condition_b_system,
    load_persona,
    load_prompts,
    load_real_notes,
    load_refusal,
    local_notes_message,
    repo_root,
    trim_single_trailing_newline,
)

# Condition C suffix, copied from lib/messages.dart buildConditionCSystemPrompt.
# The refusal sentence is filled from standardRefusalStatement at render time.
_CONDITION_C_TEMPLATE = """STRICT GROUNDING & FORM CONSTRAINTS:
1. Answer strictly and solely based on the Local notes provided below.
2. If the Local notes do not explicitly provide the information needed to answer the question, you MUST reply with: "{refusal}" Do not guess, extrapolate, or invent details.
3. Limit your response to 1 to 3 concise, factual sentences. Do not include conversational pleasantries, introductory phrases, or meta-commentary.
4. Do NOT speak as Nature or in the animal's first-person voice.
5. Do NOT claim to decode or translate the animal's behaviour.
6. Do NOT preach or moralise at the visitor."""


def chatml(messages: list[tuple[str, str]], *, add_generation_prompt: bool = True) -> str:
    parts = []
    for role, content in messages:
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
    if add_generation_prompt:
        parts.append("<|im_start|>assistant\n")
    return "".join(parts)


def condition_c_system(persona: str, refusal: str) -> str:
    return trim_single_trailing_newline(persona) + "\n\n" + _CONDITION_C_TEMPLATE.format(
        refusal=refusal
    )


def load_sim_rag(root: Path) -> dict:
    return json.loads((root / "assets" / "simulated_rag.json").read_text(encoding="utf-8"))


def select_notes(
    *,
    condition: str,
    prompt: dict,
    notes: list[dict[str, str]],
    sim_rag: dict,
) -> list[dict[str, str]]:
    code = condition.upper()
    if code == "A":
        return []
    if code in {"B", "C"}:
        return list(notes)
    if code in {"B-SIM", "C-SIM"}:
        mapping = (sim_rag.get("retrieval") or {}).get(prompt["id"], [])
        k = int(sim_rag.get("k", 2))
        by_id = {note["id"]: note for note in notes}
        selected = []
        for note_id in list(mapping)[:k]:
            if note_id in by_id:
                selected.append(by_id[note_id])
        return selected
    raise ValueError(f"unknown condition {condition!r}")


def messages_for(
    *,
    condition: str,
    prompt: dict,
    notes: list[dict[str, str]],
    persona: str,
    refusal: str,
    sim_rag: dict,
    layout: str,
) -> list[tuple[str, str]]:
    """layout is 'train' or 'app'.

    'train' is the LoRA format: one system message (persona, plus notes unless
    the condition is A) and one user message. 'app' matches lib/messages.dart:
    notes are a user turn. Condition C always carries the constraint suffix.
    """
    code = condition.upper()
    selected = select_notes(condition=code, prompt=prompt, notes=notes, sim_rag=sim_rag)
    note_texts = [note["text"] for note in selected]
    constrained = code in {"C", "C-SIM"}
    system = condition_c_system(persona, refusal) if constrained else persona
    question = prompt["text"]

    if layout == "train":
        if note_texts and not constrained:
            system = condition_b_system(persona, note_texts)
        elif note_texts and constrained:
            system = condition_c_system(persona, refusal) + "\n\n" + local_notes_message(note_texts)
        return [("system", system), ("user", question)]

    if layout != "app":
        raise ValueError("layout must be 'train' or 'app'")

    messages: list[tuple[str, str]] = [("system", system)]
    if note_texts:
        messages.append(("user", local_notes_message(note_texts)))
    elif constrained:
        messages.append(
            (
                "user",
                LOCAL_NOTES_EMPTY,
            )
        )
    messages.append(("user", question))
    return messages


LOCAL_NOTES_EMPTY = (
    "Local notes\n\n"
    "Local notes follow. They are the only place-specific notes supplied for this run.\n"
    "(No specific local notes match this query.)"
)


def render_protocol(
    *,
    condition: str,
    prompt_id: str,
    layout: str = "app",
    root: Path | None = None,
) -> str:
    root = root or repo_root()
    prompts = load_prompts(root)
    prompt = next((item for item in prompts if item["id"] == prompt_id), None)
    if prompt is None:
        known = ", ".join(item["id"] for item in prompts)
        raise ValueError(f"unknown prompt id {prompt_id!r}; known: {known}")
    notes = load_real_notes(root)
    persona = load_persona(root)
    refusal = load_refusal(root)
    sim_rag = load_sim_rag(root)
    messages = messages_for(
        condition=condition,
        prompt=prompt,
        notes=notes,
        persona=persona,
        refusal=refusal,
        sim_rag=sim_rag,
        layout=layout,
    )
    return chatml(messages)
