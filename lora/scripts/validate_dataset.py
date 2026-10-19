#!/usr/bin/env python3
"""Validate lora/data/train.jsonl and valid.jsonl."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wisent_lora.build import build_examples  # noqa: E402
from wisent_lora.protocol import load_refusal, repo_root  # noqa: E402
from wisent_lora.validate import format_report, validate_files  # noqa: E402


def main() -> None:
    root = repo_root()
    _, note_text = build_examples(root)
    report = validate_files(
        root / "lora" / "data",
        refusal=load_refusal(root),
        note_texts=note_text,
    )
    print(format_report(report))
    if not report.ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
