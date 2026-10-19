#!/usr/bin/env python3
"""Write lora/data/comparison/ from the training JSONL plus the new question bank.

Does not rewrite train.jsonl, valid.jsonl, the metadata sidecars, stats.json,
or synthetic_notes.json.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wisent_lora.comparison import (  # noqa: E402
    COMPARISON_SIZE_BOUNDS,
    build_comparison,
    write_comparison,
)
from wisent_lora.leakage import check_examples, default_probes  # noqa: E402
from wisent_lora.protocol import load_refusal, repo_root  # noqa: E402
from wisent_lora.validate import format_report, validate_examples  # noqa: E402


def main() -> None:
    root = repo_root()
    rows, note_text = build_comparison(root)
    refusal = load_refusal(root)
    report = validate_examples(
        rows,
        refusal=refusal,
        note_texts=note_text,
        size_bounds=COMPARISON_SIZE_BOUNDS,
        check_split_ratio=False,
    )
    print(format_report(report))
    if not report.ok:
        raise SystemExit(1)
    hits = check_examples(rows, default_probes(root))
    if hits:
        preview = "\n".join(f"{hit['id']}: {hit['reason']}: {hit['probe']}" for hit in hits[:30])
        raise SystemExit(f"leakage check failed ({len(hits)} hits):\n{preview}")
    print("leakage hits: 0")
    stats = write_comparison(root / "lora" / "data" / "comparison", rows)
    print(f"wrote {stats['total']} rows to lora/data/comparison/")
    print(
        "question_in_original "
        f"true={stats['question_in_original']['true']} "
        f"false={stats['question_in_original']['false']}"
    )


if __name__ == "__main__":
    main()
