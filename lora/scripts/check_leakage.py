#!/usr/bin/env python3
"""Flag training questions that overlap assets/prompts.json or the paraphrase list."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wisent_lora.leakage import check_dataset_dir  # noqa: E402
from wisent_lora.protocol import repo_root  # noqa: E402


def main() -> None:
    root = repo_root()
    hits = check_dataset_dir(root / "lora" / "data", root)
    if not hits:
        print("leakage hits: 0")
        return
    print(f"leakage hits: {len(hits)}")
    for hit in hits:
        print(f"{hit['split']} {hit['id']}: {hit['reason']}")
        print(f"  probe: {hit['probe']}")
        print(f"  question: {hit['question']}")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
