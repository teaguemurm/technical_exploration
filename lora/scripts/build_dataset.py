#!/usr/bin/env python3
"""Write lora/data train and valid JSONL and fail if validation or leakage fails."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wisent_lora.build import build_and_check  # noqa: E402
from wisent_lora.validate import CATEGORIES  # noqa: E402


def main() -> None:
    stats = build_and_check()
    print(f"examples: {stats['total']}")
    print(f"contrastive pairs: {stats['contrastive_pairs']}")
    print(f"train: {stats['train']['n']}  valid: {stats['valid']['n']}")
    for category in CATEGORIES:
        count = stats["by_category"][category]
        share = count / stats["total"]
        print(f"  {category}: {count} ({share:.1%})")
    print(f"notes origin: {stats['notes_origin']}")
    print(f"validation errors: {stats['validation_errors']}")
    print(f"leakage hits: {stats['leakage_hits']}")


if __name__ == "__main__":
    main()
