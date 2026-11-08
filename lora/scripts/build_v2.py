#!/usr/bin/env python3
"""Write lora/data/v2 and fail if validation or leakage fails.

Does not rewrite lora/data/train.jsonl, valid.jsonl, or their meta and stats files.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wisent_lora.v2 import build_and_check_v2  # noqa: E402


def main() -> None:
    stats = build_and_check_v2()
    print(f"examples: {stats['total']}")
    print(f"copied from original: {stats['copied_from_original']['total']}")
    print(f"nature_decode: {stats['nature_decode']}")
    print(f"speak_as reworded: {len(stats['speak_as_reworded'])}")
    print(f"contrastive pairs: {stats['contrastive_pairs']}")
    print(f"train: {stats['train']['n']}  valid: {stats['valid']['n']}")
    for category, count in stats["by_category"].items():
        share = count / stats["total"]
        print(f"  {category}: {count} ({share:.1%})")
    print(f"notes origin: {stats['notes_origin']}")
    print(f"validation errors: {stats['validation_errors']}")
    print(f"leakage hits: {stats['leakage_hits']}")


if __name__ == "__main__":
    main()
