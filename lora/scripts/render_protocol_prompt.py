#!/usr/bin/env python3
"""Print a ChatML prompt for one locked protocol item.

layout=app matches lib/messages.dart (notes in a user turn).
layout=train matches the LoRA data (notes inside the system message).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wisent_lora.render import render_protocol  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", required=True, help="A, B, C, B-sim, or C-sim")
    parser.add_argument("--prompt-id", required=True, help="Id from assets/prompts.json")
    parser.add_argument(
        "--layout",
        choices=("app", "train"),
        default="app",
        help="app = Flutter message roles; train = notes inside the system message",
    )
    args = parser.parse_args()
    sys.stdout.write(
        render_protocol(
            condition=args.condition,
            prompt_id=args.prompt_id,
            layout=args.layout,
        )
    )


if __name__ == "__main__":
    main()
