#!/usr/bin/env python3
"""Run the comparison set with and without a LoRA and write a side-by-side CSV.

Two backends:

* mlx — load Qwen once without an adapter and once with ``--adapter-path``
  semantics (``mlx_lm.load(..., adapter_path=...)``). Temperature 0, seed 42.
* llama — ``llama-cli`` twice per row, the second time with ``--lora``.
  The prompt is the training layout rendered as ChatML, without the assistant
  turn, and ``--no-conversation`` is set.

mlx-lm and llama-cli are not imported or required until the chosen backend
runs. This machine does not have MLX; the CSV assembly is what the unit tests
cover.

The comparison file is the full-notes condition-B layout. It does not apply a
simulated-RAG note selection.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wisent_lora.protocol import repo_root  # noqa: E402
from wisent_lora.render import chatml  # noqa: E402

FIELDNAMES = [
    "id",
    "category",
    "question_in_original",
    "notes_origin",
    "grounded_note_ids",
    "question",
    "reference",
    "output_without_adapter",
    "output_with_adapter",
]


def load_rows(path: Path, limit: int | None) -> list[dict]:
    import json

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if limit is not None:
        return rows[:limit]
    return rows


def prompt_messages(row: dict) -> list[dict]:
    return [message for message in row["messages"] if message["role"] != "assistant"]


def llama_prompt(row: dict) -> str:
    messages = [(message["role"], message["content"]) for message in prompt_messages(row)]
    return chatml(messages)


def csv_record(row: dict, without: str, with_adapter: str) -> dict[str, str]:
    metadata = row["metadata"]
    grounded = metadata.get("grounded_note_ids")
    if isinstance(grounded, list):
        grounded_text = ",".join(grounded)
    else:
        grounded_text = "" if grounded is None else str(grounded)
    return {
        "id": str(metadata.get("id", "")),
        "category": str(metadata.get("category", "")),
        "question_in_original": "true" if metadata.get("question_in_original") else "false",
        "notes_origin": str(metadata.get("notes_origin", "")),
        "grounded_note_ids": grounded_text,
        "question": row["messages"][1]["content"],
        "reference": str(metadata.get("reference_answer", row["messages"][2]["content"])),
        "output_without_adapter": without,
        "output_with_adapter": with_adapter,
    }


def write_csv(path: Path, records: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(records)


def _mlx_texts(model_id: str, rows: list[dict], adapter_path: str | None, max_tokens: int, seed: int) -> list[str]:
    import mlx.core as mx
    from mlx_lm import generate, load

    mx.random.seed(seed)
    if adapter_path:
        model, tokenizer = load(model_id, adapter_path=adapter_path)
    else:
        model, tokenizer = load(model_id)
    texts = []
    for row in rows:
        prompt = tokenizer.apply_chat_template(
            prompt_messages(row),
            tokenize=False,
            add_generation_prompt=True,
        )
        texts.append(
            generate(
                model,
                tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                temp=0.0,
                verbose=False,
            ).strip()
        )
    return texts


def run_mlx(model_id: str, adapter_path: str, rows: list[dict], max_tokens: int, seed: int) -> tuple[list[str], list[str]]:
    without = _mlx_texts(model_id, rows, None, max_tokens, seed)
    with_adapter = _mlx_texts(model_id, rows, adapter_path, max_tokens, seed)
    return without, with_adapter


def _llama_once(
    *,
    llama_cli: str,
    model: str,
    prompt: str,
    adapter: str | None,
    lora_scale: float,
    max_tokens: int,
    seed: int,
) -> str:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".txt", delete=False) as handle:
        handle.write(prompt)
        prompt_path = handle.name
    command = [
        llama_cli,
        "-m",
        model,
        "-f",
        prompt_path,
        "--temp",
        "0",
        "--top-k",
        "40",
        "--top-p",
        "0.9",
        "--repeat-penalty",
        "1.1",
        "-n",
        str(max_tokens),
        "--seed",
        str(seed),
        "-c",
        "2048",
        "--no-conversation",
    ]
    if adapter:
        command.extend(["--lora", adapter, "--lora-scale", str(lora_scale)])
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    return completed.stdout.strip()


def run_llama(
    *,
    llama_cli: str,
    model: str,
    adapter: str,
    rows: list[dict],
    lora_scale: float,
    max_tokens: int,
    seed: int,
) -> tuple[list[str], list[str]]:
    without = []
    with_adapter = []
    for row in rows:
        prompt = llama_prompt(row)
        without.append(
            _llama_once(
                llama_cli=llama_cli,
                model=model,
                prompt=prompt,
                adapter=None,
                lora_scale=lora_scale,
                max_tokens=max_tokens,
                seed=seed,
            )
        )
        with_adapter.append(
            _llama_once(
                llama_cli=llama_cli,
                model=model,
                prompt=prompt,
                adapter=adapter,
                lora_scale=lora_scale,
                max_tokens=max_tokens,
                seed=seed,
            )
        )
    return without, with_adapter


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("mlx", "llama"), required=True)
    parser.add_argument("--model", required=True, help="HF repo/path for mlx, or a GGUF path for llama")
    parser.add_argument("--adapter", required=True, help="mlx adapter directory, or a LoRA GGUF for llama")
    parser.add_argument("--llama-cli", default="llama-cli")
    parser.add_argument("--lora-scale", type=float, default=1.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    root = repo_root()
    path = args.comparison or (root / "lora" / "data" / "comparison" / "comparison.jsonl")
    rows = load_rows(path, args.limit)
    if args.backend == "mlx":
        without, with_adapter = run_mlx(args.model, args.adapter, rows, args.max_tokens, args.seed)
    else:
        without, with_adapter = run_llama(
            llama_cli=args.llama_cli,
            model=args.model,
            adapter=args.adapter,
            rows=rows,
            lora_scale=args.lora_scale,
            max_tokens=args.max_tokens,
            seed=args.seed,
        )
    write_csv(args.output, [csv_record(row, left, right) for row, left, right in zip(rows, without, with_adapter)])
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
