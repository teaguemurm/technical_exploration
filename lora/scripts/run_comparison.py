#!/usr/bin/env python3
"""Run a prompt file with and without a LoRA and write a side-by-side CSV.

Two backends:

* mlx — load Qwen once without an adapter and once with ``adapter_path``.
  Greedy sampling goes through ``make_sampler`` (``generate`` no longer takes
  ``temp``). Repeat penalty 1.1 uses ``make_logits_processors``.
* llama — ``llama-completion`` twice per row, the second time with
  ``--lora-scaled``. The prompt file is ChatML and ``-no-cnv`` sends it
  through as the raw prompt.

Sampling numbers are read from the app: ``lib/main.dart`` for temperature,
top-k, top-p, seed, max tokens, and context; ``inference_sampler_config.dart``
for the penalty window. That sampler does not set min-p, so the llama chain
is ``temperature;top_k;top_p;penalties`` and does not add the CLI's extra
samplers.

``--layout stored`` sends the file's own non-assistant messages. ``--layout app``
rebuilds condition B with ``condition_b_app_messages``. Generalisation is
``--layout app`` on ``lora/data/comparison/comparison.jsonl``. The
``question_in_original: false`` group is that result. ``--layout stored`` on
``lora/data/v2/train.jsonl`` only checks that training fit.

mlx-lm is not imported until the mlx backend runs.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wisent_lora.comparison import load_note_pool  # noqa: E402
from wisent_lora.protocol import (  # noqa: E402
    condition_b_app_messages,
    load_persona,
    load_protocol_decoding,
    repo_root,
)
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

# App sampler order from _configureSampler, excluding the seed distribution,
# which llama-completion takes as --seed. min-p is absent on purpose.
APP_SAMPLERS = "temperature;top_k;top_p;penalties"


def load_rows(path: Path, limit: int | None) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if limit is not None:
        return rows[:limit]
    return rows


def question_text(row: dict) -> str:
    users = [message["content"] for message in row["messages"] if message["role"] == "user"]
    if not users:
        raise ValueError(f"row {row.get('metadata', {}).get('id', '?')} has no user turn")
    return users[-1]


def prompt_messages(
    row: dict,
    *,
    layout: str = "stored",
    persona: str | None = None,
    note_text: dict[str, str] | None = None,
) -> list[dict]:
    if layout == "stored":
        return [message for message in row["messages"] if message["role"] != "assistant"]
    if layout != "app":
        raise ValueError("layout must be 'stored' or 'app'")
    if persona is None or note_text is None:
        raise ValueError("app layout needs the persona and the note texts")
    texts = []
    for note_id in row["metadata"].get("prompt_note_ids") or []:
        if note_id not in note_text:
            raise KeyError(f"unknown note id {note_id}")
        texts.append(note_text[note_id])
    return condition_b_app_messages(persona, texts, question_text(row))


def llama_prompt(row: dict, **kwargs) -> str:
    messages = [(message["role"], message["content"]) for message in prompt_messages(row, **kwargs)]
    return chatml(messages)


def csv_record(row: dict, without: str, with_adapter: str) -> dict[str, str]:
    metadata = row["metadata"]
    grounded = metadata.get("grounded_note_ids")
    if isinstance(grounded, list):
        grounded_text = ",".join(grounded)
    else:
        grounded_text = "" if grounded is None else str(grounded)
    assistant = row["messages"][-1]["content"]
    return {
        "id": str(metadata.get("id", "")),
        "category": str(metadata.get("category", "")),
        "question_in_original": "true" if metadata.get("question_in_original") else "false",
        "notes_origin": str(metadata.get("notes_origin", "")),
        "grounded_note_ids": grounded_text,
        "question": question_text(row),
        "reference": str(metadata.get("reference_answer", assistant)),
        "output_without_adapter": without,
        "output_with_adapter": with_adapter,
    }


def group_summary(rows: list[dict], *, source: Path, layout: str) -> dict:
    groups = {
        "true": Counter(),
        "false": Counter(),
    }
    for row in rows:
        flag = "true" if row["metadata"].get("question_in_original") else "false"
        groups[flag][str(row["metadata"].get("category", ""))] += 1

    def pack(counter: Counter) -> dict:
        return {"n": sum(counter.values()), "by_category": dict(counter)}

    return {
        "source": str(source),
        "layout": layout,
        "question_in_original_true": pack(groups["true"]),
        "question_in_original_false": pack(groups["false"]),
        "generalisation_group": "question_in_original_false",
        "note": (
            "question_in_original false is the generalisation result when source is "
            "lora/data/comparison/comparison.jsonl scored with layout app. "
            "layout stored on lora/data/v2/train.jsonl only checks training fit."
        ),
    }


def write_csv(path: Path, records: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(records)


def write_summary(csv_path: Path, summary: dict) -> Path:
    path = csv_path.with_name(csv_path.stem + ".summary.json")
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return path


def _flag_number(value: float | int) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def llama_argv(
    *,
    llama_cli: str,
    model: str,
    prompt_path: str,
    adapter: str | None,
    lora_scale: float,
    max_tokens: int,
    seed: int,
    decoding: dict,
) -> list[str]:
    command = [
        llama_cli,
        "-m",
        model,
        "-f",
        prompt_path,
        "--samplers",
        APP_SAMPLERS,
        "--temp",
        _flag_number(decoding["temperature"]),
        "--top-k",
        str(decoding["top_k"]),
        "--top-p",
        _flag_number(decoding["top_p"]),
        "--repeat-last-n",
        str(decoding["repeat_last_n"]),
        "--repeat-penalty",
        _flag_number(decoding["repeat_penalty"]),
        "--frequency-penalty",
        "0",
        "--presence-penalty",
        "0",
        "-n",
        str(max_tokens),
        "--seed",
        str(seed),
        "-c",
        str(decoding["context_size"]),
        "--no-display-prompt",
        "-co",
        "off",
        "-no-cnv",
    ]
    if adapter:
        command.extend(["--lora-scaled", f"{adapter}:{_flag_number(lora_scale)}"])
    return command


def _mlx_texts(
    model_id: str,
    rows: list[dict],
    adapter_path: str | None,
    max_tokens: int,
    seed: int,
    decoding: dict,
    *,
    layout: str,
    persona: str | None,
    note_text: dict[str, str] | None,
) -> list[str]:
    import mlx.core as mx
    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_logits_processors, make_sampler

    mx.random.seed(seed)
    if adapter_path:
        model, tokenizer = load(model_id, adapter_path=adapter_path)
    else:
        model, tokenizer = load(model_id)
    sampler = make_sampler(
        temp=float(decoding["temperature"]),
        top_p=float(decoding["top_p"]),
        top_k=int(decoding["top_k"]),
    )
    processors = make_logits_processors(
        repetition_penalty=float(decoding["repeat_penalty"]),
        repetition_context_size=int(decoding["repeat_last_n"]),
    )
    texts = []
    for row in rows:
        messages = prompt_messages(row, layout=layout, persona=persona, note_text=note_text)
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        texts.append(
            generate(
                model,
                tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                sampler=sampler,
                logits_processors=processors,
                verbose=False,
            ).strip()
        )
    return texts


def run_mlx(
    model_id: str,
    adapter_path: str,
    rows: list[dict],
    max_tokens: int,
    seed: int,
    decoding: dict,
    *,
    layout: str,
    persona: str | None,
    note_text: dict[str, str] | None,
) -> tuple[list[str], list[str]]:
    without = _mlx_texts(
        model_id, rows, None, max_tokens, seed, decoding, layout=layout, persona=persona, note_text=note_text
    )
    with_adapter = _mlx_texts(
        model_id,
        rows,
        adapter_path,
        max_tokens,
        seed,
        decoding,
        layout=layout,
        persona=persona,
        note_text=note_text,
    )
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
    decoding: dict,
) -> str:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".txt", delete=False) as handle:
        handle.write(prompt)
        prompt_path = handle.name
    command = llama_argv(
        llama_cli=llama_cli,
        model=model,
        prompt_path=prompt_path,
        adapter=adapter,
        lora_scale=lora_scale,
        max_tokens=max_tokens,
        seed=seed,
        decoding=decoding,
    )
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
    decoding: dict,
    layout: str,
    persona: str | None,
    note_text: dict[str, str] | None,
) -> tuple[list[str], list[str]]:
    without = []
    with_adapter = []
    for row in rows:
        prompt = llama_prompt(row, layout=layout, persona=persona, note_text=note_text)
        without.append(
            _llama_once(
                llama_cli=llama_cli,
                model=model,
                prompt=prompt,
                adapter=None,
                lora_scale=lora_scale,
                max_tokens=max_tokens,
                seed=seed,
                decoding=decoding,
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
                decoding=decoding,
            )
        )
    return without, with_adapter


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("mlx", "llama"), required=True)
    parser.add_argument("--model", required=True, help="HF repo/path for mlx, or a GGUF path for llama")
    parser.add_argument("--adapter", required=True, help="mlx adapter directory, or a LoRA GGUF for llama")
    parser.add_argument("--llama-cli", default="llama-completion")
    parser.add_argument("--lora-scale", type=float, default=1.0)
    parser.add_argument("--layout", choices=("stored", "app"), default="stored")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser


def _print_summary(summary: dict) -> None:
    true_group = summary["question_in_original_true"]
    false_group = summary["question_in_original_false"]
    print(f"layout {summary['layout']}")
    print(
        f"question_in_original true: {true_group['n']} "
        "(training questions under this prompt; not the generalisation result)"
    )
    print(
        f"question_in_original false: {false_group['n']} "
        "(generalisation result when the file is the comparison set scored with --layout app)"
    )
    for label, group in (("true", true_group), ("false", false_group)):
        if not group["by_category"]:
            continue
        detail = ", ".join(f"{category}={count}" for category, count in group["by_category"].items())
        print(f"  {label} by category: {detail}")


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    root = repo_root()
    decoding = load_protocol_decoding(root)
    max_tokens = decoding["max_tokens"] if args.max_tokens is None else args.max_tokens
    seed = decoding["seed"] if args.seed is None else args.seed
    path = args.comparison or (root / "lora" / "data" / "comparison" / "comparison.jsonl")
    rows = load_rows(path, args.limit)
    persona = None
    note_text = None
    if args.layout == "app":
        persona = load_persona(root)
        note_text, _, _ = load_note_pool(root)
    if args.backend == "mlx":
        without, with_adapter = run_mlx(
            args.model,
            args.adapter,
            rows,
            int(max_tokens),
            int(seed),
            decoding,
            layout=args.layout,
            persona=persona,
            note_text=note_text,
        )
    else:
        without, with_adapter = run_llama(
            llama_cli=args.llama_cli,
            model=args.model,
            adapter=args.adapter,
            rows=rows,
            lora_scale=args.lora_scale,
            max_tokens=int(max_tokens),
            seed=int(seed),
            decoding=decoding,
            layout=args.layout,
            persona=persona,
            note_text=note_text,
        )
    write_csv(args.output, [csv_record(row, left, right) for row, left, right in zip(rows, without, with_adapter)])
    summary = group_summary(rows, source=path, layout=args.layout)
    summary_path = write_summary(args.output, summary)
    print(f"wrote {len(rows)} rows to {args.output}")
    print(f"wrote {summary_path}")
    _print_summary(summary)


if __name__ == "__main__":
    main()
