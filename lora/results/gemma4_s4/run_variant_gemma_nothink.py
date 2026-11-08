#!/usr/bin/env python3
"""NOTHINK VARIANT (4 Nov 2026): identical to run_variant_gemma.py except the chat template is rendered with
enable_thinking=False. mlx-lm 0.32.2 TokenizerWrapper.apply_chat_template defaults enable_thinking to
tokenizer.has_thinking (True for Gemma 4), so the original default call put <|think|> in the system turn.
Also writes out_<name>.prompt0.txt (rendered prompt of the first row).

Copy of results/s4/run_variant.py (the Qwen s4 subset runner) with the model id
as a parameter (default google/gemma-4-E2B-it).
Same 100 eval-only rows (results/s4/subset_ids.json), same app layout
(condition_b_app_messages), same app sampler (temp 0, top-p 0.9, top-k 40,
repeat penalty 1.1 over 64, max 256, seed 42). The prompt is rendered with the
HF tokenizer's own chat template (default kwargs; for Gemma 4 = thinking off,
the same call mlx_lm.lora used for the training data).
Writes results/gemma4_s4/out_<name>.jsonl (outputs stored verbatim after .strip(),
as in the Qwen runner)."""
import argparse, json, sys, time
from pathlib import Path
D = Path(__file__).resolve().parent; LORA = D.parents[1]
sys.path.insert(0, str(LORA))
from wisent_lora.comparison import load_note_pool
from wisent_lora.protocol import condition_b_app_messages, load_persona, repo_root
p = argparse.ArgumentParser()
p.add_argument("--name", required=True)
p.add_argument("--model", default="google/gemma-4-E2B-it")
p.add_argument("--adapter", default=None)
p.add_argument("--penalty", choices=["on", "off"], default="on")
p.add_argument("--limit", type=int, default=None)
a = p.parse_args()
import mlx.core as mx
from mlx_lm import generate, load
from mlx_lm.sample_utils import make_logits_processors, make_sampler
root = repo_root(); persona = load_persona(root); note_text, _, _ = load_note_pool(root)
ids = json.load(open(LORA / "results/s4/subset_ids.json"))["ids"][: a.limit]
rows = {json.loads(l)["metadata"]["id"]: json.loads(l) for l in (LORA / "data/comparison/comparison.jsonl").read_text().splitlines() if l.strip()}
mx.random.seed(42)
t0 = time.time()
model, tok = load(a.model, adapter_path=a.adapter) if a.adapter else load(a.model)
sampler = make_sampler(temp=0.0, top_p=0.9, top_k=40)
procs = make_logits_processors(repetition_penalty=1.1, repetition_context_size=64) if a.penalty == "on" else None
out = open(D / f"out_{a.name}.jsonl", "w")
for i in ids:
    row = rows[i]
    q = [m["content"] for m in row["messages"] if m["role"] == "user"][-1]
    texts = [note_text[n] for n in row["metadata"].get("prompt_note_ids") or []]
    prompt = tok.apply_chat_template(condition_b_app_messages(persona, texts, q), tokenize=False, add_generation_prompt=True, enable_thinking=False)
    assert "<|think|>" not in prompt
    if i == ids[0]: (D / f"out_{a.name}.prompt0.txt").write_text(prompt)
    kw = dict(max_tokens=256, sampler=sampler, verbose=False)
    if procs: kw["logits_processors"] = procs
    text = generate(model, tok, prompt=prompt, **kw).strip()
    out.write(json.dumps({"id": i, "output": text}) + "\n"); out.flush()
el = time.time() - t0
json.dump({"name": a.name, "model": a.model, "adapter": a.adapter, "penalty": a.penalty, "enable_thinking": False, "seconds": round(el, 1)}, open(D / f"out_{a.name}.meta.json", "w"))
print(f"{a.name} done in {el:.1f}s", flush=True)
