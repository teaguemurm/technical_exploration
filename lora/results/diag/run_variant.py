#!/usr/bin/env python3
"""Run one diagnostic variant on the subset. Writes results/diag/out_<name>.jsonl."""
import argparse, json, sys, time
from pathlib import Path
LORA = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LORA))
from wisent_lora.comparison import load_note_pool
from wisent_lora.protocol import condition_b_app_messages, load_persona, repo_root
p = argparse.ArgumentParser()
p.add_argument("--name", required=True)
p.add_argument("--adapter", default=None)
p.add_argument("--penalty", choices=["on", "off"], required=True)
a = p.parse_args()
import mlx.core as mx
from mlx_lm import generate, load
from mlx_lm.sample_utils import make_logits_processors, make_sampler
root = repo_root(); persona = load_persona(root); note_text, _, _ = load_note_pool(root)
ids = json.load(open(Path(__file__).parent / "subset_ids.json"))["ids"]
rows = {json.loads(l)["metadata"]["id"]: json.loads(l) for l in (LORA / "data/comparison/comparison.jsonl").read_text().splitlines() if l.strip()}
mx.random.seed(42)
t0 = time.time()
model, tok = load("Qwen/Qwen2.5-1.5B-Instruct", adapter_path=a.adapter) if a.adapter else load("Qwen/Qwen2.5-1.5B-Instruct")
sampler = make_sampler(temp=0.0, top_p=0.9, top_k=40)
procs = make_logits_processors(repetition_penalty=1.1, repetition_context_size=64) if a.penalty == "on" else None
out = open(Path(__file__).parent / f"out_{a.name}.jsonl", "w")
for i in ids:
    row = rows[i]
    q = [m["content"] for m in row["messages"] if m["role"] == "user"][-1]
    texts = [note_text[n] for n in row["metadata"].get("prompt_note_ids") or []]
    prompt = tok.apply_chat_template(condition_b_app_messages(persona, texts, q), tokenize=False, add_generation_prompt=True)
    kw = dict(max_tokens=256, sampler=sampler, verbose=False)
    if procs: kw["logits_processors"] = procs
    text = generate(model, tok, prompt=prompt, **kw).strip()
    out.write(json.dumps({"id": i, "output": text}) + "\n"); out.flush()
el = time.time() - t0
json.dump({"name": a.name, "adapter": a.adapter, "penalty": a.penalty, "seconds": round(el, 1)}, open(Path(__file__).parent / f"out_{a.name}.meta.json", "w"))
print(f"{a.name} done in {el:.1f}s")
