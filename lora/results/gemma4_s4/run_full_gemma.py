#!/usr/bin/env python3
"""Full 910-row comparison for Gemma 4 E2B, same procedure as
`scripts/run_comparison.py --backend mlx --layout app` (imports its functions
unchanged: load_rows, _mlx_texts, csv_record, write_csv, group_summary,
write_summary; sampler from load_protocol_decoding = app settings).
Difference: the base (no adapter) pass is generated once and cached in
--base-cache, then reused as output_without_adapter for every adapter CSV
(run_comparison.py regenerates it per run; for Qwen the regenerated base was
identical 910/910). Model id is a parameter instead of Qwen."""
import argparse, json, sys, time
from pathlib import Path
D = Path(__file__).resolve().parent; LORA = D.parents[1]
sys.path.insert(0, str(LORA)); sys.path.insert(0, str(LORA / "scripts"))
import run_comparison as RC
from wisent_lora.comparison import load_note_pool
from wisent_lora.protocol import load_persona, load_protocol_decoding, repo_root
p = argparse.ArgumentParser()
p.add_argument("--model", default="google/gemma-4-E2B-it")
p.add_argument("--adapter", default=None)
p.add_argument("--output", type=Path, default=None)
p.add_argument("--base-cache", type=Path, required=True)
p.add_argument("--comparison", type=Path, default=LORA / "data/comparison/comparison.jsonl")
a = p.parse_args()
root = repo_root(); dec = load_protocol_decoding(root)
persona = load_persona(root); note_text, _, _ = load_note_pool(root)
rows = RC.load_rows(a.comparison, None)
print("decoding", dec, flush=True)
kw = dict(layout="app", persona=persona, note_text=note_text)
t0 = time.time()
if a.base_cache.exists():
    base = [json.loads(l)["output"] for l in a.base_cache.read_text().splitlines() if l.strip()]
    assert len(base) == len(rows), "incomplete base cache"
else:
    base = RC._mlx_texts(a.model, rows, None, int(dec["max_tokens"]), int(dec["seed"]), dec, **kw)
    a.base_cache.write_text("".join(json.dumps({"id": r["metadata"]["id"], "output": o}) + "\n" for r, o in zip(rows, base)))
    print(f"base done {time.time()-t0:.0f}s", flush=True)
if a.adapter:
    t1 = time.time()
    lora = RC._mlx_texts(a.model, rows, a.adapter, int(dec["max_tokens"]), int(dec["seed"]), dec, **kw)
    RC.write_csv(a.output, [RC.csv_record(r, l, w) for r, l, w in zip(rows, base, lora)])
    s = RC.group_summary(rows, source=a.comparison.relative_to(LORA), layout="app")
    s.update(model=a.model, adapter=a.adapter, base_from_cache=str(a.base_cache.resolve().relative_to(LORA)), seconds_adapter_pass=round(time.time() - t1, 1))
    RC.write_summary(a.output, s)
    print(f"wrote {len(rows)} rows to {a.output} ({time.time()-t1:.0f}s)", flush=True)
