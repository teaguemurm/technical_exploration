#!/usr/bin/env python3
"""Recompute the mlx-lm validation loss of every checkpoint (neither training run
saved its log). Same data (data/v2/valid.jsonl), same ChatDataset/evaluate code as
mlx_lm.lora (mask_prompt, max_seq_length 2048, batch 2, all batches).
Iter 0 = base model without adapter.
Usage: val_loss.py <model> <adapter_dir> <out.json>"""
import json, sys, time, types
from pathlib import Path
D = Path(__file__).resolve().parent; LORA = D.parents[1]
import mlx.core as mx
from mlx_lm import load
from mlx_lm.tuner.datasets import load_local_dataset, CacheDataset
from mlx_lm.tuner.trainer import evaluate
MODEL, AD, OUT = sys.argv[1], LORA / sys.argv[2], D / sys.argv[3]
cfg = types.SimpleNamespace(mask_prompt=True)
res = {}
model, tok = load(MODEL)
_, valid, _ = load_local_dataset(LORA / "data/v2", tok, cfg)
vs = CacheDataset(valid)
def ev(m): return evaluate(model=m, dataset=vs, batch_size=2, num_batches=-1, max_seq_length=2048)
t = time.time(); res["0"] = ev(model); print(MODEL, "base", res["0"], f"{time.time()-t:.0f}s", flush=True)
del model; mx.clear_cache()
model, tok = load(MODEL, adapter_path=str(AD))
res["final"] = ev(model); print("final", res["final"], flush=True)
for c in range(100, 1001, 100):
    model.load_weights(str(AD / f"{c:07d}_adapters.safetensors"), strict=False)
    res[str(c)] = ev(model); print(c, res[str(c)], flush=True)
json.dump(dict(model=MODEL, adapter_dir=sys.argv[2], n_valid=len(valid), losses=res), open(OUT, "w"), indent=1)
