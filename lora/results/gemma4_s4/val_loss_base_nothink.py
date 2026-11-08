#!/usr/bin/env python3
"""Base-only (no adapter) val loss of google/gemma-4-E2B-it on data/v2/valid.jsonl, same evaluate settings as
val_loss.py, but every chat-template call forced to enable_thinking=False. Writes gemma4_s4_val_loss_base_nothink.json."""
import json, time, types, functools
from pathlib import Path
D = Path(__file__).resolve().parent; LORA = D.parents[1]
from mlx_lm import load
from mlx_lm.tuner.datasets import load_local_dataset, CacheDataset
from mlx_lm.tuner.trainer import evaluate
model, tok = load("google/gemma-4-E2B-it")
orig = tok.apply_chat_template
def nothink(*a, **k):
    k["enable_thinking"] = False; return orig(*a, **k)
tok.apply_chat_template = nothink
_, valid, _ = load_local_dataset(LORA / "data/v2", tok, types.SimpleNamespace(mask_prompt=True))
vs = CacheDataset(valid)
s0 = tok._tokenizer.decode(vs[0][0][:40]) if isinstance(vs[0], tuple) else str(vs[0])[:200]
t = time.time(); loss = evaluate(model=model, dataset=vs, batch_size=2, num_batches=-1, max_seq_length=2048)
print("base nothink val loss", loss, f"{time.time()-t:.0f}s"); print("first example start:", repr(s0))
json.dump(dict(model="google/gemma-4-E2B-it", enable_thinking=False, n_valid=len(valid), loss_base=loss), open(D / "gemma4_s4_val_loss_base_nothink.json", "w"), indent=1)
