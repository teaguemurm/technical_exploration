#!/usr/bin/env python3
"""Full 910-row BASE pass (no adapter) for Gemma 4 E2B with thinking OFF.
Identical to the base pass of run_full_gemma.py (RC.load_rows, RC._mlx_texts, app layout, load_protocol_decoding
= app sampler, seed 42, max 256), except every tokenizer.apply_chat_template call is forced to enable_thinking=False
(mlx-lm 0.32.2 defaults it to True for Gemma 4). Done by wrapping mlx_lm.load, which _mlx_texts imports at call time.
Writes gemma4_base_full_outputs_base_nothink.jsonl (+ .meta.json, .prompt0.txt). Run with HF_HUB_OFFLINE=1."""
import json, sys, time
from pathlib import Path
D = Path(__file__).resolve().parent; LORA = D.parents[1]
sys.path.insert(0, str(LORA)); sys.path.insert(0, str(LORA / "scripts"))
import run_comparison as RC
import mlx_lm
from wisent_lora.comparison import load_note_pool
from wisent_lora.protocol import load_persona, load_protocol_decoding, repo_root
OUT = D / "gemma4_base_full_outputs_base_nothink.jsonl"
assert not OUT.exists()
_load = mlx_lm.load; seen = []
def load_nothink(*a, **k):
    model, tok = _load(*a, **k)
    orig = tok.apply_chat_template
    def act(*aa, **kk):
        kk["enable_thinking"] = False
        s = orig(*aa, **kk)
        assert "<|think|>" not in s
        if not seen: seen.append(s); (D / "gemma4_base_full_outputs_base_nothink.prompt0.txt").write_text(s)
        return s
    tok.apply_chat_template = act
    return model, tok
mlx_lm.load = load_nothink
root = repo_root(); dec = load_protocol_decoding(root)
persona = load_persona(root); note_text, _, _ = load_note_pool(root)
rows = RC.load_rows(LORA / "data/comparison/comparison.jsonl", None)
print("decoding", dec, "rows", len(rows), flush=True)
t0 = time.time()
base = RC._mlx_texts("google/gemma-4-E2B-it", rows, None, int(dec["max_tokens"]), int(dec["seed"]), dec, layout="app", persona=persona, note_text=note_text)
el = time.time() - t0
OUT.write_text("".join(json.dumps({"id": r["metadata"]["id"], "output": o}) + "\n" for r, o in zip(rows, base)))
json.dump({"model": "google/gemma-4-E2B-it", "adapter": None, "enable_thinking": False, "decoding": dec, "n": len(rows), "seconds": round(el, 1)}, open(D / "gemma4_base_full_outputs_base_nothink.meta.json", "w"), indent=1)
print(f"base nothink done {el:.0f}s", flush=True)
