"""Check that common.render (app formatter) equals the HF tokenizer chat template
(what the earlier MLX evaluations fed the model) for every row and condition."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent)); import common as C
from transformers import AutoTokenizer
res = {}
for fam, mid in [("qwen", "Qwen/Qwen2.5-1.5B-Instruct"), ("gemma", "google/gemma-4-E2B-it")]:
    tok = AutoTokenizer.from_pretrained(mid)
    for cond in "ABC":
        diff = [r["metadata"]["id"] for r in C.ROWS if C.render(r, cond, fam) != tok.apply_chat_template(C.messages(r, cond), tokenize=False, add_generation_prompt=True)]
        res[f"{fam}_{cond}"] = {"n": len(C.ROWS), "mismatch": len(diff), "examples": diff[:5]}
print(json.dumps(res, indent=1))
json.dump(res, open(Path(__file__).parent / "template_check.json", "w"), indent=1)
