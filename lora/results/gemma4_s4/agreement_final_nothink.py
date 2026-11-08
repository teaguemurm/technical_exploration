#!/usr/bin/env python3
"""Agreement of llama.cpp gemma_final_B (laptop, Q3_K_M + f16 adapter, app formatter, thinking off) with
(a) old MLX final (thinking on) and (b) new MLX final (thinking off), using outcome()/coarse()/A.norm from
results/llamacpp_compare/score.py imported unchanged (its __main__ block does not run). Also old-vs-new MLX final.
Writes agreement_final_nothink.json."""
import csv, json, sys
from collections import Counter
from pathlib import Path
D = Path(__file__).resolve().parent; LORA = D.parents[1]
sys.path.insert(0, str(LORA / "results/llamacpp_compare")); import score as S
lc = {json.loads(l)["id"]: json.loads(l)["output"] for l in open(LORA / "results/llamacpp_compare/raw/out_gemma_final_B.jsonl")}
old = {r["id"]: r["output_with_adapter"] for r in csv.DictReader(open(D / "gemma4_s4_mlx_app_comparison.csv", encoding="utf-8"))}
new = {json.loads(l)["id"]: json.loads(l)["output"] for l in open(D / "gemma4_s4_final_full_outputs_final_nothink.jsonl")}
res = {}
for pa, a, pb, b in [("llama.cpp", lc, "MLX old thinking-on", old), ("llama.cpp", lc, "MLX new thinking-off", new), ("MLX old thinking-on", old, "MLX new thinking-off", new)]:
    r = {}
    for sub, f in S.SUBSETS.items():
        ids = [i for i in S.ORDER if f(i)]
        r[sub] = {"n": len(ids), "same_outcome": sum(S.outcome(i, a[i]) == S.outcome(i, b[i]) for i in ids),
                  "same_refused_vs_answered": sum(S.coarse(i, a[i]) == S.coarse(i, b[i]) for i in ids),
                  "identical_text": sum(S.A.norm(a[i]) == S.A.norm(b[i]) for i in ids),
                  "disagreements": dict(Counter(f"{S.outcome(i, a[i])} ({pa}) vs {S.outcome(i, b[i])} ({pb})" for i in ids if S.outcome(i, a[i]) != S.outcome(i, b[i])).most_common())}
    res[f"{pa} vs {pb}"] = r
json.dump(res, open(D / "agreement_final_nothink.json", "w"), indent=1)
for k, v in res.items():
    print(k); [print("  ", s, {x: y for x, y in m.items()}) for s, m in v.items() if s in ("full", "eval_only")]
