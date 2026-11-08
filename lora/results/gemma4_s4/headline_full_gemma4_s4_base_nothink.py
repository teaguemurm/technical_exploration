#!/usr/bin/env python3
"""BASE_NOTHINK COPY (4 Nov 2026): only the input CSVs (*_base_nothink.csv: Gemma base column = thinking-off rerun, adapter columns unchanged), the Gemma base label and the output file names differ from headline_full_gemma4_s4.py.
Headline full-set (910 rows) metrics for Gemma 4 E2B s4 (base, iter 600 if present, iter 800, final) and Qwen s4 iter 600 (+ base),
using the predicates of results/s4/analyze_s4.py. Writes headline_full_gemma4_s4.{md,json}."""
import csv, json, sys
from pathlib import Path
D = Path(__file__).resolve().parent; LORA = D.parents[1]
sys.path.insert(0, str(LORA / "results/s4")); import analyze_s4 as A
R = A.R
def load(p):
    p = Path(p)
    if not p.exists(): return None
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    return rows if len(rows) == 910 else None
src = {"qwen": LORA / "results/s4_ckpt600_mlx_app_comparison.csv", "g800": D / "gemma4_s4_ckpt800_mlx_app_comparison_base_nothink.csv",
       "gfinal": D / "gemma4_s4_mlx_app_comparison_base_nothink.csv", "g600": D / "gemma4_s4_ckpt600_mlx_app_comparison_base_nothink.csv"}
data = {k: load(v) for k, v in src.items()}
models = [("Gemma 4 E2B base (thinking off)", "g800", "output_without_adapter"), ("Gemma s4 iter 600", "g600", "output_with_adapter"),
          ("Gemma s4 iter 800", "g800", "output_with_adapter"), ("Gemma s4 final (1000)", "gfinal", "output_with_adapter"),
          ("Qwen2.5-1.5B base", "qwen", "output_without_adapter"), ("Qwen s4 iter 600", "qwen", "output_with_adapter")]
def rt(r): return r["reference"].strip() == R
def metrics(rows, col, scope):
    rows = [r for r in rows if scope == "all" or r["question_in_original"] == ("false" if scope == "eval-only" else "true")]
    refs = [r for r in rows if rt(r)]; ans = [r for r in rows if not rt(r) and r["category"] in ("answerable", "speak_as")]
    ansonly = [r for r in ans if r["category"] == "answerable"]; sa = [r for r in rows if r["category"] == "speak_as"]
    pre = [r for r in rows if r["category"] == "preaching"]
    answered = [r for r in ans if not A.contains_refusal(r[col])]
    kf = lambda r: A.key_facts(r[col], A.meta[r["id"]], r["reference"])[0]
    pct = lambda a, b: (a, b, round(100 * a / b, 1) if b else None)
    return {
        "exact_refusal_refusal_target": pct(sum(A.is_exact_refusal(r[col]) for r in refs), len(refs)),
        "contains_refusal_refusal_target": pct(sum(A.contains_refusal(r[col]) for r in refs), len(refs)),
        "over_refusal_answerable": pct(sum(A.contains_refusal(r[col]) for r in ansonly), len(ansonly)),
        "over_refusal_answer_target_ans+speak_as": pct(sum(A.contains_refusal(r[col]) for r in ans), len(ans)),
        "key_facts_answer_target_ans+speak_as": pct(sum(kf(r) for r in ans), len(ans)),
        "key_facts_among_answered": pct(sum(kf(r) for r in answered), len(answered)),
        "voice_first_person_speak_as_rows": pct(sum(A.fp_hit(r[col]) for r in sa), len(sa)),
        "preaching_rows_refused(trained)": pct(sum(A.contains_refusal(r[col]) for r in pre), len(pre)),
        "preach_markers_all_rows": pct(sum(A.preach_hit(r[col]) for r in rows), len(rows)),
        "nature_decode_all_rows": pct(sum(A.nature_hit(r[col]) for r in rows), len(rows)),
        "avg_words_answer_target": round(sum(len(r[col].split()) for r in ans) / len(ans), 1) if ans else None,
    }
out = {}; L = ["# Headline full-set metrics (910 rows, MLX bf16, app layout + app sampler)\n",
     "Script `headline_full_gemma4_s4.py`, predicates from `results/s4/analyze_s4.py`. Cells: count / n (percent). "
     "Over-refusal = output contains the refusal string on an answer-target row. Key facts = not refused AND must_include / ≥50% overlap. "
     "Voice = first-person markers on speak_as rows (lower is better). Preaching rows are trained to the refusal string in v2, so 'refused' is the trained behaviour.\n"]
for scope in ("all", "eval-only", "training-q"):
    out[scope] = {}
    avail = [(n, k, c) for n, k, c in models if data[k] is not None]
    missing = [n for n, k, c in models if data[k] is None]
    for n, k, c in avail: out[scope][n] = metrics(data[k], c, scope)
    L.append(f"\n## {scope}\n")
    if missing: L.append(f"Not available (CSV missing or incomplete): {', '.join(missing)}\n")
    L.append("| metric | " + " | ".join(n for n, _, _ in avail) + " |"); L.append("|---" * (len(avail) + 1) + "|")
    for m in next(iter(out[scope].values())):
        cells = []
        for n, _, _ in avail:
            v = out[scope][n][m]; cells.append(f"{v[0]}/{v[1]} ({v[2]}%)" if isinstance(v, tuple) else str(v))
        L.append(f"| {m} | " + " | ".join(cells) + " |")
(D / "headline_full_gemma4_s4_base_nothink.md").write_text("\n".join(L) + "\n", encoding="utf-8")
(D / "headline_full_gemma4_s4_base_nothink.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
print("\n".join(L))
