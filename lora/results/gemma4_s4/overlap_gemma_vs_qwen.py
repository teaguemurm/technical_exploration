#!/usr/bin/env python3
"""Row-by-row error overlap: Gemma 4 E2B s4 (chosen checkpoint) vs Qwen s4 iter 600 on the 910-row comparison set.
Error definitions use results/s4/analyze_s4.py predicates (contains_refusal, key_facts, fp_hit, nature_hit, preach_hit):
  over_refusal   : answer-target answerable/speak_as rows whose output contains the refusal string
  missed_refusal : refusal-target rows whose output does not contain the refusal string
  inexact_refusal: refusal-target rows with the refusal string present but output not exactly it
  fact_miss      : answer-target answerable/speak_as rows, answered (no refusal string) but key facts missing
  answer_fail    : answer-target answerable/speak_as rows without key facts (over_refusal OR fact_miss)
  voice_first_person: speak_as rows with first-person markers
  nature_decode  : any row with nature/decoding markers
  preaching (separate): preaching rows answered instead of refused (deviation from the trained refusal),
                 and any row with preach markers.
For each: both / gemma only / qwen only / neither, rates, expected both by chance = pG*pQ*N, Jaccard, Cohen's kappa.
Usage: overlap_gemma_vs_qwen.py <gemma checkpoint label: 800|600|final> [more labels...]
Writes gemma_vs_qwen_s4_ckpt600.{csv,md} and gemma_vs_qwen_s4_ckpt600_both_fail.md."""
import csv, json, sys
from pathlib import Path
D = Path(__file__).resolve().parent; LORA = D.parents[1]
sys.path.insert(0, str(LORA / "results/s4"))
import analyze_s4 as A
R = A.R
def load(p): return {r["id"]: r for r in csv.DictReader(open(p, encoding="utf-8"))}
qc = load(LORA / "results/s4_ckpt600_mlx_app_comparison.csv"); ids = list(qc)
FILES = {"800": "gemma4_s4_ckpt800_mlx_app_comparison.csv", "600": "gemma4_s4_ckpt600_mlx_app_comparison.csv", "final": "gemma4_s4_mlx_app_comparison.csv"}
labels = sys.argv[1:] or ["800"]
def rt(r): return r["reference"].strip() == R
def ansq(r): return (not rt(r)) and r["category"] in ("answerable", "speak_as")
ERR = [  # name, universe predicate, error predicate(output, row), family
    ("over_refusal", ansq, lambda o, r: A.contains_refusal(o), "refusal"),
    ("missed_refusal", rt, lambda o, r: not A.contains_refusal(o), "refusal"),
    ("inexact_refusal", rt, lambda o, r: A.contains_refusal(o) and not A.is_exact_refusal(o), "refusal"),
    ("fact_miss", ansq, lambda o, r: (not A.contains_refusal(o)) and not A.key_facts(o, A.meta[r["id"]], r["reference"])[0], "facts"),
    ("answer_fail", ansq, lambda o, r: not A.key_facts(o, A.meta[r["id"]], r["reference"])[0], "facts"),
    ("voice_first_person", lambda r: r["category"] == "speak_as", lambda o, r: A.fp_hit(o), "voice"),
    ("nature_decode", lambda r: True, lambda o, r: A.nature_hit(o), "voice"),
    ("preaching_answered_not_refused", lambda r: r["category"] == "preaching", lambda o, r: not A.contains_refusal(o), "preaching (separate)"),
    ("preach_markers_any_row", lambda r: True, lambda o, r: A.preach_hit(o), "preaching (separate)"),
]
def stats(g, q):
    N = len(g); b = sum(x and y for x, y in zip(g, q)); go = sum(x and not y for x, y in zip(g, q)); qo = sum(y and not x for x, y in zip(g, q)); ne = N - b - go - qo
    pg = (b + go) / N if N else 0; pq = (b + qo) / N if N else 0
    exp = pg * pq * N; po = (b + ne) / N if N else 0; pe = pg * pq + (1 - pg) * (1 - pq)
    kappa = (po - pe) / (1 - pe) if (1 - pe) > 1e-12 else None
    jac = b / (b + go + qo) if (b + go + qo) else None
    return dict(N=N, both=b, gemma_only=go, qwen_only=qo, neither=ne, gemma_errors=b + go, qwen_errors=b + qo,
                gemma_rate=pg, qwen_rate=pq, expected_both_chance=exp, both_over_expected=(b / exp) if exp else None, jaccard=jac, cohen_kappa=kappa)
rowsout = []; both_ex = {}
for lab in labels:
    gc = load(D / FILES[lab]); assert set(gc) == set(ids)
    for scope in ("all", "eval-only", "training-q"):
        sel = [i for i in ids if scope == "all" or (qc[i]["question_in_original"] == ("false" if scope == "eval-only" else "true"))]
        for name, uni, err, fam in ERR:
            U = [i for i in sel if uni(qc[i])]
            g = [err(gc[i]["output_with_adapter"], gc[i]) for i in U]; q = [err(qc[i]["output_with_adapter"], qc[i]) for i in U]
            s = stats(g, q); rowsout.append(dict(gemma=f"gemma s4 iter {lab}" if lab != "final" else "gemma s4 final (iter 1000)", scope=scope, error=name, family=fam, **s))
            if scope == "all" and lab == labels[0]:
                both_ex[name] = [i for i, x, y in zip(U, g, q) if x and y]
                # base-model error rates for context
                gb = [err(gc[i]["output_without_adapter"], gc[i]) for i in U]; qb = [err(qc[i]["output_without_adapter"], qc[i]) for i in U]
                rowsout[-1]["gemma_base_errors"] = sum(gb); rowsout[-1]["qwen_base_errors"] = sum(qb)
cols = ["gemma", "scope", "error", "family", "N", "both", "gemma_only", "qwen_only", "neither", "gemma_errors", "qwen_errors", "gemma_rate", "qwen_rate", "expected_both_chance", "both_over_expected", "jaccard", "cohen_kappa", "gemma_base_errors", "qwen_base_errors"]
with open(D / "gemma_vs_qwen_s4_ckpt600.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
    for r in rowsout: w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
def fm(x, p=2): return "—" if x is None else (f"{x:.{p}f}" if isinstance(x, float) else str(x))
M = ["# Row-level error overlap: Gemma 4 E2B s4 vs Qwen s4 iter 600 (910-row comparison set)\n",
     "Script `overlap_gemma_vs_qwen.py`; predicates from `results/s4/analyze_s4.py`. Expected-by-chance = rate_G × rate_Q × N (independence). "
     "Jaccard = both / (both + only-G + only-Q). κ = Cohen's kappa on the binary error indicator over the row universe N. "
     "Preaching is reported separately: refusing preaching prompts is the trained v2 behaviour, so 'answered' there is a deviation from training, not a factual error.\n"]
for lab in labels:
    gl = f"gemma s4 iter {lab}" if lab != "final" else "gemma s4 final (iter 1000)"
    for scope in ("all", "eval-only", "training-q"):
        M.append(f"\n## {gl} vs qwen s4 iter 600 — {scope}\n")
        M.append("| error | family | N | both | Gemma only | Qwen only | Gemma rate | Qwen rate | expected both (chance) | both/expected | Jaccard | κ |" + (" Gemma base / Qwen base errors |" if scope == "all" and lab == labels[0] else ""))
        M.append("|---" * (12 + (1 if scope == "all" and lab == labels[0] else 0)) + "|")
        for r in rowsout:
            if r["gemma"] == gl and r["scope"] == scope:
                extra = f" {r['gemma_base_errors']} / {r['qwen_base_errors']} |" if "gemma_base_errors" in r else ""
                M.append(f"| {r['error']} | {r['family']} | {r['N']} | {r['both']} | {r['gemma_only']} | {r['qwen_only']} | {100*r['gemma_rate']:.1f}% | {100*r['qwen_rate']:.1f}% | {r['expected_both_chance']:.2f} | {fm(r['both_over_expected'],1)} | {fm(r['jaccard'])} | {fm(r['cohen_kappa'])} |" + extra)
(D / "gemma_vs_qwen_s4_ckpt600.md").write_text("\n".join(M) + "\n", encoding="utf-8")
gc = load(D / FILES[labels[0]])
B = [f"# Rows where both Gemma s4 iter {labels[0]} and Qwen s4 iter 600 fail (910-row set)\n", "Side by side: question, reference, Gemma LoRA output, Qwen LoRA output (and both bases, truncated).\n"]
for name, lst in both_ex.items():
    B.append(f"\n## {name}: {len(lst)} rows\n")
    for i in lst:
        r = qc[i]
        B.append(f"### {i} ({r['category']}, train-q={r['question_in_original']}, grounded={r['grounded_note_ids']})\n- Q: {r['question']}\n- Ref: {r['reference']}\n- **Gemma LoRA:** {gc[i]['output_with_adapter'][:400]}\n- **Qwen LoRA:** {r['output_with_adapter'][:400]}\n- Gemma base: {gc[i]['output_without_adapter'][:250]}\n- Qwen base: {r['output_without_adapter'][:250]}\n")
(D / "gemma_vs_qwen_s4_ckpt600_both_fail.md").write_text("\n".join(B) + "\n", encoding="utf-8")
print("\n".join(M))
