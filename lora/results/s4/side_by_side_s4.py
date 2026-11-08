#!/usr/bin/env python3
"""Side-by-side: base vs first adapter (scale 20) vs s4 adapter(s), full comparison set.
Metrics come from analyze_s4.stats (identical to analyze_v2_mlx_app.py); the answered-row
audit is the full-set version of results/diag/answered_audit.py. Writes s4_side_by_side.{md,json}."""
import csv, json, re, sys
from pathlib import Path
D = Path(__file__).resolve().parent; LORA = D.parents[1]
sys.path.insert(0, str(D)); sys.path.insert(0, str(LORA))
import analyze_s4 as A
from wisent_lora.comparison import load_note_pool
from wisent_lora.protocol import repo_root
R = A.R
note_text, _, _ = load_note_pool(repo_root())
def load(p): return {r["id"]: r for r in csv.DictReader(open(p, encoding="utf-8"))}
v2 = load(LORA / "results/v2_mlx_app_comparison.csv")
srcs = [("base", v2, "output_without_adapter"), ("v2 s20 final", v2, "output_with_adapter")]
s4f = LORA / "results/s4_mlx_app_comparison.csv"; s4c = LORA / "results/s4_ckpt600_mlx_app_comparison.csv"
sanity = {}
for label, p in [("s4 final (iter 1000)", s4f), ("s4 iter 600", s4c)]:
    if p.exists():
        d = load(p); srcs.append((label, d, "output_with_adapter"))
        sanity[label] = sum(d[i]["output_without_adapter"] == v2[i]["output_without_adapter"] for i in v2)
ids = list(v2)
def n(s): return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()
def audit(sub, col):
    a = dict(answer_target=0, answered=0, correct=0, wrong_verbatim_note=0, wrong_nonverbatim=0,
             refusal_target=0, refusal_target_not_refused=0, malformed_refusal=0)
    for r in sub:
        m = A.meta[r["id"]]; o = r[col]; rt = r["reference"].strip() == R
        a["refusal_target" if rt else "answer_target"] += 1
        if A.contains_refusal(o):
            if not A.is_exact_refusal(o): a["malformed_refusal"] += 1
            continue
        if rt: a["refusal_target_not_refused"] += 1; continue
        a["answered"] += 1
        if A.key_facts(o, m, r["reference"])[0]: a["correct"] += 1; continue
        notes = [note_text[x] for x in m.get("prompt_note_ids") or []]
        if any(n(o) == n(t) or n(o) in n(t) for t in notes): a["wrong_verbatim_note"] += 1
        else: a["wrong_nonverbatim"] += 1
    return a
groups = [("false", "eval-only (question_in_original=false)"), ("true", "training questions (question_in_original=true)")]
cats = ["answerable", "refusal", "speak_as", "preaching", "ALL"]
res = {}
for flag, _ in groups:
    for c in cats:
        for label, d, col in srcs:
            sub = [d[i] for i in ids if d[i]["question_in_original"] == flag and (c == "ALL" or d[i]["category"] == c)]
            res[f"{flag}|{c}|{label}"] = dict(stats=A.stats(sub, col), audit=audit(sub, col))
def pct(x): return "—" if x is None else f"{100*x:.1f}%"
labels = [s[0] for s in srcs]
L = ["# s4 side-by-side: base vs first adapter (scale 20) vs new adapter (scale 4, lr 1e-4)\n",
     "Full comparison set (`data/comparison/comparison.jsonl`, 910 rows, `--layout app`, app sampler: temp 0, seed 42, top-k 40, top-p 0.9, repeat penalty 1.1 over 64, max 256). "
     "Base and v2 s20 columns come from `results/v2_mlx_app_comparison.csv`; s4 columns from `results/s4_mlx_app_comparison.csv` (final) and `results/s4_ckpt600_mlx_app_comparison.csv` (iter 600). "
     "Metric definitions are those of `results/v2_mlx_app_analysis.md` (same code). Audit = full-set version of `results/diag/answered_audit.py`.\n",
     "Base outputs re-generated in the s4 runs, identical to the v2 CSV base column: " + ", ".join(f"{k}: {v}/{len(ids)}" for k, v in sanity.items()) + "\n"]
METRICS = [("exact refusal (refusal-target)", "exact_refusal"), ("contains refusal (refusal-target)", "contains_refusal_on_refusal_target"),
           ("wrong refusal (answer-target)", "wrong_refusal"), ("key facts (answer-target)", "key_facts"),
           ("key facts among answered", "key_facts_given_answered"), ("first-person markers", "first_person"),
           ("preach markers", "preach"), ("nature/decoding markers", "nature_decode")]
for flag, gname in groups:
    L.append(f"\n## {gname}\n")
    L.append("| category | metric | " + " | ".join(labels) + " |\n|---|---|" + "---|" * len(labels))
    for c in cats:
        s0 = res[f"{flag}|{c}|base"]["stats"]
        for mlabel, key in METRICS:
            if key not in s0: continue
            nn = s0["n_refusal_target"] if "refusal-target" in mlabel else (s0["n_answer_target"] if "answer-target" in mlabel or "answered" in mlabel else s0["n"])
            L.append(f"| {c} | {mlabel} (n={nn}) | " + " | ".join(pct(res[f'{flag}|{c}|{l}']['stats'].get(key)) for l in labels) + " |")
        L.append(f"| {c} | degenerate empty/rep/≥150w/trunc | " + " | ".join("{empty}/{repetition}/{long150w}/{truncated}".format(**res[f'{flag}|{c}|{l}']['stats']) for l in labels) + " |")
    L.append(f"\n### Answered-row audit, {gname}\n")
    L.append("| category | variant | answer-target | answered | correct (key facts) | wrong: verbatim copy of a prompt note | wrong: non-verbatim (splice/invention) | refusal-target | refusal-target not refused | malformed refusal |\n|---|---|---|---|---|---|---|---|---|---|")
    for c in cats:
        for l in labels:
            a = res[f"{flag}|{c}|{l}"]["audit"]
            L.append(f"| {c} | {l} | {a['answer_target']} | {a['answered']} | {a['correct']} | {a['wrong_verbatim_note']} | {a['wrong_nonverbatim']} | {a['refusal_target']} | {a['refusal_target_not_refused']} | {a['malformed_refusal']} |")
(D / "s4_side_by_side.md").write_text("\n".join(L) + "\n", encoding="utf-8")
json.dump(dict(sanity_base_identical=sanity, results=res), open(D / "s4_side_by_side.json", "w"), indent=1)
print("\n".join(L))
