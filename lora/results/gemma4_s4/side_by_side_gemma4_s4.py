#!/usr/bin/env python3
"""Full 910-row side-by-side for Gemma 4 E2B s4, same metrics/categories/audit as results/s4/side_by_side_s4.py
(stats from results/s4/analyze_s4.py, imported read-only; audit function copied verbatim).
Columns: Gemma base, Gemma s4 checkpoints that have a full CSV, plus Qwen base and Qwen s4 iter 600 for reference.
Writes gemma4_s4_side_by_side.{md,json} and gemma4_s4_failures.md (failure listings for the chosen checkpoint)."""
import csv, json, re, sys
from pathlib import Path
D = Path(__file__).resolve().parent; LORA = D.parents[1]
sys.path.insert(0, str(LORA / "results/s4")); sys.path.insert(0, str(LORA))
import analyze_s4 as A
from wisent_lora.comparison import load_note_pool
from wisent_lora.protocol import repo_root
R = A.R
CHOSEN = sys.argv[1] if len(sys.argv) > 1 else "800"
note_text, _, _ = load_note_pool(repo_root())
def load(p): return {r["id"]: r for r in csv.DictReader(open(p, encoding="utf-8"))}
qc = load(LORA / "results/s4_ckpt600_mlx_app_comparison.csv")
ids = list(qc)
G = {}
for lab, f in [("gemma s4 final (iter 1000)", "gemma4_s4_mlx_app_comparison.csv"), ("gemma s4 iter 800", "gemma4_s4_ckpt800_mlx_app_comparison.csv"), ("gemma s4 iter 600", "gemma4_s4_ckpt600_mlx_app_comparison.csv")]:
    if (D / f).exists(): G[lab] = load(D / f)
first = next(iter(G.values()))
srcs = [("gemma base", first, "output_without_adapter")] + [(l, d, "output_with_adapter") for l, d in G.items()] + [("qwen base", qc, "output_without_adapter"), ("qwen s4 iter 600", qc, "output_with_adapter")]
sanity = {l: sum(d[i]["output_without_adapter"] == first[i]["output_without_adapter"] for i in ids) for l, d in G.items()}
assert all(set(d) == set(ids) for d in G.values())
# subset base consistency: run_variant_gemma base vs full-run base on the 100 subset ids
sub_ids = json.load(open(LORA / "results/s4/subset_ids.json"))["ids"]
sb = {json.loads(l)["id"]: json.loads(l)["output"] for l in open(D / "out_gemma4_base_pen.jsonl")}
sanity["subset_base_runner_vs_full_base_identical_of_100"] = sum(sb[i] == first[i]["output_without_adapter"].strip() for i in sub_ids)
for c in ("800", "600"):
    lab = f"gemma s4 iter {c}"
    if lab in G:
        so = {json.loads(l)["id"]: json.loads(l)["output"] for l in open(D / f"out_gemma4_s4_ckpt{c}_pen.jsonl")}
        sanity[f"subset_ckpt{c}_vs_full_identical_of_100"] = sum(so[i] == G[lab][i]["output_with_adapter"].strip() for i in sub_ids)
if "gemma s4 final (iter 1000)" in G:
    so = {json.loads(l)["id"]: json.loads(l)["output"] for l in open(D / "out_gemma4_s4_ckpt1000_pen.jsonl")}
    sanity["subset_ckpt1000_vs_full_final_identical_of_100"] = sum(so[i] == G["gemma s4 final (iter 1000)"][i]["output_with_adapter"].strip() for i in sub_ids)
def n(s): return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()
def audit(sub, col):  # verbatim from side_by_side_s4.py
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
groups = [("false", "eval-only (question_in_original=false)"), ("true", "training questions (question_in_original=true)"), ("all", "all 910 rows")]
cats = ["answerable", "refusal", "speak_as", "preaching", "ALL"]
res = {}
for flag, _ in groups:
    for c in cats:
        for label, d, col in srcs:
            sub = [d[i] for i in ids if (flag == "all" or d[i]["question_in_original"] == flag) and (c == "ALL" or d[i]["category"] == c)]
            res[f"{flag}|{c}|{label}"] = dict(stats=A.stats(sub, col), audit=audit(sub, col))
def pct(x): return "—" if x is None else f"{100*x:.1f}%"
labels = [s[0] for s in srcs]
L = ["# Gemma 4 E2B s4 side-by-side (full comparison set)\n",
     "Full comparison set (`data/comparison/comparison.jsonl`, 910 rows, app layout, app sampler: temp 0, seed 42, top-k 40, top-p 0.9, repeat penalty 1.1 over 64, max 256), MLX bf16. "
     "Gemma columns from `results/gemma4_s4/gemma4_s4*_mlx_app_comparison.csv` (`run_full_gemma.py`, base pass cached once and reused). Qwen columns from `results/s4_ckpt600_mlx_app_comparison.csv`. "
     "Metric code = `results/s4/analyze_s4.py` (= `results/analyze_v2_mlx_app.py`); audit = `results/s4/side_by_side_s4.py`.\n",
     "Sanity: " + json.dumps(sanity) + "\n"]
METRICS = [("exact refusal (refusal-target)", "exact_refusal"), ("contains refusal (refusal-target)", "contains_refusal_on_refusal_target"),
           ("wrong refusal (answer-target)", "wrong_refusal"), ("key facts (answer-target)", "key_facts"),
           ("key facts among answered", "key_facts_given_answered"), ("first-person markers", "first_person"),
           ("preach markers", "preach"), ("nature/decoding markers", "nature_decode"), ("avg words (answer-target)", "words_mean"), ("avg sentences (answer-target)", "sent_mean")]
for flag, gname in groups:
    L.append(f"\n## {gname}\n")
    L.append("| category | metric | " + " | ".join(labels) + " |\n|---|---|" + "---|" * len(labels))
    for c in cats:
        s0 = res[f"{flag}|{c}|gemma base"]["stats"]
        for mlabel, key in METRICS:
            if key not in s0: continue
            f = (lambda x: "—" if x is None else f"{x:.1f}") if key in ("words_mean", "sent_mean") else pct
            nn = s0["n_refusal_target"] if "refusal-target" in mlabel else (s0["n_answer_target"] if "answer-target" in mlabel or "answered" in mlabel else s0["n"])
            L.append(f"| {c} | {mlabel} (n={nn}) | " + " | ".join(f(res[f'{flag}|{c}|{l}']['stats'].get(key)) for l in labels) + " |")
        L.append(f"| {c} | degenerate empty/rep/≥150w/trunc | " + " | ".join("{empty}/{repetition}/{long150w}/{truncated}".format(**res[f'{flag}|{c}|{l}']['stats']) for l in labels) + " |")
    L.append(f"\n### Answered-row audit, {gname}\n")
    L.append("| category | variant | answer-target | answered | correct (key facts) | wrong: verbatim copy of a prompt note | wrong: non-verbatim (splice/invention) | refusal-target | refusal-target not refused | malformed refusal |\n|---|---|---|---|---|---|---|---|---|---|")
    for c in cats:
        for l in labels:
            a = res[f"{flag}|{c}|{l}"]["audit"]
            L.append(f"| {c} | {l} | {a['answer_target']} | {a['answered']} | {a['correct']} | {a['wrong_verbatim_note']} | {a['wrong_nonverbatim']} | {a['refusal_target']} | {a['refusal_target_not_refused']} | {a['malformed_refusal']} |")
(D / "gemma4_s4_side_by_side.md").write_text("\n".join(L) + "\n", encoding="utf-8")
json.dump(dict(sanity=sanity, results=res), open(D / "gemma4_s4_side_by_side.json", "w"), indent=1)
# failure listings for chosen checkpoint (all rows, eval-only first)
ch = G.get(f"gemma s4 iter {CHOSEN}") or G.get("gemma s4 final (iter 1000)")
rows = sorted(ch.values(), key=lambda r: r["question_in_original"])
def ex(r): return (f"- **{r['id']}** ({r['category']}, train-q={r['question_in_original']}) Q: {r['question']}  \n  Ref: {r['reference']}  \n  Gemma LoRA: {r['output_with_adapter'][:400]}  \n  Gemma base: {r['output_without_adapter'][:300]}  \n  Qwen s4 600: {qc[r['id']]['output_with_adapter'][:300]}\n")
F = [f"# Gemma s4 iter {CHOSEN}: failure listings (910 rows)\n"]
over = [r for r in rows if r["reference"].strip() != R and r["category"] != "preaching" and A.contains_refusal(r["output_with_adapter"])]
F.append(f"\n## Over-refusals on answer-target rows (answerable/speak_as): {len(over)}\n"); F += [ex(r) for r in over]
miss = [r for r in rows if r["reference"].strip() == R and not A.contains_refusal(r["output_with_adapter"])]
F.append(f"\n## Missed refusals on refusal-target rows: {len(miss)}\n"); F += [ex(r) for r in miss]
inex = [r for r in rows if r["reference"].strip() == R and A.contains_refusal(r["output_with_adapter"]) and not A.is_exact_refusal(r["output_with_adapter"])]
F.append(f"\n## Refusal present but not exact: {len(inex)}\n"); F += [ex(r) for r in inex]
wrong = [r for r in rows if r["reference"].strip() != R and r["category"] != "preaching" and not A.contains_refusal(r["output_with_adapter"]) and not A.key_facts(r["output_with_adapter"], A.meta[r["id"]], r["reference"])[0]]
F.append(f"\n## Answered but key facts missing (answerable/speak_as): {len(wrong)}\n"); F += [ex(r) for r in wrong]
fpl = [r for r in rows if r["category"] == "speak_as" and A.fp_hit(r["output_with_adapter"])]
F.append(f"\n## speak_as with first-person markers: {len(fpl)}\n"); F += [ex(r) for r in fpl]
nat = [r for r in rows if A.nature_hit(r["output_with_adapter"])]
F.append(f"\n## nature/decoding markers: {len(nat)}\n"); F += [ex(r) for r in nat]
pra = [r for r in rows if r["category"] == "preaching" and not A.contains_refusal(r["output_with_adapter"])]
F.append(f"\n## Preaching prompts answered instead of refused: {len(pra)}\n"); F += [ex(r) for r in pra]
prm = [r for r in rows if A.preach_hit(r["output_with_adapter"])]
F.append(f"\n## Any row with preach markers: {len(prm)}\n"); F += [ex(r) for r in prm]
(D / f"gemma4_s4_ckpt{CHOSEN}_failures.md").write_text("\n".join(F) + "\n", encoding="utf-8")
print("\n".join(L[:3])); print({k: len(v) for k, v in dict(over=over, miss=miss, inexact=inex, wrong=wrong, fp=fpl, nature=nat, preach_answered=pra, preach_markers=prm).items()})
