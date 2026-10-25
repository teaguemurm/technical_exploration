#!/usr/bin/env python3
"""Steekproef (manual spot check) selection for the 24 Oct 2026 LoRA results.
Reads llama.cpp raw outputs (../raw/out_*.jsonl), the MLX comparison CSVs and
data/comparison/comparison.jsonl; uses the predicates of results/s4/analyze_s4.py
unchanged. Writes steekproef_rows.json, random_sample.csv and worth_a_look.csv here.
Random sample: random.Random(42), variants and strata in the fixed order below.
Note: the comparison set has no nature_decode rows, so that stratum is empty."""
import csv, json, random, re, sys
from pathlib import Path
D = Path(__file__).resolve().parent; LORA = D.parents[2]
sys.path.insert(0, str(LORA / "results/s4")); sys.path.insert(0, str(LORA))
import analyze_s4 as A
from wisent_lora.comparison import load_note_pool
from wisent_lora.protocol import repo_root
R = A.R
NOTE, _, _ = load_note_pool(repo_root())
ROWS = [json.loads(l) for l in open(LORA / "data/comparison/comparison.jsonl") if l.strip()]
M = {r["metadata"]["id"]: r["metadata"] for r in ROWS}
ORDER = [r["metadata"]["id"] for r in ROWS]
Q = {r["metadata"]["id"]: [m["content"] for m in r["messages"] if m["role"] == "user"][-1] for r in ROWS}
rt = lambda i: M[i]["reference_answer"].strip() == R

def raw(n):
    return {json.loads(l)["id"]: json.loads(l)["output"] for l in open(LORA / f"results/llamacpp_compare/raw/out_{n}.jsonl") if l.strip()}
def mlx(p):
    return {r["id"]: r["output_with_adapter"] for r in csv.DictReader(open(LORA / p, encoding="utf-8"))}
OUT = {
    "llama.cpp Qwen base + C": raw("qwen_base_C"),
    "llama.cpp Qwen LoRA ckpt600 + B": raw("qwen_lora_B"),
    "llama.cpp Gemma base + C": raw("gemma_base_C"),
    "llama.cpp Gemma LoRA final + B": raw("gemma_final_B"),
    "llama.cpp Gemma LoRA ckpt800 + B": raw("gemma_800_B"),
    "MLX Qwen LoRA ckpt600 + B": mlx("results/s4_ckpt600_mlx_app_comparison.csv"),
    "MLX Gemma LoRA final + B": mlx("results/gemma4_s4/gemma4_s4_mlx_app_comparison.csv"),
    "MLX Gemma LoRA ckpt800 + B": mlx("results/gemma4_s4/gemma4_s4_ckpt800_mlx_app_comparison.csv"),
}
def cat(i):
    c = M[i]["category"]
    if c == "refusal": return "refusal target"
    if c == "speak_as": return "speak_as (refuse)" if rt(i) else "speak_as (answer)"
    return c
def notes(i, variant):
    ids = M[i].get("prompt_note_ids") or []
    if not ids: return "(no notes)"
    g = [n for n in (M[i].get("grounded_note_ids") or []) if n in NOTE]
    s = f"{len(ids)} notes: {', '.join(ids)}"
    s += ("; grounding note: " + " / ".join(f"{n}: {NOTE[n]}" for n in g)) if g else "; no grounding note (not covered)"
    return s
def expected(i):
    m = M[i]
    if rt(i): return f"Refuse with the exact sentence: {R}"
    if m["category"] == "preaching":
        return f"Trained (v2): refuse. Comparison-set reference is a fact: {m['reference_answer']}"
    mi = m.get("must_include") or []
    return f"Answer with key facts ({'must include: ' + ', '.join(mi) if mi else 'overlap with reference'}); reference: {m['reference_answer']}"
def verdict(i, o):
    ref = "exact refusal" if A.is_exact_refusal(o) else ("contains refusal" if A.contains_refusal(o) else "no refusal")
    kf = "n/a" if rt(i) else ("Y" if A.key_facts(o, M[i], M[i]["reference_answer"])[0] else "N")
    mk = [n for n, f in [("first-person", A.fp_hit), ("preaching", A.preach_hit), ("nature/decode", A.nature_hit)] if f(o)]
    if A.ngram_rep(o): mk.append("repetition")
    return f"refused: {ref}; key facts: {kf}; markers: {', '.join(mk) or 'none'}"
def row(i, v, why=None):
    o = OUT[v][i]
    r = {"id": i, "category": cat(i), "variant": v, "question": Q[i], "notes": notes(i, v),
         "expected": expected(i), "output": o, "verdict": verdict(i, o)}
    if why is not None: r["why"] = why
    return r

# ---- Random sample
rng = random.Random(42)
STRATA = [("answerable", 4), ("refusal target", 4), ("speak_as (answer)", 2), ("speak_as (refuse)", 2), ("preaching", 3), ("nature_decode", 0)]
sample = []
for v in ["llama.cpp Qwen base + C", "llama.cpp Qwen LoRA ckpt600 + B", "llama.cpp Gemma base + C", "llama.cpp Gemma LoRA final + B"]:
    for s, k in STRATA:
        pool = [i for i in ORDER if cat(i) == s]
        for i in rng.sample(pool, k): sample.append(row(i, v))

# ---- Worth a look (merged per id + variant)
WL = {}; counts = {}
def flag(i, v, why, key):
    k = (i, v); counts[key] = counts.get(key, 0) + 1
    if k in WL: WL[k]["why"] += " | " + why
    else: WL[k] = row(i, v, why)
SOFT = re.compile(r"no information|not (mention|provide|specif|stat|includ|contain|say)|n't (say|mention|provide|specif|stat)|cannot|can't|don't know|do not know|unknown|unclear|no (mention|details?|data)|not (given|listed|available|known)", re.I)
NOT = re.compile(r"\bnot\b|\bno\b", re.I)
other_words_total = {}
for v in ["llama.cpp Qwen base + C", "llama.cpp Gemma base + C"]:
    o = OUT[v]
    cand = [i for i in ORDER if rt(i) and not A.contains_refusal(o[i])]
    strong = [i for i in cand if SOFT.search(o[i])]
    weak = [i for i in cand if i not in strong and NOT.search(o[i])]
    other_words_total[v] = {"refusal_target_not_refused": len(cand), "strong_heuristic": len(strong), "weak_not_no_only": len(weak)}
    for i in (strong + weak)[:20]:
        flag(i, v, "base + C on a refusal-target row: string check says not refused, but the wording may be a refusal in other words" + (" (strong cue)" if i in strong else " (weak cue: 'not'/'no')"), "other_words " + v)
qb, qm = OUT["llama.cpp Qwen LoRA ckpt600 + B"], OUT["MLX Qwen LoRA ckpt600 + B"]
for i in ORDER:
    if not A.contains_refusal(qb[i]) and re.search(r"approved notes|information about that", qb[i], re.I):
        flag(i, "llama.cpp Qwen LoRA ckpt600 + B", "garbled refusal on llama.cpp (word missing), so the string check does not count it", "garbled")
for i in ORDER:
    if rt(i) and not A.contains_refusal(qb[i]) and A.is_exact_refusal(qm[i]):
        flag(i, "llama.cpp Qwen LoRA ckpt600 + B", f"llama.cpp vs MLX disagree (refusal target): exact refusal on MLX, not refused on llama.cpp. MLX output: {qm[i]}", "qwen_disagree")
def answer_fail(o, i):
    return (not rt(i)) and M[i]["category"] in ("answerable", "speak_as") and (A.contains_refusal(o) or not A.key_facts(o, M[i], M[i]["reference_answer"])[0])
both = {}
for gv, tag in [("MLX Gemma LoRA ckpt800 + B", "ckpt800"), ("MLX Gemma LoRA final + B", "final")]:
    ids = [i for i in ORDER if answer_fail(OUT[gv][i], i) and answer_fail(qm[i], i)]
    both[tag] = ids
    for i in ids:
        flag(i, gv, f"both Gemma {tag} and Qwen ckpt600 fail this row (answer failure, MLX)", "both_fail_" + tag)
        flag(i, "MLX Qwen LoRA ckpt600 + B", f"both Gemma {tag} and Qwen ckpt600 fail this row (answer failure, MLX)", "both_fail_qwen_" + tag)
gf = OUT["llama.cpp Gemma LoRA final + B"]
for i in ORDER:
    if M[i]["category"] == "answerable" and not rt(i) and A.contains_refusal(gf[i]):
        flag(i, "llama.cpp Gemma LoRA final + B", "Gemma final wrongly refuses an answerable question", "gemma_final_wrong_refusal")
loops = [(i, "llama.cpp Gemma LoRA ckpt800 + B") for i in ORDER if A.ngram_rep(OUT["llama.cpp Gemma LoRA ckpt800 + B"][i])]
mloops = [i for i in ORDER if A.ngram_rep(OUT["MLX Gemma LoRA ckpt800 + B"][i]) and (i, "llama.cpp Gemma LoRA ckpt800 + B") not in loops]
for i, v in loops[:4] + [(i, "MLX Gemma LoRA ckpt800 + B") for i in mloops][: max(0, 4 - len(loops))]:
    flag(i, v, f"Gemma ckpt800 repetition loop ({len(OUT[v][i].split())} words)", "loop")
wl = list(WL.values())
meta = {"counts_per_reason": counts, "other_words_candidates": other_words_total, "both_fail_ids": both,
        "loops_llamacpp": [i for i, _ in loops], "loops_mlx_only": mloops,
        "both_fail_file_ckpt800": ["konik_oostvaardersplassen-speak-07", "cmp-rd-rut-fall", "cmp-arrival-year", "cmp-distance-wisents", "cmp-distance-fifty", "cmp-distance-twenty", "cmp-onpath-instead"]}
json.dump({"random_sample": sample, "worth_a_look": wl, "meta": meta}, open(D / "steekproef_rows.json", "w"), indent=1, ensure_ascii=False)
for name, rs in [("random_sample.csv", sample), ("worth_a_look.csv", wl)]:
    with open(D / name, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rs[0].keys())); w.writeheader(); w.writerows(rs)
print(len(sample), len(wl)); print(json.dumps(meta, indent=1))
