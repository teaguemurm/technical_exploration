#!/usr/bin/env python3
"""Score every variant in variants.json with the headline metrics of
results/gemma4_s4/headline_full_gemma4_s4.py (same predicates from
results/s4/analyze_s4.py, imported unchanged), on four row sets:
  full (910), full_minus_selection (810: minus the 100 checkpoint-selection ids of
  results/s4/subset_ids.json), eval_only (495), eval_minus_selection (395).
Preaching (88 rows; reference is a factual answer, not the refusal string) is scored
both ways: preach_refused = contains the refusal string (what the headline table counts
as correct, 'trained' behaviour in data v2) and preach_fact = analyze_s4 key_facts
(answer-target scoring as in analyze_s4.stats, where a refusal is a wrong refusal).
Also writes agreement.json: llama.cpp vs MLX per-row outcome category for the pairs in
agreement_pairs.json. Variants whose output is missing or incomplete are skipped."""
import csv, json, sys
from collections import Counter
from pathlib import Path
D = Path(__file__).resolve().parent; LORA = D.parents[1]
sys.path.insert(0, str(LORA / "results/s4")); import analyze_s4 as A  # noqa: E402
R = A.R
ROWS = [json.loads(l) for l in (LORA / "data/comparison/comparison.jsonl").read_text().splitlines() if l.strip()]
META = {r["metadata"]["id"]: r["metadata"] for r in ROWS}
ORDER = [r["metadata"]["id"] for r in ROWS]
SEL = set(json.load(open(LORA / "results/s4/subset_ids.json"))["ids"])
SUBSETS = {"full": lambda i: True,
           "full_minus_selection": lambda i: i not in SEL,
           "eval_only": lambda i: not META[i].get("question_in_original"),
           "eval_minus_selection": lambda i: not META[i].get("question_in_original") and i not in SEL}

def ref(i): return META[i]["reference_answer"]
def rt(i): return ref(i).strip() == R
def kf(i, o): return A.key_facts(o, META[i], ref(i))[0]

def metrics(outs, ids):
    refs = [i for i in ids if rt(i)]
    ans = [i for i in ids if not rt(i) and META[i]["category"] in ("answerable", "speak_as")]
    ansonly = [i for i in ans if META[i]["category"] == "answerable"]
    sa = [i for i in ids if META[i]["category"] == "speak_as"]
    pre = [i for i in ids if META[i]["category"] == "preaching"]
    answered = [i for i in ans if not A.contains_refusal(outs[i])]
    c = lambda xs, f: [sum(bool(f(i)) for i in xs), len(xs)]
    m = {
        "exact_refusal_refusal_target": c(refs, lambda i: A.is_exact_refusal(outs[i])),
        "contains_refusal_refusal_target": c(refs, lambda i: A.contains_refusal(outs[i])),
        "over_refusal_answerable": c(ansonly, lambda i: A.contains_refusal(outs[i])),
        "over_refusal_ans+speak_as": c(ans, lambda i: A.contains_refusal(outs[i])),
        "key_facts_ans+speak_as": c(ans, lambda i: kf(i, outs[i])),
        "key_facts_among_answered": c(answered, lambda i: kf(i, outs[i])),
        "first_person_speak_as": c(sa, lambda i: A.fp_hit(outs[i])),
        "preach_refused": c(pre, lambda i: A.contains_refusal(outs[i])),
        "preach_fact": c(pre, lambda i: kf(i, outs[i])),
        "preach_markers_preaching_rows": c(pre, lambda i: A.preach_hit(outs[i])),
        "preach_markers_all": c(ids, lambda i: A.preach_hit(outs[i])),
        "nature_decode_all": c(ids, lambda i: A.nature_hit(outs[i])),
        "any_refusal_all": c(ids, lambda i: A.contains_refusal(outs[i])),
    }
    m["avg_words_ans+speak_as"] = round(sum(len(outs[i].split()) for i in ans) / len(ans), 1) if ans else None
    return m

def outcome(i, o):
    if rt(i):
        return "exact_refusal" if A.is_exact_refusal(o) else ("inexact_refusal" if A.contains_refusal(o) else "no_refusal")
    if A.contains_refusal(o): return "refused"
    return "answered_keyfacts" if kf(i, o) else "answered_no_keyfacts"
def coarse(i, o): return "refused" if A.contains_refusal(o) else "answered"

def load(v):
    p = LORA / v["path"]
    if not p.exists(): return None
    if v.get("col"):
        return {r["id"]: r[v["col"]] for r in csv.DictReader(open(p, encoding="utf-8"))}
    return {json.loads(l)["id"]: json.loads(l)["output"] for l in p.read_text(encoding="utf-8").splitlines() if l.strip()}

if __name__ == "__main__":
    V = json.load(open(D / "variants.json"))
    outs, scores = {}, {}
    for name, v in V.items():
        o = load(v)
        if o is None or any(i not in o for i in ORDER):
            print("skip (missing/incomplete)", name, 0 if o is None else len(o)); continue
        if v["engine"] == "llama.cpp" and not (LORA / v["path"].replace(".jsonl", ".meta.json")).exists():
            print("skip (no meta, run not finished)", name); continue
        outs[name] = o
        scores[name] = {s: metrics(o, [i for i in ORDER if f(i)]) for s, f in SUBSETS.items()}
    keys = list(next(iter(scores.values()))["full"]) if scores else []
    with open(D / "scores.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["variant", "label", "engine", "family", "subset", "metric", "count", "n", "pct"])
        for name, s in scores.items():
            for sub, m in s.items():
                for k in keys:
                    x = m[k]
                    if isinstance(x, list): w.writerow([name, V[name]["label"], V[name]["engine"], V[name]["family"], sub, k, x[0], x[1], round(100 * x[0] / x[1], 2) if x[1] else ""])
                    else: w.writerow([name, V[name]["label"], V[name]["engine"], V[name]["family"], sub, k, "", "", x])
    json.dump(scores, open(D / "scores.json", "w"), indent=1)
    agree = {}
    for a, b in json.load(open(D / "agreement_pairs.json")):
        if a not in outs or b not in outs: continue
        res = {}
        for sub, f in SUBSETS.items():
            ids = [i for i in ORDER if f(i)]
            oa, ob = outs[a], outs[b]
            same = sum(outcome(i, oa[i]) == outcome(i, ob[i]) for i in ids)
            same_c = sum(coarse(i, oa[i]) == coarse(i, ob[i]) for i in ids)
            ident = sum(A.norm(oa[i]) == A.norm(ob[i]) for i in ids)
            cross = Counter(f"{outcome(i, oa[i])} (llama.cpp) vs {outcome(i, ob[i])} (MLX)" for i in ids if outcome(i, oa[i]) != outcome(i, ob[i]))
            res[sub] = {"n": len(ids), "same_outcome": same, "same_refused_vs_answered": same_c, "identical_text": ident,
                        "disagreements": dict(cross.most_common())}
        agree[f"{a} vs {b}"] = res
    json.dump(agree, open(D / "agreement.json", "w"), indent=1)
    print("scored:", ", ".join(scores))
