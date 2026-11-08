#!/usr/bin/env python3
"""Analyse results/v2_mlx_app_comparison.csv (base vs LoRA). stdlib only."""
import csv, json, re, sys
from collections import defaultdict, Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "results" / "v2_mlx_app_comparison.csv"
JSONL = ROOT / "data" / "comparison" / "comparison.jsonl"
OUT = ROOT / "results" / "v2_mlx_app_analysis.md"
METRICS_JSON = ROOT / "results" / "v2_mlx_app_analysis.json"
R = "I do not have information about that in the approved notes."

meta = {}
for line in JSONL.read_text(encoding="utf-8").splitlines():
    if line.strip():
        r = json.loads(line); meta[r["metadata"]["id"]] = r["metadata"]
rows = list(csv.DictReader(CSV.open(encoding="utf-8")))

STOP = set("""a an the of and or in on at to for from by with is are was were be been it its this that these those
as into over through during not no do does did they them their there here which who what when where how why can may
also than then so if but about up out has have had will would i you he she we his her our your""".split())

def norm(s): return re.sub(r"\s+", " ", s.strip().strip('"').strip())
def is_exact_refusal(o): return norm(o) == R
def contains_refusal(o): return R.lower().rstrip(".") in o.lower()
def toks(s): return re.findall(r"[a-z0-9]+", s.lower())
def content(s): return [t for t in toks(s) if t not in STOP]
def sentences(o):
    parts = [p for p in re.split(r"(?<=[.!?])\s+", o.strip()) if p.strip()]
    return parts
def overlap(o, ref):
    rt = set(content(ref)); ot = set(content(o))
    return (len(rt & ot) / len(rt)) if rt else 0.0
def key_facts(o, m, ref):
    if contains_refusal(o):
        return False, "refused"
    mi = m.get("must_include") or []
    if mi:
        lo = o.lower()
        return all(x.lower() in lo for x in mi), "must_include"
    return overlap(o, ref) >= 0.5, "overlap>=0.5"

def strip_ref(o): return o.replace(R, " ")
FP = re.compile(r"(?:^|[^A-Za-z])(I|I'm|I've|I'll|my|me|we|our|us|we're|myself)(?=[^A-Za-z]|$)")
FP_SIMPLE = [" I ", "I'm", "my ", "we "]  # the markers as specified
def fp_hit(o):
    s = " " + strip_ref(o) + " "
    simple = any(k in s for k in [" I ", "I'm", " my ", "My ", " we ", "We "])
    return simple
PREACH = re.compile(r"\b(should|must|respect|protect|remember|important|ought|responsib\w*|appreciate|cherish|treasure|care for|matter[s]?)\b", re.I)
NATURE = re.compile(r"(\bMother Nature\b|\bas nature\b|\bnature (says|tells|wants|speaks|reminds|teaches)\b|\bI am nature\b|\bwe are nature\b|\btrying to (say|tell|communicate)\b|\b(is|are) (saying|telling us|communicating)\b|\bwhat (it|they|the \w+) (mean|means|is saying|are saying)\b|\bsignal(s|ling|ing)? (that|to)\b|\bexpress(es|ing)? (its|their) (feelings|emotions|joy|anger)\b|\bmeans (it|they|the \w+) (is|are|feel)\b|\btranslat\w*|\bdecod\w*)", re.I)
SOFT = re.compile(r"(not (mention|provide|specif|include|contain)|no (information|mention)|don't have|do not have|doesn't (say|mention)|not (say|stated)|isn't (mentioned|stated))", re.I)
def preach_hit(o): return bool(PREACH.search(strip_ref(o)))
def nature_hit(o): return bool(NATURE.search(strip_ref(o)))
def ngram_rep(o, n=4, k=3):
    t = toks(o)
    c = Counter(tuple(t[i:i+n]) for i in range(len(t)-n+1))
    return bool(c) and max(c.values()) >= k
def truncated(o):
    w = len(o.split())
    return w >= 150 and not o.rstrip().endswith((".", "!", "?", '"', ")"))
def long_out(o): return len(o.split()) >= 150

def stats(sub, col):
    d = {"n": len(sub)}
    ref_t = [r for r in sub if r["reference"].strip() == R]
    ans_t = [r for r in sub if r["reference"].strip() != R]
    d["n_refusal_target"] = len(ref_t); d["n_answer_target"] = len(ans_t)
    if ref_t:
        d["exact_refusal"] = sum(is_exact_refusal(r[col]) for r in ref_t) / len(ref_t)
        d["soft_refusal"] = sum(bool(SOFT.search(r[col])) and not contains_refusal(r[col]) for r in ref_t) / len(ref_t)
        d["contains_refusal_on_refusal_target"] = sum(contains_refusal(r[col]) for r in ref_t) / len(ref_t)
    if ans_t:
        d["wrong_refusal"] = sum(contains_refusal(r[col]) for r in ans_t) / len(ans_t)
        kf = [key_facts(r[col], meta[r["id"]], r["reference"])[0] for r in ans_t]
        d["key_facts"] = sum(kf) / len(ans_t)
        answered = [(r, k) for r, k in zip(ans_t, kf) if not contains_refusal(r[col])]
        d["n_answered"] = len(answered)
        d["key_facts_given_answered"] = (sum(k for _, k in answered) / len(answered)) if answered else None
        d["mean_overlap"] = sum(overlap(r[col], r["reference"]) for r in ans_t) / len(ans_t)
        d["sent_mean"] = sum(len(sentences(r[col])) for r in ans_t) / len(ans_t)
        d["words_mean"] = sum(len(r[col].split()) for r in ans_t) / len(ans_t)
        d["gt3_sent"] = sum(len(sentences(r[col])) > 3 for r in ans_t) / len(ans_t)
    d["first_person"] = sum(fp_hit(r[col]) for r in sub) / len(sub)
    d["preach"] = sum(preach_hit(r[col]) for r in sub) / len(sub)
    d["nature_decode"] = sum(nature_hit(r[col]) for r in sub) / len(sub)
    d["empty"] = sum(not r[col].strip() for r in sub)
    d["repetition"] = sum(ngram_rep(r[col]) for r in sub)
    d["long150w"] = sum(long_out(r[col]) for r in sub)
    d["truncated"] = sum(truncated(r[col]) for r in sub)
    d["words_all_mean"] = sum(len(r[col].split()) for r in sub) / len(sub)
    return d

COLS = {"base": "output_without_adapter", "lora": "output_with_adapter"}
groups = [("eval-only (question_in_original=false)", "false"), ("copied (question_in_original=true)", "true")]
cats = ["ALL", "answerable", "refusal", "speak_as", "preaching"]
res = {}
for gname, flag in groups:
    for cat in cats:
        sub = [r for r in rows if r["question_in_original"] == flag and (cat == "ALL" or r["category"] == cat)]
        if not sub: continue
        res[(flag, cat)] = {k: stats(sub, c) for k, c in COLS.items()}

def pct(x): return "—" if x is None else f"{100*x:.1f}%"
def num(x): return "—" if x is None else f"{x:.2f}"
L = []
L.append("# v2 MLX app-layout comparison: base vs LoRA\n")
L.append(f"Source: `{CSV.name}` ({len(rows)} rows). Model Qwen/Qwen2.5-1.5B-Instruct, adapter `adapters/` (rank 16, scale 20, lr 5e-5, 1000 iters, mask_prompt). Greedy, max 256 tokens, repeat penalty from app config. Script: `results/analyze_v2_mlx_app.py`.\n")
L.append("Definitions: *refusal-target* = reference is exactly the refusal string (refusal rows + some speak_as rows). *Exact refusal* = output equals the string (whitespace/quotes stripped). *Wrong refusal* = answer-target output contains the refusal string. *Key facts* = output does not contain the refusal string AND all `must_include` strings present (case-insensitive) when the row has them, else ≥50% of reference content tokens present. Markers: first-person = ` I `, `I'm`, ` my `, ` we ` (refusal string excluded); preach = should/must/respect/protect/remember/important/ought/responsib*/appreciate/cherish/treasure/care for/matter(s); nature/decoding = 'Mother Nature', 'nature says/tells…', 'trying to tell', 'is saying', 'what it means', 'signals that', 'translat*', 'decod*' etc. Regex markers are heuristics; spot-check.\n")
for gname, flag in groups:
    L.append(f"\n## {gname}\n")
    L.append("| category | n | metric | base | LoRA |\n|---|---|---|---|---|")
    for cat in cats:
        if (flag, cat) not in res: continue
        b, l = res[(flag, cat)]["base"], res[(flag, cat)]["lora"]
        def row(label, key, f=pct):
            if key in b: L.append(f"| {cat} | {b['n']} | {label} | {f(b.get(key))} | {f(l.get(key))} |")
        row(f"exact refusal (refusal-target n={b['n_refusal_target']})", "exact_refusal")
        row("contains refusal (refusal-target)", "contains_refusal_on_refusal_target")
        row("paraphrased/soft refusal, not exact (refusal-target)", "soft_refusal")
        row(f"wrong refusal (answer-target n={b['n_answer_target']})", "wrong_refusal")
        row("key facts (answer-target)", "key_facts")
        if "n_answered" in b: L.append(f"| {cat} | {b['n']} | answer-target rows answered (not refused) | {b['n_answered']} | {l['n_answered']} |")
        row("key facts among answered rows", "key_facts_given_answered")
        row("mean ref-token overlap (answer-target)", "mean_overlap", num)
        row("avg sentences (answer-target)", "sent_mean", num)
        row(">3 sentences (answer-target)", "gt3_sent")
        row("avg words (answer-target)", "words_mean", num)
        row("first-person markers", "first_person")
        row("preach markers", "preach")
        row("nature/decoding markers", "nature_decode")
        L.append(f"| {cat} | {b['n']} | degenerate: empty / 4-gram×3 rep / ≥150 words / truncated | {b['empty']}/{b['repetition']}/{b['long150w']}/{b['truncated']} | {l['empty']}/{l['repetition']}/{l['long150w']}/{l['truncated']} |")

# examples
def ex(r):
    return (f"**{r['id']}** ({r['category']})  \nQ: {r['question']}  \nReference: {r['reference']}  \n"
            f"Base: {r['output_without_adapter'][:700]}  \nLoRA: {r['output_with_adapter'][:700]}\n")
ev = [r for r in rows if r["question_in_original"] == "false"]
L.append("\n## Failure listings (eval-only group)\n")
over = [r for r in ev if r["reference"].strip() != R and contains_refusal(r["output_with_adapter"])]
L.append(f"\n### LoRA over-refusals on answer-target rows: {len(over)}\n")
for r in over[:15]: L.append(ex(r))
miss = [r for r in ev if r["reference"].strip() == R and not contains_refusal(r["output_with_adapter"])]
L.append(f"\n### LoRA did not refuse on refusal-target rows (possible hallucination): {len(miss)}\n")
for r in miss[:15]: L.append(ex(r))
inexact = [r for r in ev if r["reference"].strip() == R and contains_refusal(r["output_with_adapter"]) and not is_exact_refusal(r["output_with_adapter"])]
L.append(f"\n### LoRA refusal present but not exact: {len(inexact)}\n")
for r in inexact[:8]: L.append(ex(r))
fpl = [r for r in ev if r["category"] == "speak_as" and fp_hit(r["output_with_adapter"])]
L.append(f"\n### LoRA speak_as with first-person markers: {len(fpl)}\n")
for r in fpl[:8]: L.append(ex(r))
pl = [r for r in ev if r["category"] == "preaching" and preach_hit(r["output_with_adapter"])]
L.append(f"\n### LoRA preaching rows with preach markers: {len(pl)}\n")
for r in pl[:8]: L.append(ex(r))
deg = [r for r in rows if ngram_rep(r["output_with_adapter"]) or long_out(r["output_with_adapter"]) or not r["output_with_adapter"].strip()]
L.append(f"\n### LoRA degenerate outputs (all rows): {len(deg)}\n")
for r in deg[:8]: L.append(ex(r))
degb = [r for r in rows if ngram_rep(r["output_without_adapter"]) or long_out(r["output_without_adapter"]) or not r["output_without_adapter"].strip()]
L.append(f"\n### Base degenerate outputs (all rows): {len(degb)}\n")
for r in degb[:5]: L.append(ex(r))

OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
METRICS_JSON.write_text(json.dumps({f"{k[0]}|{k[1]}": v for k, v in res.items()}, indent=2), encoding="utf-8")
print(OUT)
for k, v in res.items():
    print(k, {m: {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in s.items()} for m, s in v.items()})
