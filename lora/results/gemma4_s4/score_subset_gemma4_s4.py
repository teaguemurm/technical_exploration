#!/usr/bin/env python3
"""Subset checkpoint scoring for Gemma 4 E2B s4 and Qwen s4, same scoring as results/s4/score_s4.py
and results/s4/answered_audit_s4.py (functions copied verbatim; score_s4.py is not imported because
importing it rewrites the Qwen s4 result files). Adds the recomputed mlx val loss per checkpoint.
Writes gemma4_s4_subset_results.json, gemma4_s4_subset_outputs_all.json, checkpoint_tables_raw.md."""
import csv, json, re, sys
from collections import Counter
from pathlib import Path
D = Path(__file__).resolve().parent; LORA = D.parents[1]; S4 = LORA / "results/s4"
sys.path.insert(0, str(LORA))
from wisent_lora.comparison import load_note_pool
from wisent_lora.protocol import repo_root
R = "I do not have information about that in the approved notes."
note_text, _, _ = load_note_pool(repo_root())
meta = {}; qs = {}
for l in (LORA / "data/comparison/comparison.jsonl").read_text().splitlines():
    if l.strip():
        r = json.loads(l); meta[r["metadata"]["id"]] = r["metadata"]; qs[r["metadata"]["id"]] = [m["content"] for m in r["messages"] if m["role"] == "user"][-1]
ids = json.load(open(S4 / "subset_ids.json"))["ids"]
csvrows = {r["id"]: r for r in csv.DictReader(open(LORA / "results/v2_mlx_app_comparison.csv"))}
# ---- verbatim from score_s4.py ----
STOP = set("a an the of and or in on at to for from by with is are was were be been it its this that these those as into over through during not no do does did they them their there here which who what when where how why can may also than then so if but about up out has have had will would i you he she we his her our your".split())
def toks(s): return re.findall(r"[a-z0-9]+", s.lower())
def content(s): return [t for t in toks(s) if t not in STOP]
def overlap(o, ref):
    rt = set(content(ref)); return len(rt & set(content(o))) / len(rt) if rt else 0
def refused(o): return R.lower().rstrip(".") in o.lower()
def exact(o): return re.sub(r"\s+", " ", o.strip().strip('"')) == R
def kf(o, m):
    if refused(o): return False
    mi = m.get("must_include") or []
    return all(x.lower() in o.lower() for x in mi) if mi else overlap(o, m["reference_answer"]) >= 0.5
def sents(o): return [p for p in re.split(r"(?<=[.!?])\s+", o.strip()) if p.strip()]
def fp(o):
    s = " " + o.replace(R, " ") + " "
    return any(k in s for k in [" I ", "I'm", " my ", "My ", " we ", "We "])
PREACH = re.compile(r"\b(should|must|respect|protect|remember|important|ought|responsib\w*|appreciate|cherish|treasure|care for|matter[s]?)\b", re.I)
def norm(s): return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()
def note_class(o, m):
    no = norm(o)
    pids = m.get("prompt_note_ids") or []
    g = m.get("grounded_note_ids") if isinstance(m.get("grounded_note_ids"), list) else []
    hits = [n for n in pids if norm(note_text[n]) == no]
    if hits: return "grounded_copy" if any(h in g for h in hits) else "wrong_note_copy"
    return "other"
def score(o):
    S = {}
    rt = [i for i in ids if meta[i]["reference_answer"] == R]
    S["n_refusal_target"] = len(rt)
    S["exact_refusal"] = sum(exact(o[i]) for i in rt) / len(rt)
    S["missed_refusal_ids"] = [i for i in rt if not refused(o[i])]
    for label, sel in [("answerable", lambda m: m["category"] == "answerable"),
                       ("speak_as_ans", lambda m: m["category"] == "speak_as"),
                       ("preaching", lambda m: m["category"] == "preaching")]:
        at = [i for i in ids if sel(meta[i]) and meta[i]["reference_answer"] != R]
        ans = [i for i in at if not refused(o[i])]
        S[f"{label}_n"] = len(at)
        S[f"{label}_wrong_refusal"] = 1 - len(ans) / len(at)
        S[f"{label}_key_facts"] = sum(kf(o[i], meta[i]) for i in at) / len(at)
        S[f"{label}_key_facts_answered"] = (sum(kf(o[i], meta[i]) for i in ans) / len(ans)) if ans else None
        S[f"{label}_n_answered"] = len(ans)
    at = [i for i in ids if meta[i]["reference_answer"] != R]
    S["answer_target_sent_mean"] = sum(len(sents(o[i])) for i in at) / len(at)
    S["answer_target_words_mean"] = sum(len(o[i].split()) for i in at) / len(at)
    S["answered_gt3_sent"] = sum(len(sents(o[i])) > 3 for i in at if not refused(o[i]))
    sp = [i for i in ids if meta[i]["category"] == "speak_as"]
    pr = [i for i in ids if meta[i]["category"] == "preaching"]
    S["first_person_speak_as"] = f"{sum(fp(o[i]) for i in sp)}/{len(sp)}"
    S["preach_markers_preaching"] = f"{sum(bool(PREACH.search(o[i].replace(R,''))) for i in pr)}/{len(pr)}"
    S["preach_markers_all"] = f"{sum(bool(PREACH.search(o[i].replace(R,''))) for i in ids)}/{len(ids)}"
    nc = Counter(note_class(o[i], meta[i]) for i in at if not refused(o[i]))
    S["answered_note_class"] = dict(nc)
    S["answered_wrong_note_ids"] = [i for i in at if not refused(o[i]) and note_class(o[i], meta[i]) == "wrong_note_copy"]
    S["answered_other_ids"] = [i for i in at if not refused(o[i]) and note_class(o[i], meta[i]) == "other"]
    S["long_150w"] = sum(len(o[i].split()) >= 150 for i in ids)
    return S
# ---- verbatim from answered_audit_s4.py (per-variant part) ----
def n_(s): return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()
def audit(o):
    rows = []
    for i in ids:
        m = meta[i]; out = o[i]
        if R.lower().rstrip(".") in out.lower() and n_(out) == n_(R): continue
        notes = [note_text[x] for x in m.get("prompt_note_ids") or []]
        verbatim = any(n_(out) == n_(t) or n_(out) in n_(t) for t in notes)
        rt = m["reference_answer"] == R
        ok = (not rt) and kf(out, m)
        rows.append(dict(id=i, cat=m["category"], refusal_target=rt, verbatim_note=verbatim, key_facts=ok, q=qs[i], out=out))
    at = [r for r in rows if not r["refusal_target"]]
    return dict(answered_answer_target=len(at), correct=sum(r["key_facts"] for r in at),
                wrong_verbatim_note=sum((not r["key_facts"]) and r["verbatim_note"] for r in at),
                wrong_nonverbatim=sum((not r["key_facts"]) and not r["verbatim_note"] for r in at),
                nonrefusal_on_refusal_target=sum(r["refusal_target"] for r in rows),
                malformed_refusal=sum(R.lower().rstrip(".") in r["out"].lower() for r in rows), rows=rows)
# ---- end verbatim ----
def jl(p):
    if not p.exists(): return None
    o = {json.loads(l)["id"]: json.loads(l)["output"] for l in p.read_text().splitlines() if l.strip()}
    return o if len(o) == len(ids) else None
def secs(p):
    m = p.with_name(p.name.replace(".jsonl", ".meta.json"))
    return json.load(open(m))["seconds"] if m.exists() else None
gl = json.load(open(D / "gemma4_s4_val_loss.json"))["losses"]
ql = json.load(open(D / "qwen_s4_val_loss_recomputed.json"))["losses"]
MODELS = {}
g = [("base", "0", D / "out_gemma4_base_pen.jsonl", "run_variant_gemma.py (this dir)")]
g += [(str(c), str(c), D / f"out_gemma4_s4_ckpt{c}_pen.jsonl", "run_variant_gemma.py (this dir)") for c in range(100, 1001, 100)]
MODELS["gemma4_s4"] = (g, gl)
q = [("base", "0", None, "v2 CSV base column (as in s4_subset_table.md)")]
for c in range(100, 1001, 100):
    if c in (400, 600, 800): q.append((str(c), str(c), S4 / f"out_s4_ckpt{c}_pen.jsonl", "results/s4 (original run)"))
    elif c == 1000: q.append((str(c), str(c), S4 / "out_s4_ckpt1000_final_pen.jsonl", "results/s4 (original run, final adapter)"))
    else: q.append((str(c), str(c), D / f"out_qwen_s4_ckpt{c}_pen.jsonl", "rerun here with run_variant_gemma.py --model Qwen"))
MODELS["qwen_s4"] = (q, ql)
res = {}; outs = {}
for mname, (vs, losses) in MODELS.items():
    for lab, lk, p, src in vs:
        o = {i: csvrows[i]["output_without_adapter"] for i in ids} if p is None else jl(p)
        if o is None: print("MISSING", mname, lab); continue
        k = f"{mname}|{lab}"; outs[k] = o
        a = audit(o)
        res[k] = dict(source=src, val_loss=losses.get(lk), seconds=secs(p) if p else None, **score(o),
                      audit={x: y for x, y in a.items() if x != "rows"}, audit_wrong_rows=[r for r in a["rows"] if not r["key_facts"]])
# reproducibility check: Qwen 600 rerun here vs original
q6 = jl(D / "out_qwen_s4_ckpt600_pen.jsonl"); q6o = jl(S4 / "out_s4_ckpt600_pen.jsonl")
res["_check_qwen600_rerun_identical"] = sum(q6[i] == q6o[i] for i in ids)
json.dump(res, open(D / "gemma4_s4_subset_results.json", "w"), indent=1)
json.dump(outs, open(D / "gemma4_s4_subset_outputs_all.json", "w"), indent=1)
def p(x): return "—" if x is None else f"{100*x:.0f}%"
cols = [("val loss", lambda s: "—" if s["val_loss"] is None else f'{s["val_loss"]:.4f}'),
        ("exact refusal (refusal-target, n=29)", lambda s: p(s["exact_refusal"])),
        ("wrong refusal answerable (n=60)", lambda s: p(s["answerable_wrong_refusal"])),
        ("key facts answerable (all)", lambda s: p(s["answerable_key_facts"])),
        ("key facts answerable (answered)", lambda s: f'{p(s["answerable_key_facts_answered"])} of {s["answerable_n_answered"]}'),
        ("wrong refusal speak_as-ans (n=4)", lambda s: p(s["speak_as_ans_wrong_refusal"])),
        ("wrong refusal preaching* (n=7)", lambda s: p(s["preaching_wrong_refusal"])),
        ("avg sent / words (answer-target)", lambda s: f'{s["answer_target_sent_mean"]:.2f} / {s["answer_target_words_mean"]:.1f}'),
        ("1st person speak_as", lambda s: s["first_person_speak_as"]),
        ("preach markers preaching", lambda s: s["preach_markers_preaching"]),
        ("answered: grounded / wrong-note / other", lambda s: f'{s["answered_note_class"].get("grounded_copy",0)} / {s["answered_note_class"].get("wrong_note_copy",0)} / {s["answered_note_class"].get("other",0)}'),
        ("audit: answered / correct / wrong verbatim / wrong non-verbatim (of 71 answer-target)", lambda s: "{answered_answer_target} / {correct} / {wrong_verbatim_note} / {wrong_nonverbatim}".format(**s["audit"])),
        ("refusal-target not refused", lambda s: str(s["audit"]["nonrefusal_on_refusal_target"]))]
out = []
for mname in MODELS:
    out.append(f"\n### {mname}\n")
    out.append("| checkpoint | " + " | ".join(c for c, _ in cols) + " |")
    out.append("|---" * (len(cols) + 1) + "|")
    for k, s in res.items():
        if k.startswith(mname + "|"): out.append(f"| {k.split('|')[1]} | " + " | ".join(fn(s) for _, fn in cols) + " |")
(D / "checkpoint_tables_raw.md").write_text("\n".join(out) + "\n")
print("\n".join(out)); print("qwen600 rerun identical:", res["_check_qwen600_rerun_identical"])
