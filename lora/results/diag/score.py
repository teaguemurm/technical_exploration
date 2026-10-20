#!/usr/bin/env python3
import csv, json, re, sys
from collections import Counter
from pathlib import Path
D = Path(__file__).parent; LORA = D.parents[1]
sys.path.insert(0, str(LORA))
from wisent_lora.comparison import load_note_pool
from wisent_lora.protocol import repo_root
R = "I do not have information about that in the approved notes."
note_text, _, _ = load_note_pool(repo_root())
meta = {}; qs = {}
for l in (LORA / "data/comparison/comparison.jsonl").read_text().splitlines():
    if l.strip():
        r = json.loads(l); meta[r["metadata"]["id"]] = r["metadata"]; qs[r["metadata"]["id"]] = [m["content"] for m in r["messages"] if m["role"] == "user"][-1]
ids = json.load(open(D / "subset_ids.json"))["ids"]
csvrows = {r["id"]: r for r in csv.DictReader(open(LORA / "results/v2_mlx_app_comparison.csv"))}
VARIANTS = [
    ("base_pen (orig CSV)", None, "base, penalty 1.1"),
    ("f_base_nopen", "out_f_base_nopen.jsonl", "base, penalty off"),
    ("d_final_s20_pen (orig CSV)", None, "LoRA final, scale 20, penalty 1.1 (original)"),
    ("d_rerun_final_s20_pen", "out_d_rerun_final_s20_pen.jsonl", "same as d, rerun with diag runner (sanity)"),
    ("a_final_s20_nopen", "out_a_final_s20_nopen.jsonl", "LoRA final, scale 20, penalty off"),
    ("b_ckpt500_s20_pen", "out_b_ckpt500_s20_pen.jsonl", "LoRA iter 500, scale 20, penalty 1.1"),
    ("c_final_s10_pen", "out_c_final_s10_pen.jsonl", "LoRA final, scale 10, penalty 1.1"),
    ("e_ckpt800_s20_nopen", "out_e_ckpt800_s20_nopen.jsonl", "LoRA iter 800, scale 20, penalty off"),
]
def outputs(name, f):
    if name.startswith("base_pen"): return {i: csvrows[i]["output_without_adapter"] for i in ids}
    if name.startswith("d_final"): return {i: csvrows[i]["output_with_adapter"] for i in ids}
    p = D / f
    if not p.exists(): return None
    o = {json.loads(l)["id"]: json.loads(l)["output"] for l in p.read_text().splitlines() if l.strip()}
    return o if len(o) == len(ids) else None
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
    """for answered rows: grounded-copy / wrong-note-copy / other (paraphrase or splice)"""
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
res = {}; outs = {}
for name, f, desc in VARIANTS:
    o = outputs(name, f)
    if o is None: continue
    outs[name] = o; res[name] = dict(desc=desc, **score(o))
    mp = D / (f.replace(".jsonl", ".meta.json")) if f else None
    if mp and mp.exists(): res[name]["seconds"] = json.load(open(mp))["seconds"]
if "d_rerun_final_s20_pen" in outs:
    d0 = outputs("d_final", None); d1 = outs["d_rerun_final_s20_pen"]
    res["d_rerun_final_s20_pen"]["identical_to_orig_csv"] = sum(d0[i] == d1[i] for i in ids)
json.dump(res, open(D / "diag_results.json", "w"), indent=1)
json.dump(outs, open(D / "diag_outputs_all.json", "w"), indent=1)
def p(x): return "—" if x is None else f"{100*x:.0f}%"
cols = [("exact refusal (refusal-target)", lambda s: p(s["exact_refusal"])),
        ("wrong refusal answerable", lambda s: p(s["answerable_wrong_refusal"])),
        ("key facts answerable (all)", lambda s: p(s["answerable_key_facts"])),
        ("key facts answerable (answered)", lambda s: f'{p(s["answerable_key_facts_answered"])} of {s["answerable_n_answered"]}'),
        ("wrong refusal speak_as-ans", lambda s: p(s["speak_as_ans_wrong_refusal"])),
        ("wrong refusal preaching*", lambda s: p(s["preaching_wrong_refusal"])),
        ("avg sent / words (answer-target)", lambda s: f'{s["answer_target_sent_mean"]:.2f} / {s["answer_target_words_mean"]:.1f}'),
        ("1st person speak_as", lambda s: s["first_person_speak_as"]),
        ("preach markers preaching", lambda s: s["preach_markers_preaching"]),
        ("answered: grounded / wrong-note / other", lambda s: f'{s["answered_note_class"].get("grounded_copy",0)} / {s["answered_note_class"].get("wrong_note_copy",0)} / {s["answered_note_class"].get("other",0)}'),
        ("run time", lambda s: f'{s["seconds"]:.0f}s' if "seconds" in s else "—")]
L = ["| variant | " + " | ".join(c for c, _ in cols) + " |", "|---" * (len(cols) + 1) + "|"]
for n, s in res.items(): L.append(f"| {n} | " + " | ".join(fn(s) for _, fn in cols) + " |")
table = "\n".join(L)
print(table)
(D / "diag_table.md").write_text(table + "\n")
