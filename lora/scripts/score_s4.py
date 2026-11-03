#!/usr/bin/env python3
"""Score the scale-4 LoRA subset against the v2 comparison CSV.

Scored a fixed id subset of the 910-question comparison set: base and the
first adapter (both columns of the v2 MLX app-layout CSV), one diag
checkpoint, and the scale-4 checkpoints (lr 1e-4, iterations 400, 600, 800,
and 1000). Counts first-person and preaching markers after the refusal
sentence is removed, plus the 150-word check, key facts, wrong refusals, and
note-copy classes (grounded copy, wrong-note copy, other). Heuristics,
followed by review.

Inputs (only the comparison JSONL and the note pool are in the repo):
  lora/data/comparison/comparison.jsonl
  lora/results/v2_mlx_app_comparison.csv
  lora/results/s4/subset_ids.json          {"ids": [...]}
  lora/results/s4/out_s4_ckpt400_pen.jsonl
  lora/results/s4/out_s4_ckpt600_pen.jsonl
  lora/results/s4/out_s4_ckpt800_pen.jsonl
  lora/results/s4/out_s4_ckpt1000_final_pen.jsonl
  lora/results/diag/out_e_ckpt800_s20_nopen.jsonl
  note text from wisent_lora.comparison.load_note_pool

Outputs, written next to the subset inputs (not committed):
  lora/results/s4/s4_subset_results.json
  lora/results/s4/s4_subset_outputs_all.json
  lora/results/s4/s4_subset_table.md

The scorer originally lived two directories under ``lora/``, so ``LORA`` was
``Path(__file__).parent.parents[1]`` and the subset files sat beside the
script (``diag/`` a sibling of that directory). It now lives in
``lora/scripts/``. ``D`` still points at ``lora/results/s4/`` so the scoring
run itself is unchanged.
"""
import csv, json, re, sys
from collections import Counter
from pathlib import Path
_SCRIPTS = Path(__file__).resolve().parent
LORA = _SCRIPTS.parent
# subset_ids.json, out_s4_*.jsonl, and the s4_subset_* outputs.
D = LORA / "results" / "s4"
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
DIAG = D.parent / "diag"
VARIANTS = [
    ("base_pen (v2 CSV)", None, "base, penalty 1.1"),
    ("v2_s20_final_pen (v2 CSV)", None, "first adapter, scale 20, iter 1000, penalty 1.1"),
    ("v2_s20_ckpt800_nopen (diag e)", str(DIAG / "out_e_ckpt800_s20_nopen.jsonl"), "first adapter iter 800, penalty off"),
    ("s4_ckpt400_pen", "out_s4_ckpt400_pen.jsonl", "s4 lr1e-4 iter 400, penalty 1.1"),
    ("s4_ckpt600_pen", "out_s4_ckpt600_pen.jsonl", "s4 lr1e-4 iter 600, penalty 1.1"),
    ("s4_ckpt800_pen", "out_s4_ckpt800_pen.jsonl", "s4 lr1e-4 iter 800, penalty 1.1"),
    ("s4_ckpt1000_final_pen", "out_s4_ckpt1000_final_pen.jsonl", "s4 lr1e-4 iter 1000 (final), penalty 1.1"),
]
def outputs(name, f):
    if name.startswith("base_pen"): return {i: csvrows[i]["output_without_adapter"] for i in ids}
    if name.startswith("v2_s20_final"): return {i: csvrows[i]["output_with_adapter"] for i in ids}
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
    mp = (D / f).with_name(Path(f).name.replace(".jsonl", ".meta.json")) if f else None
    if mp and mp.exists(): res[name]["seconds"] = json.load(open(mp))["seconds"]
if False:
    d0 = outputs("d_final", None); d1 = outs["d_rerun_final_s20_pen"]
    res["d_rerun_final_s20_pen"]["identical_to_orig_csv"] = sum(d0[i] == d1[i] for i in ids)
json.dump(res, open(D / "s4_subset_results.json", "w"), indent=1)
json.dump(outs, open(D / "s4_subset_outputs_all.json", "w"), indent=1)
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
(D / "s4_subset_table.md").write_text(table + "\n")
