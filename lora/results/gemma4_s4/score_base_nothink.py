#!/usr/bin/env python3
"""Scores out_gemma4_base_nothink_pen.jsonl (and, for comparison, the old out_gemma4_base_pen.jsonl) with the
scoring code of score_subset_gemma4_s4.py, executed unchanged (source up to '# ---- end verbatim ----' is exec'd,
so its result files are NOT rewritten). Writes gemma4_s4_base_nothink_results.json and gemma4_s4_base_nothink_row.md."""
import json, os
from pathlib import Path
D = Path(__file__).resolve().parent
src = (D / "score_subset_gemma4_s4.py").read_text()
g = {"__file__": str(D / "score_subset_gemma4_s4.py"), "__name__": "scoring"}
exec(compile(src.split("# ---- end verbatim ----")[0], "score_subset_gemma4_s4.py", "exec"), g)
ids, score, audit, R = g["ids"], g["score"], g["audit"], g["R"]
os.environ["HF_HUB_OFFLINE"] = "1"
from mlx_lm.utils import load_tokenizer
from huggingface_hub import snapshot_download
tok = load_tokenizer(Path(snapshot_download("google/gemma-4-E2B-it")))
def jl(p): return {json.loads(l)["id"]: json.loads(l)["output"] for l in p.read_text().splitlines() if l.strip()}
gl = json.load(open(D / "gemma4_s4_val_loss.json"))["losses"]
res = {}
for lab, f in [("base_old_thinking", "out_gemma4_base_pen.jsonl"), ("base_nothink", "out_gemma4_base_nothink_pen.jsonl")]:
    o = jl(D / f); assert len(o) == len(ids) == 100
    a = audit(o)
    ntok = {i: len(tok.encode(o[i], add_special_tokens=False)) for i in ids}
    res[lab] = dict(file=f, **score(o), audit={x: y for x, y in a.items() if x != "rows"},
                    n_thought_text=sum(("<|channel>" in o[i]) or ("thought" in o[i].lower()[:40]) or ("<channel|>" in o[i]) for i in ids),
                    n_ge_250_tokens=sum(v >= 250 for v in ntok.values()), max_tokens_seen=max(ntok.values()),
                    preaching_rows=[dict(id=i, q=g["qs"][i], ref=g["meta"][i]["reference_answer"], out=o[i]) for i in ids if g["meta"][i]["category"] == "preaching"])
json.dump(res, open(D / "gemma4_s4_base_nothink_results.json", "w"), indent=1)
def p(x): return "—" if x is None else f"{100*x:.0f}%"
lines = []
for lab, s in res.items():
    nc = s["answered_note_class"]
    lines.append(f"| {lab} | {p(s['exact_refusal'])} | {p(s['answerable_wrong_refusal'])} | {p(s['answerable_key_facts'])} | {p(s['answerable_key_facts_answered'])} of {s['answerable_n_answered']} | {p(s['speak_as_ans_wrong_refusal'])} | {p(s['preaching_wrong_refusal'])} | {s['answer_target_sent_mean']:.2f} / {s['answer_target_words_mean']:.1f} | {s['first_person_speak_as']} | {s['preach_markers_preaching']} | {nc.get('grounded_copy',0)} / {nc.get('wrong_note_copy',0)} / {nc.get('other',0)} | " + "{answered_answer_target} / {correct} / {wrong_verbatim_note} / {wrong_nonverbatim}".format(**s["audit"]) + f" | {s['audit']['nonrefusal_on_refusal_target']} | thought={s['n_thought_text']} ≥250tok={s['n_ge_250_tokens']} max={s['max_tokens_seen']} |")
(D / "gemma4_s4_base_nothink_row.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
for r in res["base_nothink"]["preaching_rows"]: print("\n##", r["id"], "| Q:", r["q"], "\n", r["out"])
