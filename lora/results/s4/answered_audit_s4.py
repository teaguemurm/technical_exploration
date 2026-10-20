import json, re, sys
from pathlib import Path
D = Path(__file__).parent; LORA = D.parents[1]; sys.path.insert(0, str(LORA))
from wisent_lora.comparison import load_note_pool
from wisent_lora.protocol import repo_root
import score_s4 as S  # reuses kf etc. (re-runs scoring, harmless)
R = S.R
note_text, _, _ = load_note_pool(repo_root())
outs = json.load(open(D / "s4_subset_outputs_all.json")); ids = json.load(open(D / "subset_ids.json"))["ids"]
def n(s): return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()
audit = {}
for v, o in outs.items():
    rows = []
    for i in ids:
        m = S.meta[i]; out = o[i]
        if R.lower().rstrip(".") in out.lower() and n(out) == n(R): continue
        notes = [note_text[x] for x in m.get("prompt_note_ids") or []]
        verbatim = any(n(out) == n(t) or n(out) in n(t) for t in notes)
        rt = m["reference_answer"] == R
        ok = (not rt) and S.kf(out, m)
        rows.append(dict(id=i, cat=m["category"], refusal_target=rt, verbatim_note=verbatim, key_facts=ok, q=S.qs[i], out=out))
    at = [r for r in rows if not r["refusal_target"]]
    audit[v] = dict(answered_answer_target=len(at),
                    correct=sum(r["key_facts"] for r in at),
                    wrong_verbatim_note=sum((not r["key_facts"]) and r["verbatim_note"] for r in at),
                    wrong_nonverbatim=sum((not r["key_facts"]) and not r["verbatim_note"] for r in at),
                    nonrefusal_on_refusal_target=sum(r["refusal_target"] for r in rows),
                    malformed_refusal=sum(R.lower().rstrip(".") in r["out"].lower() for r in rows),
                    rows=rows)
json.dump(audit, open(D / "s4_answered_audit.json", "w"), indent=1)
for v, a in audit.items():
    print(v, {k: x for k, x in a.items() if k != "rows"})
if len(sys.argv) > 1:
    for r in audit[sys.argv[1]]["rows"]:
        if not r["key_facts"]: print("  ", r["id"], r["refusal_target"], r["verbatim_note"], "|", r["q"], "|", r["out"])
