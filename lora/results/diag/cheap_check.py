import json, sys, difflib, collections
from pathlib import Path
LORA = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LORA))
sys.path.insert(0, str(LORA / "scripts"))
from run_comparison import prompt_messages, load_rows
from wisent_lora.comparison import load_note_pool
from wisent_lora.protocol import load_persona, repo_root
R = "I do not have information about that in the approved notes."
root = repo_root()
persona = load_persona(root); note_text, _, _ = load_note_pool(root)
train = []
for f in ["train.jsonl", "valid.jsonl"]:
    for l in (LORA / "data/v2" / f).read_text().splitlines():
        if l.strip():
            r = json.loads(l); r["_file"] = f; train.append(r)
by_id = {r["metadata"]["id"]: r for r in train}
def q(r): return [m["content"] for m in r["messages"] if m["role"] == "user"][-1]
by_q = collections.defaultdict(list)
for r in train: by_q[q(r)].append(r)
comp = load_rows(LORA / "data/comparison/comparison.jsonl", None)
out = {"n_copied": 0, "matched_by_id": 0, "matched_by_question": 0, "unmatched": [], "identical": 0, "differ": [], "assistant_differs": []}
first_diff = None
for row in comp:
    m = row["metadata"]
    if not m.get("question_in_original"): continue
    out["n_copied"] += 1
    t = by_id.get(m["id"])
    if t: out["matched_by_id"] += 1
    else:
        cands = by_q.get(q(row), [])
        if cands: t = cands[0]; out["matched_by_question"] += 1
        else: out["unmatched"].append(m["id"]); continue
    built = prompt_messages(row, layout="app", persona=persona, note_text=note_text)
    stored = [x for x in t["messages"] if x["role"] != "assistant"]
    if built == stored: out["identical"] += 1
    else:
        out["differ"].append(m["id"])
        if first_diff is None:
            first_diff = "\n".join(difflib.unified_diff(json.dumps(stored, indent=1).splitlines(), json.dumps(built, indent=1).splitlines(), "train", "comparison", lineterm=""))
    if t["messages"][-1]["content"] != m.get("reference_answer"):
        out["assistant_differs"].append((m["id"], t["messages"][-1]["content"], m.get("reference_answer")))
# training target balance
cnt = collections.Counter(); ref = collections.Counter()
for r in train:
    c = r["metadata"]["category"]; cnt[c] += 1; ref[c] += r["messages"][-1]["content"] == R
out["train_valid_by_category_refusal_target"] = {c: f"{ref[c]}/{cnt[c]}" for c in cnt}
out["train_valid_refusal_share"] = sum(ref.values()) / len(train)
# eval-only prompts: are their notes texts/order derived the same way? check whether eval rows share prompt sets with train
print(json.dumps({k: (v if not isinstance(v, list) else (len(v), v[:5])) for k, v in out.items()}, indent=1))
print(first_diff or "no diffs")
json.dump(out, open(LORA / "results/diag/cheap_check.json", "w"), indent=1)
