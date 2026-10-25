"""Extract the earlier laptop (Mac14,6) app protocol runs from
exports/technical_exploration_runs_20261020_193111.json (cumulative app export, read-only):
  Gemma: the 20 Oct 2026 19:2x gemma4-format batch (promptFormatter == 'gemma4'),
         the one that replaced the superseded ChatML runs in the sheet.
  Qwen:  the 11 Oct 2026 15:4x batch (native ChatML), the last run per prompt x condition.
and check our rebuilt messages equal the app's messagesSent for every row.
Writes app_reference.json."""
import json, sys
from pathlib import Path
D = Path(__file__).resolve().parent; sys.path.insert(0, str(D))
import pcommon as P
EXP = P.ROOT / "exports/technical_exploration_runs_20261020_193111.json"
runs = [r for r in json.load(open(EXP)) if r["deviceModel"] == "Mac14,6" and not r["isHeatTest"]]
ref = {}
for r in sorted(runs, key=lambda r: r["timestamp"]):
    fam = "qwen" if r["modelFileName"].startswith("qwen") else "gemma"
    if fam == "gemma" and not (r.get("promptFormatter") == "gemma4" and r["timestamp"].startswith("2026-10-20T19")): continue
    if fam == "qwen" and not r["timestamp"].startswith("2026-10-11T15:4"): continue
    ref[f"{fam}|{r['condition']}|{r['promptId']}"] = {k: r[k] for k in ["timestamp", "modelFileName", "condition", "promptId", "messagesSent", "output", "promptTokenCount", "generatedTokenCount", "hitMaxTokens", "decodingSettings", "injectedNoteIds"]} | {"promptFormatter": r.get("promptFormatter"), "thoughtEmitted": r.get("thoughtEmitted")}
bad = 0; byp = {p["id"]: p for p in P.PROMPTS}
for k, v in ref.items():
    fam, cond, pid = k.split("|")
    mine = P.format_messages(P.messages(cond, byp[pid]))
    if mine != v["messagesSent"]:
        bad += 1; print("LAYOUT MISMATCH", k)
    if P.facts_for(cond, byp[pid]) != v["injectedNoteIds"]:
        bad += 1; print("NOTE IDS MISMATCH", k, v["injectedNoteIds"])
from collections import Counter
print(Counter(k.rsplit("|", 1)[0] for k in ref))
print("decoding settings seen:", {json.dumps(v["decodingSettings"], sort_keys=True) for v in ref.values()})
print(f"{len(ref)} reference rows, {bad} mismatches")
json.dump(ref, open(D / "app_reference.json", "w"), indent=1, ensure_ascii=False)
