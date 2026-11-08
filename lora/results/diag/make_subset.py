import json, random
from pathlib import Path
LORA = Path(__file__).resolve().parents[2]
R = "I do not have information about that in the approved notes."
rows = [json.loads(l) for l in (LORA / "data/comparison/comparison.jsonl").read_text().splitlines() if l.strip()]
ev = [r["metadata"] for r in rows if not r["metadata"].get("question_in_original")]
rng = random.Random(42)
def pick(pool, n): pool = sorted(pool, key=lambda m: m["id"]); return [m["id"] for m in rng.sample(pool, n)]
ids = []
ids += pick([m for m in ev if m["category"] == "answerable"], 60)
ids += pick([m for m in ev if m["category"] == "refusal"], 25)
ids += pick([m for m in ev if m["category"] == "speak_as" and m["reference_answer"] == R], 4)
ids += pick([m for m in ev if m["category"] == "speak_as" and m["reference_answer"] != R], 4)
ids += pick([m for m in ev if m["category"] == "preaching"], 7)
json.dump({"seed": 42, "strata": {"answerable": 60, "refusal": 25, "speak_as_refusal_target": 4, "speak_as_answer_target": 4, "preaching": 7}, "ids": ids}, open(Path(__file__).parent / "subset_ids.json", "w"), indent=1)
print(len(ids), len(set(ids)))
