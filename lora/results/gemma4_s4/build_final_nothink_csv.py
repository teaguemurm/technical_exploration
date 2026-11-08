#!/usr/bin/env python3
"""gemma4_s4_mlx_app_comparison_final_nothink.csv = gemma4_s4_mlx_app_comparison_base_nothink.csv with
output_with_adapter replaced by gemma4_s4_final_full_outputs_final_nothink.jsonl (ids checked). Both columns thinking off."""
import csv, json
from pathlib import Path
D = Path(__file__).resolve().parent
new = [json.loads(l) for l in (D / "gemma4_s4_final_full_outputs_final_nothink.jsonl").read_text().splitlines() if l.strip()]
rows = list(csv.DictReader(open(D / "gemma4_s4_mlx_app_comparison_base_nothink.csv", encoding="utf-8")))
assert len(rows) == len(new) == 910
for r, n in zip(rows, new):
    assert r["id"] == n["id"]; r["output_with_adapter"] = n["output"]
out = D / "gemma4_s4_mlx_app_comparison_final_nothink.csv"; assert not out.exists()
w = csv.DictWriter(open(out, "w", encoding="utf-8", newline=""), fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print("wrote", out.name)
