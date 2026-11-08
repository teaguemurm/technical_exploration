#!/usr/bin/env python3
"""Copies the existing Gemma full-set CSVs (final, ckpt800, ckpt600) to *_base_nothink.csv with ONLY
output_without_adapter replaced by gemma4_base_full_outputs_base_nothink.jsonl (ids checked row by row).
Adapter outputs are unchanged (adapter runs are unaffected by the base pass)."""
import csv, json
from pathlib import Path
D = Path(__file__).resolve().parent
new = [json.loads(l) for l in (D / "gemma4_base_full_outputs_base_nothink.jsonl").read_text().splitlines() if l.strip()]
old = [json.loads(l) for l in (D / "gemma4_base_full_outputs.jsonl").read_text().splitlines() if l.strip()]
for stem in ["gemma4_s4_mlx_app_comparison", "gemma4_s4_ckpt800_mlx_app_comparison", "gemma4_s4_ckpt600_mlx_app_comparison"]:
    rows = list(csv.DictReader(open(D / f"{stem}.csv", encoding="utf-8")))
    assert len(rows) == len(new) == 910
    for r, n, o in zip(rows, new, old):
        assert r["id"] == n["id"] == o["id"] and r["output_without_adapter"] == o["output"]
        r["output_without_adapter"] = n["output"]
    out = D / f"{stem}_base_nothink.csv"; assert not out.exists()
    w = csv.DictWriter(open(out, "w", encoding="utf-8", newline=""), fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("wrote", out.name)
