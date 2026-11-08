#!/usr/bin/env python3
"""Write the .summary.json that run_full_gemma.py failed to write (bug: relative --base-cache path in
base_from_cache=...relative_to(LORA); fixed in run_full_gemma.py at 19:12 for later runs). The CSV itself was
written before the crash. Same fields as run_full_gemma.py except seconds_adapter_pass (not recorded -> null).
Usage: write_missing_summary.py <csv> <adapter>"""
import sys
from pathlib import Path
D = Path(__file__).resolve().parent; LORA = D.parents[1]
sys.path.insert(0, str(LORA)); sys.path.insert(0, str(LORA / "scripts"))
import run_comparison as RC
csvp = Path(sys.argv[1]).resolve(); comp = LORA / "data/comparison/comparison.jsonl"
rows = RC.load_rows(comp, None)
s = RC.group_summary(rows, source=comp.relative_to(LORA), layout="app")
s.update(model="google/gemma-4-E2B-it", adapter=sys.argv[2], base_from_cache="results/gemma4_s4/gemma4_base_full_outputs.jsonl", seconds_adapter_pass=None,
         note="written afterwards by write_missing_summary.py; run_full_gemma.py crashed after writing the CSV")
print(RC.write_summary(csvp, s))
