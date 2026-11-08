# STATUS — llama.cpp b10481 laptop comparison

## Driver
`run_all.sh` runs every step in sequence (one `llama-server` at a time, port 8791) through `run_llamacpp.py`:
5 smoke runs (5 rows, condition B; base vs LoRA per model), then the 11 full 910-row variants in this order:
qwen_base_C, qwen_lora_B, gemma_base_C, gemma_final_B, qwen_lora_C, qwen_lora_A, gemma_final_C, gemma_final_A, gemma_800_B, gemma_800_C, gemma_800_A;
then `score.py` and `make_summary.py`.

- A step is **complete** when `raw/out_<name>.meta.json` exists and `raw/out_<name>.jsonl` has 910 lines; complete steps are skipped.
- A partial step resumes: `run_llamacpp.py` keeps the finished rows of `out_<name>.jsonl` and continues (rows are independent, `cache_prompt=false`).
- A lock dir `.driver.lock` (with pid) prevents two drivers; a stale lock (dead pid) is taken over. A leftover server on port 8791 is killed before each step.
- Log: `driver.log` (START/DONE/FAIL lines, progress every 50 rows). Server logs: `raw/server_<name>.log`.

## Resume / re-run
```
cd lora/results/llamacpp_compare
ps aux | egrep -i 'python|mlx|llama' | grep -v egrep   # nothing else should be running
nohup bash run_all.sh >> driver.log 2>&1 &
tail -f driver.log
```
Re-scoring only (no model runs): `python3.11 score.py && python3.11 make_summary.py`.

## Files
- `common.py` prompt building (A/B/C, ChatML / formatGemma4, sanitize); `run_llamacpp.py` one variant; `score.py` headline metrics on full / full minus selection / eval-only / eval minus selection + MLX agreement; `make_summary.py` → `summary.md`; `notes.md` settings & caveats (included in summary.md).
- `variants.json` variant → output path (llama.cpp jsonl + the earlier MLX CSVs); `agreement_pairs.json` llama.cpp LoRA+B vs MLX LoRA+B.
- `*_v0_unused.*.txt` = the interrupted worker's earlier run_all/score/make_summary, kept for reference, not used.

## Progress
(updated below by the worker)
