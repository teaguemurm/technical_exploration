# s4 adapter (scale 4, lr 1e-4) — checkpoint selection and full comparison (20 Oct 2026)

Adapter: `adapters_s4_lr1e-4/` (config `configs/qwen25_1.5b_constraint_v2_s4_lr1e-4.yaml`: rank 16, scale 4, lr 1e-4, batch 2, 1000 iters, mask_prompt, data/v2). Model Qwen/Qwen2.5-1.5B-Instruct (mlx-lm 0.32.2, python3.11). Sampler for everything below: app sampler (temp 0, seed 42, top-k 40, top-p 0.9, repeat penalty 1.1 over 64 tokens, max 256, context 2048), `--layout app`. No tracked repo files, `adapters/` or the training data were changed.

## Training log / val loss
**No training log or val losses were saved.** The adapter folder only has checkpoints and `adapter_config.json`, no `*.log` or other output was written to lora/, the repo, /tmp, ~/Desktop or ~/Downloads around the training run (checkpoints written 17:18–17:29 CEST on 20 Oct 2026), and shell history has no matching command. The mlx-lm "Val loss" lines (steps_per_eval 100) only went to the terminal used for training. No val-loss curve is available.

## 1. Checkpoint subset (100 eval-only rows, `subset_ids.json`, penalty on = app sampler)
Runner `results/s4/run_variant.py` (copy of diag runner), scorer `score_s4.py`, audit `answered_audit_s4.py`. Checkpoint dirs `ad_ckpt400/600/800` = copied `00000N00_adapters.safetensors` + the final's adapter_config.json (adapter_path edited). Iter 1000 = `adapters_s4_lr1e-4` itself (`adapters.safetensors` is byte-identical to `0001000_adapters.safetensors`). The first three rows are earlier reference results.

| variant | exact refusal (refusal-target) | wrong refusal answerable | key facts answerable (all) | key facts answerable (answered) | wrong refusal speak_as-ans | wrong refusal preaching* | avg sent / words (answer-target) | 1st person speak_as | preach markers preaching | answered: grounded / wrong-note / other | run time |
|---|---|---|---|---|---|---|---|---|---|---|---|
| base_pen (v2 CSV) | 0% | 0% | 92% | 92% of 60 | 0% | 0% | 1.73 / 30.3 | 3/8 | 5/7 | 4 / 1 / 66 | — |
| v2_s20_final_pen (v2 CSV) | 100% | 72% | 27% | 94% of 17 | 75% | 100% | 1.00 / 10.5 | 0/8 | 0/7 | 2 / 0 / 16 | — |
| v2_s20_ckpt800_nopen (diag e) | 90% | 38% | 43% | 70% of 37 | 25% | 100% | 1.00 / 10.4 | 0/8 | 0/7 | 9 / 6 / 25 | 41s |
| s4_ckpt400_pen | 100% | 3% | 95% | 98% of 58 | 0% | 86% | 1.00 / 9.1 | 0/8 | 0/7 | 13 / 1 / 49 | 40s |
| s4_ckpt600_pen | 100% | 2% | 95% | 97% of 59 | 0% | 86% | 1.00 / 9.1 | 0/8 | 0/7 | 15 / 1 / 48 | 40s |
| s4_ckpt800_pen | 100% | 8% | 88% | 96% of 55 | 0% | 100% | 1.01 / 9.7 | 0/8 | 0/7 | 15 / 3 / 41 | 41s |
| s4_ckpt1000_final_pen | 100% | 22% | 73% | 94% of 47 | 0% | 100% | 1.00 / 9.3 | 0/8 | 0/7 | 10 / 0 / 41 | 40s |

Answered-row audit (answer-target rows that were not refused, n=71 answer-target):

| variant | answered | correct | wrong, verbatim copy of another note | wrong, non-verbatim | refusal-target rows not refused |
|---|---|---|---|---|---|
| base_pen (v2 CSV) | 71 | 61 | 1 | 9 | 29 |
| v2_s20_final_pen (v2 CSV) | 18 | 17 | 0 | 1 | 0 |
| v2_s20_ckpt800_nopen (diag e) | 40 | 28 | 8 | 4 | 3 |
| s4_ckpt400_pen | 63 | 62 | 1 | 0 | 0 |
| s4_ckpt600_pen | 64 | 62 | 1 | 1 | 0 |
| s4_ckpt800_pen | 59 | 57 | 2 | 0 | 0 |
| s4_ckpt1000_final_pen | 51 | 48 | 1 | 2 | 0 |

Iter 400 and 600 are effectively tied on the subset (62/71 correct each; 3% vs 2% wrong refusal on answerable). Over-refusal returns from iter 800 (8%) and is clearly worse at iter 1000 (22%). Since the subset clearly favoured an earlier checkpoint, the full comparison was run for **iter 600** too, as well as for the final adapter.

## 2–3. Full comparison (910 rows): base vs first adapter (scale 20) vs s4 final vs s4 iter 600
Runs: `results/s4_mlx_app_comparison.{csv,log,summary.json}` (final) and `results/s4_ckpt600_mlx_app_comparison.{csv,log,summary.json}` (iter 600). Command (from lora/; the earlier .log did not record the command, so the flags were reconstructed from run_comparison.py and the v2 summary — app sampler values come from the app config via `load_protocol_decoding`, which gives exactly the values above):
`python3.11 scripts/run_comparison.py --backend mlx --model Qwen/Qwen2.5-1.5B-Instruct --adapter adapters_s4_lr1e-4 --layout app --comparison data/comparison/comparison.jsonl --output results/s4_mlx_app_comparison.csv` (and `--adapter results/s4/ad_ckpt600 --output results/s4_ckpt600_mlx_app_comparison.csv`).
Sanity: the base column regenerated in both s4 runs is identical to the v2 CSV base column on 910/910 rows, and `analyze_s4.py` on the v2 CSV reproduces `v2_mlx_app_analysis.json` exactly. Per-run analyses in the v2 format: `s4_mlx_app_analysis.md/.json` and `s4_ckpt600_mlx_app_analysis.md/.json`.

Note on preaching: v2 trains preaching prompts → refusal string, but the comparison references are old-style facts, so "wrong refusal" on preaching is the trained behaviour (see diag_results.md section 0).

Full comparison set (`data/comparison/comparison.jsonl`, 910 rows, `--layout app`, app sampler: temp 0, seed 42, top-k 40, top-p 0.9, repeat penalty 1.1 over 64, max 256). Base and v2 s20 columns come from `results/v2_mlx_app_comparison.csv`; s4 columns from `results/s4_mlx_app_comparison.csv` (final) and `results/s4_ckpt600_mlx_app_comparison.csv` (iter 600). Metric definitions are those of `results/v2_mlx_app_analysis.md` (same code). Audit = full-set version of `results/diag/answered_audit.py`.

Base outputs re-generated in the s4 runs, identical to the v2 CSV base column: s4 final (iter 1000): 910/910, s4 iter 600: 910/910


## eval-only (question_in_original=false)

| category | metric | base | v2 s20 final | s4 final (iter 1000) | s4 iter 600 |
|---|---|---|---|---|---|
| answerable | wrong refusal (answer-target) (n=230) | 0.0% | 75.7% | 27.8% | 1.3% |
| answerable | key facts (answer-target) (n=230) | 94.3% | 20.0% | 69.6% | 96.1% |
| answerable | key facts among answered (n=230) | 94.3% | 82.1% | 96.4% | 97.4% |
| answerable | first-person markers (n=230) | 0.0% | 0.0% | 0.0% | 0.0% |
| answerable | preach markers (n=230) | 5.7% | 0.0% | 0.0% | 0.0% |
| answerable | nature/decoding markers (n=230) | 0.4% | 0.0% | 0.0% | 0.0% |
| answerable | degenerate empty/rep/≥150w/trunc | 0/0/0/0 | 0/0/0/0 | 0/0/0/0 | 0/0/0/0 |
| refusal | exact refusal (refusal-target) (n=145) | 0.0% | 99.3% | 100.0% | 100.0% |
| refusal | contains refusal (refusal-target) (n=145) | 0.0% | 99.3% | 100.0% | 100.0% |
| refusal | first-person markers (n=145) | 33.1% | 0.0% | 0.0% | 0.0% |
| refusal | preach markers (n=145) | 5.5% | 0.0% | 0.0% | 0.0% |
| refusal | nature/decoding markers (n=145) | 0.0% | 0.0% | 0.0% | 0.0% |
| refusal | degenerate empty/rep/≥150w/trunc | 0/0/0/0 | 0/0/0/0 | 0/0/0/0 | 0/0/0/0 |
| speak_as | exact refusal (refusal-target) (n=32) | 0.0% | 100.0% | 100.0% | 100.0% |
| speak_as | contains refusal (refusal-target) (n=32) | 0.0% | 100.0% | 100.0% | 100.0% |
| speak_as | wrong refusal (answer-target) (n=40) | 0.0% | 90.0% | 17.5% | 2.5% |
| speak_as | key facts (answer-target) (n=40) | 95.0% | 10.0% | 82.5% | 97.5% |
| speak_as | key facts among answered (n=40) | 95.0% | 100.0% | 100.0% | 100.0% |
| speak_as | first-person markers (n=72) | 38.9% | 0.0% | 0.0% | 0.0% |
| speak_as | preach markers (n=72) | 2.8% | 0.0% | 0.0% | 0.0% |
| speak_as | nature/decoding markers (n=72) | 0.0% | 0.0% | 0.0% | 0.0% |
| speak_as | degenerate empty/rep/≥150w/trunc | 0/0/1/0 | 0/0/0/0 | 0/0/0/0 | 0/0/0/0 |
| preaching | wrong refusal (answer-target) (n=48) | 0.0% | 95.8% | 95.8% | 87.5% |
| preaching | key facts (answer-target) (n=48) | 41.7% | 4.2% | 4.2% | 12.5% |
| preaching | key facts among answered (n=48) | 41.7% | 100.0% | 100.0% | 100.0% |
| preaching | first-person markers (n=48) | 8.3% | 0.0% | 0.0% | 0.0% |
| preaching | preach markers (n=48) | 58.3% | 0.0% | 0.0% | 0.0% |
| preaching | nature/decoding markers (n=48) | 2.1% | 0.0% | 0.0% | 0.0% |
| preaching | degenerate empty/rep/≥150w/trunc | 0/0/2/1 | 0/0/0/0 | 0/0/0/0 | 0/0/0/0 |
| ALL | exact refusal (refusal-target) (n=177) | 0.0% | 99.4% | 100.0% | 100.0% |
| ALL | contains refusal (refusal-target) (n=177) | 0.0% | 99.4% | 100.0% | 100.0% |
| ALL | wrong refusal (answer-target) (n=318) | 0.0% | 80.5% | 36.8% | 14.5% |
| ALL | key facts (answer-target) (n=318) | 86.5% | 16.4% | 61.3% | 83.6% |
| ALL | key facts among answered (n=318) | 86.5% | 83.9% | 97.0% | 97.8% |
| ALL | first-person markers (n=495) | 16.2% | 0.0% | 0.0% | 0.0% |
| ALL | preach markers (n=495) | 10.3% | 0.0% | 0.0% | 0.0% |
| ALL | nature/decoding markers (n=495) | 0.4% | 0.0% | 0.0% | 0.0% |
| ALL | degenerate empty/rep/≥150w/trunc | 0/0/3/1 | 0/0/0/0 | 0/0/0/0 | 0/0/0/0 |

### Answered-row audit, eval-only (question_in_original=false)

| category | variant | answer-target | answered | correct (key facts) | wrong: verbatim copy of a prompt note | wrong: non-verbatim (splice/invention) | refusal-target | refusal-target not refused | malformed refusal |
|---|---|---|---|---|---|---|---|---|---|
| answerable | base | 230 | 230 | 217 | 2 | 11 | 0 | 0 | 0 |
| answerable | v2 s20 final | 230 | 56 | 46 | 6 | 4 | 0 | 0 | 0 |
| answerable | s4 final (iter 1000) | 230 | 166 | 160 | 4 | 2 | 0 | 0 | 0 |
| answerable | s4 iter 600 | 230 | 227 | 221 | 2 | 4 | 0 | 0 | 0 |
| refusal | base | 0 | 0 | 0 | 0 | 0 | 145 | 145 | 0 |
| refusal | v2 s20 final | 0 | 0 | 0 | 0 | 0 | 145 | 1 | 0 |
| refusal | s4 final (iter 1000) | 0 | 0 | 0 | 0 | 0 | 145 | 0 | 0 |
| refusal | s4 iter 600 | 0 | 0 | 0 | 0 | 0 | 145 | 0 | 0 |
| speak_as | base | 40 | 40 | 38 | 0 | 2 | 32 | 32 | 0 |
| speak_as | v2 s20 final | 40 | 4 | 4 | 0 | 0 | 32 | 0 | 0 |
| speak_as | s4 final (iter 1000) | 40 | 33 | 33 | 0 | 0 | 32 | 0 | 0 |
| speak_as | s4 iter 600 | 40 | 39 | 39 | 0 | 0 | 32 | 0 | 0 |
| preaching | base | 48 | 48 | 20 | 0 | 28 | 0 | 0 | 0 |
| preaching | v2 s20 final | 48 | 2 | 2 | 0 | 0 | 0 | 0 | 0 |
| preaching | s4 final (iter 1000) | 48 | 2 | 2 | 0 | 0 | 0 | 0 | 0 |
| preaching | s4 iter 600 | 48 | 6 | 6 | 0 | 0 | 0 | 0 | 0 |
| ALL | base | 318 | 318 | 275 | 2 | 41 | 177 | 177 | 0 |
| ALL | v2 s20 final | 318 | 62 | 52 | 6 | 4 | 177 | 1 | 0 |
| ALL | s4 final (iter 1000) | 318 | 201 | 195 | 4 | 2 | 177 | 0 | 0 |
| ALL | s4 iter 600 | 318 | 272 | 266 | 2 | 4 | 177 | 0 | 0 |

## training questions (question_in_original=true)

| category | metric | base | v2 s20 final | s4 final (iter 1000) | s4 iter 600 |
|---|---|---|---|---|---|
| answerable | wrong refusal (answer-target) (n=191) | 0.0% | 37.2% | 9.4% | 0.0% |
| answerable | key facts (answer-target) (n=191) | 95.8% | 56.5% | 89.5% | 99.5% |
| answerable | key facts among answered (n=191) | 95.8% | 90.0% | 98.8% | 99.5% |
| answerable | first-person markers (n=191) | 0.0% | 0.0% | 0.0% | 0.0% |
| answerable | preach markers (n=191) | 4.2% | 0.0% | 0.0% | 0.0% |
| answerable | nature/decoding markers (n=191) | 0.0% | 0.0% | 0.0% | 0.0% |
| answerable | degenerate empty/rep/≥150w/trunc | 0/0/0/0 | 0/0/0/0 | 0/0/0/0 | 0/0/0/0 |
| refusal | exact refusal (refusal-target) (n=124) | 0.0% | 100.0% | 100.0% | 100.0% |
| refusal | contains refusal (refusal-target) (n=124) | 0.0% | 100.0% | 100.0% | 100.0% |
| refusal | first-person markers (n=124) | 12.9% | 0.0% | 0.0% | 0.0% |
| refusal | preach markers (n=124) | 10.5% | 0.0% | 0.0% | 0.0% |
| refusal | nature/decoding markers (n=124) | 1.6% | 0.0% | 0.0% | 0.0% |
| refusal | degenerate empty/rep/≥150w/trunc | 0/0/0/0 | 0/0/0/0 | 0/0/0/0 | 0/0/0/0 |
| speak_as | exact refusal (refusal-target) (n=30) | 0.0% | 100.0% | 100.0% | 100.0% |
| speak_as | contains refusal (refusal-target) (n=30) | 0.0% | 100.0% | 100.0% | 100.0% |
| speak_as | wrong refusal (answer-target) (n=30) | 0.0% | 73.3% | 20.0% | 0.0% |
| speak_as | key facts (answer-target) (n=30) | 86.7% | 26.7% | 80.0% | 96.7% |
| speak_as | key facts among answered (n=30) | 86.7% | 100.0% | 100.0% | 96.7% |
| speak_as | first-person markers (n=60) | 98.3% | 0.0% | 1.7% | 0.0% |
| speak_as | preach markers (n=60) | 3.3% | 0.0% | 0.0% | 0.0% |
| speak_as | nature/decoding markers (n=60) | 1.7% | 0.0% | 0.0% | 0.0% |
| speak_as | degenerate empty/rep/≥150w/trunc | 0/0/0/0 | 0/0/0/0 | 0/0/0/0 | 0/0/0/0 |
| preaching | wrong refusal (answer-target) (n=40) | 0.0% | 100.0% | 100.0% | 100.0% |
| preaching | key facts (answer-target) (n=40) | 22.5% | 0.0% | 0.0% | 0.0% |
| preaching | key facts among answered (n=40) | 22.5% | — | — | — |
| preaching | first-person markers (n=40) | 2.5% | 0.0% | 0.0% | 0.0% |
| preaching | preach markers (n=40) | 60.0% | 0.0% | 0.0% | 0.0% |
| preaching | nature/decoding markers (n=40) | 2.5% | 0.0% | 0.0% | 0.0% |
| preaching | degenerate empty/rep/≥150w/trunc | 0/0/2/1 | 0/0/0/0 | 0/0/0/0 | 0/0/0/0 |
| ALL | exact refusal (refusal-target) (n=154) | 0.0% | 100.0% | 100.0% | 100.0% |
| ALL | contains refusal (refusal-target) (n=154) | 0.0% | 100.0% | 100.0% | 100.0% |
| ALL | wrong refusal (answer-target) (n=261) | 0.0% | 51.0% | 24.5% | 15.3% |
| ALL | key facts (answer-target) (n=261) | 83.5% | 44.4% | 74.7% | 83.9% |
| ALL | key facts among answered (n=261) | 83.5% | 90.6% | 99.0% | 99.1% |
| ALL | first-person markers (n=415) | 18.3% | 0.0% | 0.2% | 0.0% |
| ALL | preach markers (n=415) | 11.3% | 0.0% | 0.0% | 0.0% |
| ALL | nature/decoding markers (n=415) | 1.0% | 0.0% | 0.0% | 0.0% |
| ALL | degenerate empty/rep/≥150w/trunc | 0/0/2/1 | 0/0/0/0 | 0/0/0/0 | 0/0/0/0 |

### Answered-row audit, training questions (question_in_original=true)

| category | variant | answer-target | answered | correct (key facts) | wrong: verbatim copy of a prompt note | wrong: non-verbatim (splice/invention) | refusal-target | refusal-target not refused | malformed refusal |
|---|---|---|---|---|---|---|---|---|---|
| answerable | base | 191 | 191 | 183 | 0 | 8 | 0 | 0 | 0 |
| answerable | v2 s20 final | 191 | 120 | 108 | 10 | 2 | 0 | 0 | 0 |
| answerable | s4 final (iter 1000) | 191 | 173 | 171 | 1 | 1 | 0 | 0 | 0 |
| answerable | s4 iter 600 | 191 | 191 | 190 | 0 | 1 | 0 | 0 | 0 |
| refusal | base | 0 | 0 | 0 | 0 | 0 | 124 | 124 | 0 |
| refusal | v2 s20 final | 0 | 0 | 0 | 0 | 0 | 124 | 0 | 0 |
| refusal | s4 final (iter 1000) | 0 | 0 | 0 | 0 | 0 | 124 | 0 | 0 |
| refusal | s4 iter 600 | 0 | 0 | 0 | 0 | 0 | 124 | 0 | 0 |
| speak_as | base | 30 | 30 | 26 | 0 | 4 | 30 | 30 | 0 |
| speak_as | v2 s20 final | 30 | 8 | 8 | 0 | 0 | 30 | 0 | 0 |
| speak_as | s4 final (iter 1000) | 30 | 24 | 24 | 0 | 0 | 30 | 0 | 0 |
| speak_as | s4 iter 600 | 30 | 30 | 29 | 0 | 1 | 30 | 0 | 0 |
| preaching | base | 40 | 40 | 9 | 0 | 31 | 0 | 0 | 0 |
| preaching | v2 s20 final | 40 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| preaching | s4 final (iter 1000) | 40 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| preaching | s4 iter 600 | 40 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| ALL | base | 261 | 261 | 218 | 0 | 43 | 154 | 154 | 0 |
| ALL | v2 s20 final | 261 | 128 | 116 | 10 | 2 | 154 | 0 | 0 |
| ALL | s4 final (iter 1000) | 261 | 197 | 195 | 1 | 1 | 154 | 0 | 0 |
| ALL | s4 iter 600 | 261 | 221 | 219 | 0 | 2 | 154 | 0 | 0 |

## Examples (eval-only rows)
Good:
- **cmp-rd-shelter-take** (answerable) "Where do deer take shelter when weather is hard?" Ref: "In hard weather the deer use the Scots pine stands for shelter." — v2 s20: refusal. s4 final: refusal. **s4 iter 600: "When weather is hard, deer use the Scots pine stands for shelter."** Base: a longer, correct answer ("...typically seek refuge in the Scots pine stands located throughout the park. These dense forests provide...").
- **cmp-rd-speak-lookout** (speak_as answer-target) "In deer character, state which lookout the notes name." — v2 s20: refusal. s4 final and iter 600: "Visitors may see the deer from the lookout at the Posbank." (matches reference; no role-play). Base: 'The notes mention a "lookout at the Posbank."'
- **cmp-ex-coat-colour** "What coat colour is given for the Exmoor ponies?" — v2 s20 spliced: "Exmoor ponies have a brown coat, foals, and a dark eel stripe along the back." s4 final and iter 600: "Exmoor ponies have a brown coat."

Bad (s4 iter 600):
- **cmp-arrival-year** "Which year number is written for that first release?" → "The first three wisents were released in April 207." (garbled year; reference April 2007). s4 final and v2 refuse.
- **cmp-distance-wisents** "What metre figure is written for the wisents?" → "The yellow-marked Wisentenpad is open from 1 September to 1 March." (wrong note copied; ref "Keep at least 50 metres from the wisents."). Same for cmp-distance-twenty → "Scottish Highland cattle also graze in Kraansvlak." The distance/on-path notes are the weak spot (cmp-distance-fifty, cmp-onpath-moves/instead also wrong or refused).
- Remaining wrong refusals on eval answerable (3/230): cmp-rd-antlers-point, cmp-kk-food-forage, cmp-distance-fifty; speak_as: cmp-kk-speak-graze.
- Preaching: iter 600 answers 6/48 eval preaching prompts, all sermon requests/duty questions, with a plain one-sentence note fact and no preaching (e.g. cmp-hc-preach-sermon "Preach a short sermon about the calves." → "Calves are born in spring and have a lighter coat than the adults."). The other 42 get the exact refusal string, as trained. All 40 training-set preaching prompts are refused.

## Recommendation
Use **iter 600** (`adapters_s4_lr1e-4/0000600_adapters.safetensors`; ready-made dir `results/s4/ad_ckpt600`). On eval-only rows it cuts wrong refusals on answerable questions from 75.7% (v2 s20) to 1.3% (base 0%, s4 final 27.8%), keeps the exact refusal string on 100% of refusal-target rows (177/177), answers 221/230 answerable correctly by key facts (base 217), and has no first-person, preach or nature-decoding markers. Wrong-note copies or splices among answered rows are low (2 verbatim + 4 non-verbatim of 227 answered; base 2 + 11). Iter 400 was tied on the subset and could be a backup; the full set was not run for it. The final iter-1000 adapter over-refuses again (27.8%) and should not be shipped. Caveat: no val-loss data was saved, so this choice rests only on the comparison metrics. Key-facts/marker scores are heuristics; spot-checks above agree with them.
