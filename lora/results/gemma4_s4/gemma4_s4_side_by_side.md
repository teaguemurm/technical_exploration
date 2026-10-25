# Gemma 4 E2B s4 side-by-side (full comparison set)

Full comparison set (`data/comparison/comparison.jsonl`, 910 rows, app layout, app sampler: temp 0, seed 42, top-k 40, top-p 0.9, repeat penalty 1.1 over 64, max 256), MLX bf16. Gemma columns from `results/gemma4_s4/gemma4_s4*_mlx_app_comparison.csv` (`run_full_gemma.py`, base pass cached once and reused). Qwen columns from `results/s4_ckpt600_mlx_app_comparison.csv`. Metric code = `results/s4/analyze_s4.py` (= `results/analyze_v2_mlx_app.py`); audit = `results/s4/side_by_side_s4.py`.

Sanity: {"gemma s4 final (iter 1000)": 910, "gemma s4 iter 800": 910, "subset_base_runner_vs_full_base_identical_of_100": 100, "subset_ckpt800_vs_full_identical_of_100": 100, "subset_ckpt1000_vs_full_final_identical_of_100": 100}


## eval-only (question_in_original=false)

| category | metric | gemma base | gemma s4 final (iter 1000) | gemma s4 iter 800 | qwen base | qwen s4 iter 600 |
|---|---|---|---|---|---|---|
| answerable | wrong refusal (answer-target) (n=230) | 0.0% | 2.6% | 4.8% | 0.0% | 1.3% |
| answerable | key facts (answer-target) (n=230) | 100.0% | 96.5% | 91.7% | 94.3% | 96.1% |
| answerable | key facts among answered (n=230) | 100.0% | 99.1% | 96.3% | 94.3% | 97.4% |
| answerable | first-person markers (n=230) | 95.7% | 0.0% | 0.0% | 0.0% | 0.0% |
| answerable | preach markers (n=230) | 39.1% | 0.0% | 0.0% | 5.7% | 0.0% |
| answerable | nature/decoding markers (n=230) | 0.0% | 0.0% | 0.0% | 0.4% | 0.0% |
| answerable | avg words (answer-target) (n=230) | 136.4 | 9.2 | 11.4 | 24.8 | 8.8 |
| answerable | avg sentences (answer-target) (n=230) | 8.7 | 1.0 | 1.1 | 1.4 | 1.0 |
| answerable | degenerate empty/rep/≥150w/trunc | 0/99/82/48 | 0/0/0/0 | 0/3/1/1 | 0/0/0/0 | 0/0/0/0 |
| refusal | exact refusal (refusal-target) (n=145) | 0.0% | 99.3% | 99.3% | 0.0% | 100.0% |
| refusal | contains refusal (refusal-target) (n=145) | 0.0% | 99.3% | 99.3% | 0.0% | 100.0% |
| refusal | first-person markers (n=145) | 88.3% | 0.0% | 0.0% | 33.1% | 0.0% |
| refusal | preach markers (n=145) | 31.7% | 0.0% | 0.0% | 5.5% | 0.0% |
| refusal | nature/decoding markers (n=145) | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| refusal | degenerate empty/rep/≥150w/trunc | 0/7/131/94 | 0/0/0/0 | 0/0/0/0 | 0/0/0/0 | 0/0/0/0 |
| speak_as | exact refusal (refusal-target) (n=32) | 0.0% | 100.0% | 100.0% | 0.0% | 100.0% |
| speak_as | contains refusal (refusal-target) (n=32) | 0.0% | 100.0% | 100.0% | 0.0% | 100.0% |
| speak_as | wrong refusal (answer-target) (n=40) | 0.0% | 0.0% | 2.5% | 0.0% | 2.5% |
| speak_as | key facts (answer-target) (n=40) | 100.0% | 100.0% | 97.5% | 95.0% | 97.5% |
| speak_as | key facts among answered (n=40) | 100.0% | 100.0% | 100.0% | 95.0% | 100.0% |
| speak_as | first-person markers (n=72) | 68.1% | 0.0% | 0.0% | 38.9% | 0.0% |
| speak_as | preach markers (n=72) | 73.6% | 0.0% | 0.0% | 2.8% | 0.0% |
| speak_as | nature/decoding markers (n=72) | 1.4% | 0.0% | 0.0% | 0.0% | 0.0% |
| speak_as | avg words (answer-target) (n=40) | 147.2 | 9.9 | 10.4 | 21.1 | 9.8 |
| speak_as | avg sentences (answer-target) (n=40) | 12.1 | 1.0 | 1.0 | 1.2 | 1.0 |
| speak_as | degenerate empty/rep/≥150w/trunc | 0/7/44/35 | 0/0/0/0 | 0/0/0/0 | 0/0/1/0 | 0/0/0/0 |
| preaching | wrong refusal (answer-target) (n=48) | 0.0% | 100.0% | 100.0% | 0.0% | 87.5% |
| preaching | key facts (answer-target) (n=48) | 79.2% | 0.0% | 0.0% | 41.7% | 12.5% |
| preaching | key facts among answered (n=48) | 79.2% | — | — | 41.7% | 100.0% |
| preaching | first-person markers (n=48) | 22.9% | 0.0% | 0.0% | 8.3% | 0.0% |
| preaching | preach markers (n=48) | 79.2% | 0.0% | 0.0% | 58.3% | 0.0% |
| preaching | nature/decoding markers (n=48) | 10.4% | 0.0% | 0.0% | 2.1% | 0.0% |
| preaching | avg words (answer-target) (n=48) | 152.9 | 11.0 | 11.0 | 71.1 | 11.1 |
| preaching | avg sentences (answer-target) (n=48) | 13.5 | 1.0 | 1.0 | 3.5 | 1.0 |
| preaching | degenerate empty/rep/≥150w/trunc | 0/0/30/25 | 0/0/0/0 | 0/0/0/0 | 0/0/2/1 | 0/0/0/0 |
| ALL | exact refusal (refusal-target) (n=177) | 0.0% | 99.4% | 99.4% | 0.0% | 100.0% |
| ALL | contains refusal (refusal-target) (n=177) | 0.0% | 99.4% | 99.4% | 0.0% | 100.0% |
| ALL | wrong refusal (answer-target) (n=318) | 0.0% | 17.0% | 18.9% | 0.0% | 14.5% |
| ALL | key facts (answer-target) (n=318) | 96.9% | 82.4% | 78.6% | 86.5% | 83.6% |
| ALL | key facts among answered (n=318) | 96.9% | 99.2% | 96.9% | 86.5% | 97.8% |
| ALL | first-person markers (n=495) | 82.4% | 0.0% | 0.0% | 16.2% | 0.0% |
| ALL | preach markers (n=495) | 45.9% | 0.0% | 0.0% | 10.3% | 0.0% |
| ALL | nature/decoding markers (n=495) | 1.2% | 0.0% | 0.0% | 0.4% | 0.0% |
| ALL | avg words (answer-target) (n=318) | 140.3 | 9.5 | 11.2 | 31.3 | 9.3 |
| ALL | avg sentences (answer-target) (n=318) | 9.9 | 1.0 | 1.1 | 1.7 | 1.0 |
| ALL | degenerate empty/rep/≥150w/trunc | 0/113/287/202 | 0/0/0/0 | 0/3/1/1 | 0/0/3/1 | 0/0/0/0 |

### Answered-row audit, eval-only (question_in_original=false)

| category | variant | answer-target | answered | correct (key facts) | wrong: verbatim copy of a prompt note | wrong: non-verbatim (splice/invention) | refusal-target | refusal-target not refused | malformed refusal |
|---|---|---|---|---|---|---|---|---|---|
| answerable | gemma base | 230 | 230 | 230 | 0 | 0 | 0 | 0 | 0 |
| answerable | gemma s4 final (iter 1000) | 230 | 224 | 222 | 2 | 0 | 0 | 0 | 0 |
| answerable | gemma s4 iter 800 | 230 | 219 | 211 | 4 | 4 | 0 | 0 | 0 |
| answerable | qwen base | 230 | 230 | 217 | 2 | 11 | 0 | 0 | 0 |
| answerable | qwen s4 iter 600 | 230 | 227 | 221 | 2 | 4 | 0 | 0 | 0 |
| refusal | gemma base | 0 | 0 | 0 | 0 | 0 | 145 | 145 | 0 |
| refusal | gemma s4 final (iter 1000) | 0 | 0 | 0 | 0 | 0 | 145 | 1 | 0 |
| refusal | gemma s4 iter 800 | 0 | 0 | 0 | 0 | 0 | 145 | 1 | 0 |
| refusal | qwen base | 0 | 0 | 0 | 0 | 0 | 145 | 145 | 0 |
| refusal | qwen s4 iter 600 | 0 | 0 | 0 | 0 | 0 | 145 | 0 | 0 |
| speak_as | gemma base | 40 | 40 | 40 | 0 | 0 | 32 | 32 | 0 |
| speak_as | gemma s4 final (iter 1000) | 40 | 40 | 40 | 0 | 0 | 32 | 0 | 0 |
| speak_as | gemma s4 iter 800 | 40 | 39 | 39 | 0 | 0 | 32 | 0 | 0 |
| speak_as | qwen base | 40 | 40 | 38 | 0 | 2 | 32 | 32 | 0 |
| speak_as | qwen s4 iter 600 | 40 | 39 | 39 | 0 | 0 | 32 | 0 | 0 |
| preaching | gemma base | 48 | 48 | 38 | 0 | 10 | 0 | 0 | 0 |
| preaching | gemma s4 final (iter 1000) | 48 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| preaching | gemma s4 iter 800 | 48 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| preaching | qwen base | 48 | 48 | 20 | 0 | 28 | 0 | 0 | 0 |
| preaching | qwen s4 iter 600 | 48 | 6 | 6 | 0 | 0 | 0 | 0 | 0 |
| ALL | gemma base | 318 | 318 | 308 | 0 | 10 | 177 | 177 | 0 |
| ALL | gemma s4 final (iter 1000) | 318 | 264 | 262 | 2 | 0 | 177 | 1 | 0 |
| ALL | gemma s4 iter 800 | 318 | 258 | 250 | 4 | 4 | 177 | 1 | 0 |
| ALL | qwen base | 318 | 318 | 275 | 2 | 41 | 177 | 177 | 0 |
| ALL | qwen s4 iter 600 | 318 | 272 | 266 | 2 | 4 | 177 | 0 | 0 |

## training questions (question_in_original=true)

| category | metric | gemma base | gemma s4 final (iter 1000) | gemma s4 iter 800 | qwen base | qwen s4 iter 600 |
|---|---|---|---|---|---|---|
| answerable | wrong refusal (answer-target) (n=191) | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| answerable | key facts (answer-target) (n=191) | 100.0% | 100.0% | 99.0% | 95.8% | 99.5% |
| answerable | key facts among answered (n=191) | 100.0% | 100.0% | 99.0% | 95.8% | 99.5% |
| answerable | first-person markers (n=191) | 93.2% | 0.0% | 0.0% | 0.0% | 0.0% |
| answerable | preach markers (n=191) | 44.0% | 0.0% | 0.0% | 4.2% | 0.0% |
| answerable | nature/decoding markers (n=191) | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| answerable | avg words (answer-target) (n=191) | 138.6 | 9.2 | 11.9 | 24.4 | 9.1 |
| answerable | avg sentences (answer-target) (n=191) | 8.4 | 1.0 | 1.2 | 1.3 | 1.0 |
| answerable | degenerate empty/rep/≥150w/trunc | 0/93/73/53 | 0/0/0/0 | 0/2/2/1 | 0/0/0/0 | 0/0/0/0 |
| refusal | exact refusal (refusal-target) (n=124) | 0.0% | 99.2% | 99.2% | 0.0% | 100.0% |
| refusal | contains refusal (refusal-target) (n=124) | 0.0% | 99.2% | 99.2% | 0.0% | 100.0% |
| refusal | first-person markers (n=124) | 88.7% | 0.0% | 0.0% | 12.9% | 0.0% |
| refusal | preach markers (n=124) | 45.2% | 0.0% | 0.0% | 10.5% | 0.0% |
| refusal | nature/decoding markers (n=124) | 0.0% | 0.0% | 0.0% | 1.6% | 0.0% |
| refusal | degenerate empty/rep/≥150w/trunc | 0/3/117/96 | 0/0/0/0 | 0/0/0/0 | 0/0/0/0 | 0/0/0/0 |
| speak_as | exact refusal (refusal-target) (n=30) | 0.0% | 100.0% | 100.0% | 0.0% | 100.0% |
| speak_as | contains refusal (refusal-target) (n=30) | 0.0% | 100.0% | 100.0% | 0.0% | 100.0% |
| speak_as | wrong refusal (answer-target) (n=30) | 0.0% | 13.3% | 53.3% | 0.0% | 0.0% |
| speak_as | key facts (answer-target) (n=30) | 100.0% | 86.7% | 46.7% | 86.7% | 96.7% |
| speak_as | key facts among answered (n=30) | 100.0% | 100.0% | 100.0% | 86.7% | 96.7% |
| speak_as | first-person markers (n=60) | 70.0% | 0.0% | 0.0% | 98.3% | 0.0% |
| speak_as | preach markers (n=60) | 88.3% | 0.0% | 0.0% | 3.3% | 0.0% |
| speak_as | nature/decoding markers (n=60) | 5.0% | 0.0% | 0.0% | 1.7% | 0.0% |
| speak_as | avg words (answer-target) (n=30) | 157.1 | 10.1 | 10.7 | 46.4 | 9.7 |
| speak_as | avg sentences (answer-target) (n=30) | 13.2 | 1.0 | 1.0 | 2.5 | 1.0 |
| speak_as | degenerate empty/rep/≥150w/trunc | 0/3/38/29 | 0/0/0/0 | 0/0/0/0 | 0/0/0/0 | 0/0/0/0 |
| preaching | wrong refusal (answer-target) (n=40) | 0.0% | 100.0% | 100.0% | 0.0% | 100.0% |
| preaching | key facts (answer-target) (n=40) | 67.5% | 0.0% | 0.0% | 22.5% | 0.0% |
| preaching | key facts among answered (n=40) | 67.5% | — | — | 22.5% | — |
| preaching | first-person markers (n=40) | 27.5% | 0.0% | 0.0% | 2.5% | 0.0% |
| preaching | preach markers (n=40) | 75.0% | 0.0% | 0.0% | 60.0% | 0.0% |
| preaching | nature/decoding markers (n=40) | 2.5% | 0.0% | 0.0% | 2.5% | 0.0% |
| preaching | avg words (answer-target) (n=40) | 151.0 | 11.0 | 11.0 | 75.6 | 11.0 |
| preaching | avg sentences (answer-target) (n=40) | 14.2 | 1.0 | 1.0 | 3.6 | 1.0 |
| preaching | degenerate empty/rep/≥150w/trunc | 0/0/23/20 | 0/0/0/0 | 0/0/0/0 | 0/0/2/1 | 0/0/0/0 |
| ALL | exact refusal (refusal-target) (n=154) | 0.0% | 99.4% | 99.4% | 0.0% | 100.0% |
| ALL | contains refusal (refusal-target) (n=154) | 0.0% | 99.4% | 99.4% | 0.0% | 100.0% |
| ALL | wrong refusal (answer-target) (n=261) | 0.0% | 16.9% | 21.5% | 0.0% | 15.3% |
| ALL | key facts (answer-target) (n=261) | 95.0% | 83.1% | 77.8% | 83.5% | 83.9% |
| ALL | key facts among answered (n=261) | 95.0% | 100.0% | 99.0% | 83.5% | 99.1% |
| ALL | first-person markers (n=415) | 82.2% | 0.0% | 0.0% | 18.3% | 0.0% |
| ALL | preach markers (n=415) | 53.7% | 0.0% | 0.0% | 11.3% | 0.0% |
| ALL | nature/decoding markers (n=415) | 1.0% | 0.0% | 0.0% | 1.0% | 0.0% |
| ALL | avg words (answer-target) (n=261) | 142.6 | 9.6 | 11.6 | 34.8 | 9.4 |
| ALL | avg sentences (answer-target) (n=261) | 9.8 | 1.0 | 1.1 | 1.8 | 1.0 |
| ALL | degenerate empty/rep/≥150w/trunc | 0/99/251/198 | 0/0/0/0 | 0/2/2/1 | 0/0/2/1 | 0/0/0/0 |

### Answered-row audit, training questions (question_in_original=true)

| category | variant | answer-target | answered | correct (key facts) | wrong: verbatim copy of a prompt note | wrong: non-verbatim (splice/invention) | refusal-target | refusal-target not refused | malformed refusal |
|---|---|---|---|---|---|---|---|---|---|
| answerable | gemma base | 191 | 191 | 191 | 0 | 0 | 0 | 0 | 0 |
| answerable | gemma s4 final (iter 1000) | 191 | 191 | 191 | 0 | 0 | 0 | 0 | 0 |
| answerable | gemma s4 iter 800 | 191 | 191 | 189 | 0 | 2 | 0 | 0 | 0 |
| answerable | qwen base | 191 | 191 | 183 | 0 | 8 | 0 | 0 | 0 |
| answerable | qwen s4 iter 600 | 191 | 191 | 190 | 0 | 1 | 0 | 0 | 0 |
| refusal | gemma base | 0 | 0 | 0 | 0 | 0 | 124 | 124 | 0 |
| refusal | gemma s4 final (iter 1000) | 0 | 0 | 0 | 0 | 0 | 124 | 1 | 0 |
| refusal | gemma s4 iter 800 | 0 | 0 | 0 | 0 | 0 | 124 | 1 | 0 |
| refusal | qwen base | 0 | 0 | 0 | 0 | 0 | 124 | 124 | 0 |
| refusal | qwen s4 iter 600 | 0 | 0 | 0 | 0 | 0 | 124 | 0 | 0 |
| speak_as | gemma base | 30 | 30 | 30 | 0 | 0 | 30 | 30 | 0 |
| speak_as | gemma s4 final (iter 1000) | 30 | 26 | 26 | 0 | 0 | 30 | 0 | 0 |
| speak_as | gemma s4 iter 800 | 30 | 14 | 14 | 0 | 0 | 30 | 0 | 0 |
| speak_as | qwen base | 30 | 30 | 26 | 0 | 4 | 30 | 30 | 0 |
| speak_as | qwen s4 iter 600 | 30 | 30 | 29 | 0 | 1 | 30 | 0 | 0 |
| preaching | gemma base | 40 | 40 | 27 | 0 | 13 | 0 | 0 | 0 |
| preaching | gemma s4 final (iter 1000) | 40 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| preaching | gemma s4 iter 800 | 40 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| preaching | qwen base | 40 | 40 | 9 | 0 | 31 | 0 | 0 | 0 |
| preaching | qwen s4 iter 600 | 40 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| ALL | gemma base | 261 | 261 | 248 | 0 | 13 | 154 | 154 | 0 |
| ALL | gemma s4 final (iter 1000) | 261 | 217 | 217 | 0 | 0 | 154 | 1 | 0 |
| ALL | gemma s4 iter 800 | 261 | 205 | 203 | 0 | 2 | 154 | 1 | 0 |
| ALL | qwen base | 261 | 261 | 218 | 0 | 43 | 154 | 154 | 0 |
| ALL | qwen s4 iter 600 | 261 | 221 | 219 | 0 | 2 | 154 | 0 | 0 |

## all 910 rows

| category | metric | gemma base | gemma s4 final (iter 1000) | gemma s4 iter 800 | qwen base | qwen s4 iter 600 |
|---|---|---|---|---|---|---|
| answerable | wrong refusal (answer-target) (n=421) | 0.0% | 1.4% | 2.6% | 0.0% | 0.7% |
| answerable | key facts (answer-target) (n=421) | 100.0% | 98.1% | 95.0% | 95.0% | 97.6% |
| answerable | key facts among answered (n=421) | 100.0% | 99.5% | 97.6% | 95.0% | 98.3% |
| answerable | first-person markers (n=421) | 94.5% | 0.0% | 0.0% | 0.0% | 0.0% |
| answerable | preach markers (n=421) | 41.3% | 0.0% | 0.0% | 5.0% | 0.0% |
| answerable | nature/decoding markers (n=421) | 0.0% | 0.0% | 0.0% | 0.2% | 0.0% |
| answerable | avg words (answer-target) (n=421) | 137.4 | 9.2 | 11.6 | 24.6 | 8.9 |
| answerable | avg sentences (answer-target) (n=421) | 8.6 | 1.0 | 1.2 | 1.4 | 1.0 |
| answerable | degenerate empty/rep/≥150w/trunc | 0/192/155/101 | 0/0/0/0 | 0/5/3/2 | 0/0/0/0 | 0/0/0/0 |
| refusal | exact refusal (refusal-target) (n=269) | 0.0% | 99.3% | 99.3% | 0.0% | 100.0% |
| refusal | contains refusal (refusal-target) (n=269) | 0.0% | 99.3% | 99.3% | 0.0% | 100.0% |
| refusal | first-person markers (n=269) | 88.5% | 0.0% | 0.0% | 23.8% | 0.0% |
| refusal | preach markers (n=269) | 37.9% | 0.0% | 0.0% | 7.8% | 0.0% |
| refusal | nature/decoding markers (n=269) | 0.0% | 0.0% | 0.0% | 0.7% | 0.0% |
| refusal | degenerate empty/rep/≥150w/trunc | 0/10/248/190 | 0/0/0/0 | 0/0/0/0 | 0/0/0/0 | 0/0/0/0 |
| speak_as | exact refusal (refusal-target) (n=62) | 0.0% | 100.0% | 100.0% | 0.0% | 100.0% |
| speak_as | contains refusal (refusal-target) (n=62) | 0.0% | 100.0% | 100.0% | 0.0% | 100.0% |
| speak_as | wrong refusal (answer-target) (n=70) | 0.0% | 5.7% | 24.3% | 0.0% | 1.4% |
| speak_as | key facts (answer-target) (n=70) | 100.0% | 94.3% | 75.7% | 91.4% | 97.1% |
| speak_as | key facts among answered (n=70) | 100.0% | 100.0% | 100.0% | 91.4% | 98.6% |
| speak_as | first-person markers (n=132) | 68.9% | 0.0% | 0.0% | 65.9% | 0.0% |
| speak_as | preach markers (n=132) | 80.3% | 0.0% | 0.0% | 3.0% | 0.0% |
| speak_as | nature/decoding markers (n=132) | 3.0% | 0.0% | 0.0% | 0.8% | 0.0% |
| speak_as | avg words (answer-target) (n=70) | 151.4 | 10.0 | 10.5 | 31.9 | 9.8 |
| speak_as | avg sentences (answer-target) (n=70) | 12.6 | 1.0 | 1.0 | 1.8 | 1.0 |
| speak_as | degenerate empty/rep/≥150w/trunc | 0/10/82/64 | 0/0/0/0 | 0/0/0/0 | 0/0/1/0 | 0/0/0/0 |
| preaching | wrong refusal (answer-target) (n=88) | 0.0% | 100.0% | 100.0% | 0.0% | 93.2% |
| preaching | key facts (answer-target) (n=88) | 73.9% | 0.0% | 0.0% | 33.0% | 6.8% |
| preaching | key facts among answered (n=88) | 73.9% | — | — | 33.0% | 100.0% |
| preaching | first-person markers (n=88) | 25.0% | 0.0% | 0.0% | 5.7% | 0.0% |
| preaching | preach markers (n=88) | 77.3% | 0.0% | 0.0% | 59.1% | 0.0% |
| preaching | nature/decoding markers (n=88) | 6.8% | 0.0% | 0.0% | 2.3% | 0.0% |
| preaching | avg words (answer-target) (n=88) | 152.1 | 11.0 | 11.0 | 73.1 | 11.1 |
| preaching | avg sentences (answer-target) (n=88) | 13.8 | 1.0 | 1.0 | 3.5 | 1.0 |
| preaching | degenerate empty/rep/≥150w/trunc | 0/0/53/45 | 0/0/0/0 | 0/0/0/0 | 0/0/4/2 | 0/0/0/0 |
| ALL | exact refusal (refusal-target) (n=331) | 0.0% | 99.4% | 99.4% | 0.0% | 100.0% |
| ALL | contains refusal (refusal-target) (n=331) | 0.0% | 99.4% | 99.4% | 0.0% | 100.0% |
| ALL | wrong refusal (answer-target) (n=579) | 0.0% | 16.9% | 20.0% | 0.0% | 14.9% |
| ALL | key facts (answer-target) (n=579) | 96.0% | 82.7% | 78.2% | 85.1% | 83.8% |
| ALL | key facts among answered (n=579) | 96.0% | 99.6% | 97.8% | 85.1% | 98.4% |
| ALL | first-person markers (n=910) | 82.3% | 0.0% | 0.0% | 17.1% | 0.0% |
| ALL | preach markers (n=910) | 49.5% | 0.0% | 0.0% | 10.8% | 0.0% |
| ALL | nature/decoding markers (n=910) | 1.1% | 0.0% | 0.0% | 0.7% | 0.0% |
| ALL | avg words (answer-target) (n=579) | 141.3 | 9.6 | 11.4 | 32.9 | 9.3 |
| ALL | avg sentences (answer-target) (n=579) | 9.9 | 1.0 | 1.1 | 1.7 | 1.0 |
| ALL | degenerate empty/rep/≥150w/trunc | 0/212/538/400 | 0/0/0/0 | 0/5/3/2 | 0/0/5/2 | 0/0/0/0 |

### Answered-row audit, all 910 rows

| category | variant | answer-target | answered | correct (key facts) | wrong: verbatim copy of a prompt note | wrong: non-verbatim (splice/invention) | refusal-target | refusal-target not refused | malformed refusal |
|---|---|---|---|---|---|---|---|---|---|
| answerable | gemma base | 421 | 421 | 421 | 0 | 0 | 0 | 0 | 0 |
| answerable | gemma s4 final (iter 1000) | 421 | 415 | 413 | 2 | 0 | 0 | 0 | 0 |
| answerable | gemma s4 iter 800 | 421 | 410 | 400 | 4 | 6 | 0 | 0 | 0 |
| answerable | qwen base | 421 | 421 | 400 | 2 | 19 | 0 | 0 | 0 |
| answerable | qwen s4 iter 600 | 421 | 418 | 411 | 2 | 5 | 0 | 0 | 0 |
| refusal | gemma base | 0 | 0 | 0 | 0 | 0 | 269 | 269 | 0 |
| refusal | gemma s4 final (iter 1000) | 0 | 0 | 0 | 0 | 0 | 269 | 2 | 0 |
| refusal | gemma s4 iter 800 | 0 | 0 | 0 | 0 | 0 | 269 | 2 | 0 |
| refusal | qwen base | 0 | 0 | 0 | 0 | 0 | 269 | 269 | 0 |
| refusal | qwen s4 iter 600 | 0 | 0 | 0 | 0 | 0 | 269 | 0 | 0 |
| speak_as | gemma base | 70 | 70 | 70 | 0 | 0 | 62 | 62 | 0 |
| speak_as | gemma s4 final (iter 1000) | 70 | 66 | 66 | 0 | 0 | 62 | 0 | 0 |
| speak_as | gemma s4 iter 800 | 70 | 53 | 53 | 0 | 0 | 62 | 0 | 0 |
| speak_as | qwen base | 70 | 70 | 64 | 0 | 6 | 62 | 62 | 0 |
| speak_as | qwen s4 iter 600 | 70 | 69 | 68 | 0 | 1 | 62 | 0 | 0 |
| preaching | gemma base | 88 | 88 | 65 | 0 | 23 | 0 | 0 | 0 |
| preaching | gemma s4 final (iter 1000) | 88 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| preaching | gemma s4 iter 800 | 88 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| preaching | qwen base | 88 | 88 | 29 | 0 | 59 | 0 | 0 | 0 |
| preaching | qwen s4 iter 600 | 88 | 6 | 6 | 0 | 0 | 0 | 0 | 0 |
| ALL | gemma base | 579 | 579 | 556 | 0 | 23 | 331 | 331 | 0 |
| ALL | gemma s4 final (iter 1000) | 579 | 481 | 479 | 2 | 0 | 331 | 2 | 0 |
| ALL | gemma s4 iter 800 | 579 | 463 | 453 | 4 | 6 | 331 | 2 | 0 |
| ALL | qwen base | 579 | 579 | 493 | 2 | 84 | 331 | 331 | 0 |
| ALL | qwen s4 iter 600 | 579 | 493 | 485 | 2 | 6 | 331 | 0 | 0 |
