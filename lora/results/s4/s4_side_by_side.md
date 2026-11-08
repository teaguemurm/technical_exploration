# s4 side-by-side: base vs first adapter (scale 20) vs new adapter (scale 4, lr 1e-4)

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
