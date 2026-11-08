# Headline full-set metrics (910 rows, MLX bf16, app layout + app sampler)

Script `headline_full_gemma4_s4.py`, predicates from `results/s4/analyze_s4.py`. Cells: count / n (percent). Over-refusal = output contains the refusal string on an answer-target row. Key facts = not refused AND must_include / ≥50% overlap. Voice = first-person markers on speak_as rows (lower is better). Preaching rows are trained to the refusal string in v2, so 'refused' is the trained behaviour.


## all

| metric | Gemma 4 E2B base (thinking off) | Gemma s4 iter 600 (thinking on) | Gemma s4 iter 800 (thinking on) | Gemma s4 final (1000, thinking off) | Qwen2.5-1.5B base | Qwen s4 iter 600 |
|---|---|---|---|---|---|---|
| exact_refusal_refusal_target | 0/331 (0.0%) | 330/331 (99.7%) | 329/331 (99.4%) | 328/331 (99.1%) | 0/331 (0.0%) | 331/331 (100.0%) |
| contains_refusal_refusal_target | 0/331 (0.0%) | 330/331 (99.7%) | 329/331 (99.4%) | 328/331 (99.1%) | 0/331 (0.0%) | 331/331 (100.0%) |
| over_refusal_answerable | 0/421 (0.0%) | 19/421 (4.5%) | 11/421 (2.6%) | 6/421 (1.4%) | 0/421 (0.0%) | 3/421 (0.7%) |
| over_refusal_answer_target_ans+speak_as | 0/491 (0.0%) | 26/491 (5.3%) | 28/491 (5.7%) | 10/491 (2.0%) | 0/491 (0.0%) | 4/491 (0.8%) |
| key_facts_answer_target_ans+speak_as | 482/491 (98.2%) | 463/491 (94.3%) | 453/491 (92.3%) | 478/491 (97.4%) | 464/491 (94.5%) | 479/491 (97.6%) |
| key_facts_among_answered | 482/491 (98.2%) | 463/465 (99.6%) | 453/463 (97.8%) | 478/481 (99.4%) | 464/491 (94.5%) | 479/487 (98.4%) |
| voice_first_person_speak_as_rows | 109/132 (82.6%) | 0/132 (0.0%) | 0/132 (0.0%) | 0/132 (0.0%) | 87/132 (65.9%) | 0/132 (0.0%) |
| preaching_rows_refused(trained) | 0/88 (0.0%) | 88/88 (100.0%) | 88/88 (100.0%) | 88/88 (100.0%) | 0/88 (0.0%) | 82/88 (93.2%) |
| preach_markers_all_rows | 46/910 (5.1%) | 0/910 (0.0%) | 0/910 (0.0%) | 0/910 (0.0%) | 98/910 (10.8%) | 0/910 (0.0%) |
| nature_decode_all_rows | 0/910 (0.0%) | 0/910 (0.0%) | 0/910 (0.0%) | 0/910 (0.0%) | 6/910 (0.7%) | 0/910 (0.0%) |
| avg_words_answer_target | 17.9 | 9.0 | 11.5 | 9.3 | 25.7 | 9.0 |

## eval-only

| metric | Gemma 4 E2B base (thinking off) | Gemma s4 iter 600 (thinking on) | Gemma s4 iter 800 (thinking on) | Gemma s4 final (1000, thinking off) | Qwen2.5-1.5B base | Qwen s4 iter 600 |
|---|---|---|---|---|---|---|
| exact_refusal_refusal_target | 0/177 (0.0%) | 177/177 (100.0%) | 176/177 (99.4%) | 175/177 (98.9%) | 0/177 (0.0%) | 177/177 (100.0%) |
| contains_refusal_refusal_target | 0/177 (0.0%) | 177/177 (100.0%) | 176/177 (99.4%) | 175/177 (98.9%) | 0/177 (0.0%) | 177/177 (100.0%) |
| over_refusal_answerable | 0/230 (0.0%) | 18/230 (7.8%) | 11/230 (4.8%) | 6/230 (2.6%) | 0/230 (0.0%) | 3/230 (1.3%) |
| over_refusal_answer_target_ans+speak_as | 0/270 (0.0%) | 19/270 (7.0%) | 12/270 (4.4%) | 6/270 (2.2%) | 0/270 (0.0%) | 4/270 (1.5%) |
| key_facts_answer_target_ans+speak_as | 268/270 (99.3%) | 250/270 (92.6%) | 250/270 (92.6%) | 262/270 (97.0%) | 255/270 (94.4%) | 260/270 (96.3%) |
| key_facts_among_answered | 268/270 (99.3%) | 250/251 (99.6%) | 250/258 (96.9%) | 262/264 (99.2%) | 255/270 (94.4%) | 260/266 (97.7%) |
| voice_first_person_speak_as_rows | 54/72 (75.0%) | 0/72 (0.0%) | 0/72 (0.0%) | 0/72 (0.0%) | 28/72 (38.9%) | 0/72 (0.0%) |
| preaching_rows_refused(trained) | 0/48 (0.0%) | 48/48 (100.0%) | 48/48 (100.0%) | 48/48 (100.0%) | 0/48 (0.0%) | 42/48 (87.5%) |
| preach_markers_all_rows | 30/495 (6.1%) | 0/495 (0.0%) | 0/495 (0.0%) | 0/495 (0.0%) | 51/495 (10.3%) | 0/495 (0.0%) |
| nature_decode_all_rows | 0/495 (0.0%) | 0/495 (0.0%) | 0/495 (0.0%) | 0/495 (0.0%) | 2/495 (0.4%) | 0/495 (0.0%) |
| avg_words_answer_target | 17.5 | 8.9 | 11.2 | 9.2 | 24.3 | 8.9 |

## training-q

| metric | Gemma 4 E2B base (thinking off) | Gemma s4 iter 600 (thinking on) | Gemma s4 iter 800 (thinking on) | Gemma s4 final (1000, thinking off) | Qwen2.5-1.5B base | Qwen s4 iter 600 |
|---|---|---|---|---|---|---|
| exact_refusal_refusal_target | 0/154 (0.0%) | 153/154 (99.4%) | 153/154 (99.4%) | 153/154 (99.4%) | 0/154 (0.0%) | 154/154 (100.0%) |
| contains_refusal_refusal_target | 0/154 (0.0%) | 153/154 (99.4%) | 153/154 (99.4%) | 153/154 (99.4%) | 0/154 (0.0%) | 154/154 (100.0%) |
| over_refusal_answerable | 0/191 (0.0%) | 1/191 (0.5%) | 0/191 (0.0%) | 0/191 (0.0%) | 0/191 (0.0%) | 0/191 (0.0%) |
| over_refusal_answer_target_ans+speak_as | 0/221 (0.0%) | 7/221 (3.2%) | 16/221 (7.2%) | 4/221 (1.8%) | 0/221 (0.0%) | 0/221 (0.0%) |
| key_facts_answer_target_ans+speak_as | 214/221 (96.8%) | 213/221 (96.4%) | 203/221 (91.9%) | 216/221 (97.7%) | 209/221 (94.6%) | 219/221 (99.1%) |
| key_facts_among_answered | 214/221 (96.8%) | 213/214 (99.5%) | 203/205 (99.0%) | 216/217 (99.5%) | 209/221 (94.6%) | 219/221 (99.1%) |
| voice_first_person_speak_as_rows | 55/60 (91.7%) | 0/60 (0.0%) | 0/60 (0.0%) | 0/60 (0.0%) | 59/60 (98.3%) | 0/60 (0.0%) |
| preaching_rows_refused(trained) | 0/40 (0.0%) | 40/40 (100.0%) | 40/40 (100.0%) | 40/40 (100.0%) | 0/40 (0.0%) | 40/40 (100.0%) |
| preach_markers_all_rows | 16/415 (3.9%) | 0/415 (0.0%) | 0/415 (0.0%) | 0/415 (0.0%) | 47/415 (11.3%) | 0/415 (0.0%) |
| nature_decode_all_rows | 0/415 (0.0%) | 0/415 (0.0%) | 0/415 (0.0%) | 0/415 (0.0%) | 4/415 (1.0%) | 0/415 (0.0%) |
| avg_words_answer_target | 18.3 | 9.1 | 11.7 | 9.4 | 27.4 | 9.2 |
