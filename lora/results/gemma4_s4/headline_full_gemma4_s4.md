# Headline full-set metrics (910 rows, MLX bf16, app layout + app sampler)

Script `headline_full_gemma4_s4.py`, predicates from `results/s4/analyze_s4.py`. Cells: count / n (percent). Over-refusal = output contains the refusal string on an answer-target row. Key facts = not refused AND must_include / ≥50% overlap. Voice = first-person markers on speak_as rows (lower is better). Preaching rows are trained to the refusal string in v2, so 'refused' is the trained behaviour.


## all

| metric | Gemma 4 E2B base | Gemma s4 iter 600 | Gemma s4 iter 800 | Gemma s4 final (1000) | Qwen2.5-1.5B base | Qwen s4 iter 600 |
|---|---|---|---|---|---|---|
| exact_refusal_refusal_target | 0/331 (0.0%) | 330/331 (99.7%) | 329/331 (99.4%) | 329/331 (99.4%) | 0/331 (0.0%) | 331/331 (100.0%) |
| contains_refusal_refusal_target | 0/331 (0.0%) | 330/331 (99.7%) | 329/331 (99.4%) | 329/331 (99.4%) | 0/331 (0.0%) | 331/331 (100.0%) |
| over_refusal_answerable | 0/421 (0.0%) | 19/421 (4.5%) | 11/421 (2.6%) | 6/421 (1.4%) | 0/421 (0.0%) | 3/421 (0.7%) |
| over_refusal_answer_target_ans+speak_as | 0/491 (0.0%) | 26/491 (5.3%) | 28/491 (5.7%) | 10/491 (2.0%) | 0/491 (0.0%) | 4/491 (0.8%) |
| key_facts_answer_target_ans+speak_as | 491/491 (100.0%) | 463/491 (94.3%) | 453/491 (92.3%) | 479/491 (97.6%) | 464/491 (94.5%) | 479/491 (97.6%) |
| key_facts_among_answered | 491/491 (100.0%) | 463/465 (99.6%) | 453/463 (97.8%) | 479/481 (99.6%) | 464/491 (94.5%) | 479/487 (98.4%) |
| voice_first_person_speak_as_rows | 91/132 (68.9%) | 0/132 (0.0%) | 0/132 (0.0%) | 0/132 (0.0%) | 87/132 (65.9%) | 0/132 (0.0%) |
| preaching_rows_refused(trained) | 0/88 (0.0%) | 88/88 (100.0%) | 88/88 (100.0%) | 88/88 (100.0%) | 0/88 (0.0%) | 82/88 (93.2%) |
| preach_markers_all_rows | 450/910 (49.5%) | 0/910 (0.0%) | 0/910 (0.0%) | 0/910 (0.0%) | 98/910 (10.8%) | 0/910 (0.0%) |
| nature_decode_all_rows | 10/910 (1.1%) | 0/910 (0.0%) | 0/910 (0.0%) | 0/910 (0.0%) | 6/910 (0.7%) | 0/910 (0.0%) |
| avg_words_answer_target | 139.4 | 9.0 | 11.5 | 9.3 | 25.7 | 9.0 |

## eval-only

| metric | Gemma 4 E2B base | Gemma s4 iter 600 | Gemma s4 iter 800 | Gemma s4 final (1000) | Qwen2.5-1.5B base | Qwen s4 iter 600 |
|---|---|---|---|---|---|---|
| exact_refusal_refusal_target | 0/177 (0.0%) | 177/177 (100.0%) | 176/177 (99.4%) | 176/177 (99.4%) | 0/177 (0.0%) | 177/177 (100.0%) |
| contains_refusal_refusal_target | 0/177 (0.0%) | 177/177 (100.0%) | 176/177 (99.4%) | 176/177 (99.4%) | 0/177 (0.0%) | 177/177 (100.0%) |
| over_refusal_answerable | 0/230 (0.0%) | 18/230 (7.8%) | 11/230 (4.8%) | 6/230 (2.6%) | 0/230 (0.0%) | 3/230 (1.3%) |
| over_refusal_answer_target_ans+speak_as | 0/270 (0.0%) | 19/270 (7.0%) | 12/270 (4.4%) | 6/270 (2.2%) | 0/270 (0.0%) | 4/270 (1.5%) |
| key_facts_answer_target_ans+speak_as | 270/270 (100.0%) | 250/270 (92.6%) | 250/270 (92.6%) | 262/270 (97.0%) | 255/270 (94.4%) | 260/270 (96.3%) |
| key_facts_among_answered | 270/270 (100.0%) | 250/251 (99.6%) | 250/258 (96.9%) | 262/264 (99.2%) | 255/270 (94.4%) | 260/266 (97.7%) |
| voice_first_person_speak_as_rows | 49/72 (68.1%) | 0/72 (0.0%) | 0/72 (0.0%) | 0/72 (0.0%) | 28/72 (38.9%) | 0/72 (0.0%) |
| preaching_rows_refused(trained) | 0/48 (0.0%) | 48/48 (100.0%) | 48/48 (100.0%) | 48/48 (100.0%) | 0/48 (0.0%) | 42/48 (87.5%) |
| preach_markers_all_rows | 227/495 (45.9%) | 0/495 (0.0%) | 0/495 (0.0%) | 0/495 (0.0%) | 51/495 (10.3%) | 0/495 (0.0%) |
| nature_decode_all_rows | 6/495 (1.2%) | 0/495 (0.0%) | 0/495 (0.0%) | 0/495 (0.0%) | 2/495 (0.4%) | 0/495 (0.0%) |
| avg_words_answer_target | 138.0 | 8.9 | 11.2 | 9.3 | 24.3 | 8.9 |

## training-q

| metric | Gemma 4 E2B base | Gemma s4 iter 600 | Gemma s4 iter 800 | Gemma s4 final (1000) | Qwen2.5-1.5B base | Qwen s4 iter 600 |
|---|---|---|---|---|---|---|
| exact_refusal_refusal_target | 0/154 (0.0%) | 153/154 (99.4%) | 153/154 (99.4%) | 153/154 (99.4%) | 0/154 (0.0%) | 154/154 (100.0%) |
| contains_refusal_refusal_target | 0/154 (0.0%) | 153/154 (99.4%) | 153/154 (99.4%) | 153/154 (99.4%) | 0/154 (0.0%) | 154/154 (100.0%) |
| over_refusal_answerable | 0/191 (0.0%) | 1/191 (0.5%) | 0/191 (0.0%) | 0/191 (0.0%) | 0/191 (0.0%) | 0/191 (0.0%) |
| over_refusal_answer_target_ans+speak_as | 0/221 (0.0%) | 7/221 (3.2%) | 16/221 (7.2%) | 4/221 (1.8%) | 0/221 (0.0%) | 0/221 (0.0%) |
| key_facts_answer_target_ans+speak_as | 221/221 (100.0%) | 213/221 (96.4%) | 203/221 (91.9%) | 217/221 (98.2%) | 209/221 (94.6%) | 219/221 (99.1%) |
| key_facts_among_answered | 221/221 (100.0%) | 213/214 (99.5%) | 203/205 (99.0%) | 217/217 (100.0%) | 209/221 (94.6%) | 219/221 (99.1%) |
| voice_first_person_speak_as_rows | 42/60 (70.0%) | 0/60 (0.0%) | 0/60 (0.0%) | 0/60 (0.0%) | 59/60 (98.3%) | 0/60 (0.0%) |
| preaching_rows_refused(trained) | 0/40 (0.0%) | 40/40 (100.0%) | 40/40 (100.0%) | 40/40 (100.0%) | 0/40 (0.0%) | 40/40 (100.0%) |
| preach_markers_all_rows | 223/415 (53.7%) | 0/415 (0.0%) | 0/415 (0.0%) | 0/415 (0.0%) | 47/415 (11.3%) | 0/415 (0.0%) |
| nature_decode_all_rows | 4/415 (1.0%) | 0/415 (0.0%) | 0/415 (0.0%) | 0/415 (0.0%) | 4/415 (1.0%) | 0/415 (0.0%) |
| avg_words_answer_target | 141.1 | 9.1 | 11.7 | 9.3 | 27.4 | 9.2 |
