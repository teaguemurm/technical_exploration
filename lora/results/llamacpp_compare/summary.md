# llama.cpp b10481 laptop comparison of the constraint LoRAs (910-row comparison set)

Cells: percent (count). Metrics and predicates as `results/gemma4_s4/headline_full_gemma4_s4.md` (`results/s4/analyze_s4.py`, imported unchanged). Refusal-target = reference is exactly the refusal string (269 refusal + 62 speak_as rows = 331). 'preaching refused' = output contains the refusal string (the headline table's 'refused (trained)', i.e. refusal counted as correct); 'preaching fact' = analyze_s4 key facts on the preaching rows (reference is a factual answer; analyze_s4.stats treats them as answer-target, so a refusal there counts as wrong refusal). MLX rows are the earlier bf16 MLX runs (condition B, app layout), rescored here; they reproduce the headline numbers exactly. Condition A has no notes, so refusing there is grounded behaviour; its over-refusal is a 'how much does the LoRA refuse without notes' check.

## 1. Qwen2.5-1.5B-Instruct Q4_K_M + s4 ckpt600 LoRA

### full

| variant (full) | exact refusal (refusal-target) n=331 | over-refusal answerable n=421 | over-refusal ans+speak_as n=491 | key facts ans+speak_as n=491 | key facts among answered | 1st person speak_as (lower better) n=132 | preaching refused n=88 | preaching fact n=88 | preach markers, preaching rows n=88 | preach markers, all rows n=910 | nature/decode, all rows n=910 | avg words ans+speak_as |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen base + C | 41.1 (136) | 0.2 (1) | 0.4 (2) | 97.1 (477) | 97.5 (477/489) | 54.5 (72) | 20.5 (18) | 21.6 (19) | 35.2 (31) | 3.8 (35) | 0.0 (0) | 11.7 |
| Qwen LoRA ckpt600 + B | 93.7 (310) | 0.2 (1) | 0.4 (2) | 97.6 (479) | 98.0 (479/489) | 0.0 (0) | 86.4 (76) | 12.5 (11) | 0.0 (0) | 0.0 (0) | 0.0 (0) | 8.9 |
| Qwen LoRA ckpt600 + C | 94.3 (312) | 0.5 (2) | 0.6 (3) | 97.4 (478) | 98.0 (478/488) | 0.0 (0) | 88.6 (78) | 11.4 (10) | 0.0 (0) | 0.0 (0) | 0.0 (0) | 8.9 |
| Qwen LoRA ckpt600 + A | 99.7 (330) | 98.8 (416) | 98.6 (484) | 1.0 (5) | 71.4 (5/7) | 1.5 (2) | 100.0 (88) | 0.0 (0) | 0.0 (0) | 0.0 (0) | 0.0 (0) | 11.0 |
| MLX bf16 Qwen LoRA ckpt600 + B | 100.0 (331) | 0.7 (3) | 0.8 (4) | 97.6 (479) | 98.4 (479/487) | 0.0 (0) | 93.2 (82) | 6.8 (6) | 0.0 (0) | 0.0 (0) | 0.0 (0) | 9.0 |
| MLX bf16 Qwen base + B | 0.0 (0) | 0.0 (0) | 0.0 (0) | 94.5 (464) | 94.5 (464/491) | 65.9 (87) | 0.0 (0) | 33.0 (29) | 59.1 (52) | 10.8 (98) | 0.7 (6) | 25.7 |

### full_minus_selection

| variant (full_minus_selection) | exact refusal (refusal-target) n=302 | over-refusal answerable n=361 | over-refusal ans+speak_as n=427 | key facts ans+speak_as n=427 | key facts among answered | 1st person speak_as (lower better) n=124 | preaching refused n=81 | preaching fact n=81 | preach markers, preaching rows n=81 | preach markers, all rows n=810 | nature/decode, all rows n=810 | avg words ans+speak_as |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen base + C | 40.7 (123) | 0.3 (1) | 0.5 (2) | 97.2 (415) | 97.6 (415/425) | 55.6 (69) | 21.0 (17) | 22.2 (18) | 33.3 (27) | 3.8 (31) | 0.0 (0) | 11.7 |
| Qwen LoRA ckpt600 + B | 93.7 (283) | 0.3 (1) | 0.5 (2) | 97.7 (417) | 98.1 (417/425) | 0.0 (0) | 87.7 (71) | 11.1 (9) | 0.0 (0) | 0.0 (0) | 0.0 (0) | 9.0 |
| Qwen LoRA ckpt600 + C | 94.4 (285) | 0.6 (2) | 0.7 (3) | 97.4 (416) | 98.1 (416/424) | 0.0 (0) | 90.1 (73) | 9.9 (8) | 0.0 (0) | 0.0 (0) | 0.0 (0) | 9.0 |
| Qwen LoRA ckpt600 + A | 99.7 (301) | 98.6 (356) | 98.4 (420) | 1.2 (5) | 71.4 (5/7) | 1.6 (2) | 100.0 (81) | 0.0 (0) | 0.0 (0) | 0.0 (0) | 0.0 (0) | 10.9 |
| MLX bf16 Qwen LoRA ckpt600 + B | 100.0 (302) | 0.6 (2) | 0.7 (3) | 97.9 (418) | 98.6 (418/424) | 0.0 (0) | 93.8 (76) | 6.2 (5) | 0.0 (0) | 0.0 (0) | 0.0 (0) | 9.1 |
| MLX bf16 Qwen base + B | 0.0 (0) | 0.0 (0) | 0.0 (0) | 95.1 (406) | 95.1 (406/427) | 67.7 (84) | 0.0 (0) | 32.1 (26) | 58.0 (47) | 10.7 (87) | 0.7 (6) | 25.9 |

### eval_only

| variant (eval_only) | exact refusal (refusal-target) n=177 | over-refusal answerable n=230 | over-refusal ans+speak_as n=270 | key facts ans+speak_as n=270 | key facts among answered | 1st person speak_as (lower better) n=72 | preaching refused n=48 | preaching fact n=48 | preach markers, preaching rows n=48 | preach markers, all rows n=495 | nature/decode, all rows n=495 | avg words ans+speak_as |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen base + C | 48.0 (85) | 0.4 (1) | 0.4 (1) | 97.0 (262) | 97.4 (262/269) | 33.3 (24) | 20.8 (10) | 29.2 (14) | 29.2 (14) | 3.2 (16) | 0.0 (0) | 11.3 |
| Qwen LoRA ckpt600 + B | 92.1 (163) | 0.4 (1) | 0.7 (2) | 95.9 (259) | 96.6 (259/268) | 0.0 (0) | 77.1 (37) | 22.9 (11) | 0.0 (0) | 0.0 (0) | 0.0 (0) | 8.9 |
| Qwen LoRA ckpt600 + C | 92.7 (164) | 0.9 (2) | 1.1 (3) | 95.9 (259) | 97.0 (259/267) | 0.0 (0) | 79.2 (38) | 20.8 (10) | 0.0 (0) | 0.0 (0) | 0.0 (0) | 8.9 |
| Qwen LoRA ckpt600 + A | 99.4 (176) | 98.7 (227) | 98.9 (267) | 1.1 (3) | 100.0 (3/3) | 0.0 (0) | 100.0 (48) | 0.0 (0) | 0.0 (0) | 0.0 (0) | 0.0 (0) | 11.0 |
| MLX bf16 Qwen LoRA ckpt600 + B | 100.0 (177) | 1.3 (3) | 1.5 (4) | 96.3 (260) | 97.7 (260/266) | 0.0 (0) | 87.5 (42) | 12.5 (6) | 0.0 (0) | 0.0 (0) | 0.0 (0) | 8.9 |
| MLX bf16 Qwen base + B | 0.0 (0) | 0.0 (0) | 0.0 (0) | 94.4 (255) | 94.4 (255/270) | 38.9 (28) | 0.0 (0) | 41.7 (20) | 58.3 (28) | 10.3 (51) | 0.4 (2) | 24.3 |

### eval_minus_selection

| variant (eval_minus_selection) | exact refusal (refusal-target) n=148 | over-refusal answerable n=170 | over-refusal ans+speak_as n=206 | key facts ans+speak_as n=206 | key facts among answered | 1st person speak_as (lower better) n=64 | preaching refused n=41 | preaching fact n=41 | preach markers, preaching rows n=41 | preach markers, all rows n=395 | nature/decode, all rows n=395 | avg words ans+speak_as |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen base + C | 48.6 (72) | 0.6 (1) | 0.5 (1) | 97.1 (200) | 97.6 (200/205) | 32.8 (21) | 22.0 (9) | 31.7 (13) | 24.4 (10) | 3.0 (12) | 0.0 (0) | 11.4 |
| Qwen LoRA ckpt600 + B | 91.9 (136) | 0.6 (1) | 1.0 (2) | 95.6 (197) | 96.6 (197/204) | 0.0 (0) | 78.0 (32) | 22.0 (9) | 0.0 (0) | 0.0 (0) | 0.0 (0) | 9.0 |
| Qwen LoRA ckpt600 + C | 92.6 (137) | 1.2 (2) | 1.5 (3) | 95.6 (197) | 97.0 (197/203) | 0.0 (0) | 80.5 (33) | 19.5 (8) | 0.0 (0) | 0.0 (0) | 0.0 (0) | 9.0 |
| Qwen LoRA ckpt600 + A | 99.3 (147) | 98.2 (167) | 98.5 (203) | 1.5 (3) | 100.0 (3/3) | 0.0 (0) | 100.0 (41) | 0.0 (0) | 0.0 (0) | 0.0 (0) | 0.0 (0) | 10.9 |
| MLX bf16 Qwen LoRA ckpt600 + B | 100.0 (148) | 1.2 (2) | 1.5 (3) | 96.6 (199) | 98.0 (199/203) | 0.0 (0) | 87.8 (36) | 12.2 (5) | 0.0 (0) | 0.0 (0) | 0.0 (0) | 8.9 |
| MLX bf16 Qwen base + B | 0.0 (0) | 0.0 (0) | 0.0 (0) | 95.6 (197) | 95.6 (197/206) | 39.1 (25) | 0.0 (0) | 41.5 (17) | 56.1 (23) | 10.1 (40) | 0.5 (2) | 24.3 |

Differences vs Qwen base + C (full, percentage points):

| variant | exact refusal (refusal-target) | over-refusal answerable | over-refusal ans+speak_as | key facts ans+speak_as | key facts among answered | 1st person speak_as (lower better) | preaching refused | preaching fact | preach markers, preaching rows | preach markers, all rows | nature/decode, all rows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen LoRA ckpt600 + B − base+C | +52.6 | +0.0 | +0.0 | +0.4 | +0.4 | -54.5 | +65.9 | -9.1 | -35.2 | -3.8 | +0.0 |
| Qwen LoRA ckpt600 + C − base+C | +53.2 | +0.2 | +0.2 | +0.2 | +0.4 | -54.5 | +68.2 | -10.2 | -35.2 | -3.8 | +0.0 |
| Qwen LoRA ckpt600 + A − base+C | +58.6 | +98.6 | +98.2 | -96.1 | -26.1 | -53.0 | +79.5 | -21.6 | -35.2 | -3.8 | +0.0 |

Differences vs Qwen base + C (full_minus_selection, percentage points):

| variant | exact refusal (refusal-target) | over-refusal answerable | over-refusal ans+speak_as | key facts ans+speak_as | key facts among answered | 1st person speak_as (lower better) | preaching refused | preaching fact | preach markers, preaching rows | preach markers, all rows | nature/decode, all rows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen LoRA ckpt600 + B − base+C | +53.0 | +0.0 | +0.0 | +0.5 | +0.5 | -55.6 | +66.7 | -11.1 | -33.3 | -3.8 | +0.0 |
| Qwen LoRA ckpt600 + C − base+C | +53.6 | +0.3 | +0.2 | +0.2 | +0.5 | -55.6 | +69.1 | -12.3 | -33.3 | -3.8 | +0.0 |
| Qwen LoRA ckpt600 + A − base+C | +58.9 | +98.3 | +97.9 | -96.0 | -26.2 | -54.0 | +79.0 | -22.2 | -33.3 | -3.8 | +0.0 |

## 1. Gemma 4 E2B-it Q3_K_M + s4 LoRA (final iter 1000 and ckpt800)

### full

| variant (full) | exact refusal (refusal-target) n=331 | over-refusal answerable n=421 | over-refusal ans+speak_as n=491 | key facts ans+speak_as n=491 | key facts among answered | 1st person speak_as (lower better) n=132 | preaching refused n=88 | preaching fact n=88 | preach markers, preaching rows n=88 | preach markers, all rows n=910 | nature/decode, all rows n=910 | avg words ans+speak_as |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Gemma base + C | 59.5 (197) | 0.2 (1) | 0.8 (4) | 98.6 (484) | 99.4 (484/487) | 6.1 (8) | 23.9 (21) | 43.2 (38) | 0.0 (0) | 0.2 (2) | 0.0 (0) | 12.7 |
| Gemma LoRA final + B | 99.4 (329) | 1.7 (7) | 2.0 (10) | 97.1 (477) | 99.2 (477/481) | 0.0 (0) | 100.0 (88) | 0.0 (0) | 0.0 (0) | 0.0 (0) | 0.0 (0) | 9.2 |
| Gemma LoRA final + C | 99.7 (330) | 1.9 (8) | 2.6 (13) | 96.7 (475) | 99.4 (475/478) | 0.0 (0) | 100.0 (88) | 0.0 (0) | 0.0 (0) | 0.0 (0) | 0.0 (0) | 9.2 |
| Gemma LoRA final + A | 100.0 (331) | 100.0 (421) | 100.0 (491) | 0.0 (0) | — | 0.0 (0) | 100.0 (88) | 0.0 (0) | 0.0 (0) | 0.0 (0) | 0.0 (0) | 11.0 |
| Gemma LoRA ckpt800 + B | 99.7 (330) | 1.9 (8) | 4.1 (20) | 94.1 (462) | 98.1 (462/471) | 0.0 (0) | 100.0 (88) | 0.0 (0) | 0.0 (0) | 0.0 (0) | 0.0 (0) | 11.1 |
| Gemma LoRA ckpt800 + C | 99.4 (329) | 2.1 (9) | 6.1 (30) | 92.1 (452) | 98.0 (452/461) | 0.0 (0) | 100.0 (88) | 0.0 (0) | 0.0 (0) | 0.0 (0) | 0.0 (0) | 10.2 |
| Gemma LoRA ckpt800 + A | 100.0 (331) | 100.0 (421) | 100.0 (491) | 0.0 (0) | — | 0.0 (0) | 100.0 (88) | 0.0 (0) | 0.0 (0) | 0.0 (0) | 0.0 (0) | 11.0 |
| MLX bf16 Gemma LoRA final + B | 99.4 (329) | 1.4 (6) | 2.0 (10) | 97.6 (479) | 99.6 (479/481) | 0.0 (0) | 100.0 (88) | 0.0 (0) | 0.0 (0) | 0.0 (0) | 0.0 (0) | 9.3 |
| MLX bf16 Gemma LoRA ckpt800 + B | 99.4 (329) | 2.6 (11) | 5.7 (28) | 92.3 (453) | 97.8 (453/463) | 0.0 (0) | 100.0 (88) | 0.0 (0) | 0.0 (0) | 0.0 (0) | 0.0 (0) | 11.5 |
| MLX bf16 Gemma base + B | 0.0 (0) | 0.0 (0) | 0.0 (0) | 100.0 (491) | 100.0 (491/491) | 68.9 (91) | 0.0 (0) | 73.9 (65) | 77.3 (68) | 49.5 (450) | 1.1 (10) | 139.4 |

### full_minus_selection

| variant (full_minus_selection) | exact refusal (refusal-target) n=302 | over-refusal answerable n=361 | over-refusal ans+speak_as n=427 | key facts ans+speak_as n=427 | key facts among answered | 1st person speak_as (lower better) n=124 | preaching refused n=81 | preaching fact n=81 | preach markers, preaching rows n=81 | preach markers, all rows n=810 | nature/decode, all rows n=810 | avg words ans+speak_as |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Gemma base + C | 59.6 (180) | 0.3 (1) | 0.9 (4) | 98.4 (420) | 99.3 (420/423) | 6.5 (8) | 24.7 (20) | 42.0 (34) | 0.0 (0) | 0.1 (1) | 0.0 (0) | 12.6 |
| Gemma LoRA final + B | 99.7 (301) | 1.9 (7) | 2.3 (10) | 97.2 (415) | 99.5 (415/417) | 0.0 (0) | 100.0 (81) | 0.0 (0) | 0.0 (0) | 0.0 (0) | 0.0 (0) | 9.3 |
| Gemma LoRA final + C | 99.7 (301) | 2.2 (8) | 3.0 (13) | 96.7 (413) | 99.8 (413/414) | 0.0 (0) | 100.0 (81) | 0.0 (0) | 0.0 (0) | 0.0 (0) | 0.0 (0) | 9.3 |
| Gemma LoRA final + A | 100.0 (302) | 100.0 (361) | 100.0 (427) | 0.0 (0) | — | 0.0 (0) | 100.0 (81) | 0.0 (0) | 0.0 (0) | 0.0 (0) | 0.0 (0) | 11.0 |
| Gemma LoRA ckpt800 + B | 99.7 (301) | 2.2 (8) | 4.7 (20) | 93.4 (399) | 98.0 (399/407) | 0.0 (0) | 100.0 (81) | 0.0 (0) | 0.0 (0) | 0.0 (0) | 0.0 (0) | 11.3 |
| Gemma LoRA ckpt800 + C | 99.3 (300) | 2.5 (9) | 7.0 (30) | 91.3 (390) | 98.2 (390/397) | 0.0 (0) | 100.0 (81) | 0.0 (0) | 0.0 (0) | 0.0 (0) | 0.0 (0) | 9.8 |
| Gemma LoRA ckpt800 + A | 100.0 (302) | 100.0 (361) | 100.0 (427) | 0.0 (0) | — | 0.0 (0) | 100.0 (81) | 0.0 (0) | 0.0 (0) | 0.0 (0) | 0.0 (0) | 11.0 |
| MLX bf16 Gemma LoRA final + B | 99.7 (301) | 1.7 (6) | 2.3 (10) | 97.7 (417) | 100.0 (417/417) | 0.0 (0) | 100.0 (81) | 0.0 (0) | 0.0 (0) | 0.0 (0) | 0.0 (0) | 9.3 |
| MLX bf16 Gemma LoRA ckpt800 + B | 99.3 (300) | 3.0 (11) | 6.6 (28) | 91.3 (390) | 97.7 (390/399) | 0.0 (0) | 100.0 (81) | 0.0 (0) | 0.0 (0) | 0.0 (0) | 0.0 (0) | 11.7 |
| MLX bf16 Gemma base + B | 0.0 (0) | 0.0 (0) | 0.0 (0) | 100.0 (427) | 100.0 (427/427) | 70.2 (87) | 0.0 (0) | 72.8 (59) | 77.8 (63) | 48.6 (394) | 1.2 (10) | 139.6 |

### eval_only

| variant (eval_only) | exact refusal (refusal-target) n=177 | over-refusal answerable n=230 | over-refusal ans+speak_as n=270 | key facts ans+speak_as n=270 | key facts among answered | 1st person speak_as (lower better) n=72 | preaching refused n=48 | preaching fact n=48 | preach markers, preaching rows n=48 | preach markers, all rows n=495 | nature/decode, all rows n=495 | avg words ans+speak_as |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Gemma base + C | 63.3 (112) | 0.4 (1) | 0.4 (1) | 99.3 (268) | 99.6 (268/269) | 0.0 (0) | 10.4 (5) | 60.4 (29) | 0.0 (0) | 0.4 (2) | 0.0 (0) | 12.6 |
| Gemma LoRA final + B | 99.4 (176) | 3.0 (7) | 2.6 (7) | 96.3 (260) | 98.9 (260/263) | 0.0 (0) | 100.0 (48) | 0.0 (0) | 0.0 (0) | 0.0 (0) | 0.0 (0) | 9.2 |
| Gemma LoRA final + C | 100.0 (177) | 3.5 (8) | 3.0 (8) | 95.9 (259) | 98.9 (259/262) | 0.0 (0) | 100.0 (48) | 0.0 (0) | 0.0 (0) | 0.0 (0) | 0.0 (0) | 9.2 |
| Gemma LoRA final + A | 100.0 (177) | 100.0 (230) | 100.0 (270) | 0.0 (0) | — | 0.0 (0) | 100.0 (48) | 0.0 (0) | 0.0 (0) | 0.0 (0) | 0.0 (0) | 11.0 |
| Gemma LoRA ckpt800 + B | 100.0 (177) | 3.5 (8) | 3.3 (9) | 94.4 (255) | 97.7 (255/261) | 0.0 (0) | 100.0 (48) | 0.0 (0) | 0.0 (0) | 0.0 (0) | 0.0 (0) | 10.6 |
| Gemma LoRA ckpt800 + C | 99.4 (176) | 3.9 (9) | 3.7 (10) | 93.7 (253) | 97.3 (253/260) | 0.0 (0) | 100.0 (48) | 0.0 (0) | 0.0 (0) | 0.0 (0) | 0.0 (0) | 10.7 |
| Gemma LoRA ckpt800 + A | 100.0 (177) | 100.0 (230) | 100.0 (270) | 0.0 (0) | — | 0.0 (0) | 100.0 (48) | 0.0 (0) | 0.0 (0) | 0.0 (0) | 0.0 (0) | 11.0 |
| MLX bf16 Gemma LoRA final + B | 99.4 (176) | 2.6 (6) | 2.2 (6) | 97.0 (262) | 99.2 (262/264) | 0.0 (0) | 100.0 (48) | 0.0 (0) | 0.0 (0) | 0.0 (0) | 0.0 (0) | 9.3 |
| MLX bf16 Gemma LoRA ckpt800 + B | 99.4 (176) | 4.8 (11) | 4.4 (12) | 92.6 (250) | 96.9 (250/258) | 0.0 (0) | 100.0 (48) | 0.0 (0) | 0.0 (0) | 0.0 (0) | 0.0 (0) | 11.2 |
| MLX bf16 Gemma base + B | 0.0 (0) | 0.0 (0) | 0.0 (0) | 100.0 (270) | 100.0 (270/270) | 68.1 (49) | 0.0 (0) | 79.2 (38) | 79.2 (38) | 45.9 (227) | 1.2 (6) | 138.0 |

### eval_minus_selection

| variant (eval_minus_selection) | exact refusal (refusal-target) n=148 | over-refusal answerable n=170 | over-refusal ans+speak_as n=206 | key facts ans+speak_as n=206 | key facts among answered | 1st person speak_as (lower better) n=64 | preaching refused n=41 | preaching fact n=41 | preach markers, preaching rows n=41 | preach markers, all rows n=395 | nature/decode, all rows n=395 | avg words ans+speak_as |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Gemma base + C | 64.2 (95) | 0.6 (1) | 0.5 (1) | 99.0 (204) | 99.5 (204/205) | 0.0 (0) | 9.8 (4) | 61.0 (25) | 0.0 (0) | 0.3 (1) | 0.0 (0) | 12.3 |
| Gemma LoRA final + B | 100.0 (148) | 4.1 (7) | 3.4 (7) | 96.1 (198) | 99.5 (198/199) | 0.0 (0) | 100.0 (41) | 0.0 (0) | 0.0 (0) | 0.0 (0) | 0.0 (0) | 9.2 |
| Gemma LoRA final + C | 100.0 (148) | 4.7 (8) | 3.9 (8) | 95.6 (197) | 99.5 (197/198) | 0.0 (0) | 100.0 (41) | 0.0 (0) | 0.0 (0) | 0.0 (0) | 0.0 (0) | 9.3 |
| Gemma LoRA final + A | 100.0 (148) | 100.0 (170) | 100.0 (206) | 0.0 (0) | — | 0.0 (0) | 100.0 (41) | 0.0 (0) | 0.0 (0) | 0.0 (0) | 0.0 (0) | 11.0 |
| Gemma LoRA ckpt800 + B | 100.0 (148) | 4.7 (8) | 4.4 (9) | 93.2 (192) | 97.5 (192/197) | 0.0 (0) | 100.0 (41) | 0.0 (0) | 0.0 (0) | 0.0 (0) | 0.0 (0) | 10.7 |
| Gemma LoRA ckpt800 + C | 99.3 (147) | 5.3 (9) | 4.9 (10) | 92.7 (191) | 97.4 (191/196) | 0.0 (0) | 100.0 (41) | 0.0 (0) | 0.0 (0) | 0.0 (0) | 0.0 (0) | 9.9 |
| Gemma LoRA ckpt800 + A | 100.0 (148) | 100.0 (170) | 100.0 (206) | 0.0 (0) | — | 0.0 (0) | 100.0 (41) | 0.0 (0) | 0.0 (0) | 0.0 (0) | 0.0 (0) | 11.0 |
| MLX bf16 Gemma LoRA final + B | 100.0 (148) | 3.5 (6) | 2.9 (6) | 97.1 (200) | 100.0 (200/200) | 0.0 (0) | 100.0 (41) | 0.0 (0) | 0.0 (0) | 0.0 (0) | 0.0 (0) | 9.3 |
| MLX bf16 Gemma LoRA ckpt800 + B | 99.3 (147) | 6.5 (11) | 5.8 (12) | 90.8 (187) | 96.4 (187/194) | 0.0 (0) | 100.0 (41) | 0.0 (0) | 0.0 (0) | 0.0 (0) | 0.0 (0) | 11.6 |
| MLX bf16 Gemma base + B | 0.0 (0) | 0.0 (0) | 0.0 (0) | 100.0 (206) | 100.0 (206/206) | 70.3 (45) | 0.0 (0) | 78.0 (32) | 80.5 (33) | 43.3 (171) | 1.5 (6) | 138.1 |

Differences vs Gemma base + C (full, percentage points):

| variant | exact refusal (refusal-target) | over-refusal answerable | over-refusal ans+speak_as | key facts ans+speak_as | key facts among answered | 1st person speak_as (lower better) | preaching refused | preaching fact | preach markers, preaching rows | preach markers, all rows | nature/decode, all rows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Gemma LoRA final + B − base+C | +39.9 | +1.4 | +1.2 | -1.4 | -0.2 | -6.1 | +76.1 | -43.2 | +0.0 | -0.2 | +0.0 |
| Gemma LoRA final + C − base+C | +40.2 | +1.7 | +1.8 | -1.8 | -0.0 | -6.1 | +76.1 | -43.2 | +0.0 | -0.2 | +0.0 |
| Gemma LoRA final + A − base+C | +40.5 | +99.8 | +99.2 | -98.6 | — | -6.1 | +76.1 | -43.2 | +0.0 | -0.2 | +0.0 |
| Gemma LoRA ckpt800 + B − base+C | +40.2 | +1.7 | +3.3 | -4.5 | -1.3 | -6.1 | +76.1 | -43.2 | +0.0 | -0.2 | +0.0 |
| Gemma LoRA ckpt800 + C − base+C | +39.9 | +1.9 | +5.3 | -6.5 | -1.3 | -6.1 | +76.1 | -43.2 | +0.0 | -0.2 | +0.0 |
| Gemma LoRA ckpt800 + A − base+C | +40.5 | +99.8 | +99.2 | -98.6 | — | -6.1 | +76.1 | -43.2 | +0.0 | -0.2 | +0.0 |

Differences vs Gemma base + C (full_minus_selection, percentage points):

| variant | exact refusal (refusal-target) | over-refusal answerable | over-refusal ans+speak_as | key facts ans+speak_as | key facts among answered | 1st person speak_as (lower better) | preaching refused | preaching fact | preach markers, preaching rows | preach markers, all rows | nature/decode, all rows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Gemma LoRA final + B − base+C | +40.1 | +1.7 | +1.4 | -1.2 | +0.2 | -6.5 | +75.3 | -42.0 | +0.0 | -0.1 | +0.0 |
| Gemma LoRA final + C − base+C | +40.1 | +1.9 | +2.1 | -1.6 | +0.5 | -6.5 | +75.3 | -42.0 | +0.0 | -0.1 | +0.0 |
| Gemma LoRA final + A − base+C | +40.4 | +99.7 | +99.1 | -98.4 | — | -6.5 | +75.3 | -42.0 | +0.0 | -0.1 | +0.0 |
| Gemma LoRA ckpt800 + B − base+C | +40.1 | +1.9 | +3.7 | -4.9 | -1.3 | -6.5 | +75.3 | -42.0 | +0.0 | -0.1 | +0.0 |
| Gemma LoRA ckpt800 + C − base+C | +39.7 | +2.2 | +6.1 | -7.0 | -1.1 | -6.5 | +75.3 | -42.0 | +0.0 | -0.1 | +0.0 |
| Gemma LoRA ckpt800 + A − base+C | +40.4 | +99.7 | +99.1 | -98.4 | — | -6.5 | +75.3 | -42.0 | +0.0 | -0.1 | +0.0 |

## 2. Qwen vs Gemma (llama.cpp; full / full minus selection)

| setting | model | exact refusal | over-refusal answerable | over-refusal ans+speak_as | key facts ans+speak_as | 1st person | preaching refused | preaching fact | preach markers all | nature/decode all |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| base + C | Qwen base + C | 41.1 / 40.7 | 0.2 / 0.3 | 0.4 / 0.5 | 97.1 / 97.2 | 54.5 / 55.6 | 20.5 / 21.0 | 21.6 / 22.2 | 3.8 / 3.8 | 0.0 / 0.0 |
| base + C | Gemma base + C | 59.5 / 59.6 | 0.2 / 0.3 | 0.8 / 0.9 | 98.6 / 98.4 | 6.1 / 6.5 | 23.9 / 24.7 | 43.2 / 42.0 | 0.2 / 0.1 | 0.0 / 0.0 |
| LoRA + B | Qwen LoRA ckpt600 + B | 93.7 / 93.7 | 0.2 / 0.3 | 0.4 / 0.5 | 97.6 / 97.7 | 0.0 / 0.0 | 86.4 / 87.7 | 12.5 / 11.1 | 0.0 / 0.0 | 0.0 / 0.0 |
| LoRA + B | Gemma LoRA final + B | 99.4 / 99.7 | 1.7 / 1.9 | 2.0 / 2.3 | 97.1 / 97.2 | 0.0 / 0.0 | 100.0 / 100.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 |
| LoRA + B | Gemma LoRA ckpt800 + B | 99.7 / 99.7 | 1.9 / 2.2 | 4.1 / 4.7 | 94.1 / 93.4 | 0.0 / 0.0 | 100.0 / 100.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 |
| LoRA + C | Qwen LoRA ckpt600 + C | 94.3 / 94.4 | 0.5 / 0.6 | 0.6 / 0.7 | 97.4 / 97.4 | 0.0 / 0.0 | 88.6 / 90.1 | 11.4 / 9.9 | 0.0 / 0.0 | 0.0 / 0.0 |
| LoRA + C | Gemma LoRA final + C | 99.7 / 99.7 | 1.9 / 2.2 | 2.6 / 3.0 | 96.7 / 96.7 | 0.0 / 0.0 | 100.0 / 100.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 |
| LoRA + C | Gemma LoRA ckpt800 + C | 99.4 / 99.3 | 2.1 / 2.5 | 6.1 / 7.0 | 92.1 / 91.3 | 0.0 / 0.0 | 100.0 / 100.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 |
| LoRA + A | Qwen LoRA ckpt600 + A | 99.7 / 99.7 | 98.8 / 98.6 | 98.6 / 98.4 | 1.0 / 1.2 | 1.5 / 1.6 | 100.0 / 100.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 |
| LoRA + A | Gemma LoRA final + A | 100.0 / 100.0 | 100.0 / 100.0 | 100.0 / 100.0 | 0.0 / 0.0 | 0.0 / 0.0 | 100.0 / 100.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 |
| LoRA + A | Gemma LoRA ckpt800 + A | 100.0 / 100.0 | 100.0 / 100.0 | 100.0 / 100.0 | 0.0 / 0.0 | 0.0 / 0.0 | 100.0 / 100.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 |

## 3. llama.cpp vs MLX (same LoRA, condition B — the prompt the MLX runs used)

| pair | subset | same outcome category | same refused-vs-answered | identical text | top disagreements |
|---|---|---:|---:|---:|---|
| qwen_lora_B vs mlx_qwen_lora_B | full | 96.0% (874/910) | 96.6% | 85.8% (781) | no_refusal (llama.cpp) vs exact_refusal (MLX): 21; answered_keyfacts (llama.cpp) vs refused (MLX): 7; answered_no_keyfacts (llama.cpp) vs answered_keyfacts (MLX): 3; answered_no_keyfacts (llama.cpp) vs refused (MLX): 2 |
| qwen_lora_B vs mlx_qwen_lora_B | full_minus_selection | 96.3% (780/810) | 96.7% | 87.0% (705) | no_refusal (llama.cpp) vs exact_refusal (MLX): 19; answered_keyfacts (llama.cpp) vs refused (MLX): 5; answered_no_keyfacts (llama.cpp) vs refused (MLX): 2; answered_no_keyfacts (llama.cpp) vs answered_keyfacts (MLX): 2 |
| gemma_final_B vs mlx_gemma_final_B | full | 99.3% (904/910) | 99.6% | 96.9% (882) | answered_keyfacts (llama.cpp) vs refused (MLX): 2; answered_no_keyfacts (llama.cpp) vs answered_keyfacts (MLX): 2; refused (llama.cpp) vs answered_keyfacts (MLX): 2 |
| gemma_final_B vs mlx_gemma_final_B | full_minus_selection | 99.3% (804/810) | 99.5% | 96.9% (785) | answered_keyfacts (llama.cpp) vs refused (MLX): 2; answered_no_keyfacts (llama.cpp) vs answered_keyfacts (MLX): 2; refused (llama.cpp) vs answered_keyfacts (MLX): 2 |
| gemma_800_B vs mlx_gemma_800_B | full | 97.8% (890/910) | 98.8% | 92.4% (841) | answered_keyfacts (llama.cpp) vs refused (MLX): 9; answered_keyfacts (llama.cpp) vs answered_no_keyfacts (MLX): 5; answered_no_keyfacts (llama.cpp) vs answered_keyfacts (MLX): 4; refused (llama.cpp) vs answered_keyfacts (MLX): 1 |
| gemma_800_B vs mlx_gemma_800_B | full_minus_selection | 97.8% (792/810) | 98.6% | 93.0% (753) | answered_keyfacts (llama.cpp) vs refused (MLX): 9; answered_keyfacts (llama.cpp) vs answered_no_keyfacts (MLX): 4; answered_no_keyfacts (llama.cpp) vs answered_keyfacts (MLX): 3; refused (llama.cpp) vs answered_keyfacts (MLX): 1 |

Outcome categories: refusal-target rows → exact_refusal / inexact_refusal / no_refusal; other rows → refused / answered_keyfacts / answered_no_keyfacts.

Per-metric difference, llama.cpp minus MLX (full set, percentage points):

| pair | exact refusal (refusal-target) | over-refusal answerable | over-refusal ans+speak_as | key facts ans+speak_as | key facts among answered | 1st person speak_as (lower better) | preaching refused | preaching fact | preach markers, preaching rows | preach markers, all rows | nature/decode, all rows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen LoRA ckpt600 + B − MLX | -6.3 | -0.5 | -0.4 | +0.0 | -0.4 | +0.0 | -6.8 | +5.7 | +0.0 | +0.0 | +0.0 |
| Gemma LoRA final + B − MLX | +0.0 | +0.2 | +0.0 | -0.4 | -0.4 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 |
| Gemma LoRA ckpt800 + B − MLX | +0.3 | -0.7 | -1.6 | +1.8 | +0.2 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 |

## 4. Settings, paths, quants, runtimes

| run | rows | gen seconds | server load s | generated tokens | max prompt tokens | stopped at 256 | adapter (server /lora-adapters) |
|---|---:|---:|---:|---:|---:|---:|---|
| smoke_gemma_800_B | 5 | 1.8 | 1.5 | 66 | 235 | 0 | gemma4-e2b-constraint-s4-lora-ckpt800-f16.gguf scale 1.0 |
| smoke_gemma_base_B | 5 | 2.1 | 2.0 | 127 | 235 | 0 | none |
| smoke_gemma_final_B | 5 | 1.9 | 1.5 | 66 | 235 | 0 | gemma4-e2b-constraint-s4-lora-f16.gguf scale 1.0 |
| smoke_qwen_base_B | 5 | 1.0 | 1.0 | 110 | 238 | 0 | none |
| smoke_qwen_lora_B | 5 | 1.1 | 1.0 | 67 | 238 | 0 | qwen25-1.5b-constraint-s4-lora-ckpt600-f16.gguf scale 1.0 |
| gemma_800_A | 910 | 238.6 | 1.5 | 11830 | 59 | 0 | gemma4-e2b-constraint-s4-lora-ckpt800-f16.gguf scale 1.0 |
| gemma_800_B | 910 | 339.0 | 1.5 | 12451 | 257 | 2 | gemma4-e2b-constraint-s4-lora-ckpt800-f16.gguf scale 1.0 |
| gemma_800_C | 910 | 421.9 | 1.5 | 12033 | 411 | 1 | gemma4-e2b-constraint-s4-lora-ckpt800-f16.gguf scale 1.0 |
| gemma_base_C | 910 | 360.7 | 1.0 | 15121 | 411 | 0 | none |
| gemma_final_A | 910 | 238.3 | 1.0 | 11830 | 59 | 0 | gemma4-e2b-constraint-s4-lora-f16.gguf scale 1.0 |
| gemma_final_B | 910 | 331.9 | 1.5 | 11486 | 257 | 0 | gemma4-e2b-constraint-s4-lora-f16.gguf scale 1.0 |
| gemma_final_C | 910 | 414.4 | 1.5 | 11489 | 411 | 0 | gemma4-e2b-constraint-s4-lora-f16.gguf scale 1.0 |
| qwen_base_C | 910 | 220.7 | 0.5 | 19171 | 398 | 0 | none |
| qwen_lora_A | 910 | 119.8 | 0.5 | 11814 | 59 | 0 | qwen25-1.5b-constraint-s4-lora-ckpt600-f16.gguf scale 1.0 |
| qwen_lora_B | 910 | 189.1 | 0.5 | 11883 | 252 | 0 | qwen25-1.5b-constraint-s4-lora-ckpt600-f16.gguf scale 1.0 |
| qwen_lora_C | 910 | 239.9 | 1.0 | 11873 | 398 | 0 | qwen25-1.5b-constraint-s4-lora-ckpt600-f16.gguf scale 1.0 |

### Setup

- **Engine**: llama.cpp b10481 (`~/llama.cpp-b10481`, commit 25ae3a9b), Metal build, `llama-server` with `-c 2048 -b 2048 -np 1 -ngl 99 --seed 42`, raw prompt to `/completion`, `cache_prompt=false` (each row starts from an empty KV cache). Mac14,6 (M2 Max MacBook Pro).
- **Models (app base GGUFs)**: `~/models/app_gguf/qwen2.5-1.5b-instruct-q4_k_m.gguf` (Q4_K_M, sha256 6a1a2eb6…407e = `knownQwenSha256` in lib/model_integrity.dart) and `~/models/app_gguf/gemma-4-E2B-it-Q3_K_M.gguf` (Q3_K_M, 2,536,786,016 bytes, sha256 086e2f5b…3085 = the LFS oid of `unsloth/gemma-4-E2B-it-GGUF/gemma-4-E2B-it-Q3_K_M.gguf` on Hugging Face). The app's run history records exactly this file name / quant (`gemma-4-E2B-it-Q3_K_M.gguf`, Q3_K_M) but no hash, and no repo doc names the source, so the Gemma file is the exact-name match from unsloth (bartowski's Q3_K_M has a different name and size). Not proven byte-identical to the app's copy.
- **Adapters** (`lora/out/`): `qwen25-1.5b-constraint-s4-lora-ckpt600-f16.gguf`, `gemma4-e2b-constraint-s4-lora-f16.gguf` (final, iter 1000), `gemma4-e2b-constraint-s4-lora-ckpt800-f16.gguf`; `--lora-scaled <path>:1`. The server's `/lora-adapters` reports each with scale 1.0 (table above).
- **How the app applies a LoRA**: `packages/llm_llamacpp` (0.7.0) `inference_isolate.dart` calls `llama_adapter_lora_init(model, loraPath)` after loading the model, then `llama_set_adapters_lora(ctx, [adapter], 1, [loraScale])` (`lora_context_adapters.dart`); `LlamaCppChatRepository` defaults `loraScale = 1.0`. `llama-server --lora-scaled` goes through the same `llama_adapter_lora_init` / `llama_set_adapters_lora` path. Note: the app UI (`lib/`) currently never passes a `loraPath`, so "the app with a LoRA" means the package API at scale 1.
- **Decoding** = app `_configureSampler` chain: temp 0 → top_k 40 → top_p 0.9 (min_keep 1) → penalties(last_n 64, repeat 1.1, freq 0, presence 0) → dist(seed 42); max 256 new tokens, ctx 2048, batch 2048 (lib/main.dart). No min-p/DRY/XTC/typical.
- **Prompts**: conditions A/B/C rebuilt exactly as `lib/messages.dart buildConditionMessages` (A: persona system + question; B: persona + notes user turn (`localNotesMessage`) + question — the same `condition_b_app_messages` the MLX runs used; C: `buildConditionCSystemPrompt(persona)` with the refusal string, notes turn or the '(No specific local notes match this query.)' block, question). Persona = `assets/system_prompt.txt`; refusal = `standardRefusalStatement`. Formatting: Qwen ChatML (what the native template gives), Gemma `formatGemma4` (`<bos><|turn>role\n…<turn|>\n…<|turn>model\n`, thinking off). `template_check.json`: app formatter = HF chat template for all 910 rows × 3 conditions × 2 models. For Gemma the literal `<bos>` is dropped from the text and the server adds BOS; token ids checked identical to the app's (text with `<bos>`, no add_special) on row 1 of every run (`bos_check` in meta).
- **Stops**: app `stopSequences` (`<|im_end|>`, `<end_of_turn>`, `<|eot_id|>`, `<turn|>`, `<|end|>`, `<|im_start|>`, `<start_of_turn>`), Gemma additionally `<eos>`; plus EOG tokens. Output = `sanitizeModelOutput` (Gemma: keepThoughts as main.dart).
- **Comparison set**: `lora/data/comparison/comparison.jsonl` (910 rows: 421 answerable, 269 refusal, 132 speak_as (62 refusal-target), 88 preaching). Selection rows = the 100 ids in `results/s4/subset_ids.json` (all eval-only; 60 answerable, 25 refusal, 8 speak_as, 7 preaching), used for the Qwen and Gemma checkpoint picks.

### Caveats for llama.cpp vs MLX

- Quantization: llama.cpp runs use Q4_K_M (Qwen) / Q3_K_M (Gemma) base weights with f16 adapters; MLX runs were bf16.
- Repetition penalty: in the app chain temp 0 comes first and keeps only the argmax token, so the 1.1 repeat penalty that follows cannot change the pick (pure greedy). llama-server uses the same order, so it reproduces this. The MLX runs applied the 1.1 repetition penalty as a logits processor *before* greedy argmax, so there it did act. This alone can make outputs differ.
- MLX outputs were only `.strip()`ed; llama.cpp outputs pass the app's `sanitizeModelOutput`.
- MLX Gemma base outputs (`output_without_adapter`, both CSVs) contain a `<|channel>thought` block in 910/910 rows, which inflates their length and key-fact hits; the llama.cpp Gemma base (app formatter, no thinking prefix) did not produce thought blocks in the smoke test. Base+C here is not compared with MLX (MLX base was condition B).

