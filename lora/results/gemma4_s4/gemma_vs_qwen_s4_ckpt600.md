# Row-level error overlap: Gemma 4 E2B s4 vs Qwen s4 iter 600 (910-row comparison set)

Script `overlap_gemma_vs_qwen.py`; predicates from `results/s4/analyze_s4.py`. Expected-by-chance = rate_G × rate_Q × N (independence). Jaccard = both / (both + only-G + only-Q). κ = Cohen's kappa on the binary error indicator over the row universe N. Preaching is reported separately: refusing preaching prompts is the trained v2 behaviour, so 'answered' there is a deviation from training, not a factual error.


## gemma s4 iter 800 vs qwen s4 iter 600 — all

| error | family | N | both | Gemma only | Qwen only | Gemma rate | Qwen rate | expected both (chance) | both/expected | Jaccard | κ | Gemma base / Qwen base errors |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| over_refusal | refusal | 491 | 1 | 27 | 3 | 5.7% | 0.8% | 0.23 | 4.4 | 0.03 | 0.05 | 0 / 0 |
| missed_refusal | refusal | 331 | 0 | 2 | 0 | 0.6% | 0.0% | 0.00 | — | 0.00 | 0.00 | 331 / 331 |
| inexact_refusal | refusal | 331 | 0 | 0 | 0 | 0.0% | 0.0% | 0.00 | — | — | — | 0 / 0 |
| fact_miss | facts | 491 | 3 | 7 | 5 | 2.0% | 1.6% | 0.16 | 18.4 | 0.20 | 0.32 | 0 / 27 |
| answer_fail | facts | 491 | 7 | 31 | 5 | 7.7% | 2.4% | 0.93 | 7.5 | 0.16 | 0.25 | 0 / 27 |
| voice_first_person | voice | 132 | 0 | 0 | 0 | 0.0% | 0.0% | 0.00 | — | — | — | 91 / 87 |
| nature_decode | voice | 910 | 0 | 0 | 0 | 0.0% | 0.0% | 0.00 | — | — | — | 10 / 6 |
| preaching_answered_not_refused | preaching (separate) | 88 | 0 | 0 | 6 | 0.0% | 6.8% | 0.00 | — | 0.00 | -0.00 | 88 / 88 |
| preach_markers_any_row | preaching (separate) | 910 | 0 | 0 | 0 | 0.0% | 0.0% | 0.00 | — | — | — | 450 / 98 |

## gemma s4 iter 800 vs qwen s4 iter 600 — eval-only

| error | family | N | both | Gemma only | Qwen only | Gemma rate | Qwen rate | expected both (chance) | both/expected | Jaccard | κ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| over_refusal | refusal | 270 | 1 | 11 | 3 | 4.4% | 1.5% | 0.18 | 5.6 | 0.07 | 0.11 |
| missed_refusal | refusal | 177 | 0 | 1 | 0 | 0.6% | 0.0% | 0.00 | — | 0.00 | 0.00 |
| inexact_refusal | refusal | 177 | 0 | 0 | 0 | 0.0% | 0.0% | 0.00 | — | — | — |
| fact_miss | facts | 270 | 3 | 5 | 3 | 3.0% | 2.2% | 0.18 | 16.9 | 0.27 | 0.41 |
| answer_fail | facts | 270 | 6 | 14 | 4 | 7.4% | 3.7% | 0.74 | 8.1 | 0.25 | 0.37 |
| voice_first_person | voice | 72 | 0 | 0 | 0 | 0.0% | 0.0% | 0.00 | — | — | — |
| nature_decode | voice | 495 | 0 | 0 | 0 | 0.0% | 0.0% | 0.00 | — | — | — |
| preaching_answered_not_refused | preaching (separate) | 48 | 0 | 0 | 6 | 0.0% | 12.5% | 0.00 | — | 0.00 | 0.00 |
| preach_markers_any_row | preaching (separate) | 495 | 0 | 0 | 0 | 0.0% | 0.0% | 0.00 | — | — | — |

## gemma s4 iter 800 vs qwen s4 iter 600 — training-q

| error | family | N | both | Gemma only | Qwen only | Gemma rate | Qwen rate | expected both (chance) | both/expected | Jaccard | κ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| over_refusal | refusal | 221 | 0 | 16 | 0 | 7.2% | 0.0% | 0.00 | — | 0.00 | 0.00 |
| missed_refusal | refusal | 154 | 0 | 1 | 0 | 0.6% | 0.0% | 0.00 | — | 0.00 | 0.00 |
| inexact_refusal | refusal | 154 | 0 | 0 | 0 | 0.0% | 0.0% | 0.00 | — | — | — |
| fact_miss | facts | 221 | 0 | 2 | 2 | 0.9% | 0.9% | 0.02 | 0.0 | 0.00 | -0.01 |
| answer_fail | facts | 221 | 1 | 17 | 1 | 8.1% | 0.9% | 0.16 | 6.1 | 0.05 | 0.09 |
| voice_first_person | voice | 60 | 0 | 0 | 0 | 0.0% | 0.0% | 0.00 | — | — | — |
| nature_decode | voice | 415 | 0 | 0 | 0 | 0.0% | 0.0% | 0.00 | — | — | — |
| preaching_answered_not_refused | preaching (separate) | 40 | 0 | 0 | 0 | 0.0% | 0.0% | 0.00 | — | — | — |
| preach_markers_any_row | preaching (separate) | 415 | 0 | 0 | 0 | 0.0% | 0.0% | 0.00 | — | — | — |

## gemma s4 iter 600 vs qwen s4 iter 600 — all

| error | family | N | both | Gemma only | Qwen only | Gemma rate | Qwen rate | expected both (chance) | both/expected | Jaccard | κ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| over_refusal | refusal | 491 | 1 | 25 | 3 | 5.3% | 0.8% | 0.21 | 4.7 | 0.03 | 0.05 |
| missed_refusal | refusal | 331 | 0 | 1 | 0 | 0.3% | 0.0% | 0.00 | — | 0.00 | 0.00 |
| inexact_refusal | refusal | 331 | 0 | 0 | 0 | 0.0% | 0.0% | 0.00 | — | — | — |
| fact_miss | facts | 491 | 1 | 1 | 7 | 0.4% | 1.6% | 0.03 | 30.7 | 0.11 | 0.19 |
| answer_fail | facts | 491 | 7 | 21 | 5 | 5.7% | 2.4% | 0.68 | 10.2 | 0.21 | 0.33 |
| voice_first_person | voice | 132 | 0 | 0 | 0 | 0.0% | 0.0% | 0.00 | — | — | — |
| nature_decode | voice | 910 | 0 | 0 | 0 | 0.0% | 0.0% | 0.00 | — | — | — |
| preaching_answered_not_refused | preaching (separate) | 88 | 0 | 0 | 6 | 0.0% | 6.8% | 0.00 | — | 0.00 | -0.00 |
| preach_markers_any_row | preaching (separate) | 910 | 0 | 0 | 0 | 0.0% | 0.0% | 0.00 | — | — | — |

## gemma s4 iter 600 vs qwen s4 iter 600 — eval-only

| error | family | N | both | Gemma only | Qwen only | Gemma rate | Qwen rate | expected both (chance) | both/expected | Jaccard | κ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| over_refusal | refusal | 270 | 1 | 18 | 3 | 7.0% | 1.5% | 0.28 | 3.6 | 0.05 | 0.06 |
| missed_refusal | refusal | 177 | 0 | 0 | 0 | 0.0% | 0.0% | 0.00 | — | — | — |
| inexact_refusal | refusal | 177 | 0 | 0 | 0 | 0.0% | 0.0% | 0.00 | — | — | — |
| fact_miss | facts | 270 | 1 | 0 | 5 | 0.4% | 2.2% | 0.02 | 45.0 | 0.17 | 0.28 |
| answer_fail | facts | 270 | 6 | 14 | 4 | 7.4% | 3.7% | 0.74 | 8.1 | 0.25 | 0.37 |
| voice_first_person | voice | 72 | 0 | 0 | 0 | 0.0% | 0.0% | 0.00 | — | — | — |
| nature_decode | voice | 495 | 0 | 0 | 0 | 0.0% | 0.0% | 0.00 | — | — | — |
| preaching_answered_not_refused | preaching (separate) | 48 | 0 | 0 | 6 | 0.0% | 12.5% | 0.00 | — | 0.00 | 0.00 |
| preach_markers_any_row | preaching (separate) | 495 | 0 | 0 | 0 | 0.0% | 0.0% | 0.00 | — | — | — |

## gemma s4 iter 600 vs qwen s4 iter 600 — training-q

| error | family | N | both | Gemma only | Qwen only | Gemma rate | Qwen rate | expected both (chance) | both/expected | Jaccard | κ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| over_refusal | refusal | 221 | 0 | 7 | 0 | 3.2% | 0.0% | 0.00 | — | 0.00 | 0.00 |
| missed_refusal | refusal | 154 | 0 | 1 | 0 | 0.6% | 0.0% | 0.00 | — | 0.00 | 0.00 |
| inexact_refusal | refusal | 154 | 0 | 0 | 0 | 0.0% | 0.0% | 0.00 | — | — | — |
| fact_miss | facts | 221 | 0 | 1 | 2 | 0.5% | 0.9% | 0.01 | 0.0 | 0.00 | -0.01 |
| answer_fail | facts | 221 | 1 | 7 | 1 | 3.6% | 0.9% | 0.07 | 13.8 | 0.11 | 0.19 |
| voice_first_person | voice | 60 | 0 | 0 | 0 | 0.0% | 0.0% | 0.00 | — | — | — |
| nature_decode | voice | 415 | 0 | 0 | 0 | 0.0% | 0.0% | 0.00 | — | — | — |
| preaching_answered_not_refused | preaching (separate) | 40 | 0 | 0 | 0 | 0.0% | 0.0% | 0.00 | — | — | — |
| preach_markers_any_row | preaching (separate) | 415 | 0 | 0 | 0 | 0.0% | 0.0% | 0.00 | — | — | — |

## gemma s4 final (iter 1000) vs qwen s4 iter 600 — all

| error | family | N | both | Gemma only | Qwen only | Gemma rate | Qwen rate | expected both (chance) | both/expected | Jaccard | κ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| over_refusal | refusal | 491 | 1 | 9 | 3 | 2.0% | 0.8% | 0.08 | 12.3 | 0.08 | 0.13 |
| missed_refusal | refusal | 331 | 0 | 2 | 0 | 0.6% | 0.0% | 0.00 | — | 0.00 | 0.00 |
| inexact_refusal | refusal | 331 | 0 | 0 | 0 | 0.0% | 0.0% | 0.00 | — | — | — |
| fact_miss | facts | 491 | 1 | 1 | 7 | 0.4% | 1.6% | 0.03 | 30.7 | 0.11 | 0.19 |
| answer_fail | facts | 491 | 5 | 7 | 7 | 2.4% | 2.4% | 0.29 | 17.0 | 0.26 | 0.40 |
| voice_first_person | voice | 132 | 0 | 0 | 0 | 0.0% | 0.0% | 0.00 | — | — | — |
| nature_decode | voice | 910 | 0 | 0 | 0 | 0.0% | 0.0% | 0.00 | — | — | — |
| preaching_answered_not_refused | preaching (separate) | 88 | 0 | 0 | 6 | 0.0% | 6.8% | 0.00 | — | 0.00 | -0.00 |
| preach_markers_any_row | preaching (separate) | 910 | 0 | 0 | 0 | 0.0% | 0.0% | 0.00 | — | — | — |

## gemma s4 final (iter 1000) vs qwen s4 iter 600 — eval-only

| error | family | N | both | Gemma only | Qwen only | Gemma rate | Qwen rate | expected both (chance) | both/expected | Jaccard | κ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| over_refusal | refusal | 270 | 1 | 5 | 3 | 2.2% | 1.5% | 0.09 | 11.2 | 0.11 | 0.19 |
| missed_refusal | refusal | 177 | 0 | 1 | 0 | 0.6% | 0.0% | 0.00 | — | 0.00 | 0.00 |
| inexact_refusal | refusal | 177 | 0 | 0 | 0 | 0.0% | 0.0% | 0.00 | — | — | — |
| fact_miss | facts | 270 | 1 | 1 | 5 | 0.7% | 2.2% | 0.04 | 22.5 | 0.14 | 0.24 |
| answer_fail | facts | 270 | 4 | 4 | 6 | 3.0% | 3.7% | 0.30 | 13.5 | 0.29 | 0.43 |
| voice_first_person | voice | 72 | 0 | 0 | 0 | 0.0% | 0.0% | 0.00 | — | — | — |
| nature_decode | voice | 495 | 0 | 0 | 0 | 0.0% | 0.0% | 0.00 | — | — | — |
| preaching_answered_not_refused | preaching (separate) | 48 | 0 | 0 | 6 | 0.0% | 12.5% | 0.00 | — | 0.00 | 0.00 |
| preach_markers_any_row | preaching (separate) | 495 | 0 | 0 | 0 | 0.0% | 0.0% | 0.00 | — | — | — |

## gemma s4 final (iter 1000) vs qwen s4 iter 600 — training-q

| error | family | N | both | Gemma only | Qwen only | Gemma rate | Qwen rate | expected both (chance) | both/expected | Jaccard | κ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| over_refusal | refusal | 221 | 0 | 4 | 0 | 1.8% | 0.0% | 0.00 | — | 0.00 | 0.00 |
| missed_refusal | refusal | 154 | 0 | 1 | 0 | 0.6% | 0.0% | 0.00 | — | 0.00 | 0.00 |
| inexact_refusal | refusal | 154 | 0 | 0 | 0 | 0.0% | 0.0% | 0.00 | — | — | — |
| fact_miss | facts | 221 | 0 | 0 | 2 | 0.0% | 0.9% | 0.00 | — | 0.00 | 0.00 |
| answer_fail | facts | 221 | 1 | 3 | 1 | 1.8% | 0.9% | 0.04 | 27.6 | 0.20 | 0.33 |
| voice_first_person | voice | 60 | 0 | 0 | 0 | 0.0% | 0.0% | 0.00 | — | — | — |
| nature_decode | voice | 415 | 0 | 0 | 0 | 0.0% | 0.0% | 0.00 | — | — | — |
| preaching_answered_not_refused | preaching (separate) | 40 | 0 | 0 | 0 | 0.0% | 0.0% | 0.00 | — | — | — |
| preach_markers_any_row | preaching (separate) | 415 | 0 | 0 | 0 | 0.0% | 0.0% | 0.00 | — | — | — |
