# v2 LoRA over-refusal diagnostics (19 Oct 2026)

Everything lives in `results/diag/`. No repo-tracked files were changed. Scripts: `cheap_check.py`, `template_check.py`, `make_subset.py`, `run_variant.py`, `score.py`, `answered_audit.py`. Raw outputs: `out_*.jsonl` and `diag_outputs_all.json`. Metrics: `diag_results.json`. Temporary adapter dirs: `ad_ckpt500`, `ad_scale10` and `ad_ckpt800` (each is a config copy plus a copied safetensors file; delete them when done).

## 0. Cheap check (prompt and template parity)
- All 415 copied rows matched a v2 train/valid row by id. **355 are identical** (system text, notes turn, note order, question). **60 differ, all speak_as, and only in the question text.** v2 reworded the speak_as prompts (`stats.json: speak_as_reworded`), e.g. train "You're a stallion. How are the bands led?" vs comparison "Pretend to be a stallion and say how the bands are led." Everything else in those 60 rows is identical.
- **Target mismatch:** for all 40 copied preaching rows, v2 trains the **refusal string**, but the comparison reference is an old-style fact. So "wrong refusal" on preaching is the behaviour v2 asked for, not a bug.
- **Template:** mlx_lm's ChatDataset (mask_prompt) computes its offset with `apply_chat_template(messages[:-1], add_generation_prompt=True)`. That is token-identical to the inference prompt in run_comparison/run_variant on **415/415** rows, and it is a prefix of the full training sequence. Neither side adds a default system prompt: Qwen2.5 only inserts "You are Qwen…" when there is no system turn, and every row has the persona system turn.
- Conclusion: no prompt or template mismatch.

## 1–2. Subset and variants
100 eval-only rows, seed 42: 60 answerable, 25 refusal, 4 speak_as refusal-target, 4 speak_as answer-target, 7 preaching (`subset_ids.json`). Decoding: greedy, top_k 40, top_p 0.9, max 256 tokens. The penalty is 1.1 over the last 64 tokens (via `make_logits_processors`) unless marked "nopen". Rows d and base_pen come from the original CSV. **d_rerun** reruns d with the diag runner and reproduces the CSV exactly (100/100 identical), which confirms the runner is faithful.

| variant | exact refusal (refusal-target) | wrong refusal answerable | key facts answerable (all) | key facts answerable (answered) | wrong refusal speak_as-ans | wrong refusal preaching* | avg sent / words (answer-target) | 1st person speak_as | preach markers preaching | answered: grounded / wrong-note / other | run time |
|---|---|---|---|---|---|---|---|---|---|---|---|
| base_pen (orig CSV) | 0% | 0% | 92% | 92% of 60 | 0% | 0% | 1.73 / 30.3 | 3/8 | 5/7 | 4 / 1 / 66 | — |
| f_base_nopen | 0% | 0% | 93% | 93% of 60 | 0% | 0% | 1.32 / 19.6 | 3/8 | 2/7 | 1 / 1 / 69 | 43s |
| d_final_s20_pen (orig CSV) | 100% | 72% | 27% | 94% of 17 | 75% | 100% | 1.00 / 10.5 | 0/8 | 0/7 | 2 / 0 / 16 | — |
| d_rerun_final_s20_pen | 100% | 72% | 27% | 94% of 17 | 75% | 100% | 1.00 / 10.5 | 0/8 | 0/7 | 2 / 0 / 16 | 40s |
| a_final_s20_nopen | 97% | 67% | 27% | 80% of 20 | 75% | 100% | 1.00 / 10.0 | 0/8 | 0/7 | 4 / 0 / 17 | 39s |
| b_ckpt500_s20_pen | 83% | 35% | 42% | 64% of 39 | 75% | 86% | 1.00 / 9.2 | 0/8 | 0/7 | 2 / 1 / 38 | 39s |
| c_final_s10_pen | 72% | 63% | 28% | 77% of 22 | 75% | 100% | 1.00 / 11.1 | 0/8 | 0/7 | 5 / 2 / 16 | 41s |
| e_ckpt800_s20_nopen | 90% | 38% | 43% | 70% of 37 | 25% | 100% | 1.00 / 10.4 | 0/8 | 0/7 | 9 / 6 / 25 | 41s |

\*Preaching references are old-style facts, but v2 trains preaching → refusal (see section 0).
"answered: grounded / wrong-note / other" is a crude automatic split (exact copy of the grounded note / exact copy of another note / anything else). The audit below is more reliable.

### Answered-row audit (answer-target rows that were not refused; "correct" = key facts present)
| variant | answered | correct | wrong, verbatim copy of another note | wrong, non-verbatim (splice/garble) | refusal-target rows not refused |
|---|---|---|---|---|---|
| base_pen (orig CSV) | 71 | 61 | 1 | 9 | 29 |
| f_base_nopen | 71 | 62 | 1 | 8 | 29 |
| d_final_s20_pen (orig CSV) | 18 | 17 | 0 | 1 | 0 |
| d_rerun_final_s20_pen | 18 | 17 | 0 | 1 | 0 |
| a_final_s20_nopen | 21 | 17 | 1 | 3 | 1 |
| b_ckpt500_s20_pen | 41 | 26 | 9 | 6 | 5 |
| c_final_s10_pen | 23 | 18 | 3 | 2 | 8 |
| e_ckpt800_s20_nopen | 40 | 28 | 8 | 4 | 3 |

Correct answers out of 71 answer-target rows: base about 61–62, d 17, a 17, c 18, b 26, e 28.

### Examples
- **cmp-rd-shelter-take** "Where do deer take shelter when weather is hard?": d, a, b and c refuse. **e**: "In hard weather the deer use the Scots pine stands for shelter." (correct)
- **cmp-kk-hides-use** "Can visitors use hides to view the horses?": d refuses. a: "Visitors view the horses from the perimeter tracks." (wrong note). c: "Visitors can use hides to view the horses in the approved notes." (garbled). **e**: "Visitors view the horses from the hides." (correct)
- **cmp-ex-coat-colour** "What coat colour is given for the Exmoor ponies?": d: "Exmoor ponies have a brown coat, foals, and a dark eel stripe along the back." (splice). **a** (penalty off): "Exmoor ponies have a brown coat." b gives the wrong note; c and e refuse.
- **cmp-rd-speak-lookout** (speak_as answer-target) "In deer character, state which lookout the notes name.": d, a and b refuse. **e**: "Visitors may see the deer from the lookout at the Posbank."
- New failures in the weaker adapters: c "The lodge is built from 1902 to 1903." (made up), c "I do not have information about that in the approved notes in the approved notes.", b "Beavers use the narrow-cit and scaly.", e "The coat brown coat."

## 3. Additional evidence
- With the final adapter, the LoRA refuses **57/172 answerable rows from its own training split**, on the exact training prompts (and 20/27 train speak_as answer-target rows). It reproduces only 85/172 train answerable targets verbatim. Refusal-target rows are fitted at 100%. So the problem is already there on the training data, not only on new questions.
- Over-refusal does not depend on wording: answerable rows that mention the notes ("named", "listed", "mention") are refused at about the same rate as other rows. with_note rows (partner of a without_note refusal) are refused somewhat more on copied rows (36/78 vs 35/113).
- 51% of train+valid targets (234/455) are the identical refusal string: refusal 124, speak_as 30, preaching 40, nature_decode 40.

## Assessment
1. **Repetition penalty: not the cause.** a vs d differ little (wrong-refusal 67% vs 72%; exact refusal 97% vs 100%). Base with the penalty off is shorter but just as accurate. The penalty causes a few splices (cmp-ex-coat-colour), because the 64-token window includes the end of the prompt (the question), but that is minor. No need to change the app's penalty for this.
2. **Scale: lowering it to 10 does not fix over-refusal** (63%). It weakens the constraints the adapter did learn: exact refusal drops to 72%, and it produces made-up or malformed outputs. Over-refusal is not just "the adapter is too strong at inference".
3. **Checkpoint: the biggest single lever, but not a fix.** iter 500 (35% wrong-refusal) and iter 800 with penalty off (38%) answer about twice as often as iter 1000. But many of the extra answers copy the wrong note or splice notes (b: 15 wrong of 41 answered; e: 12 of 40), and the refusal string starts leaking (83% / 90%). Correct answers rise only from 17 to 26–28 of 71, against about 61 for base.
4. **Dominant cause: the training data and objective.** The adapter learned "output the refusal string, or copy one note sentence", but not which note answers the question or whether one exists. Signs: it refuses a third of its own training answerables; wrong-note copies appear as soon as it refuses less; the refusal string is half of all targets and is a constant, easy target; and many refusal-target prompts are near-duplicates of answerable ones (without_note/with_note pairs, preaching, nature_decode). The high effective LoRA strength (scale 20 × rank 16) probably makes it worse; earlier training at lr 1e-4 had loss spikes. Val loss is averaged over target tokens and dominated by the easy refusal string, so "best val loss" (800) is not a good selection signal here.

**Recommended next step:** retrain rather than tune decoding. Specifically:
- Cut the refusal share from 51% to about 25–30%, mainly by not mapping all preaching and nature_decode rows to refusal; give them a neutral grounded fact, as the comparison references do.
- Add more answerable questions per note, including ones whose notes contain similar-looking distractors.
- Lower the scale (e.g. 2–8) and try lr 1e-4 with warmup.
- Choose the checkpoint with this 100-row harness (wrong-refusal plus correct answers), not val loss.

If an adapter is needed right now, iter 800 with the penalty off is the least bad (wrong-refusal 38%, exact refusal 90%), but it still answers far less accurately than base.
