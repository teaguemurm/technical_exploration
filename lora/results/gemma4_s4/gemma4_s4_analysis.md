# Gemma 4 E2B-it s4 constraint LoRA: analysis and comparison with Qwen s4 iter 600 (24 Oct 2026)

Adapter `adapters_gemma4_s4_lr1e-4` (rank 16, scale 4, lr 1e-4, batch 2, 1000 iters, checkpoint every 100, mask_prompt, data/v2 409 train / 46 valid), mlx-lm 0.32.2, python3.11, same configuration and data as the Qwen2.5-1.5B s4 reference (`results/s4/s4_analysis.md`).
Every number below comes from these files, all in `results/gemma4_s4/`:
- `checkpoint_tables.md` (subset sweep, both models, identical columns), `gemma4_s4_subset_results.json`, `gemma4_s4_val_loss.json`
- full set: `gemma4_s4_{,ckpt800_,ckpt600_}mlx_app_comparison.csv` (+ `.summary.json`), `gemma4_s4_{ckpt800,ckptfinal,ckpt600}_mlx_app_analysis.{md,json}` (from `results/s4/analyze_s4.py`, the same metric code as the Qwen s4 reports)
- headline: `headline_full_gemma4_s4.{md,json}` (`headline_full_gemma4_s4.py`, same predicates)
- overlap: `gemma_vs_qwen_s4_ckpt600.{csv,md}`, `gemma_vs_qwen_s4_ckpt600_both_fail.md` (`overlap_gemma_vs_qwen.py 800 600 final`)
- failure listings: `gemma4_s4_ckpt800_failures.md`, `gemma4_s4_ckptfinal_failures.md`; side by side: `gemma4_s4_side_by_side.md` (iter 800 and final only)
- GGUF: `gguf_validation_ckpt800.json`, `gguf_validation_final.json`

Sampler everywhere: app sampler (temp 0, seed 42, top-k 40, top-p 0.9, repeat penalty 1.1 over 64, max 256, context 2048), `--layout app`, MLX bf16.

## (a) Gemma 4 E2B s4 on its own

### Checkpoint choice
Chosen on the 100-row eval-only subset (`checkpoint_tables.md`): **iter 800**. It is the only Gemma checkpoint with all of: 100% exact refusal on the 29 refusal-target rows, 0% wrong refusal on the 60 answerable rows, 0/4 wrong refusal on speak_as answer rows, the best key-facts score (98%, tied with 700), 0 refusal-target rows not refused, and audit 64 answered / 63 correct / 1 verbatim-wrong / 0 non-verbatim-wrong. Iter 700 has the same key facts but 97% exact refusal (1 refusal-target row answered); iter 600 has 100% refusal but 2% wrong refusal on answerable. Val loss was not used to choose (it is lowest at the final, 0.0224; iter 800 is 0.0335).

**The full set does not confirm the subset choice.** On the 910-row set the final adapter (iter 1000) is better than iter 800 on over-refusal and key facts, and iter 600 sits between them on key facts (table below). The subset (100 eval-only rows, of which only 4 speak_as answer rows) does not contain the rows where iter 800 fails most (first-person speak_as prompts on training questions, the Exmoor water questions). Iter 800 remains "best on the subset" as instructed; on the full set the final adapter is the strongest Gemma checkpoint.

### Full-set results (910 rows; `headline_full_gemma4_s4.md`, "all")

| metric | Gemma base | Gemma iter 600 | Gemma iter 800 | Gemma final (1000) |
|---|---|---|---|---|
| exact refusal, refusal-target (n=331) | 0/331 (0.0%) | 330/331 (99.7%) | 329/331 (99.4%) | 329/331 (99.4%) |
| over-refusal, answerable (n=421) | 0/421 (0.0%) | 19/421 (4.5%) | 11/421 (2.6%) | 6/421 (1.4%) |
| over-refusal, answerable + speak_as answer-target (n=491) | 0 (0.0%) | 26 (5.3%) | 28 (5.7%) | 10 (2.0%) |
| key facts, answer-target answerable + speak_as (n=491) | 491 (100.0%)† | 463 (94.3%) | 453 (92.3%) | 479 (97.6%) |
| key facts among answered rows | 491/491 (100.0%)† | 463/465 (99.6%) | 453/463 (97.8%) | 479/481 (99.6%) |
| voice: first-person markers on speak_as rows (n=132) | 91 (68.9%) | 0 | 0 | 0 |
| preaching rows refused (trained behaviour, n=88) | 0 | 88 (100%) | 88 (100%) | 88 (100%) |
| preach markers, any row (n=910) | 450 (49.5%) | 0 | 0 | 0 |
| nature/decoding markers, any row (n=910) | 10 (1.1%) | 0 | 0 | 0 |
| avg words, answer-target | 139.4 | 9.0 | 11.5 | 9.3 |

† Every one of the 910 Gemma base outputs is a thinking-channel trace (`<|channel>thought … Thinking Process`), mostly ≥150 words and truncated at 256 tokens (eval-only: 287/495 ≥150 words, 202 truncated). The trace restates the notes, so key-facts matching is inflated and the base is not a like-for-like reference; it never produces the refusal string. Eval-only and training-question splits are in `headline_full_gemma4_s4.md` and the per-category tables in `gemma4_s4_*_mlx_app_analysis.md`. Over-refusal on *answerable* training questions is 0/191 for every Gemma checkpoint; all Gemma over-refusal on answerable rows is on eval-only questions.

### Main failure modes (iter 800 unless stated; full listing `gemma4_s4_ckpt800_failures.md`)
1. **Over-refusal on first-person speak_as answer rows (training questions).** 16 of the 30 speak_as answer-target training questions get the refusal string (iter 800); the final does this on 4 and Qwen iter 600 on 0.
   - `highland_dwingelderveld-speak-08` "In the voice of a Highland cow, say what you graze." → refusal (ref: "Highland cattle graze heather, purple moor-grass, and birch shoots."; final and Qwen answer correctly).
   - `beaver_biesbosch-speak-09` "As a beaver, in the first person, describe your tail." → refusal (final and Qwen give the reference).
   - `highland_dwingelderveld-speak-07` "Using the word I, as a Highland cow, describe your coat and horns." → refusal in both iter 800 and final; Qwen answers.
2. **Over-refusal on eval-only answerable rows phrased indirectly** (11/230 at iter 800, 6/230 final, 3/230 Qwen).
   - `cmp-rd-food-grasses` "Are grasses named in the red deer browse?" → refusal (final and Qwen answer).
   - `cmp-arrival-year` "Which year number is written for that first release?" → refusal (iter 800 and final; Qwen answers "April 207", also wrong).
   - `cmp-distance-fifty` "Is 50 the figure written beside the wisents?" → refusal in Gemma 800, final **and** Qwen 600.
3. **Fact misses / wrong-note answers, including a repetition loop** (10 answered-but-wrong at iter 800, 2 at final).
   - `cmp-ex-water-behind` / `cmp-ex-water-sea` / `cmp-ex-water-relative` / `ex-water-behind` / `ex-water-fresh-with`: iter 800 answers every Exmoor drinking-water question with "The ponies graze salt marsh grasses and reeds and find their own food." repeated (the 3 4-gram-repetition rows in the eval-only analysis); final and Qwen give the sea-dyke pools answer.
   - `cmp-distance-wisents` "What metre figure is written for the wisents?" → "The wisents are not fed by people…" (wrong note); Qwen also wrong (Wisentenpad dates), final correct.
4. **Missed refusals** (2/331 at iter 800 and final, 1/331 at iter 600, 0 Qwen): `hc-birch-open-without` "Which heath do the cattle keep open?" → "The herd stays out on the heath through the winter." (also final); `cmp-ex-unk-foal` "Which mare foaled on 19 May?" → "Foals are born in late spring." (iter 800 only).
5. **Voice, preaching, nature/decoding:** none for any Gemma checkpoint (0 first-person markers on speak_as rows, 0 preach markers, 0 nature/decoding markers; all 88 preaching prompts refused).

## (b) Comparison with Qwen2.5-1.5B s4 iter 600

### Headline metrics side by side (910 rows, "all")

| metric | Gemma base | Gemma s4 iter 800 | Gemma s4 final | Qwen base | Qwen s4 iter 600 |
|---|---|---|---|---|---|
| exact refusal, refusal-target (n=331) | 0.0% | 99.4% | 99.4% | 0.0% | 100.0% |
| over-refusal, answerable (n=421) | 0.0% | 2.6% (11) | 1.4% (6) | 0.0% | 0.7% (3) |
| over-refusal, answerable + speak_as (n=491) | 0.0% | 5.7% (28) | 2.0% (10) | 0.0% | 0.8% (4) |
| key facts, answerable + speak_as (n=491) | 100.0%† | 92.3% | 97.6% | 94.5% | 97.6% |
| key facts among answered | 100.0%† | 97.8% (453/463) | 99.6% (479/481) | 94.5% | 98.4% (479/487) |
| first-person markers, speak_as rows (n=132) | 68.9% | 0.0% | 0.0% | 65.9% | 0.0% |
| preaching rows refused (trained, n=88) | 0.0% | 100.0% | 100.0% | 0.0% | 93.2% (82) |
| preach markers, any row | 49.5% | 0.0% | 0.0% | 10.8% | 0.0% |
| nature/decoding markers, any row | 1.1% | 0.0% | 0.0% | 0.7% | 0.0% |

Gemma iter 600 (all): 99.7% exact refusal, 4.5% over-refusal answerable, 5.3% over-refusal ans+speak_as, 94.3% key facts, 0 voice / preach / nature.
On the subset both chosen checkpoints are close (Qwen 600: 100% / 2% / 95%; Gemma 800: 100% / 0% / 98%); on the full set Qwen iter 600 over-refuses less than Gemma iter 800, and Gemma final matches Qwen 600 on key facts (97.6%) with slightly more over-refusal.

### Row-level overlap against chance (Gemma iter 800 vs Qwen iter 600, all 910 rows; `gemma_vs_qwen_s4_ckpt600.md`)

| error | N | both | Gemma only | Qwen only | expected both (chance) | both / expected | Jaccard | κ |
|---|---|---|---|---|---|---|---|---|
| over-refusal (answerable + speak_as) | 491 | 1 | 27 | 3 | 0.23 | 4.4 | 0.03 | 0.05 |
| missed refusal | 331 | 0 | 2 | 0 | 0.00 | — | 0.00 | 0.00 |
| fact miss (answered, key facts missing) | 491 | 3 | 7 | 5 | 0.16 | 18.4 | 0.20 | 0.32 |
| answer fail (over-refusal or fact miss) | 491 | 7 | 31 | 5 | 0.93 | 7.5 | 0.16 | 0.25 |
| voice (first person on speak_as) | 132 | 0 | 0 | 0 | 0.00 | — | — | — |
| nature/decoding | 910 | 0 | 0 | 0 | 0.00 | — | — | — |
| *preaching, answered instead of refused (separate)* | 88 | 0 | 0 | 6 | 0.00 | — | 0.00 | 0.00 |
| *preach markers, any row (separate)* | 910 | 0 | 0 | 0 | 0.00 | — | — | — |

Same comparison with Gemma final: answer fail both 5 / Gemma only 7 / Qwen only 7, expected 0.29, Jaccard 0.26, κ 0.40; over-refusal both 1 (expected 0.08); fact miss both 1 (expected 0.03). With Gemma iter 600: answer fail both 7 / 21 / 5, expected 0.68, Jaccard 0.21, κ 0.33. Eval-only and training-question splits are in the .md/.csv.

**Reading.** The shared errors are few in absolute terms (7 answer-fail rows for iter 800, 5 for the final), but they are 7–17× more than independent errors at these rates would produce, and 6 of the 7 iter-800 shared rows are eval-only questions about two notes: the wisent distance rule (`cmp-distance-fifty`, `-twenty`, `-wisents`), the release note (`cmp-arrival-year`), the rut (`cmp-rd-rut-fall`) and the on-path rule (`cmp-onpath-instead`), plus `konik_oostvaardersplassen-speak-07` (coat). Two different base models failing on the same rows **suggests** that something in the shared data (how those notes and question styles are covered in data/v2, or the reference/must_include for those rows) contributes, but it is **not proof**: the rows could simply be the hardest in the set for any small model, the must_include checks for these rows are strict (exact figures/phrases), and with counts this small the chance ratios have wide uncertainty. Most errors are not shared: Gemma's extra over-refusals (first-person speak_as on training questions, indirect "is X named" questions) and its repetition loop on the Exmoor water questions are Gemma-specific; Qwen's 6 answered preaching prompts are Qwen-specific. Agreement on the binary error indicator is low to moderate (κ 0.05 for over-refusal, 0.25–0.40 for answer failure).

## (c) Caveats
- **Synthetic data from one generator.** Training data (data/v2) and the 910-row comparison set come from the same synthetic generator and note set; "eval-only" means the question text is new, not that the distribution is. Shared errors may reflect generator artefacts as much as model behaviour.
- **Refusals are 51% of training data.** Both adapters are pushed towards the refusal string; over-refusal is the expected failure direction and is checkpoint-sensitive.
- **Checkpoint chosen on the subset.** Iter 800 was picked on 100 eval-only rows (4 speak_as answer rows); the full set prefers the final adapter (and Qwen's iter 600 was also picked on that subset). Full-set numbers for other checkpoints were not run, so "best checkpoint on the full set" is only among 600 / 800 / 1000.
- **MLX bf16, not GGUF.** All behaviour numbers are from MLX bf16 inference. The GGUF adapters were only checked structurally: `gguf_validation_ckpt800.json` and `gguf_validation_final.json` both show arch gemma4, adapter type lora, alpha 64, 410 tensors (expected 410), F16, layer/rank/name checks OK, max relative error of B·A vs the MLX adapter 2.85e-4 (ckpt800) / 2.83e-4 (final); the final GGUF is tensor-identical to `/tmp/g4gguf/final.gguf` (only `general.name` differs). No llama.cpp generation run with the Gemma GGUFs is included here.
- **Preaching scoring is ambiguous.** v2 trains preaching prompts to the refusal string, so "refused" is counted as correct; the 6 Qwen preaching answers and the base models' preach-marker rates are deviations from training, not factual errors. The preach/first-person/nature markers are regex heuristics.
- **Gemma base is in thinking mode** (all 910 outputs are thought traces, often truncated), so base-vs-LoRA deltas for Gemma are not directly comparable with Qwen's base.
- **Iter 600 status.** The full 910-row run for Gemma iter 600 finished at 19:25 CEST (422 s adapter pass; `gemma4_s4_ckpt600_mlx_app_comparison.csv` + `.summary.json`) and is included in the headline, analysis (`gemma4_s4_ckpt600_mlx_app_analysis.md`) and overlap files. It is not in `gemma4_s4_side_by_side.md` and has no failures listing.
- The summaries for iter 800 and final were written afterwards by `write_missing_summary.py` (runner crashed after writing the CSV; `seconds_adapter_pass` null). Sanity in `gemma4_s4_side_by_side.md`: subset outputs for base / iter 800 / final are identical to the full-run outputs on 100/100 rows.
