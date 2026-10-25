# Protocol prompts via llama.cpp: HEURISTIC summary

**Heuristic only — regex markers, not Teague's rubric.** Predicates from `lora/results/s4/analyze_s4.py` (exact refusal = output equals the refusal string after whitespace/quote strip; wrong refusal = contains the refusal string; first-person ` I `/`I'm`/` my `/` we ` (refusal string excluded); preach = should/must/respect/protect/remember/important/…/matter(s); nature/decoding = 'Mother Nature', 'trying to tell', 'is saying', 'translat*', 'decod*' …; >3 sentences = analyze_s4 sentence split, here over all 10 outputs; degenerate = empty, 4-gram repeated ≥3×, ≥150 words, truncated, or hit 256 tokens). Key facts = not a refusal and every `relevantFactIds` fact of the prompt matched by a keyword regex defined in `make_reports.py` (`FACT_RX`) — prompts.json has no reference answers, so this replaces analyze_s4's must_include/overlap test.

Denominators: prompts with relevantFactIds are answer-target (6: what-animal, who-manages, where-stand, on-path, since-when, make-care). Prompts without facts (4) are treated as refusal-target; **exact refusal** shows `x/2` for the clear ones (morning-meal, calves) and `(+y/2)` for nature-message and speak-as, where prompts.json does not say whether a refusal or a declining factual note is the target. Markers and degenerate are out of 10. Refusal is only requested by the condition C prompt; under A/B a refusal is not instructed, so A/B refusal counts are descriptive.

| model | variant | cond | exact refusal (clear 2 + unclear 2) | wrong refusal /6 | key facts /6 | first person /10 | preach /10 | nature/decoding /10 | >3 sentences /10 | degenerate /10 | mean gen tokens |
|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen | base | A | 0/2 (+0/2) | 0/6 | 0/6 | 1/10 | 3/10 | 1/10 | 1/10 | 0/10 | 40 |
| qwen | lora | A | 2/2 (+2/2) | 6/6 | 0/6 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 13 |
| qwen | base | B | 0/2 (+0/2) | 0/6 | 3/6 | 1/10 | 3/10 | 0/10 | 2/10 | 0/10 | 43 |
| qwen | lora | B | 2/2 (+2/2) | 2/6 | 1/6 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 13 |
| qwen | base | C | 1/2 (+0/2) | 0/6 | 3/6 | 2/10 | 3/10 | 1/10 | 0/10 | 0/10 | 38 |
| qwen | lora | C | 2/2 (+2/2) | 2/6 | 2/6 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 14 |
| gemma | base | A | 0/2 (+0/2) | 0/6 | 0/6 | 1/10 | 1/10 | 0/10 | 2/10 | 0/10 | 58 |
| gemma | lora | A | 2/2 (+2/2) | 6/6 | 0/6 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 13 |
| gemma | base | B | 0/2 (+0/2) | 0/6 | 3/6 | 2/10 | 0/10 | 0/10 | 0/10 | 0/10 | 38 |
| gemma | lora | B | 2/2 (+2/2) | 2/6 | 2/6 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 15 |
| gemma | base | C | 1/2 (+1/2) | 0/6 | 3/6 | 0/10 | 1/10 | 0/10 | 0/10 | 0/10 | 23 |
| gemma | lora | C | 2/2 (+2/2) | 4/6 | 1/6 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 14 |

**Extra (clearly not part of the A/B/C request): app sim-RAG conditions, fixed k=2 notes from assets/simulated_rag.json**

| model | variant | cond | exact refusal (clear 2 + unclear 2) | wrong refusal /6 | key facts /6 | first person /10 | preach /10 | nature/decoding /10 | >3 sentences /10 | degenerate /10 | mean gen tokens |
|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen | base | B-sim-RAG | 0/2 (+0/2) | 0/6 | 3/6 | 1/10 | 2/10 | 1/10 | 2/10 | 0/10 | 52 |
| qwen | lora | B-sim-RAG | 2/2 (+2/2) | 2/6 | 1/6 | 1/10 | 1/10 | 0/10 | 0/10 | 0/10 | 14 |
| qwen | base | C-sim-RAG | 1/2 (+0/2) | 0/6 | 3/6 | 2/10 | 2/10 | 0/10 | 0/10 | 0/10 | 28 |
| qwen | lora | C-sim-RAG | 2/2 (+2/2) | 3/6 | 1/6 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 13 |
| gemma | base | B-sim-RAG | 0/2 (+0/2) | 0/6 | 3/6 | 2/10 | 1/10 | 0/10 | 1/10 | 0/10 | 41 |
| gemma | lora | B-sim-RAG | 2/2 (+2/2) | 4/6 | 2/6 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 15 |
| gemma | base | C-sim-RAG | 1/2 (+1/2) | 0/6 | 3/6 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 19 |
| gemma | lora | C-sim-RAG | 2/2 (+2/2) | 6/6 | 0/6 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 13 |

## llama.cpp base vs earlier app runs (same laptop, same quants)

Reference: `exports/technical_exploration_runs_20261020_193111.json` (Mac14,6): Qwen = 11 Oct 2026 15:4x batch, Gemma = 20 Oct 2026 19:2x gemma4-format batch (the runs the sheet's Inference tab reports). The rebuilt messages equal the app's `messagesSent` for all 100 reference rows (`app_reference.py`). exact = identical string; near = difflib ratio ≥ 0.9; details in `app_match.json`.

| model | condition | exact | near | differs | prompt tokens equal | gen tokens equal |
|---|---|---|---|---|---|---|
| qwen | A | 10 | 0 | 0 | 10/10 | 0/10 |
| qwen | B | 10 | 0 | 0 | 10/10 | 0/10 |
| qwen | B-sim-RAG | 10 | 0 | 0 | 10/10 | 0/10 |
| qwen | C | 10 | 0 | 0 | 10/10 | 0/10 |
| qwen | C-sim-RAG | 10 | 0 | 0 | 10/10 | 0/10 |
| gemma | A | 10 | 0 | 0 | 10/10 | 0/10 |
| gemma | B | 10 | 0 | 0 | 10/10 | 0/10 |
| gemma | B-sim-RAG | 10 | 0 | 0 | 10/10 | 0/10 |
| gemma | C | 10 | 0 | 0 | 10/10 | 0/10 |
| gemma | C-sim-RAG | 10 | 0 | 0 | 10/10 | 0/10 |

Generated-token counts are consistently 1 lower in llama.cpp than in the app export for all 100 base rows (the app's count includes the end-of-turn token; llama-server's `predicted_n` does not), so it is a counting difference, not a generation difference.

## Notes and deviations

- Settings follow the app protocol config (lib/main.dart: temp 0, seed 42, top_k 40, top_p 0.9, repeat 1.1, max 256 tokens, ctx 2048, batch 2048; no thinking toggle → Gemma rendered with formatGemma4, no thinking prefix). The app's export records exactly these decodingSettings for all reference rows. No deviation found.
- Engine: llama.cpp b10481 `llama-server` (the version bundled in llm_llamacpp 0.7.0), raw prompt to /completion, `cache_prompt=false`, sampler chain temperature→top_k→top_p→penalties as `_configureSampler`; stops and `sanitizeModelOutput` (Gemma keepThoughts) as the app. Reused from `../llamacpp_compare/` (common.py format/sanitize/stops; same server flags as run_llamacpp.py). Gemma: literal `<bos>` dropped from text and added by the server; token ids checked identical to the app's for every Gemma row.
- LoRA: GGUF f16 adapters at scale 1 (`--lora-scaled`), confirmed by `/lora-adapters`. The app UI itself never passes a loraPath, so 'app with LoRA' = the package API at scale 1 (see ../llamacpp_compare/notes.md). No app-made LoRA runs exist to compare against.
- Conditions A/B/C = all-notes protocol (dataset.json, 10 notes), not the 910-row comparison set. sim-RAG rows are extras.
- Under condition A no notes are supplied, so the LoRA's blanket refusal there is consistent with its constraint training; 'wrong refusal' under A counts it against the answerable prompts anyway.
- Gemma base (llama.cpp and app) produced no thought blocks.
