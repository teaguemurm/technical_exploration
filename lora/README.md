# Constraint LoRA for Qwen2.5-1.5B-Instruct

Training workspace for one question: can a LoRA impose the visitor-guide constraints that the base model ignores unless they are written into the prompt?

The Flutter app is unchanged. This folder only builds data, checks it, and describes the Mac training and conversion path. Nothing here loads a model. mlx-lm, the PEFT conversion, and llama.cpp were checked against current upstream sources and are untested on this machine.

## What the adapter is supposed to learn

Condition B is the persona plus the local notes, with no abstention hint and no behavioural bans. Condition C adds the bans and the exact refusal string from `standardRefusalStatement` in `lib/messages.dart`:

`I do not have information about that in the approved notes.`

The base Qwen2.5-1.5B answers "speak as the wisent" in the animal's voice, moralises on "what is nature trying to tell me", and abstains on about one prompt in ten. The adapter is trained on condition-B text only, so those rules are not in the prompt. The primary comparison is LoRA + B against base + C. Also run base + B, LoRA + C, and LoRA + A (no notes; an over-refusal check).

## Decisions

- **The original set puts notes in the system message.** mlx-lm chat rows are `messages` of system, user, and assistant. The system text is the persona from `assets/system_prompt.txt` plus the same `Local notes` block `lib/messages.dart` builds. There is no condition-C suffix. The app's condition B puts that notes block in a user turn instead. `data/v2/` is the set that uses that app layout. `scripts/render_protocol_prompt.py` can render both layouts. Train the original files with `--layout train`. Score the protocol with `--layout app`.
- **Loss is on the assistant turn only** (`mask_prompt: true`). mlx-lm treats the last message as the completion.
- **Metadata stays on the JSONL row.** Current mlx-lm ignores keys the chat loader does not read (`messages`, and `tools` if present). That is stated in `mlx_lm/LORA.md`, and `ChatDataset.process` only indexes those fields. A sidecar `*.meta.jsonl` repeats the metadata without the messages. Do not add `prompt` or `completion` keys; those would switch the loader to the completions format.
- **Most notes are synthetic.** Five invented packs (red deer on the Veluwezoom, Konik horses in the Oostvaardersplassen, beavers in the Biesbosch, Highland cattle on the Dwingelderveld, Exmoor ponies at Lauwersmeer) match the one-sentence tone of `assets/dataset.json`. They are marked `synthetic: true` and are not park fact sheets.
- **Real Kraansvlak notes are used only for questions the ten protocol prompts do not ask** (other grazers, the park name, the Wisentenpad dates, whether people feed the wisents). The other six notes are left out of the training prompt. They are the answers to held-out prompts, and the wisent-herd sentence shares the 4-gram "the Kraansvlak herd is" with the on-path prompt. The protocol render still injects all ten notes, which is what condition B does at test time.
- **Contrastive pairs** ask the same question with the grounded note present and with that note removed. The answer flips to the exact refusal. Both sides stay in the same split.
- **Rank 16, mlx `scale` 20, all 28 layers, 500 steps, AdamW, learning rate 1e-4.** Rank 16 is inside the planned 8–16 band. mlx-lm stores `scale`, not PEFT `lora_alpha`. The forward pass is `y + scale * (x @ lora_a @ lora_b)`. PEFT and llama.cpp use `lora_alpha / rank`, so the converter sets `lora_alpha = scale * rank` (320 at these settings). `num_layers: -1` is the mlx-lm switch for every block. Embeddings and `lm_head` are not adapted: Qwen2.5 sets `tie_word_embeddings: true`, and `convert_lora_to_gguf.py` exits if either tensor is in the adapter.
- **Train the bf16 Hugging Face repo**, not an MLX 4-bit quant. mlx-lm uses QLoRA when the model is quantized.

## Layout

```
lora/
  README.md
  requirements.txt              # stdlib only
  requirements-mac.txt          # mlx-lm and the converter
  .gitignore
  configs/qwen25_1.5b_constraint.yaml
  wisent_lora/                  # builder, checks, conversion helpers
  scripts/                      # CLIs
  tests/
  data/
    held_out_paraphrases.json   # leakage probes, not training text
    synthetic_notes.json        # generated
    train.jsonl valid.jsonl     # chat rows, metadata included
    train.meta.jsonl valid.meta.jsonl
    stats.json
    v2/                         # app condition-B layout; does not replace the files above
      train.jsonl valid.jsonl
      train.meta.jsonl valid.meta.jsonl
      stats.json
    comparison/                 # evaluation only; never training data
      comparison.jsonl
      comparison.meta.jsonl
      stats.json
  eval/protocol_eval.json       # points at assets/prompts.json
```

Generated weights (`adapters/`, `*.safetensors`, `*.gguf`) are gitignored.

## Data

Rebuild from the repo root or from `lora/` (the scripts find the repo root):

```bash
python3 lora/scripts/build_dataset.py
python3 lora/scripts/build_v2.py
python3 lora/scripts/validate_dataset.py
python3 lora/scripts/check_leakage.py
python3 -m unittest discover -s lora/tests -v
```

`build_dataset.py` rewrites `lora/data/train.jsonl` and `valid.jsonl`. `build_v2.py` writes only `lora/data/v2/` and leaves those files byte-identical.

`build_dataset.py` writes the JSONL and exits non-zero if validation or the leakage check fails. No extra packages are required.

Targets, and the set that was built:

| Category | Share | Count | Train | Valid |
|---|---:|---:|---:|---:|
| answerable | 46.0% | 191 | 172 | 19 |
| refusal | 29.9% | 124 | 111 | 13 |
| speak_as | 14.5% | 60 | 54 | 6 |
| preaching | 9.6% | 40 | 36 | 4 |
| **total** | | **415** | **373** | **42** |

78 contrastive pairs (156 of the rows). Notes origin: 370 synthetic, 30 real Kraansvlak, 15 with no notes. Valid is 10.1%. The seed is 42.

Answerable rows are one to three neutral third-person sentences whose content words come from the grounded notes. Refusal rows are the refusal string and nothing else. Speak-as rows either refuse (feelings, names, greetings) or answer a fact in the third person. Preaching rows answer with a fact and do not add a moral. Voice and preaching traps use the synthetic species only.

Leakage check, run on this set: **0 hits**. A hit is containment, a shared word 4-gram, a `difflib` ratio of at least 0.72, or a same-length window ratio of at least 0.80, against `assets/prompts.json` and `data/held_out_paraphrases.json`. The 4-gram scan covers every message, so a prompt pasted into a note fails too. Fuzzy matching is on the user question.

Validation on this set: **0 errors**. It checks the chat schema, the exact refusal string, first-person animal voice, preaching wording, sentence count, category mix, pair integrity (the withheld note must actually be required), and that every answer token is in the grounded notes.

## Training set v2

`data/v2/` is a second training set. It does not replace `train.jsonl` / `valid.jsonl`. Those files stay the original train-layout rows.

Two changes apply to the 415 copied rows. Preaching answers are the refusal string from `standardRefusalStatement` in `lib/messages.dart`, and the grounded-note ids still name the note the old factual answer came from. Every prompt is condition B as `buildConditionMessages` builds it: the system turn is the trimmed persona from `assets/system_prompt.txt`, notes are the next user turn from `localNotesMessage` (title, blank line, intro, then the note texts in `prompt_note_ids` order), and the question is the last user turn. Rows with no notes stay system plus question. Questions, non-preaching answers, note tracing, and the train/valid membership of these 415 rows match the original set. The title and intro strings are read from `lib/messages.dart`, not copied by hand.

v2 also adds 40 `nature_decode` rows. The original set has speak-as rows for an animal's first person and preaching rows for rule 6. It has nothing for the Nature clause of condition C rule 4 ("Do NOT speak as Nature") or for rule 5 ("Do NOT claim to decode or translate the animal's behaviour"), both in `buildConditionCSystemPrompt`. Twenty rows ask the model to speak as Nature. Twenty ask it to decode or translate a stare, roar, tail slap, stance, or grouping. They use the five synthetic packs, half with that pack's notes and half with no notes. Every target is the refusal string, and `grounded_note_ids` is `none`. The questions do not use the held-out `nature-message` prompt or `data/held_out_paraphrases.json`. A shared 4-gram such as "trying to tell me" is a leak, so those wordings are not in the file.

The new rows use the same 10% split rule on their own random stream, so the copied rows keep the original split. That puts 4 `nature_decode` rows in valid and 36 in train.

| | Rows | Answerable | Refusal | Speak-as | Preaching | Nature-decode |
|---|---:|---:|---:|---:|---:|---:|
| Copied | 415 | 191 | 124 | 60 | 40 | 0 |
| New traps | 40 | 0 | 0 | 0 | 0 | 40 |
| **v2 total** | **455** | **191** | **124** | **60** | **40** | **40** |

Train 409, valid 46. Notes origin: 390 synthetic, 30 real, 35 with no notes. Contrastive pairs: 78. Protocol prompts and paraphrases: 0 leakage hits.

Train it with `configs/qwen25_1.5b_constraint_v2.yaml` (`data: "data/v2"`). The original config still points at `data`.

```bash
python3 lora/scripts/build_v2.py
```

From `lora/`:

```bash
mlx_lm.lora --config configs/qwen25_1.5b_constraint_v2.yaml
```

## Comparison set

`data/comparison/` is for evaluation only. It must never be used as training data. It is not a replacement for `train.jsonl` / `valid.jsonl` or for `data/v2/`, and rebuilding it does not rewrite those files. The file copies the original training rows and then adds more questions, so training on it would both duplicate the training set and mix in the generalisation questions.

Use it to read an answer against the notes first, with no adapter, then run the same question and the same notes with the adapter. On disk the system message is still the training layout: persona plus the local-notes block, no condition-C text. Generalisation scoring does not use that stored layout. `--layout app` rebuilds condition B before the model sees the row.

It has two kinds of row. The 415 training rows are copied, messages included, and flagged `question_in_original: true` with `source_split` `train` or `valid`. The other rows are new questions on the same note strings (`assets/dataset.json` and `data/synthetic_notes.json`). No note text was added. New rows are flagged `question_in_original: false`. A contrastive pair shares a question, so both sides share the flag.

New Kraansvlak prompts use every real note except `wisent-herd`. That sentence shares the 4-gram "the kraansvlak herd is" with the on-path protocol prompt, so it is not placed in a new system message. The copied training rows still use the four real notes they were trained with.

Each row's metadata, and the sidecar `comparison.meta.jsonl`, carries the category, grounded note ids or `none`, `notes_origin` (`synthetic`, `real`, or `none`), `expected_behaviour`, `reference_answer`, and the in-original flag.

| | Rows | Answerable | Refusal | Speak-as | Preaching |
|---|---:|---:|---:|---:|---:|
| Copied from train/valid | 415 | 191 | 124 | 60 | 40 |
| New questions | 495 | 230 | 145 | 72 | 48 |
| **Total** | **910** | **421 (46.3%)** | **269 (29.6%)** | **132 (14.5%)** | **88 (9.7%)** |

153 contrastive pairs (78 copied, 75 new). 756 distinct question strings. Notes origin on the combined file: 830 synthetic, 55 real, 25 with no notes. The training validator's 300–600 size gate is unchanged; this file is checked with an 800–1000 bound. Leakage check on this file: **0 hits**. Validation: **0 errors**.

```bash
python3 lora/scripts/build_comparison.py
```

Generalisation is `--layout app` on this file. That rebuilds each prompt the way condition B does in the app: persona in the system turn, notes in a user turn, question last. The CSV still has one row per question. The script also writes `*.summary.json` beside the CSV and prints two groups. `question_in_original: false` is the generalisation result. `question_in_original: true` repeats training questions under the app layout and is not that result.

`--layout stored` on `data/v2/train.jsonl` only checks that a trained adapter can reproduce prompts it was trained on. It says nothing about generalisation. The default layout is `stored`, so a comparison-file run does not change shape unless `--layout app` is set.

Decoding is read from the app. Temperature, top-k, top-p, seed, max tokens, and context come from `lib/main.dart`. The repeat-penalty window comes from `llama_sampler_init_penalties` in `packages/llm_llamacpp/lib/src/inference_sampler_config.dart`. That function does not set min-p. mlx uses `make_sampler` and `make_logits_processors` because `generate` no longer accepts `temp`. `--limit` runs a prefix of the file. Run these from `lora/`:

```bash
# Generalisation. Read the question_in_original false group in the summary.
python3 scripts/run_comparison.py \
  --backend mlx \
  --layout app \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --adapter adapters \
  --comparison data/comparison/comparison.jsonl \
  --output out/comparison.csv

# Training-fit check only. Not a generalisation result.
python3 scripts/run_comparison.py \
  --backend mlx \
  --layout stored \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --adapter adapters \
  --comparison data/v2/train.jsonl \
  --output out/v2-train-fit.csv

# llama-completion. Same app sampler chain. -no-cnv sends the ChatML file as the prompt.
python3 scripts/run_comparison.py \
  --backend llama \
  --layout app \
  --llama-cli llama-completion \
  --model qwen2.5-1.5b-instruct-q4_k_m.gguf \
  --adapter wisent-constraint-lora-f16.gguf \
  --lora-scale 1.0 \
  --comparison data/comparison/comparison.jsonl \
  --output out/comparison.csv
```

`mlx` is not installed here. The CSV row, the app-layout prompt, the sampler arguments, and the summary split are unit-tested; a real model run is not.

## Train on the Mac

From `lora/`, on the M2 Max:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-mac.txt

mlx_lm.lora --config configs/qwen25_1.5b_constraint.yaml
```

That command trains the original set. For the app-layout set, use `configs/qwen25_1.5b_constraint_v2.yaml` instead. The run is 500 steps, batch size 4, validation on the whole valid split every 50 steps, and a checkpoint `adapters/0000050_adapters.safetensors` through `adapters/0000500_adapters.safetensors`. Keep the checkpoint with the lowest validation loss, not automatically the last one. `adapters/adapters.safetensors` is the final step. If memory is tight, set `grad_checkpoint: true` or `batch_size: 1`. Drop `rank` to 8 if the valid loss rises while the model starts parroting training sentences. Confirm the log is bf16 LoRA and not QLoRA before trusting the run.

## Convert

`scale` 20 and rank 16 become PEFT `lora_alpha` 320. Factors are transposed from mlx `(in, rank)` / `(rank, out)` to PEFT `(rank, in)` / `(out, rank)`. Names become `base_model.model.model.layers.N....lora_A.weight` and `lora_B.weight`. A `lm_head` or embedding key is an error.

```bash
python3 scripts/mlx_to_peft.py \
  --mlx-adapter adapters \
  --checkpoint adapters/0000500_adapters.safetensors \
  --output peft_adapter
```

Omit `--checkpoint` to convert `adapters/adapters.safetensors`.

llama.cpp's converter needs a directory with the base `config.json` and tokenizer, not the weights. `--base-model-id` also works and ignores a stale path inside the PEFT config; this wrapper always passes `--base`.

```bash
hf download Qwen/Qwen2.5-1.5B-Instruct \
  --exclude "*.safetensors" --exclude "*.bin" \
  --local-dir base_hf

./scripts/convert_adapter_gguf.sh \
  --converter ~/src/llama.cpp/convert_lora_to_gguf.py \
  --peft peft_adapter \
  --base base_hf \
  --output wisent-constraint-lora-f16.gguf
```

llama.cpp applies the adapter as `adapter_scale * lora_alpha / rank`. With `lora_alpha = scale * rank` and `--lora-scale 1.0`, that multiplier is the mlx `scale`. The rank it divides by is `lora_b`'s leading ggml dimension, which is the PEFT rank after the usual dimension reversal.

### Fallback if the adapter GGUF will not load

Fuse, then quantize a full model. `mlx_lm.fuse --export-gguf` does not support `model_type: qwen2` (it allows Llama, Mistral, and Mixtral only), so use llama.cpp's converter. `--dequantize` is for a quantized base. A bf16 train should omit it.

```bash
mlx_lm.fuse \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --adapter-path adapters \
  --save-path fused_hf

python ~/src/llama.cpp/convert_hf_to_gguf.py fused_hf \
  --outtype bf16 --outfile fused-bf16.gguf
llama-quantize fused-bf16.gguf fused-q4_k_m.gguf Q4_K_M
```

The fused file replaces the base model. It is not passed as `loraPath`.

## Check with llama-completion, then copy to the phone

The installed `llama-cli` 0.5.0 is the chat client. It has no `--no-conversation`, Jinja is on by default, and a pre-rendered ChatML file would be wrapped again. One-shot completion of that file is `llama-completion` from the same build, which still accepts `-no-cnv`.

The flags are the app's sampler, not llama-completion's defaults. The CLI's default chain adds dry, top-n-sigma, typical-p, min-p (default 0.05), and xtc. `_configureSampler` does not.

| Flag | Value | Source |
|---|---|---|
| `--temp` | 0 | `lib/main.dart` `_temperature` |
| `--top-k` | 40 | `lib/main.dart` `_topK` |
| `--top-p` | 0.9 | `lib/main.dart` `_topP`. The app also passes min-keep 1 to `llama_sampler_init_top_p`. This CLI has no separate min-keep flag. |
| `--repeat-penalty` | 1.1 | `lib/main.dart` `_repeatPenalty` |
| `--repeat-last-n` | 64 | second argument of `llama_sampler_init_penalties` in `packages/llm_llamacpp/lib/src/inference_sampler_config.dart` |
| `--frequency-penalty`, `--presence-penalty` | 0 | the same penalties call passes 0 when those options are unset |
| `--seed` | 42 | `lib/main.dart` `_seed` |
| `-n` | 256 | `lib/main.dart` `_maxTokens` |
| `-c` | 2048 | `lib/main.dart` `_contextSize` |
| min-p | omitted | `_configureSampler` never calls `llama_sampler_init_min_p` |
| `--samplers` | `temperature;top_k;top_p;penalties` | call order in `_configureSampler`, then the seed distribution via `--seed` |

```bash
python3 scripts/render_protocol_prompt.py \
  --condition B --prompt-id speak-as --layout app > /tmp/prompt.txt

llama-completion \
  -m qwen2.5-1.5b-instruct-q4_k_m.gguf \
  --lora-scaled wisent-constraint-lora-f16.gguf:1.0 \
  -f /tmp/prompt.txt \
  --samplers temperature;top_k;top_p;penalties \
  --temp 0 --top-k 40 --top-p 0.9 \
  --repeat-last-n 64 --repeat-penalty 1.1 \
  --frequency-penalty 0 --presence-penalty 0 \
  -n 256 --seed 42 -c 2048 \
  --no-display-prompt -co off \
  -no-cnv
```

`-no-cnv` sends the file as the full ChatML prompt. `--no-display-prompt` and `-co off` keep the completion separate from the prompt text; they are not sampling. Repeat for conditions A, B, and C, with and without `--lora-scaled`. Score by hand using `eval/protocol_eval.json`. The ten prompts are not copied into that file.

```bash
adb push wisent-constraint-lora-f16.gguf /sdcard/Download/
```

`packages/llm_llamacpp` 0.7.0 loads an adapter when `LlamaCppChatRepository` is constructed with `loraPath` and `loraScale`, or after `setLora`. The inference isolate calls `llama_adapter_lora_init` while loading the model and applies the adapter to the new context before generation. `lib/main.dart` does not pass `loraPath`. Pointing the app at the file is a separate change.

## Planned next steps

After the initial LoRA is implemented and tested with the full-notes prompt (condition B), add simulated-RAG support for the LoRA version. That needs a fixed note selection for each question in the comparison set: the relevant notes plus deliberate near misses, the same idea as `assets/simulated_rag.json` (`k` of 2). The selection can be assigned by hand or automatically. Build and score the normal full-notes version first. Simulated RAG comes later. Nothing in this folder selects a subset of notes per comparison question yet.

## Untested here

- `mlx_lm.lora` and `mlx_lm.fuse` (no Apple Silicon, no MLX).
- `mlx_to_peft.py` against a real `adapters.safetensors`. The transpose, rename, alpha, and the refusal of `lm_head` / embeddings are unit-tested on plain lists.
- `convert_lora_to_gguf.py`, `llama-quantize`, and `llama-completion` (the shell wrapper is syntax-checked only).
- Loading the GGUF adapter in the Flutter isolate, and any on-device protocol run.
- `scripts/run_comparison.py` against mlx-lm or llama-completion. The comparison CSV writer, the app-layout prompt, the sampler arguments, and the summary split are unit-tested only.
