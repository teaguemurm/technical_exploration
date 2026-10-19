# Constraint LoRA for Qwen2.5-1.5B-Instruct

Training workspace for one question: can a LoRA impose the visitor-guide constraints that the base model ignores unless they are written into the prompt?

The Flutter app is unchanged. This folder only builds data, checks it, and describes the Mac training and conversion path. Nothing here loads a model. mlx-lm, the PEFT conversion, and llama.cpp were checked against current upstream sources and are untested on this machine.

## What the adapter is supposed to learn

Condition B is the persona plus the local notes, with no abstention hint and no behavioural bans. Condition C adds the bans and the exact refusal string from `standardRefusalStatement` in `lib/messages.dart`:

`I do not have information about that in the approved notes.`

The base Qwen2.5-1.5B answers "speak as the wisent" in the animal's voice, moralises on "what is nature trying to tell me", and abstains on about one prompt in ten. The adapter is trained on condition-B text only, so those rules are not in the prompt. The primary comparison is LoRA + B against base + C. Also run base + B, LoRA + C, and LoRA + A (no notes; an over-refusal check).

## Decisions

- **Notes sit in the system message.** mlx-lm chat rows are `messages` of system, user, and assistant. The system text is the persona from `assets/system_prompt.txt` plus the same `Local notes` block `lib/messages.dart` builds. There is no condition-C suffix. The app's condition B puts that notes block in a user turn instead. `scripts/render_protocol_prompt.py` can render both layouts. Train with `--layout train`. Score the protocol with `--layout app`. If LoRA + app-B is weak while LoRA + train-B is strong, the adapter did not cross the role boundary.
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
  eval/protocol_eval.json       # points at assets/prompts.json
```

Generated weights (`adapters/`, `*.safetensors`, `*.gguf`) are gitignored.

## Data

Rebuild from the repo root or from `lora/` (the scripts find the repo root):

```bash
python3 lora/scripts/build_dataset.py
python3 lora/scripts/validate_dataset.py
python3 lora/scripts/check_leakage.py
python3 -m unittest discover -s lora/tests -v
```

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

## Train on the Mac

From `lora/`, on the M2 Max:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-mac.txt

mlx_lm.lora --config configs/qwen25_1.5b_constraint.yaml
```

That is 500 steps, batch size 4, validation on the whole valid split every 50 steps, and a checkpoint `adapters/0000050_adapters.safetensors` through `adapters/0000500_adapters.safetensors`. Keep the checkpoint with the lowest validation loss, not automatically the last one. `adapters/adapters.safetensors` is the final step. If memory is tight, set `grad_checkpoint: true` or `batch_size: 1`. Drop `rank` to 8 if the valid loss rises while the model starts parroting training sentences. Confirm the log is bf16 LoRA and not QLoRA before trusting the run.

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
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct \
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

## Check with llama-cli, then copy to the phone

Protocol decoding matches the app: temperature 0, seed 42, top-k 40, top-p 0.9, repeat penalty 1.1, 256 tokens, context 2048.

```bash
python3 scripts/render_protocol_prompt.py \
  --condition B --prompt-id speak-as --layout app > /tmp/prompt.txt

llama-cli \
  -m qwen2.5-1.5b-instruct-q4_k_m.gguf \
  --lora wisent-constraint-lora-f16.gguf \
  --lora-scale 1.0 \
  -f /tmp/prompt.txt \
  --temp 0 --top-k 40 --top-p 0.9 --repeat-penalty 1.1 \
  -n 256 --seed 42 -c 2048 \
  --no-conversation
```

`--no-conversation` (short flag `-no-cnv`) sends the file as the full ChatML prompt. Repeat for conditions A, B, and C, with and without `--lora`. Score by hand using `eval/protocol_eval.json`. The ten prompts are not copied into that file.

```bash
adb push wisent-constraint-lora-f16.gguf /sdcard/Download/
```

`packages/llm_llamacpp` 0.7.0 loads an adapter when `LlamaCppChatRepository` is constructed with `loraPath` and `loraScale`, or after `setLora`. The inference isolate calls `llama_adapter_lora_init` while loading the model and applies the adapter to the new context before generation. `lib/main.dart` does not pass `loraPath`. Pointing the app at the file is a separate change.

## Untested here

- `mlx_lm.lora` and `mlx_lm.fuse` (no Apple Silicon, no MLX).
- `mlx_to_peft.py` against a real `adapters.safetensors`. The transpose, rename, alpha, and the refusal of `lm_head` / embeddings are unit-tested on plain lists.
- `convert_lora_to_gguf.py`, `llama-quantize`, and `llama-cli` (the shell wrapper is syntax-checked only).
- Loading the GGUF adapter in the Flutter isolate, and any on-device protocol run.
