# Constraint LoRA

Training workspace for one question: can a LoRA impose the visitor-guide constraints that the base model ignores unless they are written into the prompt?

Two instruction models were trained on the same v2 set: Qwen2.5-1.5B-Instruct and Gemma 4 E2B-it. The Flutter app loads a matching adapter GGUF with the base model at scale `1.0`. Scored MLX and llama.cpp runs are under `results/`. Generated weights (`adapters/`, `*.safetensors`, `*.gguf`) stay gitignored.

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
- **Rank 16, AdamW, every layer.** The original Qwen config (`configs/qwen25_1.5b_constraint.yaml`) is mlx `scale` 20, 500 steps, learning rate 1e-4, batch size 4. The adapters that were scored are the later configs in [Trained adapters](#trained-adapters). mlx-lm stores `scale`, not PEFT `lora_alpha`. The forward pass is `y + scale * (x @ lora_a @ lora_b)`. PEFT and llama.cpp use `lora_alpha / rank`, so the converter sets `lora_alpha = scale * rank` (320 at scale 20, 64 at scale 4). `num_layers: -1` is the mlx-lm switch for every block. Embeddings and `lm_head` are not adapted: Qwen2.5 sets `tie_word_embeddings: true`, and `convert_lora_to_gguf.py` exits if either tensor is in the adapter. Gemma 4 leaves out the same tensors plus its per-layer embeddings. Convert that adapter with `scripts/mlx_to_peft_gemma4.py`, because mlx names the weights `language_model.model.layers.N...`.
- **Train the bf16 Hugging Face repo**, not an MLX 4-bit quant. mlx-lm uses QLoRA when the model is quantized.

## Layout

```
lora/
  README.md
  requirements.txt              # stdlib only
  requirements-mac.txt          # mlx-lm and the converter
  .gitignore
  configs/qwen25_1.5b_constraint.yaml
  configs/qwen25_1.5b_constraint_v2.yaml
  configs/qwen25_1.5b_constraint_v2_s4_lr1e-4.yaml
  configs/gemma4_e2b_constraint_v2_s4_lr1e-4.yaml
  wisent_lora/                  # builder, checks, conversion helpers
  scripts/                      # CLIs, including mlx_to_peft_gemma4.py
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

Three changes apply to the 415 copied rows. Preaching answers are the refusal string from `standardRefusalStatement` in `lib/messages.dart`, and the grounded-note ids still name the note the old factual answer came from. Stiff speak-as questions are reworded into visitor speech in this set only. Each of those rows keeps its id, category, answer, note ids, and split. Every prompt is condition B as `buildConditionMessages` builds it: the system turn is the trimmed persona from `assets/system_prompt.txt`, notes are the next user turn from `localNotesMessage` (title, blank line, intro, then the note texts in `prompt_note_ids` order), and the question is the last user turn. Rows with no notes stay system plus question. Aside from the reworded speak-as questions, the questions match the original set. Non-preaching answers, note tracing, and the train/valid membership of these 415 rows match the original set. The title and intro strings are read from `lib/messages.dart`, not copied by hand. The same from/to pairs are in `data/v2/stats.json` under `speak_as_reworded`.

All 60 speak-as questions were reworded:

- `red_deer_veluwe-speak-01`: "Answer in the first person as a red deer and say how you feel today." → "You're a red deer for a minute. How are you feeling today?"
- `red_deer_veluwe-speak-02`: "Take the role of a stag on the Veluwezoom and describe your mood." → "You're a stag on the Veluwezoom now. What's your mood like?"
- `red_deer_veluwe-speak-03`: "What frightens you, if you answer as a red deer?" → "Pretend you're a red deer: what scares you?"
- `red_deer_veluwe-speak-04`: "Tell me your personal name, speaking as a stag." → "You're a stag. What should I call you?"
- `red_deer_veluwe-speak-05`: "Describe your hunger in the voice of a red deer." → "You're a red deer. How hungry are you?"
- `red_deer_veluwe-speak-06`: "Welcome the reader to the Veluwezoom in character as a stag." → "You're a stag. Welcome me to the Veluwezoom."
- `red_deer_veluwe-speak-07`: "Using the word I, as a red deer, say which plants you eat." → "Pretend you're a red deer: which plants do you eat?"
- `red_deer_veluwe-speak-08`: "In the voice of a stag, say what happens to your antlers in late winter." → "You're a stag. What happens to your antlers in late winter?"
- `red_deer_veluwe-speak-09`: "As a deer, in the first person, say where you drink." → "Pretend you're one of the deer: where do you go for water?"
- `red_deer_veluwe-speak-10`: "Role-play a deer and say where you shelter in hard weather." → "You're a deer now. Where do you shelter when the weather turns rough?"
- `red_deer_veluwe-speak-11`: "Pretend to be a stag and say at what time of day you roar." → "You're a stag. What time of day do you roar?"
- `red_deer_veluwe-speak-12`: "Answer as a hind, in the first person, and say who stays together." → "You're a hind. Who do you stay with?"
- `konik_oostvaardersplassen-speak-01`: "Answer in the first person as a Konik horse and say how you feel today." → "You're a Konik for a minute. How are you feeling today?"
- `konik_oostvaardersplassen-speak-02`: "Take the role of a mare in the Oostvaardersplassen and describe your mood." → "You're a mare in the Oostvaardersplassen now. How are you feeling?"
- `konik_oostvaardersplassen-speak-03`: "What frightens you, if you answer as a Konik?" → "Pretend you're a Konik: what scares you?"
- `konik_oostvaardersplassen-speak-04`: "Tell me your personal name, speaking as a Konik horse." → "You're a Konik horse. What's your name?"
- `konik_oostvaardersplassen-speak-05`: "Describe your hunger in the voice of a Konik." → "You're a Konik. How hungry are you right now?"
- `konik_oostvaardersplassen-speak-06`: "Welcome the reader to the marsh in character as a stallion." → "You're a stallion. Welcome me out onto the marsh."
- `konik_oostvaardersplassen-speak-07`: "Using the word I, as a Konik, describe your coat." → "Pretend you're a Konik: what does your coat look like?"
- `konik_oostvaardersplassen-speak-08`: "In the voice of a Konik, say what you graze." → "You're a Konik. What do you graze on?"
- `konik_oostvaardersplassen-speak-09`: "As a horse, in the first person, say where you drink." → "Pretend you're one of the horses: where do you go for water?"
- `konik_oostvaardersplassen-speak-10`: "Role-play a mare and say when foals are born." → "You're a mare. When are the foals born?"
- `konik_oostvaardersplassen-speak-11`: "Pretend to be a stallion and say how the bands are led." → "You're a stallion. How are the bands led?"
- `konik_oostvaardersplassen-speak-12`: "Answer as a Konik, in the first person, and say where the band shelters." → "You're a Konik. Where does the band shelter?"
- `beaver_biesbosch-speak-01`: "Answer in the first person as a beaver and say how you feel tonight." → "You're a beaver for a minute. How are you feeling tonight?"
- `beaver_biesbosch-speak-02`: "Take the role of a beaver in the Biesbosch and describe your mood." → "You're a beaver in the Biesbosch now. What's your mood?"
- `beaver_biesbosch-speak-03`: "What frightens you, if you answer as a beaver?" → "Pretend you're a beaver: what scares you?"
- `beaver_biesbosch-speak-04`: "Tell me your personal name, speaking as a beaver." → "You're a beaver. What's your name?"
- `beaver_biesbosch-speak-05`: "Describe your hunger in the voice of a beaver." → "You're a beaver. How hungry are you?"
- `beaver_biesbosch-speak-06`: "Welcome the reader to the creeks in character as a beaver." → "You're a beaver. Welcome me onto the creeks."
- `beaver_biesbosch-speak-07`: "Using the word I, as a beaver, say what the lodge is built from." → "Pretend you're a beaver: what is the lodge built from?"
- `beaver_biesbosch-speak-08`: "In the voice of a beaver, say what you eat." → "You're a beaver. What do you eat?"
- `beaver_biesbosch-speak-09`: "As a beaver, in the first person, describe your tail." → "Pretend you're a beaver: what's your tail like?"
- `beaver_biesbosch-speak-10`: "Role-play a beaver and say when you are mostly active." → "You're a beaver. When are you mostly active?"
- `beaver_biesbosch-speak-11`: "Pretend to be a beaver and say where kits are born." → "You're a beaver. Where are the kits born?"
- `beaver_biesbosch-speak-12`: "Answer as a beaver, in the first person, and say whether you dam every creek." → "You're a beaver. Do you dam every creek?"
- `highland_dwingelderveld-speak-01`: "Answer in the first person as a Highland cow and say how you feel today." → "You're a Highland cow for a minute. How do you feel today?"
- `highland_dwingelderveld-speak-02`: "Take the role of a Highland animal on the Dwingelderveld and describe your mood." → "You're a Highland cow now. How are you feeling today?"
- `highland_dwingelderveld-speak-03`: "What frightens you, if you answer as a Highland cow?" → "Pretend you're a Highland cow: what scares you?"
- `highland_dwingelderveld-speak-04`: "Tell me your personal name, speaking as a Highland cow." → "You're a Highland cow. What name do you go by?"
- `highland_dwingelderveld-speak-05`: "Describe your hunger in the voice of a Highland cow." → "You're a Highland cow. How hungry are you?"
- `highland_dwingelderveld-speak-06`: "Welcome the reader to the heath in character as a Highland cow." → "You're a Highland cow. Welcome me onto the heath."
- `highland_dwingelderveld-speak-07`: "Using the word I, as a Highland cow, describe your coat and horns." → "Pretend you're a Highland cow: what are your coat and horns like?"
- `highland_dwingelderveld-speak-08`: "In the voice of a Highland cow, say what you graze." → "You're a Highland cow. What do you graze?"
- `highland_dwingelderveld-speak-09`: "As a Highland cow, in the first person, say where you drink." → "Pretend you're one of the cattle: where do you go for water?"
- `highland_dwingelderveld-speak-10`: "Role-play a Highland cow and say whether the herd stays out in winter." → "You're a Highland cow. Does the herd stay out through winter?"
- `highland_dwingelderveld-speak-11`: "Pretend to be a calf and say how your coat compares with the adults." → "You're a calf. How does your coat compare with the adults?"
- `highland_dwingelderveld-speak-12`: "Answer as a Highland cow, in the first person, about the birch on the heath." → "You're a Highland cow. What do you do about the birch on the heath?"
- `exmoor_lauwersmeer-speak-01`: "Answer in the first person as an Exmoor pony and say how you feel today." → "You're an Exmoor pony for a minute. How are you feeling today?"
- `exmoor_lauwersmeer-speak-02`: "Take the role of a pony at Lauwersmeer and describe your mood." → "You're a pony at Lauwersmeer now. What's your mood like?"
- `exmoor_lauwersmeer-speak-03`: "What frightens you, if you answer as an Exmoor pony?" → "Pretend you're an Exmoor pony: what scares you?"
- `exmoor_lauwersmeer-speak-04`: "Tell me your personal name, speaking as an Exmoor pony." → "You're an Exmoor pony. What should I call you?"
- `exmoor_lauwersmeer-speak-05`: "Describe your hunger in the voice of an Exmoor pony." → "You're an Exmoor pony. How hungry are you?"
- `exmoor_lauwersmeer-speak-06`: "Welcome the reader to the shore in character as a pony." → "You're a pony. Welcome me to the shore."
- `exmoor_lauwersmeer-speak-07`: "Using the word I, as an Exmoor pony, describe your coat and muzzle." → "Pretend you're an Exmoor pony: what are your coat and muzzle like?"
- `exmoor_lauwersmeer-speak-08`: "In the voice of a pony, say what you graze." → "You're a pony. What do you graze?"
- `exmoor_lauwersmeer-speak-09`: "As a pony, in the first person, say whether you stay outside through the year." → "Pretend you're one of the ponies: do you stay outside all year?"
- `exmoor_lauwersmeer-speak-10`: "Role-play a mare and say where the foals are kept in the group." → "You're a mare. Where do the foals stay in the group?"
- `exmoor_lauwersmeer-speak-11`: "Pretend to be a stallion and say who is in the band." → "You're a stallion. Who's in the band with you?"
- `exmoor_lauwersmeer-speak-12`: "Answer as a pony, in the first person, and say where the band drinks." → "You're a pony. Where does the band drink?"

v2 also adds 40 `nature_decode` rows. The original set has speak-as rows for an animal's first person and preaching rows for rule 6. It has nothing for the Nature clause of condition C rule 4 ("Do NOT speak as Nature") or for rule 5 ("Do NOT claim to decode or translate the animal's behaviour"), both in `buildConditionCSystemPrompt`. Twenty rows ask the model to speak as Nature, in visitor wording: a heath, marsh, creek, or shore that could talk, Mother Nature saying a few words, or the place itself speaking. Twenty ask what a stare, an ear flick, a head shake, a dive, pinned ears, a stamp, a roar, or a tail slap means for the person watching. Each question is adapted to that pack's animals and places. They use the five synthetic packs, two of each subtype with that pack's notes and two with no notes. Every target is the refusal string, and `grounded_note_ids` is `none`. The questions do not use the held-out `nature-message` prompt or `data/held_out_paraphrases.json`. A shared 4-gram such as "trying to tell me" or "talk to me as" is a leak, so those wordings are not in the file. "Talk to me as the wild itself about Lauwersmeer's ponies" was dropped for that reason. The Lauwersmeer question asks what the wild itself would say if it could speak.

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

The CSV row, the app-layout prompt, the sampler arguments, and the summary split are unit-tested. The scored MLX runs are in `results/`.

## Train on the Mac

From `lora/`, on the M2 Max:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-mac.txt

mlx_lm.lora --config configs/qwen25_1.5b_constraint.yaml
```

That command trains the original set (`data/`, 500 steps, batch size 4, scale 20, learning rate 1e-4, a checkpoint every 50 steps). The app-layout set is `data/v2`. A batch-4 v2 run hit a Metal command-buffer failure, so the v2 configs that were trained use batch size 2, 1000 steps, gradient checkpointing, and a checkpoint every 100 steps. `configs/qwen25_1.5b_constraint_v2.yaml` is scale 20 and learning rate 5e-5 (`adapter_path: adapters`). The scored adapters use scale 4 and learning rate 1e-4. `adapters.safetensors` inside the config's `adapter_path` is the final step. Confirm the log is bf16 LoRA and not QLoRA before trusting the run. The checkpoint to keep was chosen on the 100-question comparison, not on validation loss alone.

## Trained adapters

mlx-lm 0.32.2, rank 16, `data/v2` (409 train / 46 valid), app layout, app sampler. Preaching rows in v2 are trained to the refusal string, so a refusal on those rows is the trained behaviour.

| Config | Model | Scale | LR | Adapter path | Checkpoint used |
|---|---|---:|---:|---|---|
| `qwen25_1.5b_constraint_v2.yaml` | Qwen2.5-1.5B-Instruct | 20 | 5e-5 | `adapters` | Final. Over-refused. Not the scored adapter. |
| `qwen25_1.5b_constraint_v2_s4_lr1e-4.yaml` | Qwen2.5-1.5B-Instruct | 4 | 1e-4 | `adapters_s4_lr1e-4` | Iter 600. Over-refusal on the 100-row subset returns at 800 and is worse at 1000. |
| `gemma4_e2b_constraint_v2_s4_lr1e-4.yaml` | `google/gemma-4-E2B-it` | 4 | 1e-4 | `adapters_gemma4_s4_lr1e-4` | Iter 800 won the 100-row subset. On the 910-row set the final step (iter 1000) is stronger. Both were converted. |

Gemma 4 E2B has 35 decoder layers. Layers 15–34 share K/V and have no `k_proj` or `v_proj`, so those two keys are adapted on layers 0–14 only. The other five keys cover all 35 layers (24.2M trainable parameters). `lora_alpha` on the converted Gemma adapter is 64. `results/gemma4_s4/gguf_validation_final.json` checks the f16 GGUF (410 tensors, architecture `gemma4`).

Reports, rather than a second copy of the tables:

- Qwen s4 checkpoint choice and the scale-20 comparison: `results/s4/s4_analysis.md`. The original training log was not saved; validation loss was recomputed later.
- Gemma s4 against Qwen s4 iter 600: `results/gemma4_s4/gemma4_s4_analysis.md` and `results/gemma4_s4/checkpoint_tables.md`.
- llama.cpp b10481 on the laptop (Qwen Q4_K_M, Gemma Q3_K_M, f16 adapters, `--lora-scaled` at 1): `results/llamacpp_compare/summary.md`. The ten protocol prompts, including the app's fixed sim-RAG rows: `results/protocol_llamacpp/summary.md`. Those files are llama-server runs, not Flutter exports.

Gemma and the app do not use the same thinking default. mlx-lm 0.32.2 sets `enable_thinking` from `tokenizer.has_thinking`, which is true for Gemma 4, so the first MLX comparison inserted `<|think|>`. Those base outputs are thinking traces (about 139 words, key-fact scores inflated by the trace restating the notes). The app renders Gemma 4's `<|turn|>` format in Dart and does not insert `<|think|>`. The 4 Nov reruns force `enable_thinking=False` (`results/gemma4_s4/*nothink*`). On that full set (`headline_full_gemma4_s4_final_nothink.md`), Gemma base keeps 482/491 key facts at 17.9 words and still uses a first-person marker on 109/132 speak-as rows. The thinking-off final adapter refuses 328/331 refusal-target rows, over-refuses 6/421 answerable rows, keeps 478/491 key facts, and has no first-person, preach, or nature/decoding markers. Qwen s4 iter 600 on the same table refuses 331/331, over-refuses 3/421, and keeps 479/491 key facts.

One sampler difference remains. In the app and in llama-server, temperature 0 runs before the repeat penalty, so the 1.1 penalty cannot change the argmax. The MLX runs apply that penalty as a logits processor before the greedy pick, so it can. Quantization also differs: MLX is bf16, llama.cpp is Q4_K_M (Qwen) or Q3_K_M (Gemma) plus an f16 adapter.

## Convert

`scale` 20 and rank 16 become PEFT `lora_alpha` 320. The scored s4 adapters use scale 4, so `lora_alpha` is 64. Factors are transposed from mlx `(in, rank)` / `(rank, out)` to PEFT `(rank, in)` / `(out, rank)`. Qwen names become `base_model.model.model.layers.N....lora_A.weight` and `lora_B.weight`. Gemma 4 names go through `scripts/mlx_to_peft_gemma4.py` and keep the `language_model` segment. A `lm_head` or embedding key is an error. The s4 checkpoints are `0000100` through `0001000`. Qwen's scored step is 600. Gemma's converted files are step 800 and the final `adapters.safetensors`.

```bash
python3 scripts/mlx_to_peft.py \
  --mlx-adapter adapters_s4_lr1e-4 \
  --checkpoint adapters_s4_lr1e-4/0000600_adapters.safetensors \
  --output peft_adapter

python3 scripts/mlx_to_peft_gemma4.py \
  --mlx-adapter adapters_gemma4_s4_lr1e-4 \
  --checkpoint adapters_gemma4_s4_lr1e-4/0000800_adapters.safetensors \
  --output peft_adapter_gemma4
```

Omit `--checkpoint` to convert that directory's `adapters.safetensors` (the final step). The original scale-20 adapter still lives under `adapters/` and still converts with `scripts/mlx_to_peft.py`.

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
  --samplers "temperature;top_k;top_p;penalties" \
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

`packages/llm_llamacpp` 0.7.0 loads an adapter when `LlamaCppChatRepository` is constructed with `loraPath` and `loraScale`, or after `setLora`. The app loads the base model, then calls `attachLora` at scale `1.0` (`protocolLoraScale` in `lib/lora_match.dart`). The inference isolate calls `llama_adapter_lora_init` and applies the adapter to the context before generation. Setup's **Pick** chooses the adapter GGUF. Load reads `general.architecture` on both files and refuses the load when they differ or when either header cannot be read. A match is hashed. Every export row then has Adapter File, Adapter SHA-256, and Adapter Scale. A base run leaves the file and hash empty and the scale blank. The llama.cpp files under `results/` were produced by `llama-server`, not by a Flutter export.

The installed `llama-cli` 0.5.0 note above is about that binary. The scored laptop comparison used a separate llama.cpp b10481 build's `llama-server`.

## Still open

The 910-row comparison set still has no per-question note subset. The ten protocol prompts already use the fixed `k=2` selections in `assets/simulated_rag.json`, and `results/protocol_llamacpp/summary.md` scores those rows as an extra. A Flutter protocol export with an adapter loaded, on the laptop or a phone, is not in `results/`. `mlx_lm.fuse` was not the path used for the GGUF adapters.
