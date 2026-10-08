# Technical Exploration: Local LLM Protocol Benchmark

Offline Flutter application for running the local on-device LLM evaluation protocol across 10 prompts, 10 local dataset notes, and 3 experimental conditions (plus an optional retrieval approximation pass).

Outputs are scored by hand post-hoc; non-functional metrics (timings, token rates, memory, thermal throttling, and battery) export directly into the evaluation measurement sheet.

---

## 1. Protocol Architecture & Locked Conditions

The protocol uses the locked prompts and notes in the `assets/` directory:
- **`assets/prompts.json`**: Exactly 10 locked visitor prompts.
- **`assets/dataset.json`**: Exactly 10 locked local notes about the Kraansvlak wisent herd.
- **`assets/simulated_rag.json`**: Locked fixed top-$k$ note rankings ($k=2$) approximating retrieval, including near misses (`no-feeding` for `morning-meal`, `other-grazers` for `what-animal`).
- **`assets/system_prompt.txt`**: Base park visitor persona.

### Conditions
- **Condition A (Base only)**: System prompt only. No dataset notes.
- **Condition B (Dataset)**: System prompt + all 10 dataset notes. No abstention hints.
- **Condition C (Constraints)**: System prompt + all 10 dataset notes + strict grounding + exact refusal rule (`"I do not have information about that in the approved notes."`) + 1–3 sentence limit + behavioural bans:
  1. Ban on speaking as Nature or in the animal's first-person voice.
  2. Ban on claiming to decode or translate the animal's behaviour.
  3. Ban on preaching or moralising at the visitor.
- **Simulated RAG Variants** (`B-sim-RAG` and `C-sim-RAG`):
  - Labelled as **"simulated RAG (fixed note selection)"**.
  - Off by default. When enabled, Condition B and C inject only the fixed top-$k$ notes ($k=2$) from `assets/simulated_rag.json`.
  - Nothing in the B variants hints at abstaining. Prompts without a list receive 0 notes (no keyword-matching fallback).

### Reproducible Decoding Settings (Locked Protocol Defaults)
- **Temperature**: `0.0` (greedy, deterministic decoding; any sampled pass is a separate, explicitly labelled option)
- **Seed**: `42`
- **Top-K**: `40`
- **Top-P**: `0.9`
- **Repeat Penalty**: `1.1`
- **Max Tokens**: `256`
- **Context Size**: `2048`
- **Reasoning / Thinking**: Suppressed for Qwen3 / reasoning fallback models (`/no_think` appended, `<think>` blocks stripped).

---

## 2. Step-by-Step Protocol Execution

### Step 1: Cold Model Load
1. Launch the app.
2. Tap **Pick GGUF** (or enter the path directly in device storage).
3. Tap **Load**.
4. The first load after launching the app is recorded as **Cold Load** (`isColdLoad = true`).
5. The app measures absolute Process RSS memory before load, load duration in milliseconds, and idle RAM overhead with the model loaded.
6. *Note*: If you load another model later, the app first unloads and disposes the current model, pauses briefly for memory settling, measures pre-load RAM, and records the load as **Warm Load** (`isColdLoad = false`).

### Step 2: Running the Protocol Batch

#### Core Protocol Pass (30 runs: 10 prompts × A/B/C)
1. Ensure the **"Simulated RAG (fixed note selection)"** toggle is **OFF**.
2. Tap **"Run Batch Pass (30 runs)"**.
3. The app executes all 10 prompts in locked sequence, grouped by prompt:
   $$\text{Prompt } i \longrightarrow \text{Condition A} \longrightarrow \text{Condition B} \longrightarrow \text{Condition C}$$
4. Every run record is persisted to disk immediately as it finishes. Live progress and a **Stop** button are displayed.

#### Extended Protocol Pass with Simulated RAG (50 runs)
1. Toggle **ON** **"Simulated RAG (fixed note selection)"**.
2. Tap **"Run Batch Pass (50 runs)"**.
3. The app executes all 10 prompts in prompt-grouped order:
   $$\text{Prompt } i \longrightarrow \text{A} \longrightarrow \text{B} \longrightarrow \text{B-sim-RAG} \longrightarrow \text{C} \longrightarrow \text{C-sim-RAG}$$
4. All records save with condition tags `B-sim-RAG` and `C-sim-RAG`, notes mode `sim-rag`, and retrieval $k=2$.

### Step 3: Running the 10-Run Heat Test
To measure sustained thermal throttling and SoC battery draw:
1. Load the model (warm load).
2. Tap **"10-Run Heat Test"**.
3. The app executes the same prompt and condition (`who-manages`, Condition `B`) 10 times back to back.
4. Battery level (%) and battery temperature (°C) are captured before and after each run.
5. Thermal throttle percentage is calculated as token rate relative to Run 1 of that heat test:
   $$\text{Throttle } \% = \frac{\text{TPS}_{\text{Run 1}} - \text{TPS}_{\text{Current}}}{\text{TPS}_{\text{Run 1}}} \times 100$$
   *(Throttle % is computed strictly within this heat test, never across the protocol batch).*

---

## 3. Telemetry & Exporting Results

Every run saves to disk immediately in `runs_history.jsonl` within the application documents directory and survives closing the app.

### Exporting Files
Tap the download icon in the AppBar:
- **Export to Files (CSV & JSON)**: Writes `technical_exploration_runs_<timestamp>.csv` and `.json` to device storage.
- **Copy CSV to Clipboard**: Copies complete CSV data formatted to paste directly into the evaluation spreadsheet:
  - **Tab 1: Load and idle**: Device Model, OS Version, Model File, Architecture, Quantization, Backend, GPU Layers, Load Type, Load Time (ms), RAM Before Load (MB), RAM Idle Model (MB), RAM Overhead (MB), Timestamp.
  - **Tab 2: Inference**: Run Index, Run Since Load, Is Cold Load, Timestamp, Device Model, OS Version, Model File, Architecture, Quantization, Backend, Condition, Notes Mode, k, Injected Notes, Prompt ID, Prompt Text, Temperature, Seed, Top-K, Top-P, Repeat Penalty, Max Tokens, Context Size, Prompt Tokens, Gen Tokens, Hit Max Tokens, TTFT (ms), Gen Duration (ms), TPS (tok/s), RAM Peak (MB), Battery Level Before (%), Battery Level After (%), Battery Temp Before (°C), Battery Temp After (°C), Throttle (%), Is Heat Test, Thinking Disabled, Output.
- **Copy JSON to Clipboard**: Copies raw structured JSON.
- **Clear History**: Clears the persisted disk log for a fresh session.

---

## 4. Building and Running on Devices

### macOS / Desktop
```bash
flutter run -d macos --release
```

### Android (Fairphone 5 & Galaxy Fold)
The Android application uses the package ID `nl.wisent.technicalexploration`.
Storage access uses `FileType.any` with `READ_EXTERNAL_STORAGE` (SDK $\le$ 32), allowing `.gguf` files to be selected directly from the `Downloads` directory without being hidden.

1. **Build APK**:
   ```bash
   flutter build apk --release
   ```
   Output: `build/app/outputs/flutter-apk/app-release.apk`

2. **Push GGUF Model to Phone**:
   ```bash
   adb push path/to/model.gguf /sdcard/Download/
   ```

3. **Install and Run**:
   ```bash
   adb install build/app/outputs/flutter-apk/app-release.apk
   # Or run directly with live console logs:
   flutter run -d <android-device-id> --release
   ```
