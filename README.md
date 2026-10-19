# Technical Exploration: Local LLM Protocol Benchmark

Offline Flutter application for running the local on-device LLM evaluation protocol across 10 prompts, 10 local dataset notes, and 3 experimental conditions (plus an optional retrieval approximation pass).

Outputs are scored by hand post-hoc; non-functional metrics (timings, token rates, memory, thermal throttling, and battery) export directly into the evaluation measurement sheet and benchmark reports.

---

## 1. Protocol Architecture & Locked Conditions

The protocol uses the locked prompts and notes in the `assets/` directory:
- **`assets/prompts.json`**: Exactly 10 locked visitor prompts.
- **`assets/dataset.json`**: Exactly 10 locked local notes about the Kraansvlak wisent herd.
- **`assets/simulated_rag.json`**: Locked fixed top-$k$ note rankings ($k=2$) approximating retrieval, including near misses (`no-feeding` for `morning-meal`, `other-grazers` for `what-animal`).
- **`assets/system_prompt.txt`**: Base park visitor persona.

### Experimental Conditions
- **Condition A (Base only)**: System prompt only. No dataset notes are injected.
- **Condition B (Dataset)**: System prompt + all 10 dataset notes injected into context. No abstention hints.
- **Condition C (Constraints)**: System prompt + all 10 dataset notes + strict grounding + exact refusal rule (`"I do not have information about that in the approved notes."`) + 1–3 sentence limit + behavioural bans:
  1. Ban on speaking as Nature or in the animal's first-person voice.
  2. Ban on claiming to decode or translate the animal's behaviour.
  3. Ban on preaching or moralising at the visitor.
- **Simulated RAG Variants** (`B-sim-RAG` and `C-sim-RAG`):
  - Labelled as **"simulated RAG (fixed note selection)"**.
  - Off by default. When enabled, Condition B and C inject only the fixed top-$k$ notes ($k=2$) from `assets/simulated_rag.json`.
  - Prompts without a retrieval mapping receive 0 notes (no keyword-matching fallback).

### Reproducible Decoding Settings (Locked Protocol Defaults)
All runs execute with deterministic decoding settings to eliminate stochastic variance across passes:
- **Temperature**: `0.0` (greedy, deterministic decoding)
- **Seed**: `42`
- **Top-K**: `40`
- **Top-P**: `0.9`
- **Repeat Penalty**: `1.1`
- **Max Tokens**: `256`
- **Context Size**: `2048`
- **Reasoning / Thinking**: No thinking toggle is applied. The export column is `N/A`. Output sanitizing still strips a `<think>` block if a model emits one.

---

## 2. Measurement Methodology & Telemetry Formulas

This section provides the exact technical definitions, instrumentation hooks, and mathematical formulas used for all recorded metrics. You can cite or adapt these directly for technical reporting and research papers.

### 2.1 Time to First Token (TTFT)
* **Definition**: Latency in milliseconds from the moment prompt inference begins until the model produces its very first content token chunk.
* **Instrumentation**: Measured using Dart's high-resolution `Stopwatch` (`Stopwatch()..start()`). The timestamp is captured when the first non-empty text chunk arrives on the inference stream:
  $$\text{TTFT (ms)} = t_{\text{first\_token}} - t_{\text{inference\_start}}$$
* **Scope**: Includes model context initialization, prompt tokenization (`llama_tokenize`), prompt prefill / evaluation batching (`llama_decode`), and sampling the initial token.

### 2.2 Tokens Per Second (TPS / Decoding Rate)
* **Definition**: Generation throughput in tokens per second, measuring purely the autoregressive decoding phase.
* **Formula**:
  $$\text{TPS} = \frac{\text{evalCount}}{\Delta t_{\text{generation}}} = \frac{\text{evalCount}}{(t_{\text{done}} - t_{\text{first\_token}})}$$
* **Notes**:
  - TTFT / prompt evaluation latency is strictly excluded from $\Delta t_{\text{generation}}$ so that prompt length does not artificially penalize the generation speed metric.
  - If generation is completed or proactively stopped on a stop delimiter, the interval is measured up to the stopping point.

### 2.3 Memory & Resident Set Size (RSS)
Physical RAM consumption is sampled at multiple points using native OS interfaces:
* **Operating System Hooks**:
  - **Linux / Android**: Read directly from the kernel procfs:
    - `/proc/self/status` $\rightarrow$ `VmRSS` (instantaneous physical memory resident in RAM)
    - `/proc/self/status` $\rightarrow$ `VmHWM` (High Water Mark / peak physical memory tracked by the Linux kernel)
    - `/proc/meminfo` $\rightarrow$ `MemTotal` and `MemAvailable` (total and free system RAM)
  - **macOS / iOS**: Read via `ProcessInfo.currentRss` and `ProcessInfo.maxRss` (`getrusage` system call).
* **Tracked Metrics**:
  - **Baseline RAM (`ramBeforeLoadBytes`)**: Process RSS measured immediately before the model is loaded into memory.
  - **Idle Model RAM (`ramIdleBytes`)**: Process RSS measured after model loading and initial context allocation, following a settling pause.
  - **Model Memory Overhead**: $\Delta \text{RAM}_{\text{idle}} = \text{RAM}_{\text{idle}} - \text{RAM}_{\text{baseline}}$
  - **Peak RAM (`ramPeakBytes`)**: Peak resident memory sampled across the full inference lifecycle (sampled at inference start, at TTFT prefill completion, on each streamed token chunk, and at run finalization).

### 2.4 Thermal Throttling (%)
* **Definition**: Relative drop in generation throughput observed during sustained execution, indicating SoC hardware thermal throttling.
* **Instrumentation**: Measured exclusively during the dedicated 10-Run Heat Test (`who-manages`, Condition B) relative to Run 1:
  $$\text{Throttle}_i (\%) = \frac{\text{TPS}_{\text{Run 1}} - \text{TPS}_i}{\text{TPS}_{\text{Run 1}}} \times 100$$
* **Max Thermal Throttle**: The maximum sustained performance drop recorded across the entire 10-run session:
  $$\text{Max Throttle} (\%) = \max_{i \in [1, 10]} \left( \max(0, \text{Throttle}_i) \right)$$
  *(Speedups where $\text{TPS}_i > \text{TPS}_1$ yield $0\%$ throttle rather than negative percentages).*

### 2.5 Battery Level & Temperature
* **Android**: `BatteryManager.EXTRA_LEVEL` / `EXTRA_SCALE` for the percentage, and `EXTRA_TEMPERATURE` (tenths of a degree) divided by 10 for °C. These are the export values.
* **Linux**: battery sysfs only (`battery/capacity` or `BAT0/capacity`, and the matching `temp` file). `thermal_zone0` is not a battery temperature and is not exported.
* **SoC thermal zone**: On Android, each run logs the zone the old reader would have used (often `thermal_zone0`) with its type, raw value, and °C, plus the hottest zone. That log is logcat only (`soc-thermal`). It is what produced the 92–95 °C figures and it is how a throttled run is tied to TTFT and tok/s. It is not an export column.
* Captured immediately before (`batteryLevelBefore`, `batteryTempBeforeCelsius`) and after each run (`batteryLevelAfter`, `batteryTempAfterCelsius`).

### 2.6 Token Counting & Context Tracking
* **`promptTokenCount`**: Exact number of input tokens tokenized by the model's native vocabulary via `llama_tokenize`.
* **`generatedTokenCount` (`evalCount`)**: Exact number of output tokens produced during generation, reported by llama.cpp.
* **`hitMaxTokens`**: Boolean flag set to `true` if `evalCount >= maxTokens (256)`, indicating the generation hit the protocol ceiling rather than an end-of-turn token.

---

## 3. How Application Features Work

### 3.1 Single Condition Execution
1. Select a prompt from the catalog dropdown dialog.
2. Select an experimental condition (A, B, or C).
3. Tap **Run (<Condition>)**.
4. The app builds the condition messages, streams inference chunks into the live output card, samples memory, computes TTFT and TPS, and appends the result to history.

### 3.2 Compare All (A vs B vs C / All 5)
- Executes all active conditions for the currently selected prompt sequentially.
- Automatically clears prior comparison cards at the start so only the current prompt's outputs are shown.
- Renders a multi-tab comparison card (`ComparisonView`) with side-by-side metrics (TTFT, TPS, Tokens, Peak RAM) and one-click copy buttons for prompt and response comparison.

### 3.3 Protocol Batch Pass (30 runs or 50 runs)
Designed for hands-off, automated protocol data collection:
- **Core Mode (30 runs)**: 10 prompts × 3 conditions (A $\rightarrow$ B $\rightarrow$ C).
- **Extended Mode (50 runs)**: 10 prompts × 5 conditions (A $\rightarrow$ B $\rightarrow$ B-sim $\rightarrow$ C $\rightarrow$ C-sim).
- **Execution Workflow**:
  ```mermaid
  flowchart TD
      A["Start Batch Pass"] --> B["Select Prompt i"]
      B --> C["Clear Comparison Cache"]
      C --> D["Run Condition A"]
      D --> E["Run Condition B"]
      E --> F["Run Condition C"]
      F --> G{"More Prompts?"}
      G -- Yes --> B
      G -- No --> H["Batch Complete"]
  ```
- **Context Isolation**: Every individual run initializes a fresh `llama_context` in the persistent worker isolate, preventing KV-cache leakage or prompt contamination between runs.
- **Inter-run Stabilisation**: A 300 ms pause is inserted between runs to allow memory reclamation and thermal stabilization.
- **Clean Interruption**: Tapping **Stop** immediately cancels the active inference stream, safely stops the batch loop, and reports the exact number of completed runs.

### 3.4 10-Run Heat Test
- Dedicated thermal stress test executing the identical prompt and condition (`who-manages`, Condition `B`) 10 times consecutively with a brief 200 ms interval.
- Isolates thermal and battery degradation under sustained load on mobile SoCs (e.g. Fairphone 5 or Samsung Galaxy Fold).
- Computes per-run throttle and displays the **Peak Max Thermal Throttle** observed across the test.

### 3.5 Stop Token Handling & Output Sanitization
- Models emitting split multi-token stop sequences (such as `<|im_end|>`, `<end_of_turn>`, `<|eot_id|>`, `<turn|>`, or `<|im_start|>`) are proactively intercepted during streaming.
- Generation is stopped early when a delimiter appears, preventing runaway synthetic multi-turn dialogs (e.g., generating fake user and assistant turns).
- Output is sanitized via `sanitizeModelOutput()`, which strips reasoning tags and truncates cleanly at the first stop delimiter.

### 3.6 Persistence & History Migration
- Runs are immediately appended to `runs_history.jsonl` on disk as each inference completes.
- On macOS, `RunPersistenceManager.loadRuns()` automatically checks for legacy runs recorded under previous sandboxed container paths (`~/Library/Containers/...`) and merges them into the active `~/Documents/runs_history.jsonl`, deduplicating records by timestamp.

---

## 4. Telemetry Export & Spreadsheet Column Mapping

Tapping the download icon in the AppBar provides instant export:
- **Export Files (CSV & JSON)**: Saves timestamped files to the `/exports` directory.
- **Copy CSV to Clipboard**: Formatted for direct pasting into the evaluation spreadsheet.
- **Copy JSON to Clipboard**: Complete raw structured JSON.

### CSV Column Reference Dictionary
| Column Name | Type | Description |
|---|---|---|
| `Run Index` | int | Monotonic run index across the entire session |
| `Run Since Load` | int | Counter of runs executed since the current model was loaded |
| `Load Type` | string | `Cold` only for protocol run 1 after a cold load when sanity did not run. When the sanity prompts ran, every protocol row is `Warm`, including run 1. |
| `Timestamp` | ISO8601 | Exact local timestamp of run completion |
| `Device Model` | string | Hardware identifier (e.g. `Fairphone 5`, `Mac14,6`) |
| `OS Version` | string | Operating system name and kernel/build version |
| `Model File` | string | GGUF file name |
| `Architecture` | string | Model architecture from GGUF metadata (e.g. `qwen2`, `gemma4`) |
| `Quantization` | string | Quantization type (e.g. `Q4_K_M`, `Q3_K_M`) |
| `Backend` | string | Compute backend (`GPU` or `CPU`) |
| `Condition` | string | Condition code (`A`, `B`, `C`, `B-sim-RAG`, `C-sim-RAG`) |
| `Notes Mode` | string | Dataset injection mode (`none`, `all`, `sim-rag`) |
| `k` | int | Retrieval note count ($k=0$ for A/B/C; $k=2$ for sim-RAG) |
| `Injected Notes` | string | Semicolon-delimited list of injected note IDs |
| `Prompt ID` | string | Identifier from `prompts.json` (e.g. `speak-as`, `who-manages`) |
| `Prompt Text` | string | Full visitor question submitted to the model |
| `Temperature` | double | Decoding temperature (`0.0`) |
| `Seed` | int | Deterministic seed (`42`) |
| `Top-K` | int | Top-K sampling threshold (`40`) |
| `Top-P` | double | Top-P nucleus sampling threshold (`0.9`) |
| `Repeat Penalty`| double | Repetition penalty (`1.1`) |
| `Max Tokens` | int | Protocol maximum output token limit (`256`) |
| `Context Size` | int | Context window size (`2048`) |
| `Prompt Tokens` | int | Exact count of prompt tokens evaluated |
| `Gen Tokens` | int | Exact count of response tokens generated (`evalCount`) |
| `Hit Max Tokens`| string | `Yes` if output reached 256 tokens; `No` if stopped normally |
| `TTFT (ms)` | int | Time to first token in milliseconds |
| `Gen Duration (ms)`| int | Duration of token generation phase excluding TTFT |
| `TPS (tok/s)` | double | Generation rate in tokens per second |
| `RAM Peak (MB)` | double | Peak resident memory during inference in megabytes |
| `Battery Before (%)`| int | Device battery level before run (mobile) |
| `Battery After (%)` | int | Device battery level after run (mobile) |
| `Temp Before (°C)`| double | Battery temperature before the run, from `BatteryManager` on Android |
| `Temp After (°C)` | double | Battery temperature after the run, from `BatteryManager` on Android |
| `Throttle (%)` | double | Thermal throttle percentage (Heat Test only) |
| `Is Heat Test` | string | `Yes` if part of 10-run heat test; `No` otherwise |
| `Thinking Disabled`| string | `N/A` unless a thinking toggle was actually applied for that model. This app does not apply one. |
| `Output` | string | Sanitized model response text |

---

## 5. Known limitations

Gemma 4 E2B runs on a ChatML fallback, not its real chat template. The GGUF ships an 18 KB Jinja template (`<|turn|>` / `<turn|>`). `llama_chat_apply_template` is not a Jinja parser and does not recognise that template, and this build does not link `common/jinja`, so both the token counter and the inference isolate fall back to ChatML. Qwen is unaffected: its template is ChatML and the logged prompt lengths match the Mac batch. The fallback can change how Gemma follows the visitor notes, and it may explain some of Gemma's over-refusals. It is a validity limitation of the 11 Oct Gemma batch. Protocol message text is unchanged. Switching Gemma onto its real template would make new Gemma answers incomparable with that batch, so this round does not do it.

After every load the app runs two fixed sanity prompts ("Name one European wild animal in three words." and "What is 2+2?"). The animal answer must contain a recognisable animal word, so Latin-script soup fails as well as mixed-script soup. The math answer must contain 4. A failure blocks the protocol batch. Those prompts exercise the loaded model, so when they ran every protocol row is `Warm` (`isColdLoad` false), including run 1. A Mac-only "Skip sanity check" toggle, off by default, skips them so the first protocol run after a cold load can still be `Cold`. The toggle is ignored on Android.

Load does not hash the GGUF. Hashing reads the whole file and warms the page cache, which makes the following load time incomparable. "Verify model" hashes once and stores the result in `model_sha256_cache.json` in the exports directory, keyed by path, size, and mtime. A later load of that same file reuses the stored hash and still blocks a known-Qwen mismatch. An unverified file loads and is logged as `sha256=unverified`. Verifying and then immediately loading still warms the cache.

The CPU configuration for a batch (winning backend, KleidiAI SME, threads, KV type, flash attention, mmap, GPU layers) is printed to logcat and shown under the model line. It is not a new export column. `variant=` is the basename of each registered `libggml-cpu-*.so` (`dladdr` on the backend's `ggml_backend_set_n_threads`), not the `CPU` device name. Threads, KV type, and flash attention are the values this app's repository passes. It does not pass them unless a build define says so, so the line says `unset`. It does not read `llama_context_default_params`. `sme=` is the value this process set with `setenv` before backend init, or `unset`.

## 6. Building and Running on Devices

### macOS / Desktop
```bash
flutter run -d macos --release
```

### Android (Fairphone 5 & Galaxy Fold)
The Android application uses the package ID `nl.wisent.technicalexploration`.
Storage access uses `FileType.any` with user file selection, allowing `.gguf` files to be selected directly from device storage.

1. **Build APK**:
   ```bash
   flutter build apk --release
   ```
   Output: `build/app/outputs/flutter-apk/app-release.apk`

   Two extra release builds for the Fold. Each also writes `app-release.apk`; keep the distinctly named copy, because the next build overwrites `app-release.apk`.

   KleidiAI SME off. `GGML_KLEIDIAI_SME=0` is set with `setenv` before backend init:
   ```bash
   flutter build apk --release --dart-define=GGML_KLEIDIAI_SME=0
   ```
   Output: `build/app/outputs/flutter-apk/app-sme0-release.apk`

   No `libggml-cpu-android_armv9.2_*.so` (the armv9.0 backend and the others stay in the APK):
   ```bash
   flutter build apk --release --dart-define=EXCLUDE_ARMV92=1
   ```
   Output: `build/app/outputs/flutter-apk/app-no-armv92-release.apk`

   Those two APKs both failed sanity: each prompt hit the token cap. SME KleidiAI and the armv9.2 libraries are ruled out on their own. Do not treat them as the next protocol build.

   The next three builds each change one thing. `app-release.apk` is still overwritten; keep the named copy. `packages/llm_llamacpp` is the vendored 0.7.0 package. The nommap APK is the only one that sets `load_mode` to `NONE` in the inference isolate. The others leave llama.cpp's `AUTO` load mode.

   No mmap. Threads stay unset, every CPU library stays, GPU layers stay 0:
   ```bash
   flutter build apk --release --dart-define=NO_MMAP=true
   ```
   Output: `build/app/outputs/flutter-apk/app-nommap-release.apk`

   Only the armv8.0 and armv8.2 CPU libraries, with `threads=4` passed into the repository:
   ```bash
   flutter build apk --release --dart-define=CPU_LIBS=armv8 --dart-define=N_THREADS=4
   ```
   Output: `build/app/outputs/flutter-apk/app-armv8-threads4-release.apk`

   Same libraries, with the thread count left unset:
   ```bash
   flutter build apk --release --dart-define=CPU_LIBS=armv8
   ```
   Output: `build/app/outputs/flutter-apk/app-armv8-release.apk`

   Vulkan diagnostic. `nGpuLayers=99` is not the protocol default. Run the sanity prompts only:
   ```bash
   flutter build apk --release --dart-define=N_GPU_LAYERS=99
   ```
   Output: `build/app/outputs/flutter-apk/app-vulkan99-release.apk`

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

   Installing a newer APK over the installed app keeps the run history. `runs_history.jsonl` is in the app documents directory, and Android leaves that directory in place when the package id and signing key stay the same and the app is not uninstalled. Uninstall deletes it.

   Export Files on Android still writes that private copy, then copies the finished CSV, JSON, and `model_sha256_cache.json` (when the cache exists) to `Download/technical-exploration/`. That folder shows up in the Files app and in Windows Explorer over USB. The snackbar names the files. Share sends those copies through the Android share sheet. A name that is already in that folder is overwritten for the hash cache and given a numeric suffix for a timestamped CSV or JSON. Desktop export is unchanged.
