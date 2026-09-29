# technical_exploration

One offline screen. Pick a local GGUF, load it, choose a prompt from `assets/prompts.json`, and optionally attach every line in `assets/dataset.json`.

The expected model is Qwen2.5-1.5B-Instruct, GGUF, Q4_K_M. It is not downloaded and not included in this project.

```bash
flutter run -d macos
```

Edit `assets/system_prompt.txt`, `assets/prompts.json`, and `assets/dataset.json` without changing Dart. Restart the app so the asset bundle picks up the files.
