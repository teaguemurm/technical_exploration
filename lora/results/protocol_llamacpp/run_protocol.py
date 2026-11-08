#!/usr/bin/env python3
"""One model x adapter variant over the 10 protocol prompts x conditions
(A, B, B-sim-RAG, C, C-sim-RAG, app batch order), llama.cpp b10481 llama-server.
Server flags, sampler chain, stops, BOS handling and sanitize are identical to
../llamacpp_compare/run_llamacpp.py (app settings). Resumable: rows already in
raw/out_<name>.jsonl are kept and skipped (cache_prompt=false, rows independent)."""
import argparse, json, signal, subprocess, sys, time, urllib.request
from pathlib import Path
D = Path(__file__).resolve().parent; sys.path.insert(0, str(D))
import pcommon as P
C = P.C
LLAMA = Path.home() / "llama.cpp-b10481/build/bin/llama-server"
ap = argparse.ArgumentParser()
ap.add_argument("--family", choices=["qwen", "gemma"], required=True)
ap.add_argument("--gguf", required=True); ap.add_argument("--lora", default=None)
ap.add_argument("--name", required=True); ap.add_argument("--port", type=int, default=8792)
a = ap.parse_args()
out_dir = D / "raw"; out_dir.mkdir(exist_ok=True)
cmd = [str(LLAMA), "-m", a.gguf, "-c", "2048", "-b", "2048", "-np", "1", "-ngl", "99",
       "--seed", "42", "--port", str(a.port), "--host", "127.0.0.1", "--no-webui"]
if a.lora: cmd += ["--lora-scaled", f"{a.lora}:1"]
log_path = out_dir / f"server_{a.name}.log"
srv = subprocess.Popen(cmd, stdout=open(log_path, "w"), stderr=subprocess.STDOUT)
base = f"http://127.0.0.1:{a.port}"
def post(path, body):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r: return json.loads(r.read())
try:
    for _ in range(600):
        try:
            with urllib.request.urlopen(base + "/health", timeout=2) as r:
                if json.loads(r.read()).get("status") == "ok": break
        except Exception: pass
        if srv.poll() is not None: raise SystemExit(f"server exited, see {log_path}")
        time.sleep(0.5)
    loras = json.loads(urllib.request.urlopen(base + "/lora-adapters").read())
    stops = C.GEMMA4_STOPS + [s for s in C.APP_STOPS if s not in C.GEMMA4_STOPS] if a.family == "gemma" else C.APP_STOPS
    sampling = dict(n_predict=256, temperature=0.0, top_k=40, top_p=0.9, min_p=0.0, min_keep=1,
                    repeat_penalty=1.1, repeat_last_n=64, frequency_penalty=0.0, presence_penalty=0.0,
                    samplers=["temperature", "top_k", "top_p", "penalties"], seed=42,
                    dry_multiplier=0.0, xtc_probability=0.0, typical_p=1.0, top_n_sigma=-1.0,
                    stop=stops, cache_prompt=False, stream=False)
    out_path = out_dir / f"out_{a.name}.jsonl"
    prev = []
    if out_path.exists():
        for l in out_path.read_text(encoding="utf-8").splitlines():
            try: prev.append(json.loads(l))
            except Exception: break
    out_path.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in prev), encoding="utf-8")
    done = {(x["condition"], x["prompt_id"]) for x in prev}
    if prev: print(f"{a.name}: resuming after {len(prev)} rows", flush=True)
    out = open(out_path, "a", encoding="utf-8"); bos_ok = True
    for p in P.PROMPTS:
        for cond in P.CONDS:
            if (cond, p["id"]) in done: continue
            prompt = P.render(cond, p, a.family)
            if a.family == "gemma":
                sent = prompt[len("<bos>"):]
                app_ids = post("/tokenize", {"content": prompt, "add_special": False, "parse_special": True})["tokens"]
                sent_ids = post("/tokenize", {"content": sent, "add_special": True, "parse_special": True})["tokens"]
                if app_ids != sent_ids: raise SystemExit(f"BOS/token mismatch {cond} {p['id']}")
            else:
                sent = prompt
            t0 = time.time()
            res = post("/completion", dict(prompt=sent, **sampling))
            wall = time.time() - t0
            raw = res.get("content", ""); tm = res.get("timings", {})
            out.write(json.dumps({"name": a.name, "condition": cond, "prompt_id": p["id"], "prompt_text": p["text"],
                                  "output": C.sanitize(raw, a.family), "raw": raw, "stop_type": res.get("stop_type"),
                                  "stopping_word": res.get("stopping_word"), "prompt_n": tm.get("prompt_n"),
                                  "predicted_n": tm.get("predicted_n"), "prompt_ms": tm.get("prompt_ms"),
                                  "predicted_ms": tm.get("predicted_ms"), "wall_s": round(wall, 3)}, ensure_ascii=False) + "\n")
            out.flush()
    out.close()
    json.dump({"name": a.name, "family": a.family, "gguf": a.gguf, "lora": a.lora, "lora_adapters_endpoint": loras,
               "server_cmd": cmd, "sampling": sampling, "rows": len(P.PROMPTS) * len(P.CONDS)},
              open(out_dir / f"out_{a.name}.meta.json", "w"), indent=1)
    print(f"{a.name}: done, loras={loras}", flush=True)
finally:
    srv.send_signal(signal.SIGTERM)
    try: srv.wait(20)
    except Exception: srv.kill()
