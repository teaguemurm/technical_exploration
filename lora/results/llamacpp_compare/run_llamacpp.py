#!/usr/bin/env python3
"""Run one variant (model x adapter x condition) over the comparison set with
llama-server from llama.cpp b10481 (commit 25ae3a9b, the version in llm_llamacpp 0.7.0).

Prompt text = common.render (app formatter; equal to the HF template for all rows,
see template_check.json), sent as a raw prompt to /completion (special tokens parsed).
Gemma: the literal '<bos>' is removed from the text and the server adds BOS
(add_special); the smoke test checks the token ids are identical to the app's
(text with '<bos>', add_special=false).
Sampler = app chain (_configureSampler): temperature 0 -> top_k 40 -> top_p 0.9
(min_keep 1) -> penalties(last_n 64, repeat 1.1, freq 0, presence 0) -> dist(seed 42),
n_predict 256, n_ctx 2048, n_batch 2048. No min-p/dry/xtc/typical. cache_prompt=false
so every row is evaluated from an empty KV cache (as the app does per run).
Output = sanitize() (lib/messages.dart sanitizeModelOutput) of the returned text.
Writes out_<name>.jsonl (+ .meta.json, server log)."""
import argparse, json, os, signal, subprocess, sys, time, urllib.request
from pathlib import Path
D = Path(__file__).resolve().parent
sys.path.insert(0, str(D)); import common as C

LLAMA = Path.home() / "llama.cpp-b10481/build/bin/llama-server"
p = argparse.ArgumentParser()
p.add_argument("--family", choices=["qwen", "gemma"], required=True)
p.add_argument("--gguf", required=True)
p.add_argument("--lora", default=None)
p.add_argument("--lora-scale", type=float, default=1.0)
p.add_argument("--cond", choices=list("ABC"), required=True)
p.add_argument("--name", required=True)
p.add_argument("--limit", type=int, default=None)
p.add_argument("--port", type=int, default=8791)
p.add_argument("--outdir", default=str(D / "raw"))
a = p.parse_args()
out_dir = Path(a.outdir); out_dir.mkdir(parents=True, exist_ok=True)
rows = C.ROWS[: a.limit] if a.limit else C.ROWS

cmd = [str(LLAMA), "-m", a.gguf, "-c", "2048", "-b", "2048", "-np", "1", "-ngl", "99",
       "--seed", "42", "--port", str(a.port), "--host", "127.0.0.1", "--no-webui"]
if a.lora:
    cmd += ["--lora-scaled", f"{a.lora}:{a.lora_scale:g}"]
log_path = out_dir / f"server_{a.name}.log"
log = open(log_path, "w")
srv = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
base = f"http://127.0.0.1:{a.port}"

def post(path, body, timeout=600):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())

try:
    t_load = time.time()
    for _ in range(600):
        try:
            with urllib.request.urlopen(base + "/health", timeout=2) as r:
                if json.loads(r.read()).get("status") == "ok": break
        except Exception:
            pass
        if srv.poll() is not None: raise SystemExit(f"server exited, see {log_path}")
        time.sleep(0.5)
    load_s = time.time() - t_load
    loras = json.loads(urllib.request.urlopen(base + "/lora-adapters").read())
    stops = C.GEMMA4_STOPS + [s for s in C.APP_STOPS if s not in C.GEMMA4_STOPS] if a.family == "gemma" else C.APP_STOPS
    sampling = dict(n_predict=256, temperature=0.0, top_k=40, top_p=0.9, min_p=0.0, min_keep=1,
                    repeat_penalty=1.1, repeat_last_n=64, frequency_penalty=0.0, presence_penalty=0.0,
                    samplers=["temperature", "top_k", "top_p", "penalties"], seed=42,
                    dry_multiplier=0.0, xtc_probability=0.0, typical_p=1.0, top_n_sigma=-1.0,
                    stop=stops, cache_prompt=False, stream=False)
    # BOS check on the first row (Gemma): app tokens vs what we send
    first = C.render(rows[0], a.cond, a.family)
    bos_check = None
    if a.family == "gemma":
        app_ids = post("/tokenize", {"content": first, "add_special": False, "parse_special": True})["tokens"]
        sent_ids = post("/tokenize", {"content": first[len("<bos>"):], "add_special": True, "parse_special": True})["tokens"]
        bos_check = {"identical": app_ids == sent_ids, "n": len(app_ids), "first": app_ids[:3]}
        if app_ids != sent_ids: raise SystemExit(f"BOS/token mismatch {app_ids[:5]} vs {sent_ids[:5]}")
    else:
        ids = post("/tokenize", {"content": first, "add_special": True, "parse_special": True})["tokens"]
        ids0 = post("/tokenize", {"content": first, "add_special": False, "parse_special": True})["tokens"]
        bos_check = {"identical": ids == ids0, "n": len(ids), "first": ids[:3]}
    # resume: keep complete rows already written (each row is independent: cache_prompt=false)
    out_path = out_dir / f"out_{a.name}.jsonl"
    prev = []
    if out_path.exists():
        for l in out_path.read_text(encoding="utf-8").splitlines():
            try: prev.append(json.loads(l))
            except Exception: break
    want = [r["metadata"]["id"] for r in rows]
    prev = [x for x, w in zip(prev, want) if x.get("id") == w]
    out_path.write_text("".join(json.dumps(x) + "\n" for x in prev), encoding="utf-8")
    done = {x["id"] for x in prev}
    n_gen = sum(x.get("predicted_n") or 0 for x in prev); max_prompt = max([x.get("prompt_n") or 0 for x in prev] + [0])
    stopped_limit = sum(x.get("stop_type") == "limit" for x in prev)
    if prev: print(f"{a.name}: resuming after {len(prev)} rows", flush=True)
    out = open(out_path, "a")
    t0 = time.time(); n_new = 0
    for r in rows:
        if r["metadata"]["id"] in done: continue
        n_new += 1
        if n_new % 50 == 0: print(f"{a.name}: {len(done) + n_new}/{len(rows)} rows, {time.time() - t0:.0f}s", flush=True)
        prompt = C.render(r, a.cond, a.family)
        sent = prompt[len("<bos>"):] if a.family == "gemma" else prompt
        res = post("/completion", dict(prompt=sent, **sampling))
        raw = res.get("content", "")
        tm = res.get("timings", {})
        n_gen += tm.get("predicted_n", 0); max_prompt = max(max_prompt, tm.get("prompt_n", 0))
        stopped_limit += int(bool(res.get("stop_type") == "limit"))
        out.write(json.dumps({"id": r["metadata"]["id"], "output": C.sanitize(raw, a.family), "raw": raw,
                              "stop_type": res.get("stop_type"), "stopping_word": res.get("stopping_word"),
                              "prompt_n": tm.get("prompt_n"), "predicted_n": tm.get("predicted_n")}) + "\n")
        out.flush()
    el = time.time() - t0
    meta = {"name": a.name, "family": a.family, "gguf": a.gguf, "lora": a.lora, "lora_scale": a.lora_scale if a.lora else None,
            "cond": a.cond, "rows": len(rows), "seconds": round(el, 1), "rows_this_session": n_new, "resumed_after_rows": len(prev), "server_load_seconds": round(load_s, 1),
            "generated_tokens": n_gen, "max_prompt_tokens": max_prompt, "stopped_at_n_predict": stopped_limit,
            "lora_adapters_endpoint": loras, "bos_check": bos_check, "server_cmd": cmd,
            "sampling": {k: v for k, v in sampling.items()}}
    out.close()
    json.dump(meta, open(out_dir / f"out_{a.name}.meta.json", "w"), indent=1)
    print(f"{a.name}: {len(rows)} rows in {el:.1f}s, {n_gen} tokens, max prompt {max_prompt}, loras={loras}", flush=True)
finally:
    srv.send_signal(signal.SIGTERM)
    try: srv.wait(20)
    except Exception: srv.kill()
