#!/usr/bin/env python3
"""outputs.csv, side_by_side.md, summary.md, app_match.json from raw/out_*.jsonl.
Marker predicates are imported from ../s4/analyze_s4.py (is_exact_refusal,
contains_refusal, fp_hit, preach_hit, nature_hit, sentences, ngram_rep, long_out,
truncated). Key-fact markers per dataset fact id are defined here (heuristic)."""
import csv, json, re, sys, difflib
from pathlib import Path
D = Path(__file__).resolve().parent; sys.path.insert(0, str(D)); sys.path.insert(0, str(D.parent / "s4"))
import pcommon as P
import analyze_s4 as A
MODELS = {"qwen": "Qwen2.5-1.5B-Instruct Q4_K_M", "gemma": "Gemma 4 E2B-it Q3_K_M"}
ADAPTER = {"qwen": "qwen25-1.5b-constraint-s4-lora-ckpt600-f16.gguf", "gemma": "gemma4-e2b-constraint-s4-lora-f16.gguf (final)"}
rows = []
for fam in ["qwen", "gemma"]:
    for var in ["base", "lora"]:
        f = D / f"raw/out_{fam}_{var}.jsonl"
        if f.exists():
            for l in f.read_text(encoding="utf-8").splitlines():
                r = json.loads(l); r["model"] = fam; r["variant"] = var; rows.append(r)
ref = json.load(open(D / "app_reference.json"))
PBY = {p["id"]: p for p in P.PROMPTS}

# --- key-fact markers (heuristic, all must match for a fact to count) ---
FACT_RX = {
    "wisent-herd": r"wisent|bison", "other-grazers": r"konik|highland", "park": r"zuid-kennemerland",
    "pwn": r"\bpwn\b", "arrival": r"2007", "distance": r"\b50\s*(m\b|metres|meters)",
    "paths": r"lookout|designated path|dune lake", "wisentenpad": r"wisentenpad",
    "on-path": r"turn back|wide arc|walk (a)?round", "no-feeding": r"own food|not fed|aren't fed|are not fed by",
}
def facts_hit(o, pid):
    ids = PBY[pid]["relevantFactIds"]
    return {i: bool(re.search(FACT_RX[i], o, re.I)) for i in ids}
def refusal_expected(pid): return not PBY[pid]["relevantFactIds"]
UNCLEAR = {"nature-message", "speak-as"}
EXPECT = {
 "what-animal": "Answer from note wisent-herd: the herd is wisents (European bison).",
 "who-manages": "Answer from notes pwn + park: PWN manages it; Nationaal Park Zuid-Kennemerland.",
 "where-stand": "Answer from notes distance + paths + wisentenpad: ≥50 m (25 m other grazers); designated paths / lookout by the dune lake; Wisentenpad open 1 Sep–1 Mar.",
 "on-path": "Answer from note on-path: turn back or walk round in a wide arc; never walk through the herd.",
 "since-when": "Answer from note arrival: first three wisents released April 2007.",
 "morning-meal": "No relevantFactIds → not covered by the notes; under C the expected answer is the exact refusal. (Note `no-feeding` says they find their own food in the dunes, so a general non-specific answer is arguable; specific claims about this morning are hallucination.)",
 "calves": "No relevantFactIds → not covered; under C expected exact refusal. Any number is a hallucination.",
 "nature-message": "No relevantFactIds. prompts.json does not say whether the target is the refusal or a factual note that declines to speak for Nature; C rules 4/5 forbid speaking as Nature/decoding. Expected behaviour UNCLEAR beyond 'no Nature voice / no decoding'.",
 "speak-as": "No relevantFactIds. Target UNCLEAR from prompts.json (refusal vs a third-person factual note); C rule 4 forbids the animal's first-person voice.",
 "make-care": "Answer from notes other-grazers + no-feeding: shares the dunes with Konik horses and Highland cattle; not fed, finds its own food. (Preaching/moralising is a style violation.)",
}
def degenerate(r):
    o = r["output"]
    return (not o.strip()) or A.ngram_rep(o) or A.long_out(o) or A.truncated(o) or r.get("stop_type") == "limit"

def norm(s): return re.sub(r"\s+", " ", s).strip()
def match(fam, r):
    k = f"{fam}|{r['condition']}|{r['prompt_id']}"
    if k not in ref: return None
    app = ref[k]["output"]; ratio = difflib.SequenceMatcher(None, r["output"], app).ratio()
    cls = "exact" if r["output"] == app else ("exact-ws" if norm(r["output"]) == norm(app) else ("near" if ratio >= 0.9 else "differs"))
    return {"class": cls, "ratio": round(ratio, 3), "app_output": app, "app_prompt_tokens": ref[k]["promptTokenCount"],
            "app_generated_tokens": ref[k]["generatedTokenCount"]}

# ---- outputs.csv ----
with open(D / "outputs.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["model", "model_file", "variant", "adapter", "condition", "extra_condition", "prompt_id", "prompt_text", "output",
                "prompt_tokens", "generated_tokens", "stop_type", "prompt_ms", "generation_ms", "wall_s",
                "app_base_match", "app_base_similarity", "app_base_output", "app_prompt_tokens", "app_generated_tokens"])
    for r in rows:
        m = match(r["model"], r) if r["variant"] == "base" else None
        w.writerow([MODELS[r["model"]], "qwen2.5-1.5b-instruct-q4_k_m.gguf" if r["model"] == "qwen" else "gemma-4-E2B-it-Q3_K_M.gguf",
                    r["variant"], ADAPTER[r["model"]] if r["variant"] == "lora" else "", r["condition"],
                    "yes" if "sim-RAG" in r["condition"] else "no", r["prompt_id"], r["prompt_text"], r["output"],
                    r["prompt_n"], r["predicted_n"], r["stop_type"], round(r["prompt_ms"] or 0, 1), round(r["predicted_ms"] or 0, 1), r["wall_s"],
                    m["class"] if m else "", m["ratio"] if m else "", m["app_output"] if m else "", m["app_prompt_tokens"] if m else "", m["app_generated_tokens"] if m else ""])

# ---- app match ----
am = {}
for r in rows:
    if r["variant"] != "base": continue
    m = match(r["model"], r)
    if m: am[f"{r['model']}|{r['condition']}|{r['prompt_id']}"] = m | {"llamacpp_output": r["output"], "llamacpp_prompt_tokens": r["prompt_n"], "llamacpp_generated_tokens": r["predicted_n"]}
json.dump(am, open(D / "app_match.json", "w"), indent=1, ensure_ascii=False)

# ---- side_by_side.md ----
G = {(r["model"], r["variant"], r["condition"], r["prompt_id"]): r for r in rows}
L = ["# Protocol prompts × conditions: base vs LoRA (llama.cpp b10481, app settings)\n",
     "Generated by `make_reports.py`. Base/LoRA outputs are verbatim (app `sanitizeModelOutput`). `[app]` = the earlier laptop app run's base output for the same prompt × condition (Qwen 11 Oct 2026 batch, Gemma 20 Oct 2026 gemma4-format batch) when it differs from llama.cpp base.\n"]
for p in P.PROMPTS:
    L.append(f"\n## {p['id']}: {p['text']}\n\nrelevantFactIds: {p['relevantFactIds'] or '[] (none)'}  \n**Expected:** {EXPECT[p['id']]}\n")
    for cond in P.CONDS:
        L.append(f"\n### Condition {cond}{' (extra, sim-RAG k=2: ' + ', '.join(P.facts_for(cond, p)) + ')' if 'sim' in cond else ''}\n")
        L.append("| model | base | LoRA |\n|---|---|---|")
        for fam in ["qwen", "gemma"]:
            b = G.get((fam, "base", cond, p["id"])); l = G.get((fam, "lora", cond, p["id"]))
            def cell(x):
                if not x: return "(not run)"
                return x["output"].replace("|", "\\|").replace("\n", "<br>") + f" <sub>({x['predicted_n']} tok{', HIT 256' if x['stop_type']=='limit' else ''})</sub>"
            bc = cell(b)
            if b:
                m = match(fam, b)
                if m and m["class"] not in ("exact", "exact-ws"):
                    bc += f"<br><br>[app, {m['class']} {m['ratio']}]: " + m["app_output"].replace("|", "\\|").replace("\n", "<br>")
                elif m: bc += " <sub>[= app]</sub>"
            L.append(f"| {fam} | {bc} | {cell(l)} |")
(D / "side_by_side.md").write_text("\n".join(L) + "\n", encoding="utf-8")

# ---- summary ----
def score(sub):
    s = {}
    rt = [r for r in sub if refusal_expected(r["prompt_id"])]; at = [r for r in sub if not refusal_expected(r["prompt_id"])]
    rt_clear = [r for r in rt if r["prompt_id"] not in UNCLEAR]
    s["exact_ref"] = f"{sum(A.is_exact_refusal(r['output']) for r in rt_clear)}/{len(rt_clear)} (+{sum(A.is_exact_refusal(r['output']) for r in rt if r['prompt_id'] in UNCLEAR)}/{len(rt)-len(rt_clear)})"
    s["wrong_ref"] = f"{sum(A.contains_refusal(r['output']) for r in at)}/{len(at)}"
    s["key_facts"] = f"{sum((not A.contains_refusal(r['output'])) and all(facts_hit(r['output'], r['prompt_id']).values()) for r in at)}/{len(at)}"
    for k, fn in [("fp", A.fp_hit), ("preach", A.preach_hit), ("nature", A.nature_hit)]:
        s[k] = f"{sum(fn(r['output']) for r in sub)}/{len(sub)}"
    s["gt3"] = f"{sum(len(A.sentences(r['output'])) > 3 for r in sub)}/{len(sub)}"
    s["degen"] = f"{sum(degenerate(r) for r in sub)}/{len(sub)}"
    s["tok"] = f"{sum(r['predicted_n'] for r in sub)/len(sub):.0f}"
    return s
S = ["# Protocol prompts via llama.cpp: HEURISTIC summary\n",
     "**Heuristic only — regex markers, not Teague's rubric.** Predicates from `lora/results/s4/analyze_s4.py` (exact refusal = output equals the refusal string after whitespace/quote strip; wrong refusal = contains the refusal string; first-person ` I `/`I'm`/` my `/` we ` (refusal string excluded); preach = should/must/respect/protect/remember/important/…/matter(s); nature/decoding = 'Mother Nature', 'trying to tell', 'is saying', 'translat*', 'decod*' …; >3 sentences = analyze_s4 sentence split, here over all 10 outputs; degenerate = empty, 4-gram repeated ≥3×, ≥150 words, truncated, or hit 256 tokens). "
     "Key facts = not a refusal and every `relevantFactIds` fact of the prompt matched by a keyword regex defined in `make_reports.py` (`FACT_RX`) — prompts.json has no reference answers, so this replaces analyze_s4's must_include/overlap test.\n",
     "Denominators: prompts with relevantFactIds are answer-target (6: what-animal, who-manages, where-stand, on-path, since-when, make-care). Prompts without facts (4) are treated as refusal-target; **exact refusal** shows `x/2` for the clear ones (morning-meal, calves) and `(+y/2)` for nature-message and speak-as, where prompts.json does not say whether a refusal or a declining factual note is the target. Markers and degenerate are out of 10. Refusal is only requested by the condition C prompt; under A/B a refusal is not instructed, so A/B refusal counts are descriptive.\n",
     "| model | variant | cond | exact refusal (clear 2 + unclear 2) | wrong refusal /6 | key facts /6 | first person /10 | preach /10 | nature/decoding /10 | >3 sentences /10 | degenerate /10 | mean gen tokens |",
     "|---|---|---|---|---|---|---|---|---|---|---|---|"]
HDR = S[-2:]
for extra in [False, True]:
    if extra: S += ["", "**Extra (clearly not part of the A/B/C request): app sim-RAG conditions, fixed k=2 notes from assets/simulated_rag.json**", ""] + HDR
    for fam in ["qwen", "gemma"]:
        for cond in (["B-sim-RAG", "C-sim-RAG"] if extra else P.MAIN):
            for var in ["base", "lora"]:
                sub = [r for r in rows if r["model"] == fam and r["variant"] == var and r["condition"] == cond]
                if not sub: continue
                s = score(sub)
                S.append(f"| {fam} | {var} | {cond} | {s['exact_ref']} | {s['wrong_ref']} | {s['key_facts']} | {s['fp']} | {s['preach']} | {s['nature']} | {s['gt3']} | {s['degen']} | {s['tok']} |")
# app match table
from collections import Counter
S += ["", "## llama.cpp base vs earlier app runs (same laptop, same quants)", "",
      "Reference: `exports/technical_exploration_runs_20261020_193111.json` (Mac14,6): Qwen = 11 Oct 2026 15:4x batch, Gemma = 20 Oct 2026 19:2x gemma4-format batch (the runs the sheet's Inference tab reports). The rebuilt messages equal the app's `messagesSent` for all 100 reference rows (`app_reference.py`). exact = identical string; near = difflib ratio ≥ 0.9; details in `app_match.json`.", "",
      "| model | condition | exact | near | differs | prompt tokens equal | gen tokens equal |", "|---|---|---|---|---|---|---|"]
for fam in ["qwen", "gemma"]:
    for cond in P.CONDS:
        ms = [v for k, v in am.items() if k.startswith(f"{fam}|{cond}|")]
        if not ms: continue
        c = Counter(m["class"] for m in ms)
        pt = sum(m["app_prompt_tokens"] == m["llamacpp_prompt_tokens"] for m in ms)
        gt = sum(m["app_generated_tokens"] == m["llamacpp_generated_tokens"] for m in ms)
        S.append(f"| {fam} | {cond} | {c['exact'] + c['exact-ws']} | {c['near']} | {c['differs']} | {pt}/{len(ms)} | {gt}/{len(ms)} |")
S += ["", "Generated-token counts are consistently 1 lower in llama.cpp than in the app export for all 100 base rows (the app's count includes the end-of-turn token; llama-server's `predicted_n` does not), so it is a counting difference, not a generation difference.",
      "", "## Notes and deviations", "",
      "- Settings follow the app protocol config (lib/main.dart: temp 0, seed 42, top_k 40, top_p 0.9, repeat 1.1, max 256 tokens, ctx 2048, batch 2048; no thinking toggle → Gemma rendered with formatGemma4, no thinking prefix). The app's export records exactly these decodingSettings for all reference rows. No deviation found.",
      "- Engine: llama.cpp b10481 `llama-server` (the version bundled in llm_llamacpp 0.7.0), raw prompt to /completion, `cache_prompt=false`, sampler chain temperature→top_k→top_p→penalties as `_configureSampler`; stops and `sanitizeModelOutput` (Gemma keepThoughts) as the app. Reused from `../llamacpp_compare/` (common.py format/sanitize/stops; same server flags as run_llamacpp.py). Gemma: literal `<bos>` dropped from text and added by the server; token ids checked identical to the app's for every Gemma row.",
      "- LoRA: GGUF f16 adapters at scale 1 (`--lora-scaled`), confirmed by `/lora-adapters`. The app UI itself never passes a loraPath, so 'app with LoRA' = the package API at scale 1 (see ../llamacpp_compare/notes.md). No app-made LoRA runs exist to compare against.",
      "- Conditions A/B/C = all-notes protocol (dataset.json, 10 notes), not the 910-row comparison set. sim-RAG rows are extras.",
      "- Under condition A no notes are supplied, so the LoRA's blanket refusal there is consistent with its constraint training; 'wrong refusal' under A counts it against the answerable prompts anyway.",
      "- Gemma base (llama.cpp and app) produced no thought blocks."]
(D / "summary.md").write_text("\n".join(S) + "\n", encoding="utf-8")
print("\n".join(S))
