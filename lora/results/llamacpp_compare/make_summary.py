#!/usr/bin/env python3
"""Build summary.md from scores.json, agreement.json, raw/*.meta.json, smoke/ and notes.md."""
import json
from pathlib import Path
D = Path(__file__).resolve().parent
S = json.load(open(D / "scores.json")); V = json.load(open(D / "variants.json")); A = json.load(open(D / "agreement.json"))
COLS = [("exact refusal (refusal-target)", "exact_refusal_refusal_target"),
        ("over-refusal answerable", "over_refusal_answerable"),
        ("over-refusal ans+speak_as", "over_refusal_ans+speak_as"),
        ("key facts ans+speak_as", "key_facts_ans+speak_as"),
        ("key facts among answered", "key_facts_among_answered"),
        ("1st person speak_as (lower better)", "first_person_speak_as"),
        ("preaching refused", "preach_refused"),
        ("preaching fact", "preach_fact"),
        ("preach markers, preaching rows", "preach_markers_preaching_rows"),
        ("preach markers, all rows", "preach_markers_all"),
        ("nature/decode, all rows", "nature_decode_all")]
def pc(x): return None if not x or not x[1] else 100 * x[0] / x[1]
def cell(x): return "—" if pc(x) is None else f"{pc(x):.1f} ({x[0]})"
def f1(v): return "—" if v is None else f"{v:.1f}"
def dd(a, b): return "—" if a is None or b is None else f"{a - b:+.1f}"
def table(fam, sub):
    names = [n for n, v in V.items() if v["family"] == fam and n in S]
    if not names: return "(no complete variants yet)"
    ns = S[names[0]][sub]
    hdr = [f"{c} n={ns[k][1]}" if k != "key_facts_among_answered" else c for c, k in COLS]
    L = [f"| variant ({sub}) | " + " | ".join(hdr) + " | avg words ans+speak_as |", "|---|" + "---:|" * (len(COLS) + 1)]
    for n in names:
        m = S[n][sub]
        L.append(f"| {V[n]['label']} | " + " | ".join((cell(m[k]) if k != "key_facts_among_answered" or pc(m[k]) is None else f"{pc(m[k]):.1f} ({m[k][0]}/{m[k][1]})") for _, k in COLS) + f" | {m['avg_words_ans+speak_as']} |")
    return "\n".join(L)
def delta_table(fam, base, others, sub):
    if base not in S: return ""
    L = [f"Differences vs {V[base]['label']} ({sub}, percentage points):", "",
         "| variant | " + " | ".join(c for c, _ in COLS) + " |", "|---|" + "---:|" * len(COLS)]
    for n in others:
        if n not in S: continue
        L.append(f"| {V[n]['label']} − base+C | " + " | ".join(dd(pc(S[n][sub][k]), pc(S[base][sub][k])) for _, k in COLS) + " |")
    return "\n".join(L)
out = ["# llama.cpp b10481 laptop comparison of the constraint LoRAs (910-row comparison set)", "",
       "Cells: percent (count). Metrics and predicates as `results/gemma4_s4/headline_full_gemma4_s4.md` (`results/s4/analyze_s4.py`, imported unchanged). "
       "Refusal-target = reference is exactly the refusal string (269 refusal + 62 speak_as rows = 331). "
       "'preaching refused' = output contains the refusal string (the headline table's 'refused (trained)', i.e. refusal counted as correct); "
       "'preaching fact' = analyze_s4 key facts on the preaching rows (reference is a factual answer; analyze_s4.stats treats them as answer-target, so a refusal there counts as wrong refusal). "
       "MLX rows are the earlier bf16 MLX runs (condition B, app layout), rescored here; they reproduce the headline numbers exactly. "
       "Condition A has no notes, so refusing there is grounded behaviour; its over-refusal is a 'how much does the LoRA refuse without notes' check.", ""]
for fam, title, base, others in [("qwen", "Qwen2.5-1.5B-Instruct Q4_K_M + s4 ckpt600 LoRA", "qwen_base_C", ["qwen_lora_B", "qwen_lora_C", "qwen_lora_A"]),
                                 ("gemma", "Gemma 4 E2B-it Q3_K_M + s4 LoRA (final iter 1000 and ckpt800)", "gemma_base_C",
                                  ["gemma_final_B", "gemma_final_C", "gemma_final_A", "gemma_800_B", "gemma_800_C", "gemma_800_A"])]:
    out += [f"## 1. {title}", ""]
    for sub in ["full", "full_minus_selection", "eval_only", "eval_minus_selection"]:
        out += [f"### {sub}", "", table(fam, sub), ""]
    for sub in ["full", "full_minus_selection"]:
        out += [delta_table(fam, base, others, sub), ""]
out += ["## 2. Qwen vs Gemma (llama.cpp; full / full minus selection)", ""]
KEY = [("exact refusal", "exact_refusal_refusal_target"), ("over-refusal answerable", "over_refusal_answerable"),
       ("over-refusal ans+speak_as", "over_refusal_ans+speak_as"), ("key facts ans+speak_as", "key_facts_ans+speak_as"),
       ("1st person", "first_person_speak_as"), ("preaching refused", "preach_refused"), ("preaching fact", "preach_fact"),
       ("preach markers all", "preach_markers_all"), ("nature/decode all", "nature_decode_all")]
out.append("| setting | model | " + " | ".join(k for k, _ in KEY) + " |"); out.append("|---|---|" + "---:|" * len(KEY))
for lab, trip in [("base + C", ["qwen_base_C", "gemma_base_C"]), ("LoRA + B", ["qwen_lora_B", "gemma_final_B", "gemma_800_B"]),
                  ("LoRA + C", ["qwen_lora_C", "gemma_final_C", "gemma_800_C"]), ("LoRA + A", ["qwen_lora_A", "gemma_final_A", "gemma_800_A"])]:
    for n in trip:
        if n not in S: continue
        out.append(f"| {lab} | {V[n]['label']} | " + " | ".join(f"{f1(pc(S[n]['full'][k]))} / {f1(pc(S[n]['full_minus_selection'][k]))}" for _, k in KEY) + " |")
out += ["", "## 3. llama.cpp vs MLX (same LoRA, condition B — the prompt the MLX runs used)", ""]
out.append("| pair | subset | same outcome category | same refused-vs-answered | identical text | top disagreements |\n|---|---|---:|---:|---:|---|")
for k, res in A.items():
    for sub in ["full", "full_minus_selection"]:
        v = res[sub]
        dis = "; ".join(f"{a}: {b}" for a, b in list(v["disagreements"].items())[:4])
        out.append(f"| {k} | {sub} | {100*v['same_outcome']/v['n']:.1f}% ({v['same_outcome']}/{v['n']}) | {100*v['same_refused_vs_answered']/v['n']:.1f}% | {100*v['identical_text']/v['n']:.1f}% ({v['identical_text']}) | {dis} |")
out += ["", "Outcome categories: refusal-target rows → exact_refusal / inexact_refusal / no_refusal; other rows → refused / answered_keyfacts / answered_no_keyfacts.", "",
        "Per-metric difference, llama.cpp minus MLX (full set, percentage points):", ""]
out.append("| pair | " + " | ".join(c for c, _ in COLS) + " |\n|---|" + "---:|" * len(COLS))
for k in A:
    a, b = k.split(" vs ")
    out.append(f"| {V[a]['label']} − MLX | " + " | ".join(dd(pc(S[a]['full'][key]), pc(S[b]['full'][key])) for _, key in COLS) + " |")
out += ["", "## 4. Settings, paths, quants, runtimes", ""]
rt = ["| run | rows | gen seconds | server load s | generated tokens | max prompt tokens | stopped at 256 | adapter (server /lora-adapters) |", "|---|---:|---:|---:|---:|---:|---:|---|"]
for sub in ["smoke", "raw"]:
    for m in sorted((D / sub).glob("out_*.meta.json")):
        j = json.load(open(m))
        la = "; ".join(f"{x.get('path','').split('/')[-1]} scale {x.get('scale')}" for x in j.get("lora_adapters_endpoint") or []) or "none"
        res = f" (resumed after {j['resumed_after_rows']})" if j.get("resumed_after_rows") else ""
        rt.append(f"| {j['name']} | {j['rows']} | {j['seconds']}{res} | {j['server_load_seconds']} | {j['generated_tokens']} | {j['max_prompt_tokens']} | {j['stopped_at_n_predict']} | {la} |")
out += rt + [""]
if (D / "notes.md").exists(): out.append((D / "notes.md").read_text())
(D / "summary.md").write_text("\n".join(out) + "\n")
print("wrote summary.md")
