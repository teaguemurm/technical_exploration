#!/usr/bin/env python3
"""Validate Gemma 4 E2B LoRA GGUFs with llama.cpp b10481's gguf reader (run with ~/llama.cpp-b10481/.venv/bin/python).
Checks: metadata (general.architecture=gemma4, general.type=adapter, adapter.type=lora, alpha), tensor count,
per-layer modules (k/v only on layers 0..14 because of KV sharing: 35 layers, 20 shared), A/B shapes (rank 16),
and that the LoRA factors equal the mlx checkpoint (B@A up to f16 rounding) when an mlx file is given.
Usage: validate_gguf.py <gguf> [<mlx safetensors>] [--compare <other gguf>]"""
import sys, re, json, collections
import numpy as np
from gguf import GGUFReader
def meta(r):
    out = {}
    for k, f in r.fields.items():
        if k.startswith("GGUF.") : continue
        v = f.parts[f.data[0]] if f.data else None
        try:
            out[k] = bytes(v).decode() if f.types and f.types[0].name == "STRING" else (v.tolist()[0] if hasattr(v, "tolist") else v)
        except Exception: out[k] = str(v)
    return out
def tensors(r): return {t.name: t for t in r.tensors}
args = sys.argv[1:]; cmp = None
if "--compare" in args: i = args.index("--compare"); cmp = args[i+1]; del args[i:i+2]
path = args[0]; mlx = args[1] if len(args) > 1 else None
r = GGUFReader(path); m = meta(r); T = tensors(r)
res = dict(file=path, n_tensors=len(T), metadata={k: v for k, v in m.items() if not k.startswith("tokenizer")})
pat = re.compile(r"^blk\.(\d+)\.(\w+)\.weight\.lora_([ab])$")
per = collections.defaultdict(set); bad = []
for n, t in T.items():
    mm = pat.match(n)
    if not mm: bad.append(n); continue
    per[mm.group(2)].add(int(mm.group(1)))
res["modules_layers"] = {k: (min(v), max(v), len(v)) for k, v in sorted(per.items())}
res["unexpected_names"] = bad
exp = {"attn_q": 35, "attn_output": 35, "ffn_gate": 35, "ffn_up": 35, "ffn_down": 35, "attn_k": 15, "attn_v": 15}
res["layer_check"] = all(len(per.get(k, ())) == v for k, v in exp.items()) and per["attn_k"] == set(range(15)) and per["attn_v"] == set(range(15))
res["expected_n_tensors"] = 2 * sum(exp.values())
ranks = {tuple(sorted(t.shape.tolist())) for n, t in T.items() if n.endswith("lora_a")}
res["rank_ok"] = all(16 in s for s in ranks)
res["dtypes"] = sorted({t.tensor_type.name for t in T.values()})
res["arch_ok"] = m.get("general.architecture") == "gemma4"
if mlx:
    from safetensors.numpy import load_file
    W = load_file(mlx)
    name_map = {"q_proj": "attn_q", "k_proj": "attn_k", "v_proj": "attn_v", "o_proj": "attn_output", "gate_proj": "ffn_gate", "up_proj": "ffn_up", "down_proj": "ffn_down"}
    maxrel = 0.0; n = 0
    for k in W:
        mm = re.match(r"language_model\.model\.layers\.(\d+)\.\w+\.(\w+)\.lora_a$", k)
        if not mm: continue
        L, mod = mm.groups(); g = f"blk.{L}.{name_map[mod]}.weight"
        a_m = W[k].astype(np.float32); b_m = W[k[:-1] + "b"].astype(np.float32)   # (in,r), (r,out)
        a_g = np.asarray(T[g + ".lora_a"].data, dtype=np.float32); b_g = np.asarray(T[g + ".lora_b"].data, dtype=np.float32)
        d_m = (a_m @ b_m).T; d_g = b_g @ a_g
        rel = np.linalg.norm(d_m - d_g) / (np.linalg.norm(d_m) + 1e-12); maxrel = max(maxrel, rel); n += 1
    res["mlx_delta_compare"] = dict(n_modules=n, max_rel_err_BA=float(maxrel))
if cmp:
    r2 = GGUFReader(cmp); T2 = tensors(r2); m2 = meta(r2)
    same = set(T) == set(T2) and all(np.array_equal(np.asarray(T[n].data), np.asarray(T2[n].data)) for n in T)
    diffmeta = {k: (m.get(k), m2.get(k)) for k in set(m) | set(m2) if m.get(k) != m2.get(k) and not k.startswith("tokenizer")}
    res["compare"] = dict(other=cmp, tensors_identical=bool(same), metadata_differences=diffmeta)
print(json.dumps(res, indent=1, default=str))
