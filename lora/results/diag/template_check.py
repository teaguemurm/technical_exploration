import json, sys
from pathlib import Path
LORA = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LORA)); sys.path.insert(0, str(LORA / "scripts"))
from run_comparison import prompt_messages, load_rows
from wisent_lora.comparison import load_note_pool
from wisent_lora.protocol import load_persona, repo_root
from transformers import AutoTokenizer
from mlx_lm.utils import load_tokenizer  # noqa
from mlx_lm import load
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
root = repo_root(); persona = load_persona(root); nt, _, _ = load_note_pool(root)
train = {}
for f in ["train.jsonl", "valid.jsonl"]:
    for l in (LORA / "data/v2" / f).read_text().splitlines():
        if l.strip(): r = json.loads(l); train[r["metadata"]["id"]] = r
print("train rows w/o system turn:", sum(r["messages"][0]["role"] != "system" for r in train.values()))
comp = load_rows(LORA / "data/comparison/comparison.jsonl", None)
only_q = 0; other = 0; tok_eq = 0; n = 0
for row in comp:
    m = row["metadata"]
    if not m.get("question_in_original"): continue
    t = train[m["id"]]
    built = prompt_messages(row, layout="app", persona=persona, note_text=nt)
    stored = t["messages"][:-1]
    if built != stored:
        if built[:-1] == stored[:-1]: only_q += 1
        else: other += 1
    # training masked-prefix tokens vs inference prompt tokens (on stored messages)
    train_prefix = tok.apply_chat_template(stored, add_generation_prompt=True, tokenize=True, return_dict=False)
    full = tok.apply_chat_template(t["messages"], tokenize=True, return_dict=False)
    infer = tok.encode(tok.apply_chat_template(stored, add_generation_prompt=True, tokenize=False), add_special_tokens=False)
    n += 1; tok_eq += (train_prefix == infer and full[:len(train_prefix)] == train_prefix)
print("diff only in question:", only_q, "other diffs:", other)
print("token-identical prefix (train mask offset == inference prompt, and prefix of full seq):", tok_eq, "/", n)
s = tok.apply_chat_template(t["messages"], tokenize=False)
print("default system injected when system present?", "You are Qwen" in s)
print("default system injected when no system?", "You are Qwen" in tok.apply_chat_template([{"role":"user","content":"hi"}], tokenize=False, add_generation_prompt=True))
print(repr(s[-400:]))
