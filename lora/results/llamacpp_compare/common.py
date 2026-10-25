"""Shared prompt building for the llama.cpp laptop comparison (new file; does not
change any earlier script).

Conditions, per comparison row (data/comparison/comparison.jsonl, 910 rows):
  A = lib/messages.dart buildConditionMessages(conditionA): persona system turn + question.
  B = condition_b_app_messages (the exact function the earlier MLX runs used):
      persona system turn, the row's notes (prompt_note_ids) as a user turn via
      localNotesMessage, question last.
  C = buildConditionMessages(conditionC): system = buildConditionCSystemPrompt(persona)
      (wisent_lora.render.condition_c_system, refusal string from lib/messages.dart),
      the row's notes as a user turn, or the '(No specific local notes match this query.)'
      block when the row has no notes, question last.
Formatting = what packages/llm_llamacpp 0.7.0 sends:
  Qwen: native llama_chat_apply_template -> ChatML (wisent_lora.render.chatml).
  Gemma 4: formatGemma4 (chat_template_format.dart), thinking off, '<bos>' in text.
"""
import json, re, sys
from pathlib import Path
D = Path(__file__).resolve().parent
LORA = D.parents[1]
sys.path.insert(0, str(LORA))
from wisent_lora.comparison import load_note_pool
from wisent_lora.protocol import condition_b_app_messages, load_persona, load_refusal, local_notes_message, repo_root
from wisent_lora.render import chatml, condition_c_system, LOCAL_NOTES_EMPTY

ROOT = repo_root()
PERSONA = load_persona(ROOT)
REFUSAL = load_refusal(ROOT)
NOTE_TEXT, _, _ = load_note_pool(ROOT)
COMPARISON = LORA / "data/comparison/comparison.jsonl"
ROWS = [json.loads(l) for l in COMPARISON.read_text(encoding="utf-8").splitlines() if l.strip()]
SELECTION_IDS = json.load(open(LORA / "results/s4/subset_ids.json"))["ids"]

# lib/messages.dart stopSequences (+ '<eos>' for Gemma 4, sanitizeModelOutput keepThoughts)
APP_STOPS = ['<|im_end|>', '<end_of_turn>', '<|eot_id|>', '<turn|>', '<|end|>', '<|im_start|>', '<start_of_turn>']
GEMMA4_STOPS = ['<turn|>', '<end_of_turn>', '<eos>']

def question(row):
    return [m["content"] for m in row["messages"] if m["role"] == "user"][-1]

def note_texts(row):
    return [NOTE_TEXT[n] for n in row["metadata"].get("prompt_note_ids") or []]

def messages(row, cond):
    q = question(row); texts = note_texts(row)
    if cond == "A":
        return [{"role": "system", "content": PERSONA}, {"role": "user", "content": q}]
    if cond == "B":
        return condition_b_app_messages(PERSONA, texts, q)
    if cond == "C":
        m = [{"role": "system", "content": condition_c_system(PERSONA, REFUSAL)}]
        m.append({"role": "user", "content": local_notes_message(texts) if texts else LOCAL_NOTES_EMPTY})
        m.append({"role": "user", "content": q})
        return m
    raise ValueError(cond)

def format_gemma4(msgs):
    out = "<bos>"
    for m in msgs:
        role = {"system": "system", "user": "user", "assistant": "model"}[m["role"]]
        out += f"<|turn>{role}\n{m['content']}<turn|>\n"
    return out + "<|turn>model\n"

def render(row, cond, family):
    msgs = messages(row, cond)
    if family == "qwen":
        return chatml([(m["role"], m["content"]) for m in msgs])
    if family == "gemma":
        return format_gemma4(msgs)
    raise ValueError(family)

_THOUGHT = [re.compile(p, re.I) for p in [r'<(think|thought)>[\s\S]*?</\1>', r'<\|thought\|>[\s\S]*?</\|thought\|>',
            r'<\|channel\>thought[\s\S]*?(<channel\|>|<\|channel>|$)', r'<(think|thought)>[\s\S]*$', r'<\|thought\|>[\s\S]*$']]

def sanitize(text, family):
    """lib/messages.dart sanitizeModelOutput; keepThoughts for Gemma 4 (as main.dart)."""
    keep = family == "gemma"
    if not keep:
        for p in _THOUGHT: text = p.sub("", text)
        text = text.strip()
    stops = APP_STOPS + (["<eos>"] if keep else [])
    idx = [text.find(s) for s in stops if text.find(s) != -1]
    if idx: text = text[: min(idx)]
    return text.strip()
