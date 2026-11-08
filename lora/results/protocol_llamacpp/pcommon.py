"""Protocol (10 prompts, assets/prompts.json) prompt building for llama.cpp.
Mirrors lib/messages.dart buildConditionMessages / selectFactsForCondition /
buildConditionCSystemPrompt / localNotesMessage exactly, for conditions
A, B, C (all 10 notes from assets/dataset.json) and the sim-RAG extras
B-sim-RAG, C-sim-RAG (assets/simulated_rag.json, k=2).
Formatting / stops / sanitize are reused from ../llamacpp_compare/common.py
(Qwen ChatML, Gemma formatGemma4 with thinking off)."""
import json, re, sys
from pathlib import Path
D = Path(__file__).resolve().parent
ROOT = D.parents[2]
sys.path.insert(0, str(D.parent / "llamacpp_compare"))
import common as C  # chatml via wisent_lora, format_gemma4, sanitize, stops

def _trim1(s):
    if s.endswith("\r\n"): return s[:-2]
    if s.endswith("\n"): return s[:-1]
    return s

MSG_DART = (ROOT / "lib/messages.dart").read_text(encoding="utf-8")
REFUSAL = re.search(r"standardRefusalStatement =\s*'([^']*)'", MSG_DART).group(1)
NOTES_TITLE = re.search(r"localNotesTitle = '([^']*)'", MSG_DART).group(1)
NOTES_INTRO = re.search(r"localNotesIntro =\s*'([^']*)'", MSG_DART).group(1)
PERSONA = _trim1((ROOT / "assets/system_prompt.txt").read_text(encoding="utf-8"))  # parseCatalog trims one newline
PROMPTS = json.loads((ROOT / "assets/prompts.json").read_text(encoding="utf-8"))
FACTS = json.loads((ROOT / "assets/dataset.json").read_text(encoding="utf-8"))
SIMRAG = json.loads((ROOT / "assets/simulated_rag.json").read_text(encoding="utf-8"))
FACT = {f["id"]: f["text"] for f in FACTS}
CONDS = ["A", "B", "B-sim-RAG", "C", "C-sim-RAG"]  # app batch order per prompt (sim-RAG on)
MAIN = ["A", "B", "C"]

def cond_c_system(base):
    return (f"{_trim1(base)}\n\n"
            "STRICT GROUNDING & FORM CONSTRAINTS:\n"
            "1. Answer strictly and solely based on the Local notes provided below.\n"
            "2. If the Local notes do not explicitly provide the information needed to answer the question, "
            f"you MUST reply with: \"{REFUSAL}\" Do not guess, extrapolate, or invent details.\n"
            "3. Limit your response to 1 to 3 concise, factual sentences. "
            "Do not include conversational pleasantries, introductory phrases, or meta-commentary.\n"
            "4. Do NOT speak as Nature or in the animal's first-person voice.\n"
            "5. Do NOT claim to decode or translate the animal's behaviour.\n"
            "6. Do NOT preach or moralise at the visitor.")

def notes_msg(lines):
    return "\n".join([NOTES_TITLE, "", NOTES_INTRO, *lines])

def facts_for(cond, p):
    if cond == "A": return []
    if cond in ("B", "C"): return [f["id"] for f in FACTS]
    ids = SIMRAG["retrieval"].get(p["id"]) or []
    return [i for i in ids[: SIMRAG.get("k", 2)] if i in FACT]

def messages(cond, p):
    q = p["text"]; ids = facts_for(cond, p)
    if cond == "A":
        return [("system", PERSONA), ("user", q)]
    if cond.startswith("B"):
        m = [("system", PERSONA)]
        if ids: m.append(("user", notes_msg([FACT[i] for i in ids])))
        return m + [("user", q)]
    m = [("system", cond_c_system(PERSONA))]
    m.append(("user", notes_msg([FACT[i] for i in ids]) if ids else f"{NOTES_TITLE}\n\n{NOTES_INTRO}\n(No specific local notes match this query.)"))
    return m + [("user", q)]

def format_messages(msgs):  # lib/messages.dart formatMessages == export field messagesSent
    return "\n\n".join(f"{r}\n{c}" for r, c in msgs)

def render(cond, p, family):
    msgs = messages(cond, p)
    if family == "qwen":
        from wisent_lora.render import chatml
        return chatml(msgs)
    return C.format_gemma4([{"role": r, "content": c} for r, c in msgs])
