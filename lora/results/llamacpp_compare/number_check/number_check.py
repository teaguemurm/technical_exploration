"""Digit-corruption check on the 910-row comparison outputs (llama.cpp vs MLX).
For each output that is not the exact refusal, numbers (digit runs) are compared with the
numbers in that row's prompt (notes + question) and reference answer.
  split   : digit groups separated by spaces/commas-free single spaces whose concatenation is not
            itself a prompt number but is a subsequence of one (e.g. '2 0 7', '2 07' vs 2007)
  dropped : a single digit run not in the prompt/reference but a proper subsequence of a prompt
            number (e.g. 5 vs 50, 207 vs 2007, 20 vs 2007)
  other   : a digit run not in prompt/reference and not explained above (hallucinated number)
Denominator 'num rows' = answerable rows whose reference contains a digit.
"""
import json, re, csv, collections, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
R = "I do not have information about that in the approved notes."
rows = {}
for l in open(os.path.join(ROOT, 'data/comparison/comparison.jsonl')):
    d = json.loads(l); m = d['metadata']; msg = d['messages']
    prompt = ' '.join(x['content'] for x in msg if x['role'] != 'assistant')
    ref = [x['content'] for x in msg if x['role'] == 'assistant'][0]
    rows[m['id']] = dict(cat=m['category'], prompt=prompt, ref=ref)

def nums(s): return re.findall(r'\d+', s)
def subseq(a, b):
    it = iter(b); return all(c in it for c in a)
def check(rid, out):
    o = out.strip().strip('"').strip()
    if o == R: return None
    r = rows[rid]; known = set(nums(r['prompt'])) | set(nums(r['ref']))
    flags = []
    # split digits: sequences like '2 0 7' / '2 07' / '20 07'
    for m in re.finditer(r'\b\d+(?: \d+)+\b', o):
        parts = m.group(0).split(' '); joined = ''.join(parts)
        if not all(p in known for p in parts) and any(subseq(joined, k) and len(k) >= len(joined) for k in known):
            flags.append(('split', m.group(0)))
    spans = [m.span() for m in re.finditer(r'\b\d+(?: \d+)+\b', o)]
    for m in re.finditer(r'\d+', o):
        if any(a <= m.start() < b for a, b in spans) and any(f[0]=='split' for f in flags): continue
        n = m.group(0)
        if n in known: continue
        if any(len(k) > len(n) and subseq(n, k) for k in known): flags.append(('dropped', n))
        else: flags.append(('other', n))
    return flags

variants = {}
RAW = os.path.join(ROOT, 'results/llamacpp_compare/raw')
for f in sorted(os.listdir(RAW)):
    if f.startswith('out_') and f.endswith('.jsonl'):
        v = 'llama.cpp ' + f[4:-6]
        variants[v] = {json.loads(l)['id']: json.loads(l)['output'] for l in open(os.path.join(RAW, f))}
for name, path in [('qwen ckpt600', 'results/s4_ckpt600_mlx_app_comparison.csv'),
                   ('gemma final', 'results/gemma4_s4/gemma4_s4_mlx_app_comparison.csv'),
                   ('gemma ckpt800', 'results/gemma4_s4/gemma4_s4_ckpt800_mlx_app_comparison.csv')]:
    rr = list(csv.DictReader(open(os.path.join(ROOT, path))))
    variants['MLX ' + name + ' LoRA'] = {x['id']: x['output_with_adapter'] for x in rr}
    if name != 'gemma ckpt800':
        variants['MLX ' + name.split()[0] + ' base'] = {x['id']: x['output_without_adapter'] for x in rr}

numrows = [i for i, r in rows.items() if r['cat'] == 'answerable' and re.search(r'\d', r['ref'])]
report = collections.OrderedDict(); examples = collections.defaultdict(list)
for v, outs in variants.items():
    c = collections.Counter(); answered = 0; rowsflag = collections.Counter()
    for rid in numrows:
        fl = check(rid, outs.get(rid, ''))
        if fl is None: continue
        answered += 1
        kinds = {k for k, _ in fl}
        for k in kinds: rowsflag[k] += 1
        if kinds & {'split', 'dropped'}: rowsflag['split_or_dropped'] += 1
        if fl: examples[v].append((rid, fl, outs[rid][:160]))
        refn = set(nums(rows[rid]['ref'])); on = set(nums(outs[rid]))
        if refn <= on: rowsflag['ref_numbers_all_present'] += 1
        elif not (kinds & {'split', 'dropped'}): rowsflag['ref_number_missing_no_corruption'] += 1
    # corruption flags on ALL 910 rows
    allc = collections.Counter()
    for rid in rows:
        fl = check(rid, outs.get(rid, '')) or []
        kinds = {k for k, _ in fl}
        if kinds & {'split', 'dropped'}: allc['split_or_dropped'] += 1
        if 'split' in kinds: allc['split'] += 1
    report[v] = dict(num_rows=len(numrows), answered=answered, split=rowsflag['split'], dropped=rowsflag['dropped'],
                     split_or_dropped=rowsflag['split_or_dropped'], other=rowsflag['other'],
                     refused=len(numrows)-answered, correct_numbers=rowsflag['ref_numbers_all_present'],
                     number_missing=rowsflag['ref_number_missing_no_corruption'],
                     all910_split_or_dropped=allc['split_or_dropped'], all910_split=allc['split'])
json.dump(dict(report=report, examples=examples), open('number_check.json', 'w'), indent=1, ensure_ascii=False)
print('variant | ' + ' | '.join(next(iter(report.values())).keys()))
print('NOTE: MLX gemma base outputs include the thinking text (thinking mode), so its numbers are not comparable.')
for v, d in report.items(): print(v, '|', ' | '.join(str(x) for x in d.values()))
