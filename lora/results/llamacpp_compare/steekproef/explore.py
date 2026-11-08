import json,csv,sys,re,collections
from pathlib import Path
D=Path(__file__).resolve().parent; LORA=D.parents[2]
sys.path.insert(0,str(LORA/'results/s4')); import analyze_s4 as A
R=A.R
rows=[json.loads(l) for l in open(LORA/'data/comparison/comparison.jsonl') if l.strip()]
M={r['metadata']['id']:r['metadata'] for r in rows}
def raw(n): return {json.loads(l)['id']:json.loads(l) for l in open(LORA/f'results/llamacpp_compare/raw/out_{n}.jsonl')}
qB=raw('qwen_lora_B'); g8=raw('gemma_800_B')
mlxq={r['id']:r['output_with_adapter'] for r in csv.DictReader(open(LORA/'results/s4_ckpt600_mlx_app_comparison.csv'))}
mlx8={r['id']:r['output_with_adapter'] for r in csv.DictReader(open(LORA/'results/gemma4_s4/gemma4_s4_ckpt800_mlx_app_comparison.csv'))}
rt=lambda i:M[i]['reference_answer'].strip()==R
# garbled: refusal-like but not contains_refusal
for i,x in qB.items():
  o=x['output']
  if not A.contains_refusal(o) and re.search(r'approved notes|information about that',o,re.I): print('GARBLED',i,rt(i),M[i]['category'],repr(o))
print('qwen rt not refused', sum(1 for i in qB if rt(i) and not A.contains_refusal(qB[i]['output'])))
dis=[i for i in qB if rt(i) and not A.contains_refusal(qB[i]['output']) and A.is_exact_refusal(mlxq[i])]
print('dis21',len(dis))
for nm,src in [('llama800',{i:x['output'] for i,x in g8.items()}),('mlx800',mlx8)]:
  lp=[i for i,o in src.items() if A.ngram_rep(o) or len(o.split())>=100]
  print(nm,len(lp),[(i,len(src[i].split())) for i in lp][:12])
