# 学習前のモデルが、同じシステム文でどんな書き方で答えるかを見る（ソロの教訓「答えの形は元のモデルに寄せる」）。
#   使い方: python probe_base.py <モデル> <データ.jsonl> [件数]   → 出力と採点を表示
import json, sys, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from common import score

name, path = sys.argv[1], sys.argv[2]; n = int(sys.argv[3]) if len(sys.argv) > 3 else 12
tok = AutoTokenizer.from_pretrained(name); model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, device_map='cuda')
rows = [json.loads(l) for l in open(path, encoding='utf-8')][:n]; ok = 0
for x in rows:
    msgs = x['messages'][:2]; gold = json.loads(x['messages'][2]['content'])
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors='pt', return_dict=True)['input_ids'].to('cuda')
    g = model.generate(ids, max_new_tokens=120, do_sample=False); out = tok.decode(g[0, ids.shape[1]:], skip_special_tokens=True)
    s = score(out, gold); ok += s['correct']
    print(f"--- {x['messages'][1]['content']}  [正解 {gold['action']} {gold['instrument'] or gold['candidates']}]  {'○' if s['correct'] else '×'} {s['error'] or ''}\n{out}")
print(f'\n正解 {ok}/{len(rows)}')
