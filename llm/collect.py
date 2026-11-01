# RFT と DPO の材料集め: モデルに 1 問 n 回答えさせ（温度 1.0）、採点器（common.score）で合格・不合格に分ける。
#   - RFT 用: 合格した答え（問題ごとに違う答えを最大 2 つ）→ data/rft_<name>.jsonl（SFT と同じ形式）
#   - DPO 用: 同じ問題の 合格（chosen）と 不合格（rejected）の組 → data/dpo_<name>.jsonl
#   答えはモデル自身が出したものだけ。合否はプログラムが決める。
#   使い方: python collect.py --adapter runs/sft-160 --data data/rl_u.jsonl --name sft160 [--n 8]
import argparse, collections, json, random
from common import score
from evaluate import load_model, answer

ap = argparse.ArgumentParser(); ap.add_argument('--adapter'); ap.add_argument('--start'); ap.add_argument('--data', required=True); ap.add_argument('--name', required=True)
ap.add_argument('--n', type=int, default=8); ap.add_argument('--temp', type=float, default=1.0); ap.add_argument('--seed', type=int, default=0)
a = ap.parse_args(); rng = random.Random(a.seed)
import torch; torch.manual_seed(a.seed)
tok, m = load_model(a.adapter, a.start); rows = [json.loads(l) for l in open(a.data, encoding='utf-8')]
outs = answer(tok, m, [r['messages'][:2] for r in rows], bs=32, sample=True, n=a.n, temperature=a.temp)
rft, dpo = [], []; stat = collections.defaultdict(lambda: [0, 0, 0])   # kind → [問題数, 1 つでも合格した問題, 合格した答えの数]
for r, os_ in zip(rows, outs):
    good, bad = [], []
    for o in os_:
        (good if score(o, r['gold'])['correct'] else bad).append(o.strip())
    k = r['kind']; stat[k][0] += 1; stat[k][1] += bool(good); stat[k][2] += len(good)
    for g in list(dict.fromkeys(good))[:2]: rft.append({'messages': r['messages'][:2] + [{'role': 'assistant', 'content': g}], 'src': r['utterance'], 'kind': k})
    if good and bad:
        for b in list(dict.fromkeys(bad))[:2]: dpo.append({'prompt': r['messages'][:2], 'chosen': [{'role': 'assistant', 'content': rng.choice(good)}], 'rejected': [{'role': 'assistant', 'content': b}], 'kind': k})
for name, xs in ((f'data/rft_{a.name}.jsonl', rft), (f'data/dpo_{a.name}.jsonl', dpo)):
    with open(name, 'w', encoding='utf-8') as f:
        for x in xs: f.write(json.dumps(x, ensure_ascii=False) + '\n')
print(f'問題 {len(rows)} → RFT {len(rft)} 件、DPO {len(dpo)} 組')
for k, (n, solved, g) in sorted(stat.items()): print(f'  {k}: {solved}/{n} 問で合格あり（合格の答え {g}/{n * a.n}）')
