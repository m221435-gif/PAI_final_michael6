# 採点: モデル（学習前、または LoRA アダプター）に問題を解かせ、common.score で自動採点。種類（kind）ごとの正答率を出す。
#   使い方: python evaluate.py --name sft-80 --adapter runs/sft-80 --data data/test_checked.jsonl [--data2 ...]  → results/<name>.json
#           --adapter を付けなければ学習前のモデル
import argparse, collections, glob, json, os, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from common import score

BASE = glob.glob(os.path.expanduser('~/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/*/'))[0]


def load_model(adapter=None, start=None):
    tok = AutoTokenizer.from_pretrained(BASE, padding_side='left'); m = AutoModelForCausalLM.from_pretrained(start or BASE, dtype=torch.bfloat16, device_map='cuda')
    if adapter:
        from peft import PeftModel
        m = PeftModel.from_pretrained(m, adapter); m = m.merge_and_unload()
    return tok, m.eval()


@torch.no_grad()
def answer(tok, m, msgs_list, bs=64, sample=False, n=1, temperature=1.0):
    """msgs_list: [[system, user], ...] → 出力文字列のリスト（n>1 なら問題ごとに n 個のリスト）"""
    outs = []
    for b in range(0, len(msgs_list), bs):
        texts = [tok.apply_chat_template(ms, add_generation_prompt=True, tokenize=False) for ms in msgs_list[b:b + bs]]
        enc = tok(texts, return_tensors='pt', padding=True).to('cuda')
        g = m.generate(**enc, max_new_tokens=64, do_sample=sample, temperature=temperature if sample else None, top_p=None, top_k=None, num_return_sequences=n, pad_token_id=tok.eos_token_id)
        dec = [tok.decode(r, skip_special_tokens=True) for r in g[:, enc['input_ids'].shape[1]:]]
        outs += [dec[k * n:(k + 1) * n] for k in range(len(texts))] if n > 1 else dec
    return outs


def run(tok, m, rows):
    outs = answer(tok, m, [r['messages'][:2] for r in rows]); per = collections.defaultdict(lambda: [0, 0]); errs = collections.Counter(); detail = []
    for r, o in zip(rows, outs):
        s = score(o, r['gold']); k = r.get('kind', '?'); per[k][0] += s['correct']; per[k][1] += 1
        if not s['correct']: errs[s['error']] += 1
        detail.append({'id': r.get('id'), 'utterance': r['messages'][1]['content'], 'gold': r['gold'], 'out': o, **s})
    tot = sum(v[0] for v in per.values()), sum(v[1] for v in per.values())
    return {'total': tot, 'acc': round(tot[0] / max(1, tot[1]), 4), 'per_kind': {k: v for k, v in sorted(per.items())}, 'errors': dict(errs), 'detail': detail}


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--name', required=True); ap.add_argument('--adapter'); ap.add_argument('--data', nargs='+', required=True); ap.add_argument('--start', help='土台のモデル（DPO なら runs/dpo-XX_start）')
    a = ap.parse_args(); tok, m = load_model(a.adapter, a.start); res = {}
    for p in a.data:
        rows = [json.loads(l) for l in open(p, encoding='utf-8')]
        for r in rows:   # SFT 形式（messages に答えが入っている）も受ける
            if 'gold' not in r: r['gold'] = json.loads(r['messages'][2]['content'])
        res[os.path.basename(p)] = run(tok, m, rows)
        print(f"[{a.name}] {os.path.basename(p)}: {res[os.path.basename(p)]['total'][0]}/{res[os.path.basename(p)]['total'][1]} = {res[os.path.basename(p)]['acc']*100:.1f}%", {k: f'{v[0]}/{v[1]}' for k, v in res[os.path.basename(p)]['per_kind'].items()}, flush=True)
    os.makedirs('results', exist_ok=True); json.dump({'name': a.name, 'adapter': a.adapter, **res}, open(f'results/{a.name}.json', 'w'), ensure_ascii=False, indent=1)
