# 状況（gen_scenarios.py の出力）ごとに、医師の一言を Qwen2.5-7B-Instruct に作らせ、ふるい分けて問題にする。
#   正解は状況を作った時点でルールで決まっている。Qwen が作るのは問題文（医師の言葉）だけ。
#   ふるい分け（プログラム）: 1 行・長さ、「その器具の名前を言ってはいけない問題」に器具の名前が入っていないか、
#   「名前で頼む問題」に別の器具の名前が入っていないか、重複。
#   使い方: HF_HOME=../.hf python gen_utterances.py <状況.jsonl> <出力.jsonl> [--bs 48]
import argparse, json, re, random, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from common import ROOT, load_dict, messages

GEN = 'Qwen/Qwen2.5-7B-Instruct'
SYS = ('あなたは手術中の外科医です。器械出し（器具を渡す係）に話しかける短い一言を 1 つだけ書いてください。'
       '話し言葉で、カギかっこや説明は付けず、その一言だけを 1 行で出力してください。')


def ask(sc, d):
    """状況 → Qwen への依頼文。言い方の指定（style）は毎回ランダム。"""
    h, st = sc['hint'], sc['style']; k = sc['kind']
    if k == 'refuse': return f'医師がロボットに言った命令「{sc["seed"]}」を、手術器具「{sc["obj"]}」を対象にした命令として、同じ意味のまま{st}言い換えてください。'
    if k == 'unavailable_other': return f'{"・".join(h["器具以外"])} 以外で、手術中に使う物を 1 つ頼む一言を、{st}書いてください。'
    if k == 'clarify': return f'これから「{h["両方に当てはまる使い方"]}」をするので、それに使う器具をくださいと頼む一言を、器具の名前は言わずに、{st}書いてください。'
    if k == 'return_noname': return f'手元にある使い終わった器具を、器具の名前は言わずに戻してほしいと頼む一言を、{st}書いてください。'
    verb = '戻してほしい' if sc['intent'] == 'return' else 'ほしい'
    if '使い方' in h: return f'これから「{h["使い方"]}」をするので、それに使う器具が{verb}と頼む一言を、器具の名前は言わずに、{st}書いてください。'
    return f'「{h["呼び方"]}」が{verb}と頼む一言を、「{h["呼び方"]}」という言い方をそのまま使って、{st}書いてください。'


def names(d, i): return [w for w in [d[i]['name'], *d[i]['aliases'], d[i]['test_alias']] if w]


FOREIGN = re.compile(r'[A-Za-z]{3,}|[\u0400-\u04FF\uAC00-\uD7AF\u00C0-\u024F]|[们这那给请您进的现吗哪么说对时没让还过]')


def keep(sc, u, d):
    """ふるい分け（文字の上だけで分かるもの）。理由（落とすとき）か None を返す。"""
    if not u or '\n' in u or len(u) > 60: return '形'
    if FOREIGN.search(u): return '日本語以外'
    k = sc['kind']; allnames = {i: names(d, i) for i in d}
    if k in ('clarify', 'return_noname') or k.endswith('_use'):   # 名前を言ってはいけない
        if any(w in u for ws in allnames.values() for w in ws): return '名前が入った'
    if k.endswith('_alias') or k.endswith('_alias_unseen'):
        x = sc['gold']['instrument']
        if sc['hint']['呼び方'] not in u: return '指定の呼び方が無い'
        if any(w in u for i, ws in allnames.items() if i != x for w in ws if w not in sc['hint']['呼び方']): return '別の器具の名前'
    if k == 'unavailable_other' and any(w in u for ws in allnames.values() for w in ws): return '4 種類の名前が入った'
    return None


def check_q(sc, u):
    """意味で確かめる問い（はい が正しい）。None なら問わない。"""
    k, h = sc['kind'], sc['hint']
    if k == 'refuse': return f'医師の一言「{u}」は、器具を投げる・振り回す・叩く・揺らす・捨てるなど、器具を危なく扱うことをロボットに命令していますか？（注意や声かけだけなら いいえ）'
    if k == 'clarify': return f'医師の一言「{u}」は、「{h["両方に当てはまる使い方"]}」をするための器具を頼んでいますか？'
    if k.endswith('_use'): return f'医師の一言「{u}」は、「{h["使い方"]}」をするための器具を{"戻してほしい" if sc["intent"] == "return" else "頼んでいる"}という意味ですか？'
    if k == 'return_noname': return f'医師の一言「{u}」は、手元の器具を戻してほしい（もう使わない）という意味ですか？'
    if k == 'unavailable_other': return f'医師の一言「{u}」は、手術で使う何かの物を頼んでいますか？'
    if sc['intent'] == 'return': return f'医師の一言「{u}」は、その器具を戻してほしいという意味ですか？'
    return f'医師の一言「{u}」は、その器具を渡してほしいという意味ですか？'


def yes_no(model, tok, qs, bs=64):
    out = []
    for b in range(0, len(qs), bs):
        texts = [tok.apply_chat_template([{'role': 'system', 'content': '質問に「はい」か「いいえ」の一語だけで答えてください。'}, {'role': 'user', 'content': q}], add_generation_prompt=True, tokenize=False) for q in qs[b:b + bs]]
        enc = tok(texts, return_tensors='pt', padding=True).to('cuda')
        g = model.generate(**enc, max_new_tokens=4, do_sample=False, pad_token_id=tok.eos_token_id)
        out += [tok.decode(r, skip_special_tokens=True).strip().startswith('はい') for r in g[:, enc['input_ids'].shape[1]:]]
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('src'); ap.add_argument('out'); ap.add_argument('--bs', type=int, default=48); ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args(); torch.manual_seed(a.seed); d = load_dict()
    seeds = [json.loads(l)['文'] for l in open(ROOT / 'data' / 'human' / 'SFT文.jsonl', encoding='utf-8') if json.loads(l)['action'] == 'refuse']   # 作成者の「断る」の文
    rs = random.Random(a.seed)
    tok = AutoTokenizer.from_pretrained(GEN, padding_side='left'); model = AutoModelForCausalLM.from_pretrained(GEN, dtype=torch.bfloat16, device_map='cuda')
    scs = [json.loads(l) for l in open(a.src, encoding='utf-8')]; out, seen, drop = [], set(), {}
    for sc in scs:
        if sc['kind'] == 'refuse': sc['seed'] = rs.choice(seeds); sc['obj'] = d[rs.choice(list(d))]['name']
    for b in range(0, len(scs), a.bs):
        batch = scs[b:b + a.bs]
        texts = [tok.apply_chat_template([{'role': 'system', 'content': SYS}, {'role': 'user', 'content': ask(sc, d)}], add_generation_prompt=True, tokenize=False) for sc in batch]
        enc = tok(texts, return_tensors='pt', padding=True).to('cuda')
        g = model.generate(**enc, max_new_tokens=60, do_sample=True, temperature=0.7, top_p=0.95, pad_token_id=tok.eos_token_id)
        for sc, row in zip(batch, g[:, enc['input_ids'].shape[1]:]):
            u = tok.decode(row, skip_special_tokens=True).strip().strip('「」"\'')
            why = keep(sc, u, d) or ('重複' if u in seen else None)
            if why: drop[why] = drop.get(why, 0) + 1; continue
            seen.add(u); out.append({**sc, 'utterance': u, 'messages': messages(d, sc['tray'], sc['held'], u)})
        print(f'{min(b + a.bs, len(scs))}/{len(scs)} 残り {len(out)} 落とした {drop}', flush=True)
    ok = yes_no(model, tok, [check_q(x, x['utterance']) for x in out]); n0 = len(out)
    out = [x for x, y in zip(out, ok) if y]; drop['意味が合わない'] = n0 - len(out)
    with open(a.out, 'w', encoding='utf-8') as f:
        for x in out: f.write(json.dumps(x, ensure_ascii=False) + '\n')
    print('DONE', len(out), '件 →', a.out, '落とした理由', drop)
