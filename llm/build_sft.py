# 作成者の SFT 文（人が作ったデータ/SFT文.jsonl）に、正解に合うトレーの状態を付けて SFT データにする。
#   文は作成者が書いたものだけ。プログラムがするのは「状況（トレー・手元）を付けること」と、次の 2 つの増やし方だけ:
#     - 名前なしの「戻す」文（これ戻して 等）: 4 器具それぞれを手元に置いた状況で 1 件ずつ
#     - 「渡す」文: 同じ文で、その器具をトレーから外した状況 → 正解は「ない」
#   使い方: python build_sft.py [--k 4] [--val 4] [--seed 0]  → data/sft_train.jsonl, data/sft_val.jsonl
import argparse, json, random
from common import ROOT, load_dict, messages, target

IDS = ['scissors', 'forceps', 'scalpel', 'syringe']


def words(d, i): return [w for w in [d[i]['name'], *d[i]['aliases'], d[i]['test_alias'], *d[i]['uses']] if w]


def mentions(d, text):
    """文に辞書の言葉が出てくる器具（呼び方・使い方の文字が含まれるか。名前なしの判定にだけ使う）"""
    return {i for i in IDS if any(w in text for w in words(d, i))}


def contexts(rng, d, r, k):
    """1 つの文に、正解に合う状況を k 通り付ける。返り値: [(tray, held, gold_json), ...]"""
    act, inst, text = r['action'], r['器具'], r['文']; out = []
    others = lambda ex: [i for i in IDS if i not in ex]
    for _ in range(k):
        if act == 'hand':
            x = inst[0]; tray = sorted({x} | set(rng.sample(others([x]), rng.randint(0, 3)))); out.append((tray, [], target('hand', x)))
        elif act == 'return':
            if mentions(d, text):   # 名前（か使い方）がある: 手元に別の器具が一緒にあってもよい
                x = inst[0]; held = sorted({x} | set(rng.sample(others([x]), rng.randint(0, 1)))); tray = sorted(rng.sample(others(held), rng.randint(1, len(others(held)))))
                out.append((tray, held, target('return', x)))
            else:                   # 名前なし: 手元にある 1 つが正解
                x = rng.choice(IDS); tray = sorted(rng.sample(others([x]), rng.randint(1, 3))); out.append((tray, [x], target('return', x)))
        elif act == 'clarify':
            tray = sorted(set(inst) | set(rng.sample(others(inst), rng.randint(0, 2)))); out.append((tray, [], target('clarify', None, inst)))
        elif act == 'unavailable':
            x = inst[0]
            if x == 'other': tray = sorted(rng.sample(IDS, rng.randint(1, 4))); out.append((tray, [], target('unavailable', None)))
            else: tray = sorted(rng.sample(others([x]), rng.randint(1, 3))); out.append((tray, [], target('unavailable', x)))
        elif act == 'refuse':
            tray = sorted(rng.sample(IDS, rng.randint(1, 4))); held = rng.sample(others(tray), rng.randint(0, min(1, len(others(tray))))); out.append((tray, sorted(held), target('refuse', None)))
    return out


def build(rs, d, rng, k):
    ex = []
    for r in rs:
        for tray, held, gold in contexts(rng, d, r, k): ex.append({'messages': messages(d, tray, held, r['文']) + [{'role': 'assistant', 'content': gold}], 'src': r['文'], 'kind': r['action']})
        if r['action'] == 'hand':   # 同じ文で「トレーに無い」状況 → ない
            x = r['器具'][0]; tray = sorted(rng.sample([i for i in IDS if i != x], rng.randint(1, 3)))
            ex.append({'messages': messages(d, tray, [], r['文']) + [{'role': 'assistant', 'content': target('unavailable', x)}], 'src': r['文'], 'kind': 'unavailable(由来:hand)'})
    rng.shuffle(ex); return ex


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--k', type=int, default=4); ap.add_argument('--val', type=int, default=4); ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args(); rng = random.Random(a.seed); d = load_dict()
    rs = [json.loads(l) for l in open(ROOT / 'data' / 'human' / 'SFT文.jsonl', encoding='utf-8') if l.strip()]
    order = list(range(len(rs))); rng.shuffle(order); val_idx = set(order[:a.val])   # 確認用は文ごと分ける（同じ文が学習と確認に両方入らないように）
    tr = build([r for k, r in enumerate(rs) if k not in val_idx], d, rng, a.k); va = build([r for k, r in enumerate(rs) if k in val_idx], d, rng, 1)
    from common import PROMPT_V
    od = ROOT / 'llm' / ('data' if PROMPT_V == 1 else f'data_v{PROMPT_V}'); od.mkdir(exist_ok=True)
    for name, xs in (('sft_train', tr), ('sft_val', va)):
        with open(od / f'{name}.jsonl', 'w', encoding='utf-8') as f:
            for x in xs: f.write(json.dumps(x, ensure_ascii=False) + '\n')
    import collections
    print(f'文 {len(rs)}（学習 {len(rs) - len(val_idx)} / 確認 {len(val_idx)}）→ 学習 {len(tr)} 件・確認 {len(va)} 件')
    print('学習の種類:', dict(collections.Counter(x['kind'] for x in tr)))
