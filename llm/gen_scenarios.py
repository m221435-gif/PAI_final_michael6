# 状況と正解をランダムに作る（文章はまだ作らない。医師の言葉は gen_utterances.py で Qwen に出力させる）。
#   使い方: python gen_scenarios.py <件数> <出力.jsonl> [--split train|test] [--seed N]
#   学習とテストで呼び方を分ける: 辞書の「テスト専用の呼び方」はテスト（--split test）だけで使う。空の器具はテストでも学習と同じ呼び方を使う。
import argparse, json, random
from common import load_dict, target

STYLES = ['丁寧に', '短く一言で', '急いでいる様子で', '略語や現場の言い方で', '前置きや別の話をはさんで', '音声認識の聞き間違いが少し混じった文で']
INTENT_W = {'hand': 0.35, 'return': 0.15, 'clarify': 0.15, 'unavailable': 0.2, 'refuse': 0.15}


def alias_pool(d, i, split):
    t = d[i].get('test_alias')
    if split == 'test' and t: return [t]
    return list(dict.fromkeys([d[i]['name']] + [x for x in d[i]['aliases'] if x != t]))


def shared_uses(d, i, j): return sorted(set(d[i]['uses']) & set(d[j]['uses']))


def confusable_pairs(d):
    """聞き返す問題を作れる組 = 間違えやすい器具に指定され、かつ「使う場面・動作」に共通の言葉がある組。"""
    ps = set()
    for i, v in d.items():
        for j in v['confusable']:
            if j in d and j != i and shared_uses(d, i, j): ps.add(tuple(sorted((i, j))))
    return sorted(ps)


def one(d, rng, split):
    ids = list(d)
    intent = rng.choices(list(INTENT_W), weights=list(INTENT_W.values()))[0]
    pairs = confusable_pairs(d)
    if intent == 'clarify' and not pairs: intent = 'hand'
    if intent == 'refuse':   # 危ない・器械出しでない指示（作成者の SFT 文「なげろ」「ふりまわして」等と同じ種類）
        tray = sorted(rng.sample(ids, rng.randint(1, len(ids))))
        return {'tray': tray, 'held': [], 'intent': 'refuse', 'kind': 'refuse', 'hint': {}, 'style': rng.choice(STYLES), 'gold': json.loads(target('refuse')), 'split': split}
    if intent == 'unavailable' and rng.random() < 0.5:   # 4 種類以外の物を頼む
        tray = sorted(rng.sample(ids, rng.randint(1, len(ids))))
        return {'tray': tray, 'held': [], 'intent': 'unavailable', 'kind': 'unavailable_other', 'hint': {'器具以外': [d[i]['name'] for i in ids]}, 'style': rng.choice(STYLES), 'gold': json.loads(target('unavailable', None)), 'split': split}
    if intent == 'clarify':
        x, y = rng.choice(pairs)
        rest = [i for i in ids if i not in (x, y)]
        tray = sorted({x, y} | set(rng.sample(rest, rng.randint(0, len(rest)))))
        held = []
        gold = json.loads(target('clarify', None, (x, y))); kind = 'clarify'
        hint = {'器具': [d[x]['name'], d[y]['name']], '両方に当てはまる使い方': rng.choice(shared_uses(d, x, y))}
    else:
        x = rng.choice(ids); rest = [i for i in ids if i != x]
        others = rng.sample(rest, rng.randint(1, len(rest)))
        if intent == 'hand':      tray, held = sorted({x} | set(others)), []
        elif intent == 'return':  tray, held = sorted(others), [x]
        else:                     tray, held = sorted(others), []          # unavailable: x はどこにもない
        if intent == 'return' and rng.random() < 0.5 and len(tray) > 1:     # 手元に別の器具もある場合
            o = rng.choice(tray); tray = [t for t in tray if t != o]; held = sorted(held + [o])
        gold = json.loads(target(intent, x))
        # 使い方で頼むのは、その場の他の器具（返すなら手元、それ以外はトレー）に当てはまらない使い方があるときだけ（あると本当は聞き返すのが正解になる）
        near = held if intent == 'return' else tray
        uses = [u for u in d[x]['uses'] if not any(u in d[o]['uses'] for o in near if o != x)]
        if uses and rng.random() < 0.4: hint = {'器具': [d[x]['name']], '使い方': rng.choice(uses)}; kind = f'{intent}_use'
        else:
            hint = {'器具': [d[x]['name']], '呼び方': rng.choice(alias_pool(d, x, split))}; kind = f'{intent}_alias'
            if split == 'test' and hint['呼び方'] == d[x].get('test_alias'): kind += '_unseen'   # 学習で使っていない呼び方
        if intent == 'return' and len(held) == 1 and rng.random() < 0.3: hint = {'器具': [d[x]['name']], '名前なし': True}; kind = 'return_noname'
    return {'tray': tray, 'held': held, 'intent': intent, 'kind': kind, 'hint': hint, 'style': rng.choice(STYLES), 'gold': gold, 'split': split}


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('n', type=int); ap.add_argument('out')
    ap.add_argument('--split', default='train'); ap.add_argument('--seed', type=int, default=0); ap.add_argument('--dict')
    a = ap.parse_args(); d = load_dict(a.dict) if a.dict else load_dict(); rng = random.Random(a.seed)
    with open(a.out, 'w', encoding='utf-8') as f:
        for k in range(a.n): f.write(json.dumps({'id': f'{a.split}_{k:05d}', **one(d, rng, a.split)}, ensure_ascii=False) + '\n')
    print(f'{a.n} 件 → {a.out}（器具 {len(d)} 種類、間違えやすい組 {len(confusable_pairs(d))}）')
