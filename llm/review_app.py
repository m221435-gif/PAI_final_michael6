# テスト問題のチェック画面（ブラウザ）。Qwen 7B が作った問題文と、ルールで決めた正解を 1 件ずつ見せ、作成者が判定する。
#   「OK」= 文も正解も合っている / 「捨てる」= 文がおかしい / 正解だけ違うときは、下で正しい答えを選んで「直して OK」。
#   保存: 人が作ったデータ/テスト確認.jsonl（id ごとに最後の判定が有効）。チェック済みの問題だけが「人が確かめたテスト」になる。
#   使い方: .venv/bin/python review_app.py [data/test_u.jsonl] → http://127.0.0.1:7863
import json, sys, random, gradio as gr
from common import ROOT

SRC = ROOT / 'llm' / (sys.argv[1] if len(sys.argv) > 1 else 'data/test_u.jsonl')
OUT = ROOT / 'data' / 'human' / 'テスト確認.jsonl'
IDS = ['scissors', 'forceps', 'scalpel', 'syringe']; JP = {'scissors': 'ハサミ', 'forceps': 'ピンセット', 'scalpel': 'メス', 'syringe': '注射器'}
ACT = {'hand': '渡す', 'return': '戻す', 'clarify': '聞き返す', 'unavailable': 'ない', 'refuse': '断る'}; ACT_R = {v: k for k, v in ACT.items()}
items = [json.loads(l) for l in open(SRC, encoding='utf-8')]
random.Random(0).shuffle(items)   # 種類が偏らないように混ぜた順で見せる（毎回同じ順）


def done():
    try: return {r['id']: r for r in map(json.loads, open(OUT, encoding='utf-8'))}
    except FileNotFoundError: return {}


def gold_jp(g):
    a = ACT[g['action']]
    if g['action'] == 'clarify': return f"{a}（{'・'.join(JP[i] for i in g.get('candidates', []))}）"
    if g.get('instrument'): return f"{a}（{JP[g['instrument']]}）"
    return a + ('（4 種類以外の物）' if g['action'] == 'unavailable' else '')


def show(_=None):
    dn = done(); rest = [x for x in items if x['id'] not in dn]
    head = f"チェック済み **{len(dn)}** 件（OK {sum(r['判定'] != '捨てる' for r in dn.values())}・捨てる {sum(r['判定'] == '捨てる' for r in dn.values())}）／ 全 {len(items)} 件"
    if not rest: return head, '全部チェックしました', '', '', ''
    x = rest[0]; fmt = lambda ids: '、'.join(JP[i] for i in ids) or 'なし'
    return head, f"## 「{x['utterance']}」", f"トレー: {fmt(x['tray'])}　／　医師の手元: {fmt(x['held'])}", f"### 正解（ルールで決めたもの）: {gold_jp(x['gold'])}", x['id']


def record(xid, verdict, gold=None):
    if not xid: return show()
    x = next(i for i in items if i['id'] == xid)
    with open(OUT, 'a', encoding='utf-8') as f: f.write(json.dumps({'id': xid, '文': x['utterance'], '判定': verdict, 'gold': gold or x['gold'], 'kind': x['kind']}, ensure_ascii=False) + '\n')
    return show()


def fix(xid, act, one, two):
    if not xid or not act: return show()
    a = ACT_R[act]; g = {'action': a, 'instrument': one if a in ('hand', 'return') or (a == 'unavailable' and one) else None}
    if a == 'clarify': g['candidates'] = sorted(two)
    return record(xid, '直して OK', g)


with gr.Blocks(title='テスト問題のチェック') as demo:
    gr.Markdown('## テスト問題のチェック\n医師の言葉と、その状況での正解を見て判定してください。**200〜300 件**が目安です（途中でやめても保存されています）。\n'
                '- 文も正解も合っている → **OK**　／　文がおかしい・不自然 → **捨てる**\n- 文は良いが正解が違う → 下で正しい答えを選んで **直して OK**')
    head = gr.Markdown(); utt = gr.Markdown(); ctx = gr.Markdown(); gold = gr.Markdown(); xid = gr.Textbox(visible=False)
    with gr.Row():
        b_ok = gr.Button('OK', variant='primary'); b_bad = gr.Button('捨てる', variant='stop')
    with gr.Accordion('正解が違うとき', open=False):
        act = gr.Radio(list(ACT_R), label='正しい動作'); one = gr.Radio(IDS, label='器具（渡す・戻す・ない。4 種類以外の物の「ない」は選ばない）'); two = gr.CheckboxGroup(IDS, label='候補 2 つ（聞き返す）')
        b_fix = gr.Button('直して OK')
    outs = [head, utt, ctx, gold, xid]
    demo.load(show, None, outs); b_ok.click(lambda i: record(i, 'OK'), xid, outs); b_bad.click(lambda i: record(i, '捨てる'), xid, outs); b_fix.click(fix, [xid, act, one, two], outs)

if __name__ == '__main__':
    demo.launch(server_name='127.0.0.1', server_port=int(sys.argv[2]) if len(sys.argv) > 2 else 7863)
