# SFT 用の文を書く画面（ブラウザで開く）。作成者が医師の言葉を打ち、正解の動作と器具をクリックで選んで「追加」する。
#   保存先: 人が作ったデータ/SFT文.jsonl（1 行 1 件）。トレーの状態は build_sft.py が正解に合うように自動で決める。
#   Qwen は使わない（GPU 不要）。使い方: .venv/bin/python sft_app.py → http://127.0.0.1:7861
import json, collections, gradio as gr
from common import ROOT

OUT = ROOT / 'data' / 'human' / 'SFT文.jsonl'
IDS = ['scissors', 'forceps', 'scalpel', 'syringe']
ACT = {'渡す': 'hand', '戻す': 'return', '聞き返す': 'clarify', 'ない': 'unavailable', '断る': 'refuse'}
OTHER = '4 種類以外の物'


def rows():
    try: return [json.loads(l) for l in open(OUT, encoding='utf-8') if l.strip()]
    except FileNotFoundError: return []


def summary():
    rs = rows(); ca = collections.Counter(r['動作'] for r in rs); ci = collections.Counter(i for r in rs for i in r['器具'])
    head = f"**{len(rs)} 件**　動作: " + '、'.join(f'{k} {ca.get(k, 0)}' for k in ACT) + '　器具: ' + '、'.join(f'{i} {ci.get(i, 0)}' for i in IDS)
    table = [[k + 1, r['文'], r['動作'], '・'.join(r['器具'])] for k, r in enumerate(rs)][::-1]
    return head, table


def add(text, act, one, two):
    text = (text or '').strip()
    if not text: return '文が空です', *summary(), text
    if act is None: return '正解の動作を選んでください', *summary(), text
    inst = list(two) if act == '聞き返す' else [] if act == '断る' else (['other'] if one == OTHER else [one] if one else [])
    if one == OTHER and act != 'ない': return '「4 種類以外の物」は「ない」のときだけ選べます', *summary(), text
    if act == '聞き返す' and len(inst) != 2: return '聞き返すときは、候補の器具を 2 つ選んでください', *summary(), text
    if act not in ('聞き返す', '断る') and len(inst) != 1: return '器具を 1 つ選んでください', *summary(), text
    with open(OUT, 'a', encoding='utf-8') as f: f.write(json.dumps({'文': text, '動作': act, 'action': ACT[act], '器具': inst}, ensure_ascii=False) + '\n')
    return f'追加しました: {text}', *summary(), ''


def undo():
    rs = rows()
    if rs:
        with open(OUT, 'w', encoding='utf-8') as f:
            for r in rs[:-1]: f.write(json.dumps(r, ensure_ascii=False) + '\n')
    return '最後の 1 件を消しました' if rs else '消すものがありません', *summary()


with gr.Blocks(title='SFT 用の文') as demo:
    gr.Markdown('## SFT 用の文を書く\n医師の言葉を 1 つ打って、**正解の動作**と**器具**を選んで「追加」。20〜30 件が目安です。\n'
                '- 渡す・戻す・ない → 器具を 1 つ　／　聞き返す → どちらとも取れる器具を 2 つ　／　断る（危ない指示）→ 器具は選ばない\n'
                '- ない: トレーから外された器具を頼む文（器具を選ぶ）と、4 種類以外の物を頼む文（「4 種類以外の物」を選ぶ）の両方\n'
                '- トレーに何があるかは自動で決めます（「ない」を選べば、その器具はトレーに無い状態にします）')
    text = gr.Textbox(label='医師の言葉', placeholder='ここに打つ（Enter でも追加できます）')
    act = gr.Radio(list(ACT), label='正解の動作')
    one = gr.Radio(IDS + [OTHER], label='器具（渡す・戻す・ない）。4 種類以外の物を頼む「ない」は「4 種類以外の物」')
    two = gr.CheckboxGroup(IDS, label='候補の器具 2 つ（聞き返す）')
    with gr.Row():
        b_add = gr.Button('追加', variant='primary'); b_undo = gr.Button('最後の 1 件を消す')
    msg = gr.Markdown(); head = gr.Markdown(summary()[0])
    table = gr.Dataframe(headers=['番号', '文', '動作', '器具'], value=summary()[1], interactive=False)
    b_add.click(add, [text, act, one, two], [msg, head, table, text]); text.submit(add, [text, act, one, two], [msg, head, table, text])
    b_undo.click(undo, None, [msg, head, table])

if __name__ == '__main__':
    demo.launch(server_name='127.0.0.1', server_port=7861)
