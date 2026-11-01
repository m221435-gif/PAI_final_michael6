# 器具辞書を作る画面（ブラウザで開く）。作成者が自分で書く（Qwen は使わない、10/31 作成者「Qwen はいらん」）。
#   1 欄に 1 語。空の欄は無視する。保存先: 人が作ったデータ/器具辞書.csv（学習・テストが読む）
#   使い方: .venv/bin/python dict_app.py [ポート] → http://127.0.0.1:7862
import csv, sys, gradio as gr
from common import DICT_CSV, split_list

IDS = ['scissors', 'forceps', 'scalpel', 'syringe']   # 検出モデルのクラス名（変えない）
N_ALIAS, N_USE = 10, 10                                # 欄の数


def load():
    try: return {r['ID']: r for r in csv.DictReader(open(DICT_CSV, encoding='utf-8-sig'))}
    except FileNotFoundError: return {}


def pad(xs, n): xs = list(xs)[:n]; return xs + [''] * (n - len(xs))


with gr.Blocks(title='器具辞書づくり') as demo:
    ex = load(); comps = []
    gr.Markdown('## 器具辞書づくり\n1 つの欄に 1 語ずつ書きます。空の欄はそのままで大丈夫です。全部終わったら一番下の「保存」（途中で押しても上書きされるだけです）。\n'
                '- **呼び方**: 器具そのものの名前（略語・カタカナ・英語・俗称・商品名など、手術室で実際に言うもの）\n'
                '- **使う場面・動作**: その器具で何をしたいか（「〜やつ」は付けず、動作や場面だけ）。**その器具だけに当てはまるもの**を 2 つ以上あると安心です\n'
                '- **テスト専用の呼び方**: 学習には使わず、「学習していない言い方にも対応できるか」の確認だけに使う 1 語')
    for i in IDS:
        r = ex.get(i, {}); al = split_list(r.get('現場での呼び方', '')); us = split_list(r.get('使う場面・動作', ''))
        test0 = r.get('テスト専用の呼び方', '')
        with gr.Tab(i):
            name = gr.Textbox(label='正式名（代表の名前 1 つ）', value=r.get('正式名', ''))
            gr.Markdown('**呼び方**')
            with gr.Row(): aliases = [gr.Textbox(show_label=False, value=v, placeholder=f'呼び方 {k + 1}', min_width=120) for k, v in enumerate(pad(al, N_ALIAS))]
            gr.Markdown('**使う場面・動作**')
            with gr.Row(): uses = [gr.Textbox(show_label=False, value=v, placeholder=f'場面・動作 {k + 1}', min_width=120) for k, v in enumerate(pad(us, N_USE))]
            test = gr.Textbox(label='テスト専用の呼び方（1 つ）', value=test0)
            conf = gr.CheckboxGroup(label='間違えやすい器具（片方に付ければ両方に付きます）', choices=[j for j in IDS if j != i], value=split_list(r.get('間違えやすい器具', '')))
            memo = gr.Textbox(label='メモ（空でよい）', value=r.get('メモ', ''))
        comps.append((i, name, aliases, uses, test, conf, memo))
    status = gr.Markdown()
    flat = [c for (_, name, al, us, test, conf, memo) in comps for c in [name, *al, *us, test, conf, memo]]

    def save(*vals):
        it = iter(vals); data = {}
        for (i, *_r) in comps:
            name = next(it).strip(); al = [next(it).strip() for _ in range(N_ALIAS)]; us = [next(it).strip() for _ in range(N_USE)]
            test = next(it).strip(); conf = list(next(it)); memo = next(it)
            al = [x for x in dict.fromkeys(al) if x and x != test]; us = [x for x in dict.fromkeys(us) if x]
            data[i] = {'正式名': name, '呼び方': al, 'テスト': test, '使う': us, '間違えやすい': conf, 'メモ': memo}
        for i in IDS:   # 間違えやすい器具は両方向にそろえる
            for j in data[i]['間違えやすい']:
                if i not in data[j]['間違えやすい']: data[j]['間違えやすい'].append(i)
        with open(DICT_CSV, 'w', encoding='utf-8-sig', newline='') as fh:
            w = csv.writer(fh); w.writerow(['ID', '正式名', '現場での呼び方', 'テスト専用の呼び方', '使う場面・動作', '間違えやすい器具', 'メモ'])
            for i in IDS: d = data[i]; w.writerow([i, d['正式名'], '、'.join(d['呼び方']), d['テスト'], '、'.join(d['使う']), '、'.join(d['間違えやすい']), d['メモ']])
        # 気づいたことを返す（使い方が他の器具と全部重なっている等）
        notes = []
        for i in IDS:
            own = [u for u in data[i]['使う'] if not any(u in data[j]['使う'] for j in IDS if j != i)]
            if len(data[i]['呼び方']) < 1: notes.append(f'{i}: 呼び方がありません（1 つ以上必要）')
            if data[i]['使う'] and not own: notes.append(f'{i}: 使う場面・動作がすべて他の器具と重なっています（使い方で頼む問題は全部「聞き返す」になります）')
        return '保存しました → 器具辞書.csv' + ('\n\n' + '\n'.join('- ' + n for n in notes) if notes else '')
    gr.Button('保存', variant='primary').click(save, inputs=flat, outputs=status)

if __name__ == '__main__':
    demo.launch(server_name='127.0.0.1', server_port=int(sys.argv[1]) if len(sys.argv) > 1 else 7862)
