# 器械出し LLM の共通部品: 器具辞書の読み込み、命令の形、プロンプトの組み立て、採点。
#   命令（LLM の出力）は 1 行の JSON:
#     {"action": "hand" | "return" | "clarify" | "unavailable" | "refuse", "instrument": <器具ID or null>, "candidates": [<器具ID>, ...]}
#   hand: トレーから渡す / return: 医師の手元からトレーへ戻す / clarify: どれか聞き返す / unavailable: トレーにない（4 種類以外の物も含む、そのときは instrument=null）
#   refuse: 危ない・器械出しでない指示（投げる・振り回す等）を断る。instrument=null
import csv, json, os, pathlib, re

ROOT = pathlib.Path(__file__).resolve().parent.parent
DICT_CSV = ROOT / 'data' / 'human' / '器具辞書.csv'
ACTIONS = ('hand', 'return', 'clarify', 'unavailable', 'refuse')


def split_list(s):
    return [x.strip() for x in re.split(r'[、,，/／]', s or '') if x.strip()]


def load_dict(path=DICT_CSV):
    """器具辞書.csv → {ID: {'name', 'aliases', 'uses', 'confusable', 'memo'}}"""
    out = {}
    with open(path, encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            i = (r.get('ID') or '').strip()
            if not i: continue
            out[i] = {'name': r['正式名'].strip() or i, 'aliases': split_list(r['現場での呼び方']), 'test_alias': (r.get('テスト専用の呼び方') or '').strip(),
                      'uses': split_list(r['使う場面・動作']), 'confusable': split_list(r['間違えやすい器具']),
                      'memo': (r.get('メモ') or '').strip()}
    return out


PROMPT_V = int(os.environ.get('PROMPT_V', '1'))   # 1 = 器具は正式名と ID だけ / 2 = 器具カタログ（辞書の呼び方・使う場面）も見せる（テスト専用の呼び方は見せない）


def catalog(d):
    return '器具カタログ:\n' + '\n'.join(f"- {v['name']}({i}): 呼び方 {'、'.join(v['aliases']) or 'なし'}／使う場面 {'、'.join(v['uses']) or 'なし'}" for i, v in d.items())


def system_prompt(d, tray, held):
    """学習・推論で同じものを使う。v1 は器具を正式名と ID だけ見せる。v2 は器具カタログも見せる（2 つ以上に当てはまる言葉なら聞き返す）。"""
    fmt = lambda ids: '、'.join(f'{d[i]["name"]}({i})' for i in ids) or 'なし'
    if PROMPT_V == 2:
        return ('あなたは手術室の器械出しロボットです。医師の言葉を、ロボットへの命令 1 行の JSON に変換してください。\n'
                '形式: {"action": "hand|return|clarify|unavailable|refuse", "instrument": 器具ID または null, "candidates": [器具ID, ...]}\n'
                'hand=トレーから渡す、return=医師の手元からトレーへ戻す、clarify=どの器具か分からないので候補を挙げて聞き返す、'
                'unavailable=頼まれた器具がトレーにない、refuse=器具を投げる・振り回すなど危ない扱いの指示なので断る。\n'
                + catalog(d) + '\n使う場面が 2 つ以上の器具に当てはまるときは clarify。\n'
                f'トレーにある器具: {fmt(tray)}\n医師の手元にある器具: {fmt(held)}')
    return ('あなたは手術室の器械出しロボットです。医師の言葉を、ロボットへの命令 1 行の JSON に変換してください。\n'
            '形式: {"action": "hand|return|clarify|unavailable|refuse", "instrument": 器具ID または null, "candidates": [器具ID, ...]}\n'
            'hand=トレーから渡す、return=医師の手元からトレーへ戻す、clarify=どの器具か分からないので候補を挙げて聞き返す、'
            'unavailable=頼まれた器具がトレーにない、refuse=危ない指示や器械出しでない指示なので断る。\n'
            f'トレーにある器具: {fmt(tray)}\n医師の手元にある器具: {fmt(held)}')


def messages(d, tray, held, utterance):
    return [{'role': 'system', 'content': system_prompt(d, tray, held)}, {'role': 'user', 'content': utterance}]


def target(action, instrument=None, candidates=()):
    """正解の命令（ルールで決まる）を JSON 文字列にする。
    形は学習前の Qwen2.5-1.5B が自然に書く形に合わせる（10/31 probe_base.py: 1 行、action → instrument の順、candidates は空なら書かない）。
    この形は学習の前に固定し、途中で変えない（ソロの教訓）。"""
    o = {'action': action, 'instrument': instrument}
    if candidates: o['candidates'] = sorted(candidates)
    return json.dumps(o, ensure_ascii=False)


def parse(text):
    """LLM の出力から最初の JSON を取り出す。読めなければ None。"""
    m = re.search(r'\{.*?\}', text or '', re.S)
    if not m: return None
    try: o = json.loads(m.group(0))
    except json.JSONDecodeError: return None
    if not isinstance(o, dict) or o.get('action') not in ACTIONS: return None
    o.setdefault('instrument', None); o.setdefault('candidates', [])
    return o


def score(output_text, gold):
    """採点。gold は target() の dict 版。返り値: {'valid', 'correct', 'error'}
       correct の条件: action が一致し、hand/return は instrument が一致、clarify は candidates が集合として一致。"""
    o = parse(output_text)
    if o is None: return {'valid': False, 'correct': False, 'error': 'invalid_json'}
    if o['action'] != gold['action']: return {'valid': True, 'correct': False, 'error': f'action:{gold["action"]}->{o["action"]}'}
    if gold['action'] in ('hand', 'return') and o['instrument'] != gold['instrument']:
        return {'valid': True, 'correct': False, 'error': 'wrong_instrument'}
    if gold['action'] == 'clarify' and set(o.get('candidates') or []) != set(gold['candidates']):
        return {'valid': True, 'correct': False, 'error': 'wrong_candidates'}
    return {'valid': True, 'correct': True, 'error': None}
