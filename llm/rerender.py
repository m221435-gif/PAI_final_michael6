# 問題ファイル（tray・held・utterance を持つもの）の messages を、今の PROMPT_V の説明文で作り直す。
#   使い方: PROMPT_V=2 python rerender.py data/test_checked.jsonl data_v2/test_checked.jsonl
import json, sys
from common import load_dict, messages
d = load_dict(); src, dst = sys.argv[1], sys.argv[2]
with open(dst, 'w', encoding='utf-8') as f:
    for l in open(src, encoding='utf-8'):
        x = json.loads(l); x['messages'] = messages(d, x['tray'], x['held'], x['utterance']); f.write(json.dumps(x, ensure_ascii=False) + '\n')
print('→', dst)
