# デモの通し: 台本（医師のセリフ）→ 学習した LLM が命令 JSON に変換 → Isaac Sim のロボットが実行（動画つき）。
#   台本: 人が作ったデータ/デモ台本.txt（1 行 1 セリフ。「#カウント」の行は器具カウント）
#   LLM に見せるトレー・手元の状態は、それまでの命令から更新する（渡したら手元へ、戻したらトレーへ）。
#   使い方: llm/.venv/bin/python demo_pipeline.py --out runs/demo1 [--adapter ...] [--start ...]
import argparse, json, os, pathlib, subprocess, sys
HERE = pathlib.Path(__file__).resolve().parent; sys.path.insert(0, str(HERE / 'llm')); os.environ.setdefault('PROMPT_V', '2'); os.environ.setdefault('HF_HUB_OFFLINE', '1')
from common import load_dict, messages, parse
from evaluate import load_model, answer

ap = argparse.ArgumentParser(); ap.add_argument('--out', default='runs/demo1'); ap.add_argument('--script', default=str(HERE / 'data' / 'human' / 'デモ台本.txt'))
ap.add_argument('--start', default=str(HERE / 'llm/runs/v2b_dpo-c_start')); ap.add_argument('--adapter', default=str(HERE / 'llm/runs/v2b_dpo-c'))
ap.add_argument('--llm-only', action='store_true', help='ロボットは動かさず、LLM の命令だけ出す')
ap.add_argument('--camera', default='fixed', choices=['fixed', 'wrist']); ap.add_argument('--weights', default='models/det_v2.pt'); ap.add_argument('--seed', type=int, default=0)
a = ap.parse_args(); d = load_dict(); IDS = list(d)
tok, m = load_model(a.adapter, a.start)
tray, held = list(IDS), []; cmds = []
for line in [l.strip() for l in open(a.script, encoding='utf-8') if l.strip()]:
    if line.startswith('#カウント'): cmds.append({'say': '器具カウントお願い', 'action': 'count'}); continue
    out = answer(tok, m, [messages(d, sorted(tray), sorted(held), line)])[0]; c = parse(out) or {'action': '?'}
    print(f'{line}  →  {out.strip()}   （トレー {sorted(tray)} 手元 {sorted(held)}）', flush=True)
    if c['action'] == 'hand' and c.get('instrument') in tray: tray.remove(c['instrument']); held.append(c['instrument'])
    if c['action'] == 'return' and c.get('instrument') in held: held.remove(c['instrument']); tray.append(c['instrument'])
    import re as _re   # 字幕には LLM が実際に出したキーだけを載せる（parse が補った candidates: [] は載せない）
    m_ = _re.search(r'\{.*?\}', out, _re.S); raw = json.loads(m_.group(0)) if m_ else {}
    cmds.append({'say': line, **{k: raw[k] for k in ('action', 'instrument', 'candidates') if k in raw}, 'llm_raw': out.strip()})
outdir = HERE / 'robot' / a.out; outdir.mkdir(parents=True, exist_ok=True)
with open(outdir / 'commands.jsonl', 'w', encoding='utf-8') as f:
    for c in cmds: f.write(json.dumps(c, ensure_ascii=False) + '\n')
print('命令 →', outdir / 'commands.jsonl')
if a.llm_only: sys.exit(0)
del m; import torch; torch.cuda.empty_cache()
env = dict(os.environ, OMNI_KIT_ACCEPT_EULA='YES', ACCEPT_EULA='Y', PRIVACY_CONSENT='N')
subprocess.run([os.path.expanduser('~/sasorobot/isaac-sim/python.sh'), 'handover_sim.py', '--stage', 'serve', '--commands', str(outdir / 'commands.jsonl'), '--out', a.out,
                '--per-type', '1', '--spread', '0.2', '--perception', 'detector', '--weights', a.weights, '--cam-z', '0.72', '--cam-res', '1280x720', '--camera', a.camera, '--seed', str(a.seed)],
               cwd=HERE / 'robot', env=env, stdout=open(outdir / 'sim.log', 'w'), stderr=subprocess.STDOUT)
print('動画 →', outdir / 'demo.mp4')
