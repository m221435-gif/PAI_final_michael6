# 提出用の動画を組み立てる: タイトル → 仕組み → デモ（Isaac Sim の録画）→ 学習の結果 → まとめ。1280x720、30fps。
#   使い方: llm/.venv/bin/python make_video.py robot/runs/demo4/demo.mp4 提出/PAI最終課題.mp4
import glob, os, subprocess, sys, tempfile
from PIL import Image, ImageDraw, ImageFont
DEMO, OUT = sys.argv[1], sys.argv[2]; CAMERA = sys.argv[3] if len(sys.argv) > 3 else 'fixed'; W, H = 1280, 720
FONT = (glob.glob('/usr/share/fonts/**/NotoSansCJK-Bold.ttc', recursive=True) + glob.glob('/usr/share/fonts/**/NotoSansCJK*.ttc', recursive=True))[0]
F = lambda n: ImageFont.truetype(FONT, n, index=0)
BG, INK, SUB, ACC = (252, 252, 251), (11, 11, 11), (82, 81, 78), (42, 120, 214)


def slide(lines, img=None):
    im = Image.new('RGB', (W, H), BG); d = ImageDraw.Draw(im); y = 60
    if img:
        g = Image.open(img).convert('RGB'); g.thumbnail((W - 120, H - 150)); im.paste(g, ((W - g.size[0]) // 2, 120))
    for txt, size, col in lines:
        if txt == '---': y += 18; continue
        d.text((80, y), txt, font=F(size), fill=col); y += int(size * 1.55)
    return im


tmp = tempfile.mkdtemp(); parts = []
def still(name, im, sec):
    p = f'{tmp}/{name}.png'; im.save(p); v = f'{tmp}/{name}.mp4'
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-loop', '1', '-i', p, '-t', str(sec), '-r', '30', '-vf', 'format=yuv420p', '-c:v', 'libx264', v], check=True); parts.append(v)

still('1_title', slide([('', 60, INK), ('言葉で頼むと器具を手渡す', 54, INK), ('器械出しロボット', 54, ACC), ('---', 0, INK),
                        ('手術室の言い方を理解する LLM × 神視点なしの知覚と物理把持', 26, SUB), ('Isaac Sim / xArm6 / Qwen2.5-1.5B（SFT → RFT → DPO）', 24, SUB)]), 5)
still('2_how', slide([('仕組み', 40, ACC), ('---', 0, INK),
                      ('① 医師の言葉 →  LLM が命令（JSON）に変換', 30, INK), ('　 渡す ／ 戻す ／ 聞き返す ／ ない ／ 危ない指示は断る', 26, SUB),
                      (('② 手首のカメラ（RGB＋深度）で台の上から見る' if CAMERA == 'wrist' else '② 作業台の上の固定カメラ（RGB＋深度）') + ' →  検出モデルで器具を探す（正解の位置は使わない）', 26, INK),
                      ('③ 深度から位置と向き →  指で挟んで持ち上げ、手渡し台・トレーへ', 30, INK),
                      ('④ 最後に器具カウント（数え漏れの防止）', 30, INK), ('---', 0, INK),
                      ('画面下: 医師の言葉・LLM の出力・ロボットの返事　右上: ロボットのカメラの映像', 24, SUB),
                      ('※ 斜めからの画角は説明用の表示で、ロボットはこのカメラを使っていない', 22, SUB)]), 11)
# デモは 1280x720 に合わせる（そのまま等速）
v = f'{tmp}/3_demo.mp4'; subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', DEMO, '-vf', f'scale={W}:{H},fps=30,format=yuv420p', '-c:v', 'libx264', '-an', v], check=True); parts.append(v)
still('4_result', slide([('LLM の学習（人が確かめたテスト 126 問）', 34, ACC)], img='図/学習の段階と正答率.png'), 11)
still('5_end', slide([('まとめ', 40, ACC), ('---', 0, INK),
                      ('・人が書いた 37 文から SFT → RFT → DPO：正答率 11% → 85%', 28, INK),
                      ('・どちらとも取れる言い方は聞き返す（15/15）、危ない指示は断る（16/17）', 28, INK),
                      ('・カメラ画像と深度だけで器具を見つけ、物理で掴んで手渡し', 28, INK), (('　 手首カメラ' if CAMERA == 'wrist' else '　 固定カメラ') + '：器具の置き方 3 通りで 8 命令すべて成功（24/24）', 26, INK), ('---', 0, INK),
                      ('課題: 「使い方で頼む」は DPO で下がった（17→12/22）、実機での確認、音声入力', 24, SUB)]), 9)
lst = f'{tmp}/list.txt'; open(lst, 'w').write(''.join(f"file '{p}'\n" for p in parts))
os.makedirs(os.path.dirname(OUT) or '.', exist_ok=True)
subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'concat', '-safe', '0', '-i', lst, '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', OUT], check=True)
d = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', OUT], capture_output=True, text=True).stdout.strip()
print('完成:', OUT, f'{float(d):.1f} 秒')
