# 学習の段階ごとの正答率（人が確かめたテスト 126 問）を 1 枚の折れ線にする → 図/学習の段階と正答率.png
#   色: dataviz の既定パレット（categorical 1〜4、validate_palette.js で確認済み。コントラストが低い色があるので線の端に必ず文字ラベルを付ける）
import json, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import glob
f = (glob.glob('/usr/share/fonts/**/NotoSansCJK-Regular.ttc', recursive=True) or glob.glob('/usr/share/fonts/**/NotoSansCJK*.ttc', recursive=True))[0]
fm.fontManager.addfont(f); plt.rcParams['font.family'] = fm.FontProperties(fname=f).get_name()
STAGES = [('学習前', 'F_base'), ('SFT', 'F_v2c_sft-160'), ('RFT', 'F_v2c_rft-150'), ('DPO', 'F_v2c_dpo-c')]
R = {s: json.load(open(f'llm/results/{n}.json'))['test_final.jsonl'] for s, n in STAGES}
def acc(r, kinds):
    a = sum(r['per_kind'].get(k, [0, 0])[0] for k in kinds); b = sum(r['per_kind'].get(k, [0, 0])[1] for k in kinds); return 100 * a / b, b
SERIES = [('全体', None, '#0b0b0b', 'o'), ('聞き返す', ['clarify'], '#2a78d6', 's'), ('使い方で頼む', ['hand_use'], '#eb6834', '^'),
          ('危ない指示を断る', ['refuse'], '#1baf7a', 'D'), ('学習していない呼び方', ['hand_alias_unseen', 'return_alias_unseen', 'unavailable_alias_unseen'], '#eda100', 'v')]
ends = []   # 線の端のラベル（重ならないように後でずらす）
fig, ax = plt.subplots(figsize=(10, 5.6), dpi=150); fig.patch.set_facecolor('#fcfcfb'); ax.set_facecolor('#fcfcfb'); x = range(len(STAGES))
for name, kinds, col, mk in SERIES:
    ys, ns = zip(*[(R[s]['acc'] * 100, R[s]['total'][1]) if kinds is None else acc(R[s], kinds) for s, _ in STAGES])
    ax.plot(x, ys, color=col, lw=3 if kinds is None else 2, marker=mk, ms=8, mec='#fcfcfb', mew=2, zorder=3 if kinds is None else 2, label=f'{name}（{ns[0]} 問）')
    ends.append([ys[-1], ys[-1], f'{name} {ys[-1]:.0f}%', kinds is None])
    if kinds is None:
        for xi, y in zip(x, ys): ax.annotate(f'{y:.1f}%', (xi, y), xytext=(14, -14) if xi == 0 else (0, 10), textcoords='offset points', ha='left' if xi == 0 else 'center', fontsize=10, color='#0b0b0b')
ends.sort(key=lambda e: e[0])
for k in range(1, len(ends)):   # 5 ポイント以上離す
    if ends[k][1] - ends[k - 1][1] < 5: ends[k][1] = ends[k - 1][1] + 5
for y0, y, txt, bold in ends:
    ax.annotate(txt, (x[-1], y0), xytext=(x[-1] + 0.12, y), textcoords='data', va='center', fontsize=11, color='#0b0b0b' if bold else '#52514e', fontweight='bold' if bold else 'normal')
ax.set_xticks(list(x), [s for s, _ in STAGES], fontsize=12); ax.set_ylim(0, 105); ax.set_ylabel('正答率（%）', color='#52514e'); ax.set_xlim(-0.2, len(STAGES) - 0.2)
ax.grid(axis='y', color='#e4e3df', lw=0.8); ax.set_axisbelow(True)
for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
for sp in ('left', 'bottom'): ax.spines[sp].set_color('#c9c8c3')
ax.tick_params(colors='#52514e'); ax.legend(loc='lower right', frameon=False, fontsize=10)
ax.set_title('医師の言葉 → 命令：学習の段階ごとの正答率（人が確かめたテスト 126 問、Qwen2.5-1.5B）', fontsize=13, loc='left', color='#0b0b0b')
plt.subplots_adjust(right=0.78); plt.savefig('図/学習の段階と正答率.png', facecolor='#fcfcfb'); print('saved')
