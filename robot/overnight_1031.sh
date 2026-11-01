#!/usr/bin/env bash
# 10/31 夜の作業（作成者に任されたもの）: 青い机の場面で 学習画像（自動アノテーション）→ 検出モデル 2 本（固定 / 手首）→ デモ → 成功率 → 動画
cd "$(dirname "$0")"; export TABLE_RGB=0.02,0.08,0.28 OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PRIVACY_CONSENT=N HF_HUB_OFFLINE=1
ISAAC=~/sasorobot/isaac-sim/python.sh; L=overnight_1031.log; log(){ echo "[$(date +%H:%M)] $*" | tee -a $L; }
log 開始
for cam in fixed wrist; do
  for seed in 11 12; do
    (cd robot && timeout 7200 $ISAAC handover_sim.py --stage dataset --camera $cam --n 220 --seed $seed --out runs/ds_blue_${cam}_$seed --per-type 1 --spread 0.2 --cam-z 0.72 --cam-res 1280x720 > runs/ds_blue_${cam}_$seed.log 2>&1)
    log "画像 $cam seed $seed: $(ls robot/runs/ds_blue_${cam}_$seed/images/train 2>/dev/null | wc -l) 学習 / $(ls robot/runs/ds_blue_${cam}_$seed/images/val 2>/dev/null | wc -l) 確認"
  done
  D=robot/runs/ds_blue_$cam; rm -rf $D; mkdir -p $D/images/{train,val} $D/labels/{train,val}
  for seed in 11 12; do for sp in train val; do cp robot/runs/ds_blue_${cam}_$seed/images/$sp/* $D/images/$sp/; cp robot/runs/ds_blue_${cam}_$seed/labels/$sp/* $D/labels/$sp/; done; done
  printf "path: %s\ntrain: images/train\nval: images/val\nnc: 4\nnames: ['scissors', 'forceps', 'scalpel', 'syringe']\n" "$(cd $D && pwd)" > $D/data.yaml
  (cd robot && .venv-det/bin/python train_det.py runs/ds_blue_$cam/data.yaml det_blue_$cam 2>&1 | grep -E "DET_DONE|Error" | tee -a ../$L)
done
for cam in fixed wrist; do
  W=$(pwd)/robot/models/det_blue_$cam.pt
  for seed in 0 1 2; do
    for try_ in 1 2; do
      timeout 3600 llm/.venv/bin/python demo_pipeline.py --out runs/demo_blue_${cam}_s$seed --camera $cam --weights $W --seed $seed --start llm/runs/v2c_dpo-c_start --adapter llm/runs/v2c_dpo-c > robot/demo_blue_${cam}_s$seed.log 2>&1
      [ -f robot/runs/demo_blue_${cam}_s$seed/report.json ] && break; log "デモ $cam seed $seed を やり直し"
    done
    log "デモ $cam seed $seed: $(python3 -c "import json;r=json.load(open('robot/runs/demo_blue_${cam}_s$seed/report.json'));print(r.get('ok'),'/',r.get('n'),[x['say'] for x in r['results'] if not x['ok']])" 2>&1)"
  done
  llm/.venv/bin/python make_video.py robot/runs/demo_blue_${cam}_s0/demo.mp4 提出/PAI最終課題_michael6_blue_$cam.mp4 2>&1 | tee -a $L
done
google-chrome --headless --disable-gpu --no-pdf-header-footer --print-to-pdf="$(pwd)/提出/説明資料_PAI最終課題_michael6.pdf" "file://$(pwd)/提出/説明資料.html" >/dev/null 2>&1 && log "PDF 作成"
bash build_repo.sh >> $L 2>&1 && log "公開用リポジトリ 組み立て"
log 終了
