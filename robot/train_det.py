# 検出モデルの学習（ultralytics YOLOv8n）。det_v2 を出発点に、新しい場面の自動アノテーション画像で学習し直す。
#   使い方: .venv-det/bin/python train_det.py <data.yaml> <名前>  → models/<名前>.pt
import sys, shutil, pathlib
from ultralytics import YOLO
data, name = sys.argv[1], sys.argv[2]; here = pathlib.Path(__file__).resolve().parent
m = YOLO(str(here / '../../ロボットアーム_ピックアップ/models/det_v2.pt'))
r = m.train(data=data, imgsz=1280, epochs=60, batch=8, device=0, project=str(here / 'runs/det_train'), name=name, exist_ok=True, patience=20, verbose=False, plots=False)
best = here / 'runs/det_train' / name / 'weights' / 'best.pt'; (here / 'models').mkdir(exist_ok=True); shutil.copy(best, here / 'models' / f'{name}.pt')
met = YOLO(str(best)).val(data=data, imgsz=1280, verbose=False); print(f'DET_DONE {name} mAP50-95={met.box.map:.3f} mAP50={met.box.map50:.3f}')
