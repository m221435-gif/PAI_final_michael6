"""検出モデルのワーカー（Isaac の venv に ultralytics を入れないための別プロセス）。
  anaconda の python で起動し、標準入力から画像パスを 1 行ずつ受け取り、検出結果を JSON 1 行で返す。
  python tools/detector_worker.py <weights.pt> <conf>
"""
import sys, json
from ultralytics import YOLO
m = YOLO(sys.argv[1]); conf = float(sys.argv[2]) if len(sys.argv) > 2 else 0.4
print('READY', sys.argv[1], flush=True)
for line in sys.stdin:
    p = line.strip()
    if not p: continue
    r = m.predict(p, imgsz=1280, conf=conf, verbose=False)[0]
    out = [{'cls': int(c), 'name': r.names[int(c)], 'conf': round(float(cf), 3), 'xyxy': [round(float(v), 1) for v in b]} for c, cf, b in zip(r.boxes.cls, r.boxes.conf, r.boxes.xyxy)]
    print(json.dumps(out), flush=True)
