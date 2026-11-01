"""器械出しロボット（PAI 最終課題）: ロボットアーム_ピックアップ/scripts/pickup_sim.py を元に、
  「命令（LLM の出力 JSON）を 1 つずつ受け取り、頼まれた器具だけを手渡し台へ／手渡し台からトレーへ戻す」形にしたもの。
  元のスクリプトは変えていない。ロボット・器具・検出モデルのファイルは ロボットアーム_ピックアップ/ のものを読むだけ。
  --stage serve --commands <命令.jsonl> : 命令を順に実行（動画 + 結果 JSON）。命令の形は llm/common.py と同じ
    {"action": "hand"|"return"|"clarify"|"unavailable"|"count", "instrument": <ID>, "candidates": [...], "say": <医師の言葉（字幕用）>}

以下は元のスクリプトの説明:
ロボットアーム ピックアップ（Isaac Sim 5.1）: xArm7 ＋ 固定 RealSense ＋ 散らばった器具 4 種 → 種類ごとの箱へ。

  OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PRIVACY_CONSENT=N .venv-isaac/bin/python scripts/pickup_sim.py --stage scene --out runs/v1
  --stage scene : 場面を組んで静止画（俯瞰・RealSense RGB/深度/ラベル）を保存
  --stage ik    : 1 個目の器具の真上まで腕を動かして静止画
  --stage full  : 全部の器具を箱へ（動画 + 結果 JSON）
器具は Isaac for Healthcare の USD（cm 単位 → scale 0.01）。知覚は段階 1 = 深度で「机より上の塊」を切り出し、
種類はシミュレータの正解ラベル（semantic）で読む（後で検出モデルに差し替える）。
"""
import argparse, json, math, os, time
ap = argparse.ArgumentParser()
ap.add_argument('--stage', default='scene', choices=['scene', 'ik', 'grasp', 'poke', 'cube', 'full', 'serve', 'dataset'])
ap.add_argument('--out', default='runs/v1'); ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--per-type', type=int, default=2); ap.add_argument('--max-items', type=int, default=99)
ap.add_argument('--spread', type=float, default=0.25, help='器具を散らばせる y の半幅 [m]（x は 0.28〜0.68）'); ap.add_argument('--min-gap', type=float, default=0.07, help='器具どうしの最小間隔 [m]')
ap.add_argument('--video', action='store_true', default=True); ap.add_argument('--target-type', default=None); ap.add_argument('--grasp-mode', default='physics', choices=['attach', 'physics'])
ap.add_argument('--arm-maxforce', type=float, default=300.0); ap.add_argument('--grip-maxforce', type=float, default=150.0); ap.add_argument('--grip-stiff', type=float, default=20000.0)   # 指は N（直動）
ap.add_argument('--item-mass', type=float, default=0.15); ap.add_argument('--physics-dt', type=float, default=1/240); ap.add_argument('--solver-iters', type=int, default=32)
ap.add_argument('--perception', default='depth', choices=['semantic', 'depth', 'detector']); ap.add_argument('--no-mimic', action='store_true')
ap.add_argument('--weights', default='models/det_v1.pt', help='--perception detector の重み（YOLO 4 クラス）'); ap.add_argument('--det-conf', type=float, default=0.4)
ap.add_argument('--cam-z', type=float, default=None, help='RealSense の高さ [m]。未指定なら 0.95（spread で上げる）。検出モデルの学習画像は 0.7'); ap.add_argument('--cam-res', default='640x480', help='RealSense の解像度。学習画像は 1280x720')
ap.add_argument('--no-park', action='store_true', help='知覚の前に腕をカメラの視野からよけない')
ap.add_argument('--dof', type=int, default=6, choices=[6, 7], help='xArm の軸数。研究室の実機は 6 軸（作成者 2026-10-17）。7 で xArm7 版')
# ---- 実機の厳しさを再現する設定（既定はオフ。--real でまとめてオン）。先生「そんな簡単に分かるはずがない」への答え（2026-10-17）
ap.add_argument('--camera', default='fixed', choices=['fixed', 'wrist'], help='fixed = 机の上の固定カメラ / wrist = 手首（URDF の camera_link、実機のカメラ台と同じ場所）')
ap.add_argument('--start', default='home', choices=['home', 'init'], help='init = 実機の開始姿勢 [0,0,0,0,-90,0]（手先水平）から始めて、まず撮影姿勢へ動く')
ap.add_argument('--scan-pose', default='0.40,0,0.40', help='wrist のときの撮影姿勢: TCP の x,y,z [m]、手先真下（実機の scan_pose_mm と同じ）')
ap.add_argument('--real', action='store_true', help='深度ノイズ 1.5mm・量子化 1.4mm・金属面の抜け 30%・カメラ誤差 3mm/0.5°・摩擦 1.0 をまとめてオン')
ap.add_argument('--depth-noise', type=float, default=0.0, help='深度のガウスノイズ σ [mm]（0.7 m で。距離の 2 乗で増える。D435 は 1〜2）')
ap.add_argument('--depth-quant', type=float, default=0.0, help='深度の量子化幅 [mm]（0.7 m で。視差 1/8 画素相当 ≈ 1.4）')
ap.add_argument('--depth-dropout', type=float, default=0.0, help='器具の画素のうち深度が抜ける割合（金属の反射）0〜1')
ap.add_argument('--calib-err', default='0,0', help='カメラのキャリブレーション誤差 "mm,deg"（平行移動と z 回転。実行ごとに固定の乱数）')
ap.add_argument('--friction', type=float, default=2.0, help='器具・指の摩擦係数（これまで 2.0。実機のゴムパッド×金属は 1.0 前後）')
ap.add_argument('--item-collider', default='sdf', choices=['box', 'convexHull', 'sdf']); ap.add_argument('--sdf-res', type=int, default=400)
ap.add_argument('--commands', default=None, help='serve: 命令の JSONL（1 行 1 命令）')
ap.add_argument('--n', type=int, default=300, help='dataset: 画像の枚数'); ap.add_argument('--val-every', type=int, default=5)
ap.add_argument('--handoff', default='present', choices=['place', 'present'], help='place = 手渡し台に置くだけ（A）/ present = 持ち手を医師に向けて差し出してから置く（B）')
a = ap.parse_args()
SERVE = a.stage in ('serve', 'dataset')   # 器械出しの場面（serve = 命令を実行、dataset = 検出モデルの学習画像を作る）
if a.real: a.depth_noise, a.depth_quant, a.depth_dropout, a.calib_err, a.friction = 1.5, 1.4, 0.3, '3,0.5', 1.0
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, 'pickup_assets')   # ロボット・器具・検出モデル（読むだけ）
OUT = os.path.join(HERE, a.out); os.makedirs(OUT, exist_ok=True)
t_start = time.time()
from isaacsim import SimulationApp
app = SimulationApp({"headless": True, "width": 1280, "height": 720})
import numpy as np, imageio.v2 as iio
from pxr import Usd, UsdGeom, UsdPhysics, Gf, Sdf, UsdLux, UsdShade
import omni.usd
from omni.physx.scripts import utils as physx_utils
from pxr import PhysxSchema
from isaacsim.core.api import World
from isaacsim.core.api.objects import FixedCuboid, VisualCuboid
from isaacsim.core.prims import SingleArticulation, SingleXFormPrim, SingleRigidPrim
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.semantics import add_update_semantics
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.sensors.camera import Camera
import isaacsim.core.utils.numpy.rotations as rot_utils
from isaacsim.robot_motion.motion_generation import LulaKinematicsSolver, ArticulationKinematicsSolver
from scipy.spatial.transform import Rotation as R

rng = np.random.default_rng(a.seed)
world = World(stage_units_in_meters=1.0, physics_dt=a.physics_dt, rendering_dt=1/30)
stage = omni.usd.get_context().get_stage()
GROUND = -1.05 if SERVE else -0.75   # 器械出しの場面: 机の天板（z=0）が床から 105 cm（医師の腰よりやや上）。ロボット・机・器具の位置は変えない
world.scene.add_default_ground_plane(z_position=GROUND)
# ---- 照明 ----
dome = UsdLux.DomeLight.Define(stage, Sdf.Path('/World/DomeLight')); dome.CreateIntensityAttr(300.0)
sun = UsdLux.DistantLight.Define(stage, Sdf.Path('/World/Sun')); sun.CreateIntensityAttr(1500.0); sun.CreateAngleAttr(1.0)
UsdGeom.Xformable(sun.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(-50, 20, 0))
# ---- 机（天板 z=0 が作業面）と箱 4 つ ----
TABLE_RGB = [float(v) for v in os.environ.get('TABLE_RGB', '').split(',') if v]   # 机の色（10/31 夜: 青い覆布の色で銀の器具を目立たせる版）
world.scene.add(FixedCuboid(prim_path='/World/table', name='table', position=np.array([0.45, 0.0, -0.025]), scale=np.array([1.1, 2 * a.spread + 0.9, 0.05]), color=np.array(TABLE_RGB if len(TABLE_RGB) == 3 else [0.12, 0.30, 0.32])))   # 手術台のドレープ風の濃い色（銀色の器具が見えるように）
TABLE_MAT = UsdShade.Material.Define(stage, Sdf.Path('/World/PhysMat/table_slick')); _tm = UsdPhysics.MaterialAPI.Apply(TABLE_MAT.GetPrim()); _tm.CreateStaticFrictionAttr(0.3); _tm.CreateDynamicFrictionAttr(0.25); _tm.CreateRestitutionAttr(0.0)
UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath('/World/table')).Bind(TABLE_MAT, UsdShade.Tokens.weakerThanDescendants, 'physics')
TYPES = ['scissors', 'forceps', 'scalpel', 'syringe']
TYPE_JP = {'scissors': 'ハサミ', 'forceps': 'ピンセット', 'scalpel': 'メス', 'syringe': '注射器'}
BIN_COLORS = {'scissors': [0.9, 0.2, 0.2], 'forceps': [0.2, 0.7, 0.2], 'scalpel': [0.2, 0.3, 0.9], 'syringe': [0.9, 0.8, 0.1]}
BIN_Y = a.spread + 0.19   # 箱は散らばり領域の外側
BIN_POS = {'scissors': np.array([0.30, -BIN_Y, 0.0]), 'forceps': np.array([0.50, -BIN_Y, 0.0]), 'scalpel': np.array([0.30, BIN_Y, 0.0]), 'syringe': np.array([0.50, BIN_Y, 0.0])}
BIN_W, BIN_L, BIN_H, WALL = 0.16, 0.24, 0.05, 0.006
def make_bin(name, center, color):
    cx, cy, _ = center; col = np.array(color)
    world.scene.add(FixedCuboid(prim_path=f'/World/bins/{name}/floor', name=f'bin_{name}_floor', position=np.array([cx, cy, 0.003]), scale=np.array([BIN_W, BIN_L, 0.006]), color=col))
    for i, (dx, dy, sx, sy) in enumerate([(BIN_W/2, 0, WALL, BIN_L), (-BIN_W/2, 0, WALL, BIN_L), (0, BIN_L/2, BIN_W, WALL), (0, -BIN_L/2, BIN_W, WALL)]):
        world.scene.add(FixedCuboid(prim_path=f'/World/bins/{name}/wall{i}', name=f'bin_{name}_wall{i}', position=np.array([cx+dx, cy+dy, BIN_H/2]), scale=np.array([sx, sy, BIN_H]), color=col*0.8))
if SERVE:   # 器械出し: 医師側（-y）に手渡し台、反対側（+y）に返却トレー。器具は机（器械台）に散らばっている
    # 手渡し台と返却トレーは「壁のない平らな台」（10/31: 箱だと器具が重なり、壁ぎわで検出できなかった）。検出モデルは机の上の器具で一番よく学習しているので、机に近い落ち着いた色にする
    MAT_W, MAT_L, MAT_T = 0.34, 0.22, 0.003; MAT_Y = a.spread + 0.17
    HANDOFF_POS = np.array([0.45, -MAT_Y, MAT_T]); TRAY_POS = np.array([0.45, MAT_Y, MAT_T])
    MAT_COLS = ([0.06, 0.07, 0.08], [0.05, 0.07, 0.12]) if len(TABLE_RGB) == 3 else ([0.22, 0.36, 0.30], [0.20, 0.28, 0.40])   # 青い机なら台は明るい灰色系
    for nm, c, col in (('handoff', HANDOFF_POS, MAT_COLS[0]), ('tray', TRAY_POS, MAT_COLS[1])):
        world.scene.add(FixedCuboid(prim_path=f'/World/mats/{nm}', name=f'mat_{nm}', position=np.array([c[0], c[1], MAT_T / 2]), scale=np.array([MAT_W, MAT_L, MAT_T]), color=np.array(col)))
    # 机の下の台（見た目だけ。机が浮いて見えないように）
    _ped = UsdGeom.Cube.Define(stage, Sdf.Path('/World/table_base')); _ped.CreateSizeAttr(1.0); _px = UsdGeom.Xformable(_ped.GetPrim())
    _px.AddTranslateOp().Set(Gf.Vec3d(0.45, 0.0, (GROUND - 0.05) / 2)); _px.AddScaleOp().Set(Gf.Vec3f(0.95, 2 * a.spread + 0.75, -GROUND - 0.05)); _ped.CreateDisplayColorAttr([(0.55, 0.60, 0.62)])
    def add_camera_stand():
        def cube(name, c, size, col):
            g = UsdGeom.Cube.Define(stage, Sdf.Path(f'/World/camera_stand/{name}')); g.CreateSizeAttr(1.0); xf = UsdGeom.Xformable(g.GetPrim())
            xf.AddTranslateOp().Set(Gf.Vec3d(*map(float, c))); xf.AddScaleOp().Set(Gf.Vec3f(*map(float, size))); g.CreateDisplayColorAttr([col])
        cz = a.cam_z if a.cam_z else 0.95; py = a.spread + 0.43               # 支柱は返却トレー側の作業台の端（カメラの視野の外）、アームを横から伸ばしてカメラを真上に吊るす
        cube('pole', (0.46, py, (cz + 0.10) / 2), (0.03, 0.03, cz + 0.10), (0.25, 0.25, 0.27))
        cube('boom', (0.46, py / 2, cz + 0.085), (0.025, py + 0.03, 0.025), (0.25, 0.25, 0.27))
        cube('cam_body', (0.46, 0.0, cz + 0.03), (0.025, 0.09, 0.025), (0.12, 0.12, 0.13))   # RealSense D435 くらいの大きさ（横長）
        cube('cam_mount', (0.46, 0.0, cz + 0.06), (0.015, 0.015, 0.04), (0.25, 0.25, 0.27))
        cube('foot', (0.46, py, 0.004), (0.14, 0.14, 0.008), (0.25, 0.25, 0.27))
    if a.camera == 'fixed': add_camera_stand()
    PRESENT_P = np.array([0.44, -0.45, 0.26])   # 差し出す位置（医師の右手のすぐ前）
    def add_doctor(x=0.45, y=-(a.spread + 0.62)):
        """見た目だけの医師（当たり判定なし）。手渡し台の向こう側に立ち、手渡し台の方へ右手を差し出す。
        ロボットの俯瞰カメラの視野（机の高さで |y| < 約 0.49）の外に置くので、知覚には写らない。"""
        def mat(name, col, rough=0.6):
            m = UsdShade.Material.Define(stage, Sdf.Path(f'/World/Looks/doc_{name}')); sh = UsdShade.Shader.Define(stage, Sdf.Path(f'/World/Looks/doc_{name}/Shader'))
            sh.CreateIdAttr('UsdPreviewSurface'); sh.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*col)); sh.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(rough)
            m.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), 'surface'); return m
        SCRUB, SKIN, CAP, MASK, GLOVE = mat('scrub', (0.22, 0.45, 0.62)), mat('skin', (0.86, 0.70, 0.58)), mat('cap', (0.30, 0.52, 0.72)), mat('mask', (0.80, 0.88, 0.92)), mat('glove', (0.55, 0.70, 0.85), 0.4)
        def part(name, kind, pos, m, r=None, h=None, scale=None, quat=None):
            path = Sdf.Path(f'/World/doctor/{name}')
            g = {'cyl': UsdGeom.Cylinder, 'cap': UsdGeom.Capsule, 'sph': UsdGeom.Sphere, 'box': UsdGeom.Cube}[kind].Define(stage, path)
            if r is not None: g.CreateRadiusAttr(r)
            if h is not None: g.CreateHeightAttr(h)
            if kind == 'box': g.CreateSizeAttr(1.0)
            xf = UsdGeom.Xformable(g.GetPrim()); xf.AddTranslateOp().Set(Gf.Vec3d(*map(float, pos)))
            if quat is not None: xf.AddOrientOp().Set(Gf.Quatf(*map(float, quat)))
            if scale is not None: xf.AddScaleOp().Set(Gf.Vec3f(*map(float, scale)))
            UsdShade.MaterialBindingAPI.Apply(g.GetPrim()).Bind(m)
        def limb(name, p0, p1, r, m):   # p0→p1 を結ぶカプセル（Capsule の軸は z）
            p0, p1 = np.array(p0, float), np.array(p1, float); v = p1 - p0; L = float(np.linalg.norm(v))
            q = R.align_vectors([v / L], [[0, 0, 1]])[0].as_quat(); part(name, 'cap', (p0 + p1) / 2, m, r=r, h=max(0.01, L - 2 * r), quat=(q[3], q[0], q[1], q[2]))
        floor = GROUND
        for sx, nm in ((-0.08, 'leg_l'), (0.08, 'leg_r')): part(nm, 'cyl', (x + sx, y, floor + 0.42), SCRUB, r=0.075, h=0.84)
        part('torso', 'cap', (x, y, floor + 1.17), SCRUB, r=0.17, h=0.42); part('neck', 'cyl', (x, y, floor + 1.52), SKIN, r=0.05, h=0.08)
        part('head', 'sph', (x, y, floor + 1.64), SKIN, r=0.105); part('cap_hat', 'sph', (x, y, floor + 1.69), CAP, r=0.11, scale=(1, 1, 0.75))
        part('mask', 'box', (x, y + 0.085, floor + 1.61), MASK, scale=(0.15, 0.04, 0.09))
        sh_r, sh_l = (x + 0.21, y + 0.02, floor + 1.40), (x - 0.21, y + 0.02, floor + 1.40)
        hand = (x + 0.03, y + 0.29, 0.20)                                   # 差し出す位置の少し下・手前に右手を出す（俯瞰カメラに写らない位置）
        elbow = (x + 0.18, y + 0.15, (floor + 1.40 + 0.20) / 2 - 0.03)   # 肩と手の間（少し下）
        limb('arm_r_up', sh_r, elbow, 0.05, SCRUB); limb('arm_r_low', elbow, hand, 0.042, GLOVE); part('hand_r', 'sph', hand, GLOVE, r=0.05, scale=(1.0, 1.3, 0.55))
        limb('arm_l_up', sh_l, (x - 0.25, y + 0.05, floor + 1.10), 0.05, SCRUB); limb('arm_l_low', (x - 0.25, y + 0.05, floor + 1.10), (x - 0.22, y + 0.15, floor + 0.88), 0.042, GLOVE)
    add_doctor()
    def add_patient(y=-(a.spread + 1.12)):
        def mat(name, col):
            m = UsdShade.Material.Define(stage, Sdf.Path(f'/World/Looks/pt_{name}')); sh = UsdShade.Shader.Define(stage, Sdf.Path(f'/World/Looks/pt_{name}/Shader'))
            sh.CreateIdAttr('UsdPreviewSurface'); sh.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*col)); sh.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(0.7)
            m.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), 'surface'); return m
        BED, DRAPE, SKIN, CAP = mat('bed', (0.30, 0.31, 0.34)), mat('drape', (0.10, 0.35, 0.45)), mat('skin', (0.86, 0.70, 0.58)), mat('cap', (0.30, 0.52, 0.72))
        def box(name, c, size, m):
            g = UsdGeom.Cube.Define(stage, Sdf.Path(f'/World/patient/{name}')); g.CreateSizeAttr(1.0); xf = UsdGeom.Xformable(g.GetPrim())
            xf.AddTranslateOp().Set(Gf.Vec3d(*map(float, c))); xf.AddScaleOp().Set(Gf.Vec3f(*map(float, size))); UsdShade.MaterialBindingAPI.Apply(g.GetPrim()).Bind(m)
        def sph(name, c, r, m, scale=(1, 1, 1)):
            g = UsdGeom.Sphere.Define(stage, Sdf.Path(f'/World/patient/{name}')); g.CreateRadiusAttr(r); xf = UsdGeom.Xformable(g.GetPrim())
            xf.AddTranslateOp().Set(Gf.Vec3d(*map(float, c))); xf.AddScaleOp().Set(Gf.Vec3f(*map(float, scale))); UsdShade.MaterialBindingAPI.Apply(g.GetPrim()).Bind(m)
        top = GROUND + 0.85                                                  # 手術台の上面（床から 85 cm）
        box('bed', (-0.25, y, (GROUND + top) / 2), (1.70, 0.58, top - GROUND), BED)
        box('drape', (-0.15, y, top + 0.07), (1.30, 0.62, 0.14), DRAPE)      # 覆布（体にかけた布）
        sph('head', (-0.92, y, top + 0.11), 0.10, SKIN); sph('cap', (-0.95, y, top + 0.13), 0.105, CAP, (1.0, 1.0, 0.8))
    add_patient()
else:
    for t in TYPES: make_bin(t, BIN_POS[t], BIN_COLORS[t])
# ---- ロボット ----
ROBOT_USD = os.path.join(ROOT, f'sim/xarm{a.dof}_pargrip.usd')   # プリムの名前 /World/xarm7 は 6 軸でもそのまま（名前だけ）   # 平行開閉グリッパ版（純正の指リンク機構はシミュレータで閉じきれず・めり込むため置き換え）
stage.DefinePrim('/World/xarm7', 'Xform').GetReferences().AddReference(ROBOT_USD, '/UF_ROBOT')   # USD に defaultPrim が無いので参照先プリムを明示
print('[sim] robot children:', [c.GetName() for c in stage.GetPrimAtPath('/World/xarm7').GetChildren()][:12], flush=True)
# 当たり判定の修正: URDF 取り込みで CollisionAPI が collisions/<link>/node_STL_BINARY_ の Xform に付き、中の Mesh に付いていない
# （指が机や器具をすり抜けた）。Mesh 側に付け直す。
def fix_robot_colliders(root_path='/World/xarm7'):
    n = 0
    for link in stage.GetPrimAtPath(root_path).GetChildren():
        col = stage.GetPrimAtPath(link.GetPath().pathString + '/collisions')
        if not col.IsValid(): continue
        tops = set()
        for d in list(Usd.PrimRange(col, Usd.TraverseInstanceProxies())):   # 先に集めてから instanceable を外す（走査中に変えると壊れる）
            if d.IsInstanceProxy():
                top = d
                while top.IsInstanceProxy(): top = top.GetParent()
                tops.add(top.GetPath())
        for tp in tops: stage.GetPrimAtPath(tp).SetInstanceable(False)
        for d in Usd.PrimRange(col):
            if d.GetTypeName() == 'Mesh':
                is_hand = any(k in link.GetName() for k in ('finger', 'knuckle', 'gripper'))
                UsdPhysics.CollisionAPI.Apply(d); UsdPhysics.MeshCollisionAPI.Apply(d).CreateApproximationAttr('convexDecomposition' if is_hand else 'convexHull'); n += 1
                if is_hand:
                    pc = PhysxSchema.PhysxCollisionAPI.Apply(d); pc.CreateContactOffsetAttr(0.003); pc.CreateRestOffsetAttr(0.0)
                    cd = PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(d); cd.CreateMaxConvexHullsAttr(16); cd.CreateHullVertexLimitAttr(32); cd.CreateShrinkWrapAttr(True)
            elif d.HasAPI(UsdPhysics.CollisionAPI) and d.GetTypeName() in ('Xform', 'Scope', ''):
                d.RemoveAPI(UsdPhysics.CollisionAPI)
    # 質量 0 のリンク（link_eef、カメラのフレーム類）は関節系の力学を壊す（接触が効かない・たわむ）ので小さい質量を与える
    nz = 0
    for link in stage.GetPrimAtPath(root_path).GetChildren():
        if link.HasAPI(UsdPhysics.RigidBodyAPI):
            ma = UsdPhysics.MassAPI.Apply(link); m = ma.GetMassAttr().Get()
            if m is None or m <= 0.0: ma.CreateMassAttr(0.02); nz += 1
    if a.no_mimic:
        for jn in ('left_finger_joint', 'left_inner_knuckle_joint', 'right_outer_knuckle_joint', 'right_finger_joint', 'right_inner_knuckle_joint'):
            jp = stage.GetPrimAtPath(root_path + '/joints/' + jn)
            for sch in list(jp.GetAppliedSchemas()):
                if 'Mimic' in sch: jp.RemoveAPI(PhysxSchema.PhysxMimicJointAPI, sch.split(':')[-1]) if ':' in sch else None
            dr_ = UsdPhysics.DriveAPI.Apply(jp, 'angular'); dr_.CreateMaxForceAttr(a.grip_maxforce); dr_.CreateStiffnessAttr(a.grip_stiff); dr_.CreateDampingAttr(a.grip_stiff / 10)
        print('[sim] mimic joints removed; all finger joints driven directly', flush=True)
    print(f'[sim] robot colliders fixed on {n} meshes; zero-mass links fixed: {nz}', flush=True)
    art = PhysxSchema.PhysxArticulationAPI.Apply(stage.GetPrimAtPath(root_path + '/root_joint')); art.CreateSolverPositionIterationCountAttr(32); art.CreateSolverVelocityIterationCountAttr(4)
fix_robot_colliders()
def add_pad_colliders():
    """左右の指に、つかみ面側・指先側の薄い箱コライダを足す（メッシュの凸分解は接触が柔らかく 1cm 以上めり込んだ）。
    内側／指先側の判定はグリッパ基部（xarm_gripper_base_link）の座標系で行う: 内側 = 基部の中心軸に近い面、指先 = 基部 +z（TCP 方向）に遠い端。"""
    Mb = np.array(UsdGeom.Xformable(stage.GetPrimAtPath('/World/xarm7/xarm_gripper_base_link')).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
    Mb_inv = np.linalg.inv(Mb)
    for f in ('left_finger', 'right_finger'):
        link = stage.GetPrimAtPath(f'/World/xarm7/{f}')
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy, UsdGeom.Tokens.guide])
        lb = cache.ComputeLocalBound(stage.GetPrimAtPath(f'/World/xarm7/{f}/collisions')).ComputeAlignedRange(); mn, mx = np.array(lb.GetMin()), np.array(lb.GetMax())
        for d in Usd.PrimRange(stage.GetPrimAtPath(f'/World/xarm7/{f}/collisions')):
            if d.HasAPI(UsdPhysics.CollisionAPI): d.RemoveAPI(UsdPhysics.CollisionAPI)
        M = np.array(UsdGeom.Xformable(link).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
        def to_base(pl): return (np.array([pl[0], pl[1], pl[2], 1.0]) @ M @ Mb_inv)[:3]   # 指ローカル → グリッパ基部ローカル
        ext = mx - mn; thick = 0.006; cen = (mn + mx) / 2
        cands = []
        for axis in (0, 1):
            for side, val in ((-1, mn[axis] + thick / 2), (+1, mx[axis] - thick / 2)):
                pl = cen.copy(); pl[axis] = val; b = to_base(pl); cands.append((float(np.hypot(b[0], b[1])), axis, side, val))
        _, in_axis, in_side, in_val = min(cands)
        z_lo_b = to_base(np.array([cen[0], cen[1], mn[2]]))[2]; z_hi_b = to_base(np.array([cen[0], cen[1], mx[2]]))[2]
        tip_at_max = z_hi_b > z_lo_b
        pad_len = ext[2] * 0.6; zc = (mx[2] - pad_len / 2) if tip_at_max else (mn[2] + pad_len / 2)
        center = cen.copy(); center[in_axis] = in_val; center[2] = zc
        size = ext.copy() * 0.9; size[in_axis] = thick; size[2] = pad_len
        pad = UsdGeom.Cube.Define(stage, Sdf.Path(f'/World/xarm7/{f}/pad_collider')); pad.CreateSizeAttr(1.0)
        xf = UsdGeom.Xformable(pad.GetPrim()); xf.AddTranslateOp().Set(Gf.Vec3d(*map(float, center))); xf.AddScaleOp().Set(Gf.Vec3f(*map(float, size)))
        UsdGeom.Imageable(pad.GetPrim()).CreatePurposeAttr(UsdGeom.Tokens.guide)
        UsdPhysics.CollisionAPI.Apply(pad.GetPrim()); pc = PhysxSchema.PhysxCollisionAPI.Apply(pad.GetPrim()); pc.CreateContactOffsetAttr(0.005); pc.CreateRestOffsetAttr(0.0)
        bc = to_base(center); print(f'[sim] pad collider on {f}: inner={"xy"[in_axis]}{"+" if in_side>0 else "-"} tip_at_max_z={tip_at_max} center_in_gripper_base={np.round(bc,3).tolist()} (基部中心軸からの距離 {np.hypot(bc[0],bc[1])*1000:.0f}mm, 基部 z {bc[2]*1000:.0f}mm; TCP は z=172mm) size={np.round(size,3).tolist()}', flush=True)
if stage.GetPrimAtPath('/World/xarm7/left_finger').IsValid(): add_pad_colliders()
# 駆動の強さ: URDF の effort 上限（20〜50 Nm）だと肘（joint4）が自重に負ける（実測 64 Nm 必要）ので、練習用に上限と剛性を上げる
for i in range(1, a.dof + 1):
    dr = UsdPhysics.DriveAPI(stage.GetPrimAtPath(f'/World/xarm7/joints/joint{i}'), 'angular'); dr.GetMaxForceAttr().Set(a.arm_maxforce); dr.GetStiffnessAttr().Set(60000.0); dr.GetDampingAttr().Set(5000.0)
for _pj in ('left_pad_joint', 'right_pad_joint'):
    _g = UsdPhysics.DriveAPI.Apply(stage.GetPrimAtPath(f'/World/xarm7/joints/{_pj}'), 'linear'); _g.CreateMaxForceAttr(a.grip_maxforce); _g.CreateStiffnessAttr(a.grip_stiff); _g.CreateDampingAttr(a.grip_stiff / 10)
robot = world.scene.add(SingleArticulation(prim_path='/World/xarm7/root_joint', name='xarm7'))   # ArticulationRootAPI は root_joint に付いている
# ---- 器具（Isaac for Healthcare、cm 単位 → 0.01）----
I4H = os.path.join(ROOT, 'sim/assets/i4h'); OR = 'shared_OR_without_Mark/Collected_surgery_room_movie_with_heart_adjusted/Assets/Surgery Room/'
ASSETS = {'scissors': I4H + '/' + OR + 'SurgicalInstruments_A/Model/sm_scissors_a01_01.usd',   # SurgicalScissors.usd は内部に剛体 API を持っていて二重になるので不採用
          'forceps': I4H + '/' + OR + 'SurgicalInstruments_A/Model/sm_forceps_a01_01.usd',
          'scalpel': I4H + '/' + OR + 'SurgicalInstruments_A/Model/sm_scalpelshandle_a01_01.usd',
          'syringe': I4H + '/' + OR + 'InsulinSyringe_A/Model/sm_insulinsyringe_a01_01.usd'}
items = []
def make_steel_material(path='/World/Looks/steel', color=(0.62, 0.63, 0.66)):
    mat = UsdShade.Material.Define(stage, Sdf.Path(path)); sh = UsdShade.Shader.Define(stage, Sdf.Path(path + '/Shader'))
    sh.CreateIdAttr('UsdPreviewSurface'); sh.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    sh.CreateInput('metallic', Sdf.ValueTypeNames.Float).Set(0.8); sh.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(0.35); sh.CreateInput('opacity', Sdf.ValueTypeNames.Float).Set(1.0)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), 'surface'); return mat
ap2 = [v for v in os.environ.get('STEEL_RGB', '').split(',') if v]
STEEL = make_steel_material(color=tuple(float(v) for v in ap2) if len(ap2) == 3 else ((0.75, 0.55, 0.20) if SERVE and len(TABLE_RGB) != 3 else (0.62, 0.63, 0.66)))   # serve は金色（見やすさ）   # 環境変数 STEEL_RGB で器具の色を変えられる（見やすさの確認用）
# 高摩擦の物理マテリアル（器具と指に適用）
PHYS_MAT = UsdShade.Material.Define(stage, Sdf.Path('/World/PhysMat/grippy')); _pm = UsdPhysics.MaterialAPI.Apply(PHYS_MAT.GetPrim()); _pm.CreateStaticFrictionAttr(a.friction); _pm.CreateDynamicFrictionAttr(a.friction * 0.9); _pm.CreateRestitutionAttr(0.0)
_pmx = PhysxSchema.PhysxMaterialAPI.Apply(PHYS_MAT.GetPrim()); _pmx.CreateFrictionCombineModeAttr('max')
def bind_phys_material(prim):
    for d in Usd.PrimRange(prim):
        if d.GetTypeName() == 'Mesh' or d.HasAPI(UsdPhysics.CollisionAPI):
            UsdShade.MaterialBindingAPI.Apply(d).Bind(PHYS_MAT, UsdShade.Tokens.weakerThanDescendants, 'physics')
            pc = PhysxSchema.PhysxCollisionAPI.Apply(d); pc.CreateContactOffsetAttr(0.003); pc.CreateRestOffsetAttr(0.0)
def bind_material(prim, mat):
    for d in Usd.PrimRange(prim):
        if d.GetTypeName() in ('Mesh', 'Xform'):
            UsdShade.MaterialBindingAPI.Apply(d).Bind(mat, UsdShade.Tokens.strongerThanDescendants)
for fp in ('left_pad', 'right_pad', 'left_finger', 'right_finger'):
    fprim = stage.GetPrimAtPath(f'/World/xarm7/{fp}')
    if fprim.IsValid(): bind_phys_material(fprim)
for fp in ('left_finger', 'right_finger'):
    padp = stage.GetPrimAtPath(f'/World/xarm7/{fp}/pad_collider')
    if padp.IsValid(): UsdShade.MaterialBindingAPI.Apply(padp).Bind(PHYS_MAT, UsdShade.Tokens.weakerThanDescendants, 'physics')
def spawn_items():
    k = 0
    for t in TYPES:
        for j in range(a.per_type):
            if k >= a.max_items: return
            path = f'/World/items/{t}_{j}'
            # 根元 = 剛体（スケール無し）、その下の model = cm→m の 0.01 スケール付き参照。
            # 根元にスケールを掛けると reset 後に二重に掛かって 1/10000 になった（Isaac の既定状態の再適用）ので分離する。
            prim = stage.DefinePrim(path, 'Xform')
            model = stage.DefinePrim(path + '/model', 'Xform')                      # 縮尺だけを持つ中間 Xform（参照先の既存 xformOp と衝突しないように分ける）
            UsdGeom.Xformable(model).AddScaleOp().Set(Gf.Vec3f(0.01, 0.01, 0.01))
            asset = stage.DefinePrim(path + '/model/asset', 'Xform'); asset.GetReferences().AddReference(ASSETS[t])
            for d in Usd.PrimRange(prim):   # アセット内に既にある物理 API を外す（根元で一括設定するため）
                for api in (UsdPhysics.RigidBodyAPI, UsdPhysics.CollisionAPI, UsdPhysics.MassAPI):
                    if d.HasAPI(api): d.RemoveAPI(api)
            placed_xy = [it['xy'] for it in items]
            for _try in range(200):   # 他の器具と最小間隔を空けて置く
                x = rng.uniform(0.28, 0.68); y = rng.uniform(-a.spread, a.spread)
                if all(np.hypot(x - px, y - py) >= a.min_gap for px, py in placed_xy): break
            yaw = rng.uniform(-math.pi, math.pi)
            xf = SingleXFormPrim(path, position=np.array([x, y, 0.03 + 0.01*k]), orientation=rot_utils.euler_angles_to_quats(np.array([0, 0, yaw])))
            if a.item_collider == 'box':
                # 器具 1 個 = bbox に合わせた箱コライダ（薄い金属器具のメッシュ由来の当たりは接触が不安定だった）。見た目はメッシュのまま
                UsdPhysics.RigidBodyAPI.Apply(prim)
                cache0 = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render]); lb = cache0.ComputeLocalBound(stage.GetPrimAtPath(path + '/model')).ComputeAlignedRange()
                lmn, lmx = np.array(lb.GetMin()), np.array(lb.GetMax()); size = np.maximum(lmx - lmn, 0.004); cen = (lmn + lmx) / 2
                box = UsdGeom.Cube.Define(stage, Sdf.Path(path + '/collider')); box.CreateSizeAttr(1.0)
                xo = UsdGeom.Xformable(box.GetPrim()); xo.AddTranslateOp().Set(Gf.Vec3d(*map(float, cen))); xo.AddScaleOp().Set(Gf.Vec3f(*map(float, size)))
                UsdGeom.Imageable(box.GetPrim()).CreatePurposeAttr(UsdGeom.Tokens.guide); UsdPhysics.CollisionAPI.Apply(box.GetPrim())
            else:
                physx_utils.setRigidBody(prim, a.item_collider, False)
                if a.item_collider == 'sdf':
                    for d in Usd.PrimRange(prim):
                        if d.GetTypeName() == 'Mesh' and d.HasAPI(UsdPhysics.CollisionAPI):
                            sdf = PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(d); sdf.CreateSdfResolutionAttr(a.sdf_res)
            mass = UsdPhysics.MassAPI.Apply(prim); mass.CreateMassAttr(a.item_mass)
            _pb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim); _pb.CreateSolverPositionIterationCountAttr(a.solver_iters); _pb.CreateSolverVelocityIterationCountAttr(4); _pb.CreateEnableCCDAttr(True)
            add_update_semantics(prim, semantic_label=t, type_label='class'); bind_material(prim, STEEL); bind_phys_material(prim)
            items.append({'name': f'{t}_{j}', 'type': t, 'path': path, 'rigid': SingleRigidPrim(path, name=f'item_{t}_{j}'), 'xy': (x, y)}); k += 1
spawn_items()
def item_bbox(path):
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy])
    r = cache.ComputeWorldBound(stage.GetPrimAtPath(path)).ComputeAlignedRange(); mn, mx = np.array(r.GetMin()), np.array(r.GetMax()); return mn, mx
for it in items:
    mn, mx = item_bbox(it['path']); meshes = [d for d in Usd.PrimRange(stage.GetPrimAtPath(it['path'])) if d.GetTypeName() == 'Mesh']
    vis = [UsdGeom.Imageable(m).ComputeVisibility() for m in meshes][:3]; purp = [UsdGeom.Imageable(m).ComputePurpose() for m in meshes][:3]
    print(f"[sim] item {it['name']}: size_m={np.round(mx-mn,3).tolist()} zrange=({mn[2]:.3f},{mx[2]:.3f}) meshes={len(meshes)} vis={vis} purpose={purp}", flush=True)
# ---- カメラ: RealSense D435 想定（固定、作業面を見下ろす）。縦画角 42° ----
CAM_POS = np.array([0.46, 0.0, a.cam_z if a.cam_z else 0.95 + max(0.0, (a.spread - 0.25) * 1.6)]); CAM_W, CAM_H = [int(v) for v in a.cam_res.split('x')]
if a.camera == 'wrist':   # 手首カメラ: URDF の camera_link（x 前方 = 手先方向、z 上）の子にする。Camera の向きの規約（+X 前方・+Z 上）と一致するので向きはそのまま
    cam = Camera(prim_path='/World/xarm7/camera_link/realsense', translation=np.zeros(3), orientation=np.array([1.0, 0.0, 0.0, 0.0]), frequency=30, resolution=(CAM_W, CAM_H))
else:
    cam = Camera(prim_path='/World/realsense', position=CAM_POS, frequency=30, resolution=(CAM_W, CAM_H), orientation=rot_utils.euler_angles_to_quats(np.array([0, 90, 0]), degrees=True))
cam.set_focal_length(1.93); cam.set_horizontal_aperture(3.896); cam.set_vertical_aperture(3.896 * CAM_H / CAM_W); cam.set_clipping_range(0.05, 5.0)   # D435 RGB: 1.93 mm, 1/2.8" センサ相当 → HFOV≈69°, VFOV≈42°
spec = Camera(prim_path='/World/spectator', position=np.array([1.35, -1.05, 0.95]) * (1.0 + max(0.0, a.spread - 0.25)), frequency=30, resolution=(1280, 720))
spec.set_focal_length(2.2); spec.set_horizontal_aperture(3.896); spec.set_clipping_range(0.05, 10.0)
def look_at(camera, target):
    pos, _ = camera.get_world_pose(); f = target - pos; f = f / np.linalg.norm(f)
    up = np.array([0, 0, 1.0]); r = np.cross(f, up); r /= np.linalg.norm(r); u = np.cross(r, f)
    # Isaac の Camera は +X 前方・+Z 上（USD カメラの -Z 前方からの変換は内部で処理）
    Rm = np.stack([f, -r, u], axis=1)
    q = R.from_matrix(Rm).as_quat(); camera.set_world_pose(position=pos, orientation=np.array([q[3], q[0], q[1], q[2]]))
world.reset()
cam.initialize(); cam.add_distance_to_image_plane_to_frame(); cam.add_semantic_segmentation_to_frame(); cam.add_instance_segmentation_to_frame()
spec.initialize()
close = Camera(prim_path='/World/closeup', position=np.array([0.9, -0.5, 0.35]), frequency=30, resolution=(960, 540)); close.set_focal_length(4.0); close.set_horizontal_aperture(3.896); close.set_clipping_range(0.05, 10.0); close.initialize()
if a.camera == 'fixed': look_at(cam, np.array([0.46, 0.0, 0.0]))
if SERVE: spec.set_world_pose(position=np.array([1.50, 0.62, 1.10])); look_at(spec, np.array([0.30, -0.08, 0.05]))
else: look_at(spec, np.array([0.45, 0.0, 0.15]))
look_at(close, np.array([0.47, 0.0, 0.03]))
# ---- 腕: Lula IK（TCP = link_tcp）----
lula = LulaKinematicsSolver(robot_description_path=os.path.join(ROOT, f'scripts/xarm{a.dof}_pargrip_robot_description.yaml'), urdf_path=os.path.join(ROOT, f'sim/xarm_urdf/xarm{a.dof}_pargrip_d435i.urdf'))
base_p, base_q = robot.get_world_pose(); lula.set_robot_base_pose(base_p, base_q)
aks = ArticulationKinematicsSolver(robot, lula, 'link_tcp')
dof_names = list(robot.dof_names); ARM = [dof_names.index(f'joint{i}') for i in range(1, a.dof + 1)]; GRIPS = [dof_names.index('left_pad_joint'), dof_names.index('right_pad_joint')]; GRIP = GRIPS[0]
GRIP_CLOSED = 0.0425   # m（片側ストローク）
print('[sim] dofs:', dof_names, flush=True)
def quat_down(yaw):
    q = R.from_euler('xyz', [math.pi, 0.0, yaw]).as_quat(); return np.array([q[3], q[0], q[1], q[2]])
# 初期姿勢: TCP を (0.35, 0, 0.30) で真下向きにする関節角を IK で求める（机に当たらない「構え」）
_seed = np.array([0.0, -0.4, 0.0, 0.7, 0.0, 1.1, 0.0]) if a.dof == 7 else np.array([0.0, -0.3, -0.6, 0.0, 0.9, 0.0])   # xArm6: J2 肩・J3 肘・J5 手首ピッチで手先を下向きに
_q, _ok = lula.compute_inverse_kinematics('link_tcp', np.array([0.35, 0.0, 0.30]), quat_down(0.0), warm_start=_seed)
HOME = np.asarray(_q) if _ok else _seed
print('[sim] HOME from IK ok=', _ok, np.round(HOME, 3).tolist(), flush=True)
# 知覚のときの「よける」姿勢: 机の横（y=-0.42）の上空に手先を上げて、固定カメラの視野から腕を外す（実機でも写真を撮る前に腕をどける）
PARK_P = np.array([0.08, -0.42, 0.45]); _okp = False
for PARK_P in (np.array([0.08, -0.42, 0.45]), np.array([0.12, -0.38, 0.40]), np.array([0.15, -0.35, 0.35]), np.array([0.20, -0.40, 0.30])):   # 届く候補を順に
    _qp, _okp = lula.compute_inverse_kinematics('link_tcp', PARK_P, quat_down(0.0), warm_start=HOME)
    if _okp: break
print('[sim] PARK from IK ok=', _okp, None if not _okp else np.round(np.asarray(_qp), 3).tolist(), flush=True)
def go_park():
    if a.no_park or not _okp: return
    ok = ik_to(PARK_P, quat_down(0.0), 80, speed=0.3); step(20)
    if not ok: print('[sim] go_park: 届かなかった', flush=True)
SCAN_P = np.array([float(v) for v in a.scan_pose.split(',')])
_qs, _oks = lula.compute_inverse_kinematics('link_tcp', SCAN_P, quat_down(0.0), warm_start=HOME); print('[sim] SCAN from IK ok=', _oks, None if not _oks else np.round(np.asarray(_qs), 3).tolist(), flush=True)
def go_scan():
    ok = ik_to(SCAN_P, quat_down(0.0), 80, speed=0.3); step(20)
    if not ok: print('[sim] go_scan: 届かなかった', flush=True)
    if a.camera == 'wrist':
        cp, cq = cam.get_world_pose(); print('[sim] wrist camera at scan: pos', np.round(cp, 3).tolist(), flush=True)
    return ok
def go_view():
    """知覚の前の姿勢: 手首カメラなら撮影姿勢へ、固定カメラなら視野からよける。
    器械出し（serve）で手首カメラのときは、見る場所（REGION）の真上に行く: 作業台 / 手渡し台 / 返却トレー"""
    if a.camera != 'wrist': return go_park()
    if SERVE:
        r = REGION['name']; tgt = {'handoff': np.array([0.42, -0.33, 0.40]), 'tray': np.array([0.42, 0.33, 0.40])}.get(r, SCAN_P)
        ok = ik_to(tgt, quat_down(0.0), 80, speed=0.3); step(20)
        if not ok: print(f'[sim] go_view({r}): 届かなかった', flush=True)
        return ok
    return go_scan()
INIT_Q = np.array([0.0, 0.0, 0.0, 0.0, -math.pi / 2, 0.0]) if a.dof == 6 else np.array([0.0, 0.0, 0.0, 0.0, 0.0, -math.pi / 2, 0.0])   # 実機の開始姿勢（手先水平・前向き）
START_Q = INIT_Q if a.start == 'init' else HOME
robot.set_joint_positions(np.concatenate([START_Q, np.zeros(2)]).astype(np.float32)); robot.set_joint_velocities(np.zeros(a.dof + 2, dtype=np.float32))
def set_arm(q):
    robot.apply_action(ArticulationAction(joint_positions=np.asarray(q, dtype=np.float32), joint_indices=np.array(ARM)))
ATTACH = {'joint': None, 'item': None}
def attach_nearest(radius=0.04):
    """TCP の近く（xy 半径 radius）にある器具をグリッパ（link_tcp の親 xarm_gripper_base_link）に固定ジョイントで留める。"""
    tcp, _ = SingleXFormPrim('/World/xarm7/link_tcp').get_world_pose()
    cands = []
    for it in items:
        mn, mx = item_bbox(it['path']); c = (mn + mx) / 2
        pts = np.array([[mn[0], mn[1]], [mx[0], mx[1]], [c[0], c[1]]])
        # 細長い器具: bbox の中心だけでなく TCP が bbox の xy 範囲内（余裕 1.5cm）かで判定
        inside = (mn[0] - 0.015 <= tcp[0] <= mx[0] + 0.015) and (mn[1] - 0.015 <= tcp[1] <= mx[1] + 0.015)
        if inside: cands.append((np.linalg.norm(c[:2] - tcp[:2]), it))
    if not cands: return None
    it = min(cands, key=lambda z: z[0])[1]
    jp = Sdf.Path('/World/xarm7/xarm_gripper_base_link/grasp_joint')
    if stage.GetPrimAtPath(jp).IsValid(): stage.RemovePrim(jp)
    j = UsdPhysics.FixedJoint.Define(stage, jp)
    j.CreateBody0Rel().SetTargets([Sdf.Path('/World/xarm7/xarm_gripper_base_link')]); j.CreateBody1Rel().SetTargets([Sdf.Path(it['path'])])
    # 現在の相対姿勢をジョイントのローカルフレームに入れる
    g_p, g_q = SingleXFormPrim('/World/xarm7/xarm_gripper_base_link').get_world_pose(); i_p, i_q = it['rigid'].get_world_pose()
    Rg = R.from_quat([g_q[1], g_q[2], g_q[3], g_q[0]]); Ri = R.from_quat([i_q[1], i_q[2], i_q[3], i_q[0]])
    rel_p = Rg.inv().apply(np.asarray(i_p) - np.asarray(g_p)); rel_R = (Rg.inv() * Ri); rq = rel_R.as_quat()
    j.CreateLocalPos0Attr().Set(Gf.Vec3f(*map(float, rel_p))); j.CreateLocalRot0Attr().Set(Gf.Quatf(float(rq[3]), float(rq[0]), float(rq[1]), float(rq[2])))
    j.CreateLocalPos1Attr().Set(Gf.Vec3f(0, 0, 0)); j.CreateLocalRot1Attr().Set(Gf.Quatf(1, 0, 0, 0))
    j.CreateExcludeFromArticulationAttr().Set(True)
    ATTACH['joint'] = jp; ATTACH['item'] = it; return it
def detach():
    if ATTACH['joint'] is not None and stage.GetPrimAtPath(ATTACH['joint']).IsValid(): stage.RemovePrim(ATTACH['joint'])
    ATTACH['joint'] = None; ATTACH['item'] = None
def set_grip(v):   # 0.0 = 開く（85mm）, GRIP_CLOSED = 閉じる。v は 0〜1 の割合でも、m でも受ける
    v_m = v * GRIP_CLOSED if v <= 1.0 and v > GRIP_CLOSED else v
    robot.apply_action(ArticulationAction(joint_positions=np.array([v_m, v_m], dtype=np.float32), joint_indices=np.array(GRIPS)))
def grip_opening():   # いまの指の内側の隙間（m）
    q = robot.get_joint_positions(); return float(0.085 - q[GRIPS[0]] - q[GRIPS[1]])
frames = []
VID = {'w': None, 'path': None, 'n': 0}   # 動画はためずに逐次書き出す（ためると 1 周で 10GB 超・途中で止まると全部消える）
CAP = {'lines': []}   # 動画の字幕（医師の言葉・LLM の命令・ロボットの返事）
_FONT = {}
def draw_caption(f):
    from PIL import Image, ImageDraw, ImageFont
    if 'f' not in _FONT:
        import glob
        cands = glob.glob('/usr/share/fonts/**/NotoSansCJK*-Regular.ttc', recursive=True) + glob.glob('/usr/share/fonts/**/NotoSansCJK*.ttc', recursive=True)
        _FONT['f'] = ImageFont.truetype(cands[0], 26, index=0) if cands else ImageFont.load_default()
    if not CAP['lines']: return f
    im = Image.fromarray(np.ascontiguousarray(f)); dr = ImageDraw.Draw(im, 'RGBA'); H = im.size[1]; n = len(CAP['lines'])
    dr.rectangle([0, H - 40 * n - 16, im.size[0], H], fill=(0, 0, 0, 150))
    for i, (txt, col) in enumerate(CAP['lines']): dr.text((20, H - 40 * n - 8 + 40 * i), txt, font=_FONT['f'], fill=col)
    return np.asarray(im)
LASTDET = {'boxes': [], 'until': -1}   # 直近の検出結果（小窓に数秒だけ描く）
JPN = {'scissors': 'ハサミ', 'forceps': 'ピンセット', 'scalpel': 'メス', 'syringe': '注射器'}
def draw_pip(f):
    if not SERVE: return f
    from PIL import Image, ImageDraw
    o = cam.get_rgba()
    if o is None or getattr(o, 'ndim', 0) != 3: return f
    W0, H0 = o.shape[1], o.shape[0]; pw = 448; ph = int(pw * H0 / W0); sc = pw / W0
    im = Image.fromarray(np.ascontiguousarray(f)); pip = Image.fromarray(np.ascontiguousarray(o[:, :, :3])).resize((pw, ph))
    dr = ImageDraw.Draw(pip)
    if getattr(step, 'n', 0) <= LASTDET['until']:
        col = {'scissors': (230, 60, 60), 'forceps': (60, 190, 60), 'scalpel': (60, 120, 230), 'syringe': (230, 190, 30)}
        for xyxy, nm, cf in LASTDET['boxes']:
            x0, y0, x1, y1 = [v * sc for v in xyxy]; c = col.get(nm, (255, 255, 255)); dr.rectangle([x0, y0, x1, y1], outline=c, width=2)
            if 'f' in _FONT:
                from PIL import ImageFont
                fs = ImageFont.truetype(_FONT['f'].path, 14); dr.text((x0, max(0, y0 - 16)), f'{JPN.get(nm, nm)} {cf:.2f}', font=fs, fill=c)
    x = im.size[0] - pw - 12; y = 12; d2 = ImageDraw.Draw(im); d2.rectangle([x - 3, y - 3, x + pw + 2, y + ph + 22], fill=(0, 0, 0))
    im.paste(pip, (x, y))
    if 'f' in _FONT:
        from PIL import ImageFont
        d2.text((x + 6, y + ph + 2), 'ロボットの手首カメラ（検出モデルの結果）' if a.camera == 'wrist' else 'ロボットのカメラ：作業台の上の固定カメラ（検出の結果）', font=ImageFont.truetype(_FONT['f'].path, 15), fill=(255, 255, 255))
    return np.asarray(im)
HOLD = {'label': None}   # 掴んでいる間だけ出すラベル（ロボットが検出で判断した種類）
def draw_hold(f):
    if not HOLD['label'] or not SERVE: return f
    from PIL import Image, ImageDraw, ImageFont
    tcp, _ = SingleXFormPrim('/World/xarm7/link_tcp').get_world_pose()
    uv = spec.get_image_coords_from_world_points(np.array([tcp], dtype=np.float32))[0]; u, v = float(uv[0]), float(uv[1])
    im = Image.fromarray(np.ascontiguousarray(f)); d = ImageDraw.Draw(im, 'RGBA'); ft = ImageFont.truetype(_FONT['f'].path, 20); t = HOLD['label']
    tw = d.textlength(t, font=ft); x0, y0 = u + 18, v - 40
    d.rounded_rectangle([x0 - 8, y0 - 4, x0 + tw + 8, y0 + 26], radius=8, fill=(20, 20, 20, 170)); d.text((x0, y0), t, font=ft, fill=(255, 220, 90))
    d.line([(u + 6, v - 6), (x0 - 8, y0 + 12)], fill=(255, 220, 90, 200), width=2)
    return np.asarray(im)
def vid_append(f):
    f = draw_caption(f); f = draw_pip(f); f = draw_hold(f)   # draw_caption が最初にフォントを読み込む（字幕は下、小窓は右上なので重ならない）
    if VID['w'] is None:   # このPCの Isaac には imageio-ffmpeg が無いので、ffmpeg に生の画素を流し込む（Isaac 本体は触らない）
        import subprocess
        VID['path'] = f"{OUT}/{'demo' if a.stage in ('full', 'serve') else 'grasp'}.mp4"; h, w = f.shape[:2]
        VID['w'] = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{w}x{h}', '-r', '30', '-i', '-',
                                     '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '20', VID['path']], stdin=subprocess.PIPE)
    VID['w'].stdin.write(np.ascontiguousarray(f, dtype=np.uint8).tobytes()); VID['n'] += 1
def vid_close():
    if VID['w'] is not None: VID['w'].stdin.close(); VID['w'].wait(); VID['w'] = None; print(f"[sim] video {VID['n']} frames -> {VID['path']}", flush=True)
def write_video(path, frs, fps=30):
    import subprocess, tempfile, shutil
    try:
        iio.mimwrite(path, frs, fps=fps, codec='libx264', quality=8); print(f'[sim] video {len(frs)} frames -> {path}', flush=True); return
    except Exception as e:
        print('[sim] imageio video failed:', repr(e)[:100], '→ ffmpeg CLI', flush=True)
    d = tempfile.mkdtemp()
    for i, f in enumerate(frs): iio.imwrite(f'{d}/{i:05d}.png', f)
    r = subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(fps), '-i', f'{d}/%05d.png', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', path]); shutil.rmtree(d)
    print(f'[sim] video (ffmpeg) rc={r.returncode} {len(frs)} frames -> {path}', flush=True)
def step(n=1, record=True):
    for _ in range(n):
        world.step(render=True)
        step.n = getattr(step, 'n', 0) + 1
        if record and a.video and (step.n % 8 == 0):
            f = spec.get_rgba()
            if f is not None and getattr(f, 'size', 0) > 0 and f.ndim == 3: vid_append(np.asarray(f)[:, :, :3])
            elif VID['n'] == 0 and world.current_time_step_index % 40 == 0: print('[sim] (video) close.get_rgba() ->', None if f is None else getattr(f, 'shape', type(f)), 'step', world.current_time_step_index, flush=True)
def ik_to(pos, quat, steps=60, tol=0.006, speed=0.25, seg=0.01):
    """TCP を直線経路で目標へ。1cm 刻みの経路点ごとに IK を解いて関節の列を作り、全体を 1 本のなめらかな速度プロファイル（cos）で再生する。
    （関節空間だけの補間は手先が振れる／区切って止めると継ぎ目の揺れで掴んだ器具が落ちた）"""
    tcp_now, _ = aks.compute_end_effector_pose(); start_p = np.asarray(tcp_now, dtype=np.float64); goal = np.asarray(pos, dtype=np.float64)
    dist = float(np.linalg.norm(goal - start_p)); n_wp = max(1, int(math.ceil(dist / seg)))
    q_cur = robot.get_joint_positions()[np.array(ARM)].astype(np.float64); qs = []
    # 手先の向き（手首の回転）が今と大きく違うときは、先にその場で向きだけ合わせる。
    # 直線経路の 1 歩目に回転を押し込むと下の「関節の飛び」判定に掛かって腕が一歩も動かなかった（2026-10-17 の 12 個版 v21/v22 が全滅した原因）
    q1, ok1 = lula.compute_inverse_kinematics('link_tcp', start_p, np.asarray(quat, dtype=np.float64), warm_start=np.asarray(q_cur, dtype=np.float64))
    if ok1 and np.abs(np.asarray(q1, dtype=np.float64) - q_cur).max() > 0.3:
        q1 = np.asarray(q1, dtype=np.float64); dj = float(np.abs(q1 - q_cur).max()); n_rot = int(max(0.4, dj / 1.5) / a.physics_dt)
        for i in range(1, n_rot + 1):
            w = 0.5 - 0.5 * math.cos(math.pi * i / n_rot); set_arm(q_cur + (q1 - q_cur) * w); step(1)
        for _ in range(int(0.25 / a.physics_dt)):
            set_arm(q1); step(1)
            if np.abs(robot.get_joint_positions()[np.array(ARM)] - q1).max() < 0.01: break
        q_cur = robot.get_joint_positions()[np.array(ARM)].astype(np.float64)
    for k in range(1, n_wp + 1):
        wp = start_p + (goal - start_p) * (k / n_wp)
        seed = qs[-1] if qs else q_cur
        qk, ok = lula.compute_inverse_kinematics('link_tcp', wp, np.asarray(quat, dtype=np.float64), warm_start=np.asarray(seed, dtype=np.float64))
        if not ok: return False
        qk = np.asarray(qk, dtype=np.float64)
        if np.abs(qk - seed).max() > 0.6:   # それでも枝が飛んだら中止
            print(f'[sim]   ik_to: joint jump {np.abs(qk - seed).max():.2f} rad at wp {k}/{n_wp}', flush=True); return False
        qs.append(qk)
    qs = np.array(qs); q0 = robot.get_joint_positions()[np.array(ARM)].astype(np.float64)
    dur = max(0.4, dist / speed); n_steps = int(dur / a.physics_dt)
    for i in range(1, n_steps + 1):
        t = i / n_steps; w = 0.5 - 0.5 * math.cos(math.pi * t)          # 0→1 なめらかに
        f = w * n_wp; k = min(int(math.floor(f)), n_wp - 1); frac = f - k   # 経路点の間を線形に
        q_a = q0 if k == 0 else qs[k - 1]; q_b = qs[k]
        robot.apply_action(ArticulationAction(joint_positions=(q_a + (q_b - q_a) * frac).astype(np.float32), joint_indices=np.array(ARM))); step(1)
    for _ in range(int(0.25 / a.physics_dt)):
        robot.apply_action(ArticulationAction(joint_positions=qs[-1].astype(np.float32), joint_indices=np.array(ARM))); step(1)
        if np.abs(robot.get_joint_positions()[np.array(ARM)] - qs[-1]).max() < 0.01: break
    tcp_p, _ = aks.compute_end_effector_pose(); err = float(np.linalg.norm(np.asarray(tcp_p) - goal))
    if err > tol: print(f'[sim]   ik_to: tcp err {err*100:.1f} cm (target {np.round(pos,3).tolist()})', flush=True)
    return err < tol
def finger_bottom_offset():
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy, UsdGeom.Tokens.guide])
    zs = [np.array(cache.ComputeWorldBound(stage.GetPrimAtPath(f'/World/xarm7/{f}')).ComputeAlignedRange().GetMin())[2] for f in ('left_pad', 'right_pad')]
    tcp, _ = SingleXFormPrim('/World/xarm7/link_tcp').get_world_pose(); return float(tcp[2] - min(zs))
def tcp_pos():
    p, _ = aks.compute_end_effector_pose(); return np.asarray(p)
# ---- 初期姿勢・落下して落ち着くまで ----
set_arm(START_Q); set_grip(0.0); step(120, record=False); print('[sim] start', a.start, 'tcp=', tcp_pos().round(3).tolist(), 'camera', a.camera, flush=True)
def save_views(tag):
    rgb = cam.get_rgba()[:, :, :3]; iio.imwrite(f'{OUT}/{tag}_realsense_rgb.png', rgb.astype(np.uint8))
    d = cam.get_depth()
    if d is not None:
        dd = np.nan_to_num(d, nan=0.0, posinf=0.0); dd = np.clip((dd - 0.5) / 0.6, 0, 1); iio.imwrite(f'{OUT}/{tag}_realsense_depth.png', (255 * (1 - dd)).astype(np.uint8))
    fr = cam.get_current_frame(); seg = fr.get('semantic_segmentation')
    if seg is not None:
        data = seg['data'] if isinstance(seg, dict) else seg
        pal = (np.array([[0,0,0],[230,50,50],[50,180,50],[50,80,230],[230,200,30],[200,200,200],[120,120,120],[255,0,255]]))
        iio.imwrite(f'{OUT}/{tag}_realsense_semantic.png', pal[np.asarray(data) % len(pal)].astype(np.uint8))
        info = seg.get('info', {}) if isinstance(seg, dict) else {}
        json.dump({str(k): v for k, v in (info.get('idToLabels', {}) or {}).items()}, open(f'{OUT}/{tag}_semantic_labels.json', 'w'), ensure_ascii=False, indent=1)
    s = spec.get_rgba()[:, :, :3]; iio.imwrite(f'{OUT}/{tag}_spectator.png', s.astype(np.uint8))
    c = close.get_rgba()[:, :, :3]; iio.imwrite(f'{OUT}/{tag}_closeup.png', c.astype(np.uint8))
    print(f'[sim] saved views {tag}', flush=True)
for it in items[:3]:
    mn, mx = item_bbox(it['path']); pos, _ = it['rigid'].get_world_pose(); sc = it['rigid'].get_local_scale()
    xo = UsdGeom.Xformable(stage.GetPrimAtPath(it['path'])); ops = [(o.GetOpName(), o.Get()) for o in xo.GetOrderedXformOps()]
    print(f"[sim] after-reset {it['name']}: pos={np.round(pos,3).tolist()} scale={np.round(sc,4).tolist()} bbox_size={np.round(mx-mn,4).tolist()} zrange=({mn[2]:.4f},{mx[2]:.4f}) ops={ops}", flush=True)
save_views('00_scene')
report = {'stage': a.stage, 'seed': a.seed, 'items': [{'name': it['name'], 'type': it['type'], 'pos0': it['rigid'].get_world_pose()[0].round(3).tolist()} for it in items]}
# ---- 知覚（段階 1）: 深度で机より上の塊、種類は semantic ラベル、向きは PCA ----
SIZE_TABLE = {'scissors': (0.17, 0.09), 'forceps': (0.18, 0.010), 'scalpel': (0.15, 0.014), 'syringe': (0.10, 0.022)}   # (長さ, 幅) m。資料のアセット実寸から
def classify_by_size(length, width):
    best, bd = None, 1e9
    for t, (L, W) in SIZE_TABLE.items():
        dd = ((length - L) / 0.03) ** 2 + ((width - W) / 0.015) ** 2
        if dd < bd: best, bd = t, dd
    return best
def perceive_depth():
    """深度だけで: 机面(z=0)より 3mm 以上高い画素 → 連結成分 → 3D 点群 → 長さ・幅 → 大きさで種類。正解ラベルは使わない。"""
    from scipy import ndimage
    d = degrade_depth(np.asarray(cam.get_depth())); H, W = d.shape
    ys, xs = np.mgrid[0:H, 0:W]; ok = np.isfinite(d) & (d > 0.05)
    pts = np.full((H, W, 3), np.nan, dtype=np.float64)
    p = cam.get_world_points_from_image_coords(np.stack([xs[ok], ys[ok]], 1).astype(np.float32), d[ok]); pts[ok] = apply_calib_err(np.asarray(p))
    z = pts[..., 2]; x, y = pts[..., 0], pts[..., 1]
    mask = ok & (z > 0.003) & (z < 0.03) & (x > 0.22) & (x < 0.74) & (np.abs(y) < a.spread + 0.06)   # 作業面の中（箱と腕を除く）、机から 3mm〜3cm
    lab, n = ndimage.label(mask); found = []
    for i in range(1, n + 1):
        m = lab == i
        if m.sum() < 40: continue
        P = pts[m]; P = P[np.isfinite(P).all(1)]
        top = P[P[:, 2] >= np.percentile(P[:, 2], 60)];  top = top if len(top) >= 5 else P
        cen = top.mean(0); c = top[np.argmin(np.linalg.norm(top[:, :2] - cen[:2], axis=1))].copy(); c[2] = cen[2]
        xy = P[:, :2] - P[:, :2].mean(0); w, v = np.linalg.eigh(np.cov(xy.T)); axis = v[:, np.argmax(w)]
        yaw = math.atan2(axis[1], axis[0]); length = float(np.ptp(xy @ axis)); width = float(np.ptp(xy @ v[:, np.argmin(w)]))
        found.append({'type': classify_by_size(length, width), 'pos': c, 'yaw': yaw, 'length': length, 'width': width, 'top_z': float(np.percentile(P[:, 2], 95)), 'pixels': int(m.sum())})
    return found
DET = {'proc': None, 'n': 0}
rng_real = np.random.default_rng(a.seed + 1000)
_cm, _cd = [float(v) for v in a.calib_err.split(',')]
CALIB_T = rng_real.uniform(-_cm, _cm, 3) / 1000.0; CALIB_TH = math.radians(rng_real.uniform(-_cd, _cd))   # 実行ごとに固定
if a.depth_noise or a.depth_quant or a.depth_dropout or _cm or _cd: print(f'[sim] real-ish: noise {a.depth_noise}mm quant {a.depth_quant}mm dropout {a.depth_dropout} calib dt={np.round(CALIB_T*1000,1).tolist()}mm dth={math.degrees(CALIB_TH):.2f}deg friction {a.friction}', flush=True)
def degrade_depth(d):
    """シミュレータの完璧な深度を RealSense 風に劣化させる: ガウスノイズ（距離の 2 乗）、量子化、金属面の抜け（丸い穴）。"""
    if not (a.depth_noise or a.depth_quant or a.depth_dropout): return d
    import cv2
    d = np.array(d, dtype=np.float64, copy=True); ok = np.isfinite(d) & (d > 0.05)
    if a.depth_dropout > 0:   # 机より手前（器具）の画素に穴を開ける
        table = np.nanmedian(d[ok]) if ok.any() else 0.7; near = ok & (d < table - 0.0012)
        ys_, xs_ = np.nonzero(near); hole = np.zeros(d.shape, np.uint8)
        n_h = int(a.depth_dropout * len(xs_) / 50)   # 穴 1 つ ≈ 50 画素
        for k in rng_real.integers(0, max(1, len(xs_)), n_h) if len(xs_) else []:
            cv2.circle(hole, (int(xs_[k]), int(ys_[k])), int(rng_real.integers(2, 6)), 1, -1)
        d[hole.astype(bool)] = np.nan
    if a.depth_noise > 0: d[ok] += rng_real.normal(0.0, 1.0, ok.sum()) * (a.depth_noise / 1000.0) * (d[ok] / 0.7) ** 2
    if a.depth_quant > 0: q = (a.depth_quant / 1000.0) * (d[ok] / 0.7) ** 2; d[ok] = np.round(d[ok] / q) * q
    return d
def apply_calib_err(pts):
    """キャリブレーション誤差: 机の座標に直した点を、カメラ直下 (0.46, 0) まわりに少し回して少しずらす（本当の位置とズレた所を掴みに行く）。"""
    if CALIB_TH == 0 and not CALIB_T.any(): return pts
    c = np.array([0.46, 0.0, 0.0]); q = pts - c; cth, sth = math.cos(CALIB_TH), math.sin(CALIB_TH)
    out = np.empty_like(q); out[..., 0] = cth * q[..., 0] - sth * q[..., 1]; out[..., 1] = sth * q[..., 0] + cth * q[..., 1]; out[..., 2] = q[..., 2]
    return out + c + CALIB_T
def detector_start():
    """検出モデルを別プロセス（anaconda の python、ultralytics 入り）で起動。Isaac の venv は触らない。"""
    import subprocess
    py = os.path.join(HERE, '.venv-det/bin/python')   # ultralytics 入りの別環境
    DET['proc'] = subprocess.Popen([py, os.path.join(HERE, 'detector_worker.py'), os.path.join(ROOT, a.weights), str(a.det_conf)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=open(f'{OUT}/detector_worker.log', 'w'), text=True, bufsize=1)
    line = DET['proc'].stdout.readline().strip(); print('[sim] detector:', line, flush=True)
    if not line.startswith('READY'): raise RuntimeError('detector worker が起動しなかった: ' + line)
def thin_fallback(dt, rgb, d, ok, inbox):
    """深度で器具の点が取れないとき（薄いメス柄・金属の抜け）: 四角の中の机の画素の深度の中央値で四角の中心を 3D にし（机の上の点）、向きは RGB で机より明るい画素の主軸。"""
    x0, y0, x1, y1 = [int(round(v)) for v in dt['xyxy']]; H, W = d.shape; x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W - 1, x1), min(H - 1, y1)
    tb = inbox & ok
    if tb.sum() < 20: return None
    dmed = float(np.median(d[tb])); cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    gray = rgb[y0:y1 + 1, x0:x1 + 1].astype(np.float64).mean(2); sel = gray > (np.median(gray) + 12)
    yy, xx = np.nonzero(sel)
    if len(xx) >= 30:
        cx, cy = x0 + xx.mean(), y0 + yy.mean(); xy = np.stack([xx - xx.mean(), yy - yy.mean()], 1); w, v = np.linalg.eigh(np.cov(xy.T)); ax = v[:, np.argmax(w)]
    else: ax = np.array([1.0, 0.0]) if (x1 - x0) >= (y1 - y0) else np.array([0.0, 1.0])
    pix = np.array([[cx, cy], [cx + 20 * ax[0], cy + 20 * ax[1]]], dtype=np.float32)
    P = apply_calib_err(np.asarray(cam.get_world_points_from_image_coords(pix, np.array([dmed, dmed], dtype=np.float32))))
    c = P[0].copy(); c[2] = 0.002; dv = P[1] - P[0]; yaw = math.atan2(dv[1], dv[0])
    print(f"[sim]   thin_fallback: {dt['name']} 深度の点が無い → 四角の中心を机の上に ({c[0]:.3f},{c[1]:.3f}) yaw {yaw:.2f}", flush=True)
    return {'type': dt['name'], 'conf': dt['conf'], 'pos': c, 'yaw': yaw, 'length': float((x1 - x0) / 1000.0), 'width': 0.0, 'top_z': 0.003, 'pixels': int(tb.sum()) // 4, 'box': dt['xyxy'], 'fallback': True}
REGION = {'name': 'table'}   # 知覚する場所: table = 器械台（散らばり領域）/ handoff = 手渡し台の中 / all = 全部
def region_mask(x, y):
    r = REGION['name']
    if SERVE and r == 'table': return (x > 0.22) & (x < 0.74) & (np.abs(y) < min(a.spread + 0.06, MAT_Y - MAT_L / 2 - 0.02))   # 台の手前で止める（台の上の器具を二重に数えない）
    if not SERVE or r == 'table': return (x > 0.22) & (x < 0.74) & (np.abs(y) < a.spread + 0.06)
    def in_bin(c): return (np.abs(x - c[0]) < MAT_W / 2) & (np.abs(y - c[1]) < MAT_L / 2)
    if r == 'handoff': return in_bin(HANDOFF_POS)
    if r == 'tray': return in_bin(TRAY_POS)
    return ((x > 0.22) & (x < 0.74) & (np.abs(y) < a.spread + 0.06)) | in_bin(HANDOFF_POS) | in_bin(TRAY_POS)
def perceive_detector():
    """RGB を検出モデルに掛けて種類と四角 → 四角の中の深度（机より 3mm〜3cm 高い画素）→ 3D 位置・向き（PCA）。正解ラベルは使わない。"""
    import cv2
    if DET['proc'] is None: detector_start()
    rgb = np.asarray(cam.get_rgba())[:, :, :3].astype(np.uint8); p_in = f'{OUT}/_det_in.png'; iio.imwrite(p_in, rgb)
    DET['proc'].stdin.write(p_in + '\n'); DET['proc'].stdin.flush(); dets = json.loads(DET['proc'].stdout.readline())
    def _iou(b1, b2):
        ix = max(0.0, min(b1[2], b2[2]) - max(b1[0], b2[0])); iy = max(0.0, min(b1[3], b2[3]) - max(b1[1], b2[1])); inter = ix * iy
        return inter / ((b1[2] - b1[0]) * (b1[3] - b1[1]) + (b2[2] - b2[0]) * (b2[3] - b2[1]) - inter + 1e-9)
    keep = []   # 同じ器具に種類違いの四角が 2 つ付くことがある → 重なり IoU>0.6 は自信の高い方だけ残す
    for dt in sorted(dets, key=lambda d: -d['conf']):
        if all(_iou(dt['xyxy'], k['xyxy']) < 0.6 for k in keep): keep.append(dt)
    dets = keep
    d = degrade_depth(np.asarray(cam.get_depth())); H, W = d.shape
    ys, xs = np.mgrid[0:H, 0:W]; ok = np.isfinite(d) & (d > 0.05)
    pts = np.full((H, W, 3), np.nan, dtype=np.float64)
    p = cam.get_world_points_from_image_coords(np.stack([xs[ok], ys[ok]], 1).astype(np.float32), d[ok]); pts[ok] = apply_calib_err(np.asarray(p))
    z = pts[..., 2]; x, y = pts[..., 0], pts[..., 1]
    inbins = region_mask(x, y) & ~((x > 0.22) & (x < 0.74) & (np.abs(y) < a.spread + 0.06))
    mask = ok & (z > np.where(inbins, MAT_T + 0.0015, 0.0015)) & (z < 0.036) & region_mask(x, y)   # 種類は検出モデルが言うので、深度は 1.5mm 以上で拾う（メス柄 3mm が 3mm 閾値で落ちた）
    # 画素がどの四角に属すか（重なった所は面積の小さい四角を優先）
    owner = np.full((H, W), -1, dtype=np.int32)
    for i in sorted(range(len(dets)), key=lambda i: -(dets[i]['xyxy'][2] - dets[i]['xyxy'][0]) * (dets[i]['xyxy'][3] - dets[i]['xyxy'][1])):
        x0, y0, x1, y1 = [int(round(v)) for v in dets[i]['xyxy']]; owner[max(0, y0):min(H, y1 + 1), max(0, x0):min(W, x1 + 1)] = i
    found = []; vis = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR); col = {'scissors': (60, 60, 230), 'forceps': (60, 200, 60), 'scalpel': (230, 120, 60), 'syringe': (40, 200, 230)}
    for i, dt in enumerate(dets):
        x0, y0, x1, y1 = [int(round(v)) for v in dt['xyxy']]; cv2.rectangle(vis, (x0, y0), (x1, y1), col.get(dt['name'], (255, 255, 255)), 2); cv2.putText(vis, f"{dt['name']} {dt['conf']:.2f}", (x0, max(12, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col.get(dt['name'], (255, 255, 255)), 1)
        m = mask & (owner == i)
        if m.sum() < 40:
            if dt['conf'] < 0.7: continue   # 深度の点が無く自信も低いものは器具ではない（グリッパーの写り込み等、10/31）
            fb = thin_fallback(dt, rgb, d, ok, owner == i)
            if fb is not None: found.append(fb)
            continue
        P = pts[m]; P = P[np.isfinite(P).all(1)]
        top = P[P[:, 2] >= np.percentile(P[:, 2], 60)]; top = top if len(top) >= 5 else P
        cen = top.mean(0); c = top[np.argmin(np.linalg.norm(top[:, :2] - cen[:2], axis=1))].copy(); c[2] = cen[2]
        xy = P[:, :2] - P[:, :2].mean(0); w, v = np.linalg.eigh(np.cov(xy.T)); axis = v[:, np.argmax(w)]
        yaw = math.atan2(axis[1], axis[0]); length = float(np.ptp(xy @ axis)); width = float(np.ptp(xy @ v[:, np.argmin(w)]))
        found.append({'type': dt['name'], 'conf': dt['conf'], 'pos': c, 'yaw': yaw, 'length': length, 'width': width, 'top_z': float(np.percentile(P[:, 2], 95)), 'pixels': int(m.sum()), 'box': dt['xyxy']})
    found = [f for f in found if bool(region_mask(np.array(f['pos'][0]), np.array(f['pos'][1])))]
    LASTDET['boxes'] = [(dt['xyxy'], dt['name'], dt['conf']) for dt in dets]; LASTDET['until'] = getattr(step, 'n', 0) + int(4.0 / a.physics_dt)
    cv2.imwrite(f"{OUT}/det_{DET['n']:02d}.png", vis); DET['n'] += 1
    print(f'[sim] detector: {len(dets)} boxes -> {len(found)} with depth', [(dt['name'], dt['conf']) for dt in dets], flush=True)
    return found
def perceive():
    if a.perception == 'depth': return perceive_depth()
    if a.perception == 'detector': return perceive_detector()
    fr = cam.get_current_frame(); d = np.asarray(cam.get_depth()); seg = fr.get('semantic_segmentation')
    data = np.asarray(seg['data']); id2label = {int(k): (v.get('class') if isinstance(v, dict) else str(v)) for k, v in (seg['info']['idToLabels'] or {}).items()}
    inst = fr.get('instance_segmentation'); idata = np.asarray(inst['data']) if inst is not None else None
    uniq, cnt = np.unique(data, return_counts=True); print('[sim] semantic ids:', {int(u): (id2label.get(int(u), '?'), int(c)) for u, c in zip(uniq, cnt)}, flush=True)
    found = []
    for sid, label in id2label.items():
        if label not in TYPES: continue
        m = (data == sid)
        if idata is not None:  # インスタンスごとに分ける
            groups = [(m & (idata == iid)) for iid in np.unique(idata[m]) if iid != 0]
        else: groups = [m]
        for g in groups:
            ys, xs = np.nonzero(g)
            if len(xs) < 40: continue
            pts = cam.get_world_points_from_image_coords(np.stack([xs, ys], 1).astype(np.float32), d[ys, xs])
            pts = np.asarray(pts); pts = pts[np.isfinite(pts).all(1)]
            if len(pts) < 20: continue
            top = pts[pts[:, 2] >= np.percentile(pts[:, 2], 60)]
            if len(top) < 5: top = pts
            cen = top.mean(0); c = top[np.argmin(np.linalg.norm(top[:, :2] - cen[:2], axis=1))].copy(); c[2] = cen[2]   # 重心に一番近い「物体上の点」
            xy = pts[:, :2] - pts[:, :2].mean(0); w, v = np.linalg.eigh(np.cov(xy.T)); axis = v[:, np.argmax(w)]
            yaw = math.atan2(axis[1], axis[0]); length = float(np.ptp(xy @ axis)); width = float(np.ptp(xy @ v[:, np.argmin(w)]))
            found.append({'type': label, 'pos': c, 'yaw': yaw, 'length': length, 'width': width, 'top_z': float(np.percentile(pts[:, 2], 95)), 'pixels': int(len(xs))})
    return found
def grasp_closed_enough():
    return grip_opening() > 0.004   # 完全に閉じ切っていなければ何かを挟んでいる
if a.stage in ('ik', 'full'):
    if a.stage == 'full' or a.camera == 'wrist': go_view()   # 手首カメラは撮影姿勢（俯瞰）に行ってから見る
    det = perceive(); print(f'[sim] perceived {len(det)}:', [(x['type'], x['pos'].round(3).tolist(), round(x['yaw'], 2), round(x['length'], 3), x['pixels']) for x in det], flush=True)
    report['perceived'] = [{'type': x['type'], 'pos': x['pos'].round(4).tolist(), 'yaw': round(x['yaw'], 3), 'length': round(x['length'], 3), 'pixels': x['pixels']} for x in det]
    # 真値と照合
    def center_of(it):
        mn, mx = item_bbox(it['path']); return (mn + mx) / 2
    truth = {it['name']: center_of(it) for it in items}
    errs = []
    for x in det:
        best = min(((n, np.linalg.norm(p - x['pos'])) for n, p in truth.items()), key=lambda z: z[1], default=None)
        if best: errs.append({'det': x['type'], 'nearest': best[0], 'err_cm': round(float(best[1]) * 100, 2)})
    # 種類の正誤（最寄りの真の器具の種類と比べる）
    allt = [(n, p) for n, p in truth.items()]
    hits = 0
    for x in det:
        n_near = min(allt, key=lambda z: np.linalg.norm(z[1][:2] - x['pos'][:2]))[0]; hits += int(n_near.startswith(x['type']))
    report['type_accuracy'] = f'{hits}/{len(det)}'; print(f'[sim] 種類の正答 {hits}/{len(det)}（知覚={a.perception}）', flush=True)
    report['perception_check'] = errs; print('[sim] perception vs truth (cm):', errs, flush=True)
if a.stage == 'ik' and det:
    x = det[0]; above = x['pos'] + np.array([0, 0, 0.15])
    action, ik_ok = aks.compute_inverse_kinematics(np.asarray(above, dtype=np.float64), np.asarray(quat_down(x['yaw']), dtype=np.float64))
    print('[sim] ik solution ok=', ik_ok, 'joint_positions=', None if action.joint_positions is None else np.round(action.joint_positions, 3).tolist(), 'indices=', action.joint_indices, flush=True)
    ok = ik_to(above, quat_down(x['yaw']), steps=120)
    q_now = robot.get_joint_positions(); tcp_prim = SingleXFormPrim('/World/xarm7/link_tcp'); tcp_w, _ = tcp_prim.get_world_pose()
    print('[sim] ik above first item ok=', ok, 'lula_tcp=', tcp_pos().round(3).tolist(), 'prim_tcp=', np.round(tcp_w, 3).tolist(), 'target=', above.round(3).tolist(), 'q_arm=', np.round(q_now[ARM], 2).tolist(), flush=True)
    report['ik_test'] = {'ok': ok, 'tcp': tcp_pos().round(4).tolist(), 'target': above.round(4).tolist()}
    save_views('01_ik')
if a.stage == 'poke':
    # 当たり判定の診断: 器具のメッシュの Collision API と、閉じたグリッパで真上から突いて動くか
    it = [i for i in items if i['name'] == (a.target_type or 'forceps_1')][0]
    for d in Usd.PrimRange(stage.GetPrimAtPath(it['path'])):
        if d.GetTypeName() == 'Mesh':
            appr = UsdPhysics.MeshCollisionAPI(d).GetApproximationAttr().Get() if d.HasAPI(UsdPhysics.MeshCollisionAPI) else None
            print('[diag] mesh', d.GetPath().pathString.split('/')[-1], 'collision=', d.HasAPI(UsdPhysics.CollisionAPI), 'approx=', appr, flush=True)
    for fp in ('left_finger', 'right_finger', 'xarm_gripper_base_link'):
        for d in Usd.PrimRange(stage.GetPrimAtPath(f'/World/xarm7/{fp}'), Usd.TraverseInstanceProxies()):
            if d.HasAPI(UsdPhysics.CollisionAPI) or d.GetTypeName() == 'Mesh':
                appr = UsdPhysics.MeshCollisionAPI(d).GetApproximationAttr().Get() if d.HasAPI(UsdPhysics.MeshCollisionAPI) else None
                print('[diag] finger prim', d.GetPath().pathString.replace('/World/xarm7/', ''), d.GetTypeName(), 'collision=', d.HasAPI(UsdPhysics.CollisionAPI), 'approx=', appr, 'enabled=', UsdPhysics.CollisionAPI(d).GetCollisionEnabledAttr().Get() if d.HasAPI(UsdPhysics.CollisionAPI) else None, flush=True)
    # 机を突く: 指先が机より下に行けたら、指に当たり判定が無い
    set_grip(GRIP_CLOSED); step(60); ik_to(np.array([0.45, 0.0, 0.12]), quat_down(0.0), 80); ok_t = ik_to(np.array([0.45, 0.0, finger_bottom_offset() - 0.015]), quat_down(0.0), 80)
    tcp, _ = SingleXFormPrim('/World/xarm7/link_tcp').get_world_pose(); print(f'[diag] table poke: target finger_bottom=-15mm → reached finger_bottom={(tcp[2]-finger_bottom_offset())*1000:.1f}mm (0 より下なら指は机をすり抜けている)', flush=True)
    ik_to(np.array([0.45, 0.0, 0.15]), quat_down(0.0), 60)
    p0, _ = it['rigid'].get_world_pose(); mn, mx = item_bbox(it['path']); print('[diag] item pos', np.round(p0, 4).tolist(), 'bbox', np.round(mn, 3).tolist(), np.round(mx, 3).tolist(), flush=True)
    cpos = (mn + mx) / 2
    set_grip(GRIP_CLOSED); step(60)
    ik_to(cpos + np.array([0, 0, 0.12]), quat_down(0.0), 80)
    ik_to(np.array([cpos[0], cpos[1], finger_bottom_offset() + 0.003]), quat_down(0.0), 80)
    tcp, _ = SingleXFormPrim('/World/xarm7/link_tcp').get_world_pose(); p1, _ = it['rigid'].get_world_pose()
    print(f'[diag] poke at bbox center: tcp z={tcp[2]*1000:.1f}mm finger_bottom={(tcp[2]-finger_bottom_offset())*1000:.1f}mm (item top {mx[2]*1000:.1f}mm) | item moved {np.linalg.norm(p1-p0)*1000:.1f} mm', flush=True)
    save_views('poke')
if a.stage == 'cube':
    for ln in ('xarm_gripper_base_link', 'left_pad', 'right_pad', 'link7', 'link_tcp'):
        lp = stage.GetPrimAtPath(f'/World/xarm7/{ln}')
        if not lp.IsValid(): print('[sim] link missing', ln); continue
        m = UsdPhysics.MassAPI(lp).GetMassAttr().Get() if lp.HasAPI(UsdPhysics.MassAPI) else None
        rb = UsdPhysics.RigidBodyAPI(lp); print(f'[sim] link {ln}: mass={m} rigidBodyAPI={lp.HasAPI(UsdPhysics.RigidBodyAPI)} enabled={rb.GetRigidBodyEnabledAttr().Get() if lp.HasAPI(UsdPhysics.RigidBodyAPI) else None} kinematic={rb.GetKinematicEnabledAttr().Get() if lp.HasAPI(UsdPhysics.RigidBodyAPI) else None} schemas={[a_ for a_ in lp.GetAppliedSchemas() if "Physx" in a_ or "Physics" in a_][:6]}', flush=True)
    # PhysX に実在する当たり判定を直接問い合わせる（指先・器具・机）
    from omni.physx import get_physx_scene_query_interface
    sq = get_physx_scene_query_interface()
    def overlap_at(pos, r=0.01):
        hits = []
        def cb(hit): hits.append(hit.rigid_body + ' | ' + hit.collision); return True
        sq.overlap_sphere(r, [float(pos[0]), float(pos[1]), float(pos[2])], cb, False); return hits
    lf, _ = SingleXFormPrim('/World/xarm7/left_pad').get_world_pose(); tcp0, _ = SingleXFormPrim('/World/xarm7/link_tcp').get_world_pose()
    print('[sim] overlap at left_pad origin:', overlap_at(lf, 0.012), flush=True)
    print('[sim] overlap at tcp (fingertip height):', overlap_at(tcp0, 0.02), flush=True)
    it0 = items[0]; mn, mx = item_bbox(it0['path']); print('[sim] overlap at item', it0['name'], ':', overlap_at((mn + mx) / 2, 0.01), flush=True)
    print('[sim] overlap at table:', overlap_at(np.array([0.45, 0.1, 0.0]), 0.01), flush=True)
    from isaacsim.core.api.objects import DynamicCuboid
    cube = world.scene.add(DynamicCuboid(prim_path='/World/probe_cube', name='probe_cube', position=np.array([0.45, 0.0, 0.0125]), scale=np.array([0.025, 0.025, 0.025]), color=np.array([0.9, 0.3, 0.1]), mass=0.1))
    bind_phys_material(stage.GetPrimAtPath('/World/probe_cube')); step(60)
    cp, _ = cube.get_world_pose(); print('[sim] cube at', np.round(cp, 3).tolist(), flush=True)
    def fi(tag):
        tcp, _ = SingleXFormPrim('/World/xarm7/link_tcp').get_world_pose()
        print(f"[sim] {tag}: tcp={np.round(tcp,3).tolist()} opening={grip_opening()*1000:.1f}mm finger_bottom={(tcp[2]-finger_bottom_offset())*1000:.1f}mm", flush=True)
    # (1) 弱い腕で机を突く（指の当たり判定の確認）
    set_grip(GRIP_CLOSED); step(60); ik_to(np.array([0.35, -0.15, 0.12]), quat_down(0.0), 80); ik_to(np.array([0.35, -0.15, finger_bottom_offset() - 0.015]), quat_down(0.0), 80); fi('table poke (target finger_bottom -15mm)')
    ik_to(np.array([0.35, -0.15, 0.15]), quat_down(0.0), 60)
    # (1b) 開いた左指の指先を立方体の真上に置いて押し下げる（指と物の接触テスト）
    set_grip(0.0); step(30); lf0, _ = SingleXFormPrim('/World/xarm7/left_pad').get_world_pose(); tcp0, _ = SingleXFormPrim('/World/xarm7/link_tcp').get_world_pose(); dxy = (lf0 - tcp0)[:2]
    tgt = np.array([cp[0] - dxy[0], cp[1] - dxy[1], 0.12]); ik_to(tgt, quat_down(0.0), 80); ik_to(np.array([tgt[0], tgt[1], finger_bottom_offset() + 0.040]), quat_down(0.0), 60)
    ik_to(np.array([tgt[0], tgt[1], finger_bottom_offset() + 0.010]), quat_down(0.0), 80); fi('press fingertip on cube top (cube top 25mm; finger_bottom should stop ~25mm)'); cpp, _ = cube.get_world_pose(); print('[sim] cube after press:', np.round(cpp, 4).tolist(), flush=True)
    ik_to(tgt, quat_down(0.0), 60)
    # (2) 立方体を掴む
    set_grip(0.0); step(30); ik_to(cp + np.array([0, 0, 0.15]), quat_down(0.0), 80); gz = 0.001; ik_to(np.array([cp[0], cp[1], gz]), quat_down(0.0), 80); fi('descend')
    set_grip(GRIP_CLOSED)
    for _ in range(40):
        o_prev = grip_opening(); step(5)
        if abs(grip_opening() - o_prev) < 0.0005: break
    def zr(path, toks=(UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy, UsdGeom.Tokens.guide)):
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), list(toks)); r_ = cache.ComputeWorldBound(stage.GetPrimAtPath(path)).ComputeAlignedRange(); return (round(float(r_.GetMin()[2])*1000,1), round(float(r_.GetMax()[2])*1000,1), round(float(r_.GetMin()[1])*1000,1), round(float(r_.GetMax()[1])*1000,1))
    print('[sim] z/y ranges(mm) at closed: left pad', zr('/World/xarm7/left_pad'), '| right pad', zr('/World/xarm7/right_pad'), '| cube', zr('/World/probe_cube'), flush=True)
    fi('closed'); save_views('cube_closed')
    cp2, _ = cube.get_world_pose(); print('[sim] cube moved during close:', round(float(np.linalg.norm(cp2 - cp)) * 1000, 1), 'mm', flush=True)
    tcpq = SingleXFormPrim('/World/xarm7/link_tcp').get_world_pose()[1]; Rt = R.from_quat([tcpq[1], tcpq[2], tcpq[3], tcpq[0]]).as_matrix()
    for axname, ax in (('tcp-x', Rt[:, 0]), ('tcp-y', Rt[:, 1])):
        for sgn in (+1, -1):
            q_ = cp2 + sgn * 0.0155 * ax; q_[2] = cp2[2] + 0.005
            print(f'[sim] overlap 3mm outside cube face ({axname}{"+" if sgn>0 else "-"}):', [h.split(' | ')[0].split('/')[-1] for h in overlap_at(q_, 0.004)], flush=True)
    print('[sim] overlap at cube center:', [h.split(' | ')[0].split('/')[-1] for h in overlap_at(cp2, 0.012)], flush=True)
    lf2, _ = SingleXFormPrim('/World/xarm7/left_pad').get_world_pose(); rf2, _ = SingleXFormPrim('/World/xarm7/right_pad').get_world_pose()
    print('[sim] finger origins:', np.round(lf2, 3).tolist(), np.round(rf2, 3).tolist(), '| cube', np.round(cp2, 3).tolist(), flush=True)
    for f in ('left_pad', 'right_pad'):
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy, UsdGeom.Tokens.guide]); r_ = cache.ComputeWorldBound(stage.GetPrimAtPath(f'/World/xarm7/{f}')).ComputeAlignedRange()
        print(f'[sim] {f} collision bbox x[{r_.GetMin()[0]:.3f},{r_.GetMax()[0]:.3f}] y[{r_.GetMin()[1]:.3f},{r_.GetMax()[1]:.3f}] z[{r_.GetMin()[2]:.3f},{r_.GetMax()[2]:.3f}]', flush=True)
    ik_to(cp + np.array([0, 0, 0.15]), quat_down(0.0), 70); fi('lift'); cz = float(cube.get_world_pose()[0][2]); print('[sim] cube z after lift:', round(cz, 4), '→', '掴めた' if cz > 0.05 else '空振り', flush=True); save_views('cube_lift')
if a.stage == 'grasp':
    det = perceive(); det = [z for z in det if (a.target_type is None or z['type'] == a.target_type)]; x = sorted(det, key=lambda z: -z['pixels'])[0]
    print('[sim] grasp test on', x['type'], 'pos', x['pos'].round(3).tolist(), 'yaw', round(x['yaw'], 2), 'top_z', round(x['top_z'], 4), 'len/width', round(x['length'], 3), round(x['width'], 3), flush=True)
    _item = min(items, key=lambda it: np.linalg.norm(it['rigid'].get_world_pose()[0][:2] - x['pos'][:2]))
    _p, _q = _item['rigid'].get_world_pose(); _Rm = R.from_quat([_q[1], _q[2], _q[3], _q[0]]).as_matrix()
    # ローカル bbox の長辺方向をワールドへ
    _cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render]); _lb = _cache.ComputeLocalBound(stage.GetPrimAtPath(_item['path'])).ComputeAlignedRange(); _ext = np.array(_lb.GetMax()) - np.array(_lb.GetMin())
    _ax_local = np.eye(3)[int(np.argmax(_ext[:2]))]; _ax_w = _Rm @ _ax_local; true_yaw = math.atan2(_ax_w[1], _ax_w[0])
    def _angdiff(a1, a2): d = abs((a1 - a2 + math.pi/2) % math.pi - math.pi/2); return math.degrees(d)
    print(f"[sim] TRUE long-axis yaw={true_yaw:.2f} rad | perceived yaw={x['yaw']:.2f} | diff={_angdiff(true_yaw, x['yaw']):.0f} deg (0 なら知覚は正しい)", flush=True)
    yaw = x['yaw']   # 指の閉じる軸は TCP の x 軸向き。器具の長手に直角に閉じるには yaw をそのまま使う（+90° だと平行になる、実測）
    def finger_info(tag):
        q = robot.get_joint_positions(); lf, _ = SingleXFormPrim('/World/xarm7/left_pad').get_world_pose(); rf, _ = SingleXFormPrim('/World/xarm7/right_pad').get_world_pose()
        tcp, _ = SingleXFormPrim('/World/xarm7/link_tcp').get_world_pose()
        v = (lf - rf)[:2]; ax = np.array([math.cos(x['yaw']), math.sin(x['yaw'])]); ang = math.degrees(math.acos(min(1.0, abs(float(np.dot(v, ax))) / (np.linalg.norm(v) + 1e-9))))
        tag = f"{tag} | finger_bottom_z={(tcp[2]-finger_bottom_offset())*1000:.1f}mm"
        axt = np.array([math.cos(true_yaw), math.sin(true_yaw)]); angt = math.degrees(math.acos(min(1.0, abs(float(np.dot(v, axt))) / (np.linalg.norm(v) + 1e-9)))); tag = f"{tag} | close-axis vs TRUE axis={angt:.0f}deg" 
        print(f"[sim] {tag}: tcp={np.round(tcp,3).tolist()} opening={grip_opening()*1000:.1f}mm pad_gap={np.linalg.norm(lf-rf)*1000:.0f}mm close-axis vs item-axis={ang:.0f}deg (90 が正しい)", flush=True)
    set_grip(0.0); step(20); finger_info('open')
    pre = x['pos'] + np.array([0, 0, 0.15]); ik_to(pre, quat_down(yaw), 80); finger_info('pre'); save_views('10_pre')
    gz = 0.001; print(f'[sim] grasp tcp z={gz*1000:.1f}mm (item top {x["top_z"]*1000:.1f}mm; パッド先端 = TCP 面。器具の全高を挟むため机すれすれ)', flush=True)
    grasp = np.array([x['pos'][0], x['pos'][1], gz]); ok = ik_to(grasp, quat_down(yaw), 80); finger_info(f'descend ok={ok} gz={gz:.4f}'); save_views('11_descend')
    set_grip(GRIP_CLOSED)
    for _ in range(40):
        o_prev = grip_opening(); step(5)
        if abs(grip_opening() - o_prev) < 0.0005: break
    finger_info('closed'); save_views('12_closed')
    def wr(path):
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy, UsdGeom.Tokens.guide]); r_ = cache.ComputeWorldBound(stage.GetPrimAtPath(path)).ComputeAlignedRange()
        return [round(float(v)*1000) for v in r_.GetMin()], [round(float(v)*1000) for v in r_.GetMax()]
    _it = min(items, key=lambda it: np.linalg.norm(it['rigid'].get_world_pose()[0][:2] - x['pos'][:2]))
    print('[sim] world bbox(mm) at closed: item collider', wr(_it['path'] + '/collider') if stage.GetPrimAtPath(_it['path'] + '/collider').IsValid() else '-', '| item visual', wr(_it['path'] + '/model'), '| left pad', wr('/World/xarm7/left_pad'), '| right pad', wr('/World/xarm7/right_pad'), flush=True)
    if a.grasp_mode == 'attach':
        got = attach_nearest(); print('[sim] attach:', None if got is None else got['name'], flush=True); step(5)
    _it0 = min(items, key=lambda it: np.linalg.norm(it['rigid'].get_world_pose()[0][:2] - x['pos'][:2]))
    for dz in (0.01, 0.02, 0.04, 0.08):   # 少しずつ持ち上げて、どこで滑るかを見る
        ik_to(np.array([grasp[0], grasp[1], gz + dz]), quat_down(yaw), 40)
        print(f'[sim]   lift +{dz*100:.0f}cm: item z={_it0["rigid"].get_world_pose()[0][2]*1000:.1f}mm opening={grip_opening()*1000:.1f}mm', flush=True)
    ik_to(pre, quat_down(yaw), 70); step(int(0.4 / a.physics_dt)); finger_info('lift')
    item = min(items, key=lambda it: np.linalg.norm(it['rigid'].get_world_pose()[0][:2] - x['pos'][:2]))
    lifted_z = float(item['rigid'].get_world_pose()[0][2]); print('[sim] item z after lift:', round(lifted_z, 4), '→', '掴めた' if lifted_z > 0.05 else '空振り', flush=True); save_views('13_lift')
    if lifted_z > 0.05:
        b = BIN_POS[x['type']]; CARRY = yaw   # 掴んだ向きのまま運ぶ（手首を回すと細い器具が指の間でねじれて落ちた）
        ik_to(b + np.array([0, 0, 0.20]), quat_down(CARRY), 90, speed=0.15); ik_to(b + np.array([0, 0, 0.10]), quat_down(CARRY), 60, speed=0.15); detach(); set_grip(0.0); step(60); ik_to(b + np.array([0, 0, 0.20]), quat_down(CARRY), 50); step(30)
        pf = item['rigid'].get_world_pose()[0]; in_bin = bool(abs(pf[0] - b[0]) < BIN_W/2 + 0.02 and abs(pf[1] - b[1]) < BIN_L/2 + 0.02 and pf[2] < 0.08)
        print('[sim] placed at', np.round(pf, 3).tolist(), '→', '箱に入った' if in_bin else '箱の外', flush=True); save_views('14_placed')
    set_arm(HOME); step(40)
    _f = close.get_rgba(); print('[sim] frames recorded:', VID['n'], '| close.get_rgba shape:', None if _f is None else getattr(_f, 'shape', type(_f)), flush=True)
    vid_close()
if a.stage == 'full':
    results = []; placed = {t: 0 for t in TYPES}; attempts = 0
    GRASP_Z_OFF = 0.02   # 器具の上面から TCP（指先）をどれだけ下げるか（薄い器具は机すれすれ 2mm で止める）
    failed_at = []   # 失敗した場所 (x, y)。同じ場所を何度も掴みに行かない（2026-10-17 夜の 12 個版は同じ塊に 36 回行って終わった）
    while attempts < 3 * len(items):
        go_view(); det = perceive()
        det = [x for x in det if abs(x['pos'][1]) < a.spread + 0.08]   # 箱の中のものは対象外
        det = [x for x in det if sum(float(np.linalg.norm(np.asarray(f) - np.asarray(x['pos'][:2]))) < 0.04 for f in failed_at) < 2]   # 同じ場所（4cm 以内）は 2 回まで
        if not det: break
        x = sorted(det, key=lambda z: -z['pixels'])[0]; attempts += 1
        yaw = (x['yaw'] + math.pi / 2) % math.pi - math.pi / 2   # 指は対称なので ±90° に収める（手首の回転を減らす）。指の閉じる軸は TCP の x 軸向き。器具の長手に直角に閉じるには yaw をそのまま使う（+90° だと平行になる、実測）   # 指を長手方向に対して直角に
        pre = x['pos'] + np.array([0, 0, 0.15]); gz = 0.001; grasp = np.array([x['pos'][0], x['pos'][1], gz])
        set_grip(0.0); step(10)
        ok1 = ik_to(pre, quat_down(yaw), 80); ok2 = ik_to(grasp, quat_down(yaw), 80)
        set_grip(GRIP_CLOSED); step(60)
        if a.grasp_mode == 'attach': attach_nearest(); step(5)
        ik_to(pre, quat_down(yaw), 70)
        held = grasp_closed_enough()
        # 持ち上がったか: その器具の真値の高さで判定（検証用。実機では把持センサ）
        item = min(items, key=lambda it: np.linalg.norm(it['rigid'].get_world_pose()[0][:2] - x['pos'][:2]))
        lifted = bool(item['rigid'].get_world_pose()[0][2] > 0.06)
        b = BIN_POS[x['type']]; over = b + np.array([0, 0, 0.20]); drop = b + np.array([0, 0, 0.10])
        ik_to(over, quat_down(yaw), 90, speed=0.15); ik_to(drop, quat_down(yaw), 60, speed=0.15); detach(); set_grip(0.0); step(40); ik_to(over, quat_down(yaw), 50)
        step(30)
        p_final = item['rigid'].get_world_pose()[0]
        def _in(bpos): return bool(abs(p_final[0] - bpos[0]) < BIN_W/2 + 0.02 and abs(p_final[1] - bpos[1]) < BIN_L/2 + 0.02 and p_final[2] < 0.08)
        in_some_bin = _in(b); correct_bin = _in(BIN_POS[item['type']])   # 採点: 本当の種類の箱に入ったか（知覚の種類ではなく）
        results.append({'attempt': attempts, 'perceived_type': x['type'], 'true_type': item['type'], 'item': item['name'], 'ik_ok': bool(ok1 and ok2), 'lifted': lifted, 'in_bin': in_some_bin, 'correct_bin': correct_bin, 'final': p_final.round(3).tolist()})
        print(f"[sim] #{attempts} 知覚={TYPE_JP[x['type']]} 実際={TYPE_JP[item['type']]} {item['name']}: lifted={lifted} 箱に入った={in_some_bin} 正しい箱={correct_bin}", flush=True)
        if correct_bin: placed[item['type']] += 1
        if not in_some_bin: failed_at.append((float(x['pos'][0]), float(x['pos'][1])))
    set_arm(HOME); step(60)
    report['results'] = results; report['placed'] = placed; report['items_total'] = len(items)
    report['success'] = sum(r['correct_bin'] for r in results); report['in_any_bin'] = sum(r['in_bin'] for r in results)
    save_views('99_end')
    vid_close()
if a.stage == 'serve':
    # ---- 器械出し: 命令（LLM の出力）を 1 つずつ実行する。知覚は検出モデル＋深度（正解は使わない）、掴みは物理。
    #      シミュレータの正解（どの器具がどこにあるか）は「採点」にだけ使う（results の true_type / in_handoff など）。
    JP = {'scissors': 'ハサミ', 'forceps': 'ピンセット', 'scalpel': 'メス', 'syringe': '注射器'}
    W, Y, C = (255, 255, 255), (255, 220, 90), (120, 230, 255)
    def say(doctor, cmd, robot_line): CAP['lines'] = [(f'医師: {doctor}', W), (f'LLM: {cmd}', Y), (f'ロボット: {robot_line}', C)]
    def wait(sec): step(int(sec / a.physics_dt))
    def truth_near(p):
        it = min(items, key=lambda it: np.linalg.norm(it['rigid'].get_world_pose()[0][:2] - p[:2])); return it
    def where(it):
        p = it['rigid'].get_world_pose()[0]
        for nm, c in (('handoff', HANDOFF_POS), ('tray', TRAY_POS)):
            if abs(p[0] - c[0]) < MAT_W / 2 + 0.02 and abs(p[1] - c[1]) < MAT_L / 2 + 0.02 and p[2] < 0.06: return nm
        return 'table' if p[2] < 0.05 else 'air'
    def pick_and_place(x, floor_z, dest):
        """x = 知覚した器具。掴んで dest（箱の中心）へ。present なら置く前に医師側で止めて見せる。返り値: 掴めたか（指が閉じ切っていないか）"""
        yaw = (x['yaw'] + math.pi / 2) % math.pi - math.pi / 2
        pre = x['pos'] + np.array([0, 0, 0.15]); pre[2] = floor_z + 0.15; grasp = np.array([x['pos'][0], x['pos'][1], floor_z + 0.001])
        set_grip(0.0); step(10); ik_to(pre, quat_down(yaw), 80); ik_to(grasp, quat_down(yaw), 80)
        set_grip(GRIP_CLOSED); step(60); ik_to(pre, quat_down(yaw), 70)
        held = grasp_closed_enough()
        if not held: return False
        HOLD['label'] = JPN.get(x['type'], x['type'])   # ロボットが検出で判断した種類
        over = dest + np.array([0, 0, 0.20])
        ik_to(over, quat_down(yaw), 90, speed=0.15)
        if a.handoff == 'present' and dest[1] < 0:   # 手渡し台（医師側 = -y）: 医師の手の前まで差し出して止まり、それから台に置く
            ik_to(PRESENT_P, quat_down(yaw), 80, speed=0.15); wait(1.5); ik_to(over, quat_down(yaw), 60, speed=0.15)   # 医師側で一度止めて差し出す（持ち手の向きは今後）
        ik_to(dest + np.array([0, 0, 0.04]), quat_down(yaw), 60, speed=0.15); set_grip(0.0); HOLD['label'] = None; step(40); ik_to(over, quat_down(yaw), 50); step(20)
        return True
    SLOT_D = [(-0.08, -0.05), (0.08, 0.05), (-0.08, 0.05), (0.08, -0.05)]   # 台の上の置き場所 (dx, dy)。重ねて置くと上から 1 つに見えて数え間違えた（10/31）
    SLOTS = {'handoff': {}, 'tray': {}}               # 箱の名前 → {置き場所の番号: 器具の種類}（ロボットが自分で置いた記録）
    def slot_pos(box, kind):
        used = SLOTS[box]; k = next(i for i in range(len(SLOT_D)) if i not in used); used[k] = kind
        c = HANDOFF_POS if box == 'handoff' else TRAY_POS; return c + np.array([SLOT_D[k][0], SLOT_D[k][1], 0.0])
    def free_slot(box, kind):
        for k, v in list(SLOTS[box].items()):
            if v == kind: del SLOTS[box][k]; return
    def fetch(kind, region, floor_z, dest, tries=2):
        """region から kind の器具を探して dest へ。返り値: (結果の文字列, 動かした器具の真値 or None)"""
        for t in range(tries):
            REGION['name'] = region; go_view(); seen = perceive(); det = [z for z in seen if z['type'] == kind]
            if not det and region == 'handoff' and len(seen) == 1:
                # 手渡し台はロボット自身が置いた場所。器具が 1 つだけ見えていれば、検出の種類が違っても（箱の色のくせで誤検出しやすい）それを戻す
                print(f"[sim]   手渡し台に 1 つだけ（検出は {seen[0]['type']}）→ 自分が置いた器具として戻す", flush=True); det = seen
            if not det: return ('見つからない', None)
            x = sorted(det, key=lambda z: -z.get('conf', 0))[0]; it = truth_near(x['pos'])
            if pick_and_place(x, floor_z, dest): return ('ok', it)
            print(f'[sim]   掴めなかった（{t + 1} 回目）→ 見直して再挑戦', flush=True)
        return ('掴めない', it)
    cmds = [json.loads(l) for l in open(a.commands, encoding='utf-8') if l.strip()]
    results = []; set_arm(HOME); step(30)
    for n, c in enumerate(cmds, 1):
        act, inst = c.get('action'), c.get('instrument'); cmd_txt = json.dumps({k: v for k, v in c.items() if k in ('action', 'instrument', 'candidates')}, ensure_ascii=False)
        doc = c.get('say', ''); res = {'n': n, 'say': doc, 'cmd': c}
        print(f'[sim] 命令 {n}: {doc} → {cmd_txt}', flush=True)
        if act == 'hand' and inst in JP:
            say(doc, cmd_txt, f'{JP[inst]}を探して渡します'); wait(1.0)
            r, it = fetch(inst, 'table', 0.0, slot_pos('handoff', inst))
            if r != 'ok': free_slot('handoff', inst)
            res.update({'robot': r, 'moved': it and it['type'], 'moved_to': it and where(it), 'ok': bool(it and it['type'] == inst and where(it) == 'handoff')})
            say(doc, cmd_txt, f'{JP[inst]}をどうぞ' if res['ok'] else f'{JP[inst]}を渡せませんでした（{r}）'); wait(2.0)
        elif act == 'return' and inst in JP:
            say(doc, cmd_txt, f'{JP[inst]}をトレーに戻します'); wait(1.0)
            r, it = fetch(inst, 'handoff', MAT_T, slot_pos('tray', inst))
            if r == 'ok': free_slot('handoff', inst)
            else: free_slot('tray', inst)
            res.update({'robot': r, 'moved': it and it['type'], 'moved_to': it and where(it), 'ok': bool(it and it['type'] == inst and where(it) == 'tray')})
            say(doc, cmd_txt, f'{JP[inst]}を戻しました' if res['ok'] else f'戻せませんでした（{r}）'); wait(2.0)
        elif act == 'clarify':
            say(doc, cmd_txt, f"{'と'.join(JP[i] for i in c.get('candidates', []) if i in JP)}、どちらですか？"); res['ok'] = True; wait(3.0)
        elif act == 'unavailable':
            say(doc, cmd_txt, (f'{JP[inst]}はトレーにありません' if inst in JP else 'その器具はトレーにありません')); res['ok'] = True; wait(3.0)
        elif act == 'refuse':
            say(doc, cmd_txt, '危ないので、その指示はできません'); res['ok'] = True; wait(3.0)
        elif act == 'count':   # 器具カウント: 机・手渡し台・トレーにある器具を数える（検出モデル）
            # 数え漏れ（体内遺残）を防ぐのが目的なので、見つけた器具の「個数」で数える（種類の見分けは別。メスとピンセットは見間違えることがある）
            if a.camera == 'wrist':
                det = []
                for r_ in ('table', 'handoff', 'tray'):
                    REGION['name'] = r_; go_view()
                    for z_ in perceive():
                        # 同じ器具を別の場所から二重に数えない: 8 cm 以内は同じ器具。台のふちにまたがった器具は見る場所で位置がずれるので、同じ種類なら 15 cm 以内も同じ器具とみなす
                        dup = lambda o: (np.linalg.norm(np.asarray(z_['pos'][:2]) - np.asarray(o['pos'][:2])) < (0.15 if o['type'] == z_['type'] else 0.08))
                        if not any(dup(o) for o in det): det.append(z_)
            else:
                REGION['name'] = 'all'; go_view(); det = perceive()
            res.update({'counted': [z['type'] for z in det], 'ok': len(det) == len(items)})
            say(doc, cmd_txt, f'器具カウント {len(det)}/{len(items)}' + ('、そろっています' if res['ok'] else f'、{len(items) - len(det)} 個足りません')); wait(3.0)
        else:
            res['ok'] = False; say(doc, cmd_txt, '命令を読めませんでした'); wait(2.0)
        res['where_now'] = {it['type']: where(it) for it in items}; results.append(res)
        print(f"[sim]   結果: {res.get('robot', '')} ok={res['ok']} 器具の場所={res['where_now']}", flush=True)
    CAP['lines'] = []; set_arm(HOME); step(60); save_views('99_end')
    report['results'] = results; report['ok'] = sum(r['ok'] for r in results); report['n'] = len(results); vid_close()
if a.stage == 'dataset':
    # ---- 検出モデルの学習画像: 同じ場面で器具の置き方・腕の姿勢を変えて、ロボットのカメラ（固定 or 手首）の RGB と
    #      シミュレータの正解の枠（YOLO 形式）を保存する。正解の枠は器具の semantic ラベル（種類）から自動で付ける。
    #      10/31 作成者: 以前の確認で直すところが 0 件だったので自動アノテーションで学習してよい
    import cv2
    cam.add_bounding_box_2d_tight_to_frame()
    for sp in ('train', 'val'): os.makedirs(f'{OUT}/images/{sp}', exist_ok=True); os.makedirs(f'{OUT}/labels/{sp}', exist_ok=True)
    CID = {t: i for i, t in enumerate(TYPES)}; n_box = 0
    def rand_place():
        """器具を置き直す: 机の上（多め）／手渡し台／返却トレーに、重ならないように。向きはランダム"""
        placed = []
        for it in items:
            for _t in range(200):
                where_ = rng.choice(['table', 'table', 'table', 'handoff', 'tray'])
                if where_ == 'table': x, y, z = rng.uniform(0.28, 0.66), rng.uniform(-a.spread, a.spread), 0.012
                else:
                    c = HANDOFF_POS if where_ == 'handoff' else TRAY_POS; x, y, z = c[0] + rng.uniform(-MAT_W / 2 + 0.06, MAT_W / 2 - 0.06), c[1] + rng.uniform(-MAT_L / 2 + 0.04, MAT_L / 2 - 0.04), MAT_T + 0.012
                if all(np.hypot(x - px, y - py) > 0.09 for px, py in placed): break
            placed.append((x, y))
            it['rigid'].set_world_pose(position=np.array([x, y, z]), orientation=rot_utils.euler_angles_to_quats(np.array([0, 0, rng.uniform(-math.pi, math.pi)])))
            it['rigid'].set_linear_velocity(np.zeros(3)); it['rigid'].set_angular_velocity(np.zeros(3))
    def rand_arm():
        """腕の姿勢: 固定カメラなら よけた姿勢・構え・作業台の上のいろいろな所（腕が写り込む画像も入れる）。手首カメラなら 3 か所の見る姿勢を少しずつずらす"""
        if a.camera == 'wrist':
            base = [SCAN_P, np.array([0.42, -0.33, 0.40]), np.array([0.42, 0.33, 0.40])][int(rng.integers(0, 3))]
            tgt = base + np.array([rng.uniform(-0.04, 0.04), rng.uniform(-0.04, 0.04), rng.uniform(-0.04, 0.05)]); yaw = rng.uniform(-0.35, 0.35)
        else:
            k = rng.random()
            if k < 0.35: tgt, yaw = PARK_P, 0.0
            else: tgt, yaw = np.array([rng.uniform(0.25, 0.65), rng.uniform(-0.40, 0.40), rng.uniform(0.12, 0.40)]), rng.uniform(-1.5, 1.5)
        q, ok = lula.compute_inverse_kinematics('link_tcp', tgt, quat_down(yaw), warm_start=HOME)
        q = np.asarray(q) if ok else (np.asarray(_qp) if _okp else HOME)
        robot.set_joint_positions(np.concatenate([q, np.zeros(2)]).astype(np.float32)); set_arm(q); set_grip(float(rng.choice([0.0, GRIP_CLOSED * 0.5])))
    for k in range(a.n):
        rand_place(); rand_arm(); step(45, record=False)
        rgb = np.asarray(cam.get_rgba())[:, :, :3].astype(np.uint8); fr = cam.get_current_frame(); bb = fr.get('bounding_box_2d_tight')
        if bb is None: print('[sim] bbox が取れない'); break
        data, info = bb['data'], bb['info']; id2 = {int(i_): (v['class'] if isinstance(v, dict) else str(v)) for i_, v in info['idToLabels'].items()}
        Hh, Ww = rgb.shape[:2]; lines = []
        for row in data:
            lab = id2.get(int(row['semanticId']), '')
            if lab not in CID: continue
            x0, y0, x1, y1 = float(row['x_min']), float(row['y_min']), float(row['x_max']), float(row['y_max'])
            if (x1 - x0) < 4 or (y1 - y0) < 4 or float(row['occlusionRatio']) > 0.6: continue   # 小さすぎ・ほとんど隠れているものは外す
            lines.append(f"{CID[lab]} {(x0 + x1) / 2 / Ww:.6f} {(y0 + y1) / 2 / Hh:.6f} {(x1 - x0) / Ww:.6f} {(y1 - y0) / Hh:.6f}")
        sp = 'val' if k % a.val_every == 0 else 'train'; nm = f'{a.camera}_{a.seed}_{k:04d}'
        cv2.imwrite(f'{OUT}/images/{sp}/{nm}.png', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)); open(f'{OUT}/labels/{sp}/{nm}.txt', 'w').write('\n'.join(lines) + ('\n' if lines else ''))
        n_box += len(lines)
        if k % 20 == 0: print(f'[sim] dataset {k}/{a.n} boxes {len(lines)}', flush=True)
    open(f'{OUT}/data.yaml', 'w').write(f"path: {OUT}\ntrain: images/train\nval: images/val\nnc: 4\nnames: {TYPES}\n")
    report['dataset'] = {'n': a.n, 'boxes': n_box, 'camera': a.camera}
report['elapsed_s'] = round(time.time() - t_start, 1)
json.dump(report, open(f'{OUT}/report.json', 'w'), ensure_ascii=False, indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
print('[sim] report ->', f'{OUT}/report.json', flush=True)
app.close(); print('SIM_DONE', flush=True)
