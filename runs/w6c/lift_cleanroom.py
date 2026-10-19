"""Cleanroom lift probe: the only numbers this script prints are reproducible
from the committed files by a fresh process. No in-memory state.

Lead b: per-pad WORLD-VERTICAL slide axes (R^T zhat from the verifier).
Lead a: active-N press recipe through the whole climb (wrist-x servo, N target
8-15 N, logged per frame).

Usage: runs/.venv-mj/bin/python runs/w6c/lift_cleanroom.py [--noser]
"""
import argparse
import json
import pathlib
import sys

import numpy as np
import mujoco

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import grounding.train_w6c as W  # noqa: E402

BASE_XML = ROOT / 'grounding/assets/allegro_floating.xml'
VAR_XML = ROOT / 'grounding/assets/allegro_floating_lifttest.xml'
ENGAGE = ROOT / 'runs/w6c/v21_engage_checkpoint.npz'

PADS = ['ff_pad', 'mf_pad', 'ff_pad_left', 'mf_pad_left']
# per-pad local axes whose world direction is vertical at the press pose
WORLDVERT = {
    'ff_pad': (0.985, 0.149, 0.091),
    'mf_pad': (0.863, -0.223, 0.452),
    'ff_pad_left': (0.982, 0.157, 0.106),
    'mf_pad_left': (0.837, -0.225, 0.498),
}


def build_variant(worldvert: bool) -> None:
    src = BASE_XML.read_text()
    for p in PADS:
        axis = '0 0 1'
        if worldvert:
            axis = ' '.join(f'{v:.3f}' for v in WORLDVERT[p])
        old = f'<body name="{p}" pos="0 0 0">'
        new = (f'<body name="{p}" pos="0 0 0">'
               f'<joint name="{p}_liftz" type="slide" axis="{axis}" '
               f'range="-0.08 0.08" damping="0.5"/>')
        assert old in src, p
        src = src.replace(old, new, 1)
    acts = '\n'.join(
        f'<position name="{p}_liftz_act" joint="{p}_liftz" kp="2000" '
        f'ctrlrange="-0.08 0.08"/>' for p in PADS)
    src = src.replace('</mujoco>', f'<actuator>{acts}</actuator>\n</mujoco>')
    VAR_XML.write_text(src)


def run(noser: bool) -> dict:
    import hashlib
    hashes = {
        'lifttest_xml': hashlib.sha256(VAR_XML.read_bytes()).hexdigest()[:16],
        'engage': hashlib.sha256(ENGAGE.read_bytes()).hexdigest()[:16],
    }
    m = mujoco.MjModel.from_xml_path(str(VAR_XML))
    eng = np.load(ENGAGE)
    d = mujoco.MjData(m)
    nq_old = eng['qpos'].shape[0]
    nu_old = eng['ctrl'].shape[0]
    d.qpos[:nq_old] = eng['qpos']
    d.ctrl[:nu_old] = eng['ctrl']
    mujoco.mj_forward(m, d)

    bid = int(m.body('object').id)
    half_z = 0.04
    z_rest = 0.6999
    pads = PADS
    jwz = [int(m.jnt_qposadr[m.joint('wrist_z').id]),
           int(m.jnt_qposadr[m.joint('wrist_z_left').id])]
    wz0 = [float(d.qpos[jwz[0]]), float(d.qpos[jwz[1]])]
    act = {p: int(m.actuator(f'{p}_liftz_act').id) for p in pads}

    N_TGT = 12.0
    KP_X = 2e-4
    X_CAP = 2e-3
    x_acc = [0.0, 0.0]
    slip0 = d.xpos[bid][:2].copy()
    peak = -9e9
    slip = 0.0
    yaw = 0.0
    clears = []
    nlog = []
    unstable = False
    for f in range(300):
        rate = 3.3e-5 if f < 30 else 1e-4
        d.ctrl[2] += rate
        d.ctrl[24] += rate
        if not noser:
            fake = type('E', (), {'model': m, 'data': d})()
            fake.obj_geom = int(m.geom('object_geom').id)
            n_r, n_l = W.pad_normal_forces(fake)
            # right: ctrl x increase presses harder; left: decrease presses
            x_acc[0] = float(np.clip(
                x_acc[0] + KP_X * (N_TGT - n_r), -X_CAP * 10, X_CAP * 10))
            x_acc[1] = float(np.clip(
                x_acc[1] - KP_X * (N_TGT - n_l), -X_CAP * 10, X_CAP * 10))
            d.ctrl[0] = float(eng['ctrl'][0]) + x_acc[0]
            d.ctrl[22] = float(eng['ctrl'][22]) + x_acc[1]
        for i, p in enumerate(pads):
            hand = 0 if i < 2 else 1
            d.ctrl[act[p]] = float(d.qpos[jwz[hand]] - wz0[hand])
        for _ in range(25):
            mujoco.mj_step(m, d)
        if not np.isfinite(d.qpos).all():
            unstable = True
            break
        c = d.xpos[bid][2] - half_z - z_rest
        clears.append(c)
        peak = max(peak, c)
        slip = max(slip, float(np.linalg.norm(d.xpos[bid][:2] - slip0)))
        yaw = max(yaw, abs(float(2 * np.arctan2(
            d.xquat[bid][3], d.xquat[bid][0])) * 57.2958))
        if f % 30 == 0 or f == 299:
            fake = type('E', (), {'model': m, 'data': d})()
            fake.obj_geom = int(m.geom('object_geom').id)
            n_r, n_l = W.pad_normal_forces(fake)
            nlog.append({'f': f, 'clear_mm': round(c * 1e3, 2),
                         'N': [round(n_r, 2), round(n_l, 2)]})
    clears_arr = np.array(clears)
    above = clears_arr >= 0.005
    best_run = 0
    run = 0
    for v in above:
        run = run + 1 if v else 0
        best_run = max(best_run, run)
    return {
        'worldvert_axes': True,
        'active_N_press': not noser,
        'N_target': None if noser else N_TGT,
        'unstable': unstable,
        'peak_mm': round(peak * 1e3, 2),
        'window_5mm_f': int(best_run),
        'slip_mm': round(slip * 1e3, 1),
        'yaw_deg': round(yaw, 1),
        'final_mm': round(float(clears_arr[-1]) * 1e3, 2) if len(clears_arr) else None,
        'n_log': nlog,
        'hashes': hashes,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--noser', action='store_true',
                    help='disable the active-N press (lead b only)')
    args = ap.parse_args()
    build_variant(worldvert=True)
    res = run(noser=args.noser)
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
