"""E1 quasistatic lift path from the VERIFIED vise (cleanroom).

Start: the frame-0 static solve of checkpoint 1aec5c14... on the base XML -
the replay must reproduce N 9.72/6.53 N (the round-3 verifier values) before
anything else counts.

Three control laws, each 120 frames @ 20 Hz (6 s):
  (a) pure wrist-z quasistatic climb (0.1 mm/frame), fingers position-held
  (b) wrist-z climb + curl press hold (N target locked in 9-12 N, servoed)
  (c) (b) + micro-tilt: wrist-rx ramped ~10 deg over the climb (pinch geometry)

Per-frame log: box up_z, lift height, per-hand N, max penetration, slip, yaw.
Gate: peak > 5 mm AND window >= 20 frames >= 5 mm AND penetration <= 0.5 mm
      AND >= 2 pad-contacts maintained.

Usage: runs/.venv-mj/bin/python runs/w6c/e1_quasistatic.py
"""
import hashlib
import json
import pathlib
import sys

import numpy as np
import mujoco

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import grounding.train_w6c as W  # noqa: E402

BASE_XML = ROOT / 'grounding/assets/allegro_floating.xml'
ENGAGE = ROOT / 'runs/w6c/v21_engage_checkpoint.npz'
CTRL_HZ = 20
FRAMES = 120
Z_RATE = 1e-4          # 0.1 mm/frame on the actuator setpoint
N_TGT = 10.5           # mid of the locked band 9-12
TILT_RAD = 0.175       # ~10 deg

FINGER_ACTS = {7: 'ffj1', 8: 'ffj2', 11: 'mfj1', 12: 'mfj2',
               29: 'ffj1_left', 30: 'ffj2_left', 33: 'mfj1_left',
               34: 'mfj2_left', 37: 'rfj1_left', 38: 'rfj2_left'}
BASE_DEM = {7: 0.10, 8: 0.10, 11: 0.10, 12: 0.10, 29: 0.10, 30: 0.10,
            33: 0.10, 34: 0.10, 37: 0.30, 38: 0.30}


def setup():
    m = mujoco.MjModel.from_xml_path(str(BASE_XML))
    z = np.load(ENGAGE)
    d = mujoco.MjData(m)
    d.qpos[:] = z['qpos']
    d.ctrl[:] = z['ctrl']
    mujoco.mj_forward(m, d)
    return m, d, z


def measure(m, d, bid, slip0):
    fake = type('E', (), {'model': m, 'data': d})()
    fake.obj_geom = int(m.geom('object_geom').id)
    n_r, n_l = W.pad_normal_forces(fake)
    pen = 0.0
    n_padc = 0
    for ci in range(d.ncon):
        c = d.contact[ci]
        g1, g2 = int(c.geom1), int(c.geom2)
        b1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g1])) or ''
        b2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g2])) or ''
        if (b1.startswith('object')) != (b2.startswith('object')):
            pen = min(pen, float(c.dist))
            hand = b2 if b1.startswith('object') else b1
            if any(k in hand for k in ('pad', 'distal', 'medial', 'tip')):
                n_padc += 1
    return {
        'n_r': round(n_r, 3), 'n_l': round(n_l, 3),
        'clear_mm': round((float(d.xpos[bid][2]) - 0.04 - 0.6999) * 1e3, 3),
        'pen_mm': round(-pen * 1e3, 3),
        'pad_contacts': n_padc,
        'slip_mm': round(float(np.linalg.norm(d.xpos[bid][:2] - slip0)) * 1e3, 2),
        'yaw_deg': round(abs(float(2 * np.arctan2(
            d.xquat[bid][3], d.xquat[bid][0])) * 57.2958), 2),
    }


def run_law(m, d, law: str):
    bid = int(m.body('object').id)
    jwz = [int(m.jnt_qposadr[m.joint('wrist_z').id]),
           int(m.jnt_qposadr[m.joint('wrist_z_left').id])]
    jwrx = int(m.jnt_qposadr[m.joint('wrist_rx').id])
    jwrx_l = int(m.jnt_qposadr[m.joint('wrist_rx_left').id])
    wz0 = [float(d.qpos[jwz[0]]), float(d.qpos[jwz[1]])]
    jadr = {i: int(m.jnt_qposadr[m.joint(jn).id]) for i, jn in FINGER_ACTS.items()}
    lo_c, hi_c = m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1]
    sub = max(1, int(round((1.0 / CTRL_HZ) / m.opt.timestep)))
    dem = {i: 0.0 for i in FINGER_ACTS}
    slip0 = d.xpos[bid][:2].copy()
    frames = []
    start = measure(m, d, bid, slip0)
    for f in range(FRAMES):
        d.ctrl[2] += Z_RATE
        d.ctrl[24] += Z_RATE
        if law in ('b', 'c'):
            fake = type('E', (), {'model': m, 'data': d})()
            fake.obj_geom = int(m.geom('object_geom').id)
            n_r, n_l = W.pad_normal_forces(fake)
            for i, jn in FINGER_ACTS.items():
                n_side = n_r if i < 22 else n_l
                dem[i] = float(np.clip(
                    dem[i] + 0.02 * (N_TGT - n_side) / N_TGT, -0.2, 0.8))
                d.ctrl[i] = float(np.clip(
                    d.qpos[jadr[jn]] + BASE_DEM[i] + dem[i], lo_c[i], hi_c[i]))
        if law == 'c':
            tilt = TILT_RAD * min(1.0, (f + 1) / FRAMES)
            d.ctrl[3] = float(d.qpos[jwrx]) + tilt
            d.ctrl[25] = float(d.qpos[jwrx_l]) - tilt
        for _ in range(sub):
            mujoco.mj_step(m, d)
        frames.append(measure(m, d, bid, slip0))
    return start, frames


def main() -> None:
    m, d, z = setup()
    fake = type('E', (), {'model': m, 'data': d})()
    fake.obj_geom = int(m.geom('object_geom').id)
    n0r, n0l = W.pad_normal_forces(fake)
    start_match = abs(n0r - 9.72) < 0.5 and abs(n0l - 6.53) < 0.5
    print(f'START replay: N=({n0r:.2f},{n0l:.2f}) '
          f'(verifier round-3 cited 9.72/6.53; match={start_match})', flush=True)
    if not start_match:
        print('FOUNDATION DISCREPANCY: the cited 9.72/6.53 is not reproducible '
              'in-repo (no artifact in runs/verify_lift contains it); running '
              'all laws from THIS repo-bit-reproducible start, results '
              'start-conditional.', flush=True)
    out = {'mujoco': mujoco.__version__,
           'hashes': {
               'base_xml': hashlib.sha256(
                   BASE_XML.read_bytes()).hexdigest()[:16],
               'engage': hashlib.sha256(
                   ENGAGE.read_bytes()).hexdigest()[:16]},
           'ctrl_hz': CTRL_HZ, 'frames': FRAMES,
           'z_rate_m_per_frame': Z_RATE, 'n_target': N_TGT,
           'tilt_rad': TILT_RAD,
           'start_replay': {'n': [round(n0r, 2), round(n0l, 2)],
                            'matches_verifier_citation': bool(start_match)},
           'laws': {}}
    for law in ('a', 'b', 'c'):
        m, d, z = setup()
        start, frames = run_law(m, d, law)
        clears = np.array([fr['clear_mm'] for fr in frames])
        above = clears >= 5.0
        best_run = 0
        run = 0
        for v in above:
            run = run + 1 if v else 0
            best_run = max(best_run, run)
        peak = float(clears.max())
        max_pen = max(fr['pen_mm'] for fr in frames)
        min_padc = min(fr['pad_contacts'] for fr in frames)
        gate = (peak > 5.0 and best_run >= 20
                and max_pen <= 0.5 and min_padc >= 2)
        out['laws'][law] = {
            'start': start,
            'peak_mm': round(peak, 2),
            'window_5mm_f': int(best_run),
            'max_pen_mm': round(max_pen, 3),
            'min_pad_contacts': int(min_padc),
            'final_slip_mm': frames[-1]['slip_mm'],
            'final_yaw_deg': frames[-1]['yaw_deg'],
            'gate_pass': bool(gate),
            'frames': frames[::10] + [frames[-1]],
        }
        print(f"LAW {law}: peak={peak:.2f}mm window={best_run} "
              f"pen={max_pen:.3f}mm padc>={min_padc} gate={'PASS' if gate else 'FAIL'}",
              flush=True)
    print(json.dumps(out, ensure_ascii=False))


if __name__ == '__main__':
    main()
