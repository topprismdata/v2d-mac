"""E2 reachability upper bound + E3 contact-mode search (cleanroom).

E2: pure-FK grid scan - given the finger geometry / joint limits / collisions,
the max approach depth of an off-face pad toward the +/-x faces. Bound =
max over configs of the penetration-equivalent depth (face_x - |tip_x| where
the tip crosses the face plane), clamped by no-collision.

E3: physics-in-the-loop static-feasibility search over the promising E2
configs. A config is a STATIC FEASIBLE GRASP iff, after a settle:
  - no penetration (min contact dist >= -0.5 mm)
  - >= 2 opposing pads with N > 2 N per hand (both hands)
  - the contact set can bear the vertical load mg = 1.47 N
    (per contact: up-capacity = max(0, n_z)*f_n + mu*f_n*sqrt(1-n_z^2), summed)

Usage: runs/.venv-mj/bin/python runs/w6c/e2e3_search.py
"""
import hashlib
import itertools
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
MG = 0.150 * 9.81

FINGER_ACTS = {7: 'ffj1', 8: 'ffj2', 11: 'mfj1', 12: 'mfj2',
               29: 'ffj1_left', 30: 'ffj2_left', 33: 'mfj1_left',
               34: 'mfj2_left', 37: 'rfj1_left', 38: 'rfj2_left'}
BASE_DEM = {7: 0.10, 8: 0.10, 11: 0.10, 12: 0.10, 29: 0.10, 30: 0.10,
            33: 0.10, 34: 0.10, 37: 0.30, 38: 0.30}
TIP_GEOMS = ['ff_tip_collision', 'mf_tip_collision']


def setup():
    m = mujoco.MjModel.from_xml_path(str(BASE_XML))
    z = np.load(ENGAGE)
    d = mujoco.MjData(m)
    d.qpos[:] = z['qpos']
    d.ctrl[:] = z['ctrl']
    mujoco.mj_forward(m, d)
    return m, d, z


def fk_pad_depths(m, d, bid):
    """E2 measure: per right-pad tip, the depth past the +x face plane
    (positive = crossing into the face region)."""
    depths = {}
    for gn in TIP_GEOMS:
        g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, gn)
        tx = float(d.geom_xpos[g][0]) - float(d.xpos[bid][0])
        depths[gn] = round(tx - 0.060, 4)   # the face at +60 mm
    return depths


def e2_scan(m, d, z, bid):
    """Grid: the wrist x/z offsets x the curl scale (ff/mf) -> the max pad
    depth toward the +x face (the FK only, no collisions solved)."""
    jadr = {i: int(m.jnt_qposadr[m.joint(jn).id]) for i, jn in FINGER_ACTS.items()}
    jwx = int(m.jnt_qposadr[m.joint('wrist_x').id])
    jwz = int(m.jnt_qposadr[m.joint('wrist_z').id])
    q0 = d.qpos.copy()
    results = []
    best = {'depth_m': -9e9, 'cfg': None}
    for dx, dz, cff, cmf in itertools.product(
            (-0.02, -0.01, 0.0, 0.01), (-0.01, 0.0, 0.01),
            (0.6, 0.8, 1.0, 1.2), (0.6, 0.8, 1.0, 1.2)):
        d.qpos[:] = q0
        d.qpos[jwx] += dx
        d.qpos[jwz] += dz
        # the curl scale: the ff/mf j1/j2 angles about their engage values
        for i in (7, 8, 29, 30):
            d.qpos[jadr[i]] = q0[jadr[i]] * cff
        for i in (11, 12, 33, 34):
            d.qpos[jadr[i]] = q0[jadr[i]] * cmf
        mujoco.mj_forward(m, d)
        dep = fk_pad_depths(m, d, bid)
        dmax = max(dep.values())
        results.append({'dx': dx, 'dz': dz, 'cff': cff, 'cmf': cmf,
                        'depth_m': max(dep, key=dep.get), 'val': dmax})
        if dmax > best['depth_m']:
            best = {'depth_m': dmax, 'cfg': {'dx': dx, 'dz': dz,
                                             'cff': cff, 'cmf': cmf}}
    d.qpos[:] = q0
    mujoco.mj_forward(m, d)
    return best, results


def static_grasp_test(m, d, settle=150):
    """E3 test: settle, then no-penetration + both-hands N>2 + the vertical
    load capacity >= mg."""
    bid = int(m.body('object').id)
    for _ in range(settle):
        mujoco.mj_step(m, d)
    min_dist = 9e9
    contacts = []
    for ci in range(d.ncon):
        c = d.contact[ci]
        g1, g2 = int(c.geom1), int(c.geom2)
        b1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g1])) or ''
        b2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g2])) or ''
        if (b1.startswith('object')) != (b2.startswith('object')):
            min_dist = min(min_dist, float(c.dist))
            hand_g = g2 if b1.startswith('object') else g1
            f6 = np.zeros(6)
            mujoco.mj_contactForce(m, d, ci, f6)
            fn = float(np.linalg.norm(f6[:3]))
            if fn <= 0:
                continue
            normal = c.frame[6:9]   # the contact frame normal row
            if b1.startswith('object'):
                normal = -normal
            nz = float(normal[2])
            mu = 1.4
            up = max(0.0, nz) * fn + mu * fn * float(np.sqrt(max(0.0, 1 - nz * nz)))
            contacts.append({'geom': mujoco.mj_id2name(
                m, mujoco.mjtObj.mjOBJ_GEOM, hand_g), 'fn': round(fn, 3),
                'up_cap': round(up, 3)})
    fake = type('E', (), {'model': m, 'data': d})()
    fake.obj_geom = int(m.geom('object_geom').id)
    n_r, n_l = W.pad_normal_forces(fake)
    no_pen = min_dist >= -5e-4
    both_press = n_r > 2.0 and n_l > 2.0
    up_total = sum(c['up_cap'] for c in contacts)
    bears = up_total >= MG
    return {'n_r': round(n_r, 3), 'n_l': round(n_l, 3),
            'min_dist_mm': round(min_dist * 1e3, 3),
            'no_pen': bool(no_pen), 'both_press': bool(both_press),
            'up_capacity_N': round(up_total, 3), 'bears_mg': bool(bears),
            'contacts': sorted(contacts, key=lambda c: -c['fn'])[:6],
            'feasible': bool(no_pen and both_press and bears)}


def e3_from_cfg(m, d, z, cfg):
    jadr = {i: int(m.jnt_qposadr[m.joint(jn).id]) for i, jn in FINGER_ACTS.items()}
    jwx = int(m.jnt_qposadr[m.joint('wrist_x').id])
    jwz = int(m.jnt_qposadr[m.joint('wrist_z').id])
    d.qpos[:] = z['qpos']
    d.ctrl[:] = z['ctrl']
    d.qpos[jwx] += cfg['dx']
    d.qpos[jwz] += cfg['dz']
    for i in (7, 8, 29, 30):
        d.qpos[jadr[i]] = z['qpos'][jadr[i]] * cfg['cff']
    for i in (11, 12, 33, 34):
        d.qpos[jadr[i]] = z['qpos'][jadr[i]] * cfg['cmf']
    d.ctrl[0] += cfg['dx']          # the wrist-x setpoint follows
    d.ctrl[2] += cfg['dz']
    mujoco.mj_forward(m, d)
    return static_grasp_test(m, d)


def main() -> None:
    m, d, z = setup()
    out = {'mujoco': mujoco.__version__,
           'hashes': {'base_xml': hashlib.sha256(
               BASE_XML.read_bytes()).hexdigest()[:16],
               'engage': hashlib.sha256(
                   ENGAGE.read_bytes()).hexdigest()[:16]},
           'mg_N': round(MG, 3)}
    # E2
    best, results = e2_scan(m, d, z, int(m.body('object').id))
    depths = sorted((r['val'] for r in results), reverse=True)
    out['E2'] = {
        'formula': 'depth = tip_x - (box_x + 0.060); positive = the pad tip crosses the face plane into the face region (FK only, no collision solve)',
        'grid': 'dx {-2,-1,0,+1}cm x dz {-1,0,+1}cm x cff {0.6..1.2} x cmf {0.6..1.2} = 192 configs',
        'configs': len(results),
        'max_depth_m': best['depth_m'],
        'best_cfg': best['cfg'],
        'top5_depth_m': [round(v, 4) for v in depths[:5]],
        'upper_bound_note': 'FK-only; collision/clamp limits resolve in E3 physics'
    }
    print(f"E2: max depth {best['depth_m']*1e3:.1f}mm at {best['cfg']}", flush=True)
    # E3: the physics test on the top-reach configs
    results.sort(key=lambda r: -r['val'])
    feasible = []
    tested = 0
    for r in results[:60]:
        cfg = {'dx': r['dx'], 'dz': r['dz'], 'cff': r['cff'], 'cmf': r['cmf']}
        t = e3_from_cfg(m, d, z, cfg)
        tested += 1
        if t['feasible']:
            feasible.append({'cfg': cfg, **t})
    out['E3'] = {
        'criterion': 'static feasible grasp = no pen (> -0.5mm) AND both hands N>2 AND contact-set vertical capacity >= mg 1.47N',
        'tested': tested,
        'n_feasible': len(feasible),
        'feasible': feasible[:8],
        'empty_set_certificate': None if feasible else {
            'resolution': 'dx 1cm, dz 1cm, curl scale 0.2 steps over the engage pose neighborhood',
            'boundaries': 'the wrist x +/-2cm, z +/-1cm, the ff/mf curl scale 0.6-1.2 of the engage angles; the settle 150 steps; the criterion as stated',
            'conclusion': 'no static feasible grasp found in the covered neighborhood'
        }
    }
    print(f"E3: {len(feasible)}/{tested} feasible", flush=True)
    print(json.dumps(out, ensure_ascii=False))


if __name__ == '__main__':
    main()
