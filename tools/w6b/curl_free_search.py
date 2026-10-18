"""r13.6: curl-free joint search — UR5e native verified grasp.

Joint parameterisation: UR5e 6R arm + 16 finger curl DoFs per hand.
Criterion: >=2 pads within 25mm of the face rect per hand.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import json

import numpy as np
import mujoco

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
m = mujoco.MjModel.from_xml_path(os.path.join(REPO, 'grounding/assets/allegro_floating_ur5e.xml'))
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)

obj = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'object')
half = m.geom_size[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, 'object_geom')]

def jid(n): return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
def gid(n): return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)

FINGERS = ('ff','mf','rf','th')
ARM = ('shoulder_pan','shoulder_lift','elbow','wrist_1','wrist_2','wrist_3')

arm_qadr = {s: [m.jnt_qposadr[jid(f'ur5e{sfx}_{n}_joint')] for n in ARM]
            for s, sfx in (('r','_r'), ('l','_l'))}
arm_rng = {s: [m.jnt_range[jid(f'ur5e{sfx}_{n}_joint')] for n in ARM]
           for s, sfx in (('r','_r'), ('l','_l'))}
pads = {s: [gid(f'{p}_tip{sfx}_collision') for p in FINGERS]
        for s, sfx in (('r',''), ('l','_left'))}
finger_qadr = {s: [m.jnt_qposadr[jid(f'{p}j{k}{sfx}')]
                   for p in FINGERS for k in range(4)]
               for s, sfx in (('r',''), ('l','_left'))}

rng = np.random.default_rng(77)
# r19 cut 8: FWD-AWARE native pairs — gate1's forward-axis criterion
# (pad-centroid vs face inward <= 35 deg) and the no-through rule are
# HARD constraints of the search itself (attitude is an arm-pose
# property; local polish cannot switch branches — measured r18)
FWD_MAX_DEG = 35.0
N = int(os.environ.get('CURL_FREE_N', '8000'))
found = {'r': 0, 'l': 0}
best = {'r': 1e9, 'l': 1e9}
best_q = {'r': None, 'l': None}
best_fingers = {'r': None, 'l': None}
pairs = []
palms = {'r': mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'palm'),
         'l': mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'palm_left')}
inward = {'r': np.array([-1.0, 0.0, 0.0]),
          'l': np.array([1.0, 0.0, 0.0])}

for trial in range(N):
    for s in ('r','l'):
        for qa in arm_qadr[s]:
            d.qpos[qa] = rng.uniform(-3.14, 3.14)
    for s in ('r','l'):
        for qa in finger_qadr[s]:
            if (qa - arm_qadr[s][0]) % 4 == 0:
                d.qpos[qa] = rng.uniform(0.0, 0.3)
            else:
                d.qpos[qa] = rng.uniform(0.0, 0.9)
    mujoco.mj_forward(m, d)
    zc = d.xpos[obj][2]

    for s, fs in (('r', 1), ('l', -1)):
        sv = [0]*4
        dy = [0]*4
        dz = [0]*4
        dxs = [0]*4
        for i, g in enumerate(pads[s]):
            p = d.geom_xpos[g]
            dx = (p[0] - fs*half[0]) * fs
            dyv = max(abs(p[1]) - half[1], 0.0)
            dzv = max((zc-half[2])-p[2], p[2]-(zc+half[2]), 0.0)
            dxs[i] = dx * 1000
            sv[i] = np.hypot(np.hypot(dx, dyv), dzv) * 1000
        vals = sorted(sv)
        # r19 cut 8: the fwd-axis criterion is a HARD admission
        # constraint (attitude is arm-pose property — polish cannot fix
        # it, measured r18).  The through depth is RECORDED for ranking
        # (the r18 polish enforces no-through at R5; shallow throughs on
        # non-touching pads are polish-fixable)
        ctr = np.mean([d.geom_xpos[g] for g in pads[s]], axis=0)
        fw = ctr - d.xpos[palms[s]]
        fw = fw / max(float(np.linalg.norm(fw)), 1e-12)
        fwd_deg = float(np.degrees(np.arccos(np.clip(
            float(fw @ inward[s]), -1.0, 1.0))))
        through_mm = round(min(dxs), 3)
        if fwd_deg > FWD_MAX_DEG:
            continue
        if vals[0] < 10.0 and vals[1] < 25.0:
            found[s] += 1
            if vals[1] < best[s]:
                best[s] = vals[1]
                best_q[s] = d.qpos[arm_qadr[s]].copy()
                # r18: persist the FULL pair (arm + fingers) — the pad
                # state depends on both; arm-only seeds broke r13.7
                pairs.append({
                    'side': 'right' if s == 'r' else 'left',
                    'arm': d.qpos[arm_qadr[s]].copy().tolist(),
                    'fingers': d.qpos[finger_qadr[s]].copy().tolist(),
                    'first_mm': round(float(vals[0]), 3),
                    'second_mm': round(float(vals[1]), 3),
                    'fwd_deg': round(fwd_deg, 2),
                    'through_mm': through_mm,
                })

print(f'N={N}: right found {found["r"]}, left {found["l"]}')
for s in ('right','left'):
    k = 'r' if s=='right' else 'l'
    qs = str(np.round(best_q[k],3).tolist()) if best_q[k] is not None else 'none'
    print(f'{s}: best 2nd-pad {best[k]:.1f} mm | q = {qs}')
out = os.path.join(REPO, 'runs/w6b/native_seed_pairs.json')
with open(out, 'w') as fh:
    json.dump({'N': N, 'n_pairs': len(pairs), 'pairs': pairs}, fh)
print(f'wrote {out} ({len(pairs)} pairs)')
