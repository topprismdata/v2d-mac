"""r13.6: curl-free joint search — UR5e native verified grasp.

Joint parameterisation: UR5e 6R arm + 16 finger curl DoFs per hand.
Criterion: >=2 pads within 25mm of the face rect per hand.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
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
N = 8000
found = {'r': 0, 'l': 0}
best = {'r': 1e9, 'l': 1e9}
best_q = {'r': None, 'l': None}

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
        for i, g in enumerate(pads[s]):
            p = d.geom_xpos[g]
            dx = (p[0] - fs*half[0]) * fs
            dyv = max(abs(p[1]) - half[1], 0.0)
            dzv = max((zc-half[2])-p[2], p[2]-(zc+half[2]), 0.0)
            sv[i] = np.hypot(np.hypot(dx, dyv), dzv) * 1000
        vals = sorted(sv)
        if vals[0] < 10.0 and vals[1] < 25.0:
            found[s] += 1
            if vals[1] < best[s]:
                best[s] = vals[1]
                best_q[s] = d.qpos[arm_qadr[s]].copy()

print(f'N={N}: right found {found["r"]}, left {found["l"]}')
for s in ('right','left'):
    k = 'r' if s=='right' else 'l'
    qs = str(np.round(best_q[k],3).tolist()) if best_q[k] is not None else 'none'
    print(f'{s}: best 2nd-pad {best[k]:.1f} mm | q = {qs}')
