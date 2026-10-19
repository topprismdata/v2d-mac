"""r13.8: continuity-seeded curl-free grasp search.

Step 1: run the position-only DLS to get q_arm_terminal for each hand.
Step 2: curl-free search with a continuity cost lambda*||q_arm - q_term||.
Step 3: select the feasible solution with the minimum joint jump.
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

FINGERS = ('ff','mf','rf','th')
ARM = ('shoulder_pan','shoulder_lift','elbow','wrist_1','wrist_2','wrist_3')

def jid(n): return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
def gid(n): return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)
def bid(n): return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)

arm = {}
for s, sfx in (('r','_r'), ('l','_l')):
    jids = [jid(f'ur5e{sfx}_{n}_joint') for n in ARM]
    arm[s] = {'qadr': [m.jnt_qposadr[j] for j in jids],
              'vjad': [m.jnt_dofadr[j] for j in jids],
              'lo': m.jnt_range[jids][:, 0],
              'hi': m.jnt_range[jids][:, 1],
              'palm_bid': bid(f'palm{sfx}'),
              'pad_gids': [gid(f'{p}_tip{sfx}_collision') for p in FINGERS],
              'finger_qadr': [m.jnt_qposadr[jid(f'{p}j{k}{sfx}')]
                              for p in FINGERS for k in range(4)]}

anchors = {'r': np.array([+0.11, 0.0, 0.76]),
           'l': np.array([-0.11, 0.0, 0.76])}

# ---- Step 1: position-only DLS to get q_terminal ----
q_term = {}
for s in ('r','l'):
    a = arm[s]
    q = np.zeros(6)
    tgt = anchors[s]
    for _ in range(80):
        d.qpos[a['qadr']] = q
        mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d)
        p = d.xpos[a['palm_bid']]
        e = tgt - p
        if np.linalg.norm(e) < 1e-3: break
        jacp = np.zeros((3, m.nv))
        mujoco.mj_jacBody(m, d, jacp, None, a['palm_bid'])
        J = jacp[:, a['vjad']]
        dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(3), e)
        q = np.clip(q + dq, a['lo'], a['hi'])
    q_term[s] = q.copy()
    print(f'q_terminal {s}: {np.round(q,3).tolist()}')

# ---- Step 2: curl-free search with continuity cost ----
rng = np.random.default_rng(88)
LAM = 1.0
N = 8000
results = {s: [] for s in ('r','l')}

for trial in range(N):
    for s in ('r','l'):
        a = arm[s]
        fq = a['finger_qadr']
        for qa in fq:
            d.qpos[qa] = rng.uniform(0.0, 0.9)
        for i, qa in enumerate(a['qadr']):
            d.qpos[qa] = rng.uniform(a['lo'][i], a['hi'][i])
    mujoco.mj_forward(m, d)
    zc = d.xpos[obj][2]

    for s, fs in (('r', 1), ('l', -1)):
        a = arm[s]
        dists = []
        for g in a['pad_gids']:
            p = d.geom_xpos[g]
            sv = (p[0] - fs*half[0]) * fs
            dyv = max(abs(p[1]) - half[1], 0.0)
            dzv = max((zc-half[2])-p[2], p[2]-(zc+half[2]), 0.0)
            dists.append(np.hypot(np.hypot(sv, dyv), dzv) * 1000)
        vals = sorted(dists)
        if vals[0] < 10.0 and vals[1] < 25.0:
            q_arm = d.qpos[a['qadr']].copy()
            jump = float(np.linalg.norm(q_arm - q_term[s])) * 180 / np.pi
            results[s].append({'jump_deg': jump, 'pad2_mm': vals[1],
                               'q': q_arm.copy()})

# ---- report ----
for s in ('r','l'):
    results[s].sort(key=lambda x: x['jump_deg'])
    print(f'\n=== {s} ===')
    print(f'feasible count: {len(results[s])}')
    for r in results[s][:5]:
        print(f'  jump {r["jump_deg"]:.1f} deg | pad2 {r["pad2_mm"]:.1f} mm | q = {np.round(r["q"],3).tolist()}')
    if results[s]:
        print(f'  min jump: {results[s][0]["jump_deg"]:.1f} deg (>60 = TRUE_EXCLUSIVE)')
    else:
        print('  NO feasible solutions')
