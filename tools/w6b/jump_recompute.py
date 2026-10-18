"""r13.8b: jump recompute with wrap normalisation + proper q_terminal."""
import numpy as np, mujoco, os

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
    fqa = [m.jnt_qposadr[jid(f'{p}j{k}{sfx}')] for p in FINGERS for k in range(4)]
    arm[s] = {'qadr': [m.jnt_qposadr[j] for j in jids],
              'vjad': [m.jnt_dofadr[j] for j in jids],
              'lo': m.jnt_range[jids][:, 0], 'hi': m.jnt_range[jids][:, 1],
              'palm': bid(f'palm{sfx}'),
              'pads': [gid(f'{p}_tip{sfx}_collision') for p in FINGERS],
              'fqa': fqa}

anchors = {'r': np.array([0.11, 0, 0.76]), 'l': np.array([-0.11, 0, 0.76])}

q_term = {}
for s in ('r','l'):
    a = arm[s]
    q = np.zeros(6)
    tgt = anchors[s]
    for _ in range(150):
        d.qpos[a['qadr']] = q
        mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d)
        p = d.xpos[a['palm']]
        e = tgt - p
        if np.linalg.norm(e) < 1e-3: break
        jacp = np.zeros((3, m.nv))
        mujoco.mj_jacBody(m, d, jacp, None, a['palm'])
        J = jacp[:, a['vjad']]
        dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(3), e)
        q = np.clip(q + dq, a['lo'], a['hi'])
    q_term[s] = q.copy()
    print(f'{s} q_terminal: {np.round(q,3).tolist()} | final |e|={np.linalg.norm(d.xpos[a["palm"]]-tgt):.5f}')

def wrap_dist(a, b):
    return (a - b + np.pi) % (2*np.pi) - np.pi

rng = np.random.default_rng(88)
results = {s: [] for s in ('r','l')}
N = 8000
for trial in range(N):
    for s in ('r','l'):
        a = arm[s]
        for qa in a['fqa']:
            d.qpos[qa] = rng.uniform(0, 0.9)
        for i, qa in enumerate(a['qadr']):
            d.qpos[qa] = rng.uniform(a['lo'][i], a['hi'][i])
    mujoco.mj_forward(m, d)
    zc = d.xpos[obj][2]
    for s, fs in (('r',1),('l',-1)):
        a = arm[s]
        dists = []
        for g in a['pads']:
            p = d.geom_xpos[g]
            sv = (p[0] - fs*half[0]) * fs
            dyv = max(abs(p[1])-half[1], 0)
            dzv = max((zc-half[2])-p[2], p[2]-(zc+half[2]), 0)
            dists.append(np.hypot(np.hypot(sv,dyv),dzv)*1000)
        vals = sorted(dists)
        if vals[0] < 10 and vals[1] < 25:
            q_arm = d.qpos[a['qadr']].copy()
            raw = wrap_dist(q_arm, q_term[s])
            jd = float(np.linalg.norm(raw)) * 180 / np.pi
            pj = {ARM[i]: round(float(raw[i]),3) for i in range(6)}
            results[s].append({'jd': jd, 'pj': pj})

for s in ('r','l'):
    results[s].sort(key=lambda x: x['jd'])
    print(f'\n{s} wrap-normalised jump (best 3):')
    for r in results[s][:3]:
        print(f'  {r["jd"]:.1f} deg | {r["pj"]}')
    if results[s]:
        v = 'FEASIBLE' if results[s][0]['jd'] < 60 else 'TRUE_EXCLUSIVE'
        print(f'  min: {results[s][0]["jd"]:.1f} deg -> {v}')
