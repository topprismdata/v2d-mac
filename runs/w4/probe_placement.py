"""Task A probe: initial-penetration + resting-stability of out-of-reach placements.

Measurement method (recorded verbatim in the W4-A report):
  * placement candidates are written straight into data.qpos on a bare model
    (no env reset), then mj_forward() -> the contact table IS the state at the
    instant the box is "born" (penetration shows up as negative contact dist).
  * resting stability: after the env's own 400-step settle (home hand pose,
    ctrl mirrored onto qpos, qvel=0) the box free-runs 399 further steps with
    the hand held still; report |p-p0| and |v| at the end.
  * reachability: full env, 4 seeds x 100 random flail steps, count any
    finger-box contact pair (the C6 criterion).
"""
from __future__ import annotations

import pathlib
import sys

import mujoco
import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from grounding.envs.lift_hold_env import MJCF, LiftHoldEnv  # noqa: E402

m = mujoco.MjModel.from_xml_path(str(MJCF))
d = mujoco.MjData(m)
OBJ_GEOM = int(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "object_geom"))
HALF = np.array(m.geom_size[OBJ_GEOM])


def support_geoms():
    out = []
    for g in range(m.ngeom):
        if int(m.geom_bodyid[g]) != 0 or g == OBJ_GEOM:
            continue
        if int(m.geom_type[g]) != int(mujoco.mjtGeom.mjGEOM_BOX):
            continue
        out.append((mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g),
                    np.array(m.geom_pos[g]), np.array(m.geom_size[g])))
    return out


SUPPORTS = support_geoms()
print("support geoms (name, pos, half):")
for n, p, h in SUPPORTS:
    print(f"  {n:12s} pos={p} half={h} top={p[2] + h[2]:.4f} "
          f"x=[{p[0]-h[0]:.3f},{p[0]+h[0]:.3f}] y=[{p[1]-h[1]:.3f},{p[1]+h[1]:.3f}]")
print(f"box half = {HALF}")


def center_height(x, y):
    best = -np.inf
    for _n, p, h in SUPPORTS:
        if abs(x - p[0]) <= h[0] and abs(y - p[1]) <= h[1]:
            best = max(best, float(p[2] + h[2]))
    return best


def footprint_height(x, y):
    """max top over geoms whose footprint OVERLAPS the box footprint"""
    fx0, fx1 = x - HALF[0], x + HALF[0]
    fy0, fy1 = y - HALF[1], y + HALF[1]
    best = -np.inf
    for _n, p, h in SUPPORTS:
        gx0, gx1 = p[0] - h[0], p[0] + h[0]
        gy0, gy1 = p[1] - h[1], p[1] + h[1]
        if fx1 > gx0 and gx1 > fx0 and fy1 > gy0 and gy1 > fy0:
            best = max(best, float(p[2] + h[2]))
    return best


def birth_contacts(x, y, z):
    mujoco.mj_resetData(m, d)
    adr = int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "object_free")])
    d.qpos[adr:adr + 7] = [x, y, z, 1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(m, d)
    rows = []
    for i in range(d.ncon):
        c = d.contact[i]
        rows.append((mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, int(c.geom1)),
                     mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, int(c.geom2)),
                     float(c.dist)))
    return rows


cands = [(0.0, 0.35, "A current bug (center-only height)"),
         (0.0, 0.35, "B footprint-aware at y=0.35"),
         (0.0, 0.18, "C fully on pedestal (y=0.18)"),
         (0.375, 0.0, "D fully on slab, x=0.375"),
         (0.39, 0.0, "E fully on slab, x=0.39"),
         (0.36, 0.0, "F fully on slab, x=0.36")]

print("\n=== birth contact tables ===")
for x, y, label in cands:
    if label.startswith("B"):
        z = footprint_height(x, y) + HALF[2]
    elif label.startswith("E") or label.startswith("F"):
        z = center_height(x, y) + HALF[2]
    else:
        z = center_height(x, y) + HALF[2]
    rows = birth_contacts(x, y, z)
    neg = [r for r in rows if r[2] < -1e-9]
    print(f"{label:42s} @({x},{y},{z:.4f}) ncon={len(rows)} "
          f"negative={[f'{a}|{b} {c:.4f}' for a, b, c in neg]}")


def settle_and_drift(x, y, z, *, n_settle=400, n_free=399):
    """Replicate reset()'s hand pose + ctrl mirror, then free-run the box."""
    mujoco.mj_resetData(m, d)
    env_ref = _env_cls()
    d.qpos[env_ref.wrist_qpos_adr[0]] = 0.0
    pre_curl = np.array([0.0, 0.9, 0.7, 0.4] * 3 + [0.8, 0.5, 0.4, 0.3])
    d.qpos[env_ref.finger_qpos_adr] = pre_curl
    d.qpos[env_ref.wrist_qpos_adr[2]] = -0.02
    d.qvel[:] = 0.0
    d.ctrl[6:] = pre_curl
    d.ctrl[0:3] = d.qpos[env_ref.wrist_qpos_adr[0:3]]
    d.ctrl[3:6] = 0.0
    adr = env_ref.obj_qpos_adr
    d.qpos[adr:adr + 7] = [x, y, z, 1.0, 0.0, 0.0, 0.0]
    d.qvel[env_ref.obj_qvel_adr:env_ref.obj_qvel_adr + 6] = 0.0
    for _ in range(n_settle):
        mujoco.mj_step(m, d)
    p0 = d.xpos[env_ref.obj_body].copy()
    vmax = 0.0
    worst = 0.0
    for _ in range(n_free):
        mujoco.mj_step(m, d)
        worst = max(worst, float(np.linalg.norm(d.xpos[env_ref.obj_body] - p0)))
        vmax = max(vmax, float(np.linalg.norm(d.qvel[env_ref.obj_qvel_adr:env_ref.obj_qvel_adr + 3])))
    p1 = d.xpos[env_ref.obj_body].copy()
    v1 = d.qvel[env_ref.obj_qvel_adr:env_ref.obj_qvel_adr + 3].copy()
    env_ref.close()
    return p0, p1, v1, worst, vmax


def _env_cls():
    return LiftHoldEnv(evaluator_only=False, domain_rand=False)


print("\n=== settle(400) + free-run(399) drift, home hand pose ===")
plans = [(0.0, 0.35, 0.74, "A current bug"),
         (0.0, 0.35, 0.80, "B footprint-aware y=0.35"),
         (0.0, 0.18, 0.80, "C on pedestal y=0.18"),
         (0.375, 0.0, 0.74, "D slab x=0.375"),
         (0.39, 0.0, 0.74, "E slab x=0.39"),
         (0.0, 0.18, 0.7601, "C+ pedestal y=0.18 +0.1mm")]
for x, y, z, label in plans:
    p0, p1, v1, worst, vmax = settle_and_drift(x, y, z)
    print(f"{label:30s} p0={np.round(p0,4)} p1={np.round(p1,4)} "
          f"|p1-p0|={np.linalg.norm(p1-p0)*1000:.3f}mm worst={worst*1000:.3f}mm "
          f"|v1|={np.linalg.norm(v1):.2e} vmax={vmax:.2e}")


def flail_reach(x, y, z, seeds=(3, 7, 11, 13), n=100):
    """C6 protocol with a forced placement: count finger-box contacts."""
    total = 0
    worst = 0.0
    for seed in seeds:
        env = _env_cls()
        env.reset(seed=seed)
        mujoco.mj_resetData(env.model, env.data)
        d2, m2 = env.data, env.model
        pre_curl = np.array([0.0, 0.9, 0.7, 0.4] * 3 + [0.8, 0.5, 0.4, 0.3])
        d2.qpos[env.wrist_qpos_adr[0]] = 0.0
        d2.qpos[env.finger_qpos_adr] = pre_curl
        d2.qpos[env.wrist_qpos_adr[2]] = -0.02
        d2.qvel[:] = 0.0
        d2.ctrl[6:] = pre_curl
        d2.ctrl[0:3] = d2.qpos[env.wrist_qpos_adr[0:3]]
        d2.ctrl[3:6] = 0.0
        d2.qpos[env.obj_qpos_adr:env.obj_qpos_adr + 7] = [x, y, z, 1, 0, 0, 0]
        d2.qvel[env.obj_qvel_adr:env.obj_qvel_adr + 6] = 0.0
        for _ in range(400):
            mujoco.mj_step(m2, d2)
        start = d2.qpos[env.obj_qpos_adr:env.obj_qpos_adr + 3].copy()
        rng = np.random.default_rng(seed)
        for _ in range(n):
            env.step(rng.uniform(-1, 1, 22).astype(np.float32))
            total += any(p[0].split("#")[0].startswith(("ff_", "mf_", "rf_", "th_"))
                         for p in env.trace["contact_pairs"][-1])
        worst = max(worst, float(np.linalg.norm(
            d2.qpos[env.obj_qpos_adr:env.obj_qpos_adr + 3] - start)))
        env.close()
    return total, worst


print("\n=== C6 flail reachability per placement (4 seeds x 100 steps) ===")
for x, y, z, label in [(0.0, 0.35, 0.74, "A bug (current)"),
                       (0.0, 0.35, 0.80, "B footprint y=0.35"),
                       (0.0, 0.18, 0.80, "C pedestal y=0.18"),
                       (0.375, 0.0, 0.74, "D slab x=0.375"),
                       (0.39, 0.0, 0.74, "E slab x=0.39"),
                       (0.36, 0.0, 0.74, "F slab x=0.36")]:
    hits, mv = flail_reach(x, y, z)
    print(f"{label:24s} finger-box contacts={hits:3d}  worst_move={mv*1000:.3f}mm")
