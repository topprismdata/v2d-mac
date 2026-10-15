"""Wrap-grasp lift test: take the wrap cells with the strongest held grasp,
pull straight up, and report the frozen-gate outcome + full face decomposition
of finger contact force (which faces carry the lift)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path("/Users/guohongbin/v2d-mac")
sys.path.insert(0, str(REPO))

from grounding.envs.lift_hold_env import LiftHoldEnv  # noqa: E402
from grounding.reachable_scan import (  # noqa: E402
    _action, _penetration_tol, _tip_pads, close_amp, gate_eval,
)

N_DESC = 60
N_CLOSE = 80


def face_of(env, pos) -> str:
    import mujoco
    m, d = env.model, env.data
    half = m.geom_size[env.obj_geom]
    bp = d.xpos[env.obj_body].copy()
    bxm = d.xmat[env.obj_body].reshape(3, 3).copy()
    rel = bxm.T @ (np.asarray(pos) - bp)
    if rel[2] > half[2] * 0.5 and abs(rel[0]) <= half[0] + 1e-9 and abs(rel[1]) <= half[1] + 1e-9:
        return "top"
    if abs(abs(rel[0]) - half[0]) <= half[0] * 0.5:
        return "x_pos" if rel[0] > 0 else "x_neg"
    if abs(abs(rel[1]) - half[1]) <= half[1] * 0.5:
        return "y_pos" if rel[1] > 0 else "y_neg"
    if rel[2] < -half[2] * 0.5:
        return "bottom"
    return "other"


def wrap_and_lift(cell, lift_z=0.16, lift_rate=0.004, close=(1.0, -0.6, 0.0),
                  verbose=False):
    x, y, z, rx, ry, rz = cell
    env = LiftHoldEnv(episode_len=400, domain_rand=False)
    env.reset(seed=0)
    rot = [rx, ry, rz]
    open_a = np.full(16, 0.6)
    close_a = close_amp(*close)
    tgt = [x, y, z]

    def step_to(t, f):
        env.step(_action(env, t, rot, f))

    high = [x, y, z + 0.10]
    for _ in range(30):
        step_to(high, open_a)
    for _ in range(N_DESC):
        step_to(tgt, open_a)
    for i in range(N_CLOSE):
        f = open_a + (close_a - open_a) * min(1.0, (i + 1) / (N_CLOSE * 0.7))
        step_to(tgt, f)
    for _ in range(N_CLOSE - int(N_CLOSE * 0.7)):
        step_to(tgt, close_a)
    for _ in range(25):
        step_to(tgt, close_a)

    # face decomposition at the grasp state (before any lift)
    m, d = env.model, env.data
    grasp_faces: dict[str, float] = {}
    for i in range(d.ncon):
        con = d.contact[i]
        g1, g2 = int(con.geom1), int(con.geom2)
        if env.obj_geom not in (g1, g2):
            continue
        other = g2 if g1 == env.obj_geom else g1
        if other not in env.finger_geom_ids:
            continue
        f6 = np.zeros(6)
        mujoco.mj_contactForce(m, d, i, f6)
        mag = float(np.linalg.norm(f6[:3]))
        grasp_faces[face_of(env, con.pos)] = grasp_faces.get(face_of(env, con.pos), 0.0) + mag
    if verbose:
        labs = sorted({p[0] for p in env.trace["contact_pairs"][-1]})
        print("  grasp geoms:", labs)
        print("  grasp face N:", {k: round(v, 2) for k, v in grasp_faces.items()})

    cur = float(env.data.xpos[env.hand_root][2])
    for i in range(int(np.ceil(lift_z / lift_rate)) + 80):
        goal = min(tgt[2] + lift_z, cur + lift_rate)
        step_to([x, y, goal], close_a)
        cur = float(env.data.xpos[env.hand_root][2])

    ev = gate_eval(env)
    env.close()
    return ev


if __name__ == "__main__":
    import mujoco  # noqa: F401
    cells = [
        (0.0, 0.0, 0.84, 1.2, -1.0, -0.6),
        (0.0, 0.0, 0.87, 0.6, 1.2, 0.6),
        (0.0, 0.0, 0.84, 1.2, 1.2, 0.0),
        (0.0, 0.0, 0.87, 1.2, 1.0, 0.0),
        (0.0, 0.0, 0.87, 1.2, 0.0, -0.6),
    ]
    for c in cells:
        print(f"cell {c}")
        ev = wrap_and_lift(c, verbose=True)
        g = ev
        print(f"  peak={g['peak_lift_m']*1000:7.1f}mm duty={g['finger_force_duty_cycle']:.2f} "
              f"meanF={g['finger_force_mean_N']:.2f}N clean@peak={not g['table_touching_at_peak']} "
              f"anch={g['max_anchored_step_m']*1000:.1f}mm grav={g['max_free_fall_gravity_error']:.2f} "
              f"PASS={g['gate_pass']}")
