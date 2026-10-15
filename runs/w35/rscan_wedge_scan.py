"""Wedge-mapper: full wrap+lift per cell, judged by the frozen gate.

Physics (established this round): a box resting flush on the pedestal can only
be lifted by finger contacts through SIDE/END-wall friction; top-face contacts
contribute zero lift (their friction is horizontal). A single hand therefore
needs the wedge topology: a wall pad (friction up) opposed by a top pad
(horizontal friction reaction). This scan runs the complete
transit->descend->ramp-close->settle->lift trajectory for every (xy, z, rot)
cell and reports the gate result plus the per-face force decomposition, so a
passing cell comes with its own load-decomposition evidence."""
from __future__ import annotations

import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

REPO = Path("/Users/guohongbin/v2d-mac")
sys.path.insert(0, str(REPO))

import mujoco  # noqa: E402

from grounding.envs.lift_hold_env import LiftHoldEnv  # noqa: E402
from grounding.reachable_scan import (  # noqa: E402
    _action, close_amp, gate_eval,
)

OUT = REPO / "runs/w35/rscan_wedge_scan.json"
N_TRANSIT = 30
N_DESC = 60
N_CLOSE = 80
N_SETTLE = 25


def face_of(env, pos) -> str:
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


def face_forces(env) -> dict[str, float]:
    m, d = env.model, env.data
    out: dict[str, float] = {}
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
        f = face_of(env, con.pos)
        out[f] = out.get(f, 0.0) + mag
    return out


def run_cell(cell, lift_z=0.16, lift_rate=0.004, close=(1.0, -0.6, 0.0)):
    x, y, z, rx, ry, rz = cell
    env = LiftHoldEnv(episode_len=420, domain_rand=False)
    env.reset(seed=0)
    rot = [rx, ry, rz]
    open_a = np.full(16, 0.6)
    close_a = close_amp(*close)
    tgt = [x, y, z]

    def step_to(t, f):
        env.step(_action(env, t, rot, f))

    high = [x, y, z + 0.10]
    for _ in range(N_TRANSIT):
        step_to(high, open_a)
    for _ in range(N_DESC):
        step_to(tgt, open_a)
    for i in range(N_CLOSE):
        f = open_a + (close_a - open_a) * min(1.0, (i + 1) / (N_CLOSE * 0.7))
        step_to(tgt, f)
    for _ in range(N_CLOSE - int(N_CLOSE * 0.7)):
        step_to(tgt, close_a)
    for _ in range(N_SETTLE):
        step_to(tgt, close_a)
    grasp_faces = face_forces(env)

    cur = float(env.data.xpos[env.hand_root][2])
    for i in range(int(np.ceil(lift_z / lift_rate)) + 80):
        goal = min(tgt[2] + lift_z, cur + lift_rate)
        step_to([x, y, goal], close_a)
        cur = float(env.data.xpos[env.hand_root][2])

    ev = gate_eval(env)
    env.close()
    g = {k: v for k, v in ev.items() if k != "finger_pairs"}
    tot = sum(grasp_faces.values())
    return {"cell": list(cell), "gate": g,
            "grasp_faces": {k: round(v, 2) for k, v in grasp_faces.items()},
            "wall_N": round(sum(v for k, v in grasp_faces.items()
                                if k.startswith(("x", "y"))), 2),
            "top_N": round(grasp_faces.get("top", 0.0), 2),
            "total_N": round(tot, 2)}


_CELLS = None


def _work(cell):
    try:
        return run_cell(cell)
    except Exception as e:  # noqa: BLE001
        return {"cell": list(cell), "error": str(e)}


def main() -> None:
    global _CELLS
    xs = [-0.10, -0.06, -0.02, 0.0, 0.02, 0.06, 0.10]
    ys = [-0.10, -0.06, -0.02, 0.0, 0.02, 0.06, 0.10]
    zs = [0.80, 0.83, 0.86]
    rxs = [1.2, 1.5, 1.8]
    rys = [-1.2, -0.6, 0.0, 0.6, 1.2]
    rzs = [-1.2, -0.6, 0.0, 0.6, 1.2]
    _CELLS = [(x, y, z, rx, ry, rz) for x in xs for y in ys for z in zs
              for rx in rxs for ry in rys for rz in rzs]
    with mp.Pool(8) as pool:
        rows = pool.map(_work, _CELLS, chunksize=2)
    ok = [r for r in rows if "error" not in r]
    ok.sort(key=lambda r: (not r["gate"]["gate_pass"],
                           -min(r["wall_N"], 2.0 * max(r["top_N"], 0.5)),
                           -r["gate"]["peak_lift_m"]))
    OUT.write_text(json.dumps({"n": len(rows), "rows": ok}, indent=2))
    passing = [r for r in ok if r["gate"]["gate_pass"]]
    print(f"cells {len(ok)}  GATE PASS {len(passing)} -> {OUT}")
    for r in ok[:25]:
        g = r["gate"]
        print(f"  xy=({r['cell'][0]:+.2f},{r['cell'][1]:+.2f}) z={r['cell'][2]} rot={r['cell'][3:]} "
              f"peak={g['peak_lift_m']*1000:6.1f}mm duty={g['finger_force_duty_cycle']:.2f} "
              f"meanF={g['finger_force_mean_N']:5.2f} clean@pk={not g['table_touching_at_peak']} "
              f"wall={r['wall_N']:5.1f} top={r['top_N']:5.1f} "
              f"faces={r['grasp_faces']}")


if __name__ == "__main__":
    main()
