"""Wrap-topology mapper: hand descends OVER the box and closes. For each
(rot, z) cell, classifies which pad touches which box face (strict criterion).
Purpose: find the reachable cells whose grasp contains a TOP-face contact plus
a WALL contact (the wedge) - the only single-hand topology that can lift a box
resting flush on a pedestal without dual-wall opposition (K1 is class-B)."""
from __future__ import annotations

import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

REPO = Path("/Users/guohongbin/v2d-mac")
sys.path.insert(0, str(REPO))

from grounding.envs.lift_hold_env import LiftHoldEnv  # noqa: E402
from grounding.reachable_scan import (  # noqa: E402
    _action, _penetration_tol, _tip_pads, close_amp, strict_contacts,
)

OUT = REPO / "runs/w35/rscan_wrap_topology.json"
N_DESC = 60
N_CLOSE = 80


def probe_cell(cell) -> dict:
    x, y, z, rx, ry, rz = cell
    from grounding.envs.lift_hold_env import LiftHoldEnv
    env = LiftHoldEnv(episode_len=N_DESC + N_CLOSE + 20, domain_rand=False)
    env.reset(seed=0)
    pads = _tip_pads(env)
    tol = _penetration_tol(env)
    rot = [rx, ry, rz]
    open_a = np.full(16, 0.6)
    close = close_amp(1.0, -0.6, 0.0)
    tgt = [x, y, z]

    def step_to(t, f):
        env.step(_action(env, t, rot, f))

    high = [x, y, z + 0.10]
    for _ in range(30):
        step_to(high, open_a)
    for _ in range(N_DESC):
        step_to(tgt, open_a)
    for i in range(N_CLOSE):
        f = open_a + (close - open_a) * min(1.0, (i + 1) / (N_CLOSE * 0.7))
        step_to(tgt, f)
    for _ in range(N_CLOSE - int(N_CLOSE * 0.7)):
        step_to(tgt, close)

    sc = strict_contacts(env, pads, tol)
    ff = float(env.trace["finger_force"][-1])
    held = bool(env.trace["finger_held"][-1])
    env.close()
    return {"cell": list(cell), "class": sc["class"], "F": round(ff, 2),
            "held": held, "n_strict_x": sc["n_strict_x"], "top": sc["top_touch"],
            "pads": {k: {"face": v["face"], "gap_mm": round(v["gap_m"] * 1000, 2),
                         "below": v["below_top"], "touch": v["strict_touch"]}
                     for k, v in sc["pads"].items()}}


_CELLS = None


def _work(cell):
    try:
        return probe_cell(cell)
    except Exception as e:  # noqa: BLE001
        return {"cell": list(cell), "error": str(e)}


def main() -> None:
    global _CELLS
    zs = [0.84, 0.87, 0.90]
    rxs = [0.0, 0.6, 1.2]
    rys = [-1.0, -0.5, 0.0, 0.5, 1.0, 1.2]
    rzs = [-0.6, 0.0, 0.6]
    _CELLS = [(0.0, 0.0, z, rx, ry, rz) for z in zs for rx in rxs
              for ry in rys for rz in rzs]
    with mp.Pool(8) as pool:
        rows = pool.map(_work, _CELLS, chunksize=2)
    ok = [r for r in rows if "error" not in r]
    wedges = [r for r in ok if r["top"] and r["n_strict_x"] >= 1]
    held = [r for r in ok if r["held"] and r["F"] > 1.0]
    OUT.write_text(json.dumps({"n": len(rows), "wedges": wedges,
                               "held": held, "rows": rows}, indent=2))
    print(f"cells {len(ok)}  wedge(top+xwall) {len(wedges)}  held(F>1N) {len(held)} -> {OUT}")
    for r in sorted(held, key=lambda r: -r["F"])[:12]:
        faces = {k: v["face"] for k, v in r["pads"].items() if v["touch"] or v["gap_mm"] < 5}
        print(f"  z={r['cell'][2]} rot={r['cell'][3:]} F={r['F']} held={r['held']} "
              f"cls={r['class']} {faces}")


if __name__ == "__main__":
    main()
