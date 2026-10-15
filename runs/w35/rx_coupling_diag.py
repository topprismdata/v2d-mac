"""Wave 3.5-C evidence: quantify WHY rx widening cannot produce bilateral contact.

rx_widen_scan.py answered the gate (K1 = 0 over 2.63M poses). This script
characterises the failure for the adjudication:

  1. joint metric (kin_shortfall's `joint_m`): over feasible poses, the best
     value of max(long_finger_gap_below_top, thumb_metric) where thumb_metric
     is the thumb gap below the top, or how far ABOVE the top it bottoms out.
     standoff wave's best simultaneous pose: thumb 13.702 mm above the top.
  2. state of the long fingers in every thumb-below-top pose (are they near
     their wall, and where did they go?).
  3. travel extrapolation: rx = 2.2, 2.6 (beyond the widened 1.8) - does more
     travel keep improving the joint metric, or is the coupling structural?
  4. coordinate-descent refinement seeded from the best rows (coarse-grid
     guard, same as the w35 method).
"""
from __future__ import annotations

import json
import multiprocessing as mp
import pathlib
import sys
import time

import numpy as np

REPO = pathlib.Path("/Users/guohongbin/v2d-mac")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "runs/w35"))

import mujoco  # noqa: E402
import rx_widen_scan as base  # noqa: E402  (module import only; no main)

XML = REPO / "grounding/assets/allegro_floating.xml"
OUT_JSON = REPO / "runs/w35/rx_coupling_diag.json"
mj = mujoco
CAP = 0.05
NEAR = 0.015


def joint_score(base_pos, rot, fcmd):
    """kin_shortfall-style joint metric + full pad state."""
    model, data = base.S["model"], base.S["data"]
    box, palm, pads = base.S["box"], base.S["palm"], base.S["pads"]
    tol = base.S["tol"]
    data.qpos[:] = model.qpos0
    data.qpos[base.S["wqadr"]] = np.concatenate([base_pos, rot])
    data.qpos[base.S["fqadr"]] = fcmd
    mj.mj_forward(model, data)
    bp = data.xpos[base.S["obj_body"]]
    top = float(bp[2] + model.geom_size[box][2])
    pen = 0.0
    for i in range(data.ncon):
        pen = max(pen, -float(data.contact[i].dist))
    if pen > tol:
        return None
    palm_gap = base.gdist(palm, box, 0.15)
    if palm_gap <= 0.0:
        return None
    long_gap, long_short, long_near = CAP, CAP, []
    long_state = {}
    for f in ("ff", "mf", "rf"):
        g = data.geom_xpos[pads[f]]
        d = base.gdist(pads[f], box)
        if g[2] < top:
            long_gap = min(long_gap, d)
            if d <= NEAR:
                long_near.append(f)
            long_state[f] = {"below": True, "gap_m": round(d, 5),
                             "pos": [round(float(v), 4) for v in g]}
        else:
            long_short = min(long_short, float(g[2] - top))
            long_state[f] = {"below": False, "above_top_m": round(float(g[2] - top), 5),
                             "pos": [round(float(v), 4) for v in g]}
    tg = data.geom_xpos[pads["th"]]
    th_below = bool(tg[2] < top)
    th_d = base.gdist(pads["th"], box)
    th_metric = min(th_d, CAP) if th_below else min(float(tg[2] - top), CAP)
    joint = max(long_gap, th_metric)
    return {
        "base": [float(v) for v in base_pos], "rot": [float(v) for v in rot],
        "cmd": fcmd.tolist()[:1] and None,  # keep payload lean; cmd passed separately
        "joint_m": joint, "long_gap_m": long_gap, "long_short_m": long_short,
        "long_near": long_near, "long_state": long_state,
        "th_below": th_below, "th_metric_m": th_metric, "th_gap_m": th_d,
        "pen_m": float(pen), "palm_gap_m": float(palm_gap),
    }


def run_chunk(args):
    rxs, grid, keep = args
    best_joint = []
    th_rows = []
    n = n_ok = n_th = 0
    for cl in grid["CLS"]:
        for ct in grid["CTS"]:
            for t0 in grid["T0S"]:
                fcmd = base.finger_cmd(cl, ct, t0)
                for ry in grid["RYS"]:
                    for rz in grid["RZS"]:
                        for rx in rxs:
                            rot = np.array([rx, ry, rz])
                            for z in grid["ZS"]:
                                for x in grid["XYS"]:
                                    for y in grid["XYS"]:
                                        sc = joint_score(np.array([x, y, z]), rot, fcmd)
                                        n += 1
                                        if sc is None:
                                            continue
                                        n_ok += 1
                                        sc["cmd"] = [cl, ct, t0]
                                        if sc["th_below"]:
                                            n_th += 1
                                            th_rows.append(sc)
                                        best_joint.append(sc)
    best_joint.sort(key=lambda r: r["joint_m"])
    th_rows.sort(key=lambda r: r["th_metric_m"])
    return {"rxs": rxs, "n": n, "ok": n_ok, "th_below": n_th,
            "best_joint": best_joint[:keep], "th_rows": th_rows[:keep]}


def descend(seed):
    """Greedy coordinate descent on the joint metric from one seed pose."""
    s = dict(seed)
    cur = joint_score(np.array(s["base"]), np.array(s["rot"]),
                      base.finger_cmd(*s["cmd"]))
    if cur is None:
        return None
    steps = [0.02, 0.01, 0.005, 0.002]
    rstep = [0.15, 0.07, 0.03]
    improved = True
    evals = 1
    while improved and evals < 4000:
        improved = False
        for ai, span in ((0, steps), (1, steps), (2, steps)):
            for st in span:
                for sgn in (1.0, -1.0):
                    cand = dict(s)
                    cand["base"] = list(s["base"])
                    cand["base"][ai] = s["base"][ai] + sgn * st
                    r = joint_score(np.array(cand["base"]), np.array(s["rot"]),
                                    base.finger_cmd(*s["cmd"]))
                    evals += 1
                    if r is not None and r["joint_m"] < cur["joint_m"] - 1e-6:
                        s, cur, improved = cand, r, True
        for ai in range(3):
            for st in rstep:
                for sgn in (1.0, -1.0):
                    cand = dict(s)
                    cand["rot"] = list(s["rot"])
                    cand["rot"][ai] = s["rot"][ai] + sgn * st
                    r = joint_score(np.array(cand["base"]), np.array(s["rot"]),
                                    base.finger_cmd(*s["cmd"]))
                    evals += 1
                    if r is not None and r["joint_m"] < cur["joint_m"] - 1e-6:
                        s, cur, improved = cand, r, True
        for ai, span in ((0, (0.25, 0.1)), (1, (0.25, 0.1)), (2, (0.25, 0.1))):
            for st in span:
                for sgn in (1.0, -1.0):
                    cand = dict(s)
                    cand["cmd"] = list(s["cmd"])
                    cand["cmd"][ai] = max(-1.0, min(1.0, s["cmd"][ai] + sgn * st))
                    r = joint_score(np.array(cand["base"]), np.array(s["rot"]),
                                    base.finger_cmd(*cand["cmd"]))
                    evals += 1
                    if r is not None and r["joint_m"] < cur["joint_m"] - 1e-6:
                        s, cur, improved = cand, r, True
    cur["descend_evals"] = evals
    return cur


def main() -> None:
    t0 = time.time()
    base._init()
    grid = {
        "CLS": (-1.0, -0.5, 0.0, 0.5, 1.0),
        "CTS": (-1.0, -0.5, 0.0, 0.5, 1.0),
        "T0S": (-1.0, 0.0, 1.0),
        "RYS": (-1.259, -0.9, 0.9, 1.2),
        "RZS": (-0.4, 0.0, 0.4),
        "ZS": np.arange(-0.163, 0.1601, 0.04),
        "XYS": np.arange(-0.06, 0.0601, 0.03),
    }
    chunk_jobs = [
        ([0.9], 30), ([1.2], 30), ([1.5], 30), ([1.8], 30),
        ([2.2], 30), ([2.6], 30), ([-1.8], 30),
    ]
    with mp.Pool(processes=7, initializer=base._init) as pool:
        results = pool.map(run_chunk, [(rxs, grid, keep) for rxs, keep in chunk_jobs])

    per_rx = []
    seeds = []
    th_best_by_rx = {}
    for r in results:
        rx = r["rxs"][0]
        per_rx.append({
            "rx": rx, "n": r["n"], "ok": r["ok"], "th_below": r["th_below"],
            "best_joint_m": r["best_joint"][0]["joint_m"] if r["best_joint"] else None,
            "best_joint": r["best_joint"][:3],
        })
        th_best_by_rx[str(rx)] = r["th_rows"][:6]
        seeds.extend(r["best_joint"][:2])

    print("coordinate descent from", len(seeds), "seeds ...", flush=True)
    descended = []
    for sd in seeds:
        d = descend(sd)
        if d is not None:
            descended.append(d)
    descended.sort(key=lambda r: r["joint_m"])

    out = {
        "wave": "3.5-C coupling diagnosis (rx widened travel)",
        "mjcf": "grounding/assets/allegro_floating.xml",
        "r5_tolerance_m": base.S["tol"],
        "grid": "same as rx_widen_scan; rx subsets {0.9,1.2,1.5,1.8} + extrapolation {2.2,2.6} + {-1.8}",
        "per_rx": per_rx,
        "thumb_below_rows_by_rx": th_best_by_rx,
        "descended_best": descended[:8],
        "standoff_reference": "runs/w35/verdict.json: best simultaneous pose thumb pad 13.702e-3 m ABOVE box top at all three heights",
        "elapsed_s": round(time.time() - t0, 1),
    }
    OUT_JSON.write_text(json.dumps(out, indent=2))
    slim = {k: v for k, v in out.items() if k not in ("thumb_below_rows_by_rx",)}
    print(json.dumps(slim, indent=2)[:6000], flush=True)


if __name__ == "__main__":
    main()
