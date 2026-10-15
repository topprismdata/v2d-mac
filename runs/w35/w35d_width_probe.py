"""Wave 3.5-D task 3 evidence: at what box width does true-K1 open?

true-K1 (side-checked, stricter than the joint metric): some long finger below
box top within R5 tol of ONE x-wall, AND thumb below top within R5 tol of the
box on the OPPOSITE x-side, palm clear, pen <= tol.

Method: in-memory patch of object_geom x-half-size over
{0.060 (committed), 0.055, 0.050, 0.045, 0.040, 0.035}, committed rz grid
{-0.4, 0, 0.4} plus the best rz family from w35d_rz_scan, neighborhood grid,
tied descent from each width's best rows. Reports the width at which
true-K1 first opens and the margin curve.

Output: runs/w35/w35d_width_probe.json
"""
from __future__ import annotations

import itertools
import json
import multiprocessing as mp
import pathlib
import sys
import time

import numpy as np

REPO = pathlib.Path("/Users/guohongbin/v2d-mac")
sys.path.insert(0, str(REPO / "runs/w35"))

import mujoco  # noqa: E402
import w35d_common as C  # noqa: E402

OUT = REPO / "runs/w35/w35d_width_probe.json"

RXS = [0.9, 1.2, 1.5, 1.8]
RYS = [-1.2, -0.9, 0.9, 1.2]
RZS = [-0.4, 0.0, 0.4]
ZS = [round(v, 4) for v in np.arange(-0.163, 0.1601, 0.04)]
XYS = [round(v, 4) for v in np.arange(-0.06, 0.0601, 0.03)]
CMDS = list(itertools.product((-1.0, -0.5, 0.0, 0.5, 1.0),
                              (-1.0, -0.5, 0.0, 0.5, 1.0),
                              (-1.0, 0.0, 1.0)))
WIDTHS = [0.060, 0.055, 0.050, 0.045, 0.040, 0.035]


def patch_width(half_x: float):
    def f(model, data=None):
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "object_geom")
        size = model.geom_size[gid].copy()
        size[0] = half_x
        model.geom_size[gid] = size
    return f


# extend w35d_common._setup with a width patch hook via wrapper
_orig_setup = C._setup


def setup_with_width(thumb_patch=None, half_x: float | None = None):
    S = _orig_setup(thumb_patch)
    if half_x is not None:
        patch_width(half_x)(S["model"])
    return S


WIDTH_STATE: dict = {}


def worker_init(half_x):
    global S
    import w35d_common as CC
    CC.S = setup_with_width(None, half_x)
    WIDTH_STATE["half_x"] = half_x


def true_k1(sc) -> bool:
    """Side-checked K1 predicate on a joint_score row."""
    tol = C.S["tol"]
    if sc is None or sc["pen_m"] > tol:
        return False
    longs = [st for st in sc["long_state"].values()
             if st.get("below") and st["gap_m"] <= tol]
    if not longs:
        return False
    th_side = sc["th_side"]
    th_below = sc["th_below"]
    th_d = sc["th_gap_m"]
    return bool(th_below and th_side != 0
                and any(st["side"] == -th_side for st in longs)
                and th_d <= tol)


def run_width_chunk(args):
    half_x, rxs = args
    if C.S is None or WIDTH_STATE.get("half_x") != half_x:
        worker_init(half_x)
    tol = C.S["tol"]
    rows = []
    stats = {"n": 0, "ok": 0, "true_k1": 0, "near": 0}
    for (cl, ct, t0) in CMDS:
        fq = C.cmd16(cl, ct, t0)
        for ry in RYS:
            for rz in RZS:
                for rx in rxs:
                    for z in ZS:
                        for x in XYS:
                            for y in XYS:
                                sc = C.joint_score(np.array([x, y, z]),
                                                   np.array([rx, ry, rz]), fq)
                                stats["n"] += 1
                                if sc is None:
                                    continue
                                stats["ok"] += 1
                                sc["cmd"] = [cl, ct, t0]
                                if true_k1(sc):
                                    stats["true_k1"] += 1
                                if sc["joint_m"] <= C.NEAR:
                                    stats["near"] += 1
                                rows.append(sc)
    rows.sort(key=lambda r: r["joint_m"])
    # keep best raw rows AND all true-K1 rows
    k1_rows = [r for r in rows if true_k1(r)][:20]
    return {"half_x": half_x, "stats": stats,
            "best": rows[:10], "k1_rows": k1_rows}


def main() -> None:
    t0 = time.time()
    chunks = []
    for hx in WIDTHS:
        for i in range(4):  # split rx buckets across processes
            chunks.append((hx, [RXS[i]]))
    with mp.Pool(processes=8) as pool:
        parts = pool.map(run_width_chunk, chunks)

    by_width = {}
    for p in parts:
        hx = p["half_x"]
        b = by_width.setdefault(hx, {"stats": {"n": 0, "ok": 0, "true_k1": 0,
                                               "near": 0},
                                     "best": [], "k1_rows": []})
        for k in b["stats"]:
            b["stats"][k] += p["stats"][k]
        b["best"].extend(p["best"])
        b["k1_rows"].extend(p["k1_rows"])
    for hx, b in by_width.items():
        b["best"].sort(key=lambda r: r["joint_m"])
        b["best"] = b["best"][:10]
        b["k1_rows"].sort(key=lambda r: r["joint_m"])
        b["k1_rows"] = b["k1_rows"][:10]

    # descent from each width's best raw row
    desc = {}
    for hx, b in sorted(by_width.items()):
        worker_init(hx)
        r0 = b["best"][0]
        rr = C.descend({"base": list(r0["base"]), "rot": list(r0["rot"]),
                        "cmd": list(r0["cmd"])}, "tied", None)
        if rr is not None:
            rr["true_k1"] = true_k1(rr)
        desc[str(hx)] = rr

    # first width where true K1 opens (raw grid OR descent)
    open_width = None
    for hx in sorted(by_width):
        if by_width[hx]["stats"]["true_k1"] > 0 or (
                desc.get(str(hx)) and desc[str(hx)].get("true_k1")):
            open_width = hx
            break

    out = {
        "wave": "3.5-D task3 box-width probe (true-K1 vs box x-half-size)",
        "date": "2026-10-15",
        "committed_half_x_m": 0.06,
        "widths": WIDTHS,
        "grid": {"rxs": RXS, "rys": RYS, "rzs": RZS, "zs_n": len(ZS),
                 "xy_n": len(XYS), "cmd_n": len(CMDS)},
        "per_width": {f"{hx:.3f}": {
            "stats": b["stats"],
            "best_joint_m_mm": round(b["best"][0]["joint_m"] * 1000, 4),
            "best": b["best"][:6], "k1_rows": b["k1_rows"][:6],
            "descended": desc.get(str(hx)),
        } for hx, b in sorted(by_width.items())},
        "first_true_k1_width_m": open_width,
        "elapsed_s": round(time.time() - t0, 1),
    }
    OUT.write_text(json.dumps(out, indent=1))
    for hx in sorted(by_width):
        b = by_width[hx]
        d = desc.get(str(hx))
        print(f"hx {hx:.3f}: n={b['stats']['n']} true_k1={b['stats']['true_k1']} "
              f"best_mm={b['best'][0]['joint_m']*1000:.3f} "
              f"desc_mm={d['joint_m']*1000:.3f} desc_true_k1={d.get('true_k1') if d else None}",
              flush=True)
    print("first true-K1 width:", open_width)
    print("elapsed", out["elapsed_s"], "s")


if __name__ == "__main__":
    main()
