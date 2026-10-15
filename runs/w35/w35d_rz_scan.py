"""Wave 3.5-D task 2: box-hand relative roll (rz) full-circle search.

Prior scans held rz in {-0.4, 0, 0.4}. This scan covers rz in [-pi, pi] at
13 steps (pi/6) x the best-family neighborhood
(rx {0.9,1.2,1.5,1.8}, ry {-1.259,-0.9,-0.3,0.3,0.9,1.2}),
plus per-rz coordinate descent from each bucket's best rows with continuous rz.

Question: does ANY roll break the thumb-under-top / long-finger-at-wall
mutual exclusion (K1-grade or NEAR-grade simultaneous pose)?

Note: MJCF wrist_rz travel is [-1.57, 1.57]; rows with |rz| > 1.57 are
out-of-travel and flagged - they answer "is this just search coverage" vs
"is it reachable in-travel".

Output: runs/w35/w35d_rz_scan.json
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

import w35d_common as C  # noqa: E402

OUT = REPO / "runs/w35/w35d_rz_scan.json"
PI = float(np.pi)

RXS = [0.9, 1.2, 1.5, 1.8]
RYS = [-1.259, -0.9, -0.3, 0.3, 0.9, 1.2]
RZS = [round(v, 4) for v in np.linspace(-PI, PI, 13)]
ZS = [round(v, 4) for v in np.arange(-0.163, 0.1601, 0.04)]
XYS = [round(v, 4) for v in np.arange(-0.06, 0.0601, 0.03)]
CMDS = list(itertools.product((-1.0, -0.5, 0.0, 0.5, 1.0),
                              (-1.0, -0.5, 0.0, 0.5, 1.0),
                              (-1.0, 0.0, 1.0)))
KEEP = 10
TRAVEL = 1.57


def run_rz_chunk(args):
    rz, rxs, rys, cmds, zs, xys, keep = args
    if C.S is None:
        C.init_worker(None)
    tol = C.S["tol"]
    rows = []
    stats = {"n": 0, "ok": 0, "th_below": 0, "sim": 0, "grade": 0, "near": 0}
    for (cl, ct, t0) in cmds:
        fq = C.cmd16(cl, ct, t0)
        for ry in rys:
            for rx in rxs:
                for z in zs:
                    for x in xys:
                        for y in xys:
                            sc = C.joint_score(np.array([x, y, z]),
                                               np.array([rx, ry, rz]), fq)
                            stats["n"] += 1
                            if sc is None:
                                continue
                            stats["ok"] += 1
                            sc["cmd"] = [cl, ct, t0]
                            if sc["th_below"]:
                                stats["th_below"] += 1
                            if sc["th_below"] and sc["long_near"]:
                                stats["sim"] += 1
                            if sc["joint_m"] <= tol:
                                stats["grade"] += 1
                            if sc["joint_m"] <= C.NEAR:
                                stats["near"] += 1
                            rows.append(sc)
    rows.sort(key=lambda r: r["joint_m"])
    return {"rz": rz, "in_travel": bool(abs(rz) <= TRAVEL),
            "stats": stats, "best": rows[:keep]}


def main() -> None:
    t0 = time.time()
    C.init_worker(None)
    tol = C.S["tol"]
    chunks = [(rz, RXS, RYS, CMDS, ZS, XYS, KEEP) for rz in RZS]
    with mp.Pool(processes=8) as pool:
        parts = pool.map(run_rz_chunk, chunks)

    buckets = []
    for p in parts:
        st = p["stats"]
        buckets.append({
            "rz": p["rz"], "in_travel": p["in_travel"], "stats": st,
            "best_joint_m_mm": round(p["best"][0]["joint_m"] * 1000, 4)
            if p["best"] else None,
            "best": p["best"],
        })
        print(f"rz {p['rz']:+.4f} in_travel={p['in_travel']} n={st['n']} "
              f"ok={st['ok']} th_below={st['th_below']} sim={st['sim']} "
              f"grade={st['grade']} near={st['near']} "
              f"best_mm={buckets[-1]['best_joint_m_mm']}", flush=True)

    # per-rz descent with continuous rz from that bucket's best rows
    desc = []
    for p in parts:
        for r in p["best"][:2]:
            sd = {"base": list(r["base"]), "rot": list(r["rot"]),
                  "cmd": list(r["cmd"])}
            C.init_worker(None)
            rr = C.descend(sd, "tied", None)
            if rr is not None:
                rr["seed_rz"] = p["rz"]
                rr["rz_in_travel_after"] = bool(abs(rr["rot"][2]) <= TRAVEL)
                desc.append(rr)
    desc.sort(key=lambda r: r["joint_m"])

    all_grade = [r for r in desc if r["joint_m"] <= tol]
    all_near = [r for r in desc if r["joint_m"] <= C.NEAR]
    out = {
        "wave": "3.5-D task2 rz full-circle scan",
        "date": "2026-10-15",
        "tol_m": tol, "near_m": C.NEAR,
        "grid": {"rxs": RXS, "rys": RYS, "rz_steps": RZS, "zs_n": len(ZS),
                 "xy_n": len(XYS), "cmd_n": len(CMDS),
                 "poses_total": len(RZS) * len(RYS) * len(RXS) * len(ZS)
                 * len(XYS) ** 2 * len(CMDS)},
        "buckets": buckets,
        "descended_best": desc[:12],
        "descend_grade_hits": len(all_grade),
        "descend_near_hits": len(all_near),
        "best_in_travel": next((r for r in desc
                                if abs(r["rot"][2]) <= TRAVEL), None),
        "elapsed_s": round(time.time() - t0, 1),
    }
    OUT.write_text(json.dumps(out, indent=1))
    print("desc best mm:", [(round(r["joint_m"] * 1000, 3), round(r["rot"][2], 3))
                            for r in desc[:8]])
    print("grade/near hits after descent:", len(all_grade), len(all_near))
    print("elapsed", out["elapsed_s"], "s")


if __name__ == "__main__":
    main()
