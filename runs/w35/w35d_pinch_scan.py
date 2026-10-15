"""Wave 3.5-D final coverage probe: per-finger independent commands.

Every prior scan tied ff/mf/rf to one curl command. A two-finger pinch (ff
down to the +x wall, mf/rf parked clear, thumb to the -x wall) needs
independent assignment. This scan:

  stage 1 grid: ff x mfrf x thumb-curl(t1=t2=t3) x thj0 = 5^4 commands,
      rot rx {0.9,1.2,1.5} x ry {-1.259,-0.9,-0.3,0.3,0.9,1.2} x rz {0},
      z 8, xy 5x5  -> 1.5M poses;
  stage 2: 7-D descent (ff, mfrf, thj0..thj3 independent) from the best rows.

K1-true (side-checked) and joint metric both tracked.
Output: runs/w35/w35d_pinch_scan.json
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

OUT = REPO / "runs/w35/w35d_pinch_scan.json"

RXS = [0.9, 1.2, 1.5]
RYS = [-1.259, -0.9, -0.3, 0.3, 0.9, 1.2]
ZS = [round(v, 4) for v in np.arange(-0.163, 0.1601, 0.04)]
XYS = [round(v, 4) for v in np.arange(-0.06, 0.0601, 0.03)]
GRID5 = (-1.0, -0.5, 0.0, 0.5, 1.0)
CMDS = list(itertools.product(GRID5, GRID5, GRID5, GRID5))  # ff, mfrf, ct, t0
KEEP = 12

S_T = None  # per-process init flag


def fq7(cmd):
    """cmd 4-D (ff, mfrf, ctn, t0n): long-finger split + tied thumb curl.
    cmd 6-D (ff, mfrf, t0n, t1n, t2n, t3n): per-joint thumb. Normalized [-1,1]."""
    ff, rest = cmd[0], cmd[1]
    a = np.empty(16)
    a[0] = ff * 0.3
    a[1:4] = ff
    a[4] = rest * 0.3
    a[5:8] = rest
    a[8] = rest * 0.3
    a[9:12] = rest
    if len(cmd) == 4:
        a[12] = cmd[3]
        a[13:16] = cmd[2]
    else:
        a[12:16] = cmd[2:6]
    return C.q_from_cmd(a)


def score7(base_pos, rot, cmd):
    return C.joint_score(np.asarray(base_pos), np.asarray(rot), fq7(cmd))


def true_k1(sc, tol):
    if sc is None:
        return False
    longs = [st for st in sc["long_state"].values()
             if st.get("below") and st["gap_m"] <= tol]
    if not longs:
        return False
    return bool(sc["th_below"] and sc["th_side"] != 0
                and any(st["side"] == -sc["th_side"] for st in longs)
                and sc["th_gap_m"] <= tol)


def chunk(args):
    cmds, rxs = args
    if C.S is None:
        C.init_worker(None)
    tol = C.S["tol"]
    rows = []
    st = {"n": 0, "ok": 0, "th_below": 0, "grade": 0, "near": 0, "true_k1": 0}
    for c in cmds:
        for ry in RYS:
            for rx in rxs:
                for z in ZS:
                    for x in XYS:
                        for y in XYS:
                            sc = score7((x, y, z), (rx, ry, 0.0), c)
                            st["n"] += 1
                            if sc is None:
                                continue
                            st["ok"] += 1
                            sc["cmd"] = list(c)
                            if sc["th_below"]:
                                st["th_below"] += 1
                            if sc["joint_m"] <= tol:
                                st["grade"] += 1
                            if sc["joint_m"] <= C.NEAR:
                                st["near"] += 1
                            if true_k1(sc, tol):
                                st["true_k1"] += 1
                            rows.append(sc)
    rows.sort(key=lambda r: r["joint_m"])
    return {"rxs": rxs, "stats": st, "best": rows[:KEEP],
            "k1_rows": [r for r in rows if true_k1(r, tol)][:KEEP]}


def descend7(seed):
    s = dict(seed)
    cur = score7(s["base"], s["rot"], s["cmd"])
    if cur is None:
        return None
    steps = [0.02, 0.01, 0.005, 0.002]
    rstep = [0.15, 0.07, 0.03]
    improved = True
    evals = 1
    while improved and evals < 8000:
        improved = False
        for ai in range(3):
            for st in steps:
                for sgn in (1.0, -1.0):
                    cand = dict(s)
                    cand["base"] = list(s["base"])
                    cand["base"][ai] = s["base"][ai] + sgn * st
                    r = score7(cand["base"], cand["rot"], cand["cmd"])
                    evals += 1
                    if r is not None and r["joint_m"] < cur["joint_m"] - 1e-6:
                        s, cur, improved = cand, r, True
        for ai in range(3):
            for st in rstep:
                for sgn in (1.0, -1.0):
                    cand = dict(s)
                    cand["rot"] = list(s["rot"])
                    cand["rot"][ai] = s["rot"][ai] + sgn * st
                    r = score7(cand["base"], cand["rot"], cand["cmd"])
                    evals += 1
                    if r is not None and r["joint_m"] < cur["joint_m"] - 1e-6:
                        s, cur, improved = cand, r, True
        for ai in range(len(s["cmd"])):
            for st in (0.25, 0.1, 0.04):
                for sgn in (1.0, -1.0):
                    cand = dict(s)
                    cand["cmd"] = list(s["cmd"])
                    cand["cmd"][ai] = max(-1.0, min(1.0, s["cmd"][ai] + sgn * st))
                    r = score7(cand["base"], cand["rot"], cand["cmd"])
                    evals += 1
                    if r is not None and r["joint_m"] < cur["joint_m"] - 1e-6:
                        s, cur, improved = cand, r, True
    cur["descend_evals"] = evals
    cur["cmd"] = list(s["cmd"])
    return cur


def main() -> None:
    t0 = time.time()
    chunks = [(CMDS[i::8], [RXS[i % 3]]) for i in range(8)]
    with mp.Pool(processes=8) as pool:
        parts = pool.map(chunk, chunks)
    agg = {"n": 0, "ok": 0, "th_below": 0, "grade": 0, "near": 0, "true_k1": 0}
    best = []
    k1 = []
    for p in parts:
        for k in agg:
            agg[k] += p["stats"][k]
        best.extend(p["best"])
        k1.extend(p["k1_rows"])
    best.sort(key=lambda r: r["joint_m"])
    k1.sort(key=lambda r: r["joint_m"])
    print("stage1:", json.dumps(agg), "best_mm",
          round(best[0]["joint_m"] * 1000, 3) if best else None, flush=True)

    C.init_worker(None)
    tol = C.S["tol"]
    desc = []
    seeds = best[:6] + k1[:4]
    seen = set()
    for r in seeds:
        key = (tuple(r["base"]), tuple(r["rot"]), tuple(r["cmd"]))
        if key in seen:
            continue
        seen.add(key)
        rr = descend7({"base": list(r["base"]), "rot": list(r["rot"]),
                       "cmd": list(r["cmd"])})
        if rr is not None:
            rr["true_k1"] = true_k1(rr, tol)
            desc.append(rr)
    desc.sort(key=lambda r: r["joint_m"])

    out = {
        "wave": "3.5-D pinch probe (per-finger independent commands)",
        "date": "2026-10-15",
        "tol_m": tol,
        "grid": {"ff_x_mfrf_x_ct_x_t0": len(CMDS), "rxs": RXS, "rys": RYS,
                 "z_n": len(ZS), "xy_n": len(XYS), "rz_fixed": 0.0,
                 "poses_total": len(CMDS) * len(RYS) * len(RXS) * len(ZS)
                 * len(XYS) ** 2},
        "stage1": {"stats": agg,
                   "best": best[:KEEP], "k1_rows": k1[:KEEP]},
        "stage2_descended": desc[:12],
        "stage2_grade_hits": sum(1 for r in desc if r["joint_m"] <= tol),
        "stage2_true_k1_hits": sum(1 for r in desc if r["true_k1"]),
        "elapsed_s": round(time.time() - t0, 1),
    }
    OUT.write_text(json.dumps(out, indent=1))
    print("desc best mm:", [(round(r["joint_m"] * 1000, 3), r["true_k1"],
                             [round(v, 2) for v in r["cmd"]]) for r in desc[:8]])
    print("stage2 grade/true-k1:", out["stage2_grade_hits"], out["stage2_true_k1_hits"])
    print("elapsed", out["elapsed_s"], "s")


if __name__ == "__main__":
    main()
