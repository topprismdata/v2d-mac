"""Wave 3.5-D: characterize the thumb-below-top (M1) pose family.

Over the coupling-diag grid restricted to rx in {1.5, 1.8, 2.2, 2.6}, collect
every feasible pose with the thumb pad below the box top and record the thumb
position/side, the long-finger state, and the geometry relative to the box's
y-extent (half 0.12 m). Answers: when the thumb gets under the top, where is
it, and can any long finger simultaneously be at a wall?

Output: runs/w35/w35d_thbelow_rows.json
"""
from __future__ import annotations

import itertools
import json
import multiprocessing as mp
import pathlib
import sys

import numpy as np

REPO = pathlib.Path("/Users/guohongbin/v2d-mac")
sys.path.insert(0, str(REPO / "runs/w35"))

import w35d_common as C  # noqa: E402

OUT = REPO / "runs/w35/w35d_thbelow_rows.json"

RXS = [1.5, 1.8, 2.2, 2.6]
RYS = [-1.2, -0.9, 0.9, 1.2]
RZS = [-0.4, 0.0, 0.4]
ZS = [round(v, 4) for v in np.arange(-0.163, 0.1601, 0.04)]
XYS = [round(v, 4) for v in np.arange(-0.06, 0.0601, 0.03)]
CMDS = list(itertools.product((-1.0, -0.5, 0.0, 0.5, 1.0),
                              (-1.0, -0.5, 0.0, 0.5, 1.0),
                              (-1.0, 0.0, 1.0)))
BOX_Y_HALF = 0.12


def chunk(args):
    cmds, rys, rzs, rxs = args
    if C.S is None:
        C.init_worker(None)
    rows = []
    n = thb = 0
    for cl, ct, t0 in cmds:
        fq = C.cmd16(cl, ct, t0)
        for ry in rys:
            for rz in rzs:
                for rx in rxs:
                    for z in ZS:
                        for x in XYS:
                            for y in XYS:
                                sc = C.joint_score(np.array([x, y, z]),
                                                   np.array([rx, ry, rz]), fq)
                                n += 1
                                if sc is None:
                                    continue
                                if sc["th_below"]:
                                    thb += 1
                                    sc["cmd"] = [cl, ct, t0]
                                    rows.append(sc)
    return {"n": n, "th_below": thb, "rows": rows}


def main() -> None:
    groups = [CMDS[i::8] for i in range(8)]
    chunks = [(g, RYS, RZS, RXS) for g in groups]
    with mp.Pool(processes=8) as pool:
        parts = pool.map(chunk, chunks)
    n = sum(p["n"] for p in parts)
    thb = sum(p["th_below"] for p in parts)
    rows = [r for p in parts for r in p["rows"]]
    rows.sort(key=lambda r: r["th_gap_m"])

    slim = [{k: r[k] for k in ("th_pos", "th_side", "th_gap_m", "long_state",
                               "long_near", "cmd", "rot", "base", "joint_m",
                               "pen_m", "palm_gap_m")} for r in rows[:60]]
    ys = [r["th_pos"][1] for r in rows]
    inside = sum(1 for y in ys if abs(y) <= BOX_Y_HALF)
    near_long = sum(1 for r in rows if r["long_near"])
    out = {
        "wave": "3.5-D M1 family characterization (thumb below top)",
        "date": "2026-10-15",
        "grid": {"rxs": RXS, "rys": RYS, "rzs": RZS, "zs_n": len(ZS),
                 "xy_n": len(XYS), "cmd_n": len(CMDS)},
        "poses": n, "feasible_th_below": thb,
        "th_y_min_mm": round(min(ys) * 1000, 1),
        "th_y_max_mm": round(max(ys) * 1000, 1),
        "th_inside_box_y_extent": inside,
        "rows_with_long_near_wall": near_long,
        "best_rows_by_th_gap": slim,
    }
    OUT.write_text(json.dumps(out, indent=1))
    print(f"poses {n} th-below {thb}; thumb y range [{out['th_y_min_mm']}, "
          f"{out['th_y_max_mm']}] mm; |y|<=120mm: {inside}/{thb}; "
          f"rows with a long finger <=15mm from wall: {near_long}")
    for r in rows[:6]:
        print("  th_gap %7.3f mm | th %s side %+d | long %s" % (
            r["th_gap_m"] * 1000, r["th_pos"], r["th_side"],
            {f: (s.get("below"),
                 round(1000 * (s["gap_m"] if s.get("below") else s["above_top_m"]), 1))
             for f, s in r["long_state"].items()}))


if __name__ == "__main__":
    main()
