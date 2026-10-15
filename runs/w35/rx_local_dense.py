"""Wave 3.5-C: dense local probe around the descended near-miss family.

rx_coupling_diag.py's descent reached joint=0.33 mm (thumb 0.33 mm ABOVE the
box top, ff pad 0.31 mm from its wall). K1 needs the thumb pad BELOW the top
and both sides within R5 TOL (0.116 mm). This probe densely samples the local
neighbourhood of the three best descended poses so the K1 negative is not an
artefact of grid coarseness.
"""
from __future__ import annotations

import itertools
import json
import pathlib
import sys
import time

import numpy as np

REPO = pathlib.Path("/Users/guohongbin/v2d-mac")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "runs/w35"))

import rx_coupling_diag as diag  # noqa: E402  (reuses joint_score)
import rx_widen_scan as base  # noqa: E402

OUT_JSON = REPO / "runs/w35/rx_local_dense.json"


def stage1():
    """base x rot dense, cmd at centre; three centres."""
    centres = [
        {"base": [0.06, 0.035, -0.012], "rot": [1.2, 1.2, 0.0], "cmd": [-1.0, -0.5, 0.0]},
        {"base": [0.058, 0.041, -0.005], "rot": [1.2, 1.2, 0.0], "cmd": [-1.0, -0.5, 0.0]},
        {"base": [0.06, -0.06, -0.013], "rot": [0.9, 1.2, 0.4], "cmd": None},
    ]
    # fill missing cmd from the pre-descent grid rows (descend() rows carry cmd=None)
    d = json.load(open(REPO / "runs/w35/rx_coupling_diag.json"))
    if centres[2]["cmd"] is None:
        for entry in d["per_rx"]:
            for row in entry["best_joint"]:
                if (abs(row["rot"][0] - 0.9) < 1e-6 and abs(row["rot"][2] - 0.4) < 1e-6
                        and abs(row["base"][1] - (-0.06)) < 2e-3):
                    centres[2]["cmd"] = row["cmd"]
                    break
            if centres[2]["cmd"]:
                break
        assert centres[2]["cmd"], "no seed row with cmd found for centre 3"
    bases = np.arange(-0.025, 0.0251, 0.005)
    rots = np.arange(-0.2, 0.201, 0.1)
    rows, hits = [], []
    n = 0
    for c in centres:
        fcmd = base.finger_cmd(*c["cmd"])
        cb, cr = np.array(c["base"]), np.array(c["rot"])
        for db in itertools.product(bases, bases, bases):
            for dr in itertools.product(rots, rots, rots):
                sc = diag.joint_score(cb + np.array(db), cr + np.array(dr), fcmd)
                n += 1
                if sc is None:
                    continue
                sc["cmd"] = list(c["cmd"])
                rows.append(sc)
                if sc["th_below"] and sc["long_gap_m"] <= base.S["tol"]:
                    hits.append(sc)
    rows.sort(key=lambda r: r["joint_m"])
    return n, rows, hits, centres


def stage2(rows):
    """cmd fine scan around the best stage-1 rows."""
    cmds = np.arange(-0.25, 0.251, 0.125)
    out, n = [], 0
    for r in rows[:20]:
        b, rt = np.array(r["base"]), np.array(r["rot"])
        c0 = np.array(r["cmd"])
        for dc in itertools.product(cmds, cmds, cmds):
            c = np.clip(c0 + np.array(dc), -1.0, 1.0)
            sc = diag.joint_score(b, rt, base.finger_cmd(*c))
            n += 1
            if sc is None:
                continue
            sc["cmd"] = [float(v) for v in c]
            out.append(sc)
    out.sort(key=lambda r: r["joint_m"])
    return n, out


def main() -> None:
    t0 = time.time()
    base._init()
    tol = base.S["tol"]
    n1, rows1, hits1, centres = stage1()
    print(f"stage1: {n1} poses, feasible {len(rows1)}, "
          f"K1-grade hits (th below top AND long side <=TOL): {len(hits1)}", flush=True)
    best1 = rows1[0]
    print(f"stage1 best joint={best1['joint_m']*1000:.3f}mm long={best1['long_gap_m']*1000:.3f}mm "
          f"th={'below' if best1['th_below'] else 'ABOVE'} {best1['th_metric_m']*1000:.3f}mm", flush=True)
    n2, rows2 = stage2(rows1[:20])
    print(f"stage2: {n2} poses", flush=True)
    best2 = rows2[0]
    print(f"stage2 best joint={best2['joint_m']*1000:.3f}mm long={best2['long_gap_m']*1000:.3f}mm "
          f"th={'below' if best2['th_below'] else 'ABOVE'} {best2['th_metric_m']*1000:.3f}mm "
          f"touch=[{best2['long_near']}]", flush=True)
    k1_grade = [r for r in rows1 + rows2
                if r["th_below"] and r["long_gap_m"] <= tol and r["th_gap_m"] <= tol]
    out = {
        "wave": "3.5-C dense local probe around descended near-miss family",
        "r5_tolerance_m": tol,
        "centres": centres,
        "stage1_poses": n1, "stage1_feasible": len(rows1),
        "stage2_poses": n2,
        "K1_grade_hits": len(k1_grade),
        "K1_pass": len(k1_grade) > 0,
        "stage1_best": rows1[:10],
        "stage2_best": rows2[:10],
        "best_th_below_rows": sorted([r for r in rows1 + rows2 if r["th_below"]],
                                     key=lambda r: r["th_gap_m"])[:10],
        "elapsed_s": round(time.time() - t0, 1),
    }
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items()
                      if k not in ("stage1_best", "stage2_best", "best_th_below_rows", "centres")},
                     indent=2))


if __name__ == "__main__":
    main()
