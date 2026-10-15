"""Wave 3.5-D task 1 quantification: K1 rescan under widened THUMB ranges.

The audit (w35d_range_audit) showed MJCF == canonical URDF exactly, so "range
narrowed vs URDF" is already refuted at source level. This scan closes the
stronger question: is ANY thumb range the binding constraint? Three in-memory
patches (no MJCF edit):

  W0 stock        - thj ranges as committed (== URDF)
  W1 thj0 +/-.7   - opposition opening doubled: thj0 [-0.437, 2.096]
  W2 all x2 span  - every thumb joint span doubled about its centre
  W3 unbounded    - every thumb joint [-pi, pi]

Grid = rx_coupling_diag grid (rx {0.9,1.2,1.5,1.8}, ry {-1.2,-0.9,0.9,1.2},
rz {-0.4,0,0.4}, z 8 steps, xy 5x5, cmd 5x5x3 tied). Then:
  * tied-builder coordinate descent from the best rows of each variant;
  * UNTIED-thumb descent (thj0..3 independent, 5-D command) from the best
    rows - this tests the "opposition cone parameterization" of the SEARCH
    (thj1=thj2=thj3 tied in finger_cmd) rather than the MJCF.

Output: runs/w35/w35d_thumb_rescan.json
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

OUT = REPO / "runs/w35/w35d_thumb_rescan.json"
PI = float(np.pi)

RXS = [0.9, 1.2, 1.5, 1.8]
RYS = [-1.2, -0.9, 0.9, 1.2]
RZS = [-0.4, 0.0, 0.4]
ZS = [round(v, 4) for v in np.arange(-0.163, 0.1601, 0.04)]
XYS = [round(v, 4) for v in np.arange(-0.06, 0.0601, 0.03)]
CMDS = list(itertools.product((-1.0, -0.5, 0.0, 0.5, 1.0),
                              (-1.0, -0.5, 0.0, 0.5, 1.0),
                              (-1.0, 0.0, 1.0)))
KEEP = 12


def variants() -> dict[str, tuple[tuple[float, float], ...] | None]:
    model = mujoco.MjModel.from_xml_path(str(C.XML))
    base = {}
    for n in C.THJ:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
        base[n] = (float(model.jnt_range[jid][0]), float(model.jnt_range[jid][1]))
    w1 = dict(base)
    w1["thj0"] = (base["thj0"][0] - 0.7, base["thj0"][1] + 0.7)
    w2 = {}
    for n, (lo, hi) in base.items():
        c = 0.5 * (lo + hi)
        w2[n] = (c - (hi - lo), c + (hi - lo))
    w3 = {n: (-PI, PI) for n in base}
    return {"W0_stock": None,
            "W1_thj0_pm0.7": tuple(w1[n] for n in C.THJ),
            "W2_all_span_x2": tuple(w2[n] for n in C.THJ),
            "W3_unbounded_pm_pi": tuple(w3[n] for n in C.THJ)}


def run_variant(patch):
    chunks = []
    for rx in RXS:
        chunks.append((patch, [rx], RYS, RZS, CMDS, ZS, XYS, KEEP))
    with mp.Pool(processes=min(8, len(chunks))) as pool:
        parts = pool.map(C.run_chunk, chunks)
    merged = {}
    for p in parts:
        for rx, d in p["per_rx"].items():
            merged[rx] = d
    return {"per_rx": merged}


def seeds_from(res, per_bucket=2):
    seeds = []
    for rx, d in res["per_rx"].items():
        for r in d["best"][:per_bucket]:
            seeds.append({"base": list(r["base"]), "rot": list(r["rot"]),
                          "cmd": list(r["cmd"]), "from_rx": rx})
    return seeds


def main() -> None:
    t0 = time.time()
    vs = variants()
    out = {"wave": "3.5-D task1 thumb-range rescan",
           "date": "2026-10-15",
           "grid": {"rxs": RXS, "rys": RYS, "rzs": RZS, "zs_n": len(ZS),
                    "xy_n": len(XYS), "cmd_n": len(CMDS),
                    "poses_per_variant": len(RXS) * len(RYS) * len(RZS)
                    * len(ZS) * len(XYS) ** 2 * len(CMDS)},
           "variants": {}, "tol_m": None}

    C.init_worker(None)
    out["tol_m"] = C.S["tol"]

    summary = {}
    descended = []
    for tag, patch in vs.items():
        res = run_variant(patch)
        agg = {"n": 0, "ok": 0, "th_below": 0, "sim": 0, "grade": 0, "near": 0}
        best_rows = []
        for rx, d in res["per_rx"].items():
            for k in agg:
                agg[k] += d["stats"][k]
            best_rows.extend(d["best"][:4])
        best_rows.sort(key=lambda r: r["joint_m"])
        out["variants"][tag] = {
            "thumb_patch": None if patch is None else
            {n: [round(lo, 4), round(hi, 4)]
             for n, (lo, hi) in zip(C.THJ, patch)},
            "agg": agg, "best": best_rows[:KEEP],
            "per_rx_stats": {rx: d["stats"] for rx, d in res["per_rx"].items()},
        }
        summary[tag] = {"n": agg["n"], "ok": agg["ok"],
                        "grade": agg["grade"], "near": agg["near"],
                        "sim": agg["sim"],
                        "best_joint_m_mm": round(best_rows[0]["joint_m"] * 1000, 4)
                        if best_rows else None}
        print(tag, json.dumps(summary[tag]), flush=True)
        # descend: tied from this variant's best rows (seed commands are tied
        # triples by construction)
        for sd in seeds_from(res)[:3]:
            C.init_worker(patch)
            r = C.descend(sd, "tied", patch)
            if r is not None:
                r["variant"] = tag
                descended.append(r)

    # untied-thumb descent: full per-joint thumb freedom, STOCK ranges.
    C.init_worker(None)
    untied_seeds = []
    for tag in ("W0_stock", "W2_all_span_x2", "W3_unbounded_pm_pi"):
        for r in out["variants"][tag]["best"][:3]:
            cl, ct, t0 = r["cmd"]
            untied_seeds.append({"base": list(r["base"]), "rot": list(r["rot"]),
                                 "cmd": [cl, t0, ct, ct, ct], "variant": tag})
    untied = []
    for sd in untied_seeds:
        r = C.descend(sd, "untied", None)
        if r is not None:
            untied.append(r)
    untied.sort(key=lambda r: r["joint_m"])
    descended.sort(key=lambda r: r["joint_m"])

    out["descended_tied_best"] = descended[:10]
    out["descended_untied_thumb_best"] = untied[:10]
    out["summary"] = summary
    out["elapsed_s"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps(out["summary"], indent=1))
    print("tied best mm:", [(round(r["joint_m"] * 1000, 3), r["variant"]) for r in descended[:6]])
    print("untied best mm:", [round(r["joint_m"] * 1000, 3) for r in untied[:6]])
    print("elapsed", out["elapsed_s"], "s")


if __name__ == "__main__":
    main()
