"""Wave 3.5-C: K1 kinematic verification on the widened wrist_rx travel.

Method identical to runs/w35/kin_diag.py (static qpos placement + mj_forward +
mj_geomDistance / contact scoring), re-targeted at the one DOF the standoff
wave identified as the binding constraint: wrist_rx was never scanned past
|0.6| (kin_shortfall) or held at 0 (kin_diag), while its travel cap is +/-1.2.

New in this scan:
  * rx grid to +/-1.8 (the widened travel);
  * ablation for free: hits with |rx| <= 1.2 vs |rx| > 1.2 attribute any K1
    success to the widening itself;
  * thumb-below-box-top rate over feasible poses, and the simultaneity count
    (thumb below top AND an opposite-side long-finger pad below top within
    15 mm of its wall) - the metric that was 0.1-0.7% and never simultaneous
    in the standoff wave.

K1 (gate): at least one feasible pose (pen <= R5 tol, palm clear of box) with
pads touching BOTH box x-walls within the R5 tolerance, below the box top.
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

import mujoco  # noqa: E402
import importlib.util  # noqa: E402

XML = REPO / "grounding/assets/allegro_floating.xml"
OUT_JSON = REPO / "runs/w35/rx_widen_scan.json"

spec = importlib.util.spec_from_file_location(
    "baf_w35c", REPO / "grounding/assets/build_allegro_floating.py")
baf = importlib.util.module_from_spec(spec)
sys.modules["baf_w35c"] = baf
spec.loader.exec_module(baf)

mj = mujoco

WRIST = ("wrist_x", "wrist_y", "wrist_z", "wrist_rx", "wrist_ry", "wrist_rz")
FINGERS = tuple(f"{f}j{i}" for f in ("ff", "mf", "rf", "th") for i in range(4))
FROMTO = np.zeros(6)
NEAR = 0.015          # "near the wall" band, same as kin_diag
CAP = 0.05


def _setup():
    model = mj.MjModel.from_xml_path(str(XML))
    data = mj.MjData(model)
    tol = baf.penetration_tolerance(mjcf_path=XML)
    wqadr = np.array([model.jnt_qposadr[mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, n)]
                      for n in WRIST], dtype=int)
    fqadr = np.array([model.jnt_qposadr[mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, n)]
                      for n in FINGERS], dtype=int)
    flo = np.array([model.jnt_range[mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, n)][0]
                    for n in FINGERS])
    fhi = np.array([model.jnt_range[mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, n)][1]
                    for n in FINGERS])
    box = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, "object_geom")
    palm = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, "palm_collision")
    obj_body = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "object")
    pads = {}
    for f in ("ff", "mf", "rf", "th"):
        bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, f"{f}_tip")
        pads[f] = [g for g in range(model.ngeom)
                   if model.geom_bodyid[g] == bid
                   and model.geom_type[g] == mj.mjtGeom.mjGEOM_CAPSULE][0]
    return dict(model=model, data=data, tol=tol, wqadr=wqadr, fqadr=fqadr,
                flo=flo, fhi=fhi, box=box, palm=palm, obj_body=obj_body, pads=pads)


S: dict | None = None


def _init():
    global S
    S = _setup()


def gdist(g1: int, g2: int, distmax: float = 0.06) -> float:
    return float(mj.mj_geomDistance(S["model"], S["data"], g1, g2, distmax, FROMTO))


def finger_cmd(cl: float, ct: float, t0: float) -> np.ndarray:
    a = np.empty(16)
    for k, f in enumerate(("ff", "mf", "rf")):
        a[4 * k: 4 * k + 4] = cl
        a[4 * k] = cl * 0.3
    a[12:16] = ct
    a[12] = t0
    return S["flo"] + (a + 1.0) * 0.5 * (S["fhi"] - S["flo"])


def score(base, rot, fcmd):
    model, data = S["model"], S["data"]
    box, palm, pads = S["box"], S["palm"], S["pads"]
    tol = S["tol"]
    data.qpos[:] = model.qpos0
    data.qpos[S["wqadr"]] = np.concatenate([base, rot])
    data.qpos[S["fqadr"]] = fcmd
    mj.mj_forward(model, data)
    bp = data.xpos[S["obj_body"]]
    box_top = float(bp[2] + model.geom_size[box][2])
    pen = 0.0
    for i in range(data.ncon):
        c = data.contact[i]
        if -float(c.dist) > pen:
            pen = -float(c.dist)
    gap = {f: gdist(g, box) for f, g in pads.items()}
    side, below = {}, {}
    neg_near, pos_near, neg_t, pos_t = [], [], [], []
    for f in pads:
        side[f] = int(np.sign(float(data.geom_xpos[pads[f]][0]) - float(bp[0])))
        below[f] = bool(float(data.geom_xpos[pads[f]][2]) < box_top)
        if not below[f]:
            continue
        if side[f] < 0:
            if gap[f] <= NEAR:
                neg_near.append(f)
            if gap[f] <= tol:
                neg_t.append(f)
        elif side[f] > 0:
            if gap[f] <= NEAR:
                pos_near.append(f)
            if gap[f] <= tol:
                pos_t.append(f)
    palm_gap = gdist(palm, box, 0.15)
    ok = pen <= tol and palm_gap > 0.0
    both_near = bool(neg_near and pos_near)
    both_touch = bool(neg_t and pos_t)
    # ranking key, same semantics as kin_diag: 0 for a touch hit, 0.5 for
    # near-only, else the worse of the two sides' best near-gap
    if both_touch:
        key = 0.0
    elif both_near:
        key = 0.5
    else:
        key = max(min((gap[f] for f in neg_near), default=1.0),
                  min((gap[f] for f in pos_near), default=1.0))
    # simultaneity: thumb pad below the top AND some opposite-side long finger
    # below the top and within 15 mm of its wall
    sim = bool(below["th"] and neg_near and
               any(below[f] for f in ("ff", "mf", "rf")))
    return {
        "gap": gap, "side": side, "below": below,
        "pen": pen, "palm_gap": palm_gap,
        "neg_touch": neg_t, "pos_touch": pos_t,
        "neg_near": neg_near, "pos_near": pos_near,
        "ok": ok, "both_near": both_near, "both_touch": both_touch, "sim": sim,
        "key": key,
    }


def run_chunk(args):
    rx, grid = args
    n = n_ok = n_both_near = n_both_touch = n_th_below = n_sim = 0
    hits, sim_rows, best_near = [], [], None
    for cl in grid["CLS"]:
        for ct in grid["CTS"]:
            for t0 in grid["T0S"]:
                fcmd = finger_cmd(cl, ct, t0)
                for ry in grid["RYS"]:
                    for rz in grid["RZS"]:
                        rot = np.array([rx, ry, rz])
                        for z in grid["ZS"]:
                            for x in grid["XYS"]:
                                for y in grid["XYS"]:
                                    sc = score(np.array([x, y, z]), rot, fcmd)
                                    n += 1
                                    if not sc["ok"]:
                                        continue
                                    n_ok += 1
                                    if sc["both_near"]:
                                        n_both_near += 1
                                    if sc["both_touch"]:
                                        n_both_touch += 1
                                        if len(hits) < 40:
                                            hits.append({
                                                "base": [float(x), float(y), float(z)],
                                                "rot": [float(rx), float(ry), float(rz)],
                                                "cmd": [cl, ct, t0],
                                                "gap_m": {k: float(v) for k, v in sc["gap"].items()},
                                                "pen_m": float(sc["pen"]),
                                                "palm_gap_m": float(sc["palm_gap"]),
                                                "touch": [sc["neg_touch"], sc["pos_touch"]],
                                                "key": float(sc["key"]),
                                            })
                                    if sc["below"]["th"]:
                                        n_th_below += 1
                                    if sc["sim"]:
                                        n_sim += 1
                                        if len(sim_rows) < 20:
                                            sim_rows.append({
                                                "base": [float(x), float(y), float(z)],
                                                "rot": [float(rx), float(ry), float(rz)],
                                                "cmd": [cl, ct, t0],
                                                "th_gap_m": float(sc["gap"]["th"]),
                                                "long_near": sc["pos_near"],
                                                "pen_m": float(sc["pen"]),
                                            })
                                    if sc["both_near"] and (best_near is None or
                                                            sc["key"] < best_near["key"]):
                                        best_near = {
                                            "base": [float(x), float(y), float(z)],
                                            "rot": [float(rx), float(ry), float(rz)],
                                            "cmd": [cl, ct, t0],
                                            "key": float(sc["key"]),
                                            "gap_m": {k: float(v) for k, v in sc["gap"].items()},
                                            "pen_m": float(sc["pen"]),
                                            "palm_gap_m": float(sc["palm_gap"]),
                                        }
    return {"rx": rx, "n": n, "ok": n_ok, "both_near": n_both_near,
            "both_touch": n_both_touch, "th_below": n_th_below, "sim": n_sim,
            "hits": hits, "sim_rows": sim_rows, "best_near": best_near}


def main() -> None:
    t0 = time.time()
    grid = {
        "CLS": (-1.0, -0.5, 0.0, 0.5, 1.0),
        "CTS": (-1.0, -0.5, 0.0, 0.5, 1.0),
        "T0S": (-1.0, 0.0, 1.0),
        "RYS": (-1.259, -0.9, 0.9, 1.2),
        "RZS": (-0.4, 0.0, 0.4),
        "ZS": np.arange(-0.163, 0.1601, 0.04),
        "XYS": np.arange(-0.06, 0.0601, 0.03),
    }
    RXS = [round(v, 3) for v in np.linspace(-1.8, 1.8, 13)]
    per_rx = (len(grid["CLS"]) * len(grid["CTS"]) * len(grid["T0S"])
              * len(grid["RYS"]) * len(grid["RZS"]) * len(grid["ZS"])
              * len(grid["XYS"]) ** 2)
    print(f"rx grid: {RXS}\nposes per rx: {per_rx}, total: {per_rx * len(RXS)}",
          flush=True)

    with mp.Pool(processes=min(8, len(RXS)), initializer=_init) as pool:
        results = pool.map(run_chunk, [(rx, grid) for rx in RXS])

    tol = baf.penetration_tolerance(mjcf_path=XML)
    total = sum(r["n"] for r in results)
    agg = {k: sum(r[k] for r in results)
           for k in ("ok", "both_near", "both_touch", "th_below", "sim")}
    hits = sorted([h for r in results for h in r["hits"]],
                  key=lambda h: h["key"])
    sim_rows = sorted([h for r in results for h in r["sim_rows"]],
                      key=lambda h: h["th_gap_m"])
    best_near = [r["best_near"] for r in results if r["best_near"]]
    best_near = sorted(best_near, key=lambda h: h["key"])[:5]
    old_travel_hits = [h for h in hits
                       if abs(h["rot"][0]) <= 1.2 + 1e-9]
    widened_only = [h for h in hits if abs(h["rot"][0]) > 1.2 + 1e-9]

    out = {
        "wave": "3.5-C wrist_rx travel widening",
        "mjcf": "grounding/assets/allegro_floating.xml",
        "r5_tolerance_m": tol,
        "method": "kin_diag.py scoring, rx grid extended to +/-1.8 (linspace 13), "
                  "ry in {-1.259,-0.9,0.9,1.2}, rz in {-0.4,0,0.4}, z 9 x xy 5x5, "
                  "finger cmds 5x5x3; feasible = pen<=R5tol and palm_gap>0 (K2/K3 embedded)",
        "placements": total,
        "feasible_ok": agg["ok"],
        "both_sides_near": agg["both_near"],
        "both_sides_touch_K1": agg["both_touch"],
        "thumb_below_top_when_feasible": agg["th_below"],
        "thumb_below_top_rate_when_feasible": (agg["th_below"] / agg["ok"]) if agg["ok"] else 0.0,
        "thumb_and_opposite_long_near_simultaneous": agg["sim"],
        "K1_pass": agg["both_touch"] > 0,
        "k1_hits_inside_old_travel": len(old_travel_hits),
        "k1_hits_requiring_widening": len(widened_only),
        "best_k1_hits": hits[:10],
        "simultaneity_rows": sim_rows[:10],
        "best_both_near_poses": best_near,
        "per_rx": [{k: r[k] for k in ("rx", "n", "ok", "both_near", "both_touch",
                                      "th_below", "sim")} for r in results],
        "elapsed_s": round(time.time() - t0, 1),
    }
    OUT_JSON.write_text(json.dumps(out, indent=2))
    slim = {k: v for k, v in out.items()
            if k not in ("best_k1_hits", "simultaneity_rows", "best_both_near_poses")}
    print(json.dumps(slim, indent=2), flush=True)
    if hits:
        print("best K1 hit:", json.dumps(hits[0], indent=2))


if __name__ == "__main__":
    main()
