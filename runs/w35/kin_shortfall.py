"""Last-gap kinematic scan: pitch (ry) x roll (rx) combined, with a proper
simultaneous-both-sides objective and the thumb's vertical shortfall as the
failure-mode metric. Complements runs/w35/kin_diag.py (which fixed rx=0).

For every feasible command pose:
  long_metric = min gap among ff/mf/rf pads below the box top on their wall
                (capped at 0.05 m); +inf shortfalls recorded separately
  th_metric   = th pad gap if below the box top, else (th_z - box_top), i.e.
                the vertical distance the thumb still sits ABOVE the top
  joint       = max(long_metric, th_metric)   -> minimised

    python runs/w35/kin_shortfall.py <h_mm> <xml>
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

REPO = pathlib.Path("/Users/guohongbin/v2d-mac")
sys.path.insert(0, str(REPO))

import mujoco  # noqa: E402
import importlib.util  # noqa: E402

H_MM = int(sys.argv[1])
XML = REPO / sys.argv[2]
OUT_JSON = REPO / f"runs/w35/kinshort_ped{H_MM}.json"

spec = importlib.util.spec_from_file_location(
    "baf_sf", REPO / "grounding/assets/build_allegro_floating.py")
baf = importlib.util.module_from_spec(spec)
sys.modules["baf_sf"] = baf
spec.loader.exec_module(baf)

TOL = baf.penetration_tolerance(mjcf_path=XML)
mj = mujoco
model = mj.MjModel.from_xml_path(str(XML))
data = mj.MjData(model)

WRIST = ("wrist_x", "wrist_y", "wrist_z", "wrist_rx", "wrist_ry", "wrist_rz")
wqadr = np.array([model.jnt_qposadr[mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, n)]
                  for n in WRIST], dtype=int)
FINGERS = tuple(f"{f}j{i}" for f in ("ff", "mf", "rf", "th") for i in range(4))
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
FROMTO = np.zeros(6)
CAP = 0.05


def gdist(g1, g2, distmax=0.06):
    return float(mj.mj_geomDistance(model, data, g1, g2, distmax, FROMTO))


def finger_cmd(cl, ct, t0):
    a = np.empty(16)
    for k in range(3):
        a[4 * k: 4 * k + 4] = cl
        a[4 * k] = cl * 0.3
    a[12:16] = ct
    a[12] = t0
    return flo + (a + 1.0) * 0.5 * (fhi - flo)


def main():
    CLS = (-1.0, 0.0, 0.5, 1.0)
    CTS = (-1.0, 0.0, 0.5, 1.0)
    T0S = (-1.0, 0.5, 1.0)
    RYS = (-1.2, -0.9, 0.9, 1.2)
    RXS = (-0.6, 0.0, 0.6)
    RZS = (-0.4, 0.0, 0.4)
    zs = np.arange(-0.163, 0.1601, 0.05)
    xys = np.arange(-0.06, 0.0601, 0.04)

    n = n_ok = 0
    best = None
    cond_best = None   # conditional: some long finger within 15 mm of its wall
    for cl in CLS:
        for ct in CTS:
            for t0 in T0S:
                fcmd = finger_cmd(cl, ct, t0)
                for ry in RYS:
                    for rx in RXS:
                        for rz in RZS:
                            rot = np.array([rx, ry, rz])
                            for z in zs:
                                for x in xys:
                                    for y in xys:
                                        data.qpos[:] = model.qpos0
                                        data.qpos[wqadr] = np.concatenate(
                                            [np.array([x, y, z]), rot])
                                        data.qpos[fqadr] = fcmd
                                        mj.mj_forward(model, data)
                                        n += 1
                                        bp = data.xpos[obj_body]
                                        top = float(bp[2] + model.geom_size[box][2])
                                        pen = 0.0
                                        for i in range(data.ncon):
                                            c = data.contact[i]
                                            pen = max(pen, -float(c.dist))
                                        if pen > TOL:
                                            continue
                                        if gdist(palm, box, 0.15) <= 0.0:
                                            continue
                                        n_ok += 1
                                        long_gap, long_short = CAP, CAP
                                        long_near = False
                                        for f in ("ff", "mf", "rf"):
                                            g = data.geom_xpos[pads[f]]
                                            below = bool(g[2] < top)
                                            d = gdist(pads[f], box)
                                            if below:
                                                long_gap = min(long_gap, d, CAP)
                                                if d <= 0.015:
                                                    long_near = True
                                            else:
                                                long_short = min(
                                                    long_short,
                                                    float(g[2] - top), CAP)
                                        tg = data.geom_xpos[pads["th"]]
                                        th_below = bool(tg[2] < top)
                                        th_d = gdist(pads["th"], box)
                                        th_metric = (min(th_d, CAP) if th_below
                                                     else min(float(tg[2] - top), CAP))
                                        joint = max(long_gap, th_metric)
                                        row = {
                                            "base": [float(x), float(y), float(z)],
                                            "rot": [float(rx), float(ry), float(rz)],
                                            "cmd": [cl, ct, t0],
                                            "long_gap_m": long_gap,
                                            "long_short_m": long_short,
                                            "th_below": th_below,
                                            "th_metric_m": th_metric,
                                            "joint_m": joint,
                                            "pen_m": pen,
                                        }
                                        if best is None or joint < best["joint_m"]:
                                            best = row
                                        if long_near and (cond_best is None or
                                                          th_metric < cond_best["th_metric_m"]):
                                            cond_best = row
    out = {
        "pedestal_h_mm": H_MM,
        "mjcf": str(XML.relative_to(REPO)),
        "r5_tolerance_m": TOL,
        "placements": n,
        "feasible_ok": n_ok,
        "best_joint": best,
        "conditional_long_near_best_th": cond_best,
    }
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"ped{H_MM}: placements={n} feasible={n_ok}")
    print("  best_joint:", json.dumps(best))
    print("  cond_long_near_best_th:", json.dumps(cond_best))


if __name__ == "__main__":
    main()
