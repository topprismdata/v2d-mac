"""Focused kinematic diagnostic: can ANY command pose pinch the box bilaterally?

Extends runs/w35/kinematics.py with an independent thumb-abduction dimension
(the first grid tied thj0 to the thumb curl scalar and never gave it an extreme),
a finer yaw set, and a control run on the committed 60 mm scene whose known
truth (clean_lift_verdict.json: opposite-wall pad pinch = 0/60000) validates
the detector itself.

    python runs/w35/kin_diag.py <h_mm> <xml-relative-to-repo>
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

REPO = pathlib.Path("/Users/guohongbin/v2d-mac")
sys.path.insert(0, str(REPO))

import mujoco  # noqa: E402

H_MM = int(sys.argv[1])
XML = REPO / sys.argv[2]
OUT_JSON = REPO / f"runs/w35/kindiag_ped{H_MM}.json"

import importlib.util  # noqa: E402
spec = importlib.util.spec_from_file_location(
    "baf_diag", REPO / "grounding/assets/build_allegro_floating.py")
baf = importlib.util.module_from_spec(spec)
sys.modules["baf_diag"] = baf
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
pedestal = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, "pedestal")
obj_body = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "object")
pads = {}
for f in ("ff", "mf", "rf", "th"):
    bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, f"{f}_tip")
    gids = [g for g in range(model.ngeom)
            if model.geom_bodyid[g] == bid
            and model.geom_type[g] == mj.mjtGeom.mjGEOM_CAPSULE]
    pads[f] = gids[0]
FROMTO = np.zeros(6)
NEAR = 0.015


def gdist(g1: int, g2: int, distmax: float = 0.06) -> float:
    return float(mj.mj_geomDistance(model, data, g1, g2, distmax, FROMTO))


def finger_cmd(cl: float, ct: float, t0: float) -> np.ndarray:
    """a-space command; the thumb's j0 abduction is independent."""
    a = np.empty(16)
    for k, f in enumerate(("ff", "mf", "rf")):
        a[4 * k: 4 * k + 4] = cl
        a[4 * k] = cl * 0.3
    a[12:16] = ct
    a[12] = t0
    return flo + (a + 1.0) * 0.5 * (fhi - flo)


def score(base, rot, fcmd):
    data.qpos[:] = model.qpos0
    data.qpos[wqadr] = np.concatenate([base, rot])
    data.qpos[fqadr] = fcmd
    mj.mj_forward(model, data)
    bp = data.xpos[obj_body]
    box_top = float(bp[2] + model.geom_size[box][2])
    pen = 0.0
    for i in range(data.ncon):
        c = data.contact[i]
        if -float(c.dist) > pen:
            pen = -float(c.dist)
    gap = {f: gdist(g, box) for f, g in pads.items()}
    info = {"side": {}, "below": {}}
    neg_near, pos_near, neg_t, pos_t = [], [], [], []
    for f in pads:
        info["side"][f] = int(np.sign(float(data.geom_xpos[pads[f]][0]) - float(bp[0])))
        info["below"][f] = bool(float(data.geom_xpos[pads[f]][2]) < box_top)
        if not info["below"][f]:
            continue
        if info["side"][f] < 0:
            if gap[f] <= NEAR:
                neg_near.append(f)
            if gap[f] <= TOL:
                neg_t.append(f)
        elif info["side"][f] > 0:
            if gap[f] <= NEAR:
                pos_near.append(f)
            if gap[f] <= TOL:
                pos_t.append(f)
    palm_gap = gdist(palm, box, 0.15)
    return {
        "gap": gap, **info, "pen": pen, "palm_gap": palm_gap,
        "neg_near": neg_near, "pos_near": pos_near,
        "neg_touch": neg_t, "pos_touch": pos_t,
        "ok": pen <= TOL and palm_gap > 0.0,
        "both_near": bool(neg_near and pos_near),
        "both_touch": bool(neg_t and pos_t),
        "key": (0.0 if (neg_t and pos_t) else
                (0.5 if (neg_near and pos_near) else
                 max(min((gap[f] for f in neg_near), default=1.0),
                     min((gap[f] for f in pos_near), default=1.0)))),
    }


def main() -> None:
    CLS = (-1.0, -0.5, 0.0, 0.5, 1.0)
    CTS = (-1.0, -0.5, 0.0, 0.5, 1.0)
    T0S = (-1.0, -0.5, 0.5, 1.0)
    RYS = (-1.2, -0.9, -0.6, 0.6, 0.9, 1.2)
    RZS = (-0.4, 0.0, 0.4)
    zs = np.arange(-0.163, 0.1601, 0.04)
    xys = np.arange(-0.06, 0.0601, 0.03)

    n = 0
    n_ok = 0
    n_both_near = 0
    n_both_touch = 0
    best = None
    thumb_below_any = 0
    for cl in CLS:
        for ct in CTS:
            for t0 in T0S:
                fcmd = finger_cmd(cl, ct, t0)
                for ry in RYS:
                    for rz in RZS:
                        rot = np.array([0.0, ry, rz])
                        for z in zs:
                            for x in xys:
                                for y in xys:
                                    sc = score(np.array([x, y, z]), rot, fcmd)
                                    n += 1
                                    if not sc["ok"]:
                                        continue
                                    n_ok += 1
                                    if sc["both_near"]:
                                        n_both_near += 1
                                    if sc["both_touch"]:
                                        n_both_touch += 1
                                    if sc["below"]["th"]:
                                        thumb_below_any += 1
                                    if best is None or sc["key"] < best["key"]:
                                        best = sc
    out = {
        "pedestal_h_mm": H_MM,
        "mjcf": str(XML.relative_to(REPO)),
        "r5_tolerance_m": TOL,
        "placements": n,
        "feasible_ok": n_ok,
        "both_sides_near": n_both_near,
        "both_sides_touch": n_both_touch,
        "thumb_below_top_when_feasible": thumb_below_any,
        "best": best,
    }
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "best"}, indent=2),
          flush=True)
    if best:
        b = best
        print(f"best key={b['key']*1000:.2f}mm gap_mm="
              f"{ {k: round(v*1000,2) for k,v in b['gap'].items()} } "
              f"below={b['below']} pen={b['pen']*1e3:.3f}mm palm={b['palm_gap']*1e3:.1f}mm")


if __name__ == "__main__":
    main()
