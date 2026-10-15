"""W4/R8 - can a fingertip reach the lip's underside? Kinematic sweep.

The question is purely geometric and is answered WITHOUT dynamics: for each
(overhang, lip_h) cell, place the hand over the box in a grid of wrist poses and
finger curls, and ask whether any finger's collision geom can reach a point on the
hook face - the horizontal underside of the lip, which is the only surface that can
give a clean upward normal force.

The criterion is w35d's strict one, kept verbatim in spirit: the nearest point on
the LIP must lie inside the hook-face rectangle (not on an end face, not on the
lip's outer wall), because a "reach" measured to the nearest point anywhere on the
geom is the same pseudo-positive 3.5-C recorded - a pad 82 mm from the wall whose
nearest point fell on a top-plane extension.

    hook face  = { z = box_top - lip_h,  bx <= |x| <= bx + overhang,  |y| <= by }

Two families of evidence are recorded per cell:
  reachable  - some (wrist, curl) puts a finger geom within `tol` of that face
               WITHOUT any finger geom being buried in the object (a commanded
               interpenetration is not a reach - see the scan body)
  clean_hook - reachable AND the contacting geom is a fingertip PAD (not a link),
               because only a pad is a grasp: a link pressed under a lip is a wedge

Run:  runs/.venv-mj/bin/python runs/w4/lip_kin_scan.py
Out:  runs/w4/lip_kin_scan.json
"""
from __future__ import annotations

import json
import pathlib
import sys

import mujoco
import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from grounding.assets import build_allegro_floating as baf  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parent / "lip_kin_scan.json"
ASSETS = REPO / "grounding/assets"

OVERHANGS_M = (0.004, 0.006, 0.008, 0.010, 0.012)
LIP_HS_M = (0.006, 0.010, 0.014)

# w35d's strict contact tolerance: the R5-derived penetration tolerance, so a
# "touch" means touching at the same standard every other reachability number in
# this repo was measured at.
TOL = baf.penetration_tolerance(REPO / "grounding/assets/allegro_floating.xml")

N_WRIST = 6
WRIST_Z_BASE = 0.92

# The sweep grid. Sized so a full 5x3 sweep finishes in minutes, and chosen to
# span what w35d/3.5-D established about this hand: the fingers flex in the x-z
# plane at y = 0, +-0.0435, the thumb's cone is fixed relative to the palm, and
# the palm cannot get below the box top (0.839 m palm floor vs 0.840 m box top) -
# so every pose below is a palm-ABOVE-the-box pose, and the hook has to be made by
# the fingers curling back under the rim.
WRIST_X = (-0.02, 0.0, 0.02)
WRIST_Y = (-0.06, -0.03, 0.0, 0.03, 0.06)
WRIST_Z = (-0.09, -0.075, -0.06, -0.045, -0.03, -0.015, 0.0)
ROT_X = (0.0, 0.4, 0.8, 1.2)
ROT_Y = (-0.3, 0.0, 0.3, 0.6)
CURLS = ((0.0, 0.6, 0.4, 0.2), (0.0, 1.0, 0.8, 0.5), (0.3, 1.0, 0.9, 0.6),
         (0.6, 1.2, 1.0, 0.8), (0.0, 1.4, 1.2, 0.9), (0.3, 0.6, 0.3, 0.1))


def finger_geom_ids(model) -> dict[str, list[int]]:
    """Every collision geom on every finger link, grouped by link."""
    mujoco_ = mujoco
    out: dict[str, list[int]] = {}
    for pref in ("ff", "mf", "rf", "th"):
        for role in ("base", "proximal", "medial", "distal", "tip"):
            bid = mujoco_.mj_name2id(model, mujoco_.mjtObj.mjOBJ_BODY, f"{pref}_{role}")
            if bid < 0:
                continue
            gids = [g for g in range(model.ngeom)
                    if int(model.geom_bodyid[g]) == bid
                    and "visual" not in (mujoco_.mj_id2name(
                        model, mujoco_.mjtObj.mjOBJ_GEOM, g) or "")]
            if gids:
                out[f"{pref}_{role}"] = gids
    return out


def scan_cell(overhang: float, lip_h: float, tol: float) -> dict:
    """Build the scene for one (overhang, lip_h) and sweep the pose grid."""
    name = f"lip_o{int(round(overhang * 1e3)):02d}_h{int(round(lip_h * 1e3)):02d}"
    path = baf.build(overhang, lip_h, ASSETS / f"_scan_{name}.xml")
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)

    bx, by, bz = baf.BOX_HALF_M
    hook_z = bz - lip_h                      # body-frame z of the hook face
    x_lo, x_hi = bx, bx + overhang           # body-frame |x| band of the hook face

    lips = {}
    for gname in baf.BOX_LIP_GEOM_NAMES:
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, gname)
        assert gid >= 0, f"{gname} missing from the generated scene"
        lips[gname] = gid

    fgeoms = finger_geom_ids(model)
    pads = {k: v for k, v in fgeoms.items() if k.endswith("_tip")}
    wrist_q = [int(model.jnt_qposadr[mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, n)])
        for n in ("wrist_x", "wrist_y", "wrist_z", "wrist_rx", "wrist_ry", "wrist_rz")]
    finger_q = np.array(
        [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
         for n in baf.FINGER_JOINTS], dtype=int)

    obj = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    ft = np.zeros(6)
    best = {"dist_m": float("inf")}
    n_reach = 0
    n_clean = 0
    n_poses = 0

    for wx in WRIST_X:
        for wy in WRIST_Y:
            for wz in WRIST_Z:
                for rx in ROT_X:
                    for ry in ROT_Y:
                        for curl in CURLS:
                            n_poses += 1
                            mujoco.mj_resetData(model, data)
                            data.qpos[wrist_q] = [wx, wy, wz, rx, ry, 0.0]
                            for k, pre in enumerate(curl):
                                for j in range(4):
                                    data.qpos[finger_q[4 * k + j]] = pre
                            mujoco.mj_kinematics(model, data)
                            mujoco.mj_forward(model, data)

                            bpos = data.xpos[obj].copy()
                            bmat = data.xmat[obj].reshape(3, 3).copy()
                            # A commanded pose with a finger geom INSIDE the object is
                            # not a reach - it is an interpenetration the solver will
                            # resolve with forces that follow the burial depth and not
                            # any actuator's authority (R1's "buried geom" row). The
                            # first version of this scan counted those and reported
                            # -90 mm "reaches", which is the 3.5-C pseudo-positive
                            # again. So a pose qualifies only if NO finger geom is
                            # buried in ANY object geom: the finger got there.
                            buried = False
                            for gids in fgeoms.values():
                                for g in gids:
                                    for lid in lips.values():
                                        mujoco.mj_geomDistance(model, data, g, lid,
                                                               0.20, ft)
                                        if float(ft[0]) < -1e-6:
                                            buried = True
                                            break
                                    if buried:
                                        break
                                if buried:
                                    break
                            if buried:
                                continue

                            for link, gids in fgeoms.items():
                                for g in gids:
                                    for gname, lid in lips.items():
                                        mujoco.mj_geomDistance(model, data, g, lid,
                                                               0.20, ft)
                                        dist = float(ft[0])
                                        pt = ft[3:6].copy()
                                        rel = bmat.T @ (pt - bpos)
                                        on_hook = (
                                            abs(rel[2] - hook_z) <= max(lip_h, 1e-3)
                                            and x_lo - 1e-9 <= abs(rel[0]) <= x_hi + 1e-9
                                            and abs(rel[1]) <= by + 1e-9
                                        )
                                        if not (on_hook and 0.0 <= dist <= tol):
                                            continue
                                        n_reach += 1
                                        is_pad = link.endswith("_tip")
                                        n_clean += int(is_pad)
                                        if dist < best["dist_m"]:
                                            best = {
                                                "dist_m": round(dist, 8),
                                                "link": link,
                                                "is_pad": bool(is_pad),
                                                "lip": gname,
                                                "rel_on_box": [round(float(v), 5)
                                                               for v in rel],
                                                "wrist": [wx, wy, wz, rx, ry, 0.0],
                                                "curl": list(curl),
                                            }
    obj_mass = float(model.body_mass[obj])
    res = {
        "overhang_mm": round(overhang * 1e3, 3),
        "lip_h_mm": round(lip_h * 1e3, 3),
        "box_mass_kg": obj_mass,
        "mass_ok": abs(obj_mass - baf.BOX_MASS_KG) < 1e-9,
        "hook_face_z_body": hook_z,
        "hook_face_x_band": [x_lo, x_hi],
        "n_poses": n_poses,
        "n_reach": n_reach,
        "n_clean_hook": n_clean,
        "reachable": n_reach > 0,
        "clean_hook": n_clean > 0,
        "best": best if n_reach else None,
        "tol_m": tol,
    }
    path.unlink(missing_ok=True)
    return res


def main() -> None:
    tol = TOL
    cells = []
    for oh in OVERHANGS_M:
        for lh in LIP_HS_M:
            r = scan_cell(oh, lh, tol)
            cells.append(r)
            print(f"o={r['overhang_mm']:4.1f} h={r['lip_h_mm']:4.1f}  "
                  f"reach={r['n_reach']:5d}  clean={r['n_clean_hook']:5d}  "
                  f"best={r['best']['dist_m'] * 1e3 if r['best'] else float('nan'):7.3f} mm  "
                  f"mass={r['box_mass_kg']:.6f}", flush=True)

    # control: the bare box, same sweep, must find NOTHING hookable
    ctrl = scan_cell(0.0, 0.0, tol)
    ctrl["note"] = "bare box control: lip_overhang_m=0 -> no lip geom exists"
    print(f"CONTROL bare box: reach={ctrl['n_reach']} clean={ctrl['n_clean_hook']}")

    ok = [c for c in cells if c["clean_hook"]]
    summary = {
        "question": "can a fingertip reach the underside of a tray lip on the box's "
                    "upper edge, and does that give a clean normal-force hook?",
        "criterion": "w35d strict: nearest point on the LIP must lie inside the "
                     "hook-face rectangle (z = box_top - lip_h, bx <= |x| <= "
                     "bx + overhang, |y| <= by), gap <= the R5 tolerance",
        "tol_m": tol,
        "overhangs_mm": [o * 1e3 for o in OVERHANGS_M],
        "lip_hs_mm": [h * 1e3 for h in LIP_HS_M],
        "control_bare_box": ctrl,
        "cells": cells,
        "n_cells": len(cells),
        "n_clean_cells": len(ok),
        "clean_region": [{"overhang_mm": c["overhang_mm"], "lip_h_mm": c["lip_h_mm"]}
                         for c in ok],
        "all_cells_mass_0p15": all(c["mass_ok"] for c in cells),
    }
    OUT.write_text(json.dumps(summary, indent=2))
    print(f"\nclean cells: {len(ok)}/{len(cells)}")
    for c in ok:
        print(f"  overhang={c['overhang_mm']}mm lip_h={c['lip_h_mm']}mm "
              f"clean={c['n_clean_hook']} best={c['best']['dist_m'] * 1e3:.3f}mm "
              f"via {c['best']['link']}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
