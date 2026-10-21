#!/usr/bin/env python3
"""Search the palm-PLATE grasp analytically (FK for the hand's internal
geometry, closed form for the palm's pose).

Why: the fingertip pads are 6 mm-radius discs carried by force-limited
servos (0.04-0.18 N.m).  Pressing a disc onto a flat face shears the
contact, shoves the box and saturates the servo, so the fingers collapse
(up to 0.87 rad measured) and only one pad stays on the face.  The palm's
plate (41 x 113 x 95 mm, normal = the palm's x axis) can be the contact
surface instead: then the ARM carries the grasp force and the contact is
face-on.

Method:
  1. FK the hand at a set of curls -> each fingertip's and the thumb's
     offset in the PALM frame, and the plate's centre offset + normal in
     the palm frame (all measured, not assumed);
  2. sweep the palm's orientation with the plate's normal facing the end
     face, and the palm's position so the plate is tangent to the face's
     plane;
  3. score every fingertip and the thumb against the box's AABB and the
     table's top; report the poses where the plate touches and nothing
     else collides.

usage: palm_plate_search.py [scene.xml]
"""
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

SCENE = Path(sys.argv[1] if len(sys.argv) > 1 else
              "grounding/assets/allegro_floating_ur5e_side_r37.xml")
OUT = Path("runs/w6b/palm_plate_search.json")

m = mujoco.MjModel.from_xml_path(str(SCENE))
d = mujoco.MjData(m)


def bid(n):
    i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
    assert i >= 0, n
    return i


PALM = bid("palm")
PLATE_G = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "palm_collision")
TIPS = ["ff_tip", "mf_tip", "rf_tip", "th_tip"]
TIP_B = {t: bid(t) for t in TIPS}
TIP_R = 0.006
FJ = [(f, k) for f in ("ff", "mf", "rf", "th") for k in range(4)]
FJ_ID = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{f}j{k}")
         for f, k in FJ]
BOX_G = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "object_geom")
BOX_HALF = np.array(m.geom_size[BOX_G])
BOX_C = np.array([0.0, 0.0, 0.74])
TABLE_TOP = 0.70


def hand_offsets(curl):
    """the tips' offsets in the palm frame + the plate's frame, at a curl."""
    d.qpos[:] = m.qpos0
    for j in FJ_ID:
        lo, hi = m.jnt_range[j]
        base = 0.5 * (lo + hi)
        d.qpos[m.jnt_qposadr[j]] = base + curl * 0.35 * (hi - base)
    mujoco.mj_forward(m, d)
    Rp = np.array(d.xmat[PALM]).reshape(3, 3)
    Pp = np.array(d.xpos[PALM])
    tips = {t: (Rp.T @ (np.array(d.xpos[TIP_B[t]]) - Pp)) for t in TIPS}
    Rg = np.array(d.geom_xmat[PLATE_G]).reshape(3, 3)
    plate_n = Rp.T @ (Rg[:, 0])
    plate_c = Rp.T @ (np.array(d.geom_xpos[PLATE_G]) - Pp)
    return tips, plate_n, plate_c


def clearance_to_box(p):
    dx = max(0.0, abs(p[0] - BOX_C[0]) - BOX_HALF[0])
    dy = max(0.0, abs(p[1] - BOX_C[1]) - BOX_HALF[1])
    dz = max(0.0, abs(p[2] - BOX_C[2]) - BOX_HALF[2])
    return float(np.linalg.norm([dx, dy, dz]) - TIP_R)


def main():
    face_n = np.array([1.0, 0.0, 0.0])          # the right hand's end face
    global PLATE_HALF_X
    PLATE_HALF_X = float(m.geom_size[PLATE_G][0])
    face_x = BOX_C[0] + BOX_HALF[0]
    cands = []
    for curl in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        tips, plate_n0, plate_c0 = hand_offsets(curl)
        # the plate's normal in the palm frame -> the palm's frame is chosen
        # so that the plate's world normal is -face_n (toward the face)
        for ang in range(0, 360, 15):
            for pitch in (-70, -50, -30, -10, 10, 30, 50, 70):
                a = np.deg2rad(ang)
                p = np.deg2rad(pitch)
                z_axis = np.array([0.0, np.cos(p) * np.cos(a),
                                   np.cos(p) * np.sin(a)])
                z_axis /= np.linalg.norm(z_axis)
                x_axis = -face_n
                y_axis = np.cross(z_axis, x_axis)
                y_axis /= np.linalg.norm(y_axis)
                z_axis = np.cross(x_axis, y_axis)
                R = np.stack([x_axis, y_axis, z_axis], axis=1)
                # the plate's world normal must equal the palm's x axis
                # (the plate's normal is the palm's x) -> the frame above is
                # already the plate-normal frame; the roll about the normal
                # is the angle we sweep.
                # the palm's origin: the plate's centre is at
                # p_palm + R @ plate_c0; require the plate's outer surface
                # (at +plate_half along the normal) to sit on the face's
                # plane.
                for lat in (0.0, 0.02, -0.02):
                    # the plate's tangential offset (along y) is free
                    plate_c_w = R @ plate_c0
                    # the tangential direction = the palm's y (world)
                    t_hat = R[:, 1]
                    # the plate's CENTRE must sit one half-thickness
                    # OUTSIDE the face's plane (its normal points at the
                    # box), so the plate's outer surface is tangent
                    p_palm = (np.array([face_x + PLATE_HALF_X, 0.0,
                                        BOX_C[2]])
                              - plate_c_w
                              + t_hat * lat)
                    # the plate's outer surface: the plane offset
                    surf = float((p_palm + plate_c_w)[0] - face_x)
                    rec = {"curl": curl, "ang": ang, "pitch": pitch,
                           "lat_mm": lat * 1e3,
                           "palm": [round(float(v), 5) for v in p_palm],
                           "R": [[round(float(v), 6) for v in row] for row in R]}
                    worst_box = 1e9
                    worst_tab = 1e9
                    per = {}
                    for t in TIPS:
                        pw = p_palm + R @ tips[t]
                        cb = clearance_to_box(pw)
                        ct = float(pw[2] - TIP_R - TABLE_TOP)
                        per[t] = (round(cb * 1e3, 3), round(ct * 1e3, 3))
                        worst_box = min(worst_box, cb)
                        worst_tab = min(worst_tab, ct)
                    rec.update({"surf_gap_mm": round(surf * 1e3, 3),
                                "box_clear_mm": round(worst_box * 1e3, 3),
                                "table_clear_mm": round(worst_tab * 1e3, 3),
                                "tips": per})
                    cands.append(rec)
    # the plate's surface offset: the plate's half-thickness along its
    # normal (the local x half-size)
    plate_half = float(m.geom_size[PLATE_G][0])
    for rec in cands:
        # the plate's outer surface = its centre + plate_half along the
        # normal (the palm's x = -face_n)
        # the plate's outer surface (the one that faces the box) is HALF a
        # thickness along the normal, and the normal points at the box
        rec["surf_gap_mm"] = round(
            (rec["surf_gap_mm"] * 1e-3 - plate_half) * 1e3, 3)
    feas = [r for r in cands
            if abs(r["surf_gap_mm"]) < 2.0
            and r["box_clear_mm"] > 2.0 and r["table_clear_mm"] > 2.0]
    print(f"grid {len(cands)} poses, feasible {len(feas)}")
    for r in feas[:20]:
        print(f"  curl={r['curl']:.1f} ang={r['ang']:3d} pitch={r['pitch']:+4d} "
              f"lat={r['lat_mm']:+5.1f} surf={r['surf_gap_mm']:+6.2f} "
              f"box={r['box_clear_mm']:+7.2f} tab={r['table_clear_mm']:+7.2f} "
              f"{r['tips']}")
    if not feas:
        # the least-bad: rank by the worst clearance (box and table) and
        # show the surface gap so the trade-off is visible
        rank = sorted(cands, key=lambda r: -min(r["box_clear_mm"],
                                                r["table_clear_mm"]))
        print("no feasible pose; the 10 least-bad (clearance-limited):")
        for r in rank[:10]:
            print(f"  curl={r['curl']:.1f} ang={r['ang']:3d} "
                  f"pitch={r['pitch']:+4d} lat={r['lat_mm']:+5.1f} "
                  f"surf={r['surf_gap_mm']:+7.2f} "
                  f"box={r['box_clear_mm']:+8.2f} tab={r['table_clear_mm']:+8.2f} "
                  f"{r['tips']}")
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(
            {"scene": str(SCENE), "n": len(cands), "n_feasible": 0,
             "least_bad": rank[:40]}, indent=1))
        print(f"wrote {OUT}")
        return
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(
        {"scene": str(SCENE), "n": len(cands), "n_feasible": len(feas),
         "feasible": feas[:40]}, indent=1))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
