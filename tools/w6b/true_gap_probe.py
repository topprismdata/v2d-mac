#!/usr/bin/env python3
"""The TRUE signed gap from each fingertip pad to the BOX, measured with
MuJoCo's own mj_geomDistance, versus the infinite-plane support formula.

Why (r63): the support function along the face normal is the exact
distance to an INFINITE plane.  The box's end face is FINITE (240 x 80 mm),
and a fingertip can sit OFF that rectangle (measured: the ring finger's
closest point is 45 mm above the face's top edge) -- the plane formula
then reports a small penetration where the pad is nowhere near the face.

This script measures both quantities for each finger over a curl sweep, and
reports the closest point's landing face, so the "which surface is the pad
actually on" question is answered by measurement rather than assumed.

usage: true_gap_probe.py [scene.xml] [report.json]
"""
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

SCENE = Path(sys.argv[1] if len(sys.argv) > 1 else
              "grounding/assets/allegro_floating_ur5e_side_r37.xml")
REPORT = Path(sys.argv[2] if len(sys.argv) > 2 else "runs/w6b/report.json")

m = mujoco.MjModel.from_xml_path(str(SCENE))
d = mujoco.MjData(m)
rep = json.loads(REPORT.read_text())

ARM = ("shoulder_pan", "shoulder_lift", "elbow",
       "wrist_1", "wrist_2", "wrist_3")
AJ = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"ur5e_r_{n}_joint")
      for n in ARM]
AQ = [int(m.jnt_qposadr[j]) for j in AJ]
FJi = {f: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM,
                            f"{f}_tip_collision") for f in ("ff", "mf", "rf")}
FJj = {f: [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{f}j{k}")
           for k in range(4)] for f in ("ff", "mf", "rf")}
OBJ_G = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "object_geom")
OBJ_B = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "object")
HALF = np.array(m.geom_size[OBJ_G])
BOX_TOP = 0.74 + float(HALF[2])


def state(wrist, fing):
    d.qpos[:] = m.qpos0
    d.qpos[AQ] = wrist
    for i, f in enumerate(("ff", "mf", "rf")):
        for k in range(4):
            d.qpos[int(m.jnt_qposadr[FJj[f][k]])] = fing[i * 4 + k]
    mujoco.mj_forward(m, d)


def measure(f):
    g = FJi[f]
    ft = np.zeros(6)
    dist = mujoco.mj_geomDistance(m, d, g, OBJ_G, 0.5, ft)
    p_pad, p_box = ft[:3], ft[3:]
    # which face of the box did the closest point land on?
    rel = p_box - np.array(d.xpos[OBJ_B])
    ax = int(np.argmax(np.abs(rel / np.maximum(HALF, 1e-9))))
    face = ("+x", "-x", "+y", "-y", "+z", "-z")[ax]
    on_end_face = (abs(rel[0]) > HALF[0] - 1e-4)
    # the plane-based support formula (what the pipeline used)
    R = np.array(d.geom_xmat[g]).reshape(3, 3)
    he = np.array(m.geom_size[g])
    sup = float(sum(he[i] * abs(float(R[:, i] @ np.array([1.0, 0, 0])))
                     for i in range(3)))
    cx = float(d.geom_xpos[g][0]) - (float(d.xpos[OBJ_B][0]) + float(HALF[0]))
    return dist, cx - sup, face, on_end_face, p_box


def main():
    tc = rep["per_side"]["right"]["terminal_commands"]
    wrist = np.asarray(tc["wrist_rad"], float)
    fing = np.asarray(tc["finger_rad"], float)
    print(f"box: half extents {np.round(HALF*1e3,1)} mm, top z = "
          f"{BOX_TOP*1e3:.1f} mm")
    state(wrist, fing)
    print("\n=== the command state, both gap measures ===")
    for f in ("ff", "mf", "rf"):
        dist, plane, face, on_end, p_box = measure(f)
        print(f"  {f}: TRUE dist {dist*1e3:+8.3f} mm | plane formula "
              f"{plane*1e3:+8.3f} mm | closest point on the box's "
              f"{face} face at z={p_box[2]*1e3:7.1f} mm | on the END face: "
              f"{on_end}")
    print("\n=== per-finger curl sweep, the TRUE distance ===")
    for f in ("ff", "mf", "rf"):
        i = ("ff", "mf", "rf").index(f)
        print(f"  {f}:")
        for s in (0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0):
            f2 = fing.copy()
            f2[i * 4:(i + 1) * 4] = fing[i * 4:(i + 1) * 4] * s
            state(wrist, f2)
            dist, plane, face, on_end, p_box = measure(f)
            print(f"    scale {s:4.2f}: true {dist*1e3:+8.3f} mm  "
                  f"(plane {plane*1e3:+8.3f})  on {face} "
                  f"{'END-FACE' if on_end else 'side/top '} "
                  f"z={p_box[2]*1e3:7.1f}")


if __name__ == "__main__":
    main()
