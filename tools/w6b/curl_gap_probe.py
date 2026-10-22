#!/usr/bin/env python3
"""How does a fingertip's SURFACE gap to the end face depend on its curl?

The r63 measurement says the shipped command state embeds the three pads
2.6 / 5.0 / 6.4 mm into the face (the corrected support-function gap), and
that the SPREAD (3.8 mm) comes from the pads' different TILTS: a flat face
sees each pad at its own support extent, so pads at different angles cannot
touch it simultaneously.

The wrist roll/pitch cannot fix that (it moves all three pads together).
The per-finger CURL can: each finger's curl rotates its own pad.  This
script measures the relation so the seat can invert it:

    surface_gap(finger, curl) -> monotone?

usage: curl_gap_probe.py [scene.xml] [report.json]
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
PAD = {f: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"{f}_tip_collision")
       for f in ("ff", "mf", "rf")}
FJ = {f: [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{f}j{k}")
          for k in range(4)] for f in ("ff", "mf", "rf")}
OBJ = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "object")
OG = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "object_geom")
HALF = float(m.geom_size[OG][0])
FACE_X = HALF


def support(gid):
    R = np.array(d.geom_xmat[gid]).reshape(3, 3)
    he = np.array(m.geom_size[gid])
    return float(sum(he[i] * abs(float(R[:, i] @ np.array([1.0, 0, 0])))
                     for i in range(3)))


def gaps(wrist, finger_cmd_act):
    """finger_cmd_act: 12 values in ACTUATOR order (ffa0..3, mfa0..3, rfa0..3)
    as they appear in the report's terminal command."""
    mujoco.mj_resetData(m, d)
    d.qpos[:] = m.qpos0
    d.qpos[AQ] = wrist
    # the actuator -> joint map (the report's order is the actuator order)
    for i, f in enumerate(("ff", "mf", "rf")):
        for k in range(4):
            j = FJ[f][k]
            d.qpos[int(m.jnt_qposadr[j])] = finger_cmd_act[i * 4 + k]
    mujoco.mj_forward(m, d)
    out = {}
    for f, g in PAD.items():
        x = float(d.geom_xpos[g][0])
        out[f] = (x - FACE_X) - support(g)
    return out


def main():
    tc = rep["per_side"]["right"]["terminal_commands"]
    wrist = np.asarray(tc["wrist_rad"], float)
    fing = np.asarray(tc["finger_rad"], float)
    print("baseline (right) commanded finger values per finger:")
    for i, f in enumerate(("ff", "mf", "rf")):
        print(f"  {f}: {np.round(fing[i*4:(i+1)*4], 4).tolist()}")
    g0 = gaps(wrist, fing)
    print(f"  surface gaps: " +
          ", ".join(f"{f}={g0[f]*1e3:+.3f}mm" for f in g0))
    print("\n=== per-finger curl sweep (the finger's own 4 joints scaled) ===")
    for f in ("ff", "mf", "rf"):
        i = ("ff", "mf", "rf").index(f)
        print(f"  {f}:")
        prev = None
        for s in (0.6, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4, 1.6, 1.8, 2.0):
            f2 = fing.copy()
            base = np.array([fing[i*4 + k] for k in range(4)])
            f2[i*4:(i+1)*4] = base * s
            g = gaps(wrist, f2)[f]
            mono = ""
            if prev is not None:
                mono = " (increasing)" if g > prev else " (decreasing)"
            prev = g
            print(f"    scale {s:4.2f}: gap {g*1e3:+8.3f} mm{mono}")
    print("\n=== the OTHER fingers' gaps change too (the wrist is fixed) ===")
    for s in (0.8, 1.0, 1.2):
        f2 = fing.copy()
        i = 1
        base = np.array([fing[i*4 + k] for k in range(4)])
        f2[i*4:(i+1)*4] = base * s
        g = gaps(wrist, f2)
        print(f"  mf scale {s:4.2f}: " +
              ", ".join(f"{k}={v*1e3:+8.3f}" for k, v in g.items()))


if __name__ == "__main__":
    main()
