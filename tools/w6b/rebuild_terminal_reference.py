#!/usr/bin/env python3
"""r117: rebuild the side cell's terminal reference from the delivered report.

THE DEFECT THIS FIXES
---------------------
The pipeline's DEFAULT terminal comes from
`runs/w6b/palm_pose_refine.json` (retarget_dual.py's W6B_R63_REFERENCE
default).  That file lived in the gitignored `runs/` tree and was never
committed, so:

  * a fresh clone had no terminal reference at all, the pipeline fell
    back to the ceiling-cell seed pairs, and the delivered numbers could
    not be reproduced outside this working tree;
  * the analysis tools (the lever attribution, the squeeze search, the
    physics selection) silently read whatever happened to be on disk, so
    re-running the terminal search for another cell REPLACED the
    delivered cell's terminal and silently moved every number that
    depended on it (observed: the attribution's residual-explained
    displacement went 16.225 mm -> 10.591 mm for the right).

THE REPAIR
----------
The delivered terminal is exactly the one recorded in the delivered
report: the terminal's commanded arm and finger values.  The arm's
forward kinematics DETERMINES the palm's pose, so the reference can be
rebuilt exactly -- no search needed, no candidate lottery:

    palm_pos, palm_R = FK(r37 scene, the delivered arm joints)

Usage:
    runs/.venv-retarget/bin/python tools/w6b/rebuild_terminal_reference.py
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys

import mujoco
import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
SCENE = "grounding/assets/allegro_floating_ur5e_side_r37.xml"
REPORT = "runs/w6b/report.json"
OUT = REPO / "runs/w6b/palm_pose_refine.json"

ARM = ("shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2",
       "wrist_3")
FINGERS = ("ff", "mf", "rf", "th")


def committed_report() -> dict:
    """The delivered report as COMMITTED (not the working tree's copy)."""
    blob = subprocess.run(["git", "show", f"HEAD:{REPORT}"], cwd=REPO,
                          capture_output=True, text=True, check=True).stdout
    return json.loads(blob)


def main() -> int:
    rep = committed_report()
    m = mujoco.MjModel.from_xml_path(str(REPO / SCENE))
    d = mujoco.MjData(m)
    sides = {}
    for side, prefix in (("right", "ur5e_r"), ("left", "ur5e_l")):
        tc = rep["per_side"][side]["terminal_commands"]
        arm = [float(v) for v in tc["wrist_rad"]]
        fing_flat = [float(v) for v in tc["finger_rad"]]
        assert len(arm) == 6 and len(fing_flat) == 16
        # the palm body: "palm" (right) and "palm_left" (left)
        palm = mujoco.mj_name2id(
            m, mujoco.mjtObj.mjOBJ_BODY,
            "palm" if side == "right" else "palm_left")
        if palm < 0:
            raise SystemExit(f"no palm body for {side}")
        aq = []
        for a in ARM:
            j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT,
                                  f"{prefix}_{a}_joint")
            if j < 0:
                raise SystemExit(f"no joint {prefix}_{a}_joint")
            aq.append(int(m.jnt_qposadr[j]))
        # the finger joints, in the pipeline's order (ff, mf, rf, th) x 4
        fq = []
        for k, f in enumerate(FINGERS):
            for j in range(4):
                name = f"{f}j{j}"
                jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
                if jid < 0:
                    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT,
                                            f"{name}_left")
                if jid < 0:
                    raise SystemExit(f"no finger joint {name} ({side})")
                fq.append(int(m.jnt_qposadr[jid]))
        # the forward kinematics of the delivered arm
        d.qpos[:] = m.qpos0
        d.qpos[aq] = arm
        d.qpos[fq] = fing_flat
        mujoco.mj_kinematics(m, d)
        pos = np.array(d.xpos[palm], float)
        Rq = np.zeros(9)
        mujoco.mju_quat2Mat(Rq, d.xquat[palm])
        R = np.array(Rq).reshape(3, 3)
        fingers = {f: fing_flat[4 * k:4 * k + 4]
                   for k, f in enumerate(FINGERS)}
        sides[side] = {
            "best": [{
                "source": ("rebuilt from the delivered report's terminal "
                           "commands by "
                           "tools/w6b/rebuild_terminal_reference.py"),
                "palm_pos": [float(v) for v in pos],
                "palm_R": [[float(v) for v in row] for row in R],
                "arm": arm,
                "fingers": fingers,
                "palm_pos_mm": [round(float(v) * 1e3, 2) for v in pos],
            }],
        }
        print(f"{side}: the palm at "
              f"{[round(float(v)*1e3, 2) for v in pos]} mm, the arm "
              f"{[round(v, 4) for v in arm]}")
    out = {
        "scene": SCENE,
        "note": ("the side cell's terminal reference.  The pipeline's "
                 "default terminal comes from this file (the "
                 "W6B_R63_REFERENCE default), so it must be in the repo: "
                 "without it a fresh clone falls back to the ceiling-cell "
                 "seed pairs and cannot reproduce the delivered numbers "
                 "(r117)."),
        "rebuilt_from": REPORT,
        "sides": sides,
    }
    OUT.write_text(json.dumps(out, indent=1))
    print("written:", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
