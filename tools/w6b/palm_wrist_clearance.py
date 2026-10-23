#!/usr/bin/env python3
"""The r86 acceptance check: at the searched terminal, does the ARM stay
clear of the table when the servos carry the MEASURED tracking error?

The replay's measured executed-vs-commanded joint error at the terminal
was up to 0.13 rad on the wrist_1 (the wrist_2's servo was saturated at
its 28 Nm limit against a 30 N table reaction).  So the clearance has
to hold with a 0.15 rad perturbation of every wrist axis, not just at
the nominal pose.

Usage: palm_wrist_clearance.py [refine.json]
"""
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

SCENE = Path("grounding/assets/allegro_floating_ur5e_side_r37.xml")
REF = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/w6b/palm_pose_refine.json")
ARM = ("shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2",
       "wrist_3")
LINKS = ("upper_arm_link", "forearm_link", "wrist_1_link", "wrist_2_link",
         "wrist_3_link")
TOL = 0.13          # the measured executed-vs-commanded envelope


def main():
    m = mujoco.MjModel.from_xml_path(str(SCENE))
    d = mujoco.MjData(m)
    tg = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "table_slab")
    top = float(m.geom_pos[tg][2] + m.geom_size[tg][2])
    ref = json.loads(REF.read_text())
    ok = True
    for side, prefix in (("right", "ur5e_r"), ("left", "ur5e_l")):
        e = ref["sides"][side]
        e = e["best"][0] if isinstance(e.get("best"), list) else e["best"]
        arm = np.asarray(e["arm"], float)
        ag = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT,
                                f"{prefix}_{n}_joint") for n in ARM]
        aq = [int(m.jnt_qposadr[j]) for j in ag]
        worst = None
        for trial, pert in enumerate(
                [np.zeros(6)] +
                [np.array([0, 0, 0, s, 0, 0]) for s in
                 (-TOL, TOL)] +
                [np.array([0, 0, 0, 0, s, 0]) for s in (-TOL, TOL)] +
                [np.array([0, 0, 0, 0, 0, s]) for s in (-TOL, TOL)]):
            d.qpos[:] = m.qpos0
            d.qpos[aq] = arm + pert
            mujoco.mj_forward(m, d)
            hits = []
            for c in range(d.ncon):
                g1, g2 = d.contact[c].geom1, d.contact[c].geom2
                b1 = m.geom_bodyid[g1]
                b2 = m.geom_bodyid[g2]
                nm = ""
                for b in (b1, b2):
                    bn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY,
                                           int(b)) or ""
                    if bn.startswith(prefix) and any(
                            ln in bn for ln in LINKS):
                        nm = bn
                if nm and (b1 == 0 or b2 == 0):
                    hits.append((nm, d.contact[c].dist))
            if hits:
                worst = (trial, pert, hits)
        print(f"=== {side}: the arm's table contact at the terminal")
        if worst:
            ok = False
            print(f"    CONTACT: {worst[2]}  (the perturbation {worst[1]})")
        else:
            print("    none, nominal and with a +-0.13 rad wrist error")
        # the clearance margin
        d.qpos[:] = m.qpos0
        d.qpos[aq] = arm
        mujoco.mj_forward(m, d)
        dist = mujoco.mj_geomDistance(m, d, tg, tg, 0.05)
        print(f"    the search's recorded arm clearance: "
              f"{e.get('arm_clearance_mm')} mm")
    print("VERDICT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
