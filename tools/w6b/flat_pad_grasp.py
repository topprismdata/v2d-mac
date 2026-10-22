#!/usr/bin/env python3
"""!! INVALIDATED BY r65 -- DO NOT TRUST ITS OUTPUT !!

This search used the rotation error
    e_r = R0 @ (0.5 * [R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]])
which is sin(theta)*axis and VANISHES at a 180 deg error.  The IK
therefore "converged" at a completely flipped orientation (measured:
max|R_achieved - R_target| = 1.588 with a reported |e_r| = 0.00 deg).
Every number this script produced is invalid.

The replacement is tools/w6b/palm_pose_refine.py (the proper
scipy as_rotvec error, the gate's own metric, the table clearance and
the natural-branch scoring).  See runs/w6b/r84_side_cell_ledger.md.

"""The r63 terminal grasp, end to end, measured: the searched flat-pad
arm pose + the known-good grasp's fingers + the per-finger curl
equalisation.

The r63 measurements that make this the right combination:
  * the searched ARM pose (tools/w6b/palm_pose_reach.py) puts the palm
    where the known-good floating grasp's palm was, with the finger axes
    at 17-24 deg from the end face's normal (the shipped side cell has
    them at 90 deg -- the pads present their edge and the distal links
    hit the box first);
  * the known-good grasp's FINGER posture is the one whose pads land on
    the face;
  * the per-finger curl equalisation (r63e) then brings all three pads
    to the SAME surface gap.

This script measures the combination WITHOUT the full pipeline: the
pads' true distance to the box, whether their closest point is on the
end face, the finger axes, the joint-limit violations, and the box's
displacement when the pose is released in a physics hold.

usage: flat_pad_grasp.py [scene.xml]
"""
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import retarget.retarget_dual as R  # noqa: E402

SCENE = Path(sys.argv[1] if len(sys.argv) > 1 else
              "grounding/assets/allegro_floating_ur5e_side_r37.xml")
R.SCENE_KIND = "ur5e"

m = mujoco.MjModel.from_xml_path(str(SCENE))
d = mujoco.MjData(m)
specs = {s: R.SideSpec.build(s) for s in R.VIDEOS}
for s in specs:
    specs[s].bind(m, d)
_, box_geom, _, _ = R.obj_ids(m)
OBJ_B = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "object")
HALF = np.array(m.geom_size[box_geom])

reach = json.loads(Path("runs/w6b/palm_pose_reach.json").read_text())["sides"]
out = {}
for side, spec in specs.items():
    v = reach.get(side, {})
    arm = np.asarray(v.get("arm", np.zeros(6)), float)
    fg = v.get("fingers", {})
    # the finger command, in ACTUATOR order (ff, mf, rf, th x 4)
    fingers = []
    for f in ("ff", "mf", "rf"):
        fingers += list(fg.get(f, [0.0] * 4))
    fingers += [0.0] * 4          # the thumb is not part of the pad set
    fingers = np.array(fingers, float)

    def state(w, f):
        d.qpos[:] = m.qpos0
        for _jn in range(m.njnt):
            nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, _jn)
            if nm and nm.startswith("object"):
                qa = int(m.jnt_qposadr[_jn])
                d.qpos[qa:qa + 7] = [0, 0, 0.74, 1, 0, 0, 0]
                break
        d.qpos[spec.wrist_qadr] = w
        for i, a in enumerate(np.asarray(spec.finger_act_id).ravel()):
            j = int(m.actuator_trnid[int(a)][0])
            d.qpos[int(m.jnt_qposadr[j])] = f[i]
        mujoco.mj_forward(m, d)

    pads = [g for g in spec.tip_geoms
            if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, int(g)) or "")
            .endswith(("_tip_collision", "_tip_left_collision"))
            and (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, int(g)) or "")
            .startswith(("ff_tip", "mf_tip", "rf_tip"))]

    def measure(w, f):
        state(w, f)
        rows = []
        for g in pads:
            gi = int(g)
            ft = np.zeros(6)
            dist = float(mujoco.mj_geomDistance(m, d, gi, box_geom, 1.0, ft))
            rel = ft[3:6] - np.array(d.xpos[OBJ_B])
            on_end = bool(abs(abs(rel[0]) - float(HALF[0])) < 2e-3)
            nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, gi) or ""
            base = nm.replace("_tip_collision", "").replace(
                "_tip_left_collision", "")
            dn = (base + "_distal_collision" if side == "right"
                  else base + "_distal_left_collision")
            di = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, dn)
            Rd = np.array(d.geom_xmat[di]).reshape(3, 3)
            n_in = np.array([-spec.face_sign, 0.0, 0.0])
            ang = float(np.degrees(np.arccos(abs(float(Rd[:, 2] @ n_in)))))
            rows.append({"pad": nm, "surface_gap_mm": round(dist * 1e3, 4),
                         "on_end_face": on_end, "axis_deg": round(ang, 2)})
        viol = 0.0
        for g in pads:
            pass
        for i, a in enumerate(np.asarray(spec.finger_act_id).ravel()):
            j = int(m.actuator_trnid[int(a)][0])
            lo, hi = m.jnt_range[j]
            viol = max(viol, float(lo - d.qpos[int(m.jnt_qposadr[j])]),
                       float(d.qpos[int(m.jnt_qposadr[j])] - hi))
        return rows, viol

    print(f"=== {side} ===")
    r0, v0 = measure(arm, fingers)
    print("  the searched arm pose + the known-good fingers:")
    for r in r0:
        print(f"    {r['pad']:22s} gap {r['surface_gap_mm']:+9.3f} mm  "
              f"on the end face {str(r['on_end_face']):5s}  "
              f"axis {r['axis_deg']:5.1f} deg")
    print(f"    the finger joint-limit violation: {v0:.4f} rad")

    # the per-finger curl equalisation
    f_eq, eq = R.equalize_pad_curls(m, d, spec, arm[None, :], fingers[None, :],
                                    -0.05e-3)
    r1, v1 = measure(arm, f_eq)
    print("  after the per-finger curl equalisation:")
    for r in r1:
        print(f"    {r['pad']:22s} gap {r['surface_gap_mm']:+9.3f} mm  "
              f"on the end face {str(r['on_end_face']):5s}  "
              f"axis {r['axis_deg']:5.1f} deg")
    print(f"    the spread: {eq.get('spread_mm')} mm; the joint-limit "
          f"violation: {v1:.4f} rad")

    # the physics hold: does the grasp hold the box?
    d.qpos[:] = m.qpos0
    for _jn in range(m.njnt):
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, _jn)
        if nm and nm.startswith("object"):
            qa = int(m.jnt_qposadr[_jn])
            d.qpos[qa:qa + 7] = [0, 0, 0.74, 1, 0, 0, 0]
            break
    d.qpos[spec.wrist_qadr] = arm
    for i, a in enumerate(np.asarray(spec.finger_act_id).ravel()):
        j = int(m.actuator_trnid[int(a)][0])
        d.qpos[int(m.jnt_qposadr[j])] = f_eq[i]
    mujoco.mj_forward(m, d)
    # hold the arm at its pose (the arm's servos) and let the physics run
    for t in range(600):
        for i, a in enumerate(np.asarray(spec.wrist_act_id).ravel()):
            d.ctrl[int(a)] = arm[i]
        for i, a in enumerate(np.asarray(spec.finger_act_id).ravel()):
            d.ctrl[int(a)] = f_eq[i]
        mujoco.mj_step(m, d)
    r2, v2 = measure(arm, f_eq)
    disp = float(np.linalg.norm(np.array(d.xpos[OBJ_B]) -
                                np.array([0, 0, 0.74]))) * 1e3
    print(f"  after a 600-step physics hold:")
    for r in r2:
        print(f"    {r['pad']:22s} gap {r['surface_gap_mm']:+9.3f} mm  "
              f"on the end face {str(r['on_end_face']):5s}  "
              f"axis {r['axis_deg']:5.1f} deg")
    print(f"    the box's displacement: {disp:.3f} mm")
    n_on = sum(1 for r in r2 if r["on_end_face"]
               and abs(r["surface_gap_mm"]) <= 0.116)
    print(f"    => {n_on} pads on the end face within R5")
    out[side] = {"arm": [float(x) for x in arm],
                 "fingers_known_good": [float(x) for x in fingers],
                 "fingers_equalized": [float(x) for x in f_eq],
                 "rows_before": r0, "rows_after_equalize": r1,
                 "rows_after_hold": r2,
                 "joint_violation": v1,
                 "box_displacement_mm": disp,
                 "n_pads_on_end_face": n_on}
Path("runs/w6b/r63_flat_pad_grasp.json").write_text(json.dumps(out, indent=1))
print("\nwrote runs/w6b/r63_flat_pad_grasp.json")
