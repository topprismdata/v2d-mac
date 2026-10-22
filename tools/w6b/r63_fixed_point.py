#!/usr/bin/env python3
"""Iterate the r63 terminal re-registration to its fixed point, offline.

The r63 stack has two coupled steps:
  * the ORIENTATION re-registration (the arm IK for a palm pose whose
    finger axes point INTO the end face -- the measured root cause of the
    side cell's one-pad grasp), and
  * the per-finger CURL equalisation (each finger's curl sets its own
    pad's distance to the face).

They are coupled: the curl changes where the pads sit, and the arm pose
changes the geometry the curl is solved in.  This harness iterates them
to a fixed point and reports the MEASURED outcome at each step, so the
iteration is observable rather than assumed.

usage: r63_fixed_point.py [scene.xml] [n_iterations]
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
N_ITER = int(sys.argv[2]) if len(sys.argv) > 2 else 4

m = mujoco.MjModel.from_xml_path(str(SCENE))
d = mujoco.MjData(m)
# the side cell is a UR5e scene: the SideSpec must be built for it
R.SCENE_KIND = "ur5e"
specs = {s: R.SideSpec.build(s) for s in R.VIDEOS}
for s in specs:
    specs[s].bind(m, d)

# the reference: the known-good floating grasp's palm pose
vg = R.measure_verified_grasp()

# a start pose: the SEARCHED reachable flat-pad pose (its own arm joints
# and the known-good grasp's finger posture).  The arm's home pose is
# ~580 mm from the box, so the re-registration has no room to work in.
arm_home = {}
finger_home = {}
_ref = json.loads(Path("runs/w6b/palm_pose_reach.json").read_text())
for s, spec in specs.items():
    v = _ref["sides"].get(s, {})
    arm_home[s] = np.asarray(v.get("arm", np.zeros(6)), float)
    fg = v.get("fingers", {})
    # the searched file holds ff/mf/rf only; the thumb keeps its current
    # command (the thumb is not part of the pad set)
    _cur = np.array(d.qpos)[spec.finger_qadr].copy()
    _vals = []
    for _f in ("ff", "mf", "rf", "th"):
        if _f in fg:
            _vals += list(fg[_f])
        else:
            _vals += list(_cur[("ff", "mf", "rf", "th").index(_f) * 4:
                              ("ff", "mf", "rf", "th").index(_f) * 4 + 4])
    finger_home[s] = np.array(_vals, float)

out = {"scene": str(SCENE), "iterations": []}
for s, spec in specs.items():
    face_sign = spec.face_sign
    n_in = np.array([-face_sign, 0.0, 0.0])
    palm = spec._palm_id
    # only the COLLISION pads (the set also holds the visual geoms, which
    # have no collision and whose frames are the mesh's, not the pad's)
    tip_g = {g: int(g) for g in spec.tip_geoms
             if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, int(g)) or "")
             .endswith(("_tip_collision", "_tip_left_collision"))}
    _, box_geom, _, _ = R.obj_ids(m)

    def state(w, f):
        d.qpos[:] = m.qpos0
        d.qpos[spec.wrist_qadr] = w
        for i, a in enumerate(np.asarray(spec.finger_act_id).ravel()):
            j = int(m.actuator_trnid[int(a)][0])
            d.qpos[int(m.jnt_qposadr[j])] = f[i]
        mujoco.mj_forward(m, d)

    def axes():
        out_d = {}
        for g in tip_g:
            nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
            base = nm.replace("_tip_collision", "").replace(
                "_tip_left_collision", "")
            dn = (base + "_distal_collision" if face_sign > 0
                  else base + "_distal_left_collision")
            di = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, dn)
            Rd = np.array(d.geom_xmat[di]).reshape(3, 3)
            out_d[g] = float(np.degrees(np.arccos(
                abs(float(Rd[:, 2] @ n_in)))))
        return out_d

    def gaps():
        out_d = {}
        for g in tip_g:
            ft = np.zeros(6)
            out_d[g] = float(mujoco.mj_geomDistance(m, d, int(g), box_geom,
                                                     1.0, ft))
        return out_d

    w = arm_home[s].copy()
    f = finger_home[s].copy()
    print(f"=== {s} ===")
    for it in range(N_ITER):
        # 1. the orientation re-registration
        w_new, f_new, rec = R.reorient_terminal_pads(
            m, d, spec, w[None, :], f[None, :],
            np.asarray(vg[s]["R"], float), np.asarray(vg[s]["pos"], float))
        w = w_new[-1].copy()
        f = f_new[-1].copy()
        # 2. the per-finger curl equalisation
        f_eq, eq = R.equalize_pad_curls(m, d, spec, w[None, :], f[None, :],
                                        -0.05e-3)
        f = f_eq
        # 3. MEASURE
        state(w, f)
        ax = axes()
        gp = gaps()
        worst = max(ax.values()) if ax else 999.0
        n_on = 0
        for g, dd in gp.items():
            ft = np.zeros(6)
            mujoco.mj_geomDistance(m, d, int(g), box_geom, 1.0, ft)
            rel = ft[3:6] - np.array(d.xpos[
                mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "object")])
            if abs(abs(rel[0]) - float(m.geom_size[box_geom][0])) < 2e-3:
                n_on += 1
        print(f"  iter {it}: worst axis {worst:5.1f} deg, "
              f"axes {[round(v,1) for v in ax.values()]}, "
              f"gaps(mm) {[round(v*1e3,3) for v in gp.values()]}, "
              f"on the end face {n_on}/3")
        out["iterations"].append({
            "side": s, "iteration": it, "worst_axis_deg": worst,
            "axes_deg": {k: round(v, 2) for k, v in ax.items()},
            "gaps_mm": {k: round(v * 1e3, 4) for k, v in gp.items()},
            "n_on_end_face": n_on,
            "arm": [round(float(v), 6) for v in w],
            "fingers": [round(float(v), 6) for v in f],
            "reorient": rec,
        })

Path("runs/w6b/r63_fixed_point.json").write_text(json.dumps(out, indent=1))
print("\nwrote runs/w6b/r63_fixed_point.json")
