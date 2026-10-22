#!/usr/bin/env python3
"""Replay ONLY the terminal hold from a committed report's commands.

Why: the pipeline's terminal state is only reported as an aggregate (the
executed rect, one contact force, the per-pad gaps).  The question that
decides the next design step is temporal: does the contact force JUMP at
first touch (a commanded interpenetration) or BUILD UP over the hold (a
servo-error / sag effect)?  Those have different fixes, and only the
per-frame record tells them apart.

This loads the side scene, takes the terminal wrist+finger commands from a
report (runs/w6b/report.json), replays the hold with the same control law
(position servos, the terminal gravity cancel), and prints per frame:
  * the end-face contact force per pad (from MuJoCo's own contact list),
  * the pads' signed SURFACE gap to the face plane,
  * the box's displacement,
  * the finger joint tracking error (commanded vs executed),
  * the arm's actuator forces.

usage: terminal_probe.py [report.json] [scene.xml] [n_frames]
"""
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

REPORT = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/w6b/report.json")
SCENE = Path(sys.argv[2] if len(sys.argv) > 2 else
              "grounding/assets/allegro_floating_ur5e_side_r37.xml")
NFR = int(sys.argv[3] if len(sys.argv) > 3 else 60)

rep = json.loads(REPORT.read_text())
m = mujoco.MjModel.from_xml_path(str(SCENE))
d = mujoco.MjData(m)

ARM = ("shoulder_pan", "shoulder_lift", "elbow",
       "wrist_1", "wrist_2", "wrist_3")
SIDES = {"right": ("ur5e_r", "", 1.0), "left": ("ur5e_l", "_left", -1.0)}

obj_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "object")
box_g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "object_geom")
half = np.array(m.geom_size[box_g])
PADS = {"right": ["ff_tip_collision", "mf_tip_collision", "rf_tip_collision"],
        "left": ["ff_tip_left_collision", "mf_tip_left_collision",
                 "rf_tip_left_collision"]}


def arm_joints(prefix):
    out = []
    for n in ARM:
        j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT,
                               f"{prefix}_{n}_joint")
        out.append((j, int(m.jnt_qposadr[j]), int(m.jnt_dofadr[j])))
    return out


def finger_joints(suffix):
    out = []
    for f in ("ff", "mf", "rf", "th"):
        for k in range(4):
            j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT,
                                   f"{f}j{k}{suffix}")
            out.append((j, int(m.jnt_qposadr[j])))
    return out


def act_ids(prefix, suffix):
    out = []
    for n in ARM:
        out.append(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR,
                                     f"{prefix}_{n}"))
    fj = []
    for f in ("ff", "mf", "rf", "th"):
        for k in range(4):
            fj.append(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR,
                                        f"ffa{k}{suffix}".replace(
                                            "ffa", f"{f}a")))
    return out, fj


def pad_surface_gap(gid_, face_plane, sign):
    """the pad's signed surface distance to the face plane (MuJoCo's own
    contact distance when in contact, else the geometric gap)."""
    best = None
    for i in range(d.ncon):
        c = d.contact[i]
        if int(c.geom1) == gid_ or int(c.geom2) == gid_:
            best = c if best is None or c.dist < best.dist else best
    if best is not None:
        return float(best.dist), True
    return None, False


def main():
    per_side = rep["per_side"]
    for side, (prefix, suffix, sign) in SIDES.items():
        tc = per_side[side]["terminal_commands"]
        wrist = np.asarray(tc["wrist_rad"], float)
        finger = np.asarray(tc["finger_rad"], float)
        aj = arm_joints(prefix)
        fj = finger_joints(suffix)
        a_act, f_act = act_ids(prefix, suffix)
        # reset: the box's free joint must be at the table rest pose
        # (0, 0, 0.74) -- otherwise the probe's face plane is 0.74 m off
        # the pads and every contact reading is a structural zero.
        mujoco.mj_resetData(m, d)
        d.qpos[:] = m.qpos0
        _ba = int(m.jnt_qposadr[mujoco.mj_name2id(
            m, mujoco.mjtObj.mjOBJ_JOINT, "object")]) if False else None
        for _jn in range(m.njnt):
            _nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, _jn)
            if _nm and _nm.startswith("object"):
                _qa = int(m.jnt_qposadr[_jn])
                d.qpos[_qa:_qa + 7] = [0.0, 0.0, 0.74, 1.0, 0.0, 0.0, 0.0]
                break
        for _i, (j, qa, _dof) in enumerate(aj):
            d.qpos[qa] = wrist[_i]
        for (j, qa), v in zip(fj, finger):
            d.qpos[qa] = v
        mujoco.mj_forward(m, d)
        print(f"=== {side} ===")
        _bc = np.array(d.xpos[obj_body])
        _fp = _bc[0] + sign * half[0]
        _pb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY,
                                "palm" if side == "right" else "palm_left")
        print(f"  box centre {np.round(_bc*1e3,2)}  face plane x = {_fp*1e3:.2f} mm")
        print(f"  palm origin {np.round(np.array(d.xpos[_pb])*1e3,2)}")
        for n in PADS[side]:
            g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)
            px = np.array(d.geom_xpos[g])
            print(f"    {n:24s} centre {np.round(px*1e3,2)}  "
                  f"dx from the face plane {float((px[0]-_fp)*sign)*1e3:+8.3f} mm  "
                  f"geom_size {np.round(m.geom_size[g]*1e3,2)} "
                  f"type {m.geom_type[g]}")
        print(f"  command: wrist {np.round(wrist, 4).tolist()}")
        print(f"  pad geom ids: " +
              ", ".join(f"{n}={mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)}"
                        for n in PADS[side]))
        for t in range(NFR):
            for aid, v in zip(a_act, wrist):
                d.ctrl[aid] = v
            for aid, v in zip(f_act, finger):
                d.ctrl[aid] = v
            # the terminal gravity cancel (the shipped hold's control law)
            d.qfrc_applied[:] = 0.0
            tau = np.zeros(m.nv)
            mujoco.mj_rne(m, d, 0, tau)
            for _, _, dof in aj:
                d.qfrc_applied[dof] = -tau[dof]
            for j, qa in fj:
                dof = int(m.jnt_dofadr[j])
                d.qfrc_applied[dof] = -tau[dof]
            for _ in range(20):
                mujoco.mj_step(m, d)
            box_c = np.array(d.xpos[obj_body])
            face_plane = box_c[0] + sign * half[0]
            forces = []
            for n in PADS[side]:
                g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)
                f = 0.0
                gap = None
                for i in range(d.ncon):
                    c = d.contact[i]
                    if int(c.geom1) == g or int(c.geom2) == g:
                        cf = np.zeros(6)
                        mujoco.mj_contactForce(m, d, i, cf)
                        nrm = np.array([-sign, 0.0, 0.0])
                        f = max(f, float(abs(np.asarray(cf[:3]) @ nrm)))
                        gap = float(c.dist)
                forces.append((n.replace("_collision", ""),
                               round(f, 3), None if gap is None
                               else round(gap * 1e3, 3)))
            fjerr = max(abs(float(d.qpos[qa]) - v)
                        for (_, qa), v in zip(fj, finger))
            disp = float(np.linalg.norm(box_c - np.array([0.0, 0.0, 0.74]))) * 1e3
            af = np.asarray(d.actuator_force)[a_act]
            if t % 5 == 0 or t == NFR - 1:
                print(f"   t={t:3d} box_disp={disp:6.2f}mm "
                      f"fjerr_max={fjerr:.4f} armF="
                      f"{np.round(af, 1).tolist()} pads={forces}")


if __name__ == "__main__":
    main()
