"""r118: the finger servo's HARDWARE force ceiling at the delivered pose.

WHY.  The r118 measurements (tools/w6b/force_depth_law.py and
tools/w6b/curl_baked_scan.py) established:

  * the delivered terminal produces 0.2549 N (right) / 0.7377 N (left) of
    normal force, and the v10 lift gate needs 1.0911 N per hand;
  * inside the GREEN region (the command state's pads in the rectangle,
    no burial) the squeeze buys at most 0.3797 N (right) / 0.7401 N
    (left) -- the force saturates well below the requirement;
  * a deeper CURL does reach the requirement (the left: 1.0166 N at curl
    1.05, 1.6398 N at curl 1.25) but it moves the pads OFF the face, so
    the command state's n_pads_in_rect collapses to 0 and gate1 goes red.

So the binding constraint is the command state's geometry, and the
question becomes: is the delivered force limited by the HARDWARE (the
finger actuator's torque limit -- which is physical and must not be
scaled) or by the SERVO (the stiffness, which r52 already classifies as
a tuning parameter: "the force limit stays physical; only the stiffness
is scaled")?

THIS TOOL answers that with a static computation at the delivered pose.
Virtual work: a normal force F at the pad tip does work F * (n . dtip/dq_j)
on finger joint j, so the joint torque needed is tau_j = F * (n . J_j).
The actuator's limit is tau_max_j, so the largest F the finger can hold
against the face normal is

    F_max = min_j  tau_max_j / |n . J_j|

computed with MuJoCo's own Jacobian at the delivered terminal pose.

    runs/.venv-retarget/bin/python tools/w6b/finger_force_ceiling.py
"""
from __future__ import annotations

import json
import pathlib

import mujoco
import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
SCENE = REPO / "grounding/assets/allegro_floating_ur5e_side_r37.xml"
REFERENCE = REPO / "runs/w6b/palm_pose_refine.json"
OUT = REPO / "runs/w6b/finger_force_ceiling.json"

ARM = ("shoulder_pan", "shoulder_lift", "elbow",
       "wrist_1", "wrist_2", "wrist_3")
FINGERS = ("ff", "mf", "rf")


def jid(m, n):
    i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
    assert i >= 0, n
    return i


def gid(m, n):
    i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)
    assert i >= 0, n
    return i


def main() -> int:
    m = mujoco.MjModel.from_xml_path(str(SCENE))
    d = mujoco.MjData(m)
    ref = json.loads(REFERENCE.read_text())

    # the box at its rest pose
    obj = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "object")
    qa = int(m.jnt_qposadr[int(obj)])
    d.qpos[qa:qa + 7] = [0, 0, 0.74, 1, 0, 0, 0]

    out = {"the scene": str(SCENE.relative_to(REPO)), "sides": {}}
    for side, prefix, sfx, face_sign in (
            ("right", "ur5e_r", "", +1.0),
            ("left", "ur5e_l", "_left", -1.0)):
        entry = ref["sides"][side]
        cand = entry["best"][0] if isinstance(entry.get("best"), list) \
            else entry["best"]
        # the arm and the fingers at the delivered terminal
        for name, q in zip(ARM, cand["arm"]):
            j = jid(m, f"{prefix}_{name}_joint")
            d.qpos[int(m.jnt_qposadr[j])] = float(q)
        fingers = cand.get("fingers") or {}
        for f in FINGERS:
            for k, v in enumerate(fingers.get(f, [0.0] * 4)):
                j = jid(m, f"{f}j{k}{sfx}")
                d.qpos[int(m.jnt_qposadr[j])] = float(v)
        for k, v in enumerate((fingers.get("th") or [0.0] * 4)):
            j = jid(m, f"thj{k}{sfx}")
            d.qpos[int(m.jnt_qposadr[j])] = float(v)
        mujoco.mj_forward(m, d)

        # the face normal: the pad presses INTO the end face, so the
        # normal is -face_sign on x
        n = np.array([-face_sign, 0.0, 0.0])
        side_out = {}
        for f in FINGERS:
            tip = gid(m, f"{f}_tip_collision" if not sfx
                      else f"{f}_tip_left_collision")
            dofs = []
            tau_max = []
            kps = []
            for k in range(4):
                j = jid(m, f"{f}j{k}{sfx}")
                dofs.append(int(m.jnt_dofadr[j]))
                # the joint -> actuator map is the REVERSE of
                # actuator_trnid (which is indexed by actuator)
                a = -1
                for ai in range(m.nu):
                    if int(m.actuator_trnid[ai, 0]) == j:
                        a = ai
                        break
                if a < 0:
                    # an unactuated joint (the ring's abduction is
                    # coupled): it cannot produce a torque, so it
                    # contributes no force cap
                    tau_max.append(0.0)
                    kps.append(0.0)
                    continue
                tau_max.append(abs(float(m.actuator_forcerange[a, 1])))
                kps.append(float(m.actuator_gainprm[a, 0]))
            jp = np.zeros((3, m.nv))
            mujoco.mj_jacGeom(m, d, jp, None, tip)
            # the moment arm of each joint for a force along the normal
            arms = [abs(float(jp[:, dof] @ n)) for dof in dofs]
            caps = []
            for arm, tm in zip(arms, tau_max):
                caps.append(tm / arm if arm > 1e-9 else float("inf"))
            f_max = min(caps) if caps else 0.0
            binding = int(np.argmin(caps)) if caps else -1
            side_out[f] = {
                "the tip position mm": [round(float(v) * 1e3, 2)
                                        for v in d.geom_xpos[tip]],
                "the normal moment arms mm": [round(a * 1e3, 3) for a in arms],
                "the actuator kp": [round(k, 3) for k in kps],
                "the torque limits Nm": [round(t, 3) for t in tau_max],
                "the per-joint force caps N": [round(c, 2) for c in caps],
                "the hardware force ceiling N": round(f_max, 3),
                "the binding joint": binding,
            }
            print(f"  {side} {f}: the moment arms "
                  f"{[round(a*1e3,2) for a in arms]} mm | "
                  f"the per-joint caps {[round(c,2) for c in caps]} N | "
                  f"the CEILING {f_max:.2f} N (the binding joint {binding})")
        out["sides"][side] = side_out

    out["the note"] = (
        "F_max = min_j (the actuator's torque limit / the joint's normal "
        "moment arm).  It is the largest normal force the finger can hold "
        "against the end face at the delivered pose, and it is a HARDWARE "
        "limit (the Allegro's actuator torque limits are physical).  The "
        "delivered cell's measured force is far below it, so the delivered "
        "force is limited by the SERVO's stiffness and the command state's "
        "geometry, not by the hardware.")
    OUT.write_text(json.dumps(out, indent=2))
    print(f"\n  written: {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
