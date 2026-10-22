#!/usr/bin/env python3
"""Can the side cell's UR5e reach the KNOWN-GOOD palm pose?

The r63 measurement: the known-good floating grasp (the gate-3 seed,
gate1 PASS) has the index finger's axis 5.4 deg from the end face's
normal -- the pad lies FLAT on the box.  The side cell has all three
fingers at 90 deg: the pads present their 12 mm EDGE and the 16 mm
distal links hit the box first.

The floating scene and the side cell have the SAME hand and the SAME
box, so the known-good palm POSE is a measured, working configuration in
the same task frame.  The only question is whether the side-mount arm
can REACH it.

This measures that, and the reachable neighbourhood of it:
  * the arm IK for the known-good palm pose (position + rotation), with
    the joint-limit margin and the residual;
  * the sweep over the palm pose's roll about the face normal, its
    standoff from the face and its lateral/height offset, so the
    reachable set is MAPPED rather than guessed;
  * for every reachable candidate, the MEASURED finger-axis angles to
    the face normal.

A NEGATIVE result here is a legitimate, publishable feasibility bound
(the review asks for exactly that when the path is infeasible).

usage: palm_pose_reach.py [scene.xml]
"""
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

SCENE = Path(sys.argv[1] if len(sys.argv) > 1 else
              "grounding/assets/allegro_floating_ur5e_side_r37.xml")
FLOAT_SCENE = Path("grounding/assets/allegro_floating.xml")
SEED = Path("runs/w6/gate3_qpos.json")
OUT = Path("runs/w6b/palm_pose_reach.json")

ARM = ("shoulder_pan", "shoulder_lift", "elbow",
       "wrist_1", "wrist_2", "wrist_3")
FINGERS = ("ff", "mf", "rf")


def main():
    sm = mujoco.MjModel.from_xml_path(str(SCENE))
    sd = mujoco.MjData(sm)
    fm = mujoco.MjModel.from_xml_path(str(FLOAT_SCENE))
    fd = mujoco.MjData(fm)

    def jid(m, n):
        i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
        assert i >= 0, n
        return i

    def gid(m, n):
        i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)
        assert i >= 0, n
        return i

    fd.qpos[:] = np.asarray(json.loads(SEED.read_text())["qpos"], float)
    for _jn in range(fm.njnt):
        nm = mujoco.mj_id2name(fm, mujoco.mjtObj.mjOBJ_JOINT, _jn)
        if nm and nm.startswith("object"):
            qa = int(fm.jnt_qposadr[_jn])
            fd.qpos[qa:qa + 7] = [0, 0, 0.74, 1, 0, 0, 0]
            break
    mujoco.mj_forward(fm, fd)
    good = {}
    for side, sfx in (("right", ""), ("left", "_left")):
        pb = mujoco.mj_name2id(fm, mujoco.mjtObj.mjOBJ_BODY, "palm" + sfx)
        Rq = np.zeros(9)
        mujoco.mju_quat2Mat(Rq, fd.xquat[pb])
        fingers = {}
        for f in FINGERS:
            fingers[f] = [float(fd.qpos[int(fm.jnt_qposadr[
                jid(fm, f"{f}j{k}{sfx}")])]) for k in range(4)]
        good[side] = (np.array(fd.xpos[pb]), Rq.reshape(3, 3).copy(), fingers)
        print(f"the known-good {side} palm: pos "
              f"{np.round(fd.xpos[pb]*1e3,1)} mm")

    OBJ_G = gid(sm, "object_geom")
    HALF = np.array(sm.geom_size[OBJ_G])
    out = {"scene": str(SCENE), "sides": {}}

    for side, prefix, sfx, face_sign in (
            ("right", "ur5e_r", "", +1.0),
            ("left", "ur5e_l", "_left", -1.0)):
        AJ = [jid(sm, f"{prefix}_{n}_joint") for n in ARM]
        AQ = [int(sm.jnt_qposadr[j]) for j in AJ]
        AD = [int(sm.jnt_dofadr[j]) for j in AJ]
        ALO = sm.jnt_range[AJ][:, 0]
        AHI = sm.jnt_range[AJ][:, 1]
        PALM = mujoco.mj_name2id(sm, mujoco.mjtObj.mjOBJ_BODY, "palm" + sfx)
        FJ = {f: [jid(sm, f"{f}j{k}{sfx}") for k in range(4)]
              for f in FINGERS}
        DIST = {f: gid(sm, f"{f}_distal_collision" if side == "right"
                       else f"{f}_distal_left_collision") for f in FINGERS}
        n_in = np.array([-face_sign, 0.0, 0.0])
        gp, gR, gfingers = good[side]
        Rg = gR.copy()
        if side == "left":
            Rg = np.diag([-1.0, 1.0, 1.0]) @ Rg @ np.diag([-1.0, 1.0, 1.0])

        def state(q, fingers):
            mujoco.mj_resetData(sm, sd)
            sd.qpos[:] = sm.qpos0
            for _jn in range(sm.njnt):
                nm = mujoco.mj_id2name(sm, mujoco.mjtObj.mjOBJ_JOINT, _jn)
                if nm and nm.startswith("object"):
                    qa = int(sm.jnt_qposadr[_jn])
                    sd.qpos[qa:qa + 7] = [0, 0, 0.74, 1, 0, 0, 0]
                    break
            sd.qpos[AQ] = q
            for f in FINGERS:
                for k, j in enumerate(FJ[f]):
                    sd.qpos[int(sm.jnt_qposadr[j])] = fingers[f][k]
            mujoco.mj_forward(sm, sd)

        def ik(target_pos, target_R, q0):
            q = q0.copy()
            for _ in range(300):
                sd.qpos[:] = sm.qpos0
                sd.qpos[AQ] = q
                mujoco.mj_kinematics(sm, sd)
                mujoco.mj_comPos(sm, sd)
                e_p = target_pos - np.array(sd.xpos[PALM])
                Rq = np.zeros(9)
                mujoco.mju_quat2Mat(Rq, sd.xquat[PALM])
                R = np.array(Rq).reshape(3, 3)
                Rrel = target_R.T @ R
                e_r = target_R @ (0.5 * np.array([Rrel[2, 1] - Rrel[1, 2],
                                                   Rrel[0, 2] - Rrel[2, 0],
                                                   Rrel[1, 0] - Rrel[0, 1]]))
                err = np.concatenate([e_p, e_r])
                if np.linalg.norm(e_p) < 3e-4 and np.linalg.norm(e_r) < 4e-3:
                    break
                jacp = np.zeros((3, sm.nv))
                jacr = np.zeros((3, sm.nv))
                mujoco.mj_jacBody(sm, sd, jacp, jacr, PALM)
                J = np.vstack([jacp[:, AD], jacr[:, AD]])
                try:
                    dq = np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), err)
                except np.linalg.LinAlgError:
                    break
                q = np.clip(q + J.T @ dq, ALO, AHI)
            sd.qpos[:] = sm.qpos0
            sd.qpos[AQ] = q
            mujoco.mj_kinematics(sm, sd)
            mujoco.mj_comPos(sm, sd)
            Rq = np.zeros(9)
            mujoco.mju_quat2Mat(Rq, sd.xquat[PALM])
            R = np.array(Rq).reshape(3, 3)
            Rrel = target_R.T @ R
            ep = float(np.linalg.norm(target_pos - np.array(sd.xpos[PALM])))
            er = float(np.linalg.norm(0.5 * np.array(
                [Rrel[2, 1] - Rrel[1, 2], Rrel[0, 2] - Rrel[2, 0],
                 Rrel[1, 0] - Rrel[0, 1]])))
            mg = float(min(np.min(q - ALO), np.min(AHI - q)))
            return q, ep, er, mg

        q_direct, ep, er, mg = ik(gp, Rg, np.zeros(6))
        state(q_direct, gfingers)
        angs = {}
        for f in FINGERS:
            Rd = np.array(sd.geom_xmat[DIST[f]]).reshape(3, 3)
            angs[f] = float(np.degrees(np.arccos(
                abs(float(Rd[:, 2] @ n_in)))))
        print(f"\n=== {side} ===")
        print(f"  the known-good palm pose, straight into the side cell:")
        print(f"    IK residual |e_p| = {ep*1e3:.2f} mm, |e_r| = "
              f"{np.degrees(er):.2f} deg, joint margin {mg:.4f} rad")
        print(f"    the finger axes vs the face normal: "
              f"{ {k: round(v,1) for k, v in angs.items()} } deg")
        print(f"    => {'REACHABLE' if ep < 5e-3 and er < 0.05 and mg > 0.02 else 'NOT reachable at the known-good pose'}")

        best = None
        rng = np.random.default_rng(99)
        K = np.array([[0, -n_in[2], n_in[1]],
                      [n_in[2], 0, -n_in[0]],
                      [-n_in[1], n_in[0], 0]])
        for standoff in (0.0, 0.01, 0.02, 0.03, 0.04, 0.06, 0.08):
            for roll in range(0, 360, 15):
                for lat in (-0.02, 0.0, 0.02):
                    for zc in (0.72, 0.74, 0.76):
                        tp = gp + np.array([face_sign * standoff, lat,
                                            zc - 0.74])
                        ang = np.deg2rad(roll)
                        cr, sr = np.cos(ang), np.sin(ang)
                        Rt = (np.eye(3) + sr * K + (1 - cr) * (K @ K)) @ Rg
                        bestq = None
                        for q_try in [np.zeros(6)] + [rng.uniform(ALO, AHI)
                                                     for _ in range(3)]:
                            qq, epp, errp, mgp = ik(tp, Rt, q_try)
                            if bestq is None or (epp + errp) < bestq[1]:
                                bestq = (qq, epp + errp, epp, errp, mgp)
                        q, ep, er, mg = (bestq[0], bestq[2], bestq[3], bestq[4])
                        if ep > 8e-3 or er > 0.06 or mg < 0.03:
                            continue
                        state(q, gfingers)
                        a2 = {}
                        for f in FINGERS:
                            Rd = np.array(sd.geom_xmat[DIST[f]]).reshape(3, 3)
                            a2[f] = float(np.degrees(np.arccos(
                                abs(float(Rd[:, 2] @ n_in)))))
                        worst = max(a2.values())
                        if best is None or worst < best[0]:
                            best = (worst, standoff, roll, lat, zc, q.copy(),
                                    ep, er, mg, a2, Rt.copy(), tp.copy())
        if best is None:
            print("  the reachable neighbourhood: EMPTY")
            out["sides"][side] = {"known_good_reachable": False,
                                  "neighbourhood": "empty"}
            continue
        (worst, standoff, roll, lat, zc, q, ep, er, mg, a2, Rt,
         tp) = best
        print(f"  the best reachable pose in the neighbourhood:")
        print(f"    standoff {standoff*1e3:.0f} mm, roll {roll} deg, "
              f"lateral {lat*1e3:+.0f} mm, height {zc*1e3:.0f} mm")
        print(f"    IK residual |e_p| = {ep*1e3:.2f} mm, |e_r| = "
              f"{np.degrees(er):.2f} deg, joint margin {mg:.4f} rad")
        print(f"    the finger axes vs the face normal: "
              f"{ {k: round(v,1) for k, v in a2.items()} } deg")
        print(f"    the arm joints: {np.round(q,4).tolist()}")
        out["sides"][side] = {
            "known_good_reachable": bool(ep < 5e-3 and er < 0.05 and mg > 0.02),
            "known_good_e_p_mm": ep * 1e3,
            "known_good_e_r_deg": float(np.degrees(er)),
            "known_good_joint_margin": mg,
            "standoff_mm": standoff * 1e3, "roll_deg": roll,
            "lateral_mm": lat * 1e3, "height_mm": zc * 1e3,
            "e_p_mm": ep * 1e3, "e_r_deg": float(np.degrees(er)),
            "joint_margin_rad": mg,
            "finger_axis_deg": {k: round(v, 2) for k, v in a2.items()},
            "arm": [round(float(v), 6) for v in q],
            "palm_R": [[round(float(v), 6) for v in row] for row in Rt],
            "palm_pos": [round(float(v), 6) for v in tp],
            "fingers": {f: [round(float(v), 6) for v in gfingers[f]]
                        for f in FINGERS},
        }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
