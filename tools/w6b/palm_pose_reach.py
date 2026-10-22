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

"""Find the side cell's terminal palm POSE for a FLAT-pad grasp.

THE MEASURED ROOT CAUSE (r63): in the known-good floating grasp (the
gate-3 seed, gate1 PASS) the index finger's axis is 5.4 deg from the end
face's normal, so the pad's 12 x 12 mm FLAT face presses flat on the box.
In the side cell all three finger axes stand at 90.0 deg: the fingers lie
IN the face plane, the pads present their 12 mm EDGE, and the 16 mm
distal links reach the box before the pads do (the executed contact list
is on the distal links).  An edge-on pad cannot hold -- the reaction is
taken by the finger's own force-limited servo (0.04-0.18 N.m), so the
fingers collapse (measured up to 0.87 rad) and only the deepest pad
stays on the face.

THE FORMULATION.  Measured in the known-good grasp:
  * the palm's WORLD rotation R0 (its fingers point into the face),
  * the fingertips' offset in the PALM frame (the pads sit ~[70, -3, 92]
    mm from the palm origin), and
  * the fingertips' axis in the PALM frame.
For a candidate palm rotation R (R0 rolled about the face normal -- the
one free parameter) the palm's position is
    p = face_point - R @ tip_offset
which puts the pads ON the face by construction; the palm's standoff
along the face normal is then swept, because the finger curl alone
closes the gap only to ~15 mm (measured).

For every candidate the arm IK is solved, and MEASURED: the residual,
the joint-limit margin, the finger-axis angles to the face normal, each
pad's true distance to the box (MuJoCo's own mj_geomDistance) and
whether the pad's closest point is on the end face.

A NEGATIVE result is a legitimate, publishable feasibility bound (the
review asks for exactly that when the path is infeasible).

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


def jid(m, n):
    i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
    assert i >= 0, n
    return i


def gid(m, n):
    i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)
    assert i >= 0, n
    return i


def main():
    sm = mujoco.MjModel.from_xml_path(str(SCENE))
    sd = mujoco.MjData(sm)
    fm = mujoco.MjModel.from_xml_path(str(FLOAT_SCENE))
    fd = mujoco.MjData(fm)

    fd.qpos[:] = np.asarray(json.loads(SEED.read_text())["qpos"], float)
    for _jn in range(fm.njnt):
        nm = mujoco.mj_id2name(fm, mujoco.mjtObj.mjOBJ_JOINT, _jn)
        if nm and nm.startswith("object"):
            qa = int(fm.jnt_qposadr[_jn])
            fd.qpos[qa:qa + 7] = [0, 0, 0.74, 1, 0, 0, 0]
            break
    mujoco.mj_forward(fm, fd)

    OBJ_G = gid(sm, "object_geom")
    OBJ_B = mujoco.mj_name2id(sm, mujoco.mjtObj.mjOBJ_BODY, "object")
    HALF = np.array(sm.geom_size[OBJ_G])
    TABLE_G = gid(sm, "table_slab")
    # r64: the table's top face -- a hand geom below it is inside the
    # table (measured r63: the searched pose put the palm 28-56 mm
    # THROUGH the slab, so the arm jammed and the executed state
    # diverged by 350-450 mm).
    TABLE_TOP = float(sm.geom_pos[TABLE_G][2]
                       + sm.geom_size[TABLE_G][2])
    out = {"scene": str(SCENE), "sides": {}}

    for side, prefix, sfx, face_sign in (
            ("right", "ur5e_r", "", +1.0),
            ("left", "ur5e_l", "_left", -1.0)):
        n_in = np.array([-face_sign, 0.0, 0.0])
        # ---------- the known-good grasp, measured ----------
        PB = mujoco.mj_name2id(fm, mujoco.mjtObj.mjOBJ_BODY, "palm" + sfx)
        Rq = np.zeros(9)
        mujoco.mju_quat2Mat(Rq, fd.xquat[PB])
        R0 = np.array(Rq).reshape(3, 3).copy()
        Pp = np.array(fd.xpos[PB]).copy()
        offs = []
        fingers = {}
        for f in FINGERS:
            pg = gid(fm, f"{f}_tip_collision" if not sfx
                     else f"{f}_tip_left_collision")
            offs.append(R0.T @ (np.array(fd.geom_xpos[pg]) - Pp))
            fingers[f] = [float(fd.qpos[int(fm.jnt_qposadr[
                jid(fm, f"{f}j{k}{sfx}")])]) for k in range(4)]
        off_mean = np.mean(offs, axis=0)
        print(f"the known-good {side} grasp: the palm at "
              f"{np.round(Pp*1e3,1)} mm; the mean fingertip offset in the "
              f"PALM frame {np.round(off_mean*1e3,1)} mm")

        # ---------- the side cell ----------
        AJ = [jid(sm, f"{prefix}_{n}_joint") for n in ARM]
        AQ = [int(sm.jnt_qposadr[j]) for j in AJ]
        AD = [int(sm.jnt_dofadr[j]) for j in AJ]
        ALO = sm.jnt_range[AJ][:, 0]
        AHI = sm.jnt_range[AJ][:, 1]
        PALM = mujoco.mj_name2id(sm, mujoco.mjtObj.mjOBJ_BODY, "palm" + sfx)
        FJ = {f: [jid(sm, f"{f}j{k}{sfx}") for k in range(4)]
              for f in FINGERS}
        PAD = {f: gid(sm, f"{f}_tip_collision" if side == "right"
                      else f"{f}_tip_left_collision") for f in FINGERS}
        DIST = {f: gid(sm, f"{f}_distal_collision" if side == "right"
                       else f"{f}_distal_left_collision") for f in FINGERS}

        def state(q, f):
            mujoco.mj_resetData(sm, sd)
            sd.qpos[:] = sm.qpos0
            for _jn in range(sm.njnt):
                nm = mujoco.mj_id2name(sm, mujoco.mjtObj.mjOBJ_JOINT, _jn)
                if nm and nm.startswith("object"):
                    qa = int(sm.jnt_qposadr[_jn])
                    sd.qpos[qa:qa + 7] = [0, 0, 0.74, 1, 0, 0, 0]
                    break
            sd.qpos[AQ] = q
            for ff in FINGERS:
                for k, j in enumerate(FJ[ff]):
                    sd.qpos[int(sm.jnt_qposadr[j])] = f[ff][k]
            mujoco.mj_forward(sm, sd)

        def ik(tp, Rt, q0):
            q = q0.copy()
            for _ in range(300):
                sd.qpos[:] = sm.qpos0
                sd.qpos[AQ] = q
                mujoco.mj_kinematics(sm, sd)
                mujoco.mj_comPos(sm, sd)
                e_p = tp - np.array(sd.xpos[PALM])
                Rq = np.zeros(9)
                mujoco.mju_quat2Mat(Rq, sd.xquat[PALM])
                R = np.array(Rq).reshape(3, 3)
                Rrel = Rt.T @ R
                e_r = Rt @ (0.5 * np.array([Rrel[2, 1] - Rrel[1, 2],
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
            Rrel = Rt.T @ R
            ep = float(np.linalg.norm(tp - np.array(sd.xpos[PALM])))
            er = float(np.linalg.norm(0.5 * np.array(
                [Rrel[2, 1] - Rrel[1, 2], Rrel[0, 2] - Rrel[2, 0],
                 Rrel[1, 0] - Rrel[0, 1]])))
            mg = float(min(np.min(q - ALO), np.min(AHI - q)))
            return q, ep, er, mg

        def solve_curl(q, f, which):
            """the per-finger curl scale that brings THAT pad to the box"""
            base = np.array(f[which])
            lo_s, hi_s = 0.05, 3.0
            prev_s, prev_g = None, None
            for sc in np.linspace(lo_s, hi_s, 41):
                cand = {g: list(f[g]) for g in FINGERS}
                cand[which] = list(base * sc)
                state(q, cand)
                ft = np.zeros(6)
                gg = float(mujoco.mj_geomDistance(sm, sd, PAD[which], OBJ_G,
                                                  1.0, ft))
                if prev_g is not None and prev_g * gg <= 0:
                    lo_s, hi_s = prev_s, float(sc)
                    break
                prev_s, prev_g = float(sc), gg
            else:
                return 1.0, False
            for _ in range(18):
                mid = 0.5 * (lo_s + hi_s)
                cand = {g: list(f[g]) for g in FINGERS}
                cand[which] = list(base * mid)
                state(q, cand)
                ft = np.zeros(6)
                gm = float(mujoco.mj_geomDistance(sm, sd, PAD[which], OBJ_G,
                                                  1.0, ft))
                cand = {g: list(f[g]) for g in FINGERS}
                cand[which] = list(base * lo_s)
                state(q, cand)
                ft = np.zeros(6)
                gl = float(mujoco.mj_geomDistance(sm, sd, PAD[which], OBJ_G,
                                                  1.0, ft))
                if gl * gm <= 0:
                    hi_s = mid
                else:
                    lo_s = mid
            return 0.5 * (lo_s + hi_s), True

        K = np.array([[0, -n_in[2], n_in[1]],
                      [n_in[2], 0, -n_in[0]],
                      [-n_in[1], n_in[0], 0]])
        face_pts = [(0.0, 0.75), (0.03, 0.75), (-0.03, 0.75),
                    (0.0, 0.765), (0.0, 0.735), (0.02, 0.765),
                    (-0.02, 0.765), (0.0, 0.71), (0.0, 0.78)]
        rng = np.random.default_rng(99)
        best = None
        for roll in range(0, 360, 10):
            ang = np.deg2rad(float(roll))
            cr, sr = np.cos(ang), np.sin(ang)
            Rz = (np.eye(3) + sr * K + (1 - cr) * (K @ K))
            Rt = Rz @ R0
            for fy, fz in face_pts:
                face_pt = np.array([face_sign * float(HALF[0]), fy, fz])
                tp0 = face_pt - Rt @ off_mean
                for extra in (0.0, 0.005, 0.010, 0.015, 0.020,
                              -0.005, -0.010, -0.015, -0.020,
                              -0.025, -0.030, -0.035, -0.040, -0.045,
                              -0.050, -0.055, -0.060, -0.065, -0.070):
                    tp = tp0 + np.array([face_sign * extra, 0.0, 0.0])
                    bq = None
                    for q_try in [np.zeros(6)] + [
                            rng.uniform(ALO, AHI) for _ in range(4)]:
                        qq, epp, errp, mgp = ik(tp, Rt, q_try)
                        if bq is None or (epp + errp) < bq[1]:
                            bq = (qq, epp + errp, epp, errp, mgp)
                    q, ep, er, mg = bq[0], bq[2], bq[3], bq[4]
                    # r63: a tight IK residual is REQUIRED -- a pose the
                    # arm cannot actually hold is not a solution (the
                    # right hand's first pass had a 3.9 mm residual, i.e.
                    # the palm was 3.9 mm from where the pads need it)
                    if ep > 1.5e-3 or er > 0.01 or mg < 0.03:
                        continue
                    # the fingers for this candidate
                    fs = {f: list(fingers[f]) for f in FINGERS}
                    for f in FINGERS:
                        sc, found = solve_curl(q, fs, f)
                        fs[f] = list(np.array(fingers[f]) * sc)
                    state(q, fs)
                    # r64: the TABLE clearance -- no hand geom may be
                    # below the slab's top face (the pads must be on the
                    # box's end face, which is ABOVE the table).
                    _clear = True
                    _worst_clear = 0.0
                    for _gi2 in range(sm.ngeom):
                        _gn2 = mujoco.mj_id2name(sm, mujoco.mjtObj.mjOBJ_GEOM,
                                                 _gi2)
                        if not _gn2:
                            continue
                        if ("collision" not in _gn2
                                and "tip" not in _gn2):
                            continue
                        if _gn2 in ("object_geom", "object_chamfer_negx",
                                    "object_chamfer_posx"):
                            continue
                        _bt2 = float(sm.geom_pos[_gi2][2]
                                     + sm.geom_size[_gi2][2])
                        _zc2 = float(sd.geom_xpos[_gi2][2])
                        _bot2 = _zc2 - _bt2
                        _worst_clear = min(_worst_clear, _bot2 - TABLE_TOP)
                        if _bot2 < TABLE_TOP - 1e-3:
                            _clear = False
                    if not _clear:
                        continue
                    a2 = {}
                    for f in FINGERS:
                        Rd = np.array(sd.geom_xmat[DIST[f]]).reshape(3, 3)
                        a2[f] = float(np.degrees(np.arccos(
                            abs(float(Rd[:, 2] @ n_in)))))
                    pd2 = {}
                    for f in FINGERS:
                        ft = np.zeros(6)
                        dd = float(mujoco.mj_geomDistance(sm, sd, PAD[f],
                                                          OBJ_G, 1.0, ft))
                        rel = ft[3:6] - np.array(sd.xpos[OBJ_B])
                        pd2[f] = (dd, bool(abs(abs(rel[0]) -
                                                float(HALF[0])) < 2e-3))
                    reach = max(abs(v[0]) for v in pd2.values())
                    # the pads must be ON THE END FACE, not merely touching
                    # the box somewhere (the ring finger can land on the
                    # top edge -- the finite-face effect of r63b)
                    n_end = sum(1 for v in pd2.values() if v[1])
                    worst = (max(a2.values()) + 40.0 * min(reach, 20.0)
                             + 30.0 * (3 - n_end)
                             - 60.0 * max(_worst_clear, 0.0))
                    if best is None or worst < best[0]:
                        best = (worst, roll, fy, fz, float(extra), q.copy(),
                                ep, er, mg, a2, pd2, Rt.copy(), tp.copy(), fs)
        if best is None:
            print(f"=== {side}: NO reachable flat-pad pose in the sweep ===")
            out["sides"][side] = {"feasible": False}
            continue
        (worst, roll, fy, fz, extra, q, ep, er, mg, a2, pd2, Rt, tp,
         fs) = best
        state(q, fs)
        Rq = np.zeros(9)
        mujoco.mju_quat2Mat(Rq, sd.xquat[PALM])
        Rach = np.array(Rq).reshape(3, 3).copy()
        pach = np.array(sd.xpos[PALM]).copy()
        print(f"=== {side} ===")
        print(f"  the best reachable pose: roll {roll} deg, the face point "
              f"(y {fy*1e3:+.0f}, z {fz*1e3:.0f}) mm, the palm standoff "
              f"{extra*1e3:+.0f} mm")
        print(f"    IK residual |e_p| = {ep*1e3:.3f} mm, |e_r| = "
              f"{np.degrees(er):.2f} deg, joint margin {mg:.4f} rad")
        print(f"    the finger axes vs the face normal: "
              f"{ {k: round(v,1) for k, v in a2.items()} } deg")
        print(f"    the pads' distance to the box: "
              f"{ {k: round(v[0]*1e3,2) for k, v in pd2.items()} } mm")
        print(f"    the pads on the end face: "
              f"{sum(1 for v in pd2.values() if v[1])} of 3")
        print(f"    the arm joints: {np.round(q,4).tolist()}")
        out["sides"][side] = {
            "feasible": True,
            "roll_deg": roll, "face_y_mm": fy * 1e3, "face_z_mm": fz * 1e3,
            "palm_standoff_mm": extra * 1e3,
            "e_p_mm": ep * 1e3, "e_r_deg": float(np.degrees(er)),
            "joint_margin_rad": mg,
            "finger_axis_deg": {k: round(v, 2) for k, v in a2.items()},
            "pad_distance_mm": {k: round(v[0] * 1e3, 3)
                                for k, v in pd2.items()},
            "pad_on_end_face": {k: bool(v[1]) for k, v in pd2.items()},
            "arm": [round(float(v), 6) for v in q],
            "palm_R": [[round(float(v), 6) for v in row] for row in Rach],
            "palm_pos": [round(float(v), 6) for v in pach],
            "fingers": {f: [round(float(v), 6) for v in fs[f]]
                        for f in FINGERS},
        }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
