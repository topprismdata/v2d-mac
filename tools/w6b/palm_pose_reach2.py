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

WHY A NEW SEARCH (r64).  The r63 search placed the palm so that the MEAN
fingertip offset landed on the face, then SCALED each finger's curl to
close the remaining gap.  Both steps are wrong:
  * scaling the curl MOVES the tips, so the placement (which assumed
    the known-good curl) no longer holds -- measured: the scaled curls
    left the pads 5-10 mm short;
  * the known-good grasp is NOT flat: its ring-finger tip sits 7.5 mm
    INSIDE the face (measured on the gate-3 seed), so reusing its
    offsets reproduces a buried pad.
And the search ignored the TABLE: the r63 winner drove the palm 28-56 mm
THROUGH the table slab, so the arm jammed (measured: the executed pad
distance diverged to 354-441 mm).

WHAT THIS SEARCH DOES.  For a candidate palm orientation (the known-good
palm rotation rolled about the face normal) and a candidate palm
position:
  1. the ARM IK solves the 6 arm joints for that palm pose;
  2. each FINGER's 4 joints are IK'd (a real Jacobian IK on the tip
     geom) so the pad's flat face reaches its OWN target point on the
     box's end face -- the targets are ON the face plane, so no pad is
     buried by construction;
  3. everything is MEASURED: the pad's signed distance to the box, its
     closest point's face, the finger's axis against the face normal,
     the table clearance, the joint limits and the IK residuals.

A negative result is a legitimate, publishable feasibility bound.

usage: palm_pose_reach2.py [scene.xml]
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
OUT = Path("runs/w6b/palm_pose_reach2.json")

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
    TABLE_TOP = float(sm.geom_pos[TABLE_G][2] + sm.geom_size[TABLE_G][2])
    FACE_X = float(HALF[0])
    out = {"scene": str(SCENE), "table_top_m": TABLE_TOP,
           "face_x_m": FACE_X, "sides": {}}

    for side, prefix, sfx, face_sign in (
            ("right", "ur5e_r", "", +1.0),
            ("left", "ur5e_l", "_left", -1.0)):
        n_in = np.array([-face_sign, 0.0, 0.0])   # the inward face normal
        # ---------- the known-good grasp, measured ----------
        PB = mujoco.mj_name2id(fm, mujoco.mjtObj.mjOBJ_BODY, "palm" + sfx)
        Rq = np.zeros(9)
        mujoco.mju_quat2Mat(Rq, fd.xquat[PB])
        R0 = np.array(Rq).reshape(3, 3).copy()
        Pp = np.array(fd.xpos[PB]).copy()
        offs, fingers = {}, {}
        for f in FINGERS:
            pg = gid(fm, f"{f}_tip_collision" if not sfx
                     else f"{f}_tip_left_collision")
            offs[f] = R0.T @ (np.array(fd.geom_xpos[pg]) - Pp)
            fingers[f] = [float(fd.qpos[int(fm.jnt_qposadr[
                jid(fm, f"{f}j{k}{sfx}")])]) for k in range(4)]
        off_mean = np.mean([offs[f] for f in FINGERS], axis=0)
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
        FQ = {f: [int(sm.jnt_qposadr[j]) for j in FJ[f]] for f in FINGERS}
        FD = {f: [int(sm.jnt_dofadr[j]) for j in FJ[f]] for f in FINGERS}
        FLO = {f: sm.jnt_range[FJ[f]][:, 0] for f in FINGERS}
        FHI = {f: sm.jnt_range[FJ[f]][:, 1] for f in FINGERS}
        PAD = {f: gid(sm, f"{f}_tip_collision" if side == "right"
                      else f"{f}_tip_left_collision") for f in FINGERS}
        DIST = {f: gid(sm, f"{f}_distal_collision" if side == "right"
                       else f"{f}_distal_left_collision") for f in FINGERS}
        HAND_GEOMS = [i for i in range(sm.ngeom)
                      if (mujoco.mj_id2name(sm, mujoco.mjtObj.mjOBJ_GEOM, i)
                          or "") and
                      ("collision" in (mujoco.mj_id2name(
                          sm, mujoco.mjtObj.mjOBJ_GEOM, i) or "")
                       or "tip" in (mujoco.mj_id2name(
                           sm, mujoco.mjtObj.mjOBJ_GEOM, i) or ""))
                      and (mujoco.mj_id2name(sm, mujoco.mjtObj.mjOBJ_GEOM, i)
                           not in ("object_geom", "object_chamfer_negx",
                                   "object_chamfer_posx"))]

        def state(q, fv):
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
                sd.qpos[FQ[ff]] = fv[ff]
            mujoco.mj_forward(sm, sd)

        def ik_arm(tp, Rt, q0):
            q = q0.copy()
            for _ in range(200):
                sd.qpos[:] = sm.qpos0
                sd.qpos[AQ] = q
                mujoco.mj_kinematics(sm, sd)
                mujoco.mj_comPos(sm, sd)
                e_p = tp - np.array(sd.xpos[PALM])
                Rq = np.zeros(9)
                mujoco.mju_quat2Mat(Rq, sd.xquat[PALM])
                Rrel = Rt.T @ np.array(Rq).reshape(3, 3)
                e_r = Rt @ (0.5 * np.array([Rrel[2, 1] - Rrel[1, 2],
                                            Rrel[0, 2] - Rrel[2, 0],
                                            Rrel[1, 0] - Rrel[0, 1]]))
                if np.linalg.norm(e_p) < 3e-4 and np.linalg.norm(e_r) < 4e-3:
                    break
                jacp = np.zeros((3, sm.nv))
                jacr = np.zeros((3, sm.nv))
                mujoco.mj_jacBody(sm, sd, jacp, jacr, PALM)
                J = np.vstack([jacp[:, AD], jacr[:, AD]])
                try:
                    dq = np.linalg.solve(J @ J.T + 1e-4 * np.eye(6),
                                         np.concatenate([e_p, e_r]))
                except np.linalg.LinAlgError:
                    break
                q = np.clip(q + J.T @ dq, ALO, AHI)
            sd.qpos[:] = sm.qpos0
            sd.qpos[AQ] = q
            mujoco.mj_kinematics(sm, sd)
            mujoco.mj_comPos(sm, sd)
            Rq = np.zeros(9)
            mujoco.mju_quat2Mat(Rq, sd.xquat[PALM])
            Rrel = Rt.T @ np.array(Rq).reshape(3, 3)
            ep = float(np.linalg.norm(tp - np.array(sd.xpos[PALM])))
            er = float(np.linalg.norm(0.5 * np.array(
                [Rrel[2, 1] - Rrel[1, 2], Rrel[0, 2] - Rrel[2, 0],
                 Rrel[1, 0] - Rrel[0, 1]])))
            mg = float(min(np.min(q - ALO), np.min(AHI - q)))
            return q, ep, er, mg

        def ik_finger(f, target, fv):
            """drive this finger's tip geom to the target point"""
            x = np.asarray(fv[f], float).copy()
            best = None
            for _ in range(120):
                state_q = None
                sd.qpos[AQ] = _ARM_Q[0]
                sd.qpos[FQ[f]] = x
                mujoco.mj_forward(sm, sd)
                cur = np.array(sd.geom_xpos[PAD[f]])
                e = target - cur
                if np.linalg.norm(e) < 5e-5:
                    best = x.copy()
                    break
                jacp = np.zeros((3, sm.nv))
                mujoco.mj_jacGeom(sm, sd, jacp, None, PAD[f])
                J = jacp[:, FD[f]]
                try:
                    dq = np.linalg.solve(J @ J.T + 1e-6 * np.eye(3), e)
                except np.linalg.LinAlgError:
                    break
                x = np.clip(x + J.T @ dq, FLO[f], FHI[f])
            state_q = None
            sd.qpos[AQ] = _ARM_Q[0]
            sd.qpos[FQ[f]] = x
            mujoco.mj_forward(sm, sd)
            res = float(np.linalg.norm(
                target - np.array(sd.geom_xpos[PAD[f]])))
            return x, res

        K = np.array([[0, -n_in[2], n_in[1]],
                      [n_in[2], 0, -n_in[0]],
                      [-n_in[1], n_in[0], 0]])
        # the three pads' targets ON the face plane: spread in y, one z
        tgt_sets = [
            [(0.015, 0.752), (-0.007, 0.765), (-0.071, 0.760)],
            [(0.015, 0.745), (-0.007, 0.758), (-0.060, 0.752)],
            [(0.020, 0.755), (-0.005, 0.768), (-0.065, 0.758)],
            [(0.010, 0.740), (-0.010, 0.752), (-0.055, 0.748)],
        ]
        rng = np.random.default_rng(7)
        _ARM_Q = [None]

        # ---------- stage 1: the ARM IK over the coarse sweep ----------
        coarse = []
        for roll in range(0, 360, 15):
            ang = np.deg2rad(float(roll))
            cr, sr = np.cos(ang), np.sin(ang)
            Rz = (np.eye(3) + sr * K + (1 - cr) * (K @ K))
            Rt = Rz @ R0
            for tset in tgt_sets:
                tmean = np.mean([np.array([face_sign * FACE_X, y, z])
                                 for y, z in tset], axis=0)
                for extra in (0.0, 0.010, 0.020, 0.030, 0.040, 0.050,
                              -0.010, -0.020, -0.030):
                    tp = tmean - Rt @ off_mean + np.array(
                        [face_sign * extra, 0.0, 0.0])
                    bq = None
                    for q_try in [np.zeros(6)] + [
                            rng.uniform(ALO, AHI) for _ in range(3)]:
                        qq, epp, errp, mgp = ik_arm(tp, Rt, q_try)
                        if bq is None or (epp + errp) < bq[1]:
                            bq = (qq, epp + errp, epp, errp, mgp)
                    q, ep, er, mg = bq[0], bq[2], bq[3], bq[4]
                    if ep > 2e-3 or er > 0.01 or mg < 0.05:
                        continue
                    coarse.append((ep + er, roll, tset, extra, q.copy(),
                                   Rt.copy(), tp.copy(), mg))
        coarse.sort(key=lambda t: t[0])
        print(f"    {side}: {len(coarse)} arm-reachable candidates")

        # ---------- stage 2: the per-finger IK + the measurements ----------
        best = None
        n_fik_fail = 0
        n_clear_fail = 0
        n_clear_hist = []
        for (_c, roll, tset, extra, q, Rt, tp, mg) in coarse[:40]:
            _ARM_Q[0] = q.copy()
            fv = {f: list(fingers[f]) for f in FINGERS}
            fv_res = {}
            fok = True
            for fi, f in enumerate(FINGERS):
                tgt = np.array([face_sign * FACE_X, tset[fi][0],
                                tset[fi][1]])
                x, res = ik_finger(f, tgt, fv)
                if res > 4e-3:
                    fok = False
                    fv_res[f] = res
                    break
                fv[f] = list(x)
            if not fok:
                n_fik_fail += 1
                if len(fv_res) and n_fik_fail <= 6:
                    print(f"    [fik] roll {roll} extra {extra*1e3:+.0f} "
                          f"the finger residuals (mm): "
                          f"{ {k: round(v*1e3,1) for k, v in fv_res.items()} }")
                continue
            state(q, fv)
            # the table clearance
            clear = 0.0
            ok_clear = True
            for gi in HAND_GEOMS:
                hz = float(sm.geom_size[gi][2])
                bot = float(sd.geom_xpos[gi][2]) - hz
                clear = min(clear, bot - TABLE_TOP)
                if bot < TABLE_TOP - 1e-3:
                    ok_clear = False
            if not ok_clear:
                n_clear_fail += 1
                n_clear_hist.append(clear)
                continue
            # the measurements
            a2, pd2 = {}, {}
            for f in FINGERS:
                Rd = np.array(sd.geom_xmat[DIST[f]]).reshape(3, 3)
                a2[f] = float(np.degrees(np.arccos(
                    abs(float(Rd[:, 2] @ n_in)))))
                ft = np.zeros(6)
                dd = float(mujoco.mj_geomDistance(sm, sd, PAD[f], OBJ_G,
                                                  1.0, ft))
                rel = ft[3:6] - np.array(sd.xpos[OBJ_B])
                pd2[f] = (dd, bool(abs(abs(rel[0]) - FACE_X) < 2e-3))
            reach = max(abs(v[0]) for v in pd2.values())
            n_end = sum(1 for v in pd2.values() if v[1])
            worst = (max(a2.values()) + 60.0 * min(reach, 0.02)
                     + 30.0 * (3 - n_end) - 40.0 * max(clear, 0.0))
            if best is None or worst < best[0]:
                best = (worst, roll, tset, extra, q.copy(), fv, a2, pd2,
                        ep, er, mg, clear, Rt.copy())
        if best is None:
            print(f"=== {side}: NO flat-pad pose with table clearance ===")
            print(f"    the per-finger IK failures: {n_fik_fail}, "
                  f"the table-clearance failures: {n_clear_fail}")
            if n_clear_hist:
                print(f"    the clearance histogram (mm): "
                      f"{sorted(round(v*1e3,1) for v in n_clear_hist)[:12]}")
            out["sides"][side] = {"feasible": False}
            continue
        (worst, roll, tset, extra, q, fv, a2, pd2, ep, er, mg, clear,
         Rt) = best
        state(q, fv)
        Rq = np.zeros(9)
        mujoco.mju_quat2Mat(Rq, sd.xquat[PALM])
        print(f"=== {side} ===")
        print(f"  the best pose: roll {roll} deg, the palm standoff "
              f"{extra*1e3:+.0f} mm, the targets {tset}")
        print(f"    the arm IK |e_p| = {ep*1e3:.3f} mm, |e_r| = "
              f"{np.degrees(er):.2f} deg, the joint margin {mg:.4f} rad")
        print(f"    the finger axes vs the face normal: "
              f"{ {k: round(v,1) for k, v in a2.items()} } deg")
        print(f"    the pads' distance to the box: "
              f"{ {k: round(v[0]*1e3,2) for k, v in pd2.items()} } mm")
        print(f"    the pads on the end face: "
              f"{sum(1 for v in pd2.values() if v[1])} of 3")
        print(f"    the table clearance: {clear*1e3:.1f} mm")
        print(f"    the arm joints: {np.round(q,4).tolist()}")
        out["sides"][side] = {
            "feasible": True,
            "roll_deg": roll, "targets": tset,
            "palm_standoff_mm": extra * 1e3,
            "e_p_mm": ep * 1e3, "e_r_deg": float(np.degrees(er)),
            "joint_margin_rad": mg, "table_clearance_mm": clear * 1e3,
            "finger_axis_deg": {k: round(v, 2) for k, v in a2.items()},
            "pad_distance_mm": {k: round(v[0] * 1e3, 3)
                                for k, v in pd2.items()},
            "pad_on_end_face": {k: bool(v[1]) for k, v in pd2.items()},
            "arm": [round(float(v), 6) for v in q],
            "fingers": {f: [round(float(v), 6) for v in fv[f]]
                        for f in FINGERS},
        }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
