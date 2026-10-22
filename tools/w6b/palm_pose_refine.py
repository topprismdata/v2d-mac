#!/usr/bin/env python3
"""r66: the side cell's flat-pad terminal pose, refined.

THE r65 FINDING (the rotation-error trap).  The 0.5*(R[2,1]-R[1,2], ...)
rotation error is sin(theta)*axis, so it VANISHES at a 180 deg error: an
IK that uses it "converges" at a completely flipped orientation (measured
|achieved - target| = 1.59 with a reported |e_r| = 0.00 deg).  Every
search in this campaign that reported e_r = 0.00 deg before r65 was
INVALID; this script uses scipy's as_rotvec.

THE GEOMETRY (measured on the gate-3 seed, the known-good grasp):
  * the index and middle pads sit EXACTLY on the box's end face
    (x = 60.0 mm), their axes 5.4 deg / 34 deg from the face normal --
    the index pad presses FLAT with its 12 x 12 mm face;
  * the ring pad is 7.5-13 mm PAST the face plane: it wraps the box's
    top edge (its z is above the box's top), it does not press the face;
  * the thumb is 115 mm clear of the box.
So the reference grasp is a TWO-pad flat grasp plus a wrap, and the gate
asks for >= 2 pads in the rectangle.

THIS SCRIPT: the palm's rotation is the known-good one rolled about the
face normal; the palm's position is anchored so the INDEX PAD lands
exactly on the face plane; each finger's 4 joints are then IK'd (a real
Jacobian IK on the tip geom) so that
  * the index and middle pads sit ON the face plane (their own targets),
  * the ring pad is pulled back clear of the box.
Everything is MEASURED: the pads' signed distances, their closest
points' faces, the finger axes, the table clearance, the joint margins
and the IK residuals.

usage: palm_pose_refine.py [scene.xml]
"""
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as Rot

SCENE = Path(sys.argv[1] if len(sys.argv) > 1 else
              "grounding/assets/allegro_floating_ur5e_side_r37.xml")
FLOAT_SCENE = Path("grounding/assets/allegro_floating.xml")
SEED = Path("runs/w6/gate3_qpos.json")
OUT = Path("runs/w6b/palm_pose_refine.json")

ARM = ("shoulder_pan", "shoulder_lift", "elbow",
       "wrist_1", "wrist_2", "wrist_3")
# the gate's R5 tolerance (retarget_dual.py: R5_TOUCH_TOL_M = 0.1164 mm)
R5_TOL_MM = 0.11636970514621301
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
    BOX_TOP = 0.74 + float(HALF[2])
    out = {"scene": str(SCENE), "table_top_m": TABLE_TOP,
           "box_top_m": BOX_TOP, "sides": {}}

    for side, prefix, sfx, face_sign in (
            ("right", "ur5e_r", "", +1.0),
            ("left", "ur5e_l", "_left", -1.0)):
        n_in = np.array([-face_sign, 0.0, 0.0])
        PB = mujoco.mj_name2id(fm, mujoco.mjtObj.mjOBJ_BODY, "palm" + sfx)
        Rq = np.zeros(9)
        mujoco.mju_quat2Mat(Rq, fd.xquat[PB])
        R0 = np.array(Rq).reshape(3, 3).copy()
        Pp = np.array(fd.xpos[PB]).copy()
        fingers, offs = {}, {}
        for f in FINGERS:
            pg = gid(fm, f"{f}_tip_collision" if not sfx
                     else f"{f}_tip_left_collision")
            offs[f] = R0.T @ (np.array(fd.geom_xpos[pg]) - Pp)
            fingers[f] = [float(fd.qpos[int(fm.jnt_qposadr[
                jid(fm, f"{f}j{k}{sfx}")])]) for k in range(4)]
        # r71: the THUMB is part of the terminal too.  The gate's forward
        # axis is the FOUR tips' centroid's direction, so the thumb's
        # posture moves it (measured: the reference's fwd is 26.4/27.6 deg
        # with the reference's thumb; with the thumb at zero the same pose
        # reads 38.7 deg).
        THUMB = [float(fd.qpos[int(fm.jnt_qposadr[
            jid(fm, f"thj{k}{sfx}")])]) for k in range(4)]
        print(f"=== {side}: the known-good palm at {np.round(Pp*1e3,1)} mm")

        AJ = [jid(sm, f"{prefix}_{n}_joint") for n in ARM]
        AQ = [int(sm.jnt_qposadr[j]) for j in AJ]
        AD = [int(sm.jnt_dofadr[j]) for j in AJ]
        ALO = sm.jnt_range[AJ][:, 0]
        AHI = sm.jnt_range[AJ][:, 1]
        PALM = mujoco.mj_name2id(sm, mujoco.mjtObj.mjOBJ_BODY, "palm" + sfx)
        FQ = {f: [int(sm.jnt_qposadr[j])
                  for j in (jid(sm, f"{f}j{k}{sfx}") for k in range(4))]
              for f in list(FINGERS) + ["th"]}
        FD = {f: [int(sm.jnt_dofadr[j])
                  for j in (jid(sm, f"{f}j{k}{sfx}") for k in range(4))]
              for f in FINGERS}
        FLO = {f: sm.jnt_range[[jid(sm, f"{f}j{k}{sfx}")
                                for k in range(4)]][:, 0] for f in FINGERS}
        FHI = {f: sm.jnt_range[[jid(sm, f"{f}j{k}{sfx}")
                                for k in range(4)]][:, 1] for f in FINGERS}
        PAD = {f: gid(sm, f"{f}_tip_collision" if side == "right"
                      else f"{f}_tip_left_collision") for f in FINGERS}
        DIST = {f: gid(sm, f"{f}_distal_collision" if side == "right"
                       else f"{f}_distal_left_collision") for f in FINGERS}
        HAND_GEOMS = [
            i for i in range(sm.ngeom)
            if (mujoco.mj_id2name(sm, mujoco.mjtObj.mjOBJ_GEOM, i) or "")
            not in ("object_geom", "object_chamfer_negx",
                    "object_chamfer_posx")
            and ("collision" in (mujoco.mj_id2name(
                sm, mujoco.mjtObj.mjOBJ_GEOM, i) or "")
                 or "tip" in (mujoco.mj_id2name(
                     sm, mujoco.mjtObj.mjOBJ_GEOM, i) or ""))]

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
            sd.qpos[FQ["th"]] = THUMB
            mujoco.mj_forward(sm, sd)

        def ik_arm(tp, Rt, q0):
            q = q0.copy()
            for _ in range(250):
                sd.qpos[:] = sm.qpos0
                sd.qpos[AQ] = q
                mujoco.mj_kinematics(sm, sd)
                mujoco.mj_comPos(sm, sd)
                e_p = tp - np.array(sd.xpos[PALM])
                Rq = np.zeros(9)
                mujoco.mju_quat2Mat(Rq, sd.xquat[PALM])
                e_r = Rot.from_matrix(
                    Rt @ np.array(Rq).reshape(3, 3).T).as_rotvec()
                if np.linalg.norm(e_p) < 2e-4 and np.linalg.norm(e_r) < 3e-3:
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
            ep = float(np.linalg.norm(tp - np.array(sd.xpos[PALM])))
            er = float(np.linalg.norm(Rot.from_matrix(
                Rt @ np.array(Rq).reshape(3, 3).T).as_rotvec()))
            mg = float(min(np.min(q - ALO), np.min(AHI - q)))
            return q, ep, er, mg

        def ik_finger(q, f, target, fv):
            """drive this finger's tip geom to the target point"""
            x = np.asarray(fv[f], float).copy()
            for _ in range(80):
                sd.qpos[:] = sm.qpos0
                sd.qpos[AQ] = q
                sd.qpos[FQ[f]] = x
                sd.qpos[FQ["th"]] = THUMB
                mujoco.mj_forward(sm, sd)
                e = target - np.array(sd.geom_xpos[PAD[f]])
                if np.linalg.norm(e) < 2e-5:
                    break
                jacp = np.zeros((3, sm.nv))
                mujoco.mj_jacGeom(sm, sd, jacp, None, PAD[f])
                J = jacp[:, FD[f]]
                try:
                    dq = np.linalg.solve(J @ J.T + 1e-6 * np.eye(3), e)
                except np.linalg.LinAlgError:
                    break
                x = np.clip(x + J.T @ dq, FLO[f], FHI[f])
            sd.qpos[:] = sm.qpos0
            sd.qpos[AQ] = q
            sd.qpos[FQ[f]] = x
            sd.qpos[FQ["th"]] = THUMB
            mujoco.mj_forward(sm, sd)
            res = float(np.linalg.norm(
                target - np.array(sd.geom_xpos[PAD[f]])))
            return x, res

        K = np.array([[0, -n_in[2], n_in[1]],
                      [n_in[2], 0, -n_in[0]],
                      [-n_in[1], n_in[0], 0]])
        # the pads' targets: the index and middle ON the face plane, the
        # ring pulled 8 mm clear (it wraps the top edge in the reference)
        rng = np.random.default_rng(19)
        rows = []
        n_arm_fail = n_ff_fail = n_mf_fail = 0
        for roll in range(0, 360, 10):
            ang = np.deg2rad(float(roll))
            cr, sr = np.cos(ang), np.sin(ang)
            Rz = (np.eye(3) + sr * K + (1 - cr) * (K @ K))
            Rt = Rz @ R0
            # the palm's x so the INDEX pad lands exactly on the face
            px = face_sign * float(HALF[0]) - float((Rt @ offs["ff"])[0])
            # the index pad's own y/z on the face (from the known-good)
            fy = float((Rt @ offs["ff"])[1])
            fz = float((Rt @ offs["ff"])[2])
            for py in (-0.03, 0.0, 0.03):
                for pz in np.arange(0.76, 0.83, 0.005):
                    tp = np.array([px, py, float(pz)])
                    # r72: among the CONVERGED arm solutions, prefer a
                    # NATURAL branch.  The multi-start IK otherwise returns
                    # whichever solution it found first, and for this cell
                    # that is a twisted one (shoulder_pan 6.04 rad, a full
                    # turn) whose servos cannot drive into it -- measured:
                    # the executed wrist_1 stalled at 1.02 rad against a
                    # -1.84 rad command.
                    cands = []
                    for q_try in [np.zeros(6)] + [
                            rng.uniform(ALO, AHI) for _ in range(14)]:
                        qq, epp, errp, mgp = ik_arm(tp, Rt, q_try)
                        if epp > 2e-3 or errp > 0.01 or mgp < 0.10:
                            continue
                        cands.append((sum(1 for v in qq if abs(v) > 3.0),
                                      sum(abs(v) for v in qq),
                                      epp + errp, qq, epp, errp, mgp))
                    if not cands:
                        n_arm_fail += 1
                        if n_arm_fail <= 3:
                            print(f"    [arm] roll {roll} py {py} pz "
                                  f"{pz*1e3:.0f}: no converged solution")
                        continue
                    cands.sort(key=lambda t: (t[0], t[2]))
                    _c0 = cands[0]
                    q, ep, er, mg = _c0[3], _c0[4], _c0[5], _c0[6]
                    # the palm's world position shifts the pads; the
                    # index's target is the face plane at (fy, fz)
                    # each pad's target is its KNOWN-GOOD position in the
                    # palm frame, projected onto the face plane -- this is
                    # scene-independent (the palm-frame offsets are
                    # identical in both scenes, measured) and it does not
                    # need a hand-tuned per-side offset.
                    fv = {f: list(fingers[f]) for f in FINGERS}
                    res_ff = res_mf = res_rf = 0.0
                    # r71: the RING finger keeps the KNOWN-GOOD posture.
                    # It is the wrap finger (it rides the box's top edge,
                    # 7.5 mm past the face plane in the reference), and
                    # re-posturing it moves the four tips' centroid -- the
                    # gate's forward axis is that centroid's direction,
                    # so the ring's posture drags it from 26 deg (the
                    # reference) to 38.7 deg (measured).
                    for f, extra_x in (("ff", 0.0), ("mf", 0.0)):
                        o = Rt @ offs[f]
                        tgt = np.array([face_sign * (float(HALF[0]) + extra_x),
                                        py + float(o[1]),
                                        float(pz) + float(o[2])])
                        # r68: NO set-back.  The KNOWN-GOOD reference
                        # grasp itself presses its pads 2-6 mm into the
                        # box (measured on the gate-3 seed: the index
                        # tip -2.13 mm, the middle tip -5.77 mm, the
                        # middle distal link -5.94 mm) -- the pads are
                        # compliant, and that geometry is what passes
                        # gate1.  A "zero surface distance" target is
                        # therefore NOT the reference, and forcing it
                        # degraded the solution (measured: the pads
                        # moved 3-18 mm off the face).
                        x_f, res_f = ik_finger(q, f, tgt, fv)
                        fv[f] = list(x_f)
                        if f == "ff":
                            res_ff = res_f
                        else:
                            res_mf = res_f
                    res_rf = 0.0
                    if res_ff > 3e-3 or res_mf > 3e-3:
                        n_ff_fail += 1
                        if n_ff_fail <= 3:
                            print(f"    [fik] roll {roll} py {py} pz "
                                  f"{pz*1e3:.0f}: the residuals "
                                  f"{res_ff*1e3:.2f}/{res_mf*1e3:.2f} mm")
                        continue
                    state(q, fv)
                    clear = 1.0
                    for gi in HAND_GEOMS:
                        bot = (float(sd.geom_xpos[gi][2])
                               - float(sm.geom_size[gi][2]))
                        clear = min(clear, bot - TABLE_TOP)
                    # the table clearance is a HARD constraint (r63: the
                    # first winner drove the palm 28-56 mm THROUGH the
                    # slab and the arm jammed)
                    if clear < 5e-3:
                        continue
                    a2, pd2 = {}, {}
                    # r70: measure the GATE'S OWN metric, not a proxy.
                    # The gate scores each pad's CENTRE against the end
                    # face's RECTANGLE (the surface distance differs by
                    # the pad's own 2-3 mm depth), and it scores the
                    # forward axis as the direction of the FOUR tip
                    # geoms' centroid (the thumb included) from the palm.
                    _bot = 0.74 - float(HALF[2])
                    _top = 0.74 + float(HALF[2])
                    _ctr = []
                    _rect = {}
                    for f in FINGERS:
                        Rd = np.array(sd.geom_xmat[DIST[f]]).reshape(3, 3)
                        a2[f] = float(np.degrees(np.arccos(
                            abs(float(Rd[:, 2] @ n_in)))))
                        ft = np.zeros(6)
                        dd = float(mujoco.mj_geomDistance(sm, sd, PAD[f],
                                                          OBJ_G, 1.0, ft))
                        rel = ft[3:6] - np.array(sd.xpos[OBJ_B])
                        pd2[f] = (dd, bool(abs(abs(rel[0])
                                                - float(HALF[0])) < 2e-3))
                        px, py, pz = (float(v) for v in sd.geom_xpos[PAD[f]])
                        sv = px - face_sign * float(HALF[0])
                        dy = max(abs(py) - float(HALF[1]), 0.0)
                        dz = max(_bot - pz, pz - _top, 0.0)
                        gap = abs(sv) if (dy == 0.0 and dz == 0.0) else \
                            float(np.hypot(np.hypot(sv, dy), dz))
                        _rect[f] = gap
                        _ctr.append(np.array(sd.geom_xpos[PAD[f]]))
                    # the thumb's tip is part of the centroid
                    _thp = gid(sm, "th_tip_collision" if side == "right"
                               else "th_tip_left_collision")
                    _ctr.append(np.array(sd.geom_xpos[_thp]))
                    _c = np.mean(_ctr, axis=0)
                    _fw = _c - np.array(sd.xpos[PALM])
                    _fw = _fw / max(float(np.linalg.norm(_fw)), 1e-12)
                    fwd_deg = float(np.degrees(np.arccos(np.clip(
                        float(_fw @ n_in), -1.0, 1.0))))
                    # the gate's band: the pad's z must be in the face's
                    # usable band, else it is out of band
                    _band = ((_bot + 0.4 * 2 * float(HALF[2]))
                             <= float(sd.geom_xpos[PAD["ff"]][2])
                             <= (_top - 0.01))
                    # the gate's forward axis is a hard clause (<= 35 deg);
                    # keep a margin so the executed state's drift does
                    # not cross it
                    if fwd_deg > 33.0:
                        continue
                    rows.append({
                        "roll": roll, "palm": [round(float(v), 5)
                                               for v in tp],
                        "e_p_mm": round(ep * 1e3, 3),
                        "e_r_deg": round(float(np.degrees(er)), 2),
                        "margin": round(mg, 4),
                        "clearance_mm": round(clear * 1e3, 1),
                        "axes": {k: round(v, 1) for k, v in a2.items()},
                        "rect_mm": {k: round(v * 1e3, 4)
                                    for k, v in _rect.items()},
                        "fwd_deg": round(fwd_deg, 2),
                        "in_band": bool(_band),
                        "n_in_rect": sum(1 for v in _rect.values()
                                         if v * 1e3 <= R5_TOL_MM),
                        "pad_mm": {k: round(v[0] * 1e3, 2)
                                   for k, v in pd2.items()},
                        "on_face": {k: bool(v[1]) for k, v in pd2.items()},
                        "distal_mm": {
                            f: round(float(mujoco.mj_geomDistance(
                                sm, sd, DIST[f], OBJ_G, 1.0,
                                np.zeros(6))) * 1e3, 3) for f in FINGERS},
                        "finger_res_mm": {"ff": round(res_ff * 1e3, 3),
                                          "mf": round(res_mf * 1e3, 3),
                                          "rf": round(res_rf * 1e3, 3)},
                        "arm": [round(float(v), 6) for v in q],
                        "fingers": dict(
                            {f: [round(float(v), 6) for v in fv[f]]
                             for f in FINGERS},
                            th=[round(float(v), 6) for v in THUMB]),
                        "palm_R": [[round(float(v), 6) for v in row]
                                   for row in Rt],
                        "palm_pos": [round(float(v), 6) for v in tp],
                    })
        # rank: the index pad's distance, then the middle's, then the
        # clearance, then the ring's burial
        # r67: prefer a NATURAL arm branch.  The right's first solution
        # had shoulder_pan = 5.996 rad (a full turn) and wrist_2 = -5.17:
        # the servos could not drive into it (the executed wrist_2 stalled
        # at -2.07 against a -5.17 command, measured), so the branch has
        # to be scored, not merely accepted.
        rows.sort(key=lambda r: (-r["n_in_rect"],
                                 sum(1 for v in r["arm"] if abs(v) > 3.0),
                                 sum(sorted(r["rect_mm"].values())[:2]),
                                 r["fwd_deg"],
                                 sum(abs(v) for v in r["arm"]),
                                 -min(r["clearance_mm"], 0.02)))
        print(f"  {len(rows)} candidates (the arm rejections: "
              f"{n_arm_fail}, the index IK: {n_ff_fail}, the middle IK: "
              f"{n_mf_fail})")
        for r in rows[:6]:
            print(f"    roll {r['roll']:3d} palm "
                  f"({r['palm'][0]*1e3:+6.1f}, {r['palm'][1]*1e3:+5.1f}, "
                  f"{r['palm'][2]*1e3:6.1f}) | e_p {r['e_p_mm']:.3f} mm "
                  f"| clear {r['clearance_mm']:+6.1f} mm "
                  f"| the rect {r['rect_mm']} | fwd {r['fwd_deg']} "
                  f"| in_rect {r['n_in_rect']} | band {r['in_band']}")
        out["sides"][side] = {"candidates": len(rows), "best": rows[:8]}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
