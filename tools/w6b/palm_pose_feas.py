#!/usr/bin/env python3
"""Can the SIDE cell's arm reach the KNOWN-GOOD palm pose?

THE r64 FINDING.  The known-good floating grasp (the gate-3 seed) is NOT
a three-flat-pad grasp: measured on that seed,
  * the index and middle pads sit EXACTLY on the box's end face
    (x = 60.0 mm, the face plane), their axes 5.4 deg and 28 deg from
    the face normal -- the index pad presses FLAT (its 12 x 12 mm face);
  * the ring pad is 7.5 mm INSIDE the face plane and 20 mm ABOVE the
    box's top -- it wraps the top edge, it does not press on the face;
  * the thumb is 115 mm clear of the box (it is not a grasping finger
    here).
So the reference grasp is a TWO-pad flat grasp plus a wrap, and that is
what the gate asks for (>= 2 pads in the rectangle).

Because the pads' offsets in the PALM frame are fixed by the finger
joints, the palm's world ROTATION is pinned: rolling it about the face
normal is the only freedom, and the roll moves the index and middle pads
off a common face plane.  So the side cell must reach the KNOWN-GOOD
palm orientation, at a position whose height clears the table (the side
cell's table top is z = 0.70; the floating scene has NO table, which is
why the floating pose may sit lower without a collision).

THIS SCRIPT sweeps the palm's position (the y offset along the face and
the z height) with the palm's rotation FIXED at the known-good one, and
for each candidate:
  * solves the side cell's arm IK for that palm pose,
  * MEASURES the table clearance (the hand's lowest geom vs the slab),
  * MEASURES each pad's signed distance to the box and whether its
    closest point is on the end face,
  * MEASURES each finger's axis against the face normal.

usage: palm_pose_feas.py [scene.xml]
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
OUT = Path("runs/w6b/palm_pose_feas.json")

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
    out = {"scene": str(SCENE), "table_top_m": TABLE_TOP, "sides": {}}

    for side, prefix, sfx, face_sign in (
            ("right", "ur5e_r", "", +1.0),
            ("left", "ur5e_l", "_left", -1.0)):
        n_in = np.array([-face_sign, 0.0, 0.0])
        PB = mujoco.mj_name2id(fm, mujoco.mjtObj.mjOBJ_BODY, "palm" + sfx)
        Rq = np.zeros(9)
        mujoco.mju_quat2Mat(Rq, fd.xquat[PB])
        R0 = np.array(Rq).reshape(3, 3).copy()
        Pp = np.array(fd.xpos[PB]).copy()
        fingers = {}
        for f in FINGERS:
            fingers[f] = [float(fd.qpos[int(fm.jnt_qposadr[
                jid(fm, f"{f}j{k}{sfx}")])]) for k in range(4)]
        print(f"=== {side}: the known-good palm at {np.round(Pp*1e3,1)} mm")

        AJ = [jid(sm, f"{prefix}_{n}_joint") for n in ARM]
        AQ = [int(sm.jnt_qposadr[j]) for j in AJ]
        AD = [int(sm.jnt_dofadr[j]) for j in AJ]
        ALO = sm.jnt_range[AJ][:, 0]
        AHI = sm.jnt_range[AJ][:, 1]
        PALM = mujoco.mj_name2id(sm, mujoco.mjtObj.mjOBJ_BODY, "palm" + sfx)
        FQ = {f: [int(sm.jnt_qposadr[j])
                  for j in (jid(sm, f"{f}j{k}{sfx}") for k in range(4))]
              for f in FINGERS}
        PAD = {f: gid(sm, f"{f}_tip_collision" if side == "right"
                      else f"{f}_tip_left_collision") for f in FINGERS}
        DIST = {f: gid(sm, f"{f}_distal_collision" if side == "right"
                       else f"{f}_distal_left_collision") for f in FINGERS}
        HAND_GEOMS = [
            i for i in range(sm.ngeom)
            if (mujoco.mj_id2name(sm, mujoco.mjtObj.mjOBJ_GEOM, i) or "") not in
            ("object_geom", "object_chamfer_negx", "object_chamfer_posx")
            and ("collision" in (mujoco.mj_id2name(
                sm, mujoco.mjtObj.mjOBJ_GEOM, i) or "")
                 or "tip" in (mujoco.mj_id2name(
                     sm, mujoco.mjtObj.mjOBJ_GEOM, i) or ""))]

        def state(q):
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
                sd.qpos[FQ[ff]] = fingers[ff]
            mujoco.mj_forward(sm, sd)

        def ik_arm(tp, Rt, q0):
            # r65: the rotation error MUST be the proper rotation vector.
            # The 0.5*(R[2,1]-R[1,2], ...) form is sin(theta)*axis, so it
            # VANISHES at a 180 deg error -- the IK then "converges" at a
            # completely flipped orientation (measured: |R_achieved -
            # R_target| = 1.59 with the reported |e_r| = 0.00 deg).
            from scipy.spatial.transform import Rotation as _Rot
            q = q0.copy()
            for _ in range(250):
                sd.qpos[:] = sm.qpos0
                sd.qpos[AQ] = q
                mujoco.mj_kinematics(sm, sd)
                mujoco.mj_comPos(sm, sd)
                e_p = tp - np.array(sd.xpos[PALM])
                Rq = np.zeros(9)
                mujoco.mju_quat2Mat(Rq, sd.xquat[PALM])
                e_r = _Rot.from_matrix(
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
            er = float(np.linalg.norm(_Rot.from_matrix(
                Rt @ np.array(Rq).reshape(3, 3).T).as_rotvec()))
            mg = float(min(np.min(q - ALO), np.min(AHI - q)))
            return q, ep, er, mg

        rng = np.random.default_rng(31)
        rows = []
        offs = {}
        for f in FINGERS:
            pg = gid(fm, f"{f}_tip_collision" if not sfx
                     else f"{f}_tip_left_collision")
            offs[f] = R0.T @ (np.array(fd.geom_xpos[pg]) - Pp)
        K = np.array([[0, -n_in[2], n_in[1]],
                      [n_in[2], 0, -n_in[0]],
                      [-n_in[1], n_in[0], 0]])
        for roll in range(0, 360, 30):
            ang = np.deg2rad(float(roll))
            cr, sr = np.cos(ang), np.sin(ang)
            Rz = (np.eye(3) + sr * K + (1 - cr) * (K @ K))
            Rt = Rz @ R0
            px = None
            for f in ("ff", "mf"):
                wx = (Rt @ offs[f])[0]
                cand = face_sign * float(HALF[0]) - wx
                px = cand if px is None else 0.5 * (px + cand)
            for py in (-0.06, -0.03, 0.0, 0.03, 0.06):
                for pz in np.arange(0.74, 0.94, 0.01):
                    tp = np.array([px, py, float(pz)])
                    bq = None
                    for q_try in [np.zeros(6)] + [
                            rng.uniform(ALO, AHI) for _ in range(3)]:
                        qq, epp, errp, mgp = ik_arm(tp, Rt, q_try)
                        if bq is None or (epp + errp) < bq[1]:
                            bq = (qq, epp + errp, epp, errp, mgp)
                    q, ep, er, mg = bq[0], bq[2], bq[3], bq[4]
                    if ep > 3e-3 or er > 0.02 or mg < 0.05:
                        continue
                    state(q)
                    clear = 1.0
                    for gi in HAND_GEOMS:
                        hz = float(sm.geom_size[gi][2])
                        bot = float(sd.geom_xpos[gi][2]) - hz
                        clear = min(clear, bot - TABLE_TOP)
                    a2, pd2 = {}, {}
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
                    n_end = sum(1 for v in pd2.values() if v[1])
                    rows.append({
                        "roll": roll, "palm_x": round(px, 4),
                        "palm_y": py, "palm_z": round(float(pz), 4),
                        "e_p_mm": round(ep * 1e3, 3),
                        "e_r_deg": round(float(np.degrees(er)), 2),
                        "margin": round(mg, 4),
                        "clearance_mm": round(clear * 1e3, 1),
                        "axes": {k: round(v, 1) for k, v in a2.items()},
                        "pad_mm": {k: round(v[0] * 1e3, 2)
                                   for k, v in pd2.items()},
                        "on_face": {k: bool(v[1]) for k, v in pd2.items()},
                        "n_on_face": n_end,
                        "arm": [round(float(v), 6) for v in q],
                    })
        rows.sort(key=lambda r: (-r["n_on_face"],
                                 -min(r["clearance_mm"], 0.05),
                                 max(r["axes"].values()),
                                 max(abs(v) for v in r["pad_mm"].values())))
        print(f"  {len(rows)} reachable candidates")
        for r in rows[:8]:
            print(f"    roll {r['roll']:3d} palm "
                  f"({r['palm_x']*1e3:+6.1f}, {r['palm_y']*1e3:+5.1f}, "
                  f"{r['palm_z']*1e3:6.1f}) mm | e_p {r['e_p_mm']:.3f} mm "
                  f"| clear {r['clearance_mm']:+6.1f} mm "
                  f"| axes {r['axes']} | pads {r['pad_mm']} "
                  f"| on the face {r['n_on_face']}/3")
        out["sides"][side] = {
            "roll_candidates": len(rows),
            "best": rows[:12],
        }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
