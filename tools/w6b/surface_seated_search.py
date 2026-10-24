"""r118: the SURFACE-seated terminal search -- the geometry fix.

THE ROOT CAUSE (measured in this campaign, r118).  The delivered
terminal's reference puts the PAD CENTRE on the end face plane:

    the right ff: the centre x = 60.00 mm (the face plane is 60.00), and
        the TRUE surface gap (MuJoCo's own mj_geomDistance) is -2.122 mm
    the right mf: the centre x = 60.00 mm, the true surface gap
        -5.776 mm

The pad is a 12 x 12 x 3 mm BOX, so a centre on the plane means the pad
is already 2.1 mm (ff) / 5.8 mm (mf) INSIDE the box.  That is why:

  * the command state's gate1 sees TWO pads in the rectangle (their
    centres are on the plane) while the EXECUTED state has only ONE pad
    in contact (the ff; the mf ends up 5.64 mm away) -- the box cannot
    host two pads that are each buried centimetres deep;
  * the measured normal force is small: the pads press from a geometry
    the box has already absorbed.

The delivered searcher (tools/w6b/palm_pose_refine.py) optimises the
pad CENTRE against the face plane.  It has no surface notion at all.

WHAT THIS TOOL DOES.  It re-searches the terminal with the SURFACE as
the criterion: the per-finger IK targets MuJoCo's own signed surface
distance (mj_geomDistance, negative = penetrating), so the pad's contact
surface sits on the face at the commanded preload instead of its centre.
The curl is BAKED INTO the IK (the flexion joints of the seed are scaled
before the IK runs), so a deeper curl -- the only knob in the r118
measurements that reached the lift's required force -- no longer walks
the tip off the face.

    runs/.venv-retarget/bin/python tools/w6b/surface_seated_search.py
"""
from __future__ import annotations

import json
import os
import pathlib

import mujoco
import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
SCENE = pathlib.Path(os.environ.get(
    "W6B_S118_SCENE",
    "grounding/assets/allegro_floating_ur5e_side_r37.xml"))
FLOAT_SCENE = pathlib.Path("grounding/assets/allegro_floating.xml")
SEED = pathlib.Path("runs/w6/gate3_qpos.json")
DELIVERED = pathlib.Path("runs/w6b/palm_pose_refine.json")
OUT = pathlib.Path(os.environ.get(
    "W6B_S118_OUT", "runs/w6b/surface_seated_candidates.json"))
ANCHOR_FILE = pathlib.Path(os.environ.get(
    "W6B_S118_ANCHOR", "runs/w6b/palm_pose_anchor.json"))

ARM = ("shoulder_pan", "shoulder_lift", "elbow",
       "wrist_1", "wrist_2", "wrist_3")
FINGERS = ("ff", "mf", "rf")
CURLS = [float(v) for v in os.environ.get(
    "W6B_S118_CURLS", "1.00,1.05,1.10,1.15,1.20").split(",")]
PRELOADS_MM = [float(v) for v in os.environ.get(
    "W6B_S118_PRELOADS_MM", "0.0,0.05,0.10").split(",")]
ROLLS = [int(v) for v in os.environ.get(
    "W6B_S118_ROLLS", "0,20,40,60,80,100,120,140,160,180,200,220,240,"
                      "260,280,300,320,340").split(",")]
ANCHOR_TOL = 0.60


def jid(m, n):
    i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
    assert i >= 0, n
    return i


def gid(m, n):
    i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)
    assert i >= 0, n
    return i


def delivered_arm(side: str) -> np.ndarray:
    ref = json.loads(DELIVERED.read_text())
    entry = ref["sides"][side]
    cand = entry["best"][0] if isinstance(entry.get("best"), list) \
        else entry["best"]
    return np.asarray(cand["arm"], float)


def main() -> int:
    sm = mujoco.MjModel.from_xml_path(str(SCENE))
    sd = mujoco.MjData(sm)
    fm = mujoco.MjModel.from_xml_path(str(FLOAT_SCENE))
    fd = mujoco.MjData(fm)

    fd.qpos[:] = np.asarray(json.loads(SEED.read_text())["qpos"], float)
    for _jn in range(fm.njnt):
        n2 = mujoco.mj_id2name(fm, mujoco.mjtObj.mjOBJ_JOINT, _jn)
        if n2 and n2.startswith("object"):
            qa = int(fm.jnt_qposadr[_jn])
            fd.qpos[qa:qa + 7] = [0, 0, 0.74, 1, 0, 0, 0]
            break
    mujoco.mj_forward(fm, fd)

    OBJ_G = gid(sm, "object_geom")
    OBJ_B = mujoco.mj_name2id(sm, mujoco.mjtObj.mjOBJ_BODY, "object")
    HALF = np.array(sm.geom_size[OBJ_G])
    out = {
        "scene": str(SCENE),
        "the criterion": (
            "the pad's CONTACT SURFACE (MuJoCo's mj_geomDistance signed "
            "distance to the box, negative = penetrating) is at the "
            "commanded preload; the pad's CENTRE is therefore offset from "
            "the face plane by the pad's own extent along the normal"),
        "sides": {},
    }

    anchor = json.loads(ANCHOR_FILE.read_text()) \
        if ANCHOR_FILE.is_file() else None

    for side, prefix, sfx, face_sign in (
            ("right", "ur5e_r", "", +1.0),
            ("left", "ur5e_l", "_left", -1.0)):
        n_in = np.array([-face_sign, 0.0, 0.0])
        PB = mujoco.mj_name2id(fm, mujoco.mjtObj.mjOBJ_BODY, "palm" + sfx)
        Rq = np.zeros(9)
        mujoco.mju_quat2Mat(Rq, fd.xquat[PB])
        R0 = np.array(Rq).reshape(3, 3).copy()
        offs, fingers = {}, {}
        for f in FINGERS:
            pg = gid(fm, f"{f}_tip_collision" if not sfx
                     else f"{f}_tip_left_collision")
            offs[f] = R0.T @ (np.array(fd.geom_xpos[pg]) - fd.xpos[PB])
            fingers[f] = [float(fd.qpos[int(fm.jnt_qposadr[
                jid(fm, f"{f}j{k}{sfx}")])]) for k in range(4)]
        THUMB = [float(fd.qpos[int(fm.jnt_qposadr[
            jid(fm, f"thj{k}{sfx}")])]) for k in range(4)]

        AJ = [jid(sm, f"{prefix}_{n}_joint") for n in ARM]
        AQ = [int(sm.jnt_qposadr[j]) for j in AJ]
        AD = [int(sm.jnt_dofadr[j]) for j in AJ]
        FQ = {f: [int(sm.jnt_qposadr[jid(sm, f"{f}j{k}{sfx}")])
                  for k in range(4)] for f in FINGERS}
        FQ["th"] = [int(sm.jnt_qposadr[jid(sm, f"thj{k}{sfx}")])
                    for k in range(4)]
        FD = {f: [int(sm.jnt_dofadr[jid(sm, f"{f}j{k}{sfx}")])
                  for k in range(4)] for f in FINGERS}
        FLO = {f: np.array([sm.jnt_range[jid(sm, f"{f}j{k}{sfx}"), 0]
                            for k in range(4)]) for f in FINGERS}
        FHI = {f: np.array([sm.jnt_range[jid(sm, f"{f}j{k}{sfx}"), 1]
                            for k in range(4)]) for f in FINGERS}
        PAD = {f: gid(sm, f"{f}_tip_collision" if not sfx
                      else f"{f}_tip_left_collision") for f in FINGERS}
        PALM = mujoco.mj_name2id(sm, mujoco.mjtObj.mjOBJ_BODY, "palm" + sfx)
        THP = gid(sm, "th_tip_collision" if side == "right"
                  else "th_tip_left_collision")
        _anchor = np.asarray(anchor[side], float) \
            if anchor and side in anchor else None
        _seed_arm = delivered_arm(side)

        def reset_obj():
            mujoco.mj_resetData(sm, sd)
            sd.qpos[:] = sm.qpos0
            for _jn in range(sm.njnt):
                n2 = mujoco.mj_id2name(sm, mujoco.mjtObj.mjOBJ_JOINT, _jn)
                if n2 and n2.startswith("object"):
                    qa = int(sm.jnt_qposadr[_jn])
                    sd.qpos[qa:qa + 7] = [0, 0, 0.74, 1, 0, 0, 0]
                    break

        def place(q, fv, f=None):
            reset_obj()
            sd.qpos[AQ] = q
            for ff in FINGERS:
                sd.qpos[FQ[ff]] = fv[ff]
            sd.qpos[FQ["th"]] = THUMB
            mujoco.mj_forward(sm, sd)

        def surf(f):
            """(the signed surface distance, the outward normal at the
            closest point)"""
            ft = np.zeros(6)
            d = float(mujoco.mj_geomDistance(sm, sd, PAD[f], OBJ_G, 0.5, ft))
            v = np.asarray(ft[0:3]) - np.asarray(ft[3:6])
            nv = float(np.linalg.norm(v))
            n_hat = v / nv if nv > 1e-9 else n_in.copy()
            return d, n_hat

        def ik_arm(tp, q0):
            q = np.asarray(q0, float).copy()
            for _ in range(200):
                reset_obj()
                sd.qpos[AQ] = q
                mujoco.mj_forward(sm, sd)
                e = tp - np.array(sd.xpos[PALM])
                if np.linalg.norm(e) < 1e-5:
                    break
                jr = np.zeros((3, sm.nv))
                jp = np.zeros((3, sm.nv))
                mujoco.mj_jacBody(sm, sd, jp, jr, PALM)
                J = jp[:, AD]
                try:
                    dq = np.linalg.solve(J @ J.T + 1e-6 * np.eye(3), e)
                except np.linalg.LinAlgError:
                    break
                q = np.clip(q + J.T @ dq, sm.jnt_range[AJ, 0],
                            sm.jnt_range[AJ, 1])
            reset_obj()
            sd.qpos[AQ] = q
            mujoco.mj_forward(sm, sd)
            return q, float(np.linalg.norm(tp - np.array(sd.xpos[PALM])))

        def ik_finger(q, f, target_sv, fv):
            """drive the pad's SURFACE to the signed gap target_sv"""
            x = np.asarray(fv[f], float).copy()
            res = float("nan")
            for _ in range(90):
                place(q, fv, f)
                sd.qpos[FQ[f]] = x
                mujoco.mj_forward(sm, sd)
                d, n_hat = surf(f)
                res = abs(d - target_sv)
                if res < 3e-5:
                    break
                # move the pad along the surface normal to close the gap
                e = n_hat * (d - target_sv)
                jacp = np.zeros((3, sm.nv))
                mujoco.mj_jacGeom(sm, sd, jacp, None, PAD[f])
                J = jacp[:, FD[f]]
                try:
                    dq = np.linalg.solve(J @ J.T + 1e-6 * np.eye(3), e)
                except np.linalg.LinAlgError:
                    break
                x = np.clip(x + J.T @ dq, FLO[f], FHI[f])
            place(q, fv, f)
            sd.qpos[FQ[f]] = x
            mujoco.mj_forward(sm, sd)
            d, _ = surf(f)
            return x, abs(d - target_sv)

        rows = []
        rng = np.random.default_rng(19)
        for curl in CURLS:
            seed_f = {}
            for f in FINGERS:
                v = list(fingers[f])
                for j in (1, 2, 3):
                    v[j] *= curl
                seed_f[f] = v
            for preload_mm in PRELOADS_MM:
                target_sv = -preload_mm * 1e-3
                for roll in ROLLS:
                    ang = np.deg2rad(float(roll))
                    cr, sr = np.cos(ang), np.sin(ang)
                    K = np.array([[0, -1.0, 0], [1.0, 0, 0], [0, 0, 0]])
                    Rz = (np.eye(3) + sr * K + (1 - cr) * (K @ K))
                    Rt = Rz @ R0
                    for py in (-0.03, 0.0, 0.03):
                        for pz in np.arange(0.76, 0.83, 0.01):
                            px = (face_sign * (float(HALF[0])
                                               - preload_mm * 1e-3)
                                  - float((Rt @ offs["ff"])[0]))
                            tp = np.array([px, py, float(pz)])
                            arm_seeds = []
                            if _anchor is not None:
                                arm_seeds.append(_anchor)
                            arm_seeds.append(_seed_arm)
                            for _ in range(3):
                                arm_seeds.append(_seed_arm + rng.normal(
                                    0.0, 0.20, 6))
                            got = []
                            for q_try in arm_seeds:
                                q, ep = ik_arm(tp, q_try)
                                if ep > 2e-3:
                                    continue
                                if _anchor is not None:
                                    delta = float(np.max(np.abs(np.angle(
                                        np.exp(1j * (q - _anchor))))))
                                    if delta > ANCHOR_TOL:
                                        continue
                                got.append(q)
                                if len(got) >= 2:
                                    break
                            for q in got:
                                fv = {f: list(seed_f[f]) for f in FINGERS}
                                res_ff = res_mf = 0.0
                                for f in ("ff", "mf"):
                                    x_f, res_f = ik_finger(q, f,
                                                           target_sv, fv)
                                    fv[f] = list(x_f)
                                    if f == "ff":
                                        res_ff = res_f
                                    else:
                                        res_mf = res_f
                                if res_ff > 3e-3 or res_mf > 3e-3:
                                    continue
                                place(q, fv)
                                gaps = {f: surf(f)[0] for f in FINGERS}
                                _bot = 0.74 - float(HALF[2])
                                _top = 0.74 + float(HALF[2])
                                _ctr = []
                                _rect = {}
                                for f in FINGERS:
                                    px, py, pz = (float(v) for v in
                                                  sd.geom_xpos[PAD[f]])
                                    sv = px - face_sign * float(HALF[0])
                                    dy = max(abs(py) - float(HALF[1]), 0.0)
                                    dz = max(_bot - pz, pz - _top, 0.0)
                                    gap = abs(sv) if (dy == 0.0 and dz == 0.0) \
                                        else float(np.hypot(np.hypot(sv, dy),
                                                           dz))
                                    _rect[f] = gap
                                    _ctr.append(np.array(sd.geom_xpos[PAD[f]]))
                                _ctr.append(np.array(sd.geom_xpos[THP]))
                                _c = np.mean(_ctr, axis=0)
                                _fw = _c - np.array(sd.xpos[PALM])
                                _fw = _fw / max(float(np.linalg.norm(_fw)), 1e-12)
                                fwd_deg = float(np.degrees(np.arccos(
                                    np.clip(float(_fw @ n_in), -1.0, 1.0))))
                                if fwd_deg > 33.0:
                                    continue
                                rows.append({
                                    "the curl": curl,
                                    "the preload mm": preload_mm,
                                    "the roll": roll,
                                    "the palm": [round(float(v), 5)
                                                 for v in tp],
                                    "the palm_R": [[round(float(v), 6)
                                                    for v in r] for r in Rt],
                                    "the surface gaps mm": {
                                        f: round(gaps[f] * 1e3, 4)
                                        for f in FINGERS},
                                    "the rect mm": {f: round(v * 1e3, 4)
                                                    for f, v in _rect.items()},
                                    "the fwd deg": round(fwd_deg, 2),
                                    "the arm": [round(float(v), 6)
                                                for v in q],
                                    # the pipeline reads the reference's
                                    # "fingers" dict with the keys ff, mf,
                                    # rf, th -- so the thumb goes under
                                    # "th" (the pipeline's default for a
                                    # missing key is all zeros)
                                    "the fingers": {
                                        f: [round(float(v), 6) for v in fv[f]]
                                        for f in FINGERS},
                                    "the thumb": [round(float(v), 6)
                                                  for v in THUMB],
                                    "the ik residuals mm": {
                                        "ff": round(res_ff * 1e3, 4),
                                        "mf": round(res_mf * 1e3, 4)},
                                })
        rows.sort(key=lambda r: (sum(sorted(r["the rect mm"].values())[:2]),
                                 r["the fwd deg"]))
        print(f"=== {side}: {len(rows)} surface-seated candidates")
        for r in rows[:10]:
            print(f"    the curl {r['the curl']} / the preload "
                  f"{r['the preload mm']} / the roll {r['the roll']}: "
                  f"the surface gaps {r['the surface gaps mm']} | "
                  f"the rect {r['the rect mm']} | the fwd "
                  f"{r['the fwd deg']} deg")
        out["sides"][side] = {"candidates": len(rows), "best": rows[:24]}
        # write after EVERY side: a long search must not lose its work
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(out, indent=1))
        if not rows:
            print("    NO candidate: the surface-seated IK does not converge")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    print(f"\n  written: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
