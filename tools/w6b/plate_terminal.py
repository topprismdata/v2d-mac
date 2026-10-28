"""r123 (grant B): wire a feasible PALM-PLATE pose into the right's
terminal reference and run the pipeline.

The v10 review's authorization B makes the palm plate a legitimate
grasp contact surface (an independent design branch; the original hand
configuration stays the immutable baseline).  The plate search
(runs/w6b/palm_plate_search.json) found 24/3456 feasible poses on the
side cell: the plate tangent to the end face, the fingertips 9-158 mm
clear of the box, the table 89 mm clear.  The plate's normal force is
carried by the ARM's servo, not the 0.04-0.18 N.m finger servos.

THIS TOOL bakes one feasible pose (chosen by index into the feasible
list) into the RIGHT side of a copy of the delivered terminal
reference: the palm pose + rotation from the search row, the finger
joints set to the search row's curl (the same base + curl*0.35*(hi-base)
transform the search used).  The LEFT side keeps a baked thin-pad curl
(its winning ladder step from r122: curl 1.25 -> 1.391 N, gate6 PASS).

usage:
    runs/.venv-retarget/bin/python tools/w6b/plate_terminal.py IDX [SCENE]
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as _Rot

REPO = pathlib.Path(__file__).resolve().parents[2]
PIPE = REPO / "runs/.venv-retarget/bin/python"
SCRIPT = REPO / "retarget/retarget_dual.py"
REFERENCE = REPO / "runs/w6b/palm_pose_refine.json"
SEARCH = REPO / "runs/w6b/palm_plate_search.json"
WORK = REPO / "runs/w6b/plate_refs"
REPORT = REPO / "runs/w6b/report.json"

SCENE = sys.argv[2] if len(sys.argv) > 2 else \
    "grounding/assets/allegro_floating_ur5e_side_r41_mixed.xml"
LEFT_CURL = 1.25

BASE = {
    "W6B_SCENE": "ur5e",
    "W6B_SCENE_XML": SCENE,
    "W6B_R23_FLYOVER_MM": "250",
    "W6B_R22_MARGIN_MM": "8",
    "W6B_R75_SIDE_APPROACH": "1",
    "W6B_R63_TARGET": "1",
    "W6B_R18_NATIVE_TARGET": "1",
    "W6B_R50_CURL": "1.25",
    "W6B_R50_PRELOAD_MM": "0.05",
    "W6B_R39_TGC": "0",
    "W6B_R36_GSTATIC": "1",
    "W6B_R36_I": "0",
    "W6B_R39_HOLD": "60",
    "W6B_R110_CURL_SCALE": "1.0",
    "W6B_R109_SQUEEZE_MM": "0",
}


def curl_fingers(m, curl):
    """the search's own curl->joint transform, per 4-joint group"""
    out = {}
    for f in ("ff", "mf", "rf", "th"):
        v = []
        for k in range(4):
            j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{f}j{k}")
            lo, hi = m.jnt_range[j]
            base = 0.5 * (lo + hi)
            v.append(base + curl * 0.35 * (hi - base))
        out[f] = v
    return out
def build_ref(idx: int) -> pathlib.Path:
    row = json.loads(SEARCH.read_text())["feasible"][idx]
    m = mujoco.MjModel.from_xml_path(str(REPO / SCENE))
    ref = json.loads(REFERENCE.read_text())
    r = ref["sides"]["right"]
    cand = r["best"][0] if isinstance(r.get("best"), list) else r["best"]
    # the r63 target reads palm_pos / palm_R (the pipeline's key names);
    # 'the palm' / 'the palm_R' are the ANALYSIS outputs' key names
    # r124: the search's pose puts the plate CENTRE at z = 0.7877, above
    # the box top (0.78), so the first run contacted the box's TOP EDGE
    # (174.8 N, on_end_face False).  Recentre: the plate's centre offset
    # in the palm frame is measured by FK at the row's curl, then the
    # palm is placed so the plate's centre sits at the end face's centre
    # (x = standoff off the face plane, z = 0.74).
    _m = m
    _d = mujoco.MjData(_m)
    _d.qpos[:] = _m.qpos0
    _cf = curl_fingers(_m, float(row["curl"]))
    for _f in ("ff", "mf", "rf", "th"):
        for _k in range(4):
            _j = mujoco.mj_name2id(_m, mujoco.mjtObj.mjOBJ_JOINT,
                                   f"{_f}j{_k}")
            _d.qpos[int(_m.jnt_qposadr[_j])] = _cf[_f][_k]
    mujoco.mj_forward(_m, _d)
    _palm_b = mujoco.mj_name2id(_m, mujoco.mjtObj.mjOBJ_BODY, "palm")
    _palm_p = np.asarray(_d.xpos[_palm_b])
    _palm_R = np.asarray(_d.xmat[_palm_b]).reshape(3, 3)
    _plate_g = mujoco.mj_name2id(_m, mujoco.mjtObj.mjOBJ_GEOM,
                                 "palm_collision")
    _plate_c = np.asarray(_d.geom_xpos[_plate_g]) - _palm_p
    _plate_half = float(_m.geom_size[_plate_g][0])
    _R = np.asarray(row["R"], float)
    _c_w = _R @ (_palm_R.T @ _plate_c)     # the plate centre in the ROW frame
    _box_half = np.asarray(_m.geom_size[
        mujoco.mj_name2id(_m, mujoco.mjtObj.mjOBJ_GEOM, "object_geom")])
    _face_x = float(_box_half[0])          # the box centre x = 0
    # the plate is TALLER than the face (the plate is 95-113 mm across,
    # the face is 80 mm), so a plate centred at the face's centre dips
    # BELOW the table top (0.70) and smashes into the table (measured:
    # 448.9 N against the table while 13.3 mm off the box).  Raise the
    # centre so the plate's BOTTOM EDGE sits 1 mm above the table; the
    # plate then covers the face's full 80 mm height.
    _TABLE_TOP = 0.70
    _FACE_CENTRE_Z = 0.74   # the box centre z: the face spans 0.70-0.78
    _plate_halfz = float(sum(abs(_R[2, k]) * float(_m.geom_size[_plate_g][k])
                             for k in range(3)))
    _plate_z = _TABLE_TOP + _plate_halfz + 0.001
    _standoff = float(os.environ.get("W6B_PLATE_STANDOFF_MM", "0.1")) * 1e-3
    _palm_pos = [
        _face_x + _plate_half + _standoff - _c_w[0],
        0.0194 - _c_w[1],
        _plate_z - _c_w[2],
    ]
    cand["palm_pos"] = [float(v) for v in _palm_pos]
    cand["palm_R"] = [[float(v) for v in rr] for rr in row["R"]]
    cand["fingers"] = curl_fingers(m, float(row["curl"]))
    # r125: the reference's arm IS the terminal IK's start seed (the
    # pipeline's arm_start).  The stale delivered arm (solved for the OLD
 #   ceiling mount) seeds a WOUND solution at the new base -- measured:
    # wrist_3 commanded 4.49 rad while the servo executed -1.69 (a 6.19
    # rad = 2*pi winding), and the whole execution tracked 0.5 m off.
    # Solve the arm HERE, multi-restart, position+orientation, and
    # unwrap every joint into its [lo, hi] limit band before seeding.
    _d2 = mujoco.MjData(_m)
    _AJ = [mujoco.mj_name2id(_m, mujoco.mjtObj.mjOBJ_JOINT,
                              f"ur5e_r_{n}_joint") for n in
           ("shoulder_pan", "shoulder_lift", "elbow", "wrist_1",
            "wrist_2", "wrist_3")]
    _AQ = np.array([int(_m.jnt_qposadr[j]) for j in _AJ])
    _AD = np.array([int(_m.jnt_dofadr[j]) for j in _AJ])
    _LO = np.array([_m.jnt_range[j][0] for j in _AJ])
    _HI = np.array([_m.jnt_range[j][1] for j in _AJ])
    _c_local = _palm_R.T @ _plate_c     # plate centre in the palm frame
    _sz = np.asarray(_m.geom_size[_plate_g], float)
    _cand_arm = np.asarray(cand.get("arm", np.zeros(6)), float)

    _TIP_B = [mujoco.mj_name2id(_m, mujoco.mjtObj.mjOBJ_BODY,
                                f"{f}_tip") for f in ("ff", "mf", "rf", "th")]
    _BOX_G = mujoco.mj_name2id(_m, mujoco.mjtObj.mjOBJ_GEOM, "object_geom")
    _BOX_H = np.asarray(_m.geom_size[_BOX_G], float)
    _BOX_C = np.asarray(_d.geom_xpos[_BOX_G], float).copy()
    _PLATE_IN_X = (_face_x + _plate_half + _standoff) - _plate_half
    _TBL_G = mujoco.mj_name2id(_m, mujoco.mjtObj.mjOBJ_GEOM, "table_slab")
    _TC133 = float(os.environ.get("W6B_PLATE_TABLE_CLEAR_MM", "15.0")) * 1e-3
    _BOTTOM_CLEAR = float(os.environ.get("W6B_PLATE_BOTTOM_MM", "15.0")) * 1e-3
    _HAND_CLEAR = float(os.environ.get("W6B_HAND_CLEAR_MM", "5.0")) * 1e-3
    _ARM_B = {mujoco.mj_name2id(_m, mujoco.mjtObj.mjOBJ_BODY,
                                f"ur5e_r_{n}_link") for n in
              ("shoulder", "upper_arm", "forearm", "wrist_1", "wrist_2",
               "wrist_3")}
    _ARM_G = [g for g in range(_m.ngeom)
              if int(_m.geom_bodyid[g]) in _ARM_B]
    # r136: the HAND's own geoms must also clear the box -- only the
    # plate may touch it.  Measured with the r135 seed: the thumb base
    # pressed the box's chamfer at 9.3 N (0.28 mm in), which is a contact
    # the gate6 does not count but which loads the box and would falsify
    # a lift.
    _hr = mujoco.mj_name2id(_m, mujoco.mjtObj.mjOBJ_BODY, "hand_root")
    _HAND_B = {_hr}
    for _b136 in range(_m.nbody):
        _p136 = int(_m.body_parentid[_b136])
        while _p136 > 0:
            if _p136 == _hr:
                _HAND_B.add(_b136)
                break
            _p136 = int(_m.body_parentid[_p136])
    # r136b: only geoms that CAN collide with the box count.  Measured:
    # the wmount_* adapter hardware and every *_visual geom carry
    # contype=conaffinity=0, and gating on them (the wmount cap penetrates
    # the box's AABB by 17.2 mm) made all 36 rolls infeasible even though
    # the roll-0 solve was 50 mm clear of the table and 50 mm clear of the
    # box with its fingertips.
    _HAND_G = [g for g in range(_m.ngeom)
               if int(_m.geom_bodyid[g]) in _HAND_B
               and int(_m.geom_contype[g]) & int(_m.geom_contype[_BOX_G])
               and int(_m.geom_conaffinity[g]) & int(
                   _m.geom_conaffinity[_BOX_G])
               and (mujoco.mj_id2name(_m, mujoco.mjtObj.mjOBJ_GEOM, g)
                    or "") != "palm_collision"]

    def _fk(qq):
        _dd = mujoco.MjData(_m)
        _dd.qpos[:] = _m.qpos0
        _dd.qpos[_AQ] = np.clip(qq, _LO, _HI)
        mujoco.mj_forward(_m, _dd)
        return (np.asarray(_dd.xpos[_palm_b], float).copy(),
                np.asarray(_dd.xmat[_palm_b], float).reshape(3, 3).copy())

    def _clear(qq):
        """(arm-link/table, fingertip/box, worst tip-behind-plate, margin)"""
        _dd = mujoco.MjData(_m)
        _dd.qpos[:] = _m.qpos0
        _dd.qpos[_AQ] = np.clip(qq, _LO, _HI)
        for _f in ("ff", "mf", "rf", "th"):
            for _k in range(4):
                _j = mujoco.mj_name2id(_m, mujoco.mjtObj.mjOBJ_JOINT,
                                       f"{_f}j{_k}")
                _dd.qpos[int(_m.jnt_qposadr[_j])] = _cf[_f][_k]
        mujoco.mj_forward(_m, _dd)
        _tc = min((mujoco.mj_geomDistance(_m, _dd, _g, _TBL_G, 0.05, None)
                   for _g in _ARM_G), default=0.05)
        _hb = min((mujoco.mj_geomDistance(_m, _dd, _g, _BOX_G, 0.05, None)
                   for _g in _HAND_G), default=0.05)
        _ht = min((mujoco.mj_geomDistance(_m, _dd, _g, _TBL_G, 0.05, None)
                   for _g in _HAND_G), default=0.05)
        _tips = np.asarray([_dd.xpos[b] for b in _TIP_B], float)
        _tb = min(float(np.linalg.norm(np.maximum(
            np.abs(_t - _BOX_C) - _BOX_H, 0.0))) for _t in _tips)
        _bh = min(float(_t[0]) - _PLATE_IN_X for _t in _tips)
        _mg = float(np.min(np.minimum(np.clip(qq, _LO, _HI) - _LO,
                                      _HI - np.clip(qq, _LO, _HI))))
        return float(_tc), _tb, _bh, _mg, float(_hb), float(_ht)

    def _unwrap(qq):
        qq = np.asarray(qq, float).copy()
        for _i in range(6):
            while qq[_i] > _HI[_i] and qq[_i] - 2 * np.pi >= _LO[_i]:
                qq[_i] -= 2 * np.pi
            while qq[_i] < _LO[_i] and qq[_i] + 2 * np.pi <= _HI[_i]:
                qq[_i] += 2 * np.pi
        return np.clip(qq, _LO, _HI)

    def _solve_arm(_tp, _tR, _n_restart=240, _roll_free=True, _seed=None):
        """multi-restart damped 6D solve; the roll about the palm x is a
        GAUGE in stage 1 (projected out) and FIXED in stage 2."""
        _rng = np.random.default_rng(7)
        _bb = None
        for _t in range(_n_restart):
            mujoco.mj_resetData(_m, _d2)
            _d2.qpos[:] = _m.qpos0
            if _seed is not None and _t == 0:
                _d2.qpos[_AQ] = np.clip(_seed, _LO, _HI)
            else:
                for _i in range(6):
                    _d2.qpos[_AQ[_i]] = _rng.uniform(_LO[_i], _HI[_i])
            _d2.qpos[_AQ] = np.clip(_d2.qpos[_AQ], _LO, _HI)
            for _f in ("ff", "mf", "rf", "th"):
                for _k in range(4):
                    _j = mujoco.mj_name2id(_m, mujoco.mjtObj.mjOBJ_JOINT,
                                           f"{_f}j{_k}")
                    _d2.qpos[int(_m.jnt_qposadr[_j])] = _cf[_f][_k]
            for _it in range(600):
                mujoco.mj_forward(_m, _d2)
                _ep = _tp - np.asarray(_d2.xpos[_palm_b])
                _Rp = np.asarray(_d2.xmat[_palm_b]).reshape(3, 3)
                # r131: the rotation error is the ROTVEC of the relative
                # rotation.  The old skew form 0.5*(Rp^T R - R^T Rp) is
                # EXACTLY ZERO at a 180 deg error (the r65 artifact that
                # already invalidated palm_pose_reach.json): with it the
                # baker "converged" with the plate 180 deg off and the
                # pipeline had to swing the arm 3.78 rad to reach the
                # ref's own palm_R.
                _er_l = _Rot.from_matrix(_Rp.T @ _tR).as_rotvec()
                if _roll_free:
                    _er = _Rp @ np.array([0.0, float(_er_l[1]),
                                          float(_er_l[2])])
                else:
                    _er = _Rp @ _er_l
                if np.linalg.norm(_ep) < 1e-5 and np.linalg.norm(_er) < 1e-4:
                    break
                _jp = np.zeros((3, _m.nv))
                _jr = np.zeros((3, _m.nv))
                mujoco.mj_jacBody(_m, _d2, _jp, _jr, _palm_b)
                _J = np.vstack([_jp[:, _AD], _jr[:, _AD]])
                # r131b: damped least squares with a trust region.  The
                # undamped solve steps 1/lambda times the residual near a
                # singularity, threw the arm to the joint limits and left
                # the corrected-metric solve 618 mm short.
                _e6 = np.concatenate([_ep, _er])
                _JJ = _J.T @ _J + 1e-4 * np.eye(6)
                _dq = np.linalg.solve(_JJ, _J.T @ _e6)
                _mx = float(np.abs(_dq).max())
                if _mx > 0.1:
                    _dq *= 0.1 / _mx
                _d2.qpos[_AQ] = np.clip(_d2.qpos[_AQ] + _dq, _LO, _HI)
            mujoco.mj_forward(_m, _d2)
            _ep = float(np.linalg.norm(_tp - np.asarray(_d2.xpos[_palm_b])))
            _Rp = np.asarray(_d2.xmat[_palm_b]).reshape(3, 3)
            _er_l = _Rot.from_matrix(_Rp.T @ _tR).as_rotvec()
            _er = float(np.linalg.norm(
                np.array([0.0, _er_l[1], _er_l[2]]) if _roll_free
                else _er_l))
            # r131d: a LIMIT MARGIN term (the search's own >0.3 rad
            # criterion).  Without it the cheapest gauge roll put wrist_3
            # at 5.925 rad against its limit and the backward approach
            # stalled (all 301 frames held the terminal).
            _marg = np.minimum(_d2.qpos[_AQ] - _LO, _HI - _d2.qpos[_AQ])
            _mp = float(np.sum(np.maximum(0.0, 0.3 - _marg)))
            # r132: the FINGERTIPS must clear the box AND stay behind the
            # plate's inner face.  The roll gauge has a continuum of
            # solutions and only some of them keep the fingers on the
            # plate's far side; the chosen one put all four tips 38-68 mm
            # in FRONT of a plate whose inner face was at 101 mm, so the
            # middle tip touched the box's top edge at 1.35 N and held the
            # whole hand 50 mm short of its command.
            _tp132 = np.asarray([_d2.xpos[b] for b in _TIP_B], float)
            _short = 0.0
            for _t132 in _tp132:
                _d132 = np.linalg.norm(np.maximum(
                    np.abs(_t132 - _BOX_C) - _BOX_H, 0.0))
                _short += float(max(0.0, 0.009 - _d132))
                _short += float(max(0.0, (_PLATE_IN_X + 0.005)
                                    - float(_t132[0])))
            # r133b: the ARM LINKS must clear the table.  Only the arm's
            # own bodies are checked -- the hand is AT the box by design.
            _tc133 = 0.0
            for _g133 in _ARM_G:
                _dd133 = mujoco.mj_geomDistance(_m, _d2, _g133, _TBL_G,
                                                _TC133, None)
                if _dd133 < _TC133:
                    _tc133 += (_TC133 - _dd133)
            # r133c/r134b: HARD-GATE ordering, CONVERGENCE FIRST.  With
            # only the geometric gate, a non-converged solve parked at the
            # joint limits (424 mm off target, 50 mm clear of everything)
            # was CHEAPER than a converged solve that penetrated the table
            # -- measured: all 36 rolls returned margin 0.00 at the limits.
            # A non-converged solve now costs 1e6 more than any converged
            # one, and the clearances rank only among converged solves.
            _conv = 0.0 if (_ep <= 2e-3 and _er <= 0.02) else 1e6
            _cost = (_conv + 1e3 * _tc133 + 1e2 * _short
                     + _ep + 0.05 * _er + 0.02 * _mp
                     + 1e-6 * float(np.linalg.norm(
                         _d2.qpos[_AQ] - _cand_arm)))
            if _bb is None or _cost < _bb[0]:
                _bb = (_cost, _ep, _er, _d2.qpos[_AQ].copy())
        return _bb

    # r134: ROLL SWEEP.  The roll about the plate's normal is a gauge for
    # the CONTACT (the plate is a centred box, invariant under a 180 deg
    # turn about that normal) but it is not free in practice: it sets the
    # plate's vertical extent, which side the fingertips sit on, and
    # whether the wrist links clear the table.  Measured before this sweep:
    # stage 1's cheapest roll made stage 2's target unreachable (the solve
    # stopped 163 mm short), and the pre-sweep roll buried the executed
    # forearm/wrist meshes 0.2-38.5 mm IN the table, where the arm was
    # held -- 3x the actuator force range changed nothing (-450/+450 N.m
    # saturated, the joint error stayed 0.25/0.45 rad).
    # Every 10 deg: re-derive the plate's centre for that roll (the bottom
    # edge 1 mm above the table, or the face's centre, whichever is higher
    # -- the face spans 0.70-0.78 so a 45 mm plate centred at 0.74 covers
    # 717-763 mm, fully on the face), solve the full 6D pose warm-started
    # from the previous roll, and keep the feasible solves.
    _R_row = np.asarray(row["R"], float)
    _prev = np.asarray(cand.get("arm", np.zeros(6)), float)
    _sweep = []
    for _th_deg in np.arange(0.0, 360.0, 10.0):
        _Rr = _R_row @ _Rot.from_rotvec(
            [float(np.deg2rad(_th_deg)), 0.0, 0.0]).as_matrix()
        _halfz = float(sum(abs(_Rr[2, k]) * _sz[k] for k in range(3)))
        # r135: the plate's BOTTOM CLEARANCE above the table is a
        # parameter, not 1 mm.  Measured with 1 mm: the plate grazes the
        # table on the way in, the contact jams wrist_1 (the constraint
        # torque -28.25 N.m against the actuator's +-28 limit, 362.7 N at
        # 1.35 mm of penetration) and the arm cannot finish its last
        # 0.47 rad, so the plate ends 51.8 mm off the face.  The face
        # spans 0.70-0.78; a 147 mm plate whose bottom is at 0.715 still
        # covers 65 of the face's 80 mm.
        _plz = max(_TABLE_TOP + _halfz + _BOTTOM_CLEAR, _FACE_CENTRE_Z)
        _cw = _Rr @ _c_local
        _tp = np.array([_face_x + _plate_half + _standoff - _cw[0],
                        0.0194 - _cw[1],
                        _plz - _cw[2]])
        _b = _solve_arm(_tp, _Rr, 24, False, _seed=_prev)
        _a = _unwrap(_b[3])
        _tc, _tb, _bh, _mg, _hb, _ht = _clear(_a)
        # r134c: the gate is the FINGERTIP/BOX clearance, not "the tips
        # behind the plate's plane".  Measured: every converged roll puts
        # the tips 62 mm in FRONT of the plate's inner face plane, but
        # laterally offset -- at roll 0 they are 36.8 mm clear of the box
        # while the plate's own face is 3 mm into it, so the plate still
        # touches first.  The behind-plane number is reported, not gated.
        _ok = (_b[1] <= 2e-3 and _b[2] <= 0.02 and _tc >= _TC133
               and _tb >= 0.009 and _mg >= 0.3
               and _hb >= _HAND_CLEAR and _ht >= _HAND_CLEAR)
        _sweep.append({
            "roll_deg": float(_th_deg),
            "pos_err_mm": round(_b[1] * 1e3, 4),
            "rot_err_rad": round(_b[2], 5),
            "plate_halfz_mm": round(_halfz * 1e3, 2),
            "plate_centre_z_mm": round(_plz * 1e3, 2),
            "arm_table_clear_mm": round(_tc * 1e3, 2),
            "tip_box_clear_mm": round(_tb * 1e3, 2),
            "tip_behind_plate_mm": round(_bh * 1e3, 2),
            "joint_margin_rad": round(_mg, 3),
            "hand_box_clear_mm": round(_hb * 1e3, 2),
            "hand_table_clear_mm": round(_ht * 1e3, 2),
            "feasible": bool(_ok),
            "arm": [round(float(v), 4) for v in _a],
            "palm_pos": [round(float(v), 5) for v in _tp],
        })
        _prev = _a
    _feas = [r for r in _sweep if r["feasible"]]
    if not _feas:
        raise SystemExit(
            "no feasible plate roll: " + json.dumps(
                [{k: r[k] for k in ("roll_deg", "pos_err_mm",
                                    "rot_err_rad",
                                    "arm_table_clear_mm",
                                    "tip_box_clear_mm",
                                    "tip_behind_plate_mm",
                                    "hand_box_clear_mm",
                                    "hand_table_clear_mm",
                                    "joint_margin_rad")} for r in _sweep],
                indent=1))
    # among the feasible rolls, keep the one with the worst-case clearance
    # as large as possible (the pose least dependent on tolerance)
    _pick = max(_feas, key=lambda r: (
        min(r["arm_table_clear_mm"], r["tip_box_clear_mm"],
            r["hand_box_clear_mm"], r["hand_table_clear_mm"]),
        r["joint_margin_rad"], -r["pos_err_mm"]))
    _arm = np.asarray(_pick["arm"], float)
    _tp2 = np.asarray(_pick["palm_pos"], float)
    _R_pick = _R_row @ _Rot.from_rotvec(
        [float(np.deg2rad(_pick["roll_deg"])), 0.0, 0.0]).as_matrix()
    _halfz2 = _pick["plate_halfz_mm"] * 1e-3
    _p_ach, _r_ach = _fk(_arm)
    cand["roll_sweep"] = _sweep
    cand["roll_pick_deg"] = _pick["roll_deg"]
    print(f"  [plate-roll-sweep] {len(_sweep)} rolls, {len(_feas)} feasible, "
          f"picked {_pick['roll_deg']:.0f} deg "
          f"(arm/table {_pick['arm_table_clear_mm']:+.2f} mm, hand/box "
          f"{_pick['hand_box_clear_mm']:+.2f} mm, tip/box "
          f"{_pick['tip_box_clear_mm']:+.2f} mm, tip behind the plate "
          f"{_pick['tip_behind_plate_mm']:+.2f} mm, margin "
          f"{_pick['joint_margin_rad']:.3f} rad, pos err "
          f"{_pick['pos_err_mm']:.4f} mm)")
    cand["arm"] = [float(v) for v in _arm]
    # the chosen solution's own violations, so the recipe is auditable
    _dd = mujoco.MjData(_m)
    _dd.qpos[:] = _m.qpos0
    _dd.qpos[_AQ] = np.clip(_arm, _LO, _HI)
    for _f in ("ff", "mf", "rf", "th"):
        for _k in range(4):
            _j = mujoco.mj_name2id(_m, mujoco.mjtObj.mjOBJ_JOINT,
                                   f"{_f}j{_k}")
            _dd.qpos[int(_m.jnt_qposadr[_j])] = _cf[_f][_k]
    mujoco.mj_forward(_m, _dd)
    _tc, _tb2, _bh2, _mg2, _hb, _ht = _clear(np.clip(_arm, _LO, _HI))
    _tips = np.asarray([_dd.xpos[b] for b in _TIP_B], float)
    _tclear = min(float(np.linalg.norm(np.maximum(
        np.abs(_t - _BOX_C) - _BOX_H, 0.0))) for _t in _tips)
    print(f"  [plate-seed-clear] arm-link/table {_tc*1e3:+.2f} mm "
          f"(floor {_TC133*1e3:.0f}) | hand/box {_hb*1e3:+.2f} mm and "
          f"hand/table {_ht*1e3:+.2f} mm (floor {_HAND_CLEAR*1e3:.0f}) | "
          f"fingertip/box {_tclear*1e3:+.2f} mm (floor 9.0) | tips behind "
          f"the plate's inner face: "
          f"{[round(float(t[0]-_PLATE_IN_X)*1e3,1) for t in _tips]} mm")
    print(f"  [plate-seed-fk] pos {np.round(_p_ach*1e3,2).tolist()} "
          f"target {np.round(_tp2*1e3,2).tolist()} "
          f"| pos err {np.linalg.norm(_p_ach-_tp2)*1e3:.4f} mm "
          f"| rot err (incl roll) "
          f"{float(np.linalg.norm(_Rot.from_matrix(_r_ach.T @ _R_pick).as_rotvec())):.5f} rad"
          f" | plate halfz {_halfz2*1e3:.1f} mm bottom "
          f"{(float(_p_ach[2]) + float((_r_ach @ _c_local)[2]) - _halfz2)*1e3:.1f} mm")
    # the ref's pose IS the seed's pose: the pipeline's IK then starts at
    # zero residual and no terminal reconfiguration is needed
    cand["palm_pos"] = [float(v) for v in _p_ach]
    cand["palm_R"] = [[float(v) for v in rr] for rr in _r_ach]
    print(f"  [plate-arm-seed] roll {_pick['roll_deg']:.0f} deg, pos "
          f"{_pick['pos_err_mm']:.4f} mm, rot {_pick['rot_err_rad']:.5f} rad "
          f"| arm {np.round(_arm, 3).tolist()}")
    # the left: its winning thin-pad curl (r122), same transform as the
    # thinpad scan
    l = ref["sides"]["left"]
    lcand = l["best"][0] if isinstance(l.get("best"), list) else l["best"]
    lf = lcand.get("fingers") or {}
    for f in ("ff", "mf", "rf", "th"):
        v = list(lf.get(f, [0.0] * 4))
        for j in (1, 2, 3):
            v[j] *= LEFT_CURL
        lf[f] = v
    lcand["fingers"] = lf
    WORK.mkdir(parents=True, exist_ok=True)
    p = WORK / f"plate_{idx}.json"
    p.write_text(json.dumps(ref, indent=1))
    return p, row


def run(ref_path: pathlib.Path) -> dict:
    env = dict(os.environ)
    env.update(BASE)
    env["W6B_R63_REFERENCE"] = str(ref_path)
    r = subprocess.run([str(PIPE), str(SCRIPT)], cwd=REPO, env=env,
                       capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(r.stdout[-1500:] + r.stderr[-1500:] + "\n")
        raise SystemExit("the pipeline failed")
    return json.loads(REPORT.read_text())


def main() -> int:
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    ref, row = build_ref(idx)
    rep = run(ref)
    out = {"scene": SCENE, "plate_row": row, "left_curl": LEFT_CURL}
    for s in ("right", "left"):
        ps = rep["per_side"][s]
        g6 = ps["gate6_executed_contact"]
        pl = ps["gate1_final_touch"]["executed_state_evidence"].get(
            "palm_plate_executed_state")
        out[s] = {
            "gate6": {k: g6[k] for k in
                      ("n_real_contacts", "normal_force_sum_N",
                       "lift_need_N", "pass")},
            "per_pad": g6["per_pad"],
            "palm_plate": pl,
            "box_motion_mm": rep.get("box_motion", {}).get(
                "final_displacement_mm"),
        }
        print(f"{s}: g6 F={g6['normal_force_sum_N']} "
              f"need={g6['lift_need_N']} real={g6['n_real_contacts']} "
              f"pass={g6['pass']} | plate={pl} | "
              f"box={rep.get('box_motion', {}).get('final_displacement_mm')}")
    print("red gates:", [k for k, v in rep["verdict_checks"].items() if not v])
    pathlib.Path("runs/w6b/plate_terminal.json").write_text(
        json.dumps(out, indent=1))
    ref.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
