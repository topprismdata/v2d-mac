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
    _tgt_p = np.asarray(_palm_pos, float)
    _tgt_R = np.asarray(row["R"], float)
    _rng = np.random.default_rng(7)
    _best = None
    for _t in range(240):
        mujoco.mj_resetData(_m, _d2)
        _d2.qpos[:] = _m.qpos0
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
            _ep = _tgt_p - np.asarray(_d2.xpos[_palm_b])
            _Rp = np.asarray(_d2.xmat[_palm_b]).reshape(3, 3)
            # r131: the rotation error is the ROTVEC of the relative
            # rotation.  The old skew-vector form, 0.5*(Rp^T R - R^T Rp),
            # is EXACTLY ZERO at a 180 deg error (the r65 artifact that
            # already invalidated palm_pose_reach.json): the solver
            # therefore "converged" with the right hand's plate rotated
            # 180 deg from the intended face orientation, and the
            # pipeline -- which uses the correct rotvec metric -- then
            # had to swing the arm 3.78 rad to reach it.
            _er = _Rot.from_matrix(_Rp.T @ _tgt_R).as_rotvec()
            if np.linalg.norm(_ep) < 1e-5 and np.linalg.norm(_er) < 1e-4:
                break
            _jp = np.zeros((3, _m.nv))
            _jr = np.zeros((3, _m.nv))
            mujoco.mj_jacBody(_m, _d2, _jp, _jr, _palm_b)
            _J = np.vstack([_jp[:, _AD], _jr[:, _AD]])
            # r131b: damped least squares with a trust region.  The
            # undamped solve steps 1/lambda times the residual near a
            # singularity, which threw the arm to the joint limits and
            # left the corrected-metric solve 618 mm short.
            _e6 = np.concatenate([_ep, _er])
            _JJ = _J.T @ _J + 1e-4 * np.eye(6)
            _dq = np.linalg.solve(_JJ, _J.T @ _e6)
            _mx = float(np.abs(_dq).max())
            if _mx > 0.1:
                _dq *= 0.1 / _mx
            _d2.qpos[_AQ] = np.clip(_d2.qpos[_AQ] + _dq, _LO, _HI)
        mujoco.mj_forward(_m, _d2)
        _ep = float(np.linalg.norm(
            _tgt_p - np.asarray(_d2.xpos[_palm_b])))
        _Rp = np.asarray(_d2.xmat[_palm_b]).reshape(3, 3)
        _er = float(np.linalg.norm(
            _Rot.from_matrix(_Rp.T @ _tgt_R).as_rotvec()))
        # r125c: ties at ~0 residual are broken by the DISTANCE to the
        # delivered arm -- the free tie-break once picked an elbow-DOWN
        # branch whose approach swept the box (the box was shoved 21 mm
        # and the plate ended 26 mm short)
        _qdel = np.asarray(cand.get("arm", np.zeros(6)), float)
        _cost = (_ep + 0.05 * _er
                 + 1e-6 * float(np.linalg.norm(_d2.qpos[_AQ] - _qdel)))
        if _best is None or _cost < _best[0]:
            _best = (_cost, _ep, _er, _d2.qpos[_AQ].copy())
    _arm = _best[3]
    # unwrap: shift by +-2*pi into the limit band where possible
    for _i in range(6):
        while _arm[_i] > _HI[_i] and _arm[_i] - 2 * np.pi >= _LO[_i]:
            _arm[_i] -= 2 * np.pi
        while _arm[_i] < _LO[_i] and _arm[_i] + 2 * np.pi <= _HI[_i]:
            _arm[_i] += 2 * np.pi
    cand["arm"] = [float(v) for v in np.clip(_arm, _LO, _HI)]
    print(f"  [plate-arm-seed] residual pos {_best[1]*1000:.3f} mm "
          f"rot {_best[2]:.5f} | arm {np.round(_arm, 3).tolist()}")
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
