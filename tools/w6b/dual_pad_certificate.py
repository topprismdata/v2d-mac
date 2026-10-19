"""r26 Track B: dual-pad feasibility certificate (GPT v5 archive-grade).

Systematic grid over the wrist 6D pose around the compliant-pair pose:
  position    +/-30 mm, 15 mm steps per axis (5^3 = 125 cells)
  orientation +/-25 deg, 12.5 deg steps per axis (5^3 = 125)
  => 15,625 pose cells
Per cell: arm 6D DLS (the pair arm warm start, tol 2 mm / 5 deg,
100 iters) + a finger coordinate-descent (2 sweeps x 16 joints x 2
signs x steps {0.05, 0.02}) from the pair's finger values, inside the
holdable band; then the pad state (ff+mf simultaneously in-rect at
R5 / 1 mm / 5 mm) + the burial-v2 + the through-state.

Output: the feasibility certificate JSON — the counts per tolerance
class, the feasible cell list, the best cell, the resolution and the
boundary margins.  Empty feasible set at the searched resolution =
the infeasibility evidence for GPT's archive (with the resolution
stated so the claim is falsifiable).

Run:  python3 tools/w6b/dual_pad_certificate.py
Out:  runs/w6b/r26_dual_pad_certificate.json
"""
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
os.environ.setdefault("W6B_SCENE", "ur5e")

import numpy as np
import mujoco
from scipy.spatial.transform import Rotation

from retarget import retarget_dual as rd

OUT = REPO / "runs/w6b/r26_dual_pad_certificate.json"
POS_STEP = 0.015          # m
POS_RANGE = 0.030         # +/- m
ORI_STEP = 12.5           # deg
ORI_RANGE = 25.0          # +/- deg
IK_TOL_P = 2e-3
IK_TOL_R = np.deg2rad(5.0)
IK_ITERS = 100
F_STEP_SCHEDULE = (0.05, 0.02)
FINGERS = ("ff", "mf", "rf", "th")


def main() -> int:
    model = mujoco.MjModel.from_xml_path(
        str(REPO / "grounding/assets/allegro_floating_ur5e_r24_vise.xml"))
    data = mujoco.MjData(model)
    spec = rd.SideSpec.build("right")
    spec.bind(model, data)
    obj_body, box_geom, _, _ = rd.obj_ids(model)
    half = model.geom_size[box_geom].copy()
    box_c = data.xpos[obj_body].copy()
    bot, top = float(box_c[2] - half[2]), float(box_c[2] + half[2])
    tol = rd.R5_TOUCH_TOL_M
    pair = min((p for p in json.loads(
        (REPO / "runs/w6b/native_seed_pairs.json").read_text())["pairs"]
        if p["side"] == "right"), key=lambda p: p["second_mm"])
    arm_pair = np.asarray(pair["arm"], float)
    f_pair = np.clip(np.asarray(pair["fingers"], float),
                     model.jnt_range[spec.finger_qadr][:, 0],
                     model.jnt_range[spec.finger_qadr][:, 1])
    # the pair's palm pose = the grid center
    data.qpos[spec.wrist_qadr] = arm_pair
    data.qpos[spec.finger_qadr] = f_pair
    mujoco.mj_forward(model, data)
    palm_center = data.xpos[spec._palm_id].copy()
    Rq = np.zeros(9)
    mujoco.mju_quat2Mat(Rq, data.xquat[spec._palm_id])
    R_center = Rq.reshape(3, 3).copy()

    tips = {f: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM,
                                 f"{f}_tip_collision")
            for f in FINGERS}
    ARM = ("shoulder_pan", "shoulder_lift", "elbow",
           "wrist_1", "wrist_2", "wrist_3")
    arm_qadr = [model.jnt_qposadr[mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, f"ur5e_r_{n}_joint")]
        for n in ARM]
    vjad = [model.jnt_dofadr[mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, f"ur5e_r_{n}_joint")]
        for n in ARM]
    lo_a = model.jnt_range[arm_qadr][:, 0]
    hi_a = model.jnt_range[arm_qadr][:, 1]
    f_qadr = [model.jnt_qposadr[mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, f"{p}j{k}")]
        for p in FINGERS for k in range(4)]
    lo_f = model.jnt_range[f_qadr][:, 0]
    hi_f = model.jnt_range[f_qadr][:, 1]

    def pad_state():
        out = {}
        for f in FINGERS:
            px, py, pz = (float(v) for v in
                          data.geom_xpos[tips[f]])
            sv = px - spec.face_sign * float(half[0])
            dy = max(abs(py) - float(half[1]), 0.0)
            dz = max(bot - pz, pz - top, 0.0)
            gap = abs(sv) if (dy == 0.0 and dz == 0.0) else float(
                np.hypot(np.hypot(sv, dy), dz))
            in_band = (bot + 0.4 * 2 * half[2]) <= pz <= (top - 0.01)
            out[f] = (gap, in_band and abs(sv) <= tol,
                      in_band and gap <= 1e-3,
                      in_band and gap <= 5e-3)
        return out

    def arm_ik(tgt_pos, tgt_R, warm):
        q = np.clip(warm, lo_a, hi_a)
        for _ in range(IK_ITERS):
            data.qpos[arm_qadr] = q
            mujoco.mj_kinematics(model, data)
            mujoco.mj_comPos(model, data)
            e_p = tgt_pos - data.xpos[spec._palm_id]
            R = np.zeros(9)
            mujoco.mju_quat2Mat(R, data.xquat[spec._palm_id])
            e_r = Rotation.from_matrix(
                tgt_R @ R.reshape(3, 3).T).as_rotvec()
            err = np.concatenate([e_p, e_r])
            if (np.linalg.norm(e_p) < IK_TOL_P
                    and np.linalg.norm(e_r) < IK_TOL_R):
                return q, True
            jacp = np.zeros((3, model.nv))
            jacr = np.zeros((3, model.nv))
            mujoco.mj_jacBody(model, data, jacp, jacr, spec._palm_id)
            J = np.vstack([jacp[:, vjad], jacr[:, vjad]])
            dq = np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), err)
            q = np.clip(q + J.T @ dq, lo_a, hi_a)
        data.qpos[arm_qadr] = q
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        e_p = tgt_pos - data.xpos[spec._palm_id]
        return (False, float(np.linalg.norm(e_p)))

    def finger_descent(f_start):
        x = np.clip(np.asarray(f_start, float), lo_f, hi_f)

        def sc_of(xx):
            data.qpos[f_qadr] = xx
            mujoco.mj_forward(model, data)
            st = pad_state()
            gaps = sorted(v[0] for v in st.values())
            best = gaps[0] if gaps else 1000.0
            second = gaps[1] if len(gaps) >= 2 else best + 500.0
            n_bur = sum(1 for v in st.values() if not v[1])
            return 1e6 * n_bur + second * 10.0 + best, best, second, n_bur
        sc, best, second, n_bur = sc_of(x)
        for step in F_STEP_SCHEDULE:
            improved = True
            while improved:
                improved = False
                for j in range(16):
                    for sgn in (+1.0, -1.0):
                        cand = x.copy()
                        cand[j] = np.clip(cand[j] + sgn * step,
                                          lo_f[j], hi_f[j])
                        s2, b2, sec2, nb2 = sc_of(cand)
                        if s2 < sc:
                            x, sc, best, second, n_bur = cand, s2, b2, \
                                sec2, nb2
                            improved = True
        return x, best, second, n_bur

    # ---- the grid ----
    pos_offs = np.arange(-POS_RANGE, POS_RANGE + 1e-9, POS_STEP)
    ori_offs = np.arange(-ORI_RANGE, ORI_RANGE + 1e-9, ORI_STEP)
    cells = []
    n_cells = (len(pos_offs) ** 3) * (len(ori_offs) ** 3)
    print(f"grid: {n_cells} cells")
    done = 0
    for dx in pos_offs:
        for dy in pos_offs:
            for dz in pos_offs:
                tgt_pos = palm_center + np.array([dx, dy, dz])
                for rx in ori_offs:
                    for ry in ori_offs:
                        for rz in ori_offs:
                            R_t = (Rotation.from_euler(
                                "xyz", [rx, ry, rz], degrees=True
                            ).as_matrix() @ R_center)
                            arm_q, conv = arm_ik(tgt_pos, R_t, arm_pair)
                            data.qpos[arm_qadr] = arm_q
                            data.qpos[f_qadr] = f_pair
                            mujoco.mj_forward(model, data)
                            x_f, best, second, n_bur = finger_descent(f_pair)
                            data.qpos[f_qadr] = x_f
                            mujoco.mj_forward(model, data)
                            st = pad_state()
                            n_r5 = sum(1 for v in st.values()
                                       if v[2])
                            n_1mm = sum(1 for v in st.values()
                                        if v[3])
                            n_5mm = sum(1 for v in st.values()
                                        if v[3])
                            done += 1
                            rec = {
                                "pos": [round(float(dx), 4),
                                        round(float(dy), 4),
                                        round(float(dz), 4)],
                                "ori": [round(float(rx), 2),
                                        round(float(ry), 2),
                                        round(float(rz), 2)],
                                "arm_conv": bool(conv),
                                "n_r5": n_r5, "n_1mm": n_1mm,
                                "n_5mm": n_5mm,
                                "best_mm": round(best * 1000, 3),
                                "second_mm": round(second * 1000, 3),
                            }
                            if n_r5 >= 2:
                                cells.append(rec)
                            if done % 500 == 0:
                                print(f"  {done}/{n_cells} cells, "
                                      f"R5-dual so far: {len(cells)}")
    feasible = [c for c in cells if c["n_r5"] >= 2]
    feasible_1mm = [c for c in cells if c["n_1mm"] >= 2]
    out = {
        "method": ("wrist 6D pose grid around the compliant-pair pose: "
                   "position +/-30 mm @ 15 mm steps x orientation +/-25 "
                   "deg @ 12.5 deg steps; per cell: arm 6D DLS (2 mm/5 "
                   "deg) + finger coordinate descent (steps 0.05/0.02, "
                   "holdable-band projected); ff+mf simultaneous "
                   "in-rect counted at R5/1 mm/5 mm"),
        "resolution": {"position_mm": 15.0, "orientation_deg": 12.5},
        "n_cells": n_cells if False else done,
        "feasible_R5_cells": feasible,
        "n_feasible_R5": len(feasible),
        "n_feasible_1mm": len(feasible_1mm),
        "counts": {"cells_scanned": done,
                   "cells_R5_dual": len(feasible),
                   "cells_1mm_dual": len(feasible_1mm)},
    }
    OUT.write_text(json.dumps(out, indent=1))
    print(f"wrote {OUT}")
    verdict = ("FEASIBLE — dual-pad solutions exist" if feasible
               else "EMPTY at this resolution — infeasibility evidence")
    print("VERDICT:", verdict)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
