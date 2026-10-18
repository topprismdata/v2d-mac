"""r15 step 1: curl-last feasibility probe (Main's ticket, run before any
pipeline change).

Question: with BOTH arms frozen at the registered path's natural end
state (the r12.8 position-only DLS convergence point — the 15.9/1.6 mm
pose), does a finger curl set exist per side such that
  (a) >= 2 pads sit in the end-face rectangle at gate1's frozen
      tolerance (R5 1.1637e-4 m) with touch z in the frozen mid band,
  (b) the touching fingers' curl tip-path aligns with the face inward
      normal at cos >= 0.5 (the W6c placement-solve physics criterion:
      N_max = forcerange/arm x cos(theta)),
  (c) no finger geom dives into the slab deeper than 1 cm (the "不离桌
      过深" bound)?
A solution set becomes the new NATIVE seed (arm = path end, zero jump);
no solution -> the measured palm-pose gap goes back to Main for a
ruling.  W6c's cos-theta scanner method, domain = the UR-flange hand.

Run:  python3 tools/w6b/curl_last_probe.py
Out:  runs/w6b/r15_curl_probe.json + the printed solution table.
"""
import itertools
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
os.environ["W6B_SCENE"] = "ur5e"

import numpy as np
import mujoco
from scipy.spatial.transform import Rotation

from retarget import retarget_dual as rd

COS_MIN = 0.5
DEPTH_BOUND_M = 0.01          # max penetration below the slab top
CURL_GRID = (0.0, 0.15, 0.3, 0.45, 0.6, 0.8, 1.0, 1.15)
FINGERS = ("ff", "mf", "rf", "th")


def solve_arm_to_path_end(model, data, side: str, path: np.ndarray):
    """The production position-only DLS; returns the path-end arm q."""
    suffix = "" if side == "right" else "_left"
    palm = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "palm" + suffix)
    key = "r" if side == "right" else "l"
    jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                              f"ur5e_{key}_{n}_joint")
            for n in ("shoulder_pan", "shoulder_lift", "elbow",
                      "wrist_1", "wrist_2", "wrist_3")]
    qadr = [model.jnt_qposadr[j] for j in jids]
    vjad = [model.jnt_dofadr[j] for j in jids]
    lo = model.jnt_range[jids][:, 0]
    hi = model.jnt_range[jids][:, 1]
    q = np.zeros(6)
    for tgt in path:
        for _ in range(60):
            data.qpos[qadr] = q
            mujoco.mj_kinematics(model, data)
            mujoco.mj_comPos(model, data)
            e = tgt - data.xpos[palm]
            if np.linalg.norm(e) < 1e-3:
                break
            jacp = np.zeros((3, model.nv))
            mujoco.mj_jacBody(model, data, jacp, None, palm)
            J = jacp[:, vjad]
            q = np.clip(q + J.T @ np.linalg.solve(
                J @ J.T + 1e-4 * np.eye(3), e), lo, hi)
    data.qpos[qadr] = q
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)
    return q, float(np.linalg.norm(path[-1] - data.xpos[palm]))


def finger_probe(model, data, side: str):
    """Sweep per-finger seed scalings at the frozen arm; return the
    solution table."""
    suffix = "" if side == "right" else "_left"
    spec = rd.SideSpec.build(side)
    spec.bind(model, data)
    obj_body, box_geom, _, _ = rd.obj_ids(model)
    half = model.geom_size[box_geom].copy()
    face_x, half_y = float(half[0]), float(half[1])
    side_sign = spec.face_sign
    box_c = data.xpos[obj_body].copy()
    box_bot = float(box_c[2] - half[2])
    box_top = float(box_c[2] + half[2])
    tol = rd.R5_TOUCH_TOL_M

    f_jids = {f: [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                                    f"{f}a{k}{suffix}") for k in range(4)]
              for f in FINGERS}
    f_qadr = {f: [model.jnt_qposadr[j] for j in f_jids[f]] for f in FINGERS}
    lo16 = np.array([model.jnt_range[j][0] for j in
                     [j for f in FINGERS for j in f_jids[f]]])
    hi16 = np.array([model.jnt_range[j][1] for j in
                     [j for f in FINGERS for j in f_jids[f]]])
    seed = np.clip(np.asarray(rd.UR5E_TERMINAL_FINGERS[side], float),
                   lo16, hi16)

    # tip pad geoms per finger (gate1's pad set) + all collision geoms
    tip_g = {}
    for f in FINGERS:
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM,
                                f"{f}_tip{suffix}_collision")
        tip_g[f] = gid
    all_collision = {f: [] for f in FINGERS}
    for g in range(model.ngeom):
        n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        if n.endswith("_collision"):
            for f in FINGERS:
                if n.startswith(f + "_"):
                    all_collision[f].append(g)

    arm_qadr = [model.jnt_qposadr[mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT,
        f"ur5e_{'r' if side == 'right' else 'l'}_{n}_joint")]
        for n in ("shoulder_pan", "shoulder_lift", "elbow",
                  "wrist_1", "wrist_2", "wrist_3")]
    arm_q = data.qpos[arm_qadr].copy()

    def evaluate(curl16):
        for f in FINGERS:
            data.qpos[f_qadr[f]] = curl16[
                [FINGERS.index(f) * 4 + k for k in range(4)]]
        mujoco.mj_forward(model, data)
        pads = {}
        for f in FINGERS:
            g = tip_g[f]
            if g < 0:
                continue
            px, py, pz = (float(v) for v in data.geom_xpos[g])
            sv = px - side_sign * face_x
            dy = max(abs(py) - half_y, 0.0)
            dz = max(box_bot - pz, pz - box_top, 0.0)
            gap = abs(sv) if (dy == 0.0 and dz == 0.0) else float(
                np.hypot(np.hypot(sv, dy), dz))
            in_band = (box_bot + 0.4 * 2 * half[2]) <= pz <= (box_top - 0.01)
            pads[f] = {"gap_m": gap, "in_band": bool(in_band), "z": pz}
        # cos theta per finger: tip-path tangent under further closing
        # vs the face inward normal
        cos_t = {}
        for f in FINGERS:
            g = tip_g[f]
            if g < 0:
                continue
            base = data.geom_xpos[g].copy()
            step = seed[[FINGERS.index(f) * 4 + k for k in range(4)]] * 0.05
            for k in range(4):
                data.qpos[f_qadr[f][k]] = np.clip(
                    curl16[FINGERS.index(f) * 4 + k] + step[k],
                    lo16[FINGERS.index(f) * 4 + k],
                    hi16[FINGERS.index(f) * 4 + k])
            mujoco.mj_forward(model, data)
            d = data.geom_xpos[g] - base
            nrm = float(np.linalg.norm(d))
            cos_t[f] = (abs(float((d / nrm) @ np.array(
                [-side_sign, 0.0, 0.0]))) if nrm > 1e-9 else 0.0)
            # restore this finger
            for k in range(4):
                data.qpos[f_qadr[f][k]] = curl16[
                    FINGERS.index(f) * 4 + k]
        depth = {}
        for f in FINGERS:
            zs = [float(data.geom_xpos[g][2]) for g in all_collision[f]]
            depth[f] = (0.70 - min(zs)) if zs else 0.0
        return pads, cos_t, depth

    solutions = []
    best = {"rect_m": np.inf}
    grid = list(itertools.product(CURL_GRID, repeat=4))
    for combo in grid:
        curl16 = np.array([combo[fi] * seed[fi * 4 + k]
                           for fi in range(4) for k in range(4)])
        curl16 = np.clip(curl16, lo16, hi16)
        pads, cos_t, depth = evaluate(curl16)
        in_rect = [f for f in FINGERS
                   if pads[f]["in_band"]
                   and pads[f]["gap_m"] <= tol
                   and cos_t[f] >= COS_MIN]
        deep = [f for f in FINGERS if depth[f] > DEPTH_BOUND_M]
        rect = min(pads[f]["gap_m"] for f in FINGERS)
        rec = {"curl": {f: round(float(combo[fi]), 2)
                        for fi, f in enumerate(FINGERS)},
               "pads": {f: {"gap_mm": round(pads[f]["gap_m"] * 1000, 3),
                            "in_band": pads[f]["in_band"],
                            "cos": round(cos_t[f], 3),
                            "depth_mm": round(depth[f] * 1000, 1)}
                        for f in FINGERS},
               "n_in_rect": len(in_rect), "deep": deep}
        if rect < best["rect_m"]:
            best = {"rect_m": rect, "rec": rec}
        if len(in_rect) >= 2 and not deep:
            solutions.append(rec)
    return solutions, best


def main() -> int:
    print("=== r15 curl-last feasibility probe ===")
    paths = rd_pths()
    model = mujoco.MjModel.from_xml_path(
        str(rd.REPO / "grounding/assets/allegro_floating_ur5e.xml"))
    data = mujoco.MjData(model)
    report = {"criteria": {
        "pads": ">=2 tip pads in the end-face rectangle at gate1's R5 "
                "tolerance, touch z in the frozen mid band",
        "cos": f">= {COS_MIN} curl tip-path vs face inward normal",
        "depth": f"no finger geom deeper than {DEPTH_BOUND_M*1000:.0f} mm "
                 "below the slab top",
    }, "per_side": {}}
    solved_all = True
    for side in ("right", "left"):
        q, err = solve_arm_to_path_end(model, data, side, paths[side])
        print(f"{side}: path-end arm solve residual {err*1000:.3f} mm")
        solutions, best = finger_probe(model, data, side)
        report["per_side"][side] = {
            "arm_q": [round(float(v), 4) for v in q],
            "n_grid": len(list(itertools.product(CURL_GRID, repeat=4))),
            "n_solutions": len(solutions),
            "solutions": solutions[:12],
            "best_any": best,
        }
        ok = len(solutions) > 0
        solved_all = solved_all and ok
        print(f"{side}: solutions={len(solutions)} "
              f"best_rect={best['rect_m']*1000:.3f} mm")
    report["solved"] = bool(solved_all)
    out = rd.REPO / "runs/w6b/r15_curl_probe.json"
    out.write_text(json.dumps(report, indent=1))
    print(f"wrote {out}")
    print("VERDICT:", "SOLVED — new native seed available"
          if solved_all else "NO SOLUTION — palm-pose gap reported")
    return 0


def rd_pths():
    """Registered paths (shared with the r14 search)."""
    sys.path.insert(0, str(REPO / "tools/w6b"))
    import base_pose_search as bps
    return bps.registered_paths()


if __name__ == "__main__":
    raise SystemExit(main())
