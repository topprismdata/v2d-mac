"""W4 side-check - the R8 lip scan's distance was read from the wrong slot.

`runs/w4/lip_kin_scan.py` measures a fingertip-to-lip gap as ``ft[0]`` from the
``fromto`` buffer of ``mj_geomDistance``. That slot is NOT the distance: with a
verified two-box scene (100 mm apart) the call returns 0.100000 and fills
``fromto`` as ``[point_on_geom1(3), point_on_geom2(3)]``. So ``ft[0]`` is the
x-coordinate of the finger-side nearest point, and R8's criterion
``0.0 <= dist <= 1.16e-4`` ("a touch within the R5 tolerance") actually asked
for the finger's nearest point to have a world x-coordinate within 0.116 mm of
zero - an unrelated razor-thin condition, which is why every cell read 0.

This re-runs the same question with the RETURN VALUE as the distance and the
same strict criteria (nearest point inside the lip-face rectangle, pads only,
nothing buried), plus the same coarse-grid + local-refinement search the W4-B
pedestal scan uses, so "0" and "not reachable" can be told apart.

Scene generation goes through the committed parameterisation
``build_alip_overhang_m, lip_h_m)``; the emitted XML is read into memory and
deleted, and the model is compiled from the string. Nothing in the repository
is modified.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
import time

import mujoco
import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from grounding.assets import build_allegro_floating as baf  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parent / "lip_kin_recheck.json"
ASSETS = REPO / "grounding/assets"
TOL = baf.penetration_tolerance(ASSETS / "allegro_floating.xml")
WALL_BAND_M = 0.001
BURY_PENALTY_M = 0.5
DIST_SKIP_M = 0.05
N_STARTS = 300
N_ITERS = 400
POLISH_ITERS = 300
SIGMA0, SIGMA_MIN, POLISH_SIGMA0 = 0.35, 0.004, 0.012
SEED = 20261015

CELLS = ((0.004, 0.006), (0.008, 0.010), (0.012, 0.014))


def lip_scene(overhang: float, lip_h: float):
    """XML text for one lip setting, compiled in memory; no file left behind."""
    # PID-unique: build() writes this path in two passes, so a fixed name lets a
    # concurrent run (or a rerun) delete the file out from under the second pass
    tmp = ASSETS / f"_w4_recheck_tmp_{os.getpid()}.xml"
    try:
        baf.build(overhang, lip_h, tmp)
        text = tmp.read_text()
    finally:
        tmp.unlink(missing_ok=True)
    stl_dir = (ASSETS / "../../runs/upstream/menagerie/wonik_allegro/assets").resolve()
    assets = {p.name: p.read_bytes() for p in stl_dir.glob("*.stl")}
    model = mujoco.MjModel.from_xml_string(text, assets)
    return model, mujoco.MjData(model), text


class Rig:
    def __init__(self, model, data, overhang: float, lip_h: float) -> None:
        self.m, self.d = model, data
        self.obj_adr = int(model.jnt_qposadr[mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, "object_free")])
        self.box_z = baf.TABLE_TOP_Z_M + baf.PEDESTAL_H_M + baf.BOX_HALF_M[2]
        self.bx, self.by, _bz = baf.BOX_HALF_M
        self.lip_h = lip_h
        self.hook_z = baf.BOX_HALF_M[2] - lip_h      # box-frame z of the lip face
        self.x_lo, self.x_hi = self.bx, self.bx + overhang
        # objective() reads these, so they have to live on the rig, not just be
        # constructor arguments: without them the hook-face test below raises
        # NameError on the first pose
        self.lip_h = lip_h
        self.overhang = overhang
        self.lip_geoms = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n)
                          for n in baf.BOX_LIP_GEOM_NAMES]
        self.obj_geoms = self.lip_geoms + [mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, "object_geom")]
        self.pad_geoms, self.hand_geoms = [], []
        for pref in ("ff", "mf", "rf", "th"):
            for role in ("base", "proximal", "medial", "distal", "tip"):
                bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{pref}_{role}")
                if bid < 0:
                    continue
                for g in range(model.ngeom):
                    if int(model.geom_bodyid[g]) == bid and "visual" not in (
                            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""):
                        self.hand_geoms.append(g)
                        if role == "tip":
                            self.pad_geoms.append((f"{pref}_{role}", g))
        palm = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "palm")
        self.hand_geoms += [g for g in range(model.ngeom)
                            if int(model.geom_bodyid[g]) == palm and "visual" not in (
                                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "")]
        names = ("wrist_x", "wrist_y", "wrist_z", "wrist_rx", "wrist_ry", "wrist_rz")
        wj = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in names]
        fj = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in baf.FINGER_JOINTS]
        self.wrist_q = [int(model.jnt_qposadr[j]) for j in wj]
        self.finger_q = np.array([model.jnt_qposadr[j] for j in fj], dtype=int)
        adr = np.array(wj + fj, dtype=int)
        self.lo = model.jnt_range[adr, 0].astype(float).copy()
        self.hi = model.jnt_range[adr, 1].astype(float).copy()
        self.span = np.where(self.hi > self.lo, self.hi - self.lo, 1.0)
        self.ft = np.zeros(6)

    def set_pose(self, pose: np.ndarray) -> None:
        self.d.qpos[self.wrist_q] = pose[:6]
        self.d.qpos[self.finger_q] = pose[6:]
        self.d.qpos[self.obj_adr + 2] = self.box_z

    def objective(self, pose: np.ndarray) -> tuple[float, dict]:
        mujoco.mj_resetData(self.m, self.d)
        self.set_pose(pose)
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_forward(self.m, self.d)
        ft = self.ft
        buried = False
        for g in self.hand_geoms:
            for og in self.obj_geoms:
                if float(mujoco.mj_geomDistance(self.m, self.d, g, og, 0.05, ft)) < -1e-6:
                    buried = True
                    break
            if buried:
                break
        recs = []
        for name, g in self.pad_geoms:
            for lg in self.lip_geoms:
                dist = float(mujoco.mj_geomDistance(self.m, self.d, g, lg, 0.05, ft))
                if dist >= DIST_SKIP_M:
                    continue
                qx, qy, qz = (float(v) for v in ft[3:6])
                rel_z = qz - self.box_z
                # The hook face is the lip's UNDERSIDE at rel_z = hook_z. The
                # z-band must be the contact tolerance, NOT max(lip_h, ...): a band
                # as wide as the lip itself admits a pad resting on the lip's TOP
                # face (rel_z = bz), which presses DOWN and is the opposite
                # mechanism. That mistake reported clean hooks for all three cells
                # with the nearest point at rel_z = 0.04 (the box top), 64-71 mm
                # above the actual hook face. The face test is a strict rectangle
                # test, same standard as w35d's wall criterion.
                off = (max(0.0, abs(rel_z - self.hook_z) - WALL_BAND_M)
                       + max(0.0, self.x_lo - abs(qx) - WALL_BAND_M)
                       + max(0.0, abs(qx) - self.x_hi - WALL_BAND_M)
                       + max(0.0, abs(qy) - self.by - WALL_BAND_M))
                        recs.append({"pad": name, "lip_geom": mujoco.mj_id2name(
                    self.m, mujoco.mjtObj.mjOBJ_GEOM, lg), "dist_m": dist,
                    "gap_m": dist + off, "rel_on_box": [round(qx, 5), round(qy, 5),
                                                         round(rel_z, 5)]})
        best = min((r["gap_m"] for r in recs), default=float("inf"))
        j = best + (BURY_PENALTY_M if buried else 0.0)
        return float(j), {"best_gap_m": best, "buried": buried, "recs": recs}


def _local(rig, start, rng, iters, sigma0, sigma_min):
    best_j, _ = rig.objective(start)
    best = start.copy()
    cur, cur_j = start.copy(), best_j
    sigma = sigma0
    for _ in range(iters):
        cand = np.clip(cur + rng.normal(0.0, sigma, 22) * rig.span, rig.lo, rig.hi)
        j, _ = rig.objective(cand)
        if j <= cur_j:
            cur, cur_j = cand, j
        if j < best_j:
            best_j, best = j, cand.copy()
        sigma = max(sigma_min, sigma * 0.97)
    return best_j, best


def refine(rig, start, rng):
    j, p = _local(rig, start, rng, N_ITERS, SIGMA0, SIGMA_MIN)
    j2, p2 = _local(rig, p, rng, POLISH_ITERS, POLISH_SIGMA0, 0.0008)
    return (j2, p2) if j2 < j else (j, p)


def main() -> None:
    cells = []
    for oh, lh in CELLS:
        t0 = time.time()
        model, data, _text = lip_scene(oh, lh)
        rig = Rig(model, data, oh, lh)
        rng = np.random.default_rng(SEED + int(oh * 1e3) + int(lh * 1e3))
        best = (float("inf"), None)
        for _ in range(N_STARTS):
            j, p = refine(rig, rng.uniform(rig.lo, rig.hi), rng)
            if j < best[0]:
                best = (j, p)
        j, info = rig.objective(best[1])
        cells.append({
            "overhang_mm": oh * 1e3, "lip_h_mm": lh * 1e3,
            "best_J_m": j, "best_gap_m": info["best_gap_m"],
            "buried": info["buried"],
            "clean_hook": bool(j <= TOL and not info["buried"]),
            "best_pads": info["recs"],
            "pose": [float(v) for v in best[1]],
            "tol_m": TOL, "seconds": round(time.time() - t0, 1),
        })
        print(f"overhang={oh * 1e3:4.1f} lip_h={lh * 1e3:4.1f}  best gap="
              f"{info['best_gap_m'] * 1e3:8.3f} mm  clean={cells[-1]['clean_hook']}"
              f"  [{cells[-1]['seconds']}s]", flush=True)
    summary = {
        "question": "R8 asked whether a fingertip pad can reach a tray lip's "
                    "underside. Re-asked with mj_geomDistance's RETURN VALUE as "
                    "the gap (R8 read fromto[0], which is the finger-side point's "
                    "x-coordinate).",
        "criterion": "nearest point inside the lip-face rectangle, gap <= the R5 "
                     "tolerance, pads only, no hand geom buried",
        "tol_m": TOL, "cells": cells,
        "any_clean": any(c["clean_hook"] for c in cells),
    }
    OUT.write_text(json.dumps(summary, indent=2))
    print(f"any clean: {summary['any_clean']} -> {OUT}")


if __name__ == "__main__":
    main()
