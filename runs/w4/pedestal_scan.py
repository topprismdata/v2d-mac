"""W4-B - does a lower pedestal open a two-wall fingertip-pad grasp? (K1 scan)

The pedestal height had never been swept downward. At 60 mm the box top is at
0.840 m and the palm floor is 0.839 m, so the palm can never get below the box
top and the pads have to reach the side walls from below the rim. At 0 mm the box
top drops to 0.780 m - 59 mm below the palm floor - which is the configuration a
human is in when picking up a box off a desk: come down from above and wrap the
walls. The question is whether this hand has that path, and w35d already
answered "no" for the 60 mm case over ~19 M poses.

Method
------
* the scene is compiled IN MEMORY from the committed XML text with only the
  pedestal geom resized. The repository generator and the XML file are never
  touched and no variant is written to disk.
* the search is a two-stage optimise-then-count over the full 22-dim commanded
  pose (6 wrist + 16 finger), all inside the model's own joint ranges:
    stage 1  structured coarse grid;
    stage 2  randomised local refinement (shrinking radius) on the objective
             below, seeded from the coarse grid AND from random starts, so a
             narrow basin cannot hide behind a coarse grid.
* objective J(pose): the worst side's wall gap, where a pad's wall gap is the
  distance from that pad geom to the corresponding side-wall RECTANGLE (the
  pad-to-box gap plus how far the box-side nearest point is from being on that
  rectangle). Burial of ANY hand geom, or a buried pad, is penalised by +0.5 m
  so the search cannot "win" by interpenetrating.
* K1 (the kinematic gate): J <= the w35d/R5 tolerance, i.e. one pose with a
  fingertip pad touching the +x wall AND a (different) fingertip pad touching
  the -x wall, both nearest points inside their wall rectangles, no finger geom
  buried in the object.

Everything measured is written to runs/w4/pedestal_scan_kin.json.
"""
from __future__ import annotations

import json
import pathlib
import re
import sys
import time

import mujoco
import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from grounding.assets import build_allegro_floating as baf  # noqa: E402

MJCF = REPO / "grounding/assets/allegro_floating.xml"
OUT = pathlib.Path(__file__).resolve().parent / "pedestal_scan_kin.json"

PEDESTAL_HEIGHTS_M = (0.0, 0.020, 0.040, 0.060)
TABLE_TOP_M = 0.70
BURY_PENALTY_M = 0.5
DIST_SKIP_M = 0.05           # beyond this a pad cannot be a candidate

TOL = baf.penetration_tolerance(MJCF)      # w35d/R5 strict touch tolerance
WALL_BAND_M = 0.001                       # how close to the wall plane counts
PEN_GAP = 0.0                             # burial threshold, same as R8

# --- stage 1: structured coarse grid --------------------------------------
GW = (-0.10, -0.05, 0.0, 0.05, 0.10)       # wrist x
GY = (-0.06, -0.03, 0.0, 0.03, 0.06)       # wrist y
GZ = (-0.15, -0.12, -0.09, -0.06, -0.03, 0.0)
GRX = (0.0, 0.4, 0.8, 1.2, 1.6)
GRY = (-0.6, -0.3, 0.0, 0.3, 0.6)
GC = (0.0, 0.5, 0.9, 1.3)                  # long-finger curl scalar

# --- stage 2: refinement budget -------------------------------------------
N_STARTS = 400
N_ITERS = 400
SIGMA0 = 0.35                             # rad / m, relative to joint range
SIGMA_MIN = 0.004
POLISH_ITERS = 300                       # extra tiny-sigma polish from the best pose
POLISH_SIGMA0 = 0.012
SEED = 20261015

PEDESTAL_RE = re.compile(r'<geom name="pedestal"[^>]*?/>', re.DOTALL)


def scene_xml(pedestal_h: float) -> str:
    text = MJCF.read_text()
    match = PEDESTAL_RE.search(text)
    assert match, "pedestal geom line not found"
    old = match.group(0)
    if pedestal_h <= 0.0:
        # no pedestal: park the geom far below the table where it cannot touch
        # anything (a zero-size box still generates degenerate contacts)
        new = old.replace('pos="0 0 0.7300"', 'pos="0 0 -5.0000"')
        assert new != old
    else:
        new = re.sub(r'size="[^"]*"', f'size="0.30 0.30 {pedestal_h / 2:.4f}"', old)
        new = re.sub(r'pos="0 0 [^"]*"',
                     f'pos="0 0 {TABLE_TOP_M + pedestal_h / 2:.4f}"', new)
    return text.replace(old, new)


def compile_scene(pedestal_h: float):
    stl_dir = (MJCF.parent / "../../runs/upstream/menagerie/wonik_allegro/assets").resolve()
    assets = {p.name: p.read_bytes() for p in stl_dir.glob("*.stl")}
    model = mujoco.MjModel.from_xml_string(scene_xml(pedestal_h), assets)
    return model, mujoco.MjData(model)


class Rig:
    """Everything the objective needs, resolved once per scene."""

    def __init__(self, model, data, pedestal_h: float) -> None:
        self.m, self.d = model, data
        self.obj_adr = int(model.jnt_qposadr[mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, "object_free")])
        self.box_z = TABLE_TOP_M + pedestal_h + baf.BOX_HALF_M[2]
        self.bx, self.by, self.bz = baf.BOX_HALF_M
        self.ogeom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "object_geom")
        self.pad_geoms: list[tuple[str, int]] = []
        self.hand_geoms: list[int] = []
        for pref in ("ff", "mf", "rf", "th"):
            for role in ("base", "proximal", "medial", "distal", "tip"):
                bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{pref}_{role}")
                if bid < 0:
                    continue
                gids = [g for g in range(model.ngeom)
                        if int(model.geom_bodyid[g]) == bid
                        and "visual" not in (mujoco.mj_id2name(
                            model, mujoco.mjtObj.mjOBJ_GEOM, g) or "")]
                for g in gids:
                    self.hand_geoms.append(g)
                    if role == "tip":
                        self.pad_geoms.append((f"{pref}_{role}", g))
        # the palm is not a pad, but burying it is just as non-physical
        palm_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "palm")
        self.hand_geoms += [g for g in range(model.ngeom)
                            if int(model.geom_bodyid[g]) == palm_bid
                            and "visual" not in (mujoco.mj_id2name(
                                model, mujoco.mjtObj.mjOBJ_GEOM, g) or "")]
        names = ("wrist_x", "wrist_y", "wrist_z", "wrist_rx", "wrist_ry", "wrist_rz")
        wrist_j = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in names]
        finger_j = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
                    for n in baf.FINGER_JOINTS]
        assert min(wrist_j + finger_j) >= 0, "a commanded joint did not resolve"
        self.wrist_q = [int(model.jnt_qposadr[j]) for j in wrist_j]
        self.finger_q = np.array([model.jnt_qposadr[j] for j in finger_j], dtype=int)
        adr = np.array(wrist_j + finger_j, dtype=int)
        # Bounds are the ACTUATOR command ranges, not jnt_range: the env's actions
        # map onto ctrlrange, so a pose outside it is not a pose the controller can
        # ask for. jnt_range is the mechanical stop, which a position servo can
        # transiently pass (R7 measured exactly that on mfj1) but cannot hold.
        cmd_lo = np.empty(22)
        cmd_hi = np.empty(22)
        for k, j in enumerate(wrist_j + finger_j):
            acts = [a for a in range(model.nu) if int(model.actuator_trnid[a, 0]) == j]
            if acts:
                cmd_lo[k], cmd_hi[k] = model.actuator_ctrlrange[acts[0]]
            else:
                cmd_lo[k], cmd_hi[k] = model.jnt_range[j]
        self.jnt_lo = model.jnt_range[adr, 0].astype(float).copy()
        self.jnt_hi = model.jnt_range[adr, 1].astype(float).copy()
        self.lo = cmd_lo
        self.hi = cmd_hi
        self.span = np.where(self.hi > self.lo, self.hi - self.lo, 1.0)
        self.ft = np.zeros(6)      # [point_on_geom1, point_on_geom2] (dist = RETURN)

    def pose_to_qpos(self, pose: np.ndarray) -> None:
        q = self.d.qpos
        q[self.wrist_q] = pose[:6]
        q[self.finger_q] = pose[6:]
        q[self.obj_adr + 2] = self.box_z

    def wall_gaps(self, pose: np.ndarray) -> tuple[float, float, list[dict], bool]:
        """(best -x gap, best +x gap, per-pad records, any_hand_geom_buried)

        mj_geomDistance returns the SIGNED distance and fills
        ``fromto = [point_on_geom1, point_on_geom2]`` - it does NOT store the
        distance in fromto[0] (verified against two boxes at a known 100 mm
        gap: return 0.100000, fromto[0:3] = the geom1 face point). Reading
        fromto[0] as the distance silently turns the measurement into a world
        coordinate.
        """
        mujoco.mj_resetData(self.m, self.d)
        self.pose_to_qpos(pose)
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_forward(self.m, self.d)
        ft = self.ft
        buried = False
        for g in self.hand_geoms:
            if float(mujoco.mj_geomDistance(self.m, self.d, g, self.ogeom, 0.05, ft)) < -1e-6:
                buried = True
                break
        recs: list[dict] = []
        for name, g in self.pad_geoms:
            dist = float(mujoco.mj_geomDistance(self.m, self.d, g, self.ogeom, 0.05, ft))
            if dist >= DIST_SKIP_M:          # farther than this: never a candidate
                continue
            qx, qy, qz = (float(v) for v in ft[3:6])   # the point ON the box
            side = 1 if qx >= 0.0 else -1
            # deviation from the WALL rectangle: the contact point has to sit ON
            # that face (|x| = half_x), not merely inside its y/z projection -
            # a nearest point on the top face does not count as a wall contact.
            off_face = max(0.0, abs(abs(qx) - self.bx) - WALL_BAND_M) \
                + max(0.0, abs(qy) - self.by - WALL_BAND_M) \
                + max(0.0, abs(qz - self.box_z) - self.bz - WALL_BAND_M)
            recs.append({"pad": name, "side": side, "dist_m": dist,
                         "gap_m": dist + off_face, "off_face_m": off_face,
                         "q_box": [round(qx, 5), round(qy, 5), round(qz - self.box_z, 5)]})
        best = {1: float("inf"), -1: float("inf")}
        for r in recs:
            best[r["side"]] = min(best[r["side"]], r["gap_m"])
        return best[-1], best[1], recs, buried

    def objective(self, pose: np.ndarray) -> tuple[float, dict]:
        neg, pos, recs, buried = self.wall_gaps(pose)
        j = max(neg, pos)
        if buried:
            j += BURY_PENALTY_M
        return float(j), {"neg_m": neg, "pos_m": pos, "buried": buried, "recs": recs}


def coarse_poses(lo: np.ndarray, hi: np.ndarray, rng: np.random.Generator):
    for wx in GW:
        for wy in GY:
            for wz in GZ:
                for rx in GRX:
                    for ry in GRY:
                        for cl in GC:
                            for ct in (0.3, 0.7, 1.1):
                                pose = np.zeros(22)
                                pose[:6] = [wx, wy, wz, rx, ry, 0.0]
                                for k in range(3):
                                    pose[6 + 4 * k: 6 + 4 * k + 4] = cl
                                    pose[6 + 4 * k] = 0.3 * cl
                                pose[18:22] = ct
                                pose[18] = 0.35 * ct + 0.1
                                yield pose
    # plus uniform random poses so the coarse stage is not trapped in its grid
    for _ in range(20000):
        yield rng.uniform(lo, hi)


def _local(rig: Rig, start: np.ndarray, rng: np.random.Generator,
           iters: int, sigma0: float, sigma_min: float) -> tuple[float, np.ndarray]:
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


def refine(rig: Rig, start: np.ndarray, rng: np.random.Generator) -> tuple[float, np.ndarray]:
    j, p = _local(rig, start, rng, N_ITERS, SIGMA0, SIGMA_MIN)
    j2, p2 = _local(rig, p, rng, POLISH_ITERS, POLISH_SIGMA0, 0.0008)
    return (j2, p2) if j2 < j else (j, p)


def scan(pedestal_h: float) -> dict:
    t0 = time.time()
    model, data = compile_scene(pedestal_h)
    rig = Rig(model, data, pedestal_h)
    rng = np.random.default_rng(SEED + int(pedestal_h * 1e3))

    n_coarse = 0
    coarse_best = (float("inf"), None, None)
    coarse_hits: list[dict] = []
    for pose in coarse_poses(rig.lo, rig.hi, rng):
        n_coarse += 1
        j, info = rig.objective(pose)
        if j < coarse_best[0]:
            coarse_best = (j, pose.copy(), info)
        if j <= TOL and not info["buried"]:
            coarse_hits.append({"stage": "coarse", "j_m": j,
                                "pose": [float(v) for v in pose]})

    # refinement: coarse-grid seeds (the best few) plus random starts
    seeds = [coarse_best[1]] if coarse_best[1] is not None else []
    best_ref = (float("inf"), None)
    for s in seeds:
        j, p = refine(rig, s, rng)
        if j < best_ref[0]:
            best_ref = (j, p)
    n_rand = 0
    for _ in range(N_STARTS - len(seeds)):
        j, p = refine(rig, rng.uniform(rig.lo, rig.hi), rng)
        n_rand += 1
        if j < best_ref[0]:
            best_ref = (j, p)

    hits = list(coarse_hits)
    if best_ref[1] is not None:
        j, info = rig.objective(best_ref[1])
        if j <= TOL and not info["buried"]:
            hits.append({"stage": "refine", "j_m": j,
                         "pose": [float(v) for v in best_ref[1]]})

    j_best, info_best = rig.objective(best_ref[1]) if best_ref[1] is not None else (float("nan"), {})
    single = min((r["gap_m"] for r in info_best.get("recs", [])), default=float("inf"))
    return {
        "pedestal_mm": round(pedestal_h * 1e3, 3),
        "box_rest_z_m": round(rig.box_z, 6),
        "box_top_m": round(rig.box_z + rig.bz, 6),
        "palm_floor_m": None,
        "n_coarse_poses": n_coarse,
        "n_refine_starts": N_STARTS,
        "coarse_best_J_m": None if coarse_best[0] == float("inf") else coarse_best[0],
        "best_J_m": None if not np.isfinite(j_best) else j_best,
        "best_neg_x_gap_m": info_best.get("neg_m"),
        "best_pos_x_gap_m": info_best.get("pos_m"),
        "best_single_wall_gap_m": single,
        "best_pose": None if best_ref[1] is None else [float(v) for v in best_ref[1]],
        "best_pads": info_best.get("recs"),
        "K1_hits": len(hits),
        "K1_two_wall_pads": len(hits) > 0,
        "K1_examples": hits[:3],
        "tol_m": TOL,
        "seconds": round(time.time() - t0, 1),
    }


def main() -> None:
    cells = []
    for h in PEDESTAL_HEIGHTS_M:
        r = scan(h)
        cells.append(r)
        print(f"pedestal={r['pedestal_mm']:5.1f} mm  box_top={r['box_top_m']:.3f}  "
              f"best J={r['best_J_m'] * 1e3 if r['best_J_m'] is not None else float('nan'):8.3f} mm  "
              f"(-x {r['best_neg_x_gap_m'] * 1e3 if r['best_neg_x_gap_m'] is not None else float('nan'):8.3f} / "
              f"+x {r['best_pos_x_gap_m'] * 1e3 if r['best_pos_x_gap_m'] is not None else float('nan'):8.3f})  "
              f"K1={r['K1_hits']}  [{r['seconds']}s]", flush=True)
    summary = {
        "question": "does lowering the pedestal (box top below the palm floor) open "
                    "a pose with fingertip pads on BOTH side walls?",
        "criterion": "K1: J = max(best -x wall gap, best +x wall gap) <= the w35d/R5 "
                     "tolerance, pads only, no hand geom buried",
        "tol_m": TOL,
        "wall_band_m": WALL_BAND_M,
        "search": {"coarse_grid": {"wx": list(GW), "wy": list(GY), "wz": list(GZ),
                                   "rx": list(GRX), "ry": list(GRY), "curl": list(GC)},
                   "random_coarse": 20000,
                   "refine_starts": N_STARTS, "refine_iters": N_ITERS,
                   "sigma0": SIGMA0, "seed": SEED},
        "cells": cells,
        "any_K1": any(c["K1_two_wall_pads"] for c in cells),
    }
    OUT.write_text(json.dumps(summary, indent=2))
    print(f"any K1: {summary['any_K1']}  -> {OUT}")


if __name__ == "__main__":
    main()
