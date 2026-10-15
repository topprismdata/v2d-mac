"""Wave 3.5 kinematic validation: does a clean dual-side pad grasp pose exist?

base_solve.py's method, re-targeted from retarget-replay to grasp-existence:
place qpos directly (no dynamics), mj_forward, and score with MuJoCo's own
contact list + mj_geomDistance. A pose is a *clean grasp* only if ALL of:

  K1 dual-side pad contact : at least one distal fingertip pad within the touch
                             tolerance of the box on the -x wall AND one on the
                             +x wall, both below the box top (side walls).
  K2 no burial             : deepest penetration over ALL contacts <= the
                             variant's own R5 baseline (k=4 x measured sink).
  K3 palm clear of box     : mj_geomDistance(palm_collision, object_geom) > 0.
  K4 base inside the caps  : the solved wrist slide command lies inside the
                             historical ctrlrange (x/y +/-0.18, z -0.163..+0.16).

Staged search: coarse grid (slides x rot x curl) -> coordinate-descent
refinement with halving steps. Deterministic. Writes runs/w35/kin_ped<h>.json.

    python runs/w35/kinematics.py 0.10 runs/ped100/allegro_floating.xml
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import numpy as np

REPO = pathlib.Path("/Users/guohongbin/v2d-mac")
sys.path.insert(0, str(REPO))

import mujoco  # noqa: E402

H = float(sys.argv[1])
XML = REPO / sys.argv[2]
OUT_JSON = REPO / f"runs/w35/kin_ped{int(round(H * 1000))}.json"

spec = importlib.util.spec_from_file_location(
    "baf_kin", REPO / "grounding/assets/build_allegro_floating.py")
baf = importlib.util.module_from_spec(spec)
sys.modules["baf_kin"] = baf
spec.loader.exec_module(baf)

TOL = baf.penetration_tolerance(mjcf_path=XML)   # variant's own R5 baseline
model = mujoco.MjModel.from_xml_path(str(XML))
data = mujoco.MjData(model)

mj = mujoco
WRIST = ("wrist_x", "wrist_y", "wrist_z", "wrist_rx", "wrist_ry", "wrist_rz")
wqadr = np.array([model.jnt_qposadr[mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, n)]
                  for n in WRIST], dtype=int)
FINGERS = tuple(f"{f}j{i}" for f in ("ff", "mf", "rf", "th") for i in range(4))
fqadr = np.array([model.jnt_qposadr[mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, n)]
                  for n in FINGERS], dtype=int)
flo = np.array([model.jnt_range[mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, n)][0]
                for n in FINGERS])
fhi = np.array([model.jnt_range[mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, n)][1]
                for n in FINGERS])
box = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, "object_geom")
palm = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, "palm_collision")
pedestal = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, "pedestal")
pads: dict[str, int] = {}
for f in ("ff", "mf", "rf", "th"):
    bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, f"{f}_tip")
    gids = [g for g in range(model.ngeom)
            if model.geom_bodyid[g] == bid
            and model.geom_type[g] == mj.mjtGeom.mjGEOM_CAPSULE]
    assert len(gids) == 1
    pads[f] = gids[0]

obj_body = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "object")
FROMTO = np.zeros(6)


def gdist(g1: int, g2: int, distmax: float = 0.10) -> float:
    return float(mj.mj_geomDistance(model, data, g1, g2, distmax, FROMTO))


def finger_cmd(c_long: float, c_th: float) -> np.ndarray:
    """Env-style mapping: a=-1 is straight (lo), a=+1 is the curl stop (hi);
    the j0 abduction gets a third of the curl scalar (scan's J0_GAIN)."""
    a = np.empty(16)
    a[0:12] = c_long
    a[12:16] = c_th
    a[[0, 4, 8, 12]] *= 0.3
    return flo + (a + 1.0) * 0.5 * (fhi - flo)


PRE_CURL = np.array([0.0, 0.9, 0.7, 0.4] * 3 + [0.8, 0.5, 0.4, 0.3])

# ------------------------------------------------------------- scene facts
def scene_facts() -> dict:
    data.qpos[:] = model.qpos0
    mj.mj_forward(model, data)
    bp = data.xpos[obj_body].copy()
    half = model.geom_size[box].astype(float)
    pgp = data.geom_xpos[palm].copy()
    phalf = model.geom_size[palm].astype(float)
    facts = {
        "box_centre_m": bp.tolist(),
        "box_top_z_m": float(bp[2] + half[2]),
        "pedestal_top_z_m": float(model.geom_pos[pedestal][2]
                                  + model.geom_size[pedestal][2]),
        "palm_collision_world_pos_m": pgp.tolist(),
        "palm_collision_half_size_m": phalf.tolist(),
        "palm_plate_bottom_z_m": float(pgp[2] - phalf[2]),
        "palm_box_gap_at_home_m": gdist(palm, box),
        "palm_pedestal_gap_at_home_m": gdist(palm, pedestal),
        "pad_z_at_home_m": {f: float(data.geom_xpos[g][2]) for f, g in pads.items()},
        "contacts_at_home": contact_summary(),
        "r5_resting_sink_m": baf.measure_resting_sink(mjcf_path=XML),
        "r5_tolerance_m": TOL,
        "wrist_caps_m": {"x": [-0.18, 0.18], "y": [-0.18, 0.18],
                         "z": [-0.163, 0.16]},
    }
    return facts


def contact_summary() -> dict:
    rows = []
    for i in range(data.ncon):
        c = data.contact[i]
        n1 = mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, int(c.geom1)) or f"geom{c.geom1}"
        n2 = mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, int(c.geom2)) or f"geom{c.geom2}"
        rows.append((n1, n2, float(-c.dist)))
    return rows


# ------------------------------------------------------------------ scoring
def place(base: np.ndarray, rot: np.ndarray, fcmd: np.ndarray) -> None:
    data.qpos[:] = model.qpos0
    data.qpos[wqadr] = np.concatenate([base, rot])
    data.qpos[fqadr] = fcmd
    mj.mj_forward(model, data)


def score(base: np.ndarray, rot: np.ndarray, fcmd: np.ndarray,
          cl: float = 0.0, ct: float = 0.0) -> dict:
    place(base, rot, fcmd)
    bp = data.xpos[obj_body]
    box_top = float(bp[2] + model.geom_size[box][2])
    pen_all = 0.0
    pen_box = 0.0
    for i in range(data.ncon):
        c = data.contact[i]
        d = -float(c.dist)
        pen_all = max(pen_all, d)
        if box in (int(c.geom1), int(c.geom2)):
            pen_box = max(pen_box, d)
    pad_gap = {f: gdist(g, box, 0.05) for f, g in pads.items()}
    side = {}
    below = {}
    for f, g in pads.items():
        side[f] = int(np.sign(float(data.geom_xpos[g][0]) - float(bp[0])))
        below[f] = bool(float(data.geom_xpos[g][2]) < box_top)
    # strict: within the R5 touch tolerance (the K1 gate)
    neg = [f for f in pads if side[f] < 0 and below[f] and pad_gap[f] <= TOL]
    pos = [f for f in pads if side[f] > 0 and below[f] and pad_gap[f] <= TOL]
    touching = [f for f in pads if pad_gap[f] <= TOL and below[f]]
    # proximity tier (base_solve's GRASP_TIP_RADIUS): what a coarse grid can rank
    NEAR = 0.015
    near = [f for f in pads if pad_gap[f] <= NEAR and below[f]]
    near_neg = [f for f in near if side[f] < 0]
    near_pos = [f for f in near if side[f] > 0]
    neg_near = [f for f in pads if side[f] < 0 and below[f] and pad_gap[f] <= NEAR]
    pos_near = [f for f in pads if side[f] > 0 and below[f] and pad_gap[f] <= NEAR]
    palm_gap = gdist(palm, box)
    palm_ped = gdist(palm, pedestal)
    K1 = bool(neg and pos)
    K2 = bool(pen_all <= TOL)
    K3 = bool(palm_gap > 0.0)
    K4 = True  # search bounds guarantee it; margins reported separately
    mean_touch_gap = (float(np.mean([pad_gap[f] for f in touching]))
                      if touching else 1.0)
    mean_near_gap = (float(np.mean([pad_gap[f] for f in near]))
                     if near else 1.0)
    return {
        "base": base.tolist(), "rot": rot.tolist(),
        "curl_long": float(cl), "curl_th": float(ct),
        "pad_gap_m": {k: float(v) for k, v in pad_gap.items()},
        "pad_side": side, "pad_below_top": below,
        "neg_side_pads": neg, "pos_side_pads": pos,
        "near_pads": near, "near_neg": near_neg, "near_pos": near_pos,
        "neg_near": neg_near, "pos_near": pos_near,
        "pen_max_m": float(pen_all), "pen_box_m": float(pen_box),
        "palm_box_gap_m": float(palm_gap),
        "palm_pedestal_gap_m": float(palm_ped),
        "K1": K1, "K2": K2, "K3": K3, "K4": K4,
        "clean": bool(K1 and K2 and K3 and K4),
        # coarse ranks by proximity; the refinement then drives gaps to <= TOL
        "rank": (0 if (near_neg and near_pos) else 1,
                 -len(near), -round(mean_near_gap, 6),
                 -len(touching), -round(mean_touch_gap, 6),
                 round(pen_all, 6)),
    }


# ------------------------------------------------------------------- search
ROT_RY = (-1.20, -0.60, 0.0, 0.60, 1.20)
ROT_RX = (-1.20, 0.0, 1.20)
ROT_RZ = (-1.57, -0.80, 0.0, 0.80, 1.57)
CURLS = ((-1.0, -1.0), (0.0, -1.0), (0.0, 0.0), (0.5, 0.5), (1.0, 1.0),
         (1.0, -1.0), (0.3, 0.3), (0.0, 1.0))
CAPS = np.array([0.18, 0.18, 0.163])


def search() -> dict:
    facts = scene_facts()
    print(json.dumps(facts, indent=2), flush=True)

    n = 0
    n_both_near = 0
    n_both_touch = 0
    best_both = None   # placement minimising max(min neg gap, min pos gap)
    pool: list[dict] = []   # top-K by both-side closeness, for multi-start
    zx = np.arange(-0.163, 0.1601, 0.03)
    xy = np.arange(-0.12, 0.1201, 0.03)
    for ry in ROT_RY:
        for rx in ROT_RX:
            for rz in ROT_RZ:
                rot0 = np.array([rx, ry, rz])
                for cl, ct in CURLS:
                    fcmd = finger_cmd(cl, ct)
                    for z in zx:
                        for x in xy:
                            for y in xy:
                                base = np.array([x, y, z])
                                sc = score(base, rot0, fcmd, cl, ct)
                                n += 1
                                if not (sc["K2"] and sc["K3"]):
                                    continue
                                gn = min(
                                    (sc["pad_gap_m"][f] for f in sc["neg_near"]),
                                    default=1.0)
                                gp = min(
                                    (sc["pad_gap_m"][f] for f in sc["pos_near"]),
                                    default=1.0)
                                key = max(gn, gp)
                                if best_both is None or key < best_both[0]:
                                    best_both = (key, sc)
                                if sc["near_neg"] and sc["near_pos"]:
                                    n_both_near += 1
                                    pool.append((key, sc))
                                if sc["neg_side_pads"] and sc["pos_side_pads"]:
                                    n_both_touch += 1
    pool.sort(key=lambda t: t[0])
    print(f"[coarse] {n} placements | both-sides-near={n_both_near} "
          f"| both-sides-touch={n_both_touch} "
          f"| best max-side-gap={None if best_both is None else f'{best_both[0]*1000:.2f}mm'}",
          flush=True)

    refined = None
    starts = [sc for _, sc in pool[:20]] or (
        [best_both[1]] if best_both is not None else [])
    for cand in starts:
        state = np.array(cand["base"] + cand["rot"]
                         + [cand["curl_long"], cand["curl_th"]])
        steps = np.array([0.02, 0.02, 0.02, 0.10, 0.10, 0.10, 0.25, 0.25])

        def evaluate(s: np.ndarray):
            if np.any(np.abs(s[0:2]) > 0.18 + 1e-9) or not (-0.163 <= s[2] <= 0.16):
                return None
            if np.any(np.abs(s[3:6]) > np.array([1.20, 1.20, 1.57]) + 1e-9):
                return None
            return score(s[0:3].copy(), s[3:6].copy(),
                         finger_cmd(s[6], s[7]), s[6], s[7])

        def objective(sc: dict) -> float:
            gn = min((sc["pad_gap_m"][f] for f in sc["neg_near"]), default=1.0)
            gp = min((sc["pad_gap_m"][f] for f in sc["pos_near"]), default=1.0)
            tie = sum(sorted(sc["pad_gap_m"].values())[:3])
            return round(max(gn, gp), 6) + round(tie, 6) * 1e-3

        cur_obj = objective(cand)
        for _ in range(9):
            improved = False
            for dim in range(8):
                for sgn in (-1, 1):
                    trial = state.copy()
                    trial[dim] += sgn * steps[dim]
                    sc = evaluate(trial)
                    if sc is None or not (sc["K2"] and sc["K3"]):
                        continue
                    o = objective(sc)
                    if o < cur_obj - 1e-9:
                        state, cur_obj, cand, improved = trial, o, sc, True
            if not improved:
                steps *= 0.5
                if steps[0] < 0.002:
                    break
        if refined is None or (cand["clean"], -objective(cand)) > \
                (refined["clean"], -objective(refined)):
            refined = cand

    out = {
        "pedestal_h_m": H,
        "mjcf": str(XML.relative_to(REPO)),
        "criteria": {
            "K1": "distal pads within R5 tol on BOTH x-walls, below box top",
            "K2": f"max penetration over all contacts <= {TOL:.4e} m (variant R5 baseline)",
            "K3": "mj_geomDistance(palm_collision, object_geom) > 0",
            "K4": "wrist slide command inside historical ctrlrange",
        },
        "scene_facts": facts,
        "coarse_placements": n,
        "coarse_both_sides_near": n_both_near,
        "coarse_both_sides_touch": n_both_touch,
        "coarse_best_max_side_gap_m": (None if best_both is None
                                       else float(best_both[0])),
        "best": refined,
        "clean_grasp_exists_kinematically": bool(refined and refined["clean"]),
    }
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in
                      ("clean_grasp_exists_kinematically", "best")},
                     indent=2)[:2000], flush=True)
    print(f"wrote {OUT_JSON}", flush=True)
    return out


if __name__ == "__main__":
    search()
