"""Wave 3.5-D shared scan infrastructure.

Self-contained replica of the runs/w35 method (static qpos placement +
mj_forward + mj_geomDistance / contact scoring). The joint metric is IDENTICAL
to rx_coupling_diag.joint_score; the pad-state scoring follows
rx_widen_scan.score. No MJCF file is modified; thumb-range variants are
in-memory model patches for diagnosis only.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

REPO = pathlib.Path("/Users/guohongbin/v2d-mac")
sys.path.insert(0, str(REPO))

import mujoco  # noqa: E402
import importlib.util  # noqa: E402

XML = REPO / "grounding/assets/allegro_floating.xml"

spec = importlib.util.spec_from_file_location(
    "baf_w35d", REPO / "grounding/assets/build_allegro_floating.py")
baf = importlib.util.module_from_spec(spec)
sys.modules["baf_w35d"] = baf
spec.loader.exec_module(baf)

mj = mujoco

WRIST = ("wrist_x", "wrist_y", "wrist_z", "wrist_rx", "wrist_ry", "wrist_rz")
FINGERS = tuple(f"{f}j{i}" for f in ("ff", "mf", "rf", "th") for i in range(4))
THJ = FINGERS[12:16]
FROMTO = np.zeros(6)
NEAR = 0.015          # "near the wall" band, same as kin_diag
CAP = 0.05

S: dict | None = None


def _setup(thumb_patch: tuple[tuple[float, float], ...] | None = None) -> dict:
    model = mj.MjModel.from_xml_path(str(XML))
    if thumb_patch is not None:
        for name, (lo, hi) in zip(THJ, thumb_patch):
            jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, name)
            model.jnt_range[jid] = (lo, hi)
    data = mj.MjData(model)
    tol = baf.penetration_tolerance(mjcf_path=XML)
    wqadr = np.array([model.jnt_qposadr[mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, n)]
                      for n in WRIST], dtype=int)
    fqadr = np.array([model.jnt_qposadr[mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, n)]
                      for n in FINGERS], dtype=int)
    flo = np.array([model.jnt_range[mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, n)][0]
                    for n in FINGERS])
    fhi = np.array([model.jnt_range[mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, n)][1]
                    for n in FINGERS])
    box = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, "object_geom")
    palm = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, "palm_collision")
    obj_body = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "object")
    pads = {}
    for f in ("ff", "mf", "rf", "th"):
        bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, f"{f}_tip")
        pads[f] = [g for g in range(model.ngeom)
                   if model.geom_bodyid[g] == bid
                   and model.geom_type[g] == mj.mjtGeom.mjGEOM_CAPSULE][0]
    return dict(model=model, data=data, tol=tol, wqadr=wqadr, fqadr=fqadr,
                flo=flo, fhi=fhi, box=box, palm=palm, obj_body=obj_body,
                pads=pads, thumb_patch=thumb_patch)


def init_worker(thumb_patch=None):
    global S
    S = _setup(thumb_patch)


def gdist(g1: int, g2: int, distmax: float = 0.06) -> float:
    return float(mj.mj_geomDistance(S["model"], S["data"], g1, g2, distmax, FROMTO))


def strict_wall_eval(base_pos, rot, fq):
    """Re-evaluate one pose with a STRICT wall-contact definition.

    A pad touching side s (sign of pad_x - box_x) counts as a strict x-wall
    contact iff mj_geomDistance's nearest point on the BOX falls inside that
    side-face rectangle (|py - bp_y| <= y_half AND pz <= top), the pad centre
    is below the top plane, and the distance is <= R5 tol. Distances to the
    y-end faces or to the unbounded extension of the top plane do NOT count.
    Returns dict(strict_contacts, true_dual_k1, sc) or None if infeasible.
    """
    model, data = S["model"], S["data"]
    box, palm, pads = S["box"], S["palm"], S["pads"]
    tol = S["tol"]
    data.qpos[:] = model.qpos0
    data.qpos[S["wqadr"]] = np.concatenate([base_pos, rot])
    data.qpos[S["fqadr"]] = fq
    mj.mj_forward(model, data)
    bp = data.xpos[S["obj_body"]]
    half = model.geom_size[box]
    top = float(bp[2] + half[2])
    pen = 0.0
    for i in range(data.ncon):
        pen = max(pen, -float(data.contact[i].dist))
    if pen > tol:
        return None
    if gdist(palm, box, 0.15) <= 0.0:
        return None
    ft = np.zeros(6)
    strict = {}
    for f, g in pads.items():
        d = float(mj.mj_geomDistance(model, data, g, box, 0.06, ft))
        pbox = ft[3:6]  # nearest point on the box
        side = int(np.sign(float(data.geom_xpos[g][0]) - float(bp[0])))
        on_face = abs(float(pbox[0]) - (float(bp[0]) + side * half[0])) <= tol
        in_rect = (abs(float(pbox[1]) - float(bp[1])) <= half[1]) and (float(pbox[2]) <= top + tol)
        below = bool(float(data.geom_xpos[g][2]) < top)
        strict[f] = {"side": side, "gap_m": d, "below": below,
                     "strict_wall": bool(below and d <= tol and on_face and in_rect),
                     "nearest_on_box": [round(float(v), 5) for v in pbox]}
    sides = [f for f, st in strict.items() if st["strict_wall"]]
    neg = [f for f in sides if strict[f]["side"] < 0]
    pos = [f for f in sides if strict[f]["side"] > 0]
    return {"strict_contacts": strict, "strict_dual_k1": bool(neg and pos),
            "pen_m": float(pen)}


def cmd16(cl: float, ct: float, t0: float) -> np.ndarray:
    """rx_widen_scan.finger_cmd: long fingers tied (j0 = cl*0.3), thumb tied
    with thj0 = t0. Commands normalized to [-1, 1]."""
    a = np.empty(16)
    for k, _f in enumerate(("ff", "mf", "rf")):
        a[4 * k: 4 * k + 4] = cl
        a[4 * k] = cl * 0.3
    a[12:16] = ct
    a[12] = t0
    return S["flo"] + (a + 1.0) * 0.5 * (S["fhi"] - S["flo"])


def q_from_cmd(a: np.ndarray) -> np.ndarray:
    return S["flo"] + (np.asarray(a) + 1.0) * 0.5 * (S["fhi"] - S["flo"])


def joint_score(base_pos, rot, fq: np.ndarray):
    """rx_coupling_diag.joint_score with the thumb command kept explicit.

    fq: 16 absolute joint positions (rad).
    """
    model, data = S["model"], S["data"]
    box, palm, pads = S["box"], S["palm"], S["pads"]
    tol = S["tol"]
    data.qpos[:] = model.qpos0
    data.qpos[S["wqadr"]] = np.concatenate([base_pos, rot])
    data.qpos[S["fqadr"]] = fq
    mj.mj_forward(model, data)
    bp = data.xpos[S["obj_body"]]
    top = float(bp[2] + model.geom_size[box][2])
    pen = 0.0
    for i in range(data.ncon):
        pen = max(pen, -float(data.contact[i].dist))
    if pen > tol:
        return None
    palm_gap = gdist(palm, box, 0.15)
    if palm_gap <= 0.0:
        return None
    long_gap, long_short = CAP, CAP
    long_near = []
    long_state = {}
    for f in ("ff", "mf", "rf"):
        g = data.geom_xpos[pads[f]]
        d = gdist(pads[f], box)
        if g[2] < top:
            long_gap = min(long_gap, d)
            if d <= NEAR:
                long_near.append(f)
            long_state[f] = {"below": True, "gap_m": round(d, 5),
                             "side": int(np.sign(float(g[0]) - float(bp[0]))),
                             "pos": [round(float(v), 4) for v in g]}
        else:
            long_short = min(long_short, float(g[2] - top))
            long_state[f] = {"below": False,
                             "above_top_m": round(float(g[2] - top), 5),
                             "pos": [round(float(v), 4) for v in g]}
    tg = data.geom_xpos[pads["th"]]
    th_below = bool(tg[2] < top)
    th_d = gdist(pads["th"], box)
    th_metric = min(th_d, CAP) if th_below else min(float(tg[2] - top), CAP)
    joint = max(long_gap, th_metric)
    return {
        "base": [float(v) for v in base_pos], "rot": [float(v) for v in rot],
        "joint_m": joint, "long_gap_m": long_gap, "long_short_m": long_short,
        "long_near": long_near, "long_state": long_state,
        "th_below": th_below, "th_metric_m": th_metric, "th_gap_m": th_d,
        "th_side": int(np.sign(float(tg[0]) - float(bp[0]))),
        "th_pos": [round(float(v), 4) for v in tg],
        "pen_m": float(pen), "palm_gap_m": float(palm_gap),
    }


def descend(seed, thumb_builder="tied", thumb_patch=None):
    """Greedy coordinate descent on the joint metric from one seed pose.

    thumb_builder:
      "tied"   - 3-D command (cl, ct, t0), rx_widen_scan.finger_cmd semantics
      "untied" - 5-D command (cl, thj0, thj1, thj2, thj3 normalized), long
                 fingers still tied; per-joint thumb freedom. If ranges were
                 patched, the untied builder uses the patched flo/fhi.
    Seeds carry commands in normalized [-1, 1] space.
    """
    if S is None or S.get("thumb_patch") != thumb_patch:
        init_worker(thumb_patch)

    def fq_of(cmd):
        if thumb_builder == "tied":
            return cmd16(cmd[0], cmd[1], cmd[2])
        a = np.empty(16)
        for k in range(3):
            a[4 * k: 4 * k + 4] = cmd[0]
            a[4 * k] = cmd[0] * 0.3
        a[12:16] = cmd[1:5]
        return q_from_cmd(a)

    s = dict(seed)
    cur = joint_score(np.array(s["base"]), np.array(s["rot"]), fq_of(s["cmd"]))
    if cur is None:
        return None
    steps = [0.02, 0.01, 0.005, 0.002]
    rstep = [0.15, 0.07, 0.03]
    cdim = 3 if thumb_builder == "tied" else 5
    csteps = tuple((0.25, 0.1) for _ in range(cdim))
    improved = True
    evals = 1
    while improved and evals < 6000:
        improved = False
        for ai, span in ((0, steps), (1, steps), (2, steps)):
            for st in span:
                for sgn in (1.0, -1.0):
                    cand = dict(s)
                    cand["base"] = list(s["base"])
                    cand["base"][ai] = s["base"][ai] + sgn * st
                    r = joint_score(np.array(cand["base"]),
                                    np.array(cand["rot"]), fq_of(cand["cmd"]))
                    evals += 1
                    if r is not None and r["joint_m"] < cur["joint_m"] - 1e-6:
                        s, cur, improved = cand, r, True
        for ai in range(3):
            for st in rstep:
                for sgn in (1.0, -1.0):
                    cand = dict(s)
                    cand["rot"] = list(s["rot"])
                    cand["rot"][ai] = s["rot"][ai] + sgn * st
                    r = joint_score(np.array(cand["base"]),
                                    np.array(cand["rot"]), fq_of(cand["cmd"]))
                    evals += 1
                    if r is not None and r["joint_m"] < cur["joint_m"] - 1e-6:
                        s, cur, improved = cand, r, True
        for ai in range(cdim):
            for st in tuple(csteps[ai]) + (csteps[ai][-1] / 2.5,):
                for sgn in (1.0, -1.0):
                    cand = dict(s)
                    cand["cmd"] = list(s["cmd"])
                    cand["cmd"][ai] = max(-1.0, min(1.0, s["cmd"][ai] + sgn * st))
                    r = joint_score(np.array(cand["base"]),
                                    np.array(cand["rot"]), fq_of(cand["cmd"]))
                    evals += 1
                    if r is not None and r["joint_m"] < cur["joint_m"] - 1e-6:
                        s, cur, improved = cand, r, True
    cur["descend_evals"] = evals
    cur["cmd"] = list(s["cmd"])
    cur["descend_builder"] = thumb_builder
    cur["descend_ranges"] = "patched" if thumb_patch else "stock"
    return cur


def run_chunk(args):
    """Chunk = (thumb_patch, rxs, rys, rzs, cmd_grid, zs, xys, keep).

    cmd_grid: list of (cl, ct, t0) normalized triples (tied builder only).
    Returns per-rx-bucket best rows.
    """
    thumb_patch, rxs, rys, rzs, cmd_grid, zs, xys, keep = args
    if S is None or S.get("thumb_patch") != thumb_patch:
        init_worker(thumb_patch)
    best = {rx: [] for rx in rxs}
    th_rows = {rx: [] for rx in rxs}
    stats = {rx: {"n": 0, "ok": 0, "th_below": 0, "sim": 0,
                  "grade": 0, "near": 0} for rx in rxs}
    tol = S["tol"]
    for (cl, ct, t0) in cmd_grid:
        fq = cmd16(cl, ct, t0)
        for ry in rys:
            for rz in rzs:
                for rx in rxs:
                    rot = (rx, ry, rz)
                    for z in zs:
                        for x in xys:
                            for y in xys:
                                sc = joint_score(np.array([x, y, z]), np.array(rot), fq)
                                st = stats[rx]
                                st["n"] += 1
                                if sc is None:
                                    continue
                                st["ok"] += 1
                                sc["cmd"] = [cl, ct, t0]
                                if sc["th_below"]:
                                    st["th_below"] += 1
                                    th_rows[rx].append(sc)
                                if (sc["th_below"] and sc["long_near"]):
                                    st["sim"] += 1
                                if sc["joint_m"] <= tol:
                                    st["grade"] += 1
                                if sc["joint_m"] <= NEAR:
                                    st["near"] += 1
                                best[rx].append(sc)
    out = {}
    for rx in rxs:
        best[rx].sort(key=lambda r: r["joint_m"])
        th_rows[rx].sort(key=lambda r: r["th_metric_m"])
        out[str(rx)] = {"stats": stats[rx], "best": best[rx][:keep],
                        "th_rows": th_rows[rx][:keep]}
    return {"thumb_patch": thumb_patch, "per_rx": out}
