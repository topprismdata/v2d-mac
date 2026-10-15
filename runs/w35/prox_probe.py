"""Wave 3.5-E probe: extend the palm's explicit stiff <pair> scope to the finger
proximal / medial collision geoms -- evidence only, no generator edit.

The frozen smoke5 replay (retarget/retarget_smoke.py) is reused verbatim; the only
thing this script does is (a) re-run it against in-memory XML variants that carry
extra <pair> lines and (b) wrap mujoco.mj_step so every contact of every physics
substep is recorded per geom pair.

Measured per variant:
  R  smoke5 replay      - per-geom-pair peak normal force / peak penetration, the
                          deepest hand<->box penetration and its geom, mfj1's servo
                          windup (ctrl - qpos), realised-qpos limit violations.
  L  per-geom 10 N ramp - the smoke5 command stream is amplified by a scalar lambda
                          until the *named* geom carries 10 N normal load against
                          the box; the penetration at that load is the R3 budget
                          number for that geom.

Run:  runs/.venv-retarget/bin/python runs/w35/prox_probe.py
"""
from __future__ import annotations

import json
import pathlib
import pickle
import sys
import tempfile
import time

import numpy as np

REPO = pathlib.Path("/Users/guohongbin/v2d-mac")
sys.path.insert(0, str(REPO))

import mujoco  # noqa: E402

from retarget import retarget_smoke as rs  # noqa: E402

BASE_XML = REPO / "grounding/assets/allegro_floating.xml"
CACHE = REPO / "runs/w35/prox_probe_anchor.pkl"
PAIR_TMPL = ('    <pair geom1="{g}" geom2="object_geom" solref="0.004 1" '
             'solimp="0.95 0.99 0.001 0.5 2" condim="4"/>')

FINGERS = ("ff", "mf", "rf", "th")
LINKS = ("proximal", "medial", "distal", "tip")
# candidate collision geoms, "全部指骨" scope (base excluded: it is excluded from
# palm contact by the MJCF's own <exclude> list and never touches the box)
ALL_CANDIDATES = [f"{f}_{l}_collision" for f in FINGERS for l in LINKS]
PROX = [g for g in ALL_CANDIDATES if "_proximal_" in g]
PROX_MED = PROX + [g for g in ALL_CANDIDATES if "_medial_" in g]
PROX_MED_DIS = PROX_MED + [g for g in ALL_CANDIDATES if "_distal_" in g]

VARIANTS = {
    "V0_committed": [],
    "V1_proximal": PROX,
    "V2_proximal_medial": PROX_MED,
    "V3_all_phalanxes": PROX_MED_DIS,
}

TARGET_N = 10.0
LAMBDA_LO, LAMBDA_HI, LAMBDA_STEP = 1.0, 3.0, 0.1
N_BISECT = 14


# --------------------------------------------------------------------- variants
def variant_text(geoms) -> str:
    base = BASE_XML.read_text()
    palm = ('    <pair geom1="palm_collision" geom2="object_geom" '
            'solref="0.004 1" solimp="0.95 0.99 0.001 0.5 2" condim="4"/>\n')
    assert palm in base, "committed palm pair not found"
    if not geoms:
        return base
    extra = "".join(PAIR_TMPL.format(g=g) for g in geoms)
    return base.replace(palm, palm + extra, 1)


def load_variant(text):
    """meshdir is relative to the XML's own directory: write the variant beside
    the committed file, load, delete (same trick as runs/w35/palm_probe.py)."""
    with tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False,
                                     dir=str(BASE_XML.parent)) as fh:
        fh.write(text)
        tmp = pathlib.Path(fh.name)
    try:
        model = mujoco.MjModel.from_xml_path(str(tmp))
    finally:
        tmp.unlink(missing_ok=True)
    return model, mujoco.MjData(model)


# ------------------------------------------------------------------- anchor cache
def anchor_and_qpos():
    if CACHE.exists():
        with CACHE.open("rb") as fh:
            return pickle.load(fh)
    seq = rs.load_source_sequence(rs.N_FRAMES)
    joints21, _ = rs.mano_skeleton(seq)
    retargeter, _cfg = rs.build_retargeter()
    qpos = rs.retarget_sequence(joints21, retargeter)
    tip_pos = rs.retargeted_tip_positions(qpos, retargeter)
    _, model, data = rs.load_scene()
    rs._mj_forward(model, data)
    anchor = rs.solve_wrist_anchor(qpos, tip_pos, model, data)
    base = rs.solve_base_placement(qpos, anchor, model, data)
    anchor = dict(anchor, ctrl=base["ctrl"], finger_cmd=base["finger_cmd"])
    payload = {"qpos": np.asarray(qpos), "anchor": anchor}
    with CACHE.open("wb") as fh:
        pickle.dump(payload, fh)
    return payload


# ------------------------------------------------------------- step instrumentation
class ContactRecorder:
    """Wraps mujoco.mj_step and accumulates, per geom pair and per control frame:
    peak normal force, peak total contact force, peak penetration, peak |dist|."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.steps = 0
        self.pair_force_n = {}      # (name1, name2) -> peak normal force
        self.pair_force_total = {}
        self.pair_pen = {}
        self.per_step = []          # one list per physics substep
        self.ctrl_per_step = []
        self.qpos_per_step = []
        self.qvel_per_step = []
        self._names = {}
        self.model = None
        self.data = None

    def name(self, gid):
        n = self._names.get(gid)
        if n is None:
            n = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, int(gid))
            self._names[gid] = n
        return n

    def attach(self, model, data):
        self.model, self.data = model, data

    def capture(self):
        m, d = self.model, self.data
        rec = []
        for i in range(d.ncon):
            c = d.contact[i]
            f = np.zeros(6)
            mujoco.mj_contactForce(m, d, i, f)
            key = (self.name(c.geom1), self.name(c.geom2))
            pen = max(0.0, -float(c.dist))
            rec.append((key, float(f[0]), float(np.linalg.norm(f[:3])), pen))
            self.pair_force_n[key] = max(self.pair_force_n.get(key, 0.0), float(f[0]))
            self.pair_force_total[key] = max(
                self.pair_force_total.get(key, 0.0), float(np.linalg.norm(f[:3])))
            self.pair_pen[key] = max(self.pair_pen.get(key, 0.0), pen)
        self.per_step.append(rec)
        self.ctrl_per_step.append(d.ctrl.copy())
        self.qpos_per_step.append(d.qpos.copy())
        self.qvel_per_step.append(d.qvel.copy())
        self.steps += 1


REC = ContactRecorder()
_ORIG_STEP = mujoco.mj_step


def _hooked_step(model, data):
    _ORIG_STEP(model, data)
    REC.attach(model, data)
    REC.capture()


def install_hook():
    mujoco.mj_step = _hooked_step


def uninstall_hook():
    mujoco.mj_step = _ORIG_STEP


# ------------------------------------------------------------------- one replay
def run_replay(model, data, qpos, anchor):
    """Replay with the recorder attached; returns per-frame folded evidence."""
    REC.reset()
    REC.attach(model, data)
    saved = rs.load_scene
    rs.load_scene = lambda: (mujoco, model, data)
    install_hook()
    try:
        rep = rs.replay(qpos, anchor)
    finally:
        uninstall_hook()
        rs.load_scene = saved

    n_frames = rep["n_frames"]
    frame_skip = int(round(1.0 / (rs.CTRL_HZ * model.opt.timestep)))
    n_main = n_frames * frame_skip
    steps = REC.per_step[-n_main:]
    per_frame_force = {}
    per_frame_total = {}
    per_frame_pen = {}
    frame_ctrl = []
    frame_qpos = []
    for fi in range(n_frames):
        sl = slice(fi * frame_skip, (fi + 1) * frame_skip)
        for key, fn, ft, pen in (r for rec in steps[sl] for r in rec):
            per_frame_force[key] = max(per_frame_force.get(key, 0.0), fn)
            per_frame_total[key] = max(per_frame_total.get(key, 0.0), ft)
            per_frame_pen[(fi, key)] = max(per_frame_pen.get((fi, key), 0.0), pen)
        frame_ctrl.append(REC.ctrl_per_step[-n_main:][sl][-1])
        frame_qpos.append(REC.qpos_per_step[-n_main:][sl][-1])
    final_force = {}
    final_pen = {}
    frame_pair_fp = {}
    for fi in range(n_frames):
        for key, fn, ft, pen in steps[fi * frame_skip + frame_skip - 1]:
            final_force[key] = fn
            final_pen[key] = pen
            frame_pair_fp[(fi, key)] = (fn, pen)
    ev = {"n_main_steps": n_main, "frame_skip": frame_skip,
          "per_frame_force_N": per_frame_force,
          "per_frame_force_total_N": per_frame_total,
          "per_frame_pen_m": per_frame_pen,
          "final_substep_force_N": final_force,
          "final_substep_pen_m": final_pen,
          "frame_pair_force_pen": frame_pair_fp,
          "frame_ctrl": np.asarray(frame_ctrl),
          "frame_qpos": np.asarray(frame_qpos),
          "frame_qvel": np.asarray([q for q in REC.qvel_per_step[-n_main:][frame_skip - 1::frame_skip]]),
          "n_frames": n_frames}
    return rep, ev


def gate_metric(ev):
    """The gate's own view: penetration and force sampled at the END of each
    control frame, which is exactly where retarget_smoke._finger_box_contact
    reads them.  An impact spike inside a frame is not what the gate sees."""
    out = {}
    n = ev["n_frames"]
    for (n1, n2) in sorted(ev["per_frame_force_N"]):
        if "object_geom" not in (n1, n2):
            continue
        other = n2 if n1 == "object_geom" else n1
        if other.endswith("_visual"):
            continue
        best_fi, best_pen, best_fn = -1, 0.0, 0.0
        for fi in range(n):
            fn, pen = ev["frame_pair_force_pen"].get((fi, (n1, n2)), (0.0, 0.0))
            if pen > best_pen:
                best_fi, best_pen, best_fn = fi, pen, fn
        out[other] = {"frame": best_fi, "pen_m": best_pen, "force_N": best_fn,
                      "peak_pen_in_frame_m": max(
                          ev["per_frame_pen_m"].get((fi, (n1, n2)), 0.0)
                          for fi in range(n)),
                      "peak_force_in_frame_N": ev["per_frame_force_N"][(n1, n2)]}
    return out


def box_contact_evidence(ev):
    """Fold the per-frame table down to the hand<->box picture the gate sees."""
    out = {"deepest_m": 0.0, "deepest_geom": None, "deepest_frame": None,
           "deepest_force_N": 0.0, "per_geom_m": {}, "per_geom_force_N": {},
           "per_geom_deepest_frame": {}, "per_geom_frames_in_contact": {}}
    n_frames = ev["n_frames"]
    for (n1, n2) in sorted(ev["per_frame_force_N"]):
        if "object_geom" not in (n1, n2):
            continue
        other = n2 if n1 == "object_geom" else n1
        if other.endswith("_visual"):
            continue
        force = ev["per_frame_force_N"][(n1, n2)]
        pen = max(ev["per_frame_pen_m"].get((fi, (n1, n2)), 0.0)
                  for fi in range(n_frames))
        out["per_geom_m"][other] = pen
        out["per_geom_force_N"][other] = force
        out["per_geom_deepest_frame"][other] = int(max(
            range(n_frames),
            key=lambda fi: ev["per_frame_pen_m"].get((fi, (n1, n2)), 0.0)))
        out["per_geom_frames_in_contact"][other] = int(sum(
            1 for fi in range(n_frames)
            if ev["per_frame_pen_m"].get((fi, (n1, n2)), 0.0) > 0.0
            or ev["per_frame_force_N"].get((n1, n2), 0.0) > 0.0))
        if pen > out["deepest_m"]:
            out["deepest_m"] = pen
            out["deepest_geom"] = other
            out["deepest_frame"] = out["per_geom_deepest_frame"][other]
            out["deepest_force_N"] = force
    return out


def servo_windup(ev):
    """|ctrl - qpos| per finger joint: the position servo's elastic windup, which
    for a MuJoCo position actuator is exactly tau_load / kp."""
    qadr = np.array([MODEL.jnt_qposadr[rs._joint_id(MODEL, n)]
                     for n in rs.MJCF_FINGER_JOINTS])
    e = np.abs(ev["frame_ctrl"][:, ACT_FINGER] - ev["frame_qpos"][:, qadr])
    out = {}
    for i, n in enumerate(rs.MJCF_FINGER_JOINTS):
        act = ACT_OF[n.replace("j", "a")]
        kp = float(MODEL.actuator_gainprm[act, 0])
        fr = float(MODEL.actuator_forcerange[act, 1])
        out[n] = {"max_abs_rad": float(e[:, i].max()),
                  "at_frame": int(np.argmax(e[:, i])),
                  "kp_Nm_per_rad": kp,
                  "implied_tau_Nm": float(e[:, i].max() * kp),
                  "forcerange_Nm": fr,
                  "saturated_at_worst": bool(e[:, i].max() * kp >= fr - 1e-9),
                  "windup_cap_rad": fr / kp}
    return out


def limit_violations(qpos_history):
    jids = [rs._joint_id(MODEL, n) for n in rs.MJCF_FINGER_JOINTS]
    qadr = np.array([MODEL.jnt_qposadr[j] for j in jids])
    lo = np.array([MODEL.jnt_range[j][0] for j in jids])
    hi = np.array([MODEL.jnt_range[j][1] for j in jids])
    q = np.asarray(qpos_history)[:, qadr]
    viol = {}
    for i, n in enumerate(rs.MJCF_FINGER_JOINTS):
        over = np.maximum(0.0, q[:, i] - hi[i])
        under = np.maximum(0.0, lo[i] - q[:, i])
        ex = float(np.max(np.concatenate([over, under])))
        if ex > 0:
            frames = sorted(set(np.where((over > 0) | (under > 0))[0].tolist()))
            viol[n] = {"excess_rad": ex,
                       "bound": "upper" if over.max() >= under.max() else "lower",
                       "range": [float(lo[i]), float(hi[i])],
                       "max_qpos_rad": float(q[:, i].max()),
                       "frames": frames}
    return viol


# ------------------------------------------------------------- 10 N per-geom ramp
def geom_box_force_and_pen(ev, geom, quasi_static=False):
    """(peak normal, peak total, peak pen) for one geom against the box.

    ``quasi_static`` swaps the per-frame peak for the last substep of the frame:
    a finger landing on the box produces an impact spike an order of magnitude
    above the load it holds, and the R3 budget is about the load it holds.
    """
    fn = ft = 0.0
    pen = 0.0
    for (n1, n2), f in ev["per_frame_force_N"].items():
        if "object_geom" not in (n1, n2):
            continue
        other = n2 if n1 == "object_geom" else n1
        if other != geom:
            continue
        fn = max(fn, f)
        ft = max(ft, ev["per_frame_force_total_N"][(n1, n2)])
    if quasi_static:
        # end-of-frame force and penetration from the SAME frame: the frame the
        # contact is deepest in.  (final_substep_* holds one value per pair --
        # the last frame's -- which is not an operating point.)
        for (fi, key), (ff, pp) in ev["frame_pair_force_pen"].items():
            other = key[1] if key[0] == "object_geom" else key[0]
            if other == geom and pp > pen:
                pen, fn = pp, ff
    else:
        for (fi, key), p in ev["per_frame_pen_m"].items():
            other = key[1] if key[0] == "object_geom" else key[0]
            if other == geom:
                pen = max(pen, p)
    return fn, ft, pen


def force_pen_curve(ev, geom):
    """The (end-of-frame normal force, end-of-frame penetration) samples of one
    geom against the box, one per control frame, in force order.

    A geom's quasi-static contact force in MuJoCo is a function of its penetration
    alone (solref/solimp fix it; the velocity term is what the end-of-frame
    sampling removes), so this curve is the geom's contact law sampled by the clip
    and it can be read at 10 N without a hand-built 10 N press."""
    rows = []
    for (fi, (n1, n2)), (fn, pen) in ev["frame_pair_force_pen"].items():
        other = n2 if n1 == "object_geom" else n1
        if other == geom:
            rows.append((fn, pen, fi))
    rows.sort()
    return [{"force_N": r[0], "pen_m": r[1], "frame": r[2]} for r in rows]


def pen_at_force(curve, target=TARGET_N):
    """Penetration at ``target`` N, read off the measured curve.

    Interpolated between the two samples that bracket the load; when the curve
    stops short of it, linearly extrapolated from its top two samples, and the
    result says which it did."""
    pts = [(c["force_N"], c["pen_m"]) for c in curve if c["force_N"] > 0.0]
    if len(pts) < 2:
        return {"pen_m": None, "method": "curve too short",
                "max_force_N": pts[-1][0] if pts else 0.0}
    for (f0, p0), (f1, p1) in zip(pts, pts[1:]):
        if f0 <= target <= f1:
            w = (target - f0) / (f1 - f0)
            return {"pen_m": p0 + w * (p1 - p0), "method": "interpolated",
                    "bracket_N": [f0, f1], "max_force_N": pts[-1][0]}
    (f0, p0), (f1, p1) = pts[-2], pts[-1]
    w = (target - f1) / (f1 - f0)
    return {"pen_m": p1 + w * (p1 - p0), "method": "extrapolated",
            "extrapolated_from_N": [f0, f1], "max_force_N": f1}


def matched_operating_point(ev, geom):
    """The deepest end-of-frame contact of one geom against the box, with the force
    read from that SAME frame.  Force and penetration taken from different frames
    is not an operating point - across a clip the two walk all over each other -
    so the force-penetration law is only meaningful frame by frame."""
    best = (0.0, 0.0, -1)
    for (fi, (n1, n2)), (fn, pen) in ev["frame_pair_force_pen"].items():
        other = n2 if n1 == "object_geom" else n1
        if other == geom and pen > best[1]:
            best = (fn, pen, fi)
    return {"force_N": best[0], "pen_m": best[1], "frame": best[2]}


def pen_at_force_matched(points, target=TARGET_N):
    """Penetration at ``target`` N on a matched (force, penetration) sweep."""
    pts = sorted((p["force_N"], p["pen_m"]) for p in points if p["force_N"] > 0.0)
    if len(pts) < 2:
        return {"pen_m": None, "method": "sweep too short",
                "max_force_N": pts[-1][0] if pts else 0.0}
    for (f0, p0), (f1, p1) in zip(pts, pts[1:]):
        if f0 <= target <= f1:
            w = (target - f0) / (f1 - f0)
            return {"pen_m": p0 + w * (p1 - p0), "method": "interpolated",
                    "bracket_N": [f0, f1], "max_force_N": pts[-1][0]}
    (f0, p0), (f1, p1) = pts[-2], pts[-1]
    w = (target - f1) / (f1 - f0)
    return {"pen_m": p1 + w * (p1 - p0), "method": "extrapolated",
            "extrapolated_from_N": [f0, f1], "max_force_N": f1}


def ramp_lambda(model_factory, qpos, anchor, geom, log):
    """Amplify the clip's finger command until ``geom`` carries 10 N on the box.

    Two force metrics are carried at every sample, because they answer different
    questions: the peak normal force inside a control frame (what a fast landing
    does) and the end-of-frame normal force (what the finger holds).  The search
    bisects on the peak metric and reports BOTH ends of the final bracket, so the
    10 N answer is bounded from below and above rather than interpolated.
    """
    def at(lam):
        a = dict(anchor)
        a["finger_cmd"] = np.clip(np.asarray(anchor["finger_cmd"]) * lam,
                                  MODEL.actuator_ctrlrange[ACT_FINGER, 0],
                                  MODEL.actuator_ctrlrange[ACT_FINGER, 1])
        model, data = model_factory()
        _rep, ev = run_replay(model, data, qpos, a)
        peak = geom_box_force_and_pen(ev, geom, quasi_static=False)
        qs = geom_box_force_and_pen(ev, geom, quasi_static=True)
        op = matched_operating_point(ev, geom)
        log.append({"lambda": lam, "peak_force_N": peak[0], "peak_pen_m": peak[2],
                    "qs_force_N": qs[0], "qs_pen_m": qs[2],
                    "matched_force_N": op["force_N"], "matched_pen_m": op["pen_m"],
                    "matched_frame": op["frame"]})
        return peak, qs

    lo, hi = None, None
    lam = LAMBDA_LO
    while lam <= LAMBDA_HI + 1e-9:
        peak, qs = at(round(lam, 4))
        if peak[0] >= TARGET_N:
            hi = round(lam, 4)
            if lo is None:
                return _bracket_result(geom, None, round(lam, 4), log)
            break
        lo = round(lam, 4)
        lam += LAMBDA_STEP
    if hi is None:
        last = log[-1]
        return {"geom": geom, "reached_10N": False,
                "peak_force_at_lambda_max_N": last["peak_force_N"],
                "peak_pen_at_lambda_max_m": last["peak_pen_m"],
                "qs_force_at_lambda_max_N": last["qs_force_N"],
                "qs_pen_at_lambda_max_m": last["qs_pen_m"],
                "lambda_max": LAMBDA_HI, "trace": log}
    for _ in range(N_BISECT):
        mid = round(0.5 * (lo + hi), 4)
        peak, qs = at(mid)
        if peak[0] >= TARGET_N:
            hi = mid
        else:
            lo = mid
    at(hi)
    return _bracket_result(geom, lo, hi, log)


def _bracket_result(geom, lo, hi, log):
    lo_row = (None if lo is None else
              next(r for r in log if abs(r["lambda"] - lo) < 1e-9))
    hi_row = (None if hi is None else
              next(r for r in log if abs(r["lambda"] - hi) < 1e-9))
    out = {"geom": geom, "reached_10N": True, "lambda_lo": lo, "lambda_hi": hi}
    if lo_row:
        out.update({"peak_force_below_N": lo_row["peak_force_N"],
                    "peak_pen_below_m": lo_row["peak_pen_m"],
                    "qs_force_below_N": lo_row["qs_force_N"],
                    "qs_pen_below_m": lo_row["qs_pen_m"]})
    if hi_row:
        out.update({"peak_force_above_N": hi_row["peak_force_N"],
                    "peak_pen_above_m": hi_row["peak_pen_m"],
                    "qs_force_above_N": hi_row["qs_force_N"],
                    "qs_pen_above_m": hi_row["qs_pen_m"]})
    out["matched_operating_points"] = [
        {"lambda": r["lambda"], "force_N": r["matched_force_N"],
         "pen_m": r["matched_pen_m"], "frame": r["matched_frame"]}
        for r in log if r["matched_pen_m"] > 0.0]
    out["pen_at_10N_matched_sweep"] = pen_at_force_matched(
        out["matched_operating_points"])
    out["trace"] = log
    return out


# --------------------------------------------------------------------------- main
MODEL = None
ACT_OF = {}
ACT_FINGER = None
REF_QPOS = None
REF_QVEL = None
REF_WRIST = None
REF_OBJ = []


def main():
    global MODEL, ACT_FINGER
    t0 = time.time()
    payload = anchor_and_qpos()
    qpos, anchor = payload["qpos"], payload["anchor"]
    MODEL = rs.load_scene()[1]
    ACT_OF.update({mujoco.mj_id2name(MODEL, mujoco.mjtObj.mjOBJ_ACTUATOR, a): a
                   for a in range(MODEL.nu)})
    ACT_FINGER = np.array([ACT_OF[f"{p}a{k}"] for p in FINGERS for k in range(4)])
    print(f"anchor ready ({time.time() - t0:.1f}s)", flush=True)

    out = {"probe": "w35e_proximal", "method": __doc__,
           "target_normal_load_N": TARGET_N,
           "variants": {}, "candidate_geoms": ALL_CANDIDATES}

    for name in (sys.argv[1:] or list(VARIANTS)):
        geoms = VARIANTS[name]
        t = time.time()
        model, data = load_variant(variant_text(geoms))
        rep, ev = run_replay(model, data, qpos, anchor)
        boxev = box_contact_evidence(ev)
        global REF_QPOS, REF_WRIST
        if REF_QPOS is None:
            REF_QPOS = np.asarray(rep["qpos_history"]).copy()
            REF_QVEL = ev["frame_qvel"].copy()
            REF_WRIST = ev["frame_ctrl"].copy()
            REF_OBJ.append(np.asarray(rep["object_world"], dtype=float))
        rec = {
            "pairs_added": geoms,
            "npair": int(model.npair),
            "smoke5_max_penetration_m": rep["max_penetration_m"],
            "smoke5_contact_duty": rep["contact_duty"],
            "box_contact": boxev,
            "gate_metric_end_of_frame": gate_metric(ev),
            "force_penetration_curve_at_10N": {
                g: pen_at_force(force_pen_curve(ev, g))
                for g in ALL_CANDIDATES
                if g in {k[0] if k[1] == "object_geom" else k[1]
                         for k in ev["per_frame_force_N"]}
            },
            "pair_peak_normal_force_N": {f"{a}|{b}": v
                                         for (a, b), v in sorted(REC.pair_force_n.items())},
            "pair_peak_total_force_N": {f"{a}|{b}": v
                                        for (a, b), v in sorted(REC.pair_force_total.items())},
            "pair_peak_penetration_m": {f"{a}|{b}": v
                                        for (a, b), v in sorted(REC.pair_pen.items())},
            "servo_windup": servo_windup(ev),
            "joint_limit_violations": limit_violations(rep["qpos_history"]),
            "palm_deviation_max_m": rep["execution_deviation_m"]["max_m"],
            "palm_deviation_within_cap": rep["execution_deviation_m"]["within_cap"],
            "actuator_force_saturation_total": rep["force_saturation_total"],
        }
        out["variants"][name] = rec
        for g, c in rec["force_penetration_curve_at_10N"].items():
            if c["pen_m"] is not None:
                print(f"    10N {g}: {(c['pen_m']) * 1e3:.4f} mm ({c['method']}, "
                      f"curve max {c['max_force_N']:.2f} N)")
        print(f"{name}: pen={rec['smoke5_max_penetration_m'] * 1e3:.4f} mm "
              f"deepest={boxev['deepest_geom']} {boxev['deepest_m'] * 1e3:.4f} mm "
              f"viol={list(rec['joint_limit_violations'])} "
              f"palm_dev={rec['palm_deviation_max_m'] * 1e3:.2f} mm "
              f"({time.time() - t:.1f}s)", flush=True)

    # cross-pair influence: peak force of every pair, variant vs V0
    ref = out["variants"]["V0_committed"]["pair_peak_normal_force_N"]
    for name, rec in out["variants"].items():
        cur = rec["pair_peak_normal_force_N"]
        delta = {}
        for k in sorted(set(ref) | set(cur)):
            a, b = ref.get(k), cur.get(k)
            if a is None or b is None:
                delta[k] = {"before_N": a, "after_N": b}
            else:
                delta[k] = {"before_N": a, "after_N": b, "delta_N": b - a,
                            "bit_identical": bool(a == b)}
        rec["cross_pair_peak_force_delta"] = delta

    # ---- per-geom 10 N ramp, for the candidates that can reach 10 N
    out["ten_newton_ramp"] = {}
    reachable = sorted(
        g for g in ALL_CANDIDATES
        if max((v for k, v in ref.items() if g in k.split("|")), default=0.0) > 0.5)
    out["geoms_ramped"] = reachable
    for vname in VARIANTS:
        factory = lambda g=VARIANTS[vname]: load_variant(variant_text(g))
        out["ten_newton_ramp"][vname] = {}
        for g in reachable:
            r = ramp_lambda(factory, qpos, anchor, g, [])
            out["ten_newton_ramp"][vname][g] = r
            if r["reached_10N"]:
                print(f"  ramp {vname} {g}: peak {r.get('peak_force_below_N', 0):.2f}"
                      f"-{r.get('peak_force_above_N', 0):.2f} N -> pen "
                      f"{(r.get('peak_pen_below_m') or 0) * 1e3:.4f}-"
                      f"{(r.get('peak_pen_above_m') or 0) * 1e3:.4f} mm | hold "
                      f"{r.get('qs_force_above_N', 0):.2f} N -> "
                      f"{(r.get('qs_pen_above_m') or 0) * 1e3:.4f} mm", flush=True)
            else:
                print(f"  ramp {vname} {g}: 10 N unreachable, peak "
                      f"{r['peak_force_at_lambda_max_N']:.2f} N -> "
                      f"{r['peak_pen_at_lambda_max_m'] * 1e3:.4f} mm", flush=True)

    dest = REPO / "runs/w35/w35e_proximal.json"
    dest.write_text(json.dumps(out, indent=2, default=float) + "\n")
    print(f"wrote {dest} ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
