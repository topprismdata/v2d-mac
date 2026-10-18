"""r14: UR5e base-pose fixture search — fixed and running.

r14 bug history (the "written but never converged" version): it drove a
hand-rolled ws() = base + Rm @ site_xpos against the VANILLA menagerie
URDF.  Two fatal defects, both measured:

  1. wrong model: the production scene's embedded arm is a rebuilt tree
     whose FK is NOT a fixed transform of the vanilla FK (measured:
     world_scene = Rz(pi) @ p_vanilla + (0.470, 0, 1.000), 1.7 m off at
     q0) — every candidate inherited the mismatch and converged to
     nothing;
  2. no step control: raw GN steps of |dq| ~ 4 rad/iteration slammed the
     wrists into their +-2pi stops and stalled.

Fix: the search runs the PRODUCTION SCENE per candidate (composer-
equivalent edit of the ur5e_r/ur5e_l_base mount pos/quat, hands ride
under wrist_3) and the DLS is retarget_dual.ur5e_ik_placement's exact
loop (mj_kinematics + mj_comPos, jacBody on the palm body, 1e-4 DLS,
joint-range clip, 1 mm tolerance, 60 inner iterations, path warm start).
The evaluation path is the REGISTERED palm task (the pipeline's own
face_anchor + measured MediaPipe lateral assembly, 260 control frames),
both arms solved in the same candidate scene, fingers at the verified
terminal curl — the collision leg sees the real symmetric fixture.

Frozen acceptance (documented before the run, zero relaxation):
  C1 reach      : >= 98% path frames with final palm error < 1 mm
                  (the production solver tolerance)
  C2 jump       : max per-frame joint jump <= 60 deg, wrap-normalized
  C3 continuity : max |second difference| <= 0.5 rad/frame^2
                  (branch-flip detector; calibrated above the shipped
                  incumbent's measured 0.413 — its commands demonstrably
                  execute through the production rate projection)
  C4 collision  : zero contacts between either side's arm+hand geoms and
                  static support (table/yaw stops/floor/gantry) over the
                  swept path; hand-vs-object is gate2's business, not this
                  filter's

A 6D variant of the sweep (r12.9 solver, verified-grasp orientation,
backward warm-started from the approach end) was probed and ABANDONED:
it converges on ~0% of frames at every probed fixture — the registered
path + grasp orientation is plain-DLS-intractable here, which is the
documented r12.1 wall and the reason production ships position-only
swing + terminal IK.  The shipped C4 therefore sweeps the position-only
commands and reports whether the first support contact is an ARM-LINK
(fixture geometry, base-pose's own fault) or a HAND-GEOM (task
geometry: the path rides at constant z=0.76, ~6 cm over the slab, and
the terminal-curl hand hangs deeper — inherent under any base).

Run:  python3 tools/w6b/base_pose_search.py
Out:  runs/w6b/r14_base_pose_search.json + the printed candidate table.
"""
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np
import mujoco
from scipy.signal import medfilt

SCENE = REPO / "grounding/assets/allegro_floating_ur5e.xml"
OUT_JSON = REPO / "runs/w6b/r14_base_pose_search.json"

# frozen criteria (see module docstring)
REACH_FRAC_MIN = 0.98
REACH_TOL_M = 1e-3
JUMP_MAX_RAD = np.deg2rad(60.0)
# calibrated from the shipped incumbent's measured 0.413 rad/frame^2
# (see C3 docstring); catches IK branch flips, not MP-lateral jitter
CONTINUITY_MAX_RAD = 0.5
DLS_LAM = 1e-4
DLS_ITERS = 60

ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow",
              "wrist_1", "wrist_2", "wrist_3")
ORIENTATIONS = {  # the 3 mount orientations on the ceiling column
    "identity": "1 0 0 0",     # the shipped r12 mount (arm folds down)
    "rx180": "0 1 0 0",        # true hang-down mount
    "rz180": "0 0 0 1",        # arm yawed to face the centre
}
X_COL = 0.235
FINGER_ORDER = tuple(f"{p}a{k}" for p in ("ff", "mf", "rf", "th")
                     for k in range(4))


def candidate_scene(pos_y: float, pos_z: float, quat: str):
    """Composer-equivalent mount edit; returns a loaded MjModel."""
    tree = ET.parse(SCENE)
    root = tree.getroot()
    world = root.find("worldbody")
    for name, sx in (("ur5e_r_base", +1.0), ("ur5e_l_base", -1.0)):
        body = world.find(f".//body[@name='{name}']")
        body.set("pos", f"{sx * X_COL} {-sx * pos_y} {pos_z}")
        body.set("quat", quat)
    # same directory so the mesh/include paths still resolve
    tmp = SCENE.parent / "_r14_candidate.xml"
    tree.write(tmp)
    try:
        return mujoco.MjModel.from_xml_path(str(tmp))
    finally:
        tmp.unlink()


def registered_paths():
    """The pipeline's own palm task per side: face_anchor + measured
    MediaPipe lateral (lock-trimmed, k=5 median-filtered, resampled onto
    the control timeline), verbatim from retarget_chain_side's ur5e
    branch.  Returns {side: (T,3)}."""
    import os
    import time
    os.environ["W6B_SCENE"] = "ur5e"
    from retarget import retarget_dual as rd
    # the MP-lateral npz read is intermittent under sibling repo activity
    # (legacy 1-D fallback fires when the file read comes up empty) —
    # retry and fail loud instead of solving the wrong path
    streams_rec = wrist_lateral = None
    for attempt in range(5):
        _, streams_rec, _, wrist_lateral, _, _ = rd.load_streams()
        if np.asarray(wrist_lateral["right"]).ndim == 2:
            break
        time.sleep(1.0)
    else:
        raise RuntimeError(
            "load_streams returned the 1-D legacy lateral 5x — "
            "runs/w6b/wrist_landmarks.npz unreadable (sibling activity?)")
    n_ctrl = int(round(
        (streams_rec["right"]["n_frames_executed"] - 1)
        / rd.VIDEO_FPS * rd.CTRL_HZ)) + 1
    out = {}
    for side in ("right", "left"):
        face_sign = +1.0 if side == "right" else -1.0
        lat = np.asarray(wrist_lateral[side], float)
        if lat.shape[0] != n_ctrl:
            lat = rd.resample_qpos22(lat, n_ctrl)
        inf = np.abs(lat).max(axis=1)
        lock = int(np.argmax(inf <= 0.3)) if (inf <= 0.3).any() else 0
        if lock:
            lat = np.array(lat, float)
            lat[:lock] = lat[lock]
        if len(lat) >= 5:
            lat = np.stack([medfilt(lat[:, c], 5) for c in range(3)],
                           axis=1)
        anchor = np.array([face_sign * 0.11, 0.0, 0.76])
        path = anchor[None, :] + lat
        if side == "left":
            path[:, 1] += 0.027  # r12.7 Focus B left pad-band bias
        out[side] = path
    return out


def _arm_addrs(model, side: str):
    key = "r" if side == "right" else "l"
    jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                              f"ur5e_{key}_{n}_joint") for n in ARM_JOINTS]
    if any(j < 0 for j in jids):
        raise RuntimeError(f"{side}: arm joints missing in candidate scene")
    return (jids,
            [model.jnt_qposadr[j] for j in jids],
            [model.jnt_dofadr[j] for j in jids],
            model.jnt_range[jids][:, 0], model.jnt_range[jids][:, 1])


def _finger_addrs(model, side: str):
    sfx = "" if side == "right" else "_left"
    jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{n}{sfx}")
            for n in FINGER_ORDER]
    return [model.jnt_qposadr[j] for j in jids]


def solve_side(model, data, side: str, path: np.ndarray):
    """ur5e_ik_placement's exact DLS loop on the candidate scene."""
    palm = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                             "palm" + ("" if side == "right" else "_left"))
    if palm < 0:
        raise RuntimeError(f"{side}: palm body missing")
    jids, qadr, vjad, lo, hi = _arm_addrs(model, side)

    from retarget import retarget_dual as rd
    f_jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                                f"{n}{'' if side == 'right' else '_left'}")
              for n in FINGER_ORDER]
    curl = np.clip(np.asarray(rd.UR5E_TERMINAL_FINGERS[side], float),
                   model.jnt_range[f_jids][:, 0],
                   model.jnt_range[f_jids][:, 1])

    q = np.zeros(6)
    errs = np.empty(len(path))
    qs = np.empty((len(path), 6))
    for t, tgt in enumerate(path):
        for _ in range(DLS_ITERS):
            data.qpos[qadr] = q
            mujoco.mj_kinematics(model, data)
            mujoco.mj_comPos(model, data)
            e = tgt - data.xpos[palm]
            if np.linalg.norm(e) < REACH_TOL_M:
                break
            jacp = np.zeros((3, model.nv))
            mujoco.mj_jacBody(model, data, jacp, None, palm)
            J = jacp[:, vjad]
            dq = J.T @ np.linalg.solve(J @ J.T + DLS_LAM * np.eye(3), e)
            q = np.clip(q + dq, lo, hi)
        data.qpos[qadr] = q
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        errs[t] = np.linalg.norm(tgt - data.xpos[palm])
        qs[t] = q
    return errs, qs, curl


def wrap(dq):
    """Wrap-normalized joint delta to (-pi, pi]."""
    return (dq + np.pi) % (2 * np.pi) - np.pi


def _descendant_geoms(model, root_body: str):
    """Geom ids of every body under (and including) root_body."""
    rb = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, root_body)
    under = {rb}
    changed = True
    while changed:
        changed = False
        for b in range(model.nbody):
            if b not in under and int(model.body_parentid[b]) in under:
                under.add(b)
                changed = True
    return {g for g in range(model.ngeom)
            if int(model.geom_bodyid[g]) in under}


def collision_scan(model, data, sides_qs, curls):
    """(first_hit, arm_frames, any_frames) over the swept path.
    first_hit = (frame, moving_geom_id, support_geom_id) or None.
    arm_frames counts frames whose moving geom is an UNNAMED arm-link
    collision geom (the rebuilt ur5e trees leave link geoms unnamed;
    hand/finger geoms carry *_collision names) — the fixture-dependent
    signal."""
    support = {}
    for g in range(model.ngeom):
        n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        if n == "table_slab" or n == "floor" or n.startswith(
                ("yaw_stop", "gantry")):
            support[g] = n
    moving = set()
    for root in ("ur5e_r_base", "ur5e_l_base"):
        moving |= _descendant_geoms(model, root)
    f_qadr = {s: _finger_addrs(model, s) for s in sides_qs}
    n_frames = next(iter(sides_qs.values())).shape[0]
    first = None
    arm_frames = 0
    any_frames = 0
    for t in range(n_frames):
        for side, qs in sides_qs.items():
            _, qadr, _, _, _ = _arm_addrs(model, side)
            data.qpos[qadr] = qs[t]
            data.qpos[f_qadr[side]] = curls[side]
        mujoco.mj_forward(model, data)
        frame_hit = False
        arm_hit = False
        for i in range(data.ncon):
            c = data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            if (g1 in moving and g2 in support) or \
               (g1 in support and g2 in moving):
                gm = g1 if g1 in moving else g2
                gs = g2 if g1 in moving else g1
                if first is None:
                    first = (t, gm, gs)
                frame_hit = True
                if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM,
                                     gm) is None:
                    arm_hit = True
        any_frames += int(frame_hit)
        arm_frames += int(arm_hit)
    return first, arm_frames, any_frames


def evaluate(pos_y: float, pos_z: float, quat_name: str, paths):
    model = candidate_scene(pos_y, pos_z, ORIENTATIONS[quat_name])
    data = mujoco.MjData(model)
    out = {"mode": quat_name, "by": round(float(pos_y), 3),
           "bz": round(float(pos_z), 3)}
    try:
        per_side = {}
        sides_qs, curls = {}, {}
        for side in ("right", "left"):
            errs, qs, curl = solve_side(model, data, side, paths[side])
            jumps = np.abs(wrap(np.diff(qs, axis=0))).max(initial=0.0)
            cont = np.abs(np.diff(qs, n=2, axis=0)).max(initial=0.0)
            sides_qs[side] = qs
            curls[side] = curl
            per_side[side] = {
                "reach_frac": float(np.mean(errs < REACH_TOL_M)),
                "max_err_mm": float(errs.max() * 1000),
                "max_jump_deg": float(np.rad2deg(jumps)),
                "max_cont_rad": float(cont),
            }
        hit_pos, arm_frames, any_frames = collision_scan(
            model, data, sides_qs, curls)
        # split the first hit: ARM-LINK contact (unnamed ur5e link geom =
        # fixture geometry, the base pose's own fault) vs HAND-GEOM
        # contact (task geometry: the registered path rides at constant
        # z=0.76, ~6 cm over the slab, and the terminal-curl hand hangs
        # deeper — inherent to the task under ANY base pose)
        kind = "none"
        if hit_pos is not None:
            kind = ("arm_link"
                    if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM,
                                         int(hit_pos[1])) is None
                    else "hand_geom")
        out["per_side"] = per_side
        out["collision_first_hit"] = (None if hit_pos is None else
                                      {"frame": int(hit_pos[0]),
                                       "geoms": [
                                           mujoco.mj_id2name(
                                               model, mujoco.mjtObj.mjOBJ_GEOM,
                                               int(hit_pos[1])) or "arm_link",
                                           mujoco.mj_id2name(
                                               model, mujoco.mjtObj.mjOBJ_GEOM,
                                               int(hit_pos[2]))]})
        out["collision_kind"] = kind
        out["contact_frames"] = int(any_frames)
        out["arm_link_contact_frames"] = int(arm_frames)
        out["C1"] = all(v["reach_frac"] >= REACH_FRAC_MIN
                        for v in per_side.values())
        out["C2"] = all(v["max_jump_deg"] <= 60.0
                        for v in per_side.values())
        out["C3"] = all(v["max_cont_rad"] <= CONTINUITY_MAX_RAD
                        for v in per_side.values())
        out["C4"] = hit_pos is None
        out["pass"] = out["C1"] and out["C2"] and out["C3"] and out["C4"]
    except Exception as exc:  # a candidate that cannot even load/solve
        out["error"] = repr(exc)
        out["pass"] = False
    return out


def main() -> int:
    print("=== r14 base-pose fixture search (scene-faithful, fixed DLS) ===")
    paths = registered_paths()
    print(f"registered path: {paths['right'].shape[0]} control frames/side")
    results = []
    for quat_name in ORIENTATIONS:
        for by in np.arange(-0.20, 0.21, 0.05):
            for bz in np.arange(0.70, 1.01, 0.05):
                r = evaluate(by, bz, quat_name, paths)
                results.append(r)
                ps = r.get("per_side", {})
                print(f"{quat_name:9s} y={by:+.2f} z={bz:.2f} "
                      f"C1={r.get('C1')} C2={r.get('C2')} "
                      f"C3={r.get('C3')} C4={r.get('C4')} "
                      f"pass={r.get('pass')} "
                      f"err_mm={[round(v['max_err_mm'], 2) for v in ps.values()]} "
                      f"jump={[round(v['max_jump_deg'], 1) for v in ps.values()]} "
                      f"hit={r.get('collision_first_hit')}")
    passing = [r for r in results if r.get("pass")]
    print(f"total: {len(results)}, passing: {len(passing)}")
    for r in passing:
        print("PASS:", json.dumps(r))
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({
        "frozen_criteria": {
            "C1_reach": f">= {REACH_FRAC_MIN} frames < {REACH_TOL_M*1e3:.0f} mm",
            "C2_jump": "<= 60 deg wrap-normalized",
            "C3_continuity": f"<= {CONTINUITY_MAX_RAD} rad/frame^2",
            "C4_collision": "no arm/hand vs static support contact",
        },
        "grid": "y in [-0.20, 0.20] step 0.05 x z in [0.70, 1.00] "
                "step 0.05 x {identity, rx180, rz180}, x=+-0.235",
        "results": results,
        "passing": passing,
    }, indent=1))
    print(f"wrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
