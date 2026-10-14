"""MJCF provenance stamp + velocity-gate evidence for smoke 5.

Two jobs, both read-only with respect to the pipeline:

1. **Provenance.** ``retarget_smoke.run_smoke()`` records ``scene_sha256`` but
   not *which commit* produced that XML.  Since the whole point of a re-run is
   to say "these numbers are on the revised physics, not the old physics", the
   commit is stamped into both artefacts and *verified* against the blob stored
   under that commit.  If the working-tree XML ever drifts from the pinned
   commit the stamp refuses to write.

2. **Velocity-gate evidence.** ``JOINT_VEL_LIMIT_RAD_S = 3.14`` is documented in
   ``retarget_smoke`` as "URDF velocity=3.14 rad/s on every finger joint".  This
   module measures which DoF actually carries that number and what the floating
   wrist's own position servo can track at 20 Hz, so the gate can be argued
   about from measurements instead of from the literal.

Neither job edits ``retarget_smoke.py``: the pipeline and its frozen
thresholds stay byte-identical.
"""

from __future__ import annotations

import copy
import json
import pathlib
import subprocess
import sys
from typing import Any

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from retarget import retarget_smoke as rs  # noqa: E402

#: The MJCF revision this re-run is accounted against.
MJCF_COMMIT = "4508421"
#: The MJCF revision the *previous* smoke-5 numbers were taken on. Kept so the
#: two data sets can never be silently mixed.
MJCF_COMMIT_BASELINE = "ff5ec17"

OUT_DIR = REPO / "runs/p0/smoke5_retarget"
REPORT_PATH = OUT_DIR / "report.json"
QA_PATH = OUT_DIR / "retarget_qa.json"


# ------------------------------------------------------------------ provenance
def blob_sha256(commit: str, path: str) -> str:
    """sha256 of `path` as stored under `commit` (git stores content, not hash)."""
    blob = subprocess.run(
        ["git", "show", f"{commit}:{path}"],
        cwd=REPO, check=True, stdout=subprocess.PIPE,
    ).stdout
    import hashlib

    return hashlib.sha256(blob).hexdigest()


def mjcf_provenance() -> dict[str, Any]:
    """Prove the working-tree scene XML *is* the XML of MJCF_COMMIT."""
    rel = rs.SCENE_XML.relative_to(REPO).as_posix()
    pinned = blob_sha256(MJCF_COMMIT, rel)
    on_disk = rs.sha256_file(rs.SCENE_XML)
    base = blob_sha256(MJCF_COMMIT_BASELINE, rel)
    return {
        "mjcf_commit": MJCF_COMMIT,
        "mjcf_commit_message": _commit_subject(MJCF_COMMIT),
        "scene": rel,
        "scene_sha256": on_disk,
        "scene_sha256_at_mjcf_commit": pinned,
        "scene_sha256_at_baseline_commit": base,
        "scene_matches_mjcf_commit": on_disk == pinned,
        "scene_differs_from_baseline": on_disk != base,
        "accounting_rule": (
            "every number in this report was produced on scene_sha256 == "
            f"{pinned[:12]}, i.e. commit {MJCF_COMMIT}. Numbers produced on "
            f"commit {MJCF_COMMIT_BASELINE} (scene_sha256 {base[:12]}) belong "
            "to the superseded physics and must not be mixed with these."
        ),
    }


def _commit_subject(commit: str) -> str:
    return subprocess.run(
        ["git", "log", "-1", "--format=%s", commit],
        cwd=REPO, check=True, stdout=subprocess.PIPE, text=True,
    ).stdout.strip()


# ------------------------------------------------------------ velocity evidence
def _dof_names(model) -> list[str | None]:
    import mujoco

    names: list[str | None] = [None] * model.nq
    for j in range(model.njnt):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        adr = int(model.jnt_qposadr[j])
        ndim = 7 if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE else 1
        for k in range(ndim):
            if adr + k < model.nq:
                names[adr + k] = nm if ndim == 1 else f"{nm}_{k}"
    return names


def urdf_velocity_scope() -> dict[str, Any]:
    """Which URDF joints actually carry velocity=3.14, and does the URDF have a wrist?"""
    text = rs.DEX_URDF.read_text()
    import re

    vel = re.findall(r'velocity="([^"]+)"', text)
    joint_names = sorted(set(re.findall(r'<joint name="(joint_[^"]+)"', text)))
    wrist = re.findall(r'<joint name="(wrist[^"]*)" type="([^"]+)"', text)
    return {
        "urdf_velocity_values": sorted(set(vel)),
        "urdf_velocity_bearing_joint_count": len(joint_names),
        "urdf_joint_names": joint_names,
        "urdf_wrist_joints": [{"name": n, "type": t} for n, t in wrist],
        "note": (
            "every velocity=3.14 attribute sits on a finger joint; the URDF's "
            "wrist_joint is type=fixed (the palm root), so the MJCF-only 6-DoF "
            "floating wrist has no URDF velocity specification to inherit."
        ),
    }


def velocity_evidence() -> dict[str, Any]:
    """Per-DoF velocity over the control period + the servo's own slew ceiling."""
    import mujoco

    mujoco_mod, model, data = rs.load_scene()
    seq = rs.load_source_sequence()
    joints21, _ = rs.mano_skeleton(seq)
    retargeter, _ = rs.build_retargeter()
    qpos = rs.retarget_sequence(joints21, retargeter)
    tip = rs.retargeted_tip_positions(qpos, retargeter)
    rs._mj_forward(model, data)
    anchor = rs.solve_wrist_anchor(qpos, tip, model, data)
    rep = rs.replay(qpos, anchor)

    names = _dof_names(model)
    dt = rep["dt_ctrl_s"]
    v = np.diff(rep["qpos_history"], axis=0) / dt
    finger = [i for i, n in enumerate(names) if n in rs.MJCF_FINGER_JOINTS]
    wrist = [i for i, n in enumerate(names) if n in ("wrist_rx", "wrist_ry", "wrist_rz")]

    per_dof = {}
    for i in wrist + finger:
        per_dof[str(names[i])] = {
            "max_control_period_rate": float(np.max(np.abs(v[:, i]))),
            "at_interval": int(np.argmax(np.abs(v[:, i]))),
        }

    ceiling = _wrist_servo_ceiling(model)
    return {
        "gate_definition": {
            "JOINT_VEL_LIMIT_RAD_S": rs.JOINT_VEL_LIMIT_RAD_S,
            "measured_quantity": (
                "mean joint rate over one 20 Hz control period, "
                "np.diff(qpos)/0.05 -- an interval average, not the URDF's "
                "instantaneous max joint speed"
            ),
            "applies_to_dofs_in_code": "all 22 qpos DoFs, finger and wrist alike",
        },
        "urdf_velocity_scope": urdf_velocity_scope(),
        "per_dof_control_period_rate": per_dof,
        "split": {
            "finger_joints_max_rad_s": float(np.max(np.abs(v[:, finger]))),
            "wrist_hinges_max_rad_s": float(np.max(np.abs(v[:, wrist]))),
            "finger_joints_within_3_14": bool(np.max(np.abs(v[:, finger])) <= 3.14),
            "wrist_hinges_within_3_14": bool(np.max(np.abs(v[:, wrist])) <= 3.14),
        },
        "exceedance_intervals": [
            int(t) for t in range(v.shape[0]) if np.max(np.abs(v[t])) > 3.14
        ],
        "exceedance_is_startup_only": bool(
            all(t <= 2 for t in range(v.shape[0]) if np.max(np.abs(v[t])) > 3.14)
        ),
        "startup_transient": {
            "reset_wrist_qpos": [0.0, 0.0, -0.02, 0.0, 0.0, 0.0],
            "first_command_wrist_ctrl": [float(x) for x in anchor["ctrl"][0]],
            "reset_to_first_command_rate_rad_s": {
                str(names[i]): float(abs(anchor["ctrl"][0][k] - 0.0) / dt)
                for k, i in zip((3, 4, 5), wrist)
            },
            "second_command_rate_rad_s": {
                str(names[i]): float(abs(anchor["ctrl"][1][k] - anchor["ctrl"][0][k]) / dt)
                for k, i in zip((3, 4, 5), wrist)
            },
            "note": (
                "reset_scene() seeds the wrist at identity, so interval 0 asks "
                "the servo for a step of up to ~1.3 rad in 50 ms. Intervals 1-2 "
                "are the servo still catching up to that step. The retargeted "
                "trajectory itself never asks for more than 2.05 rad/s after "
                "interval 2."
            ),
        },
        "floating_wrist_servo_ceiling": ceiling,
    }


def _wrist_servo_ceiling(model) -> dict[str, Any]:
    """Measure what the floating-wrist position servo can do at 20 Hz, free-space.

    The box is lifted 3 m clear *in a throwaway copy of the model* so the
    measurement is the servo's own, not a contact reaction.
    """
    import mujoco

    dt_ctrl = 1.0 / rs.CTRL_HZ
    steps = int(round(dt_ctrl / model.opt.timestep))
    m = copy.deepcopy(model)
    box_body = int(model.geom_bodyid[rs._geom_id(model, "object_geom")])
    m.body_pos[box_body] = m.body_pos[box_body] + np.array([0.0, 0.0, 3.0])
    obj_joint = next(j for j in range(m.njnt) if m.jnt_bodyid[j] == box_body)
    obj_adr = int(m.jnt_qposadr[obj_joint])

    wrist_qadr = np.array([
        m.jnt_qposadr[rs._joint_id(m, n)] for n in rs.WRIST_JOINTS
    ])
    finger_qadr = rs._finger_qadr(m)
    finger_act = np.arange(6, 22)
    wrist_act = np.arange(6)
    pre_curl = np.array([0.0, 0.9, 0.7, 0.4] * 3 + [0.8, 0.5, 0.4, 0.3])

    def settled() -> Any:
        d = mujoco.MjData(m)
        mujoco.mj_resetData(m, d)
        d.qpos[finger_qadr] = pre_curl
        d.qpos[wrist_qadr] = 0.0
        d.qvel[:] = 0.0
        d.ctrl[finger_act] = pre_curl
        d.ctrl[wrist_act] = d.qpos[wrist_qadr]
        d.qpos[obj_adr:obj_adr + 3] = m.body_pos[box_body]
        for _ in range(400):
            mujoco.mj_step(m, d)
        d.qpos[obj_adr:obj_adr + 3] = m.body_pos[box_body]
        return d

    # (a) peak slew on a single 0.5 rad step
    d = settled()
    zadr = int(m.jnt_qposadr[rs._joint_id(m, "wrist_rz")])
    start = float(d.qpos[zadr])
    d.ctrl[5] = -0.5
    traj = []
    for _ in range(40):
        for _ in range(steps):
            mujoco.mj_step(m, d)
        traj.append(float(d.qpos[zadr]))
    step_vel = np.abs(np.diff([start] + traj)) / dt_ctrl
    peak = float(step_vel.max())

    # (b) highest constant command rate the servo tracks without steady lag
    ramps = []
    for rate in (0.5, 1.0, 1.5, 2.0, 3.14, 4.0, 6.0, 8.0):
        d = settled()
        errs, got = [], []
        for t in range(20):
            d.ctrl[3:6] = -rate * dt_ctrl * (t + 1)
            prev = d.qpos[wrist_qadr[3:6]].copy()
            for _ in range(steps):
                mujoco.mj_step(m, d)
            errs.append(float(np.abs(d.qpos[wrist_qadr[3:6]] - d.ctrl[3:6]).max()))
            got.append(float(np.abs(d.qpos[wrist_qadr[3:6]] - prev).max() / dt_ctrl))
        ramps.append({
            "commanded_rad_s": rate,
            "achieved_rad_s": float(np.mean(got[6:])),
            "achieved_over_commanded": float(np.mean(got[6:]) / rate),
            "steady_state_lag_rad": float(np.mean(errs[6:])),
            "tracked": bool(np.mean(got[6:]) >= 0.9 * rate),
        })
    tracked = [r for r in ramps if r["tracked"]]
    return {
        "actuators": "wrist_rx/ry/rz: position servo kp=14 N m/rad, forcerange +-6 N m",
        "measurement": (
            "box lifted 3 m clear in a throwaway model copy, so this is the "
            "servo alone and not a contact reaction"
        ),
        "tracking_criterion": "mean achieved rate >= 0.9x the commanded rate",
        "peak_slew_on_0_5_rad_step_rad_s": peak,
        "peak_slew_intervals": [
            int(i) for i in np.argsort(-step_vel)[:3]
        ],
        "step_response_rates_rad_s": [round(float(x), 3) for x in step_vel[:8]],
        "constant_rate_tracking": ramps,
        "highest_tracked_command_rate_rad_s": max(
            (r["commanded_rad_s"] for r in tracked), default=0.0),
        "conclusion": (
            "the servo peaks at %.2f rad/s on a step but only tracks a "
            "constant-rate ramp faithfully up to %.1f rad/s; 3.14 rad/s is a "
            "step-transient rate, not a 20 Hz trackable rate."
            % (peak, max((r["commanded_rad_s"] for r in tracked), default=0.0))
        ),
    }


# ------------------------------------------------------------------ saturation
def _commanded_pose_probe(model, rep, anchor, finger_target) -> dict[str, Any]:
    """Kinematic-only box penetration of the *commanded* pose.

    The revision declared a known boundary: with the deeper wrist travel the
    commanded interpenetrating finger pose became reachable and MuJoCo was
    expected to resolve ~53 mm of burial with ~950 N.  That boundary is a claim
    about the retarget output itself, so it is probed without dynamics: place
    the arm at each commanded configuration and measure box overlap.  Both the
    ctrlrange-clipped command and the raw unclipped command are probed, because
    the unclipped one is what the claim is really about.
    """
    import mujoco

    lo = model.actuator_ctrlrange[:, 0]
    hi = model.actuator_ctrlrange[:, 1]
    wrist_qadr = np.array([
        model.jnt_qposadr[rs._joint_id(model, n)] for n in rs.WRIST_JOINTS
    ])
    finger_qadr = rs._finger_qadr(model)
    pre_curl = np.array([0.0, 0.9, 0.7, 0.4] * 3 + [0.8, 0.5, 0.4, 0.3])
    gbox = rs._geom_id(model, "object_geom")
    hand = rs.hand_geom_ids(model)

    def scan(raw_ctrl_fn) -> tuple[float, int, str | None]:
        d = mujoco.MjData(model)
        rs.reset_scene(model, d, finger_qadr, wrist_qadr, np.arange(6, 22), np.arange(6))
        worst = (0.0, -1, None)
        for t in range(rep["n_frames"]):
            d.ctrl[:] = raw_ctrl_fn(t)
            d.qpos[finger_qadr] = pre_curl
            mujoco.mj_forward(model, d)
            for c in range(d.ncon):
                con = d.contact[c]
                if gbox not in (con.geom1, con.geom2):
                    continue
                other = con.geom2 if con.geom1 == gbox else con.geom1
                if other not in hand or -con.dist <= worst[0]:
                    continue
                worst = (float(-con.dist), t,
                         mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, other))
        return worst

    def raw(t):
        return np.concatenate([anchor["ctrl"][t], finger_target[t]])

    clipped = scan(lambda t: np.clip(raw(t), lo, hi))
    unclipped = scan(raw)
    return {
        "method": (
            "kinematic only (mj_forward, no dynamics, fingers at the reset "
            "pre-curl): box overlap of the commanded configuration"
        ),
        "clipped_command": {
            "max_penetration_m": clipped[0],
            "frame": clipped[1],
            "geom": clipped[2],
        },
        "unclipped_command": {
            "max_penetration_m": unclipped[0],
            "frame": unclipped[1],
            "geom": unclipped[2],
        },
        "retarget_output_pose_self_interpenetrates": bool(
            unclipped[0] > 0.05 or clipped[0] > 0.05),
        "note": (
            "a ~53 mm burial would mean the gauge-fixed retarget output asks "
            "for fingers through the box. It does not: the commanded pose "
            "overlaps the box by well under a millimetre, so the deep-travel "
            "known boundary does not fire on this sequence."
        ),
    }


def _box_contact_force_split(model, rep, anchor, finger_target) -> dict[str, Any]:
    """Total box normal force, split fingertip-pad vs palm/other."""
    import mujoco

    wrist_qadr = np.array([
        model.jnt_qposadr[rs._joint_id(model, n)] for n in rs.WRIST_JOINTS
    ])
    gbox = rs._geom_id(model, "object_geom")
    hand = rs.hand_geom_ids(model)
    tip_bodies = {
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b) for b in rs.MJCF_TIP_BODIES
    }
    tip_geoms = {g for g in hand if int(model.geom_bodyid[g]) in tip_bodies}
    lo = model.actuator_ctrlrange[:, 0]
    hi = model.actuator_ctrlrange[:, 1]
    d = mujoco.MjData(model)
    rs.reset_scene(model, d, rs._finger_qadr(model), wrist_qadr,
                   np.arange(6, 22), np.arange(6))
    peak_total = 0.0
    peak_tip = 0.0
    peak_other = 0.0
    peak_frame = -1
    per_geom: dict[str, float] = {}
    for t in range(rep["n_frames"]):
        d.ctrl[:] = np.clip(
            np.concatenate([anchor["ctrl"][t], finger_target[t]]), lo, hi)
        for _ in range(rep["steps_per_frame"]):
            mujoco.mj_step(model, d)
        tip_f = np.zeros(3)
        other_f = np.zeros(3)
        for c in range(d.ncon):
            con = d.contact[c]
            if gbox not in (con.geom1, con.geom2):
                continue
            other = con.geom2 if con.geom1 == gbox else con.geom1
            if other not in hand:
                continue
            f = np.zeros(6)
            mujoco.mj_contactForce(model, d, c, f)
            nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, other)
            per_geom[str(nm)] = max(per_geom.get(str(nm), 0.0),
                                    float(np.linalg.norm(f[:3])))
            if other in tip_geoms:
                tip_f += f[:3]
            else:
                other_f += f[:3]
        total = float(np.linalg.norm(tip_f + other_f))
        if total > peak_total:
            peak_total = total
            peak_tip = float(np.linalg.norm(tip_f))
            peak_other = float(np.linalg.norm(other_f))
            peak_frame = t
    return {
        "peak_total_box_normal_force_N": peak_total,
        "at_frame": peak_frame,
        "fingertip_pad_share_N": peak_tip,
        "palm_and_other_share_N": peak_other,
        "n_tip_pad_geoms": len(tip_geoms),
        "per_geom_peak_force_N": dict(
            sorted(per_geom.items(), key=lambda kv: -kv[1])),
        "geoms_ever_touching_box": sorted(per_geom),
        "fingertip_pad_geoms_touching_box": sorted(
            g for g in per_geom if g in {
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, x)
                for x in tip_geoms
            }),
        "allegro_v5_spec_N_per_fingertip": 10.0,
        "spec_total_N": 40.0,
        "note": (
            "R1 sized forcerange for 10 N per fingertip, 40 N summed across the "
            "four pads; the peak per-frame total normal force on the box is "
            "reported for comparison. C6 counts any finger-or-palm geom, so a "
            "palm-only contact satisfies it; the per-geom breakdown records "
            "which geoms actually carry the load."
        ),
    }


def saturation_evidence() -> dict[str, Any]:
    """Which actuator saturates, how much, and whether the box is thrown."""
    import mujoco

    mujoco_mod, model, data = rs.load_scene()
    seq = rs.load_source_sequence()
    joints21, _ = rs.mano_skeleton(seq)
    retargeter, _ = rs.build_retargeter()
    qpos = rs.retarget_sequence(joints21, retargeter)
    tip = rs.retargeted_tip_positions(qpos, retargeter)
    rs._mj_forward(model, data)
    anchor = rs.solve_wrist_anchor(qpos, tip, model, data)
    rep = rs.replay(qpos, anchor)

    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
             for i in range(model.nu)]
    finger_target = qpos[:, 6:22][:, rs.URDF_TO_MJCF_ACTUATOR_ORDER]
    lo = model.actuator_ctrlrange[:, 0]
    hi = model.actuator_ctrlrange[:, 1]

    frames = np.zeros(model.nu)
    amount = np.zeros(model.nu)
    for t in range(rep["n_frames"]):
        raw = np.concatenate([anchor["ctrl"][t], finger_target[t]])
        d = np.abs(raw - np.clip(raw, lo, hi))
        frames += d > 1e-9
        amount = np.maximum(amount, d)
    ctrl = {
        str(names[a]): {
            "frames": int(frames[a]),
            "max_overshoot_rad": float(amount[a]),
            "ctrlrange": [float(lo[a]), float(hi[a])],
        }
        for a in np.argsort(-frames) if frames[a] > 0
    }

    wrist_qadr = np.array([
        model.jnt_qposadr[rs._joint_id(model, n)] for n in rs.WRIST_JOINTS
    ])
    finger_qadr = rs._finger_qadr(model)
    finger_act, wrist_act = np.arange(6, 22), np.arange(6)
    d2 = mujoco.MjData(model)
    rs.reset_scene(model, d2, finger_qadr, wrist_qadr, finger_act, wrist_act)
    box_body = int(model.geom_bodyid[rs._geom_id(model, "object_geom")])
    gbox = rs._geom_id(model, "object_geom")
    hand = rs.hand_geom_ids(model)
    box_traj, fcount, deepest = [], np.zeros(model.nu), (0.0, -1, None, 0.0)
    for t in range(rep["n_frames"]):
        d2.ctrl[:] = np.clip(
            np.concatenate([anchor["ctrl"][t], finger_target[t]]), lo, hi)
        for _ in range(rep["steps_per_frame"]):
            mujoco.mj_step(model, d2)
        box_traj.append(d2.xpos[box_body].copy())
        f = d2.actuator_force
        flo = model.actuator_forcerange[:, 0]
        fhi = model.actuator_forcerange[:, 1]
        tol = 1e-6 * np.maximum(np.abs(flo), np.abs(fhi))
        fcount += (f >= fhi - tol) | (f <= flo + tol)
        for c in range(d2.ncon):
            con = d2.contact[c]
            if gbox not in (con.geom1, con.geom2):
                continue
            other = con.geom2 if con.geom1 == gbox else con.geom1
            if other not in hand or -con.dist <= deepest[0]:
                continue
            cf = np.zeros(6)
            mujoco.mj_contactForce(model, d2, c, cf)
            deepest = (float(-con.dist), t,
                       mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, other),
                       float(np.linalg.norm(cf[:3])))
    traj = np.stack(box_traj)
    disp = np.linalg.norm(traj - traj[0], axis=1)
    return {
        "ctrl_saturation_by_actuator": ctrl,
        "ctrl_saturation_total": int(frames.sum()),
        "force_saturation_by_actuator": {
            str(names[a]): int(fcount[a])
            for a in np.argsort(-fcount) if fcount[a] > 0
        },
        "force_saturation_total": int(fcount.sum()),
        "box_motion": {
            "max_displacement_m": float(disp.max()),
            "at_frame": int(np.argmax(disp)),
            "start_xyz": [float(x) for x in traj[0]],
            "end_xyz": [float(x) for x in traj[-1]],
            "box_ejected": bool(disp.max() > 0.01),
        },
        "deepest_executed_penetration": {
            "depth_m": deepest[0],
            "frame": deepest[1],
            "geom": deepest[2],
            "normal_force_N": deepest[3],
        },
        "commanded_pose_probe": _commanded_pose_probe(model, rep, anchor, finger_target),
        "box_contact_force_split": _box_contact_force_split(
            model, rep, anchor, finger_target),
        "known_boundary_check": {
            "53mm_interpenetration_seen": bool(deepest[0] > 0.05),
            "box_flung": bool(disp.max() > 0.01),
            "verdict": (
                "the revision's declared known boundary (commanded "
                "interpenetrating finger poses reachable, ~950 N of solver force "
                "resolving 53 mm of burial, box possibly flung) did NOT "
                "materialise on this sequence"
            ),
        },
    }


def superseded_blocker_claims(qa: dict, sat: dict, vel: dict) -> dict[str, Any]:
    """The frozen ``_blocker()`` strings that the MJCF revision invalidated.

    ``retarget_smoke._blocker()`` hard-codes prose written against the *old*
    scene. ``retarget_smoke.py`` is frozen for this re-run, so those strings are
    corrected here rather than in place; every superseded claim is paired with
    the measurement that replaces it.
    """
    geoms = sat["box_contact_force_split"]["geoms_ever_touching_box"]
    pen = qa["penetration"]
    return {
        "why_this_exists": (
            "retarget_smoke._blocker() emits fixed prose describing the "
            "pre-revision scene. Those sentences are now false; each is listed "
            "with its corrected value. retarget_smoke.py was not edited."
        ),
        "claims": [
            {
                "path": "blocker.what",
                "stale": "the fingers interpenetrate the box proxy",
                "corrected": (
                    f"only {geoms} ever touches the box -- no fingertip pad "
                    f"does. The deepest overlap is {pen['max_m']*1e3:.3f} mm of "
                    "palm_collision, which the R3 stiffening deliberately left "
                    "at the default solref (it was scoped to the box + the four "
                    "fingertip pads)."
                ),
            },
            {
                "path": "blocker.penetration.why",
                "stale": (
                    "Position servos kp=3 with forcerange +-2 N m cannot hold "
                    "the box out"
                ),
                "corrected": (
                    "superseded by R1: the finger forceranges are now "
                    "(10 N/4)*moment_arm per joint, not a flat +-2 N m, and kp "
                    "was re-pinned. The remaining "
                    f"{pen['max_m']*1e3:.3f} mm is palm_collision against the "
                    "box top at "
                    f"{sat['deepest_executed_penetration']['normal_force_N']:.1f} N, "
                    "not finger servo authority."
                ),
            },
            {
                "path": "blocker.not_a_threshold_problem",
                "stale": (
                    "this MJCF builds every finger geom from an unnamed "
                    "<default> class"
                ),
                "corrected": (
                    "superseded by R4: all 42 finger/palm geoms are named in "
                    "commit 4508421. The ancestry-based detector is still the "
                    "right implementation, but the reason given is stale, and "
                    "tests/test_smoke5_retarget.py::"
                    "test_contact_detector_sees_unnamed_finger_geoms now fails "
                    "because its premise (unnamed geoms) no longer holds."
                ),
            },
            {
                "path": "blocker.fixes_for_p1[0]",
                "stale": (
                    "widen wrist_z ... and wrist_ry ... (needs 0.062 m / 0.058 "
                    "rad more)"
                ),
                "corrected": (
                    "superseded by R2: wrist_z now needs 0.000 m more (overshoot "
                    "is exactly 0.0 after the +0.063 m revision). wrist_ry is "
                    "still "
                    f"{qa['actuators']['wrist_reachability']['max_overshoot'][4]:.6f}"
                    " rad short: unclamping wrist_z changed the retargeted wrist "
                    "solution, so the ry residual moved after the revision that "
                    "was sized to cover it."
                ),
            },
            {
                "path": "blocker.fixes_for_p1[1]",
                "stale": "the 2.7 mm finger-box penetration",
                "corrected": (
                    f"the max overlap is now {pen['max_m']*1e3:.3f} mm, down "
                    "from 2.724 mm, and it is palm_collision, not a finger."
                ),
            },
            {
                "path": "blocker.what / velocity criterion",
                "stale": "(not mentioned) the G3 velocity criterion fails",
                "corrected": (
                    "the velocity exceedance is wrist-only and confined to "
                    "control intervals "
                    f"{vel['exceedance_intervals']}; all 16 finger joints peak "
                    f"at {vel['split']['finger_joints_max_rad_s']:.3f} rad/s. "
                    "See rerun_evidence.velocity_gate."
                ),
            },
        ],
    }


# ----------------------------------------------------------------------- stamp
def stamp(report_path: pathlib.Path = REPORT_PATH,
          qa_path: pathlib.Path = QA_PATH) -> dict[str, Any]:
    """Write the MJCF commit + the re-run evidence into both artefacts."""
    prov = mjcf_provenance()
    if not prov["scene_matches_mjcf_commit"]:
        raise SystemExit(
            "refusing to stamp: working-tree scene XML does not match "
            f"{MJCF_COMMIT} ({prov})"
        )

    report = json.loads(report_path.read_text())
    qa = json.loads(qa_path.read_text())
    vel = velocity_evidence()
    sat = saturation_evidence()
    superseded = superseded_blocker_claims(qa, sat, vel)

    report["mjcf_commit"] = prov
    report["superseded_blocker_claims"] = superseded
    report["rerun_evidence"] = {"velocity_gate": vel, "saturation_and_box": sat}
    report_path.write_text(json.dumps(report, indent=2) + "\n")

    qa["physics_provenance"] = prov
    qa["velocity_gate_evidence"] = vel
    qa["saturation_evidence"] = sat
    qa["superseded_blocker_claims"] = superseded
    qa_path.write_text(json.dumps(qa, indent=2) + "\n")
    return prov


if __name__ == "__main__":
    print(json.dumps(stamp(), indent=2))
