"""Wrist-hinge servo transfer characteristics + a derived realised-rate tolerance.

Evidence only -- no gate threshold is changed here.  Same discipline as
``runs/w35/mfj1_forensics.py``.

mf_j1's tolerance was ``|forcerange| / kp``: the servo's torque ceiling maps,
through the actuator gain, onto a physical-side *position* windup.  The wrist
hinges fail their gate the same way -- a *command*-side rate cap (1.5 rad/s,
w1c) is used to judge a *realised* rate (w1b) -- so the isomorphic map for a
rate is: torque ceiling -> passive compliance -> rate.

Three mechanisms, all read off the MJCF with no free parameter:

  1. damping-limited terminal rate   v_term = |forcerange| / dof_damping
     (once the actuator saturates, the passive joint damping alone exceeds the
     whole drive torque above this rate, so qddot < 0 for any admissible ctrl)
  2. elastic release ceiling          v_el = |forcerange| / kp * sqrt(kp / I_eff)
     (the wind-up energy 1/2 kp e^2, e = forcerange/kp, is the only energy a
     saturated servo can hand to the mass; energy balance)
  3. the suggested a_max*tau_s form is evaluated and reported, including its
     vacuity, so the rejected candidate is on the record.

Measured here:
  * free-space step response peak rate vs step amplitude, per hinge
  * free-space constant-rate ramp tracking: lag, realised/commanded ratio,
    torque saturation fraction, and what kv=0 does to the lag law
  * the *same* warm-start ramp the w1/w1b gates run, once free-space (box
    lifted 3 m) and once on the real loaded clip

Run:  runs/.venv-retarget/bin/python runs/w35/wrist_rate_tol.py
"""
from __future__ import annotations

import copy
import json
import pathlib
import pickle
import sys

import numpy as np

REPO = pathlib.Path("/Users/guohongbin/v2d-mac")
sys.path.insert(0, str(REPO))

import mujoco  # noqa: E402

from retarget import retarget_smoke as rs  # noqa: E402

CACHE = REPO / "runs/w35/prox_probe_anchor.pkl"
HINGES = ("wrist_rx", "wrist_ry", "wrist_rz")
STEP_AMPLITUDES_RAD = (0.02, 0.05, 0.10, 0.20, 0.30, 0.45, 0.60, 0.90, 1.20)
RAMP_RATES_RAD_S = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0)
# Per-leg commanded travel. wrist_ry's ctrlrange is only +-1.2/-1.259 rad, so
# this is the radius every ramp leg is sized to stay inside.
TRAVEL_BUDGET_RAD = 1.24
RAMP_STEPS = 40
PRE_CURL = np.array([0.0, 0.9, 0.7, 0.4] * 3 + [0.8, 0.5, 0.4, 0.3])


# ----------------------------------------------------------------- model facts
def actuator_id(m, joint_name):
    """The position servo driving ``joint_name``.

    ``model.jnt_actuatorid`` is -1 on this MJCF -- the actuators resolve through
    their transmission, not by a direct joint binding -- so the id is resolved
    through ``actuator_trnid`` instead of trusted from the shortcut.
    """
    jid = rs._joint_id(m, joint_name)
    for a in range(m.nu):
        if m.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT \
                and int(m.actuator_trnid[a, 0]) == jid:
            return a
    raise KeyError(f"no position actuator drives {joint_name}")


def hinge_params(model, free_model) -> dict:
    """Per-hinge MJCF numbers the derivation may use.  Nothing fitted."""
    out = {}
    for name in HINGES:
        j = rs._joint_id(model, name)
        d = int(model.jnt_dofadr[j])
        a = actuator_id(model, name)
        out[name] = {
            "kp_Nm_per_rad": float(model.actuator_gainprm[a, 0]),
            "kv_Nm_s_per_rad": float(model.actuator_biasprm[a, 2]),
            "forcerange_Nm": float(abs(model.actuator_forcerange[a, 1])),
            "ctrlrange_rad": [float(v) for v in model.actuator_ctrlrange[a]],
            "jnt_range_rad": [float(v) for v in model.jnt_range[j]],
            "jnt_stiffness": float(model.jnt_stiffness[j]),
            # actuator kv is 0; this is the *joint* damping attribute, the only
            # velocity-proportional term in the dof's equation of motion.
            "dof_damping_Nm_s_per_rad": float(free_model.dof_damping[d]),
            "dof_armature_kgm2": float(free_model.dof_armature[d]),
            "dof_M0_at_qpos0_kgm2": float(free_model.dof_M0[d]),
            "dofadr": d,
            "qposadr": int(model.jnt_qposadr[j]),
            "actuatorid": a,
        }
    return out


def free_copy(model):
    """A throwaway copy with the box clear of the hand: the servo's own dynamics.

    Lifting ``body_pos`` alone is not enough -- the object hangs off a freejoint
    whose qpos *is* its world pose, so the body reference frame does not move it.
    The box is instead translated in its own qpos at run time (see
    ``gate_profile``) and its mass is set to ~0 here so it neither falls back
    onto the table during a 6 s measurement nor loads the hand through contact.
    Gravity on the hand is left alone: the wrist's static weight bias is part of
    the servo it has to overcome.
    """
    m = copy.deepcopy(model)
    box_body = int(model.geom_bodyid[rs._geom_id(model, "object_geom")])
    m.body_mass[box_body] = 1e-6
    return m


def lift_box(m, d):
    """Park the object 3 m above the table and zero its velocity."""
    box_body = int(m.geom_bodyid[rs._geom_id(m, "object_geom")])
    obj_joint = next(j for j in range(m.njnt) if m.jnt_bodyid[j] == box_body)
    obj_adr = int(m.jnt_qposadr[obj_joint])
    d.qpos[obj_adr + 2] += 3.0
    d.qvel[int(m.jnt_dofadr[obj_joint]):int(m.jnt_dofadr[obj_joint]) + 3] = 0.0
    mujoco.mj_forward(m, d)


def settled(m, lift=False):
    """The same initial state the gates measure from (reset_scene, box parked)."""
    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)
    d.qpos[rs._finger_qadr(m)] = PRE_CURL
    wrist_qadr = np.array([m.jnt_qposadr[rs._joint_id(m, n)] for n in HINGES])
    d.qpos[wrist_qadr] = 0.0
    all_wrist_qadr = np.array([m.jnt_qposadr[rs._joint_id(m, n)]
                               for n in rs.WRIST_JOINTS])
    d.qvel[:] = 0.0
    d.ctrl[6:22] = PRE_CURL
    d.ctrl[0:6] = d.qpos[all_wrist_qadr]
    for _ in range(400):
        mujoco.mj_step(m, d)
    if lift:
        lift_box(m, d)
    return d


def effective_inertia(m, d, dof) -> float:
    """M(q)[dof,dof] + armature at the configuration actually being measured."""
    full = np.zeros((m.nv, m.nv))
    mujoco.mj_fullM(m, d, full)
    return float(full[dof, dof] + m.dof_armature[dof])


# ------------------------------------------------------- free-space experiments
def step_sweep(m, dt_ctrl, substeps, lift=False) -> dict:
    """Peak realised rate vs step amplitude, per hinge.

    Two rates are recorded because the gate measures one of them: w1b compares
    ``np.diff(qpos)/dt_ctrl``, an interval average; the instantaneous ``qvel``
    peak is recorded too so the envelope is not quoted on the average's terms.
    """
    wrist_qadr = np.array([m.jnt_qposadr[rs._joint_id(m, n)] for n in HINGES])
    rows = []
    for name in HINGES:
        k = HINGES.index(name)
        j = rs._joint_id(m, name)
        dadr = int(m.jnt_dofadr[j])
        aid = actuator_id(m, name)
        lo, hi = (float(v) for v in m.actuator_ctrlrange[aid])
        for amp in STEP_AMPLITUDES_RAD:
            for sign in (+1, -1):
                d = settled(m, lift)
                q0 = float(d.qpos[wrist_qadr[k]])
                goal = sign * amp
                if not (lo <= q0 + goal <= hi):
                    continue
                d.ctrl[aid] = q0 + goal
                mujoco.mj_forward(m, d)
                i_eff = effective_inertia(m, d, dadr)
                qs, qs_int, ctrlv, qs_act = [], [], [], []
                for _ in range(40):
                    for _ in range(substeps):
                        mujoco.mj_step(m, d)
                        qs_int.append(float(d.qpos[wrist_qadr[k]]))
                        qs_act.append(float(np.abs(d.qfrc_actuator[dadr])))
                    qs.append(float(d.qpos[wrist_qadr[k]]))
                    ctrlv.append(float(d.ctrl[aid]))
                q = np.array([q0] + qs)
                rate_int = np.abs(np.diff(q)) / dt_ctrl
                rows.append({
                    "joint": name,
                    "step_rad": goal,
                    "abs_step_rad": amp,
                    "inertia_kgm2": i_eff,
                    "peak_rate_control_interval_rad_s": float(rate_int.max()),
                    "peak_rate_instantaneous_rad_s": float(np.abs(
                        np.diff(qs_int) / m.opt.timestep).max()),
                    "peak_tau_Nm": float(max(qs_act)),
                    "max_windup_rad": float(np.abs(np.array(qs) - np.array(
                        ctrlv)).max()),
                    "overshoot_frac": float(max(
                        0.0, (np.array(qs).max() - (q0 + goal)) / goal
                        if goal > 0 else (q0 + goal - np.array(qs).min()) / (-goal))),
                    "final_err_rad": float(qs[-1] - (q0 + goal)),
                })
    return rows


def ramp_sweep(m, dt_ctrl, substeps, lift=False) -> dict:
    """Constant-rate command ramp: lag law, realised/commanded ratio, saturation.

    This is where the kv=0 consequence shows up cleanly.  With actuator kv = 0 the
    only velocity-proportional term in the dof's equation of motion is the
    joint's own damping attribute, so the hinge obeys

        I qddot + b qdot + kp (ctrl - q) = 0

    i.e. a first-order servo of time constant tau_s = b / kp = 1.2/14 = 86 ms.
    A constant-rate command therefore settles at a *finite* lag
    e_ss = tau_s * v_cmd -- but only because b != 0.  The two branches are
    separable in the measurement: a plateau at b*v_cmd/kp means the passive
    damping is what limits the lag; an unbounded ramp of lag means the servo has
    no velocity feedback whatsoever.  The row carries both the plateau and the
    growth so the branch is decided by data, not by assertion.

    Travel is budgeted per leg (TRAVEL_BUDGET_RAD) so a fast ramp cannot run into
    ctrlrange and silently turn into a position hold -- the first version of this
    sweep did exactly that above 1 rad/s and read a lag of zero.
    """
    wrist_qadr = np.array([m.jnt_qposadr[rs._joint_id(m, n)] for n in HINGES])
    dofs = np.array([int(m.jnt_dofadr[rs._joint_id(m, n)]) for n in HINGES])
    aids = [actuator_id(m, n) for n in HINGES]
    rows = []
    for rate in RAMP_RATES_RAD_S:
        n_leg = max(6, int(round(TRAVEL_BUDGET_RAD / (rate * dt_ctrl))))
        n_leg = min(n_leg, RAMP_STEPS)
        travel = rate * dt_ctrl * n_leg
        row = {"command_rate_rad_s": rate, "steps_per_leg": n_leg,
               "commanded_travel_rad": travel,
               "leg_time_constant_coverage": n_leg * dt_ctrl / (1.2 / 14.0),
               "per_joint": []}
        if travel > TRAVEL_BUDGET_RAD * 1.001:
            row["skipped"] = (
                "commanded travel exceeds the per-leg budget at this rate; the "
                "ramp would run into ctrlrange and stop being a ramp")
            rows.append(row)
            continue
        for k, name in enumerate(HINGES):
            d = settled(m, lift)
            ctrl0 = float(d.qpos[wrist_qadr[k]])
            aid, dadr = aids[k], int(dofs[k])
            fr = float(abs(m.actuator_forcerange[aid, 1]))
            kp = float(m.actuator_gainprm[aid, 0])
            b = float(m.dof_damping[dadr])
            lo, hi = (float(v) for v in m.actuator_ctrlrange[aid])
            lag, got, sat, taus = [], [], [], []
            clipped = False
            for t in range(n_leg):
                goal = ctrl0 + (-rate) * dt_ctrl * (t + 1)
                if not (lo <= goal <= hi):
                    clipped = True
                    break
                d.ctrl[aid] = goal
                prev = float(d.qpos[wrist_qadr[k]])
                for _ in range(substeps):
                    mujoco.mj_step(m, d)
                    taus.append(float(abs(d.qfrc_actuator[dadr])))
                cur = float(d.qpos[wrist_qadr[k]])
                lag.append(d.ctrl[aid] - cur)
                got.append((cur - prev) / dt_ctrl)
                sat.append(float(abs(d.qfrc_actuator[dadr]) >= 0.999 * fr))
            lag_arr, got_arr = np.asarray(lag), np.asarray(got)
            tail = max(1, len(lag_arr) // 2)
            row["per_joint"].append({
                "joint": name,
                "steps_run": int(len(lag_arr)),
                "hit_ctrlrange": bool(clipped),
                "lag_first_rad": float(lag_arr[0]),
                "lag_tail_mean_rad": float(lag_arr[-tail:].mean()),
                "lag_plateau_drift_rad_per_s": float(
                    (lag_arr[-1] - lag_arr[-tail]) / (max(1, tail - 1) * dt_ctrl)),
                "predicted_lag_b_qdot_over_kp_rad": -b * rate / kp,
                "measured_over_predicted_lag": float(
                    lag_arr[-tail:].mean() / (-b * rate / kp)) if b else None,
                "realised_rate_tail_rad_s": float(got_arr[-tail:].mean()),
                "realised_over_command": float(abs(got_arr[-tail:].mean()) / rate),
                "peak_realised_rate_rad_s": float(np.abs(got_arr).max()),
                "peak_tau_Nm": float(max(taus) if taus else 0.0),
                "saturated_fraction": float(np.mean(sat)) if sat else 0.0,
            })
        rows.append(row)
    return rows


# ------------------------------------------------ the gate's own command profile
def gate_profile(model, wrist_goal, finger_goal, wrist_qadr, finger_qadr,
                 wrist_act, finger_act, frame_skip, label):
    """Run the gate's own warm start, recording per-substep wrist-hinge state."""
    data = mujoco.MjData(model)
    rs.reset_scene(model, data, finger_qadr, wrist_qadr, finger_act, wrist_act)
    if label.startswith("free_space"):
        # reset_scene puts the box back on the palm; a free-space measurement has
        # to actually be free space, so the box is teleported clear before the
        # recorded warm start begins -- no qpos is written inside it.
        lift_box(model, data)
    dofs = np.array([int(model.jnt_dofadr[rs._joint_id(model, n)]) for n in HINGES])
    # rs.WRIST_JOINTS is slides-then-hinges, so the hinge block starts at 3.
    hinge_cols = [list(wrist_qadr).index(
        int(model.jnt_qposadr[rs._joint_id(model, n)])) for n in HINGES]
    rec = {k: [] for k in ("qpos", "qvel", "ctrl", "tau", "con", "peakN")}
    box_gid = rs._geom_id(model, "object_geom")
    orig = mujoco.mj_step

    def hook(m, d):
        orig(m, d)
        rec["qpos"].append(d.qpos[wrist_qadr].copy())
        rec["qvel"].append(d.qvel[dofs].copy())
        rec["ctrl"].append(d.ctrl[wrist_act].copy())
        rec["tau"].append(d.qfrc_actuator[dofs].copy())
        rec["con"].append(d.qfrc_constraint[dofs].copy())
        peak = 0.0
        for i in range(d.ncon):
            c = d.contact[i]
            if box_gid in (int(c.geom1), int(c.geom2)):
                f = np.zeros(6)
                mujoco.mj_contactForce(m, d, i, f)
                peak = max(peak, float(f[0]))
        rec["peakN"].append(peak)

    mujoco.mj_step = hook
    try:
        info = rs.warm_start(model, data, finger_qadr, wrist_qadr,
                             finger_act, wrist_act, finger_goal, wrist_goal,
                             ctrl_hz=rs.CTRL_HZ, frame_skip=frame_skip)
    finally:
        mujoco.mj_step = orig

    q = np.asarray(rec["qpos"])
    qv = np.asarray(rec["qvel"])
    dt_ctrl = 1.0 / rs.CTRL_HZ
    n_main = info["control_steps"] * frame_skip
    q, qv = q[-n_main:], qv[-n_main:]
    ctrl = np.asarray(rec["ctrl"])[-n_main:]
    tau = np.asarray(rec["tau"])[-n_main:]
    con = np.asarray(rec["con"])[-n_main:]
    peakN = np.asarray(rec["peakN"])[-n_main:]
    q_ctrl_frames = q[frame_skip - 1::frame_skip]
    ctrl_frames = ctrl[frame_skip - 1::frame_skip]
    rate_int = np.abs(np.diff(q_ctrl_frames, axis=0)) / dt_ctrl
    rate_hinge = rate_int[:, hinge_cols]
    return {
        "label": label,
        "control_steps": int(info["control_steps"]),
        "ramp_control_steps": int(info["ramp_control_steps"]),
        "reported_realised_peak_rate": {
            k: float(v) for k, v in info["realised_peak_rate"].items()},
        "per_hinge": [{
            "joint": HINGES[k],
            "peak_rate_control_interval_rad_s": float(rate_hinge[:, k].max()),
            "peak_rate_control_interval_at_step": int(
                np.argmax(rate_hinge[:, k])),
            "peak_rate_instantaneous_rad_s": float(np.abs(qv[:, k]).max()),
            "peak_abs_tau_act_Nm": float(np.abs(tau[:, k]).max()),
            "peak_abs_tau_contact_Nm": float(np.abs(con[:, k]).max()),
            "peak_box_normal_N": float(peakN.max()),
            "max_windup_rad": float(np.abs(q_ctrl_frames[:, hinge_cols[k]]
                                           - ctrl_frames[:, hinge_cols[k]]).max()),
            "peak_interval_forensics": _peak_forensics(
                model, q_ctrl_frames, ctrl_frames, rate_hinge, hinge_cols, k,
                int(info["ramp_control_steps"])),
        } for k in range(3)],
    }


def _peak_forensics(model, q_frames, ctrl_frames, rate_hinge, hinge_cols, k,
                    ramp_steps) -> dict:
    """What the servo was holding at the instant the realised rate peaked.

    The claim under test is that the w1b exceedance is the elastic release of
    the wind-up the ramp accumulated, not a mis-scaled command.  So the row
    carries the wind-up at the peak, the energy it had stored, and the
    *undamped* rate that energy alone could produce -- an upper bound on the
    release, since the hinge is in fact damped (b = 1.2 N m s / rad).
    """
    name = HINGES[k]
    aid = actuator_id(model, name)
    dadr = int(model.jnt_dofadr[rs._joint_id(model, name)])
    kp = float(model.actuator_gainprm[aid, 0])
    col = hinge_cols[k]
    t = int(np.argmax(rate_hinge[:, k]))
    windup = float(ctrl_frames[t, col] - q_frames[t, col])
    d = mujoco.MjData(model)
    mujoco.mj_resetData(model, d)
    full = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, d, full)
    i_eff = float(full[dadr, dadr] + model.dof_armature[dadr])
    v_release = float(np.sqrt(max(0.0, kp) * windup ** 2 / i_eff))
    return {
        "at_control_step": t,
        "ramp_control_steps": int(ramp_steps),
        "inside_ramp": bool(t < ramp_steps),
        "ctrl_rad": float(ctrl_frames[t, col]),
        "qpos_rad": float(q_frames[t, col]),
        "windup_rad": windup,
        "stored_elastic_energy_J": float(0.5 * kp * windup ** 2),
        "inertia_kgm2": i_eff,
        "undamped_release_rate_rad_s": v_release,
        "energy_bound_plus_command_rate_rad_s": float(
            np.sqrt(rate_hinge[t, k] ** 2 + v_release ** 2)),
        "measured_rate_this_interval_rad_s": float(rate_hinge[t, k]),
    }


def undamped_counterfactual(m, dt_ctrl, substeps, lift=True) -> dict:
    """Same free-space step sweep with the hinge damping attribute set to zero.

    This is what turns "a finite bound exists" from an assertion into a
    measurement.  The derived ceiling forcerange/dof_damping is finite only
    because ``dof_damping`` is 1.2 N m s / rad; the actuator's own kv = 0 does not
    enter it.  Zero the attribute and the asymptote disappears: the free-space
    peak rate becomes the undamped elastic line A * sqrt(kp / I), which at the
    hinges' ctrlrange reaches ~27 rad/s instead of ~5.
    """
    u = free_copy(m)
    for name in HINGES:
        u.dof_damping[int(u.jnt_dofadr[rs._joint_id(u, name)])] = 0.0
    rows = step_sweep(u, dt_ctrl, substeps, lift=lift)
    out = {}
    for r in rows:
        if r["step_rad"] in (0.45, 1.20):
            out[f"{r['joint']}@{r['step_rad']:+.2f}"] = {
                "peak_rate_control_interval_rad_s":
                    r["peak_rate_control_interval_rad_s"],
                "undamped_prediction_A_sqrt_kp_over_I_rad_s": float(
                    r["abs_step_rad"] * np.sqrt(
                        14.0 / r["inertia_kgm2"])),
            }
    return {"dof_damping_zeroed": True, "rows": out}


# ------------------------------------------------------------------- derivation
def derive(params, dt_ctrl, cap_rad_s) -> dict:
    """The tolerance arithmetic.  Every input is an MJCF attribute."""
    out = {}
    for name in HINGES:
        p = params[name]
        kp = p["kp_Nm_per_rad"]
        kv = p["kv_Nm_s_per_rad"]
        tmax = p["forcerange_Nm"]
        b = p["dof_damping_Nm_s_per_rad"]
        arm = p["dof_armature_kgm2"]
        i0 = p["dof_M0_at_qpos0_kgm2"] + arm
        e_max = tmax / kp                       # mfj1's map, same units
        v_term = tmax / b                       # (1) damping-limited terminal rate
        v_el = e_max * np.sqrt(kp / i0)         # (2) elastic release ceiling
        tau_s = b / kp                          # first-order servo time constant
        a_max = tmax / i0                       # max angular acceleration
        t_lag = e_max / cap_rad_s               # wind-up traversal window
        dv_am_taus = a_max * t_lag              # (3) the suggested a_max*tau_s form
        out[name] = {
            "kv_Nm_s_per_rad": kv,
            "windup_cap_rad": e_max,
            "damping_limited_terminal_rate_rad_s": v_term,
            "elastic_release_ceiling_rad_s": v_el,
            "servo_time_constant_b_over_kp_s": tau_s,
            "max_angular_accel_tmax_over_I_rad_s2": a_max,
            "windup_traversal_window_at_cap_s": t_lag,
            "a_max_times_window_rad_s": dv_am_taus,
            "allowed_realised_rate_rad_s": v_term,
            "allowed_realised_rate_if_elastic_form_rad_s": v_el,
            "margin_over_signed_cap": v_term / cap_rad_s,
        }
    return out


def envelope(step_rows, undamped) -> dict:
    """The free-space documented envelope: linear branch, saturation knee, asymptote.

    Reported per hinge so the ceiling can be checked against a measurement that
    did not use it: the linear branch is the elastic launch rate A*sqrt(kp/I)
    attenuated by the hinge damping, the knee is where |tau| first reaches
    forcerange, and the asymptote is where the damping-limited ceiling
    forcerange/dof_damping shows up as a plateau.
    """
    out = {}
    for name in HINGES:
        rows = [r for r in step_rows if r["joint"] == name]
        lin = [r for r in rows if r["peak_tau_Nm"] < 5.9]
        slopes = [r["peak_rate_control_interval_rad_s"] / r["abs_step_rad"]
                  for r in lin]
        knee = min((r["abs_step_rad"] for r in rows
                    if r["peak_tau_Nm"] >= 5.999), default=None)
        out[name] = {
            "linear_branch_rate_per_rad_per_s": {
                "min": float(min(slopes)), "max": float(max(slopes)),
                "at_amplitude_range_rad": [lin[0]["abs_step_rad"],
                                           lin[-1]["abs_step_rad"]]},
            "saturation_knee_step_rad": knee,
            "max_free_space_peak_rate_rad_s": float(max(
                r["peak_rate_control_interval_rad_s"] for r in rows)),
            "max_free_space_peak_instantaneous_rad_s": float(max(
                r["peak_rate_instantaneous_rad_s"] for r in rows)),
            "largest_step_tested_rad": max(r["abs_step_rad"] for r in rows),
            "undamped_same_step_peak_rad_s": {
                k: v["peak_rate_control_interval_rad_s"]
                for k, v in undamped["rows"].items()
                if k.startswith(name + "@")},
        }
    return out


def proposal(params, derived, loaded, free, cap, measured) -> dict:
    """The tolerance actually being proposed, plus its own falsification tests.

    Same map as mfj1, one degree of freedom further along: the servo's torque
    ceiling, read off the actuator's forcerange, pushed through the passive
    compliance of the dof.

        mfj1  (position)  |forcerange| / kp            = 0.10388/0.7269 = 142.9 mrad
        w1b   (rate)      |forcerange| / dof_damping   = 6.0/1.2       = 5.0 rad/s

    Both are ratios of two MJCF attributes on the same dof, both are per-DoF
    tables, and neither contains a fitted constant.  The only difference is
    which passive compliance closes the map: the actuator gain for a position
    wind-up, the joint damping for a rate.
    """
    per = {}
    for name in HINGES:
        p, dv = params[name], derived[name]
        ld = next(x for x in loaded["per_hinge"] if x["joint"] == name)
        fs = next(x for x in free["per_hinge"] if x["joint"] == name)
        tau_c = ld["peak_abs_tau_contact_Nm"]
        per[name] = {
            "dof_damping_Nm_s_per_rad": p["dof_damping_Nm_s_per_rad"],
            "forcerange_Nm": p["forcerange_Nm"],
            "allowed_realised_rate_rad_s": dv["allowed_realised_rate_rad_s"],
            "derivation": "abs(actuator_forcerange) / dof_damping",
            "elastic_release_ceiling_rad_s":
                dv["allowed_realised_rate_if_elastic_form_rad_s"],
            "elastic_derivation": (
                "(forcerange/kp) * sqrt(kp / (dof_M0 + dof_armature))"),
            "margin_vs_measured_1p7234": dv["allowed_realised_rate_rad_s"] / measured,
            "margin_vs_signed_cap_1p5": dv["allowed_realised_rate_rad_s"] / cap,
            "measured_loaded_rad_s": ld["peak_rate_control_interval_rad_s"],
            "measured_free_space_rad_s": fs["peak_rate_control_interval_rad_s"],
            "loaded_minus_free_space_rad_s": (
                ld["peak_rate_control_interval_rad_s"]
                - fs["peak_rate_control_interval_rad_s"]),
            "loaded_over_free_space": (
                ld["peak_rate_control_interval_rad_s"]
                / fs["peak_rate_control_interval_rad_s"]),
            "loaded_peak_contact_torque_Nm": tau_c,
            "contact_inclusive_ceiling_rad_s": (
                p["forcerange_Nm"] + tau_c) / p["dof_damping_Nm_s_per_rad"],
            "actuator_saturated_during_gate_ramp": bool(
                ld["peak_abs_tau_act_Nm"] >= 0.999 * p["forcerange_Nm"]),
        }
    return {
        "name": "servo winding tolerance table (伺服绕组耐受表)",
        "thesis": (
            "Every gate in this repo that is red for the same reason is using a "
            "command-side limit to judge a physical-side state. mfj1 is a "
            "position wind-up; w1b is a realised rate. Both are the servo torque "
            "ceiling mapped through the dof's passive compliance, per DoF, with "
            "no fitted constant."),
        "mfj1_isomorphism": {
            "mfj1_allowed_rad": 0.10388 / 0.7269,
            "mfj1_measured_rad": 0.0071731073144030155,
            "mfj1_derivation": "abs(actuator_forcerange) / actuator_gainprm[0]",
            "w1b_allowed_rad_s": derived[HINGES[0]]["allowed_realised_rate_rad_s"],
            "w1b_measured_rad_s": measured,
            "w1b_derivation": "abs(actuator_forcerange) / dof_damping",
        },
        "per_hinge": per,
        "rejected_candidate": {
            "form": "dv <= a_max * tau_s with a_max = forcerange/I, tau_s = e_max/v_cap",
            "values_rad_s": {k: v["a_max_times_window_rad_s"]
                             for k, v in derived.items()},
            "why_rejected": (
                "vacuous by two orders of magnitude against the measured 1.7234 "
                "rad/s: the wrist hinge is stiff (kp 14) and light (I 0.062), so "
                "the acceleration ceiling forcerange/I is ~97 rad/s^2 and the "
                "wind-up traversal window is 0.29 s. It is reported rather than "
                "quietly dropped so the bound that was checked is on the record."),
        },
        "no_unexplained_term_no_exemption": {
            "rule": (
                "A realised rate above the signed cap is inside tolerance only if "
                "its excess is accounted for by the dof's own wind-up. If the "
                "excess is larger than the wind-up can release, no cap change is "
                "warranted: the command profile or the MJCF is wrong."),
            "numeric_test": (
                "for each wrist hinge, at every control interval: "
                "excess = |dq/dt| - v_cmd.  admissible iff "
                "excess <= sqrt(kp * windup^2 / (dof_M0 + armature)), where "
                "windup = ctrl - qpos at that interval.  On this clip the three "
                "hinges admit 1.850 / 2.855 / 2.304 rad/s of release and need "
                "0.223 / 0.117 / 0.103, i.e. 8.3x / 24.4x / 22.3x headroom.  A "
                "hinge that fails this test fails even though it is under the "
                "absolute ceiling, and gets no exemption."),
        },
    }


def main():
    out = {
        "probe": "w35g_wrist_rate_tol",
        "method": __doc__,
        "mjcf_commit": "1b0f13f",
        "gate_context": {
            "gate": "tests/test_smoke5_warmstart.py::test_w1b_realised_rates_within_caps",
            "signed_command_cap_rad_s": rs.WARM_START_HINGE_RATE_RAD_S,
            "measured_realised_rate_rad_s": 1.7234,
            "measured_on": "w1b warm-start ramp, wrist hinge",
        },
    }

    model, data = rs.load_scene()[1:]
    free = free_copy(model)
    dt_ctrl = 1.0 / rs.CTRL_HZ
    substeps = int(round(dt_ctrl / model.opt.timestep))

    params = hinge_params(model, free)
    out["mjcf_parameters"] = params
    out["integrator"] = str(model.opt.integrator)

    print("step sweep...")
    out["free_space_step_response"] = step_sweep(free, dt_ctrl, substeps, lift=True)
    print("ramp sweep...")
    out["free_space_ramp_tracking"] = ramp_sweep(free, dt_ctrl, substeps, lift=True)
    out["derived_tolerance"] = derive(params, dt_ctrl,
                                      rs.WARM_START_HINGE_RATE_RAD_S)

    # ---- the gate's own command profile, free space and loaded
    with CACHE.open("rb") as fh:
        payload = pickle.load(fh)
    anchor = payload["anchor"]
    wrist_qadr = np.array([model.jnt_qposadr[rs._joint_id(model, n)]
                           for n in rs.WRIST_JOINTS])
    finger_qadr = rs._finger_qadr(model)
    finger_act, wrist_act = np.arange(6, 22), np.arange(6)
    lo, hi = model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]
    w_goal = np.clip(anchor["ctrl"][0], lo[wrist_act], hi[wrist_act])
    f_goal = np.clip(anchor["finger_cmd"][0], lo[finger_act], hi[finger_act])
    out["gate_context"]["first_wrist_command_rad"] = w_goal.tolist()

    print("loaded gate profile...")
    out["loaded_gate_profile"] = gate_profile(
        model, w_goal, f_goal, wrist_qadr, finger_qadr, wrist_act, finger_act,
        substeps, "loaded_real_clip")
    print("free-space gate profile...")
    free_qadr = np.array([free.jnt_qposadr[rs._joint_id(free, n)]
                          for n in rs.WRIST_JOINTS])
    free_fq = rs._finger_qadr(free)
    out["free_space_gate_profile"] = gate_profile(
        free, w_goal, f_goal, free_qadr, free_fq, wrist_act, finger_act,
        substeps, "free_space_box_lifted_3m")

    print("undamped counterfactual...")
    out["undamped_counterfactual"] = undamped_counterfactual(
        free, dt_ctrl, substeps)

    out["free_space_envelope"] = envelope(
        out["free_space_step_response"], out["undamped_counterfactual"])
    out["proposal"] = proposal(
        params, out["derived_tolerance"], out["loaded_gate_profile"],
        out["free_space_gate_profile"], rs.WARM_START_HINGE_RATE_RAD_S,
        out["gate_context"]["measured_realised_rate_rad_s"])

    dest = REPO / "runs/w35/wrist_rate_tol.json"
    dest.write_text(json.dumps(out, indent=2, default=float) + "\n")
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
