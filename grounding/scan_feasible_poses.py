"""Feasible-pose scan + behaviour-cloning dataset for Smoke 4 (revised MJCF).

    python grounding/scan_feasible_poses.py                # full 4000-pose scan
    python grounding/scan_feasible_poses.py --n-poses 200  # quick probe

Why this exists
---------------
Round 2 of smoke4 proved the scene is *reachable* by a script (534 of a 4000-pose
sweep cleared 5 cm while a finger held the box), but that sweep was never written
down, so nothing about it could be reused. PPO cannot find the grasp on its own:
the reward is not the problem, exploring a 22-dim contact manifold is. This script
redoes the sweep on the revised MJCF (commit 4508421) and **persists** it:

    runs/p0/smoke4_mujoco_ppo/feasible_poses.npz

holding (a) the sampled parameters of every candidate, (b) the measured evidence
for every candidate, (c) the (obs, action) pairs of the feasible ones, and (d) a
JSON metadata blob with the exact scan configuration. Everything runs through
``LiftHoldEnv.step`` exactly like PPO does - the scan never writes qpos - so every
recorded trajectory is replayable through the same interface the acceptance test
uses.

Why a correlated-random rollout and not an open-loop script
-----------------------------------------------------------
The revised actuators changed the hand's *dynamics*, not just its strength. The
finger forcerange is now 2.5 N m of tip force per joint instead of 2 N m of joint
torque, so a saturated joint moves ~20x slower than before: measured ~0.03 rad per
0.3 s of control on ``ffj1``. Traversing the finger ctrlrange now takes tens of
seconds, well past a 250-step episode. A fixed open-loop script that assumed fast
fingers therefore drives the hand into a pose it cannot reach in time and finds
nothing - that is exactly what the first version of this scan measured (0 feasible
in 150 poses, best peak lift 9 mm). A correlated random walk with a scheduled
wrist-lift bias explores the reachable set instead, which is also the distribution
PPO samples from, so the (obs, action) pairs it records are usable as BC targets.

Per candidate (all constants below are recorded in the npz metadata):

  * a 16-dim finger setpoint parameterised as 4 curl scalars x 4 joints, where
    joint 0 is damped (its ctrlrange is ~7x narrower than j1-j3);
  * 5 wrist biases (x, y, rx, ry, rz) that are switched off once the lift bias
    starts, so the hand stabilises before it pulls;
  * a lift bias on wrist_z that ramps linearly from 0 to ``lift_gain`` between
    ``close_steps`` and ``close_steps + lift_ramp``;
  * an Ornstein-Uhlenbeck jitter on every action dimension, so consecutive actions
    are correlated like a policy's, not white noise.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

OUT_NPZ = REPO / "runs/p0/smoke4_mujoco_ppo/feasible_poses.npz"
OUT_META = REPO / "runs/p0/smoke4_mujoco_ppo/feasible_poses_meta.json"

# ---------------------------------------------------------------- scan config
SCAN_SEED = 20251015
DEFAULT_EPISODE_LEN = 600     # 12 s: see "horizon" below
PARAM_NAMES = (
    "curl_ff", "curl_mf", "curl_rf", "curl_th",
    "wrist_bx", "wrist_by", "wrist_brx", "wrist_bry", "wrist_brz",
    "curl_jitter", "wrist_jitter",
    "desc_steps", "desc_gain", "close_steps", "lift_ramp", "lift_gain",
)
J0_GAIN = 0.3                 # finger joint 0 gets a third of the curl scalar
DESC_STEPS_RANGE = (8, 60)
DESC_GAIN_RANGE = (0.3, 1.0)
CLOSE_STEPS_RANGE = (60, 300)
LIFT_RAMP_RANGE = (40, 250)
LIFT_GAIN_RANGE = (0.05, 0.4)
CURL_JITTER_RANGE = (0.0, 0.15)
WRIST_JITTER_RANGE = (0.02, 0.2)
WRIST_LATERAL_RANGE = (-0.35, 0.35)  # wrist x/y bias, clipped by ctrlrange
WRIST_ROT_BIAS_RANGE = (-1.5, 1.5)   # rx/ry/rz: the hand must be able to yaw, or
                                    # every candidate is one-sided (four fingers
                                    # on the box's +x face, no opposition) and only
                                    # levers the box up on an edge.
OU_THETA = 0.12               # per-step mean reversion of the jitter process

# ---------------------------------------------------------- feasibility gates
# Mirrors tests/test_smoke4_accept.py, which is read-only. A pose is "feasible"
# only if the evidence the acceptance gate demands is present.
GATE_LIFT_M = 0.05
GATE_MIN_FINGER_STEPS = 5
GATE_FORCE_DUTY = 0.50
GATE_FORCE_MEAN_N = 1.4715
GATE_MAX_ANCHORED_STEP_M = 0.01
GATE_MAX_GRAVITY_ERR = 0.60
GATE_HOLD_STEPS = 20
DT = 0.02


def _rollout_actions(params: np.ndarray, n_steps: int, rng: np.random.Generator) -> np.ndarray:
    """Correlated-random action trajectory for one candidate.

    Schedule on wrist_z (the only axis that matters for the lift): descend the
    palm onto the box, hold while the fingers close, then ramp the lift. The
    descent is not optional - measured on the reset pose, a fully curled fingertip
    sits 5-8 cm ABOVE the box top and a straight one 6 mm above the box's +x
    edge, so without descending the hand the fingers never reach a side face and
    the only contact available is the palm pressing the top, which cannot lift.
    """
    (c_ff, c_mf, c_rf, c_th, bx, by, brx, bry, brz, curl_jit, wrist_jit,
     desc_steps, desc_gain, close_steps, lift_ramp, lift_gain) = params

    fingers = np.array([c_ff, c_mf, c_rf, c_th])
    joint_gain = np.array([J0_GAIN, 1.0, 1.0, 1.0])
    finger_bias = np.repeat(fingers, 4) * np.tile(joint_gain, 4)

    desc_steps = int(desc_steps)
    t_close = desc_steps + int(close_steps)
    lift_ramp = max(1.0, float(lift_ramp))
    t = np.arange(n_steps, dtype=np.float64)

    wz = np.where(t < desc_steps, -desc_gain, 0.0)
    wz += lift_gain * np.clip((t - t_close) / lift_ramp, 0.0, 1.0)
    # wrist lateral/rotational bias is released once the closing phase ends
    release = np.clip((t - t_close) / max(1.0, 0.25 * lift_ramp), 0.0, 1.0)
    wrist_bias = np.outer(release, np.array([bx, by, 0.0, brx, bry, brz]))

    eps = rng.normal(size=(n_steps, 22))
    ou = np.zeros((n_steps, 22))
    ou[0] = eps[0]
    for i in range(1, n_steps):
        ou[i] = (1.0 - OU_THETA) * ou[i - 1] + OU_THETA * eps[i]

    acts = np.empty((n_steps, 22))
    acts[:, 6:] = finger_bias + curl_jit * ou[:, 6:]
    acts[:, 0:3] = wrist_bias[:, 0:3] + wrist_jit * ou[:, 0:3]
    acts[:, 3:6] = wrist_bias[:, 3:6] + wrist_jit * ou[:, 3:6]
    acts[:, 2] = np.clip(wz + wrist_jit * ou[:, 2], -1.0, 1.0)
    return np.clip(acts, -1.0, 1.0)


def _penetration(env) -> float:
    """Deepest box-vs-anything contact in the current state (MuJoCo reports dist
    negative when overlapping). Feeds the interpenetration diagnostic."""
    d = env.data
    worst = 0.0
    for i in range(d.ncon):
        con = d.contact[i]
        if env.obj_geom in (int(con.geom1), int(con.geom2)):
            worst = max(worst, -float(con.dist))
    return worst


def _finger_saturation(env) -> float:
    """Largest |ctrl| / |forcerange| over the finger actuators (1.0 == at limit)."""
    lo = np.abs(env.ctrl_low[6:]).copy()
    hi = np.abs(env.ctrl_high[6:]).copy()
    lo[lo == 0] = 1e-12
    hi[hi == 0] = 1e-12
    ctrl = np.abs(env.data.ctrl[6:])
    return float(np.max(np.maximum(ctrl / lo, ctrl / hi)))


def _evidence(lift, finger_force, held, supported, vz, pen, sat, pairs_at_end, env) -> dict:
    peak_i = int(np.argmax(lift))
    peak = float(lift[peak_i])
    win = slice(0, peak_i + 1)
    duty = float(np.mean(finger_force[win] > 0.0)) if peak_i >= 0 else 0.0
    mean_force = float(np.mean(finger_force[win])) if peak_i >= 0 else 0.0
    n_finger_steps = int(np.sum(held > 0.5))

    anchored = (supported > 0.5) | (held > 0.5)
    per_step = np.abs(np.diff(lift, prepend=lift[0]))
    anchored_disp = per_step[anchored]
    max_anchored = float(np.max(anchored_disp)) if anchored_disp.size else 0.0
    free = ~anchored
    dvz = np.diff(vz, prepend=0.0)
    free_dv = dvz[1:][free[1:]]
    grav_err = float(np.max(np.abs(free_dv - (-9.81 * DT)))) if free_dv.size else 0.0

    peak_pairs = pairs_at_end[peak_i] if peak_i < len(pairs_at_end) else []
    touching = any(name in env.support_geoms for name, _ in peak_pairs)
    hold_streak = max(env.trace["hold_streak"]) if env.trace["hold_streak"] else 0.0

    ev = {
        "peak_lift_m": peak,
        "peak_step": peak_i,
        "n_finger_contact_steps": n_finger_steps,
        "finger_force_duty_cycle": duty,
        "finger_force_mean_N": mean_force,
        "max_anchored_step_m": max_anchored,
        "max_free_fall_gravity_error_m_s": grav_err,
        "table_touching_at_peak": bool(touching),
        "max_hold_streak": float(hold_streak),
        "max_penetration_m": float(np.max(pen)) if pen.size else 0.0,
        "max_finger_actuator_saturation": float(np.max(sat)) if sat.size else 0.0,
        "max_box_speed_m_s": float(np.max(np.abs(vz))),
        "final_lift_m": float(lift[-1]) if lift.size else 0.0,
        "peak_contact_force_N": float(np.max(finger_force)) if finger_force.size else 0.0,
    }
    ev["feasible"] = bool(
        peak > GATE_LIFT_M
        and n_finger_steps >= GATE_MIN_FINGER_STEPS
        and duty >= GATE_FORCE_DUTY
        and mean_force >= GATE_FORCE_MEAN_N
        and not touching
        and max_anchored < GATE_MAX_ANCHORED_STEP_M
        and grav_err < GATE_MAX_GRAVITY_ERR
        and ev["max_hold_streak"] >= GATE_HOLD_STEPS
    )
    return ev


def _roll_out(env, actions: np.ndarray, collect: bool = False):
    obs, _ = env.reset(seed=env._scan_seed)
    lift, ff, held, sup, vz, pen, sat, pairs = [], [], [], [], [], [], [], []
    bc_obs: list[np.ndarray] = []
    bc_act: list[np.ndarray] = []
    for a in actions:
        obs, _, term, trunc, _ = env.step(a)
        lift.append(float(env.trace["lift"][-1]))
        ff.append(float(env.trace["finger_force"][-1]))
        held.append(float(env.trace["finger_held"][-1]))
        sup.append(float(env.trace["supported"][-1]))
        vz.append(float(env.data.qvel[env.obj_qvel_adr + 2]))
        pen.append(_penetration(env))
        sat.append(_finger_saturation(env))
        pairs.append([tuple(p) for p in env.trace["contact_pairs"][-1]])
        if collect:
            bc_obs.append(obs.copy())
            bc_act.append(np.asarray(a, dtype=np.float32))
        if term or trunc:
            break
    ev = _evidence(np.asarray(lift), np.asarray(ff), np.asarray(held), np.asarray(sup),
                   np.asarray(vz), np.asarray(pen), np.asarray(sat), pairs, env)
    if collect:
        return ev, np.asarray(bc_obs, dtype=np.float32), np.asarray(bc_act, dtype=np.float32)
    return ev, None, None


def scan(n_poses: int, seed: int, episode_len: int) -> dict:
    from grounding.envs.lift_hold_env import LiftHoldEnv  # noqa: PLC0415

    rng = np.random.default_rng(seed)
    env = LiftHoldEnv(episode_len=episode_len)
    env._scan_seed = 0

    params = np.empty((n_poses, len(PARAM_NAMES)), dtype=np.float64)
    params[:, 0:4] = rng.uniform(-1.0, 1.0, size=(n_poses, 4))
    params[:, 4:6] = rng.uniform(*WRIST_LATERAL_RANGE, size=(n_poses, 2))
    params[:, 6:9] = rng.uniform(*WRIST_ROT_BIAS_RANGE, size=(n_poses, 3))
    params[:, 9] = rng.uniform(*CURL_JITTER_RANGE, size=n_poses)
    params[:, 10] = rng.uniform(*WRIST_JITTER_RANGE, size=n_poses)
    params[:, 11] = rng.uniform(*DESC_STEPS_RANGE, size=n_poses)
    params[:, 12] = rng.uniform(*DESC_GAIN_RANGE, size=n_poses)
    params[:, 13] = rng.uniform(*CLOSE_STEPS_RANGE, size=n_poses)
    params[:, 14] = rng.uniform(*LIFT_RAMP_RANGE, size=n_poses)
    params[:, 15] = rng.uniform(*LIFT_GAIN_RANGE, size=n_poses)

    ev_keys = ("peak_lift_m", "peak_step", "n_finger_contact_steps",
               "finger_force_duty_cycle", "finger_force_mean_N",
               "max_anchored_step_m", "max_free_fall_gravity_error_m_s",
               "table_touching_at_peak", "max_hold_streak", "max_penetration_m",
               "max_finger_actuator_saturation", "max_box_speed_m_s", "final_lift_m",
               "peak_contact_force_N", "feasible")
    metrics: dict[str, list] = {k: [] for k in ev_keys}

    bc_obs: list[np.ndarray] = []
    bc_act: list[np.ndarray] = []
    bc_pose: list[int] = []
    traj_len: list[int] = []

    t0 = time.time()
    for i in range(n_poses):
        # each candidate draws its own OU path from the scan rng, in sequence
        actions = _rollout_actions(params[i], episode_len, rng)
        env._scan_seed = seed + 1_000 + i
        ev, obs_traj, act_traj = _roll_out(env, actions)
        for k in ev_keys:
            metrics[k].append(ev[k])
        if ev["feasible"]:
            env._scan_seed = seed + 1_000 + i
            _, obs_traj, act_traj = _roll_out(env, actions, collect=True)
            bc_obs.append(obs_traj)
            bc_act.append(act_traj)
            bc_pose.extend([i] * obs_traj.shape[0])
            traj_len.append(obs_traj.shape[0])
        if (i + 1) % 200 == 0:
            print(f"[scan] {i + 1}/{n_poses} feasible={sum(metrics['feasible'])} "
                  f"best_lift={max(metrics['peak_lift_m']):.4f} "
                  f"max_pen={max(metrics['max_penetration_m']):.4f} "
                  f"elapsed={time.time() - t0:.0f}s", flush=True)
    env.close()

    peak = np.asarray(metrics["peak_lift_m"])
    hold = np.asarray(metrics["max_hold_streak"])
    finger_steps = np.asarray(metrics["n_finger_contact_steps"])
    feasible = np.asarray(metrics["feasible"], dtype=bool)
    meta = {
        "mjcf_commit": "4508421",
        "scan_seed": seed,
        "reset_seed_base": seed + 1_000,
        "n_poses": n_poses,
        "episode_len": episode_len,
        "episode_seconds": episode_len * DT,
        "horizon_note": (
            "600 steps = 12 s. The revised finger forcerange makes a saturated joint "
            "~20x slower than the pre-revision model (~0.03 rad per 0.3 s on ffj1), so "
            "the hand needs seconds, not the 250 steps round 2 used, to close and lift. "
            "No acceptance threshold depends on the horizon; the official evaluator "
            "(runs/p0/assets/tissue_box_lift_hold.json) sets no time limit."),
        "param_names": list(PARAM_NAMES),
        "ranges": {
            "curl_scalars": [-1.0, 1.0],
            "wrist_lateral_bias": list(WRIST_LATERAL_RANGE),
            "wrist_rot_bias": list(WRIST_ROT_BIAS_RANGE),
            "curl_jitter": list(CURL_JITTER_RANGE),
            "wrist_jitter": list(WRIST_JITTER_RANGE),
            "desc_steps": list(DESC_STEPS_RANGE),
            "desc_gain": list(DESC_GAIN_RANGE),
            "close_steps": list(CLOSE_STEPS_RANGE),
            "lift_ramp": list(LIFT_RAMP_RANGE),
            "lift_gain": list(LIFT_GAIN_RANGE),
            "j0_gain": J0_GAIN,
            "ou_theta": OU_THETA,
        },
        "gates": {
            "lift_m": GATE_LIFT_M,
            "min_finger_contact_steps": GATE_MIN_FINGER_STEPS,
            "force_duty_cycle": GATE_FORCE_DUTY,
            "force_mean_N": GATE_FORCE_MEAN_N,
            "max_anchored_step_m": GATE_MAX_ANCHORED_STEP_M,
            "max_free_fall_gravity_error_m_s": GATE_MAX_GRAVITY_ERR,
            "min_hold_steps": GATE_HOLD_STEPS,
        },
        "n_feasible": int(feasible.sum()),
        "n_poses_lift_gt_5cm": int(np.sum(peak > GATE_LIFT_M)),
        "n_poses_lift_gt_5cm_finger_held": int(
            np.sum((peak > GATE_LIFT_M) & (finger_steps >= GATE_MIN_FINGER_STEPS))),
        "n_poses_hold_ge_20": int(np.sum(hold >= GATE_HOLD_STEPS)),
        "best_peak_lift_m": float(peak.max()) if peak.size else 0.0,
        "n_bc_transitions": int(sum(traj_len)),
        "n_bc_trajectories": len(traj_len),
        "max_penetration_m_overall": float(np.max(metrics["max_penetration_m"])),
        "max_box_speed_m_s_overall": float(np.max(metrics["max_box_speed_m_s"])),
        "wall_seconds": time.time() - t0,
    }
    OUT_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT_NPZ,
        params=params,
        param_names=np.array(PARAM_NAMES),
        **{k: np.asarray(metrics[k]) for k in ev_keys},
        feasible_idx=np.flatnonzero(feasible),
        bc_obs=np.concatenate(bc_obs).astype(np.float32) if bc_obs else np.zeros((0, 64), np.float32),
        bc_act=np.concatenate(bc_act).astype(np.float32) if bc_act else np.zeros((0, 22), np.float32),
        bc_pose=np.asarray(bc_pose, dtype=np.int32),
        bc_trajectory_len=np.asarray(traj_len, dtype=np.int32),
        meta_json=np.array(json.dumps(meta, indent=2)),
    )
    OUT_META.write_text(json.dumps(meta, indent=2))
    return meta


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-poses", type=int, default=4000)
    p.add_argument("--seed", type=int, default=SCAN_SEED)
    p.add_argument("--episode-len", type=int, default=DEFAULT_EPISODE_LEN)
    a = p.parse_args()
    meta = scan(a.n_poses, a.seed, a.episode_len)
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
