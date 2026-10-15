"""Task 1 - does a *clean* grasp-and-lift exist in this scene, and can we collect it?

Three bugs in the inherited controller made the previous "no clean lift exists"
verdict an artefact rather than a fact about the scene. All three are measured in
`_diagnose()` below and pinned by tests/test_smoke4_demos.py:

1. **Finger action polarity was inverted.** `_apply_action` maps the finger half
   of the action onto each actuator's ctrlrange as `lo + (a+1)/2*(hi-lo)`, so
   a = -1 is the *extended* finger (ctrl at its lower bound) and a = +1 is the
   *curled* one. The old probe drove `close[N_WRIST:] = -1.0`, labelled in its
   docstring as "wind to the curl stop" - i.e. it commanded the fingers to
   STRAIGHTEN to their stops while approaching the box. Measured at the box
   (wrist at ctrl_z=-0.02, box half-x = 0.060 m): at a = -1.0 the ff tip sits at
   x = +0.102 m, 42 mm *outside* the +x wall; at a = 0.0 it sits at x = +0.059 m,
   on the wall. The reported 10-16 N of "finger force" and the 24-29 mm of "box
   lift" were therefore the box being levered by fingertips that were never on
   it in a grasp, which is exactly what the acceptance gates reject.

2. **The middle-finger and thumb joint indices aliased to the last finger.**
   `lift_hold_env` resolves finger joints by the names `mfa0..mfa3`/`tha0..tha3`,
   which are the *actuator* names in this MJCF; the joint names are
   `mfj0..mfj3`/`thj0..thj3`. `mj_name2id(..., mjOBJ_JOINT, "mfa0")` returns -1,
   and `jnt_qposadr[-1]` silently reads element 28 - so 8 of the 16 finger joint
   addresses collapsed onto the last one. The reset pre-shape therefore never
   reached the middle finger or the thumb, and the observation vector carried 7
   duplicated entries instead of 16 distinct joint angles (measured: 9 unique of
   16). Any policy trained on that observation was blind to half the hand.

3. **wrist_z has a hard floor set by the palm, not the ctrlrange.** The palm
   collision geom's bottom face sits at wrist_z - 0.0297 m, and at the reset pose
   that is 0.8393 m against a box top of 0.8400 m - the palm rests ON the box.
   Commanding ctrl_z = -0.163 (the ctrlrange floor) yields wrist_z = 0.8687 m,
   indistinguishable from ctrl_z = -0.020 (0.8698 m). The inherited
   "slope 0.545 m per unit ctrl" was fitted across that saturated region and is
   wrong by ~1.8x: over the linear region the measured slope is 0.982, and x is
   1:1.

With the polarity fixed, a clean bilateral pinch IS kinematically reachable - the
grasp axis is the box's 120 mm x span (fingers on the +x wall, thumb on the -x
wall), and `_kinematic_candidates` finds poses with pads on both walls at matched
heights and the palm clear. **But none of them survives contact**, and the reason
is the finding this module exists to record: to get a pad onto the far wall, 12-13
of the 16 finger collision geoms end up *inside* the box. The pads are spheres on
the end of three phalanges; they cannot reach past their own finger. So every
"bilateral contact" pose is an interpenetration, and the contact solver resolves
it by ejecting the box rather than by gripping it.

`--stage verdict` quantifies this. The measured bottleneck is geometric, not
exploration and not reward shaping:

  * palm plate 95.2 mm wide vs a 120 mm box, sitting directly over it - the palm
    can never be placed clear beside the box, so it is always either pressing on
    the box top or interpenetrating it;
  * the palm's z position servo (kp 180, forcerange +-45 N) drives ~26 N into the
    box top against a 1.47 N box - an 18x overload that pins the box to the
    pedestal whenever the palm is down;
  * the only rotation that brings the pads below the box top (wrist_rx) tilts the
    three long fingers out of level by up to 78 mm, so they rake the box and rock
    it onto an edge; the rotation that keeps them level (wrist_rz) cannot lower
    them;
  * box bottom sits exactly on the pedestal - zero clearance, so there is no
    under-cut for fingers to get beneath.

Because no clean trajectory exists, there is no demonstration set to collect and
no BC/PPO warm start that would not be warm-starting from a fabrication. The
verdict is FAIL with evidence, and the scene change is the owner's call.

    python grounding/collect_clean_demos.py --stage diagnose
    python grounding/collect_clean_demos.py --stage kinematic
    python grounding/collect_clean_demos.py --stage sweep --workers 10
    python grounding/collect_clean_demos.py --stage verdict

Not an acceptance test: it answers "does the clean lift exist and where", which is
an input to training, not a verdict on it.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import pathlib
import sys
from dataclasses import asdict, dataclass

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
OUT_DIR = REPO / "runs/p0/smoke4_mujoco_ppo"
SWEEP_JSON = OUT_DIR / "clean_lift_sweep.json"
DEMO_NPZ = OUT_DIR / "clean_demos.npz"
DIAG_JSON = OUT_DIR / "clean_lift_diagnose.json"
VERDICT_JSON = OUT_DIR / "clean_lift_verdict.json"

MJCF_COMMIT = "4ee6f1d"
N_WRIST = 6
N_FINGER = 16

FINGER_JOINTS = ("ffj0", "ffj1", "ffj2", "ffj3",
                 "mfj0", "mfj1", "mfj2", "mfj3",
                 "rfj0", "rfj1", "rfj2", "rfj3",
                 "thj0", "thj1", "thj2", "thj3")
# the three long fingers press the +x wall; the thumb presses the -x wall
LONG_FINGER_ACTS = tuple(range(N_WRIST, N_WRIST + 12))   # ff, mf, rf
THUMB_ACTS = tuple(range(N_WRIST + 12, N_WRIST + N_FINGER))

WRIST_STEP = np.array([0.012, 0.012, 0.008, 0.10, 0.10, 0.10])

# the criterion, verbatim from the task: finger contact sustained AND the box at
# least this far off the pedestal AND held there at least this many steps
CLEAN_LIFT_M = 0.005
CLEAN_HOLD_STEPS = 20


# --------------------------------------------------------------------------- env
def _make_env(episode_len: int = 1200):
    from grounding.envs.lift_hold_env import LiftHoldEnv  # noqa: PLC0415
    return LiftHoldEnv(episode_len=episode_len)


def action_from_ctrl(env, ctrl_target: np.ndarray) -> np.ndarray:
    """Invert `_apply_action` so a caller can name a target ctrl vector directly.

    The wrist half is a *delta* applied to the current ctrl, so a P-controller
    has to convert the residual into the per-step delta the env will add.
    """
    tgt = np.asarray(ctrl_target, dtype=np.float64)
    a = np.zeros(22)
    a[0:6] = np.clip((tgt[0:6] - env.data.ctrl[0:6]) / WRIST_STEP, -1.0, 1.0)
    lo, hi = env.ctrl_low[N_WRIST:], env.ctrl_high[N_WRIST:]
    a[N_WRIST:] = np.clip(2.0 * (tgt[N_WRIST:] - lo) / (hi - lo) - 1.0, -1.0, 1.0)
    return a


def ctrl_from_amp(env, amps: np.ndarray) -> np.ndarray:
    """Finger ACTION units -> finger ctrl, i.e. the forward map of the same thing."""
    lo, hi = env.ctrl_low[N_WRIST:], env.ctrl_high[N_WRIST:]
    return lo + (np.asarray(amps, dtype=np.float64) + 1.0) * 0.5 * (hi - lo)


# ----------------------------------------------------------------- kinematics
def _kin_setup(env):
    """Static forward-kinematics handle: no dynamics, no contact response."""
    import mujoco  # noqa: PLC0415
    m, d = env.model, env.data
    qa = np.array([m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]
                   for n in FINGER_JOINTS])
    palm = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "palm_collision")
    tips = {}
    for pref in ("ff_tip", "mf_tip", "rf_tip", "th_tip"):
        gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, pref + "_collision")
        tips[pref] = gid
    return m, d, qa, palm, tips


def fk_state(env, kin, box: np.ndarray, wrist: np.ndarray, amps: np.ndarray) -> None:
    """Put the model in an exact pose (no integration, no dynamics)."""
    import mujoco  # noqa: PLC0415
    m, d, qa, _, _ = kin
    mujoco.mj_resetData(m, d)
    d.qpos[:] = 0.0
    d.qpos[0:7] = [box[0], box[1], box[2], 1.0, 0.0, 0.0, 0.0]
    d.qpos[env.wrist_qpos_adr] = wrist
    fctrl = ctrl_from_amp(env, amps)
    d.qpos[qa] = fctrl
    d.ctrl[:6] = wrist
    d.ctrl[N_WRIST:] = fctrl
    mujoco.mj_forward(m, d)


def palm_clearance(env, kin, box: np.ndarray, half: np.ndarray) -> float:
    """Signed AABB gap between the palm collision geom and the box.

    Negative means the palm is inside the box. Approximating the rotated palm
    by its world-axis-aligned bounding box is conservative: it can only report a
    collision that a real contact solver would also find, never miss one that
    matters for "the palm is not what is holding the box up".
    """
    m, d, _, palm, _ = kin
    R = d.geom_xmat[palm].reshape(3, 3)
    hp = np.asarray(m.geom_size[palm])
    corners = np.array([[sx * hp[0], sy * hp[1], sz * hp[2]]
                        for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    W = corners @ R.T + d.geom_xpos[palm]
    ov = np.minimum(W.max(0), box + half) - np.maximum(W.min(0), box - half)
    return float(ov.min())


def pad_report(env, kin, box: np.ndarray, half: np.ndarray) -> dict:
    """Per-pad wall margin and height, for the three long fingers and the thumb."""
    m, d, _, _, tips = kin
    out = {}
    for name, gid in tips.items():
        p = d.geom_xpos[gid]
        r = float(m.geom_size[gid][0])
        out[name] = {
            # signed distance of the pad surface from the box surface in x:
            # <= 0 means the pad overlaps the wall in x
            "wall_margin": float(abs(p[0]) - half[0] - r),
            "dz": float(p[2] - box[2]),
            "dy": float(p[1] - box[1]),
        }
    return out


# --------------------------------------------------------------------- configs
@dataclass(frozen=True)
class Pose:
    """A scripted grasp attempt. Every field is a searched dimension."""
    wx: float          # wrist ctrl x  (grasp axis is world x)
    wy: float
    wz: float          # wrist ctrl z  (-0.163 is the ctrlrange floor)
    rx: float
    ry: float
    rz: float
    a_long: float      # action for ff/mf/rf  (+1 curled, -1 extended)
    a_thumb: float     # action for th
    a_spread: float = 0.0   # per-finger stagger, to break pad-height symmetry
    sequence: str = "descend_then_close"
    lift_rate: float = 1.0  # cap on wrist ctrl rise per step (slow lift)
    n_app: int = 300
    n_desc: int = 200
    n_close: int = 160
    n_lift: int = 400
    lift_z: float = 0.14


def amps_for(p: Pose) -> np.ndarray:
    """16 finger actions: the long fingers and the thumb differ, by construction."""
    a = np.empty(16)
    a[0:12] = p.a_long
    a[12:16] = p.a_thumb
    if p.a_spread:
        # stagger the three long fingers slightly in depth so their pads land at
        # matched height on the wall instead of raking across it
        a[0:4] = p.a_long - p.a_spread
        a[4:8] = p.a_long
        a[8:12] = p.a_long + p.a_spread
    return a


# --------------------------------------------------------------------- rollout
def run_trial(env, kin, p: Pose, *, box, half) -> dict:
    """Scripted approach -> close -> lift. Returns lift/contact evidence.

    The sequence knob is what the old probe never varied: whether the wrist
    descends before or while the fingers close. Closing at the box top drives
    the pads down the wall from above; descending first wedges them against the
    pedestal lip and levers the box.
    """
    env.reset(seed=0)
    base = np.zeros(22)

    def step_to(ctrl_t, amps=None):
        tgt = np.zeros(22)
        tgt[0:6] = ctrl_t
        tgt[N_WRIST:] = ctrl_from_amp(env, amps if amps is not None else amps_for(p))
        env.step(action_from_ctrl(env, tgt))

    wrist_hi = np.array([p.wx, p.wy, max(p.wz, 0.06), p.rx, p.ry, p.rz])
    wrist_lo = np.array([p.wx, p.wy, p.wz, p.rx, p.ry, p.rz])

    # stage 1 - approach above the box with the fingers extended
    for _ in range(p.n_app):
        step_to(wrist_hi, np.full(16, -1.0))

    # stage 2/3 - the sequence under test
    if p.sequence == "descend_then_close":
        for _ in range(p.n_desc):
            step_to(wrist_lo, np.full(16, -1.0))
        for _ in range(p.n_close):
            step_to(wrist_lo)
    elif p.sequence == "close_then_descend":
        for _ in range(p.n_close):
            step_to(wrist_hi)
        for _ in range(p.n_desc):
            step_to(wrist_lo)
    elif p.sequence == "simultaneous":
        n = max(p.n_desc, p.n_close)
        for i in range(n):
            f = -1.0 + (amps_for(p)[0] + 1.0) * min(1.0, (i + 1) / n)
            step_to(wrist_lo, np.full(16, f))
        for _ in range(40):
            step_to(wrist_lo)
    else:
        raise ValueError(f"unknown sequence {p.sequence!r}")

    # stage 4 - slow pull straight up, fingers held on their setpoint
    lift_goal = np.array([p.wx, p.wy, p.lift_z, p.rx, p.ry, p.rz])
    cur = np.array(env.data.ctrl[0:6])
    streak = best_streak = 0
    clean_steps = 0
    for _ in range(p.n_lift):
        cur = np.minimum(lift_goal, cur + p.lift_rate * WRIST_STEP)
        step_to(cur)
        on_support = any(pp[0] in env.support_geoms for pp in env.trace["contact_pairs"][-1])
        held = bool(env.trace["finger_held"][-1])
        lift = float(env.trace["lift"][-1])
        clear = (not on_support) and held and lift >= CLEAN_LIFT_M
        streak = streak + 1 if clear else 0
        best_streak = max(best_streak, streak)
        clean_steps += int(clear)

    lifts = np.asarray(env.trace["lift"], dtype=float)
    peak_i = int(np.argmax(lifts))
    pairs_at_peak = [list(pp) for pp in env.trace["contact_pairs"][peak_i]]
    on_support_at_peak = any(pp[0] in env.support_geoms for pp in pairs_at_peak)
    obj = np.asarray(env.trace["obj_z"], dtype=float)
    return {
        "cfg": asdict(p),
        "peak_lift_m": float(lifts[peak_i]),
        "peak_step": peak_i,
        "final_lift_m": float(lifts[-1]),
        "box_on_support_at_peak": bool(on_support_at_peak),
        "pairs_at_peak": pairs_at_peak,
        "n_pairs_at_peak": len(pairs_at_peak),
        "steps_off_support": int(sum(
            1 for pp in env.trace["contact_pairs"]
            if not any(x[0] in env.support_geoms for x in pp))),
        "clean_streak_steps": int(best_streak),
        "clean_steps": int(clean_steps),
        "peak_finger_force_N": float(max(env.trace["finger_force"])),
        "held_fraction": float(np.mean(env.trace["finger_held"])),
        "max_step_disp_m": float(np.max(np.abs(np.diff(obj)))) if len(obj) > 1 else 0.0,
        "clean": bool(best_streak >= CLEAN_HOLD_STEPS),
    }


# ------------------------------------------------------------------- diagnosis
def diagnose() -> dict:
    """Measure the three defects that made the previous verdict untrustworthy."""
    env = _make_env(episode_len=50)
    import mujoco  # noqa: PLC0415
    m, d = env.model, env.data
    env.reset(seed=0)
    box = d.xpos[env.obj_body].copy()
    half = np.asarray(m.geom_size[env.obj_geom])
    kin = _kin_setup(env)

    # (1) polarity
    pol = {}
    for a in (-1.0, -0.5, 0.0, 0.5, 1.0):
        fk_state(env, kin, box, np.array([0, 0, -0.02, 0, 0, 0]), np.full(16, a))
        pol[f"{a:+.2f}"] = float(d.xpos[env.tip_bodies[0]][0] - box[0])

    # (2) joint aliasing. The right question is not "are these 16 values
    # distinct?" - a hand at a symmetric pose genuinely repeats values - but
    # "does each observation slot read its own joint?", i.e. do the indices
    # alias. Report both.
    correct = np.array([m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]
                        for n in FINGER_JOINTS])
    env.reset(seed=0)
    obs = env._obs()
    fq = obs[20:36]
    actual_q = d.qpos[env.finger_qpos_adr]
    obs_faithful = bool(np.allclose(fq, actual_q.astype(np.float32), atol=1e-6))

    # (3) wrist_z floor set by the palm. Measure the palm at the FLOOR pose, not
    # after the sweep has raised the wrist - otherwise the number describes a pose
    # the hand is never in and the "floor" looks far above the box.
    wz = {}
    for cz in (-0.163, -0.10, -0.02, 0.04, 0.10, 0.16):
        d.ctrl[2] = cz
        for _ in range(500):
            mujoco.mj_step(m, d)
        wz[f"{cz:+.3f}"] = float(d.xpos[env.hand_root][2])
    mujoco.mj_resetData(m, d)
    env.reset(seed=0)
    d.ctrl[2] = -0.163
    for _ in range(500):
        mujoco.mj_step(m, d)
    wrist_floor_z = float(d.xpos[env.hand_root][2])
    palm = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "palm_collision")
    R = d.geom_xmat[palm].reshape(3, 3)
    hp = m.geom_size[palm]
    corners = np.array([[sx * hp[0], sy * hp[1], sz * hp[2]]
                        for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    W = corners @ R.T + d.geom_xpos[palm]

    diag = {
        "mjcf_commit": MJCF_COMMIT,
        "box_centre_m": box.tolist(),
        "box_half_m": half.tolist(),
        "palm_bottom_z_m": float(W[:, 2].min()),
        "box_top_z_m": float(box[2] + half[2]),
        "ff_tip_x_rel_box_by_action": pol,
        "finger_joint_adr_env": env.finger_qpos_adr.tolist(),
        "finger_joint_adr_correct": correct.tolist(),
        "finger_adr_aliases": bool(len(set(env.finger_qpos_adr.tolist())) != 16),
        "obs_finger_q_matches_qpos": obs_faithful,
        "obs_finger_q_unique_of_16": int(len(set(np.round(fq, 6).tolist()))),
        "wrist_z_by_ctrl_z": wz,
        "wrist_z_floor_m": wrist_floor_z,
        "wrist_z_slope_linear": float((wz["+0.160"] - wz["-0.020"]) / (0.160 + 0.020)),
    }
    DIAG_JSON.write_text(json.dumps(diag, indent=2))
    return diag


# ----------------------------------------------------------------------- sweep
def _kinematic_candidates(limit: int = 400) -> list[Pose]:
    """Cheap pre-filter: poses where all four pads are on a wall and the palm is clear.

    Ordered cheapest-first because the palm test depends only on the wrist pose:
    the palm rests on the box top at every low wrist height, so it rejects most of
    the grid before any of the 120 finger-amplitude combinations are evaluated.
    """
    env = _make_env(episode_len=10)
    import mujoco  # noqa: PLC0415
    m, d = env.model, env.data
    env.reset(seed=0)
    box = d.xpos[env.obj_body].copy()
    half = np.asarray(m.geom_size[env.obj_geom])
    kin = _kin_setup(env)

    found: list[Pose] = []
    amp_long = np.arange(-0.6, 1.01, 0.15)
    amp_thumb = np.arange(-1.0, 0.41, 0.15)
    for rx in np.linspace(-1.2, 1.2, 13):
        for ry in np.linspace(-1.2, 1.2, 13):
            for rz in np.linspace(-1.4, 1.4, 9):
                for wz in (-0.163, -0.10, -0.02, 0.06):
                    for wx in (-0.05, 0.0, 0.05):
                        wrist = np.array([wx, 0.0, wz, rx, ry, rz])
                        # palm is finger-independent: test once per wrist pose
                        fk_state(env, kin, box, wrist, np.zeros(16))
                        if palm_clearance(env, kin, box, half) < 0.001:
                            continue
                        for al in amp_long:
                            for at in amp_thumb:
                                amps = np.concatenate([np.full(12, al), np.full(4, at)])
                                fk_state(env, kin, box, wrist, amps)
                                pr = pad_report(env, kin, box, half)
                                longs = [pr["ff_tip"], pr["mf_tip"], pr["rf_tip"]]
                                if not all(x["wall_margin"] <= 0.0 for x in longs):
                                    continue
                                if pr["th_tip"]["wall_margin"] > 0.0:
                                    continue
                                dz = np.array([x["dz"] for x in longs] + [pr["th_tip"]["dz"]])
                                # pads must be inside the box's own height band, and
                                # the two sides must be at comparable heights or the
                                # pinch rocks the box onto an edge instead of lifting
                                if np.any(np.abs(dz) > half[2] + 0.005):
                                    continue
                                if abs(dz[:3].mean() - dz[3]) > 0.020:
                                    continue
                                found.append(Pose(wx=round(wx, 4), wy=0.0,
                                                  wz=round(wz, 4), rx=round(rx, 4),
                                                  ry=round(ry, 4), rz=round(rz, 4),
                                                  a_long=round(float(al), 3),
                                                  a_thumb=round(float(at), 3)))
                                if len(found) >= limit:
                                    return found
    return found


def _trial_worker(args):
    idx, cfg = args
    env = _make_env(episode_len=1200)
    import mujoco  # noqa: PLC0415
    m, d = env.model, env.data
    env.reset(seed=0)
    box = d.xpos[env.obj_body].copy()
    half = np.asarray(m.geom_size[env.obj_geom])
    try:
        return idx, run_trial(env, _kin_setup(env), Pose(**cfg), box=box, half=half)
    except Exception as exc:  # pragma: no cover - defensive, reported not swallowed
        return idx, {"cfg": cfg, "clean": False, "error": repr(exc)}


def sweep(cands: list[Pose], workers: int = 8) -> list[dict]:
    """Expand each kinematic candidate over the sequence / lift-rate / stagger knobs."""
    jobs = []
    for i, c in enumerate(cands):
        for seq in ("descend_then_close", "close_then_descend", "simultaneous"):
            for rate in (0.35, 1.0):
                for spread in (0.0, 0.06):
                    cfg = asdict(c)
                    cfg.update(sequence=seq, lift_rate=rate, a_spread=spread)
                    jobs.append((len(jobs), cfg))
    print(f"sweeping {len(jobs)} trials on {workers} workers")
    out = []
    with mp.Pool(workers) as pool:
        for n, (idx, row) in enumerate(pool.imap_unordered(
                _trial_worker, jobs, chunksize=4), 1):
            out.append(row)
            if n % 500 == 0:
                print(f"  {n}/{len(jobs)} clean={sum(r.get('clean', False) for r in out)}")
    return out


def _record_trajectory(env, p: Pose) -> tuple[list, list]:
    """Replay one Pose's scripted rollout, recording the (obs, action) pairs.

    Deterministic by construction: the caller resets the env to seed 0 and the
    controller is a pure function of the Pose, so these are exactly the
    obs/action pairs `run_trial` stepped with. Extracted verbatim from
    `save_demo_set` so `bc_demo_set` records the identical layout from trials
    that never achieved a clean lift.
    """
    base_amp = amps_for(p)
    obs_all: list = []
    act_all: list = []

    def step_to(ctrl_t, amps):
        tgt = np.zeros(22)
        tgt[0:6] = ctrl_t
        tgt[N_WRIST:] = ctrl_from_amp(env, amps)
        a = action_from_ctrl(env, tgt)
        obs, *_ = env.step(a)
        obs_all.append(obs.copy())
        act_all.append(a.astype(np.float32))

    wrist_hi = np.array([p.wx, p.wy, max(p.wz, 0.06), p.rx, p.ry, p.rz])
    wrist_lo = np.array([p.wx, p.wy, p.wz, p.rx, p.ry, p.rz])
    for _ in range(p.n_app):
        step_to(wrist_hi, np.full(16, -1.0))
    if p.sequence == "descend_then_close":
        for _ in range(p.n_desc):
            step_to(wrist_lo, np.full(16, -1.0))
        for _ in range(p.n_close):
            step_to(wrist_lo, base_amp)
    elif p.sequence == "close_then_descend":
        for _ in range(p.n_close):
            step_to(wrist_hi, base_amp)
        for _ in range(p.n_desc):
            step_to(wrist_lo, base_amp)
    else:
        n = max(p.n_desc, p.n_close)
        for i in range(n):
            f = -1.0 + (base_amp[0] + 1.0) * min(1.0, (i + 1) / n)
            step_to(wrist_lo, np.full(16, f))
        for _ in range(40):
            step_to(wrist_lo, base_amp)
    lift_goal = np.array([p.wx, p.wy, p.lift_z, p.rx, p.ry, p.rz])
    cur = np.array(env.data.ctrl[0:6])
    for _ in range(p.n_lift):
        cur = np.minimum(lift_goal, cur + p.lift_rate * WRIST_STEP)
        step_to(cur, base_amp)
    return obs_all, act_all


def save_demo_set(rows: list[dict], top: int = 12) -> dict:
    """Materialise the winning trajectories as obs/action pairs for behaviour cloning.

    Format matches the `bc_*` slots already reserved in feasible_poses.npz:
    flat obs/actions with a per-trajectory length, plus the pose each came from.
    """
    good = sorted([r for r in rows if r.get("clean")],
                  key=lambda r: -r["clean_streak_steps"])[:top]
    if not good:
        return {"saved": 0, "reason": "no clean trajectory in the sweep"}

    env = _make_env(episode_len=1200)
    obs_all, act_all, lens = [], [], []
    for r in good:
        env.reset(seed=0)
        obs, act = _record_trajectory(env, Pose(**r["cfg"]))
        obs_all.extend(obs)
        act_all.extend(act)
        lens.append(len(obs))

    obs_arr = np.asarray(obs_all, dtype=np.float32)
    act_arr = np.asarray(act_all, dtype=np.float32)
    pose_arr = np.asarray([[Pose(**g["cfg"]).wx, Pose(**g["cfg"]).wy,
                            Pose(**g["cfg"]).wz, Pose(**g["cfg"]).rx,
                            Pose(**g["cfg"]).ry, Pose(**g["cfg"]).rz,
                            Pose(**g["cfg"]).a_long, Pose(**g["cfg"]).a_thumb]
                           for g in good], dtype=np.float32)
    np.savez_compressed(DEMO_NPZ, bc_obs=obs_arr, bc_act=act_arr,
                        bc_pose=pose_arr, bc_trajectory_len=np.asarray(lens, dtype=np.int32),
                        mjcf_commit=np.array(MJCF_COMMIT))
    return {"saved": len(good), "frames": int(len(obs_all)), "path": str(DEMO_NPZ),
            "best_streak": int(max(r["clean_streak_steps"] for r in good))}


# ------------------------------------------------------------ BC plumbing set
# The committed verdict on MJCF 4508421 is that NO clean lift exists, so
# `save_demo_set` can never emit a set in this scene. `bc_demo_set` records the
# best AVAILABLE scripted trials regardless of the clean flag so the
# BC -> PPO -> eval pipeline can be validated end to end against real env
# telemetry. The npz records `selection="best_available_not_clean"`: these
# arrays are training-plumbing data, NOT demonstration evidence of a clean
# grasp, and must never be cited as one.
_BC_SEQUENCES = ("descend_then_close", "close_then_descend", "simultaneous")


def _plumbing_pose(rng: np.random.Generator, quick: bool) -> Pose:
    """One seeded sample from the same field `_kinematic_candidates` searches.

    Bounds are copied from that grid so the plumbing trials exercise the same
    pose space the feasibility sweep did; they are not re-tuned.
    """
    kw: dict = {}
    if quick:
        # short phase budgets for schema/pipeline tests; episode_len 220 covers
        # 20 + max(20, 20) + 40 + 40 = 120 steps in every sequence
        kw = dict(n_app=20, n_desc=20, n_close=20, n_lift=40, lift_z=0.10)
    return Pose(
        wx=float(rng.choice((-0.05, 0.0, 0.05))),
        wy=0.0,
        wz=float(rng.choice((-0.163, -0.10, -0.02, 0.06))),
        rx=round(float(rng.uniform(-1.2, 1.2)), 4),
        ry=round(float(rng.uniform(-1.2, 1.2)), 4),
        rz=round(float(rng.uniform(-1.4, 1.4)), 4),
        a_long=round(float(rng.uniform(-0.6, 1.0)), 3),
        a_thumb=round(float(rng.uniform(-1.0, 0.4)), 3),
        a_spread=float(rng.choice((0.0, 0.06))),
        sequence=str(rng.choice(_BC_SEQUENCES)),
        lift_rate=float(rng.choice((0.35, 1.0))),
        **kw,
    )


def bc_demo_set(n_trials: int = 32, top: int = 12, out: str | pathlib.Path | None = None,
                seed: int = 0, quick: bool = False) -> dict:
    """Collect a BC dataset from scripted trials WITHOUT requiring a clean lift.

    Pipeline-validation input only: runs `n_trials` poses from the same field
    the feasibility sweep searches, ranks them by off-pedestal streak then peak
    lift, and records the `top` best as obs/action pairs in the exact D4 layout
    `save_demo_set` uses (`bc_obs`/`bc_act`/`bc_pose`/`bc_trajectory_len`/
    `mjcf_commit`), plus a `selection` marker declaring them non-clean.
    """
    import mujoco  # noqa: PLC0415

    out = pathlib.Path(out) if out is not None else OUT_DIR / "bc_plumbing_demos.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    episode_len = 220 if quick else 1200
    env = _make_env(episode_len=episode_len)
    m, d = env.model, env.data
    env.reset(seed=0)
    box = d.xpos[env.obj_body].copy()
    half = np.asarray(m.geom_size[env.obj_geom])
    kin = _kin_setup(env)

    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_trials):
        row = run_trial(env, kin, _plumbing_pose(rng, quick), box=box, half=half)
        rows.append(row)
        print(f"  trial {i + 1}/{n_trials}: streak={row['clean_streak_steps']} "
              f"peak={row['peak_lift_m'] * 1000:.1f}mm clean={row['clean']}", flush=True)

    rows.sort(key=lambda r: (-r["clean_streak_steps"], -r["peak_lift_m"]))
    chosen = rows[:max(1, min(top, len(rows)))]

    obs_all, act_all, lens, pose_rows = [], [], [], []
    for r in chosen:
        p = Pose(**r["cfg"])
        env.reset(seed=0)
        obs, act = _record_trajectory(env, p)
        obs_all.extend(obs)
        act_all.extend(act)
        lens.append(len(obs))
        pose_rows.append([p.wx, p.wy, p.wz, p.rx, p.ry, p.rz, p.a_long, p.a_thumb])

    np.savez_compressed(
        out,
        bc_obs=np.asarray(obs_all, dtype=np.float32),
        bc_act=np.asarray(act_all, dtype=np.float32),
        bc_pose=np.asarray(pose_rows, dtype=np.float32),
        bc_trajectory_len=np.asarray(lens, dtype=np.int32),
        mjcf_commit=np.array(MJCF_COMMIT),
        selection=np.array("best_available_not_clean"),
    )
    return {
        "path": str(out),
        "n_trials": n_trials,
        "saved": len(chosen),
        "frames": int(len(obs_all)),
        "best_streak": int(chosen[0]["clean_streak_steps"]),
        "n_clean_trials": int(sum(r["clean"] for r in rows)),
        "selection": "best_available_not_clean",
        "quick": bool(quick),
        "mjcf_commit": MJCF_COMMIT,
    }


# -------------------------------------------------------------------- verdict
def _verdict_counts(samples: int = 60000, seed: int = 3) -> tuple:
    """Scan random hand poses and count the four things that decide feasibility.

    Returns ``(n_contact, n_pads_only, n_bilat, buried_hist)``. Split out from
    `verdict` so a caller can re-derive the counts at a different sample size
    without overwriting the recorded verdict file.
    """
    """Quantify WHY no clean lift exists, so the scene change can be argued from data.

    The question is not "did my controller fail" but "is a pad-only grasp on this
    box geometrically reachable at all". A pose that puts a pad on the far wall is
    only a grasp if the rest of the finger stays outside the box; this counts how
    many of the 16 finger collision geoms are inside for each candidate, and
    whether pads ever land on OPPOSITE walls at matched height with no other
    finger geom in contact.
    """
    env = _make_env(episode_len=10)
    import mujoco  # noqa: PLC0415
    m, d = env.model, env.data
    env.reset(seed=0)
    box = d.xpos[env.obj_body].copy()
    half = np.asarray(m.geom_size[env.obj_geom])
    qa = np.array([m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]
                   for n in FINGER_JOINTS])
    lo = env.ctrl_low[N_WRIST:]
    hi = env.ctrl_high[N_WRIST:]
    tip_geom = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n + "_collision")
                for n in ("ff_tip", "mf_tip", "rf_tip", "th_tip")}
    pad_names = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) for g in tip_geom.values()}
    d.qpos[0:7] = [*box, 1.0, 0.0, 0.0, 0.0]
    rng = np.random.default_rng(seed)

    n_contact = n_pads_only = n_bilat = 0
    buried_hist: list[int] = []
    for _ in range(samples):
        w = [rng.uniform(-0.18, 0.18), rng.uniform(-0.18, 0.18),
             rng.uniform(-0.163, 0.16), rng.uniform(-1.2, 1.2),
             rng.uniform(-1.259, 1.2), rng.uniform(-1.57, 1.57)]
        amps = rng.uniform(-1, 1, N_FINGER)
        d.qpos[env.wrist_qpos_adr] = w
        d.qpos[qa] = lo + (amps + 1.0) * 0.5 * (hi - lo)
        mujoco.mj_forward(m, d)
        hits = set()
        for i in range(d.ncon):
            c = d.contact[i]
            if env.obj_geom not in (int(c.geom1), int(c.geom2)):
                continue
            o = int(c.geom2) if int(c.geom1) == env.obj_geom else int(c.geom1)
            hits.add(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, o))
        fing = {h for h in hits if h and h.endswith("_collision") and h != "palm_collision"}
        if not fing:
            continue
        n_contact += 1
        buried = fing - pad_names
        buried_hist.append(len(buried))
        if buried:
            continue
        n_pads_only += 1
        lx = np.array([d.geom_xpos[tip_geom[n]][0] - box[0]
                       for n in ("ff_tip", "mf_tip", "rf_tip")])
        thx = float(d.geom_xpos[tip_geom["th_tip"]][0] - box[0])
        lz = np.array([d.geom_xpos[tip_geom[n]][2] - box[2]
                       for n in ("ff_tip", "mf_tip", "rf_tip")])
        thz = float(d.geom_xpos[tip_geom["th_tip"]][2] - box[2])
        # A pad on a WALL means its surface straddles that wall: for a pad of
        # radius r the centre must sit within r of the wall plane. Checking only
        # the sign of x counts a pad sitting 50 mm INSIDE the box as "on the far
        # side", which is how two clearly degenerate poses were scored as pinches.
        def on_wall(x, r):
            return abs(abs(x) - half[0]) <= r
        r_l = float(m.geom_size[tip_geom["mf_tip"]][0])
        r_t = float(m.geom_size[tip_geom["th_tip"]][0])
        opposite = ((min(lx) > 0 > thx) or (max(lx) < 0 < thx))
        if not opposite:
            continue
        if not (all(on_wall(x, r_l) for x in lx) and on_wall(thx, r_t)):
            continue
        # and every pad must be within the box's own height band, or the pinch
        # grips an edge rather than a wall
        if np.any(np.abs(lz) > half[2]) or abs(thz) > half[2]:
            continue
        if abs(float(lz.mean()) - thz) <= 0.015:
            n_bilat += 1

    return n_contact, n_pads_only, n_bilat, buried_hist


def verdict(samples: int = 60000, seed: int = 3) -> dict:
    """Quantify WHY no clean lift exists, so the scene change can be argued from data.

    See `_verdict_counts` for the scan itself.
    """
    env = _make_env(episode_len=10)
    import mujoco  # noqa: PLC0415
    m, d = env.model, env.data
    n_contact, n_pads_only, n_bilat, buried_hist = _verdict_counts(samples, seed)

    # the palm-overload number, measured by settling with the z servo at its floor
    env.reset(seed=0)
    t = np.zeros(22)
    t[0:6] = [0.0, 0.0, -0.163, 0.0, 0.0, 0.0]
    t[N_WRIST:] = ctrl_from_amp(env, np.full(N_FINGER, -1.0))
    for _ in range(400):
        env.step(action_from_ctrl(env, t))
    palm_force = 0.0
    for i in range(d.ncon):
        c = d.contact[i]
        if env.obj_geom not in (int(c.geom1), int(c.geom2)):
            continue
        o = int(c.geom2) if int(c.geom1) == env.obj_geom else int(c.geom1)
        if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, o) or "") == "palm_collision":
            f = np.zeros(6)
            mujoco.mj_contactForce(m, d, i, f)
            palm_force += float(np.linalg.norm(f[:3]))

    box_weight = float(m.body_mass[env.obj_body]) * 9.81
    out = {
        "mjcf_commit": MJCF_COMMIT,
        "samples": samples,
        "poses_with_finger_contact": n_contact,
        "poses_touching_with_pads_only": n_pads_only,
        "poses_with_opposite_wall_pad_pinch": n_bilat,
        "median_buried_geoms_when_contacting": float(np.median(buried_hist)) if buried_hist else 0.0,
        "max_buried_geoms": int(max(buried_hist)) if buried_hist else 0,
        "palm_force_into_box_N": palm_force,
        "box_weight_N": box_weight,
        "palm_overload_ratio": palm_force / box_weight,
        "clean_lift_exists": bool(n_bilat > 0),
    }
    VERDICT_JSON.write_text(json.dumps(out, indent=2))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=("diagnose", "kinematic", "sweep", "report", "verdict",
                             "bc-demos", "all"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--n-trials", type=int, default=32,
                    help="bc-demos stage: how many scripted trials to run")
    ap.add_argument("--top", type=int, default=12,
                    help="bc-demos stage: how many trials to record into the npz")
    ap.add_argument("--demo-out", type=str, default=None,
                    help="bc-demos stage: npz output path (default "
                         "runs/p0/smoke4_mujoco_ppo/bc_plumbing_demos.npz)")
    ap.add_argument("--quick", action="store_true",
                    help="bc-demos stage: short phase budgets for schema/pipeline "
                         "tests; trials are ~120 steps instead of ~1060")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.stage == "bc-demos":
        print(json.dumps(bc_demo_set(n_trials=args.n_trials, top=args.top,
                                     out=args.demo_out, quick=args.quick), indent=2))
        return

    if args.stage in ("diagnose", "all"):
        diag = diagnose()
        print("diagnosis (MJCF %s):" % MJCF_COMMIT)
        print(f"  palm bottom {diag['palm_bottom_z_m']:.4f} m vs box top "
              f"{diag['box_top_z_m']:.4f} m -> palm rests on the box")
        print(f"  wrist_z floor {diag['wrist_z_floor_m']:.4f} m "
              f"(ctrlrange floor -0.163 reaches only this)")
        print(f"  ff_tip x by action: {diag['ff_tip_x_rel_box_by_action']}")
        print(f"  finger joint aliases: {diag['finger_adr_aliases']}, "
              f"obs reads its own joints: {diag['obs_finger_q_matches_qpos']}")

    if args.stage in ("kinematic", "all"):
        cands = _kinematic_candidates(limit=args.limit)
        SWEEP_JSON.write_text(json.dumps({"mjcf_commit": MJCF_COMMIT,
                                          "n_candidates": len(cands),
                                          "candidates": [asdict(c) for c in cands]},
                                         indent=2))
        print(f"kinematic candidates: {len(cands)} -> {SWEEP_JSON}")

    if args.stage in ("sweep", "all"):
        data = json.loads(SWEEP_JSON.read_text())
        cands = [Pose(**c) for c in data["candidates"]]
        rows = sweep(cands, workers=args.workers)
        n_clean = sum(r.get("clean", False) for r in rows)
        data["rows"] = rows
        data["n_trials"] = len(rows)
        data["n_clean"] = n_clean
        SWEEP_JSON.write_text(json.dumps(data, indent=2))
        print(f"clean lifts: {n_clean}/{len(rows)}")
        best = sorted([r for r in rows if r.get("clean")],
                      key=lambda r: -r["clean_streak_steps"])
        for r in best[:10]:
            c = r["cfg"]
            print(f"  streak={r['clean_streak_steps']:3d} peak={r['peak_lift_m']*1000:6.1f}mm "
                  f"wx={c['wx']:+.3f} wz={c['wz']:+.3f} rx={c['rx']:+.2f} ry={c['ry']:+.2f} "
                  f"rz={c['rz']:+.2f} aL={c['a_long']:+.2f} aT={c['a_thumb']:+.2f} "
                  f"seq={c['sequence']} rate={c['lift_rate']}")

    if args.stage in ("report", "all"):
        data = json.loads(SWEEP_JSON.read_text())
        rows = data.get("rows", [])
        good = [r for r in rows if r.get("clean")]
        info = save_demo_set(rows) if good else {"saved": 0,
                                                  "reason": "no clean trajectory"}
        data["demo_set"] = info
        SWEEP_JSON.write_text(json.dumps(data, indent=2))
        print("demo set:", info)

    if args.stage in ("verdict", "all"):
        v = verdict()
        print("verdict (MJCF %s):" % MJCF_COMMIT)
        print(f"  random poses with any finger contact : {v['poses_with_finger_contact']}")
        print(f"  ... touching with PADS ONLY           : {v['poses_touching_with_pads_only']}")
        print(f"  ... a clean opposite-wall pad pinch   : {v['poses_with_opposite_wall_pad_pinch']}")
        print(f"  finger geoms buried in the box when contacting: "
              f"median {v['median_buried_geoms_when_contacting']:.0f}, max {v['max_buried_geoms']}")
        print(f"  palm presses {v['palm_force_into_box_N']:.1f} N into a "
              f"{v['box_weight_N']:.2f} N box ({v['palm_overload_ratio']:.0f}x)")
        print(f"  CLEAN LIFT EXISTS: {v['clean_lift_exists']}")


if __name__ == "__main__":
    main()
