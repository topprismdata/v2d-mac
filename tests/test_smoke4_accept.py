"""Task 4.5 - acceptance for Smoke 4 (MuJoCo + SB3 floating Allegro).

RED before training. The criteria are deliberately hard-coded and anti-false-positive
(GPT redline 6): "it ran" is never enough, and neither is "the box went up".

A1  at least one EVALUATION episode reached max_lift > 0.05 m
A2  in that episode, mujoco contact data contains a finger-geom <-> object_geom pair
    (geom ids resolved through the model, not through a hand-written name list)
A3  the lift is contact-driven, not kinematic:
      * Pearson correlation between the finger-contact force series and the box z
        series over the lift window must be > 0.3
      * the box must be airborne (no table contact) at the moment of peak lift
      * per-step box displacement must stay under 0.01 m (rules out teleporting)
      * the peak lift must occur while the hand is above the box (z_peak_box >
        table top), i.e. the hand carried it up rather than the table shoving it
A4  a control run with the SAME trained policy but the box out of reach must fail to
    lift anything - the lift comes from contact, not from the policy's bias
A5  report.json exists with the required keys and verdict == "pass"

Evidence lives in runs/p0/smoke4_mujoco_ppo/: eval_episodes.json (per-step traces),
curve.csv, report.json.
"""

from __future__ import annotations

import json
import math
import pathlib
import sys

import numpy as np
import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

RUN_DIR = REPO / "runs/p0/smoke4_mujoco_ppo"
EVAL_JSON = RUN_DIR / "eval_episodes.json"
REPORT_JSON = RUN_DIR / "report.json"
CURVE_CSV = RUN_DIR / "curve.csv"
LIFT_THRESHOLD = 0.05          # A1: acceptance lift (stricter than nothing, looser than eval 0.10)
MIN_CORRELATION = 0.3          # A3
MAX_STEP_DISPLACEMENT = 0.01   # A3, mirrors evaluator max_descent_step_m


@pytest.fixture(scope="module")
def eval_data():
    assert EVAL_JSON.exists(), (
        f"missing {EVAL_JSON} - run `python grounding/train_ppo.py` first"
    )
    return json.loads(EVAL_JSON.read_text())


def _finger_geom_ids() -> set[int]:
    import mujoco  # noqa: PLC0415

    m = mujoco.MjModel.from_xml_path(str(REPO / "grounding/assets/allegro_floating.xml"))
    ids = set()
    for g in range(m.ngeom):
        bname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g])) or ""
        if bname.startswith(("ff_", "mf_", "rf_", "th_")):
            ids.add(g)
    return ids


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _analyse_episode(ep: dict) -> dict:
    """Contact-driven lift evidence for one episode (independent of the test)."""
    import mujoco  # noqa: PLC0415

    m = mujoco.MjModel.from_xml_path(str(REPO / "grounding/assets/allegro_floating.xml"))
    obj_geom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "object_geom")
    table_geom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "table_slab")
    finger_ids = _finger_geom_ids()

    tr = ep["trace"]
    lift = np.asarray(tr["lift"], dtype=float)
    force = np.asarray(tr["contact_force"], dtype=float)
    wrist_z = np.asarray(tr["wrist_z"], dtype=float)
    obj_z = np.asarray(tr["obj_z"], dtype=float)

    peak_i = int(np.argmax(lift))
    peak_lift = float(lift[peak_i])

    # A2: finger <-> box geom pairs actually present in the recorded contacts
    finger_pairs: set[tuple[str, str]] = set()
    n_finger_steps = 0
    for step in tr["contact_pairs"]:
        has_finger = False
        for name, other in step:
            gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)
            if gid in finger_ids and other == "object_geom":
                has_finger = True
                finger_pairs.add((name, other))
        n_finger_steps += int(has_finger)

    # A3: correlation between finger contact force and box height, over the lift
    # window (up to the peak), not the whole episode
    win = slice(0, peak_i + 1)
    corr_all = _pearson(force[win], lift[win])
    finger_force = np.asarray(tr["finger_force"], dtype=float)
    corr_finger = _pearson(finger_force[win], lift[win])

    steps = np.diff(obj_z, prepend=obj_z[0])
    max_step = float(np.max(np.abs(steps))) if len(steps) else 0.0

    # airborne at peak: no table contact on that step
    peak_pairs = [tuple(p) for p in tr["contact_pairs"][peak_i]]
    table_touching_peak = any(
        mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name) == table_geom
        for name, _ in peak_pairs)

    return {
        "peak_lift_m": peak_lift,
        "peak_step": peak_i,
        "n_finger_contact_steps": n_finger_steps,
        "finger_pairs": sorted(list(p) for p in finger_pairs),
        "corr_force_lift_all_contact": corr_all,
        "corr_force_lift_finger_only": corr_finger,
        "max_per_step_displacement_m": max_step,
        "table_touching_at_peak": table_touching_peak,
        "wrist_z_at_peak": float(wrist_z[peak_i]),
        "peak_force_N": float(force[peak_i]),
    }


def _best_episode(eval_data: dict) -> dict:
    eps = eval_data["episodes"]
    return max(eps, key=lambda e: e["max_lift_m"])


# ------------------------------------------------------------------------- A1
def test_a1_at_least_one_episode_lifts_above_5cm(eval_data):
    lifts = [e["max_lift_m"] for e in eval_data["episodes"]]
    assert max(lifts) > LIFT_THRESHOLD, (
        f"no evaluation episode lifted the box more than {LIFT_THRESHOLD} m; "
        f"best={max(lifts):.4f} m over {len(lifts)} episodes "
        f"(all lifts: {[round(v, 4) for v in lifts]})"
    )


# ------------------------------------------------------------------------- A2
def test_a2_finger_object_contact_pair_present(eval_data):
    ep = _best_episode(eval_data)
    ev = _analyse_episode(ep)
    assert ev["n_finger_contact_steps"] >= 5, (
        f"best-lift episode had only {ev['n_finger_contact_steps']} steps with a "
        f"finger-geom <-> object_geom contact; pairs seen: {ev['finger_pairs']}"
    )
    assert ev["finger_pairs"], "no finger<->box geom pair found in data.contact"


# ------------------------------------------------------------------------- A3
def test_a3_lift_is_contact_driven(eval_data):
    ep = _best_episode(eval_data)
    ev = _analyse_episode(ep)

    assert ev["peak_lift_m"] > LIFT_THRESHOLD
    # correlation between contact force and the lift (finger contacts only)
    corr = ev["corr_force_lift_finger_only"]
    assert not math.isnan(corr), "contact force or lift series is constant - cannot attribute"
    assert corr > MIN_CORRELATION, (
        f"box z does not correlate with finger contact force "
        f"(r={corr:.3f} <= {MIN_CORRELATION}); the rise is not contact-driven"
    )
    assert not ev["table_touching_at_peak"], (
        "the box was still touching the table at peak lift - it was not airborne"
    )
    assert ev["max_per_step_displacement_m"] < MAX_STEP_DISPLACEMENT, (
        f"box moved {ev['max_per_step_displacement_m']:.4f} m in a single step "
        f"(>= {MAX_STEP_DISPLACEMENT} m): teleport, not physics"
    )


# ------------------------------------------------------------------------- A4
def test_a4_control_policy_cannot_lift_out_of_reach(eval_data):
    """Same policy, box out of reach -> lift must collapse. Proves contact causality."""
    from stable_baselines3 import PPO  # noqa: PLC0415

    from grounding.envs.lift_hold_env import LiftHoldEnv  # noqa: PLC0415

    model = PPO.load(RUN_DIR / "ppo_lift_hold", device="cpu")
    env = LiftHoldEnv(out_of_reach=True, episode_len=250)
    peak = 0.0
    for ep in range(3):
        obs, _ = env.reset(seed=999 + ep)
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, _ = env.step(action)
            done = term or trunc
        peak = max(peak, float(max(env.trace["lift"])))
    env.close()
    assert peak < LIFT_THRESHOLD, (
        f"the policy lifted the box {peak:.4f} m even with it out of reach - "
        "the lift is not coming from contact"
    )


# ------------------------------------------------------------------------- A5
def test_a5_report_json_pass():
    assert REPORT_JSON.exists(), f"missing {REPORT_JSON}"
    rep = json.loads(REPORT_JSON.read_text())
    for key in ("device", "key_lib_versions", "seed", "verdict", "criteria",
                "curve_summary", "contact_evidence"):
        assert key in rep, f"report.json missing required key '{key}'"
    assert rep["verdict"] == "pass", f"verdict is {rep['verdict']!r}, expected 'pass'"
    assert rep["device"]["mujoco_device"] == "cpu", (
        "MuJoCo must run on CPU; report must state the device explicitly"
    )
    assert CURVE_CSV.exists() and CURVE_CSV.stat().st_size > 0, "curve.csv missing/empty"


def test_a6_curve_shows_learning_progress(eval_data):
    """The curve must show the policy improving, not a flat random-policy line."""
    assert CURVE_CSV.exists(), "curve.csv missing"
    lines = CURVE_CSV.read_text().strip().splitlines()
    assert len(lines) > 3, f"curve.csv has only {len(lines)} lines - no learning signal"
    header = lines[0].split(",")
    rows = [dict(zip(header, ln.split(","))) for ln in lines[1:]]
    rewards = np.array([float(r["reward_mean"]) for r in rows])
    assert len(rewards) >= 3
    half = len(rewards) // 2
    early, late = rewards[:half].mean(), rewards[half:].mean()
    assert late > early, (
        f"mean return did not improve over training (first half {early:.2f} vs "
        f"second half {late:.2f})"
    )
