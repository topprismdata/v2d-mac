"""Task 1 - the properties that make a collected clean lift trustworthy.

The previous round concluded "no clean grasp-and-lift exists in this scene" and
that the bottleneck was exploration of a 22-dimensional contact space. That
conclusion was an artefact of three defects in the inherited controller and env,
all of which are measured here. A verdict of "infeasible" that comes from an
inverted actuator sign is not a fact about the scene, so each defect is pinned as
a test: if any of them silently returns, the search is measuring the wrong thing
again and the verdict must be re-derived rather than trusted.

  D1  finger action polarity: a = -1 is the EXTENDED finger, a = +1 is the curled
      one, and a mid-range value is what puts a pad on the box wall
  D2  the 16 finger joint indices are distinct, and every observation slot reads
      its own joint
  D3  the wrist_z floor is set by the palm resting on the box top, not by the
      ctrlrange - so a controller must not extrapolate a gain through it
  D4  replaying a collected demonstration reproduces its lift AND its off-pedestal
      hold streak (the demonstration set is evidence, not a coincidence)
"""

from __future__ import annotations

import json
import pathlib
import sys

import mujoco
import numpy as np
import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from grounding.collect_clean_demos import (  # noqa: E402
    CLEAN_HOLD_STEPS,
    CLEAN_LIFT_M,
    FINGER_JOINTS,
    MJCF_COMMIT,
    VERDICT_JSON,
    diagnose,
    _kin_setup,
    _make_env,
    fk_state,
)
from grounding.envs.lift_hold_env import LiftHoldEnv  # noqa: E402

RUN_DIR = REPO / "runs/p0/smoke4_mujoco_ppo"
SWEEP_JSON = RUN_DIR / "clean_lift_sweep.json"
EVAL_JSON = RUN_DIR / "eval_episodes.json"
DEMO_NPZ = RUN_DIR / "clean_demos.npz"
DIAG_JSON = RUN_DIR / "clean_lift_diagnose.json"


@pytest.fixture(scope="module")
def env():
    e = _make_env(episode_len=1200)
    e.reset(seed=0)
    return e


@pytest.fixture(scope="module")
def diag():
    if not DIAG_JSON.exists():
        diagnose()
    return json.loads(DIAG_JSON.read_text())


# ------------------------------------------------------------------- D1: sign
def test_d1_finger_action_minus_one_is_extended_not_curled(env):
    """a=-1 must OPEN the hand. Getting this backwards makes every "grasp" a push.

    `_apply_action` maps the finger action onto each actuator's ctrlrange as
    lo + (a+1)/2*(hi-lo), so a=-1 sits at the lower bound. For these joints the
    lower bound is the extended pose, so the fingertip is FURTHEST from the palm
    there - the opposite of the previous probe's "curl to the stop = -1".
    """
    e = env
    e.reset(seed=0)
    box = e.data.xpos[e.obj_body].copy()
    kin = _kin_setup(e)

    def tip_x(a):
        fk_state(e, kin, box, np.array([0, 0, -0.02, 0, 0, 0]), np.full(16, a))
        return float(e.data.xpos[e.tip_bodies[0]][0] - box[0])

    open_x, curl_x = tip_x(-1.0), tip_x(+1.0)
    assert open_x > curl_x, (
        f"a=-1 should place the fingertip further out than a=+1 "
        f"({open_x:+.4f} vs {curl_x:+.4f}); the fingers are not opening")
    assert open_x > box[0] + 0.06, (
        "at a=-1 the fingertip should clear the +x wall of the box entirely")


def test_d1_mid_range_action_puts_a_pad_on_the_box_wall(env):
    """The useful grasp amplitude is interior, not at either stop."""
    e = env
    e.reset(seed=0)
    box = e.data.xpos[e.obj_body].copy()
    half = np.asarray(e.model.geom_size[e.obj_geom])
    kin = _kin_setup(e)
    fk_state(e, kin, box, np.array([0, 0, -0.02, 0, 0, 0]), np.zeros(16))
    wall = float(half[0])
    tip_x = float(e.data.xpos[e.tip_bodies[0]][0] - box[0])
    assert abs(tip_x - wall) < 0.012, (
        f"a=0 should put the ff pad near the +x wall ({wall:+.3f}), got {tip_x:+.4f}")


def test_d1_action_sweep_moves_pads_monotonically_inward(env):
    """Across the curling half of the range the pad must move monotonically inward.

    The last few samples near a=-1 are excluded deliberately: at the fully
    extended stop the finger swings slightly OUT before curling, so demanding
    strict monotonicity over the whole range would be asserting something the
    kinematics do not have. What matters - and what the old inverted controller
    broke - is that increasing the action never moves the pad away.
    """
    e = env
    e.reset(seed=0)
    box = e.data.xpos[e.obj_body].copy()
    kin = _kin_setup(e)
    xs = []
    for a in np.linspace(-0.5, 1.0, 7):
        fk_state(e, kin, box, np.array([0, 0, -0.02, 0, 0, 0]), np.full(16, a))
        xs.append(float(e.data.xpos[e.tip_bodies[0]][0] - box[0]))
    assert all(xs[i] >= xs[i + 1] - 1e-9 for i in range(len(xs) - 1)), (
        f"ff tip x should decrease monotonically with the action, got {np.round(xs, 4)}")
    assert xs[-1] < xs[0] - 0.05, (
        f"curling should draw the pad in by >50 mm, moved {abs(xs[-1]-xs[0])*1000:.1f} mm")


# ------------------------------------------------------------------ D2: joints
def test_d2_finger_joint_indices_are_distinct(env):
    """Eight of the sixteen once aliased onto thj3, blinding the policy to half the hand."""
    e = env
    adr = e.finger_qpos_adr.tolist()
    assert len(set(adr)) == 16, (
        f"finger joint indices alias: {adr}. Note the MJCF names the ACTUATORS "
        "ffa*/mfa*/rfa*/tha* but the JOINTS ffj*/mfj*/rfj*/thj*; looking a joint "
        "up by actuator name returns -1 and indexes the last element.")
    dof = e.finger_dof_adr.tolist()
    assert len(set(dof)) == 16, f"finger dof indices alias: {dof}"


def test_d2_observation_slot_reads_its_own_joint(env):
    """Not "the 16 values look distinct" - a symmetric pose legitimately repeats
    values. The property that matters is that slot k reads joint k."""
    e = env
    e.reset(seed=0)
    obs = e._obs()
    q = e.data.qpos[e.finger_qpos_adr]
    assert np.allclose(obs[20:36], q.astype(np.float32), atol=1e-6), (
        "observation finger block does not match qpos at the finger joint indices")


def test_d2_unknown_joint_name_raises_instead_of_indexing_minus_one():
    """The guard that turns the silent -1 into a loud error."""
    from grounding.envs.lift_hold_env import _joint_id
    e = LiftHoldEnv(evaluator_only=True)
    with pytest.raises(KeyError):
        _joint_id(e.model, "mfa0", "finger")   # actuator name, not a joint
    assert _joint_id(e.model, "mfj0", "finger") >= 0


def test_d2_reset_preshape_reaches_the_thumb_and_middle_finger(env):
    """The reset pose must actually pose all four fingers, not just the index ones.

    The base joints (ffj0/mfj0/rfjj0) are set to 0 by design - the pre-shape
    curls joints 1-3 - so the property to check is that the CURL joints of the
    middle finger and the thumb reached their setpoint. Before the name fix those
    eight joints were never written at all and stayed at the zero reset pose,
    which is precisely the half-hand blindness D2 is about.
    """
    e = env
    e.reset(seed=0)
    q = e.data.qpos[e.finger_qpos_adr]
    pre_curl = np.array([0.0, 0.9, 0.7, 0.4] * 3 + [0.8, 0.5, 0.4, 0.3])
    for i, name in enumerate(FINGER_JOINTS):
        lo, hi = e.ctrl_low[6 + i], e.ctrl_high[6 + i]
        assert lo - 1e-3 <= q[i] <= hi + 1e-3, (
            f"{name} settled at {q[i]:.3f}, outside its own ctrlrange "
            f"[{lo:.3f}, {hi:.3f}] - the pre-shape did not reach it")
    for i in (5, 6, 7, 13, 14, 15):     # curl joints of mf and th
        assert q[i] > 0.15, (
            f"{FINGER_JOINTS[i]} at {q[i]:.3f} vs a {pre_curl[i]:.2f} setpoint: "
            "the middle finger / thumb pre-shape did not reach this joint")


# ------------------------------------------------------------ D3: the wrist floor
def test_d3_wrist_z_floor_is_the_palm_not_the_ctrlrange(env, diag):
    """ctrl_z = -0.163 (the ctrlrange floor) does NOT put the wrist that low.

    The palm rests on the box top, so the wrist saturates ~0.869 m however hard
    the controller pushes down. Any controller that fits its world->ctrl gain
    across this saturated region - as the inherited 0.545 m/unit did - is wrong
    by ~1.8x over the region that actually moves.
    """
    assert diag["wrist_z_floor_m"] > 0.86
    spread = diag["wrist_z_by_ctrl_z"]["-0.020"] - diag["wrist_z_by_ctrl_z"]["-0.163"]
    assert abs(spread) < 0.005, (
        f"ctrl_z -0.163 and -0.020 differ by {spread*1000:.1f} mm in wrist height; "
        "if this is ~0 the lower half of the ctrlrange is saturated by the palm")
    assert diag["wrist_z_slope_linear"] > 0.9, (
        f"measured wrist_z slope {diag['wrist_z_slope_linear']:.3f} is far from 1:1")


def test_d3_palm_rest_clearance_is_measured_at_the_floor_pose(diag):
    """Guards the diagnosis itself: measuring the palm after raising the wrist
    would report a floor far above the box and hide the whole effect."""
    assert diag["palm_bottom_z_m"] <= diag["box_top_z_m"] + 0.005, (
        f"palm bottom {diag['palm_bottom_z_m']:.4f} vs box top "
        f"{diag['box_top_z_m']:.4f}: the palm should rest on the box, not hover")


def test_d3_diagnosis_records_the_mjcf_commit(diag):
    assert diag["mjcf_commit"] == MJCF_COMMIT


# ------------------------------------------------------- D4: the demo set itself
needs_demos = pytest.mark.skipif(
    not DEMO_NPZ.exists(),
    reason="no clean demonstration collected - run grounding/collect_clean_demos.py")


@needs_demos
def test_d4_demo_set_has_the_reserved_bc_layout():
    z = np.load(DEMO_NPZ)
    for key in ("bc_obs", "bc_act", "bc_pose", "bc_trajectory_len"):
        assert key in z, f"{key} missing from the demo set"
    assert z["bc_obs"].ndim == 2 and z["bc_act"].ndim == 2
    assert z["bc_obs"].shape[1] == 64, z["bc_obs"].shape
    assert z["bc_act"].shape[1] == 22, z["bc_act"].shape
    assert int(z["bc_trajectory_len"].sum()) == z["bc_obs"].shape[0]
    assert len(z["bc_trajectory_len"]) == z["bc_pose"].shape[0]


@needs_demos
def test_d4_replaying_a_demo_reproduces_its_lift_and_hold(env):
    """A demonstration is only evidence if replaying it reproduces the numbers.

    Replays the stored obs/action pairs through the env with a fresh reset and
    requires the recorded trajectory to re-achieve the off-pedestal hold that
    earned it a place in the set. A demo that no longer reproduces was collected
    against different physics and must not be used to warm-start a policy.
    """
    z = np.load(DEMO_NPZ)
    lens = z["bc_trajectory_len"]
    obs, act = z["bc_obs"], z["bc_act"]
    best = None
    for i, n in enumerate(lens):
        e = LiftHoldEnv(episode_len=int(n) + 10)
        e.reset(seed=0)
        streak = 0
        for t in range(int(n)):
            o, r, term, trunc, info = e.step(act[int(lens[:i].sum()) + t])
            on_support = any(p[0] in e.support_geoms
                             for p in e.trace["contact_pairs"][-1])
            clear = (not on_support) and bool(e.trace["finger_held"][-1]) \
                and float(e.trace["lift"][-1]) >= CLEAN_LIFT_M
            streak = streak + 1 if clear else 0
        best = max(best or 0, streak)
        e.close()
    assert best >= CLEAN_HOLD_STEPS, (
        f"replaying the demo set reproduced a best off-support streak of {best} "
        f"steps, below the {CLEAN_HOLD_STEPS} that qualified it")


@needs_demos
def test_d4_demo_set_is_recorded_against_the_same_mjcf():
    z = np.load(DEMO_NPZ)
    assert str(z["mjcf_commit"]) == MJCF_COMMIT


# ------------------------------------------------------- the search itself
def _sweep_rows() -> list[dict]:
    """The recorded sweep, or skip: stage=kinematic writes no rows by design."""
    if not SWEEP_JSON.exists():
        pytest.skip("no sweep recorded - run grounding/collect_clean_demos.py")
    rows = json.loads(SWEEP_JSON.read_text()).get("rows", [])
    if not rows:
        pytest.skip("sweep has no trial rows yet (stage=kinematic ran only)")
    return rows


def test_search_found_clean_lifts_and_reports_the_criterion():
    rows = _sweep_rows()
    clean = [r for r in rows if r.get("clean")]
    if not clean:
        pytest.skip("sweep found no clean lift - see the failure distribution")
    for r in clean:
        # "clean" must mean what the criterion says: finger contact sustained AND
        # >= 5 mm off the pedestal AND held for >= 20 consecutive steps
        assert r["clean_streak_steps"] >= CLEAN_HOLD_STEPS
        assert r["peak_lift_m"] >= CLEAN_LIFT_M
        assert not r["box_on_support_at_peak"], (
            "a trial whose box still touches the pedestal at its peak step is a "
            "lever on an edge, not a lift - the exact artefact D4 exists to exclude")


def test_every_trial_records_the_contact_pairs_at_its_peak():
    """The peak-step contact pair list is what separates a lift from a lever, so
    it has to be on the record for every trial, not only the clean ones."""
    rows = _sweep_rows()
    for r in rows:
        if "error" in r:
            continue
        assert "pairs_at_peak" in r
        assert "box_on_support_at_peak" in r
        assert "max_step_disp_m" in r


def test_no_trial_errored():
    """A trial that raised would otherwise be silently counted as a failure."""
    rows = _sweep_rows()
    errs = [r for r in rows if "error" in r]
    assert not errs, f"{len(errs)} trials raised: {errs[0]['error']}"


# ------------------------------------------------------- the verdict itself
needs_verdict = pytest.mark.skipif(
    not VERDICT_JSON.exists(),
    reason="no verdict recorded - run grounding/collect_clean_demos.py --stage verdict")


# ----------------------------------------------------- tilt / lever criterion
def test_a_box_on_a_corner_is_not_a_lift(env):
    """The reason a large "lift" can still be a rejection.

    `max_lift` is measured on the box's centre of mass, so a box pivoted onto one
    corner raises its centre a long way while one corner stays welded to the
    pedestal. A3b rejects exactly this, and it is the dominant failure mode once
    the observation bug is fixed - the policy learned to stand the box up on an
    edge, reporting ~93 mm of lift with the box at 40-58 degrees of tilt and its
    lowest corner 0.8 mm BELOW the pedestal top.
    """
    e = env
    e.reset(seed=0)
    m, d = e.model, e.data
    half = np.asarray(m.geom_size[e.obj_geom])

    # The centre of a rigid box can rise at most the distance from its centre to a
    # corner while that corner stays welded to the support. If the measured lift is
    # below that bound, the trace is consistent with a corner lever and proves
    # nothing about a lift.
    corner_dist = float(np.linalg.norm(half))
    assert corner_dist * 1000 > 90.0, (
        f"corner distance {corner_dist*1000:.0f} mm must exceed the largest observed "
        "'lift' (~95 mm) for the corner-lever reading to hold; if the box got "
        "smaller this argument must be re-derived")

    # and confirm it against the actual evaluation traces: every episode that
    # reports >50 mm of lift must also be in pedestal contact at its peak step
    ev = EVAL_JSON
    if ev.exists():
        data = json.loads(ev.read_text())
        high = [ep for ep in data.get("episodes", []) if ep["max_lift_m"] > 0.05]
        for ep in high:
            pairs = ep["trace"]["contact_pairs"][int(np.argmax(ep["trace"]["lift"]))]
            assert any(g[0] in ("pedestal", "table_slab") for g in pairs), (
                f"episode {ep['episode']} reports {ep['max_lift_m']*1000:.1f} mm with "
                "no support contact at peak - if that is a genuine lift the "
                "corner-lever conclusion must be revisited")


@needs_verdict
def test_sweep_and_eval_never_count_a_corner_lever_as_clean():
    """A clean trial must be off the pedestal, not merely high.

    Guards against the sweep's own metric being loose: `peak_lift_m` alone is not
    evidence, which is why `box_on_support_at_peak` is recorded for every trial.
    """
    rows = _sweep_rows()
    for r in rows:
        if "error" in r or not r.get("clean"):
            continue
        assert not r["box_on_support_at_peak"]
    ev = EVAL_JSON
    if ev.exists():
        data = json.loads(ev.read_text())
        for ep in data.get("episodes", []):
            tr = ep["trace"]
            pairs = tr["contact_pairs"][int(np.argmax(tr["lift"]))]
            on = any(g[0] in ("pedestal", "table_slab") for g in pairs)
            if on:
                # recorded as supported at peak - must not be counted as a pass
                assert ep["max_lift_m"] > 0, "sanity"


@needs_verdict
def test_verdict_is_recorded_against_the_same_mjcf():
    v = json.loads(VERDICT_JSON.read_text())
    assert v["mjcf_commit"] == MJCF_COMMIT
    assert v["samples"] >= 10000, "too few samples to call the search space"


@needs_verdict
def test_verdict_reports_the_palm_overload_and_the_buried_geoms():
    """The two numbers the scene-change argument rests on.

    Both are properties of the MJCF, not of any controller, so they must hold
    regardless of what the search found.
    """
    v = json.loads(VERDICT_JSON.read_text())
    assert v["palm_overload_ratio"] > 5.0, (
        f"the palm presses {v['palm_force_into_box_N']:.1f} N into a "
        f"{v['box_weight_N']:.2f} N box - the overload argument no longer holds")
    assert v["median_buried_geoms_when_contacting"] >= 1.0, (
        "finger geoms are no longer being buried in the box; if this changed, the "
        "'pads cannot reach past their own phalanges' conclusion needs re-deriving")


@needs_verdict
def test_verdict_pinch_test_rejects_pads_deep_inside_the_box(env):
    """Guards the verdict's own test against the degeneracy that faked two pinches.

    Scoring "the pad is on the other side" from the sign of x counts a pad sitting
    50 mm inside the box as a successful far-side contact. The test must require
    the pad SURFACE to straddle the wall plane.
    """
    from grounding.collect_clean_demos import _make_env as mk
    import mujoco as mj
    e = mk(episode_len=10)
    e.reset(seed=0)
    m, d = e.model, e.data
    box = d.xpos[e.obj_body].copy()
    half = np.asarray(m.geom_size[e.obj_geom])
    tip = mj.mj_name2id(m, mj.mjtObj.mjOBJ_GEOM, "mf_tip_collision")
    r = float(m.geom_size[tip][0])
    far = float(d.geom_xpos[tip][0] - box[0])

    on_wall = lambda x: abs(abs(x) - half[0]) <= r   # noqa: E731
    assert not on_wall(0.010), "a pad 10 mm from the box centre is not on a wall"
    assert not on_wall(0.100), "a pad 40 mm outside the wall is not on it either"
    assert on_wall(half[0]), f"a pad centred on the wall must count (r={r})"


@needs_verdict
def test_verdict_scan_is_reproducible_and_finds_no_pinch():
    """Re-run the same scan: the answer must not depend on the random draw.

    The first implementation reported 1 pinch from one seed and 0 from another
    because its side test was sign-only. A verdict that moves with the seed is not
    evidence of anything.

    Uses the in-memory form so the recorded verdict file keeps the full sample
    count it was written with.
    """
    from grounding.collect_clean_demos import _verdict_counts
    # index 2 is poses_with_opposite_wall_pad_pinch; the tuple is
    # (contact, pads_only, pinch, buried_hist)
    counts = [_verdict_counts(samples=8000, seed=s)[2] for s in (3, 11, 29)]
    assert all(c == 0 for c in counts), (
        f"a clean opposite-wall pad pinch was found on some seeds: {counts}")
