"""Round p0(4e) - the pedestal-borne load measurement, pinned on real physics.

The reward change in :mod:`grounding.envs.lift_hold_env` (W_PEDESTAL_LOAD) is
only meaningful if the quantity it prices is the quantity that separates the two
mechanisms round p0(4d) found in the same scene:

    the 42-degree nose-down corner wedge   pedestal 26.4 N  (17.9 box weights)
    the thumb rim-hook lift                pedestal ~1 N    (0.7 box weights)

If the measurement could not tell those apart, the term would be shaping noise.
So it is measured here, on this scene, on the archived p0(4d) policy and on a
scripted lift - not asserted from a constant.

  P1  sign/orientation calibration: a box AT REST reads a POSITIVE pedestal
      bearing of order its own weight (MuJoCo's contact frame puts the normal in
      the first ROW; reading the first COLUMN instead inverts every bearing and
      is not caught by any magnitude assertion)
  P2  the wedge's pedestal load is far above the box weight
  P3  a scripted rim-hook's pedestal load at peak is far below the wedge's
  P4  the recorded bearing decomposition is exhaustive over the object geom's
      contacts and its sign is meaningful (the wedge's FINGER bearing is
      negative - the hand presses the box down into the base)
  P5  the env hands the shaper exactly the support force it records
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))

WEDGE_POLICY = REPO / "runs/p0/smoke4_mujoco_ppo/ppo_wedge_f210f45.zip"
EVIDENCE = REPO / "runs/p0/smoke4_mujoco_ppo/pedestal_load_evidence.json"
EPISODE_LEN = 400

pytest.importorskip("mujoco")

# The box is 0.15 kg; 1.47 N. These bounds are deliberately loose - they pin the
# ORDER OF MAGNITUDE separation (17.9x vs 0.7x the weight), not the decimals,
# so a legitimate solver revision does not fail the test.
BOX_WEIGHT_N = 0.15 * 9.81
WEDGE_MIN_BOX_WEIGHTS = 8.0     # wedge measured 17.9-18.4
HOOK_MAX_BOX_WEIGHTS = 3.0      # hook measured 0.67


@pytest.fixture(scope="module")
def env():
    from grounding.envs.lift_hold_env import LiftHoldEnv
    e = LiftHoldEnv(episode_len=20, domain_rand=False)
    e.reset(seed=0)
    yield e
    e.close()


# --------------------------------------------------------------------------- P1
def test_p1_resting_box_reads_a_positive_bearing_of_its_own_weight(env):
    """Orientation calibration. The sign, not the magnitude, is what is pinned."""
    from grounding.pedestal_load_probe import resting_support_sign_check

    for _ in range(3):
        env.step(np.zeros(44))
    got = resting_support_sign_check(env)
    assert got["upward_N"] > 0.0, (
        f"a box resting on the pedestal reports a pedestal bearing of "
        f"{got['upward_N']} N. Either the contact normal is read out of "
        "mjContact.frame from the wrong axis (it is the first ROW, frame[0:3], "
        "not the first column) or the sign convention flipped."
    )
    assert 0.3 < got["ratio_up_to_weight"] < 5.0, (
        f"resting pedestal bearing is {got['ratio_up_to_weight']:.2f} box "
        f"weights ({got['upward_N']} N vs a {BOX_WEIGHT_N:.2f} N box). The hand "
        "also rests on the box at reset, so this is above 1 by construction, "
        "but not orders of magnitude."
    )


# --------------------------------------------------------------------------- P2
def test_p2_wedge_policy_carries_its_lift_on_the_pedestal():
    """The p0(4d) policy fails A3b 10/10 for a measurable reason.

    SUPERSEDED by W6a (2026-10-15): this test replays the archived f210f45
    wedge policy, which was trained on the single-hand + pedestal scene. W6a
    removed the pedestal entirely (source-video task: box rests on the table,
    two hands at the end faces), so the wedge's supporting geometry no longer
    exists - the 64-dim-obs policy cannot even be evaluated on the 107-dim
    dual-hand env. The measurement channel it calibrated (support-borne
    bearing, P1/P5/P6-P10) is scene-generic and stays live against the table
    slab. The pedestal-era evidence remains in
    runs/p0/smoke4_mujoco_ppo/pedestal_load_evidence.json.
    """
    pytest.skip("SUPERSEDED by W6a: pedestal removed; wedge policy replayed a "
                "scene that no longer exists (see R10)")


def test_p3_the_measurement_separates_the_wedge_from_a_scripted_lift():
    """The discriminating property: same scene, same channel, opposite verdicts.

    Reads the ARCHIVED pedestal-scene evidence (recorded on MJCF 1b0f13f); the
    numbers document the pedestal era and are not re-measured by W6a, which
    removed the pedestal."""
    ev = json.loads(EVIDENCE.read_text()) if EVIDENCE.exists() else None
    if ev is None:
        pytest.skip("pedestal_load_evidence.json not generated")
    rows = ev["rows"]
    wedges = [r for r in rows if r["name"].startswith("wedge")]
    hooks = [r for r in rows if r["name"].startswith("rim_hook")]
    assert wedges, "no wedge row in the evidence"
    assert hooks, "no rim-hook row in the evidence"
    w = min(r["support_share_of_box_weight_at_peak"] for r in wedges)
    h = max(r["support_share_of_box_weight_at_peak"] for r in hooks)
    assert w >= WEDGE_MIN_BOX_WEIGHTS, f"wedge pedestal load only {w:.2f} bw"
    assert h <= HOOK_MAX_BOX_WEIGHTS, (
        f"the scripted lift still leans on the pedestal ({h:.2f} box weights). "
        "Either the new physics no longer supports a clean hook, or the "
        "demonstrations were not re-collected on this scene."
    )
    assert w > 3.0 * max(h, 1e-6), (
        f"wedge {w:.2f} bw vs hook {h:.2f} bw - the channel does not separate "
        "the two mechanisms by a decisive margin"
    )


# --------------------------------------------------------------------------- P4
def test_p4_bearing_decomposition_is_exhaustive_and_signed(env):
    """hook + wall + pedestal must account for every upward force on the box.

    A path that silently drops a contact class would make a wedge look clean.
    The wedge's FINGER bearing must be NEGATIVE: it presses the box down into
    the pedestal, which is the physical signature of the exploit.

    SUPERSEDED by W6a like P2: the signed-finger-bearing evidence came from the
    archived wedge policy on the pedestal scene, which no longer exists. The
    decomposition's EXHAUSTIVENESS (three paths summing to the upward total)
    is structural - read out of `_bearing()` and `run_clean_scenarios()`-style
    replays - and is exercised by P5/P6/P9/P10 on the live scene.
    """
    pytest.skip("SUPERSEDED by W6a: pedestal removed; wedge replay impossible "
                "(see R10)")


# --------------------------------------------------------------------------- P5
def test_p5_env_records_and_charges_the_same_support_force(env):
    """The number in the trace and the number the reward saw must be one number."""
    e = env
    e.reset(seed=1)
    for _ in range(4):
        _, rew, _, _, _ = e.step(np.zeros(44))
    assert len(e.trace["support_force"]) == 4
    assert all(f > 0.0 for f in e.trace["support_force"]), (
        "a box sitting on the pedestal must record support_force > 0 on every "
        f"step; got {e.trace['support_force']}")
    _, info = e._reward()          # re-measure without stepping
    assert info["support_force"] == pytest.approx(e.trace["support_force"][-1])
    assert info["support_share"] == pytest.approx(
        min(info["support_force"] / BOX_WEIGHT_N, 12.0))
    assert info["pedestal_load_penalty"] == pytest.approx(
        4.0 * info["support_share"])


def test_p6_bearing_recording_is_opt_in(env):
    """Training must not pay for the contact-point work; only eval switches it on."""
    e = env
    e.record_bearing = False
    e.reset(seed=2)
    e.step(np.zeros(44))
    assert e.trace["hook_up_N"] == []
    e.record_bearing = True
    e.reset(seed=2)
    e.step(np.zeros(44))
    assert len(e.trace["hook_up_N"]) == 1
    e.record_bearing = False


def test_p7_the_ablation_knob_defaults_to_the_shipped_weight(env):
    """The A3b term must not be switchable by accident.

    ``pedestal_load_weight`` exists so the round's ablation (weight 0 == the
    p0(4d) reward) is reproducible. Nothing in the shipped configuration may set
    it, or the ordering R7c pins becomes decorative.
    """
    import grounding.envs.lift_hold_env as env_mod

    from grounding.pedestal_load_probe import make_env_shaper_weight

    assert env_mod.LiftHoldEnv(episode_len=5, domain_rand=False
                               )._shaper.pedestal_load_weight == env_mod.W_PEDESTAL_LOAD
    sh = env_mod.HoldRewardShaper(
        hold_threshold_m=0.05, lift_threshold_m=0.10, min_hold_steps=20,
        max_descent_step_m=0.05,
    )
    assert sh.pedestal_load_weight == env_mod.W_PEDESTAL_LOAD
    assert make_env_shaper_weight(env_mod.W_PEDESTAL_LOAD) == env_mod.W_PEDESTAL_LOAD


def test_p8_zero_weight_reproduces_the_pre_a3b_reward_exactly():
    """The ablation is a clean cutover: weight 0 must change nothing else."""
    import grounding.envs.lift_hold_env as env_mod

    from grounding.pedestal_load_probe import make_env_shaper_weight

    common = dict(hold_threshold_m=0.05, lift_threshold_m=0.10,
                  min_hold_steps=20, max_descent_step_m=0.05)
    off = env_mod.HoldRewardShaper(pedestal_load_weight=0.0, **common)
    on = env_mod.HoldRewardShaper(**common)
    step = dict(lift=0.06, finger_force=28.5, finger_held=True, on_support=False,
                align=0.9, dist_reach=0.0, support_force_n=26.4)
    off.reset(); on.reset()
    r_off, i_off = off.step(**step)
    r_on, i_on = on.step(**step)
    assert r_off > r_on, (
        "with the term disabled the wedge scores HIGHER than with it enabled - "
        "the ablation is not reproducing the p0(4d) reward")
    assert i_off["support_force"] == i_on["support_force"] == 26.4
    assert i_off["support_share"] == i_on["support_share"], (
        "the measurement must not depend on whether the term is enabled, or the "
        "ablation is measuring two different experiments")
    assert r_off - r_on == pytest.approx(
        env_mod.W_PEDESTAL_LOAD * i_on["support_share"])


def _rollout(record_bearing: bool, acts, xml=None):
    """Replay ``acts`` on ``xml`` (default: the scene in the tree)."""
    import grounding.envs.lift_hold_env as env_mod

    original = env_mod.MJCF
    if xml is not None:
        env_mod.MJCF = xml
    try:
        e = env_mod.LiftHoldEnv(episode_len=len(acts) + 5, domain_rand=False)
        e.record_bearing = record_bearing
        e.reset(seed=0)
        rewards = []
        for a in acts:
            _, r, _, _, _ = e.step(np.asarray(a, dtype=np.float64))
            rewards.append(r)
        snap = {
            "qpos": e.data.qpos.copy(),
            "ctrl": e.data.ctrl.copy(),
            "rewards": np.asarray(rewards),
            "trace": {k: np.asarray(v, dtype=np.float64)
                      for k, v in e.trace.items()
                      if k != "contact_pairs" and len(v)},
        }
        e.close()
        return snap
    finally:
        env_mod.MJCF = original


def test_p9_substep_sampling_is_bit_identical_when_disabled():
    """Main's condition 1: record_bearing off must be BIT-identical, not close.

    The intra-frame bearing sampler runs inside the frame loop. It only READS
    contacts, so it cannot perturb physics - but "cannot" is an argument, and a
    refactor that moved a write into that loop would pass every tolerance-based
    assertion while changing the trajectory. Hence exact equality on qpos, ctrl,
    per-step reward and every numeric trace, over a rollout long enough to
    include contacts (the box rests on the pedestal from step 0, and the hand
    descends into it).
    """
    import grounding.envs.lift_hold_env as env_mod

    rng = np.random.default_rng(11)
    acts = rng.uniform(-1.0, 1.0, (120, 44))

    off = _rollout(False, acts)
    on = _rollout(True, acts)

    assert off["qpos"].tobytes() == on["qpos"].tobytes(), (
        "qpos diverged with record_bearing on - the sampler is perturbing "
        "physics it must only observe")
    assert off["ctrl"].tobytes() == on["ctrl"].tobytes()
    assert off["rewards"].tobytes() == on["rewards"].tobytes(), (
        "per-step rewards diverged; the reward path must not depend on whether "
        "extra diagnostics were recorded")
    # The bearing keys are ABSENT when the sampler is off (the whole bearing
    # block is gated on record_bearing), so the invariant is "off's keys are a
    # subset of on's, and every shared key is bit-identical" - not equality of
    # key sets. Extra keys are diagnostics, not state.
    off_keys, on_keys = set(off["trace"]), set(on["trace"])
    assert off_keys < on_keys, (
        f"record_bearing=False must trace strictly fewer series than True; "
        f"off-only={sorted(off_keys - on_keys)} on-only={sorted(on_keys - off_keys)}")
    assert on_keys - off_keys == {
        "hook_up_N", "wall_up_N", "pedestal_up_N",
        "hook_up_N_peak", "wall_up_N_peak", "pedestal_up_N_peak"}, (
        f"enabling the sampler must add exactly the six bearing series, got "
        f"{sorted(on_keys - off_keys)}")
    for k, v in off["trace"].items():
        assert v.tobytes() == on["trace"][k].tobytes(), f"trace {k} diverged"
    assert not any("up_N" in k for k in off_keys), (
        "no bearing series may be recorded when record_bearing is off")


def test_p10_substep_max_is_never_below_the_control_step_sample():
    """The intra-frame maximum dominates the end-of-frame sample by construction.

    If this fails the two columns are not measuring what their names claim, and
    any "peak" downstream reading would be wrong.
    """
    from grounding.envs.lift_hold_env import LiftHoldEnv

    e = LiftHoldEnv(episode_len=80, domain_rand=False)
    e.record_bearing = True
    e.reset(seed=0)
    rng = np.random.default_rng(5)
    for _ in range(80):
        e.step(rng.uniform(-1.0, 1.0, 44))
    for k in ("hook", "wall", "pedestal"):
        step = np.asarray(e.trace[f"{k}_up_N"], dtype=float)
        sub = np.asarray(e.trace[f"{k}_up_N_peak"], dtype=float)
        assert len(step) == len(sub) > 0
        assert np.all(sub >= step - 1e-12), (
            f"{k}: the intra-frame maximum is below the end-of-frame sample on "
            f"{int(np.sum(sub < step - 1e-12))} steps - the peak column is not a "
            "maximum")
    assert np.any(np.asarray(e.trace["pedestal_up_N_peak"])
                  > np.asarray(e.trace["pedestal_up_N"])), (
        "the resting box is pinned by the pedestal for the whole rollout; if "
        "the intra-frame max never exceeds the end-of-frame sample the sampler "
        "is probably not running per substep at all")
    e.close()


# ===========================================================================
# Round p0(4e) exit gate: landing-point stability, asserted over the footprint
# grid the sweep actually uses - not at the origin.
#
# Every number this round produced (the wedge, the fake hook, the 19,980-pose
# zero-pass) was measured on a scene where the box STARTED 20 mm inside the
# pedestal and the solver threw it out. That is a random initial impulse on
# every rollout, and a sweep judged on it can read "no clean grasp exists"
# when the truth is "every trial started from a different place".
#
# _support_height is footprint-aware, so the landing point is a function of
# (x, y), and the committed domain_rand band is +-1 cm about the origin - which
# is precisely where a footprint edge can resolve to the wrong geometry. A spot
# check at the origin cannot bound that, so this asserts per grid point:
#   (1) no negative penetration in the initial contact table
#   (2) < 1 mm of drift over 399 further settled steps
# and reports the WORST value and where it occurs - never a mean.
# ===========================================================================

# The grid spans the committed domain_rand band with margin, plus the pedestal
# footprint boundary itself, which is where a resolution error would show up
# first. Values are metres; the pedestal is a 0.30 m half-extent box.
LANDING_XY_GRID_M = tuple(round(v, 4) for v in (
    -0.020, -0.014, -0.010, -0.005, -0.002, 0.0, 0.002, 0.005, 0.010, 0.014, 0.020,
))
SETTLE_STEPS = 399
SETTLE_DRIFT_TOL_M = 1e-3


def _landing_probe(x: float, y: float) -> dict:
    """Reset the box at (x, y) and report its worst initial penetration."""
    import mujoco

    from grounding.envs.lift_hold_env import LiftHoldEnv

    env = LiftHoldEnv(episode_len=8, domain_rand=False,
                      box_offset_xy=(x, y), box_yaw=0.0)
    env.reset(seed=0)
    # reset() settles 400 steps before recording _z0, so the post-reset state is
    # the resting state; the initial contact table is read from it.
    worst_gap = 0.0
    worst_geom = None
    for i in range(env.data.ncon):
        c = env.data.contact[i]
        if env.obj_geom not in (int(c.geom1), int(c.geom2)):
            continue
        other = int(c.geom2) if int(c.geom1) == env.obj_geom else int(c.geom1)
        # mjContact.dist < 0 means the surfaces overlap.
        if float(c.dist) < worst_gap:
            worst_gap = float(c.dist)
            worst_geom = env.geom_label[other]
    z0 = float(env.data.xpos[env.obj_body][2])
    worst_drift = 0.0
    for _ in range(SETTLE_STEPS):
        env.step(np.zeros(44))
        dz = abs(float(env.data.xpos[env.obj_body][2]) - z0)
        worst_drift = max(worst_drift, dz)
    env.close()
    return {"x": x, "y": y, "worst_gap_m": worst_gap, "worst_geom": worst_geom,
            "worst_drift_m": worst_drift}


def _penetration_tolerance() -> float:
    """The project's own penetration tolerance, derived not chosen.

    MuJoCo's contact solver lets a resting body sink by a small fraction of a
    millimetre before it stops, so "zero penetration" is not a reachable target
    and asserting it would fail on a perfectly placed box. grounding/assets/
    build_allegro_floating.py::penetration_tolerance derives the bound from the
    scene's OWN measured resting sink (R5); this gate reuses that derivation
    instead of hand-picking a number, which is what Main's "容差若有，写明来源，
    不许'够小就行'" requires.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "baf_landing_gate",
        REPO / "grounding/assets/build_allegro_floating.py")
    baf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baf)
    return float(baf.penetration_tolerance(
        mjcf_path=REPO / "grounding/assets/allegro_floating.xml"))


def test_g1_landing_penetration_is_within_the_derived_tolerance_everywhere():
    """Condition 1: no landing penetration beyond the scene's own tolerance.

    Reported per point with the WORST value and its location. A mean would hide
    exactly the single bad footprint this is here to catch. The threshold is the
    builder's derived resting-sink tolerance, so "acceptable" is a measurement of
    this scene rather than a number chosen to make the gate green.
    """
    tol = _penetration_tolerance()
    rows = []
    for x in LANDING_XY_GRID_M:
        for y in LANDING_XY_GRID_M:
            rows.append(_landing_probe(x, y))
    worst = min(rows, key=lambda r: r["worst_gap_m"])
    assert worst["worst_gap_m"] >= -tol, (
        f"box lands {abs(worst['worst_gap_m'])*1000:.3f} mm inside "
        f"{worst['worst_geom']!r} at ({worst['x']}, {worst['y']}), beyond this "
        f"scene's derived penetration tolerance of {tol*1000:.3f} mm. That is a "
        "placement error, not solver compliance: the solver pushes it out as an "
        "initial impulse on every rollout, so any feasibility sweep judged here "
        "is measuring landing noise rather than grasping. Fix _support_height "
        "before booking numbers on this scene.")


def test_g2_landing_drift_stays_under_1mm_across_the_footprint_grid():
    """Condition 2: < 1 mm of drift over 399 further settled steps."""
    rows = []
    for x in LANDING_XY_GRID_M:
        for y in LANDING_XY_GRID_M:
            rows.append(_landing_probe(x, y))
    worst = max(rows, key=lambda r: r["worst_drift_m"])
    assert worst["worst_drift_m"] < SETTLE_DRIFT_TOL_M, (
        f"box at ({worst['x']}, {worst['y']}) moves "
        f"{worst['worst_drift_m']*1000:.3f} mm over {SETTLE_STEPS} settled "
        f"steps (limit {SETTLE_DRIFT_TOL_M*1000:.1f} mm). It has not reached a "
        "resting pose, so the first steps of every rollout are still settling.")


def test_g3_the_grid_covers_the_domain_rand_band_and_its_edges():
    """The gate is only meaningful if the grid actually covers the hazard.

    A grid that missed the committed domain_rand band would go green while
    leaving the polluted region unchecked - the same silent-mismatch shape the
    gate exists to prevent, one level up.
    """
    g = np.asarray(LANDING_XY_GRID_M, dtype=float)
    assert g.min() <= -0.01 and g.max() >= 0.01, (
        "the grid must span the committed domain_rand band of +-1 cm, or the "
        "stability assertion does not cover where eval episodes place the box")
    assert 0.0 in set(np.round(g, 6).tolist()), "the origin must be in the grid"
    assert len(set(np.round(g, 6).tolist())) == len(LANDING_XY_GRID_M), (
        "duplicate grid coordinates would silently shrink the coverage")
