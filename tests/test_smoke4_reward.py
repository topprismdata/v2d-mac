"""Smoke 4 round 2 - reward-shaping properties for lift & HOLD.

The previous round trained a *flick*: 2M steps produced 0.28 m peak lifts, but the
acceptance gates A3a(i)/(ii) measure finger contact force over the lift window and
came back at duty 0.03 / mean 0.38 N (bars: 0.50 and 1.47 N). The lift was real
physics - the hand hit the box and threw it - just not a grasp. The cause is
structural in the reward, so these tests pin the three properties that make the
evaluator's ``lift_hold`` semantics ("lift AND hold for min_hold_steps") the thing
being maximised:

  R1  lift-then-drop  <  lift-and-hold        (the point of the task)
  R2  fling-and-fall  <  0                    (no profit in throwing the box away)
  R3  the reward of a hold is monotonically non-decreasing in the number of
      consecutive hold steps (holding longer must never be worse)

The shaper is a pure function of measured quantities (lift, contact force, held
flag, ...), so these run without MuJoCo and without a policy.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from grounding.envs.lift_hold_env import (  # noqa: E402
    BOX_WEIGHT_N,
    HoldRewardShaper,
    _load_evaluator,
)

EV = _load_evaluator()
LIFT_GATE = 0.05     # A1 acceptance lift
HOLD_THR = EV["hold_threshold_m"]
MIN_HOLD = int(EV["min_hold_steps"])
DT = 0.02            # control step, 50 Hz


def shaper() -> HoldRewardShaper:
    return HoldRewardShaper(
        hold_threshold_m=HOLD_THR,
        lift_threshold_m=EV["lift_threshold_m"],
        min_hold_steps=MIN_HOLD,
        max_descent_step_m=EV["max_descent_step_m"],
        box_weight_n=BOX_WEIGHT_N,
    )


def gripped(lift: float, *, force: float = 6.0, align: float = 0.0,
            dist_reach: float = 0.0, on_support: bool = False,
            support_n: float = 0.0) -> dict:
    """One control step of a real grasp holding the box at height ``lift``.

    ``support_n`` is the pedestal-borne normal force reported by the contact
    channel; it defaults to 0 so every pre-A3b assertion keeps its exact value.
    """
    return {"lift": lift, "finger_force": force, "finger_held": True,
            "on_support": on_support, "align": align, "dist_reach": dist_reach,
            "act_pen": 0.0, "vel_pen": 0.0, "support_force_n": support_n}


def free(lift: float, *, force: float = 0.0, align: float = 0.0,
         dist_reach: float = 0.0, on_support: bool = False,
         support_n: float = 0.0) -> dict:
    """One control step where the box is in flight with no finger on it."""
    return {"lift": lift, "finger_force": force, "finger_held": False,
            "on_support": on_support, "align": align, "dist_reach": dist_reach,
            "act_pen": 0.0, "vel_pen": 0.0, "support_force_n": support_n}


def run(steps: list[dict]) -> tuple[float, list[float]]:
    sh = shaper()
    sh.reset()
    rewards = []
    for m in steps:
        rew, _ = sh.step(**m)
        rewards.append(rew)
    return sum(rewards), rewards


# --------------------------------------------------------------------------- R1
def test_r1_lift_and_hold_beats_lift_then_drop():
    """Same 5 cm lift, identical approach; only the hold differs."""
    n = 40
    # approach + grasp, lift to 6 cm, then hold for n steps
    hold_steps = [gripped(HOLD_THR + 0.01) for _ in range(n)]
    # same thing, but the box is released after 3 steps and falls
    drop_steps = [gripped(HOLD_THR + 0.01) for _ in range(3)]
    drop_steps += [free(HOLD_THR + 0.01 - 0.02 * k) for k in range(n - 3)]

    r_hold, per_hold = run(hold_steps)
    r_drop, _ = run(drop_steps)

    assert r_hold > r_drop, (
        f"lift+hold returned {r_hold:.1f}, lift+drop returned {r_drop:.1f}; "
        "the reward does not prefer the thing the evaluator asks for"
    )
    assert r_hold > 0.0, "a genuine lift-and-hold must be profitable"


def test_r1b_hold_longer_than_min_hold_steps_earns_the_bonus():
    """Reaching the evaluator's min_hold_steps must be worth a discrete bonus."""
    short, _ = run([gripped(HOLD_THR + 0.01) for _ in range(MIN_HOLD - 1)])
    full, _ = run([gripped(HOLD_THR + 0.01) for _ in range(MIN_HOLD)])
    per_step = full / MIN_HOLD - short / (MIN_HOLD - 1)
    assert per_step > 0.0, (
        f"the step that completes min_hold_steps={MIN_HOLD} pays {per_step:.3f}; "
        "the reward gives no reason to keep holding that long"
    )


# --------------------------------------------------------------------------- R2
def test_r2_fling_and_free_fall_is_net_negative():
    """The round-1 exploit: hit the box, let it fly, collect the height reward.

    No finger touches the box for the whole arc, so every step must cost more than
    the altitude it shows.
    """
    arc = [free(0.02 + 0.02 * k) for k in range(8)]        # thrown up
    arc += [free(max(0.0, 0.16 - 0.03 * k)) for k in range(8)]  # falls back
    total, per_step = run(arc)
    assert total < 0.0, (
        f"free flight returned {total:.1f} over the arc (mean {sum(per_step)/len(per_step):.2f}"
        "/step) - PPO would learn to throw the box instead of holding it"
    )


def test_r2b_same_altitude_is_worth_less_unheld_than_held():
    """Altitude must be paid for by contact, not merely displayed."""
    sh = shaper(); sh.reset()
    held, _ = sh.step(**gripped(0.10, force=8.0))
    sh.reset()
    thrown, _ = sh.step(**free(0.10))
    assert held > thrown, (
        f"0.10 m held returned {held:.2f} but 0.10 m in free flight returned "
        f"{thrown:.2f}: altitude without contact is not more valuable"
    )


def test_r2c_dropping_an_established_hold_is_punished_more_the_longer_it_was():
    """A drop penalty that scales with the streak lost; a flat one-shot lets the
    policy farm 'hold briefly, drop, re-grasp'."""
    def drop_after(n: int) -> float:
        sh = shaper(); sh.reset()
        for _ in range(n):
            sh.step(**gripped(HOLD_THR + 0.01))
        rew, info = sh.step(**free(HOLD_THR - 0.02))
        assert info["drop"] == 1.0, f"no drop charged after holding {n} steps"
        return rew
    assert drop_after(MIN_HOLD) < drop_after(2), (
        "dropping a full hold costs no more than dropping a 2-step hold"
    )


# --------------------------------------------------------------------------- R3
def test_r3_hold_reward_is_monotone_non_decreasing_in_hold_steps():
    totals = []
    prev = -1e18
    for n in range(1, 4 * MIN_HOLD):
        total, _ = run([gripped(HOLD_THR + 0.01) for _ in range(n)])
        assert total >= prev - 1e-9, (
            f"cumulative reward fell from {prev:.1f} (n={n-1}) to {total:.1f} "
            f"(n={n}): holding longer became worse"
        )
        prev = total
        totals.append(total)
    assert totals[-1] > totals[0], "no growth at all from holding longer"


def test_r3b_sliding_the_box_down_while_held_breaks_the_streak():
    """Evaluator semantics: a descent faster than max_descent_step_m is not a hold."""
    sh = shaper(); sh.reset()
    for _ in range(10):
        sh.step(**gripped(HOLD_THR + 0.02))
    assert sh.hold_streak >= 10
    rew, info = sh.step(**gripped(HOLD_THR + 0.02 - 3 * EV["max_descent_step_m"]))
    assert sh.hold_streak == 0, "a fast descent did not reset the hold streak"
    assert info["slip"] > 0.0, "a descent faster than the evaluator allows is not penalised"


@pytest.mark.parametrize("lift", [-0.01, 0.0, 0.049, 0.05, 0.10, 0.30])
def test_r4_reward_is_finite_and_streak_respects_the_hold_threshold(lift: float):
    sh = shaper(); sh.reset()
    rew, info = sh.step(**gripped(lift))
    assert isinstance(rew, float)
    assert rew == rew and abs(rew) != float("inf")
    assert sh.hold_streak == (1 if lift >= HOLD_THR else 0)
    assert info["hold"] == (1.0 if lift >= HOLD_THR else 0.0)


def test_r4b_hold_credit_is_not_graded_below_the_threshold():
    """Hold credit is binary, and that is deliberate.

    A graded hold (hold_value * lift / hold_threshold) was tried in round 3 to
    give the 3.2 cm plateau a gradient toward the threshold. It backfired: the
    policy settled into a stable low partial lift and a full 2M-step run finished
    with z_peak 0.0 in every quarter of the curve. Partial credit made "lift a
    little, stay safe" strictly better than the high-variance attempt to clear
    the threshold. The W_LIFT_HELD altitude term already provides the continuous
    gradient; the hold term stays all-or-nothing.
    """
    sh = shaper(); sh.reset()
    _, info = sh.step(**gripped(0.5 * HOLD_THR))
    assert info["hold"] == 0.0, (
        "hold credit became graded below the threshold - this regressed training "
        "to z_peak 0.0; see the NOTE in HoldRewardShaper.step"
    )
    sh.reset()
    _, info = sh.step(**gripped(HOLD_THR))
    assert info["hold"] == 1.0


def test_r5_no_side_approach_bonus_in_the_reward():
    """Regression guard for a reverted shaping attempt.

    W_HOOK paid a bonus for having the palm above the box centre while carrying
    it, aimed at the geometry a 4000-pose scripted sweep says clears 5 cm. It
    collapsed exploration instead of helping: training grasp fraction fell to
    0.002-0.018 across all four quarters of a 2M-step run, versus 0.49 for the
    configuration that produced the real grasp. The env must not silently grow
    such a term again.
    """
    import grounding.envs.lift_hold_env as env_mod
    assert not hasattr(env_mod, "W_HOOK"), (
        "W_HOOK was reverted for collapsing exploration; re-adding it needs a "
        "fresh measurement, not a guess"
    )


def test_r6_shaping_survives_without_a_side_approach_term():
    """The properties that matter must hold with the shipped weight set."""
    n = 30
    hold_steps = [gripped(HOLD_THR + 0.02) for _ in range(n)]
    drop_steps = [gripped(HOLD_THR + 0.02) for _ in range(3)]
    drop_steps += [free(HOLD_THR + 0.02 - 0.02 * k) for k in range(n - 3)]
    r_hold, _ = run(hold_steps)
    r_drop, _ = run(drop_steps)
    assert r_hold > r_drop

    totals = []
    prev = -1e18
    for k in range(1, 3 * MIN_HOLD):
        total, _ = run([gripped(HOLD_THR + 0.01) for _ in range(k)])
        assert total >= prev - 1e-9, "hold reward stopped being monotone"
        prev = total
        totals.append(total)
    assert totals[-1] > totals[0]

    arc = [free(0.02 + 0.02 * k) for k in range(8)]
    arc += [free(max(0.0, 0.16 - 0.03 * k)) for k in range(8)]
    total, _ = run(arc)
    assert total < 0.0, "free flight is profitable again"


# ===========================================================================
# Round p0(4e) - A3b's intent written into the objective.
#
# Acceptance gate A3b is "airborne at peak" (tests/test_smoke4_accept.py::
# test_a3_lift_is_contact_driven asserts `not table_touching_at_peak`). Nothing in
# the reward asked for it: W_LIFT_HELD pays for altitude however the box got
# there. Round p0(4d) proved the cost of that misalignment - 2M PPO steps
# converged on a 42-degree nose-down corner wedge that passes A1/A2/A3a/A3c/A4
# with 89-95 mm of sustained lift, duty 0.93-0.95 and 28-30 N of finger force,
# and fails A3b 10/10 because the PEDESTAL is carrying the box: 26.2 N through
# the base against 24.9 N through the fingers.
#
# W_PEDESTAL_LOAD prices that. The measured values below are not invented:
#   runs/p0/smoke4_mujoco_ppo/pedestal_load_evidence.json, on MJCF 1b0f13f
#     f210f45 wedge policy   pedestal 26.4-27.0 N = 17.9-18.4 box weights at peak
#     thumb rim-hook         pedestal ~1.0 N      =  0.67 box weights at peak
#   bearing decomposition at the wedge's peak, in box weights:
#     hook -9.4 / wall 0.0 / pedestal +15.7   (the fingers push the box DOWN)
# These constants are the round's design input; if a future scene change moves
# them, the ordering assertions below must be re-derived, not re-tuned.
# ===========================================================================

WEDGE_PEDESTAL_N = 26.4      # f210f45 corner wedge, at its own peak lift
WEDGE_FINGER_N = 28.5        # ... with plenty of finger force, A3a still passes
WEDGE_LIFT_M = 0.092
HOOK_PEDESTAL_N = 1.0        # clean rim-hook, at its own peak lift
HOOK_FINGER_N = 28.5
HOOK_LIFT_M = 0.053
EPISODE = 120               # long enough for the hold ramp and one-shots to show


def bearing(lift: float, force: float, support_n: float) -> list[dict]:
    """A sustained grasp at ``lift``, held for the whole episode.

    ``support_n`` is the pedestal-borne normal force the measurement channel
    reports. Everything else - lift, finger force, grip - is held CONSTANT
    across the three scenarios so the ONLY difference between them is where the
    box's weight is being carried.
    """
    return [gripped(lift, force=force, support_n=support_n) for _ in range(EPISODE)]


def test_r7_pedestal_borne_load_is_penalised_not_merely_uncounted():
    """The new term must be strictly monotone decreasing in the pedestal load.

    This is the whole mechanism: a policy that raises the box by leaning on the
    pedestal must be able to see that doing so costs it, step after step.
    """
    ref = bearing(HOOK_LIFT_M, HOOK_FINGER_N, 0.0)
    base, _ = run(ref)
    worse, _ = run(bearing(HOOK_LIFT_M, HOOK_FINGER_N, BOX_WEIGHT_N))
    worse2, _ = run(bearing(HOOK_LIFT_M, HOOK_FINGER_N, 2 * BOX_WEIGHT_N))

    assert worse < base, (
        f"one box-weight of pedestal load ({BOX_WEIGHT_N:.2f} N) costs nothing: "
        f"{base:.1f} -> {worse:.1f}. The A3b criterion is not priced.")
    assert worse2 < worse, (
        f"the penalty is not monotone in the load: {base:.1f} -> {worse:.1f} "
        f"-> {worse2:.1f}")


def test_r7b_pedestal_only_bearing_is_worth_less_than_a_real_grasp():
    """The ordering that has to invert: the wedge must lose to the honest hook.

    Both scenarios show more than the acceptance lift bar, both keep a finger on
    the box for the whole episode, and the wedge's is HIGHER - which is exactly
    why the pre-A3b reward paid it 40% more than the hook.
    """
    r_wedge, _ = run(bearing(WEDGE_LIFT_M, WEDGE_FINGER_N, WEDGE_PEDESTAL_N))
    r_hook, _ = run(bearing(HOOK_LIFT_M, HOOK_FINGER_N, HOOK_PEDESTAL_N))

    assert WEDGE_LIFT_M > HOOK_LIFT_M and WEDGE_FINGER_N == HOOK_FINGER_N, (
        "the two scenarios must differ ONLY in pedestal load and lift, or this "
        "test is not comparing what it claims to compare")
    assert r_wedge < r_hook, (
        f"pedestal-borne lift returns {r_wedge:.1f} over {EPISODE} steps, a clean "
        f"rim-hook {r_hook:.1f}. The reward still pays the fake lift more "
        "than the real grasp - the p0(4d) wedge is back on the menu.")


def test_r7c_doing_nothing_loses_to_both_and_the_hook_wins_outright():
    """Three-way ordering: idle < pedestal-borne lift < clean grasp.

    The idle scenario is a hand that never touches the box: the box stays where
    it was put, so the pedestal carries its full weight and every grasp-derived
    term pays zero. It is the floor the other two have to beat.
    """
    idle, _ = run([free(0.0, support_n=BOX_WEIGHT_N) for _ in range(EPISODE)])
    wedge, _ = run(bearing(WEDGE_LIFT_M, WEDGE_FINGER_N, WEDGE_PEDESTAL_N))
    hook, _ = run(bearing(HOOK_LIFT_M, HOOK_FINGER_N, HOOK_PEDESTAL_N))

    assert idle < wedge < hook, (
        f"idle {idle:.1f}, pedestal-borne lift {wedge:.1f}, clean grasp "
        f"{hook:.1f}. Expected idle < wedge < hook: standing still must beat "
        "cheating, and honesty must beat both.")


def test_r7d_the_existing_contact_evidence_terms_are_not_weakened():
    """Adding the pedestal term must not have been paid for by relaxing the
    contact evidence. W_GRASP, W_GRIP, the hold streak and the one-shots keep
    the values they had before the A3b term existed.

    Checked structurally: the shipped weights are the committed ones, and a
    gripped step with no pedestal load returns EXACTLY what it returned when
    ``support_force_n`` did not exist (its default is 0).
    """
    import grounding.envs.lift_hold_env as env_mod

    assert env_mod.W_GRASP == 0.5 and env_mod.W_GRIP == 6.0
    assert env_mod.W_LIFT_HELD == 150.0 and env_mod.W_HOLD_STEP == 12.0
    assert env_mod.W_HOLD_BONUS == 60.0 and env_mod.W_LIFT_BONUS == 200.0

    sh = shaper()
    sh.reset()
    _, info = sh.step(**gripped(HOLD_THR + 0.01))
    assert info["support_force"] == 0.0, (
        "the new term's input must default to zero so pre-A3b behaviour is "
        "bit-for-bit preserved wherever the pedestal is not involved")
    assert info["pedestal_load_penalty"] == 0.0


def test_r7e_pedestal_load_is_capped_so_one_corner_cannot_dominate():
    """The cap is a stated design decision, so it is pinned.

    An uncapped linear penalty on the 17.9-box-weight wedge would cost 27x what
    the hook earns and swamp every other term; the cap keeps the inversion while
    leaving the rest of the shaping readable.
    """
    import grounding.envs.lift_hold_env as env_mod

    sh = shaper()
    sh.reset()
    _, info = sh.step(**gripped(HOOK_LIFT_M, force=HOOK_FINGER_N,
                                support_n=env_mod.W_PEDESTAL_LOAD_CAP * BOX_WEIGHT_N))
    assert info["support_share"] == env_mod.W_PEDESTAL_LOAD_CAP
    assert info["pedestal_load_penalty"] == pytest.approx(
        env_mod.W_PEDESTAL_LOAD * env_mod.W_PEDESTAL_LOAD_CAP)

    sh.reset()
    _, info = sh.step(**gripped(HOOK_LIFT_M, force=HOOK_FINGER_N,
                                support_n=10 * BOX_WEIGHT_N))
    assert info["support_share"] == pytest.approx(10.0), (
        "loads below the cap must pass through unmodified - the cap is a "
        "ceiling, not a rescaling")


def test_r7f_negative_support_force_is_clamped_not_credited():
    """Contact-force readback can only be negative through a bug. Clamping it
    turns that bug into 'no penalty' instead of a free reward."""
    sh = shaper()
    sh.reset()
    _, info = sh.step(**gripped(HOOK_LIFT_M, force=HOOK_FINGER_N, support_n=-5.0))
    assert info["support_share"] == 0.0
    assert info["pedestal_load_penalty"] == 0.0
