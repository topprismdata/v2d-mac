"""Task 4.2 - floating-Allegro lift & hold gymnasium.Env.

Action (44) = two per-hand blocks [d_wrist_pos(3), d_wrist_rot(3),
finger_joint_targets(16)], right hand first: indices 0:22 drive the right hand
exactly as the single-hand env's 22-d action did, 22:44 drive the left hand the
same way. Observation = object pose+vel, per-hand wrist pose + finger
angles/velocities + per-fingertip distance to the box (right block first, left
appended), and a 4-frame history of the object z / right-wrist z.

Hard rule (PLAN 4.2 / GPT-redline): the ONLY way anything moves is through
``data.ctrl``. ``step()`` never assigns to ``data.qpos`` / ``data.qvel``; qpos is
written exclusively in ``reset()``. ``tests/test_smoke4_env.py::test_c5_*``
enforces this statically (AST scan) and dynamically (replay determinism).

W6a: the scene carries TWO hands at the box's +-x end faces (source-video task).
The right hand keeps every MJCF name it always had; the left hand appends
``_left`` to every body/joint/actuator/geom name, so body names still start with
the finger prefixes (``ff_tip_left``) and every prefix-based contact classifier
- this env's and the frozen acceptance test's - sees both hands unchanged.
Contact classification, support decomposition and the bearing split below are
therefore hand-agnostic by construction; the reward's palm-distance and align
terms keep measuring the RIGHT palm, so every shaped value means what it meant
before W6a.

Reward is shaped around the official evaluator
(``runs/p0/assets/tissue_box_lift_hold.json``):
    lift   >= 0.10 m      hold   <= 0.05 m descent / step
    final_min_lift >= -0.01 m         min_hold_steps = 20

The shaping itself lives in :class:`HoldRewardShaper` and is tested property-wise
in ``tests/test_smoke4_reward.py``: holding must beat lifting-and-dropping, and
altitude shown in free flight must never be worth more than altitude held.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any

import mujoco
import numpy as np

try:  # gymnasium is the SB3-supported API
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover
    import gym  # type: ignore[no-redef]
    from gym import spaces  # type: ignore[no-redef]

REPO = pathlib.Path(__file__).resolve().parents[2]
MJCF = REPO / "grounding/assets/allegro_floating.xml"
EVALUATOR_JSON = REPO / "runs/p0/assets/tissue_box_lift_hold.json"

N_WRIST = 6
N_FINGER = 16
N_ACT_PER_HAND = N_WRIST + N_FINGER  # 22 - the single-hand env's whole action
N_HANDS = 2
N_ACT = N_ACT_PER_HAND * N_HANDS     # 44: right hand block first, left appended
N_ARM_ACT = 6                        # W6a-v19: 3 arm servos x 2 hands (env-written, policy-invisible)
N_CTRL = N_ACT + N_ARM_ACT           # 50: the full MuJoCo ctrl width
# observation width (see LiftHoldEnv._obs): 13 object fields, two 43-field hand
# blocks, 8 history fields
# r7d/L2A (Main ruling): +2 per-side pad normal forces - the force observation
# that lets a policy see which side is transferring weight (the static-trap
# breaker signal)
N_OBS = 13 + N_HANDS * (3 + 4 + N_FINGER + N_FINGER + 4) + 8 + 2 + 4
# W6a scene naming: the right hand keeps upstream MJCF names, the left hand
# appends this suffix to every body / joint / actuator / geom name. Finger body
# names still START with their finger prefix ("ff_tip_left"), so
# FINGER_BODY_PREFIXES below classifies both hands without modification - which
# is also what keeps tests/test_smoke4_accept.py's frozen prefix rule counting
# both hands' force into the A3a 1.4715 N sum.
HAND_SUFFIXES = ("", "_left")        # right hand first (legacy action prefix)

# finger bodies whose geoms count as "hand" for the contact-evidence criterion
FINGER_BODY_PREFIXES = ("ff_", "mf_", "rf_", "th_")
OBJECT_GEOM = "object_geom"

# --- reward weights (shaped around the official evaluator) ------------------
# Round-2 rationale. Round 1 trained a FLICK, not a grasp: 2M steps gave 0.28 m
# peak lifts but A3a came back at force duty 0.03 / mean 0.38 N (bars 0.50 and
# 1.47 N) because only 2 steps ever had a finger on the box. The reward paid
# W_LIFT * max(lift, 0) for *altitude shown*, with no term for contact, so the
# cheapest way to 0.28 m was a strike and a ballistic arc - the box was in free
# flight for the entire window, which is exactly the trace the gates reject.
# A scripted probe of the same scene reaches duty 0.88-0.94 with 9-19 N of finger
# force, so the gates are physically reachable; the reward just never asked for
# them. Four changes, all in one place:
#
#  1. ALTITUDE IS PAID ONLY WHILE GRIPPED (W_LIFT_HELD). Free flight pays nothing.
#     This closes the "throw it and let gravity do the work" exploit.
#  2. A HOLD STREAK pays per step and its value GROWS with the streak (W_HOLD_STEP
#     + W_HOLD_RAMP * min(streak, min_hold_steps) / min_hold_steps), so holding
#     longer is never worse - the property the official evaluator's
#     min_hold_steps=20 encodes.
#  3. Completing min_hold_steps pays a one-shot W_HOLD_BONUS, and a drop costs
#     W_DROP_BASE + W_DROP_RAMP * fraction_of_hold_streak_lost. A flat one-shot
#     drop penalty is farmable ("hold 3 steps, drop, re-grasp, repeat"); scaling
#     with the streak lost makes banking a long hold strictly better than churning.
#  4. W_FREE_FLIGHT charges every unheld step above hold_threshold. This is the
#     direct inverse of the flick: throwing the box now costs ~20/step for ~20
#     steps, versus ~20/step of hold reward for keeping it. Note this is NOT the
#     round-1 mistake of a cliff keyed on resting contact (lift <= -5 mm) - it
#     only ever fires while the box is ABOVE hold_threshold and NOT held, which a
#     policy that simply rests on the pedestal never triggers.
#
# W_GRASP and W_ALIGN are enablers only, deliberately small. At reset the palm is
# already centred above the box and the thumb is already touching it, so both terms
# pay out from step 0 without doing anything: at the original weights (3.0 and 2.0)
# a do-nothing policy banked +1289 return while the best scripted grasp scored 968,
# i.e. the local optimum was "stand still". W_GRIP scales with force magnitude,
# which is what pulls the policy toward the 1.47 N sufficiency bar instead of a
# feather touch; it pays the full W_GRIP rate only while the hand actually carries
# the box, and W_GRIP_PRELOAD while the box still rests on its support, for the
# same "stand still and crush it" reason.
BOX_MASS_KG = 0.15
BOX_WEIGHT_N = BOX_MASS_KG * 9.81

W_REACH = 2.0          # per metre of palm->box distance reduction (one-shot-ish delta)
W_ALIGN = 0.5          # box centred under the palm (enabler only)
W_GRASP = 0.5          # fingertips in contact with the box (enabler only)
W_GRIP = 6.0           # per box-weight of finger force while the hand carries the box
W_GRIP_PRELOAD = 0.5   # ... and while the box still rests on its support (see below)
W_LIFT_HELD = 150.0    # per metre lifted *while a finger is on the box* -> 7.5 @ 5 cm
W_HOLD_STEP = 12.0     # per held step (base value of the hold streak)
W_HOLD_RAMP = 12.0     # extra per held step, growing to +100% at min_hold_steps
W_HOLD_BONUS = 60.0    # ONE-SHOT when min_hold_steps consecutive hold steps complete
W_LIFT_BONUS = 200.0   # ONE-SHOT at the evaluator's lift_threshold (0.10 m), gripped
W_DROP_BASE = 20.0     # ONE-SHOT penalty for losing an achieved hold ...
W_DROP_RAMP = 80.0     # ... scaled by the fraction of hold streak thrown away
W_FREE_FLIGHT = 20.0   # per step above hold_threshold with NO finger on the box
W_CTRL = 0.002         # action magnitude penalty
W_VEL = 0.01           # joint velocity penalty
W_OFF_TABLE = 20.0     # per step while the box is off the table AND not held
HOLD_SLACK = 0.001     # descent tolerance added on top of max_descent_step_m

# --- W_PEDESTAL_LOAD: the A3b intent, written into the objective -------------
# Round p0(4d) is the reason this line exists. Acceptance gate A3b is
# "airborne at peak" (tests/test_smoke4_accept.py::test_a3_lift_is_contact_driven
# asserts `not table_touching_at_peak`), but NOTHING in the reward asked for it:
# W_LIFT_HELD pays for altitude however the box got there. PPO therefore
# converged on a 42-degree nose-down corner wedge that displays 89-95 mm of lift,
# holds it for 400 steps, and satisfies A1/A2/A3a/A3c/A4 - while the pedestal
# carries 26.2 N and the fingers only 24.9 N. Measured: the passing set of
# scripted trajectories (the thumb rim-hook) carries 100% of its weight through
# top-face contacts and 0% through the pedestal. The reward paid the wedge more
# than the honest grasp. This term is the difference:
#
#   penalty = W_PEDESTAL_LOAD * min(support_force_N / box_weight_N, CAP)
#
# i.e. per BOX WEIGHT of normal force the pedestal/table is still supplying to
# the box, not per newton. Normalising by the box's own weight makes it the
# "share of the load the base is still carrying" that A3b is about, and makes
# the term dimensionless in scene units. It is read from the same contact
# channel the contact-evidence terms already use (`_read_contacts`), so no new
# sensing path and no new tuning knob is introduced.
#
# The CAP is not a free parameter chosen to flatter the numbers - it is what
# keeps a single squeezed corner contact from dominating the whole return. The
# f210f45 wedge measures ~17.8 box weights on the pedestal, which at the
# uncapped rate would cost 71x what an honest hook earns. The cap keeps the
# penalty decisive without letting one corner contact swamp every other term.
#
# The WEIGHT is pinned by the ordering property test (R7c), which requires
# idle < pedestal-borne lift < clean grasp. Measured on the round's three
# scenarios over a 120-step episode (tests/test_smoke4_reward.py):
#     weight   idle   wedge   clean hook
#      1.5     -180    4542       5878      <- inversion only 1.3x
#      4.0     -480     942       5578      <- shipped: inversion 5.9x
#      5.6     -852    -276       5317      <- ordering breaks past here
# 4.0 is the strongest value that keeps all three orderable, so it is the one
# that ships; the two neighbours are recorded so the choice is auditable
# rather than a guess.
W_PEDESTAL_LOAD = 4.0        # per box-weight of pedestal-borne load, per step
W_PEDESTAL_LOAD_CAP = 12.0   # ... capped at this many box weights (see above)

# --- object placement (W4-A) ---------------------------------------------
# Where the C6 control parks the box. This is a REQUEST, not the answer: the
# resolver below is allowed to slide it (see _resolve_object_placement) and the
# slide is reported in the reset info dict.
# v23: the request corner moved to the -x/-y quadrant -- the far corner from
# BOTH mounts (the +y corner now sits next to the left mount at y=+0.10).
OUT_OF_REACH_REQUEST_XY = (0.05, 0.30)  # v24: the dual-mount keep-out
# leaves exactly one feasible pocket (near (0.08, 0.32), clearance 0.290) --
# the request starts inside the slide span of that pocket.
# The box must come to rest with its WHOLE footprint on ONE support face, at
# least this far inside that face's edge. Anything else is a placement that tips
# or falls the moment it is born.
SUPPORT_EDGE_MARGIN_M = 0.005
# Slide resolution / how far the resolver will look for a legal pose.
SUPPORT_SLIDE_STEP_M = 0.005
SUPPORT_SLIDE_SPAN_M = 0.40
# The control box must additionally lie outside this radius around the hand's
# home (the world origin). Measured on this scene: wrist travel is +-0.18 m
# (actuator ctrlrange) and the furthest fingertip reached 0.2181 m from the
# origin over the C6 flail (4 seeds x 100 steps). 0.28 m clears both with
# ~60 mm to spare, and is what makes "the control hand cannot touch it" a
# property of the PLACEMENT instead of a hope about the flail.
CONTROL_CLEARANCE_M = 0.28


def _load_evaluator() -> dict[str, float]:
    data = json.loads(EVALUATOR_JSON.read_text())["evaluator"]
    return {k: float(v) for k, v in data.items() if isinstance(v, (int, float))}


class HoldRewardShaper:
    """Episode reward state + the per-step reward itself, as a pure function.

    Split out of ``LiftHoldEnv._reward`` so the shaping can be tested against
    properties (lift-and-hold > lift-and-drop, free flight < 0, hold reward
    monotone in the streak) with no MuJoCo model and no policy - see
    ``tests/test_smoke4_reward.py``.

    All the shaping decisions live in :meth:`step`; the caller only measures
    quantities (height above the reset height, finger contact force, whether a
    finger is touching the box, palm alignment, reach progress) and hands them in.
    """

    def __init__(self, *, hold_threshold_m: float, lift_threshold_m: float,
                 min_hold_steps: int, max_descent_step_m: float,
                 box_weight_n: float = BOX_WEIGHT_N,
                 pedestal_load_weight: float = W_PEDESTAL_LOAD) -> None:
        self.hold_threshold_m = float(hold_threshold_m)
        self.lift_threshold_m = float(lift_threshold_m)
        self.min_hold_steps = int(min_hold_steps)
        self.max_descent_step_m = float(max_descent_step_m)
        self.box_weight_n = float(box_weight_n)
        # Overridable ONLY to run the ablation that justifies this term: a
        # shaper with pedestal_load_weight=0 is exactly the round-p0(4d) reward.
        # It is never the default and nothing in the shipped configuration sets
        # it, so the pinned ordering properties cannot be weakened by accident.
        self.pedestal_load_weight = float(pedestal_load_weight)
        self.reset()

    def reset(self) -> None:
        self.hold_streak = 0        # consecutive evaluator-valid hold steps
        self.best_hold_streak = 0
        self.prev_lift = 0.0
        self.lift_bonus_paid = False
        self.hold_bonus_paid = False

    # ------------------------------------------------------------------ step
    def step(self, *, lift: float, finger_force: float, finger_held: bool,
             on_support: bool, align: float, dist_reach: float,
             act_pen: float = 0.0, vel_pen: float = 0.0,
             support_force_n: float = 0.0) -> tuple[float, dict]:
        """Reward one control step. Returns ``(reward, info)``.

        ``finger_held`` must be True only when a *finger* geom is actually in
        contact with the box - that is the distinction the flick exploit lives in.

        ``support_force_n`` is the summed NORMAL force the static support
        geometry (pedestal / table) is still applying to the box on this step.
        It defaults to 0 so every property test written against the pre-A3b
        shaping keeps its exact value; the env always passes it.
        """
        held = bool(finger_held)
        held_f = 1.0 if held else 0.0
        grasp = 1.0 if finger_force > 0.0 else 0.0

        # --- hold streak: evaluator semantics -----------------------------
        # A hold step is one where the box is above hold_threshold, a finger is
        # on it, and it did not descend faster than max_descent_step_m.
        descent = max(0.0, self.prev_lift - lift)
        slipping = descent > self.max_descent_step_m + HOLD_SLACK
        if lift >= self.hold_threshold_m and held and not slipping:
            self.hold_streak += 1
        else:
            self.hold_streak = 0
        self.best_hold_streak = max(self.best_hold_streak, self.hold_streak)
        self.prev_lift = lift

        hold = 1.0 if lift >= self.hold_threshold_m else 0.0

        # --- one-shots -----------------------------------------------------
        # A lift bonus is only earned while gripped: reaching 10 cm in free
        # flight is the flick, and must not bank the same reward as a hold.
        lift_bonus = 1.0 if (lift >= self.lift_threshold_m and held
                             and not self.lift_bonus_paid) else 0.0
        if lift_bonus:
            self.lift_bonus_paid = True
        min_hold = (1.0 if (self.hold_streak >= self.min_hold_steps
                            and not self.hold_bonus_paid) else 0.0)
        if min_hold:
            self.hold_bonus_paid = True

        # --- drop: one-shot, scaled by the hold streak thrown away ---------
        lost_streak = self.best_hold_streak
        drop = 1.0 if (lost_streak > 0 and not hold) else 0.0
        drop_fraction = (lost_streak / self.min_hold_steps) if lost_streak else 0.0
        if drop:
            # the streak is gone; the policy must earn the next one honestly
            self.best_hold_streak = 0
            self.lift_bonus_paid = False
            self.hold_bonus_paid = False

        # --- the streak's per-step value grows with its own length ----------
        ramp = min(self.hold_streak, self.min_hold_steps) / max(1, self.min_hold_steps)
        hold_value = W_HOLD_STEP + W_HOLD_RAMP * ramp

        # --- free flight: altitude shown with no hand on it ----------------
        free_flight = 1.0 if (lift >= self.hold_threshold_m and not held) else 0.0

        # Grip force pays the full rate only while the hand carries the box;
        # squeezing a box that still rests on its support pays the preload rate.
        carrying = 1.0 if (held and not on_support) else 0.0
        grip_rate = W_GRIP * carrying + W_GRIP_PRELOAD * (1.0 - carrying)

        # NOTE on two shaping attempts that were tried and reverted (do not
        # re-add without re-measuring):
        #  * GRADED hold, hold_value * min(lift / hold_threshold, 1). Intended to
        #    give the 3.2 cm plateau a gradient toward the 5 cm threshold. It made
        #    "lift a little, stay safe" strictly better than the high-variance
        #    attempt to clear the bar: 2M steps ended at z_peak 0.0 in every
        #    quarter of the curve.
        #  * W_HOOK, a bonus for having the palm above the box centre while
        #    carrying it, aimed at the geometry a scripted sweep says clears 5 cm.
        #    It collapsed exploration instead: training grasp fraction fell to
        #    0.002-0.018 across all four quarters, versus 0.49 for the config
        #    that produced the real grasp.
        # The continuous gradient is already supplied by W_LIFT_HELD.
        #
        # pedestal_load is the share of the box's OWN weight that the base is
        # still carrying, capped (see W_PEDESTAL_LOAD). It is the one term that
        # prices the A3b criterion: a wedge that shows 9 cm of lift while the
        # pedestal carries ~18 box weights pays for all of them, a rim-hook that
        # carries the box entirely through its top-face contacts pays nothing.
        # Everything above it - grasp, duty, hold streak, one-shots - is left at
        # its measured value; the contact evidence is untouched.
        support_share = min(max(support_force_n, 0.0) / self.box_weight_n,
                            W_PEDESTAL_LOAD_CAP)
        rew = (
            W_REACH * dist_reach
            + W_ALIGN * align
            + W_GRASP * grasp
            + grip_rate * min(finger_force / self.box_weight_n, 3.0)
            + W_LIFT_HELD * max(lift, 0.0) * held_f
            + hold_value * hold * held_f
            + W_HOLD_BONUS * min_hold
            + W_LIFT_BONUS * lift_bonus
            - (W_DROP_BASE + W_DROP_RAMP * min(drop_fraction, 1.0)) * drop
            - W_FREE_FLIGHT * free_flight
            - self.pedestal_load_weight * support_share
            - act_pen
            - vel_pen
        )
        info = {
            "lift": lift,
            "align": align,
            "grasp": grasp,
            "hold": hold * held_f,
            "min_hold": min_hold,
            "hold_streak": float(self.hold_streak),
            "drop": drop,
            "drop_fraction": drop_fraction,
            "slip": float(slipping),
            "free_flight": free_flight,
            "supported": float(on_support),
            "finger_held": held_f,
            "contact_force": float(finger_force),
            "finger_force": float(finger_force),
            "finger_contact": grasp,
            "support_force": float(support_force_n),
            "support_share": float(support_share),
            "pedestal_load_penalty": float(
                self.pedestal_load_weight * support_share),
            "pairs": [],
        }
        return float(rew), info


def _joint_id(m, name: str, role: str) -> int:
    """Resolve a joint name, refusing to return -1.

    ``mj_name2id`` returns -1 for an unknown name, and every downstream use here
    is an array index - so ``jnt_qposadr[-1]`` silently reads the LAST joint and a
    typo degrades into wrong-but-plausible numbers instead of an error. This MJCF
    names joints ``mfj*``/``thj*`` and actuators ``mfa*``/``tha*``, so mixing the
    two vocabularies is easy and silent; see the finger-joint lookup in
    :class:`LiftHoldEnv`, which this guard exists to protect.
    """
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
    if jid < 0:
        raise KeyError(
            f"{role} joint {name!r} not found in the model; joints and actuators "
            "use different names in this MJCF (joints ffj*/mfj*/rfj*/thj*, "
            "actuators ffa*/mfa*/rfa*/tha*)")
    return jid


class _ModelOnly:
    """Tiny helper for test_c1: expose the model without a full gym env."""

    def __init__(self, xml_path: pathlib.Path) -> None:
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)

    def close(self) -> None:  # pragma: no cover - symmetry with the gym env
        pass


class LiftHoldEnv(gym.Env):
    """Floating Allegro hand lifting a tissue-box proxy off a table."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(
        self,
        *,
        out_of_reach: bool = False,
        evaluator_only: bool = False,
        episode_len: int = 250,
        ctrl_hz: int = 50,
        domain_rand: bool = True,
        box_offset_xy: tuple[float, float] | None = None,
        box_yaw: float | None = None,
        pedestal_load_weight: float = W_PEDESTAL_LOAD,
        start_state: tuple[np.ndarray, np.ndarray] | None = None,
        start_noise: tuple[float, float, float] | None = None,
        grasp_bias_nm: float = 0.0,
    ) -> None:
        super().__init__()
        self.evaluator_only = evaluator_only
        if evaluator_only:
            self.model = mujoco.MjModel.from_xml_path(str(MJCF))
            self.data = mujoco.MjData(self.model)
            return

        ev = _load_evaluator()
        self.lift_threshold_m = ev["lift_threshold_m"]
        self.hold_threshold_m = ev["hold_threshold_m"]
        self.min_hold_steps = int(ev["min_hold_steps"])
        self.max_descent_step_m = ev["max_descent_step_m"]
        self.final_min_lift_m = ev["final_min_lift_m"]
        self._shaper = HoldRewardShaper(
            hold_threshold_m=ev["hold_threshold_m"],
            lift_threshold_m=ev["lift_threshold_m"],
            min_hold_steps=int(ev["min_hold_steps"]),
            max_descent_step_m=ev["max_descent_step_m"],
            box_weight_n=BOX_MASS_KG * 9.81,
            pedestal_load_weight=pedestal_load_weight,
        )

        self.model = mujoco.MjModel.from_xml_path(str(MJCF))
        self.data = mujoco.MjData(self.model)
        self.domain_rand = domain_rand
        self.out_of_reach = out_of_reach
        # W6c: optional checkpoint start - (qpos, ctrl) captured from a
        # deterministic approach run; reset() writes them (the sanctioned
        # qpos-write point) instead of the home protocol. start_noise =
        # (finger_std rad, box_xy_std m, box_yaw_std deg) perturbs the written
        # state per reset so PPO trains a distribution, not one trajectory.
        # step() still moves the world exclusively through data.ctrl.
        self._start_state = None if start_state is None else (
            np.asarray(start_state[0], dtype=np.float64),
            np.asarray(start_state[1], dtype=np.float64))
        self._start_noise = None if start_noise is None else (
            float(start_noise[0]), float(start_noise[1]),
            float(np.deg2rad(start_noise[2])))
        # W6c-r5 (Main ruling, option a): force-controlled grasp bias. A
        # constant closing torque on the 8 lift-chain curl joints (ffj1/2 +
        # mfj1/2, both hands) via qfrc_applied - the env-equivalent of a
        # force-controlled actuator, zero XML/nu change, magnitude clipped to
        # the joint's own forcerange every step (cap-respecting by
        # construction). Rationale (rounds 3-4 closure): under pure position
        # control the vertical drag demand is structurally zero while the box
        # rests (position-servo static trap); a force bias holds the contact
        # normal constant through the lift so kinetic friction (mu 1.8) can
        # carry the weight. step() writes qfrc_applied ONLY when this is
        # nonzero; qpos/qvel stay untouched (the c5 invariant holds).
        self._grasp_bias_nm = float(grasp_bias_nm)
        self._grasp_bias_sides: tuple[float, float] | None = None
        self._grasp_bias_dofs: list[tuple[int, float]] = []
        m = self.model
        self._grasp_bias_side_dofs: list[list[tuple[int, float]]] = [[], []]
        for si, sfx in enumerate(HAND_SUFFIXES):
            for f in ("ff", "mf"):
                for i in (1, 2):
                    jid = _joint_id(m, f"{f}j{i}{sfx}", "finger")
                    aid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR,
                                            f"{f}a{i}{sfx}")
                    cap = float(m.actuator_forcerange[aid][1])
                    self._grasp_bias_dofs.append(
                        (int(m.jnt_dofadr[jid]), cap))
                    self._grasp_bias_side_dofs[si].append(
                        (int(m.jnt_dofadr[jid]), cap))
        # Explicit box placement, used by the DAgger-style demonstration
        # collector (grounding/collect_hook_demos.py) to sweep the perturbation
        # grid one cell at a time. Both default to None, i.e. the committed
        # reset behaviour: domain_rand's +-1 cm / +-0.1 rad, or dead centre when
        # domain_rand is off. Keeping them as init args (rather than writing
        # qpos from outside) means the placement still goes through reset's
        # own settle loop, so a perturbed box is settled, not dropped.
        self.box_offset_xy = (None if box_offset_xy is None
                              else (float(box_offset_xy[0]), float(box_offset_xy[1])))
        self.box_yaw = None if box_yaw is None else float(box_yaw)

        # timestep 0.002 s -> 0.02 s control step at ctrl_hz=50
        self.frame_skip = max(1, int(round(1.0 / (ctrl_hz * self.model.opt.timestep))))
        self.dt = self.model.opt.timestep * self.frame_skip

        self.max_episode_steps = episode_len

        # ---- name lookups ------------------------------------------------
        m, d = self.model, self.data
        self.obj_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "object")
        self.obj_geom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, OBJECT_GEOM)
        # one root / palm per hand, right first (W6a); self.hand_root and
        # self.palm stay the RIGHT hand's for every legacy measurement channel
        self.hand_roots = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY,
                                             f"hand_root{sfx}")
                           for sfx in HAND_SUFFIXES]
        self.palms = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"palm{sfx}")
                      for sfx in HAND_SUFFIXES]
        self.hand_root = self.hand_roots[0]
        self.palm = self.palms[0]
        self.obj_qpos_adr = int(m.jnt_qposadr[mujoco.mj_name2id(
            m, mujoco.mjtObj.mjOBJ_JOINT, "object_free")])
        self.obj_qvel_adr = int(m.jnt_dofadr[mujoco.mj_name2id(
            m, mujoco.mjtObj.mjOBJ_JOINT, "object_free")])
        # per-hand wrist joint / dof addresses: 6 slides+hinges each, in the
        # action's in-block order (x, y, z, rx, ry, rz)
        self.wrist_qpos_adr = [[int(m.jnt_qposadr[mujoco.mj_name2id(
            m, mujoco.mjtObj.mjOBJ_JOINT, f"wrist_{a}{sfx}")])
            for a in ("x", "y", "z", "rx", "ry", "rz")] for sfx in HAND_SUFFIXES]
        self.wrist_dof_adr = [[int(m.jnt_dofadr[mujoco.mj_name2id(
            m, mujoco.mjtObj.mjOBJ_JOINT, f"wrist_{a}{sfx}")])
            for a in ("x", "y", "z", "rx", "ry", "rz")] for sfx in HAND_SUFFIXES]
        # JOINT names are ffj*/mfj*/rfj*/thj* (+ *_left); the MJCF names the
        # ACTUATORS ffa*/mfa*/rfa*/tha* (+ *_left), and looking a joint up by an
        # actuator name returns -1, which then indexes the LAST element of
        # jnt_qposadr - so all eight middle-finger and thumb entries silently
        # collapsed onto thj3. The reset pre-shape never reached those eight
        # joints and the observation carried 7 duplicated values instead of 16
        # distinct angles, blinding any policy to half the hand. `_joint_id`
        # makes that failure loud instead. Right-hand block first, left appended.
        finger_joint_names = tuple(
            f"{f}j{i}{sfx}" for sfx in HAND_SUFFIXES
            for f in ("ff", "mf", "rf", "th") for i in range(4))
        self.finger_joint_names = finger_joint_names
        self.finger_qpos_adr = np.array(
            [m.jnt_qposadr[_joint_id(m, n, "finger")] for n in finger_joint_names],
            dtype=int)
        self.finger_dof_adr = np.array(
            [m.jnt_dofadr[_joint_id(m, n, "finger")] for n in finger_joint_names],
            dtype=int)
        assert len(set(self.finger_qpos_adr.tolist())) == N_FINGER * N_HANDS, (
            "finger joint qpos indices alias each other - a joint name did not "
            f"resolve: {self.finger_qpos_adr.tolist()}")
        self.tip_bodies = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY,
                                             f"{f}_tip{sfx}")
                           for sfx in HAND_SUFFIXES
                           for f in ("ff", "mf", "rf", "th")]
        self.finger_geom_ids = {
            int(g) for g in range(m.ngeom)
            if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g])) or "")
            .startswith(FINGER_BODY_PREFIXES)
        }
        # collision geoms in the menagerie hand are unnamed; index them by id and
        # give the finger-side ones a stable synthetic label for the trace.
        self.geom_name_to_id: dict[str, int] = {}
        for g in range(m.ngeom):
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g)
            if name:
                self.geom_name_to_id[name] = g
        self.geom_label: list[str] = []
        for g in range(m.ngeom):
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g)
            if not name:
                bname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY,
                                          int(m.geom_bodyid[g])) or "world"
                name = f"{bname}#geom{g}"
                self.geom_name_to_id[name] = g
            self.geom_label.append(name)
        self.ctrl_low = m.actuator_ctrlrange[:N_ACT, 0].copy()
        self.ctrl_high = m.actuator_ctrlrange[:N_ACT, 1].copy()
        # world-body box geoms the object can rest on (table slab, pedestal, ...)
        self.support_geoms: set[str] = set()
        for g in range(m.ngeom):
            if int(m.geom_bodyid[g]) != 0 or g == self.obj_geom:
                continue
            if int(m.geom_type[g]) != int(mujoco.mjtGeom.mjGEOM_BOX):
                continue
            self.support_geoms.add(self.geom_label[g])
        assert self.support_geoms, "scene defines no support geometry"
        # The same geoms as axis-aligned faces (x0, x1, y0, y1, top) so that
        # placement can be asked about a FOOTPRINT rather than a point - see
        # _support_height.
        self.support_faces: list[tuple[str, float, float, float, float, float]] = []
        for g in range(m.ngeom):
            if int(m.geom_bodyid[g]) != 0 or g == self.obj_geom:
                continue
            if int(m.geom_type[g]) != int(mujoco.mjtGeom.mjGEOM_BOX):
                continue
            pos, half = m.geom_pos[g], m.geom_size[g]
            self.support_faces.append((
                self.geom_label[g],
                float(pos[0] - half[0]), float(pos[0] + half[0]),
                float(pos[1] - half[1]), float(pos[1] + half[1]),
                float(pos[2] + half[2])))
        # Box footprint: the union of EVERY geom on the object body, so a scene
        # that grows the proxy (e.g. a lip flange) cannot be placed by a stale
        # half-extent that no longer matches what is drawn.
        self.box_half_z = float(m.geom_size[self.obj_geom][2])
        self.box_half_xy = self._object_footprint_half(0.0)

        # ---- spaces ------------------------------------------------------
        self.action_space = spaces.Box(-1.0, 1.0, (N_ACT,), dtype=np.float32)
        n_obs = N_OBS
        self.n_obs = n_obs
        self.observation_space = spaces.Box(-np.inf, np.inf, (n_obs,), dtype=np.float32)

        # ---- episode bookkeeping ----------------------------------------
        self._step_count = 0
        self._z0 = 0.0
        self._z_peak = 0.0
        self._prev_dist = None
        self._prev_obj_pos = None
        self._ctrl_ref = np.zeros(N_ACT)
        self._history = np.zeros(8, dtype=np.float64)
        self._rng = np.random.default_rng(0)
        self.last_contact_pairs: list[tuple[str, str]] = []
        self.last_contact_force = 0.0
        self.last_placement: dict = {}
        # Per-step bearing decomposition is only recorded when asked for: it needs
        # the contact POINTS (to tell a top-face hook from side-wall friction),
        # not just the forces, so it costs more than the rest of the contact
        # loop. Training leaves it off; the evaluation run switches it on so
        # every episode can report where the box's weight was actually going.
        self.record_bearing = False
        # Substep-resolution bearing maxima for the step that just ran, or None
        # when record_bearing is off. Read only after step(); see the sampling
        # comment there for the alignment with other probes.
        self._substep_bearing: dict[str, float] | None = None
        self.episode_info: dict[str, Any] = {}
        # full per-step trace of the most recent episode (used by the acceptance test)
        self.trace: dict[str, list] = {}

    # ------------------------------------------------------------------ obs
    def _obs(self) -> np.ndarray:
        m, d = self.model, self.data
        obj_pos = d.xpos[self.obj_body].copy()
        obj_quat = d.xquat[self.obj_body].copy()
        obj_linvel = d.qvel[self.obj_qvel_adr: self.obj_qvel_adr + 3].copy()
        obj_angvel = d.qvel[self.obj_qvel_adr + 3: self.obj_qvel_adr + 6].copy()
        # one block per hand (right first): wrist pose relative to the box, the
        # hand's 16 finger angles/velocities, its 4 fingertip distances
        hand_blocks = []
        for h in range(N_HANDS):
            lo, hi = h * N_FINGER, (h + 1) * N_FINGER
            wrist_pos = d.xpos[self.hand_roots[h]].copy()
            wrist_quat = d.xquat[self.hand_roots[h]].copy()
            finger_q = d.qpos[self.finger_qpos_adr[lo:hi]].copy()
            finger_v = d.qvel[self.finger_dof_adr[lo:hi]].copy()
            tip_dist = np.array(
                [np.linalg.norm(d.xpos[b] - obj_pos)
                 for b in self.tip_bodies[h * 4: h * 4 + 4]], dtype=np.float64)
            hand_blocks += [wrist_pos - obj_pos, wrist_quat,
                            finger_q, finger_v, tip_dist]
        self._history = np.concatenate(
            [[obj_pos[2] - self._z0,
              d.xpos[self.hand_roots[0]][2] - self._z0],
             self._history[:-2]])

        return np.concatenate([
            obj_pos - np.array([0.0, 0.0, self._z0]),  # 3  box xyz relative to start
            obj_quat,  # 4
            obj_linvel,  # 3
            obj_angvel,  # 3
        ] + hand_blocks + [
            self._history,  # 8  4-frame history of (box dz, right-wrist dz)
            self._pad_normals_obs(),  # 2  per-side pad normal (N) (r7d/L2A)
            self._closure_extras(),  # 4  v21 coupled-closure: yaw/table-contact/slip/clear
        ]).astype(np.float32)

    def _closure_extras(self) -> np.ndarray:
        """v21 coupled-closure observables (W6cLift20): box yaw (deg/10),
        residual table support (N, /1.4715 = weight units), cumulative xy
        slip (m, /0.01), and box-bottom clearance (m, /0.005 = the gate
        scale). Named-addressed append; see N_OBS."""
        m, d = self.model, self.data
        q = d.xquat[self.obj_body]
        yaw_deg = float(np.degrees(2.0 * np.arctan2(q[3], q[0])))
        slip = float(np.linalg.norm(
            d.xpos[self.obj_body][:2] - getattr(self, "_v21_xy0",
                                                d.xpos[self.obj_body][:2])))
        clear = float(d.xpos[self.obj_body][2] - self.box_half_z
                      - (self._z0 - self.box_half_z))
        sup = 0.0
        for ci in range(d.ncon):
            c = d.contact[ci]
            g1, g2 = int(c.geom1), int(c.geom2)
            if self.obj_geom not in (g1, g2):
                continue
            other = g2 if g1 == self.obj_geom else g1
            if self.geom_label[other] in self.support_geoms:
                f = np.zeros(6)
                mujoco.mj_contactForce(m, d, ci, f)
                sup += float(np.linalg.norm(f[:3]))
        return np.array([yaw_deg / 10.0, sup / 1.4715, slip / 0.01,
                         clear / 0.005], dtype=np.float64)

    def _pad_normals_obs(self) -> np.ndarray:
        """Per-side summed normal force (N) on hand-vs-box contacts."""
        n_r, n_l = 0.0, 0.0
        for ci in range(self.data.ncon):
            c = self.data.contact[ci]
            g1, g2 = int(c.geom1), int(c.geom2)
            if self.obj_geom not in (g1, g2):
                continue
            other = g2 if g1 == self.obj_geom else g1
            if int(other) not in self.finger_geom_ids:
                continue
            f = np.zeros(6)
            mujoco.mj_contactForce(self.model, self.data, ci, f)
            left = (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY,
                                      int(self.model.geom_bodyid[other])) or ""
                    ).endswith("_left")
            if left:
                n_l += float(np.linalg.norm(f[:3]))
            else:
                n_r += float(np.linalg.norm(f[:3]))
        return np.array([n_r, n_l], dtype=np.float64)

    def observation_space_shape(self) -> tuple[int, ...]:
        return self.observation_space.shape  # type: ignore[return-value]

    # ---------------------------------------------------------------- reset
    def _reset_home_state(self, d) -> None:
        """Home-pose episode start: box placement protocol + hand pre-shape.

        Extracted verbatim from reset() when the W6c hold-seed start arrived;
        reset() chooses between this and the seed state, then runs the shared
        settle + bookkeeping."""
        m = self.model
        # --- object placement (allowed: reset may write qpos) -------------
        # out_of_reach asks for the box 0.35 m along +y, far outside the hand's
        # workspace. That request is NOT satisfiable on this scene and is not
        # honoured blindly any more: the pedestal ends at y=0.30 m and the slab
        # at y=0.45 m, so only a 150 mm-wide band of bare table exists off the
        # pedestal, and the box is 240 mm long in y. The nearest fully
        # supported pose on that axis is y=0.18 m - inside the hand's reach,
        # which makes the control meaningless (measured: 24 finger-box contacts
        # over the C6 flail at y=0.18). _resolve_object_placement therefore
        # rejects the +y axis and slides the box to +x, where the 120 mm-wide
        # box does fit in the same 150 mm band (x in [0.36, 0.39] m; the
        # max-margin pick is 0.375 m, clearing the hand by 0.315 m).
        x_off, y_off = OUT_OF_REACH_REQUEST_XY if self.out_of_reach else (0.0, 0.0)
        clearance = CONTROL_CLEARANCE_M if self.out_of_reach else 0.0
        yaw = 0.0
        quat = np.array([1.0, 0.0, 0.0, 0.0])
        if self.domain_rand and not self.out_of_reach:
            x_off += float(self._rng.uniform(-0.01, 0.01))
            y_off += float(self._rng.uniform(-0.01, 0.01))
            yaw = float(self._rng.uniform(-0.10, 0.10))
            quat = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
        # explicit placement overrides the stochastic protocol entirely (and is
        # usable with domain_rand=False, which is how the collector holds the
        # commanded pose exact and perturbs only the variable under test)
        if self.box_offset_xy is not None:
            x_off += self.box_offset_xy[0]
            y_off += self.box_offset_xy[1]
        if self.box_yaw is not None:
            yaw = self.box_yaw
            quat = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
        px, py, pz, placement = self._resolve_object_placement(
            x_off, y_off, yaw, min_clearance=clearance)
        obj_pos = np.array([px, py, pz])
        self.last_placement = placement
        d.qpos[self.obj_qpos_adr: self.obj_qpos_adr + 7] = np.concatenate([obj_pos, quat])

        # --- hands: home pose with a light grasp pre-shape ------------------
        # W6a: both hands start at the MJCF's own home (wrist joints at 0 - the
        # body euler carries the end-face tilt), fingers half-curled. The old
        # single-hand reset dropped wrist_z by 0.02 to meet the raised box; the
        # dual-hand home is already placed, so the slides stay at 0.
        # W6a-v8: matches the finger home springs (FINGER_SPRINGREF) so the
        # servos, springs and gravity balance at one posture - the reset, the
        # raw MjData settle and the executed grasp all share one equilibrium.
        pre_curl_base = np.array([
            -0.144, 0.63, 0.58, -0.014,
            0.132, 0.466, 0.64, 0.449,
            0.101, 1.057, 0.73, 0.652,
            0.606, 0.573, 0.938, 0.407,
        ])
        for h in range(N_HANDS):
            wq = self.wrist_qpos_adr[h]
            d.qpos[wq] = 0.0
            pre = pre_curl_base.copy()
            if self.domain_rand:
                pre = pre + self._rng.uniform(-0.05, 0.05, N_FINGER)
            d.qpos[self.finger_qpos_adr[h * N_FINGER:(h + 1) * N_FINGER]] = pre
        d.qvel[:] = 0.0
        # ctrl MUST mirror the qpos we just wrote: all actuators are position
        # servos, so ctrl=0 would drag every joint back to zero during settling.
        for h in range(N_HANDS):
            d.ctrl[h * N_ACT_PER_HAND: h * N_ACT_PER_HAND + N_WRIST] = 0.0
            d.ctrl[h * N_ACT_PER_HAND + N_WRIST:
                   (h + 1) * N_ACT_PER_HAND] = \
                d.qpos[self.finger_qpos_adr[h * N_FINGER:(h + 1) * N_FINGER]]

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            super().reset(seed=seed)
        d, m = self.data, self.model

        mujoco.mj_resetData(m, d)

        if self._start_state is not None:
            # --- W6c checkpoint start ----------------------------------------
            q_start, ctrl_start = self._start_state
            d.qpos[:] = q_start
            # v19: the checkpoint ctrl is N_ACT-wide (44); the model is nu 50
            # (arm chain) - clip against the FULL model ctrlrange; the tail is
            # owned by _write_arm_ik_ctrl every step
            d.ctrl[:] = np.clip(ctrl_start, m.actuator_ctrlrange[:, 0],
                                m.actuator_ctrlrange[:, 1])
            if self._start_noise is not None:
                f_std, xy_std, yaw_std = self._start_noise
                for h in range(N_HANDS):
                    fq = self.finger_qpos_adr[h * N_FINGER:(h + 1) * N_FINGER]
                    d.qpos[fq] = d.qpos[fq] + self._rng.normal(0.0, f_std, N_FINGER)
                    d.ctrl[h * N_ACT_PER_HAND + N_WRIST:
                           (h + 1) * N_ACT_PER_HAND] = d.qpos[fq]
                oa = self.obj_qpos_adr
                d.qpos[oa] += float(self._rng.normal(0.0, xy_std))
                d.qpos[oa + 1] += float(self._rng.normal(0.0, xy_std))
                yaw = float(self._rng.normal(0.0, yaw_std))
                d.qpos[oa + 3: oa + 7] = [np.cos(yaw / 2), 0.0, 0.0,
                                          np.sin(yaw / 2)]
            d.qvel[:] = 0.0
            self.last_placement = {"mode": "w6c_checkpoint_start"}
        else:
            self._reset_home_state(d)
        d.qfrc_applied[:] = 0.0  # grasp bias re-applied per step when enabled
        # Settle under gravity / contact stiffness so step 0 is not a shock and the
        # box has already reached its resting height when z0 is recorded (MuJoCo's
        # default solref lets a resting box creep ~4 mm before it stops).
        for _ in range(400):
            mujoco.mj_step(m, d)
        # NOTE: mj_step only - no qpos assignment after this point.

        self._z0 = float(d.xpos[self.obj_body][2])
        self._z_peak = self._z0
        self._step_count = 0
        self._shaper.reset()
        self._prev_obj_pos = d.xpos[self.obj_body].copy()
        self._prev_dist = float(np.linalg.norm(d.xpos[self.palm] - d.xpos[self.obj_body]))
        self._ctrl_ref = d.ctrl.copy()
        self._v21_xy0 = d.xpos[self.obj_body][:2].copy()
        self._history = np.zeros(8)
        self.last_contact_pairs = []
        self.last_contact_force = 0.0
        self._substep_bearing = None
        self.trace = {"obj_z": [], "contact_force": [], "finger_force": [],
                      "support_force": [], "hook_up_N": [], "wall_up_N": [],
                      "pedestal_up_N": [], "hook_up_N_peak": [], "wall_up_N_peak": [],
                      "pedestal_up_N_peak": [], "contact_pairs": [], "obj_vz": [],
                      "supported": [], "finger_held": [], "hold_streak": [],
                      "free_flight": [], "wrist_z": [], "lift": []}
        self._ep: dict[str, list[float]] = {"grasp": [], "hold": [], "streak": []}

        return self._obs(), {"obj_z0": self._z0, "placement": self.last_placement}

    # --------------------------------------------------------- contact probe
    def _finger_geom_ids_of(self, pairs: list[tuple[str, str]]) -> list[str]:
        """Finger-side geom names among (other, box) contact pairs."""
        out = []
        for name, _ in pairs:
            if int(self.geom_name_to_id.get(name, -1)) in self.finger_geom_ids:
                out.append(name)
        return out

    def _support_height(self, x: float, y: float,
                        *, half_xy: tuple[float, float] | None = None,
                        strict: bool = True) -> float:
        """Top surface height of the highest static geom the box FOOTPRINT touches.

        Whole-footprint, not centre-point. The centre-only version is what made
        the C6 control a false pass: ``out_of_reach`` parked the box at
        y=0.35 m, whose footprint spans y in [0.23, 0.47] m. The centre at 0.35
        is off the pedestal (which ends at y=0.30 m), so the old code returned
        the bare table top (0.70 m) and dropped the box to z=0.74 m - i.e. 60 mm
        INSIDE the pedestal. The birth contact table read
        ``pedestal|object -0.0600`` x4, the solver threw the box off the table
        in the 400 settle steps, and it came to rest on the floor 0.41 m away
        (z=0.0394). The flail then moved it 0.57 mm, under the 1e-3 m gate -
        so the control passed while measuring a box that was not even on the
        table any more.

        Taking the max over every geom the footprint OVERLAPS makes that
        unrepresentable: the box is always born resting on the highest surface
        it actually touches.
        """
        hx, hy = self.box_half_xy if half_xy is None else half_xy
        best = -np.inf
        for _name, x0, x1, y0, y1, top in self.support_faces:
            if x + hx > x0 and x - hx < x1 and y + hy > y0 and y - hy < y1:
                best = max(best, top)
        if not np.isfinite(best):
            if not strict:
                return float(-np.inf)
            raise RuntimeError(
                f"box footprint at ({x}, {y}) half ({hx}, {hy}) overlaps no "
                "support geom - it would be born in mid-air")
        return float(best)

    def _object_footprint_half(self, yaw: float) -> tuple[float, float]:
        """Axis-aligned half-extents of the object body's footprint at ``yaw``.

        Taken over every geom on the object body, then rotated, so the answer is
        the true bounding box of what will collide with the table - not the
        nominal half-extent of one geom.
        """
        m = self.model
        hw = np.zeros(2)
        for g in range(m.ngeom):
            if int(m.geom_bodyid[g]) != self.obj_body:
                continue
            if int(m.geom_type[g]) != int(mujoco.mjtGeom.mjGEOM_BOX):
                continue
            half, pos = m.geom_size[g][:2], m.geom_pos[g][:2]
            hw = np.maximum(hw, np.abs(pos) + np.abs(half))
        c, s = abs(float(np.cos(yaw))), abs(float(np.sin(yaw)))
        return (float(c * hw[0] + s * hw[1]), float(s * hw[0] + c * hw[1]))

    def _face_fit(self, x: float, y: float,
                  half_xy: tuple[float, float]) -> tuple[float, float] | None:
        """(top, edge_margin) of the support face the box rests on, or None.

        The face must (a) contain the WHOLE footprint with
        ``SUPPORT_EDGE_MARGIN_M`` to spare, and (b) be the highest face the
        footprint OVERLAPS, i.e. the one ``_support_height`` returns. (b) is not
        redundant: the pedestal sits inside the slab, so a footprint straddling
        the pedestal edge "fits" the slab with room to spare - and dropping the
        box onto the slab there buries its overhanging part in the pedestal,
        which is the bug class this whole function exists to remove.
        """
        hx, hy = half_xy
        best: tuple[float, float] | None = None
        support = self._support_height(x, y, half_xy=half_xy, strict=False)
        if not np.isfinite(support):
            return None   # the search visits poses off the table; not an error
        for _name, x0, x1, y0, y1, top in self.support_faces:
            if abs(top - support) > 1e-12:
                continue
            # margin from the FOOTPRINT to the face edge, not from the centre -
            # the centre-only version accepts a box whose corner hangs in the air.
            margin = min(x - hx - x0, x1 - (x + hx), y - hy - y0, y1 - (y + hy))
            if margin < SUPPORT_EDGE_MARGIN_M:
                continue
            best = (float(top), float(margin))
            break
        return best

    def _clearance(self, x: float, y: float,
                   half_xy: tuple[float, float]) -> float:
        """Distance from the NEAREST hand mount to the box footprint (v23: the
        mounts carry y offsets, so the origin-centred circle no longer covers
        the left palm)."""
        hx, hy = half_xy
        best = float("inf")
        for sfx in ("", "_left"):
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY,
                                    f"hand_root{sfx}")
            mx, my = (float(self.model.body_pos[bid][0]),
                      float(self.model.body_pos[bid][1]))
            gx = max(0.0, abs(x - mx) - hx)
            gy = max(0.0, abs(y - my) - hy)
            best = min(best, float(np.hypot(gx, gy)))
        return best

    def _search_placement(self, x: float, y: float, half_xy: tuple[float, float],
                          *, min_clearance: float
                          ) -> tuple[float, float, float, float] | None:
        """Find a legal pose near (x, y) in two stages.

        1. slide along ONE axis (the cheap fix: the box was nudged off the face
           by a small offset, e.g. the demo collector's +-6 mm grid);
        2. only if that finds nothing, a 2-D sweep - needed because the
           control's +y request is unsatisfiable along y at ANY distance (see
           reset), and a single-axis slide can never satisfy an infeasible axis
           that is held fixed.

        Candidates are scored by (edge margin, clearance from the hand): the
        most robust legal pose wins, and among equally robust ones the one
        furthest from the hand. Both terms are computed from the footprint.
        """
        n = int(round(SUPPORT_SLIDE_SPAN_M / SUPPORT_SLIDE_STEP_M))
        best: tuple[tuple[float, float], tuple[float, float, float, float]] | None = None

        def consider(cx: float, cy: float) -> None:
            nonlocal best
            fit = self._face_fit(cx, cy, half_xy)
            if fit is None:
                return
            clearance = self._clearance(cx, cy, half_xy)
            if clearance < min_clearance:
                return
            key = (fit[1], clearance)
            if best is None or key > best[0]:
                best = (key, (float(cx), float(cy), fit[0], fit[1]))

        for axis in (0, 1):
            for k in range(n + 1):
                for sign in ((1, -1) if k else (1,)):
                    off = [float(x), float(y)]
                    off[axis] += sign * k * SUPPORT_SLIDE_STEP_M
                    consider(*off)
        if best is None:
            for i in range(-n, n + 1):
                for j in range(-n, n + 1):
                    consider(x + i * SUPPORT_SLIDE_STEP_M, y + j * SUPPORT_SLIDE_STEP_M)
        return None if best is None else best[1]

    def _resolve_object_placement(self, x: float, y: float, yaw: float, *,
                                  min_clearance: float = 0.0
                                  ) -> tuple[float, float, float, dict]:
        """(x, y, z, report) for a placement the box can actually rest at.

        Never returns an initial overlap: the height comes from
        ``_support_height`` over the whole footprint. If the requested xy would
        leave the box hanging over an edge (or, for the control, inside the
        hand's reach) it is moved to a legal pose and the correction is REPORTED
        rather than hidden.
        """
        half_xy = self._object_footprint_half(yaw)
        fit = self._face_fit(x, y, half_xy)
        if fit is not None and self._clearance(x, y, half_xy) >= min_clearance:
            px, py, top, margin = x, y, fit[0], fit[1]
            moved = False
        else:
            found = self._search_placement(x, y, half_xy, min_clearance=min_clearance)
            if found is None:
                raise RuntimeError(
                    f"no stable placement for the box near ({x}, {y}): within "
                    f"{SUPPORT_SLIDE_SPAN_M} m no pose puts the whole footprint "
                    f"({half_xy[0]:.3f}, {half_xy[1]:.3f}) on one support face "
                    f"with {SUPPORT_EDGE_MARGIN_M * 1e3:.0f} mm to spare"
                    + (f" and {min_clearance} m of clearance from the hand"
                       if min_clearance > 0 else ""))
            px, py, top, margin = found
            moved = True
        report = {
            "requested_xy_m": [float(x), float(y)],
            "placed_xy_m": [float(px), float(py)],
            "resting_z_m": float(top + self.box_half_z),
            "edge_margin_m": float(margin),
            "clearance_m": self._clearance(px, py, half_xy),
            "moved": moved,
        }
        return float(px), float(py), float(top + self.box_half_z), report

    def _support_and_held(self, pairs: list[tuple[str, str]]) -> tuple[bool, bool]:
        """(box resting on a support surface, box held by a finger).

        A legitimately lifted box has no support contact, so "off the support"
        alone cannot distinguish a lift from the box being shoved away. Note the
        box rests on the *pedestal*, not the slab, so the support test must accept
        any static support geom - checking for "table_slab" alone made a normally
        resting box look knocked away on every single step.
        """
        on_support = any(name in self.support_geoms for name, _ in pairs)
        held = any(self.geom_name_to_id.get(name, -1) in self.finger_geom_ids
                   for name, _ in pairs)
        return on_support, held

    def _box_knocked_away(self, pairs: list[tuple[str, str]]) -> bool:
        """Box is neither resting on a support surface nor held by any finger."""
        on_support, held = self._support_and_held(pairs)
        return not on_support and not held

    def _read_contacts(self) -> tuple[list[tuple[str, str]], float, float, float]:
        """Geom pairs touching the box, the summed normal force, the finger-only
        portion of that force (what the acceptance test correlates against box
        height), and the SUPPORT-only portion - the normal force the pedestal /
        table is still applying to the box.

        The last value is what distinguishes a real lift from a corner wedge. A
        box that is genuinely off the pedestal has no support contact and reads
        0 N; the f210f45 wedge, which displays 9 cm of lift while leaning on the
        pedestal, reads tens of newtons here. It comes out of the same loop and
        the same ``mj_contactForce`` call as the existing contact evidence - no
        second sensing path, and no pair that the A2/A3a tests do not already see.
        """
        m, d = self.model, self.data
        pairs: list[tuple[str, str]] = []
        force = 0.0
        finger_force = 0.0
        support_force = 0.0
        for i in range(d.ncon):
            con = d.contact[i]
            g1, g2 = int(con.geom1), int(con.geom2)
            if self.obj_geom not in (g1, g2):
                continue
            f = np.zeros(6)
            mujoco.mj_contactForce(m, d, i, f)
            mag = float(np.linalg.norm(f[:3]))
            force += mag
            other = g2 if g1 == self.obj_geom else g1
            label = self.geom_label[other]
            if other in self.finger_geom_ids:
                finger_force += mag
            elif label in self.support_geoms:
                # support geoms and finger geoms are disjoint by construction
                # (support_geoms only collects world-body box geoms), so this is
                # an exact partition of `force`, not a guess.
                support_force += mag
            pairs.append((label, self.geom_label[self.obj_geom]))
        return pairs, force, finger_force, support_force

    def _bearing(self) -> dict[str, float]:
        """Upward force (N) on the box this step, split by bearing path.

        Three paths, mutually exclusive and exhaustive over the object geom's
        contacts:

          hook     a FINGER geom touching the box's TOP face - the rim hook, the
                   only path a clean lift is supposed to use
          wall     a FINGER geom touching a SIDE face - wall friction, the second
                   legitimate bearing path of a real pinch
          pedestal a STATIC support geom (pedestal / table) - the path A3b
                   forbids; reading it non-zero means the box is being held up
                   by the base, whatever height the policy is displaying

        The normal is taken from ``contact.frame[0:3]``, MuJoCo's row-major
        contact frame whose first row is the normal directed geom1 -> geom2, and
        flipped to point into the box so ``fn * n`` is the force applied TO it.
        ``tests/test_smoke4_pedestal_load.py`` pins that orientation with a
        resting-box calibration (a box at rest must read a positive bearing
        close to its own weight).
        """
        m, d = self.model, self.data
        bp = d.xpos[self.obj_body]
        bxm = d.xmat[self.obj_body].reshape(3, 3)
        half = m.geom_size[self.obj_geom]
        out = {"hook_up_N": 0.0, "wall_up_N": 0.0, "pedestal_up_N": 0.0}
        for i in range(d.ncon):
            con = d.contact[i]
            g1, g2 = int(con.geom1), int(con.geom2)
            if self.obj_geom not in (g1, g2):
                continue
            other = g2 if g1 == self.obj_geom else g1
            f = np.zeros(6)
            mujoco.mj_contactForce(m, d, i, f)
            if float(f[0]) == 0.0:
                continue
            n = np.asarray(con.frame, dtype=np.float64)[0:3]
            n = n / max(float(np.linalg.norm(n)), 1e-12)
            if g1 == self.obj_geom:
                n = -n
            up = float((n * float(f[0]))[2])
            label = self.geom_label[other]
            if label in self.support_geoms:
                out["pedestal_up_N"] += up
                continue
            if other not in self.finger_geom_ids:
                continue
            rel = bxm.T @ (np.asarray(con.pos) - bp)
            on_top = (abs(rel[2] - half[2]) <= max(float(half[2]) * 0.5, 1e-3)
                      and abs(rel[0]) <= half[0] + 1e-9
                      and abs(rel[1]) <= half[1] + 1e-9 and rel[2] > 0.0)
            out["hook_up_N" if on_top else "wall_up_N"] += up
        return out

    # --------------------------------------------------------------- reward
    def _reward(self) -> tuple[float, dict[str, float]]:
        """Measure the world, then delegate the shaping to HoldRewardShaper.

        The shaper owns every decision about what the episode is worth (see its
        docstring); this method only reads sensors. Round 1 kept the arithmetic
        inline, which made the shaping untestable without a policy.
        """
        d = self.data
        obj_pos = d.xpos[self.obj_body]
        lift = float(obj_pos[2] - self._z0)
        self._z_peak = max(self._z_peak, lift)

        palm_pos = d.xpos[self.palm]
        dist = float(np.linalg.norm(palm_pos - obj_pos))
        dist_reach = 0.0
        if self._prev_dist is not None:
            dist_reach = self._prev_dist - dist
        self._prev_dist = dist
        align = float(np.exp(-40.0 * float(np.linalg.norm(obj_pos[:2] - palm_pos[:2]))))

        pairs, force, finger_force, support_force = self._read_contacts()
        on_support, finger_held = self._support_and_held(pairs)

        qvel = d.qvel[self.finger_dof_adr]
        wrist_vel = np.concatenate([d.qvel[adr] for adr in self.wrist_dof_adr])
        act_pen = float(np.sum(np.square(d.ctrl - self._ctrl_ref))) * W_CTRL
        vel_pen = float(np.sum(np.square(qvel))) * W_VEL \
            + float(np.sum(np.square(wrist_vel))) * W_VEL

        rew, info = self._shaper.step(
            lift=lift,
            finger_force=finger_force,
            finger_held=finger_held,
            on_support=on_support,
            align=align,
            dist_reach=dist_reach,
            act_pen=act_pen,
            vel_pen=vel_pen,
            support_force_n=support_force,
        )
        # Sliding the box off the table used to be a free exploit: once it is off,
        # max(lift, 0) is 0 forever, so PPO learned to shove it over the edge
        # (observed: episode-mean lift of -0.13 m = box at z 0.61, below the table).
        knocked = 1.0 if self._box_knocked_away(pairs) else 0.0
        rew -= W_OFF_TABLE * knocked

        info["knocked_away"] = knocked
        info["contact_force"] = force
        info["pairs"] = pairs
        info["dist"] = dist
        return float(rew), info

    # ------------------------------------------------------------------ step
    def _apply_action(self, action: np.ndarray) -> None:
        """Map the 44-d action onto actuator setpoints. No qpos/qvel writes.

        Layout, right hand first: [d_wrist_pos(3), d_wrist_rot(3),
        finger_targets(16)] per hand - the 0:22 prefix drives the right hand
        exactly as the single-hand env's 22-d action did.
        """
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)

        ctrl = self.data.ctrl[:N_ACT].copy()
        for h in range(N_HANDS):
            base = h * N_ACT_PER_HAND
            # wrist: pose DELTA, integrated into the setpoint then clipped to ctrlrange
            dpos = a[base:base + 3] * np.array([0.012, 0.012, 0.008])
            drot = a[base + 3:base + 6] * np.array([0.10, 0.10, 0.10])
            target = ctrl[base:base + N_WRIST].copy()
            target[0:3] = target[0:3] + dpos
            target[3:6] = target[3:6] + drot
            ctrl[base:base + N_WRIST] = target
            # fingers: absolute setpoint spanning the joint ctrlrange; the
            # action's finger blocks sit at 6 + h*22 (interleaved [6,16,6,16])
            fbase = N_WRIST + h * N_ACT_PER_HAND
            lo = self.ctrl_low[base + N_WRIST: base + N_ACT_PER_HAND]
            hi = self.ctrl_high[base + N_WRIST: base + N_ACT_PER_HAND]
            ctrl[base + N_WRIST: base + N_ACT_PER_HAND] = \
                lo + (a[fbase:fbase + N_FINGER] + 1.0) * 0.5 * (hi - lo)
        self.data.ctrl[:N_ACT] = np.clip(ctrl, self.ctrl_low[:N_ACT], self.ctrl_high[:N_ACT])
        self._write_arm_ik_ctrl()
        self._ctrl_ref = self.data.ctrl.copy()

    def _write_arm_ik_ctrl(self) -> None:
        """W6a-v19: analytic 3-DoF arm IK, env-side, every step.

        Solves the closed-form 2-link-in-yaw-plane IK from each shoulder
        anchor to the hand root's live world position and writes the 6 arm
        servo targets into ctrl[N_ACT:].  The policy never sees these dims
        (its action is still N_ACT=44) and no arm state enters obs — the
        trained checkpoints stay bit-valid, the arm is a visual follower.
        """
        L = 0.16  # link length (matches the MJCF arm_upper/forearm geoms)
        for s in range(2):
            sfx = "" if s == 0 else "_left"
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY,
                                    f"hand_root{sfx}")
            target = self.data.xpos[bid]
            # v22: shoulder anchor read off the model's arm_yaw body (the left
            # mount carries a +75 mm y offset -- layout-proof, no constant dupe)
            arm_sfx = "_r" if s == 0 else "_l"   # arm bodies use _r/_l, hands use ""/_left
            yaw_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY,
                                        f"arm_yaw{arm_sfx}")
            assert yaw_bid >= 0, f"arm_yaw{arm_sfx} body missing"
            shoulder = self.model.body_pos[yaw_bid].astype(float)
            dx, dy, dz = target - shoulder
            yaw = float(np.arctan2(dy, dx))
            r = float(np.hypot(dx, dy))
            dist = float(np.clip(np.linalg.norm([dx, dy, dz]), 1e-6, 2 * L - 1e-6))
            elbow = float(np.pi - np.arccos(
                np.clip((2 * L * L - dist ** 2) / (2 * L * L), -1.0, 1.0)))
            pitch = float(np.arctan2(dz, r) - np.arccos(
                np.clip(dist ** 2 / (2 * L * dist), -1.0, 1.0)))
            base = N_ACT + s * 3
            self.data.ctrl[base + 0] = yaw
            self.data.ctrl[base + 1] = pitch
            self.data.ctrl[base + 2] = elbow

    def step(self, action: np.ndarray):
        if not isinstance(action, np.ndarray) or action.shape != (N_ACT,):
            raise ValueError(f"action must be shape ({N_ACT},), got {action!r}")
        self._apply_action(action)
        if self._grasp_bias_sides is not None:
            for si, bias in enumerate(self._grasp_bias_sides):
                for dof, cap in self._grasp_bias_side_dofs[si]:
                    self.data.qfrc_applied[dof] = float(
                        np.clip(bias, -cap, cap))
        elif self._grasp_bias_nm:
            for dof, cap in self._grasp_bias_dofs:
                self.data.qfrc_applied[dof] = float(
                    np.clip(self._grasp_bias_nm, -cap, cap))
        if self.record_bearing:
            # Substep-resolution bearing sampling. A stiffened contact's frame-
            # internal spike lives INSIDE a control step (50 Hz control = 10 x
            # 2 ms sim substeps), so a single sample taken after the frame loop
            # cannot see it: Wave 3.5F's own note records 9 N @ 0.6 mm becoming
            # 9.93 kN @ 20.8 mm as an intra-frame event. Sampling after each
            # substep and keeping the running max is what makes the third
            # bearing column mean what its name says.
            #
            # Sampling points: the END of each completed mj_step, i.e. the same
            # substep boundaries the physics is integrated on. If another probe
            # wraps mj_step and samples BEFORE it, its index i corresponds to
            # our index i-1 - a one-substep (2 ms) offset. report.json carries
            # that alignment so the two numbers are never read as one quantity.
            sub = {"hook_up_N": -np.inf, "wall_up_N": -np.inf,
                   "pedestal_up_N": -np.inf}
            for _ in range(self.frame_skip):
                mujoco.mj_step(self.model, self.data)
                b = self._bearing()
                for k, v in b.items():
                    if v > sub[k]:
                        sub[k] = v
            self._substep_bearing = {k: (0.0 if not np.isfinite(v) else v)
                                     for k, v in sub.items()}
        else:
            for _ in range(self.frame_skip):
                mujoco.mj_step(self.model, self.data)
            self._substep_bearing = None
        rew, rinfo = self._reward()

        self._ep["grasp"].append(rinfo["grasp"])
        self._ep["hold"].append(rinfo["min_hold"])
        self._ep["streak"].append(rinfo["hold_streak"])
        self._step_count += 1
        term = False
        trunc = self._step_count >= self.max_episode_steps

        # trace for the acceptance evidence
        obj_z = float(self.data.xpos[self.obj_body][2])
        self.trace["obj_z"].append(obj_z)
        self.trace["lift"].append(obj_z - self._z0)
        self.trace["wrist_z"].append(float(self.data.xpos[self.hand_root][2]))
        self.trace["contact_force"].append(rinfo["contact_force"])
        self.trace["finger_force"].append(rinfo["finger_force"])
        # A3b evidence: the pedestal-borne normal force per step. A genuine lift
        # reads ~0 here; a corner wedge reads tens of newtons while displaying
        # centimetres of lift. The acceptance report turns this trace into the
        # per-episode bearing decomposition (hook / wall friction / pedestal).
        self.trace["support_force"].append(rinfo["support_force"])
        if self.record_bearing:
            b = self._bearing()
            for k, v in b.items():
                self.trace[k].append(v)
            for k, v in self._substep_bearing.items():
                self.trace[f"{k}_peak"].append(v)
        self.trace["contact_pairs"].append([list(p) for p in rinfo["pairs"]])
        # per-step state the acceptance gates segment on: vertical velocity (for the
        # gravity-consistency check on a free-falling box) and whether the box was
        # supported / finger-held at this step.
        self.trace["obj_vz"].append(float(self.data.qvel[self.obj_qvel_adr + 2]))
        self.trace["supported"].append(rinfo["supported"])
        self.trace["finger_held"].append(rinfo["finger_held"])
        # round-2 shaping telemetry: consecutive evaluator-valid hold steps, and
        # whether the box was aloft with no finger on it (the flick exploit).
        self.trace["hold_streak"].append(rinfo["hold_streak"])
        self.trace["free_flight"].append(rinfo["free_flight"])
        self.last_contact_pairs = [(a, b) for a, b in rinfo["pairs"]]

        info = {
            "lift": rinfo["lift"],
            "grasp": rinfo["grasp"],
            "hold": rinfo["min_hold"],
            "z_peak": self._z_peak,
            "contact_force": rinfo["contact_force"],
            "support_force": rinfo["support_force"],
        }
        if term or trunc:
            info["lifecycle"] = self._episode_stats()
        return self._obs(), float(rew), bool(term), bool(trunc), info

    def _episode_stats(self) -> dict[str, float]:
        """Per-episode telemetry for the training curve (means over the episode)."""
        n = max(1, len(self.trace["lift"]))
        return {
            "lift": float(np.mean(self.trace["lift"])) if self.trace["lift"] else 0.0,
            "lift_max": float(max(self.trace["lift"])) if self.trace["lift"] else 0.0,
            "grasp": float(np.mean(self._ep["grasp"])) if self._ep["grasp"] else 0.0,
            "hold": float(np.mean(self._ep["hold"])) if self._ep["hold"] else 0.0,
            "hold_streak_max": float(max(self._ep["streak"])) if self._ep["streak"] else 0.0,
            "free_flight": float(np.mean(self.trace["free_flight"]))
            if self.trace["free_flight"] else 0.0,
            "z_peak": float(self._z_peak - self._z0),
            "contact_force": float(np.mean(self.trace["contact_force"])),
            "support_force": float(np.mean(self.trace["support_force"]))
            if self.trace["support_force"] else 0.0,
            "steps": n,
        }

    # ---------------------------------------------------------------- render
    def render(self, mode: str = "rgb_array", camera: str = "cam_overhead",
               width: int = 320, height: int = 240):
        r = mujoco.Renderer(self.model, height, width)
        r.update_scene(self.data, camera=camera)
        img = r.render()
        r.close()
        return img

    def close(self) -> None:
        pass


def make_env(rank: int = 0, **kwargs) -> LiftHoldEnv:
    """SubprocVecEnv factory (must be importable at module level)."""
    def _init() -> LiftHoldEnv:
        env = LiftHoldEnv(**kwargs)
        env.reset(seed=rank)
        return env

    return _init
