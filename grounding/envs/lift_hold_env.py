"""Task 4.2 - floating-Allegro lift & hold gymnasium.Env.

Action (22) = [d_wrist_pos(3), d_wrist_rot(3), finger_joint_targets(16)]
Observation     = object pose+vel, wrist pose, finger joint angles+velocities,
                  per-fingertip distance to the box, and a 4-frame history of the
                  object z / wrist z.

Hard rule (PLAN 4.2 / GPT-redline): the ONLY way anything moves is through
``data.ctrl``. ``step()`` never assigns to ``data.qpos`` / ``data.qvel``; qpos is
written exclusively in ``reset()``. ``tests/test_smoke4_env.py::test_c5_*``
enforces this statically (AST scan) and dynamically (replay determinism).

Reward is shaped around the official evaluator
(``runs/p0/assets/tissue_box_lift_hold.json``):
    lift   >= 0.10 m      hold   <= 0.05 m descent / step
    final_min_lift >= -0.01 m         min_hold_steps = 20
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
N_ACT = N_WRIST + N_FINGER  # 22

# finger bodies whose geoms count as "hand" for the contact-evidence criterion
FINGER_BODY_PREFIXES = ("ff_", "mf_", "rf_", "th_")
OBJECT_GEOM = "object_geom"

# --- reward weights (shaped around the official evaluator) ------------------
# Weighting rationale (learned the hard way - see report.json deviations):
#  * the height term must dominate the "touch it" term, otherwise PPO settles for a
#    permanent light grasp and never lifts;
#  * the drop penalty must NOT be a per-step cliff on resting contact: an earlier
#    W_DROP=60/step keyed on lift <= -5 mm made touching the box at all net
#    negative (a 5 cm lift is only worth 6.0/step), so PPO converged on "never
#    touch the box". W_DROP is now charged once, on the transition that loses an
#    already-achieved lift, and W_LIFT is floored so resting contact is free.
W_REACH = 2.0          # per metre of palm->box distance reduction
W_ALIGN = 3.0          # box centred under the palm
W_GRASP = 2.0          # fingertips in contact with the box (enabler, not the goal)
W_LIFT = 150.0         # per metre lifted above the start height -> 7.5 @ 5 cm
W_HOLD = 8.0           # per step held above hold_threshold
W_MIN_HOLD = 60.0      # once min_hold_steps are satisfied
W_LIFT_BONUS = 200.0   # evaluator lift_threshold (0.10 m) reached
W_DROP = 60.0          # ONE-SHOT penalty for losing an achieved lift
W_CTRL = 0.002         # action magnitude penalty
W_VEL = 0.01           # joint velocity penalty
W_OFF_TABLE = 20.0     # per step while the box is off the table AND not held
HOLD_SLACK = 0.001     # descent tolerance added on top of max_descent_step_m


def _load_evaluator() -> dict[str, float]:
    data = json.loads(EVALUATOR_JSON.read_text())["evaluator"]
    return {k: float(v) for k, v in data.items() if isinstance(v, (int, float))}


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

        self.model = mujoco.MjModel.from_xml_path(str(MJCF))
        self.data = mujoco.MjData(self.model)
        self.domain_rand = domain_rand
        self.out_of_reach = out_of_reach

        # timestep 0.002 s -> 0.02 s control step at ctrl_hz=50
        self.frame_skip = max(1, int(round(1.0 / (ctrl_hz * self.model.opt.timestep))))
        self.dt = self.model.opt.timestep * self.frame_skip

        self.max_episode_steps = episode_len

        # ---- name lookups ------------------------------------------------
        m, d = self.model, self.data
        self.obj_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "object")
        self.obj_geom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, OBJECT_GEOM)
        self.hand_root = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "hand_root")
        self.palm = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "palm")
        self.obj_qpos_adr = int(m.jnt_qposadr[mujoco.mj_name2id(
            m, mujoco.mjtObj.mjOBJ_JOINT, "object_free")])
        self.obj_qvel_adr = int(m.jnt_dofadr[mujoco.mj_name2id(
            m, mujoco.mjtObj.mjOBJ_JOINT, "object_free")])
        self.wrist_qpos_adr = [int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)])
                               for n in ("wrist_x", "wrist_y", "wrist_z",
                                         "wrist_rx", "wrist_ry", "wrist_rz")]
        self.wrist_dof_adr = [int(m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)])
                              for n in ("wrist_x", "wrist_y", "wrist_z",
                                        "wrist_rx", "wrist_ry", "wrist_rz")]
        self.finger_qpos_adr = np.array(
            [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]
             for n in ("ffj0", "ffj1", "ffj2", "ffj3", "mfa0", "mfa1", "mfa2", "mfa3",
                       "rfj0", "rfj1", "rfj2", "rfj3", "tha0", "tha1", "tha2", "tha3")],
            dtype=int)
        self.finger_dof_adr = np.array(
            [m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]
             for n in ("ffj0", "ffj1", "ffj2", "ffj3", "mfa0", "mfa1", "mfa2", "mfa3",
                       "rfj0", "rfj1", "rfj2", "rfj3", "tha0", "tha1", "tha2", "tha3")],
            dtype=int)
        self.tip_bodies = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
                           for n in ("ff_tip", "mf_tip", "rf_tip", "th_tip")]
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
        self.ctrl_low = m.actuator_ctrlrange[:, 0].copy()
        self.ctrl_high = m.actuator_ctrlrange[:, 1].copy()
        # world-body box geoms the object can rest on (table slab, pedestal, ...)
        self.support_geoms: set[str] = set()
        for g in range(m.ngeom):
            if int(m.geom_bodyid[g]) != 0 or g == self.obj_geom:
                continue
            if int(m.geom_type[g]) != int(mujoco.mjtGeom.mjGEOM_BOX):
                continue
            self.support_geoms.add(self.geom_label[g])
        assert self.support_geoms, "scene defines no support geometry"

        # ---- spaces ------------------------------------------------------
        self.action_space = spaces.Box(-1.0, 1.0, (N_ACT,), dtype=np.float32)
        # 3 box xyz + 4 quat + 3 linvel + 3 angvel + 3 wrist-box + 4 wrist quat
        # + 16 finger q + 16 finger v + 4 tip distances + 8 history = 64
        n_obs = 3 + 4 + 3 + 3 + 3 + 4 + N_FINGER + N_FINGER + 4 + 8
        self.n_obs = n_obs
        self.observation_space = spaces.Box(-np.inf, np.inf, (n_obs,), dtype=np.float32)

        # ---- episode bookkeeping ----------------------------------------
        self._step_count = 0
        self._z0 = 0.0
        self._z_peak = 0.0
        self._hold_steps = 0
        self._min_hold_counter = 0
        self._prev_dist = None
        self._prev_obj_pos = None
        self._prev_lift = 0.0
        self._ctrl_ref = np.zeros(N_ACT)
        self._history = np.zeros(8, dtype=np.float64)
        self._rng = np.random.default_rng(0)
        self.last_contact_pairs: list[tuple[str, str]] = []
        self.last_contact_force = 0.0
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
        wrist_pos = d.xpos[self.hand_root].copy()
        wrist_quat = d.xquat[self.hand_root].copy()
        finger_q = d.qpos[self.finger_qpos_adr].copy()
        finger_v = d.qvel[self.finger_dof_adr].copy()
        tip_dist = np.array(
            [np.linalg.norm(d.xpos[b] - obj_pos) for b in self.tip_bodies], dtype=np.float64)
        self._history = np.concatenate(
            [[obj_pos[2] - self._z0, wrist_pos[2] - self._z0],
             self._history[:-2]])

        return np.concatenate([
            obj_pos - np.array([0.0, 0.0, self._z0]),  # 3  box xyz relative to start
            obj_quat,  # 4
            obj_linvel,  # 3
            obj_angvel,  # 3
            wrist_pos - obj_pos,  # 3  wrist relative to box
            wrist_quat,  # 3 -> 4
            finger_q,  # 16
            finger_v,  # 16
            tip_dist,  # 4
            self._history,  # 8  4-frame history of (box dz, wrist dz)
        ]).astype(np.float32)

    def observation_space_shape(self) -> tuple[int, ...]:
        return self.observation_space.shape  # type: ignore[return-value]

    # ---------------------------------------------------------------- reset
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            super().reset(seed=seed)
        d, m = self.data, self.model

        mujoco.mj_resetData(m, d)

        # --- object placement (allowed: reset may write qpos) -------------
        # out_of_reach parks the box 0.35 m along +y: far outside the hand's
        # workspace (fingers reach ~0.06 m past the palm, and the wrist ctrlrange
        # caps travel at 0.18 m) but still over the table slab, so the box cannot
        # move. The resting height must be the height of whatever is actually
        # beneath it (the pedestal only spans +-0.30 m), so it is raycast rather
        # than assumed - otherwise the box free-falls 60 mm during settling.
        x_off, y_off = (0.0, 0.35) if self.out_of_reach else (0.0, 0.0)
        support_z = self._support_height(x_off, y_off)
        box_z = support_z + float(m.geom_size[self.obj_geom][2])
        obj_pos = np.array([x_off, y_off, box_z])
        quat = np.array([1.0, 0.0, 0.0, 0.0])
        if self.domain_rand and not self.out_of_reach:
            obj_pos[:2] += self._rng.uniform(-0.01, 0.01, 2)
            yaw = float(self._rng.uniform(-0.10, 0.10))
            quat = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
        d.qpos[self.obj_qpos_adr: self.obj_qpos_adr + 7] = np.concatenate([obj_pos, quat])

        # --- hand: home pose with a light grasp pre-shape ------------------
        d.qpos[self.wrist_qpos_adr] = 0.0
        # fingers start half-curled so the box is between them from step 0
        pre_curl = np.array([
            0.0, 0.9, 0.7, 0.4,
            0.0, 0.9, 0.7, 0.4,
            0.0, 0.9, 0.7, 0.4,
            0.8, 0.5, 0.4, 0.3,
        ])
        if self.domain_rand:
            pre_curl = pre_curl + self._rng.uniform(-0.05, 0.05, N_FINGER)
        d.qpos[self.finger_qpos_adr] = pre_curl
        d.qpos[self.wrist_qpos_adr[2]] = -0.02  # palm slightly lower to meet the box
        d.qvel[:] = 0.0
        # ctrl MUST mirror the qpos we just wrote: all 22 actuators are position
        # servos, so ctrl=0 would drag every joint back to zero during settling.
        d.ctrl[N_WRIST:] = pre_curl
        d.ctrl[0:3] = d.qpos[self.wrist_qpos_adr[0:3]]
        d.ctrl[3:6] = 0.0
        # Settle under gravity / contact stiffness so step 0 is not a shock and the
        # box has already reached its resting height when z0 is recorded (MuJoCo's
        # default solref lets a resting box creep ~4 mm before it stops).
        for _ in range(400):
            mujoco.mj_step(m, d)
        # NOTE: mj_step only - no qpos assignment after this point.

        self._z0 = float(d.xpos[self.obj_body][2])
        self._z_peak = self._z0
        self._step_count = 0
        self._hold_steps = 0
        self._min_hold_counter = 0
        self._prev_obj_pos = d.xpos[self.obj_body].copy()
        self._prev_dist = float(np.linalg.norm(d.xpos[self.palm] - d.xpos[self.obj_body]))
        self._prev_lift = 0.0
        self._ctrl_ref = d.ctrl.copy()
        self._history = np.zeros(8)
        self.last_contact_pairs = []
        self.last_contact_force = 0.0
        self.trace = {"obj_z": [], "contact_force": [], "finger_force": [],
                      "contact_pairs": [], "obj_vz": [], "supported": [],
                      "finger_held": [],
                      "wrist_z": [], "lift": []}
        self._ep: dict[str, list[float]] = {"grasp": [], "hold": []}

        return self._obs(), {"obj_z0": self._z0}

    # --------------------------------------------------------- contact probe
    def _finger_geom_ids_of(self, pairs: list[tuple[str, str]]) -> list[str]:
        """Finger-side geom names among (other, box) contact pairs."""
        out = []
        for name, _ in pairs:
            if int(self.geom_name_to_id.get(name, -1)) in self.finger_geom_ids:
                out.append(name)
        return out

    def _support_height(self, x: float, y: float) -> float:
        """Top surface height of the highest axis-aligned static geom under (x, y).

        Resolved from the model rather than assumed, because the pedestal only
        spans +-0.30 m: placing the parked box at the nominal pedestal height would
        drop it 60 mm onto the bare table during settling.
        """
        m = self.model
        best = -np.inf
        for gid in range(m.ngeom):
            if gid == self.obj_geom or int(m.geom_bodyid[gid]) != 0:
                continue  # skip the box and anything parented to a body
            if int(m.geom_type[gid]) != int(mujoco.mjtGeom.mjGEOM_BOX):
                continue
            half = m.geom_size[gid]
            pos = m.geom_pos[gid]
            if abs(x - pos[0]) <= half[0] and abs(y - pos[1]) <= half[1]:
                best = max(best, float(pos[2] + half[2]))
        if not np.isfinite(best):
            raise RuntimeError(f"no support geom under ({x}, {y})")
        return best

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

    def _read_contacts(self) -> tuple[list[tuple[str, str]], float, float]:
        """Geom pairs touching the box, the summed normal force, and the finger-only
        portion of that force (the latter is what the acceptance test correlates
        against the box height)."""
        m, d = self.model, self.data
        pairs: list[tuple[str, str]] = []
        force = 0.0
        finger_force = 0.0
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
            pairs.append((label, self.geom_label[self.obj_geom]))
        return pairs, force, finger_force

    # --------------------------------------------------------------- reward
    def _reward(self) -> tuple[float, dict[str, float]]:
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

        pairs, force, finger_force = self._read_contacts()
        grasp = 1.0 if finger_force > 0.0 else 0.0

        # evaluator semantics: hold means "stayed above hold_threshold_m without
        # descending more than max_descent_step_m per step"
        if lift >= self.hold_threshold_m:
            descent = max(0.0, self._prev_lift - lift)
            if descent <= self.max_descent_step_m + HOLD_SLACK:
                self._min_hold_counter += 1
            else:
                self._min_hold_counter = 0
        else:
            self._min_hold_counter = 0
        self._prev_lift = lift

        hold = 1.0 if lift >= self.hold_threshold_m else 0.0
        min_hold = 1.0 if self._min_hold_counter >= self.min_hold_steps else 0.0
        lift_bonus = 1.0 if lift >= self.lift_threshold_m else 0.0
        # one-shot: charge the drop penalty only on the transition out of an
        # achieved lift, never for resting contact on the table
        was_lifted = self._prev_lift >= self.hold_threshold_m
        drop = 1.0 if (was_lifted and lift < self.hold_threshold_m) else 0.0

        # Sliding the box off the table used to be a free exploit: once it is off,
        # max(lift, 0) is 0 forever, so PPO learned to shove it over the edge
        # (observed: episode-mean lift of -0.13 m = box at z 0.61, below the table).
        # Charging every abandoned-off-table step removes that escape hatch.
        knocked = 1.0 if self._box_knocked_away(pairs) else 0.0
        on_support, finger_held = self._support_and_held(pairs)

        qvel = self.data.qvel[self.finger_dof_adr]
        wrist_vel = self.data.qvel[self.wrist_dof_adr]
        act_pen = float(np.sum(np.square(self.data.ctrl - self._ctrl_ref))) * W_CTRL
        vel_pen = float(np.sum(np.square(qvel))) * W_VEL \
            + float(np.sum(np.square(wrist_vel))) * W_VEL
        rew = (
            W_REACH * dist_reach
            + W_ALIGN * align
            + W_GRASP * grasp
            + W_LIFT * max(lift, 0.0)
            + W_HOLD * hold
            + W_MIN_HOLD * min_hold
            + W_LIFT_BONUS * lift_bonus
            - W_DROP * drop
            - W_OFF_TABLE * knocked
            - act_pen
            - vel_pen
        )
        info = {
            "lift": lift,
            "align": align,
            "grasp": grasp,
            "hold": hold,
            "min_hold": min_hold,
            "drop": drop,
            "knocked_away": knocked,
            "supported": float(on_support),
            "finger_held": float(finger_held),
            "contact_force": force,
            "finger_force": finger_force,
            "finger_contact": grasp,
            "pairs": pairs,
            "dist": dist,
        }
        return float(rew), info

    # ------------------------------------------------------------------ step
    def _apply_action(self, action: np.ndarray) -> None:
        """Map the 22-d action onto actuator setpoints. No qpos/qvel writes."""
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)

        # wrist: pose DELTA, integrated into the setpoint then clipped to ctrlrange
        dpos = a[0:3] * np.array([0.012, 0.012, 0.008])
        drot = a[3:6] * np.array([0.10, 0.10, 0.10])
        target = self.data.ctrl[0:N_WRIST].copy()
        target[0:3] = target[0:3] + dpos
        target[3:6] = target[3:6] + drot

        # fingers: absolute setpoint spanning the joint ctrlrange
        lo, hi = self.ctrl_low[N_WRIST:], self.ctrl_high[N_WRIST:]
        finger_target = lo + (a[N_WRIST:] + 1.0) * 0.5 * (hi - lo)

        ctrl = np.concatenate([target, finger_target])
        self.data.ctrl[:] = np.clip(ctrl, self.ctrl_low, self.ctrl_high)
        self._ctrl_ref = self.data.ctrl.copy()

    def step(self, action: np.ndarray):
        if not isinstance(action, np.ndarray) or action.shape != (N_ACT,):
            raise ValueError(f"action must be shape ({N_ACT},), got {action!r}")
        self._apply_action(action)
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
        rew, rinfo = self._reward()

        self._ep["grasp"].append(rinfo["grasp"])
        self._ep["hold"].append(rinfo["min_hold"])
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
        self.trace["contact_pairs"].append([list(p) for p in rinfo["pairs"]])
        # per-step state the acceptance gates segment on: vertical velocity (for the
        # gravity-consistency check on a free-falling box) and whether the box was
        # supported / finger-held at this step.
        self.trace["obj_vz"].append(float(self.data.qvel[self.obj_qvel_adr + 2]))
        self.trace["supported"].append(rinfo["supported"])
        self.trace["finger_held"].append(rinfo["finger_held"])
        self.last_contact_pairs = [(a, b) for a, b in rinfo["pairs"]]

        info = {
            "lift": rinfo["lift"],
            "grasp": rinfo["grasp"],
            "hold": rinfo["min_hold"],
            "z_peak": self._z_peak,
            "contact_force": rinfo["contact_force"],
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
            "z_peak": float(self._z_peak - self._z0),
            "contact_force": float(np.mean(self.trace["contact_force"])),
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
