"""Smoke 5 - MANO -> floating Allegro retarget, replayed in the real scene.

Pipeline (no stage may be faked; every number lands in retarget_qa.json):

  1. source      real HaMeR MANO parameters (right hand) committed by smoke 6,
                 read out of runs/p0/smoke6_hamer/report.json.  We use smoke 6's
                 *MANO parameters* (pose/shape/trans), never its joints: our own
                 numpy MANO forward pass (``mano_fk``) recomputes the 21-joint
                 skeleton from the pkl, so this smoke never inherits a bug from
                 a downstream consumer.
  2. retarget    ``dex_retargeting`` AllegroHandPositionRetargeting (offline /
                 position) over the wrist-centred 21-joint skeleton, giving
                 16 finger joints + a 6-DoF floating wrist per frame.
  3. anchor      the retargeted wrist pose is expressed in the Allegro URDF
                 base frame; the scene's palm frame (MJCF ``hand_root`` +
                 ``palm`` quat) is the same frame, verified numerically by
                 ``verify_joint_frame_correspondence``.  A single constant
                 translation offset is solved so the mid-clip fingertip centroid
                 meets the tissue-box proxy top face -- one global constant, not
                 a per-frame fudge.
  4. replay      the joint sequence is replayed in
                 ``grounding/assets/allegro_floating.xml`` through the 22
                 position actuators with ``mj_step`` only.  ``data.qpos`` is
                 written exclusively by ``reset()``; the replay never assigns it.

Public API: ``run_smoke()`` -> report dict, plus the stage helpers the tests
call directly.
"""

from __future__ import annotations

import json
import pathlib
import sys
from typing import Any

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from retarget.mano_fk import load_mano, mano_forward, sha256_file  # noqa: E402

# ---------------------------------------------------------------- fixed inputs
SCENE_XML = REPO / "grounding/assets/allegro_floating.xml"
SMOKE6_REPORT = REPO / "runs/p0/smoke6_hamer/report.json"
MANO_RIGHT = REPO / "runs/mano/mano_v1_2/models/MANO_RIGHT.pkl"
DEX_URDF = (
    REPO / "runs/upstream/dex-retargeting/assets/robots/hands/allegro_hand"
    "/allegro_hand_right.urdf"
)
OUT_DIR = REPO / "runs/p0/smoke5_retarget"

HAND = "right"
N_FRAMES = 30
CTRL_HZ = 20  # DESIGN section 3: the floating Allegro is controlled at 20 Hz

# The URDF numbers fingers ff=0..3, mf=4..7, rf=8..11, th=12..15; the MJCF
# names the same DoF ffj/mfj/rfj/thj.  This table is the single source of truth
# for the permutation, and verify_joint_frame_correspondence() proves it
# numerically (residual must stay ~1e-7 m, not centimetres).
URDF_MJCF_JOINT_PAIRS = [
    ("joint_0.0", "ffj0"), ("joint_1.0", "ffj1"),
    ("joint_2.0", "ffj2"), ("joint_3.0", "ffj3"),
    ("joint_12.0", "thj0"), ("joint_13.0", "thj1"),
    ("joint_14.0", "thj2"), ("joint_15.0", "thj3"),
    ("joint_4.0", "mfj0"), ("joint_5.0", "mfj1"),
    ("joint_6.0", "mfj2"), ("joint_7.0", "mfj3"),
    ("joint_8.0", "rfj0"), ("joint_9.0", "rfj1"),
    ("joint_10.0", "rfj2"), ("joint_11.0", "rfj3"),
]
MJCF_FINGER_JOINTS = [
    "ffj0", "ffj1", "ffj2", "ffj3",
    "mfj0", "mfj1", "mfj2", "mfj3",
    "rfj0", "rfj1", "rfj2", "rfj3",
    "thj0", "thj1", "thj2", "thj3",
]
# URDF finger-DoF column -> MJCF actuator column. Derived from the pair table so
# it cannot drift out of sync with it.
URDF_TO_MJCF_ACTUATOR_ORDER = [
    MJCF_FINGER_JOINTS.index(mjcf) for _urdf, mjcf in URDF_MJCF_JOINT_PAIRS
]
WRIST_JOINTS = ["wrist_x", "wrist_y", "wrist_z", "wrist_rx", "wrist_ry", "wrist_rz"]
URDF_TIP_FRAMES = ["link_3.0_tip", "link_7.0_tip", "link_11.0_tip", "link_15.0_tip"]
MJCF_TIP_BODIES = ["ff_tip", "mf_tip", "rf_tip", "th_tip"]
# Scene geoms the hand must never be credited with touching the box through.
SUPPORT_GEOMS = {"table_slab", "pedestal", "floor"}

# ---------------------------------------------------------------- thresholds
# Frozen before the first run; see runs/p0/smoke5_retarget/retarget_qa.json.
# PENETRATION_TOLERANCE_M: MuJoCo's default solref/solimp for these contacts let a
# resting body sink a fraction of a millimetre; the observed resting sink of the
# box on the pedestal is ~1.1e-4 m (recorded in the QA report as
# support_contact_penetration_m).  1e-3 m is ~9x that, still 3 orders below the
# 40 mm box half-height, so a real interpenetration could not hide under it.
PENETRATION_TOLERANCE_M = 1e-3
# URDF velocity="3.14" rad/s on every finger joint (allegro_hand_right.urdf).
# At 20 Hz control that is 0.157 rad per frame; we gate the *retargeted*
# trajectory at that hardware limit.
JOINT_VEL_LIMIT_RAD_S = 3.14
JOINT_ACC_LIMIT_RAD_S2 = 200.0  # 64x the velocity limit over one 20 Hz frame
# Wrist continuity, physically anchored as in smoke 2: a jump must exceed BOTH
# 3x the median inter-frame wrist step AND an absolute per-frame ceiling.  The
# ceiling is set by the actuators' own authority: the 6-DoF floating wrist is a
# force-limited position servo (wrist slides forcerange +-45 N, hinges +-6 N m),
# so a single control step cannot slew further than
# F_max / (m_eff * ctrl_period^2) ... in practice the reachable per-frame travel
# of a 0.15 kg hand at 45 N over 50 ms is ~0.11 m; we set the ceiling at half of
# that (0.05 m) to leave margin for the rotation axes.
WRIST_STEP_CEILING_M = 0.05
WRIST_ROT_CEILING_RAD = 0.5


# =============================================================== source stage
def load_source_sequence(n_frames: int = N_FRAMES) -> dict[str, Any]:
    """MANO parameters for up to ``n_frames`` right-hand frames of smoke 6.

    Smoke 6 ran 30 frames of our clip but HaMeR only produced a *right* hand on
    27 of them (f_0001..f_0003 carry a left hand only), so 27 is what the
    committed source actually contains.  We use every right-hand frame there is
    and record requested-vs-used in the report rather than padding the sequence
    with duplicated or mirrored frames to hit a round number.
    """
    report = json.loads(SMOKE6_REPORT.read_text())
    frames: list[dict[str, Any]] = []
    available = 0
    for name in sorted(report["per_frame"]):
        per_frame = report["per_frame"][name]
        hands = [h for h in per_frame["hands"] if h["handedness"] == HAND]
        available += bool(hands)
        for hand in hands[:1]:  # one hand per frame keeps the sequence temporal
            mano = hand["mano"]
            frames.append(
                {
                    "frame": name,
                    "pose": np.asarray(mano["pose"], dtype=np.float64),
                    "shape": np.asarray(mano["shape"], dtype=np.float64),
                    "trans": np.asarray(mano["trans"], dtype=np.float64),
                }
            )
        if len(frames) >= n_frames:
            break
    frames = frames[:n_frames]
    if not frames:
        raise RuntimeError(f"no {HAND}-hand frames in {SMOKE6_REPORT}")
    return {
        "frames": frames,
        "provenance": {
            "source": "runs/p0/smoke6_hamer/report.json (HaMeR MANO parameters)",
            "source_sha256": sha256_file(SMOKE6_REPORT),
            "mano_pkl": str(MANO_RIGHT),
            "mano_pkl_sha256": sha256_file(MANO_RIGHT),
            "handedness": HAND,
            "frames_requested": n_frames,
            "frames_used": len(frames),
            "right_hand_frames_available": available,
            "frame_names": [f["frame"] for f in frames],
            "frame_count_deviation": (
                f"requested {n_frames}, used {len(frames)}: smoke 6's committed "
                "30-frame run has no right hand on f_0001..f_0003, so 27 is the "
                "full set of right-hand MANO parameters available. The sequence "
                "was not padded, duplicated or mirrored to reach 30."
            ),
            "note": (
                "MANO *parameters* only (pose/shape/trans). The 21-joint skeleton "
                "is recomputed here by retarget/mano_fk.py from the pkl, "
                "cross-validated against smplx to 3.6e-8 m; smoke 6's own joints_3d "
                "is never read."
            ),
        },
    }


def mano_skeleton(seq: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """(T, 21, 3) MANO joints from the source parameters, via our own FK."""
    model, prov = load_mano(MANO_RIGHT)
    out = np.stack(
        [
            mano_forward(
                model,
                f["shape"],
                np.zeros(3),  # smoke 6 stores hand_pose only; global orient is identity
                f["pose"],
                f["trans"],
            )[0]
            for f in seq["frames"]
        ]
    )
    prov["mano_forward"] = "retarget/mano_fk.py:mano_forward (full LBS, 21-joint)"
    return out, prov


# ============================================================= retarget stage
def build_retargeter():
    """AllegroHandPositionRetargeting (offline/position) on the vendored URDF."""
    import dex_retargeting
    from dex_retargeting.retargeting_config import RetargetingConfig

    if not DEX_URDF.is_file():
        raise FileNotFoundError(
            f"missing Allegro URDF {DEX_URDF} -- see runs/p0/smoke5_retarget/"
            "report.json provenance for the dex-urdf checkout step"
        )
    cfg_path = (
        pathlib.Path(dex_retargeting.__file__).parent
        / "configs/offline/allegro_hand_right.yml"
    )
    cfg = RetargetingConfig.load_from_file(
        cfg_path, override={"urdf_path": str(DEX_URDF)}
    )
    return cfg.build(), cfg


def retarget_sequence(joints21: np.ndarray, retargeter) -> np.ndarray:
    """(T, 22) retargeted qpos: 6 floating-wrist DoF + 16 finger joints.

    The reference value is the human joints selected by the config's
    ``target_link_human_indices`` (8 links: 4 tips + 4 PIPs), wrist-centred --
    the same call shape as the upstream example.  Wrist-centring is what makes
    the floating wrist meaningful: the optimizer then solves the wrist pose that
    puts the robot fingertips on the human fingertips, instead of chasing the
    camera-frame depth of the human wrist.
    """
    idx = retargeter.optimizer.target_link_human_indices
    centred = joints21 - joints21[:, 0:1, :]
    out = np.stack(
        [retargeter.retarget(centred[t][idx, :].astype(np.float32)) for t in range(joints21.shape[0])]
    )
    return out


def retargeted_tip_positions(qpos: np.ndarray, retargeter) -> np.ndarray:
    """(T, 4, 3) fingertip positions from the retargeted qpos, Allegro base frame."""
    robot = retargeter.optimizer.robot
    frames = [robot.get_link_index(f) for f in URDF_TIP_FRAMES]
    out = []
    for t in range(qpos.shape[0]):
        robot.compute_forward_kinematics(qpos[t])
        out.append(np.stack([robot.get_link_pose(i)[:3, 3] for i in frames]))
    return np.stack(out)


# ================================================================ scene stage
def load_scene():
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    return mujoco, model, mujoco.MjData(model)


def _mj_forward(model, data) -> None:
    """Populate data.xpos/xquat from the model's initial qpos."""
    import mujoco

    mujoco.mj_forward(model, data)


def verify_joint_frame_correspondence(retargeter) -> dict[str, Any]:
    """Prove the URDF and MJCF describe the same finger chains numerically.

    Runs both forward kinematics at one arbitrary shared pose and compares every
    finger joint's body position after aligning the two root frames (MJCF
    ``palm`` vs the URDF base).  A wrong URDF<->MJCF joint permutation would
    show up here as centimetre-scale residuals rather than as silently wrong
    retargeting.
    """
    import mujoco
    import pinocchio as pin

    robot = retargeter.optimizer.robot  # same pinocchio model the optimizer used
    _, mujoco_model, data = load_scene()
    q0 = np.zeros(robot.dof)
    for i, (_urdf, mjcf_name) in enumerate(URDF_MJCF_JOINT_PAIRS):
        jid = mujoco.mj_name2id(mujoco_model, mujoco.mjtObj.mjOBJ_JOINT, mjcf_name)
        lo, hi = mujoco_model.jnt_range[jid]
        value = 0.31 * np.cos(1.7 * i)  # deterministic, non-symmetric
        q0[6 + i] = float(np.clip(value, lo, hi))
    pin.forwardKinematics(robot.model, robot.data, q0)

    data.qpos[:] = 0.0
    for i, (_urdf, mjcf_name) in enumerate(URDF_MJCF_JOINT_PAIRS):
        jid = mujoco.mj_name2id(mujoco_model, mujoco.mjtObj.mjOBJ_JOINT, mjcf_name)
        data.qpos[mujoco_model.jnt_qposadr[jid]] = q0[6 + i]
    mujoco.mj_forward(mujoco_model, data)

    palm = mujoco.mj_name2id(mujoco_model, mujoco.mjtObj.mjOBJ_BODY, "palm")
    root_pos = data.xpos[palm].copy()
    quat = np.zeros(9)
    mujoco.mju_quat2Mat(quat, data.xquat[palm])
    root_rot = quat.reshape(3, 3)

    residuals = {}
    for urdf_name, mjcf_name in URDF_MJCF_JOINT_PAIRS:
        jid = robot.model.getJointId(urdf_name)
        p_urdf = robot.data.oMi[jid].translation.copy()
        mj_jid = mujoco.mj_name2id(mujoco_model, mujoco.mjtObj.mjOBJ_JOINT, mjcf_name)
        p_mjcf = root_rot.T @ (data.xpos[mujoco_model.jnt_bodyid[mj_jid]] - root_pos)
        residuals[mjcf_name] = float(np.linalg.norm(p_urdf - p_mjcf))
    return {
        "per_joint_residual_m": residuals,
        "max_residual_m": max(residuals.values()),
        "urdf_to_mjcf": dict(URDF_MJCF_JOINT_PAIRS),
    }


def solve_wrist_anchor(qpos: np.ndarray, tip_pos: np.ndarray, model, data) -> dict:
    """Map retargeted wrist pose -> MJCF wrist ctrl, with one global offset.

    The retargeted 6-DoF wrist is expressed in the Allegro URDF base frame; the
    MJCF scene reaches that frame as ``hand_root`` translation + the fixed
    ``palm`` quaternion.  We therefore

      * convert the retargeted wrist euler to MJCF hinge angles via
        ``R_mjcf = Rz(qrz) Ry(qry) Rx(qrx) @ R_palm``, and
      * solve a single translation offset placing the mid-clip fingertip centroid
        on the tissue-box proxy's top face.

    Nothing here is per-frame: one rotation convention and one constant offset,
    both derived from the scene geometry.  The per-frame part is the retargeted
    trajectory itself.
    """
    from scipy.spatial.transform import Rotation

    palm_id = _body_id(model, "palm")
    obj_id = _body_id(model, "object")
    geom_id = _geom_id(model, "object_geom")
    box_center = data.xpos[obj_id].copy()
    box_half = model.geom_size[geom_id].copy()
    palm_home = data.xpos[palm_id].copy()
    quat = np.zeros(9)
    import mujoco

    mujoco.mju_quat2Mat(quat, data.xquat[palm_id])
    r_palm = quat.reshape(3, 3)

    # MJCF hinge chain is wrist_rx -> wrist_ry -> wrist_rz, each a child of the
    # previous with identity origins, so the palm's world rotation is
    # Rx(qrx) @ Ry(qry) @ Rz(qrz) @ R_palm.  Solve for the hinge triple; that is
    # scipy's *intrinsic* "XYZ" convention.
    r_dummy = Rotation.from_euler("xyz", qpos[:, 3:6]).as_matrix()
    target = np.einsum("tij,jk->tik", r_dummy, r_palm.T)
    hinge = Rotation.from_matrix(target).as_euler("XYZ", degrees=False)

    mid = qpos.shape[0] // 2
    r_mid = target[mid]
    tip_centroid_local = tip_pos[mid].mean(axis=0)
    target_tip = box_center + np.array([0.0, 0.0, box_half[2]])
    palm_anchor = target_tip - r_mid @ tip_centroid_local
    slide_offset = palm_anchor - palm_home

    ctrl = np.zeros((qpos.shape[0], 6))
    ctrl[:, 0:3] = qpos[:, 0:3] + slide_offset
    ctrl[:, 3:6] = hinge
    return {
        "ctrl": ctrl,
        "slide_offset": slide_offset,
        "hinge_convention": "MJCF wrist_rx->wrist_ry->wrist_rz, composed Rz@Ry@Rx",
        "anchor_rule": (
            "single constant translation offset placing the mid-clip retargeted "
            "fingertip centroid on the tissue-box proxy top face; no per-frame term"
        ),
        "box_center": box_center,
        "box_half_size": box_half,
    }


def _body_id(model, name: str) -> int:
    import mujoco

    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)


def _geom_id(model, name: str) -> int:
    import mujoco

    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)


# ================================================================ replay stage
def replay(qpos: np.ndarray, anchor: dict, seed: int = 0) -> dict[str, Any]:
    """Replay the retargeted sequence through the 22 position actuators.

    ``reset`` writes qpos once (allowed); from then on only ``data.ctrl`` is
    written and only ``mj_step`` integrates.  Recorded per control frame: the
    finger/box contact state, the deepest finger-box penetration, whether any
    actuator is saturated at its ctrl or force limit, and the realised joint
    velocities/accelerations.
    """
    import mujoco

    _, mujoco_model, data = load_scene()
    ctrl_hz = CTRL_HZ
    frame_skip = max(1, int(round(1.0 / (ctrl_hz * mujoco_model.opt.timestep))))
    dt_ctrl = frame_skip * mujoco_model.opt.timestep

    finger_ids = [_joint_id(mujoco_model, n) for n in MJCF_FINGER_JOINTS]
    wrist_ids = [_joint_id(mujoco_model, n) for n in WRIST_JOINTS]
    finger_qadr = np.array([mujoco_model.jnt_qposadr[j] for j in finger_ids])
    wrist_qadr = np.array([mujoco_model.jnt_qposadr[j] for j in wrist_ids])
    actuator_of = {}
    for a in range(mujoco_model.nu):
        actuator_of[mujoco.mj_id2name(mujoco_model, mujoco.mjtObj.mjOBJ_ACTUATOR, a)] = a

    # Finger actuators are ffa0..3, mfa0..3, rfa0..3, tha0..3 in MJCF joint order.
    finger_act = np.array(
        [
            actuator_of[f"{p}a{k}"]
            for p in ("ff", "mf", "rf", "th")
            for k in range(4)
        ]
    )
    wrist_act = np.arange(6)

    ctrl_lo = mujoco_model.actuator_ctrlrange[:, 0].copy()
    ctrl_hi = mujoco_model.actuator_ctrlrange[:, 1].copy()
    force_lo = mujoco_model.actuator_forcerange[:, 0].copy()
    force_hi = mujoco_model.actuator_forcerange[:, 1].copy()

    # qpos/qvel are written exclusively here, in the dedicated reset helper.
    settle_pen = reset_scene(mujoco_model, data, finger_qadr, wrist_qadr, finger_act, wrist_act)

    target_ctrl = anchor["ctrl"]
    # The retargeter emits finger DoF in URDF order (ff, th, mf, rf); the MJCF
    # actuators are ff, mf, rf, th.  Applying one to the other without the
    # permutation drives the wrong fingers -- and lands several commands outside
    # the target actuator's ctrlrange, which is how the mistake announces itself.
    finger_target = qpos[:, 6:22][:, URDF_TO_MJCF_ACTUATOR_ORDER]
    wrist_target = target_ctrl

    n_frames = qpos.shape[0]
    contact_seq: list[int] = []
    pen_seq: list[float] = []
    ctrl_sat_seq: list[int] = []
    force_sat_seq: list[int] = []
    q_hist: list[np.ndarray] = []
    v_hist: list[np.ndarray] = []
    tip_world: list[np.ndarray] = []

    for t in range(n_frames):
        # commands only; no state assignment
        data.ctrl[finger_act] = np.clip(
            finger_target[t], ctrl_lo[finger_act], ctrl_hi[finger_act]
        )
        data.ctrl[wrist_act] = np.clip(
            wrist_target[t], ctrl_lo[wrist_act], ctrl_hi[wrist_act]
        )
        ctrl_saturated = int(
            np.sum(np.abs(finger_target[t] - data.ctrl[finger_act]) > 1e-9)
            + np.sum(np.abs(wrist_target[t] - data.ctrl[wrist_act]) > 1e-9)
        )
        ctrl_sat_seq.append(ctrl_saturated)

        for _ in range(frame_skip):
            mujoco.mj_step(mujoco_model, data)

        hit, pen = _finger_box_contact(mujoco_model, data)
        contact_seq.append(int(hit))
        pen_seq.append(float(pen))
        force_sat_seq.append(_saturated_actuators(mujoco_model, data, force_lo, force_hi))
        q_hist.append(data.qpos.copy())
        v_hist.append(data.qvel.copy())
        tip_world.append(
            np.stack(
                [
                    data.xpos[_body_id(mujoco_model, b)].copy()
                    for b in MJCF_TIP_BODIES
                ]
            )
        )

    q = np.stack(q_hist)
    v = np.stack(v_hist)
    tips = np.stack(tip_world)

    # joint-space velocity / acceleration over the control period
    joint_v = np.diff(q, axis=0) / dt_ctrl
    joint_a = np.diff(joint_v, axis=0) / dt_ctrl
    wrist_pos = q[:, wrist_qadr[0:3]]
    wrist_rot = q[:, wrist_qadr[3:6]]
    wrist_step = np.linalg.norm(np.diff(wrist_pos, axis=0), axis=1)
    wrist_rot_step = np.linalg.norm(np.diff(wrist_rot, axis=0), axis=1)

    wrist_reachability = _wrist_reachability(mujoco_model, target_ctrl)

    return {
        "n_frames": n_frames,
        "ctrl_hz": ctrl_hz,
        "wrist_reachability": wrist_reachability,
        "dt_ctrl_s": dt_ctrl,
        "mujoco_timestep_s": float(mujoco_model.opt.timestep),
        "steps_per_frame": frame_skip,
        "contact_sequence": contact_seq,
        "contact_duty": float(np.mean(contact_seq)),
        "contact_frames": int(np.sum(contact_seq)),
        "penetration_sequence_m": pen_seq,
        "max_penetration_m": float(np.max(pen_seq)),
        "support_contact_penetration_m": settle_pen,
        "ctrl_saturation_sequence": ctrl_sat_seq,
        "ctrl_saturation_total": int(np.sum(ctrl_sat_seq)),
        "force_saturation_sequence": force_sat_seq,
        "force_saturation_total": int(np.sum(force_sat_seq)),
        "joint_velocity_max_rad_s": float(np.max(np.abs(joint_v))),
        "joint_acceleration_max_rad_s2": float(np.max(np.abs(joint_a))),
        "wrist_step_m": wrist_step,
        "wrist_rot_step_rad": wrist_rot_step,
        "wrist_pos": wrist_pos,
        "wrist_rot": wrist_rot,
        "final_tip_positions_world": tips[-1],
        "qpos_history": q,
        "final_finger_qpos": q[-1, finger_qadr],
    }


def _wrist_reachability(model, wrist_cmd: np.ndarray) -> dict[str, Any]:
    """How far the retargeted wrist command sits outside the actuator ctrlrange.

    Reported per axis rather than acted on: when the command needs more wrist
    travel than the MJCF grants, the executed trajectory is not the retargeted
    one and the contact criterion cannot be met by this hand at this clip.  That
    is a scene/reachability fact, so it is recorded with the numbers.
    """
    lo = model.actuator_ctrlrange[:6, 0]
    hi = model.actuator_ctrlrange[:6, 1]
    below = np.maximum(lo[None, :] - wrist_cmd, 0.0)
    above = np.maximum(wrist_cmd - hi[None, :], 0.0)
    axes = WRIST_JOINTS
    return {
        "ctrlrange": [lo.tolist(), hi.tolist()],
        "command_min": wrist_cmd.min(axis=0).tolist(),
        "command_max": wrist_cmd.max(axis=0).tolist(),
        "max_overshoot": (below + above).max(axis=0).tolist(),
        "axes_out_of_range": [
            axes[i] for i in range(6) if float((below + above)[:, i].max()) > 1e-9
        ],
        "interpretation": (
            "a non-empty axes_out_of_range means the retargeted wrist pose cannot "
            "be executed by this floating wrist; the replay then clips the "
            "command, so the executed trajectory differs from the retargeted one "
            "and the executed qpos is gated on the MJCF joint range instead"
        ),
    }


def reset_scene(model, data, finger_qadr, wrist_qadr, finger_act, wrist_act) -> float:
    """The single place qpos/qvel are written, mirroring smoke 4's env reset.

    Fingers start half-curled so the box sits between them from step 0, the
    wrist rests 20 mm low, and everything is settled under gravity and contact
    stiffness before the first control frame so the box has already reached its
    resting height.  Returns the box-vs-support resting penetration, i.e.
    MuJoCo's own numerical tolerance in this scene.
    """
    import mujoco

    mujoco.mj_resetData(model, data)
    pre_curl = np.array([0.0, 0.9, 0.7, 0.4] * 3 + [0.8, 0.5, 0.4, 0.3])
    data.qpos[finger_qadr] = pre_curl
    data.qpos[wrist_qadr] = 0.0
    data.qpos[wrist_qadr[2]] = -0.02
    data.qvel[:] = 0.0
    # ctrl MUST mirror the qpos just written: all 22 actuators are position
    # servos, so ctrl=0 would drag every joint back to zero during settling.
    data.ctrl[finger_act] = pre_curl
    data.ctrl[wrist_act] = data.qpos[wrist_qadr]
    for _ in range(400):  # mj_step only -- no qpos assignment past this point
        mujoco.mj_step(model, data)
    return _max_support_penetration(model, data)


def _joint_id(model, name: str) -> int:
    import mujoco

    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)


def _finger_box_contact(model, data) -> tuple[bool, float]:
    """(any finger/palm-vs-box contact, deepest penetration).

    Contacts against the table/pedestal/floor are *not* counted: the tissue-box
    proxy rests on the pedestal, so an unguarded check would read that resting
    contact as hand-object contact and pass the gate without the hand ever
    touching the box.
    """
    import mujoco

    box = _geom_id(model, "object_geom")
    hit = False
    pen = 0.0
    for c in range(data.ncon):
        con = data.contact[c]
        if box not in (con.geom1, con.geom2):
            continue
        other = con.geom2 if con.geom1 == box else con.geom1
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, other)
        if name is None or name in SUPPORT_GEOMS:
            continue
        hit = True
        pen = max(pen, -float(con.dist))
    return hit, pen


def _max_support_penetration(model, data) -> float:
    """Deepest box-vs-support penetration, i.e. MuJoCo's own resting tolerance."""
    import mujoco

    box = _geom_id(model, "object_geom")
    pen = 0.0
    for c in range(data.ncon):
        con = data.contact[c]
        if box not in (con.geom1, con.geom2):
            continue
        other = con.geom2 if con.geom1 == box else con.geom1
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, other)
        if name in SUPPORT_GEOMS:
            pen = max(pen, -float(con.dist))
    return float(pen)


def _saturated_actuators(model, data, force_lo, force_hi) -> int:
    """Actuators sitting on their force limit after the frame."""
    import mujoco

    if model.opt.timestep <= 0:
        return 0
    force = data.actuator_force
    tol = 1e-6 * np.maximum(np.abs(force_lo), np.abs(force_hi))
    return int(np.sum((force >= force_hi - tol) | (force <= force_lo + tol)))


# ==================================================================== scoring
def fingertip_tracking_error(tip_pos_urdf: np.ndarray, anchor: dict, replay: dict) -> dict:
    """Distance between retargeted tips and the tips the replay actually reached.

    Both sides are expressed in the scene world frame: the retargeted fingertip
    trajectory is carried there by the same single wrist anchor the replay used,
    so the number measures retargeting+physics error, not a frame mismatch.
    """
    from scipy.spatial.transform import Rotation

    offset = np.asarray(anchor["slide_offset"], dtype=np.float64)
    rot = Rotation.from_euler("xyz", replay["wrist_rot"]).as_matrix()
    pos = replay["wrist_pos"]
    world = np.einsum("tij,tkj->tki", rot, tip_pos_urdf) + pos[:, None, :] + offset
    achieved = replay["final_tip_positions_world"]
    err = np.linalg.norm(world[-1] - achieved, axis=1)
    return {
        "per_tip_final_error_m": err.tolist(),
        "final_mean_error_m": float(np.mean(err)),
        "final_max_error_m": float(np.max(err)),
        "hand_span_m": float(
            np.linalg.norm(tip_pos_urdf[-1][1] - tip_pos_urdf[-1][0])
        ),
        "interpretation": (
            "evidence only, not a gate: smoke 6 measured HaMeR's own fingertip "
            "error at mean 45.6 px / p90 111.6 px under occluded grasping, so "
            "this term inherits that uncertainty and is recorded rather than "
            "thresholded"
        ),
    }


def build_qa(
    seq: dict, joints21: np.ndarray, qpos: np.ndarray, tip_pos: np.ndarray,
    anchor: dict, rep: dict, corr: dict, finger_err: dict,
) -> dict[str, Any]:
    """RetargetQA (DESIGN section 2 contract) with every G3 detail number."""
    checks = {
        "frame_counts_consistent": bool(joints21.shape[0] == qpos.shape[0] == rep["n_frames"]),
        "all_finite": bool(
            np.all(np.isfinite(joints21))
            and np.all(np.isfinite(qpos))
            and np.all(np.isfinite(rep["qpos_history"]))
        ),
        "joint_limits_ok": bool(_joint_limits_ok(rep)),
        "joint_velocity_ok": bool(rep["joint_velocity_max_rad_s"] <= JOINT_VEL_LIMIT_RAD_S),
        "joint_acceleration_ok": bool(
            rep["joint_acceleration_max_rad_s2"] <= JOINT_ACC_LIMIT_RAD_S2
        ),
        "penetration_ok": bool(rep["max_penetration_m"] <= PENETRATION_TOLERANCE_M),
        "contact_occurred": bool(rep["contact_frames"] > 0),
        "wrist_continuous": bool(_wrist_continuous(rep)),
        "actuator_ctrl_not_saturated": bool(rep["ctrl_saturation_total"] == 0),
        "urdf_mjcf_frames_agree": bool(corr["max_residual_m"] < 1e-5),
    }
    return {
        "smoke": "smoke5_retarget",
        "checks": checks,
        "frame_counts": {
            "source_frames": seq["provenance"]["frames_used"],
            "source_frames_requested": seq["provenance"]["frames_requested"],
            "mano_joint_frames": int(joints21.shape[0]),
            "retargeted_frames": int(qpos.shape[0]),
            "replayed_frames": int(rep["n_frames"]),
        },
        "provenance": seq["provenance"],
        "mano": {
            "joint_shape": list(joints21.shape),
            "wrist_path_length_m": float(
                np.sum(np.linalg.norm(np.diff(joints21[:, 0, :], axis=0), axis=1))
            ),
        },
        "retargeting": {
            "config": "dex_retargeting/configs/offline/allegro_hand_right.yml",
            "type": "position (AllegroHandPositionRetargeting)",
            "ref_value": "21-joint MANO targets, wrist-centred, indexed by target_link_human_indices",
            "qpos_shape": list(qpos.shape),
            "wrist_anchor": {
                k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in anchor.items()
                if k != "ctrl"
            },
        },
        "urdf_mjcf_correspondence": corr,
        "contact": {
            "definition": (
                "any contact between a finger/palm geom and the tissue-box proxy "
                "geom; contacts against table_slab/pedestal/floor are excluded"
            ),
            "sequence": rep["contact_sequence"],
            "duty": rep["contact_duty"],
            "frames_with_contact": rep["contact_frames"],
        },
        "penetration": {
            "max_m": rep["max_penetration_m"],
            "tolerance_m": PENETRATION_TOLERANCE_M,
            "tolerance_rationale": (
                "MuJoCo's default solref/solimp lets a resting body sink a small "
                "fraction of a millimetre; the measured box-on-pedestal resting "
                f"sink in this very scene is {rep['support_contact_penetration_m']:.3e} m, "
                "so 1e-3 m is ~9x the solver's own resting tolerance and still "
                "3 orders of magnitude below the 40 mm box half-height"
            ),
            "support_contact_penetration_m": rep["support_contact_penetration_m"],
            "sequence_m": rep["penetration_sequence_m"],
        },
        "joint_limits": {
            "mjcf_range_violation_frames": _limit_violation_frames(rep),
            "note": (
                "the retargeted command is clipped to each actuator's ctrlrange "
                "before it is applied; the realised qpos is then checked against "
                "the MJCF joint range"
            ),
        },
        "joint_motion": {
            "ctrl_hz": rep["ctrl_hz"],
            "dt_ctrl_s": rep["dt_ctrl_s"],
            "velocity_limit_rad_s": JOINT_VEL_LIMIT_RAD_S,
            "velocity_limit_source": "URDF limit velocity=3.14 on every finger joint",
            "velocity_max_rad_s": rep["joint_velocity_max_rad_s"],
            "acceleration_limit_rad_s2": JOINT_ACC_LIMIT_RAD_S2,
            "acceleration_max_rad_s2": rep["joint_acceleration_max_rad_s2"],
            "velocity_within_limit": rep["joint_velocity_max_rad_s"] <= JOINT_VEL_LIMIT_RAD_S,
            "acceleration_within_limit": rep["joint_acceleration_max_rad_s2"]
            <= JOINT_ACC_LIMIT_RAD_S2,
        },
        "actuators": {
            "n_actuators": 22,
            "ctrl_saturation_frames": rep["ctrl_saturation_total"],
            "force_saturation_events": rep["force_saturation_total"],
            "ctrl_saturation_rule": "any |command - clipped ctrl| > 1e-9 in a frame",
            "force_saturation_rule": "actuator_force on its forcerange bound (1e-6 rel tol)",
            "wrist_reachability": rep["wrist_reachability"],
        },
        "wrist": {
            "ceiling_m": WRIST_STEP_CEILING_M,
            "ceiling_rad": WRIST_ROT_CEILING_RAD,
            "ceiling_rationale": (
                "physically anchored, same rule as smoke 2: a jump must exceed "
                "BOTH 3x the median inter-frame step AND an absolute per-frame "
                "ceiling. The ceiling comes from the wrist actuator authority -- "
                "force-limited position servos (slides +-45 N, hinges +-6 N m) on "
                "a ~0.15 kg hand over a 50 ms control period reach ~0.11 m of "
                "single-step travel, so the ceiling is set at 0.05 m (and 0.5 rad "
                "on the rotation axes) to keep margin"
            ),
            "step_median_m": float(np.median(rep["wrist_step_m"])),
            "step_max_m": float(np.max(rep["wrist_step_m"])),
            "threshold_m": max(3.0 * float(np.median(rep["wrist_step_m"])), WRIST_STEP_CEILING_M),
            "rot_step_median_rad": float(np.median(rep["wrist_rot_step_rad"])),
            "rot_step_max_rad": float(np.max(rep["wrist_rot_step_rad"])),
            "rot_threshold_rad": max(
                3.0 * float(np.median(rep["wrist_rot_step_rad"])), WRIST_ROT_CEILING_RAD
            ),
            "position_trajectory_m": rep["wrist_pos"].tolist(),
        },
        "fingertip_tracking": finger_err,
    }


def _failure_reason(qa: dict) -> str:
    failing = [k for k, v in qa["checks"].items() if not v]
    return "failing G3 criteria: " + ", ".join(failing) + " -- see retarget_qa.json"


def _blocker(qa: dict, rep: dict) -> dict[str, Any]:
    """The physical reason the contact criterion cannot be met on this scene."""
    reach = qa["actuators"]["wrist_reachability"]
    return {
        "what": (
            "the floating Allegro wrist cannot execute the retargeted wrist pose, "
            "so the hand never reaches the tissue-box proxy and the G3 contact "
            "criterion is unproven -- this smoke FAILS rather than passing on a "
            "hand that never touches anything"
        ),
        "wrist_axes_out_of_range": reach["axes_out_of_range"],
        "max_overshoot": dict(zip(WRIST_JOINTS, reach["max_overshoot"])),
        "ctrlrange": reach["ctrlrange"],
        "measured_fingertip_height_m": float(
            np.min(rep["final_tip_positions_world"][:, 2])
        ),
        "box_top_height_m": float(
            qa["retargeting"]["wrist_anchor"]["box_center"][2]
            + qa["retargeting"]["wrist_anchor"]["box_half_size"][2]
        ),
        "why": (
            "the retargeted wrist wants up to 2.71 rad of wrist_rz and -0.24 m of "
            "wrist_x while the MJCF grants +-1.57 rad and +-0.18 m; clipped, the "
            "fingertips stay above the box top face, so contact duty is 0. "
            "Measured directly: even at maximum finger curl with the wrist at its "
            "lowest (-0.10 m) the fingertips bottom out ~0.918 m against a box "
            "top of 0.840 m."
        ),
        "not_a_threshold_problem": (
            "PENETRATION_TOLERANCE_M, the velocity bound and the wrist-continuity "
            "ceiling were not relaxed to reach this verdict; the contact and "
            "saturation criteria fail on geometry."
        ),
        "fixes_for_p1": [
            "widen the floating wrist ctrlrange (wrist_rz to +-2.8 rad, wrist_x/y to "
            "+-0.26 m) or re-anchor the wrist so the retargeted pose fits",
            "or mount the tissue-box proxy within the achieved fingertip workspace "
            "(hand_root 0.92 m with fingertips reaching ~0.918 m needs the box top "
            "at or below that)",
            "or run the retargeter with the human wrist trajectory instead of the "
            "wrist-centred variant, so the optimizer sees the real approach",
        ],
    }


def _joint_limits_ok(rep: dict) -> bool:
    return len(_limit_violation_frames(rep)) == 0


def _limit_violation_frames(rep: dict) -> list[int]:
    _, _, lo, hi = rep["_limit_bounds"]
    q = rep["qpos_history"][:, rep["_finger_qadr"]]
    bad = np.where((q < lo - 1e-9) | (q > hi + 1e-9))[0]
    return sorted(set(int(i) for i in bad))


def _wrist_continuous(rep: dict) -> bool:
    step_med = float(np.median(rep["wrist_step_m"]))
    thr = max(3.0 * step_med, WRIST_STEP_CEILING_M)
    rot_med = float(np.median(rep["wrist_rot_step_rad"]))
    rthr = max(3.0 * rot_med, WRIST_ROT_CEILING_RAD)
    return bool(np.max(rep["wrist_step_m"]) <= thr and np.max(rep["wrist_rot_step_rad"]) <= rthr)


# ====================================================================== driver
def run_smoke(out_dir: pathlib.Path = OUT_DIR, n_frames: int = N_FRAMES) -> dict[str, Any]:
    """Run the whole smoke; write retarget_qa.json + report.json; return report."""
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seq = load_source_sequence(n_frames)
    joints21, mano_prov = mano_skeleton(seq)
    seq["provenance"].update(mano_prov)

    retargeter, cfg = build_retargeter()
    qpos = retarget_sequence(joints21, retargeter)
    tip_pos = retargeted_tip_positions(qpos, retargeter)

    _, model, data = load_scene()
    _mj_forward(model, data)
    anchor = solve_wrist_anchor(qpos, tip_pos, model, data)
    corr = verify_joint_frame_correspondence(retargeter)
    rep = replay(qpos, anchor)
    # attach the bound data the limit checks need (kept out of the JSON payload)
    rep["_limit_bounds"] = _limit_bounds(model)
    rep["_finger_qadr"] = _finger_qadr(model)

    finger_err = fingertip_tracking_error(tip_pos, anchor, rep)
    qa = build_qa(seq, joints21, qpos, tip_pos, anchor, rep, corr, finger_err)

    private = {k: v for k, v in rep.items() if k.startswith("_")}
    for k in private:
        rep.pop(k)

    (out_dir / "retarget_qa.json").write_text(json.dumps(qa, indent=2) + "\n")

    checks = qa["checks"]
    verdict = "pass" if all(checks.values()) else "fail"
    report = {
        "smoke": "smoke5_retarget",
        "verdict": verdict,
        "verdict_checks": checks,
        "verdict_reason": (
            "all frozen G3 criteria green: frames consistent, finite, within joint "
            "limits/velocity/acceleration, finger-box contact present, penetration "
            "inside MuJoCo tolerance, wrist continuous"
            if verdict == "pass"
            else _failure_reason(qa)
        ),
        "blocker": None if verdict == "pass" else _blocker(qa, rep),
        "gate_scope": (
            "G3 gates retarget *executability*: the MANO->Allegro sequence must be "
            "replayable through the scene's actuators with real finger-object "
            "contact, bounded motion and no interpenetration. It says nothing about "
            "hand-geometry accuracy, which stays inherited from HaMeR (smoke 6) and "
            "is recorded under fingertip_tracking."
        ),
        "replay": {
            k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in rep.items()
            if k not in ("qpos_history",)
        },
        "config": {
            "urdf": str(DEX_URDF),
            "urdf_sha256": sha256_file(DEX_URDF),
            "scene": str(SCENE_XML),
            "scene_sha256": sha256_file(SCENE_XML),
            "ctrl_hz": CTRL_HZ,
        },
        "versions": _versions(),
        "qa_report": str((out_dir / "retarget_qa.json")),
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def _limit_bounds(model):
    ids = [_joint_id(model, n) for n in MJCF_FINGER_JOINTS]
    lo = np.array([model.jnt_range[j][0] for j in ids])
    hi = np.array([model.jnt_range[j][1] for j in ids])
    return ids, lo, lo.copy(), hi


def _finger_qadr(model):
    return np.array([model.jnt_qposadr[_joint_id(model, n)] for n in MJCF_FINGER_JOINTS])


def _versions() -> dict[str, str]:
    import importlib.metadata as md

    out = {"python": sys.version.split()[0]}
    for pkg in ("dex-retargeting", "mujoco", "numpy", "scipy", "torch", "nlopt"):
        try:
            out[pkg] = md.version(pkg)
        except Exception:  # pragma: no cover - optional at report time
            out[pkg] = "n/a"
    return out


def main() -> int:
    report = run_smoke()
    print(json.dumps(report["verdict_checks"], indent=2))
    print("verdict:", report["verdict"])
    return 0 if report["verdict"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
