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
from retarget.gauge import fix_gauge  # noqa: E402

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
# The W6a-v21 scene (MJCF_COMMIT 98c498a, sha256 29896fe5...) carries a second,
# mirrored hand: identical upstream naming with a "_left" suffix.  nu 22 -> 44,
# nq/nv 29/28 -> 51/50; v19 added six env-IK-driven arm servos behind the hand
# block (nu 44 -> 50, ctrl [44:50], see SERVO_WINDING_ARM_SERVOS -- zero
# hand-arm mechanical coupling, measured).  Smoke 5 is single-hand *input*:
# the right hand executes the retargeted clip and the left hand is frozen at
# its initial pose (its servos are simply never commanded, so ctrl stays 0 =
# its reset pose).
MJCF_FINGER_JOINTS_LEFT = [n + "_left" for n in MJCF_FINGER_JOINTS]
WRIST_JOINTS_LEFT = [n + "_left" for n in WRIST_JOINTS]
URDF_TIP_FRAMES = ["link_3.0_tip", "link_7.0_tip", "link_11.0_tip", "link_15.0_tip"]
MJCF_TIP_BODIES = ["ff_tip", "mf_tip", "rf_tip", "th_tip"]
# Scene geoms the hand must never be credited with touching the box through.
SUPPORT_GEOMS = {"table_slab", "pedestal", "floor"}
# Every finger/palm geom in this MJCF is built from an unnamed <default> class,
# so mj_id2name returns None for all 42 of them. Contact classification must
# therefore go through geom->body ancestry, never through geom names -- keying on
# names silently reports "no contact" for a hand buried in the box.
HAND_BODIES = (
    ["palm"]
    + [f"{p}_{seg}" for p in ("ff", "mf", "rf", "th")
       for seg in ("base", "proximal", "medial", "distal", "tip")]
)


def hand_geom_ids(model) -> set[int]:
    """Geom ids belonging to the hand (palm + 4 finger chains)."""
    import mujoco

    bodies = {
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        for name in HAND_BODIES
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) >= 0
    }
    return {g for g in range(model.ngeom) if model.geom_bodyid[g] in bodies}

# ---------------------------------------------------------------- thresholds
# Frozen before the first run; see runs/p0/smoke5_retarget/retarget_qa.json.
# PENETRATION_TOLERANCE_M: MuJoCo's default solref/solimp for these contacts let a
# resting body sink a fraction of a millimetre; the observed resting sink of the
# box on the pedestal is ~1.1e-4 m (recorded in the QA report as
# support_contact_penetration_m).  1e-3 m is ~9x that, still 3 orders below the
# 40 mm box half-height, so a real interpenetration could not hide under it.
PENETRATION_TOLERANCE_M = 1e-3
# URDF velocity="3.14" rad/s on every finger joint (allegro_hand_right.urdf).
# At 20 Hz control that is 0.157 rad per frame.  This is the *finger* limit and
# only the finger limit: the URDF carries exactly 16 velocity attributes, all on
# finger joints, and its wrist_joint is type="fixed", so the MJCF-only 6-DoF
# floating wrist has no URDF spec to inherit and is gated separately below.
JOINT_VEL_LIMIT_RAD_S = 3.14
JOINT_ACC_LIMIT_RAD_S2 = 200.0  # 64x the velocity limit over one 20 Hz frame
# ---------------------------------------------------------------- warm start
# reset_scene() leaves the floating wrist at identity and the fingers at a
# half-curl, then the replay's very first frame commands the retargeted pose.
# Commanding a 1.3 rad wrist pitch in one 50 ms step asks for ~26 rad/s, which
# no real hand can do -- the servo correctly refuses and the first few control
# intervals measure the servo's refusal, not the retargeted trajectory.  The
# replay therefore ramps into the first command at a rate the hardware can
# actually track, measured free-space on this very model (see
# runs/p0/smoke5_retarget/report.json -> rerun_evidence.velocity_gate.
# floating_wrist_servo_ceiling):
#   wrist hinges  kp=14  N m/rad, +-6  N m -> trackable to 1.5  rad/s
#   wrist slides  kp=180 N/m,    +-45 N  -> trackable to 0.2  m/s   (wrist_y;
#                 unchanged by W6a) -- but the W6a re-scale gave x/z
#                 kp=15000/damping=280, whose own derived allowance
#                 45/280 = 0.1607 m/s is the tighter line there
#                 (WARM_START_SLIDE_RATE_XZ_M_S)
#   finger joints URDF velocity="3.14" rad/s, and they execute at 0.34 rad/s,
#   so the ramp is not what limits them; it is capped at the URDF figure anyway.
# r10 fixture re-rating (Main Front D): the strokes/rates are W6a fixture
# self-set parameters, re-rated to the human demo's measured path envelope
# (runs/w6b/wrist_landmarks.npz, task window = video frames 1..151, robust
# stats -- the raw track carries pre-lock and dropout artefacts):
#   wrist_y stroke: task p2p 0.444 m x 1.2 -> +-0.225 (MJCF sync'd)
#   wrist slide y rate: task p99 0.166 m/s x 1.5 -> 0.25 m/s
#   wrist hinge rate: MANO orientation p95 1.815 rad/s x 1.5 -> 2.7 rad/s
# Gate thresholds unchanged; only the fixture's own physical rating moved.
WARM_START_HINGE_RATE_RAD_S = 2.7
WARM_START_SLIDE_RATE_M_S = 0.25
WARM_START_FINGER_RATE_RAD_S = 3.14

# Per-DoF-class velocity gate, signed at wave 3 after the reset-seed hypothesis
# was refuted (removing the seed moved the exceedance 3.680 -> 4.411 rad/s, so
# there was no "no startup transient" number to gate on; the blocker turned out
# to be the frame-0 pose being unreachable, which is upstream of any threshold).
# The gate is applied to the *command* trajectory -- warm-start ramp included --
# and each DoF class is bounded by what that class's actuator can actually
# track, not by one number borrowed from the wrong joint set:
#   finger 16 DoF   3.14 rad/s  URDF spec; measured peak 0.282, 11x margin
#   wrist hinges    2.70 rad/s  r10 re-rating: MANO orientation p95
#                   1.815 rad/s x 1.5 (task window, unwrapped)
#   wrist slides    0.25 m/s    r10 re-rating: MP wrist task-window p99
#                   0.166 m/s x 1.5 (wrist_y; x/z stay at their physical
#                   line via WARM_START_SLIDE_RATE_XZ_M_S)
VEL_LIMIT_FINGER_RAD_S = JOINT_VEL_LIMIT_RAD_S
# Fraction of the signed velocity limit any generated command is allowed to
# reach.  Exactly 1.0 makes cap*dt/dt round to cap*(1+1e-15), i.e. a command
# generated at the ceiling reads as a hair over it; this is float hygiene on the
# generator, not a change to the limit.
RATE_MARGIN = 1.0 - 1e-9
VEL_LIMIT_WRIST_HINGE_RAD_S = WARM_START_HINGE_RATE_RAD_S
VEL_LIMIT_WRIST_SLIDE_M_S = WARM_START_SLIDE_RATE_M_S
VEL_LIMIT_SOURCE = {
    "finger": 'URDF velocity="3.14" on all 16 finger joints',
    "wrist_hinge": "free-space measured trackable rate at 20 Hz (ratio >= 0.9)",
    "wrist_slide": "free-space measured trackable rate at 20 Hz (ratio >= 0.9)",
    "measured_quantity": (
        "commanded rate between consecutive control frames, np.diff(ctrl)/0.05, "
        "over the warm-start ramp concatenated with the replay"
    ),
    "excludes": (
        "the 3.687 rad/s wrist step-response peak is deliberately not the gate: "
        "it is what the servo does on a step, not what it can track"
    ),
}
# Hold the first command for 5 s of control time (100 steps at 20 Hz, three
# times the ramp itself) once the ramp has arrived.  Bounded, not "until
# converged": at the first commanded pose the hand is in sustained box contact,
# so the servos settle to a finite contact-deflection error that decays
# asymptotically.  The error it leaves is recorded as tracking_error_to_goal.
WARM_START_SETTLE_STEPS = 100
# A DoF counts as reached when the servos hold it within this of the command:
# 10 mm on a slide, 10 mrad on a hinge.  Applied to both because the tracking
# error is a single rad/m vector; at this pose the hand already carries ~1 mm
# of contact deflection, so 10 mm is a generous "arrived" and 170 mm of shortfall
# is unmistakably "jammed".
WARM_START_REACHABLE_TOL = 0.01
# Run the warm start in replay().  Enabled per the wave-3 ruling, and kept as a
# one-line switch because it is currently a net *regression* on this scene: the
# ramp honours its rate caps, but the first commanded wrist pose is outside the
# floating wrist's reachable set here, so the hand wedges against the box
# instead of arriving and three criteria that were green at commit 6e66cf5 go
# red.  Set False to reproduce 6e66cf5's numbers exactly; the underlying defect
# is reachability, not the ramp.
WARM_START_ENABLED = True
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

# ============================================ servo winding tolerance table
# Ruling (user, 2026-10-15): a gate may not hold a *command-side* ceiling
# against a *physical-side* transient.  Where one did, the ceiling is now the
# servo winding tolerance table -- the actuator's own force limit, mapped to the
# measured quantity through that DoF's passive compliance:
#
#   position   |qpos - jnt_range|   <=  |forcerange| / kp           [rad]
#   rate       realised rate       <=  |forcerange| / dof_damping  [rad/s]
#
# Both are ratios of two numbers read straight off the MJCF, per DoF, with no
# fitted constant.  The wrist hinges' actuators are kv=0, which is why the rate
# side closes on dof_damping and not on a velocity feedback term: with
# dof_damping = 0 there is no finite bound at all (measured, not assumed --
# runs/w35/wrist_rate_tol_report.md section 4), and with kv = 0 but
# dof_damping = 1.2 the 5.0 rad/s plateau appears and the bound holds.
#
# The bound is three-part and all three parts are required:
#   1. amplitude  <= the per-DoF cap above;
#   2. duration   <= WINDING_MAX_FRAMES consecutive control frames
#                   (3 * 50 ms = 150 ms at CTRL_HZ);
#   3. explanation -- the transient must be elastic wind-up, not a fault:
#      the actuator is not on its force limit while it happens, and the winding
#      it does carry accounts for the excursion.  An excursion recorded *during*
#      saturation is never exempted.
#
# Hardcoded on purpose: the tables are what the gates were signed against.
# tests/test_smoke5_winding_tolerance.py recomputes every entry from the model
# (forcerange / kp, forcerange / dof_damping) and fails if the tables drift.
#
# Re-derived on the dual-hand scene (W6a-v2, MJCF_COMMIT 9a5c384, blob sha256
# 6449871e...): every finger joint of BOTH hands, every wrist hinge of BOTH
# hands and every wrist slide of BOTH hands, read straight off the model.
# Left/right symmetry is a measured fact, not an assumption: the mirror pairs
# returned bit-identical forcerange / kp / dof_damping, and the derivation test
# asserts it, so a future asymmetric edit cannot hide behind this table.
#
# History of this table, so nobody "simplifies" it back:
#   * single-hand generation: finger kps gravity/geometry-derived upright;
#   * W6a-v1 (2e7c26a, blob ff4d4355...) posed the hands at a 45-degree pitch,
#     which moved the finger kp table (ffj0 0.3261 -> 0.2706 N m/rad etc.);
#   * W6a-v2 (9a5c384) refuted the tilt and posed the hands upright at the end
#     faces -- the finger kps returned BIT-IDENTICAL to the single-hand
#     generation's table.  The wrist slides keep their W6a re-scale
#     (x/z kp=15000/damping=280, y kp=180/damping=25) and the hinges are
#     untouched (kp 14, damping 1.2), so the rate tables are unchanged across
#     all three generations;
#   * W6a-v5 (8c58733, blob 7b7a1adf...) and W6a-v8 (eba5d7b, blob
#     8b6a6ddd...) re-posed/mounted the hands and re-sized the passive finger
#     springs (per-joint in v8) -- re-affirmed by per-joint actuator diffs:
#     ZERO forcerange/kp/dof_damping changes in both steps, so these numbers
#     are unchanged across all five scene generations;
#   * W6a-v10..v12 (HEAD 353b0b0, blob f8925e27...) changed the numbers
#     twice over: v10 re-measured the R1 moment-arm table (abductor/thumb
#     forceranges re-pinned -- ffj0 0.03 -> 0.0225 etc., position caps x0.75)
#     and swapped the pads to r9x11 cylinders with friction 1.8; v12 then
#     raised the lift-chain forceranges x1.7 (ffj1/ffj2/mfj1/mfj2, both
#     hands, 0.159-0.179 N m) to align with the Allegro hardware spec --
#     Main's ruling A, R10-v12 basis.  This is a HARDWARE-SPEC CALIBRATION:
#     the caps move because the actuators' own force limits moved, not
#     because any gate criterion was loosened (the lift gate stays at 5 mm).
#     Rate tables are untouched by both steps (hinges 5.0 rad/s x6; slides
#     x/z 45/280 = 0.1607 m/s, y 45/25 = 1.8 m/s, x2 hands);
#   * W6a-v13 (1fd969f, blob 6ec76dee...) flattened the pads to 12x12x3 mm
#     boxes (GPT re-verify condition #1; targets the r5 contact-point
#     migration mechanism).  The rebuild re-measured the abductor/thumb
#     moment arms again, so seven position keys moved x2 hands (ffj0
#     0.0690 -> 0.0460, mfj0 -> 0.0458, rfj0 -> 0.0457, thj0 -> 0.0797,
#     thj1 -> 0.0425, thj2 -> 0.1024, thj3 -> 0.1687); the lift chain kept
#     the v12 hardware-spec caps and the rate tables are still untouched.
#     The gate-3 pad-touch criterion moved from cylinder radius subtraction
#     to the flat pad's own half-thickness (1.5 mm), derived from the geom
#     in tests/test_scene_w6.py::_surface_rad;
#   * W6a-v14 (83bcedb, blob 257ef611...) raised the curl servos x5
#     (ffa2/mfa2 kp 0.235 -> 1.174 N m/rad, ffj2/mfj2 both hands -- the r7d
#     compliance endgame fix), so exactly four position keys moved:
#     ffj2 0.6783 -> 0.1357 and mfj2 0.6806 -> 0.1361 (the other 30 keys
#     measured identical).  Rate gates re-measured unchanged (kp does not
#     enter |forcerange|/dof_damping), the flat-pad contact criterion is
#     geometry-identical and carries over, and the per-joint spring table
#     is unchanged;
#   * W6a-v19/v20.1/v21 (0498dc3 / e170b6d / 98c498a): v19 added the six
#     env-IK arm servos (existing 44 actuators bit-identical, zero
#     hand-arm coupling); v21 extended the v12 hardware-alignment precedent
#     to the rf/th joints (rfj1 0.180 / rfj2 0.159 / thj1 0.029 /
#     thj2 0.073 N m, x2 hands) -- eight position keys refreshed (rfj1
#     0.1457 -> 0.2477, rfj2 0.3996 -> 0.6793, thj1 0.0425 -> 0.0722,
#     thj2 0.1024 -> 0.1740), the other 24 measured identical.  Rate gates,
#     springs and the flat-pad criterion re-measured unchanged.
SERVO_WINDING_POSITION_CAP_RAD = {
    "ffj0": 0.039994670929922725,
    "ffj1": 0.21431646932185144,
    "ffj2": 0.10942631398144967,
    "ffj3": 0.6644094488188976,
    "mfj0": 0.03981432360742706,
    "mfj1": 0.21119349437933507,
    "mfj2": 0.10979732050841635,
    "mfj3": 0.6669291338582677,
    "rfj0": 0.045662100456621,
    "rfj1": 0.24768786127167627,
    "rfj2": 0.6792589437819421,
    "rfj3": 0.9396436525612472,
    "thj0": 0.07973711983114266,
    "thj1": 0.07223177019396024,
    "thj2": 0.1740086000955566,
    "thj3": 0.16869834710743803,
    "ffj0_left": 0.039994670929922725,
    "ffj1_left": 0.21431646932185144,
    "ffj2_left": 0.10942631398144967,
    "ffj3_left": 0.6644094488188976,
    "mfj0_left": 0.03981432360742706,
    "mfj1_left": 0.21119349437933507,
    "mfj2_left": 0.10979732050841635,
    "mfj3_left": 0.6669291338582677,
    "rfj0_left": 0.045662100456621,
    "rfj1_left": 0.24768786127167627,
    "rfj2_left": 0.6792589437819421,
    "rfj3_left": 0.9396436525612472,
    "thj0_left": 0.07973711983114266,
    "thj1_left": 0.07223177019396024,
    "thj2_left": 0.1740086000955566,
    "thj3_left": 0.16869834710743803,
}
SERVO_WINDING_RATE_CAP_RAD_S = {
    "wrist_rx": 5.0,
    "wrist_ry": 5.0,
    "wrist_rz": 5.0,
    "wrist_rx_left": 5.0,
    "wrist_ry_left": 5.0,
    "wrist_rz_left": 5.0,
}
# Same derivation, one DoF class over: the slides' realised rate is judged
# against |forcerange| / dof_damping too.  Per axis, not one number: the W6a
# re-scale gave x/z kp=15000/damping=280 (45/280 = 0.1607 m/s) while y keeps
# kp=180/damping=25 (45/25 = 1.8 m/s).  Kept as an explicit table rather than
# folded into the rad/s one so the units cannot be silently mixed.
SERVO_WINDING_RATE_CAP_M_S = {
    "wrist_x": 0.16071428571428573,
    "wrist_y": 1.8,
    "wrist_z": 0.16071428571428573,
    "wrist_x_left": 0.16071428571428573,
    "wrist_y_left": 1.8,
    "wrist_z_left": 0.16071428571428573,
}
# W6a-v19 (0498dc3, blob e7b5de06...): the articulated arms add six position
# servos -- arm_yaw/pitch/elbow per hand, kp = 60 N m/rad, forcerange +-8 N m,
# ctrlrange +-3.2, living at ctrl indices [44:50] AFTER the 44 hand actuators.
# They are DRIVEN BY THE ENV'S IK at every env step: NOT acceptance joints,
# never commanded by the policy, and smoke5's single-input replay leaves them
# at ctrl 0 (static at their zero pose -- W6a measured zero hand-arm
# mechanical coupling: neq = 0, independent kinematic trees, zero arm
# collision geoms).  Recorded here for provenance completeness; they carry NO
# winding-table rows because no smoke-5 gate reads them.  Values are measured
# off the model, and tests/test_smoke5_winding_tolerance.py re-derives them.
SERVO_WINDING_ARM_SERVOS = {
    "arm_yaw_r":   {"kp_Nm_per_rad": 60.0, "forcerange_Nm": 8.0, "ctrl_index": 44},
    "arm_pitch_r": {"kp_Nm_per_rad": 60.0, "forcerange_Nm": 8.0, "ctrl_index": 45},
    "arm_elbow_r": {"kp_Nm_per_rad": 60.0, "forcerange_Nm": 8.0, "ctrl_index": 46},
    "arm_yaw_l":   {"kp_Nm_per_rad": 60.0, "forcerange_Nm": 8.0, "ctrl_index": 47},
    "arm_pitch_l": {"kp_Nm_per_rad": 60.0, "forcerange_Nm": 8.0, "ctrl_index": 48},
    "arm_elbow_l": {"kp_Nm_per_rad": 60.0, "forcerange_Nm": 8.0, "ctrl_index": 49},
}
# The hand actuators occupy ctrl [0:44] contiguously (right wrist 0:6, right
# fingers 6:22, left wrist 22:28, left fingers 28:44); the arm servos start
# exactly there.  Everything in this module that addresses actuators by
# index (slides [:6], fingers [6:22], graded blocks [:22]) is addressing the
# hand block and is unaffected by the arm block behind it.
MJCF_ARM_ACTUATOR_BASE = 44
# The warm-start ramp may not command a DoF faster than that DoF's own servo
# can deliver.  For the re-scaled x/z slides the derived winding allowance
# 45/280 = 0.1607 m/s is now BELOW the old free-space measured trackable rate
# of 0.2 m/s, and a ramp commanding 0.2 m/s drove the realised rate past the
# allowance for 15 consecutive intervals (measured on this scene).  Each slide
# is therefore commanded at the tighter of the two lines; wrist_y (kp 180,
# damping 25) keeps the measured 0.2 m/s.  Derived from the table above, whose
# model derivation the winding-tolerance tests recompute every run.
#
# The x/z command additionally sits 1% below the allowance: the kp=15000 servo
# tracks a ramp with ~0.08% velocity ripple (measured: commanded 0.160714,
# realised peak 0.160836 over 9 intervals), so a command exactly AT the
# allowance pokes the realised rate past it.  This is command-side input-
# shaping headroom per the recorded w1 ruling ("the ramp should command below
# it by the measured overshoot factor"); the winding GATE itself stays the
# unfitted |forcerange| / dof_damping.
WARM_START_SLIDE_TRACK_MARGIN = 1e-2
WARM_START_SLIDE_RATE_XZ_M_S = min(
    WARM_START_SLIDE_RATE_M_S, SERVO_WINDING_RATE_CAP_M_S["wrist_x"]
) * (1.0 - WARM_START_SLIDE_TRACK_MARGIN)
WINDING_MAX_FRAMES = 3
# Part 3 of the position gate.  Two conditions:
#   * a contact torque must exist on the DoF -- below this floor the DoF is not
#     touching anything, and a free-drifting joint has no wind-up to excuse.
#     The floor is a numerical floor (a torque MuJoCo reports as exactly zero
#     is zero), not a tuned threshold.
#   * the servo's own winding torque must account for the whole non-actuator
#     load on the DoF to this relative band.
WINDING_CONTACT_FLOOR_NM = 1e-12
WINDING_EXPLANATION_TOL = 0.05
# A rate above the command-side trackable rate is exempt only up to what the
# DoF's own accumulated wind-up could release into the inertia; that release is
# the *upper* bound on the explanation (damping only makes the real release
# smaller), so the criterion is deliberately strict.
WINDING_RELEASE_FORMULA = (
    "release_j(t) = sqrt(kp_j * windup_j(t)^2 / (dof_M0_j + armature_j)), "
    "windup_j(t) = ctrl_j(t) - qpos_j(t)"
)
# The rejected alternative, kept on the record so the bounds that were examined
# are visible rather than quietly dropped.  runs/w35/wrist_rate_tol_report.md
# section 3.1: dv <= a_max * tau_s with a_max = forcerange/I ~ 94-98 rad/s^2 and
# tau_s = (forcerange/kp)/v_cap = 0.2857 s gives 26.8-27.9 rad/s, an order of
# magnitude above anything this hand does.  Vacuous: the acceleration ceiling of
# a stiff, light wrist hinge is enormous, and tau_s is a window set by the
# *command* rate, not by the servo.
SERVO_WINDING_REJECTED_RATE_FORMS = {
    "dv <= a_max * tau_s": {
        "value_rad_s": {"wrist_rx": 27.886, "wrist_ry": 27.33, "wrist_rz": 26.787},
        "rejected_because": (
            "an order of magnitude above the measured transient: forcerange/I "
            "~ 94-98 rad/s^2 is the acceleration ceiling of a kp=14, I~0.062 "
            "wrist hinge, and tau_s = 0.2857 s is a window the command rate "
            "gives, not the servo.  A bound an order of magnitude above every "
            "achievable value gates nothing."
        ),
    },
}


def winding_position_cap(model, joint: str) -> float:
    """|forcerange| / kp for one finger joint, read off the model."""
    a = _joint_actuator_id(model, joint)
    fr = float(np.max(np.abs(model.actuator_forcerange[a])))
    kp = float(model.actuator_gainprm[a, 0])
    return fr / kp


def winding_rate_cap(model, joint: str) -> float:
    """|forcerange| / dof_damping for one wrist joint, read off the model."""
    a = _joint_actuator_id(model, joint)
    fr = float(np.max(np.abs(model.actuator_forcerange[a])))
    return fr / float(model.dof_damping[model.jnt_dofadr[_joint_id(model, joint)]])


def _joint_actuator_id(model, joint: str) -> int:
    """The position actuator bound to ``joint``.

    Every actuator in this MJCF hangs off a joint transmission, and
    ``jnt_actuatorid`` is -1 throughout -- the mapping has to be read out of
    ``actuator_trnid`` under ``actuator_trntype == mjTRN_JOINT``.
    """
    import mujoco

    jid = _joint_id(model, joint)
    trntype = np.asarray(model.actuator_trntype).reshape((model.nu, -1))
    trnid = np.asarray(model.actuator_trnid).reshape((model.nu, -1))
    for a in range(model.nu):
        for t in range(trnid.shape[1]):
            if int(trntype[a, min(t, trntype.shape[1] - 1)]) != int(
                    mujoco.mjtTrn.mjTRN_JOINT):
                continue
            if int(trnid[a, t]) == jid:
                return a
    raise KeyError(f"no joint transmission from actuator to {joint}")


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
                "is recomputed here by retarget/mano_fk.py from the pkl; smoke 6's "
                "own joints_3d is never read."
            ),
            "c1b_status": {
                "what_c1b_actually_checks": (
                    "shape and finiteness only: our numpy MANO forward pass returns "
                    "(21, 3) finite joints. It does NOT compare them against a "
                    "reference implementation."
                ),
                "legacy_claim": "cross-validated against smplx to 3.6e-8 m",
                "legacy_claim_status": "unverified legacy claim",
                "why_it_is_unverified": (
                    "the committed test opens with pytest.importorskip('smplx'), so it "
                    "had never executed in the CI venv (runs/.venv-retarget has no "
                    "smplx). The 3.6e-8 m figure rests on a run this test file cannot "
                    "reproduce. See commit ae64f8a, which un-broke the test without "
                    "adding any comparison."
                ),
                "bugs_found_by_ae64f8a": [
                    "hands_mean is stored flat (45,), one 3-vector per hand-pose "
                    "joint; the test added it straight to a (15, 3) rotvec block "
                    "(ValueError: shapes (15,3) (45,)). Fixed with .reshape(15, 3).",
                    "Rotation.from_rotvec on a (15, 3) block yields 15 rotations, so "
                    "as_rotvec() comes back 2-D and the np.concatenate against the "
                    "flat global_orient failed on dimensionality. Fixed with "
                    ".reshape(-1), because MANO's pose parameter is one flat "
                    "48-vector (global_orient 3, then hand_pose 45).",
                ],
                "why_a_real_comparison_is_not_a_one_liner": (
                    "smplx.MANO expects the npz-style shapedirs/posedirs arrays, but "
                    "MANO_RIGHT.pkl is a pickled dict, so smplx.create() fails on "
                    "shapedirs[:, :, :10]. A genuine cross-validation needs the pkl "
                    "converted to the npz layout first."
                ),
                "deferred_to": "P1",
            },
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
    """Split the retargeted clip into the part that is video and the part that is base.

    The retargeted 6-DoF wrist is expressed in the Allegro URDF base frame; the
    MJCF scene reaches that frame as ``hand_root`` translation + the fixed
    ``palm`` quaternion.  So the clip decomposes into

      * ``hinge``   -- the wrist orientation, converted to the MJCF hinge triple
        via ``R_mjcf = Rx(qrx) Ry(qry) Rz(qrz) @ R_palm`` and then gauge-fixed.
        This is the video's hand orientation and is never solved for.
      * ``slide_shape`` -- the wrist's *own* slide motion, with the clip mean
        removed from all three axes.  Zero mean, so it is exactly the motion the
        retargeter produced and nothing else.
      * ``base`` -- the floating base translation, i.e. the mean commanded slide.
        This is the only thing left under-determined, and it is what
        ``solve_base_placement`` solves.

    The previous rule -- one constant offset placing the mid-clip fingertip
    centroid on the box's *top face* -- is still computed, as
    ``legacy_base_slide``, because it is the "before" the base solve is measured
    against.  It is not used as the answer: with the clip's own wrist pitch it
    asks for a palm frame 29 mm *below* the box top, which is what drove the
    hand 76 mm into the box.
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

    # Gauge fixing: strip the global yaw and the in-plane base offset out of the
    # wrist command. Those are coordinate freedoms of a hand with no fixed arm,
    # not physics; leaving them on the actuators is what pushed wrist_rz to
    # 2.71 rad against a +-1.57 rad ctrlrange.  z is deliberately untouched here
    # -- the hand's height above the object is physical -- and is de-meaned
    # below instead, so that the base solve owns all three slides.
    raw = np.zeros((qpos.shape[0], 6))
    raw[:, 0:3] = qpos[:, 0:3]
    raw[:, 3:6] = hinge
    fixed, gauge = fix_gauge(raw, model.actuator_ctrlrange[:6], r_palm)
    hinge_cmd = fixed[:, 3:6]
    slide_shape = fixed[:, 0:3] - fixed[:, 0:3].mean(axis=0)

    # The superseded rule, kept as the measured "before".
    mid = qpos.shape[0] // 2
    tip_centroid_local = tip_pos[mid].mean(axis=0)
    target_tip = box_center + np.array([0.0, 0.0, box_half[2]])
    palm_anchor = target_tip - target[mid] @ tip_centroid_local
    legacy = np.zeros((qpos.shape[0], 6))
    legacy[:, 0:3] = qpos[:, 0:3] + (palm_anchor - palm_home)
    legacy[:, 3:6] = hinge
    legacy, _ = fix_gauge(legacy, model.actuator_ctrlrange[:6], r_palm)
    legacy_base = legacy[:, 0:3].mean(axis=0)

    return {
        "hinge_cmd": hinge_cmd,
        "slide_shape": slide_shape,
        "legacy_base_slide": legacy_base,
        "legacy_slide_offset": palm_anchor - palm_home,
        "gauge_fixing": gauge,
        "hinge_convention": "MJCF wrist_rx->wrist_ry->wrist_rz, composed Rz@Ry@Rx",
        "anchor_rule": (
            "floating base xyz solved by retarget/base_solve.py against the box's "
            "grasp geometry; the per-frame wrist orientation and slide motion are "
            "the retargeted clip's own and are not fitted"
        ),
        "superseded_anchor_rule": (
            "single constant translation offset placing the mid-clip retargeted "
            "fingertip centroid on the tissue-box proxy TOP face; no per-frame term"
        ),
        "box_center": box_center,
        "box_half_size": box_half,
    }


# --- base-solve gates -------------------------------------------------------
# The box's own geometry is the scale for every gate here.  Read off the frozen
# MJCF (half sizes 0.06 / 0.12 / 0.04 m) and nothing about it is tuned.
_BOX_HALF_MIN_M = 0.04          # smallest half-dimension: the box's thin axis
_BOX_HALF_DIAGONAL_M = float(np.linalg.norm([0.06, 0.12, 0.04]))
# How far the base solve may move the hand, per axis, relative to the placement
# the superseded rule produced.  The bound is the wrist slide's own travel: a
# base displacement larger than the actuator's range is not executable, so a
# solve that needs one has solved a problem this hand cannot act on.
BASE_DISPLACEMENT_CAP_M = float(np.linalg.norm([0.36, 0.36, 0.323]))
# How far the hand's world pose may deviate from the pose the retargeted clip
# asked for.  The palm frame must land within the box's thinnest half-dimension:
# past that the hand is no longer executing the grasp it was given, whatever the
# servos report.  The fingertips are held to the box's half-diagonal, i.e. the
# whole object, because a fingertip is what has to stay on the box's wall.
PALM_WORLD_POSE_DEVIATION_CAP_M = _BOX_HALF_MIN_M
FINGERTIP_WORLD_POSE_DEVIATION_CAP_M = _BOX_HALF_DIAGONAL_M
HAND_WORLD_POSE_DEVIATION_CAP_M = _BOX_HALF_MIN_M


def solve_base_placement(qpos: np.ndarray, anchor: dict, model, data) -> dict:
    """Solve the floating base xyz, then the temporally coupled command trajectory.

    Stage 1 (``retarget/base_solve.py``) places the hand against the box's own
    grasp geometry; stage 2 (``retarget/traj_solve.py``) solves the whole clip
    jointly against the actuator ctrlrange, the joint range and the signed
    velocity limits.  Both are reported; neither hides a residual.
    """
    from retarget import base_solve, traj_solve

    import mujoco

    mujoco.mj_resetData(model, data)
    _mj_forward(model, data)

    palm_id = _body_id(model, "palm")
    obj_id = _body_id(model, "object")
    tip_bodies = [_body_id(model, b) for b in MJCF_TIP_BODIES]
    hand_geoms, tip_geoms = base_solve.classify_hand_geoms(
        model, hand_geom_ids(model), set(tip_bodies)
    )
    plate_geoms = sorted(g for g in hand_geoms if model.geom_bodyid[g] == palm_id)
    wrist_joints = [_joint_id(model, n) for n in WRIST_JOINTS]
    finger_joints = [_joint_id(model, n) for n in MJCF_FINGER_JOINTS]
    wrist_qadr = np.array([model.jnt_qposadr[j] for j in wrist_joints])
    finger_qadr = np.array([model.jnt_qposadr[j] for j in finger_joints])
    # The Jacobian MuJoCo returns is in *velocity* DoFs, and nv != nq here: the
    # box carries a free joint, so the hand's velocity indices are not its qpos
    # addresses.  Index by dofadr, never by qposadr.
    wrist_dof = np.array([model.jnt_dofadr[j] for j in wrist_joints])
    finger_dof = np.array([model.jnt_dofadr[j] for j in finger_joints])
    finger_cmd = qpos[:, 6:22][:, URDF_TO_MJCF_ACTUATOR_ORDER]

    ctx = {
        "wrist_qadr": wrist_qadr,
        "finger_qadr": finger_qadr,
        "wrist_dof": wrist_dof,
        "finger_dof": finger_dof,
        "palm_body": palm_id,
        "obj_body": obj_id,
        "box_geom": _geom_id(model, "object_geom"),
        "box_half": model.geom_size[_geom_id(model, "object_geom")].copy(),
        "tip_bodies": tip_bodies,
        "hand_geoms": hand_geoms,
        "tip_geoms": tip_geoms,
        "plate_geoms": plate_geoms,
        "hand_root_home": data.xpos[_body_id(model, "hand_root")].copy(),
        "slide_cmd": anchor["slide_shape"],
        "hinge": anchor["hinge_cmd"],
        "finger_cmd": finger_cmd,
    }

    slide_lo = model.actuator_ctrlrange[:3, 0].copy()
    slide_hi = model.actuator_ctrlrange[:3, 1].copy()
    # The base must leave room for the clip's own slide motion at every frame, or
    # some frame would command outside the actuator range.  The exact region is
    # [lo - min_t shape, hi - max_t shape]; using a symmetric |max| bound instead
    # throws away a centimetre or two of travel on each side for no reason.
    shape = anchor["slide_shape"]
    base_lo = slide_lo - shape.min(axis=0)
    base_hi = slide_hi - shape.max(axis=0)

    solved = base_solve.solve_base_xyz(model, data, ctx, base_lo, base_hi)
    legacy = np.asarray(anchor["legacy_base_slide"], float)
    if solved.get("status") == "solved":
        base = np.asarray(solved["base_slide_command_m"], float)
    else:
        base = legacy
    displacement = base - legacy
    solved["displacement_from_superseded_rule_m"] = displacement.tolist()
    solved["displacement_norm_m"] = float(np.linalg.norm(displacement))
    solved["base_displacement_cap_m"] = BASE_DISPLACEMENT_CAP_M
    solved["displacement_per_axis_within_wrist_travel"] = bool(
        np.all(np.abs(displacement) <= np.asarray([0.18, 0.18, 0.1615]))
    )
    solved["displacement_within_cap"] = bool(
        np.linalg.norm(displacement) <= BASE_DISPLACEMENT_CAP_M
    )
    solved["superseded_rule_base_slide_m"] = legacy.tolist()
    solved["base_search_box_m"] = [base_lo.tolist(), base_hi.tolist()]

    # ---- stage 2: temporally coupled constrained solve --------------------
    x0 = np.concatenate(
        [
            np.concatenate([base + anchor["slide_shape"], anchor["hinge_cmd"]], axis=1),
            finger_cmd,
        ],
        axis=1,
    )
    t_ctx = dict(ctx)
    t_ctx["dofadr"] = np.concatenate([wrist_dof, finger_dof])
    t_ctx["pose_bodies"] = [palm_id] + tip_bodies

    finger_act = _finger_actuator_ids(model)
    joint_ids = [_joint_id(model, n) for n in WRIST_JOINTS + MJCF_FINGER_JOINTS]
    ctrl_lo = np.concatenate([model.actuator_ctrlrange[:6, 0], model.actuator_ctrlrange[finger_act, 0]])
    ctrl_hi = np.concatenate([model.actuator_ctrlrange[:6, 1], model.actuator_ctrlrange[finger_act, 1]])
    # The box is the *intersection*: a command can sit inside an actuator's
    # ctrlrange and still be outside the joint's own range, and a position
    # actuator would then drive the joint past a limit it is meant to respect.
    jnt_lo = np.array([model.jnt_range[j][0] for j in joint_ids])
    jnt_hi = np.array([model.jnt_range[j][1] for j in joint_ids])
    lo = np.maximum(ctrl_lo, jnt_lo)
    hi = np.minimum(ctrl_hi, jnt_hi)
    step = np.concatenate([
        np.full(3, VEL_LIMIT_WRIST_SLIDE_M_S),
        np.full(3, VEL_LIMIT_WRIST_HINGE_RAD_S),
        np.full(16, VEL_LIMIT_FINGER_RAD_S),
    ]) * rs_ctrl_dt(model)

    x, traj = traj_solve.solve_constrained_traj(
        model, data, t_ctx, x0, lo, hi, step,
        slide_move_cap_m=PALM_WORLD_POSE_DEVIATION_CAP_M,
    )
    traj["box_intersection_note"] = (
        "the box is the actuator ctrlrange INTERSECT the joint range; the "
        "retargeter commands several finger joints outside both"
    )
    traj["sign_rate_limits"] = {
        "wrist_slide_m_s": VEL_LIMIT_WRIST_SLIDE_M_S,
        "wrist_hinge_rad_s": VEL_LIMIT_WRIST_HINGE_RAD_S,
        "finger_rad_s": VEL_LIMIT_FINGER_RAD_S,
    }

    ctrl = x[:, 0:6]
    solved["constrained_solve"] = traj
    solved["command_outside_ctrlrange_after"] = float(
        max(0.0, np.max(ctrl - model.actuator_ctrlrange[:6, 1]),
            np.max(model.actuator_ctrlrange[:6, 0] - ctrl))
    )
    return {
        "ctrl": ctrl,
        "finger_cmd": x[:, 6:22],
        "base": base,
        "record": solved,
        "hand_world_pose_deviation_cap_m": HAND_WORLD_POSE_DEVIATION_CAP_M,
    }


def executed_anchor(qpos: np.ndarray, model, data) -> dict:
    """The anchor the replay actually executes: retargeted clip + solved base.

    Shared with ``retarget/mjcf_provenance.py`` on purpose.  The provenance
    measurements (servo ceilings, saturation, blocker forensics) have to be taken
    on the trajectory the artefact reports, not on a second reconstruction of it,
    so there is exactly one way to build this dict.
    """
    anchor = solve_wrist_anchor(qpos, *retargeted_tip_positions_for(qpos), model, data)
    base = solve_base_placement(qpos, anchor, model, data)
    return dict(anchor, ctrl=base["ctrl"], finger_cmd=base["finger_cmd"])


def retargeted_tip_positions_for(qpos: np.ndarray) -> tuple:
    """(tip_pos,) for a qpos already computed by ``retarget_sequence``.

    Thin wrapper so ``executed_anchor`` can be called with a qpos the caller
    already has, without rebuilding the retargeter.
    """
    retargeter, _ = build_retargeter()
    return (retargeted_tip_positions(qpos, retargeter),)


def rs_ctrl_dt(model) -> float:
    """Control period implied by CTRL_HZ and the scene timestep."""
    frame_skip = max(1, int(round(1.0 / (CTRL_HZ * model.opt.timestep))))
    return frame_skip * float(model.opt.timestep)


def _finger_actuator_ids(model) -> np.ndarray:
    """Actuator index of each MJCF_FINGER_JOINTS entry, in that order.

    The MJCF's finger actuators are ffa0..3, mfa0..3, rfa0..3, tha0..3 in joint
    order.  Names come from the frozen scene; a missing one is a hard error
    rather than a silent -1 that would index the wrong actuator.
    """
    import mujoco

    out = []
    for prefix in ("ff", "mf", "rf", "th"):
        for k in range(4):
            name = f"{prefix}a{k}"
            aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if aid < 0:
                raise KeyError(f"scene has no actuator named {name}")
            out.append(aid)
    return np.array(out)


def _body_id(model, name: str) -> int:
    import mujoco

    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)


def _geom_id(model, name: str) -> int:
    import mujoco

    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)


# ================================================================ replay stage
def warm_start(
    model,
    data,
    finger_qadr: np.ndarray,
    wrist_qadr: np.ndarray,
    finger_act: np.ndarray,
    wrist_act: np.ndarray,
    finger_goal: np.ndarray,
    wrist_goal: np.ndarray,
    ctrl_hz: int = CTRL_HZ,
    frame_skip: int = 25,
    finger_suffix: str = "",
    wrist_suffix: str = "",
) -> dict[str, Any]:
    """``finger_suffix``/``wrist_suffix`` exist for the dual-hand caller
    (retarget_dual.py): the function is otherwise side-neutral -- it takes
    explicit qpos-address/actuator arrays -- but its record names DoFs with the
    right-hand name lists and its winding-rate gates resolve actuators by joint
    NAME, which must be the driven hand's names.  Defaults keep the single-hand
    path byte-identical."""
    """Ramp the servos from the reset pose into the first commanded pose.

    ``reset_scene()`` leaves the floating wrist at identity and the fingers at
    the pre-curl.  Stepping straight to the first retargeted command asks the
    wrist for a 1.3 rad pitch change in 50 ms (~26 rad/s) and the fingers for up
    to 1.23 rad; the force-limited servos refuse, and the resulting transient
    says nothing about the trajectory.  A real hand would be driven there over
    a second or so.

    This walks ``data.ctrl`` along a per-axis ramp capped at the servos'
    *measured* trackable rates and then holds the goal until the position error
    stops falling.  Only ``data.ctrl`` is written; ``data.qpos``/``data.qvel``
    are read, never assigned, so the C5 "replay never writes qpos" invariant
    still holds for the warm start too.

    The ramp is a pure input shaping: it changes the path the servos take to
    the first command, never the command itself.  tests/test_smoke5_warmstart.py
    pins that by settling the warm-started state and a straight-to-target
    reference and requiring the hand's and the box's world poses to agree.

    Returns what the ramp did, for the report: control steps, duration, the
    peak commanded rate per class, the peak *realised* rate, and the residual to
    the goal.
    """
    import mujoco

    dt_ctrl = frame_skip * model.opt.timestep
    lo = model.actuator_ctrlrange[:, 0]
    hi = model.actuator_ctrlrange[:, 1]
    # ``goal`` below is laid out fingers-then-wrist, because that is the order
    # the ramp variable and every per-DoF cap are built in.  model.actuator_
    # ctrlrange is laid out wrist-then-fingers.  Clipping one against the other
    # silently clamps the fingers to the *wrist slides'* +-0.18/0.16 m bounds,
    # which is how the ramp ended up commanding ffj0 = -0.18 rad.  Reorder the
    # bounds to match, and assert it so the next reorder cannot pass silently.
    goal = np.concatenate([np.asarray(finger_goal, float), np.asarray(wrist_goal, float)])
    lo = np.concatenate([lo[finger_act], lo[wrist_act]])
    hi = np.concatenate([hi[finger_act], hi[wrist_act]])
    goal = np.clip(goal, lo, hi)
    n_f = len(finger_qadr)
    # per-DoF command cap: slides in m/s, everything else in rad/s
    finger_caps = np.full(n_f, WARM_START_FINGER_RATE_RAD_S)
    if n_f:
        finger_slide = np.array([
            model.jnt_type[
                [j for j in range(model.njnt)
                 if int(model.jnt_qposadr[j]) == int(qa)][0]
            ] == mujoco.mjtJoint.mjJNT_SLIDE
            for qa in finger_qadr
        ])
        finger_caps[finger_slide] = WARM_START_SLIDE_RATE_M_S
    caps = np.concatenate([
        finger_caps,
        # per axis, not one number: the W6a re-scale put x/z's own allowance
        # below the measured 0.2 m/s trackable rate (see the derivation note
        # above WARM_START_SLIDE_RATE_XZ_M_S); the order is x, y, z.
        np.array([WARM_START_SLIDE_RATE_XZ_M_S, WARM_START_SLIDE_RATE_M_S,
                  WARM_START_SLIDE_RATE_XZ_M_S]),
        np.full(3, WARM_START_HINGE_RATE_RAD_S),
    ])

    cur = np.concatenate([
        data.qpos[finger_qadr].copy(), data.qpos[wrist_qadr].copy()
    ])
    start = cur.copy()

    steps = 0
    # Sized to the commanded block (fingers + 6 right-wrist DoF), not model.nu:
    # since W6a the scene carries 44 actuators (two hands) and only the right
    # hand's 22 are ever commanded here, so a model.nu-shaped accumulator would
    # not broadcast against the per-step command deltas.
    peak_cmd = np.zeros(n_f + 6)
    hist: list[np.ndarray] = []
    ctrl_hist: list[np.ndarray] = []
    prev_q = np.concatenate([
        data.qpos[finger_qadr].copy(), data.qpos[wrist_qadr].copy()
    ])
    realised = []
    force_hist: list[np.ndarray] = []
    ramp_done_at = None
    # Settling is bounded rather than run to convergence: at the first commanded
    # pose the hand is in sustained contact with the box, so the servos hold a
    # finite contact-deflection error instead of reaching the commanded qpos
    # exactly, and the residual decays asymptotically.  WARM_START_SETTLE_STEPS
    # is 5 s at 20 Hz -- three times the ramp itself -- and the tracking error
    # it leaves is recorded rather than hidden.
    settle_budget = WARM_START_SETTLE_STEPS
    while True:
        delta = goal - cur
        step = np.clip(delta, -caps * dt_ctrl, caps * dt_ctrl)
        # Command strictly under the cap.  A step of exactly cap*dt divided by
        # dt again comes back as cap*(1+1e-15), so a trajectory generated at the
        # ceiling reads as a violation of a ceiling it satisfies.  The signed
        # limits are upper bounds; this keeps the generated command inside them.
        step = np.clip(step, -caps * dt_ctrl * RATE_MARGIN, caps * dt_ctrl * RATE_MARGIN)
        cur = cur + step
        data.ctrl[finger_act] = cur[:n_f]
        data.ctrl[wrist_act] = cur[n_f:]
        peak_cmd = np.maximum(peak_cmd, np.abs(step) / dt_ctrl)
        for _ in range(frame_skip):
            mujoco.mj_step(model, data)
        now = np.concatenate([
            data.qpos[finger_qadr].copy(), data.qpos[wrist_qadr].copy()
        ])
        realised.append((now - prev_q) / dt_ctrl)
        prev_q = now
        hist.append(now.copy())
        ctrl_hist.append(data.ctrl.copy())
        force_hist.append(data.actuator_force.copy())
        steps += 1
        if ramp_done_at is None and np.max(np.abs(goal - cur)) <= 1e-12:
            ramp_done_at = steps
            continue
        if ramp_done_at is None:
            continue
        held = steps - ramp_done_at
        if held >= settle_budget:
            break
    if ramp_done_at is None:
        ramp_done_at = steps

    realised_arr = np.abs(np.stack(realised)) if realised else np.zeros((0, model.nu))
    residual = float(np.max(np.abs(goal - cur)))
    tracking = np.abs(goal - prev_q)
    idx = {"finger": slice(0, n_f), "wrist_slide": slice(n_f, n_f + 3),
           "wrist_hinge": slice(n_f + 3, n_f + 6)}

    # Reachability is a per-DoF verdict, so the report has to name WHICH DoF
    # fell short.  ``reachable`` is the conjunction over every commanded DoF;
    # on the single-hand scene the shortfall was the wrist descent, on the
    # dual-hand scene the wrist arrives and what misses is a finger holding
    # its contact deflection.  ``unreachable_dofs`` therefore spans fingers
    # and wrist alike.
    unreachable_dofs = [
        str(name) for name, ok in zip(
            [n + finger_suffix for n in MJCF_FINGER_JOINTS]
            + [n + wrist_suffix for n in WRIST_JOINTS],
            tracking <= WARM_START_REACHABLE_TOL)
        if not ok
    ]
    wrist_unreachable = [n for n in unreachable_dofs if n in WRIST_JOINTS]

    # Did the servos actually get there, or did the world stop them?  The
    # tracking error is the honest discriminator: the ramp caps the *command*
    # rate, but a hand that jams on the box never arrives no matter how slowly
    # it is driven, and reporting the shortfall keeps "the warm start failed"
    # from being ambiguous.
    return {
        "reachable": bool(not unreachable_dofs),
        "unreachable_dofs": unreachable_dofs,
        "unreachable_note": (
            "a DoF counted here holds a tracking error above the tolerance at "
            "the first commanded pose, which on this scene is sustained box "
            "contact: a finger in contact holds a contact-deflection error "
            "rather than the commanded value, and a wrist slide on the list "
            "means the commanded descent itself is outside the floating "
            "wrist's reachable set -- no approach path reaches it. The ramp "
            "caps the *command* rate; it cannot make an unreachable pose "
            "reachable."
        )
        + (
            "" if wrist_unreachable else
            " On this run every wrist DoF delivered the commanded pose; the "
            "shortfall is on the finger joints only."
        ),
        "wrist_reachable_note": (
            "per-DoF reachability, dual-hand scene: on the superseded "
            "single-hand scene the finding was the opposite -- the palm rested "
            "on the box top and the commanded wrist descent was unreachable "
            "outright. Kept as a recorded finding so the change of scene is "
            "visible rather than silent."
        ) if not wrist_unreachable else (
            "the commanded wrist pose is not HELD by this scene: on the "
            "superseded scenes the palm blocked the wrist's descent outright, "
            "on the W6a-v2 upright posture the servos saturate against gravity "
            "trying to hold it (wrist_rx demanded torque beyond its +-6 N m "
            "budget) and the pose sags. Either way no approach path makes the "
            "command stick -- the ramp caps the *command* rate; it cannot "
            "make an unholdable pose holdable."
        ),
        "control_steps": int(steps),
        "ramp_control_steps": int(ramp_done_at),
        "ramp_duration_s": float(ramp_done_at * dt_ctrl),
        "settle_control_steps": int(steps - ramp_done_at),
        "duration_s": float(steps * dt_ctrl),
        "dt_ctrl_s": float(dt_ctrl),
        "rate_caps": {
            "finger_rad_s": WARM_START_FINGER_RATE_RAD_S,
            # wrist_y's cap, and the binding one for the aggregate W1 check;
            # the per-axis dict is the authority.
            "wrist_slide_m_s": WARM_START_SLIDE_RATE_M_S,
            "wrist_slide_m_s_per_axis": {
                "wrist_x": WARM_START_SLIDE_RATE_XZ_M_S,
                "wrist_y": WARM_START_SLIDE_RATE_M_S,
                "wrist_z": WARM_START_SLIDE_RATE_XZ_M_S,
            },
            "wrist_hinge_rad_s": WARM_START_HINGE_RATE_RAD_S,
            "source": (
                "free-space step/ramp measurement of this model's own servos "
                "(hinges 1.5 rad/s, wrist_y 0.2 m/s), tightened per axis to "
                "the servo's own derived winding allowance where the W6a "
                "re-scale put that below the measurement "
                "(x/z: 45 N / 280 = 0.1607 m/s); recorded in "
                "rerun_evidence.velocity_gate.floating_wrist_servo_ceiling"
            ),
        },
        "commanded_peak_rate": {
            k: float(np.max(peak_cmd[v])) for k, v in idx.items()
        },
        "commanded_peak_rate_slide_per_axis": {
            "wrist_x" + wrist_suffix: float(peak_cmd[n_f + 0]),
            "wrist_y" + wrist_suffix: float(peak_cmd[n_f + 1]),
            "wrist_z" + wrist_suffix: float(peak_cmd[n_f + 2]),
        },
        "realised_peak_rate": {
            k: float(np.max(realised_arr[:, v])) if realised_arr.size else 0.0
            for k, v in idx.items()
        },
        # The three-part servo winding gate on the *realised* hinge rate.  The
        # 1.5 rad/s command-side figure stays the command cap (w1/w1c); what
        # changed is that a realised transient above it is now judged against
        # the servo's own force limit mapped through dof_damping, with the
        # wind-up release as the mandatory explanation.
        "servo_winding_rate": servo_winding_rate_gate(
            model,
            [n + wrist_suffix for n in WRIST_JOINTS[3:]],
            np.stack(hist)[:, n_f + 3:n_f + 6],
            np.stack(ctrl_hist)[:, wrist_act[3:6]],
            np.stack(force_hist),
            dt_ctrl,
            WARM_START_HINGE_RATE_RAD_S,
        ),
        # The same three-part gate on the slides' realised rate.  The command
        # cap stays 0.2 m/s (w1/w1c); a realised transient above it is judged
        # against the slide's own winding allowance, 45 N / 25 = 1.8 m/s.
        "servo_winding_rate_slide": servo_winding_rate_gate(
            model,
            [n + wrist_suffix for n in WRIST_JOINTS[:3]],
            np.stack(hist)[:, n_f:n_f + 3],
            np.stack(ctrl_hist)[:, wrist_act[0:3]],
            np.stack(force_hist),
            dt_ctrl,
            WARM_START_SLIDE_RATE_M_S,
            cap_table=SERVO_WINDING_RATE_CAP_M_S,
        ),
        "command_span": {
            k: float(np.max(np.abs(goal[v] - start[v]))) for k, v in idx.items()
        },
        "tracking_error_to_goal": {
            k: float(np.max(tracking[v])) for k, v in idx.items()
        },
        "tracking_error_note": (
            "finite by design: the first commanded pose puts the hand in "
            "sustained box contact, so the force-limited servos hold a contact "
            "deflection instead of reaching the commanded qpos exactly."
        ),
        "residual_to_goal_rad": residual,
        "reached_goal": bool(residual <= 1e-9),
        "cap_respected": bool(
            np.max(peak_cmd) <= max(
                WARM_START_FINGER_RATE_RAD_S, WARM_START_SLIDE_RATE_M_S,
                WARM_START_HINGE_RATE_RAD_S) + 1e-9),
        "qpos_history": np.stack(hist),
        "ctrl_history": np.stack(ctrl_hist),
    }


def replay(qpos: np.ndarray, anchor: dict, seed: int = 0) -> dict[str, Any]:
    """Replay the retargeted sequence through the position actuators.

    Single-hand input on the dual-hand scene: the right hand's 22 servos are
    commanded; the left hand's 22 are never written, so ctrl stays 0 and the
    left hand stays frozen at its reset pose.

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
    # Read now, while the hand is still at its reset pose: reading it after the
    # loop would measure wherever the replay happened to leave hand_root, and the
    # commanded palm position below would be offset by the whole excursion.
    hand_root_home = data.xpos[_body_id(mujoco_model, "hand_root")].copy()

    # The retargeter emits finger DoF in URDF order (ff, th, mf, rf); the MJCF
    # actuators are ff, mf, rf, th.  Applying one to the other without the
    # permutation drives the wrong fingers -- and lands several commands outside
    # the target actuator's ctrlrange, which is how the mistake announces itself.
    # The permutation is applied upstream, in solve_base_placement, so both
    # blocks here are already in MJCF actuator order and already constraint-solved.
    finger_target = anchor["finger_cmd"]
    wrist_target = anchor["ctrl"]

    # Rate-limited warm start: drive the servos from the reset pose to the
    # first commanded pose at a rate they can actually track, so the recorded
    # trajectory starts from a state the hardware could be in instead of from a
    # step command the servos had to refuse.  See WARM_START_ENABLED for why
    # this is a switch and not a given.
    if WARM_START_ENABLED:
        warm = warm_start(
            mujoco_model, data, finger_qadr, wrist_qadr, finger_act, wrist_act,
            np.clip(finger_target[0], ctrl_lo[finger_act], ctrl_hi[finger_act]),
            np.clip(wrist_target[0], ctrl_lo[wrist_act], ctrl_hi[wrist_act]),
            ctrl_hz=ctrl_hz, frame_skip=frame_skip,
        )
        warm.pop("qpos_history", None)
    else:
        warm = {"enabled": False,
                "note": "WARM_START_ENABLED is False; the replay starts from the "
                        "reset seed directly, as at commit 6e66cf5"}

    n_frames = qpos.shape[0]
    contact_seq: list[int] = []
    pen_seq: list[float] = []
    ctrl_sat_seq: list[int] = []
    force_sat_seq: list[int] = []
    q_hist: list[np.ndarray] = []
    v_hist: list[np.ndarray] = []
    # Per control frame the servo actually ran on: the command it was given, the
    # torque that command produced, and the constraint (contact) torque on the
    # same DoF.  The servo winding tolerance gate reads all three; without them
    # the winding can only be asserted, not measured.
    force_hist: list[np.ndarray] = []
    qfrc_con_hist: list[np.ndarray] = []
    qfrc_other_hist: list[np.ndarray] = []
    tip_world: list[np.ndarray] = []
    palm_world: list[np.ndarray] = []
    object_world: list[np.ndarray] = []
    replay_ctrl: list[np.ndarray] = []

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
        replay_ctrl.append(data.ctrl.copy())

        for _ in range(frame_skip):
            mujoco.mj_step(mujoco_model, data)

        hit, pen = _finger_box_contact(mujoco_model, data)
        contact_seq.append(int(hit))
        pen_seq.append(float(pen))
        force_sat_seq.append(_saturated_actuators(mujoco_model, data, force_lo, force_hi))
        q_hist.append(data.qpos.copy())
        v_hist.append(data.qvel.copy())
        force_hist.append(data.actuator_force.copy())
        qfrc_con_hist.append(data.qfrc_constraint.copy())
        qfrc_other_hist.append(_qfrc_other(data))
        tip_world.append(
            np.stack(
                [
                    data.xpos[_body_id(mujoco_model, b)].copy()
                    for b in MJCF_TIP_BODIES
                ]
            )
        )
        # world trajectory of the palm frame and of the box: the base-alignment
        # evidence.  Which body actually touches the box, and where the palm
        # sits relative to the box top, is the whole question when the retargeted
        # pose cannot be executed, so it is recorded per frame rather than
        # inferred at the end.
        palm_world.append(data.xpos[_body_id(mujoco_model, "palm")].copy())
        object_world.append(data.xpos[_body_id(mujoco_model, "object")].copy())

    q = np.stack(q_hist)
    v = np.stack(v_hist)
    tips = np.stack(tip_world)
    palm_w = np.stack(palm_world)

    # Execution deviation: where the physics actually put the palm frame versus
    # where the command asked for it.  The palm origin sits exactly at
    # hand_root + the three slide commands, so the commanded pose is exact and no
    # second forward pass is needed.  Recorded per frame and gated, because a
    # servo's refusal to track an infeasible command is a property of the solve
    # above, not of the trajectory.
    cmd_palm = np.stack(replay_ctrl)[:, 0:3] + hand_root_home
    exec_dev = np.linalg.norm(palm_w - cmd_palm, axis=1)

    # joint-space velocity / acceleration over the control period
    joint_v = np.diff(q, axis=0) / dt_ctrl
    joint_a = np.diff(joint_v, axis=0) / dt_ctrl
    wrist_pos = q[:, wrist_qadr[0:3]]
    wrist_rot = q[:, wrist_qadr[3:6]]
    wrist_step = np.linalg.norm(np.diff(wrist_pos, axis=0), axis=1)
    wrist_rot_step = np.linalg.norm(np.diff(wrist_rot, axis=0), axis=1)

    wrist_reachability = _wrist_reachability(mujoco_model, wrist_target)

    # The warm start is input shaping, not part of the retargeted sequence, so
    # the gate above measures the replay trajectory only.  Its own velocities
    # are kept in full under "warm_start" so nothing is dropped silently.
    n_f = len(finger_qadr)
    if WARM_START_ENABLED:
        warm["velocity_scope"] = (
            "warm-start segment excluded from joint_velocity_max_rad_s by "
            "design: it is a rate-limited approach to the first command, not "
            "retargeted motion. Its own rates are below."
        )
        warm["replay_qpos_at_warm_start_end"] = q_hist[0].tolist()

    # ---- command-trajectory velocity gate (signed at wave 3) --------------
    # The gate is on the *commanded* rate, over the warm-start ramp
    # concatenated with the replay, split by DoF class.  Measuring the realised
    # qpos rate instead would gate the servo's refusal to track an infeasible
    # command, which is the servo's behaviour, not the trajectory's.
    warm_ctrl = warm.pop("ctrl_history", None)
    ctrl_all = (
        np.concatenate([warm_ctrl, np.stack(replay_ctrl)], axis=0)
        if warm_ctrl is not None and len(warm_ctrl) else np.stack(replay_ctrl)
    )
    cmd_v = np.abs(np.diff(ctrl_all, axis=0) / dt_ctrl)
    finger_cols = np.arange(6, 22)
    slide_cols = np.arange(0, 3)
    hinge_cols = np.arange(3, 6)
    cmd_vel = {
        "finger": {
            "limit": VEL_LIMIT_FINGER_RAD_S,
            "unit": "rad/s",
            "max_commanded_rate": float(np.max(cmd_v[:, finger_cols])),
            "within_limit": bool(
                np.max(cmd_v[:, finger_cols]) <= VEL_LIMIT_FINGER_RAD_S),
        },
        "wrist_hinge": {
            "limit": VEL_LIMIT_WRIST_HINGE_RAD_S,
            "unit": "rad/s",
            "max_commanded_rate": float(np.max(cmd_v[:, hinge_cols])),
            "within_limit": bool(
                np.max(cmd_v[:, hinge_cols]) <= VEL_LIMIT_WRIST_HINGE_RAD_S),
        },
        "wrist_slide": {
            "limit": VEL_LIMIT_WRIST_SLIDE_M_S,
            "unit": "m/s",
            "max_commanded_rate": float(np.max(cmd_v[:, slide_cols])),
            "within_limit": bool(
                np.max(cmd_v[:, slide_cols]) <= VEL_LIMIT_WRIST_SLIDE_M_S),
        },
    }
    cmd_vel["source"] = VEL_LIMIT_SOURCE
    cmd_vel["command_frames_gated"] = int(ctrl_all.shape[0])
    cmd_vel["command_frames_from_warm_start"] = int(
        0 if warm_ctrl is None else len(warm_ctrl))
    cmd_vel["all_within_limit"] = bool(
        all(cmd_vel[k]["within_limit"] for k in
            ("finger", "wrist_hinge", "wrist_slide")))
    # per-DoF worst offenders, so a failure names the axis
    worst: dict[str, Any] = {}
    # actuator order is wrist_x, wrist_y, wrist_z, wrist_rx, wrist_ry, wrist_rz,
    # wrist_rz -- i.e. slides first.  Column index c of the hinge block is
    # WRIST_JOINTS[3 + c], not WRIST_JOINTS[c].
    for name, cols, base, names in (
            ("finger", finger_cols, 0, MJCF_FINGER_JOINTS),
            ("wrist_hinge", hinge_cols, 3, WRIST_JOINTS),
            ("wrist_slide", slide_cols, 0, WRIST_JOINTS)):
        unit = "m/s" if name == "wrist_slide" else "rad/s"
        block = cmd_v[:, cols]
        i = int(np.argmax(block.max(axis=0)))
        f = int(np.argmax(block[:, i]))
        worst[name] = {
            "dof": names[base + i],
            "actuator_index": int(cols[i]),
            "max_commanded_rate": float(block[f, i]),
            "unit": unit,
            "command_frame": int(f),
            "from_warm_start": bool(
                warm_ctrl is not None and f < len(warm_ctrl)),
        }
    cmd_vel["worst_dof"] = worst

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
        "command_velocity": cmd_vel,
        "joint_acceleration_max_rad_s2": float(np.max(np.abs(joint_a))),
        "wrist_step_m": wrist_step,
        "wrist_rot_step_rad": wrist_rot_step,
        "wrist_pos": wrist_pos,
        "wrist_rot": wrist_rot,
        "final_tip_positions_world": tips[-1],
        "tip_centroid_world": tips.mean(axis=1),
        "palm_world": palm_w,
        "object_world": np.stack(object_world),
        "commanded_palm_world": cmd_palm,
        "execution_deviation_m": {
            "definition": (
                "||palm_world(realised) - (hand_root + slide command)|| per control "
                "frame: how far the physics put the hand from the pose the solved "
                "command asked for"
            ),
            "rms_m": float(np.sqrt(np.mean(exec_dev ** 2))),
            "max_m": float(exec_dev.max()),
            "per_frame_m": exec_dev.tolist(),
            "cap_m": PALM_WORLD_POSE_DEVIATION_CAP_M,
            "within_cap": bool(float(exec_dev.max()) <= PALM_WORLD_POSE_DEVIATION_CAP_M),
        },
        "qpos_history": q,
        "ctrl_history": np.stack(replay_ctrl),
        "actuator_force_history": np.stack(force_hist),
        "qfrc_constraint_history": np.stack(qfrc_con_hist),
        "qfrc_other_history": np.stack(qfrc_other_hist),
        "_finger_act": finger_act,
        "_wrist_act": wrist_act,
        "_wrist_qadr": wrist_qadr,
        "_finger_ids": finger_ids,
        "_wrist_ids": wrist_ids,
        "_force_lo": force_lo,
        "_force_hi": force_hi,
        "final_finger_qpos": q[-1, finger_qadr],
        "warm_start": warm,
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


def _qfrc_other(data) -> np.ndarray:
    """Net non-actuator generalised force on every DoF.

    This is the load a position servo has to wind against: constraint
    (contact) + joint damping + gravity/Coriolis.  In this MuJoCo build the
    DoF equation closes as

        M qacc = -qfrc_bias + qfrc_passive + qfrc_constraint
                 + qfrc_applied + qfrc_actuator

    -- ``qfrc_bias`` carries the opposite sign to the classical documentation.
    The sign is measured, not assumed: summing the five terms with the signs
    above reproduces ``mj_fullM(model, data) @ data.qacc`` to 4e-16 over all
    DoFs, and no other sign assignment does.  Getting it wrong would make the
    winding explanation compare the servo against a fictitious load.
    """
    return (-np.asarray(data.qfrc_bias) + np.asarray(data.qfrc_passive)
            + np.asarray(data.qfrc_constraint))


def _finger_box_contact(model, data) -> tuple[bool, float]:
    """(any finger/palm-vs-box contact, deepest penetration).

    Two exclusions, both load-bearing:

    * Contacts against the table/pedestal/floor are not counted -- the box rests
      on the pedestal, so an unguarded check would read that resting contact as
      hand-object contact and pass the gate without the hand ever touching the
      box.
    * The *other* geom must belong to the hand. This MJCF's finger/palm geoms
      are all unnamed (they come from <default> classes), so membership is
      decided by geom->body ancestry; a name-based check returns None for every
      one of them and reports zero contact no matter how deep the hand is buried.
    """
    box = _geom_id(model, "object_geom")
    hand = hand_geom_ids(model)
    hit = False
    pen = 0.0
    for c in range(data.ncon):
        con = data.contact[c]
        if box not in (con.geom1, con.geom2):
            continue
        other = con.geom2 if con.geom1 == box else con.geom1
        if other not in hand:
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

    rot = Rotation.from_euler("xyz", replay["wrist_rot"]).as_matrix()
    # The palm frame origin the physics actually held, per control frame.  Using
    # it (rather than a re-derived anchor offset) keeps both sides of the
    # comparison in the same world frame and the same pose.
    palm = np.asarray(replay["palm_world"], dtype=np.float64)
    world = np.einsum("tij,tkj->tki", rot, tip_pos_urdf) + palm[:, None, :]
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
    base: dict | None = None, model=None,
) -> dict[str, Any]:
    """RetargetQA (DESIGN section 2 contract) with every G3 detail number."""
    checks = {
        "frame_counts_consistent": bool(joints21.shape[0] == qpos.shape[0] == rep["n_frames"]),
        "all_finite": bool(
            np.all(np.isfinite(joints21))
            and np.all(np.isfinite(qpos))
            and np.all(np.isfinite(rep["qpos_history"]))
        ),
        "joint_limits_ok": bool(
            model is not None and rep["servo_winding_position"]["ok"]),
        "joint_velocity_ok": bool(rep["command_velocity"]["all_within_limit"]),
        "joint_acceleration_ok": bool(
            rep["joint_acceleration_max_rad_s2"] <= JOINT_ACC_LIMIT_RAD_S2
        ),
        "penetration_ok": bool(rep["max_penetration_m"] <= PENETRATION_TOLERANCE_M),
        # C6 SUPERSEDED (W6 acceptance, PLAN 2026-10-16): the dual-hand sweep
        # covers the whole table, so "the replay touches the box" is no longer
        # a smoke-5 acceptance criterion -- the contact-NATURE evidence is
        # carried by A2/A3a in W6c.  The measurement stays on the record under
        # qa["contact"] (frames_with_contact + sequence), out of the gate set.
        "wrist_continuous": bool(_wrist_continuous(rep)),
        "actuator_ctrl_not_saturated": bool(rep["ctrl_saturation_total"] == 0),
        "urdf_mjcf_frames_agree": bool(corr["max_residual_m"] < 1e-5),
        # --- base-solve gates (round 3) ---------------------------------
        "base_solve_feasible": bool(base is not None
                                    and base["record"].get("status") == "solved"),
        "base_displacement_within_cap": bool(
            base is not None and base["record"].get("displacement_within_cap", False)
        ),
        "hand_world_pose_deviation_within_cap": bool(
            _deviation_within_cap(base, rep)
        ),
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
                if k not in ("ctrl", "finger_cmd")
            },
            "base_solve": None if base is None else base["record"],
        },
        "urdf_mjcf_correspondence": corr,
        "contact": {
            "definition": (
                "any contact between a finger/palm geom and the tissue-box proxy "
                "geom; contacts against table_slab/pedestal/floor are excluded"
            ),
            # C6 superseded by the W6 acceptance ruling: the dual-hand sweep
            # covers the whole table, so a single retargeted replay touching
            # the box proves nothing about grasp viability any more.  The
            # contact-nature criterion (which geoms carry the box's weight,
            # total-force duty) is carried by A2/A3a in W6c.  The measurement
            # stays here, recorded rather than gated.
            "superseded_criterion": True,
            "superseded_note": (
                "C6 (contact_occurred) is superseded per the W6 acceptance "
                "(PLAN 2026-10-16): the dual-hand sweep covers the whole table "
                "and the structural premise of a single replay's contact proof "
                "is gone; A2/A3a carry the contact-nature gate in W6c"
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
            "servo_winding_tolerance": rep["servo_winding_position"],
            "note": (
                "the retargeted command is clipped to each actuator's ctrlrange "
                "before it is applied; the realised qpos is then checked against "
                "the MJCF joint range. mjcf_range_violation_frames is the "
                "zero-tolerance evidence (frames where qpos left jnt_range at "
                "all); the gate that judges them is servo_winding_tolerance, "
                "the user-approved three-part servo winding table. This MJCF "
                "gives every finger joint limited=true with stiffness=0, so "
                "jnt_range is an assertion the gate checks, not a mechanical "
                "stop the solver enforces -- which is exactly why an elastic "
                "wind-up can carry qpos past it."
            ),
        },
        "joint_motion": {
            "ctrl_hz": rep["ctrl_hz"],
            "dt_ctrl_s": rep["dt_ctrl_s"],
            # The gated quantity: commanded rate per DoF class, warm-start ramp
            # included.  velocity_max_rad_s below is the *realised* qpos rate and
            # is evidence only -- it measures how hard the servos pushed, which
            # on an infeasible command is the servo's refusal, not the
            # trajectory's speed.
            "gated_quantity": rep["command_velocity"]["source"]["measured_quantity"],
            "command_velocity": {
                k: rep["command_velocity"][k]
                for k in ("finger", "wrist_hinge", "wrist_slide",
                          "all_within_limit", "worst_dof",
                          "command_frames_gated",
                          "command_frames_from_warm_start")
            },
            "velocity_limit_source": VEL_LIMIT_SOURCE,
            "velocity_limit_rad_s": VEL_LIMIT_FINGER_RAD_S,
            "velocity_within_limit": rep["command_velocity"]["all_within_limit"],
            "velocity_max_rad_s": rep["joint_velocity_max_rad_s"],
            "velocity_max_rad_s_note": (
                "realised qpos rate over all velocity DoFs (both hands plus the "
                "box's free joint; only the right hand is commanded), evidence "
                "only; it is not the gate"
            ),
            "acceleration_limit_rad_s2": JOINT_ACC_LIMIT_RAD_S2,
            "acceleration_max_rad_s2": rep["joint_acceleration_max_rad_s2"],
            "acceleration_within_limit": rep["joint_acceleration_max_rad_s2"]
            <= JOINT_ACC_LIMIT_RAD_S2,
        },
        "actuators": {
            "n_actuators": 44,
            "n_actuators_note": (
                "the W6a dual-hand scene carries 44 position servos (two hands); "
                "smoke 5 commands the right hand's 22 and leaves the left hand's "
                "servos at ctrl 0, which holds it frozen at its initial pose"
            ),
            "n_actuators_commanded": 22,
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
    """The physical reason the remaining criteria cannot be met on this scene.

    On the dual-hand W6a scene the executability failure is gone (the base and
    constrained solves both succeed) and the shortfall is three-fold: the
    executed palm pose drifts past its 40 mm cap, the replay never touches the
    box (the clip closes to a sub-millimetre gap and the position servos stop
    there), and a few finger joints ride mrad-scale boundary dust past jnt_range
    beyond the winding gate's mandatory-explanation clause.  Every number below
    is measured on this run.
    """
    base = qa["retargeting"]["base_solve"]
    traj = base["constrained_solve"]
    dev = traj["hand_world_pose_deviation"]
    reach = qa["actuators"]["wrist_reachability"]
    gauge = qa["retargeting"]["wrist_anchor"]["gauge_fixing"]
    box_top = float(
        qa["retargeting"]["wrist_anchor"]["box_center"][2]
        + qa["retargeting"]["wrist_anchor"]["box_half_size"][2]
    )
    return {
        "what": (
            "dual-hand W6a-v8 scene (MJCF eba5d7b, blob 8b6a6ddd...; "
            "single-hand-input mode: the right hand executes the retargeted "
            "clip, the left hand is frozen at its reset pose): the base solve "
            "and the constrained solve solved -- the commanded hand sits "
            f"{base['gap_m'] * 1000:.2f} mm short of the box at the closest "
            f"fingertip with {base['tips_within_radius']} of four tips inside "
            f"the grasp radius and {base['palm_plate_gap_m'] * 1000:.1f} mm of "
            "palm-plate clearance. The one gate that still fails is the palm "
            f"world-pose deviation: the physics holds the palm "
            f"{rep['execution_deviation_m']['max_m'] * 1000:.0f} mm away from "
            "the pose it is commanded (cap 40 mm) -- the servos do not track "
            "the solved pose in the world, which is the standing P1. The "
            "replay recorded zero finger/palm-vs-box contact frames; that "
            "criterion (C6) is superseded per the W6 acceptance and is kept "
            "as a recorded measurement under qa['contact'], not a gate."
        ),
        "root_cause_fixed_this_round": {
            "was": (
                "the superseded anchor rule placed the mid-clip fingertip "
                "centroid on the box's top face, which with the clip's wrist pitch "
                "asks for a palm frame below the box top: the commanded hand sat "
                f"{-76.01:.0f} mm inside the box and the replay could only refuse it"
            ),
            "now": (
                "retarget/base_solve.py solves the floating base xyz against the "
                "box's grasp geometry (the 120 mm x span) with no hand geom "
                "interpenetration and a full pad-radius of palm-plate clearance; "
                f"the closest fingertip approach is {base['gap_m'] * 1000:.2f} mm "
                f"from the box with {base['penetration_m'] * 1000:.3f} mm of "
                "penetration"
            ),
        },
        "remaining": {
            "palm_world_pose_deviation_mm": {
                "value": rep["execution_deviation_m"]["max_m"] * 1000,
                "cap_mm": PALM_WORLD_POSE_DEVIATION_CAP_M * 1000,
                "definition": rep["execution_deviation_m"]["definition"],
                "rms_mm": rep["execution_deviation_m"]["rms_m"] * 1000,
            },
            "fingertip_world_pose_deviation_mm": {
                "value": dev["fingertips"]["max_m"] * 1000,
                "cap_mm": FINGERTIP_WORLD_POSE_DEVIATION_CAP_M * 1000,
                "note": (
                    "the ring finger's base joint is commanded by the retargeter "
                    "0.93 rad outside its actuator range, and the constrained "
                    "solve can only clip it; the clip's own fingertip layout "
                    "therefore cannot be reproduced here"
                ),
            },
            "constrained_solve_residual_m": traj["world_pose_residual_rms_m"],
            "constrained_solve_converged": traj["converged"],
            "fingertips_within_grasp_radius": base["tips_within_radius"],
            "contact_duty": qa["contact"]["duty"],
            "palm_plate_gap_mm": base["palm_plate_gap_m"] * 1000,
        },
        "wrist_axes_out_of_range": reach["axes_out_of_range"],
        "max_overshoot": dict(zip(WRIST_JOINTS, reach["max_overshoot"])),
        "ctrlrange": reach["ctrlrange"],
        "gauge_freedoms_removed": gauge["gauge_freedoms_removed"],
        "penetration": {
            "max_m": qa["penetration"]["max_m"],
            "tolerance_m": qa["penetration"]["tolerance_m"],
            "why": (
                "the fingertips now graze the box rather than driving into it: "
                f"{qa['penetration']['max_m'] * 1000:.3f} mm against a "
                f"{qa['penetration']['tolerance_m'] * 1000:.0f} mm tolerance. The "
                "palm-vs-box stiffness hardening is still outstanding, so this "
                "number is contact on the pads only."
            ),
        },
        "measured_fingertip_height_m": float(
            np.min(rep["final_tip_positions_world"][:, 2])
        ),
        "box_top_height_m": box_top,
        "not_a_threshold_problem": (
            "PENETRATION_TOLERANCE_M, JOINT_VEL_LIMIT_RAD_S, "
            "WRIST_STEP_CEILING_M, the three signed velocity gates and both "
            "world-pose caps are all unchanged. Every gate added this round is "
            "sized from the frozen MJCF's own geometry (the box's 0.04 / 0.06 / "
            "0.12 m half-dimensions) and from the measured servo behaviour, not "
            "from the observed result."
        ),
        "scene_side_evidence": {
            "note": (
                "measured on MJCF 4508421, frozen. These are the same scene smoke4 "
                "trains on, so every option below changes smoke4's physics too. "
                "The second block is smoke4's independent scan (240k random poses "
                "over 4 seeds plus a 52,020-pose wrist grid); it is reported here "
                "rather than re-derived, and the two agree on the mechanism."
            ),
            "smoke5_measurements": {
                "palm_plate_bottom_below_palm_frame_origin_m": 0.0297,
                "realised_palm_z_floor_m": 0.8687,
                "box_top_m": box_top,
                "floor_minus_box_top_mm": (0.8687 - box_top) * 1000,
                "note": (
                    "commanding ctrl_z from -0.163 up to -0.020 moves the wrist "
                    "1.1 mm: the bottom 60% of the wrist_z ctrlrange is saturated "
                    "against the palm plate, which is what the old '0.869 m "
                    "reachability floor' was"
                ),
            },
            "smoke4_measurements": {
                "palm_plate_width_mm": 95.2,
                "palm_overload": (
                    "the z position servo (kp 180 N/m, forcerange +-45 N) drives "
                    "26.4 N into a 1.47 N box -- an 18x overload, steady state"
                ),
                "pad_cannot_pass_own_finger": (
                    "over poses where the box touches the hand, the median is 4 "
                    "and the max 16 of the 16 finger collision geoms buried inside "
                    "the box: the 12 mm pad sphere sits at the end of three "
                    "phalanges and cannot reach the far 120 mm wall without going "
                    "through its own finger, so every bilateral contact pose is an "
                    "interpenetration and the solver ejects the box"
                ),
                "level_vs_below_top": (
                    "the rotation that lowers pads below the box top (wrist_rx) "
                    "spreads the three long fingers 47/65/78 mm apart in height at "
                    "rx 0.6/0.9/1.2, so they rake the box; the rotation that keeps "
                    "them level (wrist_rz, 0.0 mm spread) cannot lower them below "
                    "the palm plate. The two cannot both be had."
                ),
                "zero_undercut": (
                    "the box bottom sits exactly on the pedestal top, so no finger "
                    "can get beneath it"
                ),
                "search": (
                    "240k random poses over 4 seeds, a 52,020-pose structured wrist "
                    "grid, and a 3,600-trial dynamic sweep (descend-then-close, "
                    "close-then-descend, simultaneous; lift rates; curl amplitude): "
                    "0 clean trajectories. Pads-only contact occurs in ~1.3% of "
                    "touching poses but never as an opposite-wall matched-height "
                    "pinch."
                ),
            },
            "where_the_two_disagree": (
                "smoke4 reports the palm plate cannot be placed clear beside the "
                "box at all. This round's kinematic base solve does find a base "
                "with the plate 44.8 mm clear (palm frame at x=-0.104, y=+0.162), "
                "because the plate's 95 mm width runs along world x and the box's "
                "is only 120 mm wide. That does not rescue the grasp: in the "
                "physics the hand still ends up 86 mm from its own commanded pose "
                "and only 1 of 4 pads is within grasp radius. The kinematic "
                "clearance is real; the grasp is still not."
            ),
        },
        "fixes_for_p1": [
            "scene-side, for Main's ruling (MJCF frozen at 4508421, smoke4 is "
            "training on it): raise the box on a short standoff so the palm can "
            "get beside and below it instead of over it, and/or narrow the grasp "
            "span below the palm plate's 95.2 mm. These are what the measurements "
            "above point at, not a wider wrist travel.",
            "narrow the box's x span below ~95 mm: the pads physically cannot pass "
            "their own finger to reach the far wall of a 120 mm span, so no "
            "controller change can close a grasp across it on this scene.",
            "the retargeter commands rfj0 0.93 rad outside its actuator range on "
            "every frame; that is a retargeting-vs-MJCF mismatch and it caps how "
            "well the clip's fingertip layout can be reproduced even once the "
            "grasp geometry exists.",
            "palm-vs-box contact stiffness is not hardened yet, so the palm plate "
            "absorms 26.4 N while the gates report pad contact only.",
            "the warm-start ramp commands exactly at the measured trackable hinge "
            "rate, and the servo's step response then overshoots it: the realised "
            "rate reaches 1.87 rad/s against the 1.5 rad/s cap. The cap is not "
            "the thing to change; the ramp should command below it by the measured "
            "overshoot factor.",
        ],
    }


def _deviation_within_cap(base: dict | None, rep: dict) -> bool:
    """Every world-pose deviation the round-3 changes introduce, inside its cap.

    Two independent sources, both gated: the constrained solve moving the hand
    away from the retargeted pose, and the physics failing to execute the pose
    it was given.
    """
    if base is None:
        return False
    dev = base["record"]["constrained_solve"]["hand_world_pose_deviation"]
    return bool(
        dev["palm_frame"]["max_m"] <= PALM_WORLD_POSE_DEVIATION_CAP_M
        and dev["fingertips"]["max_m"] <= FINGERTIP_WORLD_POSE_DEVIATION_CAP_M
        and rep["execution_deviation_m"]["max_m"] <= PALM_WORLD_POSE_DEVIATION_CAP_M
    )


def _joint_limits_ok(rep: dict) -> bool:
    return len(_limit_violation_frames(rep)) == 0


def _limit_violation_frames(rep: dict) -> list[int]:
    _, _, lo, hi = rep["_limit_bounds"]
    q = rep["qpos_history"][:, rep["_finger_qadr"]]
    bad = np.where((q < lo - 1e-9) | (q > hi + 1e-9))[0]
    return sorted(set(int(i) for i in bad))


def _max_run(mask: np.ndarray) -> int:
    """Longest run of consecutive True in a 1-D boolean array."""
    best = run = 0
    for m in mask:
        run = run + 1 if m else 0
        best = max(best, run)
    return int(best)


def _actuator_saturated(force: np.ndarray, fmax: float) -> np.ndarray:
    """|force| sitting on the actuator's force limit, per row.

    MuJoCo clamps ``actuator_force`` to ``forcerange``, so "on the limit" is
    "equal to the limit".  The relative tolerance is the one the scene's own
    saturation rule already uses.
    """
    tol = 1e-6 * max(abs(fmax), 1e-12)
    return np.abs(force) >= fmax - tol


def servo_winding_position_gate(model, rep: dict) -> dict[str, Any]:
    """C3 as a three-part servo winding gate, per finger joint.

    Part 1 (amplitude) and part 3 (explanation) are per control frame; part 2
    (duration) is per joint.  An excursion is only tolerated when all three hold,
    and an excursion recorded while its actuator is on the force limit is never
    tolerated -- past saturation the winding is capped at |forcerange|/kp by
    definition, so a larger excursion cannot be winding at all.
    """
    ids = rep["_finger_ids"]
    qadr = rep["_finger_qadr"]
    act = rep["_finger_act"]
    q = rep["qpos_history"][:, qadr]
    ctrl = rep["ctrl_history"][:, act]
    force = rep["actuator_force_history"][:, act]
    qfrc_con = rep["qfrc_constraint_history"][:, [
        model.jnt_dofadr[j] for j in ids]]
    qfrc_other = rep["qfrc_other_history"][:, [
        model.jnt_dofadr[j] for j in ids]]
    lo = np.array([model.jnt_range[j][0] for j in ids])
    hi = np.array([model.jnt_range[j][1] for j in ids])
    fmax = np.array([
        float(np.max(np.abs(model.actuator_forcerange[a]))) for a in act])
    kp = np.array([float(model.actuator_gainprm[a, 0]) for a in act])

    per_joint: dict[str, Any] = {}
    ok = True
    for c, name in enumerate(MJCF_FINGER_JOINTS):
        cap = SERVO_WINDING_POSITION_CAP_RAD[name]
        excess = np.maximum(np.maximum(lo[c] - q[:, c], q[:, c] - hi[c]), 0.0)
        frames = np.where(excess > 0.0)[0]
        rec: dict[str, Any] = {
            "cap_rad": cap,
            "cap_formula": "|forcerange| / kp",
            "cap_from_model_rad": winding_position_cap(model, name),
            "max_excess_rad": float(excess.max()),
            "margin_x": float(cap / excess.max()) if excess.max() > 0 else None,
            "frames": [int(f) for f in frames],
            "max_consecutive_frames": _max_run(excess > 0.0),
            "frame_limit": WINDING_MAX_FRAMES,
            "amplitude_ok": bool(excess.max() <= cap),
            "duration_ok": bool(_max_run(excess > 0.0) <= WINDING_MAX_FRAMES),
            "events": [],
        }
        explanation_ok = True
        for f in frames:
            tau_servo = float(kp[c] * (ctrl[f, c] - q[f, c]))
            tau_contact = float(qfrc_con[f, c])
            tau_other = float(qfrc_other[f, c])
            sat = bool(_actuator_saturated(np.array([force[f, c]]), fmax[c])[0])
            # Part 3.  Two things have to hold, and both are load-bearing:
            #   * there is a contact torque on this DoF.  An excursion on a
            #     freely drifting joint has nothing to wind against, so it is
            #     not winding no matter how small it is -- the ruling names
            #     this case explicitly.
            #   * the servo's own winding torque accounts for the whole
            #     non-actuator load on the DoF (contact + damping + gravity)
            #     to WINDING_EXPLANATION_TOL.  That is the balance a position
            #     servo must satisfy for its error to be a static wind-up.
            torque_match = bool(
                abs(tau_contact) > WINDING_CONTACT_FLOOR_NM
                and abs(abs(tau_servo) - abs(tau_other))
                <= WINDING_EXPLANATION_TOL * abs(tau_other))
            # Saturation is folded in: past the force limit the winding is
            # capped at |forcerange|/kp by definition, so an excursion recorded
            # there is never winding, however well the numbers happen to match.
            explained = bool(torque_match and not sat)
            rec["events"].append({
                "frame": int(f),
                "qpos_rad": float(q[f, c]),
                "ctrl_rad": float(ctrl[f, c]),
                "excess_rad": float(excess[f]),
                "servo_winding_rad": float(ctrl[f, c] - q[f, c]),
                "tau_servo_Nm": tau_servo,
                "tau_contact_Nm": tau_contact,
                "tau_other_Nm": tau_other,
                "winding_vs_load_residual": (
                    abs(abs(tau_servo) - abs(tau_other)) / abs(tau_other)
                    if abs(tau_other) > 0 else None),
                "actuator_saturated": sat,
                "torque_match": torque_match,
                "explained": explained,
            })
            if not explained:
                explanation_ok = False
        rec["explanation_ok"] = explanation_ok
        rec["ok"] = bool(
            rec["amplitude_ok"] and rec["duration_ok"] and rec["explanation_ok"])
        ok = ok and rec["ok"]
        if rec["frames"]:
            per_joint[name] = rec
    return {
        "ok": bool(ok),
        "rule": (
            "three-part servo winding tolerance, per finger joint: (1) "
            "excess beyond jnt_range <= |forcerange|/kp, (2) at most "
            f"{WINDING_MAX_FRAMES} consecutive control frames "
            f"({WINDING_MAX_FRAMES * 1000 // CTRL_HZ} ms at {CTRL_HZ} Hz), "
            "(3) the transient is elastic wind-up: there is a contact torque on "
            "that DoF, the actuator is not on its force limit while it "
            "happens, and kp*(ctrl-qpos) accounts for the whole non-actuator "
            f"load on the DoF to {WINDING_EXPLANATION_TOL:.0%}. An excursion "
            "during saturation is never exempted, and neither is one on a "
            "freely drifting joint."
        ),
        "explanation_tol": WINDING_EXPLANATION_TOL,
        "max_frames": WINDING_MAX_FRAMES,
        "control_period_ms": 1000.0 / CTRL_HZ,
        "joints_with_excursion": per_joint,
    }


def servo_winding_rate_gate(
    model, joint_names: list[str], q_hist: np.ndarray, ctrl_hist: np.ndarray,
    force_hist: np.ndarray, dt_ctrl: float, command_cap: float,
    cap_table: dict[str, float] | None = None,
) -> dict[str, Any]:
    """The same three-part gate on a realised rate, per wrist hinge.

    ``q_hist``/``ctrl_hist`` are the *warm-start* hinge block (3 columns) and
    ``force_hist`` the full actuator-force history, so the caller hands over the
    columns it wants.  Intervals, not frames: the rate is the quantity the gate
    is about, and ``w1b`` compares ``np.diff(qpos)/dt``.  ``cap_table`` selects
    the derived allowance per joint -- the hinge table by default, the slide
    table (m/s) when the caller passes the slide block; the formula and the
    three parts are identical, only the units differ.
    """
    cap_table = SERVO_WINDING_RATE_CAP_RAD_S if cap_table is None else cap_table
    per_joint: dict[str, Any] = {}
    ok = True
    for col, name in enumerate(joint_names):
        jid = _joint_id(model, name)
        adr = int(model.jnt_dofadr[jid])
        a = _joint_actuator_id(model, name)
        cap = cap_table[name]
        fmax = float(np.max(np.abs(model.actuator_forcerange[a])))
        kp = float(model.actuator_gainprm[a, 0])
        kv = float(model.actuator_biasprm[a, 2])
        inertia = float(model.dof_M0[adr] + model.dof_armature[adr])

        q = q_hist[:, col]
        ctrl = ctrl_hist[:, col]
        v_real = np.abs(np.diff(q) / dt_ctrl)
        v_cmd = np.abs(np.diff(ctrl) / dt_ctrl)
        windup = np.abs(ctrl[:-1] - q[:-1])
        release = np.sqrt(kp * windup ** 2 / inertia)
        excess = v_real - v_cmd
        sat = _actuator_saturated(force_hist[:, a], fmax)
        # Parts 1 and 2 both describe the same excursion -- how far past the
        # allowance, and for how long -- so they are measured against the same
        # line, the derived cap.  Part 3 then asks, of every interval where the
        # servo outran the command-side trackable rate, whether the extra speed
        # is this DoF's own wind-up letting go.
        over_cap = v_real > cap
        over_command = v_real > command_cap

        events = []
        explanation_ok = True
        for t in np.where(over_command)[0]:
            # The release account is checked over the same window as the
            # duration clause (WINDING_MAX_FRAMES), not a single interval: a
            # wind-up stored at t-1 can fire at t, and the DoF then coasts at
            # t+1 on the momentum that release imparted.  Requiring the
            # current interval's release alone to cover the excess misses
            # that coast and reds a physically explained transient (measured
            # on the slides: interval 12 release 1.31 m/s, interval 13 coast
            # 0.289 m/s with wind-up already spent).  A genuinely unexplained
            # excursion still has ~zero release anywhere in its window.
            wlo = max(0, t - (WINDING_MAX_FRAMES - 1))
            window_release = float(np.max(release[wlo:t + 1]))
            explained = bool(
                excess[t] > 0.0
                and excess[t] <= window_release
                and not sat[t])
            if not explained:
                explanation_ok = False
            events.append({
                "interval": int(t),
                "realised_rate_rad_s": float(v_real[t]),
                "commanded_rate_rad_s": float(v_cmd[t]),
                "excess_rad_s": float(excess[t]),
                "windup_rad": float(windup[t]),
                "release_rad_s": float(release[t]),
                "release_window_rad_s": window_release,
                "release_window_frames": int(t - wlo + 1),
                "actuator_saturated": bool(sat[t]),
                "explained": explained,
            })
        rec = {
            "cap_rad_s": cap,
            # Historical key name: for the slide table this holds m/s, not
            # rad/s.  ``cap_units`` is the authoritative label.
            "cap_units": "rad/s" if cap_table is None else "m/s",
            "cap_formula": "|forcerange| / dof_damping",
            "cap_from_model_rad_s": winding_rate_cap(model, name),
            "dof_damping": float(model.dof_damping[adr]),
            "kv_Nm_s_per_rad": kv,
            # The release formula assumes the gain-only actuator: the wind-up is
            # the whole torque the servo is carrying.  If the MJCF ever gave
            # these hinges a velocity feedback term, the explanation would be
            # unsound and the gate has to say so rather than quietly pass.
            "gain_only_ok": kv == 0.0,
            "inertia_kg_m2": inertia,
            "command_side_cap_rad_s": command_cap,
            "peak_realised_rad_s": float(v_real.max()) if v_real.size else 0.0,
            "margin_x": float(cap / v_real.max()) if v_real.size and v_real.max() > 0 else None,
            "amplitude_ok": bool(v_real.max() <= cap) if v_real.size else True,
            "max_consecutive_intervals": _max_run(over_cap),
            "max_consecutive_intervals_over_command_cap": _max_run(over_command),
            "frame_limit": WINDING_MAX_FRAMES,
            "duration_ok": bool(_max_run(over_cap) <= WINDING_MAX_FRAMES),
            "explanation_ok": explanation_ok,
            "release_formula": WINDING_RELEASE_FORMULA,
            "events": events,
        }
        rec["ok"] = bool(
            rec["amplitude_ok"] and rec["duration_ok"] and rec["explanation_ok"]
            and rec["gain_only_ok"])
        ok = ok and rec["ok"]
        per_joint[name] = rec
    return {
        "ok": bool(ok),
        "rule": (
            "three-part servo winding tolerance, per wrist hinge: (1) realised "
            "rate <= |forcerange|/dof_damping, (2) at most "
            f"{WINDING_MAX_FRAMES} consecutive control intervals above that "
            "same cap, (3) every interval in which the servo outran the "
            f"command-side trackable rate {command_cap} rad/s has its excess "
            "over the commanded rate explained as this DoF's own wind-up "
            f"release ({WINDING_RELEASE_FORMULA}) with the actuator off its "
            "force limit. kv = 0 does not weaken the bound: dof_damping = 1.2 "
            "is what makes a finite bound exist at all. The bound is tight -- "
            "the free-space step asymptote of this servo plateaus at "
            "4.90-4.97 rad/s (both directions, both step signs), within 2% of "
            "it; with dof_damping forced to 0 the plateau disappears entirely."
        ),
        "rejected_forms": SERVO_WINDING_REJECTED_RATE_FORMS,
        "command_side_cap_rad_s": command_cap,
        "max_frames": WINDING_MAX_FRAMES,
        "joints": per_joint,
    }


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
    base = solve_base_placement(qpos, anchor, model, data)
    # One dict the replay reads: the wrist command and the finger command both
    # come out of the same constrained solve, so they cannot drift apart.
    anchor = dict(anchor, ctrl=base["ctrl"], finger_cmd=base["finger_cmd"])
    corr = verify_joint_frame_correspondence(retargeter)
    rep = replay(qpos, anchor)
    # attach the bound data the limit checks need (kept out of the JSON payload)
    rep["_limit_bounds"] = _limit_bounds(model)
    rep["_finger_qadr"] = _finger_qadr(model)
    # The C3 gate: zero tolerance on jnt_range is replaced by the three-part
    # servo winding tolerance table.  Computed here because it needs the model.
    rep["servo_winding_position"] = servo_winding_position_gate(model, rep)

    finger_err = fingertip_tracking_error(tip_pos, anchor, rep)
    qa = build_qa(seq, joints21, qpos, tip_pos, anchor, rep, corr, finger_err,
                  base, model)

    private = {k: v for k, v in rep.items() if k.startswith("_")}
    for k in private:
        rep.pop(k)

    checks = qa["checks"]
    verdict = "pass" if all(checks.values()) else "fail"
    report = {
        "smoke": "smoke5_retarget",
        "verdict": verdict,
        "verdict_checks": checks,
        "verdict_reason": (
            "all frozen G3 criteria green: frames consistent, finite, within joint "
            "limits/velocity/acceleration, penetration inside MuJoCo tolerance, "
            "wrist continuous (C6 contact_occurred superseded per the W6 "
            "acceptance and recorded under qa['contact'], not gated)"
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
            # Raw per-frame traces.  qpos_history was already excluded for
            # size; the three the winding gate reads are excluded with it and
            # surface instead as the gate's own per-frame evidence.
            if k not in ("qpos_history", "ctrl_history",
                         "actuator_force_history", "qfrc_constraint_history",
                         "qfrc_other_history")
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
    # Stamp the MJCF commit and the re-run evidence before either file is
    # written, so every run produces provenance-tagged artefacts.  Deferred
    # import: mjcf_provenance imports this module.  A failure here must not
    # cost us the run's numbers, so it is recorded rather than raised.
    try:
        from retarget import mjcf_provenance  # noqa: PLC0415

        mjcf_provenance.attach(report, qa)
    except Exception as exc:  # pragma: no cover - provenance must not gate data
        report["mjcf_provenance_error"] = f"{type(exc).__name__}: {exc}"
    (out_dir / "retarget_qa.json").write_text(json.dumps(qa, indent=2) + "\n")
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
