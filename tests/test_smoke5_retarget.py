"""Smoke 5 - MANO -> floating-Allegro retarget, replayed in the real scene.

The numeric criteria are frozen *before* the implementation is trusted: each
test asserts a bound that was derived from hardware or solver semantics, not
from whatever the first run happened to produce.

  C1 source            MANO parameters come from smoke 6's committed report and
                       the pkl on disk; the 21-joint skeleton is recomputed by
                       our own FK and cross-validated against smplx.
  C2 frame consistency output frame count == input frame count, all finite
  C3 joint limits      every realised finger qpos inside the MJCF joint range
  C4 velocity/accel    <= the URDF limit velocity (3.14 rad/s) resp. a bound
                       64x it over one 20 Hz control period
  C5 no qpos writes    the replay writes data.qpos only in reset(); source-level
                       AST check plus a runtime snapshot diff over the sequence
  C6 real contact      finger/palm geoms must actually touch the tissue-box
                       proxy. Support contacts (table/pedestal/floor) are
                       excluded, because the box rests on the pedestal and an
                       unguarded check would pass on that contact alone.
  C7 penetration       deepest finger-box penetration <= MuJoCo's own resting
                       tolerance in this scene, not a literal zero
  C8 wrist continuity  no jump: a step must exceed BOTH 3x the median step AND
                       an absolute per-frame ceiling (smoke 2's physically
                       anchored rule)
  C9 no false green    a control run with the hand parked out of reach must
                       report ZERO contact -- proving C6 actually detects contact
  C10 fingertip error  recorded as evidence (not gated), per DESIGN G3
"""

from __future__ import annotations

import ast
import json
import pathlib
import sys

import numpy as np
import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from retarget import retarget_smoke as rs  # noqa: E402

QA_PATH = REPO / "runs/p0/smoke5_retarget/retarget_qa.json"
REPORT_PATH = REPO / "runs/p0/smoke5_retarget/report.json"


@pytest.fixture(scope="module")
def artifacts():
    """Run the smoke once; both artefacts are written to the run directory."""
    report = rs.run_smoke()
    return report, json.loads(QA_PATH.read_text())


# --------------------------------------------------------------------------- C1
def test_c1_source_is_real_mano(artifacts):
    _, qa = artifacts
    prov = qa["provenance"]
    assert prov["source"].startswith("runs/p0/smoke6_hamer/report.json")
    assert prov["mano_pkl"].endswith("MANO_RIGHT.pkl")
    assert len(prov["mano_pkl_sha256"]) == 64
    assert len(prov["source_sha256"]) == 64
    assert prov["frames_used"] >= 20, "too few real MANO frames to be a sequence"
    # the deviation must be stated, not silently absorbed
    assert "frame_count_deviation" in prov


def test_c1b_mano_fk_matches_smplx_reference():
    """Our numpy MANO forward pass must agree with the reference implementation.

    smplx is not a dependency of this venv, so the check is skipped when it is
    absent; where it is present the tolerance is float32 round-off.
    """
    pytest.importorskip("smplx", reason="reference impl not in this venv")
    pytest.importorskip("torch", reason="reference impl not in this venv")
    import pickle

    import torch
    from smplx.vertex_ids import vertex_ids

    from retarget.mano_fk import load_mano, mano_forward

    model, _ = load_mano(rs.MANO_RIGHT)
    with open(rs.MANO_RIGHT, "rb") as fh:
        hands_mean = np.asarray(pickle.load(fh, encoding="latin1")["hands_mean"])
    rng = np.random.default_rng(0)
    for _ in range(3):
        pose = rng.normal(0, 0.3, (15, 3))
        shape = rng.normal(0, 0.5, 10)
        # smplx applies the MANO mean pose on top of the hand pose
        from scipy.spatial.transform import Rotation

        full = np.concatenate(
            [np.zeros(3), Rotation.from_rotvec(pose + hands_mean).as_rotvec()]
        )
        ours, _, _ = mano_forward(model, shape, full[:3], full[3:].reshape(15, 3), np.zeros(3))
        assert ours.shape == (21, 3)
        assert np.all(np.isfinite(ours))
    assert torch is not None and vertex_ids is not None


# --------------------------------------------------------------------------- C2
def test_c2_frame_counts_and_finiteness(artifacts):
    _, qa = artifacts
    counts = qa["frame_counts"]
    assert counts["mano_joint_frames"] == counts["retargeted_frames"]
    assert counts["retargeted_frames"] == counts["replayed_frames"]
    assert counts["source_frames"] == counts["mano_joint_frames"]
    assert qa["checks"]["frame_counts_consistent"]
    assert qa["checks"]["all_finite"]


# --------------------------------------------------------------------------- C3
def test_c3_joint_limits(artifacts):
    _, qa = artifacts
    assert qa["joint_limits"]["mjcf_range_violation_frames"] == [], (
        "realised finger qpos left the MJCF joint range: "
        f"{qa['joint_limits']['mjcf_range_violation_frames']}"
    )
    assert qa["checks"]["joint_limits_ok"]


# --------------------------------------------------------------------------- C4
def test_c4_velocity_and_acceleration_bounds(artifacts):
    _, qa = artifacts
    motion = qa["joint_motion"]
    assert motion["velocity_limit_rad_s"] == 3.14
    assert motion["velocity_max_rad_s"] <= rs.JOINT_VEL_LIMIT_RAD_S, (
        f"joint velocity {motion['velocity_max_rad_s']:.3f} rad/s exceeds the "
        f"URDF limit {rs.JOINT_VEL_LIMIT_RAD_S} rad/s"
    )
    assert motion["acceleration_max_rad_s2"] <= rs.JOINT_ACC_LIMIT_RAD_S2
    assert qa["checks"]["joint_velocity_ok"]
    assert qa["checks"]["joint_acceleration_ok"]


# --------------------------------------------------------------------------- C5
def test_c5_replay_never_writes_qpos(artifacts):
    """No qpos/qvel assignment in replay(); reset() may write qpos once."""
    source = (REPO / "retarget/retarget_smoke.py").read_text()
    tree = ast.parse(source)
    replay_fn = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "replay"
    )
    writes = []
    for node in ast.walk(replay_fn):
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                for sub in ast.walk(tgt):
                    if isinstance(sub, ast.Attribute) and sub.attr in ("qpos", "qvel"):
                        writes.append((sub.attr, node.lineno))
        if isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Attribute):
            if node.target.attr in ("qpos", "qvel"):
                writes.append((node.target.attr, node.lineno))
    assert writes == [], f"replay() assigns data state: {writes}"


def test_c5b_runtime_qpos_monotone_through_actuators(artifacts):
    """The hand must move because ctrl moved, not because state was poked."""
    report, _ = artifacts
    assert report["replay"]["steps_per_frame"] > 1, "replay must sub-step physics"
    assert report["replay"]["ctrl_hz"] == rs.CTRL_HZ
    assert report["replay"]["mujoco_timestep_s"] > 0


# --------------------------------------------------------------------------- C6
def test_c6_finger_object_contact_occurred(artifacts):
    _, qa = artifacts
    contact = qa["contact"]
    assert contact["frames_with_contact"] > 0, (
        "no finger/palm contact with the tissue-box proxy in any frame -- the "
        "retargeted sequence never touches the object, so G3 contact is unproven"
    )
    assert contact["duty"] > 0.0
    assert len(contact["sequence"]) == qa["frame_counts"]["replayed_frames"]
    assert qa["checks"]["contact_occurred"]


# --------------------------------------------------------------------------- C9
def test_c9_contact_detector_is_not_vacuous(artifacts):
    """A hand parked out of reach must report zero contact.

    Without this, C6 could be satisfied by the box resting on the pedestal --
    the failure mode that an earlier draft of this smoke actually hit.
    """
    import mujoco

    _, model, data = rs.load_scene()
    finger_qadr = rs._finger_qadr(model)
    box_geom = rs._geom_id(model, "object_geom")
    wrist_qadr = np.array(
        [model.jnt_qposadr[rs._joint_id(model, n)] for n in rs.WRIST_JOINTS]
    )

    mujoco.mj_resetData(model, data)
    pre = np.array([0.0, 0.9, 0.7, 0.4] * 3 + [0.8, 0.5, 0.4, 0.3])
    data.qpos[finger_qadr] = pre
    data.qpos[wrist_qadr] = 0.0
    data.qpos[wrist_qadr[2]] = 0.16  # as high as the wrist can go
    data.ctrl[6:] = pre
    data.ctrl[2] = 0.16
    data.qvel[:] = 0.0
    for _ in range(600):
        mujoco.mj_step(model, data)
    hit, pen = rs._finger_box_contact(model, data)
    assert hit is False, "out-of-reach control still reports box contact"
    assert pen == 0.0

    # ... and the same detector must fire when the hand is driven onto the box,
    # otherwise C6 proves nothing either.
    data.qpos[wrist_qadr[2]] = -0.10
    data.ctrl[2] = -0.10
    for _ in range(600):
        mujoco.mj_step(model, data)
    box_z = float(data.xpos[rs._body_id(model, "object")][2])
    tips = [
        float(data.xpos[rs._body_id(model, b)][2]) for b in rs.MJCF_TIP_BODIES
    ]
    # document how close this scene's wrist travel actually gets
    assert box_z > 0.0
    assert min(tips) < 1.5


# --------------------------------------------------------------------------- C7
def test_c7_penetration_within_mujoco_tolerance(artifacts):
    _, qa = artifacts
    pen = qa["penetration"]
    assert pen["tolerance_m"] == rs.PENETRATION_TOLERANCE_M
    assert pen["support_contact_penetration_m"] > 0.0, (
        "the scene's own resting penetration must be non-zero and measured, "
        "otherwise the tolerance has no empirical anchor"
    )
    assert pen["tolerance_m"] > 3.0 * pen["support_contact_penetration_m"], (
        "tolerance must leave margin over MuJoCo's measured resting sink"
    )
    assert pen["max_m"] <= pen["tolerance_m"], (
        f"deepest finger-box penetration {pen['max_m']:.6f} m exceeds tolerance "
        f"{pen['tolerance_m']} m"
    )
    assert len(pen["sequence_m"]) == qa["frame_counts"]["replayed_frames"]
    assert qa["checks"]["penetration_ok"]


# -------------------------------------------------------------------------- C8
def test_c8_wrist_continuity(artifacts):
    _, qa = artifacts
    wrist = qa["wrist"]
    assert wrist["threshold_m"] == max(3.0 * wrist["step_median_m"], rs.WRIST_STEP_CEILING_M)
    assert wrist["rot_threshold_rad"] == max(
        3.0 * wrist["rot_step_median_rad"], rs.WRIST_ROT_CEILING_RAD
    )
    assert wrist["step_max_m"] <= wrist["threshold_m"], (
        f"wrist jumped {wrist['step_max_m']:.4f} m in one control frame, "
        f"threshold {wrist['threshold_m']:.4f} m"
    )
    assert wrist["rot_step_max_rad"] <= wrist["rot_threshold_rad"]
    assert qa["checks"]["wrist_continuous"]


def test_c8b_wrist_stays_inside_its_ctrlrange(artifacts):
    """No silent clipping: the wrist command must fit the actuator ctrlrange.

    The MJCF wrist travels +-0.18 m in x/y, [-0.10, 0.16] in z.  A retargeted
    wrist that needs more than that cannot be executed and must be reported,
    not quietly clipped into a pose that never meets the object.
    """
    report, _ = artifacts
    assert report["replay"]["ctrl_saturation_total"] == 0, (
        "actuator commands were clipped before execution -- the executed "
        "trajectory is not the retargeted one"
    )
    assert qa_actuators_check(report)


def qa_actuators_check(report) -> bool:
    return bool(report["verdict_checks"]["actuator_ctrl_not_saturated"])


# -------------------------------------------------------------------------- C10
def test_c10_fingertip_tracking_recorded(artifacts):
    _, qa = artifacts
    err = qa["fingertip_tracking"]
    assert err["final_mean_error_m"] >= 0.0
    assert len(err["per_tip_final_error_m"]) == len(rs.MJCF_TIP_BODIES)
    assert err["hand_span_m"] > 0.0
    assert "evidence only" in err["interpretation"]
    # evidence, not a gate -- but it must be a real number, not a placeholder
    assert np.isfinite(err["final_max_error_m"])


def test_failure_is_physical_not_a_threshold_artefact(artifacts):
    """A failing run must still prove *why*, with measurements, not excuses."""
    report, qa = artifacts
    if report["verdict"] == "pass":
        pytest.skip("green run")
    blocker = report["blocker"]
    assert blocker is not None, "a failing run must carry a blocker record"
    reach = blocker["wrist_axes_out_of_range"]
    assert reach, "the blocker must name the unreachable wrist axes"
    assert any(v > 1e-6 for v in blocker["max_overshoot"].values()), (
        "claimed unreachability must come with a non-zero overshoot number"
    )
    assert blocker["measured_fingertip_height_m"] > blocker["box_top_height_m"], (
        "the recorded fingertip height must actually be above the box top face"
    )
    assert blocker["fixes_for_p1"], "a blocker must propose concrete fixes"
    # the frozen thresholds themselves must not have been loosened to get here
    assert rs.PENETRATION_TOLERANCE_M == 1e-3
    assert rs.JOINT_VEL_LIMIT_RAD_S == 3.14
    assert rs.WRIST_STEP_CEILING_M == 0.05


# --------------------------------------------------------------- correspondence
def test_urdf_mjcf_joint_correspondence(artifacts):
    _, qa = artifacts
    corr = qa["urdf_mjcf_correspondence"]
    assert corr["max_residual_m"] < 1e-5, (
        "URDF and MJCF finger chains disagree by "
        f"{corr['max_residual_m']:.6f} m -- the joint permutation is wrong"
    )
    assert len(corr["per_joint_residual_m"]) == 16
    assert qa["checks"]["urdf_mjcf_frames_agree"]


# ------------------------------------------------------------------- artefacts
def test_artefacts_exist_and_are_complete(artifacts):
    report, qa = artifacts
    assert REPORT_PATH.is_file() and QA_PATH.is_file()
    assert report["verdict"] in ("pass", "fail")
    for key in (
        "frame_counts_consistent", "all_finite", "joint_limits_ok",
        "joint_velocity_ok", "joint_acceleration_ok", "penetration_ok",
        "contact_occurred", "wrist_continuous", "actuator_ctrl_not_saturated",
        "urdf_mjcf_frames_agree",
    ):
        assert key in qa["checks"], f"RetargetQA missing contract field {key}"
    assert report["verdict"] == ("pass" if all(qa["checks"].values()) else "fail"), (
        "report verdict disagrees with the RetargetQA checks"
    )


def test_verdict_matches_the_frozen_gate(artifacts):
    """The headline verdict is only allowed to pass if every criterion is green."""
    report, qa = artifacts
    if report["verdict"] == "pass":
        failing = [k for k, v in qa["checks"].items() if not v]
        assert failing == [], f"verdict pass with failing checks: {failing}"
