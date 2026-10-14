"""
Smoke 2 acceptance test: VGGT-mps (Apple Silicon).

Input : runs/p0/assets/frames/f_0001.jpg .. f_0030.jpg  (30 frames)
Output: 30 camera poses (4x4) + 30 depth maps, computed on MPS.

Criteria are hard-coded below and MUST NOT be relaxed:
  (a) all 30 frames produce a finite 4x4 pose and a finite, strictly positive
      depth map  -> no NaN / Inf / non-positive depth anywhere
  (b) consecutive-frame relative pose step norm stays below 3x the sequence
      median step norm (both for the joint 6-DoF metric AND for the
      translation-only metric)
  (c) rotation and translation statistics are reported separately
  (d) the run really happened on MPS (device + non-trivial wall time),
      with an explicit op-level CPU fallback list
  (e) report.json exists, is complete, and has verdict == "pass"
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
FRAMES_DIR = REPO / "runs" / "p0" / "assets" / "frames"
OUT_DIR = REPO / "runs" / "p0" / "smoke2_vggt"
REPORT = OUT_DIR / "report.json"

N_FRAMES = 30
STEP_RATIO_LIMIT = 3.0  # criterion (b): single-step norm must be < 3x median

sys.path.insert(0, str(REPO / "perception"))


@pytest.fixture(scope="module")
def result():
    """Run the smoke once for the whole module (inference is expensive)."""
    import vggt_smoke

    return vggt_smoke.run_smoke(
        frames_dir=FRAMES_DIR,
        out_dir=OUT_DIR,
        n_frames=N_FRAMES,
        device="mps",
    )


# --------------------------------------------------------------------------
# (a) 30 frames produced, nothing NaN, depth strictly positive
# --------------------------------------------------------------------------
def test_all_30_frames_have_pose_and_depth(result):
    assert result["n_frames"] == N_FRAMES, f"expected {N_FRAMES} frames, got {result['n_frames']}"
    poses = result["poses"]
    depths = result["depths"]
    assert poses.shape[0] == N_FRAMES, f"pose count {poses.shape[0]} != {N_FRAMES}"
    assert depths.shape[0] == N_FRAMES, f"depth count {depths.shape[0]} != {N_FRAMES}"


def test_poses_are_finite_and_valid_rotation(result):
    poses = result["poses"]
    assert poses.shape[1:] == (4, 4), f"expected (N,4,4) poses, got {poses.shape}"
    assert np.isfinite(poses).all(), "NaN or Inf in camera poses"
    R = poses[:, :3, :3]
    t = poses[:, :3, 3]
    # bottom row must be [0 0 0 1]
    assert np.allclose(poses[:, 3, :], np.array([0.0, 0.0, 0.0, 1.0]), atol=1e-5), "bottom row not [0,0,0,1]"
    # R must be orthonormal with det +1
    RtR = R.transpose(0, 2, 1) @ R
    assert np.allclose(RtR, np.eye(3), atol=1e-3), "rotation blocks are not orthonormal"
    dets = np.linalg.det(R)
    assert np.all(dets > 0), "rotation determinant must be +1"
    assert np.isfinite(t).all(), "NaN or Inf in pose translation"


def test_depths_are_finite_and_strictly_positive(result):
    depths = result["depths"]
    assert depths.ndim == 3, f"expected (N,H,W) depth, got shape {depths.shape}"
    assert np.isfinite(depths).all(), "NaN or Inf in depth maps"
    assert (depths > 0).all(), f"non-positive depth values (min={depths.min()})"


def test_depths_are_not_constant_per_frame(result):
    """A constant/flat depth map means the depth head degenerated -> fail."""
    for i in range(result["depths"].shape[0]):
        d = result["depths"][i]
        assert d.std() > 1e-6, f"frame {i} depth map is constant (std={d.std()})"
        assert d.max() / d.min() > 1.01, f"frame {i} depth has no dynamic range"


def test_poses_are_not_all_identical(result):
    """All-identical poses would pass a naive NaN check but mean no geometry."""
    t = result["poses"][:, :3, 3]
    assert float(t.std(axis=0).max()) > 1e-6, "camera translation constant across all frames"
    R = result["poses"][:, :3, :3]
    rel = np.einsum("nij,nkj->nik", R[1:], R[:-1])
    angles = np.arccos(np.clip((rel[:, 0, 0] + rel[:, 1, 1] + rel[:, 2, 2] - 1) / 2, -1, 1))
    assert float(angles.max()) > 1e-5, "camera rotation identical across all frames"


# --------------------------------------------------------------------------
# (b) relative-pose step norm < 3x sequence median  (anti "teleport" red line)
# --------------------------------------------------------------------------
def _step_metrics(poses):
    """Per-step (i -> i+1) joint / rotation / translation metrics."""
    w2c = poses[:, :3, :3]
    w2c_t = poses[:, :3, 3]
    R_rel = np.einsum("nij,nkj->nik", w2c[1:], w2c[:-1])       # R_i^T R_{i+1}
    t_rel = w2c_t[1:] - np.einsum("nij,nj->ni", R_rel, w2c_t[:-1])
    rot = np.arccos(np.clip((R_rel[:, 0, 0] + R_rel[:, 1, 1] + R_rel[:, 2, 2] - 1) / 2, -1, 1))
    trans = np.linalg.norm(t_rel, axis=1)
    joint = np.sqrt(rot ** 2 + trans ** 2)
    return rot, trans, joint


def test_relative_pose_steps_have_no_outlier_beyond_3x_median(result):
    rot, trans, joint = _step_metrics(result["poses"])
    for name, steps in (("joint", joint), ("translation", trans)):
        median = float(np.median(steps))
        worst = float(steps.max())
        assert median > 0.0, f"{name} step median is 0 -> degenerate motion"
        limit = STEP_RATIO_LIMIT * median
        assert worst < limit, (
            f"{name} step outlier: max={worst:.6g} >= {STEP_RATIO_LIMIT}x median={median:.6g} "
            f"(limit={limit:.6g}); steps={np.round(steps, 6).tolist()}"
        )


def test_rotation_steps_have_no_outlier_beyond_3x_median(result):
    rot, _, _ = _step_metrics(result["poses"])
    median = float(np.median(rot))
    assert median > 0.0, "rotation step median is 0 -> degenerate motion"
    assert float(rot.max()) < STEP_RATIO_LIMIT * median, (
        f"rotation step outlier: max={rot.max():.6g} >= {STEP_RATIO_LIMIT}x median={median:.6g}"
    )


# --------------------------------------------------------------------------
# (c) rotation and translation statistics reported separately
# --------------------------------------------------------------------------
def test_rotation_and_translation_stats_reported_separately(result):
    stats = result["motion_stats"]
    for key in ("rotation_rad", "translation_m", "joint_m_rad"):
        assert key in stats, f"missing '{key}' in motion_stats (keys: {sorted(stats)})"
    for key in ("rotation_rad", "translation_m", "joint_m_rad"):
        s = stats[key]
        for field in ("mean", "median", "max", "std"):
            assert field in s, f"motion_stats['{key}'] missing '{field}'"
            assert math.isfinite(float(s[field])), f"motion_stats['{key}']['{field}'] is not finite"
    # camera units sanity: VGGT is metric-ish, translation must not be absurd
    assert 1e-6 < stats["translation_m"]["median"] < 10.0, (
        f"median step translation out of plausible range: {stats['translation_m']['median']}"
    )


def test_rot_vs_trans_statistics_are_distinct(result):
    """Separate accounting means rotation is not folded into translation."""
    stats = result["motion_stats"]
    assert stats["rotation_rad"] is not stats["translation_m"]
    assert stats["rotation_rad"]["median"] != stats["translation_m"]["median"]


# --------------------------------------------------------------------------
# (d) MPS really used, fallbacks accounted
# --------------------------------------------------------------------------
def test_ran_on_mps_with_measurable_time(result):
    assert result["device"] == "mps", f"device must be 'mps', got {result['device']!r}"
    assert result["mps_inference_seconds"] > 0.0, "MPS inference time not measured"
    assert result["device_probe"]["mps_available"] is True, "torch reports MPS unavailable"
    assert result["device_probe"]["allocated_via"] in ("mps", "cpu_then_mps")
    # CPU control run is the "MPS really on device" evidence
    assert result["cpu_control_seconds"] > 0.0, "no CPU control timing recorded"
    assert isinstance(result["cpu_fallbacks"], list), "cpu_fallbacks must be a list"


def test_report_json_complete(result):
    assert REPORT.exists(), f"missing {REPORT}"
    rep = json.loads(REPORT.read_text())
    for key in ("device", "key_lib_versions", "weights", "criteria", "verdict"):
        assert key in rep, f"report.json missing '{key}'"
    assert rep["verdict"] == "pass", f"verdict is {rep['verdict']!r}, expected 'pass'"
    assert rep["device"] == "mps"
    assert rep["weights"]["model_id"] == "facebook/VGGT-1B"
    assert rep["weights"]["revision"], "weights revision (sha) not recorded"
    for lib in ("torch", "numpy", "vggt_mps"):
        assert lib in rep["key_lib_versions"], f"key_lib_versions missing '{lib}'"
    for c in ("a_all_frames_no_nan", "b_step_norm_lt_3x_median", "c_rot_trans_separate"):
        assert c in rep["criteria"], f"report.json criteria missing '{c}'"
        assert rep["criteria"][c]["pass"] is True, f"criterion {c} not passing: {rep['criteria'][c]}"
    # report must carry the numeric quality numbers, not just booleans
    assert rep["reprojection"]["method"], "reprojection error must document its method"
    assert math.isfinite(rep["reprojection"]["median_px"]), "reprojection median must be finite"
    # anti-false-positive: the pose must beat the identity-pose control, otherwise
    # the depth/pose carry no geometric information over raw frame differencing.
    assert rep["reprojection"]["beats_identity_baseline"] is True, (
        f"VGGT reprojection {rep['reprojection']['median_px']:.2f}px is not better than the "
        f"identity-pose baseline {rep['reprojection']['identity_pose_baseline_px']:.2f}px"
    )
    assert rep["scale_consistency"]["cv"] is not None, "scale consistency missing"
    assert rep["fork_selection"]["chosen"].startswith("jmanhype/vggt-mps")


def test_evidence_overlays_exist(result):
    ev = OUT_DIR / "evidence"
    assert ev.is_dir(), f"missing evidence dir {ev}"
    for name in ("f_0001_depth_edges.jpg", "f_0015_depth_edges.jpg", "f_0030_depth_edges.jpg"):
        f = ev / name
        assert f.exists(), f"missing evidence overlay {name}"
        assert f.stat().st_size > 5_000, f"evidence overlay {name} suspiciously small"
