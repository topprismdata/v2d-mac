"""Smoke 6 acceptance: HaMeR (Mano-parameterized hand mesh recovery) on Apple MPS.

Criteria are HARD-CODED here and were frozen BEFORE the implementation existed.
They must never be relaxed; a genuinely failing criterion yields verdict=fail in
the report (this is a research bring-up, not a tuning exercise).

Run with the smoke venv interpreter:
    cd /Users/guohongbin/v2d-mac && runs/.venv-hamer/bin/python -m pytest \
        tests/test_smoke6_hamer.py -v

Frozen criteria (all numeric, none widened since 6.1):
  C1  MANO params per detected hand are non-empty and all finite
        pose  = 15x3 = 45 axis-angle values (MANO hand_pose, root excluded)
        shape = 10 betas
        trans = 3 camera translation components
  C2  median 2D hand-keypoint confidence >= 0.20 over all (frame, hand, kp)
      (HaMeR is a regressor with no confidence head; the confidence is that of
       the 2D hand-keypoint detector that drives it, frozen here as such)
  C3  mean MANO-joint reprojection error < 15 px over all (frame, hand, kp)
  C4  each of {left, right} hand detected in >= 80% of the 30 frames
  C5  handedness flips <= 2 across the 30 frames (left/right assignment by image
      x-order must stay consistent with the modal assignment)
  C6  dominant inference device == mps, and report.json enumerates cpu_fallback_ops
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

FRAMES = REPO / "runs/p0/assets/frames"
OUT_DIR = REPO / "runs/p0/smoke6_hamer"
EVIDENCE_DIR = OUT_DIR / "evidence"
REPORT = OUT_DIR / "report.json"

# ---------------------------------------------------------------- criteria --
TEST_FRAMES = [f"f_{i:04d}" for i in range(1, 31)]   # f_0001 .. f_0030
MIN_KEYPOINT_CONF_MEDIAN = 0.20      # C2
MAX_MEAN_REPROJ_ERROR_PX = 15.0       # C3
MIN_HAND_DETECTION_RATE = 0.80       # C4 (per handedness)
MAX_HANDEDNESS_FLIPS = 2             # C5
REQUIRED_DEVICE = "mps"              # C6

# C1: MANO parameter block layout
N_HAND_POSE_JOINTS = 15              # pose 15x3 = 45
N_BETAS = 10                         # shape
TRANSLATION_DIM = 3                  # trans

EVIDENCE_FRAMES = ["f_0010", "f_0020"]

REQUIRED_REPORT_KEYS = {
    "smoke",
    "verdict",
    "device",
    "cpu_fallback_ops",
    "key_lib_versions",
    "weights_version",
    "per_criterion",
    "per_frame",
    "mps_evidence",
    "evidence_files",
    "known_limitations",
}


@pytest.fixture(scope="module")
def run():
    from perception import hamer_smoke

    return hamer_smoke.run_smoke(
        frames=TEST_FRAMES, out_dir=OUT_DIR, evidence_dir=EVIDENCE_DIR
    )


def test_criterion_mano_params_finite(run):
    """C1: pose(15x3) + shape(10) + trans(3) present and finite for every hand.

    Frame-level coverage is C4's job (>=80% of frames per handedness), so this
    criterion constrains the parameters of every hand that *was* reconstructed.
    """
    assert sum(len(r["hands"]) for r in run["per_frame"].values()) > 0, (
        "no hand was reconstructed in any frame"
    )
    for name in TEST_FRAMES:
        for h in run["per_frame"][name]["hands"]:
            m = h["mano"]
            assert np.asarray(m["pose"]).shape == (N_HAND_POSE_JOINTS, 3), (
                f"{name}: MANO pose shape {np.asarray(m['pose']).shape} != (15, 3)"
            )
            assert np.asarray(m["shape"]).shape == (N_BETAS,), (
                f"{name}: MANO shape shape {np.asarray(m['shape']).shape} != (10,)"
            )
            assert np.asarray(m["trans"]).shape == (TRANSLATION_DIM,), (
                f"{name}: MANO trans shape {np.asarray(m['trans']).shape} != (3,)"
            )
            for key in ("pose", "shape", "trans"):
                arr = np.asarray(m[key], dtype=np.float64)
                assert np.isfinite(arr).all(), f"{name}: MANO {key} has non-finite values"


def test_criterion_keypoint_confidence(run):
    """C2: median 2D keypoint confidence >= 0.20."""
    med = run["per_criterion"]["keypoint_conf_median"]
    assert med >= MIN_KEYPOINT_CONF_MEDIAN, (
        f"median keypoint confidence {med:.4f} < {MIN_KEYPOINT_CONF_MEDIAN}"
    )
    assert np.isfinite(med)


def test_criterion_reprojection_error(run):
    """C3: mean MANO-joint reprojection error < 15 px."""
    err = run["per_criterion"]["mean_reprojection_error_px"]
    assert np.isfinite(err)
    assert err < MAX_MEAN_REPROJ_ERROR_PX, (
        f"mean reprojection error {err:.2f} px >= {MAX_MEAN_REPROJ_ERROR_PX} px"
    )


def test_criterion_left_right_detection_rate(run):
    """C4: left and right hand each detected in >= 80% of the 30 frames."""
    for side in ("left", "right"):
        rate = run["per_criterion"][f"{side}_detection_rate"]
        assert 0.0 <= rate <= 1.0
        assert rate >= MIN_HAND_DETECTION_RATE, (
            f"{side} hand detected in only {rate:.2%} of frames "
            f"(< {MIN_HAND_DETECTION_RATE:.0%})"
        )


def test_criterion_handedness_stability(run):
    """C5: at most 2 frames disagree with the modal x-order handedness assignment."""
    flips = run["per_criterion"]["handedness_flips"]
    assert isinstance(flips, int)
    assert flips <= MAX_HANDEDNESS_FLIPS, (
        f"{flips} handedness flips > {MAX_HANDEDNESS_FLIPS}"
    )


def test_criterion_mps_is_the_dominant_device(run):
    """C6: inference ran on mps; cpu fallback ops enumerated (may be non-empty)."""
    import torch

    assert torch.backends.mps.is_available(), "MPS not available on this machine"
    assert run["device"]["primary"] == REQUIRED_DEVICE, (
        f"primary inference device {run['device']['primary']} != {REQUIRED_DEVICE}"
    )
    assert {d.split(":")[0] for d in run["device"]["param_devices"]} == {REQUIRED_DEVICE}, (
        f"model params not all on {REQUIRED_DEVICE}: "
        f"{sorted(set(run['device']['param_devices']))}"
    )
    assert run["device"]["mps_allocated_bytes"] >= 1, (
        "no MPS memory allocated -- nothing ran on MPS"
    )
    assert isinstance(run["cpu_fallback_ops"], list), "cpu_fallback_ops must be a list"
    assert run["mps_evidence"]["mps_infer_sec"] > 0.0
    assert run["mps_evidence"]["cpu_infer_sec"] > 0.0, "CPU reference timing missing"


def test_criterion_evidence_and_report(run):
    """Two overlay figures + a complete report.json must exist and agree."""
    for name in EVIDENCE_FRAMES:
        path = EVIDENCE_DIR / f"{name}_hamer_hand_overlay.png"
        assert path.is_file(), f"missing evidence overlay {path}"

    assert REPORT.is_file(), "report.json missing"
    rep = json.loads(REPORT.read_text())
    missing = REQUIRED_REPORT_KEYS - rep.keys()
    assert not missing, f"report.json missing keys: {sorted(missing)}"
    assert rep["verdict"] == "pass", f"verdict={rep['verdict']}"
    assert rep["per_criterion"]["keypoint_conf_median"] >= MIN_KEYPOINT_CONF_MEDIAN
    assert rep["per_criterion"]["mean_reprojection_error_px"] < MAX_MEAN_REPROJ_ERROR_PX
    assert rep["per_criterion"]["left_detection_rate"] >= MIN_HAND_DETECTION_RATE
    assert rep["per_criterion"]["right_detection_rate"] >= MIN_HAND_DETECTION_RATE
    assert rep["per_criterion"]["handedness_flips"] <= MAX_HANDEDNESS_FLIPS
    assert rep["device"]["primary"] == REQUIRED_DEVICE
    assert "cpu_fallback_ops" in rep
    assert rep["known_limitations"], "known limitations must be recorded"
    # the predecessor's blocked-state artifacts must survive as gating evidence
    assert (OUT_DIR / "BLOCKER.md").is_file(), "BLOCKER.md must be preserved"
    assert (OUT_DIR / "report.blocked.json").is_file(), (
        "the blocked report must be archived as report.blocked.json"
    )


def test_criterion_evidence_matches_this_run(run):
    """Anti-false-positive: the saved overlay must be built from this run's data.

    Re-draw the overlay from the joints returned by this run and require a
    pixel-exact match against the PNG on disk.
    """
    from PIL import Image

    from perception import hamer_smoke

    for name in EVIDENCE_FRAMES:
        path = EVIDENCE_DIR / f"{name}_hamer_hand_overlay.png"
        dest = EVIDENCE_DIR / f"_verify_{name}.png"
        hamer_smoke.draw_overlay(
            FRAMES / f"{name}.jpg",
            run["per_frame"][name]["hands"],
            dest,
        )
        try:
            a = np.asarray(Image.open(dest).convert("RGB"))
            b = np.asarray(Image.open(path).convert("RGB"))
            assert a.shape == b.shape and (a == b).all(), (
                f"{name}: on-disk evidence overlay does not match this run's joints"
            )
        finally:
            dest.unlink(missing_ok=True)


if __name__ == "__main__":  # acceptance-script entry point
    raise SystemExit(pytest.main([__file__, "-v"]))
