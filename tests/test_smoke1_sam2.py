"""Smoke 1 acceptance: SAM2 (SAM 2.1 hiera-small) on Apple Silicon MPS.

Criteria are HARD-CODED here. They must never be relaxed; a genuinely failing
criterion yields verdict=fail in the report (this is a research bring-up).

Run with the smoke venv interpreter:
    cd /Users/guohongbin/v2d-mac && runs/.venv-sam2/bin/python -m pytest \
        tests/test_smoke1_sam2.py -v
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
OUT_DIR = REPO / "runs/p0/smoke1_sam2"
EVIDENCE_DIR = OUT_DIR / "evidence"
REPORT = OUT_DIR / "report.json"

# ---------------------------------------------------------------- criteria --
TEST_FRAMES = ["f_0001", "f_0010", "f_0020"]
MIN_IOU_MPS_CPU = 0.8          # DESIGN.md L7 criterion 1
MIN_MASK_AREA_FRAC = 0.005     # non-empty: >=0.5% of frame
MAX_MASK_AREA_FRAC = 0.90      # anti-degenerate: must not swallow the frame
MIN_BOX_COVERAGE = 0.10        # mask must actually cover >=10% of prompt box
MIN_MPS_ALLOC_BYTES = 1        # proof the graph really lived on MPS
REQUIRED_REPORT_KEYS = {
    "device",
    "key_lib_versions",
    "weights_version",
    "per_criterion",
    "verdict",
}


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union) if union else 0.0


@pytest.fixture(scope="module")
def run():
    from perception import sam2_smoke

    return sam2_smoke.run_smoke(frames=TEST_FRAMES, out_dir=OUT_DIR, evidence_dir=EVIDENCE_DIR)


def test_criterion_masks_non_empty(run):
    """Each frame must yield a non-degenerate binary mask on MPS."""
    for name, r in run["per_frame"].items():
        frac = r["mps"]["area_frac"]
        assert frac >= MIN_MASK_AREA_FRAC, f"{name}: mask empty (area_frac={frac:.5f})"
        assert frac <= MAX_MASK_AREA_FRAC, (
            f"{name}: mask degenerate, covers {frac:.3f} of frame (>={MAX_MASK_AREA_FRAC})"
        )


def test_criterion_mask_covers_prompt_box(run):
    """Mask must intersect the prompt box -- guards against off-target masks."""
    for name, r in run["per_frame"].items():
        cov = r["mps"]["box_coverage"]
        assert cov >= MIN_BOX_COVERAGE, (
            f"{name}: mask covers only {cov:.3f} of prompt box (<{MIN_BOX_COVERAGE})"
        )


def test_criterion_mps_vs_cpu_iou(run):
    """Same checkpoint + same prompt on MPS vs real CPU inference: IoU > 0.8."""
    for name, r in run["per_frame"].items():
        iou = r["iou_mps_cpu"]
        assert r["cpu"]["ran_inference"] is True, f"{name}: CPU path did not run inference"
        assert iou > MIN_IOU_MPS_CPU, (
            f"{name}: MPS/CPU mask IoU={iou:.4f} <= {MIN_IOU_MPS_CPU}"
        )


def test_criterion_mps_really_used(run):
    """MPS graph residency proof + params live on the mps device."""
    import torch

    assert run["device"] == "mps"
    assert torch.backends.mps.is_available(), "MPS not available on this machine"
    assert run["mps_peak_allocated_bytes"] >= MIN_MPS_ALLOC_BYTES, (
        "no MPS memory allocated during inference -- inference did not run on MPS"
    )
    assert set(run["param_devices"]) == {"mps"}, (
        f"model params not all on mps: {sorted(set(run['param_devices']))}"
    )
    assert run["mps_infer_sec"] > 0.0 and run["cpu_infer_sec"] > 0.0


def test_criterion_masks_are_frame_specific(run):
    """Anti-degenerate: three different frames must not collapse to one mask."""
    names = list(run["per_frame"].keys())
    assert len(names) == len(TEST_FRAMES)
    ious = [
        _iou(run["per_frame"][a]["mps"]["mask"], run["per_frame"][b]["mps"]["mask"])
        for i, a in enumerate(names)
        for b in names[i + 1 :]
    ]
    assert max(ious) < 0.995, f"masks identical across frames (IoU {ious}) -> constant mask"


def test_criterion_evidence_matches_reported_mask(run):
    """Anti-false-positive: the overlay must be built from the reported mask.

    A green IoU + non-empty result is worthless if the saved figure was made
    from some other run. We re-render the overlay from the mask returned by
    this run and require a *pixel-exact* match against the file on disk
    (evidence is PNG, so no encoder tolerance is needed).
    """
    from PIL import Image

    from perception import sam2_smoke

    for name in TEST_FRAMES:
        r = run["per_frame"][name]
        dest = EVIDENCE_DIR / f"_verify_{name}.png"
        sam2_smoke._write_overlay(
            FRAMES / f"{name}.jpg", r["mps"]["mask"], np.array(r["prompt_box_xyxy"]), dest
        )
        try:
            a = np.asarray(Image.open(dest).convert("RGB"))
            b = np.asarray(
                Image.open(EVIDENCE_DIR / f"{name}_mps_mask_overlay.png").convert("RGB")
            )
            assert a.shape == b.shape and (a == b).all(), (
                f"{name}: on-disk overlay does not match this run's mask"
            )
        finally:
            dest.unlink(missing_ok=True)


def test_criterion_evidence_and_report(run):
    """Overlay figures for all frames + a complete report.json must exist."""
    for name in TEST_FRAMES:
        assert (EVIDENCE_DIR / f"{name}_mps_mask_overlay.png").is_file(), (
            f"missing overlay evidence for {name}"
        )
        assert (EVIDENCE_DIR / f"{name}_mps_cpu_diff.png").is_file(), (
            f"missing MPS-vs-CPU diff evidence for {name}"
        )
    assert REPORT.is_file(), "report.json missing"
    rep = json.loads(REPORT.read_text())
    missing = REQUIRED_REPORT_KEYS - rep.keys()
    assert not missing, f"report.json missing keys: {sorted(missing)}"
    assert rep["verdict"] == "pass", f"verdict={rep['verdict']}"
    assert rep["per_criterion"]["min_iou_mps_cpu"] > MIN_IOU_MPS_CPU
    assert rep["per_criterion"]["min_mask_area_frac"] >= MIN_MASK_AREA_FRAC
    assert rep["per_criterion"]["min_box_coverage"] >= MIN_BOX_COVERAGE
    # The object-hit diagnostic must be present for every frame so a mask that
    # landed on the background cannot pass silently.
    for name in TEST_FRAMES:
        assert "object_hit" in rep["per_frame"][name]["mps"], name
        assert "median_mask_hue_deg" in rep["per_frame"][name]["mps"]["object_hit"], name
    assert rep["known_limitations"], "known limitations must be recorded"
