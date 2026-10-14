"""Smoke 3 (P0) acceptance: FoundPose coarse pose for the tissue box, no CAD asset.

The assertions recompute the geometry from the committed/evidence artifacts instead of
trusting the driver, so the test cannot pass on "the pipeline ran without error".

Run:  conda run -n foundpose-mps python -m pytest tests/test_smoke3_foundpose.py -v
      (numpy/opencv are available in any env; the FoundPose env is only needed to
      regenerate the artifacts.)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

try:  # cv2 is only needed for the mask cross-check.
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/p0/smoke3_foundpose"
EVIDENCE = OUT / "evidence"
REPORT = OUT / "report.json"
OBJ = ROOT / "perception/assets/tissue_box_proxy.obj"

IMAGE_W, IMAGE_H = 1280, 720
CRIT_IOU_MIN = 0.50
CRIT_CENTER_DIST_MAX_PX = 64.0


def _load_report() -> dict:
    assert REPORT.is_file(), f"missing {REPORT} (driver not run)"
    return json.loads(REPORT.read_text())


def _obj_vertices_m() -> np.ndarray:
    verts = []
    for line in OBJ.read_text().splitlines():
        if line.strip().startswith("v "):
            verts.append([float(x) for x in line.split()[1:4]])
    v = np.asarray(verts, np.float64)
    assert v.shape == (8, 3), f"proxy box must have 8 vertices, got {v.shape}"
    return v


def _project(T: np.ndarray, pts_m: np.ndarray, K: np.ndarray) -> np.ndarray:
    pts_cam = (T[:3, :3] @ (pts_m * 1000.0).T + T[:3, 3:4]).T
    assert np.all(pts_cam[:, 2] > 0), f"object behind the camera: z={pts_cam[:, 2]}"
    uv = (K @ pts_cam.T).T
    return uv[:, :2] / uv[:, 2:3]


def test_report_exists_with_required_fields() -> None:
    r = _load_report()
    for key in (
        "smoke",
        "verdict",
        "device",
        "key_lib_versions",
        "weights_version",
        "template_source",
        "template_count",
        "criteria",
        "errors",
        "T_camera_object",
    ):
        assert key in r, f"report.json missing field '{key}'"
    assert r["smoke"] == "smoke3_foundpose"
    assert r["verdict"] == "pass", f"verdict={r['verdict']}; criteria={r['criteria']}"
    assert r["template_source"] == "handmade_proxy_box"
    assert r["template_count"] > 0, "FoundPose produced no reference templates"


def test_no_cad_mesh_was_used() -> None:
    """The templates must come from the handmade proxy, not from any CAD asset."""
    r = _load_report()
    assert "proxy" in r["template_source_detail"].lower()
    assert "handmade" in r["template_source_detail"].lower()
    assert r["camera_assumed"]["scale_source"] == "handmade_proxy_box"


def test_inference_ran_on_mps() -> None:
    r = _load_report()
    ev = r["device_evidence"]
    assert ev["mps_available"] is True, "torch reports no MPS backend"
    assert r["device"] == "mps", f"FoundPose inference device is {r['device']}, expected mps"
    assert r["criteria"]["C6_inference_on_mps"]["pass"] is True
    assert "torch" in r["key_lib_versions"]


def test_transform_is_finite_rigid_4x4() -> None:
    r = _load_report()
    T = np.asarray(r["T_camera_object"], np.float64)
    assert T.shape == (4, 4), f"T_camera_object must be 4x4, got {T.shape}"
    assert np.all(np.isfinite(T)), "T_camera_object contains non-finite entries"
    assert np.allclose(T[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6), f"last row {T[3]}"
    assert abs(np.linalg.det(T[:3, :3]) - 1.0) < 1e-3, f"det(R)={np.linalg.det(T[:3, :3])}"

    # The pose written next to the report must match the one inside it.
    on_disk = np.loadtxt(OUT / "T_camera_object.txt")
    assert np.allclose(on_disk, T, atol=1e-6), "report pose != T_camera_object.txt"


def test_criteria_all_pass() -> None:
    r = _load_report()
    expected = {
        "C1_T_finite_rigid_4x4",
        "C2_template_count_gt_0",
        "C3_corners_inside_image",
        "C4_silhouette_iou",
        "C5_bbox_center_dist_px",
        "C6_inference_on_mps",
    }
    assert set(r["criteria"]) == expected, f"criteria set changed: {set(r['criteria'])}"
    for name, c in r["criteria"].items():
        assert c["pass"] is True, f"{name} failed: value={c['value']} threshold={c['threshold']}"


def test_reprojected_corners_inside_image() -> None:
    """Recomputes the 8 corner projections from the proxy OBJ and the assumed camera."""
    assert cv2 is not None, "opencv is required to re-check the mask"
    r = _load_report()
    T = np.asarray(r["T_camera_object"], np.float64)
    K = np.asarray(r["camera_assumed"]["K"], np.float64)
    uv = _project(T, _obj_vertices_m(), K)

    assert np.all(uv[:, 0] >= 0) and np.all(uv[:, 0] < IMAGE_W), f"x out of image: {uv[:, 0]}"
    assert np.all(uv[:, 1] >= 0) and np.all(uv[:, 1] < IMAGE_H), f"y out of image: {uv[:, 1]}"

    # Must agree with the evidence file written by the driver.
    stored = np.asarray(
        json.loads((EVIDENCE / "projected_corners_px.json").read_text()), np.float64
    )
    assert np.allclose(stored[:, :2], uv, atol=1e-3), "stored corners != recomputed corners"


def test_proxy_silhouette_aligns_with_object_mask() -> None:
    """IoU between the projected proxy box (2D convex hull) and the GrabCut mask."""
    assert cv2 is not None, "opencv is required to re-check the mask"
    r = _load_report()
    T = np.asarray(r["T_camera_object"], np.float64)
    K = np.asarray(r["camera_assumed"]["K"], np.float64)
    uv = _project(T, _obj_vertices_m(), K)

    mask = (cv2.imread(str(EVIDENCE / "mask_f0001.png"), cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
    assert mask.shape == (IMAGE_H, IMAGE_W), f"mask shape {mask.shape}"
    assert mask.sum() > 0, "object mask is empty"

    hull = np.round(cv2.convexHull(uv.astype(np.float32))).astype(np.int32)
    proj = np.zeros((IMAGE_H, IMAGE_W), np.uint8)
    cv2.fillConvexPoly(proj, hull, 1)

    inter = np.logical_and(proj, mask).sum()
    union = np.logical_or(proj, mask).sum()
    iou = float(inter) / float(max(1, union))
    assert iou >= CRIT_IOU_MIN, f"proxy/mask silhouette IoU {iou:.3f} < {CRIT_IOU_MIN}"

    ys, xs = np.nonzero(mask)
    mask_bbox = (xs.min(), ys.min(), xs.max(), ys.max())
    proj_bbox = (uv[:, 0].min(), uv[:, 1].min(), uv[:, 0].max(), uv[:, 1].max())
    dist = float(
        np.hypot(
            0.5 * (proj_bbox[0] + proj_bbox[2]) - 0.5 * (mask_bbox[0] + mask_bbox[2]),
            0.5 * (proj_bbox[1] + proj_bbox[3]) - 0.5 * (mask_bbox[1] + mask_bbox[3]),
        )
    )
    assert dist <= CRIT_CENTER_DIST_MAX_PX, f"bbox centre distance {dist:.1f} px > {CRIT_CENTER_DIST_MAX_PX}"


def test_evidence_artifacts_exist() -> None:
    for name in (
        "overlay_f0001_foundpose_proxy.png",
        "mask_f0001.png",
        "T_camera_object.json",
        "projected_corners_px.json",
        "proxy_silhouette.png",
    ):
        p = EVIDENCE / name
        assert p.is_file() and p.stat().st_size > 0, f"missing evidence {p}"


def test_cpu_components_are_recorded() -> None:
    """FoundPose ships CPU-only sub-modules; the report must say so explicitly."""
    r = _load_report()
    fb = r["cpu_fallbacks"]
    assert isinstance(fb, list) and fb, "cpu_fallbacks must not be empty"
    joined = " ".join(fb).lower()
    assert "faiss" in joined and "cpu" in joined
    assert "pnp" in joined or "opencv" in joined
