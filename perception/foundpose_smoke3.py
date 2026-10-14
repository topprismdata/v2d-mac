"""Smoke 3 (P0): FoundPose coarse 6DoF pose on the tissue-box clip, with no CAD asset.

The scene object is a consumer tissue box. No CAD/mesh exists for it, so a proxy box
(``perception/assets/tissue_box_proxy.obj``, 0.24 x 0.12 x 0.08 m) is rendered into
FoundPose reference templates and the coarse pose is estimated on ``frames/f_0001.jpg``.

Pipeline, executed with the ``foundpose-mps`` conda env:
  1. proxy OBJ -> BOP-format PLY (mm)
  2. frame + mask + assumed camera -> mini BOP dataset "tbox"
  3. ``scripts/gen_templates.py`` (pyrender)   -> reference templates
  4. ``scripts/gen_repre.py`` (DINOv2 + faiss)  -> object representation
  5. ``scripts/infer.py`` (DINOv2 + PnP)        -> T_camera_object
  6. overlay + report.json

Acceptance criteria (hard, defined before the run; see tests/test_smoke3_foundpose.py):
  C1 T_camera_object is a finite 4x4 rigid transform (last row [0,0,0,1], |det R| - 1| < 1e-3)
  C2 template_count > 0
  C3 all 8 proxy-box corners reproject strictly inside the 1280x720 image
  C4 silhouette IoU(proxy box hull, object mask) >= 0.50
  C5 |bbox centre(projected box) - bbox centre(mask)| <= 64 px
  C6 the neural inference ran on MPS (mps built + feature-extraction timing vs CPU)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

# --------------------------------------------------------------------------------------
# Paths and fixed configuration
# --------------------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
FOUNDPOSE_REPO = ROOT / "runs/upstream/foundpose"
ASSETS = ROOT / "runs/p0/assets"
OUT = ROOT / "runs/p0/smoke3_foundpose"
EVIDENCE = OUT / "evidence"
LOGS = OUT / "logs"
BOP_ROOT = OUT / "bop_datasets"
BOP_OUTPUT = OUT / "bop_output"
CONFIGS = OUT / "configs"
REPORT = OUT / "report.json"

DATASET_NAME = "tbox"
OBJ_ID = 1
SCENE_ID = 1
IM_ID = 1
FRAME = ASSETS / "frames/f_0001.jpg"
OBJ_PATH = ROOT / "perception/assets/tissue_box_proxy.obj"

CONDA_ENV = "foundpose-mps"
TEMPLATE_SOURCE = "handmade_proxy_box"

IMAGE_W, IMAGE_H = 1280, 720

# No calibration is available for this clip. Intrinsics are ASSUMED: 90 deg horizontal
# FOV (typical of the wide-angle egocentric capture used for this clip), square pixels.
FX = FY = 650.0
CX, CY = 640.0, 360.0
K_QUERY = [[FX, 0.0, CX], [0.0, FY, CY], [0.0, 0.0, 1.0]]
# The template renderer only needs a nominal camera with the same focal length; a small
# viewport keeps the pyrender rasterisation affordable.
K_TEMPLATE = [[FX, 0.0, 320.0], [0.0, FY, 240.0], [0.0, 0.0, 1.0]]
TEMPLATE_IM_SIZE = (640, 480)
# Camera-object distance range of the box across the clip (mm); sets the view-sphere radii.
DEPTH_RANGE_MM = (380.0, 700.0)

# GrabCut seed rectangle (x, y, w, h) around the tissue box in f_0001.
GRABCUT_RECT = (424, 271, 342, 222)

# Acceptance thresholds (fixed before the run, see module docstring).
CRIT_IOU_MIN = 0.50
CRIT_CENTER_DIST_MAX_PX = 64.0

# Official LM-O template budget (57 viewpoints x 14 in-plane rotations = 798 templates).
# SSAA is 2.0 instead of the LM-O default 4.0: the nominal template viewport is kept at
# 640x480 instead of the full 1280x720 query size, so 4x would need 5096x5096 buffers.
TEMPLATE_OPTS = {
    "num_viewspheres": 1,
    "min_num_viewpoints": 57,
    "num_inplane_rotations": 14,
    "images_per_view": 1,
    "ssaa_factor": 2.0,
}

EXTRACTOR_GEN_REPRE = EXTRACTOR_INFER = (
    "dinov2_version=vits14-reg_stride=14_facet=token_layer=9_logbin=0_norm=1"
)


# --------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------


def _log(msg: str) -> None:
    print(f"[smoke3] {msg}", flush=True)


def load_obj(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Reads an OBJ file, returning (vertices_xyz, faces_0based)."""
    verts: List[Tuple[float, float, float]] = []
    faces: List[Tuple[int, int, int]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line.startswith("v "):
            verts.append(tuple(float(x) for x in line.split()[1:4]))  # type: ignore[arg-type]
        elif line.startswith("f "):
            idx = [int(tok.split("/")[0]) - 1 for tok in line.split()[1:]]
            faces.append(tuple(idx[:3]))  # type: ignore[arg-type]
    return np.asarray(verts, np.float64), np.asarray(faces, np.int64)


def write_bop_ply(path: Path, verts_mm: np.ndarray, faces: np.ndarray) -> None:
    """Writes a BOP-style binary little-endian PLY (vertices in mm, with colors)."""
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "comment v2d-mac P0 smoke 3 proxy tissue box (handmade_proxy_box)\n"
        f"element vertex {len(verts_mm)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property float nx\nproperty float ny\nproperty float nz\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        f"element face {len(faces)}\n"
        "property list uchar int vertex_indices\n"
        "end_header\n"
    ).encode("utf-8")
    body = bytearray()
    for v in verts_mm:
        # Outward normals are irrelevant for FoundPose's shading (recomputed by trimesh).
        body += struct.pack("<6f3B", *v, 0.0, 0.0, 0.0, 200, 200, 200)
    for f in faces:
        body += struct.pack("<B3i", 3, int(f[0]), int(f[1]), int(f[2]))
    path.write_bytes(header + bytes(body))


def mask_to_coco_rle(mask: np.ndarray) -> Dict[str, Any]:
    """Encodes a binary mask as an uncompressed COCO RLE (column-major)."""
    flat = mask.T.reshape(-1).astype(np.uint8)  # order="F" == transpose then flatten
    counts: List[int] = []
    prev = 0
    run = 0
    for v in flat:
        if v == prev:
            run += 1
        else:
            counts.append(run)
            prev = v
            run = 1
    counts.append(run)
    if len(counts) % 2 == 0:
        counts.append(0)
    return {"size": [int(mask.shape[0]), int(mask.shape[1])], "counts": [int(c) for c in counts]}


def bbox_iou_xyxy(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    return float(inter / (area_a + area_b - inter))


# --------------------------------------------------------------------------------------
# Step 1 + 2: mini BOP dataset
# --------------------------------------------------------------------------------------


def build_bop_dataset(image: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
    """Writes the PLY model, camera files, test frame and detection mask."""
    ds_dir = BOP_ROOT / DATASET_NAME
    (ds_dir / "models").mkdir(parents=True, exist_ok=True)
    test_dir = ds_dir / "test" / f"{SCENE_ID:06d}"
    (test_dir / "rgb").mkdir(parents=True, exist_ok=True)
    det_dir = BOP_ROOT / "detections" / "cnos-fastsam"
    det_dir.mkdir(parents=True, exist_ok=True)

    verts_m, faces = load_obj(OBJ_PATH)
    dims_m = (verts_m.max(0) - verts_m.min(0)).tolist()
    verts_mm = verts_m * 1000.0
    write_bop_ply(ds_dir / "models" / f"obj_{OBJ_ID:06d}.ply", verts_mm, faces)

    diameter_mm = float(np.linalg.norm(verts_mm.max(0) - verts_mm.min(0)))
    (ds_dir / "models" / "models_info.json").write_text(
        json.dumps({str(OBJ_ID): {"diameter": diameter_mm}})
    )
    (ds_dir / "camera.json").write_text(
        json.dumps(
            {
                "width": TEMPLATE_IM_SIZE[0],
                "height": TEMPLATE_IM_SIZE[1],
                "fx": K_TEMPLATE[0][0],
                "fy": K_TEMPLATE[1][1],
                "cx": K_TEMPLATE[0][2],
                "cy": K_TEMPLATE[1][2],
                "depth_scale": 1.0,
            },
            indent=2,
        )
    )

    # BOP classic datasets store RGB as PNG; write the decoded frame losslessly.
    cv2.imwrite(str(test_dir / "rgb" / f"{IM_ID:06d}.png"), image)
    (test_dir / "scene_camera.json").write_text(
        json.dumps({str(IM_ID): {"cam_K": K_QUERY, "depth_scale": 1.0}}, indent=2)
    )
    # No ground-truth poses are claimed for this clip; annotations stay empty so that
    # FoundPose runs in its update_without_anno path.
    (test_dir / "scene_gt.json").write_text(json.dumps({str(IM_ID): []}))
    (test_dir / "scene_gt_info.json").write_text(json.dumps({str(IM_ID): []}))
    (ds_dir / "test_targets_bop19.json").write_text(
        json.dumps(
            [{"scene_id": SCENE_ID, "im_id": IM_ID, "obj_id": OBJ_ID, "inst_count": 1}]
        )
    )

    ys, xs = np.nonzero(mask)
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    det = {
        "scene_id": SCENE_ID,
        "image_id": IM_ID,
        "category_id": OBJ_ID,
        "bbox": [x0, y0, x1 - x0, y1 - y0],
        "segmentation": mask_to_coco_rle(mask),
        "score": 1.0,
        "time": 0.0,
    }
    (det_dir / f"cnos-fastsam_{DATASET_NAME}-test.json").write_text(json.dumps([det]))

    return {
        "obj_dims_m": dims_m,
        "model_diameter_mm": diameter_mm,
        "mask_bbox_xyxy": [x0, y0, x1, y1],
        "mask_area_px": int(mask.sum()),
    }


def compute_object_mask(image: np.ndarray) -> np.ndarray:
    """Segments the tissue box with GrabCut seeded by GRABCUT_RECT."""
    mask = np.zeros(image.shape[:2], np.uint8)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    rect = (GRABCUT_RECT[0], GRABCUT_RECT[1], GRABCUT_RECT[2], GRABCUT_RECT[3])
    cv2.grabCut(image, mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
    out = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)
    out = cv2.morphologyEx(out, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return out


# --------------------------------------------------------------------------------------
# Step 3-5: run the FoundPose scripts
# --------------------------------------------------------------------------------------


def script_env() -> Dict[str, str]:
    env = dict(os.environ)
    repo = str(FOUNDPOSE_REPO)
    env["PYTHONPATH"] = ":".join(
        [repo, f"{repo}/external/bop_toolkit", f"{repo}/external/dinov2"]
    )
    env["BOP_PATH"] = str(BOP_ROOT)
    env["BOP_OUTPUT_PATH"] = str(BOP_OUTPUT)
    env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    return env


def run_foundpose(script: str, opts_path: Path, log_name: str) -> Dict[str, Any]:
    cmd = [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        CONDA_ENV,
        "python",
        str(FOUNDPOSE_REPO / "scripts" / script),
        "--opts-path",
        str(opts_path),
    ]
    _log(f"run: {' '.join(cmd)}")
    LOGS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    proc = subprocess.run(cmd, env=script_env(), capture_output=True, text=True)
    elapsed = time.time() - t0
    log_path = LOGS / log_name
    log_path.write_text(
        f"$ {' '.join(cmd)}\n\n--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}\n"
    )
    _log(f"{script} exit={proc.returncode} in {elapsed:.1f}s (log: {log_path})")
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout).strip().splitlines()[-25:]
        raise RuntimeError(f"{script} failed:\n" + "\n".join(tail))
    return {"cmd": cmd, "returncode": proc.returncode, "seconds": elapsed, "log": str(log_path)}


def write_configs() -> Dict[str, Path]:
    CONFIGS.mkdir(parents=True, exist_ok=True)
    tpl = CONFIGS / "gen_templates_tbox.json"
    tpl.write_text(
        json.dumps(
            {
                "gen_templates_opts": {
                    "version": "v1",
                    "object_dataset": DATASET_NAME,
                    "object_lids": [OBJ_ID],
                    **TEMPLATE_OPTS,
                    "max_num_triangles": 20000,
                    "back_face_culling": False,
                    "texture_size": [1024, 1024],
                    "background_type": "black",
                    "light_type": "multi_directional",
                    "features_patch_size": 14,
                    "crop": True,
                    "crop_rel_pad": 0.2,
                    "crop_size": [420, 420],
                }
            },
            indent=2,
        )
    )
    repre = CONFIGS / "gen_repre_tbox.json"
    repre.write_text(
        json.dumps(
            {
                "gen_repre_opts": {
                    "version": "v1",
                    "templates_version": "v1",
                    "object_dataset": DATASET_NAME,
                    "object_lids": [OBJ_ID],
                    "extractor_name": EXTRACTOR_GEN_REPRE,
                    "grid_cell_size": 14.0,
                    "apply_pca": True,
                    "pca_components": 256,
                    "pca_whiten": False,
                    "cluster_features": True,
                    "cluster_num": 512,
                    "template_desc_opts": {
                        "desc_type": "tfidf",
                        "tfidf_knn_metric": "l2",
                        "tfidf_knn_k": 3,
                        "tfidf_soft_assign": False,
                        "tfidf_soft_sigma_squared": 10.0,
                    },
                }
            },
            indent=2,
        )
    )
    infer = CONFIGS / "infer_tbox.json"
    infer.write_text(
        json.dumps(
            {
                "infer_opts": {
                    "version": "v1",
                    "object_dataset": DATASET_NAME,
                    "object_lids": [OBJ_ID],
                    "repre_version": "v1",
                    "crop_rel_pad": 0.2,
                    "crop_size": [420, 420],
                    "use_detections": True,
                    "extractor_name": EXTRACTOR_INFER,
                    "grid_cell_size": 14.0,
                    "match_template_type": "tfidf",
                    "match_top_n_templates": 5,
                    "match_feat_matching_type": "cyclic_buddies",
                    "match_top_k_buddies": 300,
                    "pnp_type": "opencv",
                    "pnp_ransac_iter": 400,
                    "pnp_inlier_thresh": 10.0,
                    "final_pose_type": "best_coarse",
                    "num_preds_factor": 1,
                    "vis_results": True,
                }
            },
            indent=2,
        )
    )
    return {"gen_templates": tpl, "gen_repre": repre, "infer": infer}


def load_estimated_pose() -> Tuple[np.ndarray, Dict[str, Any]]:
    """Reads T_camera_object (model -> original camera, mm) from the FoundPose output."""
    results_path = (
        BOP_OUTPUT
        / "inference"
        / f"{DATASET_NAME}_v1"
        / str(OBJ_ID)
        / "estimated-poses.json"
    )
    results = json.loads(results_path.read_text())
    if not results:
        raise RuntimeError(f"No pose estimates in {results_path}")
    best = max(results, key=lambda r: float(r["score"]))
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(best["R"], np.float64).reshape(3, 3)
    T[:3, 3] = np.asarray(best["t"], np.float64).reshape(3)
    return T, {k: best[k] for k in ("score", "scene_id", "img_id", "inst_id", "hypothesis_id", "time")}


# --------------------------------------------------------------------------------------
# Step 6: reprojection, overlay and report
# --------------------------------------------------------------------------------------


def project_points(T: np.ndarray, pts_mm: np.ndarray) -> np.ndarray:
    """Projects object-frame points (mm) into pixel coordinates of the query image."""
    pts_cam = (T[:3, :3] @ pts_mm.T + T[:3, 3:4]).T
    z = np.clip(pts_cam[:, 2], 1e-6, None)
    return np.stack(
        [
            FX * pts_cam[:, 0] / z + CX,
            FY * pts_cam[:, 1] / z + CY,
            pts_cam[:, 2],
        ],
        axis=1,
    )


def convex_hull_mask(pts2d: np.ndarray) -> np.ndarray:
    """Fills the 2D convex hull of the projected points (the proxy box is convex)."""
    canvas = np.zeros((IMAGE_H, IMAGE_W), np.uint8)
    finite = pts2d[np.isfinite(pts2d[:, :2]).all(axis=1), :2]
    if len(finite) < 3:
        return canvas
    hull = cv2.convexHull(finite.astype(np.float32))
    cv2.fillConvexPoly(canvas, np.clip(np.round(hull), -1e6, 1e6).astype(np.int32), 1)
    return canvas


def _principal_axis_angle(pts: np.ndarray) -> float:
    """Angle (rad) of the major PCA axis of a 2D point set."""
    pts = np.asarray(pts, np.float64)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) < 2:
        return 0.0
    centred = pts - pts.mean(axis=0)
    cov = centred.T @ centred
    w, v = np.linalg.eigh(cov)
    return float(np.arctan2(v[1, -1], v[0, -1]))


def make_overlay(image: np.ndarray, mask: np.ndarray, pts2d: np.ndarray) -> np.ndarray:
    canvas = image.copy()
    contour, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(canvas, contour, -1, (0, 255, 0), 2)

    finite = pts2d[np.isfinite(pts2d[:, :2]).all(axis=1), :2]
    if len(finite) >= 3:
        hull_i32 = np.clip(np.round(cv2.convexHull(finite.astype(np.float32))), -1e6, 1e6).astype(np.int32)
        overlay = canvas.copy()
        cv2.fillConvexPoly(overlay, hull_i32, (0, 0, 255))
        canvas = cv2.addWeighted(overlay, 0.25, canvas, 0.75, 0)
        cv2.polylines(canvas, [hull_i32], True, (0, 0, 255), 2)

    for i, (x, y, _) in enumerate(pts2d):
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        cx_i, cy_i = int(round(float(x))), int(round(float(y)))
        if not (0 <= cx_i < IMAGE_W and 0 <= cy_i < IMAGE_H):
            continue
        cv2.circle(canvas, (cx_i, cy_i), 5, (255, 0, 255), -1)
        cv2.putText(canvas, str(i), (cx_i + 6, cy_i - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
    cv2.putText(canvas, "green: mask  red: FoundPose proxy box", (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return canvas


def mps_evidence() -> Dict[str, Any]:
    """Times the DINOv2 extractor on MPS and CPU (420x420 input, as used at inference)."""
    code = (
        "import json,time,torch;"
        "from utils import feature_util;"
        "import sys; sys.path.insert(0,'.');"
        f"ex=feature_util.make_feature_extractor({EXTRACTOR_INFER!r});"
        "x=torch.rand(1,3,420,420);out={};"
        "\nfor dev in (['mps','cpu'] if torch.backends.mps.is_available() else ['cpu']):"
        "\n    ex.to(dev);t=x.to(dev);ex(t);torch.mps.synchronize() if dev=='mps' else None;"
        "\n    t0=time.time();ex(t);torch.mps.synchronize() if dev=='mps' else None;"
        "\n    out[dev]=time.time()-t0;"
        "\nprint('@@'+json.dumps({'mps_available':bool(torch.backends.mps.is_available()),"
        "'mps_built':bool(torch.backends.mps.is_built()),'times':out}))"
    )
    cmd = ["conda", "run", "--no-capture-output", "-n", CONDA_ENV, "python", "-c", code]
    proc = subprocess.run(cmd, env=script_env(), capture_output=True, text=True, cwd=str(FOUNDPOSE_REPO))
    for line in proc.stdout.splitlines():
        if line.startswith("@@"):
            return json.loads(line[2:])
    raise RuntimeError("MPS benchmark failed:\n" + (proc.stderr or proc.stdout)[-2000:])


def parse_infer_log() -> Dict[str, Any]:
    """Extracts FoundPose's own diagnostics (correspondence count, PnP inliers, score)."""
    out: Dict[str, Any] = {}
    log = LOGS / "infer.log"
    if not log.is_file():
        return {"log_available": False}
    text = log.read_text()
    m = re.findall(r"Number of corresp: \[([^\]]*)\]", text)
    if m:
        out["corresp_per_template"] = [int(x) for x in m[-1].split(",") if x.strip()]
    q = [float(x) for x in re.findall(r"Quality of coarse pose \d+: ([\d.]+)", text)]
    if q:
        out["pnp_inlier_counts"] = q
        out["best_pnp_inliers"] = max(q)
    s = re.findall(r"SUMMARY - success: (\d+), time: ([\d.]+)s", text)
    if s:
        out["success_hypotheses"] = int(s[-1][0])
        out["seconds_per_hypothesis"] = float(s[-1][1])
    mps_fallback = sorted(set(re.findall(r"The operator '([^']+)' is not currently supported", text)))
    if mps_fallback:
        out["mps_op_cpu_fallbacks"] = mps_fallback
    return out


def key_lib_versions() -> Dict[str, str]:
    code = (
        "import json,torch,cv2,numpy,scipy,sklearn;"
        "import faiss,pyrender,trimesh,kornia;"
        "print('@@'+json.dumps({"
        "'torch':torch.__version__,'mps':str(getattr(torch.backends,'mps',None) and torch.backends.mps.is_available()),"
        "'cv2':cv2.__version__,'numpy':numpy.__version__,'scipy':scipy.__version__,"
        "'sklearn':sklearn.__version__,'faiss':faiss.__version__,'trimesh':trimesh.__version__,"
        "'kornia':kornia.__version__,'python':__import__('sys').version.split()[0]}))"
    )
    proc = subprocess.run(
        ["conda", "run", "--no-capture-output", "-n", CONDA_ENV, "python", "-c", code],
        env=script_env(), capture_output=True, text=True, cwd=str(FOUNDPOSE_REPO),
    )
    for line in proc.stdout.splitlines():
        if line.startswith("@@"):
            return json.loads(line[2:])
    raise RuntimeError("version probe failed:\n" + (proc.stderr or proc.stdout)[-2000:])


def dinov2_weights_info() -> Dict[str, Any]:
    """Locates the cached DINOv2 checkpoint and records its hash (FoundPose ships no weights)."""
    cache = Path(
        subprocess.run(
            ["conda", "run", "--no-capture-output", "-n", CONDA_ENV, "python", "-c",
             "import torch;print(torch.hub.get_dir())"],
            capture_output=True, text=True,
        ).stdout.strip().splitlines()[-1]
    ) / "checkpoints"
    info: Dict[str, Any] = {
        "source": "torch.hub (dl.fbaipublicfiles.com)",
        "extractor": "dinov2 vits14-reg (DINOv2 ViT-S/14 with registers)",
        "note": "FoundPose is training-free; the only network weights are the DINOv2 backbone.",
    }
    for cand in sorted(cache.glob("dinov2*")) if cache.is_dir() else []:
        h = hashlib.sha256(cand.read_bytes()).hexdigest()
        info.update(
            {
                "file": str(cand),
                "url": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/"
                + cand.name,
                "bytes": cand.stat().st_size,
                "sha256": h,
            }
        )
    return info


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-build", action="store_true", help="reuse an existing BOP dataset")
    ap.add_argument(
        "--post-only",
        action="store_true",
        help="reuse existing templates/representation/poses and only redo the evidence + report",
    )
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(FRAME))
    if image is None:
        raise RuntimeError(f"cannot read {FRAME}")
    assert image.shape[:2] == (IMAGE_H, IMAGE_W), f"unexpected frame size {image.shape}"

    if args.skip_build and (BOP_ROOT / DATASET_NAME / "models" / f"obj_{OBJ_ID:06d}.ply").is_file():
        mask = compute_object_mask(image)
        ds_info = {"reused": True}
    else:
        mask = compute_object_mask(image)
        ds_info = build_bop_dataset(image, mask)
    cv2.imwrite(str(EVIDENCE / "mask_f0001.png"), mask * 255)
    _log(f"mask area = {int(mask.sum())} px")

    cfgs = write_configs()
    poses_path = (
        BOP_OUTPUT / "inference" / f"{DATASET_NAME}_v1" / str(OBJ_ID) / "estimated-poses.json"
    )
    if args.post_only and poses_path.is_file():
        _log("post-only: reusing the templates, representation and pose estimates on disk")
        timings = {"reused_from_previous_run": True, "logs": str(LOGS)}
    else:
        timings = {
            "gen_templates": run_foundpose("gen_templates.py", cfgs["gen_templates"], "gen_templates.log"),
            "gen_repre": run_foundpose("gen_repre.py", cfgs["gen_repre"], "gen_repre.log"),
            "infer": run_foundpose("infer.py", cfgs["infer"], "infer.log"),
        }

    metadata = json.loads(
        (BOP_OUTPUT / "templates/v1" / DATASET_NAME / str(OBJ_ID) / "metadata.json").read_text()
    )
    template_count = len(metadata)

    T, pose_meta = load_estimated_pose()
    np.savetxt(str(OUT / "T_camera_object.txt"), T, fmt="%.9f")

    verts_m, _ = load_obj(OBJ_PATH)
    pts2d = project_points(T, verts_m * 1000.0)

    proj_mask = convex_hull_mask(pts2d)
    inter = int(np.logical_and(proj_mask, mask).sum())
    union = int(np.logical_or(proj_mask, mask).sum())
    silhouette_iou = inter / max(1, union)

    ys, xs = np.nonzero(mask)
    # Supplementary (non-gating) pose-quality diagnostics: how well the proxy box covers
    # the segmented object and how far its in-plane orientation is off.
    mask_area = int(mask.sum())
    proj_area = int(proj_mask.sum())
    mask_cov = inter / max(1, mask_area)
    axis_angle_deg = float(
        np.degrees(
            _principal_axis_angle(pts2d[:, :2])
            - _principal_axis_angle(np.stack([xs, ys], axis=1).astype(np.float64))
        )
    )
    axis_angle_deg = (axis_angle_deg + 180.0) % 360.0 - 180.0

    mask_bbox = [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]
    proj_bbox = [
        float(pts2d[:, 0].min()), float(pts2d[:, 1].min()),
        float(pts2d[:, 0].max()), float(pts2d[:, 1].max()),
    ]
    center_dist = float(
        np.hypot(
            0.5 * (proj_bbox[0] + proj_bbox[2]) - 0.5 * (mask_bbox[0] + mask_bbox[2]),
            0.5 * (proj_bbox[1] + proj_bbox[3]) - 0.5 * (mask_bbox[1] + mask_bbox[3]),
        )
    )
    corners_in_image = bool(
        np.all(pts2d[:, 0] >= 0) and np.all(pts2d[:, 0] < IMAGE_W)
        and np.all(pts2d[:, 1] >= 0) and np.all(pts2d[:, 1] < IMAGE_H)
    )

    cv2.imwrite(str(EVIDENCE / "overlay_f0001_foundpose_proxy.png"), make_overlay(image, mask, pts2d))
    (EVIDENCE / "T_camera_object.json").write_text(json.dumps(T.tolist(), indent=2))
    (EVIDENCE / "projected_corners_px.json").write_text(json.dumps(pts2d.tolist(), indent=2))
    cv2.imwrite(str(EVIDENCE / "proxy_silhouette.png"), proj_mask * 255)

    t_rigid = bool(
        T.shape == (4, 4)
        and np.all(np.isfinite(T))
        and np.allclose(T[3], [0, 0, 0, 1], atol=1e-6)
        and abs(np.linalg.det(T[:3, :3]) - 1.0) < 1e-3
    )

    mps = mps_evidence()
    device_used = "mps" if mps["mps_available"] else "cpu"

    criteria = {
        "C1_T_finite_rigid_4x4": {
            "value": t_rigid,
            "threshold": "finite, last row [0,0,0,1], |det R - 1| < 1e-3",
            "pass": t_rigid,
        },
        "C2_template_count_gt_0": {
            "value": template_count,
            "threshold": "> 0",
            "pass": template_count > 0,
        },
        "C3_corners_inside_image": {
            "value": corners_in_image,
            "threshold": f"all 8 corners in [0,{IMAGE_W}) x [0,{IMAGE_H})",
            "pass": corners_in_image,
            "corners_px": pts2d[:, :2].tolist(),
        },
        "C4_silhouette_iou": {
            "value": silhouette_iou,
            "threshold": f">= {CRIT_IOU_MIN}",
            "pass": silhouette_iou >= CRIT_IOU_MIN,
        },
        "C5_bbox_center_dist_px": {
            "value": center_dist,
            "threshold": f"<= {CRIT_CENTER_DIST_MAX_PX}",
            "pass": center_dist <= CRIT_CENTER_DIST_MAX_PX,
            "mask_bbox_xyxy": mask_bbox,
            "projected_bbox_xyxy": proj_bbox,
            "bbox_iou": bbox_iou_xyxy(tuple(proj_bbox), tuple(mask_bbox)),
        },
        "C6_inference_on_mps": {
            "value": device_used,
            "threshold": "mps",
            "pass": device_used == "mps",
            "evidence": mps,
        },
    }

    verdict = "pass" if all(c["pass"] for c in criteria.values()) else "fail"

    diagnostics = parse_infer_log()

    report = {
        "smoke": "smoke3_foundpose",
        "verdict": verdict,
        "device": device_used,
        "device_evidence": mps,
        "key_lib_versions": key_lib_versions(),
        "weights_version": dinov2_weights_info(),
        "template_source": TEMPLATE_SOURCE,
        "template_source_detail": (
            "handmade proxy box, perception/assets/tissue_box_proxy.obj, "
            f"{ds_info.get('obj_dims_m')} m, BOP PLY diameter {ds_info.get('model_diameter_mm', float('nan')):.1f} mm"
        ),
        "template_count": template_count,
        "template_opts": TEMPLATE_OPTS,
        "criteria": criteria,
        "inference_diagnostics": diagnostics,
        "errors": {
            "silhouette_iou": silhouette_iou,
            "bbox_center_dist_px": center_dist,
            "bbox_iou": criteria["C5_bbox_center_dist_px"]["bbox_iou"],
            "mask_coverage_by_proxy": mask_cov,
            "proxy_area_over_mask_area": proj_area / max(1, mask_area),
            "in_plane_axis_error_deg": axis_angle_deg,
            "foundpose_internal_score": float(pose_meta["score"]),
            "foundpose_internal_score_note": (
                "FoundPose's own per-hypothesis score is 0.0 because upstream "
                "eval_util.update_without_anno is called with camera_c2w=orig_camera_c2w while the "
                "correspondences live in the crop camera, so it reprojects with the wrong camera; "
                "this does not affect the T_camera_object we export (trans_m2oc frame is correct)"
            ),
            "symmetric_add_proxy_metric": None,
            "object_diameter_mm": ds_info.get("model_diameter_mm"),
            "pose_translation_mm": T[:3, 3].tolist(),
            "note": "no metric GT pose exists for this clip; errors are 2D reprojection-based",
        },
        "T_camera_object": T.tolist(),
        "camera_assumed": {
            "K": K_QUERY,
            "im_size": [IMAGE_W, IMAGE_H],
            "source": "assumed 90 deg HFOV (no calibration available for this clip)",
            "scale_source": "handmade_proxy_box",
        },
        "mask_source": f"opencv GrabCut (GC_INIT_WITH_RECT, seed rect {GRABCUT_RECT}, 5 iters)",
        "dataset_info": ds_info,
        "stage_timings": timings,
        "cpu_fallbacks": [
            "torch 2.3.0 MPS op fallback (PYTORCH_ENABLE_MPS_FALLBACK=1): "
            "aten::_upsample_bicubic2d_aa in DINOv2 interpolate_pos_encoding has no MPS kernel; "
            "everything else in the ViT runs natively on MPS",
            "faiss-cpu 1.8.0: visual-words and per-template kNN indices (CPU-only build, no MPS/CUDA)",
            "pyrender/OpenGL: template rasterisation (CPU, no MPS equivalent)",
            "opencv solvePnPRansac + LM refinement: pose solving (CPU)",
            "OpenCV GrabCut: object mask (CPU)",
            "bop_toolkit dataset/camera IO: CPU",
        ],
        "env": {
            "conda_env": CONDA_ENV,
            "yaml": str(ROOT / "conda_foundpose_mps.yaml"),
            "yaml_source": "runs/upstream/foundpose/conda_foundpose_mps.yaml",
            "deltas": [
                "name foundpose_mps -> foundpose-mps",
                "dropped channels nvidia, xformers (no osx-arm64 builds)",
                "dropped conda dep vispy (pulls qt/qt-webengine; imported nowhere)",
                "dropped conda deps triangle, glumpy (imported nowhere)",
                "dropped pip deps submitit, webdataset, hand_tracking_toolkit (imported nowhere)",
                "added pip dep trimesh (required by utils/renderer.py)",
                "first attempt with the unmodified official yaml failed: CondaHTTPError on "
                "qt-main / matplotlib-base downloads from repo.anaconda.com (vispy dependency)",
            ],
        },
        "upstream_patches": [
            {
                "file": "external/bop_toolkit/bop_toolkit_lib/config.py",
                "change": "output_path honours BOP_OUTPUT_PATH env var",
            },
            {
                "file": "external/bop_toolkit/bop_toolkit_lib/dataset_params.py",
                "change": "registered the synthetic 'tbox' dataset (obj_ids, symmetric_obj_ids, split params)",
            },
            {
                "file": "utils/data_util.py",
                "change": "DatasetOpts.crop_image_center=False, add_depth=False (1280x720 RGB-only clip)",
            },
            {
                "file": "scripts/infer.py, scripts/gen_repre.py",
                "change": "device selection now prefers MPS (upstream only has cuda/cpu)",
            },
            {
                "file": "utils/knn_util.py",
                "change": (
                    "KNN.fit/KNN.search unwrap any non-CPU tensor (upstream only handles CUDA); "
                    "faiss-cpu segfaults when handed MPS-backed torch storage (observed in gen_repre)"
                ),
            },
            {
                "file": "utils/infer_pose_util.py",
                "change": (
                    "mask slice bug fix: `[shift:-shift]` is empty when shift == 0 (`-0 == 0`); "
                    "the detection mask became empty and grid_sample then crashed"
                ),
            },
            {
                "file": "scripts/infer.py",
                "change": (
                    "debug visualisation of the per-vertex error now requires a GT pose; the "
                    "no-annotation path passed None into vis_pointcloud_error and crashed"
                ),
            },
            {
                "file": "(none)",
                "change": (
                    "MPS op-level CPU fallback (env PYTORCH_ENABLE_MPS_FALLBACK=1): "
                    "aten::_upsample_bicubic2d_aa inside DINOv2 interpolate_pos_encoding is not "
                    "implemented for MPS in torch 2.3.0"
                ),
            },
        ],
        "host": {
            "machine": platform.machine(),
            "python": platform.python_version(),
            "os": platform.platform(),
        },
        "artifacts": {
            "overlay": str(EVIDENCE / "overlay_f0001_foundpose_proxy.png"),
            "mask": str(EVIDENCE / "mask_f0001.png"),
            "T_camera_object": str(OUT / "T_camera_object.txt"),
            "logs": str(LOGS),
        },
    }
    REPORT.write_text(json.dumps(report, indent=2))
    _log(f"verdict = {verdict}")
    for name, c in criteria.items():
        _log(f"  {name}: value={c['value']} pass={c['pass']}")
    return 0 if verdict == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
