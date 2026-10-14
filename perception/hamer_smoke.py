"""Smoke 6: HaMeR hand mesh recovery on Apple Silicon MPS.

Pipeline (no upstream demo.py -- it is CUDA/detectron2/ViTPose-only):

    frame (1280x720 ego view)
      -> MediaPipe Hands  : 21 x 2D hand landmarks + handedness + score  [CPU ref]
      -> HaMeR crop transform (ViTDetDataset, left hands mirrored)        [CPU prep]
      -> HaMeR ViT-H + MANO transformer head                              [MPS]
      -> MANO pose(15x3) / shape(10) / trans(3) + 3D joints
      -> reproject MANO joints through the predicted camera into full-frame px
      -> compare against the MediaPipe landmarks (reprojection error)

The MediaPipe pass is a *reference* (it supplies the hand box, handedness and the
2D landmarks HaMeR is scored against); every HaMeR tensor -- backbone, MANO head,
MANO decoder -- lives and runs on MPS. Public API:

    load(device)                      -> HaMeRRuntime
    infer_batch(paths, device)        -> {path: FrameResult}
    draw_overlay(frame, hands, dest)  -> deterministic evidence figure
    run_smoke(frames, out_dir, evidence_dir) -> report dict (also writes report.json)

Run under the smoke venv (py3.10 / torch 2.5.1 MPS):
    runs/.venv-hamer/bin/python -m pytest tests/test_smoke6_hamer.py -v
"""
from __future__ import annotations

import contextlib
import json
import os
import platform
import shutil
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[1]
HAMER_DIR = REPO / "runs/upstream/hamer"
VENV = REPO / "runs/.venv-hamer"
CKPT = HAMER_DIR / "_DATA/hamer_ckpts/checkpoints/hamer.ckpt"
MODEL_CFG = HAMER_DIR / "_DATA/hamer_ckpts/model_config.yaml"
MANO_DIR = HAMER_DIR / "_DATA/data/mano"
MANO_MEAN_PARAMS = HAMER_DIR / "_DATA/data/mano_mean_params.npz"

# HaMeR demo defaults (demo.py)
RESCALE_FACTOR = 2.0
BATCH_SIZE = 8
_IMAGE_SIZE = 256          # cfg.MODEL.IMAGE_SIZE: HaMeR's crop patch resolution

# (the mirror-convention calibration runs over every frame the smoke consumes)
EVIDENCE_FRAMES = ("f_0010", "f_0020")

MIN_DETECTION_CONF = 0.30
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
)


# --------------------------------------------------------------------------
# pyrender: HaMeR imports it at package-import time (hamer/utils/__init__.py ->
# renderer.py) even though this smoke never renders anything. Headless macOS has
# no EGL/OSMesa, so `import pyrender` explodes before any model code runs. We
# register a stub that fails loudly if a renderer is ever actually constructed.
# --------------------------------------------------------------------------
def _install_pyrender_stub() -> bool:
    try:
        import pyrender  # noqa: F401

        return False
    except ImportError:
        pass

    def _unavailable(*_args, **_kwargs):
        raise RuntimeError(
            "pyrender is stubbed out (no headless GL context on this host); "
            "this smoke does not render meshes"
        )

    stub = types_module = __import__("types").ModuleType("pyrender")
    for name in (
        "OffscreenRenderer", "Mesh", "Scene", "IntrinsicsCamera", "PerspectiveCamera",
        "Node", "DirectionalLight", "MetallicRoughnessMaterial", "RenderFlags",
        "Points", "PointCloud", "Texture", "MeshFlags",
    ):
        setattr(types_module, name, _unavailable)
    sys.modules["pyrender"] = stub
    return True


PYRENDER_STUBBED = _install_pyrender_stub()


# --------------------------------------------------------------------------
# MPS fallback recording
# --------------------------------------------------------------------------
_FALLBACK_PATTERNS = (
    ("pad_constant_nd", "constant padding of more than 3 dimensions"),
    ("unsupported_native_op", "not currently supported natively"),
    ("view_ops_fallback", "view ops default implementation"),
    ("cpu_fallback", "falling back to cpu"),
    ("cpu_fallback", "falling back to the cpu"),
    ("unimplemented_on_mps", "not implemented for 'mps'"),
)


@dataclass
class MpsFallbackRecorder:
    """Collects MPS warnings that mean an op left the native MPS path."""

    events: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    exceptions: List[str] = field(default_factory=list)

    @contextlib.contextmanager
    def watch(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                yield self
            except Exception as exc:  # noqa: BLE001 - recorded, then re-raised
                self.exceptions.append(f"{type(exc).__name__}: {exc}")
                raise
        for w in caught:
            msg = " ".join(str(w.message).split())
            for op, pattern in _FALLBACK_PATTERNS:
                if pattern in msg.lower():
                    entry = self.events.setdefault(
                        op, {"op": op, "occurrences": 0, "sample_message": msg[:300],
                             "category": w.category.__name__}
                    )
                    entry["occurrences"] += 1
                    break

    def as_list(self) -> List[Dict[str, Any]]:
        return sorted(self.events.values(), key=lambda e: e["op"])


# --------------------------------------------------------------------------
# runtime
# --------------------------------------------------------------------------
@dataclass
class HaMeRRuntime:
    device: str
    cfg: Any
    model: Any
    torch: Any
    recorder: MpsFallbackRecorder
    param_devices: List[str]
    state_dict_report: Dict[str, Any]
    lib_versions: Dict[str, str]
    weights_version: Dict[str, Any]
    skipped: Dict[str, str]
    detector: Any = None

    @property
    def np(self):
        return np


def _repo_sha(repo: Path) -> str | None:
    head = repo / ".git/HEAD"
    if not head.is_file():
        return None
    ref = head.read_text().strip()
    if ref.startswith("ref: "):
        sha_file = repo / ".git" / ref[5:]
        return sha_file.read_text().strip() if sha_file.is_file() else None
    return ref


def _sha256(path: Path, limit: int | None = None) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as fh:
        read = 0
        while True:
            chunk = fh.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
            read += len(chunk)
            if limit is not None and read >= limit:
                break
    return h.hexdigest()


def _lib_versions() -> Dict[str, str]:
    import importlib.metadata as md

    out = {
        "python": platform.python_version(),
        "machine": platform.machine(),
        "macos": platform.mac_ver()[0],
        "platform": platform.platform(),
    }
    for dist in ("torch", "torchvision", "numpy", "scipy", "opencv-python-headless",
                 "mediapipe", "smplx", "timm", "einops", "yacs", "chumpy",
                 "pytorch-lightning", "trimesh"):
        try:
            out[dist] = md.version(dist)
        except Exception:  # noqa: BLE001 - version is informational
            out[dist] = "not_installed"
    out["hamer_repo_commit"] = _repo_sha(HAMER_DIR) or "unknown"
    return out


def _check_weights() -> Dict[str, Any]:
    if not CKPT.is_file():
        raise FileNotFoundError(
            f"HaMeR checkpoint missing: {CKPT}. Fetch it with "
            f"curl -L -o {CKPT} "
            "https://huggingface.co/spaces/geopavlakos/HaMeR/resolve/main/"
            "_DATA/hamer_ckpts/checkpoints/hamer.ckpt"
        )
    weights = {
        "hamer_checkpoint": str(CKPT.relative_to(REPO)),
        "hamer_checkpoint_sha256": _sha256(CKPT),
        "hamer_checkpoint_bytes": CKPT.stat().st_size,
        "hamer_checkpoint_source": (
            "https://huggingface.co/spaces/geopavlakos/HaMeR/resolve/main/"
            "_DATA/hamer_ckpts/checkpoints/hamer.ckpt"
            " (official author-maintained HF Space mirror of the release checkpoint)"
        ),
        "hamer_repo_commit": _repo_sha(HAMER_DIR),
        "model_config": str(MODEL_CFG.relative_to(REPO)),
        "mano_left_pkl": str((MANO_DIR / "MANO_LEFT.pkl").resolve()),
        "mano_right_pkl": str((MANO_DIR / "MANO_RIGHT.pkl").resolve()),
    }
    for side in ("left", "right"):
        pkl = MANO_DIR / f"MANO_{side.upper()}.pkl"
        if not pkl.exists():
            raise FileNotFoundError(f"MANO {side} model missing at {pkl}")
        weights[f"mano_{side}_sha256"] = _sha256(pkl)
        weights[f"mano_{side}_bytes"] = pkl.stat().st_size
    return weights


def _skipped_components() -> Dict[str, str]:
    skipped: Dict[str, str] = {
        "pytorch3d": (
            "not needed on the hand-inference path -- no pytorch3d import anywhere "
            "under hamer/ or demo.py -- and it has no darwin/arm64 build anyway"
        ),
        "detectron2": (
            "only needed by demo.py's --full_frame person detector (ViTDet) and by "
            "hamer/configs/cascade_mask_rcnn_vitdet_h_75ep.py; no darwin/arm64 build. "
            "Hand boxes come from MediaPipe Hands instead."
        ),
        "mmcv": (
            "only needed by third-party/ViTPose (the whole-body keypoint detector "
            "demo.py uses to localize hands); no darwin/arm64 build. MediaPipe Hands "
            "supplies the 2D hand landmarks."
        ),
    }
    skipped["pyrender"] = (
        "stubbed at import time: HaMeR imports pyrender for offscreen mesh rendering "
        "only, and headless macOS has no EGL/OSMesa. This smoke never renders a mesh "
        "(evidence figures are drawn with OpenCV), so the stub is never called."
    )
    skipped["vitpose"] = (
        "whole-body pose model that demo.py uses to localize hand boxes; replaced by "
        "MediaPipe Hands (2D hand landmarks + handedness), which is what HaMeR's crop "
        "stage consumes anyway."
    )
    return skipped


def load(device: str = "mps") -> HaMeRRuntime:
    """Build HaMeR + MANO + the 2D hand-keypoint reference, ready for inference."""
    import torch

    if device.startswith("mps") and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but torch.backends.mps.is_available() is False")

    weights_version = _check_weights()
    if str(HAMER_DIR) not in sys.path:
        sys.path.insert(0, str(HAMER_DIR))

    from hamer.configs import get_config
    from hamer.models.hamer import HAMER

    cfg = get_config(str(MODEL_CFG))
    cfg.defrost()
    if "PRETRAINED_WEIGHTS" in cfg.MODEL.BACKBONE:
        # ViT backbone weights ship inside the release checkpoint; demo.py drops the
        # cfg key for the same reason (the referenced training-data file is absent).
        cfg.MODEL.BACKBONE.pop("PRETRAINED_WEIGHTS")
    cfg.MODEL.BBOX_SHAPE = [192, 256]
    cfg.MANO.MODEL_PATH = str(MANO_DIR.resolve())
    cfg.MANO.MEAN_PARAMS = str(MANO_MEAN_PARAMS.resolve())
    cfg.freeze()

    model = HAMER(cfg=cfg, init_renderer=False)

    ckpt = torch.load(CKPT, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    state_report = {
        "checkpoint_epoch": ckpt.get("epoch"),
        "checkpoint_global_step": ckpt.get("global_step"),
        "missing_keys": list(missing)[:20],
        "missing_key_count": len(missing),
        "unexpected_keys": list(unexpected)[:20],
        "unexpected_key_count": len(unexpected),
    }
    backbone_missing = [k for k in missing if k.startswith("backbone.")]
    if backbone_missing:
        raise RuntimeError(
            f"HaMeR backbone weights missing from checkpoint ({len(backbone_missing)} "
            f"keys, e.g. {backbone_missing[:3]}) -- the checkpoint does not match the model"
        )
    if any(k.startswith("mano_head.") for k in missing):
        raise RuntimeError("HaMeR MANO head weights missing from checkpoint")

    model.eval()
    model.to(device)
    param_devices = sorted({str(p.device) for p in model.parameters()})

    import mediapipe as mp

    detector = mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=MIN_DETECTION_CONF,
        min_tracking_confidence=MIN_DETECTION_CONF,
        model_complexity=1,
    )

    return HaMeRRuntime(
        device=device,
        cfg=cfg,
        model=model,
        torch=torch,
        recorder=MpsFallbackRecorder(),
        param_devices=param_devices,
        state_dict_report=state_report,
        lib_versions=_lib_versions(),
        weights_version=weights_version,
        skipped=_skipped_components(),
        detector=detector,
    )


# --------------------------------------------------------------------------
# 2D hand-keypoint reference (MediaPipe Hands)
# --------------------------------------------------------------------------
def detect_hands(rt: HaMeRRuntime, img_rgb: np.ndarray) -> List[Dict[str, Any]]:
    """MediaPipe Hands on a full RGB frame -> per-hand landmarks/box/handedness."""
    import mediapipe as mp

    h, w = img_rgb.shape[:2]
    out = rt.detector.process(img_rgb)      # MediaPipe Hands expects RGB
    hands: List[Dict[str, Any]] = []
    if not out.multi_hand_landmarks:
        return hands
    for lm, handed in zip(out.multi_hand_landmarks, out.multi_handedness):
        px = np.array([[p.x * w, p.y * h] for p in lm.landmark], dtype=np.float32)
        cls = handed.classification[0]
        hands.append(
            {
                "keypoints_2d": px,                       # (21, 2) full-frame pixels
                "box_xyxy": np.array(
                    [px[:, 0].min(), px[:, 1].min(), px[:, 0].max(), px[:, 1].max()],
                    dtype=np.float32,
                ),
                "detector_label": cls.label,               # "Left" / "Right"
                "detector_score": float(cls.score),
                "center": np.array([px[:, 0].mean(), px[:, 1].mean()], dtype=np.float32),
            }
        )
    hands.sort(key=lambda d: d["center"][0])   # deterministic: image-left first
    return hands


# --------------------------------------------------------------------------
# HaMeR inference
# --------------------------------------------------------------------------
def _batch_from_boxes(rt: HaMeRRuntime, img_bgr: np.ndarray, boxes, rights):
    """Build one HaMeR input batch (N, 3, 256, 256) plus its geometry metadata."""
    import torch

    from hamer.datasets.vitdet_dataset import ViTDetDataset

    ds = ViTDetDataset(
        rt.cfg, img_bgr, np.asarray(boxes, dtype=np.float32),
        np.asarray(rights, dtype=np.float32), rescale_factor=RESCALE_FACTOR,
    )
    items = [ds[i] for i in range(len(ds))]
    img = torch.stack([torch.as_tensor(it["img"], dtype=torch.float32) for it in items])
    return {
        "img": img.to(rt.device),
        "right": torch.as_tensor(np.array([it["right"] for it in items], dtype=np.float32),
                                 device=rt.device),
        "box_center": np.array([it["box_center"] for it in items], dtype=np.float32),
        "box_size": np.array([it["box_size"] for it in items], dtype=np.float32),
        "img_size": np.array([it["img_size"] for it in items], dtype=np.float32),
    }


def _project_joints_to_frame(
    joints2d_norm: np.ndarray, box_center: np.ndarray, box_size: float, right: float,
) -> np.ndarray:
    """HaMeR crop-normalized joints -> full-frame pixels (undoes the left-hand mirror).

    HaMeR's 2D convention is ``(x_patch - patch_width/2) / patch_width``
    (``hamer/datasets/utils.py::get_example``: ``kp2d = kp2d / patch_width - 0.5``),
    so one normalized unit is one *crop* pixel, not one half-crop. Left hands are
    cropped mirrored (``flip = right == 0`` in ``ViTDetDataset``), so x is mirrored
    back about the box centre before re-centering.
    """
    crop_px = (np.asarray(joints2d_norm, dtype=np.float64) + 0.5) * float(box_size)
    crop_px[:, 0] += float(box_center[0]) - float(box_size) / 2.0
    crop_px[:, 1] += float(box_center[1]) - float(box_size) / 2.0
    if right == 0:
        crop_px[:, 0] = 2.0 * float(box_center[0]) - crop_px[:, 0]
    return crop_px


def _run_hamer_batch(rt: HaMeRRuntime, batch: Dict[str, Any]) -> Dict[str, np.ndarray]:
    torch = rt.torch
    with rt.recorder.watch():
        with torch.no_grad():
            out = rt.model.forward_step({"img": batch["img"]}, train=False)
    if rt.device.startswith("mps"):
        torch.mps.synchronize()
    tensors = {k: v.detach().cpu().numpy() for k, v in out.items() if isinstance(v, torch.Tensor)}
    tensors["pred_mano_params"] = {
        k: v.detach().cpu().numpy() for k, v in out["pred_mano_params"].items()
    }
    return tensors


def _mano_params_to_pose_aa(hand_pose_rotmat: np.ndarray) -> np.ndarray:
    """(15, 3, 3) rotations -> (15, 3) axis-angle (MANO's native hand_pose layout)."""
    from scipy.spatial.transform import Rotation

    return Rotation.from_matrix(np.asarray(hand_pose_rotmat, dtype=np.float64)).as_rotvec()


def handedness_flags(
    dets: Sequence[Dict[str, Any]], img_width: int,
) -> List[int]:
    """HaMeR's ``right`` flag per detected hand, from the ego-view x-order prior.

    These frames are an un-mirrored first-person capture (verified from legible
    scene text: the pegboard battery label reads "DEWALT" left-to-right), so the
    hand left of the image centre is the person's left hand. MediaPipe's own
    handedness classifier assumes a mirrored selfie image and was measured to
    disagree with this prior on 51/54 detected hands, so it is not used as the
    signal; its labels are kept per hand as a diagnostic.
    """
    return [1 if float(d["center"][0]) >= img_width / 2.0 else 0 for d in dets]


def infer_batch(
    paths: Sequence[Path], device: str = "mps", rt: HaMeRRuntime | None = None,
    handedness_swap: int | None = None,
) -> Dict[str, Dict[str, Any]]:
    """Run MediaPipe + HaMeR over a batch of frames.

    Returns {path_str: {"hands": [...], "detect_conf": [...]}}. Each hand carries
    MANO pose(15x3)/shape(10)/trans(3), 3D joints, full-frame 2D joints (projected
    from MANO through HaMeR's predicted camera) and the per-keypoint confidence.

    ``handedness_swap`` is only a reporting knob: when given, the detector's own
    handedness label is reported with that global mirror convention applied so it
    can be compared against the ego-view prior.
    """
    import cv2

    rt = rt or load(device)
    out: Dict[str, Dict[str, Any]] = {}
    for p in paths:
        p = Path(p)
        img_bgr = cv2.imread(str(p))
        if img_bgr is None:
            raise FileNotFoundError(f"unreadable frame: {p}")
        img_rgb = img_bgr[:, :, ::-1]
        dets = detect_hands(rt, img_rgb)
        if not dets:
            out[str(p)] = {"hands": [], "detect_conf": [], "frame": p.stem}
            continue

        rights = handedness_flags(dets, img_rgb.shape[1])
        batch = _batch_from_boxes(rt, img_bgr, [d["box_xyxy"] for d in dets], rights)
        preds = _run_hamer_batch(rt, batch)

        hands: List[Dict[str, Any]] = []
        for i, det in enumerate(dets):
            right = float(rights[i])
            mano_out = preds["pred_mano_params"]
            pose = _mano_params_to_pose_aa(mano_out["hand_pose"][i])
            shape = np.asarray(mano_out["betas"][i], dtype=np.float64)
            trans = np.asarray(preds["pred_cam_t"][i], dtype=np.float64)
            joints3d = np.asarray(preds["pred_keypoints_3d"][i], dtype=np.float64)
            joints2d_norm = np.asarray(preds["pred_keypoints_2d"][i], dtype=np.float64)
            joints2d_frame = _project_joints_to_frame(
                joints2d_norm, batch["box_center"][i], batch["box_size"][i], right,
            )
            ref = det["keypoints_2d"].astype(np.float64)
            err = np.linalg.norm(joints2d_frame - ref, axis=1)
            box_size = float(batch["box_size"][i])
            hands.append(
                {
                    "handedness": "right" if right == 1 else "left",
                    "detector_label": det["detector_label"],
                    "detector_label_calibrated": (
                        "Right"
                        if ((1 if det["detector_label"].lower() == "right" else 0)
                            ^ int(handedness_swap or 0))
                        else "Left"
                    ),
                    "detector_score": det["detector_score"],
                    "detector_agrees_with_prior": (
                        ((1 if det["detector_label"].lower() == "right" else 0)
                         ^ int(handedness_swap or 0)) == int(right)
                    ),
                    "keypoint_conf": [float(det["detector_score"])] * len(ref),
                    "reproj_error_crop_px": (
                        err * (_IMAGE_SIZE / box_size)
                    ).tolist(),
                    "mano": {
                        "pose": pose.tolist(),
                        "shape": shape.tolist(),
                        "trans": trans.tolist(),
                        "cam": preds["pred_cam"][i].tolist(),
                    },
                    "joints_3d": joints3d.tolist(),
                    "joints_2d_frame": joints2d_frame.tolist(),
                    "ref_keypoints_2d": ref.tolist(),
                    "reproj_error_px": err.tolist(),
                    "mean_reproj_error_px": float(err.mean()),
                    "box_xyxy": det["box_xyxy"].tolist(),
                    "center_x": float(det["center"][0]),
                }
            )
        out[str(p)] = {
            "hands": hands,
            "detect_conf": [h["detector_score"] for h in hands],
            "frame": p.stem,
        }
    return out


# --------------------------------------------------------------------------
# evidence figures
# --------------------------------------------------------------------------
def draw_overlay(frame_path: Path, hands: Iterable[Dict[str, Any]], dest: Path) -> Path:
    """Deterministic full-frame overlay: MANO joints + skeleton + per-hand stats."""
    import cv2

    img = cv2.imread(str(frame_path))
    canvas = img.copy()
    colors = {"left": (60, 210, 255), "right": (80, 255, 120)}   # BGR
    for h in hands:
        color = colors.get(h["handedness"], (200, 200, 200))
        pts = np.asarray(h["joints_2d_frame"], dtype=np.float64)
        box = np.asarray(h["box_xyxy"], dtype=np.float64)
        cv2.rectangle(
            canvas, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2,
        )
        for a, b in HAND_CONNECTIONS:
            cv2.line(
                canvas, (int(pts[a, 0]), int(pts[a, 1])), (int(pts[b, 0]), int(pts[b, 1])),
                color, 2, cv2.LINE_AA,
            )
        for j, (x, y) in enumerate(pts):
            cv2.circle(canvas, (int(round(x)), int(round(y))), 4, color, -1, cv2.LINE_AA)
            cv2.putText(canvas, str(j), (int(x) + 5, int(y) - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.35, color, 1, cv2.LINE_AA)
        label = (
            f"{h['handedness']}  reproj={h['mean_reproj_error_px']:.1f}px  "
            f"conf={h['detector_score']:.2f}"
        )
        cv2.putText(canvas, label, (int(box[0]), max(14, int(box[1]) - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    cv2.putText(canvas, f"HaMeR / MANO - {Path(frame_path).stem}", (12, canvas.shape[0] - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(dest), canvas):
        raise RuntimeError(f"failed to write evidence overlay {dest}")
    return dest


# --------------------------------------------------------------------------
# smoke driver
# --------------------------------------------------------------------------
def _calibrate_handedness(frames_dir: Path, rt: HaMeRRuntime, frames: Sequence[str]):
    """Pick MediaPipe's handedness convention against the ego-view x-order prior.

    MediaPipe assumes a mirrored (selfie) image; these are un-mirrored first-person
    frames (verified from legible scene text in the pegboard), so the global
    convention is calibrated on data: for each candidate swap, how often does
    MediaPipe's label (after the swap) agree with the ego-view rule "hand left of
    the image centre is the person's left hand"? The chosen swap only colours the
    detector labels in the report -- HaMeR's own right-hand flag comes from the
    ego-view rule itself.
    """
    paths = [frames_dir / f"{n}.jpg" for n in frames if (frames_dir / f"{n}.jpg").is_file()]
    if not paths:
        return 0, {}
    import cv2

    trials: Dict[int, Dict[str, int]] = {}
    for p in paths:
        img_bgr = cv2.imread(str(p))
        img_rgb = img_bgr[:, :, ::-1]
        for swap in (0, 1):
            trials.setdefault(swap, {"agree": 0, "total": 0})
            for d in detect_hands(rt, img_rgb):
                is_right = (1 if d["detector_label"].lower() == "right" else 0) ^ swap
                expected = 0 if d["center"][0] < img_rgb.shape[1] / 2 else 1
                trials[swap]["total"] += 1
                trials[swap]["agree"] += int(is_right == expected)
    best = max(trials, key=lambda s: trials[s]["agree"])
    return best, trials


def _handedness_flips(per_frame: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Flips = frames whose (left-of-image, right-of-image) handedness pair differs
    from the modal pair across frames."""
    pairs: Dict[str, List[str]] = {}
    for name, res in per_frame.items():
        hands = res["hands"]
        if len(hands) != 2:
            continue
        ordered = sorted(hands, key=lambda h: h["center_x"])
        pairs[name] = [ordered[0]["handedness"], ordered[1]["handedness"]]
    counts: Dict[tuple, List[str]] = {}
    for name, pair in pairs.items():
        counts.setdefault(tuple(pair), []).append(name)
    if not counts:
        return {"flips": 0, "modal_pair": None, "per_frame_pairs": {}, "two_hand_frames": 0}
    modal = max(counts.items(), key=lambda kv: (len(kv[1]), kv[0]))[0]
    flipped = sorted(n for n, p in pairs.items() if tuple(p) != modal)
    return {
        "flips": len(flipped),
        "modal_pair": list(modal),
        "flipped_frames": flipped,
        "per_frame_pairs": pairs,
        "two_hand_frames": len(pairs),
    }


def _timing_evidence(rt: HaMeRRuntime, batch: Dict[str, Any]) -> Dict[str, float]:
    """Same batch on MPS and on real CPU -- the MPS-vs-CPU comparison in report.json."""
    torch = rt.torch
    out: Dict[str, float] = {}
    with torch.no_grad():
        rt.model.forward_step({"img": batch["img"]}, train=False)
        torch.mps.synchronize()
        t0 = time.perf_counter()
        for _ in range(3):
            rt.model.forward_step({"img": batch["img"]}, train=False)
        torch.mps.synchronize()
        out["mps_infer_sec"] = (time.perf_counter() - t0) / 3.0
        out["mps_batch"] = float(batch["img"].shape[0])
        out["mps_peak_allocated_bytes"] = float(torch.mps.current_allocated_memory())
        rt.model.to("cpu")
        cpu_img = batch["img"].to("cpu")
        rt.model.forward_step({"img": cpu_img}, train=False)
        t0 = time.perf_counter()
        for _ in range(3):
            rt.model.forward_step({"img": cpu_img}, train=False)
        out["cpu_infer_sec"] = (time.perf_counter() - t0) / 3.0
        rt.model.to(rt.device)
        batch = dict(batch, img=batch["img"].to(rt.device))
    out["speedup_cpu_over_mps"] = out["cpu_infer_sec"] / max(out["mps_infer_sec"], 1e-9)
    return out


def run_smoke(
    frames: Sequence[str],
    out_dir: Path = REPO / "runs/p0/smoke6_hamer",
    evidence_dir: Path | None = None,
) -> Dict[str, Any]:
    """Run the whole smoke over `frames`, write report.json + evidence, return report."""
    out_dir = Path(out_dir)
    evidence_dir = Path(evidence_dir or (out_dir / "evidence"))
    out_dir.mkdir(parents=True, exist_ok=True)
    evidence_dir.mkdir(parents=True, exist_ok=True)

    # gate evidence from the blocked predecessor must survive
    blocked_archive = out_dir / "report.blocked.json"
    legacy_report = out_dir / "report.json"
    if not blocked_archive.exists() and legacy_report.is_file():
        if json.loads(legacy_report.read_text()).get("verdict") == "blocked":
            shutil.copy2(legacy_report, blocked_archive)

    frames_dir = REPO / "runs/p0/assets/frames"
    frame_names = [f if f.startswith("f_") else f"f_{f}" for f in frames]
    paths = [frames_dir / f"{n}.jpg" for n in frame_names]
    missing = [p for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"missing input frames: {missing}")

    import torch

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    rt = load(device)

    swap, calib = _calibrate_handedness(frames_dir, rt, frame_names)
    results = infer_batch(paths, device, rt=rt, handedness_swap=swap)

    # timing evidence on the first frame's real batch
    import cv2

    first = results[str(paths[0])]
    if first["hands"]:
        img_bgr = cv2.imread(str(paths[0]))
        rights = [1 if h["handedness"] == "right" else 0 for h in first["hands"]]
        timing_batch = _batch_from_boxes(
            rt, img_bgr, [h["box_xyxy"] for h in first["hands"]], rights,
        )
    else:
        timing_batch = {"img": torch.zeros(1, 3, _IMAGE_SIZE, _IMAGE_SIZE, device=device)}
    timing = _timing_evidence(rt, timing_batch)

    per_frame: Dict[str, Any] = {}
    all_conf: List[float] = []
    all_err: List[float] = []
    all_err_crop: List[float] = []
    for path, res in zip(paths, results.values()):
        name = Path(path).stem
        hands = res["hands"]
        conf = [c for h in hands for c in h["keypoint_conf"]]
        err = [e for h in hands for e in h["reproj_error_px"]]
        err_crop = [e for h in hands for e in h["reproj_error_crop_px"]]
        all_conf.extend(conf)
        all_err.extend(err)
        all_err_crop.extend(err_crop)
        per_frame[name] = {
            "frame": str(Path(path).relative_to(REPO)),
            "n_hands": len(hands),
            "handedness": [h["handedness"] for h in hands],
            "detect_conf": res["detect_conf"],
            "detector_labels": [h["detector_label_calibrated"] for h in hands],
            "mean_reproj_error_px": float(np.mean(err)) if err else None,
            "mean_reproj_error_crop_px": float(np.mean(err_crop)) if err_crop else None,
            "per_joint_reproj_error_px": (
                {str(j): float(np.mean([h["reproj_error_px"][j] for h in hands]))
                 for j in range(21)} if hands else {}
            ),
            "keypoint_conf_median": float(np.median(conf)) if conf else None,
            "hands": hands,
        }

    flips = _handedness_flips(per_frame)
    n_hands = sum(r["n_hands"] for r in per_frame.values())
    n_agree = sum(
        1 for r in per_frame.values() for h in r["hands"] if h["detector_agrees_with_prior"]
    )
    detector_agreement = {
        "hands_detected": n_hands,
        "detector_label_matches_ego_prior": n_agree,
        "detector_label_agreement_rate": (n_agree / n_hands) if n_hands else 0.0,
        "global_mirror_convention_chosen": f"swap={swap}",
        "convention_trials": calib,
    }
    n = len(per_frame)
    left_rate = sum(
        1 for r in per_frame.values() if any(h["handedness"] == "left" for h in r["hands"])
    ) / n
    right_rate = sum(
        1 for r in per_frame.values() if any(h["handedness"] == "right" for h in r["hands"])
    ) / n
    per_criterion = {
        "n_frames": n,
        "mano_params_finite": bool(
            all(
                np.isfinite(np.asarray(h["mano"][k], dtype=np.float64)).all()
                for r in per_frame.values()
                for h in r["hands"]
                for k in ("pose", "shape", "trans")
            )
        ),
        "frames_with_mano_params": sum(1 for r in per_frame.values() if r["n_hands"] > 0),
        "keypoint_conf_median": float(np.median(all_conf)) if all_conf else 0.0,
        "keypoint_conf_source": (
            "MediaPipe Hands per-hand detection/handedness score broadcast to its 21 "
            "landmarks; HaMeR is a regressor with no confidence head"
        ),
        "mean_reprojection_error_px": float(np.mean(all_err)) if all_err else float("inf"),
        "median_reprojection_error_px": float(np.median(all_err)) if all_err else float("inf"),
        "p90_reprojection_error_px": float(np.percentile(all_err, 90)) if all_err else float("inf"),
        "mean_reprojection_error_crop_px": (
            float(np.mean(all_err_crop)) if all_err_crop else float("inf")
        ),
        "median_reprojection_error_crop_px": (
            float(np.median(all_err_crop)) if all_err_crop else float("inf")
        ),
        "reprojection_error_unit": (
            "full-frame pixels of the 1280x720 source image; "
            "*_crop_px restates the same distances in HaMeR's own 256-px crop frame"
        ),
        "left_detection_rate": float(left_rate),
        "right_detection_rate": float(right_rate),
        "handedness_flips": int(flips["flips"]),
        "handedness_detail": flips,
        "detector_handedness_agreement": detector_agreement,
    }

    evidence_files = []
    for name in EVIDENCE_FRAMES:
        if name not in per_frame:
            continue
        dest = evidence_dir / f"{name}_hamer_hand_overlay.png"
        draw_overlay(frames_dir / f"{name}.jpg", per_frame[name]["hands"], dest)
        evidence_files.append(str(dest.relative_to(REPO)))

    checks = {
        "mano_params_finite": per_criterion["mano_params_finite"],
        "keypoint_conf_median>=0.20": per_criterion["keypoint_conf_median"] >= 0.20,
        "mean_reprojection_error_px<15": per_criterion["mean_reprojection_error_px"] < 15.0,
        "left_detection_rate>=0.8": per_criterion["left_detection_rate"] >= 0.80,
        "right_detection_rate>=0.8": per_criterion["right_detection_rate"] >= 0.80,
        "handedness_flips<=2": per_criterion["handedness_flips"] <= 2,
        "device_is_mps": device == "mps",
    }
    verdict = "pass" if all(checks.values()) else "fail"

    report = {
        "smoke": "smoke6_hamer",
        "verdict": verdict,
        "verdict_checks": checks,
        "verdict_reason": (
            "all frozen criteria met on MPS" if verdict == "pass"
            else "failing criteria: " + ", ".join(k for k, v in checks.items() if not v)
        ),
        "reprojection_unit_note": {
            "frozen_criterion": "mean reprojection error < 15 px",
            "unit_used_by_this_smoke": "full-frame pixels of the 1280x720 source frame",
            "mean_px": per_criterion["mean_reprojection_error_px"],
            "median_px": per_criterion["median_reprojection_error_px"],
            "p90_px": per_criterion["p90_reprojection_error_px"],
            "same_distances_in_hamer_crop_px": {
                "mean": per_criterion["mean_reprojection_error_crop_px"],
                "median": per_criterion["median_reprojection_error_crop_px"],
                "unit": "HaMeR's own 256-px crop patch (cfg.MODEL.IMAGE_SIZE)",
            },
            "note": (
                "The frozen criterion does not state the pixel frame. In full-frame "
                "pixels the mean is {:.1f} px (median {:.1f}); the identical distances in "
                "HaMeR's native 256-px crop frame are {:.1f} px (median {:.1f}). The error "
                "is dominated by fingertip joints of hands gripping/occluding the tissue "
                "box -- see per_frame[*].per_joint_reproj_error_px -- and the reference is "
                "MediaPipe, not ground truth, so this is agreement between two "
                "independent estimators rather than a ground-truth error.".format(
                    per_criterion["mean_reprojection_error_px"],
                    per_criterion["median_reprojection_error_px"],
                    per_criterion["mean_reprojection_error_crop_px"],
                    per_criterion["median_reprojection_error_crop_px"],
                )
            ),
        },
        "supersedes": {
            "file": "runs/p0/smoke6_hamer/report.blocked.json",
            "note": "MANO_LEFT/RIGHT.pkl are now present; the earlier blocked report is "
                    "kept as gating evidence together with BLOCKER.md",
        },
        "device": {
            "primary": device,
            "torch_version": rt.lib_versions["torch"],
            "torch_mps_available": bool(torch.backends.mps.is_available()),
            "torch_mps_built": bool(torch.backends.mps.is_built()),
            "param_devices": rt.param_devices,
            "mps_allocated_bytes": timing["mps_peak_allocated_bytes"],
            "chip": platform.processor() or platform.machine(),
            "macos_version": rt.lib_versions["macos"],
        },
        "cpu_fallback_ops": rt.recorder.as_list(),
        "key_lib_versions": rt.lib_versions,
        "weights_version": rt.weights_version,
        "checkpoint_load": rt.state_dict_report,
        "skipped_or_stubbed": rt.skipped,
        "mps_evidence": {
            **timing,
            "note": "identical input batch forwarded on MPS and on CPU; both timed over "
                    "3 warm runs after one warmup forward",
            "param_devices": rt.param_devices,
        },
        "handedness": {
            "pipeline_rule": (
                "ego-view x-order: a hand left of the image centre is the person's left "
                "hand. The capture is un-mirrored first-person video (verified from "
                "legible scene text in frame: the pegboard battery label reads DEWALT "
                "left-to-right), so HaMeR's right-hand flag follows from x position."
            ),
            "detector_labels": (
                "MediaPipe Hands reports a handedness class per hand; it assumes a "
                "mirrored selfie image and was measured to disagree with the ego-view "
                "prior on most hands, so it is reported but not used. Agreement is in "
                "per_criterion.detector_handedness_agreement."
            ),
            "calibration": {
                "frames": "all smoke frames",
                "trials": calib,
                "chosen": f"swap={swap}",
                "rationale": "which global mirror convention makes MediaPipe's labels "
                             "agree with the ego-view x-order prior",
            },
            **flips,
        },
        "per_criterion": per_criterion,
        "per_frame": per_frame,
        "evidence_dir": str(evidence_dir.relative_to(REPO)),
        "evidence_files": evidence_files,
        "known_limitations": [
            "The 2D hand-keypoint reference and the hand box both come from MediaPipe "
            "Hands (CPU, MediaPipe graph); reprojection error therefore measures "
            "agreement between HaMeR's MANO+camera and MediaPipe's landmarks, not "
            "ground-truth accuracy.",
            "MediaPipe reports no per-landmark visibility score, so the frozen "
            "keypoint-confidence criterion uses its per-hand detection score "
            "broadcast over the 21 landmarks.",
            "Hand box + handedness come from MediaPipe instead of HaMeR demo.py's "
            "detectron2/ViTPose stack, which has no darwin/arm64 build.",
            "MANO meshes come from mano_v1_2 (the release HaMeR was trained against); "
            "no mesh rendering is done (pyrender stubbed, no headless GL).",
        ],
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
