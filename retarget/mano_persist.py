"""W6b step 1: persist per-frame MANO parameters from HaMeR.

Smoke 6 proved the HaMeR wiring on MPS but kept only two overlay PNGs -- the
per-frame MANO parameters were never written to disk.  This runner reuses
smoke 6's own pipeline functions (``perception.hamer_smoke``: the MediaPipe
reference pass, the ego-view x-order handedness prior with the calibrated
``swap=1`` convention, the ViTDetDataset crop, the MPS forward) and adds the
one thing the smoke did not keep: every frame's per-hand MANO parameters.

Output ``runs/w6b/mano_params.npz`` (video timeline, fps from the source clip):

    frame_idx        (F,)   int   video frame number, 1-based (f_0001 -> 1)
    t                (F,)   f8    seconds, frame_idx-1) / fps
    fps              ()     f8    source clip fps (30.0, ffprobe)
    for side in {left, right}:
      <side>/present    (F,)    bool   hand detected on this frame
      <side>/pose       (F,15,3) f8    MANO hand_pose, axis-angle
      <side>/shape      (F,10)   f8    MANO betas
      <side>/transl     (F,3)    f8    pred_cam_t (HaMeR camera frame, metres)
      <side>/global_orient (F,3) f8   MANO global_orient, axis-angle
      <side>/conf       (F,)     f8    MediaPipe detector score
      <side>/reproj_mean_px (F,) f8  MANO->frame reprojection error (evidence)
      <side>/box_xyxy   (F,4)    f8    detection box, full-frame px

Frames where a hand is absent are zero-filled with ``present=False``; gaps are
NOT interpolated here -- what HaMeR did not see is a fact about the source and
belongs to the downstream consumer to disclose.

Run (smoke venv):
    runs/.venv-hamer/bin/python -m retarget.mano_persist
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from perception import hamer_smoke as hs  # noqa: E402

FRAMES_DIR = REPO / "runs/p0/assets/frames"
OUT_NPZ = REPO / "runs/w6b/mano_params.npz"
OUT_META = REPO / "runs/w6b/mano_params.meta.json"
SOURCE_CLIP = REPO / "runs/p0/assets/tissue_box.mp4"
# frames/f_NNNN.jpg = source frame 10*(NN-1)+1 (verified 2026-10-16 by direct
# ffmpeg extraction: source n=11 matches f_0002, source n=191 matches f_0020,
# JPEG recompression noise aside).  ffprobe: 30/1 fps, 390 frames -- so the
# effective rate of this frame set is 3.0 fps and t = (N-1)/3 s.
CLIP_FPS = 3.0
FRAME_STRIDE = 10
# smoke 6's calibrated mirror convention (51/54 detector agreement vs 3/54 for
# swap=0); frozen there, reused here so this run consumes the same wiring.
HANDEDNESS_SWAP = 1


def main() -> int:
    import cv2

    paths = sorted(FRAMES_DIR.glob("f_*.jpg"))
    if not paths:
        raise FileNotFoundError(f"no frames under {FRAMES_DIR}")
    device = "mps"
    try:
        import torch

        if not torch.backends.mps.is_available():
            device = "cpu"
    except ImportError:
        pass
    rt = hs.load(device)

    F = len(paths)
    out = {
        "frame_idx": np.arange(1, F + 1, dtype=np.int64),
        "t": (np.arange(F, dtype=np.float64)) / CLIP_FPS,
        "fps": np.float64(CLIP_FPS),
    }
    sides: dict[str, dict[str, np.ndarray]] = {}
    for side in ("left", "right"):
        sides[side] = {
            "present": np.zeros(F, bool),
            "pose": np.zeros((F, 15, 3)),
            "shape": np.zeros((F, 10)),
            "transl": np.zeros((F, 3)),
            "global_orient": np.zeros((F, 3)),
            "conf": np.zeros(F),
            "reproj_mean_px": np.zeros(F),
            "box_xyxy": np.zeros((F, 4)),
        }

    from scipy.spatial.transform import Rotation

    n_hand_frames = 0
    for i, p in enumerate(paths):
        img_bgr = cv2.imread(str(p))
        if img_bgr is None:
            raise FileNotFoundError(f"unreadable frame: {p}")
        img_rgb = img_bgr[:, :, ::-1]
        dets = hs.detect_hands(rt, img_rgb)
        if not dets:
            continue
        n_hand_frames += 1
        rights = hs.handedness_flags(dets, img_rgb.shape[1])
        batch = hs._batch_from_boxes(
            rt, img_bgr, [d["box_xyxy"] for d in dets], rights
        )
        preds = hs._run_hamer_batch(rt, batch)
        mano = preds["pred_mano_params"]
        for k, det in enumerate(dets):
            side = "right" if rights[k] == 1 else "left"
            s = sides[side]
            s["present"][i] = True
            s["pose"][i] = hs._mano_params_to_pose_aa(mano["hand_pose"][k])
            go = np.asarray(mano["global_orient"][k], dtype=np.float64).reshape(3, 3)
            s["global_orient"][i] = Rotation.from_matrix(go).as_rotvec()
            s["shape"][i] = np.asarray(mano["betas"][k], dtype=np.float64)
            s["transl"][i] = np.asarray(preds["pred_cam_t"][k], dtype=np.float64)
            s["conf"][i] = float(det["detector_score"])
            s["box_xyxy"][i] = det["box_xyxy"]
            j2d = hs._project_joints_to_frame(
                np.asarray(preds["pred_keypoints_2d"][k], dtype=np.float64),
                batch["box_center"][k], batch["box_size"][k], float(rights[k]),
            )
            err = np.linalg.norm(
                j2d - det["keypoints_2d"].astype(np.float64), axis=1
            )
            s["reproj_mean_px"][i] = float(err.mean())
        print(f"[{i + 1}/{F}] {p.stem}: "
              f"{sum(1 for k in range(len(dets)))} hand(s) "
              f"({', '.join('R' if r else 'L' for r in rights)})", flush=True)

    for side, arrs in sides.items():
        for name, arr in arrs.items():
            out[f"{side}/{name}"] = arr

    OUT_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT_NPZ, **out)

    present = {s: int(sides[s]["present"].sum()) for s in sides}
    meta = {
        "source_clip": str(SOURCE_CLIP.relative_to(REPO)),
        "clip_fps": CLIP_FPS,
        "n_frames": F,
        "frames_dir": str(FRAMES_DIR.relative_to(REPO)),
        "device": rt.device,
        "handedness_swap": HANDEDNESS_SWAP,
        "handedness_rule": (
            "ego-view x-order (smoke 6 calibrated: swap=1); a hand left of the "
            "image centre is the person's left hand"
        ),
        "frames_with_any_hand": n_hand_frames,
        "detection_counts": present,
        "wiring_provenance": (
            "perception/hamer_smoke.py functions reused unchanged: "
            "detect_hands, handedness_flags, _batch_from_boxes, "
            "_run_hamer_batch, _project_joints_to_frame; smoke 6 report "
            "runs/p0/smoke6_hamer/report.json gates this wiring"
        ),
        "checkpoint": {
            "path": str(hs.CKPT.relative_to(REPO)),
            "sha256_1kb_limit": hs._sha256(hs.CKPT, limit=1 << 20),
        },
        "lib_versions": rt.lib_versions,
        "weights_version": rt.weights_version,
        "mps_fallback_ops": rt.recorder.as_list(),
        "array_layout": (
            "per side: pose (F,15,3) axis-angle hand_pose; shape (F,10) betas; "
            "transl (F,3) pred_cam_t metres in the HaMeR camera frame "
            "(x right, y down, z forward); global_orient (F,3) axis-angle; "
            "absent detections are zero-filled with present=False, never "
            "interpolated"
        ),
    }
    OUT_META.write_text(json.dumps(meta, indent=2))
    print(f"wrote {OUT_NPZ} ({F} frames, left {present['left']}, "
          f"right {present['right']}) and {OUT_META}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
