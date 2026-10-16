"""W6b round-4c: MediaPipe wrist landmarks over the full-fps source frames.

Reuses smoke 6's MediaPipe machinery (perception.hamer_smoke.detect_hands:
the same detector, the same ego-view x-order handedness prior) and keeps the
one thing the trajectory metric needs: the per-frame WRIST landmark (pixel)
per hand, plus the hand span in the SAME landmark terms the MANO stream uses
(wrist -> middle MCP, landmarks 0 -> 9), so the wrist-referenced trajectory
error is a like-for-like wrist-to-wrist comparison in hand-span units.

Output runs/w6b/wrist_landmarks.npz (video timeline, 30 fps):
    for side in {left, right}:
      present   (F,) bool
      wrist_px  (F,2) float64  landmark 0, full-frame pixels
      span_px   (F,) float64  ||lm9 - lm0|| (wrist -> middle MCP)
      score     (F,) float64  detector score

Run (hamer venv):  runs/.venv-hamer/bin/python -m retarget.wrist_landmarks
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

FRAMES_DIR = REPO / "runs/w6b/frames_full"
OUT_NPZ = REPO / "runs/w6b/wrist_landmarks.npz"
WRIST, MIDDLE_MCP = 0, 9


def main() -> int:
    import cv2

    from perception import hamer_smoke as hs

    paths = sorted(FRAMES_DIR.glob("f_*.jpg"))
    if not paths:
        raise FileNotFoundError(f"no frames under {FRAMES_DIR}")
    rt = hs.load("mps" if _mps() else "cpu")

    F = len(paths)
    out = {}
    for side in ("left", "right"):
        out[f"{side}/present"] = np.zeros(F, bool)
        out[f"{side}/wrist_px"] = np.zeros((F, 2))
        out[f"{side}/span_px"] = np.zeros(F)
        out[f"{side}/score"] = np.zeros(F)

    for i, p in enumerate(paths):
        img = cv2.imread(str(p))
        if img is None:
            raise FileNotFoundError(p)
        dets = hs.detect_hands(rt, img[:, :, ::-1])
        rights = hs.handedness_flags(dets, img.shape[1])
        for k, det in enumerate(dets):
            side = "right" if rights[k] == 1 else "left"
            lm = det["keypoints_2d"].astype(np.float64)
            out[f"{side}/present"][i] = True
            out[f"{side}/wrist_px"][i] = lm[WRIST]
            out[f"{side}/span_px"][i] = float(
                np.linalg.norm(lm[MIDDLE_MCP] - lm[WRIST]))
            out[f"{side}/score"][i] = float(det["detector_score"])
        if (i + 1) % 30 == 0 or i == F - 1:
            print(f"[{i + 1}/{F}] {p.stem}", flush=True)

    OUT_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT_NPZ, frame_idx=np.arange(1, F + 1), fps=np.float64(30.0),
             **out)
    counts = {s: int(out[f"{s}/present"].sum()) for s in ("left", "right")}
    print(f"wrote {OUT_NPZ}: left {counts['left']}/{F}, right {counts['right']}/{F}")
    return 0


def _mps() -> bool:
    try:
        import torch

        return bool(torch.backends.mps.is_available())
    except ImportError:
        return False


if __name__ == "__main__":
    raise SystemExit(main())
