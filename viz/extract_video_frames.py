"""Extract the human-video frames the W6d panel compares against.

The source clip is ``runs/p0/assets/tissue_box.mp4`` (1280x720, 30 fps,
390 frames): both hands approach the tissue box, grasp its two 120x80 mm
end faces, lift it ~20 cm, then set it back down.  The panel's slider is
driven by the robot replay; the video side shows the source frame nearest
in time.  We extract a sparse set of source frames covering the three task
phases (接近 / 触面 / 抬起) so the comparison stays honest frame-by-frame.

Output: ``docs/video_frames/f_NNNN.jpg`` (NNNN = the exact source frame
number, so t = (N-1)/30 s is recoverable) + ``manifest.json``.

    runs/.venv-mj/bin/python viz/extract_video_frames.py
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
SRC = REPO / "runs/p0/assets/tissue_box.mp4"
OUT = REPO / "docs/video_frames"

FPS = 30.0  # ffprobe: 30/1, 390 frames, 13.0 s (pinned in PLAN.md)

# Three task phases, hand-picked on a full contact sheet of the clip.
# (name, [source frame numbers])
SEGMENTS = [
    ("接近", [1, 25, 49]),      # hands travel from rest toward the box ends
    ("触面", [67, 91, 115]),    # pads seat on the end faces, grasp closes
    ("抬起", [160, 190, 220]),  # box lifted off the table, held aloft
]


def extract(frame_no: int) -> pathlib.Path:
    dst = OUT / f"f_{frame_no:04d}.jpg"
    if dst.exists() and dst.stat().st_size > 5_000:
        return dst
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(SRC),
         "-vf", f"select=eq(n\\,{frame_no - 1})", "-frames:v", "1",
         "-q:v", "2", str(dst)],
        check=True)
    return dst


def main() -> None:
    if not SRC.exists():
        raise SystemExit(f"missing source video {SRC}")
    OUT.mkdir(parents=True, exist_ok=True)
    files: list[str] = []
    for name, nums in SEGMENTS:
        for n in nums:
            p = extract(n)
            files.append(str(p.relative_to(REPO)))
    manifest = {
        "fps": FPS,
        "src_video": str(SRC.relative_to(REPO)),
        "video_frames": files,
        "segments": [{"name": name,
                      "files": [f"docs/video_frames/f_{n:04d}.jpg"
                                for n in nums]}
                     for name, nums in SEGMENTS],
    }
    (OUT / "manifest.json").write_text(
        json.dumps(manifest, indent=1, ensure_ascii=False))
    print(f"extracted {len(files)} frames -> {OUT}")


if __name__ == "__main__":
    sys.exit(main())
