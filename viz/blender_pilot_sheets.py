"""Assemble the Blender pilot deliverables: 9-view grid + comparison sheet.

    runs/.venv-mj/bin/python viz/blender_pilot_sheets.py
"""
from __future__ import annotations

import pathlib
import subprocess

from PIL import Image, ImageDraw

REPO = pathlib.Path(__file__).resolve().parents[1]
PILOT = REPO / "viz/blender/pilot"


def frame(path: str, size=(1280, 720)) -> Image.Image:
    return Image.open(path).convert("RGB").resize(size)


def main() -> None:
    frames = [60, 190, 300]
    presets = ["front", "side", "close"]
    grid = Image.new("RGB", (1280 * 3, (720 + 40) * 3))
    d = ImageDraw.Draw(grid)
    for row, f in enumerate(frames):
        for col, k in enumerate(presets):
            p = PILOT / f"{k}_f{f:04d}.png"
            y = row * (720 + 40)
            d.rectangle([col * 1280, y, col * 1280 + 1279, y + 38],
                        fill=(16, 20, 24))
            d.text((col * 1280 + 12, y + 11),
                   f"Blender Cycles · f{f:03d} · {k} 机位 · 产品级贴皮试点",
                   fill=(232, 237, 242))
            grid.paste(frame(str(p)), (col * 1280, y + 40))
    grid.save(REPO / "docs/images/blender_pilot_grid.png")

    # comparison: original f90 vs Blender close f90 vs MuJoCo close f90
    subprocess.run(["ffmpeg", "-y", "-v", "error",
                    "-i", "runs/p0/assets/tissue_box.mp4",
                    "-vf", "select=eq(n\\,89)", "-frames:v", "1", "-q:v", "2",
                    "/tmp/orig_f90.jpg"], check=True, cwd=REPO)
    rows = [("原片 · tissue_box.mp4 第 90 帧（t=2.97 s）",
             "/tmp/orig_f90.jpg"),
            ("Blender Cycles · close 机位 · replay 第 90 帧（试点）",
             str(PILOT / "close_f0090.png")),
            ("MuJoCo 光栅化 · close 机位 · replay 第 90 帧（对照）",
             str(REPO / "docs/replay_frames/close_r0090.png"))]
    H = 44 + len(rows) * (720 + 44)
    sheet = Image.new("RGB", (1280, H))
    d = ImageDraw.Draw(sheet)
    y = 0
    for label, path in rows:
        d.rectangle([0, y, 1280, y + 43], fill=(16, 20, 24))
        d.text((16, y + 14), label, fill=(232, 237, 242))
        sheet.paste(frame(path), (0, y + 44))
        y += 720 + 44
    sheet.save(REPO / "docs/images/blender_compare.png")
    print("saved docs/images/blender_pilot_grid.png + blender_compare.png")


if __name__ == "__main__":
    main()
