"""Preview the side-entry layout from the reference clip's camera.

Renders a few replay frames of the UR5e side-mount cell with a free camera
placed against the measured composition targets in viz/composition_target.json
(box centre 50%/38%, box width 30% of frame, forearm entries at 0%/55% and
100%/60%, camera 15-20 deg down, table edge out of frame).

    runs/.venv-mj/bin/python viz/preview_side_layout.py [--frames 120,240,360]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
from PIL import Image

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "viz"))

SCENE = REPO / "grounding/assets/allegro_floating_ur5e_side_r36.xml"
REPLAY = REPO / "runs/w6b/replay_frames.json"
OUT = REPO / "runs/w6b/preview_side"
W, H = 1280, 720


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="60,180,300,389")
    ap.add_argument("--dist", type=float, default=0.90)
    ap.add_argument("--height", type=float, default=0.88, help="absolute z")
    ap.add_argument("--pitch", type=float, default=17.0)
    ap.add_argument("--azimuth", type=float, default=0.0)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    import mujoco

    m = mujoco.MjModel.from_xml_path(str(SCENE))
    d = mujoco.MjData(m)
    m.vis.global_.offwidth = W
    m.vis.global_.offheight = H
    obj_body = int(m.body("object").id)
    obj_adr = int(m.jnt_qposadr[mujoco.mj_name2id(
        m, mujoco.mjtObj.mjOBJ_JOINT, "object_free")])

    frames = json.loads(REPLAY.read_text())["frames"]
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    renderer = mujoco.Renderer(m, height=H, width=W)
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE

    for spec in args.frames.split(","):
        i = int(spec)
        fr = frames[i]
        d.qpos[:] = np.asarray(fr["qpos"])
        d.qpos[obj_adr:obj_adr + 7] = np.asarray(fr["obj"])
        mujoco.mj_forward(m, d)
        box = d.xpos[obj_body].copy()
        # camera on -y (the box's long axis runs left-right in frame), above
        # the box, pitched down by the requested angle
        hz = box[2] + args.dist * np.tan(np.radians(args.pitch))
        campos = np.array([box[0], box[1] - args.dist, hz])
        lookat = box.copy()
        fwd = lookat - campos
        fwd /= np.linalg.norm(fwd)
        right = np.cross(fwd, [0, 0, 1.0])
        right /= np.linalg.norm(right)
        up = np.cross(right, fwd)
        # MjvCamera free mode: lookat/azimuth/elevation/distance
        cam.lookat[:] = lookat
        cam.distance = float(np.linalg.norm(campos - lookat))
        cam.azimuth = float(np.degrees(np.arctan2(-fwd[1], fwd[0])) + args.azimuth)
        cam.elevation = float(np.degrees(np.arcsin(fwd[2])))
        renderer.update_scene(d, camera=cam)
        img = renderer.render()
        dst = out / f"f{i:04d}_az{args.azimuth:.0f}_p{args.pitch:.0f}.png"
        Image.fromarray(img).save(dst)
        print(f"frame {i}: box={np.round(box,3)} campos={np.round(campos,3)} "
              f"-> {dst}")
    renderer.close()


if __name__ == "__main__":
    main()
