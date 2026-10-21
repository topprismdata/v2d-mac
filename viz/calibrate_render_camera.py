"""Calibrate the side-entry render camera against the reference composition.

Renders one frame at a few (distance, lookat-offset) pairs with the fidelity
model and measures the box's extent directly in the PNG (the purple skin),
then prints the pair whose box width / centre is closest to the reference
targets in viz/composition_target.json.

    runs/.venv-mj/bin/python viz/calibrate_render_camera.py
"""
from __future__ import annotations

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
W, H = 1280, 720
FRAME = 251


def measure_png(path):
    im = np.asarray(Image.open(path).convert("RGB")).astype(int)
    r, g, b = im[:, :, 0], im[:, :, 1], im[:, :, 2]
    mask = (r > 70) & (b > 90) & (g < r - 25) & (g < b - 25)
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return {
        "width": float((xs.max() - xs.min()) / W),
        "u": float((xs.min() + xs.max()) / 2 / W),
        "v": float((ys.min() + ys.max()) / 2 / H),
    }


def main() -> None:
    import mujoco
    import scene_fidelity as sf

    tgt = json.loads((REPO / "viz/composition_target.json").read_text())
    m = sf.build_fidelity_model(SCENE,
                                hide_bodies=("yaw_stop_posy", "yaw_stop_negy"))
    m.vis.global_.offwidth = W
    m.vis.global_.offheight = H
    d = mujoco.MjData(m)
    _src = pathlib.Path(__import__("os").environ.get("V2D_CALIB_FRAMES", str(REPLAY)))
    _payload = json.loads(_src.read_text())
    frames = _payload["replay"]["frames"] if "replay" in _payload else _payload["frames"]
    fr = frames[FRAME]
    d.qpos[:] = np.asarray(fr["qpos"])
    obj_adr = int(m.jnt_qposadr[mujoco.mj_name2id(
        m, mujoco.mjtObj.mjOBJ_JOINT, "object_free")])
    d.qpos[obj_adr:obj_adr + 7] = np.asarray(fr["obj"])
    mujoco.mj_forward(m, d)

    renderer = mujoco.Renderer(m, height=H, width=W)
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    out = pathlib.Path("/tmp/calib")
    out.mkdir(exist_ok=True)
    best = None
    for dist in (0.34, 0.36, 0.38, 0.40, 0.42):
        for off in (0.0, 0.01):
            cam.lookat[:] = d.xpos[int(m.body("object").id)]
            cam.lookat[2] -= off
            cam.azimuth, cam.elevation, cam.distance = 90.0, -14.0, dist
            renderer.update_scene(d, camera=cam)
            img = renderer.render()
            p = out / f"d{dist}_o{off}.png"
            Image.fromarray(img).save(p)
            meas = measure_png(p)
            if meas is None:
                continue
            err = (abs(meas["width"] - tgt["box_width_frac"]) / 0.02
                   + abs(meas["u"] - tgt["box_center_uv"][0]) / 0.02
                   + abs(meas["v"] - tgt["box_center_uv"][1]) / 0.02)
            print(f"dist={dist} lookat_off={off}: width={meas['width']:.3f} "
                  f"u={meas['u']:.3f} v={meas['v']:.3f} err={err:.2f}")
            if best is None or err < best[0]:
                best = (err, dist, off, meas)
    renderer.close()
    print(f"\nbest: dist={best[1]} lookat_off={best[2]} err={best[0]:.2f} "
          f"width={best[3]['width']:.3f} u={best[3]['u']:.3f} "
          f"v={best[3]['v']:.3f}")
    print(f"targets: width={tgt['box_width_frac']} "
          f"u={tgt['box_center_uv'][0]} v={tgt['box_center_uv'][1]}")


if __name__ == "__main__":
    main()
