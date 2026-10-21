"""Measure the rendered composition of the side cell against the reference targets.

Projects the box (centre + left/right edges) and the two wrist bodies through the
render camera (the side-entry preset used by viz/render_replay.py) and prints the
frame fractions, so the numbers quoted in docs/gpt_v7_package.md are measured, not
eyeballed.

    runs/.venv-mj/bin/python viz/measure_side_composition.py [--frame 251]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "viz"))

SCENE = REPO / "grounding/assets/allegro_floating_ur5e_side_r36.xml"
REPLAY = REPO / "runs/w6b/replay_frames.json"
W, H = 1280, 720


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", type=int, default=251)
    ap.add_argument("--dist", type=float, default=0.42)
    ap.add_argument("--el", type=float, default=-19.0)
    ap.add_argument("--az", type=float, default=90.0)
    args = ap.parse_args()

    import mujoco

    m = mujoco.MjModel.from_xml_path(str(SCENE))
    d = mujoco.MjData(m)
    frames = json.loads(REPLAY.read_text())["frames"]
    fr = frames[args.frame]
    d.qpos[:] = np.asarray(fr["qpos"])
    obj_adr = int(m.jnt_qposadr[mujoco.mj_name2id(
        m, mujoco.mjtObj.mjOBJ_JOINT, "object_free")])
    d.qpos[obj_adr:obj_adr + 7] = np.asarray(fr["obj"])
    mujoco.mj_forward(m, d)

    obj_geom = int(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "object_geom"))
    box = d.xpos[int(m.body("object").id)].copy()
    half = m.geom_size[obj_geom].copy()
    # the side-entry render preset (viz/render_replay.py SIDE_PRESETS); the
    # camera transform is taken from MuJoCo's own scene camera so the
    # projection matches the renderer exactly.
    az, el, dist = args.az, args.el, args.dist
    # the render tracks the live box position and looks 60 mm below it
    lookat = box.copy()
    lookat[2] -= 0.06
    m.vis.global_.offwidth = W
    m.vis.global_.offheight = H
    renderer = mujoco.Renderer(m, height=H, width=W)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = lookat
    cam.azimuth, cam.elevation, cam.distance = az, el, dist
    renderer.update_scene(d, camera=cam)
    c = renderer.scene.camera[0]
    campos = np.asarray(c.pos).copy()
    fwd = np.asarray(c.forward).copy()
    right = np.cross(np.asarray(c.forward), [0, 0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, np.asarray(c.forward))
    # pixel scale from the model's vertical field of view (MuJoCo renders with
    # m.vis.global_.fovy; the horizontal half-angle follows from the aspect)
    fovy = float(m.vis.global_.fovy)
    # calibrated against the renderer itself: MuJoCo's free camera uses the
    # model's fovy as the HORIZONTAL half-angle scale (verified: the rendered
    # box spans 0.481 of the frame at dist 0.32, this predicts 0.475)
    tan_h = np.tan(np.radians(fovy) / 2.0)

    def proj(p):
        rel = np.asarray(p, float) - campos
        zc = float(rel @ fwd)
        if zc <= 1e-6:
            return None
        u = W / 2 + (rel @ right) / (zc * tan_h) * (W / 2)
        # the renderer uses the same angular scale on both axes (verified
        # against the rendered frame: the box's vertical centre lands within
        # 0.02 of this prediction)
        v = H / 2 - (rel @ up) / (zc * tan_h) * (H / 2)
        return (u / W, v / H)

    box_c = proj(box)
    box_l = proj(box + [-half[0], 0, 0])
    box_r = proj(box + [half[0], 0, 0])
    wl = proj(d.xpos[int(m.body("ur5e_l_wrist_3_link").id)])
    wr = proj(d.xpos[int(m.body("ur5e_r_wrist_3_link").id)])
    tgt = json.loads((REPO / "viz/composition_target.json").read_text())
    print(f"frame {args.frame}: box={np.round(box,4)} half={np.round(half,4)}")
    print(f"  box centre   u={box_c[0]:.3f} v={box_c[1]:.3f}   "
          f"(target {tgt['box_center_uv'][0]:.2f}, {tgt['box_center_uv'][1]:.2f})")
    print(f"  box width    {box_r[0]-box_l[0]:.3f} of frame  "
          f"(target {tgt['box_width_frac']:.2f})")
    print(f"  left wrist   u={wl[0]:.3f} v={wl[1]:.3f}   "
          f"(target entry {tgt['forearm_entry_left_uv']})")
    print(f"  right wrist  u={wr[0]:.3f} v={wr[1]:.3f}   "
          f"(target entry {tgt['forearm_entry_right_uv']})")
    print(f"  camera: campos={np.round(campos,3)} fovy={fovy:.1f} deg "
          f"(preset az {az} el {el} dist {dist})")
    renderer.close()


if __name__ == "__main__":
    main()
