"""Offscreen-render every replay frame of viz/panel_data_v2.json with MuJoCo.

For each replay frame: write the full qpos, then place the object free joint
from ``frame["obj"]`` before mj_forward.  This is the trap pinned in the W6d
contract: composed replay qpos can carry the object joint zeroed (the W6a
gate-3 configuration does exactly that), so the object must never be trusted
from qpos alone.

Camera presets (free camera, lookat tracks the live object position):
  front 正面   azimuth 90  -- mimics the human clip: box long side on,
                              hands entering from left/right ends
  side  侧面   azimuth 0   -- end-face view, one hand toward the camera
  close 特写   azimuth 55  -- tight on the grasp

    runs/.venv-mj/bin/python viz/render_replay.py
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
from PIL import Image

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

DATA = REPO / "viz/panel_data_v2.json"
OUT = REPO / "docs/replay_frames"

WIDTH, HEIGHT = 1280, 720  # match the source clip so both panes scale identically

PRESETS = [
    # (key, 中文, azimuth deg, elevation deg, distance m)
    # v5 scene adds the gantry (rail runs along y at z~1.015): dead-on az 90
    # puts the lens against the rail's end face (renders as a giant cube), so
    # the presets view from slightly off-axis -- gantry stays in frame without
    # blocking the lens.  Verified against reset + gate3 frames.
    ("front", "正面", 105.0, -30.0, 1.00),
    ("side", "侧面", -25.0, -20.0, 0.62),
    # the money shot: on-axis profile from inside the slab footprint -- both
    # hands mirrored on the box ends, rail end passes behind the lens
    ("close", "特写", 90.0, -26.0, 0.45),
]


def main() -> None:
    import mujoco
    import scene_fidelity

    payload = json.loads(DATA.read_text())
    frames = payload["replay"]["frames"]
    OUT.mkdir(parents=True, exist_ok=True)

    # rendering model = scene XML + scene-fidelity visual layer (MjSpec,
    # physics-invariant: same nq/nv/nbody/njnt/qpos addresses -- see
    # scene_fidelity.py).  The scene follows the replay qpos width
    # (floating=65, ur5e=59).  The env/physics stack is NOT involved here.
    import sys
    if str(REPO / "viz") not in sys.path:
        sys.path.insert(0, str(REPO / "viz"))
    scene_xml = scene_fidelity.scene_for_qpos_width(len(frames[0]["qpos"]))[1]
    m = scene_fidelity.build_fidelity_model(scene_xml)
    d = mujoco.MjData(m)
    # lighting hierarchy: warm key + cool fill (the v17-era global shadow-off
    # is reverted -- no more dome spheres to burn black discs); mild ambient
    # higher ambient lifts the big structure shadows (this build has no soft
    # shadows -- ambient is the only softening knob) while keeping hierarchy
    m.light_ambient[:] = 0.38
    m.light_diffuse[0] = 0.46 * np.array([1.00, 0.93, 0.82])  # warm key
    m.light_diffuse[1] = 0.30 * np.array([0.88, 0.93, 1.05])  # cool fill
    m.light_specular[0] = 0.22
    m.light_specular[1] = 0.10
    # the frozen MJCF ships a 640 px offscreen buffer; widen it in memory so
    # the panel's sim panes come out at the source clip's 1280x720
    m.vis.global_.offwidth = WIDTH
    m.vis.global_.offheight = HEIGHT
    obj_adr = int(m.jnt_qposadr[mujoco.mj_name2id(
        m, mujoco.mjtObj.mjOBJ_JOINT, "object_free")])
    obj_body = int(m.body("object").id)
    # v20.1: replay frames carry the FULL model qpos (65 = 6 arm/pad angles
    # + 51-dim base) -- the executed arm/pad angles ship with the data, so
    # the renderer writes qpos directly and no IK runs here.  The object
    # free joint is still placed from frame["obj"] (the pinned trap).

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    renderer = mujoco.Renderer(m, height=HEIGHT, width=WIDTH)
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0

    renders: dict[str, list[str]] = {}
    for key, _cn, az, el, dist in PRESETS:
        outs = []
        for i, fr in enumerate(frames):
            d.qpos[:] = np.asarray(fr["qpos"])
            # --- the trap: object comes from obj, never from qpos -------
            d.qpos[obj_adr:obj_adr + 7] = np.asarray(fr["obj"])
            mujoco.mj_forward(m, d)

            cam.lookat[:] = d.xpos[obj_body]
            cam.azimuth, cam.elevation, cam.distance = az, el, dist
            renderer.update_scene(d, camera=cam)
            img = renderer.render()
            dst = OUT / f"{key}_r{i:04d}.png"
            Image.fromarray(img).save(dst)
            outs.append(str(dst.relative_to(REPO)))
        renders[key] = outs
    renderer.close()

    payload["meta"]["renders"] = {
        "dir": "docs/replay_frames",
        "presets": [{"key": k, "label": cn} for k, cn, *_ in PRESETS],
        "frames": renders,
    }
    DATA.write_text(json.dumps(payload, indent=1, ensure_ascii=False))
    (REPO / "viz/panel_data_v2.js").write_text(
        "window.PANEL_DATA_V2 = " + json.dumps(payload, ensure_ascii=False) + ";\n")
    n = len(frames)
    print(f"rendered {len(PRESETS)} presets x {n} frames -> {OUT}")


if __name__ == "__main__":
    main()
