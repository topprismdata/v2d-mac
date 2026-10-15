"""Render the sim's winning trajectory at 4 stages + compose the side-by-side
comparison against the original human video frames.

    python runs/w5/render_compare.py   ->  /tmp/v2d_compare.png
"""
from __future__ import annotations

import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

OUT = pathlib.Path("/tmp/v2d_compare.png")


def main() -> None:
    import mujoco
    from PIL import Image, ImageDraw

    import grounding.envs.lift_hold_env as lhe
    import grounding.collect_hook_demos as chd
    from grounding.reachable_scan import run_traj

    env = lhe.LiftHoldEnv(episode_len=400, domain_rand=False)
    m, d = env.model, env.data

    # the winning configuration (limit-covering grid, 1/15625 on the bare box)
    cfg = json.loads(json.dumps(chd.BASE_CFG))
    cfg["wrist_world"] = [0.09, -0.09, 0.757]
    cfg["rot"] = [1.8, -0.6, -1.5]
    a = list(cfg["close"])
    a[12:16] = [0.0, -0.7, -0.7, -0.7]
    cfg["close"] = a
    cfg["close_params"] = [1.0, -0.7, 0.0]
    cfg["n_desc"] = 60
    cfg["lift_rate"] = 0.004

    m.vis.global_.offwidth = 960
    m.vis.global_.offheight = 540
    renderer = mujoco.Renderer(m, height=540, width=960)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.0, 0.0, 0.80]
    cam.distance = 0.9
    cam.azimuth = 155
    cam.elevation = -18

    frames: dict[str, np.ndarray] = {}
    steps: list[int] = []

    env.step_wrapped = False

    def snap(name: str) -> None:
        mujoco.mj_forward(m, d)
        renderer.update_scene(d, camera=cam)
        frames[name] = renderer.render().copy()

    orig_step = env.step

    def counting_step(action):
        r = orig_step(action)
        steps.append(len(steps))
        if len(steps) == 45 and "approach" not in frames:
            snap("approach")
        if len(steps) == 175 and "grip" not in frames:
            snap("grip")
        return r

    env.step = counting_step
    run_traj(env, cfg)

    # lift peak + hold: find them from the trace
    lift = env.trace["lift"]
    peak = int(max(range(len(lift)), key=lambda i: lift[i]))
    # re-render at peak and hold by rewinding is not possible; instead re-run
    # with snap points at those steps
    env2 = lhe.LiftHoldEnv(episode_len=400, domain_rand=False)
    orig2 = env2.step

    def counting2(action):
        r = orig2(action)
        steps.append(len(steps))
        if len(steps) >= peak and "peak" not in frames:
            mujoco.mj_forward(m, d)
            renderer.update_scene(d, camera=cam)
            frames["peak"] = renderer.render().copy()
        if len(steps) >= min(len(lift) - 1, peak + 60) and "hold" not in frames:
            mujoco.mj_forward(m, d)
            renderer.update_scene(d, camera=cam)
            frames["hold"] = renderer.render().copy()
        return r

    env2.step = counting2
    run_traj(env2, cfg)

    stages = [
        ("approach", "S3: 接近", "human f_01.jpg"),
        ("grip", "S4: 合拢", "human f_02.jpg"),
        ("peak", "S5: 提升峰值", "human f_04.jpg"),
        ("hold", "S6: 保持", "human f_05.jpg"),
    ]
    human_dir = pathlib.Path("/tmp/v2d_frames")
    human_map = {"human f_01.jpg": "f_01.jpg", "human f_02.jpg": "f_02.jpg",
                 "human f_04.jpg": "f_04.jpg", "human f_05.jpg": "f_05.jpg"}

    rows = []
    for key, label, hfile in stages:
        hf = human_dir / human_map[hfile]
        if not hf.exists():
            continue
        him = Image.open(hf).resize((640, 360))
        sim = frames.get(key)
        if sim is None:
            sim = np.full((540, 960, 3), 30, dtype=np.uint8)
        simim = Image.fromarray(sim).resize((640, 360))
        row = Image.new("RGB", (1284, 380), (18, 18, 18))
        row.paste(him, (0, 10))
        row.paste(simim, (644, 10))
        dd = ImageDraw.Draw(row)
        dd.text((8, 0), f"HUMAN  {label}", fill=(120, 220, 120))
        dd.text((652, 0), f"SIM (single Allegro)  {label}", fill=(120, 180, 255))
        rows.append(row)

    sheet = Image.new("RGB", (1284, 380 * len(rows) + 40), (10, 10, 10))
    for i, r in enumerate(rows):
        sheet.paste(r, (0, i * 380))
    dd = ImageDraw.Draw(sheet)
    dd.text((8, 380 * len(rows) + 12),
            "v2d-mac: original human demo (left) vs single-hand sim (right) - 2026-10-15",
            fill=(200, 200, 200))
    sheet.save(OUT)
    print(f"saved {OUT} ({len(rows)} stage rows); sim frames: {list(frames)}")


if __name__ == "__main__":
    main()
