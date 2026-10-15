"""Dump scene + gate-clean trajectory for the HTML evidence panel.

Replays the 1/15625 gate-clean top-hook trajectory (the exact cfg from the
limit-covering grid: BASE_CFG + seed patch, ct=-0.7, n_desc=60, lift_rate=0.004)
on the settled scene (XML 1ea0fdf5), recording per control frame:
  - every body's world pose (pos + quat) -> three.js animates boxes from this
  - lift, finger contacts, per-step reward
plus static scene geometry (boxes with world poses + sizes + rgba) and the
frozen-gate verdict + bearing decomposition at peak.

    python runs/w5/dump_panel_data.py   ->  viz/panel_data.json
"""
from __future__ import annotations

import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

OUT = REPO / "viz/panel_data.json"


def body_snapshot(m, d) -> list[dict]:
    """World pose of every body, for the renderer."""
    out = []
    for b in range(m.nbody):
        name = m.body(b).name
        if not name or name == "world":
            continue
        out.append({
            "name": name,
            "pos": [round(float(v), 5) for v in d.xpos[b]],
            "quat": [round(float(v), 6) for v in d.xquat[b]],
        })
    return out


def geom_static(m, d) -> list[dict]:
    """Static renderable geometry: type, size, world pose, color."""
    out = []
    for g in range(m.ngeom):
        gm = m.geom(g)
        gtype = int(gm.type[0]) if hasattr(gm.type, "__len__") else int(gm.type)
        types = {2: "sphere", 3: "capsule", 5: "cylinder", 6: "box", 7: "mesh"}
        size = [float(v) for v in m.geom_size[g]]
        rgba = [round(float(v), 3) for v in m.geom_rgba[g]]
        if gtype == 6:  # box: half-extents
            size = [round(v, 5) for v in m.geom_size[g]]
        elif gtype == 3:  # capsule: (radius, half-length)
            size = [round(float(m.geom_size[g][0]), 5),
                    round(float(m.geom_size[g][1]), 5)]
        elif gtype == 7:  # mesh: render as its world-aligned AABB box
            mid = int(m.geom_dataid[g])
            v0, vn = int(m.mesh_vertadr[mid]), int(m.mesh_vertnum[mid])
            verts = m.mesh_vert[v0:v0 + vn]
            size = [round(float(v), 5) for v in
                    (verts.max(axis=0) - verts.min(axis=0)) / 2.0]
        out.append({
            "name": m.geom(g).name or f"geom_{g}",
            "body": m.body(m.geom_bodyid[g]).name,
            "type": types.get(gtype, "box"),
            "size": size,
            "pos": [round(float(v), 5) for v in d.geom_xpos[g]],
            "quat": [round(float(v), 6) for v in
                     _mat_to_quat(d.geom_xmat[g].reshape(3, 3))],
            "rgba": rgba,
        })
    return out


def _mat_to_quat(mat: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix -> xyzw quaternion (shepperd's method)."""
    t = np.trace(mat)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        w, x, y, z = 0.25 * s, (mat[2, 1] - mat[1, 2]) / s, \
            (mat[0, 2] - mat[2, 0]) / s, (mat[1, 0] - mat[0, 1]) / s
    elif mat[0, 0] > mat[1, 1] and mat[0, 0] > mat[2, 2]:
        s = np.sqrt(1.0 + mat[0, 0] - mat[1, 1] - mat[2, 2]) * 2
        w, x, y, z = (mat[2, 1] - mat[1, 2]) / s, 0.25 * s, \
            (mat[0, 1] + mat[1, 0]) / s, (mat[0, 2] + mat[2, 0]) / s
    elif mat[1, 1] > mat[2, 2]:
        s = np.sqrt(1.0 + mat[1, 1] - mat[0, 0] - mat[2, 2]) * 2
        w, x, y, z = (mat[0, 2] - mat[2, 0]) / s, \
            (mat[0, 1] + mat[1, 0]) / s, 0.25 * s, (mat[1, 2] + mat[2, 1]) / s
    else:
        s = np.sqrt(1.0 + mat[2, 2] - mat[0, 0] - mat[1, 1]) * 2
        w, x, y, z = (mat[1, 0] - mat[0, 1]) / s, \
            (mat[0, 2] + mat[2, 0]) / s, (mat[1, 2] + mat[2, 1]) / s, 0.25 * s
    return np.array([x, y, z, w])


def main() -> None:
    import mujoco

    import grounding.envs.lift_hold_env as lhe
    from grounding.collect_hook_demos import BASE_CFG
    from grounding.reachable_scan import gate_eval, run_traj, \
        contact_load_decomposition

    # the exact winning configuration: limit-covering grid, non-wide branch
    cfg = json.loads(json.dumps(BASE_CFG))
    cfg["wrist_world"] = [0.09, -0.09, 0.757]
    cfg["rot"] = [1.8, -0.6, -1.5]
    a = list(cfg["close"])
    a[12:16] = [0.0, -0.7, -0.7, -0.7]
    cfg["close"] = a
    cfg["close_params"] = [1.0, -0.7, 0.0]
    cfg["n_desc"] = 60
    cfg["lift_rate"] = 0.004

    env = lhe.LiftHoldEnv(episode_len=400, domain_rand=False)
    env.record_bearing = True

    frames: list[dict] = []
    obs, act = None, None
    # run_traj drives env.step; to capture per-frame body poses we wrap step
    orig_step = env.step

    def recording_step(action):
        r = orig_step(action)
        frames.append({
            "b": body_snapshot(env.model, env.data),
            "lift": round(float(env.trace["lift"][-1]), 5),
            "reward": round(float(env.trace.get("reward", [0])[-1]), 2)
            if env.trace.get("reward") else 0.0,
        })
        return r

    env.step = recording_step
    obs, act = run_traj(env, cfg, record=True)
    ev = gate_eval(env)
    assert ev["gate_pass"], "replay of the winning configuration did not pass"
    decomp = contact_load_decomposition(env, act, ev["peak_step"])

    # static scene from the model (world poses at frame 0 are fine for the
    # static parts; hand bodies come from `frames`)
    m, d = env.model, env.data
    mujoco.mj_forward(m, d)
    static = [g for g in geom_static(m, d)
              if g["body"] in ("world", "table_slab", "pedestal", "floor")
              or g["name"] == "object_geom"]

    peak_step = int(ev["peak_step"])
    payload = {
        "meta": {
            "mjcf_commit": "1b0f13f",
            "xml_sha256": "1ea0fdf5c87ce5a161efcda7c597b0fc6647c9fe30bcf2f5d0bcb52402d24b89",
            "trajectory": "limit-covering grid winner, 1/15625, load top=1.0",
            "ctrl_hz": 50,
        },
        "static": static,
        "frames": frames,
        "peak_step": peak_step,
        "gate": {k: v for k, v in ev.items() if k != "finger_pairs"},
        "load_decomposition": decomp,
        "box_outline_m": [0.12, 0.24, 0.08],
        "six_smoke": [
            {"id": 1, "name": "SAM2 MPS", "verdict": "pass"},
            {"id": 2, "name": "VGGT-mps", "verdict": "pass"},
            {"id": 3, "name": "FoundPose", "verdict": "pass"},
            {"id": 4, "name": "HaMeR MPS", "verdict": "pass"},
            {"id": 5, "name": "dex-retarget", "verdict": "12/13",
             "note": "唯一红项 = 掌偏差 102mm → P1 retarget"},
            {"id": 6, "name": "MuJoCo PPO", "verdict": "in-progress",
             "note": "倒角扫描进行中"},
        ],
    }
    OUT.write_text(json.dumps(payload, indent=1, default=str))
    print(f"frames={len(frames)} peak={ev['peak_lift_m'] * 1000:.1f}mm "
          f"gate_pass={ev['gate_pass']} size={OUT.stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
