"""Ego-camera composition solver: r27 side-entry scene, source frame 252.

Solves the camera (behind-box distance, lens) against Main's measured
composition targets in viz/composition_target.json, then prints the
normalized projection table.

    runs/.venv-mj/bin/python viz/camera_calibration.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "viz"))
import scene_fidelity as sf  # noqa: E402

W, H, SENSOR = 1280, 720, 36.0


def main() -> None:
    tgt = json.loads((REPO / "viz/composition_target.json").read_text())
    XML = sf.SCENE_UR5E
    m = mujoco.MjModel.from_xml_path(str(XML))
    d = mujoco.MjData(m)
    data = json.loads((REPO / "viz/panel_data_v2.json").read_text())
    fr = data["replay"]["frames"][251]  # sim f252 <-> source frame 252
    d.qpos[:] = np.asarray(fr["qpos"])
    d.qpos[0:7] = np.asarray(fr["obj"])
    mujoco.mj_forward(m, d)

    box = d.geom_xpos[mujoco.mj_name2id(
        m, mujoco.mjtObj.mjOBJ_GEOM, "skin_floral")].copy()
    wl = d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY,
                                  "ur5e_l_wrist_3_link")]
    wr = d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY,
                                  "ur5e_r_wrist_3_link")]
    edge = np.array([0.0, 0.45, 0.70])
    print("box", np.round(box, 3), "| wristL", np.round(wl, 3),
          "| wristR", np.round(wr, 3))

    def project(pts, campos, target, lens):
        P = np.asarray(campos, float)
        T = np.asarray(target, float)
        fwd = T - P
        fwd /= np.linalg.norm(fwd)
        right = np.cross(fwd, [0, 0, 1.0])
        right /= np.linalg.norm(right)
        up = np.cross(right, fwd)
        fpx = lens / SENSOR * W
        out = []
        for p in pts:
            rel = np.asarray(p, float) - P
            zc = rel @ fwd
            if zc <= 1e-6:
                out.append(None)
                continue
            u = W / 2 + fpx * (rel @ right) / zc
            v = H / 2 - fpx * (rel @ up) / zc
            out.append((u / W, v / H))
        return out

    def evaluate(campos, target, lens):
        pts = [box, box + [-0.12, 0, 0], box + [0.12, 0, 0], wl, wr, edge]
        pr = project(pts, campos, target, lens)
        b, bl, br_, wl_, wr_, e = pr
        if any(p is None for p in pr):
            return 9.0, pr
        width_frac = br_[0] - bl[0]
        loss = (abs(b[0] - 0.50) / 0.05 + abs(b[1] - 0.38) / 0.05
                + abs(width_frac - 0.30) / 0.05
                + abs(wl_[0] - 0.02) / 0.05 + abs(wr_[0] - 0.98) / 0.05
                + abs(wl_[1] - 0.55) / 0.05 + abs(wr_[1] - 0.60) / 0.05)
        return loss / 7, pr

    best = (9.0, None, None)
    # the UR5E side-entry carry height differs from the human's — let z float
    # and target the composition (box v, wrist entries) directly
    for D in (0.55, 0.65, 0.75, 0.85, 0.95):
        for lens in (22, 26, 30, 35):
            for z in (0.95, 1.02, 1.08, 1.15):
                for yback in (0.30, 0.40, 0.50):
                    campos = (box[0] + 0.02, box[1] - yback, z)
                    lookat = (box[0], box[1], box[2])
                    L, pr = evaluate(campos, lookat, lens)
                    if L < best[0]:
                        best = (L, campos, (lookat, lens, pr))
    L, campos, (lookat, lens, pr) = best
    print(f"best loss {L:.3f} | campos {np.round(campos, 3)} | lens {lens}mm")
    names = ["box", "boxL", "boxR", "wristL", "wristR", "tableEdge"]
    for nm, p in zip(names, pr):
        print(f"  {nm:10s} u={p[0]:.3f} v={p[1]:.3f}")
    hd = math.hypot(campos[0] - box[0], campos[1] - box[1])
    pitch = math.degrees(math.atan2(0.88 - 0.78, hd))
    print(f"horizontal distance {hd:.3f} m | pitch down ~{pitch:.1f} deg")
    print(f"fov_h ~{2*math.degrees(math.atan(SENSOR/2/lens)):.1f} deg")


if __name__ == "__main__":
    main()
