"""v19 arm-chain junction audit (per-frame, per-side, programmatic).

v19: 3-DoF articulated arms (gantry -> yaw disc -> upper capsule -> elbow
disc -> forearm capsule -> wrist sphere), angles written per rendered frame
by the same closed-form IK the env runs (lift_hold_env._write_arm_ik_ctrl).
Because the IK is analytic and written directly into qpos, the wrist sphere
must sit exactly on the hand root at EVERY frame -- servo lag cannot appear
in the render path.

Audited (per side, all 390 frames):
  J1 arm_wrist sphere centre -> hand_root   IK residual (verdict carrier)
  J2 elbow disc centre -> forearm far end   rigid hinge (invariant)
  J3 shoulder yaw disc -> upper far end     rigid hinge (invariant)

    runs/.venv-mj/bin/python viz/audit_arm_junctions.py
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "viz"))

SAMPLES = [0, 60, 90, 190, 300, 389]


def capsule_end(g: int, m, d, which: str) -> np.ndarray:
    """World endpoint of a capsule (size [radius, half-length], local z)."""
    c = d.geom_xpos[g]
    axis = d.geom_xmat[g].reshape(3, 3)[:, 2]
    hl = float(m.geom_size[g][1])
    return c + axis * (hl if which == "pos" else -hl)


def main() -> None:
    import mujoco
    import scene_fidelity

    m = scene_fidelity.build_fidelity_model()
    d = mujoco.MjData(m)
    data = json.loads((REPO / "viz/panel_data_v2.json").read_text())
    frames = data["replay"]["frames"]

    gid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)
    bid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
    obj_adr = int(m.jnt_qposadr[mujoco.mj_name2id(
        m, mujoco.mjtObj.mjOBJ_JOINT, "object_free")])

    arms = {}
    for side, sfx in (("right", "_r"), ("left", "_l")):
        arms[side] = {
            "forearm": gid(f"arm_forearm{sfx}"),
            "wrist": gid(f"arm_wrist{sfx}"),
            "upper": gid(f"arm_upper{sfx}"),
            "elbow_disc": gid(f"arm_elbow_disc{sfx}"),
            "yaw_disc": gid(f"arm_yaw_disc{sfx}"),
            "hand": bid("hand_root" if side == "right" else "hand_root_left"),
        }

    j1 = {s: [] for s in arms}
    j2 = {s: [] for s in arms}
    j3 = {s: [] for s in arms}
    for fr in frames:
        d.qpos[:] = np.asarray(fr["qpos"])          # v20.1: full qpos incl. arm/pad angles
        d.qpos[obj_adr:obj_adr + 7] = np.asarray(fr["obj"])  # object-placement trap pinned in W6d
        mujoco.mj_forward(m, d)
        for s, parts in arms.items():
            hp = d.xpos[parts["hand"]]
            wc = d.geom_xpos[parts["wrist"]]
            j1[s].append(float(np.linalg.norm(wc - hp)))
            # forearm far end (the wrist end) vs elbow disc centre
            fe = capsule_end(parts["forearm"], m, d, "neg")
            ec = d.geom_xpos[parts["elbow_disc"]]
            j2[s].append(float(np.linalg.norm(fe - ec)))
            # upper far end vs yaw disc centre
            ue = capsule_end(parts["upper"], m, d, "neg")
            yc = d.geom_xpos[parts["yaw_disc"]]
            j3[s].append(float(np.linalg.norm(ue - yc)))

    def stat(v):
        return f"min {min(v)*1e3:8.4f} mm | max {max(v)*1e3:8.4f} mm"

    lines = ["J1 arm_wrist centre -> hand_root (executed-trajectory follow residual)"]
    for s in arms:
        lines.append(f"  {s:5s} {stat(j1[s])}")
    lines.append("J2 forearm wrist-end -> elbow disc centre (rigid hinge, mm)")
    for s in arms:
        lines.append(f"  {s:5s} {stat(j2[s])}")
    lines.append("J3 upper elbow-end -> yaw disc centre (rigid hinge, mm)")
    for s in arms:
        lines.append(f"  {s:5s} {stat(j3[s])}")

    lines += ["", "per-frame J1 follow residual (mm):", "  frame   " +
              "  ".join(f"{f:>8d}" for f in SAMPLES)]
    for s in arms:
        vals = "  ".join(f"{j1[s][f]*1e3:8.5f}" for f in SAMPLES)
        lines.append(f"  {s:5s}  {vals}")

    out = "\n".join(lines)
    print(out)
    out_dir = REPO / "runs/w6d"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "v19_junction_audit.txt").write_text(out + "\n")
    print("\nsaved -> runs/w6d/v19_junction_audit.txt")


if __name__ == "__main__":
    main()
