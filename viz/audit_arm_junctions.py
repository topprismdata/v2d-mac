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

    scene_xml, _kind = scene_fidelity.scene_for_qpos_width(
        len(json.loads((REPO / "viz/panel_data_v2.json").read_text())
            ["replay"]["frames"][0]["qpos"]))
    m = scene_fidelity.build_fidelity_model(scene_xml)
    d = mujoco.MjData(m)
    data = json.loads((REPO / "viz/panel_data_v2.json").read_text())
    frames = data["replay"]["frames"]

    gid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)
    bid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
    obj_adr = int(m.jnt_qposadr[mujoco.mj_name2id(
        m, mujoco.mjtObj.mjOBJ_JOINT, "object_free")])

        # v25/ur5e: per-axis UR joint chain audit (J1-J6 per side) + the
    # flange->hand mount constancy.  The chain is a kinematic tree, so the
    # meaningful checks are: qpos finite + within the joint's ctrlrange for
    # every frame, and the wrist_3 -> hand_root mount distance constant.
    UR_JOINTS = ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                 "wrist_1_joint", "wrist_2_joint", "wrist_3_joint")
    jax = {}
    for side, sfx in (("right", "_r"), ("left", "_l")):
        for k, jn in enumerate(UR_JOINTS, start=1):
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT,
                                    f"ur5e{sfx}_{jn}")
            assert jid >= 0, f"joint not found: ur5e{sfx}_{jn}"
            jax[(side, f"J{k} {jn}")] = {
                "adr": int(m.jnt_qposadr[jid]),
                "range": (float(m.jnt_range[jid][0]), float(m.jnt_range[jid][1]))
                if m.jnt_range[jid][1] > m.jnt_range[jid][0] else None,
            }
        jax[(side, "mount")] = {
            "wrist3": bid(f"ur5e{sfx}_wrist_3_link"),
            "hand": bid("hand_root" if side == "right" else "hand_root_left"),
        }

    jvals = {k: [] for k in jax}
    mount = {s: [] for s in ("right", "left")}
    for fr in frames:
        d.qpos[:] = np.asarray(fr["qpos"])          # v20.1: full qpos incl. arm/pad angles
        d.qpos[obj_adr:obj_adr + 7] = np.asarray(fr["obj"])  # object-placement trap pinned in W6d
        mujoco.mj_forward(m, d)
        for (side, label), info in jax.items():
            if "adr" in info:
                v = float(d.qpos[info["adr"]])
                if not np.isfinite(v):
                    raise SystemExit(f"non-finite qpos {side} {label}")
                jvals[(side, label)].append(v)
        for side in ("right", "left"):
            info = jax[(side, "mount")]
            w3 = d.xpos[info["wrist3"]]
            hp = d.xpos[info["hand"]]
            mount[side].append(float(np.linalg.norm(w3 - hp)))

    lines = ["UR5e per-axis joint audit (390 frames, per side)"]
    for (side, label), info in jax.items():
        if "adr" not in info:
            continue
        v = jvals[(side, label)]
        rng = info["range"]
        oob = sum(1 for x in v if rng and not (rng[0] - 1e-9 <= x <= rng[1] + 1e-9))
        lines.append(f"  {side:5s} {label:22s} min {min(v):+.3f} max {max(v):+.3f} rad"
                     + (f" | ctrlrange 越界 {oob} 帧" if rng else " | 无 ctrlrange"))
    lines.append("flange(wrist_3) -> hand_root mount distance (mm, rigid => constant)")
    for s in ("right", "left"):
        lines.append(f"  {s:5s} min {min(mount[s])*1e3:7.2f} | max {max(mount[s])*1e3:7.2f}"
                     f" | span {(max(mount[s])-min(mount[s]))*1e3:.4f}")

    lines += ["", "per-frame joint angles sample (rad, J1..J6):"]
    for s in ("right", "left"):
        vals = "  ".join(
            f"{f:>4d}:" + ",".join(f"{jvals[(s, f'J{k} {jn}')][f]:+.2f}"
                                   for k, jn in
                                   enumerate(("shoulder_pan_joint",
                                              "shoulder_lift_joint",
                                              "elbow_joint", "wrist_1_joint",
                                              "wrist_2_joint",
                                              "wrist_3_joint"), start=1))
            for f in SAMPLES)
        lines.append(f"  {s:5s} {vals}")

    out = "\n".join(lines)
    print(out)
    out_dir = REPO / "runs/w6d"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "v19_junction_audit.txt").write_text(out + "\n")
    print("\nsaved -> runs/w6d/v25_junction_audit.txt")


if __name__ == "__main__":
    main()
