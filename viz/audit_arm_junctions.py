"""v17 arm-chain junction audit (per-frame, per-side, programmatic).

v17 puts the arm segments back on per-body housings welded to the hands
(arm_r_forearm/elbow/upper -- rigid with the hand, so the hand junction
cannot open) and turns the shoulder into two big SPHERE SHELLS on the gantry
(gantry_junction_r R=0.22 m, gantry_junction_l R=0.19 m).  The audit
criterion is therefore COVERAGE: the upper-arm tip must stay inside the
sphere (|tip - center| <= R, margin >= 0) so the visual reads as the arm
rotating inside the shoulder housing rather than floating next to it.

Junctions audited (per side):
  J1 forearm capsule centre -> hand_root   welded (must be frame-invariant)
  J2 upper-arm tip coverage margin: R - |tip - dome centre|   MOVING
  J3 dome centre -> gantry_beam_x          static (mount offset)

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


def capsule_ends(g: int, m, d) -> tuple[np.ndarray, np.ndarray]:
    """World endpoints of a capsule (size [radius, half-length], local z)."""
    c = d.geom_xpos[g]
    axis = d.geom_xmat[g].reshape(3, 3)[:, 2]
    hl = float(m.geom_size[g][1])
    return c - axis * hl, c + axis * hl


def main() -> None:
    import mujoco
    import scene_fidelity

    m = scene_fidelity.build_fidelity_model()
    d = mujoco.MjData(m)
    data = json.loads((REPO / "viz/panel_data_v2.json").read_text())
    frames = data["replay"]["frames"]

    gid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)
    bid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)

    arms = {}
    for side, sfx in (("right", "_r"), ("left", "_l")):
        dome = gid(f"gantry_junction{sfx}")
        arms[side] = {
            "forearm": gid(f"arm{sfx}_forearm"),
            "upper": gid(f"arm{sfx}_upper"),
            "dome": dome,
            "dome_R": float(m.geom_size[dome][0]),
            "dome_c": np.array(m.geom_pos[dome], dtype=float),
            "hand": bid("hand_root" if side == "right" else "hand_root_left"),
        }
    beam = gid("gantry_beam_x")

    j1 = {s: [] for s in arms}
    j2 = {s: [] for s in arms}
    j3 = {s: [] for s in arms}
    for fr in frames:
        d.qpos[:] = np.asarray(fr["qpos"])
        d.qpos[0:7] = np.asarray(fr["obj"])  # object-placement trap pinned in W6d
        mujoco.mj_forward(m, d)
        for s, parts in arms.items():
            hp = d.xpos[parts["hand"]]
            fc = d.geom_xpos[parts["forearm"]]
            j1[s].append(float(np.linalg.norm(hp - fc)))
            e1, e2 = capsule_ends(parts["upper"], m, d)
            c = parts["dome_c"]
            tip = min((e1, e2), key=lambda e: np.linalg.norm(e - c))
            j2[s].append(parts["dome_R"] - float(np.linalg.norm(tip - c)))
            j3[s].append(float(np.linalg.norm(c - d.geom_xpos[beam])))

    def stat(v, unit=1e3):
        return f"min {min(v)*unit:7.1f} | max {max(v)*unit:8.1f}"

    lines = ["J1 forearm centre -> hand_root (welded -- must be invariant, mm)"]
    for s in arms:
        lines.append(f"  {s:5s} {stat(j1[s])}")
    lines.append("J2 upper-tip coverage margin = R - |tip-centre| (mm, >=0 covered)")
    for s in arms:
        lines.append(f"  {s:5s} {stat(j2[s])}")
    lines.append("J3 dome centre -> x-beam (static mount offset, mm)")
    for s in arms:
        lines.append(f"  {s:5s} {stat(j3[s])}")

    lines += ["", "per-frame J2 coverage margin (mm, >=0 = covered):",
              "  frame    " + "  ".join(f"{f:>7d}" for f in SAMPLES)]
    for s in arms:
        vals = "  ".join(f"{j2[s][f]*1e3:7.1f}" for f in SAMPLES)
        lines.append(f"  {s:5s}   {vals}")
    n_cover_r = sum(1 for v in j2["right"] if v >= 0)
    n_cover_l = sum(1 for v in j2["left"] if v >= 0)
    lines.append(f"  coverage: right {n_cover_r}/390 | left {n_cover_l}/390 frames")

    out = "\n".join(lines)
    print(out)
    out_dir = REPO / "runs/w6d"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "v17_junction_audit.txt").write_text(out + "\n")
    print("\nsaved -> runs/w6d/v17_junction_audit.txt")


if __name__ == "__main__":
    main()
