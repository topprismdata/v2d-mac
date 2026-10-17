"""v16 arm-chain junction audit (per-frame, per-side, programmatic).

v16 redistributes the arm segments onto the GANTRY chain (gantry_forearm /
gantry_shoulder / gantry_zstage, welded under the fixed gantry body) while
the HANDS still move on their wrist slides.  Main's question: does the
static forearm detach from the moving hand during the replay?

Junctions audited (per side):
  J1 hand_root -> gantry_forearm        MOVING (the verdict carrier)
  J2 gantry_forearm -> gantry_shoulder  static (must be frame-invariant)
  J3 gantry_shoulder -> gantry_zstage   static
  J4 gantry_zstage -> gantry_beam_x     static

Distances are exact point-to-oriented-box (hand root / segment corners
clamped into the neighbour box's local frame).  Static junctions sample the
segment's 8 corners against the neighbour box surface, which upper-bounds
the true surface gap.

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


def point_box_dist(p: np.ndarray, g: int, m, d) -> float:
    """Exact distance from point p to geom g's oriented box surface."""
    rel = p - d.geom_xpos[g]
    local = d.geom_xmat[g].reshape(3, 3).T @ rel
    clamped = np.clip(local, -m.geom_size[g], m.geom_size[g])
    return float(np.linalg.norm(local - clamped))


def box_corners(g: int, m, d) -> np.ndarray:
    signs = np.array([[x, y, z] for x in (-1, 1)
                      for y in (-1, 1) for z in (-1, 1)], dtype=float)
    local = signs * m.geom_size[g]
    return d.geom_xpos[g] + local @ d.geom_xmat[g].reshape(3, 3).T


def corner_gap(a: int, b: int, m, d) -> float:
    return min(point_box_dist(c, b, m, d) for c in box_corners(a, m, d))


def main() -> None:
    import mujoco
    import scene_fidelity

    m = scene_fidelity.build_fidelity_model()
    d = mujoco.MjData(m)
    data = json.loads((REPO / "viz/panel_data_v2.json").read_text())
    frames = data["replay"]["frames"]

    gid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)
    all_geom_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
                      for g in range(m.ngeom)]

    arms = {}
    for side, sfx in (("right", "_r"), ("left", "_l")):
        zname = f"gantry_zstage{sfx}" if f"gantry_zstage{sfx}" in all_geom_names \
            else "gantry_zstage"
        arms[side] = {
            "forearm": gid(f"gantry_forearm{sfx}"),
            "shoulder": gid(f"gantry_shoulder{sfx}"),
            "zstage": gid(zname),
            "hand": mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY,
                                      "hand_root" if side == "right"
                                      else "hand_root_left"),
        }
    beam = gid("gantry_beam_x")

    j1 = {s: [] for s in arms}
    j2 = {s: [] for s in arms}
    j3 = {s: [] for s in arms}
    j4 = {s: [] for s in arms}
    for fr in frames:
        d.qpos[:] = np.asarray(fr["qpos"])
        d.qpos[0:7] = np.asarray(fr["obj"])  # object-placement trap pinned in W6d
        mujoco.mj_forward(m, d)
        for s, parts in arms.items():
            hp = d.xpos[parts["hand"]]
            j1[s].append(point_box_dist(hp, parts["forearm"], m, d))
            j2[s].append(corner_gap(parts["forearm"], parts["shoulder"], m, d))
            j3[s].append(corner_gap(parts["shoulder"], parts["zstage"], m, d))
            j4[s].append(corner_gap(parts["zstage"], beam, m, d))

    def stat(v):
        return f"min {min(v)*1e3:7.1f} mm | max {max(v)*1e3:8.1f} mm"

    lines = ["J1 hand_root -> forearm (MOVING -- verdict carrier)"]
    for s in arms:
        lines.append(f"  {s:5s} {stat(j1[s])}")
    lines.append("J2 forearm -> shoulder (static)")
    for s in arms:
        lines.append(f"  {s:5s} {stat(j2[s])}")
    lines.append("J3 shoulder -> zstage (static)")
    for s in arms:
        lines.append(f"  {s:5s} {stat(j3[s])}")
    lines.append("J4 zstage -> x-beam (static)")
    for s in arms:
        lines.append(f"  {s:5s} {stat(j4[s])}")

    lines += ["", "per-frame J1 samples (mm):", "  frame    " +
              "  ".join(f"{f:>7d}" for f in SAMPLES)]
    for s in arms:
        vals = "  ".join(f"{j1[s][f]*1e3:7.1f}" for f in SAMPLES)
        lines.append(f"  {s:5s}   {vals}")

    out = "\n".join(lines)
    print(out)
    out_dir = REPO / "runs/w6d"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "v16_junction_audit.txt").write_text(out + "\n")
    print("\nsaved -> runs/w6d/v16_junction_audit.txt")


if __name__ == "__main__":
    main()
