"""W4-B validation - re-derive the K1 hit with a SECOND, independent measurement.

The scan's objective reads ``mj_geomDistance``. This script ignores that path
completely: it writes the reported pose into a fresh model and reports

  * every hand geom's SIGNED distance to the box (return value only), so a
    burial anywhere would show up as a negative number;
  * MuJoCo's own contact list (d.contact) at that pose - the same measurement
    the physics engine uses, not a GJK query;
  * the nearest-point-on-box for each contacting tip, in box coordinates.

Run: runs/.venv-mj/bin/python runs/w4/validate_k1.py <pedestal_mm>
"""
from __future__ import annotations

import json
import pathlib
import sys

import mujoco
import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from grounding.assets import build_allegro_floating as baf  # noqa: E402

RUNS = pathlib.Path(__file__).resolve().parent
JSON = RUNS / "pedestal_scan_kin.json"


def main() -> None:
    data = json.loads(JSON.read_text())
    import importlib.util
    spec = importlib.util.spec_from_file_location("ps", RUNS / "pedestal_scan.py")
    ps = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ps)

    for cell in data["cells"]:
        h = cell["pedestal_mm"] / 1e3
        pose = np.array(cell["best_pose"], dtype=float)
        model, mdata = ps.compile_scene(h)
        rig = ps.Rig(model, mdata, h)
        mujoco.mj_resetData(model, mdata)
        rig.pose_to_qpos(pose)
        mujoco.mj_kinematics(model, mdata)
        mujoco.mj_forward(model, mdata)
        ft = np.zeros(6)

        dists = []
        for g in rig.hand_geoms:
            d = float(mujoco.mj_geomDistance(model, mdata, g, rig.ogeom, 0.20, ft))
            dists.append((mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g)
                          or f"geom{g}", d))
        buried = [(n, d) for n, d in dists if d < -1e-9]

        contacts = []
        for i in range(mdata.ncon):
            c = mdata.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            if g1 != rig.ogeom and g2 != rig.ogeom:
                continue
            other = g2 if g1 == rig.ogeom else g1
            pt = mdata.geom_xpos[other] + (c.pos - mdata.geom_xpos[other]) * 0.0
            contacts.append({
                "geom": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, other),
                "dist_m": float(c.dist),
                "pos": [round(float(v), 5) for v in c.pos],
            })
        pads = []
        for name, g in rig.pad_geoms:
            d = float(mujoco.mj_geomDistance(model, mdata, g, rig.ogeom, 0.20, ft))
            q = [round(float(v), 5) for v in ft[3:6]]
            pads.append({"pad": name, "dist_m": round(d, 9),
                         "q_box": [q[0], q[1], round(q[2] - rig.box_z, 5)]})
        print(f"=== pedestal {cell['pedestal_mm']:.0f} mm  box_top {cell['box_top_m']:.3f} "
              f"tol {data['tol_m'] * 1e3:.4f} mm")
        print(f"    hand geoms: {len(dists)}   buried: {buried}")
        print(f"    mj contacts with the box: {len(contacts)}")
        for c in contacts:
            print(f"      {c['geom']:28s} dist={c['dist_m'] * 1e3:+.5f} mm")
        for p in pads:
            print(f"      pad {p['pad']:6s} dist={p['dist_m'] * 1e3:+.5f} mm "
                  f"q_box={p['q_box']}")
        two_wall = sorted({(-1 if p["q_box"][0] < 0 else 1) for p in pads
                           if p["dist_m"] <= data["tol_m"]})
        print(f"    sides touched within tol: {two_wall}  -> K1 = {len(two_wall) == 2}")
        print(f"    worst signed distance over all hand geoms: "
              f"{min(d for _, d in dists):+.6e} m")


if __name__ == "__main__":
    main()
