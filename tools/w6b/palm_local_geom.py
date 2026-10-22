#!/usr/bin/env python3
"""Measure the palm-local geometry that the plate grasp depends on.

For each hand (right = "palm", left = "palm_left") print, in the PALM's
own frame:
  * the plate geom (palm_collision): centre, its three axes, half extents,
    and which local axis is the plate's normal;
  * the finger bases and the four fingertip bodies at several curls;
  * the thumb tip;
so that the grasp target can be derived from measurement instead of from
an assumed convention.

usage: palm_local_geom.py [scene.xml]
"""
import sys
from pathlib import Path

import mujoco
import numpy as np

SCENE = Path(sys.argv[1] if len(sys.argv) > 1 else
              "grounding/assets/allegro_floating_ur5e_side_r37.xml")

m = mujoco.MjModel.from_xml_path(str(SCENE))
d = mujoco.MjData(m)


def bid(n):
    i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
    assert i >= 0, n
    return i


def gid(n):
    i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)
    assert i >= 0, n
    return i


SIDES = {
    "right": {"palm": "palm", "plate": "palm_collision",
              "tips": ["ff_tip", "mf_tip", "rf_tip", "th_tip"],
              "bases": ["ff_base", "mf_base", "rf_base", "th_base"],
              "jprefix": ""},
    "left": {"palm": "palm_left", "plate": "palm_left_collision",
             "tips": ["ff_tip_left", "mf_tip_left", "rf_tip_left",
                      "th_tip_left"],
             "bases": ["ff_base_left", "mf_base_left", "rf_base_left",
                       "th_base_left"],
             "jprefix": "_left"},
}


def local_of(body_id, R_p, P_p, world_pt):
    return R_p.T @ (np.asarray(world_pt) - P_p)


def main():
    for side, cfg in SIDES.items():
        pb = bid(cfg["palm"])
        pg = gid(cfg["plate"])
        print(f"=== {side} ===")
        print(f"  palm body id {pb}  plate geom id {pg}")
        print(f"  plate half extents (m) {m.geom_size[pg]}")
        print(f"  plate geom pos in palm (m) "
              f"{m.geom_pos[pg]}")
        FJ = [(f, k) for f in ("ff", "mf", "rf", "th") for k in range(4)]
        fj = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT,
                                f"{f}j{k}{cfg['jprefix']}") for f, k in FJ]
        print("  finger joint ranges (deg):")
        for (f, k), j in zip(FJ, fj):
            lo, hi = m.jnt_range[j]
            print(f"    {f}j{k}: {np.degrees(lo):7.1f} .. {np.degrees(hi):7.1f}")
        for curl in (0.0, 0.25, 0.5, 0.75, 1.0):
            d.qpos[:] = m.qpos0
            for j in fj:
                lo, hi = m.jnt_range[j]
                base = 0.5 * (lo + hi)
                d.qpos[m.jnt_qposadr[j]] = base + curl * 0.35 * (hi - base)
            mujoco.mj_forward(m, d)
            Rp = np.array(d.xmat[pb]).reshape(3, 3)
            Pp = np.array(d.xpos[pb])
            Rg = np.array(d.geom_xmat[pg]).reshape(3, 3)
            pg_c = np.array(d.geom_xpos[pg])
            # the plate's normal: the local axis with the smallest extent
            ext = m.geom_size[pg]
            k = int(np.argmin(ext))
            n_local = np.zeros(3)
            n_local[k] = 1.0
            n_world = Rg @ n_local
            n_palm = Rp.T @ n_world
            # the sign: the plate's outward face is the one AWAY from the
            # palm's interior (the finger roots) -> compare with the mean
            # direction from the palm origin to the finger bases
            bases = [bid(b) for b in cfg["bases"]]
            mean_base = np.mean([np.array(d.xpos[b]) for b in bases], axis=0)
            interior = Rp.T @ (mean_base - Pp)
            sgn = 1.0 if float(n_palm @ interior) > 0 else -1.0
            n_palm = n_palm * sgn
            print(f"  curl={curl:.2f}")
            print(f"    plate normal (palm frame) {np.round(n_palm, 4)}"
                  f"  (axis {k}, sign {sgn:+.0f})")
            print(f"    plate centre (palm) "
                  f"{np.round(local_of(pb, Rp, Pp, pg_c) * 1e3, 2)} mm")
            print(f"    plate extents along palm axes (mm) "
                  f"{np.round(ext * 1e3, 2)}")
            for t in cfg["tips"]:
                b = bid(t)
                print(f"    {t:12s} palm-frame "
                      f"{np.round(local_of(pb, Rp, Pp, d.xpos[b]) * 1e3, 2)}")


if __name__ == "__main__":
    main()
