#!/usr/bin/env python3
"""Which face of the palm slab is the GRASP surface?

The Allegro palm is a collision box (half extents 20.4 x 56.5 x 47.5 mm in
the palm's own frame).  Its three face pairs are:

    +-x : 113 x 95 mm   (the largest face)
    +-y :  41 x 95 mm
    +-z :  41 x 113 mm

A palm-plate grasp needs the face that a grasped object actually touches.
The decisive measurement is the SHIPPED, VERIFIED grasp (the floating
scene's gate-3 seed, runs/w6/gate3_qpos.json): the fingertips are in
contact with the box's end face there, so the palm slab's position and
orientation relative to that face identifies the grasp surface.

This prints, in the palm's own frame:
  * every face of the slab, its outward normal and its distance from the
    box's end-face plane (signed: positive = away from the box);
  * the fingertip positions;
so the face that faces the object is identified by measurement.

usage: palm_anatomy.py [scene.xml] [gate3_qpos.json]
"""
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

SCENE = Path(sys.argv[1] if len(sys.argv) > 1 else
              "grounding/assets/allegro_floating.xml")
SEED = Path(sys.argv[2] if len(sys.argv) > 2 else "runs/w6/gate3_qpos.json")

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


def main():
    q = json.loads(SEED.read_text())["qpos"]
    d.qpos[:] = np.asarray(q, float)
    mujoco.mj_forward(m, d)
    obj = bid("object")
    box_g = gid("object_geom")
    half = np.array(m.geom_size[box_g])
    box_c = np.array(d.xpos[obj])
    # the end face the right hand touches: the +x face
    face_plane = box_c[0] + half[0]
    normal = np.array([1.0, 0.0, 0.0])

    for side, palm_n, plate_n in (("right", "palm", "palm_collision"),
                                  ("left", "palm_left", "palm_left_collision")):
        pb = bid(palm_n)
        pg = gid(plate_n)
        Rp = np.array(d.xmat[pb]).reshape(3, 3)
        Pp = np.array(d.xpos[pb])
        Pg = np.array(d.geom_xpos[pg])
        ext = np.array(m.geom_size[pg])
        ctr_local = Rp.T @ (Pg - Pp)
        print(f"=== {side} (box face plane x = {face_plane*1e3:.1f} mm) ===")
        print(f"  palm origin {np.round(Pp*1e3, 2)}  "
              f"slab centre (palm frame) {np.round(ctr_local*1e3, 2)} mm  "
              f"half extents {np.round(ext*1e3, 2)} mm")
        # every face of the slab: normal in the palm frame, and the signed
        # distance from the face plane of the slab face that is nearest
        for k in range(3):
            for s in (+1, -1):
                n_loc = np.zeros(3)
                n_loc[k] = s
                n_world = Rp @ n_loc
                # a point on that face
                p_loc = ctr_local + n_loc * ext[k]
                p_world = Pp + Rp @ p_loc
                dist = float((p_world - np.array([face_plane, 0, 0])) @ normal)
                area = 1.0
                for j in range(3):
                    if j != k:
                        area *= 2 * ext[j]
                print(f"    face {['x','y','z'][k]}{'+' if s>0 else '-'} "
                      f"normal_world {np.round(n_world, 3)} "
                      f"area {area*1e4:7.1f} cm2  "
                      f"signed dist from the +x face plane "
                      f"{dist*1e3:+8.2f} mm")
        tips = ["ff_tip", "mf_tip", "rf_tip", "th_tip"]
        for t in tips:
            suffix = "" if side == "right" else "_left"
            try:
                b = bid(t + suffix)
            except AssertionError:
                continue
            pw = np.array(d.xpos[b])
            print(f"    {t+suffix:14s} world {np.round(pw*1e3,1)} "
                  f"dist from the +x face plane "
                  f"{float((pw-np.array([face_plane,0,0]))@normal)*1e3:+8.2f} mm  "
                  f"palm-frame {np.round(Rp.T@(pw-Pp)*1e3,1)}")


if __name__ == "__main__":
    main()
