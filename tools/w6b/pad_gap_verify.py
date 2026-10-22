#!/usr/bin/env python3
"""Verify the pad's surface-gap formula against MuJoCo's own contact solver.

The r60 seat computes a pad's signed surface distance to the end face as

    r_eff = geom_size[0] * |geom_z_axis . n_face|

i.e. it treats the pad as a 6 mm-radius DISC whose radius is projected on
the face normal.  But the pad geom is a BOX (12 x 12 x 3 mm, MuJoCo geom
type 6), so the extent of the box along a direction n is its SUPPORT
FUNCTION

    h(n) = sum_i half_extent_i * |axis_i . n|

and the correct surface gap is (centre offset along n) - h(n).

The check uses a minimal model with the SAME pad box and a face to press
against, so MuJoCo's own contact distance is the ground truth.  The pad's
tilt is swept, because a disc and a box agree only in special
orientations -- that is what discriminates the two formulas.

usage: pad_gap_verify.py
"""
import mujoco
import numpy as np

XML = """
<mujoco>
  <worldbody>
    <body name="obj" pos="0 0 0.02">
      <freejoint/>
      <geom name="obj_geom" type="box" size="0.06 0.12 0.04"/>
    </body>
    <body name="padbody" pos="0.07 0 0.02">
      <freejoint/>
      <geom name="pad" type="box" size="0.006 0.006 0.0015"/>
    </body>
  </worldbody>
</mujoco>
"""
m = mujoco.MjModel.from_xml_string(XML)
d = mujoco.MjData(m)
pg = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "pad")
PAD_Q = 7            # the pad body's free joint (after the object's)
n = np.array([1.0, 0.0, 0.0])
FACE_X = 0.06        # the object's +x face


def support(R, he):
    """the box's extent along n: sum_i h_i |axis_i . n|"""
    return float(sum(he[i] * abs(float(R[:, i] @ n)) for i in range(3)))


def disc_formula(R, he):
    """the r60 formula: geom_size[0] * |local z axis . n|"""
    return float(he[0] * abs(float(R[:, 2] @ n)))


print(f"pad geom: type {m.geom_type[pg]} (6=BOX), half extents "
      f"{np.round(m.geom_size[pg]*1e3, 2)} mm")
print(f"\n{'tilt (deg)':>11}{'MuJoCo dist':>13}{'support pred':>14}"
      f"{'r60 disc pred':>15}{'verdict':>13}")
ok_support = ok_disc = 0
for tilt in (0, 5, 10, 20, 35, 55, 75):
    mujoco.mj_resetData(m, d)
    q = np.zeros(m.nq)
    th = np.deg2rad(tilt)
    q[PAD_Q + 3:PAD_Q + 6] = [0.0, th, 0.0]
    R0 = np.array([[np.cos(th), 0, np.sin(th)],
                   [0, 1, 0],
                   [-np.sin(th), 0, np.cos(th)]])
    h = support(R0, m.geom_size[pg])
    # put the pad's support point 2 mm INSIDE the face
    q[PAD_Q] = FACE_X + h - 0.002
    q[PAD_Q + 2] = 0.02
    d.qpos[:] = q
    mujoco.mj_forward(m, d)
    R = np.array(d.geom_xmat[pg]).reshape(3, 3)
    c = np.array(d.geom_xpos[pg])
    off = float(c[0]) - FACE_X
    dist = None
    for i in range(d.ncon):
        cn = d.contact[i]
        if int(cn.geom1) == pg or int(cn.geom2) == pg:
            dist = float(cn.dist)
            break
    pred = off - support(R, m.geom_size[pg])
    predb = off - disc_formula(R, m.geom_size[pg])
    v = ""
    if dist is not None:
        if abs(pred - dist) < 1e-5:
            v = "support-OK"
            ok_support += 1
        elif abs(predb - dist) < 1e-5:
            v = "r60-OK"
            ok_disc += 1
        else:
            v = "MISMATCH"
    print(f"{tilt:11d}"
          f"{(dist*1e3 if dist is not None else float('nan')):13.4f}"
          f"{pred*1e3:14.4f}{predb*1e3:15.4f}{v:>13}")

print(f"\n=> support function correct in {ok_support} tilts; the r60 disc "
      f"formula in {ok_disc}.")
print("=> the surface gap must be (centre offset) - sum_i h_i |axis_i . n| "
      "(the BOX's support function).")
