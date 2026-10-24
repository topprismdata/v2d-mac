"""r119: the DUAL-PAD surface seat -- both pads' surfaces on one face.

THE PROBLEM (r118).  gate1's v1/v2 (the pad CENTRE within 0.1164 mm of
the face plane) and its v3 (the pad SURFACE within 0.1164 mm) are
physically exclusive for a 12 x 12 x 3 mm pad, because the two
distances differ by the pad's support along the face normal.  Measured
at the delivered pose:

    the ff: the support 2.122 mm, the centre on the plane (60.000), the
        support point at 57.878 -- 2.122 mm PAST the plane
    the mf: the support 5.776 mm, the centre on the plane (60.000), the
        support point at 54.224 -- 5.776 mm PAST the plane

So the delivered "two pads on the face" is two pad CENTRES on the
plane, with both pads' surfaces buried 2.1 / 5.8 mm inside the box.
Only one pad can physically do that.

THE FIX.  Put both pads' SURFACES on the plane instead.  Each pad's
surface is represented by its SUPPORT POINT -- the corner that is
extreme along the inward normal (the box's support function applied to
the pad), which is the point that actually touches.  Then:

  * the finger IK drives the support point onto the plane, and
  * the palm's stand-off is set by the ff's support point, so the two
    supports are free to differ (the mf can pick a different corner).

The mf's support is a CONTINUOUS function of its joints (measured:
3.445 mm at j2 = 0 to 7.394 mm at j2 = 1.2), so the mf can change its
corner to bring its support point onto the plane too -- while the ff
holds the reference corner.

The candidates are replayed in the full pipeline by
tools/w6b/surface_seated_physics.py (or the build tool's own leg).

    runs/.venv-mj/bin/python tools/w6b/dual_pad_surface_seat.py
"""
from __future__ import annotations

import json
import os
import pathlib

import mujoco
import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
SCENE = pathlib.Path(os.environ.get(
    "W6B_S119_SCENE",
    "grounding/assets/allegro_floating_ur5e_side_r37.xml"))
DELIVERED = REPO / "runs/w6b/palm_pose_refine.json"
OUT = pathlib.Path(os.environ.get(
    "W6B_S119_OUT", "runs/w6b/dual_pad_surface_seat.json"))

ARM = ("shoulder_pan", "shoulder_lift", "elbow",
       "wrist_1", "wrist_2", "wrist_3")
FINGERS = ("ff", "mf")
# the curl scales for the seed (the flexion joints are scaled)
CURLS = [float(v) for v in os.environ.get(
    "W6B_S119_CURLS", "1.00,1.05,1.10").split(",")]
# the commanded surface preload (mm, negative = into the face)
PRELOADS_MM = [float(v) for v in os.environ.get(
    "W6B_S119_PRELOADS_MM", "-0.05,0.0").split(",")]


def jid(m, n):
    i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
    assert i >= 0, n
    return i


def gid(m, n):
    i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)
    assert i >= 0, n
    return i


class Side:
    def __init__(self, m, d, side: str):
        self.m = m
        self.d = d
        self.side = side
        sfx = "" if side == "right" else "_left"
        prefix = "ur5e_r" if side == "right" else "ur5e_l"
        self.face_sign = 1.0 if side == "right" else -1.0
        self.d_in = np.array([-self.face_sign, 0.0, 0.0])
        self.AJ = [jid(m, f"{prefix}_{n}_joint") for n in ARM]
        self.AQ = [int(m.jnt_qposadr[j]) for j in self.AJ]
        self.AD = [int(m.jnt_dofadr[j]) for j in self.AJ]
        self.PALM = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY,
                                      "palm" + sfx)
        self.OBJ = gid(m, "object_geom")
        self.HALF = np.array(m.geom_size[self.OBJ])
        self.PAD = {f: gid(m, f"{f}_tip_collision" if not sfx
                           else f"{f}_tip_left_collision")
                    for f in FINGERS}
        self.FQ = {f: [int(m.jnt_qposadr[jid(m, f"{f}j{k}{sfx}")])
                      for k in range(4)] for f in FINGERS}
        self.FQ["th"] = [int(m.jnt_qposadr[jid(m, f"thj{k}{sfx}")])
                         for k in range(4)]
        self.FD = {f: [int(m.jnt_dofadr[jid(m, f"{f}j{k}{sfx}")])
                      for k in range(4)] for f in FINGERS}
        self.FLO = {f: np.array([m.jnt_range[jid(m, f"{f}j{k}{sfx}"), 0]
                                 for k in range(4)]) for f in FINGERS}
        self.FHI = {f: np.array([m.jnt_range[jid(m, f"{f}j{k}{sfx}"), 1]
                                 for k in range(4)]) for f in FINGERS}
        self.obj = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "object")
        self.qa = int(m.jnt_qposadr[int(self.obj)])

    def reset(self):
        mujoco.mj_resetData(self.m, self.d)
        self.d.qpos[:] = self.m.qpos0
        self.d.qpos[self.qa:self.qa + 7] = [0, 0, 0.74, 1, 0, 0, 0]

    def place(self, q, fv, thumb):
        self.reset()
        self.d.qpos[self.AQ] = q
        for f in FINGERS:
            self.d.qpos[self.FQ[f]] = fv[f]
        self.d.qpos[self.FQ["th"]] = thumb
        mujoco.mj_forward(self.m, self.d)

    def support_local(self, f):
        """the local corner that is extreme along the inward normal"""
        R = np.asarray(self.d.geom_xmat[self.PAD[f]]).reshape(3, 3)
        he = np.asarray(self.m.geom_size[self.PAD[f]])
        dl = R.T @ self.d_in
        return np.sign(dl) * he

    def support_point(self, f):
        """the world point of the pad that touches the face"""
        R = np.asarray(self.d.geom_xmat[self.PAD[f]]).reshape(3, 3)
        v = self.support_local(f)
        return np.asarray(self.d.geom_xpos[self.PAD[f]]) + R @ v

    def support_jac(self, f, x=None):
        """the Jacobian of the support point (the pad's local corner).

        This MuJoCo build's mj_jacGeom does not take a point, and the
        support point is a CORNER (whose local offset changes with the
        posture), so the Jacobian is taken by finite differences over
        the finger's own joints.
        """
        dofs = self.FD[f]
        base = (np.asarray(self.d.qpos)[dofs].copy() if x is None
                else np.asarray(x, float).copy())
        h = 1e-6
        cols = []
        for k in range(len(dofs)):
            xp = base.copy()
            xp[k] += h
            self.d.qpos[dofs] = xp
            mujoco.mj_forward(self.m, self.d)
            pp = self.support_point(f)
            self.d.qpos[dofs] = base
            mujoco.mj_forward(self.m, self.d)
            cols.append((pp - self.support_point(f)) / h)
        return np.array(cols).T

    def plane_x(self):
        return float(self.d.xpos[int(self.obj)][0]) + \
            self.face_sign * float(self.HALF[0])

    def surf_gap(self, f):
        """the signed distance of the pad's SURFACE to the box"""
        ft = np.zeros(6)
        dd = float(mujoco.mj_geomDistance(self.m, self.d, self.PAD[f],
                                          self.OBJ, 0.5, ft))
        v = np.asarray(ft[0:3]) - np.asarray(ft[3:6])
        nv = float(np.linalg.norm(v))
        n_hat = v / nv if nv > 1e-9 else self.d_in.copy()
        return dd, n_hat

    def contact_point(self, f):
        """the pad's TRUE closest point to the box and the signed gap"""
        ft = np.zeros(6)
        dd = float(mujoco.mj_geomDistance(self.m, self.d, self.PAD[f],
                                          self.OBJ, 0.5, ft))
        return np.asarray(ft[0:3]), dd

    def contact_jac(self, f, x=None):
        """the Jacobian of the pad's closest point, by finite differences
        over the finger's own joints"""
        dofs = self.FD[f]
        base = (np.asarray(self.d.qpos)[dofs].copy() if x is None
                else np.asarray(x, float).copy())
        h = 1e-6
        cols = []
        for k in range(len(dofs)):
            xp = base.copy()
            xp[k] += h
            self.d.qpos[dofs] = xp
            mujoco.mj_forward(self.m, self.d)
            pp, _ = self.contact_point(f)
            self.d.qpos[dofs] = base
            mujoco.mj_forward(self.m, self.d)
            p0, _ = self.contact_point(f)
            cols.append((pp - p0) / h)
        return np.array(cols).T

    def ik_contact(self, q, f, fv, thumb, iters=260):
        """drive the pad's TRUE closest point onto the END-FACE PLANE,
        and keep the contact INSIDE the end face's rectangle.

        The x constraint alone is not enough: the box's end face is
        finite (240 x 80 mm) and the pad's closest point can land on
        the TOP edge instead (measured: the mf's closest point was at
        the box's top face, z = 780 mm, so an x-only constraint put it
        on the top EDGE and the true surface gap stayed 8.2 mm).

        So the constraint is on the contact point's position in the end
        face's own frame: the normal coordinate must be zero (on the
        plane) and the two tangential coordinates must stay inside the
        face (the gate's own band).
        """
        x = np.asarray(fv[f], float).copy()
        best = None
        for _ in range(iters):
            self.place(q, fv, thumb)
            self.d.qpos[self.FQ[f]] = x
            mujoco.mj_forward(self.m, self.d)
            p, _ = self.contact_point(f)
            e = float((p[0] - self.plane_x()) * self.d_in[0])
            # the tangential offsets from the face's centre, in the
            # face's own frame (the end face is the x = +-60 plane)
            dz = float(p[2] - self.d.xpos[int(self.obj)][2])
            if abs(e) < 2e-5 and abs(dz) < 0.5 * float(self.HALF[2]):
                best = (x.copy(), abs(e))
                break
            J = self.contact_jac(f, x)
            # the normal row and the vertical row (the face's band)
            r0 = (J[0, :] * self.d_in[0])
            r1 = J[2, :]
            A = np.vstack([r0, r1])
            b = np.array([-e, -dz])
            try:
                dq = np.linalg.lstsq(A, b, rcond=None)[0]
            except np.linalg.LinAlgError:
                break
            x = np.clip(x + dq, self.FLO[f], self.FHI[f])
        self.place(q, fv, thumb)
        self.d.qpos[self.FQ[f]] = x
        mujoco.mj_forward(self.m, self.d)
        p, _ = self.contact_point(f)
        e = float((p[0] - self.plane_x()) * self.d_in[0])
        dz = float(p[2] - self.d.xpos[int(self.obj)][2])
        # the residual is the worse of the two, and the band violation
        # counts: a contact off the end face is a failure
        res = max(abs(e), abs(dz) - 0.5 * float(self.HALF[2]))
        if best is not None and best[1] < res:
            x = best[0]
            self.place(q, fv, thumb)
            self.d.qpos[self.FQ[f]] = x
            mujoco.mj_forward(self.m, self.d)
            p, _ = self.contact_point(f)
            e = float((p[0] - self.plane_x()) * self.d_in[0])
            dz = float(p[2] - self.d.xpos[int(self.obj)][2])
            res = best[1]
        return x, max(res, 0.0)


def main() -> int:
    m = mujoco.MjModel.from_xml_path(str(SCENE))
    d = mujoco.MjData(m)
    ref = json.loads(DELIVERED.read_text())
    out = {"scene": str(SCENE),
           "the criterion": (
               "BOTH pads' support points (the corner that touches) on "
               "the end-face plane, at the commanded preload"),
           "sides": {}}

    for side in ("right", "left"):
        sd = Side(m, d, side)
        entry = ref["sides"][side]
        base = entry["best"][0] if isinstance(entry.get("best"), list) \
            else entry["best"]
        thumb = list(base["fingers"].get("th", [0.0] * 4))
        rows = []
        for curl in CURLS:
            seed = {}
            for f in FINGERS:
                v = list(base["fingers"][f])
                for j in (1, 2, 3):
                    v[j] *= curl
                seed[f] = v
            for preload_mm in PRELOADS_MM:
                target_x_offset = preload_mm * 1e-3
                q = np.asarray(base["arm"], float)
                fv = {f: list(seed[f]) for f in FINGERS}
                # THE ORDER MATTERS.  The palm's stand-off along the
                # normal is set by the FF (the palm's translation moves
                # both pads, so the ff's contact point has to be the one
                # that defines the plane), and the MF then has to reach
                # the plane with its OWN corner -- which its joints can
                # do, because the support is a continuous function of
                # them (measured: 3.445 mm at j2 = 0 to 7.394 mm at
                # j2 = 1.2).
                #
                # So: (1) the palm backs off along the normal by the
                # ff's support, (2) the ff's IK lands its contact point
                # on the plane, (3) the mf's IK lands its contact point
                # on the plane, (4) iterate, because each step moves
                # the other's geometry.
                for _ in range(6):
                    # the palm's normal translation: the ff's contact
                    # point onto the plane
                    sd.place(q, fv, thumb)
                    jp = np.zeros((3, m.nv))
                    jr = np.zeros((3, m.nv))
                    mujoco.mj_jacBody(m, d, jp, jr, sd.PALM)
                    out_dir = np.array([sd.face_sign, 0.0, 0.0])
                    Jn = jp[:, sd.AD].T @ out_dir
                    Jo = jr[:, sd.AD]
                    A = np.vstack([Jn, Jo])
                    p, _ = sd.contact_point("ff")
                    e = float((p[0] - sd.plane_x()) * sd.d_in[0])
                    b = np.concatenate([np.array([-e]), np.zeros(3)])
                    dq = np.linalg.lstsq(A, b, rcond=None)[0]
                    q = np.clip(q + dq, m.jnt_range[sd.AJ, 0],
                                m.jnt_range[sd.AJ, 1])
                    # the fingers: the ff then the mf onto the plane
                    x_ff, r_ff = sd.ik_contact(q, "ff", fv, thumb)
                    fv["ff"] = list(x_ff)
                    x_mf, r_mf = sd.ik_contact(q, "mf", fv, thumb)
                    fv["mf"] = list(x_mf)
                    if max(r_ff, r_mf) < 3e-3:
                        break
                sd.place(q, fv, thumb)
                # measure: the support points and the true surface gaps
                sup = {f: sd.support_point(f) for f in FINGERS}
                con = {f: sd.contact_point(f)[0] for f in FINGERS}
                gaps = {f: sd.surf_gap(f)[0] for f in FINGERS}
                on_face = {}
                for f in FINGERS:
                    p = con[f]
                    # STRICT: the contact point must be on the END
                    # FACE, not on the edge where the end face meets
                    # the top face.  The box's top is z = 780 and the
                    # face band is 700-780, so a contact at exactly
                    # 780 is on the EDGE -- and the measured mf contacts
                    # all landed exactly there.
                    _bz = float(sd.d.xpos[int(sd.obj)][2])
                    on_face[f] = bool(
                        abs(abs(p[0]) - float(sd.HALF[0])) < 1e-3
                        and abs(p[1]) <= float(sd.HALF[1])
                        and abs(p[2] - _bz) < float(sd.HALF[2]) - 1e-3)
                # the palm pose MUST be the one the arm realises: the
                # pipeline drives its terminal IK from the reference's
                # palm_pos / palm_R, so a stale pair puts the executed
                # pads off the face (measured: the executed mf was
                # 10.5 mm away while the static contact point was on
                # the plane)
                pp = np.array(d.xpos[sd.PALM], dtype=float)
                Rq = np.zeros(9)
                mujoco.mju_quat2Mat(Rq, d.xquat[sd.PALM])
                PR = np.array(Rq).reshape(3, 3)
                rows.append({
                    "the curl": curl,
                    "the preload mm": preload_mm,
                    "the contact point x mm": {
                        f: round(float(con[f][0]) * 1e3, 4) for f in FINGERS},
                    "the support point x mm": {
                        f: round(float(sup[f][0]) * 1e3, 4) for f in FINGERS},
                    "the contact point on the face": on_face,
                    "the surface gaps mm": {f: round(float(gaps[f]) * 1e3, 4)
                                            for f in FINGERS},
                    "the pad centres mm": {
                        f: [round(float(v) * 1e3, 3)
                            for v in d.geom_xpos[sd.PAD[f]]]
                        for f in FINGERS},
                    "the palm": [round(float(v), 6) for v in pp],
                    "the palm_R": [[round(float(v), 6) for v in r]
                                   for r in PR],
                    "the arm": [round(float(v), 6) for v in q],
                    "the fingers": {
                        f: [round(float(v), 6) for v in fv[f]]
                        for f in FINGERS},
                    "the thumb": [round(float(v), 6) for v in thumb],
                    "the mf ik residual mm": round(float(r_mf) * 1e3, 4),
                    "the ff ik residual mm": round(float(r_ff) * 1e3, 4),
                })
                print(f"  {side} the curl {curl} / the preload {preload_mm}: "
                      f"the contact points "
                      f"{rows[-1]['the contact point x mm']} | the surface "
                      f"gaps {rows[-1]['the surface gaps mm']} | on the face "
                      f"{on_face}")
        out["sides"][side] = {"candidates": len(rows), "best": rows}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    print(f"\n  written: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
