#!/usr/bin/env python3
"""r209 (review: the P0 full dynamics evidence): the FULL 6D
Newton-Euler account of the box, per frame, from the recorded trace.

Linear:  sum(F_contact, world) == m * a_com
Angular: sum((p_i - com) x F_i, world) == d(I_world * w)/dt + w x I_world w
         (I_world = R diag(I_body) R^T; w from the quaternion derivative)

Everything is computed from the trace's OWN records (f6/frame/pos per
contact, pos/quat per frame) plus the model constants (mass, diagonal
body inertia) -- no proxy quantities.

Output: per-phase max |linear residual| (N) and max |angular residual|
(N.m), plus the all-frames maxima.
"""
from __future__ import annotations
import json
import math
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
TRACE = REPO / "runs/w6b/box_trace.json"
SCENE = REPO / ("grounding/assets/"
                "allegro_floating_ur5e_side_r45_ablation.xml")


def quat_R(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def quat_w(q, dq):
    # omega (world) from the quaternion and its derivative
    w, x, y, z = q
    # q_dot = 0.5 * omega_quat * q  =>  solve for omega
    # use the standard: omega_body = 2 * q_inv * dq (vector part)
    qv = np.array([x, y, z])
    qw = w
    dv = np.array([dq[1], dq[2], dq[3]])
    dw = dq[0]
    return 2.0 * (qw * dv - dw * qv + np.cross(qv, dv))


def main() -> int:
    audit_mode = "--audit" in sys.argv
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(SCENE))
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "object")
    mass = float(m.body_mass[bid])
    I_body = np.diag(m.body_inertia[bid])
    print(f"box mass {mass:.4f} kg  I_body {np.round(I_body, 8).tolist()}")

    tr = json.loads(TRACE.read_text())
    fr = tr["frames"] if isinstance(tr, dict) else tr
    n = len(fr)
    ts = np.array([f["t"] for f in fr])
    pos = np.array([f["pos"] for f in fr])
    qts = np.array([f["quat"] for f in fr])
    com = pos.copy()  # the box body frame origin IS the COM (uniform box)

    # per-frame contact resultant (world) and torque about the COM
    Fsum = np.zeros((n, 3))
    Tsum = np.zeros((n, 3))
    for i, f in enumerate(fr):
        tot_f = np.zeros(3)
        tot_t = np.zeros(3)
        R = quat_R(f["quat"])
        for c in (f.get("contacts") or []):
            if "pos" not in c:
                continue
            f6 = np.array(c["f6"])
            ax = np.array(c["frame"]).reshape(3, 3)
            f_c = f6[0:3]
            f_w = ax.T @ f_c
            if c.get("box_is_g1"):
                f_w = -f_w
            p = np.array(c["pos"])
            tot_f += f_w
            tot_t += np.cross(p - com[i], f_w)
            # r212: the contact-frame TORQUE components (f6[3:6]) --
            # condim 4 carries a torsional friction torque about the
            # normal; ignoring it was exactly the pre-lift angular gap.
            tq_c = f6[3:6]
            if c.get("box_is_g1"):
                tq_c = -tq_c
            n_ax = ax[0]
            t_ax1 = ax[1]
            t_ax2 = ax[2]
            tot_t += tq_c[0] * n_ax + tq_c[1] * t_ax1 + tq_c[2] * t_ax2
        Fsum[i] = tot_f
        Tsum[i] = tot_t

    # r209b: the PHASE-LEVEL impulse-momentum closure -- the honest
    # full-dynamics form at 20 Hz sampling (the r178 lesson: per-frame
    # finite differences are substep-noise dominated; the integrated
    # impulse form is not):
    #   sum(F_contact)*dt - m*g*dt*n  ==  m*(v_end - v_start)
    #   sum(tau_contact)*dt           ==  L_end - L_start
    # with v from central-difference positions and L the world-frame
    # angular momentum at the phase boundaries.
    DT = 0.05  # the control period (20 Hz), s
    acc = np.zeros((n, 3))
    om = np.zeros((n, 3))
    qd = np.zeros((n, 4))
    for i in range(1, n - 1):
        qd[i] = (qts[i + 1] - qts[i - 1]) / (2 * DT)
        acc[i] = (pos[i + 1] - 2 * pos[i] + pos[i - 1]) / DT ** 2
        om[i] = quat_w(qts[i], qd[i])

    def Lw(i):
        Rw = quat_R(qts[i])
        return Rw @ I_body @ Rw.T @ om[i]

    phases = [("approach+grasp", 0, 260), ("pre-lift", 260, 331),
              ("lift_accel", 331, 345), ("lift_hold", 345, n - 1)]
    gv = np.array([0.0, 0.0, -9.81])  # the gravity ACCELERATION
    W = mass * gv                     # the gravity FORCE on the box
    print(f"frames {n}  dt {DT} s  (the impulse-momentum closure)")
    worst_lin = 0.0
    worst_ang = 0.0
    for name, i0, i1 in phases:
        nfr = i1 - i0
        J = Fsum[i0:i1].sum(axis=0) * DT + W * DT * nfr
        dp = mass * (acc * 0)  # placeholder
        v0 = (pos[min(i0 + 1, n - 1)] - pos[max(i0 - 1, 0)]) / (2 * DT)
        v1 = (pos[min(i1, n - 1)] - pos[max(i1 - 2, 0)]) / (2 * DT)
        dp = mass * (v1 - v0)
        lin = np.linalg.norm(J - dp)
        # angular: sum(tau)*dt == L(i1) - L(i0)
        Ja = Tsum[i0:i1].sum(axis=0) * DT
        dL = Lw(min(i1, n - 1)) - Lw(max(i0 - 1, 0))
        ang = np.linalg.norm(Ja - dL)
        worst_lin = max(worst_lin, lin)
        worst_ang = max(worst_ang, ang)
        print(f"{name:16s} |J - m*dv| {lin:8.4f} N.s  "
              f"|Jt - dL| {ang:8.4f} N.m.s")
    print(f"ALL              |J - m*dv| {worst_lin:8.4f} N.s  "
          f"|Jt - dL| {worst_ang:8.4f} N.m.s")
    print(f"weight-momentum scale: m*g*T_lift "
          f"{mass * 9.81 * (345 - 331) * DT:.3f} N.s (the linear scale)")

    if audit_mode:
        # r211: the SUBSTEP-INTEGRATED closure -- the impulses come from
        # the audit sidecar (per-physics-substep integration), so the
        # 20 Hz sampling limit does not apply to J any more; only the
        # boundary velocity/momentum estimates remain 20 Hz.
        side = REPO / "runs/w6b/audit_impulse.json"
        aud = json.loads(side.read_text())
        af = np.array([f["F"] for f in aud["frames"]])
        at = np.array([f["T"] for f in aud["frames"]])
        ats = np.array([f["t"] for f in aud["frames"]])
        dtc = float(aud.get("dt_ctrl_s", DT))
        print("-- audit sidecar (substep-integrated impulses) --")
        for name, i0, i1 in phases:
            i1s = min(i1, len(af))
            J = af[i0:i1s].sum(axis=0) * dtc + W * dtc * (i1s - i0)
            v0 = (pos[min(i0 + 1, n - 1)] - pos[max(i0 - 1, 0)]) / (2 * DT)
            i1b = min(i1, n - 1)
            v1 = (pos[i1b] - pos[max(i1b - 2, 0)]) / (2 * DT)
            dp = mass * (v1 - v0)
            lin = np.linalg.norm(J - dp)
            Ja = at[i0:i1s].sum(axis=0) * dtc
            dL = Lw(min(i1, n - 1)) - Lw(max(i0 - 1, 0))
            ang = np.linalg.norm(Ja - dL)
            print(f"{name:16s} |J_sub - m*dv| {lin:8.4f} N.s  "
                  f"|Jt_sub - dL| {ang:8.4f} N.m.s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
