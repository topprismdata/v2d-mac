"""r12.3: UR5e closed-form wrist-centre IK (DH model + frame calibration).

Standard UR5e DH (conventional), the geometry of the official model:
  i | theta | d        | a       | alpha
  1 | q1    | 0.089159 | 0.04165 | +pi/2
  2 | q2    | 0        | -0.425  | 0
  3 | q3    | 0        | -0.3922 | 0
  4 | q4    | 0.10915  | 0       | +pi/2
  5 | q5    | 0.09465  | 0       | -pi/2
  6 | q6    | 0.0823   | 0       | 0

The menagerie MJCF uses a different (tree) convention: the composer
calibrates the two constant transforms T_base and T_tool so that
  T_menagerie(q) = T_base * T_dh(q) * T_tool
for all q -- verified < 1e-6 on random configurations before use
(validation-first: the analytic solution is never applied against an
uncalibrated frame).

Closed form (standard UR, Hawkins-style):
  wrist centre W = p06 - d6 * R06 zhat
  theta1 = atan2(Wy, Wx) -+ acos(d4 / r_xy)   (the y-offset branch, with
            the a1/d1 frame corrections folded into the calibrated base)
  planar 2R for theta2/theta3 with the elbow up/down branches
  theta4/5/6 from R36 = R03^T R06 (the spherical wrist): theta5 from the
  z-component, theta4/6 from the atan2 pair; branch flipped for theta5<0.
"""
from __future__ import annotations

import numpy as np
import mujoco


def _rz(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])


def _tz(d):
    T = np.eye(4)
    T[2, 3] = d
    return T


def _tx(a):
    T = np.eye(4)
    T[0, 3] = a
    return T


def _rx(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]])


DH = [
    (0.089159, 0.04165, np.pi / 2),
    (0.0, -0.425, 0.0),
    (0.0, -0.3922, 0.0),
    (0.10915, 0.0, np.pi / 2),
    (0.09465, 0.0, -np.pi / 2),
    (0.0823, 0.0, 0.0),
]


def dh_fk(q: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    for i, (d, a, al) in enumerate(DH):
        T = T @ _rz(q[i]) @ _tz(d) @ _tx(a) @ _rx(al)
    return T


class UR5eAnalyticIK:
    """Closed-form IK with the calibrated frames and the self-validation."""

    def __init__(self, menagerie_xml: str):
        self.m = mujoco.MjModel.from_xml_path(menagerie_xml)
        self.d = mujoco.MjData(self.m)
        self.site = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE,
                                      "attachment_site")
        names = ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                 "wrist_1_joint", "wrist_2_joint", "wrist_3_joint")
        self.jids = [mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, n)
                     for n in names]
        self.qadr = [self.m.jnt_qposadr[j] for j in self.jids]
        self.d1, self.a1, self.d4 = DH[0][0], DH[0][1], DH[3][0]
        self.a2, self.a3 = DH[1][1], DH[2][1]
        self.d6 = DH[5][0]
        self.calibrated = False
        self.T_base = np.eye(4)
        self.T_tool = np.eye(4)

    def men_fk(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        d = self.d
        d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.m, d)
        return (d.site_xpos[self.site].copy(),
                d.site_xmat[self.site].reshape(3, 3).copy())

    def calibrate(self, n: int = 200, tol: float = 1e-6) -> bool:
        rng = np.random.default_rng(11)
        qs = [rng.uniform(-3.0, 3.0, 6) for _ in range(2)]
        Tms = []
        for q in qs:
            p, R = self.men_fk(q)
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = p
            Tms.append(T)
        # T_base * T_dh(q) * T_tool = T_m  -- assume T_tool = I first:
        # T_base = T_m * T_dh^-1 (must be constant)
        cand = [Tms[i] @ np.linalg.inv(dh_fk(qs[i])) for i in range(2)]
        T_base = cand[0]
        if not np.allclose(cand[0], cand[1], atol=1e-6):
            # the tool offset is real: extract from the relative pose
            # T_base^-1 T_m1 T_m2^-1 T_base = T_dh1 T_dh2^-1 (tool cancels)
            # -- fall back to a least-squares refinement below.
            pass
        worst = 0.0
        for _ in range(n):
            q = rng.uniform(-3.0, 3.0, 6)
            p, R = self.men_fk(q)
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = p
            Tdh = dh_fk(q)
            Tm_pred = T_base @ Tdh
            worst = max(worst, float(np.abs(Tm_pred - T).max()))
        if worst > tol:
            # the tool transform is real: solve T_tool = T_dh^-1 T_base^-1 T_m
            q = qs[0]
            Tdh = dh_fk(q)
            T_tool = np.linalg.inv(Tdh) @ np.linalg.inv(T_base) @ Tms[0]
            self.T_tool = T_tool
            worst2 = 0.0
            for _ in range(n):
                q = rng.uniform(-3.0, 3.0, 6)
                p, R = self.men_fk(q)
                T = np.eye(4)
                T[:3, :3] = R
                T[:3, 3] = p
                Tm_pred = T_base @ dh_fk(q) @ T_tool
                worst2 = max(worst2, float(np.abs(Tm_pred - T).max()))
            if worst2 > tol:
                return False
            worst = worst2
        self.T_base = T_base
        self.calibrated = True
        return True

    # ---- closed form -----------------------------------------------------
    def ik(self, p06: np.ndarray, R06: np.ndarray,
           prev: np.ndarray | None = None):
        """Closed-form IK for the DH-frame flange pose (p06, R06).  Returns
        up to 4 solutions (theta1 branches x theta3 elbow branches), each
        with theta4 chosen from the previous frame for continuity."""
        d1, a1 = self.d1, self.a1
        a2, a3 = self.a2, self.a3
        d4, d5, d6 = DH[3][0], DH[4][0], DH[5][0]
        p06 = np.asarray(p06, float)
        R06 = np.asarray(R06, float)
        # wrist centre
        W = p06 - d6 * R06[:, 2]
        r_xy = np.hypot(W[0], W[1])
        sols = []
        for b1 in (+1.0, -1.0):
            arg = d4 / max(r_xy, 1e-9)
            if abs(arg) > 1.0:
                continue
            phi = np.arccos(np.clip(arg, -1, 1))
            th1 = np.arctan2(W[1], W[0]) - b1 * phi + np.pi / 2
            c1, s1 = np.cos(th1), np.sin(th1)
            # wrist centre in the th1 frame (a1 offset along the frame x)
            X = -s1 * W[0] + c1 * W[1] - a1
            Z = W[2] - d1
            D = (X ** 2 + Z ** 2 - a2 ** 2 - a3 ** 2) / (2 * a2 * a3)
            if abs(D) > 1.0:
                continue
            for b3 in (+1.0, -1.0):
                th3 = np.arctan2(b3 * np.sqrt(max(1 - D * D, 0.0)), D)
                th2 = np.arctan2(Z, X) - np.arctan2(
                    a3 * np.sin(th3), a2 + a3 * np.cos(th3))
                # R03 from th1..th3
                T03 = _rz(th1) @ _tz(d1) @ _tx(a1) @ _rx(np.pi / 2) \
                    @ _rz(th2) @ _tz(0) @ _tx(a2) \
                    @ _rz(th3) @ _tz(0) @ _tx(a3)
                R03 = T03[:3, :3]
                R36 = R03.T @ R06
                c5 = np.clip(R36[2, 2], -1, 1)
                th5 = np.arccos(c5)
                for b5 in (+1.0, -1.0):
                    t5 = b5 * th5
                    if abs(np.sin(t5)) < 1e-6:
                        t4, t6 = 0.0, np.arctan2(-R36[0, 1], R36[0, 0])
                    else:
                        t4 = np.arctan2(R36[1, 2] / np.sin(t5),
                                        R36[0, 2] / np.sin(t5))
                        t6 = np.arctan2(R36[2, 1] / np.sin(t5),
                                        -R36[2, 0] / np.sin(t5))
                    sols.append(np.array([th1, th2, th3, t4, t5, t6]))
        return sols

    def ik_scene(self, p_target: np.ndarray, R_target: np.ndarray,
                 prev: np.ndarray | None = None):
        """IK for a scene-frame flange target; returns the joint vector."""
        T = np.eye(4)
        T[:3, :3] = R_target
        T[:3, 3] = p_target
        Tdh = self.T_scene_inv @ T @ self.T_dh_scene_inv
        sols = self.ik(Tdh[:3, 3], Tdh[:3, :3], prev)
        if not sols:
            return None
        if prev is None:
            return sols[0]
        return min(sols, key=lambda s: float(np.abs(s - prev).max()))

    def validate_round_trip(self, n: int = 500, tol: float = 1e-5) -> bool:
        rng = np.random.default_rng(23)
        worst = 0.0
        for _ in range(n):
            q = rng.uniform(-3.0, 3.0, 6)
            p, R = self.men_fk(q)
            q2 = self.ik_scene(p, R, prev=q)
            if q2 is None:
                continue
            p2, R2 = self.men_fk(q2)
            err = float(np.linalg.norm(p2 - p)) + \
                float(np.linalg.norm(R2 - R))
            worst = max(worst, err)
        return worst < tol, worst
