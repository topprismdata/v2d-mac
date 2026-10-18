"""r12 go/no-go: UR5e reachability over the registered palm task path.

Ceiling-mounted UR5e at the existing shoulder flanges (+-0.235, 0, 1.000),
180 deg about x (arm hangs down).  Targets = the registered task-space palm
path (Front B2: world-centred anchor + measured MediaPipe wrist lateral,
pre-lock-trimmed, k=5 median-filtered) + flange offset anchor z 0.87.

Damped-least-squares position IK (lam 1e-3, 150 iters, 3 restarts);
converged < 2 mm.  Result (2026-10-18, both sides 100%):
  right: 130/130 frames reachable, left: 130/130.
Run:  runs/.venv-retarget/bin/python runs/w6b/ur5e/reachability_scan.py
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np
import mujoco
from scipy.signal import medfilt

UR5E = str(REPO / "runs/upstream/mujoco_menagerie/universal_robots_ur5e/ur5e.xml")


def main() -> None:
    m = mujoco.MjModel.from_xml_path(UR5E)
    d = mujoco.MjData(m)
    site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    names = ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
             "wrist_1_joint", "wrist_2_joint", "wrist_3_joint")
    jids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in names]
    qadr = [m.jnt_qposadr[j] for j in jids]
    vjad = [m.jnt_dofadr[j] for j in jids]
    rng_lo = m.jnt_range[jids][:, 0]
    rng_hi = m.jnt_range[jids][:, 1]
    Rm = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], float)

    def dls(base, tgt_world, q0):
        best = None
        for r in range(3):
            q = q0 if r == 0 else np.random.default_rng(r + 2).uniform(rng_lo, rng_hi)
            e = 9.9
            for _ in range(150):
                d.qpos[qadr] = q
                mujoco.mj_kinematics(m, d)
                mujoco.mj_comPos(m, d)
                err = Rm.T @ (tgt_world - base) - d.site_xpos[site]
                e = np.linalg.norm(err)
                if e < 2e-3:
                    return True, e, q
                jac = np.zeros((3, m.nv))
                mujoco.mj_jacSite(m, d, jac, None, site)
                J = jac[:, vjad]
                dq = np.linalg.solve(J @ J.T + 1e-3 * np.eye(3), err)
                q = np.clip(q + J.T @ dq, rng_lo, rng_hi)
            d.qpos[qadr] = q
            mujoco.mj_kinematics(m, d)
            e = np.linalg.norm(Rm.T @ (tgt_world - base) - d.site_xpos[site])
            if best is None or e < best:
                best = e
        return False, best, q

    from retarget.retarget_dual import load_streams
    streams, rec, go, wl_lat, wh, legacy = load_streams()
    rng = np.random.default_rng(1)
    for side, mount_x in (("right", 0.235), ("left", -0.235)):
        base = np.array([mount_x, 0.0, 1.0])
        lat = wl_lat[side].copy()
        inf = np.abs(lat).max(axis=1)
        lock = int(np.argmax(inf <= 0.3))
        lat[:lock] = lat[lock]
        lat = np.stack([medfilt(lat[:, c], 5) for c in range(3)], axis=1)
        anchor = np.array([0.0, 0.0, 0.87])
        targets = anchor[None, :] + lat
        q0 = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0])
        reach, n, fails = 0, 0, []
        for t in range(0, targets.shape[0], 3):
            res = dls(base, targets[t], q0)
            reach += res[0]
            n += 1
            if not res[0]:
                fails.append((t, round(res[1], 3)))
            q0 = res[2]
        verdict = "PASS" if reach == n else "FAIL"
        print(f"{side}: reachable {reach}/{n} = {100*reach/n:.1f}%  [{verdict}]"
              + (f"  fail frames {fails[:6]}" if fails else ""))


if __name__ == "__main__":
    main()
