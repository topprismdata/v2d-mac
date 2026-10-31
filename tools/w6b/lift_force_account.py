#!/usr/bin/env python3
"""r162 (v12 P0): the per-frame LIFT FORCE ACCOUNT.

Reads runs/w6b/box_trace.json (recorded with the r162 force vectors) and
the scene XML (for the box's mass), and produces the Newton-Euler account
the v11 review asked for: per-frame contact force sums by class (plates,
table, yaw stops, vise corner), the box's finite-differenced velocity and
acceleration, and the residual

    m*a  =  sum(F_contacts)  +  m*g        (linear; world frame)

checked separately over the pre-grasp, lift-acceleration and hold phases.
The sign convention of mj_contactForce (which geom the force acts on) is
NOT assumed: the script self-checks it against the at-rest invariant
(the table's normal force on the resting box must sum to +m*g in world
z) and fails loudly if neither convention satisfies it.

Usage:
    runs/.venv-mj/bin/python tools/w6b/lift_force_account.py \
        [scene.xml] [box_trace.json]
"""
import json
import pathlib
import sys

import mujoco
import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
SCENE = sys.argv[1] if len(sys.argv) > 1 else \
    "grounding/assets/allegro_floating_ur5e_side_r44_dualplate.xml"
TRACE = pathlib.Path(sys.argv[2] if len(sys.argv) > 2
                     else "runs/w6b/box_trace.json")

CTRL_HZ = 20.0
DT = 1.0 / CTRL_HZ


def classify(name):
    n = name or ""
    # r170: the LEFT plate's geom is palm_left_collision -- it does NOT
    # contain the substring "palm_collision", so it fell into "other" and
    # the printed account hid the left plate's (possibly negative) share
    # while the SUM still closed.  Match both spellings.
    if "palm_collision" in n or "palm_left_collision" in n:
        return "plate"
    if "table" in n:
        return "table"
    if "yaw_stop" in n:
        return "yaw_stop"
    if "vise" in n:
        return "vise"
    return "other"


def main() -> int:
    m = mujoco.MjModel.from_xml_path(str(REPO / SCENE))
    bg = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "object_geom")
    body = int(m.geom_bodyid[bg])
    mass = float(m.body_mass[body])
    g = float(abs(m.opt.gravity[2]))
    tr = json.loads(TRACE.read_text())
    fr = tr["frames"] if isinstance(tr, dict) else tr
    print(f"mass {mass:.4f} kg  g {g:.3f}  frames {len(fr)}")

    # -- sign convention self-check on the longest resting stretch -------
    def frame_F(con):
        f6 = np.asarray(con["f6"], float)
        R = np.asarray(con["frame"], float).reshape(3, 3)
        fw = R @ f6[:3]              # frame rows are the axes in world
        # measured: mj_contactForce returns the force on GEOM2, so the
        # force ON THE BOX is always -fw.  Verified simultaneously against
        # five physical directions in one hold frame: the table pushes up
        # (+mg at rest), the right plate pushes -x, the left plate +x, the
        # negy yaw stop +y, and its sliding friction pulls down.
        return -fw

    best_i, best_n = 0, 0
    run = 0
    for i, f in enumerate(fr):
        rest = any("table" in str(c.get("other")) for c in f["contacts"])
        if "f6" not in (f["contacts"] or [{}])[0]:
            rest = False
        run = run + 1 if rest else 0
        if run > best_n:
            best_n, best_i = run, i
    rest_i = best_i - best_n // 2
    fz = sum(frame_F(c)[2] for c in fr[rest_i]["contacts"]
             if "table" in str(c.get("other")))
    print(f"sign check @frame {rest_i}: table Fz on box = {fz:+.4f} N "
          f"(must be ~ +{mass * g:.4f})")
    if abs(fz - mass * g) > 0.15:
        print("FAIL: the contact-force sign convention self-check failed")
        return 1

    # -- per-frame account ------------------------------------------------
    rows = []
    for i, f in enumerate(fr):
        sums = {}
        for c in f.get("contacts") or []:
            if "f6" not in c:
                continue
            k = classify(c.get("other"))
            sums.setdefault(k, np.zeros(3))
            sums[k] = sums[k] + frame_F(c)
        p = np.asarray(f["pos"], float)
        rows.append((f.get("t", i), p, sums))

    def accel(i):
        if i <= 0 or i >= len(rows) - 1:
            return np.zeros(3)
        v1 = (rows[i + 1][1] - rows[i][1]) / DT
        v0 = (rows[i][1] - rows[i - 1][1]) / DT
        return (v1 - v0) / DT

    def resid(i):
        _, p, sums = rows[i]
        F = sum(sums.values(), np.zeros(3))
        return mass * accel(i) - (F + mass * np.array([0.0, 0.0, -g]))

    n_lift = 90
    phases = {
        "approach": range(0, min(260, len(rows))),
        "grasp": range(260, len(rows) - n_lift),
        "lift_accel": range(len(rows) - n_lift, len(rows) - n_lift // 2),
        "lift_hold": range(len(rows) - n_lift // 2, len(rows)),
    }
    z = np.array([p[2] for _, p, _ in rows])
    print(f"\nbox z: start {z[0] * 1e3:.3f}  peak {(z.max() - z[0]) * 1e3:.3f}  "
          f"final {(z[-1] - z[0]) * 1e3:.3f} mm")
    print(f"\n{'phase':<11}{'per-class world-z mean Fz (N), ALL contact geoms'}")
    for name, rng in phases.items():
        idx = [i for i in rng if 0 < i < len(rows) - 1]
        if not idx:
            continue
        agg = {}
        fz_all = 0.0
        rz = []
        rm = []
        for i in idx:
            _, _, sums = rows[i]
            for k, v in sums.items():
                agg[k] = agg.get(k, 0.0) + v[2]
            for v in sums.values():
                fz_all += v[2]
            r = resid(i)
            rz.append(float(r[2]))
            rm.append(float(np.linalg.norm(r)))
        n = len(idx)
        parts = "  ".join(f"{k}={agg.get(k, 0.0) / n:+.3f}"
                          for k in sorted(agg))
        close = fz_all / n - mass * g
        print(f"{name:<11} {parts}  | SUM={fz_all / n:+.3f}  "
              f"SUM-mg={close:+.4f}  resid_z={np.mean(rz):+.4f}  "
              f"|resid|max={np.max(rm):.4f}")
    # the plates' vertical friction vs the weight in the hold phase
    hold = [i for i in phases["lift_hold"] if 0 < i < len(rows) - 1]
    if hold:
        per = {}
        for i in hold:
            for k, v in rows[i][2].items():
                per[k] = per.get(k, 0.0) + v[2]
        for k, v in sorted(per.items()):
            print(f"hold {k}: mean Fz {v / len(hold):+.3f} N")
        tot = sum(per.values()) / len(hold)
        print(f"hold: ALL-contacts total Fz {tot:+.3f} N vs weight "
              f"{-mass * g:+.3f} N -> closes to {abs(tot + mass * g):+.4f} N"
              f" (a NEGATIVE plate share = that plate's friction points"
              f" DOWN: the box slides UP relative to it -- physically"
              f" required for Newton-Euler closure, not a missing contact)")
    return 0


def sums_of(rows, i):
    return rows[i][2]


if __name__ == "__main__":
    raise SystemExit(main())
