#!/usr/bin/env python3
"""r115: how low can the side cell's bases go?

The v9 review flagged the visual gap -- the forearms enter the frame
35-45 deg downward against the reference's ~5-10 deg -- and r114
located the preload's blocker: pressing the palm into the face moves
the arm onto the boundary of r86's table-clearance constraint.  Both
point at the same lever: the bases are mounted HIGH (z = 1.35 m) while
the palms work at z ~ 0.79 m, so the arm has to fold down 0.56 m over
a 0.70 m reach (~39 deg).

This probe answers, for each candidate base height, with the ARM's IK
alone (seconds, no full pipeline):

  * can the arm still reach the KNOWN-GOOD palm pose?
  * how far does the solution leave the path-verified branch (the
    anchor's tolerance is ANCHOR_TOL = 0.60 rad per joint)?
  * what does the FOREARM's entry angle become (the review's metric)?
  * does the arm still clear the table (r86's hard constraint)?

Usage: base_height_probe.py
"""
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as Rot

SCENE = Path("grounding/assets/allegro_floating_ur5e_side_r37.xml")
REFINE = Path("runs/w6b/palm_pose_refine.json")
ANCHOR_TOL = 0.60

ARM = ("shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2",
       "wrist_3")
LINKS = ("upper_arm_link", "forearm_link", "wrist_1_link", "wrist_2_link",
         "wrist_3_link")
HEIGHTS = (1.35, 1.20, 1.05, 0.90)


def variant(z):
    """A copy of the side scene with both bases at the height z.

    The scene carries RELATIVE mesh paths, so the variant has to live
    in the same directory as the original.
    """
    out = Path("grounding/assets") / f"_baseprobe_{int(round(z*100)):03d}.xml"
    txt = SCENE.read_text()
    for old in ('pos="0.70 0 1.35"', 'pos="-0.70 0 1.35"'):
        x, y, _ = old.split('"')[1].split()
        assert txt.count(old) == 1, old
        txt = txt.replace(old, f'pos="{x} {y} {z:.4f}"')
    out.write_text(txt)
    return out


def solve_arm(m, d, palm, aq, ad, tp, Rt, seed, lo, hi):
    """The refine tool's arm IK: a damped least squares on the palm's
    pose, with the joint limits as a clip."""
    q = seed.copy()
    for _ in range(250):
        d.qpos[:] = m.qpos0
        d.qpos[aq] = q
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        e_p = tp - np.array(d.xpos[palm])
        Rq = np.zeros(9)
        mujoco.mju_quat2Mat(Rq, d.xquat[palm])
        e_r = Rot.from_matrix(
            Rt @ np.array(Rq).reshape(3, 3).T).as_rotvec()
        if (np.linalg.norm(e_p) < 2e-4
                and np.linalg.norm(e_r) < 3e-3):
            break
        jacp = np.zeros((3, m.nv))
        jacr = np.zeros((3, m.nv))
        mujoco.mj_jacBody(m, d, jacp, jacr, palm)
        J = np.vstack([jacp[:, ad], jacr[:, ad]])
        try:
            dq = np.linalg.solve(J @ J.T + 1e-4 * np.eye(6),
                                  np.concatenate([e_p, e_r]))
        except np.linalg.LinAlgError:
            break
        q = np.clip(q + J.T @ dq, lo, hi)
    d.qpos[:] = m.qpos0
    d.qpos[aq] = q
    mujoco.mj_kinematics(m, d)
    mujoco.mj_comPos(m, d)
    Rq = np.zeros(9)
    mujoco.mju_quat2Mat(Rq, d.xquat[palm])
    ep = float(np.linalg.norm(tp - np.array(d.xpos[palm])))
    er = float(np.linalg.norm(Rot.from_matrix(
        Rt @ np.array(Rq).reshape(3, 3).T).as_rotvec()))
    return q, ep, er


def main():
    ref = json.loads(REFINE.read_text())
    rows = []
    for z in HEIGHTS:
        scene = variant(z)
        m = mujoco.MjModel.from_xml_path(str(scene))
        d = mujoco.MjData(m)
        tg = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "table_slab")
        top = float(m.geom_pos[tg][2] + m.geom_size[tg][2])
        row = {"base_z": z, "sides": {}}
        for side, prefix in (("right", "ur5e_r"), ("left", "ur5e_l")):
            # the palm bodies are "palm" (right) and "palm_left" (left)
            palm_id = mujoco.mj_name2id(
                m, mujoco.mjtObj.mjOBJ_BODY,
                "palm" if side == "right" else "palm_left")
            jids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT,
                                      f"{prefix}_{a}_joint") for a in ARM]
            if min(jids) < 0:
                raise SystemExit(f"missing joints for {prefix}")
            aq = [int(m.jnt_qposadr[j]) for j in jids]
            ad = [int(m.jnt_dofadr[j]) for j in jids]
            lo = m.jnt_range[jids][:, 0]
            hi = m.jnt_range[jids][:, 1]

            best = ref["sides"][side]["best"][0]
            anchor = np.asarray(best["arm"], float)
            tp = np.asarray(best["palm_pos"], float)
            Rt = np.asarray(best["palm_R"], float)
            rng = np.random.default_rng(7)
            solved = None
            for trial in range(24):
                seed = anchor if trial == 0 else (
                    anchor + rng.normal(0.0, 0.12, 6))
                q, ep, er = solve_arm(m, d, palm_id, aq, ad, tp, Rt, seed,
                                      lo, hi)
                if ep < 2e-3 and er < 0.01:
                    dq_ = q - anchor
                    dq_ = dq_ - 2 * np.pi * np.round(dq_ / (2 * np.pi))
                    cand = (float(np.abs(dq_).max()), q, ep, er)
                    if solved is None or cand[0] < solved[0]:
                        solved = cand
            if solved is None:
                row["sides"][side] = {"solved": False}
                print(f"base z {z:.2f} | {side:5s} | NOT REACHABLE")
                continue
            dq_max, q, ep, er = solved

            # the forearm's entry angle: the direction from the elbow to
            # the wrist_2 link's origin, in the world
            d.qpos[:] = m.qpos0
            d.qpos[aq] = q
            mujoco.mj_kinematics(m, d)
            elb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY,
                                    f"{prefix}_forearm_link")
            w2 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY,
                                   f"{prefix}_wrist_2_link")
            v = np.array(d.xpos[w2]) - np.array(d.xpos[elb])
            # the review's metric is the entry angle IN THE IMAGE: the
            # render camera sits at azimuth 90 deg (looking along -y), so
            # the image's horizontal is the world x and the vertical is
            # (tilted) the world z.  Using the full horizontal distance
            # would mix in the y component the camera does not show.
            entry = float(np.degrees(np.arctan2(-v[2], abs(v[0]))))
            entry_h = float(np.degrees(np.arctan2(
                -v[2], np.hypot(v[0], v[1]))))

            # r86's constraint: the arm's collision geoms vs the table.
            # A conservative margin: each arm geom's bounding sphere
            # (rbound) against the table's top plane, plus the hard
            # check (any contact with the table).
            d.qpos[:] = m.qpos0
            d.qpos[aq] = q
            mujoco.mj_forward(m, d)
            clear = 1e9
            hits = []
            for g in range(m.ngeom):
                gb = int(m.geom_bodyid[g])
                bn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, gb) or ""
                if not bn.startswith(prefix):
                    continue
                if not any(ln in bn for ln in LINKS):
                    continue
                if float(m.geom_rbound[g]) <= 0:
                    continue
                clear = min(clear, float(d.geom_xpos[g][2])
                            - float(m.geom_rbound[g]) - top)
            for c in range(d.ncon):
                g1, g2 = int(d.contact[c].geom1), int(d.contact[c].geom2)
                if int(m.geom_bodyid[g1]) == 0 or int(m.geom_bodyid[g2]) == 0:
                    for g in (g1, g2):
                        bn = mujoco.mj_id2name(
                            m, mujoco.mjtObj.mjOBJ_BODY,
                            int(m.geom_bodyid[g])) or ""
                        if bn.startswith(prefix) and any(
                                ln in bn for ln in LINKS):
                            hits.append((bn, round(float(d.contact[c].dist),
                                                   5)))
            row["sides"][side] = {
                "solved": True,
                "e_p_mm": round(ep * 1e3, 4),
                "e_r_deg": round(er * 1e3, 4),
                "branch_delta_rad": round(dq_max, 4),
                "branch_ok": bool(dq_max <= ANCHOR_TOL),
                "entry_deg_down": round(entry, 2),
                "entry_deg_down_3d": round(entry_h, 2),
                "arm_clearance_mm": round(clear * 1e3, 2),
                "table_contact": hits[:3],
                "arm": [round(float(v), 4) for v in q],
            }
            print(f"base z {z:.2f} | {side:5s} | the entry "
                  f"{entry:6.2f} deg down (the 3d "
                  f"{entry_h:5.2f}) | the branch {dq_max:.3f} rad "
                  f"{'(ok)' if dq_max <= ANCHOR_TOL else '(OVER)'} | the arm's "
                  f"clearance {clear*1e3:8.2f} mm | the e_p {ep*1e3:.3f} mm "
                  f"| the table's contacts {len(hits)}")
        rows.append(row)
    Path("runs/w6b/base_height_probe.json").write_text(
        json.dumps({"rows": rows}, indent=1))
    print("written: runs/w6b/base_height_probe.json")


if __name__ == "__main__":
    raise SystemExit(main())
