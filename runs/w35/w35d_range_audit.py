"""Wave 3.5-D task 1: thumb joint-range fidelity audit.

Questions answered with numbers:
  1. Do MJCF thj0..thj3 ranges/axes/ctrlranges match the canonical Allegro URDF?
  2. Does the MJCF thumb CHAIN reproduce URDF thumb-chain FK bit-for-bit
     (joint origins, axes, rpy) so the "opposition cone" is not distorted?
  3. For reference: long-finger ranges vs URDF (same table).

Sources:
  - MJCF: grounding/assets/allegro_floating.xml (loaded via mujoco)
  - URDF A: runs/upstream/dex-retargeting/assets/robots/hands/allegro_hand/allegro_hand_right.urdf
  - URDF B (live fetch, drake mirror of felixduvallet/allegro-hand-description):
    /tmp/allegro_drake_right.urdf (fetched 2026-10-15 from
    https://raw.githubusercontent.com/Daniella1/urdf_files_dataset/main/urdf_files/drake/allegro_hand_description/urdf/allegro_hand_description_right.urdf)

No MJCF is modified. Output: runs/w35/w35d_range_audit.json
"""
from __future__ import annotations

import json
import pathlib
import sys
import xml.etree.ElementTree as ET

import numpy as np

REPO = pathlib.Path("/Users/guohongbin/v2d-mac")
sys.path.insert(0, str(REPO))

import mujoco  # noqa: E402

XML = REPO / "grounding/assets/allegro_floating.xml"
URDF_A = (REPO / "runs/upstream/dex-retargeting/assets/robots/hands/"
          "allegro_hand/allegro_hand_right.urdf")
URDF_B = pathlib.Path("/tmp/allegro_drake_right.urdf")
OUT = REPO / "runs/w35/w35d_range_audit.json"

TH = ("thj0", "thj1", "thj2", "thj3")
URDF_TH = {"thj0": "joint_12", "thj1": "joint_13", "thj2": "joint_14",
           "thj3": "joint_15"}
URDF_TH_ALT = {k: v + ".0" for k, v in URDF_TH.items()}
FF = ("ffj0", "ffj1", "ffj2", "ffj3")
URDF_FF = {"ffj0": "joint_0.0", "ffj1": "joint_1.0", "ffj2": "joint_2.0",
           "ffj3": "joint_3.0"}


def rpy_to_mat(rpy) -> np.ndarray:
    r, p, y = rpy
    Rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)],
                   [0, np.sin(r), np.cos(r)]])
    Ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0],
                   [-np.sin(p), 0, np.cos(p)]])
    Rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0],
                   [0, 0, 1]])
    return Rz @ Ry @ Rx


def quat_to_mat(w, x, y, z) -> np.ndarray:
    n = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def parse_urdf(path: pathlib.Path, suffix: str) -> dict:
    joints = {}
    root = ET.parse(path).getroot()
    for j in root.findall("joint"):
        n = j.get("name")
        if j.get("type") not in ("revolute", "continuous", "prismatic"):
            continue
        lim = j.find("limit")
        ax = j.find("axis")
        org = j.find("origin")
        joints[n] = {
            "lower": float(lim.get("lower")) if lim is not None else None,
            "upper": float(lim.get("upper")) if lim is not None else None,
            "axis": [float(v) for v in ax.get("xyz").split()] if ax is not None else [0, 0, 1],
            "xyz": [float(v) for v in org.get("xyz").split()] if org is not None else [0, 0, 0],
            "rpy": [float(v) for v in org.get("rpy").split()] if org is not None and org.get("rpy") else [0, 0, 0],
        }
    return joints


def urdf_fk_tip(joints: dict, names: list[str], qs: list[float],
                T_world_base: np.ndarray) -> np.ndarray:
    """URDF FK: successive Trans(xyz)*R_rpy then Rot(axis, q) about child origin."""
    T = T_world_base.copy()
    for name, q in zip(names, qs):
        j = joints[name]
        T = T @ np.eye(4)
        R_off = rpy_to_mat(j["rpy"])
        T_off = np.eye(4)
        T_off[:3, :3] = R_off
        T_off[:3, 3] = j["xyz"]
        T = T @ T_off
        a = np.array(j["axis"], dtype=float)
        a = a / np.linalg.norm(a)
        K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
        R = np.eye(3) + np.sin(q) * K + (1 - np.cos(q)) * (K @ K)
        Rj = np.eye(4)
        Rj[:3, :3] = R
        T = T @ Rj
    return T[:3, 3]


def axis_world(T: np.ndarray, axis: list[float]) -> np.ndarray:
    return T[:3, :3] @ (np.array(axis, dtype=float) /
                        np.linalg.norm(axis))


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(XML))
    th_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in TH]
    ff_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in FF]

    mjcf = {}
    for n, jid in zip(TH + FF, th_ids + ff_ids):
        mjcf[n] = {
            "range_rad": [float(model.jnt_range[jid][0]), float(model.jnt_range[jid][1])],
            "axis": [float(v) for v in model.jnt_axis[jid]],
        }
    # actuator ctrlranges for thumb
    act = {}
    for nm in ("tha0", "tha1", "tha2", "tha3"):
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, nm)
        jid = model.actuator_trnid[aid][0]
        act[nm] = {"ctrlrange": [float(model.actuator_ctrlrange[aid][0]),
                                 float(model.actuator_ctrlrange[aid][1])],
                   "jnt_range": [float(model.jnt_range[jid][0]),
                                 float(model.jnt_range[jid][1])]}

    sources = {}
    urdfs = {"urdfA_dex_retargeting": (URDF_A, URDF_TH_ALT, URDF_FF),
             "urdfB_felixduvallet_drake_mirror": (URDF_B, URDF_TH, None)}
    for tag, (path, thmap, ffmap) in urdfs.items():
        jj = parse_urdf(path, "")
        table = {}
        for thn, un in thmap.items():
            j = jj[un]
            table[thn] = {"urdf_name": un, "range_rad": [j["lower"], j["upper"]],
                          "axis": j["axis"], "xyz": j["xyz"], "rpy": j["rpy"]}
        if ffmap:
            for fn, un in ffmap.items():
                j = jj[un]
                table[fn] = {"urdf_name": un, "range_rad": [j["lower"], j["upper"]],
                             "axis": j["axis"]}
        sources[tag] = table

    # diff table MJCF vs each URDF
    diffs = {}
    for tag, table in sources.items():
        rows = []
        for jn in TH + FF:
            if jn not in table:
                continue
            ur = table[jn]["range_rad"]
            mjr = mjcf[jn]["range_rad"]
            rows.append({
                "joint": jn,
                "urdf_range_rad": ur,
                "mjcf_range_rad": mjr,
                "d_lower_rad": round(mjr[0] - ur[0], 6),
                "d_upper_rad": round(mjr[1] - ur[1], 6),
                "urdf_axis": table[jn].get("axis"),
                "mjcf_axis": mjcf[jn]["axis"],
                "axis_match": bool(np.allclose(table[jn].get("axis", mjcf[jn]["axis"]),
                                               mjcf[jn]["axis"])),
            })
        diffs[tag] = rows

    # FK equivalence check: palm frame fixed, N random thumb configs.
    # MJCF: wrist at home, set thumb qpos, read th_tip world pos + joint axes.
    data = mujoco.MjData(model)
    tip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "th_tip")
    th_bodies = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b)
                 for b in ("th_base", "th_proximal", "th_medial", "th_distal")]

    rng = np.random.default_rng(7)
    N = 256
    urdf_joints = parse_urdf(URDF_A, "")
    # palm world transform in MJCF: hand_root (0,0,0.92) * palm quat (0,1,0,1)
    T_hand = np.eye(4)
    T_hand[:3, 3] = [0, 0, 0.92]
    T_palm = T_hand @ np.eye(4)
    T_palm[:3, :3] = quat_to_mat(0, 1, 0, 1)
    # URDF base_link == MJCF palm frame (palm geom matches palm_link geoms)
    urdf_names = [URDF_TH_ALT[n] for n in TH]

    max_err_mm = 0.0
    max_axis_deg = 0.0
    worst = None
    for i in range(N):
        qs = rng.uniform([model.jnt_range[j][0] for j in th_ids],
                         [model.jnt_range[j][1] for j in th_ids])
        data.qpos[:] = model.qpos0
        for jid, q in zip(th_ids, qs):
            data.qpos[model.jnt_qposadr[jid]] = q
        mujoco.mj_forward(model, data)
        tip_mj = data.xpos[tip_id].copy()
        axes_mj = [data.xaxis[b].copy() for b in th_bodies]

        T = T_palm.copy()
        axes_ur = []
        for name, q in zip(urdf_names, qs):
            j = urdf_joints[name]
            T_off = np.eye(4)
            T_off[:3, :3] = rpy_to_mat(j["rpy"])
            T_off[:3, 3] = j["xyz"]
            T = T @ T_off
            axes_ur.append(axis_world(T, j["axis"]))
            a = np.array(j["axis"], dtype=float)
            a /= np.linalg.norm(a)
            K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
            Rj = np.eye(4)
            Rj[:3, :3] = np.eye(3) + np.sin(q) * K + (1 - np.cos(q)) * (K @ K)
            T = T @ Rj
        tip_ur = T[:3, 3]
        err_mm = float(np.linalg.norm(tip_mj - tip_ur) * 1000)
        for am, au in zip(axes_mj, axes_ur):
            dot = float(np.clip(np.dot(am, au), -1, 1))
            max_axis_deg = max(max_axis_deg, float(np.degrees(np.arccos(dot))))
        if err_mm > max_err_mm:
            max_err_mm = err_mm
            worst = {"q": [round(float(v), 4) for v in qs],
                     "tip_mjcf_mm": [round(float(v) * 1000, 4) for v in tip_mj],
                     "tip_urdf_mm": [round(float(v) * 1000, 4) for v in tip_ur]}
    fk_check = {"n_samples": N, "max_tip_err_mm": round(max_err_mm, 6),
                "max_joint_axis_err_deg": round(max_axis_deg, 6),
                "worst_sample": worst}

    out = {
        "wave": "3.5-D task1 thumb range fidelity audit",
        "date": "2026-10-15",
        "mjcf": XML.name,
        "urdf_sources": {
            "urdfA_dex_retargeting": str(URDF_A.relative_to(REPO)),
            "urdfB_felixduvallet_drake_mirror":
                "https://raw.githubusercontent.com/Daniella1/urdf_files_dataset/main/"
                "urdf_files/drake/allegro_hand_description/urdf/"
                "allegro_hand_description_right.urdf (fetched 2026-10-15, converted from "
                "felixduvallet/allegro-hand-description allegro_hand_description_right.xacro)",
        },
        "mjcf_thumb_actuators": act,
        "diffs": diffs,
        "fk_equivalence_urdfA_vs_mjcf": fk_check,
        "unreachable_sources": [
            "https://www.wonikrobotics.com/upload/board/Allegro%20Hand%20V5_Rev.3,250514_6686b.pdf (HTTP 403)",
            "https://www.roboticscenter.ai/store/product/allegro-hand (no per-joint range table on page)",
            "https://www.allegrohand.com/sub/tech/t.php?idx=6 (HTTP 403)",
        ],
    }
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("mjcf_thumb_actuators", "fk_equivalence_urdfA_vs_mjcf")}, indent=1))
    for tag, rows in diffs.items():
        print(f"--- {tag}")
        for r in rows:
            print(f"  {r['joint']:5s} urdf {r['urdf_range_rad']} mjcf {r['mjcf_range_rad']} "
                  f"d=({r['d_lower_rad']:+.4f},{r['d_upper_rad']:+.4f}) axis_match={r['axis_match']}")


if __name__ == "__main__":
    main()
