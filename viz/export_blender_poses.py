"""Export the W6d replay as per-frame geom poses for Blender Cycles.

Physics truth stays in MuJoCo: this script replays the delivered qpos
(v20.1 65-wide, arm/pad angles included) through the fidelity model —
the SAME model the MuJoCo renderer uses (frozen XML + visual layer) —
and dumps every geom's world pose per frame.  Blender only consumes the
truth (photoreal skinning); nothing here touches MJCF/simulation code.

Output viz/blender/poses.json:
  meta   : frame count, fps, source, axis note
  geoms  : [{name, type, size, family}]  (family -> Blender material)
  frames : [{t, q:[65], poses:[[x,y,z,qw,qx,qy,qz] * ngeom]}]

Axis mapping: IDENTITY — MuJoCo and Blender are both right-handed z-up;
quaternions are wxyz in both (Blender rotation_quaternion same order).
The floor plane and lights are built natively in Blender, so the mujoco
'floor' plane geom is skipped.

    runs/.venv-mj/bin/python viz/export_blender_poses.py
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "viz"))

OUT_DIR = REPO / "viz/blender"
OUT = OUT_DIR / "poses.json"

FAMILIES = [
    # v25/r10 工业臂风格令（docs/images/arm_reference_real.png）:
    # 结构蓝双板 / 黑关节鼓 / 黄铜支柱 / 蓝安装盘 / 黑六角底板
    ("struct_blue", lambda n: any(k in n for k in
                                  ("plate", "mount_disc", "mounting"))),
    ("black_shell", lambda n: any(k in n for k in
                                  ("drum", "shell", "housing", "rotary",
                                   "hex_base", "base_plate"))),
    ("brass", lambda n: any(k in n for k in ("pillar", "brass", "standoff"))),
    ("poly", lambda n: "_tip_collision" in n),          # 指垫 聚氨酯
    ("floral", lambda n: n in ("skin_floral",)),        # 盒身花卉
    ("tissue", lambda n: n == "tissue"),
    ("film", lambda n: n == "slot_film"),
    ("wood", lambda n: n.startswith(("table_slab", "table_apron", "table_leg",
                                     "peg_shelf"))),
    ("peg", lambda n: n == "pegboard_wall"),
    ("frosted", lambda n: n.startswith("mon_") and n != "mon_screen"),
    ("screen", lambda n: n == "mon_screen"),
    ("prop", lambda n: n.startswith("prop_")),
    ("cable", lambda n: "_cable" in n),
]


def family_of(name: str) -> str:
    if name == "floor":
        return "skip"
    for fam, pred in FAMILIES:
        if pred(name):
            return fam
    if name.startswith(("gantry_", "mon_")):
        return "powder" if name.startswith("gantry_") else "frosted"
    if name.startswith(("table_", "peg_")):
        return "wood"
    if name.startswith("prop_"):
        return "prop"
    if name.startswith(("ff_", "mf_", "rf_", "th_", "palm", "hand_")):
        return "anodalu"   # Allegro 手保持现状（阳极铝）
    if name.startswith("arm_"):
        return "arm_segment"  # v19 旧段名（r10 前兼容）
    return "powder"        # 其余结构件默认喷塑钢


def mat_to_quat_wxyz(q: np.ndarray) -> np.ndarray:
    """Rotation matrix (3x3) -> quaternion wxyz (robust branch method)."""
    t = float(np.trace(q))
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        return np.array([0.25 * s, (q[2, 1] - q[1, 2]) / s,
                         (q[0, 2] - q[2, 0]) / s, (q[1, 0] - q[0, 1]) / s])
    i = int(np.argmax(np.diag(q)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = np.sqrt(1.0 + q[i, i] - q[j, j] - q[k, k]) * 2
    quat = np.zeros(4)
    quat[0] = (q[k, j] - q[j, k]) / s
    quat[i + 1] = 0.25 * s
    quat[j + 1] = (q[j, i] + q[i, j]) / s
    quat[k + 1] = (q[k, i] + q[i, k]) / s
    return quat


def main() -> None:
    import mujoco
    import scene_fidelity as sf

    data = json.loads((REPO / "viz/panel_data_v2.json").read_text())
    frames = data["replay"]["frames"]
    width = len(frames[0]["qpos"])

    # scene selection by qpos width: the replay carries the FULL qpos of the
    # scene it was generated on (floating=65, ur5e=59)
    scenes = {"floating": sf.SCENE_FLOATING, "ur5e": sf.SCENE_UR5E}
    scene_kind = scene = None
    for kind, xml in scenes.items():
        import mujoco as _mj
        nq = int(_mj.MjModel.from_xml_path(str(xml)).nq)
        if nq == width:
            scene_kind, scene = kind, xml
            break
    if scene is None:
        raise SystemExit(f"replay qpos width {width} matches no known scene "
                         f"(floating=65, ur5e=59)")
    print(f"scene: {scene_kind} (nq {width})")

    m = sf.build_fidelity_model(scene)
    d = mujoco.MjData(m)
    obj_adr = int(m.jnt_qposadr[mujoco.mj_name2id(
        m, mujoco.mjtObj.mjOBJ_JOINT, "object_free")])

    geoms = []
    for g in range(m.ngeom):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or f"geom_{g}"
        fam = family_of(name)
        if fam == "skip":
            continue
        geoms.append({"name": name, "type": int(m.geom_type[g]),
                      "size": [float(v) for v in m.geom_size[g]],
                      "family": fam})

    # name->index remap after skipping the floor
    keep = [g for g in range(m.ngeom)
            if family_of(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g)
                         or "") != "skip"]

    # v25/r12 prep: when the UR5e lands (menagerie universal_robots_ur5e,
    # welded into the scene by W6b), its LINK poses drive real-CAD STL meshes
    # in Blender -- far better than primitives for a photoreal arm
    UR_LINK_SUFFIXES = ("shoulder_link", "upper_arm_link", "forearm_link",
                        "wrist_1_link", "wrist_2_link", "wrist_3_link")
    ur_bodies = []
    for b in range(m.nbody):
        n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or ""
        if "ur5e" in n and n.endswith(UR_LINK_SUFFIXES):
            ur_bodies.append((b, n))
    ur_bodies = sorted(set(ur_bodies))

    out_frames = []
    for fr in frames:
        d.qpos[:] = np.asarray(fr["qpos"])
        d.qpos[obj_adr:obj_adr + 7] = np.asarray(fr["obj"])  # the pinned trap
        mujoco.mj_forward(m, d)
        poses = []
        for g in keep:
            p = d.geom_xpos[g]
            qw, qx, qy, qz = mat_to_quat_wxyz(d.geom_xmat[g].reshape(3, 3))
            poses.append([round(float(v), 6) for v in
                          (*p, qw, qx, qy, qz)])
        ur = [{"name": n,
               "pose": [round(float(v), 6) for v in
                        (*d.xpos[b], *d.xquat[b])]}   # xquat is already wxyz
              for b, n in ur_bodies]
        out_frames.append({"t": float(fr["t"]), "q": fr["qpos"],
                           "poses": poses, "ur": ur})

    fam_counts: dict[str, int] = {}
    for g in geoms:
        fam_counts[g["family"]] = fam_counts.get(g["family"], 0) + 1
    print("family coverage:", fam_counts)

    if ur_bodies:
        print(f"UR5e links exported: {[n for _, n in ur_bodies]}")
    else:
        print("UR5e links: none in model yet (r12 pending) -- ur lists empty")

    payload = {
        "meta": {
            "n_frames": len(out_frames), "fps": data["replay"]["fps"],
            "scene": scene_kind,
            "n_geoms": len(geoms),
            "source": "scene_fidelity.build_fidelity_model() @ r8 replay",
            "axis_note": ("IDENTITY mapping: mujoco and blender are both "
                          "right-handed z-up; quats wxyz both sides; floor "
                          "plane + lights are built natively in blender"),
        },
        "geoms": geoms,
        "frames": out_frames,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, separators=(",", ":")))
    print(f"wrote {OUT.relative_to(REPO)} "
          f"({OUT.stat().st_size // 1024 // 1024} MB, "
          f"{len(out_frames)} frames x {len(geoms)} geoms)")


if __name__ == "__main__":
    main()
