"""Blender Cycles photoreal renderer for the W6d replay (pilot + terminal).

Run inside Blender (NOT the mujoco venv):

    blender -b -P viz/blender_w6d.py -- \
        --frames 60,190,300 --presets front,side,close \
        --outdir viz/blender/pilot

Consumes viz/blender/poses.json (produced by viz/export_blender_poses.py —
physics truth stays in MuJoCo; this script only skins it).  Axis mapping is
identity (both right-handed z-up, quats wxyz).

Cycles + METAL GPU, area key (warm) + area fill (cool) + world; area lights
give the natural gantry penumbra MuJoCo's rasterizer could not.
"""
from __future__ import annotations

import json
import math
import pathlib
import sys

import bpy
from mathutils import Vector

HERE = pathlib.Path(__file__).resolve().parent
TEX2K = HERE / "textures_2k"

PRESETS = {  # world positions resolved from the MuJoCo free-camera presets
    # (mjv GL camera pos at f190; the lookat tracks the live box position)
    "front": {"pos": [0.156, -1.430, 1.559]},
    "side": {"pos": [-0.171, -1.304, 1.414]},
    "close": {"pos": [0.111, -0.916, 1.364]},
}

def parse_args() -> dict:
    argv = sys.argv[sys.argv.index("--") + 1:]
    opts = {"frames": "all", "presets": "front,side,close",
            "outdir": str(HERE / "pilot"), "skip_existing": "1"}
    i = 0
    while i < len(argv):
        if argv[i].startswith("--"):
            opts[argv[i][2:]] = argv[i + 1]
            i += 2
        else:
            i += 1
    return opts


def setup_cycles() -> None:
    sc = bpy.context.scene
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "METAL"
    prefs.get_devices()
    for dev in prefs.devices:
        dev.use = True
    sc.render.engine = "CYCLES"
    sc.cycles.device = "GPU"
    sc.cycles.samples = 64
    sc.cycles.use_adaptive_sampling = True
    sc.cycles.adaptive_threshold = 0.03
    sc.cycles.use_denoising = True
    sc.render.use_persistent_data = True
    sc.render.resolution_x = 1280
    sc.render.resolution_y = 720
    sc.view_settings.view_transform = "AgX"
    try:
        sc.view_settings.look = "AgX - Medium High Contrast"
    except TypeError:
        pass


def mat_principled(name, color, rough, metal=0.0, alpha=1.0,
                   coat=0.0, tex_path=None):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    bsdf = m.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Roughness"].default_value = rough
    bsdf.inputs["Metallic"].default_value = metal
    if alpha < 1.0:
        bsdf.inputs["Alpha"].default_value = alpha
    if coat:
        for cand in ("Coat Weight", "Clearcoat"):
            if cand in bsdf.inputs:
                bsdf.inputs[cand].default_value = coat
                break
    if tex_path:
        tex = m.node_tree.nodes.new("ShaderNodeTexImage")
        tex.image = bpy.data.images.load(str(tex_path))
        m.node_tree.links.new(tex.outputs["Color"],
                              bsdf.inputs["Base Color"])
        if name == "mat_wood":
            bsdf.inputs["Roughness"].default_value = 0.3
            for cand in ("Coat Weight", "Clearcoat"):
                if cand in bsdf.inputs:
                    bsdf.inputs[cand].default_value = 0.35
                    break
    return m


# v25/r12 prep: UR5e real-CAD arm (menagerie universal_robots_ur5e).
# When W6b's r12 scene welds the UR5e in, the exporter emits per-link poses
# and this loader swaps the primitive arm for the real CAD meshes.
UR5E_MESH_DIR = HERE.parent / "runs/upstream/menagerie/universal_robots_ur5e/assets"
UR5E_LINKS = {
    # both arms share the menagerie CAD (one arm's OBJs, two instances)
    "ur5e_r_shoulder_link": ["shoulder_0", "shoulder_1", "shoulder_2"],
    "ur5e_l_shoulder_link": ["shoulder_0", "shoulder_1", "shoulder_2"],
    "ur5e_r_upper_arm_link": ["upperarm_0", "upperarm_1", "upperarm_2",
                              "upperarm_3"],
    "ur5e_l_upper_arm_link": ["upperarm_0", "upperarm_1", "upperarm_2",
                              "upperarm_3"],
    "ur5e_r_forearm_link": ["forearm_0", "forearm_1", "forearm_2", "forearm_3"],
    "ur5e_l_forearm_link": ["forearm_0", "forearm_1", "forearm_2", "forearm_3"],
    "ur5e_r_wrist_1_link": ["wrist1_0", "wrist1_1", "wrist1_2"],
    "ur5e_l_wrist_1_link": ["wrist1_0", "wrist1_1", "wrist1_2"],
    "ur5e_r_wrist_2_link": ["wrist2_0", "wrist2_1", "wrist2_2"],
    "ur5e_l_wrist_2_link": ["wrist2_0", "wrist2_1", "wrist2_2"],
    "ur5e_r_wrist_3_link": ["wrist3"],
    "ur5e_l_wrist_3_link": ["wrist3"],
}


def build_materials() -> dict:
    T = HERE / "blender" / "textures_2k"
    return {
        "anodalu": mat_principled("anodalu", (0.36, 0.38, 0.43), 0.33, 1.0),
        # v25 工业臂风格令（实拍参考 docs/images/arm_reference_real.png）
        "struct_blue": mat_principled("struct_blue", (0.18, 0.435, 0.69),
                                      0.32, 0.85),   # 结构蓝 #2E6FB0
        "black_shell": mat_principled("black_shell", (0.045, 0.045, 0.052),
                                      0.42, 0.25),   # 黑关节壳/鼓
        "brass": mat_principled("brass", (0.72, 0.53, 0.18), 0.28, 1.0),
        "poly": mat_principled("poly", (0.045, 0.045, 0.055), 0.72),
        "powder": mat_principled("powder", (0.33, 0.35, 0.38), 0.48, 0.15),
        "frosted": mat_principled("frosted", (0.66, 0.67, 0.70), 0.55),
        "screen": mat_principled("screen", (0.015, 0.015, 0.02), 0.18),
        "wood": mat_principled("wood", (1, 1, 1), 0.3,
                               coat=0.35, tex_path=T / "wood.png"),
        "floral": mat_principled("floral", (1, 1, 1), 0.36,
                                 tex_path=T / "floral.png"),
        "peg": mat_principled("peg", (1, 1, 1), 0.55,
                              tex_path=T / "peg.png"),
        "tissue": mat_principled("tissue", (0.95, 0.95, 0.93), 0.85),
        "film": mat_principled("film", (0.93, 0.96, 0.99), 0.15, alpha=0.22),
        "cable": mat_principled("cable", (0.14, 0.14, 0.16), 0.5),
        "prop_level": mat_principled("prop_level", (0.95, 0.78, 0.1), 0.4),
        "prop_box_red": mat_principled("prop_box_red", (0.72, 0.15, 0.12), 0.45),
        "floor": mat_principled("floor", (0.30, 0.26, 0.22), 0.7),
        # UR5e 实机配色: 灰蓝本体 + 黑关节壳
        "ur_body": mat_principled("ur_body", (0.52, 0.56, 0.60), 0.38, 0.9),
        "ur_joint": mat_principled("ur_joint", (0.07, 0.07, 0.08), 0.45, 0.3),
    }


def add_capsule(radius: float, length: float):
    """Blender 5.2 has no primitive_capsule_add — cylinder + 2 hemispheres."""
    bpy.ops.mesh.primitive_cylinder_add(radius=radius, depth=length,
                                        vertices=32)
    ob = bpy.context.active_object
    for z in (length / 2, -length / 2):
        bpy.ops.mesh.primitive_uv_sphere_add(radius=radius, segments=24,
                                             ring_count=12,
                                             location=(0, 0, z))
        cap = bpy.context.active_object
        ob.select_set(True)
        cap.select_set(True)
        bpy.context.view_layer.objects.active = ob
        bpy.ops.object.join()
    return ob


def add_geom(objs, spec, mats):
    name, gtype, size, fam = spec["name"], spec["type"], spec["size"], spec["family"]
    if gtype == 0:          # infinite plane: skip, blender builds its own floor
        return
    if gtype == 2:          # sphere
        bpy.ops.mesh.primitive_uv_sphere_add(radius=size[0], segments=48,
                                             ring_count=24)
    elif gtype == 3:        # capsule (z axis, like mujoco)
        ob = add_capsule(size[0], 2 * size[1])
        ob.name = name
        ob.data.materials.append(mats.get(fam, mats["powder"]))
        objs[name] = ob
        return
    elif gtype == 5:        # cylinder
        bpy.ops.mesh.primitive_cylinder_add(radius=size[0],
                                            depth=2 * size[1], vertices=48)
    elif gtype == 6:        # box
        bpy.ops.mesh.primitive_cube_add(size=1.0)
    else:                   # anything exotic: skip
        return
    ob = bpy.context.active_object
    ob.name = name
    ob.scale = (size[0] * 2, size[1] * 2, size[2] * 2) if gtype == 6 \
        else (1.0, 1.0, 1.0)
    ob.data.materials.append(mats.get(fam, mats["powder"]))
    objs[name] = ob


def add_floor(mats):
    bpy.ops.mesh.primitive_plane_add(size=6.0, location=(0, 0, -0.051))
    fl = bpy.context.active_object
    fl.name = "blender_floor"
    fl.data.materials.append(mats["floor"])


def add_lights_and_world():
    w = bpy.data.worlds.new("w6d")
    bpy.context.scene.world = w
    w.use_nodes = True
    bg = w.node_tree.nodes["Background"]
    bg.inputs[0].default_value = (0.55, 0.62, 0.75, 1.0)
    bg.inputs[1].default_value = 0.22

    def area(name, pos, rot, energy, color, size):
        data = bpy.data.lights.new(name, "AREA")
        data.energy = energy
        data.color = color
        data.shape = "SQUARE"
        data.size = size
        ob = bpy.data.objects.new(name, data)
        ob.location = pos
        ob.rotation_euler = rot
        ob.visible_camera = False   # lights must never photobomb the frame
        bpy.context.scene.collection.objects.link(ob)
        return ob

    # directional key so contact shadows ground the objects; cool fill lifts
    # the shadow side instead of a flat world dome
    area("key", (-0.85, -0.85, 1.40),
         (math.radians(48), 0, math.radians(-43)), 42, (1.0, 0.90, 0.78), 0.9)
    area("fill", (1.15, -0.55, 1.00),
         (math.radians(55), 0, math.radians(115)), 10, (0.78, 0.87, 1.0), 1.4)
    area("back", (0.25, 1.25, 1.55),
         (math.radians(-32), 0, math.radians(180)), 14, (0.95, 0.96, 1.0), 1.8)
    # wall wash: the pegboard wall faces the camera and sits outside the key
    # light's cone -- without this it renders near-black
    wash = bpy.data.lights.new("wallwash", "AREA")
    wash.energy = 60
    wash.color = (0.92, 0.93, 0.96)
    wash.shape = "RECTANGLE"
    wash.size = 2.4
    wash.size_y = 1.0
    wob = bpy.data.objects.new("wallwash", wash)
    wob.location = (0.0, -0.4, 1.5)
    wob.rotation_euler = (math.radians(58), 0, 0)
    wob.visible_camera = False
    bpy.context.scene.collection.objects.link(wob)


def make_camera(name, pos):
    cam_data = bpy.data.cameras.new(name)
    cam_data.lens = 30
    cam = bpy.data.objects.new(name, cam_data)
    cam.location = pos
    bpy.context.scene.collection.objects.link(cam)
    target = bpy.data.objects.new(name + "_tgt", None)
    bpy.context.scene.collection.objects.link(target)
    con = cam.constraints.new(type="TRACK_TO")
    con.target = target
    con.track_axis = "TRACK_NEGATIVE_Z"
    con.up_axis = "UP_Y"
    return cam, target


def main() -> None:
    # pre-launch guard: a second concurrent instance on the same outdir
    # halves the CPU and the two writers race frames (v25 batch incident).
    # ps shows every live "Blender -b -P viz/blender_w6d.py"; anything that
    # is not THIS process is a racer.
    import os as _os
    import subprocess as _sp
    my_pid = str(_os.getpid())
    others = [l.strip() for l in _sp.run(
        ["ps", "-axo", "pid,command"], capture_output=True, text=True)
        .stdout.splitlines()
        if "Blender -b -P viz/blender_w6d.py" in l and "grep" not in l
        and not l.strip().startswith(my_pid)]
    if others:
        raise SystemExit("another blender_w6d.py render is already running "
                         "(concurrent instances race frames) -- aborting: "
                         + " | ".join(others[:2]))

    opts = parse_args()
    poses_path = pathlib.Path(opts.get("data", str(HERE / "blender" / "poses.json")))
    payload = json.loads(poses_path.read_text())
    frames = payload["frames"]
    want = (list(range(len(frames))) if opts["frames"] == "all"
            else [int(v) for v in opts["frames"].split(",")])
    presets = opts["presets"].split(",")
    skip_existing = opts.get("skip_existing", "0") == "1"

    # fresh scene
    bpy.ops.wm.read_factory_settings(use_empty=True)
    setup_cycles()

    mats = build_materials()
    objs: dict[str, bpy.types.Object] = {}
    for spec in payload["geoms"]:
        add_geom(objs, spec, mats)

    # UR5e real-CAD links: one joined mesh per link group, keyed
    # "ur:<link_name>" and driven by the exported link poses
    ur_link_names = [u.get("name") for u in
                     (payload["frames"][0].get("ur") or [])] \
        if payload["frames"] and "ur" in payload["frames"][0] else []
    for link, files in UR5E_LINKS.items():
        if ur_link_names and link not in ur_link_names:
            continue
        parts = []
        for f in files:
            path = UR5E_MESH_DIR / f"{f}.obj"
            if not path.exists():
                continue
            bpy.ops.wm.obj_import(filepath=str(path))
            parts.append(bpy.context.active_object)
        if not parts:
            continue
        for o in parts:
            o.select_set(True)
        bpy.context.view_layer.objects.active = parts[0]
        if len(parts) > 1:
            bpy.ops.object.join()
        mesh = bpy.context.active_object
        mesh.name = f"ur5e:{link}"
        mesh.data.materials.append(
            mats["ur_joint" if "wrist" in link else "ur_body"])
    add_floor(mats)
    add_lights_and_world()

    cams_raw = {k: make_camera(k, v["pos"]) for k, v in PRESETS.items()
                if k in presets}
    cams = {k: c for k, (c, t) in cams_raw.items()}
    cam_targets = {k: t for k, (c, t) in cams_raw.items()}

    outdir = pathlib.Path(opts["outdir"])
    outdir.mkdir(parents=True, exist_ok=True)

    sc = bpy.context.scene
    for f in want:
        fr = frames[f]
        for gi, spec in enumerate(payload["geoms"]):
            ob = objs.get(spec["name"])
            if ob is None:
                continue
            x, y, z, qw, qx, qy, qz = fr["poses"][gi]
            ob.location = (x, y, z)
            ob.rotation_mode = "QUATERNION"
            ob.rotation_quaternion = (qw, qx, qy, qz)
        for entry in fr.get("ur", []):
            ob = bpy.data.objects.get(f"ur5e:{entry['name']}")
            if ob is None:
                continue
            px, py, pz, qw, qx, qy, qz = entry["pose"]
            ob.location = (px, py, pz)
            ob.rotation_mode = "QUATERNION"
            ob.rotation_quaternion = (qw, qx, qy, qz)
        box_loc = objs["skin_floral"].location  # cameras track the box
        for k in presets:
            cam_targets[k].location = box_loc
            dst = outdir / f"{k}_f{f:04d}.png"
            # resume: skip frames already rendered by a previous pass
            if skip_existing and dst.exists() and dst.stat().st_size > 100_000:
                print(f"skipped {k} f{f:04d} (exists)", flush=True)
                continue
            sc.camera = cams[k]
            sc.render.filepath = str(dst)
            bpy.ops.render.render(write_still=True)
            print(f"rendered {k} f{f:04d}", flush=True)

    update_panel_meta(outdir, presets, want)


def update_panel_meta(outdir: pathlib.Path, presets: list, want: list) -> None:
    """Point the panel at the blender PNG sequence (meta.renders_blender);
    the MuJoCo renders stay available via ?renderer=mujoco."""
    panel_json = HERE / "panel_data_v2.json"
    if not panel_json.exists():
        print("panel_data_v2.json not found -- skip meta update")
        return
    payload = json.loads(panel_json.read_text())
    payload["meta"]["renders_blender"] = {
        "dir": str(outdir),
        "presets": [{"key": k, "label": {"front": "正面", "side": "侧面",
                                         "close": "特写"}.get(k, k)}
                    for k in presets],
        "frames": {k: [str(outdir / f"{k}_f{f:04d}.png") for f in want]
                   for k in presets},
    }
    panel_json.write_text(json.dumps(payload, indent=1, ensure_ascii=False))
    (HERE / "panel_data_v2.js").write_text(
        "window.PANEL_DATA_V2 = "
        + json.dumps(payload, ensure_ascii=False) + ";\n")
    print(f"panel meta updated: renders_blender "
          f"({len(presets)} presets x {len(want)} frames)", flush=True)


if __name__ == "__main__":
    main()
