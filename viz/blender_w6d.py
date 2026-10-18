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

PRESETS = {  # matched to the MuJoCo panel compositions (visual iteration)
    # NB: the rail carries two large end-stop blocks (gantry_limit_*, y=±0.52,
    # z up to 1.17, x ±0.55) -- cameras must sit INSIDE the rail span
    # (|y| < 0.44) or off to the x sides, or the block fills the frame black
    "front": {"pos": (-0.30, -0.40, 1.06), "look": (0.0, 0.0, 0.70)},
    "side": {"pos": (0.72, -0.42, 0.88), "look": (0.02, 0.0, 0.74)},
    "close": {"pos": (-0.28, -0.34, 0.94), "look": (0.0, 0.0, 0.76)},
}


def parse_args() -> dict:
    argv = sys.argv[sys.argv.index("--") + 1:]
    opts = {"frames": "60,190,300", "presets": "front,side,close",
            "outdir": str(HERE / "pilot")}
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


def build_materials() -> dict:
    T = HERE / "blender" / "textures_2k"
    return {
        "anodalu": mat_principled("anodalu", (0.36, 0.38, 0.43), 0.33, 1.0),
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


def make_camera(name, pos, look):
    cam_data = bpy.data.cameras.new(name)
    cam_data.lens = 30
    cam = bpy.data.objects.new(name, cam_data)
    cam.location = pos
    bpy.context.scene.collection.objects.link(cam)
    target = bpy.data.objects.new(name + "_tgt", None)
    target.location = look
    bpy.context.scene.collection.objects.link(target)
    con = cam.constraints.new(type="TRACK_TO")
    con.target = target
    con.track_axis = "TRACK_NEGATIVE_Z"
    con.up_axis = "UP_Y"
    return cam


def main() -> None:
    opts = parse_args()
    poses_path = pathlib.Path(opts.get("data", str(HERE / "blender" / "poses.json")))
    payload = json.loads(poses_path.read_text())
    frames = payload["frames"]
    want = [int(v) for v in opts["frames"].split(",")]
    presets = opts["presets"].split(",")

    # fresh scene
    bpy.ops.wm.read_factory_settings(use_empty=True)
    setup_cycles()

    mats = build_materials()
    objs: dict[str, bpy.types.Object] = {}
    for spec in payload["geoms"]:
        add_geom(objs, spec, mats)
    add_floor(mats)
    add_lights_and_world()

    cams = {k: make_camera(k, v["pos"], v["look"]) for k, v in PRESETS.items()
            if k in presets}

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
        for k in presets:
            sc.camera = cams[k]
            sc.render.filepath = str(outdir / f"{k}_f{f:04d}.png")
            bpy.ops.render.render(write_still=True)
            print(f"rendered {k} f{f:04d}")


if __name__ == "__main__":
    main()
