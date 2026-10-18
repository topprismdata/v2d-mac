"""Scene-fidelity visual layer for the W6d panel renderer (pure visuals).

Main's scene-fidelity batch: make the sim frame read like the original clip
(purple floral tissue box with white tissue in a top window, oak table,
pegboard wall + monitor in the background, warm indoor light) WITHOUT
touching physics.

Hard red line: grounding/assets/allegro_floating.xml is frozen (W6a owns it
in v13).  Everything here is applied through MuJoCo's MjSpec editing API at
MODEL-BUILD TIME inside the renderer only:

  * procedural textures (PIL, fixed seeds -> deterministic) + materials
  * re-skin of existing geoms (table slab -> wood)
  * NEW visual-only geoms: floral skin + tissue + window film as children of
    the moving ``object`` body, pegboard wall / monitor / shelf props as
    static worldbody dressing

Every added geom is contype=0 / conaffinity=0 / mass=0: collision pairs,
masses, inertias and all qpos addresses are unchanged, so the physics gates
(tests/test_scene_w6.py 13/13) stay numerically identical by construction.
"""
from __future__ import annotations

import pathlib

import mujoco
import numpy as np
from PIL import Image, ImageDraw

REPO = pathlib.Path(__file__).resolve().parents[1]
TEX_DIR = REPO / "viz/fidelity_textures"

# original-clip palette (sampled from source frame 90)
VIOLET = (98, 44, 142)
FLOWERS = [(214, 84, 110), (233, 168, 76), (88, 140, 78), (222, 118, 140)]
OAK_LIGHT = (170, 141, 106)
OAK_DARK = (142, 116, 88)
PEG_BASE = (232, 228, 214)
PEG_HOLE = (168, 162, 148)


def _png(name: str) -> pathlib.Path:
    TEX_DIR.mkdir(parents=True, exist_ok=True)
    return TEX_DIR / f"{name}.png"


def _floral_texture(path: pathlib.Path, size: int = 1024) -> None:
    """Purple base + dense scattered flowers/leaves, the tissue-box print.
    1024 px + ~3x element density vs v20 (GPT 满分批: density/saturation
    aligned toward the original print)."""
    rng = np.random.default_rng(11)
    img = Image.new("RGB", (size, size), VIOLET)
    d = ImageDraw.Draw(img)
    for _ in range(700):  # leaves
        x, y = rng.integers(0, size, 2)
        r = rng.integers(6, 20)
        a = rng.uniform(0, np.pi)
        dx, dy = r * np.cos(a), r * np.sin(a) * 0.45
        green = (66 + int(rng.integers(0, 55)), 105 + int(rng.integers(0, 70)), 58)
        d.ellipse([x - r, y - dy, x + r, y + dy], fill=green)
    for _ in range(720):  # flowers: 5 petals around a centre (Main: 700+)
        x, y = rng.integers(0, size, 2)
        pr = rng.integers(9, 20)
        col = FLOWERS[rng.integers(0, len(FLOWERS))]
        for k in range(5):
            a = 2 * np.pi * k / 5 + rng.uniform(0, 1)
            px, py = x + 0.7 * pr * np.cos(a), y + 0.7 * pr * np.sin(a)
            d.ellipse([px - pr * 0.55, py - pr * 0.55,
                       px + pr * 0.55, py + pr * 0.55], fill=col)
        d.ellipse([x - pr * 0.38, y - pr * 0.38, x + pr * 0.38, y + pr * 0.38],
                  fill=(246, 224, 148))
    img.save(path)


def _wood_texture(path: pathlib.Path, size: int = 512) -> None:
    """Light-oak grain: horizontal sinusoid streaks + fine noise."""
    rng = np.random.default_rng(5)
    # streaks vary along x so the slab-top UV maps them across the view
    xx, yy = np.mgrid[0:size, 0:size].astype(np.float64)
    grain = (np.sin(xx * 1.6 + np.sin(yy * 0.035) * 9) * 0.5 + 0.5)
    grain = grain * 0.55 + np.sin(xx * 0.25 + 1.1) * 0.2
    grain = grain * 0.8 + (np.sin(yy * 0.9 + xx * 0.05) * 0.5 + 0.5) * 0.2
    speck = rng.normal(0, 0.09, (size, size))
    g = np.clip(grain * 0.85 + speck, 0, 1)
    lo, hi = np.array(OAK_DARK, float), np.array(OAK_LIGHT, float)
    img = (lo[None, None, :] + g[:, :, None] * (hi - lo)[None, None, :])
    Image.fromarray(img.astype(np.uint8)).save(path)


def _pegboard_texture(path: pathlib.Path, size: int = 512) -> None:
    """Cream board with a regular round-hole grid."""
    img = Image.new("RGB", (size, size), PEG_BASE)
    d = ImageDraw.Draw(img)
    step, r = size // 16, size // 110  # v25: dense holes, strong contrast
    for cy in range(step // 2, size, step):
        for cx in range(step // 2, size, step):
            d.ellipse([cx - r, cy - r - 2, cx + r, cy + r - 2],
                      fill=(78, 72, 62))     # cast shadow above
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(34, 32, 29))
            d.arc([cx - r - 1, cy - r - 1, cx + r + 1, cy + r + 1],
                  180, 360, fill=(252, 250, 244))
    img.save(path)


def build_fidelity_spec():
    """allegro_floating.xml + the visual fidelity layer, via MjSpec."""
    import mujoco

    _floral_texture(_png("floral"))
    _wood_texture(_png("wood"))
    _pegboard_texture(_png("pegboard"))

    spec = mujoco.MjSpec.from_file(str(REPO / "grounding/assets/allegro_floating.xml"))
    tex2d = lambda name, path: spec.add_texture(
        name=name, type=mujoco.mjtTexture.mjTEXTURE_2D,
        height=512, width=512, file=str(path))
    tex_wood = tex2d("tex_wood", _png("wood"))
    tex_floral = tex2d("tex_floral", _png("floral"))
    tex_peg = tex2d("tex_peg", _png("pegboard"))

    def material(name, tex, repeat=(1, 1)):
        # add_material's texuniform kwarg is silently dropped in 3.16 -- set
        # it on the returned MjsMaterial or box UVs map only a fraction of
        # the texture and prints come out hugely magnified
        m_ = spec.add_material(name=name, textures=[tex.name])
        m_.texuniform = 1
        m_.texrepeat = list(repeat)
        return m_

    material("mat_wood", tex_wood)
    material("mat_floral", tex_floral, repeat=(2, 2))  # denser print
    material("mat_peg", tex_peg, repeat=(2, 2))

    # v19: the v17 shoulder domes are gone (no junction spheres in this XML);
    # the articulated arm segments get the machined-metal look so the chain
    # reads as one mechanism (skip silently if W6a renames again)
    def mat_machined(name, rgba, spec_=0.5, shin=0.6, metal_=0.0, rough=0.5):
        # metallic/roughness ride along for PBR-capable consumers; the
        # classic renderer honours rgba+specular+shininess
        m_ = spec.add_material(name=name)
        m_.rgba = list(rgba)
        m_.specular = spec_
        m_.shininess = shin
        m_.metallic = metal_
        m_.roughness = rough
        return m_

    mat_anodalu = mat_machined("mat_anodalu", (0.45, 0.47, 0.53, 1.0),
                               0.6, 0.65, metal_=0.85, rough=0.35)  # 阳极铝
    mat_poly = mat_machined("mat_poly", (0.13, 0.13, 0.15, 1.0),
                            0.25, 0.3, rough=0.85)               # 聚氨酯垫
    mat_powder = mat_machined("mat_powder", (0.47, 0.49, 0.52, 1.0),
                              0.3, 0.35, metal_=0.3, rough=0.6)  # 喷塑钢
    mat_frosted = mat_machined("mat_frosted", (0.86, 0.87, 0.89, 1.0),
                               0.2, 0.25, rough=0.75)           # 磨砂塑料
    for an in ("arm_r_forearm", "arm_r_elbow", "arm_r_upper",
               "arm_l_forearm", "arm_l_elbow", "arm_l_upper",
               "arm_yaw_disc_r", "arm_elbow_disc_r", "arm_wrist_r",
               "arm_yaw_disc_l", "arm_elbow_disc_l", "arm_wrist_l"):
        ag = spec.geom(an) if an in [x.name for x in spec.geoms] else None
        if ag is not None:
            ag.material = "mat_anodalu"
            ag.rgba = [1.0, 1.0, 1.0, 1.0]

    # fingertip pads -> dark polyurethane
    for g in spec.geoms:
        if g.name and "_tip_collision" in g.name:
            g.material = "mat_poly"
            g.rgba = [1.0, 1.0, 1.0, 1.0]

    # gantry structure -> powder-coated steel; monitor -> frosted plastic
    for g in spec.geoms:
        if g.name and g.name.startswith("gantry_"):
            g.material = "mat_powder"
            g.rgba = [1.0, 1.0, 1.0, 1.0]
        if g.name and g.name.startswith("mon_") and g.name != "mon_screen":
            # the screen itself keeps its dark rgba (an off display)
            g.material = "mat_frosted"
            g.rgba = [1.0, 1.0, 1.0, 1.0]

    # --- re-skin existing geoms (visual only, physics identical) ----------
    spec.geom("table_slab").material = "mat_wood"

    # --- tissue box dressing: children of the MOVING object body ----------
    ob = spec.body("object")

    def vis(parent, name, gtype, size, pos, material=None, rgba=None,
            transparency=0.0):
        g = parent.add_geom(name=name, type=gtype, size=size, pos=pos,
                            contype=0, conaffinity=0, mass=0)
        if material:
            g.material = material
        if rgba is not None:
            g.rgba = rgba
        if transparency:
            g.rgba[3] = transparency
        return g

    # floral skin covering box + chamfer blades (slight oversize, 2 mm class)
    vis(ob, "skin_floral", mujoco.mjtGeom.mjGEOM_BOX,
        [0.063, 0.1215, 0.0425], [0, 0, 0], material="mat_floral")
    # white tissue puffing through the top slot
    vis(ob, "tissue", mujoco.mjtGeom.mjGEOM_BOX,
        [0.030, 0.048, 0.010], [0, 0, 0.044],
        rgba=[0.98, 0.98, 0.96, 1.0])
    # translucent window film over the slot
    vis(ob, "slot_film", mujoco.mjtGeom.mjGEOM_BOX,
        [0.046, 0.078, 0.0012], [0, 0, 0.0405],
        rgba=[0.95, 0.97, 0.99, 0.20])

    # --- workbench backdrop: pegboard wall, shelf, monitor ---------------
    wb = spec.worldbody

    # --- engineering details (visual-only, mass 0) ------------------------
    # cable drag chains: small dark capsules riding the outside of each upper
    # arm and forearm segment
    for seg_body, seg_len in (("arm_upper_r", 0.08), ("arm_upper_l", 0.08),
                              ("arm_elbow_r", 0.08), ("arm_elbow_l", 0.08)):
        ab = spec.body(seg_body)
        n_seg = 4
        for k in range(n_seg):
            z = -seg_len + (2 * seg_len) * (k + 0.5) / n_seg
            cg = ab.add_geom(name=f"{seg_body}_cable{k}",
                             type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                             size=[0.0028, 0.011, 0.0],
                             pos=[0.016, 0.0, z],
                             contype=0, conaffinity=0, mass=0)
            cg.rgba = [0.16, 0.16, 0.18, 1.0]
    # bolt rings on the yaw/elbow discs (6 heads each, top face)
    for disc_body, disc_r, disc_h in (("arm_yaw_r", 0.024, 0.012),
                                      ("arm_yaw_l", 0.024, 0.012),
                                      ("arm_elbow_r", 0.020, 0.014),
                                      ("arm_elbow_l", 0.020, 0.014)):
        db = spec.body(disc_body)
        for k in range(6):
            a = 2 * np.pi * k / 6
            bg = db.add_geom(name=f"{disc_body}_bolt{k}",
                             type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                             size=[0.0028, 0.0015, 0.0],
                             pos=[disc_r * 0.72 * np.cos(a),
                                  disc_r * 0.72 * np.sin(a),
                                  disc_h + 0.0012],
                             contype=0, conaffinity=0, mass=0)
            bg.rgba = [0.30, 0.31, 0.34, 1.0]
    # table support frame: apron box ring + four legs under the 30 mm slab
    for name_, size_, pos_ in (
            ("table_apron_xp", [0.45, 0.018, 0.028], [0.0, 0.40, 0.630]),
            ("table_apron_xn", [0.45, 0.018, 0.028], [0.0, -0.40, 0.630]),
            ("table_apron_yp", [0.018, 0.40, 0.028], [0.40, 0.0, 0.630]),
            ("table_apron_yn", [0.018, 0.40, 0.028], [-0.40, 0.0, 0.630]),
            ("table_leg_pp", [0.028, 0.028, 0.330], [0.36, 0.36, 0.275]),
            ("table_leg_pn", [0.028, 0.028, 0.330], [0.36, -0.36, 0.275]),
            ("table_leg_np", [0.028, 0.028, 0.330], [-0.36, 0.36, 0.275]),
            ("table_leg_nn", [0.028, 0.028, 0.330], [-0.36, -0.36, 0.275])):
        lg = wb.add_geom(name=name_, type=mujoco.mjtGeom.mjGEOM_BOX,
                         size=size_, pos=pos_, contype=0, conaffinity=0, mass=0)
        lg.material = "mat_wood"

    def deco(name, gtype, size, pos, material=None, rgba=None, euler=None):
        g = wb.add_geom(name=name, type=gtype, size=size, pos=pos,
                        contype=0, conaffinity=0, mass=0)
        if material:
            g.material = material
        if rgba is not None:
            g.rgba = rgba
        return g

    deco("pegboard_wall", mujoco.mjtGeom.mjGEOM_BOX,
         [1.35, 0.012, 0.75], [0, 1.06, 1.18], material="mat_peg")
    # shelf strip + a few props on the wall
    deco("peg_shelf", mujoco.mjtGeom.mjGEOM_BOX,
         [0.5, 0.055, 0.008], [0.28, 1.0, 1.46], material="mat_wood")
    deco("prop_level", mujoco.mjtGeom.mjGEOM_BOX,
         [0.16, 0.014, 0.016], [0.34, 0.975, 1.52], rgba=[0.95, 0.78, 0.1, 1])
    deco("prop_box_red", mujoco.mjtGeom.mjGEOM_BOX,
         [0.045, 0.03, 0.03], [0.1, 0.985, 1.5], rgba=[0.72, 0.15, 0.12, 1])
    # monitor on the left: dark screen + white stand/arm + base
    deco("mon_screen", mujoco.mjtGeom.mjGEOM_BOX,
         [0.21, 0.012, 0.13], [-0.62, 0.42, 1.08], rgba=[0.06, 0.06, 0.07, 1])
    deco("mon_frame", mujoco.mjtGeom.mjGEOM_BOX,
         [0.225, 0.008, 0.145], [-0.62, 0.408, 1.08], rgba=[0.88, 0.88, 0.9, 1])
    deco("mon_arm", mujoco.mjtGeom.mjGEOM_BOX,
         [0.016, 0.016, 0.22], [-0.62, 0.42, 0.86], rgba=[0.82, 0.82, 0.85, 1])
    deco("mon_base", mujoco.mjtGeom.mjGEOM_BOX,
         [0.12, 0.09, 0.008], [-0.62, 0.42, 0.742], rgba=[0.85, 0.85, 0.88, 1])

    return spec


def build_fidelity_model():
    """Compile the fidelity spec; returns the MjModel for RENDERING only."""
    m = build_fidelity_spec().compile()
    # MjSpec 3.16 bug: add_material(textures=[...]) writes the 2d texture id
    # into mat_texid slot 0, which the renderer never samples (the XML asset
    # path puts 2d textures in slot 1) -- materials then render as flat rgba.
    # Re-slot every bound 2d texture into slot 1 (verified against the XML
    # path's field dump).
    for mid_ in range(m.nmat):
        if m.mat_texid[mid_][0] >= 0:
            m.mat_texid[mid_][1] = m.mat_texid[mid_][0]
            m.mat_texid[mid_][0] = -1
    return m


if __name__ == "__main__":
    import mujoco
    m0 = mujoco.MjModel.from_xml_path(str(REPO / "grounding/assets/allegro_floating.xml"))
    m = build_fidelity_model()
    same = (m.nq == m0.nq and m.nv == m0.nv and m.nbody == m0.nbody
            and m.njnt == m0.njnt)
    adr0 = m0.jnt_qposadr[mujoco.mj_name2id(m0, mujoco.mjtObj.mjOBJ_JOINT, "object_free")]
    adr = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "object_free")]
    print(f"nq {m.nq}=={m0.nq} nv {m.nv}=={m0.nv} nbody {m.nbody}=={m0.nbody} "
          f"njnt {m.njnt}=={m0.njnt} obj_adr {adr}=={adr0} -> physics-safe: {same and adr == adr0}")
    print(f"geoms {m0.ngeom} -> {m.ngeom} (+{m.ngeom - m0.ngeom} visual-only)")
