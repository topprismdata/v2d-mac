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

import numpy as np
from PIL import Image, ImageDraw

REPO = pathlib.Path(__file__).resolve().parents[1]
TEX_DIR = REPO / "viz/fidelity_textures"

# original-clip palette (sampled from source frame 90)
VIOLET = (86, 48, 122)
FLOWERS = [(214, 84, 110), (233, 168, 76), (88, 140, 78), (222, 118, 140)]
OAK_LIGHT = (200, 155, 106)
OAK_DARK = (172, 126, 82)
PEG_BASE = (232, 228, 214)
PEG_HOLE = (168, 162, 148)


def _png(name: str) -> pathlib.Path:
    TEX_DIR.mkdir(parents=True, exist_ok=True)
    return TEX_DIR / f"{name}.png"


def _floral_texture(path: pathlib.Path, size: int = 512) -> None:
    """Purple base + scattered flowers/leaves, the tissue-box print."""
    rng = np.random.default_rng(11)
    img = Image.new("RGB", (size, size), VIOLET)
    d = ImageDraw.Draw(img)
    for _ in range(70):  # leaves
        x, y = rng.integers(0, size, 2)
        r = rng.integers(8, 22)
        a = rng.uniform(0, np.pi)
        dx, dy = r * np.cos(a), r * np.sin(a) * 0.45
        green = (70 + int(rng.integers(0, 50)), 110 + int(rng.integers(0, 60)), 60)
        d.ellipse([x - r, y - dy, x + r, y + dy], fill=green)
    for _ in range(46):  # flowers: 5 petals around a centre
        x, y = rng.integers(0, size, 2)
        pr = rng.integers(7, 16)
        col = FLOWERS[rng.integers(0, len(FLOWERS))]
        for k in range(5):
            a = 2 * np.pi * k / 5 + rng.uniform(0, 1)
            px, py = x + 0.7 * pr * np.cos(a), y + 0.7 * pr * np.sin(a)
            d.ellipse([px - pr * 0.55, py - pr * 0.55,
                       px + pr * 0.55, py + pr * 0.55], fill=col)
        d.ellipse([x - pr * 0.4, y - pr * 0.4, x + pr * 0.4, y + pr * 0.4],
                  fill=(245, 225, 150))
    img.save(path)


def _wood_texture(path: pathlib.Path, size: int = 512) -> None:
    """Light-oak grain: horizontal sinusoid streaks + fine noise."""
    rng = np.random.default_rng(5)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    grain = (np.sin(yy * 0.35 + np.sin(xx * 0.012) * 6) * 0.5 + 0.5)
    grain = grain * 0.7 + np.sin(yy * 0.05 + 1.3) * 0.3
    speck = rng.normal(0, 0.05, (size, size))
    g = np.clip(grain * 0.9 + speck, 0, 1)
    lo, hi = np.array(OAK_DARK, float), np.array(OAK_LIGHT, float)
    img = (lo[None, None, :] + g[:, :, None] * (hi - lo)[None, None, :])
    Image.fromarray(img.astype(np.uint8)).save(path)


def _pegboard_texture(path: pathlib.Path, size: int = 512) -> None:
    """Cream board with a regular round-hole grid."""
    img = Image.new("RGB", (size, size), PEG_BASE)
    d = ImageDraw.Draw(img)
    step, r = size // 8, size // 64
    for cy in range(step // 2, size, step):
        for cx in range(step // 2, size, step):
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=PEG_HOLE)
            d.arc([cx - r - 1, cy - r - 1, cx + r + 1, cy + r + 1],
                  180, 360, fill=(245, 242, 230))
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

    # v17 shoulder domes: the shipped solid dark spheres read as "tube stuck
    # in a ball"; restyle (in memory) as a machined two-piece housing --
    # metallic paint + specular, an equator seam ring and a pole cap, so the
    # arm segment plunging into the sphere reads as a shoulder joint rotating
    # inside its housing
    metal = spec.add_material(name="mat_metal")
    metal.rgba = [0.60, 0.62, 0.67, 1.0]
    metal.specular = 0.55
    metal.shininess = 0.6
    metal.reflectance = 0.04
    # dome-specific: translucent polycarbonate -- the arm stays visible
    # inside the sphere, which is what makes it read as a shoulder joint
    # rotating in its housing instead of a solid ball occluding the scene.
    # NB: material alpha is ignored by the renderer, so transparency goes
    # through geom rgba directly (same path as the box slot film); no
    # material on the domes, specular stays at the model default
    for jn, R in (("gantry_junction_r", 0.220), ("gantry_junction_l", 0.190)):
        jg = spec.geom(jn)
        jg.rgba = [0.58, 0.63, 0.70, 0.35]
        pos = jg.pos
        seam = spec.worldbody.add_geom(
            name=jn + "_seam", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[R + 0.002, 0.0035, 0.0], pos=[pos[0], pos[1], pos[2]],
            contype=0, conaffinity=0, mass=0)
        seam.rgba = [0.25, 0.27, 0.31, 0.75]
        cap = spec.worldbody.add_geom(
            name=jn + "_cap", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[R * 0.45, 0.006, 0.0],
            pos=[pos[0], pos[1], pos[2] + R + 0.001],
            contype=0, conaffinity=0, mass=0)
        cap.rgba = [0.42, 0.44, 0.49, 1.0]
    # the six arm segments share the machined-metal look so the chain reads
    # as one mechanism entering the housing
    for an in ("arm_r_forearm", "arm_r_elbow", "arm_r_upper",
               "arm_l_forearm", "arm_l_elbow", "arm_l_upper"):
        ag = spec.geom(an)
        ag.material = "mat_metal"
        ag.rgba = [1.0, 1.0, 1.0, 1.0]

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
        rgba=[0.92, 0.95, 0.97, 0.30])

    # --- workbench backdrop: pegboard wall, shelf, monitor ---------------
    wb = spec.worldbody

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
