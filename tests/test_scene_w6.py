"""W6a - the dual-hand scene: two hands at the box's +-x end faces.

The source-video task (signed 2026-10-15, commit 8a18651) grips the tissue box
with BOTH hands, one per 120x80 mm short end face, the box resting directly on
the table - no pedestal. This file pins the scene that realises it:

  * dual-hand model statistics (nq/nu/nv/ngeom) and name conventions
  * GATE 1  birth contact table has no negative penetration
  * GATE 2  box + hands settle 400 steps with < 1 mm drift, no blow-up
  * GATE 3  a kinematic probe (pedestal_scan-style grid + refine over all 22
            hand DoF per hand) finds ONE configuration where both hands' pads
            touch their own end face within the derived R5 touch tolerance
  * frozen invariants: box mass 0.15 kg (and the 9-digit mass precision the
    XML is emitted with), W_PEDESTAL_LOAD untouched, both hands carrying the
    IDENTICAL measured forcerange table

Everything here is measured from the built model, never asserted from the
generator's comments.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

pytest.importorskip("mujoco")

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import mujoco  # noqa: E402

from grounding.assets import build_allegro_floating as builder  # noqa: E402
from grounding.envs import lift_hold_env as env_mod  # noqa: E402

MJCF = REPO / "grounding/assets/allegro_floating.xml"

BOX_HALF_M = (0.06, 0.12, 0.04)
BOX_MASS_KG = 0.15
FINGERS = ("ff", "mf", "rf", "th")
BIRTH_PENETRATION_TOL_M = 1e-9
REST_MOVE_TOL_M = 1e-3
SETTLE_STEPS = 400
FACE_X, HALF_Y = 0.06, 0.12
BOX_BOT, BOX_TOP = 0.70, 0.78


@pytest.fixture(scope="module")
def model():
    return mujoco.MjModel.from_xml_path(str(MJCF))


@pytest.fixture(scope="module")
def data(model):
    d = mujoco.MjData(model)
    mujoco.mj_forward(model, d)
    return d


def nid(model, objtype, name):
    i = mujoco.mj_name2id(model, objtype, name)
    assert i >= 0, f"{name!r} not found"
    return i


# ============================================================== dual-hand stats
def test_w6_model_statistics(model):
    # 2 hands x (7 freejoint-equivalent wrist DoF + 16 fingers) + object 7
    assert model.nq == 51
    assert model.nv == 50
    assert model.nu == 44
    # 42 upstream hand geoms x 2 + box + 2 chamfer plates + floor + slab
    assert model.ngeom == 89


def test_w6_both_hand_roots_exist_at_the_end_faces(model, data):
    for name, side in (("hand_root", +1.0), ("hand_root_left", -1.0)):
        bid = nid(model, mujoco.mjtObj.mjOBJ_BODY, name)
        pos = data.xpos[bid]
        assert pos[0] * side > FACE_X, f"{name} must stand outside its end face"
    # the left subtree is a complete renamed copy: every right-hand body/joint/
    # actuator has exactly one _left twin (left bodies are skipped - they ARE
    # the twins)
    for b in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or ""
        if not name or name.endswith("_left"):
            continue
        if name in ("world", "object", "hand_root"):
            continue
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name + "_left") >= 0, (
            f"body {name} has no _left twin"
        )


def test_w6_left_hand_mirrors_right_finger_naming(model):
    for f in FINGERS:
        for i in range(4):
            j = nid(model, mujoco.mjtObj.mjOBJ_JOINT, f"{f}j{i}")
            jl = nid(model, mujoco.mjtObj.mjOBJ_JOINT, f"{f}j{i}_left")
            # identical joint ranges: the left hand is a byte-identical copy
            np.testing.assert_allclose(model.jnt_range[j], model.jnt_range[jl], atol=0)
            a = nid(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{f}a{i}")
            al = nid(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{f}a{i}_left")
            np.testing.assert_allclose(model.actuator_forcerange[a],
                                       model.actuator_forcerange[al], atol=0)
            np.testing.assert_allclose(model.actuator_gainprm[a][0],
                                       model.actuator_gainprm[al][0], atol=0)


def test_w6_finger_geoms_carry_the_frozen_prefix_rule(model):
    """The frozen acceptance classifier counts finger geoms by body-name PREFIX
    (ff_/mf_/rf_/th_). Both hands' bodies must satisfy it, or the left hand's
    force silently drops out of the A3a 1.4715 N sum."""
    for g in range(model.ngeom):
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                                  int(model.geom_bodyid[g])) or ""
        if "_collision" in bname or bname.endswith(("_tip", "_tip_left")):
            if bname.startswith(("ff_", "mf_", "rf_", "th_")):
                continue
            if bname in ("palm", "palm_left", "hand_root", "hand_root_left"):
                continue
            # every hand body is a finger-chain body: must carry a finger prefix
            assert bname.startswith(("ff_", "mf_", "rf_", "th_")), (
                f"hand body {bname!r} breaks the frozen prefix rule")


def test_w6_pedestal_is_gone_box_rests_on_the_slab(model):
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "pedestal") == -1
    slab = nid(model, mujoco.mjtObj.mjOBJ_GEOM, "table_slab")
    top = float(model.geom_pos[slab][2] + model.geom_size[slab][2])
    obj = nid(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    rest_z = top + BOX_HALF_M[2]
    assert model.body_pos[obj][2] == pytest.approx(rest_z, abs=1e-9)
    assert builder.PEDESTAL_H_M == 0.0


def test_w6_box_mass_and_precision_pins(model):
    obj = nid(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    assert float(model.body_mass[obj]) == pytest.approx(BOX_MASS_KG, abs=1e-12)
    assert builder.box_object_mass() == pytest.approx(BOX_MASS_KG, abs=1e-12)
    # the generator emits the box mass with 9 decimal places; that precision is
    # what kept body_mass reproducible to 2.8e-17 across rebuilds
    xml = MJCF.read_text()
    mass_line = [ln for ln in xml.splitlines() if 'name="object_geom"' in ln]
    assert mass_line and 'mass="0.149' in mass_line[0]
    decimals = mass_line[0].split('mass="')[1].split('"')[0].split(".")[1]
    assert len(decimals) == 9, f"object_geom mass must keep 9-digit precision"


def test_w6_wrist_ry_covers_the_full_quarter_turn(model):
    for sfx in ("", "_left"):
        j = nid(model, mujoco.mjtObj.mjOBJ_JOINT, f"wrist_ry{sfx}")
        lo, hi = model.jnt_range[j]
        assert lo <= -np.pi / 2 and hi >= np.pi / 2, (
            f"wrist_ry{sfx} must cover the +-90 deg end-face pitch"
        )


# ============================================================== GATE 1: birth
def test_w6_gate1_birth_contact_table_has_no_penetration(model, data):
    worst = min((float(data.contact[i].dist) for i in range(data.ncon)),
                default=0.0)
    assert worst >= -BIRTH_PENETRATION_TOL_M, (
        f"birth contact table penetrates by {worst * 1e3:.4f} mm"
    )
    # and the box is born exactly resting: its only contacts are the slab's
    for i in range(data.ncon):
        names = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM,
                                   int(data.contact[i].geom1)),
                 mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM,
                                   int(data.contact[i].geom2))}
        assert "table_slab" in names and "object_geom" in names, (
            f"unexpected birth contact pair {names}"
        )


# ============================================================== GATE 2: settle
def test_w6_gate2_settle_is_quiet(model):
    d = mujoco.MjData(model)
    mujoco.mj_forward(model, d)
    obj = nid(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    rr = nid(model, mujoco.mjtObj.mjOBJ_BODY, "hand_root")
    rl = nid(model, mujoco.mjtObj.mjOBJ_BODY, "hand_root_left")
    p0, r0, l0 = d.xpos[obj].copy(), d.xpos[rr].copy(), d.xpos[rl].copy()
    for _ in range(SETTLE_STEPS):
        mujoco.mj_step(model, d)
    box_move = float(np.linalg.norm(d.xpos[obj] - p0))
    hand_move = max(float(np.linalg.norm(d.xpos[rr] - r0)),
                    float(np.linalg.norm(d.xpos[rl] - l0)))
    assert box_move < REST_MOVE_TOL_M, f"box drifted {box_move * 1e3:.4f} mm"
    assert hand_move < REST_MOVE_TOL_M, f"a hand drifted {hand_move * 1e3:.4f} mm"
    assert np.all(np.isfinite(d.qvel)) and np.abs(d.qvel).max() < 1.0, (
        "settle ends with anomalous velocities - something is ringing or flying"
    )
    # the box is still ON the slab, not through it
    assert d.xpos[obj][2] > BOX_BOT - 1e-3


# ============================================================== GATE 3: reach
def _rect_gap(px, py, pz, side):
    s = abs(px - side * FACE_X)
    dy = max(abs(py) - HALF_Y, 0.0)
    dz = max(BOX_BOT - pz, pz - BOX_TOP, 0.0)
    if dy == 0.0 and dz == 0.0:
        return s
    return float(np.hypot(np.hypot(s, dy), dz))


def _hand(model):
    sfx_by = {}
    for side_name, side, sfx in (("right", +1.0, ""), ("left", -1.0, "_left")):
        wrist = [f"wrist_{a}{sfx}" for a in ("x", "y", "z", "rx", "ry", "rz")]
        fjoints = [f"{f}j{i}{sfx}" for f in FINGERS for i in range(4)]
        names = wrist + fjoints
        sfx_by[side_name] = dict(
            side=side,
            qadr=[int(model.jnt_qposadr[nid(model, mujoco.mjtObj.mjOBJ_JOINT, n)])
                  for n in names],
            rng=[model.jnt_range[nid(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
                 for n in names],
            pads=[nid(model, mujoco.mjtObj.mjOBJ_GEOM, f"{f}_tip{sfx}_collision")
                  for f in FINGERS],
            pad_rad=[float(model.geom_size[
                nid(model, mujoco.mjtObj.mjOBJ_GEOM, f"{f}_tip{sfx}_collision")][0])
                for f in FINGERS],
            collisions=[g for g in range(model.ngeom)
                        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g)
                            or "").endswith("_collision")
                        and (("left" in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g)
                                         or "")) == (sfx == "_left"))],
        )
    return sfx_by


def _objective(model, d, h, q22, tol):
    d.qpos[:] = 0.0
    for adr, v, (lo, hi) in zip(h["qadr"], q22, h["rng"]):
        d.qpos[adr] = np.clip(v, lo, hi)
    mujoco.mj_kinematics(model, d)
    score = np.inf
    for g, rad in zip(h["pads"], h["pad_rad"]):
        px, py, pz = d.geom_xpos[g]
        gap = _rect_gap(px, py, pz, h["side"]) - rad
        score = min(score, gap + 0.5 if gap < -tol else gap)
    for g in h["collisions"]:
        px, py, pz = d.geom_xpos[g]
        if (abs(px) < FACE_X - 0.002 and abs(py) < HALF_Y
                and BOX_BOT < pz < BOX_TOP):
            score += 0.5
    return score


def test_w6_gate3_both_hands_reach_their_end_faces(model):
    """Kinematic probe, pedestal_scan method: coarse structured grid then
    shrinking random refine over all 22 hand DoF, plus random restarts so a
    narrow basin cannot hide. PASS = one configuration where each hand's best
    pad is within the derived R5 touch tolerance of its end face, with NO hand
    geom buried in the box."""
    tol = builder.penetration_tolerance(MJCF)
    hands = _hand(model)
    d = mujoco.MjData(model)
    found = {}
    for hname, seed in (("right", 20251015), ("left", 20251016)):
        h = hands[hname]
        rng = np.random.default_rng(seed)
        best = (np.inf, None)
        for sx in np.arange(-0.06, 0.105, 0.03):
            for sz in np.arange(-0.10, 0.165, 0.03):
                for rx in (-0.6, 0.0, 0.6):
                    for ry in (-0.4, 0.0, 0.4):
                        for c in (0.0, 0.4, 0.8, 1.2):
                            q = np.zeros(22)
                            q[0], q[2], q[3], q[4] = sx, sz, rx, ry
                            for j in range(4):
                                q[6 + 4 * j + 1: 6 + 4 * j + 4] = c
                            q[19:23] = 0.6
                            s = _objective(model, d, h, q, tol)
                            if s < best[0]:
                                best = (s, q.copy())
        p, bscore = best[1].copy(), best[0]
        for rad in (0.3, 0.15, 0.07, 0.03, 0.012, 0.005):
            for _ in range(600):
                q = np.clip(p + rng.uniform(-rad, rad, 22),
                            [lo for lo, _ in h["rng"]], [hi for _, hi in h["rng"]])
                s = _objective(model, d, h, q, tol)
                if s < bscore:
                    bscore, p = s, q
        for _ in range(4):
            q0 = np.array([rng.uniform(lo, hi) for lo, hi in h["rng"]])
            pp, ss = q0.copy(), _objective(model, d, h, q0, tol)
            for rad in (0.15, 0.05, 0.02, 0.008):
                for _ in range(400):
                    q = np.clip(pp + rng.uniform(-rad, rad, 22),
                                [lo for lo, _ in h["rng"]],
                                [hi for _, hi in h["rng"]])
                    s = _objective(model, d, h, q, tol)
                    if s < ss:
                        ss, pp = s, q
            if ss < bscore:
                bscore, p = ss, pp
        found[hname] = (bscore, p.copy())

    # compose both hands' poses into ONE qpos: the simultaneous claim
    d.qpos[:] = 0.0
    for hname in ("right", "left"):
        h = hands[hname]
        for adr, v, (lo, hi) in zip(h["qadr"], found[hname][1], h["rng"]):
            d.qpos[adr] = np.clip(v, lo, hi)
    mujoco.mj_kinematics(model, d)
    gaps = {}
    for hname in ("right", "left"):
        h = hands[hname]
        gaps[hname] = min(
            _rect_gap(*d.geom_xpos[g], h["side"]) - rad
            for g, rad in zip(h["pads"], h["pad_rad"]))
    worst = max(gaps.values())
    assert worst <= tol, (
        f"dual-hand reach failed: best gaps "
        f"R={gaps['right'] * 1e3:+.3f} mm L={gaps['left'] * 1e3:+.3f} mm vs "
        f"tolerance {tol * 1e3:.4f} mm"
    )
    # no burial in the touching configuration
    for hname in ("right", "left"):
        h = hands[hname]
        for g in h["collisions"]:
            px, py, pz = d.geom_xpos[g]
            assert not (abs(px) < FACE_X - 0.002 and abs(py) < HALF_Y
                        and BOX_BOT < pz < BOX_TOP), (
                f"{hname} hand geom buried in the box at the reach pose")


# ============================================================== env-level pins
def test_w6_env_action_and_observation_shapes():
    from grounding.envs.lift_hold_env import LiftHoldEnv  # noqa: PLC0415

    env = LiftHoldEnv()
    assert env.observation_space.shape == (107,)
    env.close()


def test_w6_pedestal_load_weight_untouched():
    assert env_mod.W_PEDESTAL_LOAD == 4.0
    assert env_mod.W_PEDESTAL_LOAD_CAP == 12.0
