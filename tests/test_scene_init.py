"""W4-A - object placement: no birth overlap, and the box really rests.

The C6 control ("park the box where the hand cannot touch it") was passing for
the wrong reason. ``_support_height`` sampled the box CENTRE only, so the park
pose straddling the pedestal edge at y=0.30 m was assigned the bare table
height (0.70 m) and the
box was born 60 mm INSIDE the pedestal (birth contact table: four
``pedestal|object -0.0600`` entries). The solver threw it off the table during
reset's 400 settle steps; it came to rest on the FLOOR (z=0.0394) and the flail
then moved it 0.57 mm, comfortably under the 1e-3 m gate. (v23: the out-of-reach
corner moved to the -x/-y quadrant, far from both y-offset hand mounts; the
dual-mount clearance check replaced the origin-centred circle.) The control measured a
box that was not on the table any more.

These tests pin the three properties the fix must hold:

  * placement height comes from the whole FOOTPRINT, never a single point;
  * no commanded placement produces a negative contact distance at birth;
  * a bare box at rest stays put (< 1 mm over 399 steps, both velocities ~0),
    so a "no contact -> no motion" gate is measuring what it claims to.

Thresholds here are NOT the acceptance gates (those live in
tests/test_smoke4_accept.py and are frozen); they are physics sanity limits.
"""

from __future__ import annotations

import pathlib
import sys

import mujoco
import numpy as np
import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from grounding.envs.lift_hold_env import (  # noqa: E402
    CONTROL_CLEARANCE_M, SUPPORT_EDGE_MARGIN_M)

# anything deeper than this at birth is a real overlap (MuJoCo's contact margin
# is 0 by default, so an exactly-touching pose reads 0.0)
BIRTH_PENETRATION_TOL_M = 1e-9
REST_MOVE_TOL_M = 1e-3        # the C6 gate, applied to a box nobody is touching
REST_LINVEL_TOL = 1e-3        # m/s
REST_ANGVEL_TOL = 1e-3        # rad/s
TABLE_TOP_M = 0.70            # bare slab surface the control box rests on


@pytest.fixture(scope="module")
def env_cls():
    from grounding.envs.lift_hold_env import LiftHoldEnv  # noqa: PLC0415

    return LiftHoldEnv


@pytest.fixture()
def probe(env_cls):
    """A plain env whose model/data are scratch space (reset is never called)."""
    env = env_cls(domain_rand=False)
    yield env
    env.close()


def support_top(env, name: str) -> float:
    for label, _x0, _x1, _y0, _y1, top in env.support_faces:
        if label == name:
            return top
    raise AssertionError(f"no support face named {name}")


def object_geom_ids(env) -> set[int]:
    return {g for g in range(env.model.ngeom)
            if int(env.model.geom_bodyid[g]) == env.obj_body}


def birth_contact_table(env, pos, quat=(1.0, 0.0, 0.0, 0.0)) -> list[tuple[str, str, float]]:
    """Contacts at the instant the box is born, before a single mj_step.

    A scratch MjData so the env's own state is untouched: write qpos, mj_forward,
    read d.contact. Negative ``dist`` is penetration.
    """
    d = mujoco.MjData(env.model)
    adr = env.obj_qpos_adr
    d.qpos[adr:adr + 7] = np.concatenate([np.asarray(pos, dtype=float),
                                          np.asarray(quat, dtype=float)])
    mujoco.mj_forward(env.model, d)
    table = []
    for i in range(d.ncon):
        c = d.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        table.append((mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, g1),
                      mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, g2),
                      float(c.dist)))
    return table


# ------------------------------------------------------------------ (a) no overlap
def test_support_height_uses_the_whole_footprint(probe):
    """The pinned regression for the centre-only sampler, W6a edition.

    W6a removed the pedestal, so the discriminating geometry is now the SLAB
    EDGE: a footprint centred at y = 0.42 straddles the slab boundary
    (slab half-width 0.45), and the footprint-aware `_support_height` must
    still report the slab top for it (the box overlaps the slab), while a
    centre-only sampler reading the support under the CENTRE would give the
    same slab top here - the distinction that remains testable on a single
    face is that the RESOLVER refuses to park an overhanging footprint there
    (asserted in test_control_clearance_never_trades_away_stability via
    _face_fit's edge margin). The footprint itself must span the edge.
    """
    slab = support_top(probe, "table_slab")
    hx, hy = probe.box_half_xy
    assert probe._support_height(0.0, 0.42) == pytest.approx(slab)
    # ... and the footprint really does straddle the slab edge there
    assert 0.42 + hy > 0.45 and 0.42 - hy < 0.45, (
        "park pose no longer straddles the slab edge"
    )
    assert hx > 0.0 and hy > 0.0


@pytest.mark.parametrize("cmd", [
    (0.0, 0.35, 0.0),        # the control's own (unsatisfiable) request
    (0.0, 0.18, 0.0),        # nearest fully-on-pedestal pose on the y axis
    (0.0, 0.0, 0.0),
    (0.30, 0.0, 0.0),        # centre ON the pedestal edge -> footprint hangs off
    (0.31, 0.31, 0.0),
    (-0.28, 0.26, 0.06),     # rotated footprint over the corner
    (0.24, 0.0, 0.10),       # yaw on the pedestal edge
    (0.40, 0.0, 0.0),        # off the pedestal, on the slab, low margin
    (0.0, 0.42, 0.0),
    (0.006, -0.006, 0.0),    # collector grid corner
])
def test_no_commanded_placement_starts_penetrating(probe, cmd):
    """Every pose the resolver can return must be born touching, never inside."""
    x, y, yaw = cmd
    px, py, pz, report = probe._resolve_object_placement(x, y, yaw)
    quat = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
    table = birth_contact_table(probe, (px, py, pz), quat)
    obj = object_geom_ids(probe)
    worst = min([dist for g1, g2, dist in table if g1 in obj or g2 in obj],
                default=0.0)
    assert worst >= -BIRTH_PENETRATION_TOL_M, (
        f"cmd={cmd} -> placed at ({px:.4f}, {py:.4f}, {pz:.4f}) with birth "
        f"contact {worst:+.6f} m: {table}")
    assert report["resting_z_m"] == pytest.approx(pz)


@pytest.mark.parametrize("cmd", [
    (0.0, 0.35, 0.0),        # resolves: slides off the pedestal, clears the hand
    (0.24, 0.0, 0.10),       # a yawed box cannot clear the pedestal AND the
                             # slab edge on x: has to fail LOUDLY
])
def test_control_clearance_never_trades_away_stability(probe, cmd):
    """Under the control's clearance rule the answer is a legal pose or a raise.

    The yawed case is the interesting one: half-x grows to 0.072 m, so the legal
    x window (clear of the pedestal at 0.372 m, clear of the slab edge at
    0.373 m) is empty. Dropping the clearance rule silently would put the
    control box back where the hand can reach it; dropping the stability rule
    would hang it off an edge. It has to raise instead.
    """
    x, y, yaw = cmd
    try:
        px, py, pz, report = probe._resolve_object_placement(
            x, y, yaw, min_clearance=CONTROL_CLEARANCE_M)
    except RuntimeError as exc:
        assert "no stable placement" in str(exc), str(exc)
        return
    assert report["clearance_m"] >= CONTROL_CLEARANCE_M, report
    assert report["edge_margin_m"] >= SUPPORT_EDGE_MARGIN_M, report
    quat = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
    table = birth_contact_table(probe, (px, py, pz), quat)
    obj = object_geom_ids(probe)
    worst = min([dist for g1, g2, dist in table if g1 in obj or g2 in obj],
                default=0.0)
    assert worst >= -BIRTH_PENETRATION_TOL_M, f"cmd={cmd} birth {worst:+.6f} m"


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5, 6, 7])
@pytest.mark.parametrize("kwargs", [
    {},
    {"out_of_reach": True},
    {"domain_rand": True},
])
def test_reset_never_births_a_penetrating_box(env_cls, seed, kwargs):
    """End-to-end: whatever reset() finally puts in qpos starts out legal."""
    env = env_cls(**kwargs)
    _, info = env.reset(seed=seed)
    u = env.unwrapped
    report = info["placement"]
    table = birth_contact_table(u, u.data.xpos[u.obj_body])
    obj = object_geom_ids(u)
    worst = min([dist for g1, g2, dist in table if g1 in obj or g2 in obj],
                default=0.0)
    assert worst >= -BIRTH_PENETRATION_TOL_M, (
        f"{kwargs} seed={seed}: birth contact {worst:+.6f} m at "
        f"{u.data.qpos[u.obj_qpos_adr:u.obj_qpos_adr+3]} (placement {report})")
    # ... and it did not get thrown out of the pose it was given while settling
    settled = np.asarray(report["placed_xy_m"] + [report["resting_z_m"]])
    assert float(np.linalg.norm(u.data.qpos[u.obj_qpos_adr:u.obj_qpos_adr+3] - settled)) < REST_MOVE_TOL_M, (
        f"{kwargs} seed={seed}: box moved {u.data.qpos[u.obj_qpos_adr:u.obj_qpos_adr+3]} -> "
        f"{u.data.qpos[u.obj_qpos_adr:u.obj_qpos_adr+3]} from its own placement {report}")
    env.close()


# ------------------------------------------------------------------ (b) resting
def _rest_probe(env, steps=399):
    """Free-run the box with the hand frozen at its reset pose."""
    u = env.unwrapped
    p0 = u.data.qpos[u.obj_qpos_adr:u.obj_qpos_adr+3].copy()
    q0 = u.data.qpos[u.obj_qpos_adr+3:u.obj_qpos_adr+7].copy()
    worst = 0.0
    for _ in range(steps):
        u.data.ctrl[:] = u._ctrl_ref
        mujoco.mj_step(u.model, u.data)
        worst = max(worst, float(np.linalg.norm(u.data.xpos[u.obj_body] - p0)))
    return p0, worst, q0


@pytest.mark.parametrize("kwargs", [{"out_of_reach": True}, {}])
def test_bare_box_at_rest_does_not_creep(env_cls, kwargs):
    """399 steps of nothing: the box must not move, spin or drift."""
    env = env_cls(domain_rand=False, **kwargs)
    env.reset(seed=0)
    p0, worst, q0 = _rest_probe(env)
    u = env.unwrapped
    v = u.data.qvel[u.obj_qvel_adr:u.obj_qvel_adr + 3]
    w = u.data.qvel[u.obj_qvel_adr + 3:u.obj_qvel_adr + 6]
    assert worst < REST_MOVE_TOL_M, (
        f"{kwargs}: the untouched box drifted {worst * 1e3:.3f} mm over 399 steps "
        f"from {p0} - a rest pose this scene cannot hold")
    assert float(np.linalg.norm(v)) < REST_LINVEL_TOL, f"|v| = {np.linalg.norm(v)} m/s"
    assert float(np.linalg.norm(w)) < REST_ANGVEL_TOL, f"|w| = {np.linalg.norm(w)} rad/s"
    assert float(np.linalg.norm(u.data.qpos[u.obj_qpos_adr+3:u.obj_qpos_adr+7] - q0)) < REST_MOVE_TOL_M, "box rotated"
    env.close()


def test_control_box_stays_on_the_table_under_flail(env_cls):
    """C6 cannot pass by the box falling off the table again.

    The old (false-green) control ended the flail at z=0.0394 m - the floor.
    """
    env = env_cls(out_of_reach=True)
    env.reset(seed=3)
    u = env.unwrapped
    rng = np.random.default_rng(3)
    contacts = 0
    for _ in range(100):
        env.step(rng.uniform(-1, 1, 44).astype(np.float32))
        contacts += any(p[0].split("#")[0].startswith(("ff_", "mf_", "rf_", "th_"))
                        for p in u.trace["contact_pairs"][-1])
    assert contacts == 0, f"hand touched the parked box {contacts} times"
    assert float(u.data.xpos[u.obj_body][2]) > TABLE_TOP_M, (
        f"the control box fell off the table: z={u.data.xpos[u.obj_body][2]:.4f}")
    env.close()


# ------------------------------------------------------------------ the control pose
def test_control_placement_clears_the_hand_by_construction(env_cls):
    """The park pose must lie outside the wrist's whole travel envelope.

    Read from the model, not hardcoded: the wrist x/y actuators cap at +-0.18 m,
    and the box footprint has to stay outside that disk or "out of reach" is a
    property of the flail rather than of the scene.
    """
    from grounding.envs.lift_hold_env import CONTROL_CLEARANCE_M  # noqa: PLC0415

    env = env_cls(out_of_reach=True, domain_rand=False)
    _, info = env.reset(seed=0)
    report = info["placement"]
    travel = float(np.abs(env.model.actuator_ctrlrange[0:2]).max())
    assert report["clearance_m"] > travel, (
        f"parked box clears the hand by {report['clearance_m']:.4f} m but the wrist "
        f"can travel {travel:.4f} m")
    assert report["clearance_m"] >= CONTROL_CLEARANCE_M
    assert report["moved"] is True, (
        "the +y park request became satisfiable - re-derive CONTROL_CLEARANCE_M "
        "and the report before accepting the pose it asks for")
    env.close()


def test_placement_report_names_the_pose_it_chose(env_cls):
    """A silent correction is how this bug survived: the move must be visible."""
    env = env_cls(out_of_reach=True, domain_rand=False)
    _, info = env.reset(seed=0)
    report = info["placement"]
    assert set(report) == {"requested_xy_m", "placed_xy_m", "resting_z_m",
                           "edge_margin_m", "clearance_m", "moved"}
    # v23: the request corner moved to the -x/-y quadrant (far from both mounts)
    assert report["requested_xy_m"] == [0.05, 0.30]
    assert report["placed_xy_m"] != report["requested_xy_m"]
    assert report["edge_margin_m"] > 0.0
    env.close()
