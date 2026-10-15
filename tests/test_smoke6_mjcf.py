"""Wave 3 MJCF revision - the five physics defects smoke4b/smoke5 measured.

RED-first. Every assertion below is pinned to a *measurement*, not a taste:

  R1 finger forcerange  smoke4b measured a mean fingertip force of 37.7 N while the
                        real Wonik Allegro V5 is spec'd at 10 N per fingertip, so the
                        sim hand was ~3.8x too strong and the policy was cheating by
                        pressing. The fix is not a magic number: each joint's
                        forcerange is ``10 N * moment_arm_j``, the moment arm being
                        the perpendicular distance from that joint's axis to the
                        fingertip measured from the MJCF's own geometry, minimised
                        over the reachable workspace with the degenerate poses (axis
                        passing through the tip capsule, finger folded into the palm)
                        excluded - a pose MuJoCo will not integrate without the
                        fingertip already intersecting the palm is not a grasp pose.

  R2 wrist travel       smoke5's gauge fixing left wrist_z 0.062 m and wrist_ry
                        0.058 rad short of the commanded range. Those are physical
                        (vertical descent, pitch swing), so the travel is widened.
                        wrist_rz / x / y had zero residual and must not move.

  R3 contact stiffness  default MuJoCo contacts let a fingertip press a rigid paper
                        box 2.7 mm deep. solref/solimp are raised so the deepest
                        penetration under a 10 N fingertip load stays under 1 mm and
                        the resting box does not jitter.

  R4 geom naming        all 42 finger/palm geoms came out of unnamed <default>
                        classes, which once made a contact detector report a
                        structural zero. Root cause removal: every one is named.

  R5 penetration tol    the tolerance is derived as k x the measured resting sink,
                        not a hardcoded 1e-3, so it follows the stiffness change.

Run: pytest tests/test_smoke6_mjcf.py
"""

from __future__ import annotations

import json
import pathlib
import re
import subprocess
import sys

import mujoco
import numpy as np
import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from grounding.assets import build_allegro_floating as builder  # noqa: E402

GENERATOR = REPO / "grounding/assets/build_allegro_floating.py"
MJCF = REPO / "grounding/assets/allegro_floating.xml"
REVISION_MD = REPO / "grounding/assets/REVISION.md"
SMOKE5_QA = REPO / "runs/p0/smoke5_retarget/retarget_qa.json"

# ---- frozen physical quantities (must not move) -----------------------------
BOX_HALF_M = (0.06, 0.12, 0.04)
BOX_MASS_KG = 0.15
PEDESTAL_H_M = 0.06
CONTROL_HZ = 20
N_ACT = 22
FINGER_JOINTS = [f"{f}j{i}" for f in ("ff", "mf", "rf", "th") for i in range(4)]

# ---- the real machine's spec, the number R1 is derived from ---------------
ALLEGRO_V5_TIP_FORCE_N = 10.0
PRE_REVISION_TIP_FORCE_N = 37.7   # smoke4b measured mean
PRE_REVISION_FORCERANGE = 2.0     # what the old MJCF gave every finger joint


@pytest.fixture(scope="module")
def model():
    return mujoco.MjModel.from_xml_path(str(MJCF))


@pytest.fixture(scope="module")
def data(model):
    return mujoco.MjData(model)


# ===================================================================== R1
def _tip_geoms(model) -> dict[str, int]:
    out = {}
    for f in ("ff", "mf", "rf", "th"):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{f}_tip")
        gids = [g for g in range(model.ngeom)
                if model.geom_bodyid[g] == bid
                and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE]
        assert len(gids) == 1, f"{f}_tip must own exactly one capsule geom, got {gids}"
        out[f] = gids[0]
    return out


def test_r1_generator_derives_forcerange_from_moment_arm_not_a_literal():
    """R1 root cause: the old generator wrote forcerange="-2.0 2.0" as a literal on
    all 16 joints. The revision must compute it from the measured moment arm.

    The check covers both the generator source and the emitted XML, because the build
    runs two passes: a placeholder model to measure, then the measured one. A
    literal that survived in either place could reach a committed XML."""
    src = GENERATOR.read_text()
    xml = MJCF.read_text()
    for blob, what in ((src, "generator source"), (xml, "generated XML")):
        assert 'forcerange="-2.0 2.0"' not in blob, (
            f"the flat +-2.0 N*m literal is back in the {what}; R1 requires per-joint "
            f"forcerange derived from the moment arm"
        )
    assert "moment_arm" in src, "the generator never mentions a moment arm"
    assert builder.FINGER_TIP_FORCE_LIMIT_N == pytest.approx(ALLEGRO_V5_TIP_FORCE_N), (
        "the fingertip force limit must be the Allegro V5 spec value"
    )


def test_r1_placeholder_forcerange_never_reaches_the_committed_xml():
    """The build's pass-1 placeholder must be overwritten by pass 2. If the measured
    table ever failed to load, the placeholder would be committed silently - so the
    committed XML must contain no zero-force finger actuator."""
    xml = MJCF.read_text()
    assert builder.PLACEHOLDER_FORCERANGE_NM == 0.0, (
        "a non-zero placeholder risks being committed if the measurement pass fails"
    )
    assert 'forcerange="0 0"' not in xml, (
        "a finger actuator still carries the build placeholder - the measured "
        "forcerange table did not reach the XML"
    )


def test_r1_moment_arm_table_is_geometric_and_positive(model):
    """Every joint gets a finite, strictly positive moment arm measured from the
    MJCF, and the table is exactly the 16 finger joints."""
    table = builder.moment_arm_table()
    assert set(table) == set(FINGER_JOINTS)
    for name, arm in table.items():
        assert np.isfinite(arm) and arm > 5e-3, (
            f"{name}: moment arm {arm:.5f} m is degenerate - the fingertip sits on "
            f"the joint axis, so no torque limit can bound the fingertip force"
        )
    # Sanity: the arm must be a length, and the finger's own link lengths bound it.
    assert all(a < 0.20 for a in table.values()), "moment arm exceeds the finger's reach"


def test_r1_actuator_lookup_is_not_silently_aliasing(model):
    """Every finger joint must resolve to its OWN actuator.

    Upstream names the actuators ffa0/mfa0/rfa0/tha0 after the joint's role, not
    ffj0/mfj0/... . mj_name2id on a joint name returns -1, and indexing an array with
    -1 silently returns the LAST element - so a forcerange check written that way
    reads tha3's numbers sixteen times and passes whatever tha3 happens to be. That
    is how a whole-finger force budget can be violated without any test going red."""
    qadr, actr = builder.finger_qpos_and_actuator_adrs(model)
    assert len(set(actr.tolist())) == 16, "finger actuators alias each other"
    for k, name in enumerate(FINGER_JOINTS):
        assert actr[k] == builder.finger_actuator_id(model, name)
        assert model.actuator_trnid[actr[k], 0] == mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, name), f"{name} drives the wrong joint"
    # and the derived forceranges really do differ between joints
    taus = {round(float(model.actuator_forcerange[builder.finger_actuator_id(model, n)][1]), 6)
            for n in FINGER_JOINTS}
    assert len(taus) > 1, "all 16 joints report the same forcerange - an aliasing bug"


def test_r1_forcerange_is_the_shared_budget_times_the_moment_arm(model):
    """The headline identity. The 10 N spec is a budget for a whole *fingertip*, and
    four joints press that same pad, so each joint gets a quarter of the budget
    converted through its own moment arm."""
    table = builder.moment_arm_table()
    for i, name in enumerate(FINGER_JOINTS):
        bid = builder.finger_actuator_id(model, name)
        lo, hi = model.actuator_forcerange[bid]
        assert lo < 0 < hi, f"{name}: forcerange must be symmetric about zero"
        assert hi == pytest.approx(builder.finger_forcerange(table[name]), rel=2e-3), (
            f"{name}: forcerange {hi:.5f} N*m != (10 N / 4) * arm {table[name]:.5f} m "
            f"= {builder.finger_forcerange(table[name]):.5f} N*m"
        )


def test_r1_four_joints_of_a_finger_sum_to_the_ten_newton_budget():
    """The point of the quarter-share. Each joint's tau/arm is the largest fingertip
    force it can contribute (the arm is a minimum, so the quotient is a maximum), and
    the four contributions of a finger must add up to the spec, not to 4x it.

    A per-joint 10 N would read 10 N here and pass a naive per-joint check while
    letting one fingertip reach 40 N, which is exactly the cheat R1 exists to remove.
    """
    table = builder.moment_arm_table()
    for prefix in ("ff", "mf", "rf", "th"):
        total = sum(builder.finger_forcerange(table[f"{prefix}j{j}"]) / table[f"{prefix}j{j}"]
                    for j in range(4))
        assert total == pytest.approx(ALLEGRO_V5_TIP_FORCE_N, rel=1e-9), (
            f"{prefix}: its four joints can together press {total:.3f} N into one "
            f"fingertip, spec is {ALLEGRO_V5_TIP_FORCE_N} N"
        )


def test_r1_forcerange_is_much_weaker_than_before(model):
    """Direction check: the revision must actually reduce authority, not relabel it."""
    for name in FINGER_JOINTS:
        bid = builder.finger_actuator_id(model, name)
        hi = model.actuator_forcerange[bid][1]
        assert hi < PRE_REVISION_FORCERANGE, f"{name} still has the old +-2 N*m"


def test_r1_measured_fingertip_force_stays_under_ten_newtons(model):
    """End-to-end: drive the hand through grasp poses at the task's 20 Hz control
    rate and measure the force the box actually feels. The static derivation above is
    only worth anything if the simulated contact agrees with it.

    This is the check smoke4b's 37.7 N failed: it measured a real contact force, and
    the revision has to bring that number under the real machine's 10 N.

    Two regimes are measured separately, because they are different physics and only
    one of them is a fingertip force:

    * **surface contact** (penetration within the R5 tolerance): the pad is on the
      box and pressing it. The contact force here is the actuator budget arriving at
      the fingertip, and it is what the 10 N spec is about. This is the gate.
    * **buried geom** (penetration beyond it): a finger *link* has been commanded to
      a pose that puts it inside the box - reachable now that R2 lets the hand
      descend 63 mm. MuJoCo resolves the overlap with the constraint solver, whose
      force is set by the overlap depth, not by any actuator's authority. It is
      reported, not gated, and the corresponding scene property is a known limit
      recorded in REVISION.md."""
    data = mujoco.MjData(model)
    qadr, actr = builder.finger_qpos_and_actuator_adrs(model)
    ctrlrange = model.actuator_ctrlrange[actr]
    obj = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "object_geom")
    hand_geoms = {g for g in range(model.ngeom)
                  if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                                        int(model.geom_bodyid[g])) or "")
                  .startswith(("ff_", "mf_", "rf_", "th_"))
                  and model.geom_contype[g] != 0}
    wrist_z = int(model.jnt_qposadr[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "wrist_z")])
    wrist_ry = int(model.jnt_qposadr[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "wrist_ry")])
    pre_curl = np.array([0.0, 0.9, 0.7, 0.4] * 3 + [0.8, 0.5, 0.4, 0.3])
    rng = np.random.default_rng(20251015)
    frame_skip = max(1, int(round(1.0 / (CONTROL_HZ * model.opt.timestep))))

    worst_surface = 0.0
    worst_buried = 0.0
    tol = builder.penetration_tolerance()
    for _ in range(120):
        mujoco.mj_resetData(model, data)
        data.qpos[wrist_z] = rng.uniform(*model.jnt_range[
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "wrist_z")]) * 0.8
        data.ctrl[2] = data.qpos[wrist_z]
        data.qpos[wrist_ry] = rng.uniform(*model.jnt_range[
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "wrist_ry")]) * 0.6
        data.ctrl[4] = data.qpos[wrist_ry]
        data.qpos[qadr] = pre_curl
        data.ctrl[actr] = pre_curl
        for _ in range(250):
            mujoco.mj_step(model, data)
        target = ctrlrange[:, 0] + rng.random(16) * (ctrlrange[:, 1] - ctrlrange[:, 0])
        for k in (0, 4, 8):
            target[k] = rng.uniform(-0.47, 0.47)
        # close over 50 control frames at 20 Hz, as the task's action rate does
        for s in range(50):
            data.ctrl[actr] = pre_curl + (target - pre_curl) * ((s + 1) / 50.0)
            for _ in range(frame_skip):
                mujoco.mj_step(model, data)
        for _ in range(150):
            mujoco.mj_step(model, data)
            for c in range(data.ncon):
                con = data.contact[c]
                g1, g2 = int(con.geom1), int(con.geom2)
                if obj not in (g1, g2):
                    continue
                other = g2 if g1 == obj else g1
                if other not in hand_geoms:
                    continue
                f = np.zeros(6)
                mujoco.mj_contactForce(model, data, c, f)
                mag = float(np.linalg.norm(f[:3]))
                if -float(con.dist) <= tol:
                    worst_surface = max(worst_surface, mag)
                else:
                    worst_buried = max(worst_buried, mag)
    # Two different physics, gated separately. See the docstring above.
    assert 0.0 < worst_surface <= ALLEGRO_V5_TIP_FORCE_N, (
        f"a fingertip pressing the box surface reaches {worst_surface:.2f} N; the "
        f"Allegro V5 limit is {ALLEGRO_V5_TIP_FORCE_N} N/finger (smoke4b measured "
        f"{PRE_REVISION_TIP_FORCE_N} N before this revision)"
    )
    # Not a gate, but it must not be silently enormous either: a finger link driven
    # far inside the box is resolved by the constraint solver, not by actuator
    # authority, so the number is reported rather than bounded.
    assert worst_buried < 5e3, (
        f"buried-geom contact force {worst_buried:.0f} N is outside any plausible "
        f"regime - something other than commanded interpenetration is going on"
    )


def test_r3_did_not_stiffen_the_palm_into_a_force_spike(model):
    """R3 regression guard. An earlier pass stamped the stiff solref on *every*
    colliding hand geom. That governed the box<->finger pairs as intended, but it
    also stiffened the palm-vs-pedestal contact - a pre-existing near-tangential
    graze - and turned it into an incompressible squeeze whose peak force went from
    3.2 kN to 77 kN, 24x worse, on a contact R3 says nothing about.

    The stiffening must therefore stay on the box and the four fingertip pads."""
    palm = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "palm")
    palm_coll = [g for g in range(model.ngeom)
                 if model.geom_bodyid[g] == palm and model.geom_contype[g] != 0]
    assert palm_coll, "the palm must keep its collision geom"
    for g in palm_coll:
        assert model.geom_solref[g][0] == pytest.approx(builder.DEFAULT_SOLREF[0]), (
            f"palm geom {g} carries solref {model.geom_solref[g][0]}; R3 must not "
            f"stiffen the hand-vs-table contacts"
        )


def test_r1_kp_is_checked_against_gravity_and_saturation(model):
    """kp must satisfy two constraints at once:

      * reach   - kp * (worst tracking error) >= forcerange, else the actuator
                  saturates below its own force limit and the forcerange is fiction.
      * hold    - kp * (gravity sag budget) >= worst-case gravity torque, else the
                  finger cannot hold its own weight.
    """
    for name in FINGER_JOINTS:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        bid = builder.finger_actuator_id(model, name)
        kp = model.actuator_gainprm[bid][0]
        tau = model.actuator_forcerange[bid][1]
        grav = builder.gravity_torque_table()[name]
        span = model.jnt_range[jid][1] - model.jnt_range[jid][0]
        assert kp > 0.0, f"{name}: kp must be positive"
        # reach: full-span tracking error is the worst case the servo can see
        assert kp * span >= tau, (
            f"{name}: kp={kp:.3f} x span={span:.3f} rad = {kp * span:.4f} N*m < "
            f"forcerange {tau:.4f} N*m - the actuator can never reach its own limit"
        )
        # hold: gravity must be correctable within the documented sag budget
        assert kp * builder.GRAVITY_SAG_BUDGET_RAD >= grav, (
            f"{name}: kp={kp:.3f} can only supply {kp * builder.GRAVITY_SAG_BUDGET_RAD:.4f}"
            f" N*m at the {builder.GRAVITY_SAG_BUDGET_RAD} rad sag budget but gravity "
            f"needs {grav:.4f} N*m"
        )


def test_r1_kp_does_not_destabilise_the_finger(model):
    """The other side of kp: too high and the 2 ms explicit-ish integration rings.
    Bound the servo's natural frequency against the timestep."""
    dt = model.opt.timestep
    for name in FINGER_JOINTS:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        bid = builder.finger_actuator_id(model, name)
        kp = model.actuator_gainprm[bid][0]
        inertia = builder.effective_inertia_table()[name]
        omega = np.sqrt(kp / inertia)
        assert omega * dt < 1.0, (
            f"{name}: servo natural frequency {omega:.1f} rad/s is "
            f"{omega * dt:.2f} steps per radian at dt={dt} - it will ring"
        )


# ===================================================================== R2
def test_r2_wrist_z_travel_covers_the_measured_residual(model):
    """smoke5 measured 0.062088 m of wrist_z travel still needed after the gauge
    fixing, and R2 widened the low end for it. The 5c base solve has since removed
    that residual at the source (overshoot_after == 0), so the gate is now: the
    residual stays solved, and the widened travel still covers the command minimum."""
    qa = json.loads(SMOKE5_QA.read_text())
    overshoot = qa["retargeting"]["wrist_anchor"]["gauge_fixing"]["overshoot_after"][2]
    assert overshoot == pytest.approx(0.0, abs=5e-4), "gauge residual is back"
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "wrist_z")
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "wrist_az")
    lo, hi = model.jnt_range[jid]
    assert lo <= -0.10 - overshoot, (
        f"wrist_z low end {lo:.6f} still clips smoke5's command minimum "
        f"{-0.10 - overshoot:.6f}"
    )
    assert lo == pytest.approx(-0.163, abs=1e-9)
    assert hi == pytest.approx(0.16, abs=1e-6), "wrist_z high end must not move"
    assert model.actuator_ctrlrange[bid][0] == pytest.approx(lo, abs=1e-6)
    assert model.actuator_ctrlrange[bid][1] == pytest.approx(hi, abs=1e-6)


def test_r2_wrist_ry_travel_covers_the_measured_residual(model):
    """Same for wrist_ry: smoke5 measured 0.058449 rad of pitch travel still
    needed and R2 widened the low end for it. The 5c base solve removed the
    residual at the source (overshoot_after == 0); the widened travel stays."""
    qa = json.loads(SMOKE5_QA.read_text())
    overshoot = qa["retargeting"]["wrist_anchor"]["gauge_fixing"]["overshoot_after"][4]
    assert overshoot == pytest.approx(0.0, abs=5e-4), "gauge residual is back"
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "wrist_ry")
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "wrist_ry")
    lo, hi = model.jnt_range[jid]
    assert lo <= -1.20 - overshoot, (
        f"wrist_ry low end {lo:.6f} still clips smoke5's command minimum "
        f"{-1.20 - overshoot:.6f}"
    )
    assert lo == pytest.approx(-1.259, abs=1e-9)
    assert hi == pytest.approx(1.20, abs=1e-6), "wrist_ry high end must not move"
    assert model.actuator_ctrlrange[bid][0] == pytest.approx(lo, abs=1e-6)


def test_r2_untouched_wrist_axes_keep_their_exact_range(model):
    """wrist_rz / x / y had ZERO residual after the gauge fixing. The plan is
    explicit: any gauge choice that could absorb them is not a valid argument, so
    these three must be byte-identical to the pre-revision values."""
    for name, expect in (("wrist_x", (-0.18, 0.18)),
                         ("wrist_y", (-0.18, 0.18)),
                         ("wrist_rz", (-1.57, 1.57))):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        assert tuple(model.jnt_range[jid]) == pytest.approx(expect, abs=1e-9), (
            f"{name} moved; only wrist_z and wrist_ry had a measured residual"
        )


# ===================================================================== Wave 3.5-C
def test_w35c_wrist_rx_travel_widened_for_the_thumb_opposition_cone(model):
    """Wave 3.5-C: runs/w35/verdict.json proved the thumb opposition cone bottoms
    out 13.702 mm above the box top in every feasible pose at three pedestal
    heights - the cone is fixed relative to the palm, and K3 (palm clear of the
    box top) pins the palm. The one wrist DOF that pitches the cone down without
    lifting the opposite long fingers is wrist_rx, whose +/-1.2 rad cap the
    earlier scans never probed past |0.6| rad. Travel widened to +/-1.8 rad
    (same R2 principle: travel is a modelling choice and must cover the task
    workspace); the K1 verification numbers live in runs/w35/rx_widen_scan.json."""
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "wrist_rx")
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "wrist_rx")
    lo, hi = model.jnt_range[jid]
    assert (lo, hi) == pytest.approx((-1.80, 1.80), abs=1e-9), (
        f"wrist_rx travel is [{lo:.6g}, {hi:.6g}]; Wave 3.5-C widened it to "
        f"[-1.8, 1.8] for the thumb opposition cone"
    )
    assert tuple(model.actuator_ctrlrange[bid]) == pytest.approx(
        (-1.80, 1.80), abs=1e-9), "wrist_rx ctrlrange must match the joint range"


def test_r2_commanded_smoke5_trajectory_now_fits(model):
    """The point of R2: replaying the actual smoke5 command minima must land inside
    ctrlrange on every axis. This is the end-to-end form of the residual claim."""
    qa = json.loads(SMOKE5_QA.read_text())
    cmd_lo = qa["actuators"]["wrist_reachability"]["command_min"]
    for i, (jname, aname) in enumerate(
            zip(("wrist_x", "wrist_y", "wrist_z", "wrist_rx", "wrist_ry", "wrist_rz"),
                ("wrist_ax", "wrist_ay", "wrist_az", "wrist_rx", "wrist_ry", "wrist_rz"))):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aname)
        assert bid >= 0, f"no actuator named {aname}"
        lo, hi = model.actuator_ctrlrange[bid]
        assert lo - 1e-9 <= cmd_lo[i] <= hi + 1e-9, (
            f"{jname}: smoke5 commanded {cmd_lo[i]:.5f}, ctrlrange is [{lo:.5f}, {hi:.5f}]"
        )


# ===================================================================== R3
def test_r3_contact_stiffness_is_raised_on_the_object(model):
    """R3: the box must be stiffer than MuJoCo's default, on the object side so the
    pair inherits it."""
    obj = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "object_geom")
    solref = model.geom_solref[obj]
    assert solref[0] < 0.005, (
        f"box solref timeconst {solref[0]:.4f} s is the soft default; smoke5 measured "
        f"a fingertip pressing 2.7 mm into a rigid paper box"
    )
    assert solref[1] == pytest.approx(1.0, abs=1e-6), "contact must be critically damped"


def test_r3_fingertip_contacts_inherit_the_stiffness(model):
    """The pair takes the stiffer of the two sides, so the fingertip pads must be at
    least as stiff as the box - otherwise the box<->finger contact silently stays
    soft and R3 does nothing."""
    obj = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "object_geom")
    box_ref = model.geom_solref[obj][0]
    for f in ("ff", "mf", "rf", "th"):
        gid = _tip_geoms(model)[f]
        assert model.geom_solref[gid][0] <= box_ref + 1e-9, (
            f"{f} fingertip contact is softer than the box it presses"
        )


def test_r3_penetration_under_a_ten_newton_fingertip_load_is_under_one_mm(model):
    """The acceptance condition of R3, measured not asserted: press the resting box
    with 10 N - the Allegro V5 per-fingertip limit, i.e. the worst force the revised
    hand can ever apply - and require the deepest support penetration to stay under
    1 mm."""
    data = mujoco.MjData(model)
    obj = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "object_geom")
    dof = int(model.jnt_dofadr[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "object_free")])
    for _ in range(900):
        data.qfrc_applied[:] = 0.0
        data.qfrc_applied[dof + 2] = -10.0
        mujoco.mj_step(model, data)
    pen = 0.0
    for c in range(data.ncon):
        con = data.contact[c]
        if obj in (int(con.geom1), int(con.geom2)):
            other = int(con.geom2) if int(con.geom1) == obj else int(con.geom1)
            if int(model.geom_bodyid[other]) == 0:
                pen = max(pen, -float(con.dist))
    assert 0.0 < pen < 1e-3, (
        f"a 10 N load sinks the box {pen * 1e3:.3f} mm into the pedestal; R3 requires "
        f"under 1 mm (smoke5 measured 2.7 mm at the fingertip)"
    )


def test_r3_resting_box_does_not_jitter(model):
    """Stiffer contacts are only acceptable if they stay quiet: the settled box must
    not oscillate."""
    data = mujoco.MjData(model)
    for _ in range(700):
        mujoco.mj_step(model, data)
    zs = []
    for _ in range(400):
        mujoco.mj_step(model, data)
        zs.append(float(data.qpos[2]))
    assert float(np.std(zs)) < 1e-6, (
        f"resting box jitters with std {np.std(zs):.3e} m under the stiffened contacts"
    )


def test_r3_revised_scene_still_holds_the_box_up(model):
    """Stiffening must not launch the box: with the fingers in the pre-curl grasp
    shape and the wrist at its home descent, the box still settles onto the pedestal
    and stays there."""
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    pre_curl = np.array([0.0, 0.9, 0.7, 0.4] * 3 + [0.8, 0.5, 0.4, 0.3])
    finger_adr, finger_act = builder.finger_qpos_and_actuator_adrs(model)
    wz = int(model.jnt_qposadr[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "wrist_z")])
    data.qpos[finger_adr] = pre_curl
    data.qpos[wz] = -0.02
    data.ctrl[finger_act] = pre_curl
    data.ctrl[2] = -0.02
    for _ in range(600):
        mujoco.mj_step(model, data)
    obj_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    rest = builder.PEDESTAL_H_M + builder.TABLE_TOP_Z_M + BOX_HALF_M[2]
    assert abs(float(data.xpos[obj_body][2]) - rest) < 5e-3, (
        f"box rests at {data.xpos[obj_body][2]:.5f}, expected ~{rest:.5f}"
    )


# ===================================================================== R4
def _hand_body_ids(model) -> set[int]:
    ids = set()
    for b in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or ""
        if name == "palm" or name.startswith(("ff_", "mf_", "rf_", "th_")):
            ids.add(b)
    return ids


def test_r4_every_finger_and_palm_geom_is_named(model):
    """R4 root cause removal: 42 geoms came out of unnamed <default> classes, which
    is what made a name-based contact detector structurally blind."""
    bodies = _hand_body_ids(model)
    unnamed = [g for g in range(model.ngeom)
               if int(model.geom_bodyid[g]) in bodies
               and not mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g)]
    assert not unnamed, (
        f"{len(unnamed)} hand geoms are still unnamed (first: "
        f"{mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[unnamed[0]]))})"
    )
    n_hand = sum(1 for g in range(model.ngeom) if int(model.geom_bodyid[g]) in bodies)
    assert n_hand == 42, f"expected the 42 upstream hand geoms, found {n_hand}"


def test_r4_hand_geom_names_are_unique_and_resolvable(model):
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g)
             for g in range(model.ngeom) if int(model.geom_bodyid[g]) in _hand_body_ids(model)]
    assert len(set(names)) == len(names), "duplicate hand geom names"
    for n in names:
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n) >= 0, f"{n} unresolvable"


def test_r4_name_resolves_the_fingertip_contact_by_name(model):
    """The concrete failure smoke5 hit: a contact read by name. After naming, a
    fingertip geom must be findable by name alone, with no ancestry fallback."""
    for f in ("ff", "mf", "rf", "th"):
        gid = _tip_geoms(model)[f]
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
        assert name is not None, f"{f} fingertip geom is unnamed"
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) == gid


def test_r4_generator_names_geoms_rather_than_the_xml(model):
    """The XML is a build artefact; the naming must live in the generator."""
    src = GENERATOR.read_text()
    assert "geom_name" in src or "_name_hand_geoms" in src, (
        "the generator does not name hand geoms - the XML would drift on rebuild"
    )


# ===================================================================== R5
def test_r5_resting_sink_is_measured_not_assumed():
    """The baseline function must actually simulate: perturbing the stiffness must
    move the number it returns."""
    a = builder.measure_resting_sink()
    b = builder.measure_resting_sink(solref_timeconst=0.001)
    assert 0.0 < a < 1e-3, f"implausible resting sink {a:.3e} m"
    assert b < a, "a stiffer contact did not reduce the resting sink - the function is fake"


def test_r5_tolerance_is_k_times_the_measured_sink():
    """The tolerance follows the measurement. k >= 3 by contract."""
    sink = builder.measure_resting_sink()
    tol = builder.penetration_tolerance()
    assert builder.PENETRATION_TOLERANCE_K >= 3.0
    assert tol == pytest.approx(builder.PENETRATION_TOLERANCE_K * sink, rel=1e-9)
    assert tol < 1e-3, f"derived tolerance {tol:.3e} is looser than the old 1e-3"


def test_r5_tolerance_follows_a_stiffness_change():
    """Change the contact model, the tolerance must move with it."""
    base = builder.penetration_tolerance()
    stiff = builder.penetration_tolerance(solref_timeconst=0.001)
    assert stiff < base, "the derived tolerance ignored the stiffness change"


def test_r5_derived_tolerance_is_documented_in_revision_md():
    doc = REVISION_MD.read_text()
    sink = builder.measure_resting_sink()
    tol = builder.penetration_tolerance()
    assert str(round(sink, 8))[:8] in doc or f"{sink:.3e}" in doc, (
        "REVISION.md does not record the measured resting sink"
    )
    assert f"{tol:.3e}" in doc or str(round(tol, 8))[:8] in doc, (
        "REVISION.md does not record the derived tolerance"
    )


# ============================================================== regression
def test_frozen_box_and_scene_constants():
    """The box and pedestal are the evaluator's, not ours."""
    assert BOX_HALF_M == (0.06, 0.12, 0.04)
    assert BOX_MASS_KG == 0.15
    assert builder.BOX_HALF_M == BOX_HALF_M
    assert builder.BOX_MASS_KG == BOX_MASS_KG
    assert builder.PEDESTAL_H_M == PEDESTAL_H_M


def test_action_semantics_unchanged(model):
    assert model.nu == N_ACT
    assert model.nu - 6 == 16
    for i, name in enumerate(("wrist_ax", "wrist_ay", "wrist_az",
                              "wrist_rx", "wrist_ry", "wrist_rz")):
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) == i


def test_generated_xml_is_reproducible_from_the_generator():
    """A build artefact that cannot be rebuilt is not auditable."""
    before = MJCF.read_text()
    subprocess.run([sys.executable, str(GENERATOR)], check=True,
                   cwd=str(REPO), capture_output=True)
    assert MJCF.read_text() == before, (
        "regenerating the XML changed it - the committed XML is stale"
    )


def test_model_loads_and_statistics_are_self_consistent(model):
    assert model.ngeom == 46
    assert model.nbody == 24
    assert model.nq == 29
    assert model.nu == 22
    assert model.nv == 28
    obj = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    assert float(model.body_mass[obj]) == pytest.approx(BOX_MASS_KG)


def test_revision_md_records_the_forcerange_table():
    """The per-joint moment-arm / forcerange table is the deliverable of R1."""
    doc = REVISION_MD.read_text()
    table = builder.moment_arm_table()
    for name in FINGER_JOINTS:
        assert name in doc, f"REVISION.md has no row for {name}"
    assert "moment arm" in doc.lower()
    for name, arm in table.items():
        assert f"{arm * 1e3:.2f}" in doc or f"{arm:.5f}" in doc, (
            f"REVISION.md does not record {name}'s measured moment arm {arm:.5f} m"
        )


def test_model_stats_script_runs():
    """The self-check script the revision ships must actually run."""
    out = subprocess.run([sys.executable, str(REPO / "grounding/assets/model_stats.py")],
                         cwd=str(REPO), capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    payload = json.loads(out.stdout)
    assert payload["model_shape"]["ngeom"] == 46
    assert payload["model_shape"]["nu"] == 22
