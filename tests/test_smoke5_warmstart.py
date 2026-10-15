"""Smoke 5 warm start -- rate-limited approach to the first commanded pose.

Two properties, both behavioural (they call ``retarget_smoke.warm_start`` and
step physics; they do not read the committed JSON):

  W1  inside the warm-start segment no DoF is driven faster than the servo's
      measured trackable rate.  The caps are per class because the units are:
      rad/s for hinges and finger joints, m/s for slides.
  W1b the servos' *realised* rates.  A command-side cap cannot bound a
      physical-side transient, so this one is the three-part servo winding
      tolerance table: |forcerange| / dof_damping per hinge, a duration window,
      and a mandatory wind-up-release explanation.
  W2  the ramp is pure input shaping -- after it, the hand sits in the same
      world pose as a reference that was driven straight to the first command.
      This is what stops a "helpful" ramp from quietly relocating the hand or
      the box on the way in.

Neither test asserts that the warm start *succeeds*.  On the superseded
single-hand scene it did not: the first commanded wrist pose was outside that
floating wrist's reachable set (the palm rested on the box top and blocked the
wrist's descent).  On the W6a dual-hand scene the re-scaled wrist slides
deliver the commanded pose and the residual tracking error sits on the finger
joints holding their contact deflection --
``test_w2c_first_command_reachability_is_recorded_per_dof`` records whichever
way the current scene falls, per DoF, rather than pretending either way.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from retarget import retarget_smoke as rs  # noqa: E402

REPORT_PATH = REPO / "runs/p0/smoke5_retarget/report.json"


def _first_command():
    """The exact (finger, wrist) command the replay's frame 0 applies."""
    mujoco_mod, model, data = rs.load_scene()
    rs._mj_forward(model, data)
    seq = rs.load_source_sequence()
    joints21, _ = rs.mano_skeleton(seq)
    retargeter, _ = rs.build_retargeter()
    qpos = rs.retarget_sequence(joints21, retargeter)
    # The executed anchor, not a re-derivation: the warm start has to ramp into
    # the trajectory the report actually contains.
    anchor = rs.executed_anchor(qpos, model, data)
    f_act = np.arange(6, 22)
    w_act = np.arange(6)
    lo, hi = model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]
    return (
        mujoco_mod, model,
        rs._finger_qadr(model),
        np.array([model.jnt_qposadr[rs._joint_id(model, n)]
                  for n in rs.WRIST_JOINTS]),
        f_act, w_act,
        np.clip(anchor["finger_cmd"][0], lo[f_act], hi[f_act]),
        np.clip(anchor["ctrl"][0], lo[w_act], hi[w_act]),
    )


def _run_warm_start():
    mujoco_mod, model, fq, wq, fa, wa, gf, gw = _first_command()
    data = mujoco_mod.MjData(model)
    frame_skip = max(1, int(round(
        1.0 / (rs.CTRL_HZ * model.opt.timestep))))
    rs.reset_scene(model, data, fq, wq, fa, wa)
    info = rs.warm_start(model, data, fq, wq, fa, wa, gf, gw,
                         ctrl_hz=rs.CTRL_HZ, frame_skip=frame_skip)
    return mujoco_mod, model, data, fq, wq, fa, wa, gf, gw, info


# --------------------------------------------------------------------------- W1
def test_w1_warm_start_never_exceeds_the_trackable_rate():
    """No DoF is commanded faster than its servo's measured trackable rate."""
    *_, info = _run_warm_start()
    caps = info["rate_caps"]
    peak = info["commanded_peak_rate"]
    assert peak["wrist_hinge"] <= caps["wrist_hinge_rad_s"] + 1e-9
    assert peak["wrist_slide"] <= caps["wrist_slide_m_s"] + 1e-9
    assert peak["finger"] <= caps["finger_rad_s"] + 1e-9
    # Per axis, not just per class: the W6a re-scale put the x/z slides' own
    # derived winding allowance (45/280 = 0.1607 m/s) below the old measured
    # 0.2 m/s, and a ramp commanding at the stale figure drove the realised
    # rate past the allowance.  The ramp must command each axis at the tighter
    # of its two lines.
    slide_caps = caps["wrist_slide_m_s_per_axis"]
    axis_peak = info["commanded_peak_rate_slide_per_axis"]
    for axis in ("wrist_x", "wrist_y", "wrist_z"):
        assert axis_peak[axis] <= slide_caps[axis] + 1e-9, axis
    xz = (45.0 / 280.0) * (1.0 - rs.WARM_START_SLIDE_TRACK_MARGIN)
    assert slide_caps["wrist_x"] == pytest.approx(xz)
    assert slide_caps["wrist_z"] == pytest.approx(xz)
    assert info["cap_respected"], "the ramp overshot its own cap"


def test_w1b_realised_rates_within_caps():
    """The servos' *realised* rates, as the three-part servo winding gate.

    W1 above holds the *command* side to the measured trackable rate.  The
    realised side is a different quantity and was never boundable by a
    command-side figure: a force-limited position servo can exceed 1.5 rad/s
    while releasing wind-up the ramp accumulated, and 1.5 rad/s was only ever
    measured in free space.  So the realised rate is judged against the servo
    winding tolerance table -- |forcerange| / dof_damping per hinge -- with
    the duration window and the wind-up release explanation mandatory.  An
    excess the DoF's own wind-up cannot account for is red even though it sits
    far inside the absolute cap.
    """
    *_, info = _run_warm_start()
    caps = info["rate_caps"]
    gate = info["servo_winding_rate"]
    assert gate["command_side_cap_rad_s"] == rs.WARM_START_HINGE_RATE_RAD_S
    assert gate["max_frames"] == rs.WINDING_MAX_FRAMES
    assert set(gate["joints"]) == set(rs.WRIST_JOINTS[3:])
    assert gate["ok"], (
        "a wrist hinge's realised rate is outside the servo winding tolerance: "
        + "; ".join(
            f"{n} peak={r['peak_realised_rad_s']:.4f} cap={r['cap_rad_s']:.2f} "
            f"run={r['max_consecutive_intervals']} "
            f"explained={r['explanation_ok']}"
            for n, r in gate["joints"].items() if not r["ok"])
    )
    for name, rec in gate["joints"].items():
        assert rec["cap_rad_s"] == rs.SERVO_WINDING_RATE_CAP_RAD_S[name]
        assert rec["cap_formula"] == "|forcerange| / dof_damping"
        assert rec["dof_damping"] == pytest.approx(1.2)
        assert rec["peak_realised_rad_s"] <= rec["cap_rad_s"], name
        assert rec["max_consecutive_intervals"] <= rs.WINDING_MAX_FRAMES, name
        for ev in rec["events"]:
            assert ev["excess_rad_s"] > 0.0
            # Window semantics: the release account covers the same
            # WINDING_MAX_FRAMES window as the duration clause -- wind-up
            # stored at t-1 fires at t and the DoF coasts at t+1.  A
            # single-interval check would red that coast.
            assert ev["excess_rad_s"] <= ev["release_window_rad_s"], (
                f"{name} interval {ev['interval']}: the excess over the "
                "commanded rate is not this DoF's wind-up releasing"
            )
            assert ev["actuator_saturated"] is False, (
                f"{name} interval {ev['interval']}: an excess recorded while the "
                "actuator is on its force limit is never exempted"
            )
    # the other two classes keep their own caps; the winding table is the
    # wrist-hinge ruling and does not silently loosen the rest.
    realised = info["realised_peak_rate"]
    assert realised["finger"] <= caps["finger_rad_s"] + 1e-9

    # The slides get the same three-part winding gate, on their own derived
    # table -- per axis, read off the W6a dual-hand model: x/z are kp=15000,
    # damping 280 -> 45/280 = 0.1607 m/s, y keeps kp=180, damping 25 ->
    # 45/25 = 1.8 m/s.  (The 0.2 m/s figure is the command cap, w1/w1c -- a
    # force-limited servo can exceed it while releasing wind-up, exactly like
    # the hinges; on the re-scaled x/z servos the derived allowance is now the
    # *tighter* line and the command cap must not lean on it.)
    slide_gate = info["servo_winding_rate_slide"]
    # ``command_side_cap_rad_s`` / ``cap_rad_s`` are the machinery's historical
    # key names; for the slide table they hold m/s.  ``cap_units`` asserts it.
    assert slide_gate["command_side_cap_rad_s"] == rs.WARM_START_SLIDE_RATE_M_S
    assert gate["max_frames"] == rs.WINDING_MAX_FRAMES
    assert set(slide_gate["joints"]) == set(rs.WRIST_JOINTS[:3])
    assert slide_gate["ok"], (
        "a wrist slide's realised rate is outside the servo winding tolerance: "
        + "; ".join(
            f"{n} peak={r['peak_realised_rad_s']:.4f} cap={r['cap_rad_s']:.2f} "
            f"run={r['max_consecutive_intervals']} "
            f"explained={r['explanation_ok']}"
            for n, r in slide_gate["joints"].items() if not r["ok"])
    )
    expected_slide_damping = {"wrist_x": 280.0, "wrist_y": 25.0, "wrist_z": 280.0}
    for name, rec in slide_gate["joints"].items():
        assert rec["cap_units"] == "m/s"
        assert rec["cap_rad_s"] == rs.SERVO_WINDING_RATE_CAP_M_S[name]
        assert rec["cap_formula"] == "|forcerange| / dof_damping"
        assert rec["dof_damping"] == pytest.approx(expected_slide_damping[name])
        assert rec["peak_realised_rad_s"] <= rec["cap_rad_s"], name
        assert rec["max_consecutive_intervals"] <= rs.WINDING_MAX_FRAMES, name
        for ev in rec["events"]:
            assert ev["excess_rad_s"] > 0.0
            # Window semantics: the release account covers the same
            # WINDING_MAX_FRAMES window as the duration clause -- wind-up
            # stored at t-1 fires at t and the DoF coasts at t+1.  A
            # single-interval check would red that coast.
            assert ev["excess_rad_s"] <= ev["release_window_rad_s"], (
                f"{name} interval {ev['interval']}: the excess over the "
                "commanded rate is not this DoF's wind-up releasing"
            )
            assert ev["actuator_saturated"] is False, (
                f"{name} interval {ev['interval']}: an excess recorded while the "
                "actuator is on its force limit is never exempted"
            )


def test_w1c_caps_are_the_measured_trackable_rates():
    """The caps are the free-space measurements, not round numbers picked to pass."""
    assert rs.WARM_START_HINGE_RATE_RAD_S == 1.5
    assert rs.WARM_START_SLIDE_RATE_M_S == 0.2
    assert rs.WARM_START_FINGER_RATE_RAD_S == rs.JOINT_VEL_LIMIT_RAD_S
    # the re-scaled x/z slides are commanded just below their own derived
    # allowance -- the 1% tracking headroom, not the stale measured figure
    assert rs.WARM_START_SLIDE_TRACK_MARGIN == 1e-2
    assert rs.WARM_START_SLIDE_RATE_XZ_M_S == pytest.approx(
        (45.0 / 280.0) * (1.0 - rs.WARM_START_SLIDE_TRACK_MARGIN))
    assert rs.WARM_START_SLIDE_RATE_XZ_M_S < rs.SERVO_WINDING_RATE_CAP_M_S["wrist_x"]
    assert rs.WARM_START_SLIDE_RATE_XZ_M_S == min(
        rs.WARM_START_SLIDE_RATE_M_S, rs.SERVO_WINDING_RATE_CAP_M_S["wrist_x"]
    ) * (1.0 - rs.WARM_START_SLIDE_TRACK_MARGIN)
    # the realised-rate ceiling is a separate, derived table; the 1.5 rad/s
    # command-side figure is not it and must not be reused as one.  On the W6a
    # dual-hand scene the tables carry both hands (6 hinges, 6 slides); the
    # warm start only ever gates the right hand's, which the left-hand keys
    # sitting in the tables cannot shortcut.
    assert rs.SERVO_WINDING_RATE_CAP_RAD_S == {
        "wrist_rx": 5.0, "wrist_ry": 5.0, "wrist_rz": 5.0,
        "wrist_rx_left": 5.0, "wrist_ry_left": 5.0, "wrist_rz_left": 5.0}
    assert rs.SERVO_WINDING_RATE_CAP_M_S == {
        "wrist_x": 45.0 / 280.0, "wrist_y": 45.0 / 25.0,
        "wrist_z": 45.0 / 280.0,
        "wrist_x_left": 45.0 / 280.0, "wrist_y_left": 45.0 / 25.0,
        "wrist_z_left": 45.0 / 280.0}
    if REPORT_PATH.exists():
        rep = json.loads(REPORT_PATH.read_text())
        ceil = rep.get("rerun_evidence", {}).get(
            "velocity_gate", {}).get("floating_wrist_servo_ceiling")
        if ceil:
            assert ceil["highest_tracked_command_rate_rad_s"] >= \
                rs.WARM_START_HINGE_RATE_RAD_S


# --------------------------------------------------------------------------- W2
def test_w2_ramp_does_not_move_the_hand_or_the_box():
    """The warm-started pose must match a straight-to-target reference.

    Both arms get the same settle budget so what is compared is the *path*, not
    how long each was given.  The reference is deliberately naive -- ctrl set to
    the first command on frame 1 -- because that is what the warm start has to
    be indistinguishable from.
    """
    mujoco_mod, model, fq, wq, fa, wa, gf, gw = _first_command()
    frame_skip = max(1, int(round(1.0 / (rs.CTRL_HZ * model.opt.timestep))))

    warm = mujoco_mod.MjData(model)
    rs.reset_scene(model, warm, fq, wq, fa, wa)
    rs.warm_start(model, warm, fq, wq, fa, wa, gf, gw,
                  ctrl_hz=rs.CTRL_HZ, frame_skip=frame_skip)

    ref = mujoco_mod.MjData(model)
    rs.reset_scene(model, ref, fq, wq, fa, wa)
    ref.ctrl[fa] = gf
    ref.ctrl[wa] = gw
    for _ in range(rs.WARM_START_SETTLE_STEPS):
        for _ in range(frame_skip):
            mujoco_mod.mj_step(model, ref)

    mujoco_mod.mj_forward(model, warm)
    mujoco_mod.mj_forward(model, ref)
    palm = mujoco_mod.mj_name2id(model, mujoco_mod.mjtObj.mjOBJ_BODY, "palm")
    d_pos = np.abs(warm.xpos[palm] - ref.xpos[palm])
    d_rot = float(np.abs(warm.xmat[palm] - ref.xmat[palm]).max())
    d_box = float(np.abs(warm.xpos[1] - ref.xpos[1]).max())

    # A 30 mm / 0.2 rad / 10 mm envelope is ~100x the hand's own contact
    # deflection at this pose and ~1000x the ramp's command quantisation, so a
    # ramp that relocated the hand or nudged the box cannot hide inside it.
    assert d_pos.max() < 0.030, f"ramp moved the hand: {d_pos} m"
    assert d_rot < 0.2, f"ramp reoriented the hand: {d_rot}"
    assert d_box < 0.010, f"ramp moved the box: {d_box} m"


def test_w2b_warm_start_writes_no_qpos():
    """The ramp is input shaping: like replay(), it may only write data.ctrl."""
    import ast

    tree = ast.parse((REPO / "retarget/retarget_smoke.py").read_text())
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "warm_start")
    writes = []
    for node in ast.walk(fn):
        targets = []
        if isinstance(node, ast.Assign):
            targets = [t for t in node.targets]
        elif isinstance(node, ast.AugAssign):
            targets = [node.target]
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                and node.func.attr in ("mju_", "mj_"):
            continue
        for tgt in targets:
            for sub in ast.walk(tgt):
                if isinstance(sub, ast.Attribute) and sub.attr in ("qpos", "qvel"):
                    writes.append((sub.attr, node.lineno))
    assert writes == [], f"warm_start assigns data state: {writes}"


# ------------------------------------------------------- the recorded finding
def test_w2c_first_command_reachability_is_recorded_per_dof():
    """Record, do not hide: which DoFs the ramp cannot deliver, and why.

    On the superseded single-hand scene the finding was the wrist: the palm
    rested on the box top and no approach path reached the commanded descent.
    On the W6a dual-hand scene the re-scaled wrist slides deliver the
    commanded pose; the residual tracking error sits on the finger joints,
    which are in sustained box contact at the first command -- a
    contact-deflection shortfall, not an unreachable command.  Whichever way
    a scene falls, ``reachable`` and ``unreachable_dofs`` must agree and the
    report must name the DoFs it could not deliver.
    """
    *_, info = _run_warm_start()
    track = info["tracking_error_to_goal"]
    assert bool(info["reachable"]) == (not info["unreachable_dofs"]), (
        "the reachable flag and the named DoF list must agree")
    assert set(info["unreachable_dofs"]) <= (
        set(rs.MJCF_FINGER_JOINTS) | set(rs.WRIST_JOINTS))
    if info["reachable"]:
        assert track["wrist_slide"] <= rs.WARM_START_REACHABLE_TOL
        assert track["wrist_hinge"] <= rs.WARM_START_REACHABLE_TOL
    else:
        assert info["unreachable_dofs"], "unreachable but no DoF named"
        assert "reachable set" in info["unreachable_note"]
        wrist_missing = [n for n in info["unreachable_dofs"]
                         if n in rs.WRIST_JOINTS]
        if wrist_missing:
            # the original single-hand finding, restated: a wrist DoF that
            # misses means the commanded pose itself is out of reach
            assert (track["wrist_slide"] > rs.WARM_START_REACHABLE_TOL
                    or track["wrist_hinge"] > rs.WARM_START_REACHABLE_TOL)
            assert "reachable set" in info["wrist_reachable_note"]
        else:
            # the dual-hand finding: the wrist delivers; what misses is a
            # finger holding its contact deflection -- recorded, not folded
            # into a wrist claim
            assert track["wrist_slide"] <= rs.WARM_START_REACHABLE_TOL
            assert track["wrist_hinge"] <= rs.WARM_START_REACHABLE_TOL
            assert all(n in rs.MJCF_FINGER_JOINTS
                       for n in info["unreachable_dofs"])


def test_w2d_warm_start_is_reported_in_the_artefact():
    """Duration and peak rate are recorded, as the wave-3 ruling requires."""
    rep = json.loads(REPORT_PATH.read_text())
    w = rep["replay"]["warm_start"]
    for key in ("control_steps", "ramp_duration_s", "duration_s",
                "commanded_peak_rate", "realised_peak_rate",
                "reachable", "tracking_error_to_goal"):
        assert key in w, f"warm start artefact missing {key}"
    assert w["ramp_control_steps"] > 0
    assert w["commanded_peak_rate"]["wrist_hinge"] <= \
        rs.WARM_START_HINGE_RATE_RAD_S + 1e-9
    assert rep["mjcf_commit"]["mjcf_commit"], "artefact must name its MJCF commit"
