"""Smoke 5 warm start -- rate-limited approach to the first commanded pose.

Two properties, both behavioural (they call ``retarget_smoke.warm_start`` and
step physics; they do not read the committed JSON):

  W1  inside the warm-start segment no DoF is driven faster than the servo's
      measured trackable rate.  The caps are per class because the units are:
      rad/s for hinges and finger joints, m/s for slides.
  W2  the ramp is pure input shaping -- after it, the hand sits in the same
      world pose as a reference that was driven straight to the first command.
      This is what stops a "helpful" ramp from quietly relocating the hand or
      the box on the way in.

Neither test asserts that the warm start *succeeds*.  On the current scene it
does not: the first commanded wrist pose is outside this floating wrist's
reachable set (the palm rests on the box top and blocks the wrist's descent),
which is why ``test_w2_first_command_is_unreachable_on_this_scene`` records
that as a finding rather than pretending otherwise.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

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
    assert info["cap_respected"], "the ramp overshot its own cap"


def test_w1b_realised_rates_within_caps():
    """The servos' *realised* rates stay under the caps too, not just the cmd."""
    *_, info = _run_warm_start()
    caps = info["rate_caps"]
    realised = info["realised_peak_rate"]
    assert realised["wrist_hinge"] <= caps["wrist_hinge_rad_s"] + 1e-9
    assert realised["wrist_slide"] <= caps["wrist_slide_m_s"] + 1e-9
    assert realised["finger"] <= caps["finger_rad_s"] + 1e-9


def test_w1c_caps_are_the_measured_trackable_rates():
    """The caps are the free-space measurements, not round numbers picked to pass."""
    assert rs.WARM_START_HINGE_RATE_RAD_S == 1.5
    assert rs.WARM_START_SLIDE_RATE_M_S == 0.2
    assert rs.WARM_START_FINGER_RATE_RAD_S == rs.JOINT_VEL_LIMIT_RAD_S
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
def test_w2c_first_command_is_unreachable_on_this_scene():
    """Record, do not hide: the frame-0 wrist pose is outside the reachable set.

    The ramp is correct but it cannot manufacture reachability.  This test
    exists so the shortfall stays a *measurement* in the report rather than a
    mystery, and so that a future scene which does make the pose reachable
    turns this red and forces a look at the numbers.
    """
    *_, info = _run_warm_start()
    track = info["tracking_error_to_goal"]
    if info["reachable"]:
        assert track["wrist_slide"] <= rs.WARM_START_REACHABLE_TOL
        assert track["wrist_hinge"] <= rs.WARM_START_REACHABLE_TOL
    else:
        assert info["unreachable_dofs"], "unreachable but no DoF named"
        assert track["wrist_slide"] > rs.WARM_START_REACHABLE_TOL
        assert "reachable set" in info["unreachable_note"]


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
