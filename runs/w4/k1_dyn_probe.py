"""W4-B dynamic probe - can the hand HOLD the two-wall pad pinch the K1 scan found?

The kinematic gate only says a pose EXISTS in which two fingertip pads touch the
two side walls within tolerance with nothing buried. Whether that pose is a
*grasp* is the frozen acceptance gates' question, and this script asks it with
the frozen code: every number below comes out of
``tests/test_smoke4_accept._analyse_episode`` - the exact function the A1-A5
tests run - fed by a scripted trajectory through the ordinary env step path.

Trajectory (world-frame control, correct polarity, through env.step):
  transit  ramp the wrist setpoint and the absolute finger setpoints to the K1
           pose; the wrist action is a DELTA (see _apply_action), so the ramp is
           an error-driven delta clipped to the actuator range
  press    hold the wrist, curl the contacting fingers further until the pads
           load up (or the ctrlrange stops them)
  lift     raise the wrist z setpoint while holding the finger setpoints

Outputs runs/w4/k1_dyn_probe.json with the per-phase A1-A5 numbers and the
bearing decomposition (hook / side-wall / support).

    runs/.venv-mj/bin/python runs/w4/k1_dyn_probe.py --pedestal-mm 60
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from grounding.envs import lift_hold_env as lhe  # noqa: E402

RUNS = pathlib.Path(__file__).resolve().parent
SCAN = RUNS / "pedestal_scan_kin.json"
OUT = RUNS / "k1_dyn_probe.json"
ASSETS = REPO / "grounding/assets"
XML = ASSETS / "allegro_floating.xml"

N_TRANSIT = 70
N_PRESS = 60
N_LIFT = 90
N_HOLD = 30
PRESS_INCREMENT_RAD = 0.03     # per control step, added to every finger joint


def scene_path(pedestal_mm: float) -> pathlib.Path | None:
    """A transient variant XML for a non-committed pedestal height."""
    if abs(pedestal_mm - 60.0) < 1e-9:
        return None            # the committed scene
    text = XML.read_text()
    old = ('<geom name="pedestal" type="box" size="0.30 0.30 0.0300" '
           'pos="0 0 0.7300"')
    assert old in text, "pedestal line not found - the probe must patch the real scene"
    h = pedestal_mm / 1e3
    if h <= 0.0:
        # no pedestal: park the geom far below the table (a zero-size box geom is
        # rejected by the compiler and would generate degenerate contacts anyway)
        new = old.replace("0 0 0.7300", "0 0 -5.0000")
    else:
        new = old.replace("0.30 0.30 0.0300", f"0.30 0.30 {h / 2:.4f}") \
                  .replace("0 0 0.7300", f"0 0 {0.70 + h / 2:.4f}")
    tmp = ASSETS / "_w4_dyn_variant.xml"
    tmp.write_text(text.replace(old, new))
    return tmp


def ctrl_to_finger_action(env, ctrl: np.ndarray) -> np.ndarray:
    """Invert _apply_action's absolute finger mapping."""
    lo, hi = env.unwrapped.ctrl_low[6:], env.unwrapped.ctrl_high[6:]
    return np.clip(2.0 * (ctrl[6:] - lo) / np.maximum(hi - lo, 1e-12) - 1.0, -1.0, 1.0)


def run(pedestal_mm: float, pose: np.ndarray) -> dict:
    tmp = scene_path(pedestal_mm)
    if tmp is not None:
        lhe.MJCF = tmp
    try:
        env = lhe.LiftHoldEnv(domain_rand=False, episode_len=400)
        u = env.unwrapped
        u.record_bearing = True
        env.reset(seed=0)
        target = pose.copy()
        target[:3] = np.clip(target[:3], u.ctrl_low[:3], u.ctrl_high[:3])
        target[3:6] = np.clip(target[3:6], u.ctrl_low[3:6], u.ctrl_high[3:6])
        target[6:] = np.clip(target[6:], u.ctrl_low[6:], u.ctrl_high[6:])
        phases = []

        def hold(n: int, name: str, finger_target=None, wrist_target=None):
            for _ in range(n):
                a = np.zeros(22)
                w_t = u.data.ctrl[:6].copy() if wrist_target is None else wrist_target[:6]
                f_t = u.data.ctrl[6:].copy() if finger_target is None else finger_target
                # wrist DELTA control, error-driven, one sixth of the error per step
                a[:3] = np.clip((w_t[:3] - u.data.ctrl[:3]) * u.frame_skip / 0.012 / 6.0,
                                -1.0, 1.0)
                a[3:6] = np.clip((w_t[3:] - u.data.ctrl[3:6]) / 0.10 / 6.0, -1.0, 1.0)
                a[6:] = ctrl_to_finger_action(env, np.concatenate(
                    [u.data.ctrl[:6], f_t]))
                env.step(a.astype(np.float32))
            phases.append((name, n, float(u.data.qpos[u.obj_qpos_adr + 2]),
                           float(u.last_contact_force)))

        # The K1 pose is an extreme wrist posture (rz up to +-1.4 rad). Ramping
        # everything at once sweeps the hand THROUGH the box; the approach has to
        # be: retract the fingers, put the wrist where the grasp is, THEN close.
        def ramp(finger_from: np.ndarray, finger_to: np.ndarray,
                 wrist_from: np.ndarray, wrist_to: np.ndarray, n: int,
                 name: str) -> np.ndarray:
            for k in range(1, n + 1):
                s_ = k / n
                goal_w = wrist_from + (wrist_to - wrist_from) * s_
                goal_f = finger_from + (finger_to - finger_from) * s_
                hold(1, f"{name}[{k}]", finger_target=goal_f, wrist_target=goal_w)
            return u.data.ctrl[6:].copy()

        ctrl0 = u.data.ctrl.copy()
        open_f = u.ctrl_low[6:] + 0.12 * (u.ctrl_high[6:] - u.ctrl_low[6:])
        ramp(open_f, open_f, ctrl0[:6], target[:6], N_TRANSIT, "approach")
        ramp(open_f, target[6:], target[:6], target[:6], N_PRESS, "close")
        # press: curl further, in the direction each finger was already going
        press_dir = np.sign(target[6:] - open_f)
        press_dir[press_dir == 0] = 1.0
        f_t = u.data.ctrl[6:].copy()
        for _ in range(N_PRESS):
            f_t = np.clip(f_t + press_dir * PRESS_INCREMENT_RAD,
                          u.ctrl_low[6:], u.ctrl_high[6:])
            hold(1, "press", finger_target=f_t)
        # lift: raise the wrist z setpoint, hold everything else
        lift_top = min(u.ctrl_high[2], u.data.ctrl[2] + 0.16)
        z0 = u.data.ctrl[2]
        for k in range(1, N_LIFT + 1):
            goal = u.data.ctrl[:6].copy()
            goal[2] = z0 + (lift_top - z0) * k / N_LIFT
            hold(1, "lift", finger_target=f_t, wrist_target=goal)
        hold(N_HOLD, "hold", finger_target=f_t)

        sys.path.insert(0, str(REPO / "tests"))
        import test_smoke4_accept as acc  # noqa: PLC0415

        ep = {"trace": u.trace, "episode": 0, "seed": 0}
        ev = acc._analyse_episode(ep)  # noqa: SLF001
        tr = u.trace
        res = {
            "pedestal_mm": pedestal_mm,
            "target_pose": [float(v) for v in target],
            "ctrl_reached": [float(v) for v in u.data.ctrl],
            "phase_trail": phases[-6:],
            "A_evidence": {k: ev[k] for k in (
                "peak_lift_m", "peak_step", "n_finger_contact_steps", "finger_pairs",
                "finger_force_duty_cycle", "finger_force_mean_N", "box_weight_N",
                "force_duty_ok", "force_sufficiency_ok",
                "max_anchored_step_m", "max_free_fall_gravity_error",
                "separation_contact_driven", "anchored_step_ok",
                "free_fall_gravity_ok", "table_touching_at_peak", "peak_force_N")},
            "A1_peak_lift_gt_0p05": bool(ev["peak_lift_m"] > 0.05),
            "A2_finger_box_pair": bool(ev["n_finger_contact_steps"] > 0),
            "A3a_force": bool(ev["force_duty_ok"] and ev["force_sufficiency_ok"]),
            "A3c_no_injection": bool(ev["anchored_step_ok"] and ev["free_fall_gravity_ok"]
                                     and ev["separation_contact_driven"]),
            "A3b_support_borne_peak_N": float(max(tr.get("support_force", [0.0]))),
            "bearing_peak_N": {
                "hook": float(max(tr.get("hook_up_N_peak", [-1e9]))),
                "wall": float(max(tr.get("wall_up_N_peak", [-1e9]))),
                "support": float(max(tr.get("pedestal_up_N_peak", [-1e9]))),
            },
            "final_obj_z_m": float(u.data.xpos[u.obj_body][2]),
            "z0_m": float(u._z0),
            "max_lift_m": float(max(tr["lift"])),
        }
        env.close()
        return res
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)
            lhe.MJCF = XML


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pedestal-mm", type=float, default=60.0)
    args = ap.parse_args()
    scan = json.loads(SCAN.read_text())
    cell = [c for c in scan["cells"]
            if abs(c["pedestal_mm"] - args.pedestal_mm) < 1e-6][0]
    pose = np.array(cell["best_pose"], dtype=float)
    res = run(args.pedestal_mm, pose)
    OUT.write_text(json.dumps(res, indent=2))
    ev = res["A_evidence"]
    print(f"pedestal {args.pedestal_mm:.0f} mm")
    print(f"  A1 peak lift      {ev['peak_lift_m'] * 1e3:8.2f} mm   (>50 mm? "
          f"{res['A1_peak_lift_gt_0p05']})")
    print(f"  A2 finger steps   {ev['n_finger_contact_steps']:8d}   pairs {ev['finger_pairs']}")
    print(f"  A3a duty/mean     {ev['finger_force_duty_cycle']:.3f} / "
          f"{ev['finger_force_mean_N']:.3f} N (>= {ev['box_weight_N']:.4f}) "
          f"-> {res['A3a_force']}")
    print(f"  A3b support peak  {res['A3b_support_borne_peak_N']:8.2f} N")
    print(f"  A3c anchored/grav {ev['max_anchored_step_m'] * 1e3:.3f} mm / "
          f"{ev['max_free_fall_gravity_error']:.4f} -> {res['A3c_no_injection']}")
    print(f"  bearing peaks     {res['bearing_peak_N']}")
    print(f"  final obj z       {res['final_obj_z_m']:.4f} m (z0 {res['z0_m']:.4f})")
    print(f"  wrote {OUT}")


if __name__ == "__main__":
    main()
