"""W4/R8 - dynamic check: does a scripted lip hook clear the FROZEN acceptance gates?

The kinematic scan (runs/w4/lip_kin_scan.py) answers "can a fingertip get to the
hook face". This answers the only question that matters: does hooking it produce a
gate-clean lift, judged by tests/test_smoke4_accept.py's own machinery imported and
called verbatim - A1 (lift > 5 cm), A2 (>= 5 steps of finger<->box contact),
A3a (finger-force duty >= 0.5 AND mean force >= the box weight), A3b (not touching
its support at peak), A3c (anchored step < 10 mm, free-fall gravity consistency,
contact-driven separation) - plus the A3b spirit this repo added on top: the peak
must not be carried by the pedestal.

Nothing here relaxes a threshold. `gate_eval` is `grounding.reachable_scan`'s, which
calls `test_smoke4_accept._analyse_episode` itself, so a scan pass IS a test pass by
construction. The scene is varied, never the gate.

Trajectory shape (world-frame wrist P-loop, correct polarity, fingers commanded as
absolute setpoints - the same conventions probe_clean_lift and reachable_scan fix):

  transit   lift the hand clear, move xy            (never sweep the palm past the box)
  descend   come down beside the box to just above the lip
  curl      RAMPED curl, so a pad arrives under the lip instead of impacting it
  settle    hold, letting the contact build
  lift      raise the wrist target at a rate limit, fingers held, and read the gates

Run:  runs/.venv-mj/bin/python runs/w4/lip_dyn_scan.py [--cells o,h ...]
Out:  runs/w4/lip_dyn_scan.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from grounding.assets import build_allegro_floating as baf  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parent / "lip_dyn_scan.json"
KIN = pathlib.Path(__file__).resolve().parent / "lip_kin_scan.json"
ASSETS = REPO / "grounding/assets"

N_WRIST = 6
WRIST_ROT_STEP = 0.10
WORLD_GAIN = np.array([0.012, 0.012, 0.00436])   # measured, probe_clean_lift

N_TRANSIT = 40
N_DESC = 60
N_CLOSE = 90
EPISODE_LEN = 400


def action(env, wrist_tgt, rot_tgt, fingers):
    """World-frame wrist P-loop + absolute finger setpoints, in action units."""
    err = np.asarray(wrist_tgt) - env.data.xpos[env.hand_root]
    a = np.zeros(22)
    a[0:3] = np.clip(err / WORLD_GAIN, -1.0, 1.0)
    a[3:6] = np.clip((np.asarray(rot_tgt) - env.data.ctrl[3:6]) / WRIST_ROT_STEP, -1.0, 1.0)
    a[N_WRIST:] = np.clip(np.asarray(fingers), -1.0, 1.0)
    return a


def close_amp(cl: float, ct: float, t0: float = 0.0) -> np.ndarray:
    """16 finger action units; long fingers tied, thumb independent."""
    a = np.full(16, 0.6)
    a[0:3] = cl * 0.3
    a[4:7] = cl
    a[8:11] = cl
    a[12] = t0
    a[13:16] = ct
    return a


def run_traj(env, cfg: dict):
    """One scripted attempt. Deterministic: reset(seed=0), a pure function of cfg."""
    env.reset(seed=0)
    w = list(cfg["wrist_world"])
    rot = cfg["rot"]
    close = np.asarray(cfg["close"])
    open_a = np.full(16, 0.6)
    lift_z, lift_rate = cfg["lift_z"], cfg["lift_rate"]
    n_close = int(cfg.get("n_close", N_CLOSE))
    settle = int(cfg.get("settle", 25))
    lift_hold = int(cfg.get("lift_hold", 60))

    def step_to(tgt, fingers_cmd):
        env.step(action(env, tgt, rot, fingers_cmd))

    # transit: hand high, then move xy, so the palm never sweeps the box
    high = [w[0], w[1], w[2] + 0.12]
    for _ in range(N_TRANSIT):
        step_to(high, open_a)

    for _ in range(int(cfg.get("n_desc", N_DESC))):
        step_to(w, open_a)

    # ramped curl: a straight jump to full curl impacts the box instead of
    # sliding a pad under the lip (measured - the impact knocks it away)
    for i in range(n_close):
        f = open_a + (close - open_a) * min(1.0, (i + 1) / (n_close * 0.7))
        step_to(w, f)
    for _ in range(n_close - int(n_close * 0.7)):
        step_to(w, close)
    for _ in range(settle):
        step_to(w, close)

    # lift at a rate-limited world z target, fingers held
    cur = float(env.data.xpos[env.hand_root][2])
    for _ in range(int(np.ceil(lift_z / lift_rate)) + lift_hold):
        goal = min(w[2] + lift_z, cur + lift_rate)
        step_to([w[0], w[1], goal], close)
        cur = float(env.data.xpos[env.hand_root][2])


def bearing_at_peak(env) -> dict:
    """A3b evidence: how the box's weight was carried at peak lift."""
    tr = env.trace
    lift = np.asarray(tr["lift"], dtype=float)
    if lift.size == 0:
        return {}
    i = int(np.argmax(lift))
    return {
        "peak_step": i,
        "peak_lift_m": float(lift[i]),
        "hook_up_N": float(tr["hook_up_N"][i]),
        "wall_up_N": float(tr["wall_up_N"][i]),
        "pedestal_up_N": float(tr["pedestal_up_N"][i]),
        "support_force_N_at_peak": float(tr["support_force"][i]),
        "box_weight_N": baf.BOX_MASS_KG * 9.81,
        "support_box_weights_at_peak": float(tr["support_force"][i]) / (baf.BOX_MASS_KG * 9.81),
    }


def make_env_for(mjcf_path: pathlib.Path):
    """Env bound to a specific MJCF, without touching the committed file's role.

    `MJCF` is a module constant read inside `LiftHoldEnv.__init__`, so patching it
    for the duration of construction is enough - and restoring it afterwards keeps
    every other module that imports it (train_ppo, the reports, the tests) reading
    the shipped scene.
    """
    import grounding.envs.lift_hold_env as env_mod
    saved = env_mod.MJCF
    env_mod.MJCF = mjcf_path
    try:
        env = env_mod.LiftHoldEnv(episode_len=EPISODE_LEN, domain_rand=False)
    finally:
        env_mod.MJCF = saved
    env.record_bearing = True
    return env


def evaluate(mjcf_path: pathlib.Path, cfg: dict) -> dict:
    from grounding.reachable_scan import gate_eval
    env = make_env_for(mjcf_path)
    try:
        run_traj(env, cfg)
        ev = gate_eval(env)
        row = {
            "cfg": {k: (list(v) if isinstance(v, (list, tuple)) else v)
                    for k, v in cfg.items()},
            "gate_pass": bool(ev["gate_pass"]),
            "peak_lift_m": ev["peak_lift_m"],
            "n_finger_contact_steps": ev["n_finger_contact_steps"],
            "finger_force_duty_cycle": ev["finger_force_duty_cycle"],
            "finger_force_mean_N": ev["finger_force_mean_N"],
            "box_weight_N": ev["box_weight_N"],
            "table_touching_at_peak": ev["table_touching_at_peak"],
            "max_anchored_step_m": ev["max_anchored_step_m"],
            "max_free_fall_gravity_error": ev["max_free_fall_gravity_error"],
            "separation_contact_driven": ev["separation_contact_driven"],
            "per_gate": {
                "A1_lift_gt_5cm": ev["peak_lift_m"] > 0.05,
                "A2_finger_contact_ge_5": ev["n_finger_contact_steps"] >= 5,
                "A3a_duty": ev["force_duty_ok"],
                "A3a_sufficiency": ev["force_sufficiency_ok"],
                "A3b_airborne_at_peak": not ev["table_touching_at_peak"],
                "A3c_anchored_step": ev["anchored_step_ok"],
                "A3c_free_fall_gravity": ev["free_fall_gravity_ok"],
                "A3c_separation": ev["separation_ok"],
            },
            "bearing": bearing_at_peak(env),
            "finger_pairs": ev["finger_pairs"],
        }
        return row
    finally:
        env.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="*", default=None,
                    help="overhang_mm,lip_h_mm cells to run; default = every "
                         "kinematically clean cell")
    ap.add_argument("--all-cells", action="store_true",
                    help="run every cell in the sweep, clean or not")
    ap.add_argument("--control", action="store_true",
                    help="also run the bare-box control at the same trajectories")
    ap.add_argument("--traj-limit", type=int, default=0,
                    help="cap trajectories per cell (0 = no cap)")
    args = ap.parse_args()

    kin = json.loads(KIN.read_text())
    if args.cells:
        want = {tuple(float(v) for v in c.split(",")) for c in args.cells}
        cells = [c for c in kin["cells"]
                 if (c["overhang_mm"], c["lip_h_mm"]) in want]
    elif args.all_cells:
        cells = list(kin["cells"])
    else:
        cells = [c for c in kin["cells"] if c["clean_hook"]]

    if not cells:
        raise SystemExit("no cells selected - run lip_kin_scan.py first")

    # trajectory grid. Deliberately spans the hook family plus its neighbours, so a
    # FAIL is reported as "no trajectory in this family clears the gates" rather
    # than "this one trajectory did not".
    xs = (-0.03, -0.02, -0.01, 0.0, 0.01)
    ys = (-0.06, -0.03, 0.0, 0.03, 0.06)
    zs = (0.79, 0.805, 0.82, 0.835, 0.85)
    rxs = (1.4, 1.7, 2.0)
    rys = (0.3, 0.6, 0.9)
    cts = (-0.3, -0.6, -0.9)
    cls = (0.4, 0.8)
    lifts = (0.16,)

    results = []
    for cell in cells:
        oh, lh = cell["overhang_mm"] * 1e-3, cell["lip_h_mm"] * 1e-3
        name = f"o{int(round(oh * 1e3)):02d}_h{int(round(lh * 1e3)):02d}"
        path = baf.build(oh, lh, ASSETS / f"_dyn_{name}.xml")
        n_traj = 0
        n_pass = 0
        best = None
        passes = []
        for x in xs:
            for y in ys:
                for z in zs:
                    for rx in rxs:
                        for ry in rys:
                            for cl in cls:
                                for ct in cts:
                                    cfg = {
                                        "wrist_world": [x, y, z],
                                        "rot": [rx, ry, 0.0],
                                        "close": close_amp(cl, ct).tolist(),
                                        "lift_z": lifts[0],
                                        "lift_rate": 0.004,
                                        "n_close": N_CLOSE,
                                        "settle": 25,
                                        "lift_hold": 60,
                                    }
                                    row = evaluate(path, cfg)
                                    n_traj += 1
                                    if row["gate_pass"]:
                                        n_pass += 1
                                        passes.append(row)
                                    if best is None or row["peak_lift_m"] > best["peak_lift_m"]:
                                        best = row
                                    if args.traj_limit and n_traj >= args.traj_limit:
                                        break
                                if args.traj_limit and n_traj >= args.traj_limit:
                                    break
                            if args.traj_limit and n_traj >= args.traj_limit:
                                break
                        if args.traj_limit and n_traj >= args.traj_limit:
                            break
                    if args.traj_limit and n_traj >= args.traj_limit:
                        break
                if args.traj_limit and n_traj >= args.traj_limit:
                    break
        results.append({
            "overhang_mm": cell["overhang_mm"],
            "lip_h_mm": cell["lip_h_mm"],
            "kin_clean_hook": cell["clean_hook"],
            "n_traj": n_traj,
            "n_pass": n_pass,
            "pass_rate": n_pass / n_traj if n_traj else 0.0,
            "best_row": best,
            "passing_rows": passes[:5],
        })
        print(f"o={cell['overhang_mm']:4.1f} h={cell['lip_h_mm']:4.1f}  "
              f"{n_pass}/{n_traj} gate-clean  "
              f"best_lift={best['peak_lift_m'] * 1000:.1f}mm  "
              f"ped@peak={best['bearing'].get('support_box_weights_at_peak', float('nan')):.2f}bw",
              flush=True)
        path.unlink(missing_ok=True)

    control = None
    if args.control:
        path = baf.build(0.0, 0.0, ASSETS / "_dyn_control.xml")
        rows = []
        for x in xs:
            for y in ys:
                for z in zs:
                    cfg = {"wrist_world": [x, y, z], "rot": [1.7, 0.6, 0.0],
                           "close": close_amp(0.8, -0.6).tolist(),
                           "lift_z": 0.16, "lift_rate": 0.004,
                           "n_close": N_CLOSE, "settle": 25, "lift_hold": 60}
                    rows.append(evaluate(path, cfg))
        control = {
            "n_traj": len(rows),
            "n_pass": sum(r["gate_pass"] for r in rows),
            "best_peak_lift_m": max(r["peak_lift_m"] for r in rows),
            "note": "bare box, same trajectory family - the control for the sweep",
        }
        path.unlink(missing_ok=True)
        print(f"CONTROL bare box: {control['n_pass']}/{control['n_traj']} gate-clean, "
              f"best {control['best_peak_lift_m'] * 1000:.1f} mm")

    payload = {
        "question": "does a scripted lip hook clear the FROZEN smoke4 acceptance "
                    "gates (A1-A5 verbatim from tests/test_smoke4_accept.py)?",
        "gate_source": "grounding.reachable_scan.gate_eval -> "
                       "test_smoke4_accept._analyse_episode, unmodified",
        "cells": results,
        "n_cells": len(results),
        "n_cells_with_any_pass": sum(1 for r in results if r["n_pass"] > 0),
        "control_bare_box": control,
    }
    OUT.write_text(json.dumps(payload, indent=2))
    print(f"\ncells with >=1 gate-clean trajectory: "
          f"{payload['n_cells_with_any_pass']}/{len(results)}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
