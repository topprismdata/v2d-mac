"""Adjudicate SceneInitFix's K1 seeds under this round's own frozen gates.

Division of labour (Main ruling). SceneInitFix answered "can the K1 dual-wall
pad pinch actually LIFT?" by searching the dynamics - press depth, closing
timing, lift curve - for the four poses it found. This module answers the
complementary question and does NOT redo that search: it takes each of those
poses as a FORCED SEED and asks what the frozen smoke4 acceptance gates say
about it under one fixed, fully disclosed scripted driver.

Why forcing them in matters: my own limit-covering grid is a 5x5x5x5x5x5 lattice.
A pose found by random-restart refinement can fall between lattice points, and
"my grid found nothing" would then be a statement about my lattice again - the
same class of error as the w35d verdict this round overturned. These four poses
are exactly the evidence that the lattice is not the whole story, so they belong
in the gate record whether or not my grid reaches them.

The 60 mm seed is the one on the CURRENT scene (box top 0.84 m). The other three
come from 0/20/40 mm pedestal variants whose box rest heights differ; they are
still judged, and a failure there is reported as "this seed does not transfer to
the current box height" rather than silently dropped.

    python grounding/k1_seed_gate.py
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

RUN_DIR = REPO / "runs/p0/smoke4_mujoco_ppo"
W4_JSON = REPO / "runs/w4/pedestal_scan_kin.json"
OUT = RUN_DIR / "k1_seed_gate.json"
WRIST_Z_BASE = 0.92          # hand_root world z = WRIST_Z_BASE + wrist_z ctrl
N_WRIST = 6


def load_seeds() -> list[dict]:
    """The four K1 poses, decomposed into wrist pose + 16 finger joint targets."""
    d = json.loads(W4_JSON.read_text())
    seeds = []
    for c in d["cells"]:
        p = np.asarray(c["best_pose"], dtype=float)
        assert p.size == 22, f"unexpected best_pose length {p.size}"
        seeds.append({
            "pedestal_mm": float(c["pedestal_mm"]),
            "box_rest_z_m": float(c["box_rest_z_m"]),
            "box_top_m": float(c["box_top_m"]),
            "wrist_world": [float(p[0]), float(p[1]), WRIST_Z_BASE + float(p[2])],
            "wrist_xyz_ctrl": [float(v) for v in p[:3]],
            "rot": [float(v) for v in p[3:6]],
            "joint_targets": [float(v) for v in p[6:22]],
            "best_J_m": float(c["best_J_m"]),
        })
    return seeds


def to_actions(env, joints: list[float]) -> np.ndarray:
    """Joint targets (rad) -> the 16 action units `_apply_action` expects.

    `_apply_action` maps a finger action a in [-1, 1] onto
    ``lo + (a + 1) * 0.5 * (hi - lo)``, so the inverse is
    ``a = 2 * (target - lo) / (hi - lo) - 1``. Needed because the K1 seeds are
    stored as joint angles from the kinematic scan, while the trajectory driver
    speaks action units.
    """
    lo = env.ctrl_low[N_WRIST:]
    hi = env.ctrl_high[N_WRIST:]
    tgt = np.clip(np.asarray(joints, dtype=float), lo, hi)
    return np.clip(2.0 * (tgt - lo) / np.maximum(hi - lo, 1e-9) - 1.0, -1.0, 1.0)


def seed_cfg(seed: dict, *, lift_rate: float = 0.004, lift_hold: int = 120,
             n_desc: int = 60, n_close: int = 80, settle: int = 25) -> dict:
    """One disclosed driver, identical for every seed.

    descend -> ramped close to the seed's own finger targets -> settle -> lift
    at a fixed rate. No per-seed tuning of press depth or timing: tuning those is
    SceneInitFix's job, and doing it here would duplicate their search and blur
    which question each of us answered.
    """
    return {
        "family": "k1_forced_seed",
        "wrist_world": list(seed["wrist_world"]),
        "rot": list(seed["rot"]),
        "close": None,               # filled per-env in `_rollout` (needs ctrlrange)
        "close_params": [1.0, 0.0, 0.0],
        "sequence": "descend_then_close",
        "close_ramp": True,
        "n_transit": 30, "n_desc": n_desc, "n_close": n_close,
        "settle": settle, "lift_z": 0.16,
        "lift_rate": lift_rate, "lift_hold": lift_hold,
    }


def _rollout(seed: dict, **kw):
    from grounding.envs.lift_hold_env import LiftHoldEnv
    from grounding.reachable_scan import gate_eval, run_traj
    from grounding.pedestal_load_probe import bearing_decomposition

    env = LiftHoldEnv(episode_len=400, domain_rand=False)
    cfg = seed_cfg(seed, **kw)
    cfg["close"] = to_actions(env, seed["joint_targets"]).tolist()
    run_traj(env, cfg)
    ev = gate_eval(env)
    bearing = bearing_decomposition(env, ev["peak_step"])
    sup = np.asarray(env.trace["support_force"], dtype=float)
    lift = np.asarray(env.trace["lift"], dtype=float)
    fing = np.asarray(env.trace["finger_force"], dtype=float)
    peak_i = int(np.argmax(lift))
    win = slice(0, peak_i + 1)
    env.close()
    weight = 1.4715
    return {
        "pedestal_mm": seed["pedestal_mm"],
        "wrist_xyz_ctrl": seed["wrist_xyz_ctrl"],
        "rot": seed["rot"],
        "gate": {k: v for k, v in ev.items() if k != "finger_pairs"},
        "finger_pairs": ev["finger_pairs"],
        "peak_lift_m": float(lift[peak_i]),
        "final_lift_m": float(lift[-1]),
        "n_steps_above_5cm": int(np.sum(lift >= 0.05)),
        "n_steps_above_5cm_airborne": int(np.sum((lift >= 0.05) & (sup < 1e-6))),
        "support_force_N_at_peak": float(sup[peak_i]),
        "support_box_weights_at_peak": round(float(sup[peak_i]) / weight, 4),
        "support_box_weights_mean_window": round(
            float(np.mean(sup[win])) / weight, 4),
        "finger_force_N_max": float(np.max(fing)),
        "bearing_at_peak": bearing,
        "driver": {k: v for k, v in cfg.items() if k != "close"},
    }


def main() -> None:
    seeds = load_seeds()
    rows = []
    for s in seeds:
        r = _rollout(s)
        r["gate_pass"] = bool(r["gate"]["gate_pass"])
        rows.append(r)
        g = r["gate"]
        print(
            f"K1 seed pedestal={s['pedestal_mm']:>4.0f}mm  "
            f"peak {r['peak_lift_m']*1000:7.1f}mm  "
            f"clean@peak={not g['table_touching_at_peak']}  "
            f"duty {g['finger_force_duty_cycle']:.2f}  "
            f"anchored {g['max_anchored_step_m']*1000:.1f}mm  "
            f"ped {r['support_box_weights_at_peak']:+6.2f} bw  "
            f"steps>5cm {r['n_steps_above_5cm']:>3d} "
            f"(airborne {r['n_steps_above_5cm_airborne']:>3d})  "
            f"PASS={r['gate_pass']}", flush=True)

    out = {
        "question": "under ONE fixed disclosed scripted driver, what do the frozen "
                    "smoke4 acceptance gates say about SceneInitFix's four K1 "
                    "dual-wall pinch poses?",
        "division_of_labour": (
            "SceneInitFix answers whether the K1 pinch can LIFT (it searches press "
            "depth / closing timing / lift curve). This module does NOT repeat "
            "that search: it runs each seed through the standard descend -> "
            "ramped close -> settle -> fixed-rate lift driver and reports the "
            "frozen gate verdict. The two conclusions are complementary and are "
            "reported separately, never merged into one number."),
        "why_forced_seeds": (
            "this round's own limit-covering grid is a 5^6 lattice; a "
            "random-restart-refined pose can fall between its points. Excluding "
            "these seeds would make 'my grid found nothing' a statement about "
            "the lattice again - the same class of error as the overturned w35d "
            "verdict."),
        "driver": "descend(60) -> ramped close(80) -> settle(25) -> lift 0.004 m/step, hold 120",
        "scene_commit": "20ca639",
        "scene_note": ("all four commits share XML 1ea0fdf5; the variable between "
                       "them is landing semantics, not physics"),
        "rows": rows,
        "n_pass": sum(int(r["gate_pass"]) for r in rows),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print(f"-> {OUT}  ({out['n_pass']}/{len(rows)} gate-clean)")


if __name__ == "__main__":
    main()
