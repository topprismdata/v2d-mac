"""W4-C - can the two-wall pad pinch actually LIFT? Search, not one straight line.

The K1 scan proved a pose exists in which two fingertip pads touch the two side
walls (all four pedestal heights). The scripted single-pass transit/press/lift
that followed it did not lift the box on any of them - which says the PATH was
wrong, not that the grasp is impossible.

This searches the path instead. The seed is the K1 pose; what varies is the
timing and depth of the close, the press, and the lift:

    approach_steps  retract the fingers and put the wrist at the K1 posture
    close_steps     ramp the absolute finger setpoints from open to the K1 pose
    press_steps     keep curling past the K1 pose (the pads are found within
                    0.02 mm of contact, so a real press has to go past it)
    press_rate      rad per control step of that press
    lift_frac       where in the press the lift starts
    lift_steps      steps to raise the wrist z setpoint
    lift_height     how far (m, clipped by the actuator range)
    lift_curl       extra curl DURING the lift (squeeze and pull up)
    hold_steps      settle at the top

Every trial is judged by ``tests/test_smoke4_accept._analyse_episode`` - the
function the frozen A1-A3c tests call - on a trace recorded through the ordinary
env step path, with the gate thresholds untouched:

    A1  peak lift > 0.05 m
    A2  a finger-geom <-> object_geom pair in the recorded contacts
    A3a finger force on >= 50% of the lift window AND mean >= 1.4715 N
    A3b the box is NOT touching its support at peak lift
    A3c < 10 mm per anchored step, free-fall dv_z within 0.6 m/s of gravity,
        and the separation from the support contact-driven

(A4/A5 are policy/report gates and are not adjudicable from a scripted
trajectory; the C6 control they rest on is pinned in tests/test_scene_init.py.)

    runs/.venv-mj/bin/python runs/w4/k1_lift_search.py --trials 800
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import pathlib
import sys
import time

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))

RUNS = pathlib.Path(__file__).resolve().parent
SCAN = RUNS / "pedestal_scan_kin.json"
OUT_JSON = RUNS / "k1_lift_search.json"
OUT_NPZ = REPO / "runs/p0/smoke4_mujoco_ppo/reachable_demo_set.npz"
MJCF_COMMIT = "1ea0fdf5c87ce5a161efcda7c597b0fc6647c9fe30bcf2f5d0bcb52402d24b89"

BOUNDS = {
    "approach_steps": (30, 90),
    "close_steps": (15, 70),
    "press_steps": (0, 60),
    "press_rate": (0.005, 0.05),
    "lift_frac": (0.0, 1.0),
    "lift_steps": (30, 120),
    "lift_height": (0.06, 0.16),
    "lift_curl": (0.0, 0.06),
    "hold_steps": (10, 40),
}
OPEN_FRAC = 0.12          # "retracted": this far along each finger's ctrl range


def sample(rng: np.random.Generator) -> dict:
    cfg = {k: float(rng.uniform(*v)) for k, v in BOUNDS.items()}
    for k in ("approach_steps", "close_steps", "press_steps", "lift_steps", "hold_steps"):
        cfg[k] = int(round(cfg[k]))
    return cfg


def jitter(cfg: dict, rng: np.random.Generator, scale: float) -> dict:
    out = dict(cfg)
    for k, (lo, hi) in BOUNDS.items():
        span = (hi - lo) * scale
        out[k] = float(np.clip(cfg[k] + rng.normal(0.0, span), lo, hi))
    for k in ("approach_steps", "close_steps", "press_steps", "lift_steps", "hold_steps"):
        out[k] = int(round(out[k]))
    return out


def ctrl_for_fingers(env, f_ctrl: np.ndarray) -> np.ndarray:
    lo, hi = env.unwrapped.ctrl_low[6:22], env.unwrapped.ctrl_high[6:22]
    return np.clip(2.0 * (f_ctrl - lo) / np.maximum(hi - lo, 1e-12) - 1.0, -1.0, 1.0)


_ENV = None
_POSE = None
_ACC = None


def _init(pose: np.ndarray) -> None:
    global _ENV, _POSE, _ACC
    from grounding.envs.lift_hold_env import LiftHoldEnv  # noqa: PLC0415
    import test_smoke4_accept as acc  # noqa: PLC0415

    _ENV = LiftHoldEnv(domain_rand=False, episode_len=600)
    _ENV.unwrapped.record_bearing = True
    _POSE = pose
    _ACC = acc


def run_cfg(cfg: dict, record: bool = False) -> dict:
    env, u, pose = _ENV, _ENV.unwrapped, _POSE
    env.reset(seed=0)
    target = np.concatenate([
        np.clip(pose[0:6], u.ctrl_low[0:6], u.ctrl_high[0:6]),
        np.clip(pose[6:22], u.ctrl_low[6:22], u.ctrl_high[6:22])])
    open_f = u.ctrl_low[6:22] + OPEN_FRAC * (u.ctrl_high[6:22] - u.ctrl_low[6:22])
    obs_all, act_all = [], []

    def drive(wrist_t: np.ndarray, finger_t: np.ndarray, n: int) -> None:
        for _ in range(n):
            a = np.zeros(22)
            a[:3] = np.clip((wrist_t[:3] - u.data.ctrl[:3]) * u.frame_skip / 0.012 / 6.0,
                            -1.0, 1.0)
            a[3:6] = np.clip((wrist_t[3:6] - u.data.ctrl[3:6]) / 0.10 / 6.0, -1.0, 1.0)
            a[6:22] = ctrl_for_fingers(env, finger_t)
            o, *_ = env.step(a.astype(np.float32))
            if record:
                obs_all.append(o.copy())
                act_all.append(a.astype(np.float32))

    ctrl0 = u.data.ctrl.copy()
    # approach: fingers retracted while the wrist walks to the K1 posture
    for k in range(1, cfg["approach_steps"] + 1):
        s = k / cfg["approach_steps"]
        drive(ctrl0[:6] + (target[:6] - ctrl0[:6]) * s, open_f, 1)
    # close: absolute finger setpoints ramp onto the K1 pose
    f_t = u.data.ctrl[6:22].copy()
    for k in range(1, cfg["close_steps"] + 1):
        f_t = open_f + (target[6:22] - open_f) * (k / cfg["close_steps"])
        drive(target[:6], f_t, 1)
    # press past the pose, in the direction each finger was closing
    press_dir = np.sign(target[6:22] - open_f)
    press_dir[press_dir == 0] = 1.0
    lift_at = int(cfg["lift_frac"] * cfg["press_steps"])
    z_top = min(u.ctrl_high[2], u.data.ctrl[2] + cfg["lift_height"])
    for k in range(1, cfg["press_steps"] + 1):
        f_t = np.clip(f_t + press_dir * cfg["press_rate"],
                      u.ctrl_low[6:22], u.ctrl_high[6:22])
        w_t = u.data.ctrl[:6].copy()
        if k > lift_at and cfg["lift_steps"] > 0:
            w_t[2] = u.data.ctrl[2] + (z_top - u.data.ctrl[2]) * min(
                1.0, (k - lift_at) / cfg["lift_steps"])
            f_t = np.clip(f_t + press_dir * cfg["lift_curl"],
                          u.ctrl_low[6:22], u.ctrl_high[6:22])
        drive(w_t, f_t, 1)
    # finish the lift ramp if the press ran out first
    for k in range(max(0, cfg["lift_steps"] - (cfg["press_steps"] - lift_at))):
        w_t = u.data.ctrl[:6].copy()
        w_t[2] = u.data.ctrl[2] + (z_top - u.data.ctrl[2]) * min(
            1.0, (k + 1) / max(1, cfg["lift_steps"]))
        f_t = np.clip(f_t + press_dir * cfg["lift_curl"], u.ctrl_low[6:22], u.ctrl_high[6:22])
        drive(w_t, f_t, 1)
    drive(u.data.ctrl[:6].copy(), f_t, cfg["hold_steps"])

    ev = _ACC._analyse_episode({"trace": u.trace, "episode": 0, "seed": 0})  # noqa: SLF001
    tr = u.trace
    gates = {
        "A1": bool(ev["peak_lift_m"] > _ACC.LIFT_THRESHOLD),
        "A2": bool(ev["n_finger_contact_steps"] > 0),
        "A3a": bool(ev["force_duty_ok"] and ev["force_sufficiency_ok"]),
        "A3b": bool(not ev["table_touching_at_peak"]),
        "A3c": bool(ev["anchored_step_ok"] and ev["free_fall_gravity_ok"]
                    and ev["separation_contact_driven"]),
    }
    out = {
        "cfg": cfg,
        "gates": gates,
        "pass_all": all(gates.values()),
        "peak_lift_m": ev["peak_lift_m"],
        "max_lift_m": float(max(tr["lift"])),
        "final_obj_z_m": float(u.data.xpos[u.obj_body][2]),
        "z0_m": float(u._z0),
        "finger_force_duty_cycle": ev["finger_force_duty_cycle"],
        "finger_force_mean_N": ev["finger_force_mean_N"],
        "max_anchored_step_m": ev["max_anchored_step_m"],
        "support_peak_N": float(max(tr.get("support_force", [0.0]))),
        "bearing_peak_N": {
            "hook": float(max(tr.get("hook_up_N_peak", [-1e9]))),
            "wall": float(max(tr.get("wall_up_N_peak", [-1e9]))),
            "support": float(max(tr.get("pedestal_up_N_peak", [-1e9]))),
        },
        "finger_pairs": ev["finger_pairs"],
        "wrist_z_at_peak_m": ev["wrist_z_at_peak"],
        "peak_step": ev["peak_step"],
    }
    if record:
        out["_obs"] = np.asarray(obs_all, dtype=np.float32)
        out["_act"] = np.asarray(act_all, dtype=np.float32)
    return out


def _work(args):
    cfg, seed = args
    rng = np.random.default_rng(seed)
    _init(_POSE_GLOBAL)
    return run_cfg(cfg)


def _init_pool(pose):
    global _POSE_GLOBAL
    _POSE_GLOBAL = pose


def score(row: dict) -> tuple:
    """Rank: pass_all, then A1+A3b (a real airborne lift), then lift height."""
    g = row["gates"]
    return (int(row["pass_all"]),
            int(g["A1"] and g["A3b"]),
            int(g["A1"]),
            row["max_lift_m"],
            -row["support_peak_N"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=800)
    ap.add_argument("--refine", type=int, default=250)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--pedestal-mm", type=float, default=60.0)
    args = ap.parse_args()

    scan = json.loads(SCAN.read_text())
    pose = np.array([c for c in scan["cells"]
                     if abs(c["pedestal_mm"] - args.pedestal_mm) < 1e-6][0]["best_pose"])

    t0 = time.time()
    rng = np.random.default_rng(20261015)
    rows = []
    with mp.Pool(args.workers, initializer=_init_pool, initargs=(pose,)) as pool:
        for i, cfg in enumerate(sample(rng) for _ in range(args.trials)):
            rows.append(pool.apply_async(_work, ((cfg, 20261015 + i),)).get())
            if (i + 1) % 100 == 0:
                best = max(rows, key=score)
                print(f"  {i + 1}/{args.trials}  best lift {best['max_lift_m']*1e3:.1f} mm "
                      f"pass={best['pass_all']} gates={best['gates']} "
                      f"[{time.time() - t0:.0f}s]", flush=True)
        rows.sort(key=score, reverse=True)
        print(f"  random done [{time.time() - t0:.0f}s]; refining around the best",
              flush=True)
        best_cfg = rows[0]["cfg"]
        for i in range(args.refine):
            scale = 0.5 * (1.0 - i / args.refine) + 0.05
            cfg = jitter(best_cfg, rng, scale)
            rows.append(pool.apply_async(_work, ((cfg, 700000 + i),)).get())

    rows.sort(key=score, reverse=True)
    top = rows[:20]
    summary = {
        "question": "starting from the K1 two-wall pad pose, does any press/close/"
                    "lift schedule actually lift the box under the frozen gates?",
        "gates": {"A1": "peak lift > 0.05 m", "A2": "a finger<->object pair exists",
                  "A3a": "duty >= 0.5 and mean >= 1.4715 N over the lift window",
                  "A3b": "no support contact at peak lift",
                  "A3c": "anchored step < 10 mm, free-fall dv_z within 0.6 m/s, "
                         "contact-driven separation"},
        "pedestal_mm": args.pedestal_mm,
        "seed_pose": [float(v) for v in pose],
        "n_trials": len(rows),
        "n_pass_all": sum(1 for r in rows if r["pass_all"]),
        "n_A1": sum(1 for r in rows if r["gates"]["A1"]),
        "best": top[0],
        "top": top,
        "seconds": round(time.time() - t0, 1),
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2))
    b = top[0]
    print(f"\ntrials {len(rows)}  A1 passes {summary['n_A1']}  "
          f"all-gate passes {summary['n_pass_all']}")
    print(f"best: lift {b['max_lift_m'] * 1e3:.1f} mm  gates {b['gates']}  "
          f"support peak {b['support_peak_N']:.1f} N  cfg {b['cfg']}")
    print(f"wrote {OUT_JSON}")

    if b["pass_all"]:
        rec = run_cfg(b["cfg"], record=True)
        ev = rec
        OUT_NPZ.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            OUT_NPZ,
            bc_obs=ev["_obs"], bc_act=ev["_act"],
            bc_pose=np.asarray([[*pose[:6], 0.0, 0.0, 0.0]], dtype=np.float32),
            bc_trajectory_len=np.array([len(ev["_obs"])], dtype=np.int32),
            mjcf_commit=np.array(MJCF_COMMIT),
            selection=np.array("k1_two_wall_pad"),
            cfg_json=np.array(json.dumps(b["cfg"])),
            gate_metrics=np.array(json.dumps(b["gates"] | {
                k: b[k] for k in ("peak_lift_m", "max_lift_m",
                                  "finger_force_duty_cycle", "finger_force_mean_N",
                                  "max_anchored_step_m", "support_peak_N",
                                  "bearing_peak_N")})),
            load_decomposition=np.array(json.dumps(b["bearing_peak_N"])),
        )
        print(f"recorded demo seed -> {OUT_NPZ}")


if __name__ == "__main__":
    main()
