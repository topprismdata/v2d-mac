"""Task 4.4 - SB3 PPO on the floating-Allegro lift/hold task.

    python grounding/train_ppo.py --total-steps 2000000 --n-envs 8 --seed 0

Curves (windowed means of the raw episode telemetry) go to
``runs/p0/smoke4_mujoco_ppo/curve.csv``; the trained policy is saved next to it and
re-evaluated with full contact tracing into ``eval_episodes.json`` for the
acceptance test. Nothing here writes qpos: SB3 only ever calls ``env.step``.

Two pipeline-validation knobs that must NOT be used for a formal run:
``--out-dir`` redirects every artifact so a plumbing run cannot overwrite the
committed smoke4 evidence, and ``--init-from-bc`` loads a behaviour-cloning warm
start (``grounding/bc_pretrain.py``) into the fresh policy before training.
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys
import time

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

OUT_DIR = REPO / "runs/p0/smoke4_mujoco_ppo"


def _make_env(rank: int, seed: int, episode_len: int,
              pedestal_load_weight: float | None = None):
    from stable_baselines3.common.monitor import Monitor  # noqa: PLC0415

    from grounding.envs.lift_hold_env import LiftHoldEnv  # noqa: PLC0415

    kwargs = {} if pedestal_load_weight is None else {
        "pedestal_load_weight": pedestal_load_weight}

    def _init():
        env = LiftHoldEnv(episode_len=episode_len, **kwargs)
        env = Monitor(env)
        env.reset(seed=seed + rank)
        env.action_space.seed(seed + rank)
        return env

    return _init


class WindowedCurve:
    """Windowed means of the per-episode telemetry -> curve.csv."""

    FIELDS = ("reward", "lift", "grasp", "hold", "z_peak", "contact_force")

    def __init__(self, path: pathlib.Path, window: int = 20) -> None:
        self.path = path
        self.window = window
        self.rows: list[dict[str, float]] = []
        self.buffer: dict[str, list[float]] = {f: [] for f in self.FIELDS}
        self._fh = path.open("w", newline="")
        self._w = csv.writer(self._fh)
        self._w.writerow(["episode", "global_step"] + [f"{f}_mean" for f in self.FIELDS])

    def push(self, episode: int, global_step: int, info: dict) -> None:
        for f in self.FIELDS:
            self.buffer[f].append(float(info.get(f, 0.0)))
        if len(self.buffer["reward"]) < self.window:
            return
        row = {"episode": episode, "global_step": global_step}
        for f in self.FIELDS:
            row[f] = float(np.mean(self.buffer[f][-self.window:]))
        self._w.writerow([row["episode"], row["global_step"]] + [row[f] for f in self.FIELDS])
        self._fh.flush()
        self.rows.append(row)

    def close(self) -> None:
        self._fh.close()


def train(total_steps: int, n_envs: int, seed: int, episode_len: int,
          early_stop_lift: float, resume: bool = False,
          out_dir: pathlib.Path = OUT_DIR,
          init_from_bc: pathlib.Path | None = None,
          pedestal_load_weight: float | None = None) -> dict:
    from stable_baselines3 import PPO  # noqa: PLC0415
    from stable_baselines3.common.callbacks import BaseCallback  # noqa: PLC0415
    from stable_baselines3.common.vec_env import SubprocVecEnv  # noqa: PLC0415

    if resume and init_from_bc is not None:
        raise ValueError("--resume and --init-from-bc are mutually exclusive: "
                         "one continues a trained policy, the other overwrites a "
                         "fresh one")

    out_dir.mkdir(parents=True, exist_ok=True)

    class CurveCallback(BaseCallback):
        def __init__(self) -> None:
            super().__init__()
            self.curve = WindowedCurve(out_dir / "curve.csv")
            self.episode = 0
            self.best = {"z_peak": -1.0, "episode": -1}
            self.history: list[dict] = []

        def _on_step(self) -> bool:
            for info in self.locals.get("infos", []):
                ep = info.get("episode")
                stats = info.get("lifecycle")
                if ep is None or stats is None:
                    continue
                self.episode += 1
                payload = {
                    "reward": ep.get("r", 0.0),
                    "lift": stats.get("lift", 0.0),
                    "grasp": stats.get("grasp", 0.0),
                    "hold": stats.get("hold", 0.0),
                    "z_peak": stats.get("z_peak", 0.0),
                    "contact_force": stats.get("contact_force", 0.0),
                }
                self.curve.push(self.episode, int(self.num_timesteps), payload)
                self.history.append({"episode": self.episode,
                                      "global_step": int(self.num_timesteps), **payload})
                if payload["z_peak"] > self.best["z_peak"]:
                    self.best = {"z_peak": payload["z_peak"], "episode": self.episode,
                                 "global_step": int(self.num_timesteps)}
            # True == keep training; returning False would abort learn() after 1 step
            return True

        def _on_training_end(self) -> None:
            self.curve.close()

    vec = SubprocVecEnv(
        [_make_env(i, seed, episode_len, pedestal_load_weight)
         for i in range(n_envs)], start_method="spawn")
    ckpt = out_dir / "ppo_lift_hold.zip"
    resumed_from = None
    if resume and ckpt.exists():
        # continue from the existing policy; total_steps is the NEW budget to add
        model = PPO.load(ckpt, env=vec, device="cpu")
        resumed_from = int(model.num_timesteps)
    else:
        model = PPO(
            "MlpPolicy", vec,
            seed=seed,
            verbose=0,
            # 8 envs x 256 steps = 2048 samples per rollout -> ~975 policy updates
            # over the 2M budget. With n_steps=1024 it was only ~244 updates, far
            # too few for a 22-dim contact-rich exploration problem.
            n_steps=256,
            batch_size=128,
            n_epochs=10,
            gae_lambda=0.95,
            gamma=0.99,
            # a little entropy helps: the grasp is only reachable by coordinated
            # 22-dim motion, and a deterministic policy gets stuck in the
            # "never touch the box" local optimum.
            ent_coef=0.01,
            learning_rate=3e-4,
            clip_range=0.2,
            policy_kwargs={"net_arch": [256, 256]},
        )
        if init_from_bc is not None:
            from grounding.bc_pretrain import load_into_sb3  # noqa: PLC0415

            bc_info = load_into_sb3(model, init_from_bc)
            print(f"[bc] warm start loaded from {init_from_bc}: "
                  f"max|delta| {json.dumps(bc_info['max_abs_delta'])}", flush=True)
    cb = CurveCallback()
    t0 = time.time()
    model.learn(total_timesteps=total_steps, callback=cb, progress_bar=False,
                reset_num_timesteps=not resume)
    wall = time.time() - t0
    model.save(out_dir / "ppo_lift_hold")
    vec.close()

    summary = {
        "total_steps_requested": total_steps,
        "cumulative_timesteps": int(model.num_timesteps),
        "resumed_from_timesteps": resumed_from,
        "init_from_bc": str(init_from_bc) if init_from_bc else None,
        "wall_seconds": wall,
        "seed": seed,
        "n_envs": n_envs,
        "episode_len": episode_len,
        "best_training_episode": cb.best,
        "n_episodes": cb.episode,
        "early_stop_target_m": early_stop_lift,
        "pedestal_load_weight": pedestal_load_weight,
    }
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def bearing_summary(trace: dict, peak_i: int, box_weight_n: float) -> dict:
    """Per-episode bearing decomposition, reported at THREE resolutions.

    Round p0(4e) asks where the box's weight was actually going - top-face hook,
    side-wall friction, pedestal - rather than only how high the policy displayed
    it. Three columns, because they answer different questions and a column
    named "peak" that is not actually a peak is worse than no column at all
    (Main ruling, wave 3.5-G):

      hold              mean over the hold window (steps whose hold_streak >= 1)
                        of the control-step sample. Quasi-static caliber; this
                        is the column the load-path claims are read from.
                        Resolution: 20 ms (one control step).
      peak_control_step max over the hold window of the control-step sample.
                        Resolution: 20 ms. CANNOT see a frame-internal spike.
      peak_substep      max over the hold window of the per-control-step
                        intra-frame maximum. Resolution: 2 ms (the sim
                        timestep), sampled at the END of each completed
                        mj_step. This is the only column that can show the
                        frame-internal amplification stiffened contacts produce
                        (Wave 3.5F: 9 N @ 0.6 mm -> 9.93 kN @ 20.8 mm).

    Sampling alignment: we sample after each completed mj_step. A probe that
    wraps mj_step and samples BEFORE it is offset by one substep (2 ms), i.e.
    their index i corresponds to our index i-1. The two numbers must not be read
    as the same quantity without applying that offset.

    Signs matter and are preserved: a negative path pushes the box DOWN, which
    is exactly what a corner wedge does with its fingers.
    """
    paths = ("hook", "wall", "pedestal")
    step = {k: np.asarray(trace[f"{k}_up_N"], dtype=float) for k in paths}
    peak = {k: np.asarray(trace.get(f"{k}_up_N_peak", []), dtype=float)
            for k in paths}
    streak = np.asarray(trace.get("hold_streak", []), dtype=float)
    i = int(peak_i)

    window = slice(0, i + 1)
    hold_win = None
    if streak.size:
        mask = streak[: i + 1] >= 1.0
        if mask.any():
            hold_win = slice(0, i + 1)
            sel = np.where(mask)[0]

    def _col(arr: np.ndarray) -> dict:
        return {
            "at_peak_step": round(float(arr[i]) / box_weight_n, 4),
            "mean_over_lift_window": round(
                float(np.mean(arr[window])) / box_weight_n, 4),
            "max_over_lift_window": round(
                float(np.max(arr[window])) / box_weight_n, 4),
        }

    def _hold(arr: np.ndarray) -> dict:
        if hold_win is None or sel.size == 0:
            return {"n_steps": 0, "mean": None}
        v = arr[sel]
        j = int(sel[int(np.argmax(v))])
        return {"n_steps": int(sel.size), "mean": round(
            float(np.mean(v)) / box_weight_n, 4), "max": round(
            float(np.max(v)) / box_weight_n, 4), "max_step": int(j)}

    out = {
        "peak_step": i,
        "box_weight_N": float(box_weight_n),
        "caliber": {
            "hold": "mean over evaluator-valid hold steps of the control-step "
                    "sample; quasi-static caliber; resolution 20 ms (1 control "
                    "step, 10 x 2 ms sim substeps)",
            "peak_control_step": "max over the lift window of the control-step "
                                 "sample; resolution 20 ms; CANNOT see a "
                                 "frame-internal spike",
            "peak_substep": "max over the lift window of the intra-frame "
                            "maximum; resolution 2 ms (the sim timestep), "
                            "sampled at the END of each completed mj_step",
            "alignment": "a probe that samples BEFORE mj_step instead of after "
                         "is offset by one substep (2 ms): its index i "
                         "corresponds to ours i-1",
        },
        "hold": {k: _hold(step[k]) for k in paths},
        "peak_control_step": {k: _col(step[k]) for k in paths},
        "peak_substep": ({
            "resolution_m_s": 0.002,
            "sampling": "end of each completed mj_step",
            **{k: _col(peak[k]) for k in paths},
        } if peak["hook"].size else {"resolution_m_s": 0.002,
                                     "sampling": "not recorded"}),
    }
    # The path decomposition at the single peak-lift step, kept verbatim: it is
    # what the A3b diagnosis is read from and what the round's own tests pin.
    out["bearing_box_weights"] = {
        k: round(float(step[k][i]) / box_weight_n, 4) for k in paths}
    total = float(sum(float(step[k][i]) for k in paths))
    out["upward_total_N"] = total
    out["net_box_weights"] = round(total / box_weight_n, 4)
    sup = np.asarray(trace.get("support_force", []), dtype=float)
    if sup.size:
        out["support_force_N_at_peak"] = float(sup[i])
        out["support_force_box_weights_at_peak"] = round(
            float(sup[i]) / box_weight_n, 4)
        out["support_force_box_weights_mean_window"] = round(
            float(np.mean(sup[window])) / box_weight_n, 4)
    return out


def evaluate(n_episodes: int, seed: int, episode_len: int,
             out_dir: pathlib.Path = OUT_DIR) -> dict:
    """Deterministic evaluation with full contact tracing."""
    from stable_baselines3 import PPO  # noqa: PLC0415

    from grounding.envs.lift_hold_env import LiftHoldEnv  # noqa: PLC0415

    model = PPO.load(out_dir / "ppo_lift_hold", device="cpu")
    env = LiftHoldEnv(episode_len=episode_len)
    # A3b evidence: record where the box's weight was actually carried on every
    # step - top-face hook / side-wall friction / pedestal. Off during training
    # (it needs contact points, not just forces), on for the eval run.
    env.record_bearing = True
    episodes = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done, total = False, 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, rew, term, trunc, info = env.step(action)
            total += rew
            done = term or trunc
        pairs = [sorted(set(tuple(p) for p in step)) for step in env.trace["contact_pairs"]]
        flat = sorted({p for step in pairs for p in step})
        peak_i = int(np.argmax(env.trace["lift"]))
        episodes.append({
            "episode": ep,
            "seed": seed + ep,
            "return": float(total),
            "z0": float(env._z0),
            "max_lift_m": float(max(env.trace["lift"])),
            "final_lift_m": float(env.trace["lift"][-1]),
            "max_contact_force_N": float(max(env.trace["contact_force"])),
            "bearing": bearing_summary(
                env.trace, peak_i,
                float(env.model.body_mass[env.obj_body]) * 9.81),
            "contact_pairs_any": [list(p) for p in flat],
            "contact_pairs_at_peak": [list(p) for p in pairs[peak_i]],
            "n_steps_with_finger_contact": int(sum(
                1 for step in pairs
                if any(g[0].split("#")[0].startswith(("ff_", "mf_", "rf_", "th_"))
                       for g in step))),
            "trace": {
                "obj_z": env.trace["obj_z"],
                "lift": env.trace["lift"],
                "wrist_z": env.trace["wrist_z"],
                "contact_force": env.trace["contact_force"],
                "finger_force": env.trace["finger_force"],
                "contact_pairs": pairs,
                "obj_vz": env.trace["obj_vz"],
                "supported": env.trace["supported"],
                "finger_held": env.trace["finger_held"],
                "support_force": env.trace["support_force"],
                "hook_up_N": env.trace["hook_up_N"],
                "wall_up_N": env.trace["wall_up_N"],
                "pedestal_up_N": env.trace["pedestal_up_N"],
                "dt": float(env.dt),
            },
        })
        print(f"[eval] ep={ep} max_lift={episodes[-1]['max_lift_m']:+.4f} m "
              f"final={episodes[-1]['final_lift_m']:+.4f} m "
              f"finger_contact_steps={episodes[-1]['n_steps_with_finger_contact']}",
              flush=True)
    env.close()

    # A4 control: same policy, box parked out of the hand's workspace. If the lift
    # still happened here, it would not be contact-driven.
    ctrl_env = LiftHoldEnv(out_of_reach=True, episode_len=episode_len)
    ctrl_peak = 0.0
    ctrl_finger_steps = 0
    for ep in range(3):
        obs, _ = ctrl_env.reset(seed=999 + ep)
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, _ = ctrl_env.step(action)
            done = term or trunc
        ctrl_peak = max(ctrl_peak, float(max(ctrl_env.trace["lift"])))
        ctrl_finger_steps += sum(
            1 for step in ctrl_env.trace["contact_pairs"]
            if any(p[0].split("#")[0].startswith(("ff_", "mf_", "rf_", "th_"))
                   for p in step))
    ctrl_env.close()
    print(f"[control out_of_reach] peak_lift={ctrl_peak:+.4f} m "
          f"finger_contact_steps={ctrl_finger_steps}", flush=True)

    result = {
        "n_episodes": n_episodes,
        "evaluator": json.loads(
            (REPO / "runs/p0/assets/tissue_box_lift_hold.json").read_text())["evaluator"],
        "best_episode": int(np.argmax([e["max_lift_m"] for e in episodes])),
        "control_out_of_reach": {
            "n_episodes": 3,
            "peak_lift_m": ctrl_peak,
            "n_finger_contact_steps": ctrl_finger_steps,
            "threshold_m": 0.05,
            "pass": bool(ctrl_peak < 0.05),
        },
        "episodes": episodes,
    }
    (out_dir / "eval_episodes.json").write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--total-steps", type=int, default=2_000_000)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--episode-len", type=int, default=250)
    p.add_argument("--eval-episodes", type=int, default=10)
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--early-stop-lift", type=float, default=0.10)
    p.add_argument("--resume", action="store_true",
                   help="continue from runs/p0/smoke4_mujoco_ppo/ppo_lift_hold.zip; "
                        "--total-steps is then the ADDITIONAL budget")
    p.add_argument("--out-dir", type=pathlib.Path, default=OUT_DIR,
                   help="artifact directory (default: the formal smoke4 dir; "
                        "pipeline-validation runs MUST redirect this so the "
                        "committed evidence is not overwritten)")
    p.add_argument("--init-from-bc", type=pathlib.Path, default=None,
                   help="load a bc_pretrain.py warm start into the fresh policy "
                        "before training (mutually exclusive with --resume)")
    p.add_argument("--pedestal-load-weight", type=float, default=None,
                   help="ABLATION ONLY. Overrides W_PEDESTAL_LOAD; 0.0 "
                        "reproduces the round-p0(4d) reward exactly. Must be "
                        "paired with --out-dir so the formal evidence is not "
                        "overwritten.")
    args = p.parse_args()

    if args.pedestal_load_weight is not None and args.out_dir == OUT_DIR:
        raise SystemExit(
            "--pedestal-load-weight is an ablation knob and would overwrite the "
            "formal evidence in " + str(OUT_DIR) + "; pass a distinct --out-dir")

    if not args.eval_only:
        s = train(args.total_steps, args.n_envs, args.seed, args.episode_len,
                  args.early_stop_lift, resume=args.resume, out_dir=args.out_dir,
                  init_from_bc=args.init_from_bc,
                  pedestal_load_weight=args.pedestal_load_weight)
        print(json.dumps(s, indent=2), flush=True)
    res = evaluate(args.eval_episodes, args.seed + 10_000, args.episode_len,
                   out_dir=args.out_dir)
    best = res["episodes"][res["best_episode"]]
    print(f"BEST max_lift={best['max_lift_m']:+.4f} m", flush=True)


if __name__ == "__main__":
    main()
