"""Task 4.4 - SB3 PPO on the floating-Allegro lift/hold task.

    python grounding/train_ppo.py --total-steps 2000000 --n-envs 8 --seed 0

Curves (windowed means of the raw episode telemetry) go to
``runs/p0/smoke4_mujoco_ppo/curve.csv``; the trained policy is saved next to it and
re-evaluated with full contact tracing into ``eval_episodes.json`` for the
acceptance test. Nothing here writes qpos: SB3 only ever calls ``env.step``.
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


def _make_env(rank: int, seed: int, episode_len: int):
    from stable_baselines3.common.monitor import Monitor  # noqa: PLC0415

    from grounding.envs.lift_hold_env import LiftHoldEnv  # noqa: PLC0415

    def _init():
        env = LiftHoldEnv(episode_len=episode_len)
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
          early_stop_lift: float) -> dict:
    from stable_baselines3 import PPO  # noqa: PLC0415
    from stable_baselines3.common.callbacks import BaseCallback  # noqa: PLC0415
    from stable_baselines3.common.vec_env import SubprocVecEnv  # noqa: PLC0415

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    class CurveCallback(BaseCallback):
        def __init__(self) -> None:
            super().__init__()
            self.curve = WindowedCurve(OUT_DIR / "curve.csv")
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
            return False

        def _on_training_end(self) -> None:
            self.curve.close()

    vec = SubprocVecEnv([_make_env(i, seed, episode_len) for i in range(n_envs)])
    model = PPO(
        "MlpPolicy", vec,
        seed=seed,
        verbose=0,
        n_steps=1024,
        batch_size=256,
        gae_lambda=0.95,
        gamma=0.99,
        ent_coef=0.0,
        learning_rate=3e-4,
        clip_range=0.2,
        policy_kwargs={"net_arch": [256, 256]},
    )
    cb = CurveCallback()
    t0 = time.time()
    model.learn(total_timesteps=total_steps, callback=cb, progress_bar=False)
    wall = time.time() - t0
    model.save(OUT_DIR / "ppo_lift_hold")
    vec.close()

    summary = {
        "total_steps_requested": total_steps,
        "wall_seconds": wall,
        "seed": seed,
        "n_envs": n_envs,
        "episode_len": episode_len,
        "best_training_episode": cb.best,
        "n_episodes": cb.episode,
        "early_stop_target_m": early_stop_lift,
    }
    (OUT_DIR / "train_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def evaluate(n_episodes: int, seed: int, episode_len: int) -> dict:
    """Deterministic evaluation with full contact tracing."""
    from stable_baselines3 import PPO  # noqa: PLC0415

    from grounding.envs.lift_hold_env import LiftHoldEnv  # noqa: PLC0415

    model = PPO.load(OUT_DIR / "ppo_lift_hold", device="cpu")
    env = LiftHoldEnv(episode_len=episode_len)
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
        episodes.append({
            "episode": ep,
            "seed": seed + ep,
            "return": float(total),
            "z0": float(env._z0),
            "max_lift_m": float(max(env.trace["lift"])),
            "final_lift_m": float(env.trace["lift"][-1]),
            "max_contact_force_N": float(max(env.trace["contact_force"])),
            "contact_pairs_any": [list(p) for p in flat],
            "contact_pairs_at_peak": [list(p) for p in pairs[int(np.argmax(env.trace["lift"]))]],
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
            },
        })
        print(f"[eval] ep={ep} max_lift={episodes[-1]['max_lift_m']:+.4f} m "
              f"final={episodes[-1]['final_lift_m']:+.4f} m "
              f"finger_contact_steps={episodes[-1]['n_steps_with_finger_contact']}",
              flush=True)
    env.close()

    result = {
        "n_episodes": n_episodes,
        "evaluator": json.loads(
            (REPO / "runs/p0/assets/tissue_box_lift_hold.json").read_text())["evaluator"],
        "best_episode": int(np.argmax([e["max_lift_m"] for e in episodes])),
        "episodes": episodes,
    }
    (OUT_DIR / "eval_episodes.json").write_text(json.dumps(result, indent=2))
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
    args = p.parse_args()

    if not args.eval_only:
        s = train(args.total_steps, args.n_envs, args.seed, args.episode_len,
                  args.early_stop_lift)
        print(json.dumps(s, indent=2), flush=True)
    res = evaluate(args.eval_episodes, args.seed + 10_000, args.episode_len)
    best = res["episodes"][res["best_episode"]]
    print(f"BEST max_lift={best['max_lift_m']:+.4f} m", flush=True)


if __name__ == "__main__":
    main()
