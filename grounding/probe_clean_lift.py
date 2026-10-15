"""Decisive feasibility probe: can this scene produce a *clean* grasp-and-lift?

The 4000-pose parameter scan reports `n_feasible = 0` on the revised MJCF
(4508421). Every pose that passes all seven numeric gates still has the box in
contact with `pedestal` at its peak step, i.e. it levers the box on the pedestal
edge rather than lifting it clear - which the `not touching` criterion rejects
by design.

This probe asks the prior question directly, with geometry instead of a blind
16-dimensional search: put the palm at box mid-height with a known lateral
clearance, curl the fingers, and pull straight up. It reports whether the box
ever leaves the support, and with what finger force.

    python grounding/probe_clean_lift.py            # grid over approach params
    python grounding/probe_clean_lift.py --quick    # single pose

Not an acceptance test: it produces feasibility evidence for the scene, which
is an input to deciding whether the scene or the controller is at fault.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

OUT = Path("runs/p0/smoke4_mujoco_ppo/clean_lift_probe.json")

N_WRIST_A = 6   # action indices 0..5 are wrist deltas; fingers start at 6


def _geom_half(env, gid: int) -> np.ndarray:
    return np.asarray(env.model.geom_size[gid], dtype=np.float64)


def _on_support(rows: list[dict], sup: set[str]) -> bool:
    return any((p[0] in sup or p[1] in sup) for p in rows["pairs"])


def tip_geom_ids(env) -> list[int]:
    import mujoco  # noqa: PLC0415
    m = env.model
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) for i in range(m.ngeom)]
    return [i for i, n in enumerate(names) if n and n.endswith("_distal_collision")]


def find_reaching_rotations(env, cands, box, box_top_z: float) -> list[tuple]:
    """Rotation probe: at reset the fingers point UP, so a zero-rotation wrist can
    never reach a box below it. Score each rotation bias by how far the
    fingertips get below the box top, and keep the ones that get there."""
    scored = []
    for rot in cands:
        env.reset(seed=0)
        for _ in range(90):
            a = np.zeros(22)
            a[3:6] = rot
            a[N_WRIST_A:] = 0.6
            env.step(a)
        zmin = float(min(env.data.geom_xpos[i][2] for i in tip_geom_ids(env)))
        scored.append((rot, zmin - box_top_z))
    return sorted(scored, key=lambda t: t[1])


def run_pose(env, *, side: float, height: float, curl_steps: int,
             lift_steps: int, approach_steps: int = 260,
             open_a: float = 0.6, rot_bias: tuple = (0.0, 0.0, 0.0)) -> dict:
    """One scripted wrap-and-pull attempt.

    `side`      lateral wrist offset from the box centre (m, along y)
    `height`    wrist height relative to the box centre (m)
    `open_a`    finger setpoint while approaching (+1 = open, -1 = curled)
    """
    env._scan_seed = 0
    env.reset(seed=0)

    box = env.data.xpos[env.obj_body].copy()
    # Control the wrist in WORLD coordinates. The action is a delta on ctrl, and
    # wrist_z is inverted relative to world z (ctrl -0.163 -> 0.869 m, ctrl +0.16
    # -> 1.045 m, measured), so a controller written against ctrl drove the hand
    # the wrong way: it pinned ctrl at the upper bound and left the fingertips
    # 20-29 cm above the box. These gains are the measured world displacement per
    # unit action: 12 mm for the slides, 8 mm * 0.545 for z.
    world_gain = np.array([0.012, 0.012, 0.00436])
    wrist_target = box + np.array([side, 0.0, height])

    def _track(target_z_offset: float = None) -> np.ndarray:
        tgt = wrist_target.copy()
        if target_z_offset is not None:
            tgt[2] = box[2] + target_z_offset
        err = tgt - env.data.xpos[env.hand_root]
        a = np.zeros(22)
        a[0:3] = np.clip(err / world_gain, -1.0, 1.0)
        a[3:6] = rot_bias
        a[N_WRIST_A:] = open_a
        return a

    # Approach: hold the rotation while tracking the wrist onto the target.
    for _ in range(approach_steps):
        env.step(_track())

    # Close the grasp: wrist holds, fingers wind to the curl stop.
    close = np.zeros(22)
    close[3:6] = rot_bias
    close[N_WRIST_A:] = -1.0
    for _ in range(curl_steps):
        env.step(close)

    # Pull straight up, tracking the wrist as it climbs (a pure +z setpoint delta
    # would push ctrl into its upper bound and stall).
    trace: list[dict] = []
    for _ in range(lift_steps):
        env.step(_track(target_z_offset=height + 0.12))
        trace.append({
            "lift": float(env.trace["lift"][-1]),
            "finger_force": float(env.trace["finger_force"][-1]),
            "pairs": [tuple(p) for p in env.trace["contact_pairs"][-1]],
            "held": bool(env.trace["finger_held"][-1]),
        })

    sup = env.support_geoms
    lifts = np.asarray([t["lift"] for t in trace])
    peak_i = int(np.argmax(lifts))
    streak = env.trace["hold_streak"]
    return {
        "side": side,
        "height": height,
        "peak_lift_m": float(lifts[peak_i]),
        "final_lift_m": float(lifts[-1]),
        "peak_step": peak_i,
        "box_on_support_at_peak": bool(_on_support(trace[peak_i], sup)),
        "ever_left_support": bool(not any(_on_support(t, sup) for t in trace)),
        "steps_off_support": int(sum(1 for t in trace if not _on_support(t, sup))),
        "peak_finger_force_N": float(max(t["finger_force"] for t in trace)),
        "held_fraction": float(np.mean([t["held"] for t in trace])),
        "hold_streak": int(max(streak)) if streak else 0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    from grounding.envs.lift_hold_env import LiftHoldEnv  # noqa: PLC0415

    env = LiftHoldEnv(episode_len=400)
    box_half = _geom_half(env, env.obj_geom)
    env.reset(seed=0)
    box = env.data.xpos[env.obj_body].copy()
    box_top = box[2] + float(box_half[2])

    # Stage 1 - which wrist rotation can even get a fingertip below the box top?
    rot_cands = [(rx, ry, rz) for rx in (-2.5, -1.6, 0.0, 1.6, 2.5)
                 for ry in (0.0,) for rz in (0.0,)]
    scored = find_reaching_rotations(env, rot_cands, box, box_top)
    print("rotation probe (fingertip z relative to box top, m):")
    for rot, dz in scored:
        print(f"  rot={rot}  {dz:+.3f}")
    reachable = [rot for rot, dz in scored if dz < -0.01]
    print(f"reaching rotations: {len(reachable)}/{len(scored)}")
    if not reachable:
        print("NO rotation reaches the box - the scene is geometrically infeasible.")
        return

    # Grasp axis: the box is 120 x 240 x 80 mm (half 0.06/0.12/0.04), so the
    # 240 mm span is the wrong axis to reach across - a hand wraps the 120 mm
    # one. Approach along x, not y.
    sides = [-(float(box_half[0]) + 0.045), -(float(box_half[0]) + 0.085)]
    heights = [-0.02, 0.0, 0.02]
    if args.quick:
        sides, heights, reachable = sides[:1], heights[:1], reachable[:1]

    rows = []
    for rot in reachable:
        for side in sides:
            for height in heights:
                r = run_pose(env, side=side, height=height, rot_bias=rot,
                             curl_steps=100, lift_steps=220)
                rows.append(r)
                print(f"rot={rot} side={side:+.3f} height={height:+.3f} "
                      f"peak={r['peak_lift_m']*1000:6.1f}mm "
                      f"clear={str(r['ever_left_support']):5s} "
                      f"off={r['steps_off_support']:3d} "
                      f"F={r['peak_finger_force_N']:5.2f}N "
                      f"held={r['held_fraction']:.2f} "
                      f"streak={r['hold_streak']}")

    best = max(rows, key=lambda r: r["peak_lift_m"])
    OUT.write_text(json.dumps({
        "mjcf_commit": "4508421",
        "box_centre_m": box.tolist(),
        "box_half_m": box_half.tolist(),
        "rotation_probe": [[list(r), dz] for r, dz in scored],
        "n_poses": len(rows),
        "n_ever_left_support": int(sum(r["ever_left_support"] for r in rows)),
        "best": best,
        "rows": rows,
    }, indent=2))
    print(f"\nclean lift in {sum(r['ever_left_support'] for r in rows)}/{len(rows)} poses -> {OUT}")


if __name__ == "__main__":
    main()
