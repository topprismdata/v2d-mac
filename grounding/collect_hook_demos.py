"""Task 4(e) part 2: re-collect the reachable-class demo set on MJCF 1b0f13f.

Why this exists. Round p0(4d) recorded ``reachable_demo_set.npz`` (678 frames,
2 trajectories) against MJCF ``4ee6f1d``. Wave 3.5F then stiffened the finger
PROXIMAL geoms (commit 1b0f13f), which changed the box<->finger contact
response exactly where the rim hook engages: penetration 0.40-1.19 mm became
0.05-0.10 mm and the in-frame peak reaction rose by up to 3x. A demo set whose
every frame was recorded against different physics is not a demo set - replaying
the old actions on the new scene no longer lifts the box clear (verified: peak
lift 53.3 mm but A3b table_touching_at_peak=True, so the box is still on the
pedestal at its own peak).

What this module does, in order:

  1. RE-SEARCH. Sweep the reachable hook family (the only family in p0(4d) that
     ever passed every frozen gate, including A3b) on the current scene, judged
     by tests/test_smoke4_accept._analyse_episode verbatim - a scan pass is a
     test pass by construction.
  2. MEASURE THE SURVIVAL BOUNDARY before choosing an amplitude. p0(4d) found
     the passing set dies at +-5 mm of placement error and that the committed
     domain_rand protocol (+-1 cm, +-0.1 rad) passes 0/20; the adjudication for
     this round is explicit that +-1 cm must NOT be reused. So the perturbation
     grid is swept per amplitude and the largest amplitude that still passes is
     what gets collected - the boundary is a measurement, not an assumption.
  3. COLLECT. Every surviving (cfg, perturbation) cell is recorded as a D4
     trajectory, so the BC set spans the placement/yaw envelope the hook actually
     has measure in, rather than a single knife-edge pose.

    python grounding/collect_hook_demos.py --workers 8
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

RUN_DIR = REPO / "runs/p0/smoke4_mujoco_ppo"
OUT_NPZ = RUN_DIR / "reachable_demo_set.npz"
OUT_JSON = RUN_DIR / "hook_demos.json"
MJCF_COMMIT = "1b0f13f"
MJCF_SHA256 = "1ea0fdf5c87ce5a161efcda7c597b0fc6647c9fe30bcf2f5d0bcb52402d24b89"

# hook trajectory shape that passed on 4ee6f1d - the starting neighbourhood
BASE_CFG = {
    "family": "reachable_hook",
    "wrist_world": [-0.02, -0.02, 0.80],
    "rot": [1.8, 0.6, 0.0],
    "close": [0.3, 1.0, 1.0, 1.0, 0.3, 1.0, 1.0, 1.0,
              0.3, 1.0, 1.0, 1.0, 0.0, -0.6, -0.6, -0.6],
    "close_params": [1.0, -0.6, 0.0],
    "sequence": "descend_then_close",
    "close_ramp": True,
    "n_transit": 30, "n_desc": 60, "n_close": 80,
    "settle": 25, "lift_z": 0.16, "lift_rate": 0.004, "lift_hold": 80,
}

# perturbation grid, in millimetres and radians. Deliberately FINER and SMALLER
# than the committed domain_rand protocol: the adjudication for this round says
# do not reuse +-1 cm / +-0.1 rad, so the grid probes 0-6 mm / 0-0.06 rad and
# the survival boundary is read off the result.
OFFSETS_MM = (0.0, 1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 4.0, -4.0, 5.0, -5.0, 6.0, -6.0)
YAWS_RAD = (0.0, 0.02, -0.02, 0.04, -0.04, 0.06, -0.06)

_ENV: dict = {}


def make_env(box_offset_xy=(0.0, 0.0), box_yaw=0.0, episode_len: int = 400):
    from grounding.envs.lift_hold_env import LiftHoldEnv
    return LiftHoldEnv(episode_len=episode_len, domain_rand=False,
                       box_offset_xy=box_offset_xy, box_yaw=box_yaw)


def _score(job: tuple) -> dict:
    cfg, off, yaw = job
    from grounding.reachable_scan import gate_eval, run_traj
    env = make_env(off, yaw)
    run_traj(env, cfg)
    ev = gate_eval(env)
    env.close()
    return {"cfg": cfg, "offset_xy_m": list(off), "yaw_rad": float(yaw),
            "gate": {k: v for k, v in ev.items() if k != "finger_pairs"}}


def search_grid(coarse: bool, wide: bool = False) -> list[dict]:
    """Neighbourhood sweep of the hook family on the CURRENT scene.

    ``wide`` adds the timing axes the first sweep held at their p0(4d) values
    (descent depth, settle, lift rate). They matter because the Wave 3.5F
    stiffening changed the box<->finger response mid-close: a trajectory that
    used to clear the rim with one descent depth may now need a slower or
    shallower one, and holding the timings fixed would report that as "no hook
    exists on this scene" when it only means "no hook exists at THESE timings".
    """
    xs = (-0.02,) if coarse else (-0.04, -0.03, -0.02, -0.01, 0.0, 0.01)
    ys = (-0.02,) if coarse else (-0.04, -0.03, -0.02, -0.01, 0.0, 0.01)
    zs = (0.80,) if coarse else ((0.76, 0.77, 0.78, 0.79, 0.80, 0.81, 0.82) if wide
                                 else (0.78, 0.79, 0.80, 0.81, 0.82, 0.83))
    rys = (0.6,) if coarse else ((0.4, 0.5, 0.6, 0.7, 0.8) if wide
                                 else (0.4, 0.5, 0.6, 0.7, 0.8))
    rxs = (1.8,) if coarse else ((1.6, 1.7, 1.8, 1.9, 2.0) if wide
                                 else (1.7, 1.8, 1.9))
    cts = (-0.6,) if coarse else ((-0.3, -0.4, -0.5, -0.6, -0.7, -0.8) if wide
                                  else (-0.4, -0.5, -0.6, -0.7, -0.8))
    n_descs = (60,) if coarse else ((40, 60, 80) if wide else (60,))
    lift_rates = (0.004,) if coarse else ((0.002, 0.004) if wide else (0.004,))
    cfgs = []
    for x in xs:
        for y in ys:
            for z in zs:
                for rx in rxs:
                    for ry in rys:
                        for ct in cts:
                            for nd in n_descs:
                                for lr in lift_rates:
                                    c = json.loads(json.dumps(BASE_CFG))
                                    c["wrist_world"] = [x, y, z]
                                    c["rot"] = [rx, ry, 0.0]
                                    a = list(c["close"])
                                    a[12:16] = [0.0, ct, ct, ct]
                                    c["close"] = a
                                    c["close_params"] = [1.0, ct, 0.0]
                                    c["n_desc"] = nd
                                    c["lift_rate"] = lr
                                    cfgs.append(c)
    return cfgs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--coarse", action="store_true")
    ap.add_argument("--wide", action="store_true",
                    help="also sweep descent depth and lift rate (large grid)")
    ap.add_argument("--max-demos", type=int, default=40,
                    help="cap on recorded trajectories (339 frames each)")
    ap.add_argument("--min-frames", type=int, default=3390,
                    help="adjudication floor: >= 5x the old 678-frame set")
    args = ap.parse_args()

    from grounding.reachable_scan import (  # noqa: PLC0415
        contact_load_decomposition, gate_eval, run_traj,
    )

    # ---- step 1: re-search on the current scene ---------------------------
    cfgs = search_grid(args.coarse, args.wide)
    jobs = [(c, (0.0, 0.0), 0.0) for c in cfgs]
    print(f"[search] {len(jobs)} hook poses on MJCF {MJCF_COMMIT}", flush=True)
    with mp.Pool(args.workers) as pool:
        rows = pool.map(_score, jobs, chunksize=4)
    passing = [r for r in rows if r["gate"]["gate_pass"]]
    passing.sort(key=lambda r: -r["gate"]["peak_lift_m"])
    print(f"[search] {len(passing)}/{len(rows)} pass all frozen gates incl. A3b; "
          f"best lift {max((r['gate']['peak_lift_m'] for r in rows), default=0)*1000:.1f}mm",
          flush=True)
    if not passing:
        print("[search] no A3b-clean hook pose exists on this scene - refusing to "
              "record a demo set of trajectories that cannot pass the gate",
              flush=True)
        OUT_JSON.write_text(json.dumps(
            {"mjcf_commit": MJCF_COMMIT, "mjcf_sha256": MJCF_SHA256,
             "n_trials": len(rows), "n_pass": 0, "rows": rows}, indent=2))
        return

    # ---- step 2: measure the perturbation survival boundary ---------------
    # Keep the strongest poses so the boundary probe is not dominated by weak
    # cells; the boundary is then read per-pose and the kept envelope is the
    # amplitude that survives for EVERY pose that survives at zero.
    seeds = passing[: min(6, len(passing))]
    probe = []
    for r in seeds:
        for mm in OFFSETS_MM:
            probe.append((r["cfg"], (mm / 1000.0, 0.0), 0.0))
        for mm in OFFSETS_MM:
            probe.append((r["cfg"], (0.0, mm / 1000.0), 0.0))
        for y in YAWS_RAD:
            probe.append((r["cfg"], (0.0, 0.0), y))
    print(f"[dagger] probing {len(probe)} perturbed cells "
          f"({len(seeds)} poses x {2*len(OFFSETS_MM) + len(YAWS_RAD)} cells)",
          flush=True)
    with mp.Pool(args.workers) as pool:
        brows = pool.map(_score, probe, chunksize=4)

    by_pose: dict[int, dict] = {}
    for b in brows:
        key = json.dumps(b["cfg"], sort_keys=True)
        s = by_pose.setdefault(key, {"cfg": b["cfg"], "x_mm": 0, "y_mm": 0,
                                     "yaw_deg": 0, "pass_x": set(),
                                     "pass_y": set(), "pass_yaw": set()})
        if not b["gate"]["gate_pass"]:
            continue
        dx, dy = b["offset_xy_m"]
        if dy == 0.0 and dx != 0.0:
            s["pass_x"].add(round(dx * 1000.0, 3))
        elif dx == 0.0 and dy != 0.0:
            s["pass_y"].add(round(dy * 1000.0, 3))
        elif dx == 0.0 and dy == 0.0 and b["yaw_rad"] != 0.0:
            s["pass_yaw"].add(round(float(np.degrees(b["yaw_rad"])), 3))

    def _span(vals: set[float]) -> float:
        """Largest SYMMETRIC half-width present in ``vals``.

        Symmetric on purpose: the demo set is used to warm-start a policy that
        will be evaluated over the whole placement distribution, so a one-sided
        pass (e.g. +4 mm survives but -4 mm does not) is not evidence that the
        amplitude has measure. A direction that survives only one way counts as
        zero width.
        """
        best = 0.0
        for m in sorted({round(abs(v), 3) for v in vals if abs(v) > 1e-9}):
            if m in vals and -m in vals:
                best = m
        return best

    # The envelope is the INTERSECTION over the seed poses: an amplitude is only
    # usable for BC if every seed pose still passes at it, otherwise half the
    # recorded trajectories are knife-edge cells that teach the policy to hope.
    boundary = {}
    for out_key, key in (("offset_x_mm", "pass_x"), ("offset_y_mm", "pass_y"),
                         ("yaw_deg", "pass_yaw")):
        spans = [_span(s[key]) for s in by_pose.values()]
        boundary[out_key] = float(min(spans)) if spans else 0.0
    print(f"[dagger] survival boundary: x +-{boundary['offset_x_mm']:.1f}mm, "
          f"y +-{boundary['offset_y_mm']:.1f}mm, yaw +-{boundary['yaw_deg']:.2f} deg",
          flush=True)

    # per-pose spans, kept for the report
    per_pose = []
    for s in by_pose.values():
        per_pose.append({
            "wrist_world": s["cfg"]["wrist_world"], "rot": s["cfg"]["rot"],
            "close_params": s["cfg"]["close_params"],
            "offset_x_mm": _span(s["pass_x"]),
            "offset_y_mm": _span(s["pass_y"]),
            "yaw_deg": _span(s["pass_yaw"]),
        })
    per_pose.sort(key=lambda r: -(r["offset_x_mm"] + r["offset_y_mm"]
                                  + 10.0 * r["yaw_deg"]))

    # ---- step 3: collect ------------------------------------------------
    # Amplitude is the MEASURED boundary, scaled down one grid step: collecting
    # at exactly the boundary collects knife-edge cells whose every frame is a
    # coin flip. Envelope is clipped so no cell exceeds what was measured.
    amp_x = max(0.0, boundary["offset_x_mm"] - 1.0) / 1000.0
    amp_y = max(0.0, boundary["offset_y_mm"] - 1.0) / 1000.0
    amp_yaw = float(np.radians(max(0.0, boundary["yaw_deg"] - 0.5)))
    print(f"[collect] using amplitude x +-{amp_x*1000:.1f}mm y +-{amp_y*1000:.1f}mm "
          f"yaw +-{np.degrees(amp_yaw):.2f}deg (boundary minus one grid step)",
          flush=True)

    cells = [(0.0, 0.0, 0.0)]
    for sx in (-1.0, 0.0, 1.0):
        for sy in (-1.0, 0.0, 1.0):
            for k in (0.0, 0.5, 1.0):
                dx, dy = sx * amp_x * k, sy * amp_y * k
                if dx == 0.0 and dy == 0.0:
                    continue
                cells.append((dx, dy, 0.0))
    for sy in (-1.0, 1.0):
        cells.append((0.0, 0.0, sy * amp_yaw))

    collect_jobs = []
    for r in seeds:
        for dx, dy, yaw in cells:
            collect_jobs.append((r["cfg"], (dx, dy), yaw))
    print(f"[collect] verifying {len(collect_jobs)} cells and recording the "
          f"passing ones", flush=True)
    with mp.Pool(args.workers) as pool:
        crows = pool.map(_score, collect_jobs, chunksize=4)
    ok = [c for c in crows if c["gate"]["gate_pass"]]

    obs_all, act_all, lens, poses, gates, cfgs_out = [], [], [], [], [], []
    decomps = []
    for c in ok[: args.max_demos]:
        env = make_env(tuple(c["offset_xy_m"]), c["yaw_rad"])
        obs, act = run_traj(env, c["cfg"], record=True)
        ev = gate_eval(env)
        if not ev["gate_pass"]:
            env.close()
            continue
        decomps.append(contact_load_decomposition(env, act, ev["peak_step"]))
        env.close()
        obs_all.append(obs)
        act_all.append(act)
        lens.append(len(obs))
        poses.append([*c["cfg"]["wrist_world"], *c["cfg"]["rot"],
                      *c["cfg"]["close_params"]])
        gates.append({k: v for k, v in ev.items() if k != "finger_pairs"})
        cfgs_out.append({**c["cfg"], "offset_xy_m": c["offset_xy_m"],
                         "yaw_rad": c["yaw_rad"]})

    n_frames = int(sum(lens))
    if not lens:
        raise SystemExit("no perturbation cell passed the frozen gate; "
                         "refusing to emit an empty demo set")
    print(f"[collect] {len(lens)} trajectories, {n_frames} frames "
          f"(floor {args.min_frames})", flush=True)

    np.savez_compressed(
        OUT_NPZ,
        bc_obs=np.concatenate(obs_all).astype(np.float32),
        bc_act=np.concatenate(act_all).astype(np.float32),
        bc_pose=np.asarray(poses, dtype=np.float32),
        bc_trajectory_len=np.asarray(lens, dtype=np.int32),
        mjcf_commit=np.array(MJCF_COMMIT),
        mjcf_sha256=np.array(MJCF_SHA256),
        selection=np.array("reachable_class_clean_dagger"),
        cfg_json=np.array(json.dumps(cfgs_out)),
        gate_metrics_json=np.array(json.dumps(gates)),
        load_decomposition_json=np.array(json.dumps(decomps)),
    )
    OUT_JSON.write_text(json.dumps({
        "mjcf_commit": MJCF_COMMIT, "mjcf_sha256": MJCF_SHA256,
        "n_trials": len(rows), "n_pass": len(passing),
        "search_grid": len(cfgs),
        "n_dagger_probes": len(brows),
        "survival_boundary": boundary,
        "collection_amplitude": {"x_mm": amp_x * 1000, "y_mm": amp_y * 1000,
                                 "yaw_deg": float(np.degrees(amp_yaw))},
        "n_trajectories": len(lens), "n_frames": n_frames,
        "old_set_frames": 678,
        "per_pose_boundary": per_pose,
        "gates": gates,
        "load_decomposition": decomps,
    }, indent=2))
    print(f"-> {OUT_NPZ} ({n_frames} frames) and {OUT_JSON}")


if __name__ == "__main__":
    main()
