"""p0(4) final slice: re-collect the gate-clean demo set on the COMMITTED chamfer scene.

Why this exists (drift found, not assumed).  runs/w5/chamfer_scan.json and
runs/w5/chamfer_bearing_audit.json measured their 27/91 survival boundary on the
pre mass-precision-fix chamfer XML (variant sha256 2179ec0b..., 6-decimal mass
fields).  Commit c7c6dac re-derived the scene with 9-decimal masses
(git show HEAD:grounding/assets/allegro_floating.xml | shasum -a 256 =
d61dc0de...).  Replaying the scan's stored grid-winner cfg on the committed
scene peaks at 47.686 mm -- BELOW the 50 mm A1 bar -- so the old 27/91 verdict
does not carry over.  This script re-measures the full 91-cell perturbation
grid on the committed XML, and records EVERY gate-passing cell as a D4
trajectory.  No cell is dropped or added by hand; the survivor set is the
measurement.

    runs/.venv-mj/bin/python runs/p0/smoke4_mujoco_ppo/collect_chamfer_demos.py
"""
from __future__ import annotations

import json
import multiprocessing as mp
import pathlib
import subprocess
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

RUN_DIR = REPO / "runs/p0/smoke4_mujoco_ppo"
SCAN = REPO / "runs/w5/chamfer_scan.json"
OUT_NPZ = RUN_DIR / "reachable_demo_set_c7c6dac.npz"
OUT_RECHECK = RUN_DIR / "chamfer91_recheck.json"
MJCF = REPO / "grounding/assets/allegro_floating.xml"
EPISODE_LEN = 400

# the committed chamfer scene (c7c6dac): 3 mm top-edge chamfer + 9dp masses
MJCF_COMMIT = "c7c6dac"

from grounding.collect_hook_demos import OFFSETS_MM, YAWS_RAD  # noqa: E402


def _xml_sha() -> str:
    raw = subprocess.run(
        ["git", "show", "HEAD:grounding/assets/allegro_floating.xml"],
        cwd=REPO, check=True, capture_output=True).stdout
    import hashlib
    return hashlib.sha256(raw).hexdigest()


def _score(job: tuple) -> dict:
    """One perturbation cell, judged by the frozen gate machinery verbatim."""
    cfg, mm, yaw = job
    from grounding.envs.lift_hold_env import LiftHoldEnv
    from grounding.reachable_scan import gate_eval, run_traj
    env = LiftHoldEnv(episode_len=EPISODE_LEN, domain_rand=False,
                      box_offset_xy=(mm / 1000.0, mm / 1000.0),
                      box_yaw=float(yaw))
    run_traj(env, cfg)
    ev = gate_eval(env)
    env.close()
    return {"off_mm": float(mm), "yaw_rad": float(yaw),
            "gate_pass": bool(ev["gate_pass"]),
            "peak_lift_mm": round(ev["peak_lift_m"] * 1000, 4),
            "n_finger_contact_steps": int(ev["n_finger_contact_steps"]),
            "finger_force_duty_cycle": round(float(ev["finger_force_duty_cycle"]), 4),
            "finger_force_mean_N": round(float(ev["finger_force_mean_N"]), 4),
            "table_touching_at_peak": bool(ev["table_touching_at_peak"]),
            "max_anchored_step_m": float(ev["max_anchored_step_m"]),
            "max_free_fall_gravity_error": float(ev["max_free_fall_gravity_error"])}


def main() -> None:
    from grounding.envs.lift_hold_env import LiftHoldEnv
    from grounding.reachable_scan import (
        contact_load_decomposition, gate_eval, run_traj)

    scan = json.loads(SCAN.read_text())
    round_ = next(r for r in scan["results"]
                  if abs(r["chamfer_mm"] - 0.003) < 1e-9)
    cfg_base = round_["passing_poses"][0]["cfg"]
    xml_sha = _xml_sha()
    assert xml_sha.startswith("d61dc0de"), \
        f"committed chamfer XML drifted again: {xml_sha}"

    # ---- 91-cell survival re-probe on the COMMITTED scene -----------------
    jobs = [(cfg_base, mm, yaw) for mm in OFFSETS_MM for yaw in YAWS_RAD]
    print(f"[recheck] {len(jobs)} cells on {MJCF_COMMIT} ({xml_sha[:12]}...)",
          flush=True)
    with mp.Pool(8) as pool:
        rows = pool.map(_score, jobs, chunksize=2)
    survivors = [r for r in rows if r["gate_pass"]]
    print(f"[recheck] {len(survivors)}/{len(rows)} cells pass the frozen gate "
          f"(pre-fix audit: 27/91)", flush=True)
    OUT_RECHECK.write_text(json.dumps({
        "mjcf_commit": MJCF_COMMIT,
        "xml_sha256": xml_sha,
        "note": ("re-measured on the committed c7c6dac scene (9dp masses); the "
                 "scan/audit 27/91 was measured on the 6dp predecessor "
                 "(2179ec0b). The knife-edge grid shifts under the mass-"
                 "precision fix; this file is the boundary of record."),
        "base_cfg": cfg_base,
        "cells": rows,
        "n_pass": len(survivors),
        "n_cells": len(rows),
    }, indent=2))

    # ---- record EVERY surviving cell (no hand-picking) --------------------
    obs_all, act_all, lens, poses, gates, cfgs_out, decomps = [], [], [], [], [], [], []
    dropped = []
    for r in survivors:
        env = LiftHoldEnv(episode_len=EPISODE_LEN, domain_rand=False,
                          box_offset_xy=(r["off_mm"] / 1000.0,
                                         r["off_mm"] / 1000.0),
                          box_yaw=float(r["yaw_rad"]))
        obs, act = run_traj(env, cfg_base, record=True)
        ev = gate_eval(env)  # re-adjudicate the recorded trajectory itself
        if not ev["gate_pass"]:
            dropped.append({"off_mm": r["off_mm"], "yaw_rad": r["yaw_rad"],
                            "why": "recorded replay failed the frozen gate"})
            env.close()
            continue
        decomps.append(contact_load_decomposition(env, act, ev["peak_step"]))
        env.close()
        obs_all.append(obs.astype(np.float32))
        act_all.append(act.astype(np.float32))
        lens.append(len(obs))
        poses.append([*cfg_base["wrist_world"], *cfg_base["rot"],
                      *cfg_base["close_params"]])
        gates.append({k: v for k, v in ev.items() if k != "finger_pairs"})
        cfgs_out.append({**cfg_base,
                         "offset_xy_m": [r["off_mm"] / 1000.0,
                                         r["off_mm"] / 1000.0],
                         "yaw_rad": float(r["yaw_rad"])})

    n_frames = int(sum(lens))
    if not lens:
        raise SystemExit("no surviving cell survived the recorded-replay "
                         "re-check; refusing to emit an empty demo set")
    print(f"[collect] {len(lens)} trajectories, {n_frames} frames"
          + (f" ({len(dropped)} dropped on re-check)" if dropped else ""),
          flush=True)

    np.savez_compressed(
        OUT_NPZ,
        bc_obs=np.concatenate(obs_all),
        bc_act=np.concatenate(act_all),
        bc_pose=np.asarray(poses, dtype=np.float32),
        bc_trajectory_len=np.asarray(lens, dtype=np.int32),
        mjcf_commit=np.array(MJCF_COMMIT),
        mjcf_sha256=np.array(xml_sha),
        selection=np.array("chamfer3mm_gate_clean_cells"),
        cfg_json=np.array(json.dumps(cfgs_out)),
        gate_metrics_json=np.array(json.dumps(gates)),
        load_decomposition_json=np.array(json.dumps(decomps)),
    )
    print(f"-> {OUT_NPZ.name}: {len(lens)} trajs / {n_frames} frames, "
          f"per-cell lifts "
          f"{[g['peak_lift_m'] and round(g['peak_lift_m']*1000,1) for g in gates]}",
          flush=True)


if __name__ == "__main__":
    main()
