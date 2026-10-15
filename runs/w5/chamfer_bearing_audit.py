"""W5 follow-up: what carries the load in the chamfer scan's passing cells?

The scan's V3 check (chamfer plates in the final-step contact pairs) came back
False for every passing trajectory, which makes the V1/V2 numbers uninterpretable:
a lift whose load path is unknown cannot be declared a real grasp.  This script
replays the surviving cells of one chamfer value with
`record_bearing=True` (sub-step sampling, Smoke4Reward's three-column caliber)
and dumps, at the PEAK-LIFT step:
  - the exact contact pairs (which geom carries)
  - the bearing decomposition (hook / wall / pedestal), hold + peak_control + peak_substep
  - whether the chamfer plates appear at peak

    python runs/w5/chamfer_bearing_audit.py --chamfer 3 --cells 27
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import grounding.envs.lift_hold_env as lhe  # noqa: E402
import grounding.collect_hook_demos as chd  # noqa: E402
from grounding.reachable_scan import gate_eval, run_traj  # noqa: E402

SCAN = REPO / "runs/w5/chamfer_scan.json"
OUT = REPO / "runs/w5/chamfer_bearing_audit.json"


def _materialize(chamfer_m: float) -> pathlib.Path:
    sys.path.insert(0, str(REPO / "grounding/assets"))
    from build_allegro_floating import build
    return build(chamfer_m=chamfer_m,
                 out=REPO / "grounding/assets/_audit_chamfer.xml")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chamfer", type=float, default=3.0, help="mm")
    args = ap.parse_args()
    chamfer_m = args.chamfer / 1000.0

    scan = json.loads(SCAN.read_text())
    round_ = next(r for r in scan["results"]
                  if abs(r["chamfer_mm"] * 1000 - args.chamfer) < 1e-6)
    surv = round_["survival"]
    seed_pose = round_["passing_poses"][0]

    variant = _materialize(chamfer_m)
    lhe.MJCF = variant

    rows = []
    # replay every cell of the full perturbation grid around the best passing
    # pose -- using the EXACT recorded cfg.  These trajectories are knife-edge
    # sensitive (a curl of -0.6 vs -0.7 flips them), so approximating the cfg
    # from BASE_CFG measures a different trajectory and proves nothing.
    passing = [p for p in round_["passing_poses"] if "cfg" in p]
    if not passing:
        raise SystemExit("scan predates full-cfg recording; re-run the scan")
    best_pose = max(passing, key=lambda p: p["peak_lift_m"])
    cfg_base = json.loads(json.dumps(best_pose["cfg"]))

    for mm in chd.OFFSETS_MM:
        for yaw in chd.YAWS_RAD:
            from grounding.envs.lift_hold_env import LiftHoldEnv
            env = LiftHoldEnv(episode_len=400, domain_rand=False,
                              box_offset_xy=(mm / 1000.0, mm / 1000.0),
                              box_yaw=float(yaw))
            env.record_bearing = True
            obs, act = run_traj(env, cfg_base, record=True)
            ev = gate_eval(env)
            peak_i = int(ev["peak_step"])
            pairs_at_peak = {tuple(p) for p in env.trace["contact_pairs"][peak_i]}
            row = {
                "off_mm": mm, "yaw_rad": yaw,
                "gate_pass": bool(ev["gate_pass"]),
                "peak_lift_mm": ev["peak_lift_m"] * 1000,
                "chamfer_at_peak": any("chamfer" in f"{a}{b}"
                                       for a, b in pairs_at_peak),
                "pairs_at_peak": sorted(f"{a}|{b}" for a, b in pairs_at_peak),
            }
            from grounding.reachable_scan import contact_load_decomposition
            row["load_decomposition"] = contact_load_decomposition(
                env, act, ev["peak_step"])
            rows.append(row)
            env.close()

    pathlib.Path(variant).unlink(missing_ok=True)
    n_pass = sum(r["gate_pass"] for r in rows)
    out = {
        "chamfer_mm": args.chamfer,
        "cells": len(rows),
        "gate_pass_cells": n_pass,
        "chamfer_at_peak_cells": sum(r["chamfer_at_peak"] for r in rows),
        "pair_names_at_peak": sorted({p for r in rows for p in r["pairs_at_peak"]}),
        "rows": rows,
    }
    OUT.write_text(json.dumps(out, indent=2, default=str))
    print(f"cells={len(rows)} gate_pass={n_pass} "
          f"chamfer_at_peak={out['chamfer_at_peak_cells']}")
    print("pair names at peak:", out["pair_names_at_peak"])
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
