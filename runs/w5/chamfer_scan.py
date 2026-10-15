"""W5: does a top-edge chamfer widen the zero-measure hook into a basin?

Baseline (20ca639, XML 1ea0fdf5): exactly 1 of 15,625 limit-covering poses
passes the frozen gates (peak 106.56 mm, duty 0.733, load decomposition
top=1.0 -- 100% of upward force through the top-face hook, 0% pedestal), and
its survival boundary is ZERO (33-cell perturbation probe: any x/y/yaw offset
kills it).  A finger balanced on a knife edge.

Ruling (PLAN, W4): add a 45-degree chamfer to the box's top +-x edges
(build_allegro_floating.build(chamfer_m=...), plate approximation documented
in box_chamfer_specs) and measure three things per chamfer value:

  V1  gate-clean trajectory count over the same limit-covering grid
      (chamfer=0 must reproduce the 1/15625 baseline -- the control)
  V2  survival boundary of the best trajectory (perturbation grid, mm/rad)
  V3  realness: the lift must route through the chamfer plates (contact pairs
      include object_chamfer_*), not through a new artifact

Gates are adjudicated by tests/test_smoke4_accept._analyse_episode verbatim
via grounding.reachable_scan.gate_eval; thresholds untouched.

    python runs/w5/chamfer_scan.py            # full: 5 chamfers x 15,625 grid
    python runs/w5/chamfer_scan.py --quick    # chamfer 0/5 mm, coarse grid
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import pathlib
import sys
import time

REPO = pathlib.Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import grounding.envs.lift_hold_env as lhe  # noqa: E402
import grounding.collect_hook_demos as chd  # noqa: E402
from grounding.reachable_scan import gate_eval, run_traj  # noqa: E402

OUT = REPO / "runs/w5/chamfer_scan.json"
ASSETS = REPO / "grounding/assets"
BARE_SHA = "1ea0fdf5c87ce5a161efcda7c597b0fc6647c9fe30bcf2f5d0bcb52402d24b89"

# The one trajectory that passed the frozen gates on the bare box.  Every
# chamfer value is probed at this pose AND across the full grid: the headline
# question is whether the basin around it stops being zero.
SEED_CFG_PATCH = {
    "wrist_world": [0.09, -0.09, 0.757],
    "rot": [1.8, -0.6, -1.5],
}

CHAMFERS_MM = (0.0, 3.0, 4.0, 5.0, 6.0)
OFFSETS_MM = chd.OFFSETS_MM
YAWS_RAD = chd.YAWS_RAD

def _init_worker(variant_xml: str) -> None:
    """Each spawn worker re-imports the module, so the variant XML must be
    re-pointed here, not just in the parent.  The parent builds the variant
    ONCE; workers only point at it -- building the same variant in every
    worker raced on the shared scratch path (empty-file ParseXML)."""
    lhe.MJCF = pathlib.Path(variant_xml)


def _materialize(chamfer_m: float) -> pathlib.Path:
    sys.path.insert(0, str(ASSETS))
    from build_allegro_floating import build
    p = build(chamfer_m=chamfer_m,
              out=ASSETS / f"_scan_chamfer_{int(chamfer_m * 1000)}.xml")
    sha = hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
    assert sha != BARE_SHA or chamfer_m == 0.0, (
        f"chamfer={chamfer_m} produced the bare-box XML -- the parameter did "
        "not reach the geometry")
    return p


def _score(job: tuple) -> dict:
    cfg, off, yaw = job
    from grounding.envs.lift_hold_env import LiftHoldEnv
    env = LiftHoldEnv(episode_len=400, domain_rand=False,
                      box_offset_xy=tuple(off), box_yaw=float(yaw))
    run_traj(env, cfg)
    ev = gate_eval(env)
    pairs = {tuple(p) for p in env.trace["contact_pairs"][-1]} if \
        env.trace["contact_pairs"] else set()
    env.close()
    chamfer_contact = any(
        "chamfer" in str(a) + str(b) for a, b in pairs)
    gate = {k: v for k, v in ev.items() if k != "finger_pairs"}
    # strict: the frozen gate's own composite verdict, never a re-derivation
    gate_ok = bool(gate["gate_pass"])
    return {"cfg": cfg, "offset_xy_m": list(off), "yaw_rad": float(yaw),
            "gate": gate, "gate_ok": gate_ok,
            "chamfer_in_contact": chamfer_contact,
            "peak_lift_m": float(gate.get("peak_lift_m", 0.0))}


def perturbation_jobs(best_cfg: dict) -> list[tuple]:
    return [(best_cfg, (mm / 1000.0, mm / 1000.0), yaw)
            for mm in OFFSETS_MM for yaw in YAWS_RAD]


def run_chamfer(chamfer_m: float, quick: bool) -> dict:
    variant = _materialize(chamfer_m)
    lhe.MJCF = variant
    t0 = time.time()
    cfgs = chd.search_grid(coarse=quick)
    # forced seed: the known passing pose from the bare box, always adjudicated
    seed = json.loads(json.dumps(cfgs[0]))
    seed.update(SEED_CFG_PATCH)
    if quick:
        cfgs = cfgs[:200]
    cfgs.insert(0, seed)
    # zero-offset jobs: the grid varies the pose, the placement is nominal
    jobs = [(c, (0.0, 0.0), 0.0) for c in cfgs]

    with mp.Pool(6, initializer=_init_worker,
                 initargs=(str(variant),)) as pool:
        rows = pool.map(_score, jobs, chunksize=8)

    passed = [r for r in rows if r["gate_ok"]]
    best = max(rows, key=lambda r: r["peak_lift_m"])

    # V2: survival boundary of the best trajectory (perturbation probe on the
    # pose closest to the seed if nothing passed, else on a passing pose)
    probe_cfg = (passed[0]["cfg"] if passed else
                 max(rows, key=lambda r: r["peak_lift_m"])["cfg"])
    probe_jobs = perturbation_jobs(probe_cfg)
    with mp.Pool(6, initializer=_init_worker,
                 initargs=(str(variant),)) as pool:
        prows = pool.map(_score, probe_jobs, chunksize=4)
    survived = [r for r in prows if r["gate_ok"]]
    boundary = {
        "survived_cells": len(survived),
        "of_cells": len(prows),
        "max_offset_mm": max((abs(r["offset_xy_m"][0]) + abs(r["offset_xy_m"][1]))
                             / 2 * 1000 for r in survived) if survived else 0.0,
        "max_yaw_rad": max(abs(r["yaw_rad"]) for r in survived) if survived else 0.0,
        "worst_cell": [{"off": r["offset_xy_m"], "yaw": r["yaw_rad"]}
                       for r in prows if not r["gate_ok"]][:3],
    }

    result = {
        "chamfer_mm": chamfer_m,
        "poses": len(rows),
        "gate_clean_count": len(passed),
        "baseline_reproduced": (chamfer_m != 0.0) or (len(passed) == 1),
        "best_peak_lift_m": best["peak_lift_m"],
        "best_gate_ok": best["gate_ok"],
        "best_load_decomposition": {
            k: v for k, v in best["gate"].items()
            if k in ("bearing_top", "bearing_wall", "bearing_pedestal",
                     "top", "wall", "pedestal", "load_decomposition")
        },
        "chamfer_in_contact_any": any(r["chamfer_in_contact"] for r in rows
                                      if r["gate_ok"]),
        "survival": boundary,
        "passing_poses": [
            {"wrist_world": r["cfg"]["wrist_world"], "rot": r["cfg"]["rot"],
             "peak_lift_m": r["peak_lift_m"],
             "chamfer_in_contact": r["chamfer_in_contact"],
             # full config: these trajectories are knife-edge sensitive, so an
             # audit replay needs the EXACT cfg, not a pose approximation
             "cfg": r["cfg"]}
            for r in passed
        ],
        "seconds": round(time.time() - t0, 1),
        "variant_sha256": hashlib.sha256(pathlib.Path(variant).read_bytes()).hexdigest(),
    }
    pathlib.Path(variant).unlink(missing_ok=True)
    return result

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", type=float, default=None, help="single chamfer, mm")
    args = ap.parse_args()
    if args.only is not None:
        chamfers = (args.only,)
    else:
        chamfers = (0.0, 5.0) if args.quick else CHAMFERS_MM
    results = [run_chamfer(c / 1000.0, args.quick) for c in chamfers]
    if args.only is not None and OUT.exists():
        merged = json.loads(OUT.read_text())
        by_mm = {round(r["chamfer_mm"] * 1000): r for r in merged["results"]}
        for r in results:
            by_mm[round(r["chamfer_mm"] * 1000)] = r
        merged["results"] = [by_mm[k] for k in sorted(by_mm)]
        OUT.write_text(json.dumps(merged, indent=2, default=str))
    else:
        OUT.write_text(json.dumps({
            "baseline_sha": BARE_SHA,
            "adjudicator": "tests/test_smoke4_accept._analyse_episode via gate_eval",
            "results": results,
        }, indent=2, default=str))
    print(f"\n{'chamfer':>8} {'clean':>6} {'best_mm':>8} {'survived':>9} "
          f"{'max_off_mm':>11}")
    for r in results:
        print(f"{r['chamfer_mm']:>8.1f} {r['gate_clean_count']:>6} "
              f"{r['best_peak_lift_m'] * 1000:>8.1f} "
              f"{r['survival']['survived_cells']:>4}/{r['survival']['of_cells']:<4} "
              f"{r['survival']['max_offset_mm']:>11.2f}")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
