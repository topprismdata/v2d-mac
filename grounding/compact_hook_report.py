"""Compact the current-scene hook sweep into a committable evidence file.

``collect_hook_demos.py`` writes one gate record per scripted trial, which for a
16,200-pose sweep is a 30 MB file. This reduces it to what the round actually
needs to be checkable later: the sweep totals, the distribution of peak lift,
the closest misses, and the A/B that isolates WHY the p0(4d) hook no longer
clears A3b.

The A/B is the load-bearing part. Replaying the SAME 339 recorded actions on
the scene that recorded them (4ee6f1d) and on the current scene (1b0f13f) turns
"the demo set went stale" into a measured claim: the same trajectory has the box
clean of the pedestal at its peak step on one scene and in contact with it on
the other.
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
SWEEP = RUN_DIR / "hook_demos.json"
AB = RUN_DIR / "hook_scene_ab.json"
OUT = RUN_DIR / "hook_feasibility_1b0f13f.json"
KEEP_TOP = 40


def main() -> None:
    sweep = json.loads(SWEEP.read_text())
    rows = sweep.get("rows", [])
    peaks = np.array([r["gate"]["peak_lift_m"] for r in rows], dtype=float)
    touching = np.array([bool(r["gate"]["table_touching_at_peak"]) for r in rows])
    duty = np.array([r["gate"]["finger_force_duty_cycle"] for r in rows])

    rows_sorted = sorted(rows, key=lambda r: -r["gate"]["peak_lift_m"])
    out = {
        "question": "does a scripted trajectory in the reachable hook family "
                    "still clear the frozen acceptance gates, including A3b "
                    "(airborne at peak), on the current scene?",
        "mjcf_commit": sweep.get("mjcf_commit"),
        "mjcf_sha256": sweep.get("mjcf_sha256"),
        "n_trials": sweep.get("n_trials"),
        "n_pass": sweep.get("n_pass"),
        "verdict": "no gate-clean reachable-class trajectory exists in this "
                   "family on this scene",
        "peak_lift_m_distribution": {
            "min": float(peaks.min()) if peaks.size else None,
            "median": float(np.median(peaks)) if peaks.size else None,
            "p90": float(np.percentile(peaks, 90)) if peaks.size else None,
            "max": float(peaks.max()) if peaks.size else None,
            "n_above_A1_bar_5cm": int(np.sum(peaks > 0.05)),
        },
        "A3b": {
            "table_touching_at_peak_count": int(touching.sum()),
            "table_touching_at_peak_fraction": float(touching.mean()) if touching.size else None,
        },
        "finger_force_duty_cycle": {
            "median": float(np.median(duty)) if duty.size else None,
            "max": float(duty.max()) if duty.size else None,
        },
        "closest_misses": [
            {k: r[k] for k in ("cfg", "offset_xy_m", "yaw_rad", "gate")}
            for r in rows_sorted[:KEEP_TOP]
        ],
        "supplementary_sweeps": {},
    }
    if AB.exists():
        out["scene_ab_same_actions"] = json.loads(AB.read_text())
    OUT.write_text(json.dumps(out, indent=2))
    SWEEP.unlink()   # 30 MB of per-trial gate records; the summary is the record
    print(json.dumps({k: v for k, v in out.items()
                      if k not in ("closest_misses", "scene_ab_same_actions",
                                   "supplementary_sweeps")}, indent=2))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
