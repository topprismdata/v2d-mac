"""Local refinement around the wedge-scan hits (gate pass at (-0.02,-0.02,0.8)
rot 1.8/0.6/0 and the 97 mm near-miss at (-0.06,-0.10,0.83) rot 1.8/-0.6/-0.6).
Uses the COMPRESSED phase profile (fits the 250-step formal eval episode) so a
refinement hit is directly recordable."""
from __future__ import annotations

import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

REPO = Path("/Users/guohongbin/v2d-mac")
sys.path.insert(0, str(REPO))

from runs.w35.rscan_wedge_scan import run_cell  # noqa: E402  (reuses full profile)

OUT = REPO / "runs/w35/rscan_refine.json"


def run_compressed(args):
    cell, lift_z, lift_rate, ct = args
    x, y, z, rx, ry, rz = cell
    # compressed phases via the same machinery, injected through run_cell's
    # module constants is invasive; call run_cell with a lifted budget and
    # accept the +80 lift tail (the recorded demo compresses phases separately
    # and is re-verified by the replay test before use).
    r = run_cell(cell, lift_z=lift_z, lift_rate=lift_rate, close=(1.0, ct, 0.0))
    r["lift"] = [lift_z, lift_rate]
    r["ct"] = ct
    return r


def main() -> None:
    jobs = []
    anchors = [
        (-0.02, -0.02, 0.80, 1.8, 0.6, 0.0),
        (-0.06, -0.10, 0.83, 1.8, -0.6, -0.6),
        (-0.10, -0.06, 0.86, 1.2, 1.2, 1.2),
        (-0.10, -0.10, 0.83, 1.8, 0.6, 1.2),
    ]
    for a in anchors:
        for dx in (-0.01, 0.0, 0.01):
            for dy in (-0.01, 0.0, 0.01):
                for dz in (-0.01, 0.0, 0.01):
                    for dry in (-0.1, 0.0, 0.1):
                        for lift_z, lift_rate in ((0.16, 0.004), (0.20, 0.004)):
                            for ct in (-0.6, -0.4):
                                jobs.append((
                                    (a[0] + dx, a[1] + dy, a[2] + dz,
                                     a[3], a[4] + dry, a[5]),
                                    lift_z, lift_rate, ct))
    with mp.Pool(8) as pool:
        rows = pool.map(run_compressed, jobs, chunksize=2)
    ok = [r for r in rows if "error" not in r]
    ok.sort(key=lambda r: (not r["gate"]["gate_pass"],
                           -min(r["gate"]["peak_lift_m"] / 0.001, 80.0)))
    OUT.write_text(json.dumps({"n": len(rows), "rows": ok}, indent=2))
    passing = [r for r in ok if r["gate"]["gate_pass"]]
    print(f"jobs {len(ok)}  GATE PASS {len(passing)} -> {OUT}")
    for r in passing[:15]:
        g = r["gate"]
        print(f"  xy=({r['cell'][0]:+.3f},{r['cell'][1]:+.3f}) z={r['cell'][2]:.3f} "
              f"rot=({r['cell'][3]},{r['cell'][4]:.1f},{r['cell'][5]}) "
              f"lift={r['lift']} ct={r['ct']} "
              f"peak={g['peak_lift_m']*1000:6.1f}mm duty={g['finger_force_duty_cycle']:.2f} "
              f"meanF={g['finger_force_mean_N']:5.2f} clean={not g['table_touching_at_peak']} "
              f"anch={g['max_anchored_step_m']*1000:.1f}mm")


if __name__ == "__main__":
    main()
