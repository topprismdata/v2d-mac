"""Hook-basin tolerance scan: the one gate-passing cell (-0.02,-0.02,0.80,
rot 1.8/0.6/0, ct -0.6) is a thumb-hook-over-the-rim lift. A hook basin worth
using for BC+PPO must survive +-5 mm placement error (the formal eval jitters
the box +-10 mm). This scan maps the basin with 5 mm steps and flags cells
whose 4-neighbours all pass."""
from __future__ import annotations

import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

REPO = Path("/Users/guohongbin/v2d-mac")
sys.path.insert(0, str(REPO))

from runs.w35.rscan_wedge_scan import run_cell  # noqa: E402

OUT = REPO / "runs/w35/rscan_hook_basin.json"


def work(job):
    cell, ct = job
    try:
        r = run_cell(cell, lift_z=0.16, lift_rate=0.004,
                     close=(1.0, ct, 0.0))
        r["ct"] = ct
        return r
    except Exception as e:  # noqa: BLE001
        return {"cell": list(cell), "ct": ct, "error": str(e)}


def main() -> None:
    jobs = []
    for dx in (-0.01, -0.005, 0.0, 0.005, 0.01):
        for dy in (-0.01, -0.005, 0.0, 0.005, 0.01):
            for dz in (-0.005, 0.0, 0.005):
                for dry in (-0.15, -0.075, 0.0, 0.075, 0.15):
                    for ct in (-0.6, -0.8, -1.0):
                        for rx in (1.75, 1.8):
                            jobs.append((
                                (-0.02 + dx, -0.02 + dy, 0.80 + dz,
                                 rx, 0.6 + dry, 0.0), ct))
    with mp.Pool(8) as pool:
        rows = pool.map(work, jobs, chunksize=2)
    ok = [r for r in rows if "error" not in r]
    ok.sort(key=lambda r: (not r["gate"]["gate_pass"], -r["gate"]["peak_lift_m"]))
    OUT.write_text(json.dumps({"n": len(rows), "rows": ok}, indent=2))

    passing = [r for r in ok if r["gate"]["gate_pass"]]
    print(f"jobs {len(ok)}  GATE PASS {len(passing)} -> {OUT}")
    # basin map in the xy plane at the anchor z/rot
    grid = {}
    for r in ok:
        if (r["cell"][2] == 0.80 and r["cell"][3] == 1.8
                and abs(r["cell"][4] - 0.6) < 1e-9 and r["ct"] == -0.6):
            grid[(round(r["cell"][0], 3), round(r["cell"][1], 3))] = (
                r["gate"]["gate_pass"], round(r["gate"]["peak_lift_m"] * 1000, 1))
    for y in (0.01, 0.005, 0.0, -0.005, -0.01):
        line = []
        for x in (-0.01, -0.005, 0.0, 0.005, 0.01):
            p = grid.get((round(-0.02 + x, 3), round(-0.02 + y, 3)))
            line.append("  ----  " if p is None else
                        f"{'PASS' if p[0] else f'{p[1]:4.0f}mm':>7}")
        print(f"  y={y:+.3f}: " + " ".join(line))
    for r in passing[:10]:
        g = r["gate"]
        print(f"  xy=({r['cell'][0]:+.3f},{r['cell'][1]:+.3f}) z={r['cell'][2]:.3f} "
              f"rot=({r['cell'][3]},{r['cell'][4]:.2f},{r['cell'][5]}) ct={r['ct']} "
              f"peak={g['peak_lift_m']*1000:5.1f} duty={g['finger_force_duty_cycle']:.2f} "
              f"anch={g['max_anchored_step_m']*1000:.1f} clean={not g['table_touching_at_peak']}")


if __name__ == "__main__":
    main()
