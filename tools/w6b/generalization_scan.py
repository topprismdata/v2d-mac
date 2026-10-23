"""v10 preparation: the approach generalisation scan.

The v9 review required evidence that the side cell's approach solution is
not tied to ONE constructed path: "replay at least three different staging
heights or horizontal approach heights, and require the correct joint
branch, no table contact and a continuous trajectory in all of them".

This script runs the committed side-cell recipe with several approach
geometries (the staging's distance along the face normal and the home's
descent height) and checks, for each:

  * the path's own palm deviation (the path tracks its own target),
  * the branch continuity (the max |dq| between two consecutive commanded
    frames -- a 2*pi branch switch reads ~6.283 rad),
  * the table/support contacts during the approach (gate3),
  * the executed terminal distance and forward axis.

    runs/.venv-retarget/bin/python tools/w6b/generalization_scan.py
"""
from __future__ import annotations

import json
import os
import pathlib
import re
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
PIPE = REPO / "runs/.venv-retarget/bin/python"
SCRIPT = REPO / "retarget/retarget_dual.py"
SCENE = "grounding/assets/allegro_floating_ur5e_side_r37.xml"

BASE = {
    "W6B_SCENE": "ur5e",
    "W6B_SCENE_XML": SCENE,
    "W6B_R23_FLYOVER_MM": "250",
    "W6B_R22_MARGIN_MM": "8",
    "W6B_R75_SIDE_APPROACH": "1",
    "W6B_R63_TARGET": "1",
    "W6B_R18_NATIVE_TARGET": "1",
    "W6B_R50_CURL": "1.25",
    "W6B_R50_PRELOAD_MM": "0.05",
    "W6B_R39_TGC": "0",
    "W6B_R36_GSTATIC": "1",
    "W6B_R36_I": "0",
    "W6B_R39_HOLD": "60",
}

# (the staging's distance along the face normal mm, the home's descent
# height mm) -- four geometries: the delivered one and three variants
# that change BOTH the horizontal approach distance and the descent
CASES = [
    ("the delivered geometry", 100, 350),
    ("a closer approach", 60, 350),
    ("a further approach", 160, 350),
    ("a higher descent", 100, 550),
    ("a lower descent", 100, 220),
]


def run(extra: dict) -> str:
    env = dict(os.environ)
    env.update(BASE)
    env.update(extra)
    r = subprocess.run([str(PIPE), str(SCRIPT)], cwd=REPO, env=env,
                       capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(r.stdout[-2000:] + r.stderr[-2000:] + "\n")
        raise SystemExit("the pipeline failed")
    return r.stdout


def main() -> int:
    rows = []
    fails = []
    for name, stage, lift in CASES:
        out = run({"W6B_R75_STAGE_MM": str(stage),
                   "W6B_R91_HOME_LIFT_MM": str(lift)})
        rep = json.loads((REPO / "runs/w6b/report.json").read_text())
        for side in ("right", "left"):
            ps = rep["per_side"][side]
            dev = ps["gate5_palm_deviation"]["side_path_palm_deviation"]["max_m"] * 1e3
            step = ps["commanded_wrist_max_joint_step_rad"]
            sup = ps["gate3_approach_clean"]["support_contact_frames"]
            m = re.search(rf"{side}: cmd rect ([\d.]+) mm fwd ([\d.]+) deg .*?"
                          r"executed rect ([\d.]+) mm.*?fwd ([\d.]+)", out, re.S)
            ex = (float(m.group(3)), float(m.group(4))) if m else (float("nan"),) * 2
            rows.append((name, stage, lift, side, dev, step, sup, ex[0], ex[1]))
            # the reviewer's three requirements
            if dev > 20.0:
                fails.append(f"{name}/{side}: the path's deviation {dev:.1f} mm")
            if step > 0.35:
                fails.append(f"{name}/{side}: the branch step {step:.3f} rad")
            if sup != 0:
                fails.append(f"{name}/{side}: {sup} support-contact frames")
        print(f"  {name}: done")
    print()
    hdr = (f"{'geometry':24s} {'stage':>5s} {'lift':>5s} {'side':>5s} "
           f"{'path dev mm':>12s} {'branch step':>12s} {'support':>8s} "
           f"{'exec mm':>8s} {'fwd':>6s}")
    print(hdr)
    for r in rows:
        print(f"{r[0]:24s} {r[1]:5d} {r[2]:5d} {r[3]:>5s} {r[4]:12.2f} "
              f"{r[5]:12.4f} {r[6]:8d} {r[7]:8.2f} {r[8]:6.1f}")
    print()
    if fails:
        print("FAIL:")
        for f in fails:
            print("  ", f)
        return 1
    print("PASS: every approach geometry keeps the branch, stays clear of "
          "the table and tracks its own target")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
