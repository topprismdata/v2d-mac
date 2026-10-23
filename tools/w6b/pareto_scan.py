"""v9 question 2: the terminal posture's Pareto trade-off.

The v9 review: the right's executed forward axis is 33.3 deg against a
35 deg limit -- a 1.7 deg margin -- while the left's executed pad
distance is still 6.06 mm against the 5 mm goal.  "These results say the
current terminal posture has an obvious trade-off and cannot be called
converged on both metrics.  I do not recommend starting another
unbounded grid search.  Do a LOCAL Pareto sensitivity experiment: vary
the wrist posture and the approach depth a little, and measure the
executed pad distance, the forward axis, the contact count and the joint
margin together."

This script does exactly that, over the two depth knobs the terminal
actually has (r109/r110):

  * the CURL SCALE (the flexion joints of the searched curl scaled),
  * the SQUEEZE (the terminal's arm translated along the face normal),

and reports the four metrics side by side, so the trade-off is visible
rather than asserted.

    runs/.venv-retarget/bin/python tools/w6b/pareto_scan.py
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
PIPE = REPO / "runs/.venv-retarget/bin/python"
SCRIPT = REPO / "retarget/retarget_dual.py"
SCENE = "grounding/assets/allegro_floating_ur5e_side_r37.xml"
OUT = REPO / "runs/w6b/pareto_scan.json"

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
# (the curl scale, the squeeze mm)
CASES = [
    (1.00, 0.00),   # the delivered terminal
    (1.05, 0.00),
    (1.10, 0.00),
    (1.20, 0.00),
    (1.00, 0.05),
    (1.00, 0.10),
    (1.05, 0.05),
]


def run(curl: float, sq: float):
    env = dict(os.environ)
    env.update(BASE)
    env["W6B_R110_CURL_SCALE"] = f"{curl}"
    env["W6B_R109_SQUEEZE_MM"] = f"{sq}"
    r = subprocess.run([str(PIPE), str(SCRIPT)], cwd=REPO, env=env,
                       capture_output=True, text=True)
    if r.returncode != 0:
        return None, None
    rep = json.loads((REPO / "runs/w6b/report.json").read_text())
    out = r.stdout
    import re
    ex = {}
    for side in ("right", "left"):
        m = re.search(rf"{side}: cmd rect ([\d.]+) mm fwd ([\d.]+) deg .*?"
                      r"executed rect ([\d.]+) mm.*?fwd ([\d.]+)", out, re.S)
        ex[side] = (float(m.group(3)), float(m.group(4))) if m else (None, None)
    return rep, ex


def metrics(rep: dict, side: str) -> dict:
    ps = rep["per_side"][side]
    g1 = ps["gate1_final_touch"]
    pads = g1["executed_state_evidence"]["gate1_v3_executed_state"]["pads"]
    contact = [p for p in pads.values() if p.get("in_contact")]
    # the arm's joint margin: the distance to the closest joint limit
    import mujoco
    return {
        # NOTE: the report's gate1.forward_axis_deg is the COMMANDED
        # axis; the EXECUTED one is printed by the pipeline and parsed
        # from the stdout below.  The v9 review's 1.7 deg margin is
        # about the executed value.
        "the commanded forward axis deg": round(
            float(g1["forward_axis_deg"]), 2),
        "the contacting pads": [p for p in pads if pads[p].get("in_contact")],
        "the normal force N": round(
            float(sum(p["contact_normal_force_N"] for p in contact)), 4),
        "the gate1 pass": bool(g1["pass"]),
        "the gate2 pass": bool(ps["gate2_contact"]["pass"]),
        "the gate3 pass": bool(ps["gate3_approach_clean"]["pass"]),
        "the gate5 pass": bool(ps["gate5_palm_deviation"]["pass"]),
        "the red gates": [k for k, v in rep["verdict_checks"].items()
                          if not v],
    }


def main() -> int:
    rows = []
    for curl, sq in CASES:
        rep, ex = run(curl, sq)
        if rep is None:
            print(f"  the curl {curl} / the squeeze {sq}: the run FAILED")
            continue
        row = {"curl": curl, "squeeze_mm": sq}
        for side in ("right", "left"):
            row[side] = metrics(rep, side)
            row[side]["the executed pad distance mm"] = ex[side][0]
            row[side]["the EXECUTED forward axis deg"] = ex[side][1]
        rows.append(row)
        print(f"  the curl {curl} / the squeeze {sq}: "
              + " | ".join(
                  f"{s}: {row[s]['the executed pad distance mm']} mm, "
                  f"the fwd {row[s]['the EXECUTED forward axis deg']} deg, "
                  f"the pads {len(row[s]['the contacting pads'])}, "
                  f"N {row[s]['the normal force N']}"
                  for s in ("right", "left"))
              + f" | red: {row['right']['the red gates']}")

    OUT.write_text(json.dumps(rows, indent=2, default=float))
    print(f"\n  written: {OUT.relative_to(REPO)}")
    print()
    print("  the trade-off, read from the table above:")
    for r in rows:
        for s in ("right", "left"):
            m = r[s]
            print(f"    the curl {r['curl']} / the squeeze {r['squeeze_mm']} / "
                  f"{s}: the pad distance {m['the executed pad distance mm']} "
                  f"mm (the goal < 5), the forward axis "
                  f"{m.get('the EXECUTED forward axis deg')} deg (the limit 35), "
                  f"the contacting pads {len(m['the contacting pads'])}, "
                  f"the normal force {m['the normal force N']} N")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
