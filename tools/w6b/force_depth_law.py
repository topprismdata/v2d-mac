"""r118: the force-vs-penetration law of the terminal contact.

WHY THIS TOOL EXISTS.  The v10 lift gate needs 1.0911 N of normal force
per hand (the 0.4 kg box, mu = 1.8, the friction pinch on the two end
faces).  The delivered terminal produces 0.2549 N (right) / 0.7377 N
(left) -- 54.5% short -- and the pareto scan found that a curl of 1.10
produces 1.1384 N on the right BUT breaks gate1.

THE TWO MEASUREMENTS THAT EXPLAIN IT (both taken in this campaign):

  * at curl 1.10 the right's index pad is commanded 0.784 mm THROUGH
    the end face (the gate1 R5 tolerance is 0.1164 mm), so the extra
    force and the red gate are the SAME displacement;
  * the r109 SQUEEZE translates the terminal's arm along the face
    normal with the palm's orientation held, so it presses without
    changing the command state's geometry -- at 0.05 mm it gives
    0.3797 N with n_pads_in_rect still 2 and no burial.

So the force is a function of the commanded penetration depth, and the
gates cap that depth.  This tool MEASURES the law instead of guessing
it: for a ladder of (curl, squeeze) pairs it records the command
state's geometry, the EXECUTED penetration depth (the executed pad's
signed surface gap), the normal force and the box's displacement.

The question the law answers: is 1.0911 N reachable INSIDE the green
region (the commanded depth <= the R5 tolerance)?

    runs/.venv-retarget/bin/python tools/w6b/force_depth_law.py
"""
from __future__ import annotations

import json
import os
import pathlib
import re
import shutil
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
PIPE = REPO / "runs/.venv-retarget/bin/python"
SCRIPT = REPO / "retarget/retarget_dual.py"
SCENE = "grounding/assets/allegro_floating_ur5e_side_r37.xml"
OUT = REPO / "runs/w6b/force_depth_law.json"
REPORT = REPO / "runs/w6b/report.json"

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

# (the curl scale, the squeeze mm) -- the depth ladder
CASES = [
    (1.00, 0.00),
    (1.00, 0.05),
    (1.00, 0.10),
    (1.00, 0.15),
    (1.00, 0.20),
    (1.00, 0.30),
    (1.05, 0.00),
    (1.05, 0.05),
    (1.10, 0.00),
]


def run(curl: float, sq: float) -> tuple[dict, str]:
    env = dict(os.environ)
    env.update(BASE)
    env["W6B_R110_CURL_SCALE"] = f"{curl}"
    env["W6B_R109_SQUEEZE_MM"] = f"{sq}"
    r = subprocess.run([str(PIPE), str(SCRIPT)], cwd=REPO, env=env,
                       capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(r.stdout[-2000:] + r.stderr[-2000:] + "\n")
        raise SystemExit(f"the pipeline failed (the curl {curl}, "
                         f"the squeeze {sq})")
    return json.loads(REPORT.read_text()), r.stdout


def row_of(rep: dict, side: str) -> dict:
    ps = rep["per_side"][side]
    g1 = ps["gate1_final_touch"]
    ex = g1["executed_state_evidence"]["gate1_v3_executed_state"]
    pads = ex["pads"]
    contact = [p for p, v in pads.items() if v.get("in_contact")]
    # the executed penetration: the deepest pad's signed surface gap
    # (negative = pressed into the face)
    deepest = min((float(v.get("surface_gap_mm", 0.0)) for v in pads.values()),
                  default=0.0)
    return {
        "the command rect mm": round(float(g1["rect_dist_mm"]), 4),
        "the command pads in the rect": int(g1["n_pads_in_rect"]),
        "the command burial": bool(g1["pad_center_buried"]),
        "the command fwd deg": round(float(g1["forward_axis_deg"]), 2),
        "the gate1 pass": bool(g1["pass"]),
        "the executed pads in contact": contact,
        "the executed n_eligible": int(ex["n_eligible"]),
        "the executed penetration mm": round(deepest, 4),
        "the normal force N": round(float(sum(
            v.get("contact_normal_force_N", 0.0) for v in pads.values()
            if v.get("in_contact"))), 4),
    }


def main() -> int:
    # the report is shared: keep the delivered one, restore it after
    keep = REPORT.with_suffix(".json.force_depth_law_keep")
    if REPORT.exists():
        shutil.copy2(REPORT, keep)
    rows = []
    try:
        for curl, sq in CASES:
            rep, _ = run(curl, sq)
            row = {"curl": curl, "squeeze_mm": sq}
            for s in ("right", "left"):
                row[s] = row_of(rep, s)
            row["the red gates"] = [k for k, v in rep["verdict_checks"].items()
                                   if not v]
            row["the box displacement mm"] = round(
                float(rep.get("box_motion", {}).get(
                    "max_displacement_mm", 0.0)), 3)
            rows.append(row)
            print(f"  the curl {curl} / the squeeze {sq}: "
                  + " | ".join(
                      f"{s}: the cmd rect {row[s]['the command rect mm']} mm "
                      f"inrect {row[s]['the command pads in the rect']} "
                      f"burial {row[s]['the command burial']} | "
                      f"the exec depth {row[s]['the executed penetration mm']} mm "
                      f"the force {row[s]['the normal force N']} N "
                      f"the pads {len(row[s]['the executed pads in contact'])}"
                      for s in ("right", "left"))
                  + f" | the box {row['the box displacement mm']} mm "
                  + f"| red: {row['the red gates']}")
    finally:
        if keep.exists():
            shutil.copy2(keep, REPORT)
            keep.unlink()

    OUT.write_text(json.dumps(rows, indent=2, default=float))
    print(f"\n  written: {OUT.relative_to(REPO)}")

    # the law: the force per mm of the executed depth, inside the green
    # region (the command state's pads are in the rect and not buried)
    print("\n  the law (the green rows only):")
    for r in rows:
        for s in ("right", "left"):
            m = r[s]
            green = (m["the command pads in the rect"] >= 2
                     and not m["the command burial"])
            if not green:
                continue
            d = m["the executed penetration mm"]
            f = m["the normal force N"]
            slope = f / abs(d) if d else float("nan")
            print(f"    the curl {r['curl']} / the squeeze {r['squeeze_mm']} / "
                  f"{s}: the executed depth {d} mm, the force {f} N, "
                  f"the slope {slope:.3f} N/mm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
