"""r118: the per-side curl scan -- the force and the geometry, separately.

THE MEASURED TENSION (r118, this campaign).  The v10 lift gate needs
1.0911 N of normal force per hand.  The pareto scan (curl 1.0-1.2, one
curl for BOTH hands) found that the curl scale is the only knob in this
family that reaches the required force -- and that it breaks gate1:

    the curl 1.10 / the right: the normal force 1.1384 N (>= 1.0911 N)
        but the index pad is commanded 0.784 mm THROUGH the end face, and
        the R5 tolerance is 0.1164 mm, so the command state's
        n_pads_in_rect drops to 0 and gate1 goes red.

The force law measured in the same campaign (tools/w6b/force_depth_law.py)
says why: the contact force is a function of the COMMANDED penetration
depth, and the gate caps that depth at the R5 tolerance.  Inside the
green window the squeeze buys 0.2549 -> 0.3797 N (right) and the
left 0.7377 -> 0.7015 N -- nowhere near 1.0911 N.

THE MIRROR ASYMMETRY.  The curl scale multiplies the flexion joints of
the commanded finger vector, and the two hands' palm frames are
mirrored: at curl 1.10 the right's index pad goes 0.784 mm INTO the
face while the left's goes 0.686 mm OUT of it.  One curl for both hands
therefore buys the right's force with the left's geometry.  The curl has
to be INDEPENDENT per side.

THIS TOOL bakes a per-side curl into the terminal reference (the same
joint transform the pipeline's r110 applies, but per side) and replays
the full pipeline for each pair, reporting the command geometry, the
executed normal force and the gates side by side.

    runs/.venv-retarget/bin/python tools/w6b/curl_baked_scan.py
"""
from __future__ import annotations

import json
import os
import pathlib
import shutil
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
PIPE = REPO / "runs/.venv-retarget/bin/python"
SCRIPT = REPO / "retarget/retarget_dual.py"
SCENE = "grounding/assets/allegro_floating_ur5e_side_r37.xml"
REFERENCE = REPO / "runs/w6b/palm_pose_refine.json"
WORK = REPO / "runs/w6b/curl_baked_refs"
OUT = REPO / "runs/w6b/curl_baked_scan.json"
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
    # the curl is baked into the reference here, so the pipeline's own
    # r110 scaling must stay neutral
    "W6B_R110_CURL_SCALE": "1.0",
    "W6B_R109_SQUEEZE_MM": "0",
}

# the curl ladder per side (the flexion joints of the 4-joint groups)
RIGHT_CURLS = [1.00, 1.02, 1.04, 1.06, 1.08, 1.10, 1.12]
LEFT_CURLS = [1.00, 1.05, 1.10, 1.15, 1.20, 1.25]


def scaled_reference(rc: float, lc: float) -> pathlib.Path:
    """a copy of the delivered reference with a per-side curl baked in"""
    ref = json.loads(REFERENCE.read_text())
    for side, scale in (("right", rc), ("left", lc)):
        entry = ref["sides"][side]
        cand = entry["best"][0] if isinstance(entry.get("best"), list) \
            else entry["best"]
        fingers = cand.get("fingers") or {}
        for f in ("ff", "mf", "rf", "th"):
            v = list(fingers.get(f, [0.0] * 4))
            # the r110 transform: the flexion joints (1, 2, 3) are scaled,
            # the spread (0) is kept
            for j in (1, 2, 3):
                v[j] *= scale
            fingers[f] = v
        cand["fingers"] = fingers
    WORK.mkdir(parents=True, exist_ok=True)
    p = WORK / f"curl_r{rc:.3f}_l{lc:.3f}.json"
    p.write_text(json.dumps(ref, indent=1))
    return p


def run(ref_path: pathlib.Path) -> dict:
    env = dict(os.environ)
    env.update(BASE)
    env["W6B_R63_REFERENCE"] = str(ref_path)
    r = subprocess.run([str(PIPE), str(SCRIPT)], cwd=REPO, env=env,
                       capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(r.stdout[-2000:] + r.stderr[-2000:] + "\n")
        raise SystemExit(f"the pipeline failed ({ref_path.name})")
    return json.loads(REPORT.read_text())


def side_metrics(rep: dict, side: str) -> dict:
    ps = rep["per_side"][side]
    g1 = ps["gate1_final_touch"]
    ex = g1["executed_state_evidence"]["gate1_v3_executed_state"]
    pads = ex["pads"]
    contact = [p for p, v in pads.items() if v.get("in_contact")]
    deepest = min((float(v.get("surface_gap_mm", 0.0)) for v in pads.values()),
                  default=0.0)
    burial = {p: v.get("through_mm") for p, v in
              g1["per_pad_burial"].items()}
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
        "the per-pad through mm": {k: round(float(v), 3)
                                   for k, v in burial.items()
                                   if v is not None},
    }


def main() -> int:
    keep = REPORT.with_suffix(".json.curl_baked_keep")
    if REPORT.exists():
        shutil.copy2(REPORT, keep)
    rows = []
    try:
        # phase 1: the right's ladder with the left held at the delivered
        # curl; phase 2: the left's ladder with the right held at the
        # delivered curl
        plan = [(rc, 1.00) for rc in RIGHT_CURLS] + \
               [(1.00, lc) for lc in LEFT_CURLS[1:]]
        for rc, lc in plan:
            ref = scaled_reference(rc, lc)
            rep = run(ref)
            row = {"the right curl": rc, "the left curl": lc}
            for s in ("right", "left"):
                row[s] = side_metrics(rep, s)
            row["the red gates"] = [k for k, v in rep["verdict_checks"].items()
                                    if not v]
            rows.append(row)
            print(f"  the curl r {rc} / l {lc}: "
                  + " | ".join(
                      f"{s}: the cmd rect {row[s]['the command rect mm']} "
                      f"inrect {row[s]['the command pads in the rect']} "
                      f"the force {row[s]['the normal force N']} N "
                      f"the pads {len(row[s]['the executed pads in contact'])}"
                      for s in ("right", "left"))
                  + f" | red: {row['the red gates']}")
            ref.unlink(missing_ok=True)
    finally:
        if keep.exists():
            shutil.copy2(keep, REPORT)
            keep.unlink()

    OUT.write_text(json.dumps(rows, indent=2, default=float))
    print(f"\n  written: {OUT.relative_to(REPO)}")
    print("\n  the per-side force vs the geometry:")
    for r in rows:
        for s in ("right", "left"):
            m = r[s]
            print(f"    the curl {r[f'the {s} curl']} / {s}: "
                  f"the force {m['the normal force N']} N, "
                  f"the command rect {m['the command rect mm']} mm, "
                  f"inrect {m['the command pads in the rect']}, "
                  f"the exec n_eligible {m['the executed n_eligible']}, "
                  f"the gate1 {'green' if m['the gate1 pass'] else 'RED'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
