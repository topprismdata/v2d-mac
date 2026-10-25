"""r120: the authorization-A thin-pad curl scan (per side).

THE GRANT.  The v10 review's grant A (docs/gpt_v10_review.md, Q11):
explore thinner/smaller fingertip pads; the original pad stays an
immutable baseline.  r40 (grounding/assets/allegro_floating_ur5e_side_
r40_thinpad.xml) = side_r37 with every pad's half-thickness halved
(1.5 -> 0.75 mm).

THE HYPOTHESIS (from r118's measured coupling): a thinner pad sits its
CENTRE 0.75 mm further out at the same commanded press depth, so the
deeper curls that reach the lift force (the right's curl 1.10 measured
1.1384 N on the ORIGINAL pad) but bust the command rect by 0.684-0.784
mm should come back inside the R5 tolerance (0.1164 mm).

THIS TOOL is curl_baked_scan.py with three changes:
  * the scene is r40_thinpad;
  * the ladders extend deeper (the thin pad may need more press to buy
    the same force -- the force-depth law shifts with the geometry);
  * each row also records gate6 (the executed-contact hard gate): the
    real-contact count, the summed normal force and the a-priori lift
    need read from the scene.

Run (in a CLEAN clone, the local refs are mid-campaign):
    runs/.venv-retarget/bin/python tools/w6b/thinpad_scan.py
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
SCENE = "grounding/assets/allegro_floating_ur5e_side_r40_thinpad.xml"
REFERENCE = REPO / "runs/w6b/palm_pose_refine.json"
WORK = REPO / "runs/w6b/thinpad_refs"
OUT = REPO / "runs/w6b/thinpad_scan.json"
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
    "W6B_R110_CURL_SCALE": "1.0",
    "W6B_R109_SQUEEZE_MM": "0",
}

RIGHT_CURLS = [1.00, 1.02, 1.04, 1.06, 1.08, 1.10, 1.12, 1.16, 1.20]
LEFT_CURLS = [1.00, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30, 1.35]


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
    g6 = ps["gate6_executed_contact"]
    contact = [p for p, v in pads.items() if v.get("in_contact")]
    deepest = min((float(v.get("surface_gap_mm", 0.0)) for v in pads.values()),
                  default=0.0)
    return {
        "the command rect mm": round(float(g1["rect_dist_mm"]), 4),
        "the command pads in the rect": int(g1["n_pads_in_rect"]),
        "the command fwd deg": round(float(g1["forward_axis_deg"]), 2),
        "the gate1 pass": bool(g1["pass"]),
        "the executed pads in contact": contact,
        "the executed n_eligible": int(ex["n_eligible"]),
        "the executed penetration mm": round(deepest, 4),
        "the normal force N": round(float(sum(
            v.get("contact_normal_force_N", 0.0) for v in pads.values()
            if v.get("in_contact"))), 4),
        "the gate6 real contacts": int(g6["n_real_contacts"]),
        "the gate6 force sum N": round(float(g6["normal_force_sum_N"]), 4),
        "the gate6 lift need N": round(float(g6["lift_need_N"]), 4),
        "the gate6 pass": bool(g6["pass"]),
    }


def main() -> int:
    keep = REPORT.with_suffix(".json.thinpad_keep")
    if REPORT.exists():
        shutil.copy2(REPORT, keep)
    rows = []
    try:
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
                      f"{s}: rect {row[s]['the command rect mm']} "
                      f"inrect {row[s]['the command pads in the rect']} "
                      f"force {row[s]['the normal force N']} N "
                      f"g6 {row[s]['the gate6 force sum N']}"
                      f"/{row[s]['the gate6 real contacts']}"
                      f"{' PASS' if row[s]['the gate6 pass'] else ''}"
                      for s in ("right", "left"))
                  + f" | red: {row['the red gates']}")
            ref.unlink(missing_ok=True)
    finally:
        if keep.exists():
            shutil.copy2(keep, REPORT)
            keep.unlink()
    OUT.write_text(json.dumps({"scene": SCENE, "rows": rows}, indent=1))
    print(f"written: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
