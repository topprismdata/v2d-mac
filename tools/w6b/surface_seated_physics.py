"""r118: the physics leg for the surface-seated candidates.

The surface-seated search (tools/w6b/surface_seated_search.py) produces
candidates whose pad CONTACT SURFACE sits on the end face (and whose
curl is baked into the finger IK).  This tool replays each candidate in
the full pipeline and measures what matters:

  * the command state's gate1 (the frozen v1/v2 CENTRE criterion) and
    the v3 SURFACE criterion,
  * the EXECUTED normal force per pad and the executed pad count,
  * which gates are red.

The reference schema is the delivered one (runs/w6b/palm_pose_refine.json):
    {"sides": {"right": {"best": [{...}]}, "left": {...}}}
with each candidate carrying "palm_pos", "palm_R", "arm" and "fingers"
(the keys ff, mf, rf, th).

    runs/.venv-retarget/bin/python tools/w6b/surface_seated_physics.py
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
CANDS = pathlib.Path(os.environ.get(
    "W6B_S118_CANDS", "runs/w6b/surface_seated_candidates.json"))
DELIVERED = REPO / "runs/w6b/palm_pose_refine.json"
WORK = REPO / "runs/w6b/surface_seated_refs"
OUT = pathlib.Path(os.environ.get(
    "W6B_S118_PHYS_OUT", "runs/w6b/surface_seated_physics.json"))
REPORT = REPO / "runs/w6b/report.json"
N_CANDS = int(os.environ.get("W6B_S118_N", "8"))

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


def delivered_entry(side: str) -> dict:
    ref = json.loads(DELIVERED.read_text())
    e = ref["sides"][side]
    return e["best"][0] if isinstance(e.get("best"), list) else e["best"]


def candidate_entry(row: dict) -> dict:
    """the searcher's row -> the reference's entry schema"""
    fingers = {f: list(row["the fingers"][f]) for f in ("ff", "mf", "rf")}
    fingers["th"] = list(row["the thumb"])
    return {
        "palm_pos": list(row["the palm"]),
        "palm_R": [list(r) for r in row["the palm_R"]],
        "arm": list(row["the arm"]),
        "fingers": fingers,
        "the curl": row["the curl"],
        "the preload mm": row["the preload mm"],
        "the surface gaps mm": row["the surface gaps mm"],
    }


def run(ref_path: pathlib.Path) -> dict:
    env = dict(os.environ)
    env.update(BASE)
    env["W6B_R63_REFERENCE"] = str(ref_path)
    r = subprocess.run([str(PIPE), str(SCRIPT)], cwd=REPO, env=env,
                       capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(r.stdout[-1500:] + r.stderr[-1500:] + "\n")
        raise SystemExit(f"the pipeline failed ({ref_path.name})")
    return json.loads(REPORT.read_text())


def metrics(rep: dict, side: str) -> dict:
    ps = rep["per_side"][side]
    g1 = ps["gate1_final_touch"]
    ex = g1["executed_state_evidence"]["gate1_v3_executed_state"]
    pads = ex["pads"]
    contact = [p for p, v in pads.items() if v.get("in_contact")]
    return {
        "the v1v2 rect mm": round(float(g1["rect_dist_mm"]), 4),
        "the v1v2 pads in the rect": int(g1["n_pads_in_rect"]),
        "the v1v2 burial": bool(g1["pad_center_buried"]),
        "the v1v2 fwd deg": round(float(g1["forward_axis_deg"]), 2),
        "the v1v2 pass": bool(g1["pass"]),
        "the v3 eligible": int(ex["n_eligible"]),
        "the v3 pass": bool(ex["pass"]),
        "the executed pads in contact": contact,
        "the normal force N": round(float(sum(
            v.get("contact_normal_force_N", 0.0) for v in pads.values()
            if v.get("in_contact"))), 4),
        "the per-pad force N": {p: round(float(v.get(
            "contact_normal_force_N", 0.0)), 4) for p, v in pads.items()},
        "the per-pad surface gap mm": {p: round(float(v.get(
            "surface_gap_mm", 0.0)), 4) for p, v in pads.items()},
    }


def main() -> int:
    if not CANDS.is_file():
        raise SystemExit(f"the candidates are missing: {CANDS}")
    cands = json.loads(CANDS.read_text())
    keep = REPORT.with_suffix(".json.surface_physics_keep")
    if REPORT.exists():
        shutil.copy2(REPORT, keep)
    rows = []
    try:
        for side in ("right", "left"):
            best = cands["sides"].get(side, {}).get("best", [])
            other = delivered_entry("left" if side == "right" else "right")
            for i, row in enumerate(best[:N_CANDS]):
                trial = {"scene": cands.get("scene"), "sides": {}}
                for s2 in ("right", "left"):
                    trial["sides"][s2] = {"best": [
                        candidate_entry(row) if s2 == side else other]}
                WORK.mkdir(parents=True, exist_ok=True)
                p = WORK / f"{side}_{i}.json"
                p.write_text(json.dumps(trial))
                try:
                    rep = run(p)
                except SystemExit:
                    print(f"  {side} candidate {i}: the pipeline FAILED")
                    continue
                m = metrics(rep, side)
                red = [k for k, v in rep["verdict_checks"].items() if not v]
                rows.append({"side": side, "index": i,
                             "the curl": row["the curl"],
                             "the preload mm": row["the preload mm"],
                             "the surface gaps mm": row["the surface gaps mm"],
                             side: m, "the red gates": red})
                print(f"  {side} #{i} (the curl {row['the curl']} / the "
                      f"preload {row['the preload mm']}): "
                      f"the v1v2 rect {m['the v1v2 rect mm']} inrect "
                      f"{m['the v1v2 pads in the rect']} | the v3 eligible "
                      f"{m['the v3 eligible']} | the force "
                      f"{m['the normal force N']} N the pads "
                      f"{len(m['the executed pads in contact'])} | red: {red}")
                p.unlink(missing_ok=True)
    finally:
        if keep.exists():
            shutil.copy2(keep, REPORT)
            keep.unlink()

    OUT.write_text(json.dumps(rows, indent=2, default=float))
    print(f"\n  written: {OUT.relative_to(REPO)}")

    # the verdict: is there a candidate with the v3 (surface) criterion
    # green AND the executed force at the lift's requirement?
    NEED = 1.0911
    print("\n  the verdict:")
    for s in ("right", "left"):
        cand_rows = [r for r in rows if r["side"] == s]
        v3_green = [r for r in cand_rows if r[s]["the v3 pass"]]
        strong = [r for r in v3_green if r[s]["the normal force N"] >= NEED]
        print(f"    {s}: {len(cand_rows)} candidates, {len(v3_green)} with the "
              f"v3 surface criterion green, {len(strong)} of those also at "
              f"{NEED} N")
        for r in cand_rows[:5]:
            print(f"      #{r['index']} the curl {r['the curl']}: the v3 "
                  f"{r[s]['the v3 pass']} ({r[s]['the v3 eligible']} pads), "
                  f"the force {r[s]['the normal force N']} N, the red "
                  f"{r['the red gates']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
