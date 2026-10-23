"""v10 step 2: the terminal search with the CONTACT NORMAL FORCE as its
criterion.

The v10 lift feasibility gate (tools/w6b/lift_feasibility.py) found that
the lift is statically INFEASIBLE with the delivered terminal: the
measured normal force is 0.2972 N (right) + 0.7177 N (left) = 1.0149 N,
so the available friction (mu = 1.8) is 1.827 N against a 3.928 N weight
-- 53.5% short.  A lift needs 1.091 N per hand, and the fingers' torque
capacity is ~27.8 N of normal force, so the requirement is reachable: the
infeasibility is a property of the CURRENT terminal configuration, not of
the hardware.

The review's instruction for exactly this situation: put the normal force
into the terminal search's criterion.  The kinematic search
(tools/w6b/palm_pose_refine.py) ranks candidates by geometry (the pads on
the face, the forward axis, the clearance); it cannot see the contact
force.  This tool adds the physics leg: each candidate is replayed in the
full pipeline and its EXECUTED normal force is measured, and the winner
is the candidate with the largest force that keeps every gate green.

    runs/.venv-retarget/bin/python tools/w6b/squeeze_search.py
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
REFINE = REPO / "runs/w6b/palm_pose_refine.json"
WORK = REPO / "runs/w6b/squeeze_candidates"
OUT = REPO / "runs/w6b/squeeze_search.json"

BASE = {
    "W6B_SCENE": "ur5e",
    "W6B_SCENE_XML": SCENE,
    "W6B_R23_FLYOVER_MM": "250",
    "W6B_R22_MARGIN_MM": "8",
    "W6B_R75_SIDE_APPROACH": "1",
    "W6B_R63_TARGET": "1",
    "W6B_R18_NATIVE_TARGET": "1",
    "W6B_R39_TGC": "0",
    "W6B_R36_GSTATIC": "1",
    "W6B_R36_I": "0",
    "W6B_R39_HOLD": "60",
}
GREEN = ("gate1_final_touch", "gate2_contact", "gate3_approach_clean",
         "gate5_palm_deviation")


def run(ref_path: pathlib.Path) -> tuple[dict, str]:
    env = dict(os.environ)
    env.update(BASE)
    env["W6B_R63_REFERENCE"] = str(ref_path)
    r = subprocess.run([str(PIPE), str(SCRIPT)], cwd=REPO, env=env,
                       capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(r.stdout[-1500:] + r.stderr[-1500:] + "\n")
        raise SystemExit("the pipeline failed")
    return json.loads((REPO / "runs/w6b/report.json").read_text()), r.stdout


def force_of(rep: dict, side: str) -> float:
    pads = rep["per_side"][side]["gate1_final_touch"]["executed_state_evidence"][
        "gate1_v3_executed_state"]["pads"]
    return float(sum(p["contact_normal_force_N"] for p in pads.values()
                     if p.get("in_contact")))


def gates_ok(rep: dict, side: str) -> bool:
    ps = rep["per_side"][side]
    return all(bool(ps[g]["pass"]) for g in GREEN)


def main() -> int:
    refine = json.loads(REFINE.read_text())
    WORK.mkdir(parents=True, exist_ok=True)
    base_best = {s: refine["sides"][s]["best"][0] for s in ("right", "left")}

    rows = []
    for side in ("right", "left"):
        cands = refine["sides"][side]["best"]
        for i, cand in enumerate(cands):
            # the candidate under test replaces this side; the other side
            # keeps its current best
            trial = {"scene": refine.get("scene"), "sides": {}}
            for s2 in ("right", "left"):
                trial["sides"][s2] = {
                    "best": [cand if s2 == side else base_best[s2]]}
            p = WORK / f"{side}_{i}.json"
            p.write_text(json.dumps(trial))
            rep, _ = run(p)
            f = force_of(rep, side)
            ok = gates_ok(rep, side)
            vc = rep["verdict_checks"]
            red = [k for k, v in vc.items() if not v]
            rows.append({"side": side, "index": i, "force_N": f,
                         "gates_ok": ok, "red": red,
                         "margin": cand.get("margin")})
            print(f"  {side} candidate {i}: the normal force {f:.4f} N, "
                  f"the gates {'ok' if ok else 'red: ' + ','.join(red)}")
            p.unlink()

    print()
    winners = {}
    for side in ("right", "left"):
        cand_rows = [r for r in rows if r["side"] == side]
        good = [r for r in cand_rows if r["gates_ok"]]
        if not good:
            print(f"  {side}: NO candidate keeps every gate green -- the "
                  f"search's criterion cannot be satisfied")
            continue
        best = max(good, key=lambda r: r["force_N"])
        winners[side] = best
        print(f"  {side}: the winner is candidate {best['index']} with "
              f"{best['force_N']:.4f} N "
              f"(the others: "
              + ", ".join(f"{r['index']}:{r['force_N']:.3f}"
                          for r in cand_rows) + ")")

    out = {"rows": rows, "winners": winners}
    OUT.write_text(json.dumps(out, indent=2, default=float))
    print(f"\n  written: {OUT.relative_to(REPO)}")
    if len(winners) == 2:
        print("  to apply the winners: "
              "runs/.venv-retarget/bin/python tools/w6b/squeeze_search.py "
              "--apply")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
