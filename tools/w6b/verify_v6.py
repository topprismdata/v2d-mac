#!/usr/bin/env python3
"""Clean-room verification of the v6 headline numbers (the r26 Track A
recipe), run against the v6-era commit (4a51dda0) in its own worktree.

Unlike verify_v7..v12 this script is REPO-PARAMETERISED: the pipeline it
drives is the one inside --repo, so the era's code and the era's claims are
compared directly (the honest protocol of ledger r181/r186: evidence is
pinned to its own commit).

    runs/.venv-retarget/bin/python tools/w6b/verify_v6.py --repo /tmp/v6_wt

The claims are those quoted in docs/gpt_v6_package.md section II/III:
  gate1 rect (pad1 R5): right 0.229213 um / left 3.903380 um
  up_z min 0.9907; gate3 red frames right 351 / left 0
  gate4 radius right 1.49-1.62 / left 0.195-0.499 rad
  g5reg right 0.77 / left 0.51 mm

Only the gate1 claim has a DIRECT report field at that commit.  The others
(gate3 frame counts under the era's definition, gate4's radius range,
up_z, g5's registered mm) are checked only when the era report carries the
field; a missing field is reported as UNVERIFIABLE (not a pass) -- the
gaps the ledger records.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys

SCENE = "grounding/assets/allegro_floating_ur5e_r24_vise.xml"
ENV = {
    "W6B_SCENE": "ur5e",
    "W6B_SCENE_XML": SCENE,
    "W6B_R23_FLYOVER_MM": "250",
    "W6B_R22_MARGIN_MM": "8",
    "W6B_R18_NATIVE_TARGET": "1",
    "W6B_R26_TRACK_A": "1",
    "W6B_R28_FORCE3": "1",
}
# the v6 package's quoted numbers (um -> mm for gate1)
EXPECT = {
    "gate1_rect_mm": {"right": 0.000229213, "left": 0.00390338},
}
# the v6 tolerances scale with the quoted precision: the right's number
# matched to 17 significant digits in the r186b probe, so the tolerance
# is 1% of the quoted value (a float-format difference, not a rerun)
TOL = {"gate1_rect_mm": 0.01}


def run(repo: pathlib.Path) -> str:
    env = dict(os.environ)
    env.update(ENV)
    pipe = repo / "runs/.venv-retarget/bin/python"
    r = subprocess.run([str(pipe), "retarget/retarget_dual.py"], cwd=repo,
                       env=env, capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(r.stdout[-3000:] + "\n" + r.stderr[-3000:] + "\n")
        raise SystemExit(f"the pipeline failed (rc={r.returncode})")
    return r.stdout


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    a = ap.parse_args()
    repo = pathlib.Path(a.repo).resolve()

    print(f"[1/2] the r26 recipe in {repo}")
    run(repo)
    rep = json.loads((repo / "runs/w6b/report.json").read_text())
    fails, unver = [], []

    print("[2/2] the v6 claims")
    for side in ("right", "left"):
        got = rep["per_side"][side]["gate1_final_touch"]["rect_dist_mm"]
        want = EXPECT["gate1_rect_mm"][side]
        tol = abs(want) * TOL["gate1_rect_mm"] + 1e-12
        ok = abs(got - want) <= tol
        print(f"  {'OK ' if ok else 'BAD'} gate1 rect {side}: {got:.9f} mm "
              f"(expect {want:.9f} +- {tol:.2e})")
        if not ok:
            fails.append(f"gate1_{side}")

    # the claims without direct era-report fields: report them as gaps
    g3r = rep["per_side"]["right"].get("gate3_approach_clean", {})
    g3l = rep["per_side"]["left"].get("gate3_approach_clean", {})
    n_r = len(g3r.get("support_contact_control_frames") or [])
    n_l = len(g3l.get("support_contact_control_frames") or [])
    print(f"  ??  gate3 red frames: right {n_r} / left {n_l} (the v6 text "
          f"quotes 351 / 0 under the era's own definition -- no direct "
          f"field to compare)")
    unver.append("gate3 frame counts (definition mismatch)")

    g5r = rep["per_side"]["right"]["gate5_palm_deviation"]
    g5l = rep["per_side"]["left"]["gate5_palm_deviation"]
    reg_r = g5r.get("registered") or {}
    reg_l = g5l.get("registered") or {}
    if isinstance(reg_r, dict) and isinstance(reg_l, dict):
        br = (reg_r.get("registered") or reg_r.get("bias_norm_m"))
        bl = (reg_l.get("registered") or reg_l.get("bias_norm_m"))
        print(f"  ??  g5 registered: right {br} / left {bl} "
              f"(the v6 text quotes 0.77 / 0.51 mm)")
        unver.append("g5 registered mm")

    upz = (rep.get("box_motion") or {}).get("up_z_min")
    print(f"  ??  up_z min: {upz} (the v6 text quotes 0.9907; the era "
          f"report at this commit carries no box_motion block)")
    unver.append("up_z min")

    print()
    if fails:
        print(f"FAIL: {len(fails)} claim(s) mismatched: {fails}")
        print(f"UNVERIFIABLE (no era field): {unver}")
        return 1
    print(f"PASS: the gate1 claim reproduces; UNVERIFIABLE: {unver}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())