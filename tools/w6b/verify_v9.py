"""Clean-room verification of the v9 headline numbers.

Runs the committed side-cell recipe in a FRESH process and checks the
results against the values quoted in docs/gpt_v9_package.md.  Nothing here
reads the session state: every number comes from a fresh run of a
committed script, and the report is re-read from disk afterwards.

    runs/.venv-retarget/bin/python tools/w6b/verify_v9.py
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

# the recipe: the side cell, the side approach, the refined target.  The
# phase split, the backward path solve and the static gravity bias are
# the pipeline's defaults (r94/r100/r101) -- they are NOT env overrides
# here, so this recipe is the committed default path.
ENV = {
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

# the numbers quoted in docs/gpt_v9_package.md
EXPECT = {
    "cmd_rect_mm": {"right": 0.0001, "left": 0.0001},
    "exec_rect_mm": {"right": 4.58, "left": 6.06},
    "exec_fwd_deg": {"right": 33.3, "left": 32.9},
    "cmd_fwd_deg": {"right": 26.3, "left": 27.6},
    "path_dev_mm": {"right": 2.60, "left": 2.59},
    "exec_dev_mm": {"right": 16.70, "left": 11.02},
    "finger_viol_rad": {"right": 0.0, "left": 8.6e-07},
    "box_mm": 2.0,
}
TOL = {
    "cmd_rect_mm": 0.01,
    "exec_rect_mm": 0.05,
    "exec_fwd_deg": 0.15,
    "cmd_fwd_deg": 0.15,
    "path_dev_mm": 0.05,
    "exec_dev_mm": 0.2,
    "finger_viol_rad": 1e-5,
    "box_mm": 0.1,
}
# the gates that must be green, and the two that must be red (with the
# red ones' reason recorded in the ledger)
GREEN = ("gate1_final_touch", "gate2_contact", "gate3_approach_clean",
         "gate5_palm_deviation")
RED = ("gate4_joint_limits", "wrist_referenced_error")


def run(env_extra: dict | None = None) -> str:
    env = dict(os.environ)
    env.update(ENV)
    env.update(env_extra or {})
    r = subprocess.run([str(PIPE), str(SCRIPT)], cwd=REPO, env=env,
                       capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(r.stdout[-3000:] + "\n" + r.stderr[-3000:] + "\n")
        raise SystemExit(f"the pipeline failed (rc={r.returncode})")
    return r.stdout


def parse_exec_rect(out: str) -> dict:
    """the executed terminal distance and forward axis per side."""
    got = {}
    for m in re.finditer(
            r"(right|left): cmd rect ([\d.]+) mm fwd ([\d.]+) deg .*?"
            r"executed rect ([\d.]+) mm.*?fwd ([\d.]+)",
            out, re.S):
        side = m.group(1)
        got[side] = {
            "cmd_rect_mm": float(m.group(2)),
            "cmd_fwd_deg": float(m.group(3)),
            "exec_rect_mm": float(m.group(4)),
            "exec_fwd_deg": float(m.group(5)),
        }
    return got


def main() -> int:
    fails = []

    def chk(name, got, want, tol):
        ok = abs(float(got) - float(want)) <= tol
        print(f"  {'OK ' if ok else 'BAD'} {name}: {got} (expect {want} +- {tol})")
        if not ok:
            fails.append(name)

    print("[1/3] the side-cell recipe, fresh process")
    out = run()
    ex = parse_exec_rect(out)
    for side in ("right", "left"):
        if side not in ex:
            fails.append(f"the {side} line was not printed")
            continue
        for k in ("cmd_rect_mm", "exec_rect_mm", "exec_fwd_deg",
                  "cmd_fwd_deg"):
            chk(f"the {side}'s {k}", ex[side][k], EXPECT[k][side], TOL[k])

    print("[2/3] the report's own records")
    rep = json.loads((REPO / "runs/w6b/report.json").read_text())
    for side in ("right", "left"):
        ps = rep["per_side"][side]
        chk(f"the {side}'s path deviation",
            ps["gate5_palm_deviation"]["side_path_palm_deviation"]["max_m"]
            * 1e3, EXPECT["path_dev_mm"][side], TOL["path_dev_mm"])
        chk(f"the {side}'s executed deviation",
            ps["gate5_palm_deviation"]["execution_dev_max_m"] * 1e3,
            EXPECT["exec_dev_mm"][side], TOL["exec_dev_mm"])
        chk(f"the {side}'s finger residual",
            ps["gate4_joint_limits"]["finger_qpos_max_violation_rad"],
            EXPECT["finger_viol_rad"][side], TOL["finger_viol_rad"])
        chk(f"the {side}'s command pad distance",
            ps["gate1_final_touch"]["rect_dist_mm"],
            EXPECT["cmd_rect_mm"][side], TOL["cmd_rect_mm"])
    chk("the box's displacement", rep["box_motion"]["final_displacement_mm"],
        EXPECT["box_mm"], TOL["box_mm"])

    print("[3/3] the gate verdicts")
    vc = rep["verdict_checks"]
    for g in GREEN:
        ok = bool(vc.get(g))
        print(f"  {'OK ' if ok else 'BAD'} {g}: {vc.get(g)} (expect True)")
        if not ok:
            fails.append(g)
    for g in RED:
        ok = not bool(vc.get(g))
        print(f"  {'OK ' if ok else 'BAD'} {g}: {vc.get(g)} (expect False, "
              f"disclosed in the ledger)")
        if not ok:
            fails.append(g)

    print()
    if fails:
        print(f"FAIL: {len(fails)} item(s): {fails}")
        return 1
    print("PASS: every v9 headline number reproduced in a fresh process")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
