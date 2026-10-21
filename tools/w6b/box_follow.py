"""r55: iterate the terminal aim onto the box's ACTUAL terminal position.

The approach pushes the box across the table (measured 44-55 mm of travel at
0.15-0.4 kg).  The terminal IK targets an absolute palm pose derived from the
box's table-rest pose, so the hands aim at where the box used to be, and the
few millimetres that survive to the terminal are booked against the pads.

This script measures the box's terminal displacement and re-runs with the
terminal aim offset by it, until the fixed point (or four iterations).

    runs/.venv-retarget/bin/python tools/w6b/box_follow.py
"""
from __future__ import annotations

import json
import os
import pathlib
import re
import subprocess

REPO = pathlib.Path(__file__).resolve().parents[2]
PIPE = REPO / "runs/.venv-retarget/bin/python"
SCRIPT = REPO / "retarget/retarget_dual.py"
SCENE = "grounding/assets/allegro_floating_ur5e_side_r37.xml"

RX = re.compile(
    r"^\s*(right|left): cmd rect ([0-9.]+) mm fwd ([0-9.]+) deg .*?"
    r"executed rect ([0-9.]+) mm.*?fwd ([0-9.]+) deg")


def run(offset):
    env = {**dict(os.environ),
           "W6B_SCENE": "ur5e", "W6B_SCENE_XML": SCENE,
           "W6B_R23_FLYOVER_MM": "250", "W6B_R22_MARGIN_MM": "8",
           "W6B_R38_SEAT": "both", "W6B_R39_TGC": "1", "W6B_R39_HOLD": "60",
           "W6B_R50_CURL": "1.25", "W6B_R50_PRELOAD_MM": "0.03",
           "W6B_R52_FINGER_TGC": "1"}
    if offset:
        env["W6B_R55_TGT_OFFSET"] = ",".join(f"{v:.6f}" for v in offset)
    out = subprocess.run([str(PIPE), str(SCRIPT)], cwd=REPO, env=env,
                         capture_output=True, text=True, timeout=1800)
    if out.returncode != 0:
        raise SystemExit(f"run failed:\n{out.stdout[-1500:]}\n{out.stderr[-1500:]}")
    res = {}
    for line in out.stdout.splitlines():
        m = RX.match(line)
        if m:
            res[m.group(1)] = {"cmd_mm": float(m.group(2)),
                               "cmd_fwd": float(m.group(3)),
                               "exec_mm": float(m.group(4)),
                               "exec_fwd": float(m.group(5))}
    rep = json.loads((REPO / "runs/w6b/report.json").read_text())
    for side in ("right", "left"):
        ev = rep["per_side"][side]["gate1_final_touch"]["executed_state_evidence"]
        res[side]["n_eligible"] = ev["n_eligible_contact_pads"]
        res[side]["F_n"] = {k: v["contact_normal_force_N"]
                            for k, v in ev["pad_contact_evidence"].items()
                            if v["in_contact"]}
        res[side]["gate1_pass"] = rep["per_side"][side]["gate1_final_touch"]["pass"]
    bm = rep.get("box_motion") or {}
    res["box_final_mm"] = bm.get("final_displacement_mm")
    res["box_max_mm"] = bm.get("max_displacement_mm")
    # the box's terminal displacement VECTOR (the aim offset)
    res["box_offset_m"] = rep.get("box_terminal_offset_m")
    return res


def main():
    offset = None
    hist = []
    for it in range(4):
        r = run(offset)
        hist.append({"iteration": it, "offset": offset, **r})
        off = r.get("box_offset_m")
        if not off:
            break
        nxt = [float(off[0]), float(off[1]), float(off[2])]
        if offset and max(abs(a - b) for a, b in zip(offset, nxt)) < 5e-4:
            break
        offset = nxt
    (REPO / "runs/w6b/box_follow.json").write_text(json.dumps(hist, indent=2))
    print("| iter | offset mm | exec mm R/L | exec fwd R/L | eligible R/L | "
          "F_n R/L | box final mm | gate1 R/L |")
    print("|" + "---|" * 8)
    for h in hist:
        o = h["offset"]
        os_ = "-" if not o else f"{o[0]*1e3:.1f},{o[1]*1e3:.1f},{o[2]*1e3:.1f}"
        print(f"| {h['iteration']} | {os_} "
              f"| {h['right']['exec_mm']}/{h['left']['exec_mm']} "
              f"| {h['right']['exec_fwd']}/{h['left']['exec_fwd']} "
              f"| {h['right']['n_eligible']}/{h['left']['n_eligible']} "
              f"| {list(h['right']['F_n'].values())}/{list(h['left']['F_n'].values())} "
              f"| {h['box_final_mm']} "
              f"| {h['right']['gate1_pass']}/{h['left']['gate1_pass']} |")
    print("\nwrote runs/w6b/box_follow.json")


if __name__ == "__main__":
    main()
