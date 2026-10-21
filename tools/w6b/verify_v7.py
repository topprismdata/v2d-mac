"""Clean-room verification of the v7 headline numbers.

Runs the two committed recipes in fresh processes and checks the results
against the values quoted in docs/gpt_v7_package.md.  Nothing here reads the
session state: every number comes from a fresh run of a committed script.

    runs/.venv-retarget/bin/python tools/w6b/verify_v7.py
"""
from __future__ import annotations

import json
import pathlib
import re
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
PIPE = REPO / "runs/.venv-retarget/bin/python"
SCRIPT = REPO / "retarget/retarget_dual.py"

SIDE = {
    "name": "side cell (r37 scene)",
    "scene": "grounding/assets/allegro_floating_ur5e_side_r37.xml",
    "env": {
        "W6B_SCENE": "ur5e",
        "W6B_SCENE_XML": "grounding/assets/allegro_floating_ur5e_side_r37.xml",
        "W6B_R23_FLYOVER_MM": "250",
        "W6B_R22_MARGIN_MM": "8",
        "W6B_R38_SEAT": "both",
        "W6B_R39_TGC": "1",
        "W6B_R39_HOLD": "60",
        "W6B_R50_CURL": "1.25",
        "W6B_R50_PRELOAD_MM": "0.03",
        "W6B_R52_FINGER_TGC": "1",
    },
    "expect_cmd_mm": {"right": 0.023, "left": 0.000},
    "expect_exec_mm": {"right": 4.39, "left": 4.59},
    "expect_fwd_deg": {"right": 33.6, "left": 29.2},
    "expect_cmd_fwd_deg": {"right": 18.7, "left": 19.1},
    "expect_gate1_pass": {"right": True, "left": True},
}

CEILING = {
    "name": "ceiling master (r26 recipe)",
    "scene": "grounding/assets/allegro_floating_ur5e_r24_vise.xml",
    "env": {
        "W6B_SCENE": "ur5e",
        "W6B_SCENE_XML": "grounding/assets/allegro_floating_ur5e_r24_vise.xml",
        "W6B_R23_FLYOVER_MM": "250",
        "W6B_R22_MARGIN_MM": "8",
        "W6B_R18_NATIVE_TARGET": "1",
        "W6B_R26_TRACK_A": "1",
        "W6B_R28_FORCE3": "1",
    },
    "expect_cmd_mm": {"right": 0.000, "left": 0.004},
    "expect_exec_mm": {"right": 152.68, "left": 16.77},
    "expect_fwd_deg": {"right": 101.7, "left": 19.6},
}

RX = re.compile(
    r"^\s*(right|left): cmd rect ([0-9.]+) mm fwd ([0-9.]+) deg "
    r"buried=(\w+) pass=(\w+) .*?"
    r"executed rect ([0-9.]+) mm.*?fwd ([0-9.]+) deg")


def run(cell):
    env = {**dict(__import__("os").environ), **cell["env"]}
    out = subprocess.run([str(PIPE), str(SCRIPT)], cwd=REPO, env=env,
                         capture_output=True, text=True, timeout=900)
    if out.returncode != 0:
        raise SystemExit(f"{cell['name']} failed:\n{out.stdout[-2000:]}\n"
                         f"{out.stderr[-2000:]}")
    res = {}
    for line in out.stdout.splitlines():
        m = RX.match(line)
        if m:
            side = m.group(1)
            res[side] = {"cmd_mm": float(m.group(2)),
                         "cmd_fwd": float(m.group(3)),
                         "buried": m.group(4) == "True",
                         "gate1_pass": m.group(5) == "True",
                         "exec_mm": float(m.group(6)),
                         "exec_fwd": float(m.group(7))}
    return res


def check(cell, res):
    ok = True
    print(f"\n== {cell['name']}")
    for side in ("right", "left"):
        if side not in res:
            print(f"   {side}: MISSING from the output")
            ok = False
            continue
        r = res[side]
        for key, exp in (("cmd_mm", cell["expect_cmd_mm"][side]),
                         ("exec_mm", cell["expect_exec_mm"][side]),
                         ("exec_fwd", cell["expect_fwd_deg"][side])):
            got = r[key]
            good = abs(got - exp) <= max(0.01, abs(exp) * 0.02)
            ok = ok and good
            print(f"   {side:5s} {key:8s} got {got:9.3f} expect {exp:9.3f} "
                  f"{'OK' if good else 'MISMATCH'}")
        if "expect_cmd_fwd_deg" in cell:
            got = r["cmd_fwd"]
            exp = cell["expect_cmd_fwd_deg"][side]
            good = abs(got - exp) <= max(0.01, abs(exp) * 0.02)
            ok = ok and good
            print(f"   {side:5s} cmd_fwd  got {got:9.3f} expect {exp:9.3f} "
                  f"{'OK' if good else 'MISMATCH'}")
        if "expect_gate1_pass" in cell:
            got = r["gate1_pass"]
            exp = cell["expect_gate1_pass"][side]
            good = got == exp
            ok = ok and good
            print(f"   {side:5s} gate1    got {str(got):>9s} expect "
                  f"{str(exp):>9s} {'OK' if good else 'MISMATCH'}")
    return ok


def composition():
    """Measure the delivered frame's box extent and compare with the target."""
    import numpy as np
    from PIL import Image
    out = {}
    for tag, path in (("sim", REPO / "docs/replay_frames/front_r0251.png"),
                      ("reference", REPO / "runs/w6b/frames_full/f_0252.jpg")):
        im = np.asarray(Image.open(path).convert("RGB")).astype(int)
        r, g, b = im[:, :, 0], im[:, :, 1], im[:, :, 2]
        m = (r > 70) & (b > 90) & (g < r - 25) & (g < b - 25)
        ys, xs = np.nonzero(m)
        h, w = im.shape[:2]
        out[tag] = {"width": float((xs.max() - xs.min()) / w),
                    "u": float((xs.min() + xs.max()) / 2 / w),
                    "v": float((ys.min() + ys.max()) / 2 / h)}
    print("\n== rendered composition (measured on the delivered PNG)")
    ok = True
    for k in ("width", "u", "v"):
        got, exp = out["sim"][k], out["reference"][k]
        good = abs(got - exp) <= 0.05
        ok = ok and good
        print(f"   {k:6s} sim {got:.3f}  reference {exp:.3f}  "
              f"{'OK' if good else 'MISMATCH'}")
    return ok


def main() -> None:
    ok = True
    for cell in (SIDE, CEILING):
        ok = check(cell, run(cell)) and ok
    ok = composition() and ok
    print("\nVERIFY:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
