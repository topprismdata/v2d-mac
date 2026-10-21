"""r55: the mass-sensitivity audit the reviewer asked for.

The 0.4 kg object is a modelling choice (a 150 g box slides 16.7 mm under the
squeeze and every millimetre of that travel is booked against the pad that did
not follow it).  The reviewer asked for the same controller across a mass
sweep, reporting the box's displacement, the contact force, the executed
distance and whether it slips.

    runs/.venv-retarget/bin/python tools/w6b/mass_sensitivity.py
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
BASE_SCENE = REPO / "grounding/assets/allegro_floating_ur5e_side_r37.xml"
MASSES = [0.15, 0.25, 0.4]

RX = re.compile(
    r"^\s*(right|left): cmd rect ([0-9.]+) mm fwd ([0-9.]+) deg .*?"
    r"executed rect ([0-9.]+) mm.*?fwd ([0-9.]+) deg")


def scene_for(mass: float) -> pathlib.Path:
    """Write the mass variant next to the base scene (committed with -f)."""
    text = BASE_SCENE.read_text()
    out = REPO / f"grounding/assets/allegro_floating_ur5e_side_r37_m{int(mass*1000):03d}.xml"
    new = re.sub(r'(name="object_geom" type="box" size="0.06 0.12 0.04" )mass="[0-9.]+"',
                 r'\1mass="%s"' % mass, text)
    if new == text:
        # the mass attribute may sit on another attribute order
        new = re.sub(r'(name="object_geom"[^>]*?mass=")[0-9.]+(")',
                     r'\g<1>%s\g<2>' % mass, text)
    new = new.replace("allegro_floating_ur5e_side_r37",
                      f"allegro_floating_ur5e_side_r37_m{int(mass*1000):03d}")
    out.write_text(new)
    return out


def run(mass: float) -> dict:
    scene = scene_for(mass)
    env = {
        **dict(__import__("os").environ),
        "W6B_SCENE": "ur5e",
        "W6B_SCENE_XML": f"grounding/assets/{scene.name}",
        "W6B_R23_FLYOVER_MM": "250",
        "W6B_R22_MARGIN_MM": "8",
        "W6B_R38_SEAT": "both",
        "W6B_R39_TGC": "1",
        "W6B_R39_HOLD": "60",
        "W6B_R50_CURL": "1.25",
        "W6B_R50_PRELOAD_MM": "0.03",
        "W6B_R52_FINGER_TGC": "1",
    }
    out = subprocess.run([str(PIPE), str(SCRIPT)], cwd=REPO, env=env,
                         capture_output=True, text=True, timeout=1800)
    if out.returncode != 0:
        raise SystemExit(f"mass {mass} failed:\n{out.stdout[-1500:]}\n{out.stderr[-1500:]}")
    res = {"mass_kg": mass}
    for line in out.stdout.splitlines():
        m = RX.match(line)
        if m:
            side = m.group(1)
            res[side] = {"cmd_mm": float(m.group(2)),
                         "cmd_fwd": float(m.group(3)),
                         "exec_mm": float(m.group(4)),
                         "exec_fwd": float(m.group(5))}
    rep = json.loads((REPO / "runs/w6b/report.json").read_text())
    res["box_motion"] = rep.get("box_motion")
    for side in ("right", "left"):
        ev = rep["per_side"][side]["gate1_final_touch"]["executed_state_evidence"]
        res[side]["eligible_pads"] = ev["eligible_contact_pads"]
        res[side]["n_eligible"] = ev["n_eligible_contact_pads"]
        res[side]["normal_forces_N"] = {
            k: v["contact_normal_force_N"]
            for k, v in ev["pad_contact_evidence"].items() if v["in_contact"]}
        res[side]["surface_gaps_mm"] = {
            k: v["pad_surface_signed_gap_mm"]
            for k, v in ev["pad_contact_evidence"].items()}
        res[side]["gate1_pass"] = rep["per_side"][side]["gate1_final_touch"]["pass"]
    return res


def main() -> None:
    rows = [run(m) for m in MASSES]
    out = REPO / "runs/w6b/mass_sensitivity.json"
    out.write_text(json.dumps(rows, indent=2))
    hdr = ("| mass | cmd mm R/L | exec mm R/L | exec fwd R/L | eligible pads R/L "
           "| F_n N R/L | box max slip mm | slip>5mm | gate1 R/L |")
    print(hdr)
    print("|" + "---|" * 9)
    for r in rows:
        bm = r.get("box_motion") or {}
        print(f"| {r['mass_kg']} kg "
              f"| {r['right']['cmd_mm']}/{r['left']['cmd_mm']} "
              f"| {r['right']['exec_mm']}/{r['left']['exec_mm']} "
              f"| {r['right']['exec_fwd']}/{r['left']['exec_fwd']} "
              f"| {r['right']['n_eligible']}/{r['left']['n_eligible']} "
              f"| {list(r['right']['normal_forces_N'].values())}/"
              f"{list(r['left']['normal_forces_N'].values())} "
              f"| {bm.get('max_displacement_mm')} "
              f"| {bm.get('slip_over_5mm')} "
              f"| {r['right']['gate1_pass']}/{r['left']['gate1_pass']} |")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
