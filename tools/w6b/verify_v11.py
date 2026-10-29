#!/usr/bin/env python3
"""v11 clean-room verification (fresh processes, fresh clone).

Headline: the TWO-HANDED PLATE LIFT.  In a fresh clone of the committed
tree, the dual-plate recipe (scene r44 + the committed plate_1 reference)
must make both palms' plates take real face contact in one run, lift the
box 6 mm over 90 frames with the box's z holding at the peak, and drop
the table's support force from the box's full weight to exactly zero --
plus bit-level determinism on a re-run -- while the r43 right-plate
master recipe still reproduces its own anchor numbers.

Usage:
    runs/.venv-retarget/bin/python tools/w6b/verify_v11.py
"""
import json
import os
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
TOOLS = REPO / "tools/w6b"
PIPE = REPO / "runs/.venv-retarget/bin/python"
RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok)))
    print(f"  {'OK ' if ok else 'FAIL'}  {name}"
          + (f"  [{detail}]" if detail else ""))


def near(a, b, tol):
    return a is not None and abs(float(a) - float(b)) <= tol


DUAL_ENV = {
    "W6B_SCENE": "ur5e",
    "W6B_SCENE_XML":
        "grounding/assets/allegro_floating_ur5e_side_r44_dualplate.xml",
    "W6B_R63_REFERENCE": "runs/w6b/plate_refs/plate_1.json",
    "W6B_LIFT_MM": "6",
    "W6B_LIFT_FRAMES": "90",
    "W6B_R109_SQUEEZE_MM": "0.4",
    "W6B_R127_LEFT_CURL_DELAY": "80",
    "W6B_R127_CURL_DELAY_RIGHT": "64",
    "W6B_R144_FINGER_LEAD": "100",
    "W6B_R38_SEAT": "left",
    "W6B_R23_FLYOVER_MM": "250",
    "W6B_R22_MARGIN_MM": "8",
    "W6B_R75_SIDE_APPROACH": "1",
    "W6B_R63_TARGET": "1",
    "W6B_R18_NATIVE_TARGET": "1",
    "W6B_R50_CURL": "1.25",
    "W6B_R39_TGC": "0",
    "W6B_R36_GSTATIC": "1",
    "W6B_R36_I": "0",
    "W6B_R39_HOLD": "60",
    "W6B_R126_BOX_TRACE": "1",
}
MASTER_ENV = dict(DUAL_ENV, **{
    "W6B_SCENE_XML":
        "grounding/assets/allegro_floating_ur5e_side_r43_plate.xml",
    "W6B_R63_REFERENCE": "runs/w6b/plate_refs/plate_0.json",
    "W6B_LIFT_MM": "0",
    "W6B_LIFT_FRAMES": "0",
    "W6B_R126_BOX_TRACE": "0",
})


def run_recipe(clone, env_extra, tag):
    env = dict(os.environ)
    env.update(env_extra)
    r = subprocess.run([str(PIPE), "retarget/retarget_dual.py"],
                       cwd=clone, env=env, capture_output=True,
                       text=True, timeout=1800)
    if r.returncode != 0:
        sys.stderr.write(r.stdout[-1200:] + r.stderr[-1200:] + "\n")
    rep = {}
    p = clone / "runs/w6b/report.json"
    if p.is_file():
        rep = json.loads(p.read_text())
    tr = []
    p = clone / "runs/w6b/box_trace.json"
    if p.is_file():
        d = json.loads(p.read_text())
        tr = d["frames"] if isinstance(d, dict) else d
    return rep, tr


def _z(trace):
    return [f["pos"][2] for f in trace]


def main() -> int:
    print("=== v11 clean-room verification (fresh processes)")
    clone = pathlib.Path("/tmp/v11_clone_check")
    subprocess.run(["rm", "-rf", str(clone)], check=False)
    subprocess.run(["git", "clone", "-q", str(REPO), str(clone)],
                   check=True)
    (clone / "runs/upstream").mkdir(parents=True, exist_ok=True)
    for tree in ("menagerie", "dex-retargeting"):
        dst = clone / "runs/upstream" / tree
        if dst.is_symlink() or dst.exists():
            subprocess.run(["rm", "-rf", str(dst)], check=False)
        src = REPO / "runs/upstream" / tree
        if src.exists():
            dst.symlink_to(src)
    (clone / "runs/mano").symlink_to(REPO / "runs/mano")
    (clone / "runs/p0/assets").mkdir(parents=True, exist_ok=True)
    (clone / "runs/p0/assets/frames").symlink_to(
        REPO / "runs/p0/assets/frames")

    # -- scenario A: the r43 right-plate master anchor --------------------
    rep, _ = run_recipe(clone, MASTER_ENV, "master")
    g6 = {s: rep.get("per_side", {}).get(s, {}).get(
        "gate6_executed_contact", {}) for s in ("right", "left")}
    check("master: gate6 passes on both sides",
          g6["right"].get("pass") is True and g6["left"].get("pass") is True,
          f"right {g6['right'].get('normal_force_sum_N')} / "
          f"left {g6['left'].get('normal_force_sum_N')} N")
    check("master: the anchor forces reproduce",
          near(g6["right"].get("normal_force_sum_N"), 4.1578, 0.02)
          and near(g6["left"].get("normal_force_sum_N"), 2.0349, 0.02),
          f"{g6['right'].get('normal_force_sum_N')} / "
          f"{g6['left'].get('normal_force_sum_N')} N")
    check("master: the box barely moves",
          near(rep.get("box_motion", {}).get("final_displacement_mm"),
               0.32, 0.05),
          str(rep.get("box_motion", {}).get("final_displacement_mm")))

    # -- scenario B: the dual-plate lift ----------------------------------
    rep, tr = run_recipe(clone, DUAL_ENV, "dual-lift")
    for s in ("right", "left"):
        g6[s] = rep.get("per_side", {}).get(s, {}).get(
            "gate6_executed_contact", {})
    fr_r = g6["right"].get("per_pad", {}).get("palm_collision", {})
    fr_l = g6["left"].get("per_pad", {}).get("palm_left_collision", {})
    check("dual: both plates take real face contact (gate6 PASS both)",
          g6["right"].get("pass") is True and g6["left"].get("pass") is True,
          f"right {g6['right'].get('normal_force_sum_N')} / "
          f"left {g6['left'].get('normal_force_sum_N')} N")
    check("dual: both plate forces clear the 1.090711 N floor",
          g6["right"].get("normal_force_sum_N", 0) >= 1.090711
          and g6["left"].get("normal_force_sum_N", 0) >= 1.090711,
          f"{g6['right'].get('normal_force_sum_N')} / "
          f"{g6['left'].get('normal_force_sum_N')} N")
    check("dual: both plates on the flat end face, inside the depth window",
          fr_r.get("on_end_face") is True and fr_l.get("on_end_face") is True
          and -0.5 <= fr_r.get("surface_gap_mm", 1) <= 0.2
          and -0.5 <= fr_l.get("surface_gap_mm", 1) <= 0.2,
          f"{fr_r.get('surface_gap_mm')} / {fr_l.get('surface_gap_mm')} mm")
    z = _z(tr)
    rise_pk = (max(z) - z[0]) * 1e3
    rise_fn = (z[-1] - z[0]) * 1e3
    check("dual: the box rises 6 mm over the 90 lift frames and HOLDS",
          abs(rise_pk - 6.0784) <= 0.02 and abs(rise_fn - rise_pk) <= 1e-6,
          f"peak {rise_pk:.4f} / final {rise_fn:.4f} mm")
    tbl = 1e9
    if tr:
        last = tr[-1].get("contacts") or []
        tbl = sum(c.get("fn", 0.0) for c in last
                  if "table" in str(c.get("other", "")))
    check("dual: the table support is exactly zero at the end",
          tbl == 0.0, f"{tbl:.3f} N")
    red = sorted(k for k, v in rep.get("verdict_checks", {}).items()
                 if not v)
    check("dual: the red-gate set is the disclosed one",
          red == ["gate1_final_touch", "gate4_joint_limits",
                  "gate5_palm_deviation", "wrist_referenced_error"],
          str(red))

    # -- determinism: run the dual recipe again, compare bit-for-bit ------
    rep2, tr2 = run_recipe(clone, DUAL_ENV, "dual-lift-again")
    z2 = _z(tr2)
    same = (near((max(z2) - z2[0]) * 1e3, rise_pk, 1e-9)
            and near((z2[-1] - z2[0]) * 1e3, rise_fn, 1e-9))
    for s in ("right", "left"):
        g = rep2.get("per_side", {}).get(s, {}).get(
            "gate6_executed_contact", {})
        same = same and near(g.get("normal_force_sum_N"), g6[s].get(
            "normal_force_sum_N"), 1e-9)
    check("dual: a second fresh run is bit-identical", same,
          f"peak {(max(z2) - z2[0]) * 1e3:.6f} mm")

    subprocess.run(["rm", "-rf", str(clone)], check=False)
    subprocess.run(["git", "checkout", "--", "runs/w6b/report.json"],
                   cwd=REPO, check=False)

    bad = [n for n, ok in RESULTS if not ok]
    print()
    if bad:
        print(f"FAIL: {len(bad)} of {len(RESULTS)} checks failed:")
        for n in bad:
            print("   -", n)
        return 1
    print("PASS: every v11 headline number reproduced in a fresh clone")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
