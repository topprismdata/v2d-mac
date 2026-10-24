"""Clean-room verification of the v10 headline numbers.

Every number quoted in docs/gpt_v10_package.md is re-derived here by a
FRESH process running a COMMITTED script, and the artifacts are re-read
from disk afterwards.  Nothing reads the session state.

    runs/.venv-retarget/bin/python tools/w6b/verify_v10.py

The v10 campaign is the lift: v9 ruled the grip too weak to lift the
box, so this round measured the static balance, tried five ways to
raise the contact force, attributed the residual to a lever arm, ran a
local Pareto scan on the terminal posture, and priced the visual gap
against the physics gap with a base-height sweep.
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
TOOLS = REPO / "tools/w6b"
RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(ok), detail))
    print(f"  {'OK ' if ok else 'FAIL'}  {name}"
          + (f": {detail}" if detail else ""))


def run_tool(script: str, env_extra: dict | None = None,
             timeout: int = 2400) -> tuple[int, str]:
    env = dict(os.environ)
    env.update(env_extra or {})
    r = subprocess.run([str(PIPE), str(TOOLS / script)], cwd=REPO, env=env,
                       capture_output=True, text=True, timeout=timeout)
    return r.returncode, r.stdout + r.stderr


def near(a: float, b: float, tol: float) -> bool:
    return abs(float(a) - float(b)) <= tol


def main() -> int:
    print("=== v10 clean-room verification (fresh processes)")

    # 1. the delivered cell still reproduces (v9's headline numbers)
    rc, out = run_tool("verify_v9.py", timeout=1200)
    check("the delivered cell's v9 headline numbers reproduce",
          rc == 0 and "PASS: every v9 headline number reproduced"
          in out,
          "verify_v9.py's exit " + str(rc))

    # 2. the lift's static balance
    rc, out = run_tool("lift_feasibility.py")
    lf = json.loads((REPO / "runs/w6b/lift_feasibility.json").read_text())
    check("the lift feasibility tool ran", rc == 0, "exit " + str(rc))
    check("the box's weight", near(lf["weight_N"], 3.9279, 1e-3),
          f"{lf['weight_N']:.4f} N (expected 3.9279)")
    check("the measured normal force",
          near(lf["measured_normal_force_N"]["right"], 0.2549, 5e-4)
          and near(lf["measured_normal_force_N"]["left"], 0.7377, 5e-4),
          f"{lf['measured_normal_force_N']['right']} / "
          f"{lf['measured_normal_force_N']['left']} N")
    deficit = 1.0 - (lf["available_friction_N"] / lf["weight_N"])
    check("the friction deficit", near(deficit, 0.545, 5e-3),
          f"{deficit*100:.1f}% short (expected 54.5%)")
    check("the required force per hand",
          near(lf["required_per_hand_N"], 1.0911, 5e-4),
          f"{lf['required_per_hand_N']:.4f} N")
    check("the fingers' torque capacity covers it",
          lf["required_squeeze_within_the_hardware_capacity"] is True
          and near(lf["normal_force_capacity_per_hand_N"]["right"],
                   27.756, 5e-3),
          f"{lf['normal_force_capacity_per_hand_N']['right']:.3f} N per hand")
    check("the static balance is infeasible with the measured state",
          lf["static_feasible_with_the_measured_state"] is False)
    check("the local execution lifts nothing",
          abs(lf["local_execution"]["dz_mm"]) < 1e-3
          and abs(lf["local_execution"]["drift_mm"]) < 1e-3,
          f"dz {lf['local_execution']['dz_mm']:.2e} mm, "
          f"drift {lf['local_execution']['drift_mm']:.2e} mm")

    # 3. the Pareto scan on the terminal posture
    rc, out = run_tool("pareto_scan.py")
    ps = json.loads((REPO / "runs/w6b/pareto_scan.json").read_text())
    rows = {(r["curl"], r["squeeze_mm"]): r for r in ps}
    base = rows[(1.0, 0.0)]
    check("the Pareto scan ran", rc == 0, "exit " + str(rc))
    check("the delivered row's executed pad distance",
          near(base["right"]["the executed pad distance mm"], 4.58, 0.05)
          and near(base["left"]["the executed pad distance mm"], 6.06, 0.05),
          f"{base['right']['the executed pad distance mm']} / "
          f"{base['left']['the executed pad distance mm']} mm")
    check("the delivered row's executed forward axis",
          near(base["right"]["the EXECUTED forward axis deg"], 33.3, 0.15)
          and near(base["left"]["the EXECUTED forward axis deg"], 32.9, 0.15),
          f"{base['right']['the EXECUTED forward axis deg']} / "
          f"{base['left']['the EXECUTED forward axis deg']} deg")
    check("the curl 1.2 row loses the right's contact",
          rows[(1.2, 0.0)]["right"]["the contacting pads"] == []
          and rows[(1.2, 0.0)]["right"]["the normal force N"] == 0.0,
          "the contacting pads " + str(
              rows[(1.2, 0.0)]["right"]["the contacting pads"]))
    check("the 0.05 mm squeeze is the cost-free improvement",
          near(rows[(1.0, 0.05)]["right"]["the executed pad distance mm"],
               4.55, 0.05)
          and near(rows[(1.0, 0.05)]["right"]["the normal force N"], 0.3797,
                   5e-4),
          f"{rows[(1.0, 0.05)]['right']['the executed pad distance mm']} mm, "
          f"{rows[(1.0, 0.05)]['right']['the normal force N']} N")

    # 4. the lever-arm attribution
    rc, out = run_tool("lever_attribution.py")
    la = json.loads((REPO / "runs/w6b/lever_attribution.json").read_text())
    check("the lever attribution ran", rc == 0, "exit " + str(rc))
    left = la["the measured"]["left"]
    right = la["the measured"]["right"]
    check("the left's predicted distance matches the measurement",
          near(left["the predicted gap from the residual mm"], 6.009, 0.01)
          and near(left["the executed pad distance mm"], 6.06, 0.05),
          f"predicted {left['the predicted gap from the residual mm']} vs "
          f"measured {left['the executed pad distance mm']} mm")
    check("the right's hypothesis is REFUTED",
          near(right["the predicted gap from the residual mm"], 12.01, 0.02)
          and near(right["the executed pad distance mm"], 4.58, 0.05),
          f"predicted {right['the predicted gap from the residual mm']} vs "
          f"measured {right['the executed pad distance mm']} mm")

    # 5. the base-height sweep
    rc, out = run_tool("base_height_probe.py")
    bp = json.loads((REPO / "runs/w6b/base_height_probe.json").read_text())
    byz = {r["base_z"]: r for r in bp["rows"]}
    check("the base probe ran", rc == 0, "exit " + str(rc))
    check("the current base reproduces the known-good branch",
          byz[1.35]["sides"]["right"]["branch_delta_rad"] < 1e-6
          and byz[1.35]["sides"]["left"]["branch_delta_rad"] < 1e-6,
          "the branch delta " +
          str(byz[1.35]["sides"]["right"]["branch_delta_rad"]))
    check("the left's entry angle flattens at a lower base",
          byz[1.35]["sides"]["left"]["entry_deg_down"] > 40.0
          and byz[1.05]["sides"]["left"]["entry_deg_down"] < 5.0,
          f"{byz[1.35]['sides']['left']['entry_deg_down']} deg -> "
          f"{byz[1.05]['sides']['left']['entry_deg_down']} deg")
    check("a lower base leaves the path-verified branch",
          byz[1.05]["sides"]["left"]["branch_delta_rad"] > 0.60
          and byz[1.05]["sides"]["right"]["branch_delta_rad"] > 0.60,
          f"{byz[1.05]['sides']['left']['branch_delta_rad']} / "
          f"{byz[1.05]['sides']['right']['branch_delta_rad']} rad")

    # 6. the low-base cell's candidate table
    rc, out = run_tool("r38_candidate_table.py")
    tb = json.loads((REPO / "runs/w6b/r38_candidate_table.json").read_text())
    rrows = {(r["side"], r["index"]): r for r in tb["rows"]}
    check("the r38 candidate table ran", rc == 0, "exit " + str(rc))
    check("the right's 1.0568 N candidate is a path artifact",
          near(rrows[("right", 4)]["force_N"], 1.0568, 5e-4)
          and rrows[("right", 4)]["path_max_mm"] > 100.0,
          f"the force {rrows[('right', 4)]['force_N']} N with the path "
          f"{rrows[('right', 4)]['path_max_mm']} mm")
    check("the low base's valid candidates are weaker than the delivered",
          rrows[("right", 2)]["path_max_mm"] < 10.0
          and rrows[("right", 2)]["force_N"] < 0.2972
          and rrows[("left", 0)]["path_max_mm"] < 10.0
          and rrows[("left", 0)]["force_N"] < 0.7177,
          f"the right {rrows[('right', 2)]['force_N']} N, the left "
          f"{rrows[('left', 0)]['force_N']} N")
    check("a zero-contact candidate passes every gate (the gate hole)",
          rrows[("right", 1)]["force_N"] == 0.0
          and rrows[("right", 1)]["red"] == [],
          "the force " + str(rrows[("right", 1)]["force_N"]) + " N, the red "
          + str(rrows[("right", 1)]["red"]))

    # 6b. the r118 root causes (the static, recomputable ones)
    fc = json.loads((REPO / "runs/w6b/finger_force_ceiling.json")
                    .read_text())
    check("the hardware force ceiling is well above the lift's need",
          fc["sides"]["right"]["ff"]["the hardware force ceiling N"] > 2.0
          and fc["sides"]["left"]["ff"]["the hardware force ceiling N"] > 2.0,
          "the right ff "
          + str(fc["sides"]["right"]["ff"]["the hardware force ceiling N"])
          + " N, the left ff "
          + str(fc["sides"]["left"]["ff"]["the hardware force ceiling N"])
          + " N")
    check("the middle pad's own ceiling is BELOW the lift's need",
          fc["sides"]["right"]["mf"]["the hardware force ceiling N"] < 1.0911
          and fc["sides"]["left"]["mf"]["the hardware force ceiling N"] < 1.0911,
          "the right mf "
          + str(fc["sides"]["right"]["mf"]["the hardware force ceiling N"])
          + " N, the left mf "
          + str(fc["sides"]["left"]["mf"]["the hardware force ceiling N"])
          + " N")
    fdl = json.loads((REPO / "runs/w6b/force_depth_law.json").read_text())
    green = [r for r in fdl
             if r["right"]["the command pads in the rect"] >= 2
             and not r["right"]["the command burial"]]
    check("the green window's force stays far below the lift's need",
          all(r["right"]["the normal force N"] < 1.0911 for r in green)
          and all(r["left"]["the normal force N"] < 1.0911 for r in green),
          "the best green row: the right "
          + str(max(r["right"]["the normal force N"] for r in green))
          + " N, the left "
          + str(max(r["left"]["the normal force N"] for r in green)) + " N")
    cb = json.loads((REPO / "runs/w6b/curl_baked_scan.json").read_text())
    left_strong = [r for r in cb if r["the left curl"] > 1.0
                   and r["left"]["the normal force N"] >= 1.0911]
    check("a deeper per-side curl reaches the required force",
          bool(left_strong),
          "the left: " + ", ".join(
              "the curl " + str(r["the left curl"]) + " -> "
              + str(r["left"]["the normal force N"]) + " N"
              for r in left_strong))
    check("but every such curl breaks the command state's geometry",
          all(r["left"]["the command pads in the rect"] == 0
              for r in left_strong),
          "the rect: " + ", ".join(
              str(r["left"]["the command rect mm"]) for r in left_strong))
    ssb = json.loads((REPO / "runs/w6b/surface_seated_build.json").read_text())
    v3_green = [r for r in ssb if r.get("the v3 eligible", 0) >= 1]
    check("the surface-seated geometry makes the v3 criterion green",
          len(v3_green) >= 4,
          str(len(v3_green)) + " of " + str(len(ssb)) + " rows")
    check("but it makes the v1v2 criterion red (they are exclusive)",
          all(r.get("the v1v2 pads in the rect") == 0 for r in v3_green),
          "the rect: " + ", ".join(
              str(r.get("the v1v2 rect mm")) for r in v3_green))

    # 7. the delivered cell reproduces in a FRESH CLONE (only the
    # committed files; the four external asset trees are linked in)
    clone = pathlib.Path("/tmp/v10_clone_check")
    subprocess.run(["rm", "-rf", str(clone)], check=False)
    rc, _ = run_tool("rebuild_terminal_reference.py")
    check("the terminal reference rebuilds", rc == 0, "exit " + str(rc))
    subprocess.run(["git", "clone", "-q", str(REPO), str(clone)], check=True)
    (clone / "runs/upstream").mkdir(parents=True, exist_ok=True)
    for tree, src in (("menagerie", REPO / "runs/upstream/menagerie"),
                      ("dex-retargeting",
                       REPO / "runs/upstream/dex-retargeting")):
        dst = clone / "runs/upstream" / tree
        if dst.is_symlink() or dst.exists():
            subprocess.run(["rm", "-rf", str(dst)], check=False)
        dst.symlink_to(src)
    (clone / "runs/mano").symlink_to(REPO / "runs/mano")
    (clone / "runs/p0/assets").mkdir(parents=True, exist_ok=True)
    (clone / "runs/p0/assets/frames").symlink_to(
        REPO / "runs/p0/assets/frames")
    env = dict(os.environ)
    env.update({
        "W6B_SCENE": "ur5e",
        "W6B_SCENE_XML": "grounding/assets/allegro_floating_ur5e_side_r37.xml",
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
    })
    r = subprocess.run([str(PIPE), "retarget/retarget_dual.py"], cwd=clone,
                       env=env, capture_output=True, text=True, timeout=1200)
    ok = r.returncode == 0
    crep = {}
    if ok:
        crep = json.loads((clone / "runs/w6b/report.json").read_text())
    def _exec(side):
        g1 = crep.get("per_side", {}).get(side, {}).get(
            "gate1_final_touch", {})
        return g1.get("executed_state_evidence", {}).get("rect_dist_mm")
    check("the fresh clone runs the delivered recipe", ok,
          "the exit " + str(r.returncode))
    check("the fresh clone reproduces the executed pad distances",
          near(_exec("right"), 4.58, 0.05) and near(_exec("left"), 6.06, 0.05),
          f"{_exec('right')} / {_exec('left')} mm")
    red = [k for k, v in crep.get("verdict_checks", {}).items() if not v]
    check("the fresh clone has the same two red gates",
          sorted(red) == ["gate4_joint_limits", "wrist_referenced_error"],
          str(red))
    subprocess.run(["rm", "-rf", str(clone)], check=False)

    # the delivered report must be restored at the end (the tools above
    # overwrite runs/w6b/report.json)
    subprocess.run(["git", "checkout", "--", "runs/w6b/report.json"],
                   cwd=REPO, check=False)

    bad = [n for n, ok, _ in RESULTS if not ok]
    print()
    if bad:
        print(f"FAIL: {len(bad)} of {len(RESULTS)} checks failed:")
        for n in bad:
            print("   -", n)
        return 1
    print("PASS: every v10 headline number reproduced in a fresh process")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
