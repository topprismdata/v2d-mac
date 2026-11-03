#!/usr/bin/env python3
"""r184: the fidelity ITERATIVE OPTIMIZER (the v13 review's authorized
"迭代优化方案" for the wrist-trajectory fidelity gate).

Coordinate descent over the fidelity-relevant env knobs, objective =
min max(per-side conf-weighted median err span frac), constraint = the
gate verdict stays green (failing_gates == []).

Each evaluation = one pipeline run in a clean clone (~10 s).  The run
recipe is the DELIVERED v12 recipe (the seated chain, R75=1) plus the
knob overrides; a candidate is accepted only if green AND strictly
better.  The accepted config is printed as the env line and recorded.

Usage:
    python3 tools/w6b/fidelity_opt.py [--clone /tmp/g6_clone] [--budget 40]
"""
import argparse
import json
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

# the delivered v12 recipe (the seated chain), cloned-relative
BASE_ENV = {
    "W6B_SCENE": "ur5e",
    "W6B_SCENE_XML": "grounding/assets/allegro_floating_ur5e_side_r45_ablation.xml",
    "W6B_R75_SIDE_APPROACH": "1",
    "W6B_R23_FLYOVER_MM": "250",
    "W6B_R22_MARGIN_MM": "8",
    "W6B_R169_ENTRY_Z_MM_LEFT": "30",
    "W6B_R155_TRIM_MM_LEFT": "-0.01",
    "W6B_R109_SQUEEZE_MM": "0.4",
    "W6B_LIFT_MM": "6",
    "W6B_LIFT_FRAMES": "90",
    "W6B_R127_LEFT_CURL_DELAY": "80",
    "W6B_R127_CURL_DELAY_RIGHT": "64",
    "W6B_R144_FINGER_LEAD": "100",
    "W6B_R38_SEAT": "left",
    "W6B_R63_REFERENCE": "runs/w6b/plate_refs/plate_2.json",
    "W6B_R63_TARGET": "1",
    "W6B_R18_NATIVE_TARGET": "1",
    "W6B_R50_CURL": "1.25",
    "W6B_R39_TGC": "0",
    "W6B_R36_GSTATIC": "1",
    "W6B_R36_I": "0",
    "W6B_R39_HOLD": "60",
    "W6B_R126_BOX_TRACE": "1",
}

# the coordinate-descent knobs: (env name, [values in try order])
KNOBS = [
    ("W6B_R22_MARGIN_MM", ["4", "12", "0", "16", "-4"]),
    ("W6B_R155_TRIM_MM_LEFT", ["0.0", "-0.02", "0.01", "-0.03"]),
    ("W6B_R23_FLYOVER_MM", ["180", "320", "120", "400"]),
    ("W6B_R169_ENTRY_Z_MM_LEFT", ["20", "40", "10", "50"]),
    ("W6B_R50_CURL", ["1.2", "1.3", "1.15"]),
]


def evaluate(clone, overrides, timeout=600):
    env = dict(os.environ)
    env.update(BASE_ENV)
    env.update(overrides)
    env["W6B_REPORT_JSON"] = "runs/w6b/report.json"
    python = os.path.join(clone, "runs/.venv-retarget/bin/python")
    if not os.path.exists(python):
        python = os.path.join(REPO, "runs/.venv-retarget/bin/python")
    try:
        subprocess.run(
            [python, "retarget/retarget_dual.py"], cwd=clone, env=env,
            capture_output=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        return None
    rp = os.path.join(clone, "runs/w6b/report.json")
    if not os.path.exists(rp):
        return None
    try:
        r = json.load(open(rp))
    except json.JSONDecodeError:
        return None
    te = r.get("trajectory_error", {}).get("wrist_referenced", {})
    per = te.get("per_side", {})
    if not per:
        return None
    fr = {s: float(per[s]["conf_weighted_median_err_span_frac"])
          for s in ("right", "left") if s in per}
    vr = str(r.get("verdict_reason", ""))
    if "failing gates: [" in vr:
        failing = sorted(vr.split("[")[1].split("]")[0].replace(
            "'", "").replace(" ", "").split(","))
    else:
        failing = []
    return {"frac": fr, "failing": failing,
            "obj": max(fr.values())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clone", default="/tmp/g6_clone")
    ap.add_argument("--budget", type=int, default=40)
    a = ap.parse_args()

    best = evaluate(a.clone, {})
    if best is None:
        print("baseline run failed", file=sys.stderr)
        sys.exit(2)
    print(f"baseline: frac={best['frac']} failing={best['failing']} "
          f"obj={best['obj']:.4f}")
    best_cfg = {}
    budget = a.budget - 1

    for knob, values in KNOBS:
        for v in values:
            if budget <= 0:
                break
            cand = dict(best_cfg)
            cand[knob] = v
            res = evaluate(a.clone, cand)
            budget -= 1
            if res is None:
                print(f"  {knob}={v}: RUN FAILED")
                continue
            better = (not res["failing"] and res["obj"] < best["obj"]
                      and (not best["failing"] or True))
            # accept green-over-green strictly better; if baseline red,
            # accept fewer failing gates then better objective
            if res["failing"] and best["failing"]:
                accept = (len(res["failing"]) < len(best["failing"])
                          or (len(res["failing"]) == len(best["failing"])
                              and res["obj"] < best["obj"]))
            else:
                accept = (not res["failing"]
                          and (bool(best["failing"])
                               or res["obj"] < best["obj"]))
            tag = "ACCEPT" if accept else "reject"
            print(f"  {knob}={v}: frac={res['frac']} "
                  f"failing={len(res['failing'])} obj={res['obj']:.4f} {tag}")
            if accept:
                best, best_cfg = res, cand
        if budget <= 0:
            break
        # second pass on the winning knob's neighbourhood
    print("\n== best ==")
    print(f"frac={best['frac']} failing={best['failing']} obj={best['obj']:.4f}")
    print("overrides:", json.dumps(best_cfg, indent=1))
    full = dict(BASE_ENV)
    full.update(best_cfg)
    print("\n".join(f"{k}={v}" for k, v in sorted(full.items())
                    if k in best_cfg or k in
                    ("W6B_R180_DECAY", "W6B_R23_FLYOVER_MM",
                     "W6B_R59_STANDOFF_MM", "W6B_R59_CLEAR_MM")))


if __name__ == "__main__":
    main()
