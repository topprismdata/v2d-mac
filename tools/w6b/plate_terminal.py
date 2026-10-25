"""r123 (grant B): wire a feasible PALM-PLATE pose into the right's
terminal reference and run the pipeline.

The v10 review's authorization B makes the palm plate a legitimate
grasp contact surface (an independent design branch; the original hand
configuration stays the immutable baseline).  The plate search
(runs/w6b/palm_plate_search.json) found 24/3456 feasible poses on the
side cell: the plate tangent to the end face, the fingertips 9-158 mm
clear of the box, the table 89 mm clear.  The plate's normal force is
carried by the ARM's servo, not the 0.04-0.18 N.m finger servos.

THIS TOOL bakes one feasible pose (chosen by index into the feasible
list) into the RIGHT side of a copy of the delivered terminal
reference: the palm pose + rotation from the search row, the finger
joints set to the search row's curl (the same base + curl*0.35*(hi-base)
transform the search used).  The LEFT side keeps a baked thin-pad curl
(its winning ladder step from r122: curl 1.25 -> 1.391 N, gate6 PASS).

usage:
    runs/.venv-retarget/bin/python tools/w6b/plate_terminal.py IDX [SCENE]
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

import mujoco
import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
PIPE = REPO / "runs/.venv-retarget/bin/python"
SCRIPT = REPO / "retarget/retarget_dual.py"
REFERENCE = REPO / "runs/w6b/palm_pose_refine.json"
SEARCH = REPO / "runs/w6b/palm_plate_search.json"
WORK = REPO / "runs/w6b/plate_refs"
REPORT = REPO / "runs/w6b/report.json"

SCENE = sys.argv[2] if len(sys.argv) > 2 else \
    "grounding/assets/allegro_floating_ur5e_side_r41_mixed.xml"
LEFT_CURL = 1.25

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


def curl_fingers(m, curl):
    """the search's own curl->joint transform, per 4-joint group"""
    out = {}
    for f in ("ff", "mf", "rf", "th"):
        v = []
        for k in range(4):
            j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{f}j{k}")
            lo, hi = m.jnt_range[j]
            base = 0.5 * (lo + hi)
            v.append(base + curl * 0.35 * (hi - base))
        out[f] = v
    return out


def build_ref(idx: int) -> pathlib.Path:
    row = json.loads(SEARCH.read_text())["feasible"][idx]
    m = mujoco.MjModel.from_xml_path(str(REPO / SCENE))
    ref = json.loads(REFERENCE.read_text())
    r = ref["sides"]["right"]
    cand = r["best"][0] if isinstance(r.get("best"), list) else r["best"]
    cand["the palm"] = [float(v) for v in row["palm"]]
    cand["the palm_R"] = [[float(v) for v in rr] for rr in row["R"]]
    cand["the fingers"] = curl_fingers(m, float(row["curl"]))
    # the left: its winning thin-pad curl (r122), same transform as the
    # thinpad scan
    l = ref["sides"]["left"]
    lcand = l["best"][0] if isinstance(l.get("best"), list) else l["best"]
    lf = lcand.get("fingers") or {}
    for f in ("ff", "mf", "rf", "th"):
        v = list(lf.get(f, [0.0] * 4))
        for j in (1, 2, 3):
            v[j] *= LEFT_CURL
        lf[f] = v
    lcand["fingers"] = lf
    WORK.mkdir(parents=True, exist_ok=True)
    p = WORK / f"plate_{idx}.json"
    p.write_text(json.dumps(ref, indent=1))
    return p, row


def run(ref_path: pathlib.Path) -> dict:
    env = dict(os.environ)
    env.update(BASE)
    env["W6B_R63_REFERENCE"] = str(ref_path)
    r = subprocess.run([str(PIPE), str(SCRIPT)], cwd=REPO, env=env,
                       capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(r.stdout[-1500:] + r.stderr[-1500:] + "\n")
        raise SystemExit("the pipeline failed")
    return json.loads(REPORT.read_text())


def main() -> int:
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    ref, row = build_ref(idx)
    rep = run(ref)
    out = {"scene": SCENE, "plate_row": row, "left_curl": LEFT_CURL}
    for s in ("right", "left"):
        ps = rep["per_side"][s]
        g6 = ps["gate6_executed_contact"]
        pl = ps["gate1_final_touch"]["executed_state_evidence"].get(
            "palm_plate_executed_state")
        out[s] = {
            "gate6": {k: g6[k] for k in
                      ("n_real_contacts", "normal_force_sum_N",
                       "lift_need_N", "pass")},
            "per_pad": g6["per_pad"],
            "palm_plate": pl,
            "box_motion_mm": rep.get("box_motion", {}).get(
                "final_displacement_mm"),
        }
        print(f"{s}: g6 F={g6['normal_force_sum_N']} "
              f"need={g6['lift_need_N']} real={g6['n_real_contacts']} "
              f"pass={g6['pass']} | plate={pl} | "
              f"box={rep.get('box_motion', {}).get('final_displacement_mm')}")
    print("red gates:", [k for k, v in rep["verdict_checks"].items() if not v])
    pathlib.Path("runs/w6b/plate_terminal.json").write_text(
        json.dumps(out, indent=1))
    ref.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
