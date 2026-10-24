#!/usr/bin/env python3
"""r116: the low-base cell's candidates, with the PATH quality.

The squeeze physics leg ranks candidates by the executed normal force,
but a candidate whose APPROACH PATH cannot be reconstructed produces a
meaningless force: on r38 the right's candidate 4 showed 1.057 N (the
closest to the 1.091 N a lift needs) while its path was 259 mm off --
the terminal was never actually reached.

So the candidate table has to carry the path's own error next to the
force and the gates.

usage: r38_candidate_table.py <refine.json> <scene.xml>
"""
import json
import os
import pathlib
import re
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
PIPE = REPO / "runs/.venv-retarget/bin/python"
SCRIPT = REPO / "retarget/retarget_dual.py"

BASE = {
    "W6B_SCENE": "ur5e",
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


def run(ref_path, scene):
    env = dict(os.environ)
    env.update(BASE)
    env["W6B_SCENE_XML"] = scene
    env["W6B_R63_REFERENCE"] = str(ref_path)
    r = subprocess.run([str(PIPE), str(SCRIPT)], cwd=REPO, env=env,
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(r.stdout[-2000:] + r.stderr[-2000:])
    rep = json.loads((REPO / "runs/w6b/report.json").read_text())
    m = re.search(r"\[dbg-side\].*", r.stdout)
    return rep, (m.group(0) if m else "")


def metrics(rep, side):
    ps = rep["per_side"][side]
    g1 = ps["gate1_final_touch"]
    ev = g1.get("executed_state_evidence", {}).get(
        "gate1_v3_executed_state", {})
    pads = ev.get("pads", {})
    force = float(sum(p.get("contact_normal_force_N", 0.0)
                      for p in pads.values() if p.get("in_contact")))
    return {
        "force_N": round(force, 4),
        "cmd_rect_mm": g1.get("rect_dist_mm"),
        "exec_rect_mm": ev.get("rect_dist_mm"),
        "cmd_fwd_deg": g1.get("forward_axis_deg"),
        "exec_fwd_deg": ev.get("forward_axis_deg"),
        "green": [g for g in GREEN if ps[g].get("pass")],
        "red": [g for g, v in ps.items()
                if isinstance(v, dict) and v.get("pass") is False],
    }


def main():
    ref_path = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else
                            "runs/w6b/palm_pose_refine_r38.json")
    scene = sys.argv[2] if len(sys.argv) > 2 else (
        "grounding/assets/allegro_floating_ur5e_side_r38.xml")
    ref = json.loads(ref_path.read_text())
    base_best = {s: ref["sides"][s]["best"][0] for s in ("right", "left")}
    out = []
    for side in ("right", "left"):
        for i, cand in enumerate(ref["sides"][side]["best"]):
            trial = {"scene": ref.get("scene"), "sides": {}}
            for s2 in ("right", "left"):
                trial["sides"][s2] = {"best": [cand if s2 == side
                                              else base_best[s2]]}
            p = pathlib.Path(f"/tmp/r116_{side}_{i}.json")
            p.write_text(json.dumps(trial))
            rep, dbg = run(p, scene)
            m = metrics(rep, side)
            path = {}
            for s2 in ("right", "left"):
                mm = re.search(rf"'{s2}': \{{'max_m': ([0-9.eE+-]+), "
                               rf"'rms_m': ([0-9.eE+-]+)", dbg)
                if mm:
                    path[s2] = (round(float(mm.group(1)) * 1e3, 2),
                                round(float(mm.group(2)) * 1e3, 2))
            row = {"side": side, "index": i, "path_max_mm":
                   path.get(side, (None, None))[0],
                   "path_rms_mm": path.get(side, (None, None))[1],
                   "other_path_max_mm": path.get(
                       "left" if side == "right" else "right",
                       (None, None))[0],
                   **m}
            out.append(row)
            print(f"{side:5s} cand {i}: the path max "
                  f"{row['path_max_mm']:8.2f} mm rms "
                  f"{row['path_rms_mm']:7.2f} mm | the force "
                  f"{row['force_N']:.4f} N | the exec rect "
                  f"{row['exec_rect_mm']} mm fwd {row['exec_fwd_deg']} | "
                  f"the red {row['red']}")
            p.unlink()
    out_path = REPO / "runs/w6b/r38_candidate_table.json"
    out_path.write_text(json.dumps({"scene": scene, "rows": out}, indent=1))
    print("written:", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
