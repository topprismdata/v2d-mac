"""v9 question 3: is the left hand's 1.48 mm excess really the mount's
lever arm?

The v9 review asked for a falsifiable attribution of the gap between the
right's 4.58 mm and the left's 6.06 mm executed pad distance, and gave
the check:

    dx ~= J(q) K_q^-1 tau_residual

i.e. the tool-tip displacement from the residual joint torques, through
the task-space Jacobian and the joint stiffness.  This script measures
the quantities that formula needs FROM THE RUN ITSELF:

  * the executed joint errors (the commanded vs the actual arm joints),
  * the servo's holding torque per joint (kp * the error),
  * the uncompensated gravity bias per joint,
  * the task-space Jacobian of the palm at the terminal,
  * the contact's reaction on the arm.

and it reports the tool-tip displacement each side's residual explains,
next to the measured one.  If the two sides' predicted displacements do
not differ by the measured 1.48 mm, the lever-arm explanation is
refuted.

    runs/.venv-retarget/bin/python tools/w6b/lever_attribution.py
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
PIPE = REPO / "runs/.venv-retarget/bin/python"
SCRIPT = REPO / "retarget/retarget_dual.py"
SCENE = "grounding/assets/allegro_floating_ur5e_side_r37.xml"

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
    "W6B_DBG_FORCE": "1",
}


def main() -> int:
    env = dict(os.environ)
    env.update(ENV)
    r = subprocess.run([str(PIPE), str(SCRIPT)], cwd=REPO, env=env,
                       capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(r.stdout[-2000:] + r.stderr[-2000:] + "\n")
        raise SystemExit("the pipeline failed")
    lines = [l for l in r.stdout.splitlines() if "dbg-force" in l]

    import mujoco
    m = mujoco.MjModel.from_xml_path(str(REPO / SCENE))
    d = mujoco.MjData(m)
    kp = {}
    for n in ("shoulder_pan", "shoulder_lift", "elbow", "wrist_1",
              "wrist_2", "wrist_3"):
        for side, pfx in (("right", "r"), ("left", "l")):
            a = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR,
                                  f"ur5e_{pfx}_{n}")
            kp[(side, n)] = float(m.actuator_gainprm[a][0])

    out = {}
    for line in lines:
        side = None
        if "right: af=" in line:
            side = "right"
        elif "left: af=" in line:
            side = "left"
        if side is None:
            continue
        # af=... con=... bias=... qpos=... cmd=... err=...
        def grab(tag):
            i = line.find(tag + "=")
            if i < 0:
                return None
            j = line.find("[", i)
            k = line.find("]", j)
            return np.array([float(v) for v in line[j + 1:k].split()])
        af = grab("af")
        con = grab("con")
        bias = grab("bias")
        qpos = grab("qpos")
        cmd = grab("cmd")
        err = grab("err")
        if err is None or bias is None:
            continue
        pfx = "r" if side == "right" else "l"
        # the arm's joints in the model's order
        jids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT,
                                  f"ur5e_{pfx}_{n}_joint") for n in
                ("shoulder_pan", "shoulder_lift", "elbow", "wrist_1",
                 "wrist_2", "wrist_3")]
        qadr = [int(m.jnt_qposadr[j]) for j in jids]
        vadr = [int(m.jnt_dofadr[j]) for j in jids]
        d.qpos[qadr] = qpos
        mujoco.mj_forward(m, d)
        palm = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY,
                                 "palm" if side == "right" else "palm_left")
        jp = np.zeros((3, m.nv))
        jr = np.zeros((3, m.nv))
        mujoco.mj_jacBody(m, d, jp, jr, palm)
        J = jp[:, vadr]
        # the residual torque per joint: the servo's holding torque from
        # the joint error, plus the uncompensated gravity bias
        kpv = np.array([kp[(side, n)] for n in
                        ("shoulder_pan", "shoulder_lift", "elbow", "wrist_1",
                         "wrist_2", "wrist_3")])
        tau_servo = kpv * err
        tau_res = tau_servo + (bias if bias is not None else 0.0)
        # dx = J Kq^-1 tau_res  (Kq is diagonal: the actuator's kp)
        dx = J @ (tau_res / np.maximum(kpv, 1e-9))
        out[side] = {
            "the joint errors rad": [round(float(v), 5) for v in err],
            "the servo's holding torque N.m": [round(float(v), 3)
                                               for v in tau_servo],
            "the uncompensated gravity bias N.m": [round(float(v), 3)
                                                   for v in bias],
            "the residual torque N.m": [round(float(v), 3) for v in tau_res],
            "the tool-tip displacement it explains mm":
                [round(float(v) * 1e3, 3) for v in dx],
            "the magnitude mm": round(float(np.linalg.norm(dx)) * 1e3, 3),
        }
        print(f"== {side}")
        print(f"   the joint errors {np.round(err, 5).tolist()}")
        print(f"   the servo's holding torque {np.round(tau_servo, 3).tolist()}")
        print(f"   the uncompensated gravity bias "
              f"{np.round(bias, 3).tolist()}")
        print(f"   the residual {np.round(tau_res, 3).tolist()}")
        print(f"   -> the tool-tip displacement it explains: "
              f"{np.round(dx * 1e3, 3).tolist()} mm "
              f"(|d| = {np.linalg.norm(dx)*1e3:.3f} mm)")

    rep = json.loads((REPO / "runs/w6b/report.json").read_text())
    meas = {}
    for side in ("right", "left"):
        meas[side] = {
            "the executed pad distance mm": round(float(
                rep["per_side"][side]["gate1_final_touch"].get(
                    "executed_rect_dist_mm", float("nan"))), 3)
            if "executed_rect_dist_mm" in rep["per_side"][side][
                "gate1_final_touch"] else None,
            "the executed forward axis deg": round(float(
                rep["per_side"][side]["gate1_final_touch"]["forward_axis_deg"]),
                2),
        }
    print()
    for side in ("right", "left"):
        if side in out:
            print(f"   the {side}: the residual explains "
                  f"{out[side]['the magnitude mm']:.3f} mm of the tool-tip "
                  f"displacement; the measured executed pad distance is "
                  f"{meas[side]}")
    out["the measured"] = meas
    (REPO / "runs/w6b/lever_attribution.json").write_text(
        json.dumps(out, indent=2, default=float))
    print("\n   written: runs/w6b/lever_attribution.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
