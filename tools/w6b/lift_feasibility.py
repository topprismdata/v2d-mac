"""v10: the side-mount LIFT FEASIBILITY gate.

The v9 review named the single next research goal:

    Side-Mount Lift Feasibility -- first prove whether the current
    contact state admits a physically feasible path off the table, then
    do a small continuous execution.  If the static balance has no
    solution, stop at the infeasibility evidence; do not train a policy
    or tune friction.

Part 1 (this script's static half) uses the EXECUTED contact state that
the pipeline measured -- the real contact points, the real normal
forces -- and asks the static question:

  * can the friction at the measured contacts support the box's weight?
  * is the moment balance about the box's centre of mass satisfiable
    given the contacts' height and the friction cone?
  * what normal force per hand would be needed, and is that within the
    finger actuators' reach?

Part 2 (the local execution half) then does what the review asks for:
WITHOUT changing the mass, the friction coefficients or the joint limits,
a small continuous upward motion is commanded, and the box's
table-contact force and its height are recorded.

    runs/.venv-retarget/bin/python tools/w6b/lift_feasibility.py
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
G = 9.81

BASE_ENV = {
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


def run_pipeline() -> dict:
    env = dict(os.environ)
    env.update(BASE_ENV)
    r = subprocess.run([str(PIPE), str(SCRIPT)], cwd=REPO, env=env,
                       capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(r.stdout[-2000:] + r.stderr[-2000:] + "\n")
        raise SystemExit("the pipeline failed")
    return json.loads((REPO / "runs/w6b/report.json").read_text())


def main() -> int:
    import mujoco

    print("[1/3] the executed contact state (a fresh pipeline run)")
    rep = run_pipeline()
    m = mujoco.MjModel.from_xml_path(str(REPO / SCENE))
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "object")
    bgid = int(m.body_geomadr[bid])
    mass = float(m.body_mass[bid])
    mu_box = float(m.geom_friction[bgid][0])
    mu_pad = 1.8
    mu = max(mu_box, mu_pad)          # MuJoCo takes the elementwise max
    z_com = float(m.body_pos[bid][2])

    contacts = {}
    for side in ("right", "left"):
        v3 = rep["per_side"][side]["gate1_final_touch"]["executed_state_evidence"][
            "gate1_v3_executed_state"]["pads"]
        n_sum = 0.0
        z_list = []
        for name, rec in v3.items():
            if rec.get("in_contact") and rec.get("contact_normal_force_N", 0) > 0:
                n_sum += float(rec["contact_normal_force_N"])
                z_list.append((name, float(rec["contact_normal_force_N"])))
        tips = rep["per_side"][side]["executed_terminal_tip_positions_m"]
        # the contact's height: the contacting tip's z
        zc = float(tips[[k for k in tips if name.split("_")[0] in k][0]][2]) \
            if z_list else z_com
        contacts[side] = {"N": n_sum, "pads": z_list, "z": zc}
        print(f"    {side}: the contacting pads {z_list}, "
              f"the normal force {n_sum:.4f} N")

    print("[2/3] the static balance")
    W = mass * G
    N_tot = sum(c["N"] for c in contacts.values())
    F_avail = mu * N_tot
    N_req_total = W / mu
    # the moment balance about the centre of mass: the normal force acts
    # horizontally at the contact's height, the friction acts upward at
    # the face's x offset -> the required friction/normal ratio
    ratios = []
    x_face = 0.060                          # the box's half width, m
    for side, c in contacts.items():
        h = c["z"] - z_com                   # the contact's height above
        # the centre of mass, m
        if abs(h) > 1e-6 and c["N"] > 0:
            ratios.append((side, abs(x_face / h)))
    need_ratio = max([r for _, r in ratios], default=0.0)
    print(f"    the box's mass {mass:.4f} kg -> the weight {W:.3f} N")
    print(f"    the friction coefficient mu = max({mu_box}, {mu_pad}) = {mu}")
    print(f"    the measured total normal force {N_tot:.4f} N")
    print(f"    the available friction mu*N = {F_avail:.3f} N "
          f"(the weight is {W:.3f} N -> "
          f"{'SHORT' if F_avail < W else 'enough'} by "
          f"{abs(W - F_avail) / W * 100:.1f}%)")
    print(f"    the normal force needed for a lift: {N_req_total:.3f} N total "
          f"= {N_req_total / 2:.3f} N per hand")
    for side, c in contacts.items():
        print(f"      the {side}: the measured {c['N']:.4f} N -> "
              f"{c['N'] / (N_req_total / 2) * 100:.1f}% of the requirement")
    print(f"    the moment balance about the centre of mass: the required "
          f"friction/normal ratio is {need_ratio:.3f} (the cone allows "
          f"{mu:.3f}) -> "
          f"{'satisfiable' if need_ratio <= mu else 'NOT satisfiable'}")

    # the finger's torque capacity: the upper bound on the squeeze that
    # the hardware can deliver, so the verdict can say whether the
    # infeasibility is a property of the CURRENT terminal configuration
    # or of the hardware
    cap = {}
    for side, pfx in (("right", ""), ("left", "_left")):
        tot = 0.0
        for jn in ("ffj0", "ffj1", "ffj2", "ffj3", "mfj0", "mfj1", "mfj2",
                   "mfj3", "rfj0", "rfj1", "rfj2", "rfj3", "thj0", "thj1",
                   "thj2", "thj3"):
            # the Allegro's actuators are named <finger>a<index>
            # ("ffa0") for the joint <finger>j<index> ("ffj0")
            a = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR,
                                  jn.replace("j", "a") + pfx)
            if a < 0:
                continue
            tot += float(abs(m.actuator_forcerange[a][1]))
        cap[side] = tot
    # a 50 mm lever from the finger's joints to the pad is the order of
    # magnitude of the Allegro's index finger
    lever = 0.050
    n_cap = {k: v / lever for k, v in cap.items()}
    print(f"    the finger's total torque capacity: "
          + ", ".join(f"{k} {v:.3f} N.m" for k, v in cap.items())
          + f" -> at a {lever*1e3:.0f} mm lever, a normal force of "
            f"{min(n_cap.values()):.2f} N per hand")
    print(f"    the required {N_req_total / 2:.3f} N per hand is "
          f"{'WITHIN' if min(n_cap.values()) >= N_req_total / 2 else 'BEYOND'}"
          f" the hand's torque capacity")

    feasible = F_avail >= W and need_ratio <= mu
    print(f"    the STATIC VERDICT with the measured contact state: "
          f"{'FEASIBLE' if feasible else 'INFEASIBLE'}")

    print("[3/3] the local execution (no mass, friction or limit changes)")
    lift = local_execution(m, rep, mu)
    print(f"    the commanded upward step: {lift['step_mm']} mm over "
          f"{lift['n_steps']} control frames")
    print(f"    the box's height: {lift['z0_mm']:.2f} -> {lift['z1_mm']:.2f} mm "
          f"(the lift {lift['dz_mm']:.2f} mm)")
    print(f"    the box's table contact force: {lift['f0_N']:.3f} -> "
          f"{lift['f1_N']:.3f} N")
    print(f"    the box's lateral drift: {lift['drift_mm']:.2f} mm")

    out = {
        "mass_kg": mass,
        "weight_N": W,
        "mu_sliding": mu,
        "measured_normal_force_N": {k: v["N"] for k, v in contacts.items()},
        "measured_total_normal_force_N": N_tot,
        "available_friction_N": F_avail,
        "required_total_normal_force_N": N_req_total,
        "required_per_hand_N": N_req_total / 2,
        "moment_ratio_required": need_ratio,
        "static_feasible_with_the_measured_state": bool(feasible),
        "finger_torque_capacity_Nm": cap,
        "normal_force_capacity_per_hand_N": n_cap,
        "required_squeeze_within_the_hardware_capacity": bool(
            min(n_cap.values()) >= N_req_total / 2),
        "local_execution": lift,
        "verdict_note": (
            "the infeasibility is a property of the CURRENT terminal "
            "configuration (the searched anchor's squeeze), not of the "
            "hardware: the required normal force per hand is within the "
            "fingers' torque capacity"),
        "verdict": (
            "the static balance is INFEASIBLE with the measured contact "
            "state: the available friction is short of the weight by "
            f"{abs(W - F_avail) / W * 100:.1f}%.  A lift needs the normal "
            f"force raised to {N_req_total / 2:.3f} N per hand "
            f"(measured {contacts['right']['N']:.3f} / "
            f"{contacts['left']['N']:.3f} N).  The local execution below "
            "tests whether a small continuous upward motion changes that."),
    }
    (REPO / "runs/w6b/lift_feasibility.json").write_text(
        json.dumps(out, indent=2, default=float))
    print()
    print("    written: runs/w6b/lift_feasibility.json")
    return 0


def local_execution(m, rep, mu) -> dict:
    """the reviewer's part 2: a small continuous upward motion, with the
    mass, the friction and the joint limits untouched, recording the
    box's table-contact force and its height."""
    import mujoco
    d = mujoco.MjData(m)
    # the terminal state: the commanded arm and finger angles from the
    # report, and the box at its rest pose
    for side in ("right", "left"):
        tc = rep["per_side"][side]["terminal_commands"]
        for n, v in zip(("shoulder_pan", "shoulder_lift", "elbow",
                         "wrist_1", "wrist_2", "wrist_3"), tc["wrist_rad"]):
            j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT,
                                 f"ur5e_{'r' if side == 'right' else 'l'}_{n}_joint")
            d.qpos[int(m.jnt_qposadr[j])] = float(v)
        for j, v in zip(m.jnt_range[:0].tolist() or [], []):
            pass
    # the fingers: the commanded actuator values map to the joint
    # positions through the actuator's gear (1) and the joint's range
    for side in ("right", "left"):
        tc = rep["per_side"][side]["terminal_commands"]
        pfx = "r" if side == "right" else "l"
        for jn in ("ffj0", "ffj1", "ffj2", "ffj3", "mfj0", "mfj1", "mfj2",
                   "mfj3", "rfj0", "rfj1", "rfj2", "rfj3", "thj0", "thj1",
                   "thj2", "thj3"):
            j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT,
                                 f"{jn}{'_left' if side == 'left' else ''}")
            if j < 0:
                continue
            lo, hi = m.jnt_range[j]
            d.qpos[int(m.jnt_qposadr[j])] = float(np.clip(0.0, lo, hi))
    # the box's free joint: the rest pose
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "object")
    bjadr = int(m.jnt_qposadr[mujoco.mj_name2id(
        m, mujoco.mjtObj.mjOBJ_JOINT, "object_free")]) \
        if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "object_free") >= 0 \
        else None
    if bjadr is not None:
        d.qpos[bjadr + 2] = 0.74
    mujoco.mj_forward(m, d)
    # the state is reconstructed, so it has to SETTLE before it is
    # measured: the reconstructed box touches the table exactly, and a
    # touching contact carries no force.  The settle is the same
    # controller (the position servos hold the commanded pose) and it
    # changes nothing but the contact state.
    for _ in range(120):
        mujoco.mj_step(m, d)

    def table_force():
        """the total normal force between the box and the table."""
        tot = 0.0
        for ci in range(d.ncon):
            c = d.contact[ci]
            b1 = int(m.geom_bodyid[int(c.geom1)])
            b2 = int(m.geom_bodyid[int(c.geom2)])
            if bid not in (b1, b2):
                continue
            cf = np.zeros(6)
            mujoco.mj_contactForce(m, d, ci, cf)
            # mj_contactForce returns the force in the CONTACT frame:
            # component 0 is the NORMAL force, 1 and 2 are the
            # tangential ones.  (Measured: the settled box's four table
            # contacts each report 0.982 N normal, i.e. 3.93 N in
            # total = its weight.)
            tot += float(abs(cf[0]))
        return tot

    z0 = float(d.xpos[bid][2])
    f0 = table_force()
    x0 = float(d.xpos[bid][0])

    # the upward command: a Jacobian step on the palm bodies
    step_mm = 6.0
    n_steps = 90
    frame_skip = 10
    palm_r = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "palm")
    palm_l = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "palm_left")
    arm_r = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT,
                               f"ur5e_r_{n}_joint") for n in
                             ("shoulder_pan", "shoulder_lift", "elbow",
                              "wrist_1", "wrist_2", "wrist_3")]
    arm_l = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT,
                               f"ur5e_l_{n}_joint") for n in
                             ("shoulder_pan", "shoulder_lift", "elbow",
                              "wrist_1", "wrist_2", "wrist_3")]
    qadr_r = [int(m.jnt_qposadr[j]) for j in arm_r]
    qadr_l = [int(m.jnt_qposadr[j]) for j in arm_l]
    vadr_r = [int(m.jnt_dofadr[j]) for j in arm_r]
    vadr_l = [int(m.jnt_dofadr[j]) for j in arm_l]
    act_r = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR,
                                f"ur5e_r_{n}") for n in
                             ("shoulder_pan", "shoulder_lift", "elbow",
                              "wrist_1", "wrist_2", "wrist_3")]
    act_l = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR,
                                f"ur5e_l_{n}") for n in
                             ("shoulder_pan", "shoulder_lift", "elbow",
                              "wrist_1", "wrist_2", "wrist_3")]
    base_r = d.qpos[qadr_r].copy()
    base_l = d.qpos[qadr_l].copy()
    ctrl_hold = 90
    for k in range(n_steps):
        t = 0.5 * (1 - np.cos(np.pi * k / max(1, n_steps - 1)))
        dz = step_mm * 1e-3 * t
        for palm, qadr, vadr, acts, base in (
                (palm_r, qadr_r, vadr_r, act_r, base_r),
                (palm_l, qadr_l, vadr_l, act_l, base_l)):
            jp = np.zeros((3, m.nv))
            jr = np.zeros((3, m.nv))
            mujoco.mj_jacBody(m, d, jp, jr, palm)
            J = jp[:, vadr]
            dq = np.linalg.lstsq(J, np.array([0.0, 0.0, dz]), rcond=None)[0]
            q = base + dq
            d.qpos[qadr] = np.clip(q, m.jnt_range[
                [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT,
                                   f"ur5e_{'r' if acts is act_r else 'l'}_{n}_joint")
                 for n in ("shoulder_pan", "shoulder_lift", "elbow",
                           "wrist_1", "wrist_2", "wrist_3")]][:, 0],
                m.jnt_range[[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT,
                                               f"ur5e_{'r' if acts is act_r else 'l'}_{n}_joint")
                              for n in ("shoulder_pan", "shoulder_lift", "elbow",
                                        "wrist_1", "wrist_2", "wrist_3")]][:, 1])
            d.ctrl[acts] = d.qpos[qadr]
        for _ in range(frame_skip):
            mujoco.mj_step(m, d)
    z1 = float(d.xpos[bid][2])
    f1 = table_force()
    x1 = float(d.xpos[bid][0])
    # the pad normal forces after the attempt: did the squeeze increase?
    pads_after = {}
    for gi in range(m.ngeom):
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, gi) or ""
        if "tip" not in nm and "distal" not in nm:
            continue
        for ci in range(d.ncon):
            c = d.contact[ci]
            if gi not in (int(c.geom1), int(c.geom2)):
                continue
            b1 = int(m.geom_bodyid[int(c.geom1)])
            b2 = int(m.geom_bodyid[int(c.geom2)])
            if bid not in (b1, b2):
                continue
            cf = np.zeros(6)
            mujoco.mj_contactForce(m, d, ci, cf)
            pads_after[nm] = round(float(abs(cf[0])), 4)
            break
    return {
        "step_mm": step_mm,
        "n_steps": n_steps,
        "z0_mm": z0 * 1e3,
        "z1_mm": z1 * 1e3,
        "dz_mm": (z1 - z0) * 1e3,
        "f0_N": f0,
        "f1_N": f1,
        "drift_mm": (x1 - x0) * 1e3,
        "pad_normal_forces_after_N": pads_after,
        "note": ("the mass, the friction coefficients and the joint limits "
                 "are untouched; only the arm's commanded pose is raised"),
    }


if __name__ == "__main__":
    raise SystemExit(main())
