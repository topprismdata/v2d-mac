"""Self-check: load the generated MJCF and dump its statistics as JSON.

Run:  python grounding/assets/model_stats.py

Prints machine-readable JSON on stdout so a test (or a human) can diff the scene
against the frozen contract without eyeballing the XML:

  * model shape (nq / nv / nu / nbody / ngeom / njnt)
  * the geometry that must not drift: box extents and mass, pedestal height
  * the six wrist axes' range and ctrlrange, so R2's travel change is visible
  * every finger actuator's kp and forcerange, so R1's derived table is visible
  * the contact stiffness actually in force on the object and on a fingertip, so
    R3 is visible
  * the derived penetration tolerance and the resting sink it came from (R5)
  * the moment arm table R1 was computed from, recomputed from the model on the
    spot rather than read back out of the XML, so the two cannot silently diverge
"""

from __future__ import annotations

import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import mujoco  # noqa: E402

from grounding.assets import build_allegro_floating as builder  # noqa: E402

MJCF = pathlib.Path(__file__).resolve().parent / "allegro_floating.xml"


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(MJCF))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    obj = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "object_geom")
    obj_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    ped = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "pedestal")

    wrists = {}
    for jname, aname in builder.WRIST_ACTUATOR_OF.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aname)
        lo, hi = model.jnt_range[jid]
        wrists[jname] = {
            "actuator": aname,
            "range": [float(lo), float(hi)],
            "ctrlrange": [float(v) for v in model.actuator_ctrlrange[aid]],
            "kp": float(model.actuator_gainprm[aid][0]),
            "travel_added": builder.WRIST_TRAVEL_ADD_M.get(jname, (0.0, 0.0, 0.0))[2],
        }

    arms = builder.moment_arm_table()
    gravity = builder.gravity_torque_table()
    inertia = builder.effective_inertia_table()
    fingers = {}
    for name in builder.FINGER_JOINTS:
        aid = builder.finger_actuator_id(model, name)
        tau = float(model.actuator_forcerange[aid][1])
        fingers[name] = {
            "moment_arm_m": arms[name],
            "forcerange_Nm": tau,
            "implied_tip_force_N": tau / arms[name],
            "kp": float(model.actuator_gainprm[aid][0]),
            "gravity_torque_Nm": gravity[name],
            "effective_inertia": inertia[name],
            "servo_bandwidth_rad_s": (model.actuator_gainprm[aid][0]
                                      / inertia[name]) ** 0.5,
        }

    tip_ff = builder._tip_geom_ids(model)["ff"]
    sink = builder.measure_resting_sink()
    payload = {
        "mjcf": str(MJCF.relative_to(REPO)),
        "model_shape": {
            "nq": int(model.nq), "nv": int(model.nv), "nu": int(model.nu),
            "nbody": int(model.nbody), "ngeom": int(model.ngeom),
            "njnt": int(model.njnt),
            "timestep": float(model.opt.timestep),
            "integrator": int(model.opt.integrator),
        },
        "geometry": {
            "box_half_extent_m": [float(v) for v in model.geom_size[obj]],
            "box_mass_kg": float(model.body_mass[obj_body]),
            "pedestal_half_height_m": float(model.geom_size[ped][2]),
            "pedestal_top_z_m": float(model.geom_pos[ped][2] + model.geom_size[ped][2]),
        },
        "wrists": wrists,
        "fingers": fingers,
        "contact": {
            "object_solref": [float(v) for v in model.geom_solref[obj]],
            "object_solimp": [float(v) for v in model.geom_solimp[obj]],
            "ff_tip_solref": [float(v) for v in model.geom_solref[tip_ff]],
            "ff_tip_solimp": [float(v) for v in model.geom_solimp[tip_ff]],
        },
        "penetration": {
            "resting_sink_m": sink,
            "tolerance_k": builder.PENETRATION_TOLERANCE_K,
            "tolerance_m": builder.PENETRATION_TOLERANCE_K * sink,
        },
        "naming": {
            "hand_geoms": int(sum(
                1 for g in range(model.ngeom)
                if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                                     int(model.geom_bodyid[g])) or "")
                in ("palm",) or
                (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                                   int(model.geom_bodyid[g])) or "")
                .startswith(builder.FINGER_PREFIXES))),
            "hand_geoms_unnamed": int(sum(
                1 for g in range(model.ngeom)
                if not mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g)
                and ((mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                                        int(model.geom_bodyid[g])) or "") == "palm"
                     or (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                                           int(model.geom_bodyid[g])) or "")
                     .startswith(builder.FINGER_PREFIXES)))),
        },
    }
    json.dump(payload, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
