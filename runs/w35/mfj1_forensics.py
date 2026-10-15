"""mf_j1 joint-limit excess: physical forensics + a derived measurement tolerance.

Evidence only -- no gate threshold is changed here.

The MJCF gives every finger joint ``limited="true"`` with ``stiffness="0"``: the
range is an assertion the smoke5 gate checks, not a mechanical stop the solver
enforces.  So a realised qpos outside jnt_range can only be (a) a command the
gate let through, or (b) the position servo's elastic windup under contact load.
For a MuJoCo position actuator the two are separable in closed form:

    actuator_force_dof = kp * (ctrl - qpos) - kv * qvel
                       = kp * e            (finger servos are gain-only, kv=0)

so the windup e is exactly tau_contact / kp, and the worst windup the actuator
can be talked into is |forcerange| / kp -- a per-joint number read straight off
the MJCF, with no free parameter.

This script measures, on the committed MJCF and on the proximal-stiffened variant:
  * mfj1's realised qpos, its command, the servo error e, the implied torque and
    the contact-side constraint torque on the same dof, per physics substep
  * how many control frames the excess lasts and what the contact force is
  * the per-joint windup cap |forcerange| / kp for all 16 finger joints

Run:  runs/.venv-retarget/bin/python runs/w35/mfj1_forensics.py
"""
from __future__ import annotations

import json
import pathlib
import pickle
import sys
import tempfile

import numpy as np

REPO = pathlib.Path("/Users/guohongbin/v2d-mac")
sys.path.insert(0, str(REPO))

import mujoco  # noqa: E402

from retarget import retarget_smoke as rs  # noqa: E402

sys.path.insert(0, str(REPO / "runs/w35"))
from prox_probe import FINGERS, load_variant, variant_text  # noqa: E402

CACHE = REPO / "runs/w35/prox_probe_anchor.pkl"
PROX = [f"{f}_proximal_collision" for f in FINGERS]
TARGET = "mfj1"


class DofRecorder:
    """Records per-substep dof state for the 16 finger joints."""

    # qpos is indexed by jnt_qposadr, the generalised-force/velocity vectors by
    # jnt_dofadr.  They differ by one here (the object's freejoint owns 7 qpos but
    # 6 dofs), so the two index sets are kept apart.
    DOF_FIELDS = ("qvel", "qfrc_actuator", "qfrc_constraint", "qfrc_passive",
                  "qfrc_bias")

    def __init__(self):
        self.rows = []
        self.ctrl = []
        self.contact = []
        self.all_contacts = []

    def reset(self):
        self.rows = []
        self.ctrl = []
        self.contact = []
        self.all_contacts = []

    def capture(self, model, data, box_gid):
        row = {"qpos": np.asarray(data.qpos[self.qposadr], dtype=float).copy()}
        for f in self.DOF_FIELDS:
            row[f] = np.asarray(getattr(data, f)[self.dofs], dtype=float).copy()
        # actuator_force is indexed by actuator, not by dof
        row["actuator_force"] = np.asarray(data.actuator_force[self.acts],
                                           dtype=float).copy()
        self.rows.append(row)
        self.ctrl.append(np.asarray(data.ctrl[self.acts], dtype=float).copy())
        box, allc = [], []
        for i in range(data.ncon):
            c = data.contact[i]
            f = np.zeros(6)
            mujoco.mj_contactForce(model, data, i, f)
            n1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(c.geom1))
            n2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(c.geom2))
            allc.append((n1, n2, float(f[0]), -float(c.dist)))
            if box_gid in (int(c.geom1), int(c.geom2)):
                other = n2 if int(c.geom1) == box_gid else n1
                box.append((other, float(f[0]), -float(c.dist)))
        self.contact.append(box)
        self.all_contacts.append(allc)


_ORIG_STEP = mujoco.mj_step
REC = DofRecorder()


def main():
    with CACHE.open("rb") as fh:
        payload = pickle.load(fh)
    qpos, anchor = payload["qpos"], payload["anchor"]

    ref_model = rs.load_scene()[1]
    jids = [rs._joint_id(ref_model, n) for n in rs.MJCF_FINGER_JOINTS]
    REC.dofs = np.array([ref_model.jnt_dofadr[j] for j in jids])
    REC.qposadr = np.array([ref_model.jnt_qposadr[j] for j in jids])
    acts = {}
    for a in range(ref_model.nu):
        acts[mujoco.mj_id2name(ref_model, mujoco.mjtObj.mjOBJ_ACTUATOR, a)] = a
    REC.acts = np.array([acts[f"{p}a{k}"] for p in FINGERS for k in range(4)])

    out = {"probe": "w35e_mfj1_forensics", "joint": TARGET,
           "method": __doc__, "variants": {}}

    for vname, geoms in (("V0_committed", []), ("V1_proximal", PROX)):
        model, data = load_variant(variant_text(geoms))
        box_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "object_geom")
        REC.reset()

        def hook(m, d):
            _ORIG_STEP(m, d)
            REC.capture(m, d, box_gid)

        saved = rs.load_scene
        rs.load_scene = lambda: (mujoco, model, data)
        mujoco.mj_step = hook
        try:
            rep = rs.replay(qpos, anchor)
        finally:
            mujoco.mj_step = _ORIG_STEP
            rs.load_scene = saved

        frame_skip = int(round(1.0 / (rs.CTRL_HZ * model.opt.timestep)))
        n_main = rep["n_frames"] * frame_skip
        sub = slice(len(REC.rows) - n_main, len(REC.rows))
        rows = REC.rows[sub]
        ctrls = REC.ctrl[sub]
        cons = REC.contact[sub]
        allc = REC.all_contacts[sub]
        fi = rs.MJCF_FINGER_JOINTS.index(TARGET)
        qadr = int(ref_model.jnt_qposadr[jids[fi]])
        dofadr = int(ref_model.jnt_dofadr[jids[fi]])
        act_id = acts[TARGET.replace("j", "a")]
        kp = float(model.actuator_gainprm[act_id, 0])
        kv = float(model.actuator_biasprm[act_id, 2])
        lo, hi = (float(v) for v in model.jnt_range[jids[fi]])
        jlim = float(model.jnt_stiffness[jids[fi]])

        q = np.asarray(rep["qpos_history"])[:, qadr]
        excess = np.maximum(0.0, q - hi)
        bad = np.where(excess > 0)[0]
        frames = sorted(set(int(f) for f in bad))
        worst_frame = int(np.argmax(excess)) if excess.size else None

        e = np.abs(np.asarray([r["qpos"][fi] for r in rows])
                   - np.asarray([c[fi] for c in ctrls]))
        qf = np.asarray([r["qpos"][fi] for r in rows])
        af = np.asarray([r["actuator_force"][fi] for r in rows])
        qc = np.asarray([r["qfrc_constraint"][fi] for r in rows])
        qp_ = np.asarray([r["qfrc_passive"][fi] for r in rows])
        qb = np.asarray([r["qfrc_bias"][fi] for r in rows])
        qv = np.asarray([r["qvel"][fi] for r in rows])

        def at(frame):
            sl = slice(frame * frame_skip, (frame + 1) * frame_skip)
            mf_contacts = {}
            for entry in cons[sl]:
                for name, fn, pen in entry:
                    if name.startswith("mf_"):
                        mf_contacts[name] = max(mf_contacts.get(name, 0.0), fn)
            mf_any = []
            for entry in allc[sl]:
                for n1, n2, fn, pen in entry:
                    if str(n1).startswith("mf_") or str(n2).startswith("mf_"):
                        mf_any.append({"pair": f"{n1}|{n2}", "force_N": fn,
                                       "pen_m": pen})
            mf_any.sort(key=lambda r: -r["force_N"])
            return {
                "frame": frame,
                "ctrl_rad": float(ctrls[frame * frame_skip][fi]),
                "qpos_rad": float(q[frame]),
                "excess_rad": float(excess[frame]),
                "servo_error_rad": float(qf[sl][-1] - ctrls[frame * frame_skip][fi]),
                "actuator_force_Nm": float(af[sl][-1]),
                "kp_times_error_Nm": float(kp * (qf[sl][-1] - ctrls[frame * frame_skip][fi])),
                "qfrc_constraint_Nm": float(qc[sl][-1]),
                "qfrc_passive_Nm": float(qp_[sl][-1]),
                "qfrc_bias_Nm": float(qb[sl][-1]),
                "qvel_rad_s": float(qv[sl][-1]),
                "mf_box_contacts_peak_normal_N": mf_contacts,
                "mf_chain_contacts": mf_any[:6],
                "max_contact_force_N": max(
                    (r["force_N"] for entry in allc[sl] for r in
                     [{"force_N": e[2]} for e in entry]), default=0.0),
            }

        per_joint = {}
        for k, name in enumerate(rs.MJCF_FINGER_JOINTS):
            j = jids[k]
            a = acts[name.replace("j", "a")]
            kp_k = float(model.actuator_gainprm[a, 0])
            fr_k = float(model.actuator_forcerange[a, 1])
            qq = np.asarray(rep["qpos_history"])[:, int(ref_model.jnt_qposadr[j])]
            ex = float(max(0.0, qq.max() - model.jnt_range[j][1],
                           model.jnt_range[j][0] - qq.min()))
            ek = np.asarray([r["qpos"][k] for r in rows])
            ck = np.asarray([c[k] for c in ctrls])
            per_joint[name] = {
                "kp_Nm_per_rad": kp_k,
                "kv_Nm_s_per_rad": float(model.actuator_biasprm[a, 2]),
                "forcerange_Nm": fr_k,
                "jnt_range_rad": [float(model.jnt_range[j][0]),
                                  float(model.jnt_range[j][1])],
                "jnt_stiffness": float(model.jnt_stiffness[j]),
                "windup_cap_rad": fr_k / kp_k,
                "measured_excess_rad": ex,
                "measured_max_servo_error_rad": float(np.abs(ek - ck).max()),
            }

        out["variants"][vname] = {
            "pairs_added": geoms,
            "target": {
                "jnt_range_rad": [lo, hi],
                "jnt_limited": bool(model.jnt_limited[jids[fi]]),
                "jnt_stiffness": jlim,
                "kp_Nm_per_rad": kp,
                "kv_Nm_s_per_rad": kv,
                "forcerange_Nm": float(model.actuator_forcerange[act_id, 1]),
                "ctrlrange_rad": [float(v) for v in model.actuator_ctrlrange[act_id]],
                "max_excess_rad": float(excess.max()),
                "violating_frames": frames,
                "violating_frame_count": len(frames),
            },
            "mf_contact_force_profile": {
                "frames_with_any_mf_contact": sorted({
                    fi for fi in range(rep["n_frames"])
                    if any(str(e[0]).startswith("mf_") or str(e[1]).startswith("mf_")
                           for entry in allc[fi * frame_skip:(fi + 1) * frame_skip]
                           for e in entry)}),
                "peak_mf_chain_normal_N": max(
                    (e[2] for entry in allc for e in entry
                     if str(e[0]).startswith("mf_") or str(e[1]).startswith("mf_")),
                    default=0.0),
                "peak_mf_box_normal_N": max(
                    (f for entry in cons for (nm, f, _) in entry
                     if nm.startswith("mf_")), default=0.0),
            },
            "worst_frame_detail": at(worst_frame) if worst_frame is not None else None,
            "all_violating_frames": [at(f) for f in frames],
            "per_joint_windup_cap": per_joint,
        }
        v = out["variants"][vname]
        print(f"{vname}: max excess {v['target']['max_excess_rad'] * 1e3:.3f} mrad "
              f"on {len(frames)} frame(s); servo cap "
              f"{v['target']['forcerange_Nm'] / kp * 1e3:.1f} mrad")
        if v["worst_frame_detail"]:
            w = v["worst_frame_detail"]
            print(f"   worst f{w['frame']}: ctrl {w['ctrl_rad']:.6f} "
                  f"qpos {w['qpos_rad']:.6f} excess {w['excess_rad'] * 1e3:.3f} mrad "
                  f"| servo error {w['servo_error_rad'] * 1e3:+.3f} mrad "
                  f"tau_act {w['actuator_force_Nm'] * 1e3:+.4f} mN.m "
                  f"qfrc_constraint {w['qfrc_constraint_Nm'] * 1e3:+.4f} mN.m "
                  f"contacts {w['mf_box_contacts_peak_normal_N']}")

    dest = REPO / "runs/w35/w35e_mfj1_forensics.json"
    dest.write_text(json.dumps(out, indent=2, default=float) + "\n")
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
