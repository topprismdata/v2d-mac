"""r118: build a SURFACE-seated terminal candidate and replay it.

WHY A SEPARATE TOOL.  The delivered terminal's search optimises the pad
CENTRE against the end-face plane.  The pad is a 12 x 12 x 3 mm BOX, so
a centre on the plane means the pad is 2.1 mm (ff) / 5.8 mm (mf) inside
the box -- which is why the command state's gate1 sees two pads in the
rectangle while the EXECUTED state has only one in contact (the ff).

This tool builds a candidate whose pad CONTACT SURFACE sits on the face,
using the pad's support function along the face normal

    support = sum_i |axis_i . n| * half_extent_i

(the exact formula of the pipeline's own _geom_support_along), and then
replays it in the full pipeline to measure the executed contact.

The measured result (r118): the surface-seated candidate executes with a
real contact (the ff's executed surface gap -0.009 .. -0.035 mm) and
gate1's v3 (the surface criterion) goes from n_eligible = 0 to 1.  The
middle pad still ends up 9.3 mm away, because the two pads' supports
differ (2.122 vs 5.776 mm) and a per-finger IK that drives the GAP does
not change the POSTURE.

    runs/.venv-retarget/bin/python tools/w6b/surface_seated_build.py
"""
from __future__ import annotations

import json
import os
import pathlib
import shutil
import subprocess
import sys

import mujoco
import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
SCENE = REPO / "grounding/assets/allegro_floating_ur5e_side_r37.xml"
DELIVERED = REPO / "runs/w6b/palm_pose_refine.json"
WORK = REPO / "runs/w6b/surface_seated_refs"
OUT = REPO / "runs/w6b/surface_seated_build.json"
REPORT = REPO / "runs/w6b/report.json"
PIPE = REPO / "runs/.venv-retarget/bin/python"
SCRIPT = REPO / "retarget/retarget_dual.py"

ARM = ("shoulder_pan", "shoulder_lift", "elbow",
       "wrist_1", "wrist_2", "wrist_3")
FINGERS = ("ff", "mf", "rf")
CURLS = [float(v) for v in os.environ.get(
    "W6B_S118_CURLS", "1.00,1.05,1.10,1.15").split(",")]
TARGET_GAP_MM = float(os.environ.get("W6B_S118_GAP_MM", "-0.05"))

BASE = {
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
    "W6B_R110_CURL_SCALE": "1.0",
    "W6B_R109_SQUEEZE_MM": "0",
}


def jid(m, n):
    i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
    assert i >= 0, n
    return i


def gid(m, n):
    i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)
    assert i >= 0, n
    return i


class Side:
    def __init__(self, m, d, side: str):
        self.m = m
        self.d = d
        self.side = side
        sfx = "" if side == "right" else "_left"
        prefix = "ur5e_r" if side == "right" else "ur5e_l"
        self.face_sign = 1.0 if side == "right" else -1.0
        self.n_in = np.array([-self.face_sign, 0.0, 0.0])
        self.AJ = [jid(m, f"{prefix}_{n}_joint") for n in ARM]
        self.AQ = [int(m.jnt_qposadr[j]) for j in self.AJ]
        self.AD = [int(m.jnt_dofadr[j]) for j in self.AJ]
        self.PALM = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY,
                                      "palm" + sfx)
        self.OBJ = gid(m, "object_geom")
        self.PAD = {f: gid(m, f"{f}_tip_collision" if not sfx
                           else f"{f}_tip_left_collision")
                    for f in FINGERS}
        self.FQ = {f: [int(m.jnt_qposadr[jid(m, f"{f}j{k}{sfx}")])
                      for k in range(4)] for f in FINGERS}
        self.FQ["th"] = [int(m.jnt_qposadr[jid(m, f"thj{k}{sfx}")])
                         for k in range(4)]
        self.FD = {f: [int(m.jnt_dofadr[jid(m, f"{f}j{k}{sfx}")])
                      for k in range(4)] for f in FINGERS}
        self.FLO = {f: np.array([m.jnt_range[jid(m, f"{f}j{k}{sfx}"), 0]
                                 for k in range(4)]) for f in FINGERS}
        self.FHI = {f: np.array([m.jnt_range[jid(m, f"{f}j{k}{sfx}"), 1]
                                 for k in range(4)]) for f in FINGERS}
        self.obj = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "object")
        self.qa = int(m.jnt_qposadr[int(self.obj)])

    def reset(self):
        mujoco.mj_resetData(self.m, self.d)
        self.d.qpos[:] = self.m.qpos0
        self.d.qpos[self.qa:self.qa + 7] = [0, 0, 0.74, 1, 0, 0, 0]

    def place(self, q, fv, thumb):
        self.reset()
        self.d.qpos[self.AQ] = q
        for f in FINGERS:
            self.d.qpos[self.FQ[f]] = fv[f]
        self.d.qpos[self.FQ["th"]] = thumb
        mujoco.mj_forward(self.m, self.d)

    def surf(self, f):
        ft = np.zeros(6)
        dd = float(mujoco.mj_geomDistance(self.m, self.d, self.PAD[f],
                                          self.OBJ, 0.5, ft))
        v = np.asarray(ft[0:3]) - np.asarray(ft[3:6])
        nv = float(np.linalg.norm(v))
        n_hat = v / nv if nv > 1e-9 else self.n_in.copy()
        return dd, n_hat

    def support(self, f):
        R = np.asarray(self.d.geom_xmat[self.PAD[f]]).reshape(3, 3)
        he = np.asarray(self.m.geom_size[self.PAD[f]])
        return float(sum(abs(R[:, i] @ self.n_in) * he[i] for i in range(3)))

    def ik_finger(self, q, f, target, fv, thumb):
        x = np.asarray(fv[f], float).copy()
        for _ in range(140):
            self.place(q, fv, thumb)
            self.d.qpos[self.FQ[f]] = x
            mujoco.mj_forward(self.m, self.d)
            dd, n_hat = self.surf(f)
            if abs(dd - target) < 3e-5:
                break
            e = n_hat * (dd - target)
            jp = np.zeros((3, self.m.nv))
            mujoco.mj_jacGeom(self.m, self.d, jp, None, self.PAD[f])
            J = jp[:, self.FD[f]]
            try:
                dq = np.linalg.solve(J @ J.T + 1e-6 * np.eye(3), e)
            except np.linalg.LinAlgError:
                break
            x = np.clip(x + J.T @ dq, self.FLO[f], self.FHI[f])
        self.place(q, fv, thumb)
        self.d.qpos[self.FQ[f]] = x
        mujoco.mj_forward(self.m, self.d)
        dd, _ = self.surf(f)
        return x, abs(dd - target)


def build(sd: Side, base: dict, curl: float, target: float) -> dict:
    fv = {}
    for f in FINGERS:
        v = list(base["fingers"][f])
        for j in (1, 2, 3):
            v[j] *= curl
        fv[f] = v
    thumb = list(base["fingers"].get("th", [0.0] * 4))
    q = np.asarray(base["arm"], float)
    for _ in range(6):
        sd.place(q, fv, thumb)
        gap, _ = sd.surf("ff")
        # translate the palm ALONG the face normal (out of the face) so
        # the ff pad's SURFACE lands on the face; the orientation is
        # held.  The direction is the face's outward normal, which is
        # -x for the right hand and +x for the left (the two palms are
        # mirrored -- using a fixed +x broke the left side: the surface
        # gap read 500 mm, i.e. the IK never converged).
        _out = np.array([sd.face_sign, 0.0, 0.0])
        jp = np.zeros((3, sd.m.nv))
        jr = np.zeros((3, sd.m.nv))
        mujoco.mj_jacBody(sd.m, sd.d, jp, jr, sd.PALM)
        Jn = jp[:, sd.AD].T @ _out
        Jo = jr[:, sd.AD]
        A = np.vstack([Jn, Jo])
        b = np.concatenate([np.array([-(gap - target)]), np.zeros(3)])
        dq = np.linalg.lstsq(A, b, rcond=None)[0]
        q = np.clip(q + dq, sd.m.jnt_range[sd.AJ, 0],
                    sd.m.jnt_range[sd.AJ, 1])
        fv2 = {f: list(fv[f]) for f in FINGERS}
        res = {}
        for f in ("ff", "mf"):
            x_f, r_f = sd.ik_finger(q, f, target, fv2, thumb)
            fv2[f] = list(x_f)
            res[f] = r_f
        fv = fv2
        if max(res.values()) < 3e-3:
            break
    sd.place(q, fv, thumb)
    pp = np.array(sd.d.xpos[sd.PALM], dtype=float)
    Rq = np.zeros(9)
    mujoco.mju_quat2Mat(Rq, sd.d.xquat[sd.PALM])
    PR = np.array(Rq).reshape(3, 3)
    return {
        "palm_pos": [round(float(v), 6) for v in pp],
        "palm_R": [[round(float(v), 6) for v in r] for r in PR],
        "arm": [round(float(v), 6) for v in q],
        "fingers": {**{f: [round(float(v), 6) for v in fv[f]]
                      for f in FINGERS},
                   "th": [round(float(v), 6) for v in thumb]},
        "the static surface gaps mm": {
            f: round(sd.surf(f)[0] * 1e3, 4) for f in FINGERS},
        "the pad centres mm": {f: [round(float(v) * 1e3, 3)
                                   for v in sd.d.geom_xpos[sd.PAD[f]]]
                               for f in FINGERS},
    }


def replay(ref_path: pathlib.Path) -> dict:
    env = dict(os.environ)
    env.update(BASE)
    env["W6B_R63_REFERENCE"] = str(ref_path)
    r = subprocess.run([str(PIPE), str(SCRIPT)], cwd=REPO, env=env,
                       capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(r.stdout[-1200:] + r.stderr[-1200:] + "\n")
        return {}
    return json.loads(REPORT.read_text())


def main() -> int:
    m = mujoco.MjModel.from_xml_path(str(SCENE))
    d = mujoco.MjData(m)
    ref = json.loads(DELIVERED.read_text())

    def entry(side):
        e = ref["sides"][side]
        return e["best"][0] if isinstance(e.get("best"), list) else e["best"]

    keep = REPORT.with_suffix(".json.surface_build_keep")
    if REPORT.exists():
        shutil.copy2(REPORT, keep)
    rows = []
    try:
        for side in ("right", "left"):
            sd = Side(m, d, side)
            base = entry(side)
            other = entry("left" if side == "right" else "right")
            for curl in CURLS:
                cand = build(sd, base, curl, TARGET_GAP_MM * 1e-3)
                trial = {"scene": str(SCENE.relative_to(REPO)), "sides": {}}
                for s2 in ("right", "left"):
                    trial["sides"][s2] = {"best": [
                        cand if s2 == side else other]}
                WORK.mkdir(parents=True, exist_ok=True)
                p = WORK / f"{side}_curl{curl:.2f}.json"
                p.write_text(json.dumps(trial, indent=1))
                rep = replay(p)
                row = {"side": side, "the curl": curl,
                       "the static surface gaps mm":
                           cand["the static surface gaps mm"]}
                if rep:
                    g = rep["per_side"][side]["gate1_final_touch"]
                    ex = g["executed_state_evidence"]["gate1_v3_executed_state"]
                    row["the v1v2 rect mm"] = round(float(g["rect_dist_mm"]), 4)
                    row["the v1v2 pads in the rect"] = int(g["n_pads_in_rect"])
                    row["the v1v2 pass"] = bool(g["pass"])
                    row["the v3 eligible"] = int(ex["n_eligible"])
                    row["the v3 pass"] = bool(ex["pass"])
                    row["the normal force N"] = round(float(sum(
                        v.get("contact_normal_force_N", 0.0)
                        for v in ex["pads"].values()
                        if v.get("in_contact"))), 4)
                    row["the per-pad surface gap mm"] = {
                        k: round(float(v.get("surface_gap_mm", 0.0)), 4)
                        for k, v in ex["pads"].items()}
                    row["the red gates"] = [k for k, v
                                            in rep["verdict_checks"].items()
                                            if not v]
                    print(f"  {side} the curl {curl}: the static gaps "
                          f"{cand['the static surface gaps mm']} | the "
                          f"executed ff gap "
                          f"{row['the per-pad surface gap mm'].get('ff_tip_collision')}"
                          f" / the left variant | the v3 "
                          f"{row['the v3 eligible']} | the force "
                          f"{row['the normal force N']} N | the red "
                          f"{row['the red gates']}")
                p.unlink(missing_ok=True)
                rows.append(row)
    finally:
        if keep.exists():
            shutil.copy2(keep, REPORT)
            keep.unlink()

    OUT.write_text(json.dumps(rows, indent=2, default=float))
    print(f"\n  written: {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
