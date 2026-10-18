"""Is the recheck's "clean hook" actually a hook, or a press from ABOVE?

`lip_kin_recheck.py` reports clean_hook=True for all three cells with the nearest
lip point at rel_z = 0.04 - which is the lip's TOP face, while the hook face
(the underside a finger must push up on) is at rel_z = bz - lip_h (0.03 for the
10 mm lip). The objective's z-band tolerance is max(lip_h, WALL_BAND_M), i.e. as
wide as the lip itself, so a pad resting ON TOP of the lip scores the same as a
pad hooked UNDER it.

Those are opposite mechanisms:
  press from above  -> pushes the box DOWN. Useless for a lift.
  hook from below   -> pushes the box UP. The clean bearing path R8 was after.

This audits each reported pose on the physics: where is the pad relative to the
lip's underside plane, and what is the contact normal's sign. Only a pad strictly
BELOW the underside counts as a hook.
"""
from __future__ import annotations

import json
import pathlib
import sys

import mujoco
import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from grounding.assets import build_allegro_floating as baf  # noqa: E402

RECHECK = pathlib.Path(__file__).resolve().parent / "lip_kin_recheck.json"
ASSETS = REPO / "grounding/assets"
OUT = pathlib.Path(__file__).resolve().parent / "hook_direction_audit.json"


def audit(overhang: float, lip_h: float, pose: list[float]) -> dict:
    tmp = ASSETS / f"_audit_{overhang * 1e3:.0f}_{lip_h * 1e3:.0f}.xml"
    baf.build(overhang, lip_h, tmp)
    stl = (ASSETS / "../../runs/upstream/menagerie/wonik_allegro/assets").resolve()
    assets = {p.name: p.read_bytes() for p in stl.glob("*.stl")}
    text = tmp.read_text()
    tmp.unlink(missing_ok=True)
    m = mujoco.MjModel.from_xml_string(text, assets)
    d = mujoco.MjData(m)

    names = ("wrist_x", "wrist_y", "wrist_z", "wrist_rx", "wrist_ry", "wrist_rz")
    wq = [int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)])
          for n in names]
    fq = [int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)])
          for n in baf.FINGER_JOINTS]
    obj = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "object")
    d.qpos[wq] = pose[:6]
    d.qpos[fq] = pose[6:]
    d.qpos[int(m.jnt_qposadr[mujoco.mj_name2id(
        m, mujoco.mjtObj.mjOBJ_JOINT, "object_free")]) + 2] = \
        baf.TABLE_TOP_Z_M + baf.PEDESTAL_H_M + baf.BOX_HALF_M[2]
    mujoco.mj_kinematics(m, d)
    mujoco.mj_forward(m, d)

    bz = baf.BOX_HALF_M[2]
    hook_z_world = baf.TABLE_TOP_Z_M + baf.PEDESTAL_H_M + bz - lip_h
    bx = baf.BOX_HALF_M[0]
    out = {
        "overhang_mm": overhang * 1e3, "lip_h_mm": lip_h * 1e3,
        "box_world_z": float(d.xpos[obj][2]),
        "hook_face_world_z": hook_z_world,
        "pads": [],
    }
    ft = np.zeros(6)
    for pref in ("ff", "mf", "rf", "th"):
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{pref}_tip")
        for g in range(m.ngeom):
            if int(m.geom_bodyid[g]) != bid:
                continue
            if "visual" in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""):
                continue
            pz = float(d.geom_xpos[g][2])
            px = float(d.geom_xpos[g][0])
            # nearest point on each lip, with the SIGNED distance from the return value
            rec = {"pad": f"{pref}_tip", "pad_world_z": pz, "pad_world_x": px,
                   "below_underside": pz < hook_z_world}
            for lname in baf.BOX_LIP_GEOM_NAMES:
                lg = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, lname)
                dist = float(mujoco.mj_geomDistance(m, d, g, lg, 0.30, ft))
                rec[f"dist_to_{lname}"] = dist
                rec[f"nearest_on_{lname}_z"] = float(ft[5])
            out["pads"].append(rec)
    return out


def main() -> None:
    data = json.loads(RECHECK.read_text())
    rows = []
    for c in data["cells"]:
        r = audit(c["overhang_mm"] * 1e-3, c["lip_h_mm"] * 1e-3, c["pose"])
        r["recheck_clean_hook"] = c["clean_hook"]
        r["recheck_best_gap_m"] = c["best_gap_m"]
        rows.append(r)
        print(f"o={r['overhang_mm']:4.1f} h={r['lip_h_mm']:4.1f} "
              f"hook_face_z={r['hook_face_world_z']:.4f} "
              f"(recheck claimed clean={r['recheck_clean_hook']})")
        for p in r["pads"]:
            touching = {k: round(v, 6) for k, v in p.items()
                        if k.startswith("dist_to_") and abs(v) < 0.05}
            if touching:
                print(f"    {p['pad']:8s} pad_z={p['pad_world_z']:.4f} "
                      f"below_underside={p['below_underside']!s:5s} {touching}")

    any_hook = False
    for r in rows:
        for p in r["pads"]:
            if p["below_underside"] and any(
                    abs(v) < 1e-3 for k, v in p.items() if k.startswith("dist_to_")):
                any_hook = True
    print()
    print(f"VERDICT: any pad actually BELOW the lip underside and touching? {any_hook}")
    OUT.write_text(json.dumps({"rows": rows, "any_true_hook": any_hook}, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
