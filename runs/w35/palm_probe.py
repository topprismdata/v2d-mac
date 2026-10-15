"""Palm<->box contact stiffening probe (Wave 3.5 auxiliary evidence).

Measures, on the COMMITTED 60 mm scene (read-only) and on a variant with an
explicit stiff `<pair>` injected, at R3's methodology:

  P1  palm->box normal-force ramp: settle the scene, drive the wrist_z servo
      below the palm-on-box floor in small steps, and at the first control
      point where the summed palm_collision<->object_geom normal force crosses
      10 N, report the deepest palm->box penetration (contact -dist).
  P2  pedestal graze regression guard: with the box's collision disabled, drive
      wrist_z to its ctrlrange floor (-0.163) for 400 steps and report the peak
      palm->pedestal contact force (R3 measured 3.2 kN soft, 77 kN stiffened).

    python runs/w35/palm_probe.py
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

REPO = pathlib.Path("/Users/guohongbin/v2d-mac")
sys.path.insert(0, str(REPO))

import mujoco  # noqa: E402

mj = mujoco
BASE_XML = REPO / "grounding/assets/allegro_floating.xml"
PAIR = ('<pair geom1="palm_collision" geom2="object_geom" '
        'solref="0.004 1" solimp="0.95 0.99 0.001 0.5 2" condim="4"/>')


def load(xml_text: pathlib.Path | str):
    """meshdir is relative to the XML's own directory, so string loads resolve
    the wrong path - mirror the generator's _verify_out: temp file beside the
    original, load, delete."""
    if isinstance(xml_text, pathlib.Path):
        xml_text = xml_text.read_text()
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False,
                                     dir=str(BASE_XML.parent)) as fh:
        fh.write(xml_text)
        tmp = pathlib.Path(fh.name)
    try:
        model = mj.MjModel.from_xml_path(str(tmp))
    finally:
        tmp.unlink(missing_ok=True)
    return model, mj.MjData(model)


def geom(model, name):
    return mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, name)


def settle(model, data, n=900):
    mj.mj_resetData(model, data)
    for _ in range(n):
        mj.mj_step(model, data)


def palm_box_state(model, data):
    pg, og = geom(model, "palm_collision"), geom(model, "object_geom")
    force = 0.0
    pen = 0.0
    for i in range(data.ncon):
        c = data.contact[i]
        if pg in (int(c.geom1), int(c.geom2)) and og in (int(c.geom1), int(c.geom2)):
            f = np.zeros(6)
            mj.mj_contactForce(model, data, i, f)
            force += float(np.linalg.norm(f[:3]))
            pen = max(pen, -float(c.dist))
    return force, pen


def probe_p1(model, data, step=0.002, max_press=0.09):
    """Ramp the wrist down until palm->box force crosses 10 N; return the
    (force, penetration) at the crossing and the deepest penetration seen."""
    settle(model, data)
    zadr = model.jnt_qposadr[mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, "wrist_z")]
    crossed = None
    deepest = 0.0
    press = 0.0
    while press <= max_press:
        data.ctrl[2] = -0.02 - press
        for _ in range(60):
            mj.mj_step(model, data)
        force, pen = palm_box_state(model, data)
        deepest = max(deepest, pen)
        if force >= 10.0:
            crossed = (force, pen, press)
            break
        press += step
    return {"crossed_at_force_N": None if crossed is None else crossed[0],
            "pen_at_10N_m": None if crossed is None else crossed[1],
            "ctrl_overshoot_m_at_10N": None if crossed is None else crossed[2],
            "deepest_pen_m": deepest,
            "reached_10N": crossed is not None}


def probe_p2(model, data, n=400):
    """R3's regression guard: palm driven to the wrist floor onto the bare
    pedestal (box collision disabled at model level, then restored)."""
    saved = (model.geom_contype.copy(), model.geom_conaffinity.copy())
    og = geom(model, "object_geom")
    model.geom_contype[og] = 0
    model.geom_conaffinity[og] = 0
    pg = geom(model, "palm_collision")
    peak = 0.0
    peak_pen = 0.0
    try:
        settle(model, data)
        data.ctrl[2] = -0.163
        for _ in range(n):
            mj.mj_step(model, data)
            for i in range(data.ncon):
                c = data.contact[i]
                if pg in (int(c.geom1), int(c.geom2)):
                    f = np.zeros(6)
                    mj.mj_contactForce(model, data, i, f)
                    peak = max(peak, float(np.linalg.norm(f[:3])))
                    peak_pen = max(peak_pen, -float(c.dist))
    finally:
        model.geom_contype[:], model.geom_conaffinity[:] = saved
    return {"palm_pedestal_peak_force_N": peak,
            "palm_pedestal_peak_pen_m": peak_pen}


def main() -> None:
    # Since Wave 3.5-B the committed XML itself carries the pair: the soft scene is
    # the committed text with the pair line stripped, the stiff scene is verbatim.
    base = BASE_XML.read_text()
    out = {}
    for label, text in (
            ("soft_committed_scene", base.replace("    " + PAIR + "\n", "", 1)),
            ("stiff_pair_injected", base)):
        model, data = load(text)
        assert (model.npair > 0) == ("stiff" in label), "pair injection failed"
        out[label] = {"P1_palm_box_10N": probe_p1(model, data),
                      "P2_pedestal_graze": probe_p2(model, data)}
        print(label, json.dumps(out[label], indent=2), flush=True)
    dest = REPO / "runs/w35/palm_probe.json"
    dest.write_text(json.dumps(out, indent=2))
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
