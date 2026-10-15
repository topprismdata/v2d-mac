"""Round p0(4e) evidence: the p0(4d) hook, A/B'd across the two scene revisions.

The re-collection search found no gate-clean reachable-class trajectory on the
current scene. This module says WHY, in a form that stays true if the physics is
revised again: it replays the EXACT action sequence round p0(4d) recorded, once
on the scene that recorded it (MJCF 4ee6f1d, sha256 dbba3d70...) and once on the
current scene (MJCF 1b0f13f, sha256 1ea0fdf5...), and reports the same
measurements both times.

Only the scene changes. Same actions, same env code, same thresholds - so any
difference in the verdict is attributable to the Wave 3.5F proximal stiffening
alone.

The secondary finding matters as much as the primary one. On BOTH scenes the
recorded hook spends only three steps above the 5 cm bar and reaches its peak
with the finger force at zero: the box is launched, not carried. It cleared A3b
on 4ee6f1d only because at the single apex step of that launch the box happened
to have no contact at all - a one-step coincidence, consistent with the
adjudication's own verdict that the p0(4d) passing set had measure ~zero. On
the current scene even that coincidence is gone. A3c's anchored-step gate was
already rejecting the trajectory's tail (14 mm in one step against a 10 mm cap).

    python grounding/hook_scene_ab.py
"""
from __future__ import annotations

import json
import pathlib
import re
import subprocess
import sys
import tempfile

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

RUN_DIR = REPO / "runs/p0/smoke4_mujoco_ppo"
DEMOS = RUN_DIR / "reachable_demo_set_f210f45.npz"
OUT = RUN_DIR / "hook_scene_ab.json"
OLD_COMMIT = "4ee6f1d"
NEW_COMMIT = "1b0f13f"
OLD_SHA = "dbba3d70d722d529f80d1935e6c22826ac5d9be964c9827c64f3062bc660b934"
NEW_SHA = "1ea0fdf5c87ce5a161efcda7c597b0fc6647c9fe30bcf2f5d0bcb52402d24b89"
LIFT_BAR_M = 0.05


def _materialise(xml_text: str) -> pathlib.Path:
    """Write an XML copy whose meshdir is absolute, then return its path.

    The 4ee6f1d scene carries a RELATIVE ``meshdir`` (``../../runs/upstream/...``)
    that only resolves from ``grounding/assets/``. Re-homing the file therefore
    breaks it. Rewriting meshdir to an absolute path in a throwaway COPY is the
    fix: the frozen asset in ``grounding/assets/`` is never touched.
    """
    src = REPO / "grounding/assets/allegro_floating.xml"
    abs_dir = str((src.parent / "../../runs/upstream/menagerie/wonik_allegro/assets")
                  .resolve())
    fixed = re.sub(r'meshdir="[^"]*"', f'meshdir="{abs_dir}"', xml_text, count=1)
    assert f'meshdir="{abs_dir}"' in fixed, "meshdir rewrite did not apply"
    path = pathlib.Path(tempfile.mkdtemp(prefix="scene_ab_")) / "scene.xml"
    path.write_text(fixed)
    return path


def _old_scene_path() -> pathlib.Path:
    xml = subprocess.run(["git", "show", f"{OLD_COMMIT}:grounding/assets/allegro_floating.xml"],
                         cwd=REPO, capture_output=True, text=True, check=True).stdout
    return _materialise(xml)


def _replay(xml_path: pathlib.Path, acts: np.ndarray) -> dict:
    import grounding.envs.lift_hold_env as env_mod  # noqa: PLC0415

    original = env_mod.MJCF
    env_mod.MJCF = xml_path
    try:
        env = env_mod.LiftHoldEnv(episode_len=int(len(acts)) + 5, domain_rand=False)
        env.reset(seed=0)
        for a in acts:
            env.step(np.asarray(a, dtype=np.float64))
        lift = np.asarray(env.trace["lift"], dtype=float)
        sup = np.asarray(env.trace["support_force"], dtype=float)
        fing = np.asarray(env.trace["finger_force"], dtype=float)
        obj_z = np.asarray(env.trace["obj_z"], dtype=float)
        pi = int(np.argmax(lift))
        step = np.abs(np.diff(obj_z, prepend=obj_z[0]))
        out = {
            "peak_lift_m": float(lift[pi]),
            "peak_step": pi,
            "n_steps": int(len(lift)),
            "support_force_N_at_peak": float(sup[pi]),
            "A3b_airborne_at_peak": bool(sup[pi] < 1e-6),
            "n_steps_above_5cm": int(np.sum(lift >= LIFT_BAR_M)),
            "n_steps_above_5cm_and_airborne": int(
                np.sum((lift >= LIFT_BAR_M) & (sup < 1e-6))),
            "finger_force_N_at_peak": float(fing[pi]),
            "max_single_step_displacement_m": float(step.max()),
            "final_lift_m": float(lift[-1]),
            "max_lift_over_first_half_m": float(np.max(lift[:max(1, len(lift) // 2)])),
            "max_lift_over_last_quarter_m": float(np.max(lift[(3 * len(lift)) // 4:])),
        }
        env.close()
        return out
    finally:
        env_mod.MJCF = original


def main() -> None:
    if not DEMOS.exists():
        raise SystemExit(f"missing {DEMOS}")
    z = np.load(DEMOS, allow_pickle=False)
    lens = np.asarray(z["bc_trajectory_len"], dtype=int)
    old = _old_scene_path()
    new = REPO / "grounding/assets/allegro_floating.xml"

    rows = []
    ofs = 0
    for i, L in enumerate(lens):
        acts = z["bc_act"][ofs: ofs + int(L)]
        ofs += int(L)
        rows.append({
            "trajectory": i,
            "frames": int(L),
            f"scene_{OLD_COMMIT}": _replay(old, acts),
            f"scene_{NEW_COMMIT}": _replay(new, acts),
        })
        print(json.dumps(rows[-1], indent=1), flush=True)

    out = {
        "question": "do the round-p0(4d) recorded hook actions still clear A3b "
                    "on the current scene?",
        "method": "identical action sequences, identical env code and thresholds, "
                  "only grounding/assets/allegro_floating.xml differs",
        "recorded_on": {"commit": OLD_COMMIT, "sha256": OLD_SHA},
        "replayed_on": {"commit": NEW_COMMIT, "sha256": NEW_SHA},
        "lift_bar_m": LIFT_BAR_M,
        "rows": rows,
        "summary": {
            "A3b_pass_old": [r[f"scene_{OLD_COMMIT}"]["A3b_airborne_at_peak"] for r in rows],
            "A3b_pass_new": [r[f"scene_{NEW_COMMIT}"]["A3b_airborne_at_peak"] for r in rows],
            "steps_above_5cm_old": [r[f"scene_{OLD_COMMIT}"]["n_steps_above_5cm"] for r in rows],
            "steps_above_5cm_new": [r[f"scene_{NEW_COMMIT}"]["n_steps_above_5cm"] for r in rows],
            "finger_force_N_at_peak_old": [r[f"scene_{OLD_COMMIT}"]["finger_force_N_at_peak"] for r in rows],
            "finger_force_N_at_peak_new": [r[f"scene_{NEW_COMMIT}"]["finger_force_N_at_peak"] for r in rows],
            "reading": (
                "the hook raises the box only 3 steps above the 5 cm bar and "
                "reaches that peak with ZERO finger force - the box is launched, "
                "not carried. Its A3b pass on 4ee6f1d was the apex step of that "
                "launch happening to have no contact, i.e. a single-step "
                "coincidence, consistent with the p0(4d) adjudication that the "
                "passing set has measure ~zero. On 1b0f13f that coincidence is "
                "gone: the same actions leave the box in contact with the "
                "pedestal at their own peak."),
        },
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
