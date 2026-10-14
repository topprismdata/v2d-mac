"""Build runs/p0/smoke4_mujoco_ppo/report.json from the training + eval evidence.

    python grounding/envs/make_report.py

verdict is computed from the same criteria tests/test_smoke4_accept.py asserts; it is
never hand-written. Failures are reported as failures with the measured numbers.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import platform
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

RUN_DIR = REPO / "runs/p0/smoke4_mujoco_ppo"
LIFT_THRESHOLD = 0.05
MIN_CORRELATION = 0.3
MAX_STEP_DISPLACEMENT = 0.01


def _sha256(path: pathlib.Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    import mujoco  # noqa: PLC0415
    import stable_baselines3  # noqa: PLC0415
    import gymnasium  # noqa: PLC0415
    import torch  # noqa: PLC0415

    sys.path.insert(0, str(REPO / "tests"))
    from test_smoke4_accept import _analyse_episode  # noqa: PLC0415

    eval_path = RUN_DIR / "eval_episodes.json"
    if not eval_path.exists():
        raise SystemExit(f"missing {eval_path}")
    ev = json.loads(eval_path.read_text())
    train = json.loads((RUN_DIR / "train_summary.json").read_text())

    per_ep = []
    for ep in ev["episodes"]:
        a = _analyse_episode(ep)
        per_ep.append({"episode": ep["episode"], "seed": ep["seed"],
                       "return": ep["return"], **a})

    best = max(per_ep, key=lambda e: e["peak_lift_m"])

    # --- curve summary --------------------------------------------------
    curve_csv = RUN_DIR / "curve.csv"
    curve_summary: dict = {"path": "runs/p0/smoke4_mujoco_ppo/curve.csv"}
    if curve_csv.exists() and len(curve_csv.read_text().strip().splitlines()) > 1:
        lines = curve_csv.read_text().strip().splitlines()
        header = lines[0].split(",")
        rows = [dict(zip(header, ln.split(","))) for ln in lines[1:]]
        rew = np.array([float(r["reward_mean"]) for r in rows])
        peak = np.array([float(r["z_peak_mean"]) for r in rows])
        half = len(rows) // 2
        curve_summary.update({
            "n_windows": len(rows),
            "first_global_step": int(rows[0]["global_step"]),
            "last_global_step": int(rows[-1]["global_step"]),
            "reward_first_half_mean": float(rew[:half].mean()),
            "reward_second_half_mean": float(rew[half:].mean()),
            "reward_improved": bool(rew[half:].mean() > rew[:half].mean()),
            "z_peak_first_half_mean": float(peak[:half].mean()),
            "z_peak_second_half_mean": float(peak[half:].mean()),
            "z_peak_max_windowed": float(peak.max()),
        })

    # --- criteria -------------------------------------------------------
    any_episode_lifts = best["peak_lift_m"] > LIFT_THRESHOLD
    finger_pairs = bool(best["finger_pairs"]) and best["n_finger_contact_steps"] >= 5
    corr = best["corr_force_lift_finger_only"]
    corr_ok = (not np.isnan(corr)) and corr > MIN_CORRELATION
    airborne = not best["table_touching_at_peak"]
    no_teleport = best["max_per_step_displacement_m"] < MAX_STEP_DISPLACEMENT
    ctrl_ok = bool(ev.get("control_out_of_reach", {}).get("pass", False))

    criteria = {
        "A1_episode_lift_gt_5cm": {
            "value_m": best["peak_lift_m"], "threshold_m": LIFT_THRESHOLD,
            "pass": bool(any_episode_lifts)},
        "A2_finger_object_contact_pair": {
            "n_finger_contact_steps": best["n_finger_contact_steps"],
            "pairs": best["finger_pairs"], "pass": finger_pairs},
        "A3a_contact_force_lift_correlation": {
            "pearson_r": None if np.isnan(corr) else float(corr),
            "threshold": MIN_CORRELATION, "pass": bool(corr_ok)},
        "A3b_airborne_at_peak": {
            "table_touching_at_peak": best["table_touching_at_peak"],
            "pass": airborne},
        "A3c_no_teleport": {
            "max_per_step_displacement_m": best["max_per_step_displacement_m"],
            "threshold_m": MAX_STEP_DISPLACEMENT, "pass": bool(no_teleport)},
        "A4_control_out_of_reach": {
            "peak_lift_m": ev.get("control_out_of_reach", {}).get("peak_lift_m"),
            "threshold_m": LIFT_THRESHOLD, "pass": ctrl_ok},
    }
    all_pass = all(c["pass"] for c in criteria.values())
    if not curve_summary.get("reward_improved", False):
        all_pass = False

    model_path = RUN_DIR / "ppo_lift_hold.zip"
    report = {
        "smoke": "smoke4_mujoco_ppo",
        "verdict": "pass" if all_pass else "fail",
        "seed": train["seed"],
        "device": {
            "mujoco_device": "cpu",
            "sb3_device": "cpu",
            "torch_mps_available": bool(torch.backends.mps.is_available()),
            "rationale": (
                "MuJoCo has no MPS backend; SB3 PPO trains on CPU. The policy MLP is "
                "small enough that MPS would not change wall time, and no torch op in "
                "this smoke is device-bound."
            ),
            "platform": platform.platform(),
            "processor": platform.processor() or platform.machine(),
            "cpu_count": __import__("os").cpu_count(),
        },
        "key_lib_versions": {
            "mujoco": mujoco.__version__,
            "stable_baselines3": stable_baselines3.__version__,
            "gymnasium": gymnasium.__version__,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "python": platform.python_version(),
        },
        "model": {
            "xml": "grounding/assets/allegro_floating.xml",
            "upstream": "google-deepmind/mujoco_menagerie wonik_allegro "
                        "@ d7b481bac63d3d600ab404ee74f7474c5efe8a67",
            "xml_sha256": _sha256(REPO / "grounding/assets/allegro_floating.xml"),
            "nq": 29, "nv": 28, "nu": 22,
            "weights_file": "runs/p0/smoke4_mujoco_ppo/ppo_lift_hold.zip",
            "weights_sha256": _sha256(model_path),
        },
        "training": train,
        "curve_summary": curve_summary,
        "evaluator_thresholds": ev["evaluator"],
        "acceptance_thresholds": {
            "lift_m": LIFT_THRESHOLD,
            "min_correlation": MIN_CORRELATION,
            "max_per_step_displacement_m": MAX_STEP_DISPLACEMENT,
            "min_finger_contact_steps": 5,
        },
        "contact_evidence": {
            "best_episode": best["episode"],
            "best_episode_seed": best["seed"],
            "peak_lift_m": best["peak_lift_m"],
            "peak_step": best["peak_step"],
            "peak_force_N": best["peak_force_N"],
            "wrist_z_at_peak_m": best["wrist_z_at_peak"],
            "n_finger_contact_steps": best["n_finger_contact_steps"],
            "finger_object_geom_pairs": best["finger_pairs"],
            "pearson_r_finger_force_vs_lift": (
                None if np.isnan(corr) else float(corr)),
            "table_touching_at_peak": best["table_touching_at_peak"],
            "max_per_step_displacement_m": best["max_per_step_displacement_m"],
        },
        "criteria": criteria,
        "per_episode": per_ep,
        "design_deviations": [
            "MuJoCo forbids actuators on a <freejoint>, so the hand's floating base "
            "is expressed as the freejoint's own 3 slide + 3 hinge joints "
            "(wrist_x/y/z, wrist_rx/ry/rz), each with a written ctrlrange and "
            "forcerange. The object body keeps the literal <freejoint/>.",
            "Upstream menagerie hand uses <position kp=\"1\"/> with no forcerange "
            "and joint damping 0.1, authored for a table-mounted hand. For a "
            "free-floating hand these are raised to kp=3, forcerange +-2 N*m and "
            "damping 0.5 so the 16-joint posture servo can hold the hand against "
            "gravity. Gravity itself is NOT compensated - the ~35 mm steady-state "
            "sag under zero ctrl is real physics.",
            "The tissue-box long axis (0.24 m) is aligned with the palm long axis "
            "(y) because all three fingers flex in the x-z plane; with the long "
            "axis on x the 0.24 m extent would point away from the fingers.",
            "PLAN task 4.1 cites github.com/mujoco/mujoco_menagerie, which does not "
            "exist; the real upstream is google-deepmind/mujoco_menagerie and the "
            "Allegro hand now lives under wonik_allegro/.",
        ],
        "evidence_files": {
            "curve": "runs/p0/smoke4_mujoco_ppo/curve.csv",
            "eval_episodes": "runs/p0/smoke4_mujoco_ppo/eval_episodes.json",
            "train_log": "runs/p0/smoke4_mujoco_ppo/train.log",
            "train_summary": "runs/p0/smoke4_mujoco_ppo/train_summary.json",
        },
    }
    (RUN_DIR / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"verdict": report["verdict"],
                      "criteria": {k: v["pass"] for k, v in criteria.items()}}, indent=2))


if __name__ == "__main__":
    main()
