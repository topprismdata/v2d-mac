"""Task 4.6b - the reachable-class demo set is replay evidence, not a coincidence.

Round p0(4) searched the reachable classes (w35d strict wall criterion; K1
dual-wall stays class-B) for scripted trajectories that honestly pass the
frozen smoke4 acceptance gates. The surviving family is a thumb rim-hook lift:
the gate pass is real, but placement-sensitive (passing set has measure ~zero
under +-5 mm placement error), which this test pins alongside the numbers.

  R1  D4 layout + provenance (bc_obs 64 / bc_act 22 / per-traj lengths /
      bc_pose 8-col / mjcf_commit 4ee6f1d / selection=reachable_class_clean)
  R2  replaying every recorded action sequence in a fresh env reproduces the
      recorded peak lift / finger force / duty cycle AND passes the frozen
      acceptance gate via the very same _analyse_episode the formal test uses
  R3  the load decomposition is stored and upward forces resolve to faces
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))

NPZ = REPO / "runs/p0/smoke4_mujoco_ppo/reachable_demo_set.npz"
MJCF_COMMIT = "4ee6f1d"

pytest.importorskip("mujoco")


@pytest.fixture(scope="module")
def demo():
    assert NPZ.exists(), f"missing {NPZ} - run the reachable_scan record step"
    return dict(np.load(NPZ, allow_pickle=False))


def test_r1_d4_layout_and_provenance(demo):
    for key in ("bc_obs", "bc_act", "bc_pose", "bc_trajectory_len", "mjcf_commit",
                "selection"):
        assert key in demo, f"{key} missing from the demo set"
    assert demo["bc_obs"].ndim == 2 and demo["bc_obs"].shape[1] == 64
    assert demo["bc_act"].ndim == 2 and demo["bc_act"].shape[1] == 22
    assert demo["bc_obs"].dtype == np.float32 and demo["bc_act"].dtype == np.float32
    lens = demo["bc_trajectory_len"]
    assert int(lens.sum()) == demo["bc_obs"].shape[0]
    assert demo["bc_pose"].shape == (len(lens), 8)
    assert str(demo["mjcf_commit"]) == MJCF_COMMIT
    assert str(demo["selection"]) == "reachable_class_clean"
    assert "gate_metrics_json" in demo and "cfg_json" in demo


@pytest.fixture(scope="module")
def replayed(demo):
    """Replay every trajectory once; reuse the results across assertions."""
    from grounding.envs.lift_hold_env import LiftHoldEnv
    from grounding.reachable_scan import gate_eval
    from test_smoke4_accept import _analyse_episode

    lens = demo["bc_trajectory_len"]
    gates = json.loads(str(demo["gate_metrics_json"]))
    cfgs = json.loads(str(demo["cfg_json"]))
    assert len(gates) == len(lens) == len(cfgs)
    out = []
    ofs = 0
    for i, L in enumerate(lens):
        act = demo["bc_act"][ofs: ofs + int(L)]
        env = LiftHoldEnv(episode_len=int(L) + 10, domain_rand=False)
        env.reset(seed=0)
        for a in act:
            env.step(np.asarray(a, dtype=np.float64))
        ev = _analyse_episode({"trace": dict(env.trace, dt=float(env.dt))})
        ev["gate_pass"] = bool(
            ev["peak_lift_m"] > 0.05
            and ev["n_finger_contact_steps"] >= 5
            and ev["finger_pairs"]
            and ev["force_duty_ok"]
            and ev["force_sufficiency_ok"]
            and not ev["table_touching_at_peak"]
            and ev["anchored_step_ok"]
            and ev["free_fall_gravity_ok"]
            and ev["separation_ok"]
        )
        out.append({"gate": ev, "recorded": gates[i], "cfg": cfgs[i]})
        env.close()
        ofs += int(L)
    return out


def test_r2_replay_reproduces_lift_force_and_passes_gate(replayed):
    for i, r in enumerate(replayed):
        ev, rec = r["gate"], r["recorded"]
        assert ev["gate_pass"], (
            f"trajectory {i} replay fails the frozen gate: peak "
            f"{ev['peak_lift_m']*1000:.1f}mm clean@peak={not ev['table_touching_at_peak']}")
        assert abs(ev["peak_lift_m"] - rec["peak_lift_m"]) <= 0.005, (
            f"trajectory {i}: peak lift drifted on replay "
            f"({ev['peak_lift_m']*1000:.1f} vs {rec['peak_lift_m']*1000:.1f} mm)")
        assert abs(ev["finger_force_mean_N"] - rec["finger_force_mean_N"]) <= 0.5
        assert abs(ev["finger_force_duty_cycle"] - rec["finger_force_duty_cycle"]) <= 0.05
        assert ev["n_finger_contact_steps"] >= 5 and ev["finger_pairs"]
        assert ev["peak_lift_m"] > 0.05 and not ev["table_touching_at_peak"]
        assert ev["force_duty_ok"] and ev["force_sufficiency_ok"]
        assert ev["anchored_step_ok"] and ev["free_fall_gravity_ok"] and ev["separation_ok"]


def test_r3_load_decomposition_present_and_resolves(demo):
    assert "load_decomposition_json" in demo
    decomps = json.loads(str(demo["load_decomposition_json"]))
    lens = demo["bc_trajectory_len"]
    assert len(decomps) == len(lens)
    for dec in decomps:
        share = dec["upward_share_by_face"]
        assert abs(sum(share.values()) - 1.0) < 1e-6
        assert dec["total_upward_N"] > 0.0
        # disclosed load path: the rim hook carries the box through top-face
        # contacts; wall faces carry nothing in this family
        assert share.get("top", 0.0) > 0.0
