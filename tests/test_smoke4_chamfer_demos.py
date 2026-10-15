"""p0(4) final slice - the chamfer-scene demo set is replay evidence, not a claim.

The demo set (reachable_demo_set_c7c6dac.npz) is recorded on the COMMITTED
c7c6dac scene (xml sha256 d61dc0de..., 3 mm top-edge chamfer + 9-decimal mass
fields).  Every trajectory is one cell of the 91-cell perturbation grid around
the chamfer scan's grid-winner cfg, kept because the frozen gate passed on the
recorded frames themselves.  These tests pin three things:

  r1  provenance/layout (D4 keys, MJCF commit + sha, selection marker)
  r2  the recorded frames are BIT-EXACTLY reproducible by replaying the cfg -
      same lift curve, same contacts, same bearing decomposition - and each
      replay still passes the frozen gate verbatim
  r3  the set is the measured survivor boundary of the committed scene
      (chamfer91_recheck.json), including the honest negative: the pre-fix
      grid-winner cell (0, 0) does NOT survive the mass-precision fix and is
      not in the set (drift measured, 27/91 -> 22/91)

    runs/.venv-mj/bin/python -m pytest tests/test_smoke4_chamfer_demos.py -q
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

RUN_DIR = REPO / "runs/p0/smoke4_mujoco_ppo"
NPZ = RUN_DIR / "reachable_demo_set_c7c6dac.npz"
RECHECK_JSON = RUN_DIR / "chamfer91_recheck.json"
MJCF_COMMIT = "c7c6dac"
MJCF_SHA256 = "d61dc0de3ca313f2eaf9971ac5eef550642073a5ea8386fc4c46aa787ea0823e"
SELECTION = "chamfer3mm_gate_clean_cells"

pytest.importorskip("mujoco")


@pytest.fixture(scope="module")
def demo():
    assert NPZ.exists(), f"missing {NPZ} - run collect_chamfer_demos.py"
    return dict(np.load(NPZ, allow_pickle=False))


@pytest.fixture(scope="module")
def cfgs(demo) -> list[dict]:
    return json.loads(str(demo["cfg_json"]))


@pytest.fixture(scope="module")
def gates(demo) -> list[dict]:
    return json.loads(str(demo["gate_metrics_json"]))


@pytest.fixture(scope="module")
def decomps(demo) -> list[dict]:
    return json.loads(str(demo["load_decomposition_json"]))


# --------------------------------------------------------------------- r1
def test_r1_d4_layout_and_provenance(demo):
    for key in ("bc_obs", "bc_act", "bc_pose", "bc_trajectory_len",
                "mjcf_commit", "mjcf_sha256", "selection", "cfg_json",
                "gate_metrics_json", "load_decomposition_json"):
        assert key in demo, f"demo npz missing D4 key '{key}'"
    assert str(demo["mjcf_commit"]) == MJCF_COMMIT
    assert str(demo["mjcf_sha256"]) == MJCF_SHA256, (
        "demo set was not recorded against the committed c7c6dac scene")
    assert str(demo["selection"]) == SELECTION
    lens = demo["bc_trajectory_len"]
    assert int(lens.sum()) == len(demo["bc_obs"]) == len(demo["bc_act"])
    assert demo["bc_obs"].shape[1] == 64 and demo["bc_act"].shape[1] == 22
    assert len(lens) >= 2, "a one-trajectory set is the zero-measure mistake "
    "this round exists to replace"


# --------------------------------------------------------------------- r2
@pytest.fixture(scope="module")
def replays(demo, cfgs):
    """Replay every recorded cfg once; share across assertions."""
    from grounding.envs.lift_hold_env import LiftHoldEnv
    from grounding.reachable_scan import gate_eval, run_traj

    lens = [int(v) for v in demo["bc_trajectory_len"]]
    starts = np.concatenate([[0], np.cumsum(lens)])
    out = []
    for i, cfg in enumerate(cfgs):
        env = LiftHoldEnv(episode_len=400, domain_rand=False,
                          box_offset_xy=tuple(cfg["offset_xy_m"]),
                          box_yaw=float(cfg["yaw_rad"]))
        obs, act = run_traj(env, cfg, record=True)
        ev = gate_eval(env)
        out.append({
            "obs": obs, "act": act, "gate": ev,
            "rec_obs": demo["bc_obs"][starts[i]:starts[i + 1]],
            "rec_act": demo["bc_act"][starts[i]:starts[i + 1]],
        })
        env.close()
    return out


def test_r2_replay_is_bit_exact(replays):
    """The frames are physics, not a log: replaying the cfg must reproduce the
    recorded observations and actions exactly (MuJoCo CPU determinism)."""
    for i, r in enumerate(replays):
        assert r["obs"].shape == r["rec_obs"].shape, f"traj {i} length drifted"
        assert np.array_equal(r["act"], r["rec_act"]), (
            f"traj {i}: replayed actions are not bit-identical to the recording")
        assert np.array_equal(r["obs"], r["rec_obs"]), (
            f"traj {i}: replayed observations are not bit-identical to the "
            "recording - the recorded lift/contacts/forces are not reproducible")


def test_r2b_every_replay_still_passes_the_frozen_gate(replays, gates):
    for i, r in enumerate(replays):
        ev = r["gate"]
        assert ev["gate_pass"], (
            f"traj {i} no longer passes the frozen gate on replay "
            f"(peak {ev['peak_lift_m']*1000:.1f} mm)")
        rec = gates[i]
        assert abs(rec["peak_lift_m"] - ev["peak_lift_m"]) < 1e-9, (
            f"traj {i}: recorded peak lift {rec['peak_lift_m']} != replayed "
            f"{ev['peak_lift_m']}")
        assert rec["table_touching_at_peak"] == ev["table_touching_at_peak"]
        assert ev["anchored_step_ok"] and ev["free_fall_gravity_ok"] \
            and ev["separation_ok"]


def test_r2c_bearing_decomposition_matches_replay(replays, cfgs, decomps):
    """The recorded -x-wall face split is re-derived from a replay, not trusted
    from the collector: the whole lift reaction must ride the -x wall with zero
    top-face and zero pedestal participation (per cell)."""
    from grounding.envs.lift_hold_env import LiftHoldEnv
    from grounding.reachable_scan import contact_load_decomposition, run_traj
    for i, cfg in enumerate(cfgs):
        env = LiftHoldEnv(episode_len=400, domain_rand=False,
                          box_offset_xy=tuple(cfg["offset_xy_m"]),
                          box_yaw=float(cfg["yaw_rad"]))
        run_traj(env, cfg)  # decomposition resets and re-steps internally
        ld = contact_load_decomposition(env, replays[i]["act"],
                                        int(replays[i]["gate"]["peak_step"]))
        env.close()
        rec = decomps[i]
        for k in ("top", "x_pos", "x_neg", "y_pos", "y_neg", "other"):
            assert abs(ld[k] - rec[k]) < 1e-9, (
                f"traj {i}: bearing path '{k}' drifted: recorded {rec[k]} vs "
                f"replayed {ld[k]}")
        assert rec["x_neg"] == 1.0, (
            f"traj {i}: lift reaction is not 100% -x wall ({rec})")
        assert rec["top"] == 0.0, f"traj {i}: top face bears load ({rec})"


# --------------------------------------------------------------------- r3
def test_r3_set_is_the_measured_boundary_not_a_hand_pick(demo, cfgs):
    assert RECHECK_JSON.exists(), (
        f"missing {RECHECK_JSON} - the 91-cell re-probe is the boundary of record")
    recheck = json.loads(RECHECK_JSON.read_text())
    assert recheck["mjcf_commit"] == MJCF_COMMIT
    assert recheck["n_cells"] == 91
    survivors = {(r["off_mm"], r["yaw_rad"]) for r in recheck["cells"]
                 if r["gate_pass"]}
    recorded = {(round(c["offset_xy_m"][0] * 1000.0, 6), c["yaw_rad"])
                for c in cfgs}
    assert recorded == survivors, (
        "recorded cells != measured survivor cells: the set was narrowed or "
        "padded by hand")
    assert len(survivors) == len(cfgs) == int(len(demo["bc_trajectory_len"]))


def test_r3b_prefix_winner_cell_is_honestly_absent(cfgs):
    """Drift pinned: on the committed 9dp-mass scene the scan's grid-winner
    cell (0, 0) peaks at 47.7 mm - below the 50 mm bar - so it is NOT in the
    set.  If a future scene change brings it back, this test demands the set be
    re-collected rather than silently padded."""
    cells = [(round(c["offset_xy_m"][0] * 1000.0, 6), c["yaw_rad"]) for c in cfgs]
    assert (0.0, 0.0) not in cells
    yaws = sorted({c["yaw_rad"] for c in cfgs})
    assert all(y > 0.0 for y in yaws), "unexpected yaw=0 survivor"
    assert len({(c["offset_xy_m"][0], c["yaw_rad"]) for c in cfgs}) >= 2


def test_r3c_dead_cells_still_die(replays, cfgs):
    """Negative control: cells the recheck declared dead must still fail the
    frozen gate on replay - the boundary is a measurement in BOTH directions."""
    recheck = json.loads(RECHECK_JSON.read_text())
    dead = [(r["off_mm"], r["yaw_rad"]) for r in recheck["cells"]
            if not r["gate_pass"]]
    assert dead, "recheck claims every cell lives; boundary unmeasured"
    from grounding.envs.lift_hold_env import LiftHoldEnv
    from grounding.reachable_scan import gate_eval, run_traj
    base_cfg = json.loads(str(dict(np.load(NPZ, allow_pickle=False))["cfg_json"]))[0]
    for mm, yaw in (dead[0], dead[len(dead) // 2], dead[-1]):
        env = LiftHoldEnv(episode_len=400, domain_rand=False,
                          box_offset_xy=(mm / 1000.0, mm / 1000.0),
                          box_yaw=float(yaw))
        run_traj(env, {k: v for k, v in base_cfg.items()
                       if k not in ("offset_xy_m", "yaw_rad")})
        ev = gate_eval(env)
        env.close()
        assert not ev["gate_pass"], (
            f"cell ({mm} mm, {yaw} rad) is recorded dead in the recheck but "
            f"passes now (peak {ev['peak_lift_m']*1000:.1f} mm) - the scene or "
            "the gate moved; re-collect the demo set")
