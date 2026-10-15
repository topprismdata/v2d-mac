"""P0 plumbing - the BC -> PPO -> eval pipeline, pinned against silent drift.

Scene-agnostic plumbing validation (commit 622d64b, MJCF 4508421): the demo
collector can emit a dataset, the BC warm start transfers into an SB3 MlpPolicy,
a PPO checkpoint round-trips prediction-identically, and the deterministic
evaluation writes the formal ``eval_episodes.json`` schema. Nothing here makes a
performance claim: the demo sets used on MJCF 4508421 are
``selection="best_available_not_clean"`` (no clean lift exists - see the
committed verdict), so the warm start is proven to LOAD, not to help.

  P1  the BC dataset schema matches the D4 layout test_smoke4_demos.py pins
      (bc_obs N x 64 / bc_act N x 22 / per-trajectory lengths / mjcf_commit)
  P2  the BC state dict transfers into SB3 and measurably changes the policy
  P3  a PPO checkpoint saves and reloads prediction-identically
  P4  the deterministic evaluation writes the formal evaluation-report schema
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

pytest.importorskip("mujoco")
pytest.importorskip("stable_baselines3")

from grounding.bc_pretrain import (  # noqa: E402
    ACT_DIM,
    NET_ARCH,
    OBS_DIM,
    BCPolicyMLP,
    load_bc_dataset,
    load_into_sb3,
    train_bc,
)
from grounding.collect_clean_demos import (  # noqa: E402
    MJCF_COMMIT,
    _make_env,
    bc_demo_set,
)
from grounding import train_ppo  # noqa: E402
from grounding.envs.lift_hold_env import LiftHoldEnv  # noqa: E402

D4_KEYS = {"bc_obs", "bc_act", "bc_pose", "bc_trajectory_len", "mjcf_commit"}
EVAL_TOP_KEYS = {"n_episodes", "evaluator", "best_episode", "control_out_of_reach",
                 "episodes"}
EVAL_EPISODE_KEYS = {"episode", "seed", "return", "z0", "max_lift_m", "final_lift_m",
                     "max_contact_force_N", "contact_pairs_any",
                     "contact_pairs_at_peak", "n_steps_with_finger_contact", "trace"}
EVAL_TRACE_KEYS = {"obj_z", "lift", "wrist_z", "contact_force", "finger_force",
                   "contact_pairs", "obj_vz", "supported", "finger_held", "dt"}
CONTROL_KEYS = {"n_episodes", "peak_lift_m", "n_finger_contact_steps",
                "threshold_m", "pass"}


@pytest.fixture(scope="module")
def demo_npz(tmp_path_factory) -> pathlib.Path:
    """A tiny real BC dataset straight from the collector (quick trials)."""
    out = tmp_path_factory.mktemp("demos") / "bc_demos.npz"
    info = bc_demo_set(n_trials=2, top=2, out=out, seed=7, quick=True)
    assert info["saved"] == 2
    return out


@pytest.fixture(scope="module")
def bc_pt(demo_npz: pathlib.Path, tmp_path_factory) -> pathlib.Path:
    """The BC warm start trained on that dataset (few epochs, CPU)."""
    out = tmp_path_factory.mktemp("bc") / "bc_policy.pt"
    meta = train_bc(demo_npz, out, epochs=3, batch_size=64, device="cpu", seed=0)
    assert np.isfinite(meta["final_loss"])
    return out


@pytest.fixture(scope="module")
def trained_out(tmp_path_factory, bc_pt: pathlib.Path) -> pathlib.Path:
    """A short BC-initialised PPO run, saved into a scratch artifact dir."""
    out_dir = tmp_path_factory.mktemp("ppo_run")
    train_ppo.train(total_steps=256, n_envs=2, seed=0, episode_len=60,
                    early_stop_lift=0.10, resume=False, out_dir=out_dir,
                    init_from_bc=bc_pt)
    return out_dir


# ------------------------------------------------------------------ P1: dataset
def test_p1_bc_dataset_schema(demo_npz: pathlib.Path):
    """The collector's npz must keep the D4 layout the demo tests pin."""
    z = np.load(demo_npz, allow_pickle=False)
    assert D4_KEYS <= set(z.files), f"D4 keys missing: {D4_KEYS - set(z.files)}"
    assert z["bc_obs"].dtype == np.float32 and z["bc_obs"].ndim == 2
    assert z["bc_obs"].shape[1] == OBS_DIM, z["bc_obs"].shape
    assert z["bc_act"].dtype == np.float32 and z["bc_act"].shape[1] == ACT_DIM
    lens = z["bc_trajectory_len"]
    assert lens.dtype == np.int32 and lens.ndim == 1
    assert int(lens.sum()) == z["bc_obs"].shape[0]
    assert len(lens) == z["bc_pose"].shape[0]
    assert z["bc_pose"].shape[1] == 8  # wx wy wz rx ry rz a_long a_thumb
    assert str(z["mjcf_commit"]) == MJCF_COMMIT
    assert str(z["selection"]) == "best_available_not_clean"


def test_p1_loader_rejects_missing_keys(tmp_path):
    """A truncated npz must fail loudly, not hand BC a silent half-dataset."""
    bad = tmp_path / "bad.npz"
    np.savez_compressed(bad, bc_obs=np.zeros((4, OBS_DIM), np.float32))
    with pytest.raises(ValueError, match="missing keys"):
        load_bc_dataset(bad)


def test_p1_loader_rejects_dim_drift(tmp_path):
    bad = tmp_path / "bad.npz"
    np.savez_compressed(bad, bc_obs=np.zeros((4, 63), np.float32),
                        bc_act=np.zeros((4, ACT_DIM), np.float32),
                        bc_trajectory_len=np.array([4], np.int32),
                        bc_pose=np.zeros((1, 8), np.float32),
                        mjcf_commit=np.array(MJCF_COMMIT))
    with pytest.raises(ValueError, match=r"bc_obs must be"):
        load_bc_dataset(bad)


# ------------------------------------------------------------- P2: BC transfer
def test_p2_bc_state_dict_transfers_into_sb3(bc_pt: pathlib.Path):
    """Warm start must load into a fresh MlpPolicy and MOVE it.

    Verifies the exact contract train_ppo.train relies on: policy_net/action_net
    key-for-key, and a measurably non-zero delta (an all-zero delta would mean
    the payload was an untrained net and "warm start" was a fiction).
    """
    import torch  # noqa: PLC0415

    from stable_baselines3 import PPO  # noqa: PLC0415

    env = LiftHoldEnv(episode_len=30)
    model = PPO("MlpPolicy", env, seed=0, policy_kwargs={"net_arch": [256, 256]},
                n_steps=32, batch_size=32)
    obs0 = np.zeros(OBS_DIM, np.float32)
    act_before, _ = model.predict(obs0, deterministic=True)

    info = load_into_sb3(model, bc_pt)

    assert set(info["max_abs_delta"]) == {"policy_net", "action_net"}
    assert all(d > 0.0 for d in info["max_abs_delta"].values())
    assert act_before.shape == (ACT_DIM,)
    act_after, _ = model.predict(obs0, deterministic=True)
    assert not np.array_equal(act_before, act_after), \
        "policy output unchanged after a non-zero-delta BC load"

    payload = torch.load(bc_pt, map_location="cpu", weights_only=False)
    meta = payload["meta"]
    assert meta["obs_dim"] == OBS_DIM and meta["act_dim"] == ACT_DIM
    assert meta["net_arch"] == list(NET_ARCH)
    assert meta["mjcf_commit"] == MJCF_COMMIT
    env.close()


def test_p2_bc_module_shapes_match_sb3():
    """The standalone BC net must be shape-identical to the SB3 stack it targets."""
    import torch  # noqa: PLC0415

    from stable_baselines3 import PPO  # noqa: PLC0415

    env = LiftHoldEnv(episode_len=30)
    policy = PPO("MlpPolicy", env, seed=0,
                 policy_kwargs={"net_arch": [256, 256]}).policy
    bc = BCPolicyMLP.build()
    # SB3 submodule state dicts are unprefixed; compare module-for-module
    assert set(bc.policy_net.state_dict()) == set(policy.mlp_extractor.policy_net.state_dict())
    assert set(bc.action_net.state_dict()) == set(policy.action_net.state_dict())
    bc_shapes = {k: tuple(v.shape) for k, v in bc.policy_net.state_dict().items()}
    bc_shapes.update({f"action_net.{k}": tuple(v.shape)
                      for k, v in bc.action_net.state_dict().items()})
    sb3_shapes = {k: tuple(v.shape)
                  for k, v in policy.mlp_extractor.policy_net.state_dict().items()}
    sb3_shapes.update({f"action_net.{k}": tuple(v.shape)
                       for k, v in policy.action_net.state_dict().items()})
    assert bc_shapes == sb3_shapes
    env.close()


# ------------------------------------------------------- P3: checkpoint roundtrip
def test_p3_checkpoint_round_trip_is_lossless(trained_out: pathlib.Path):
    """save -> load must reproduce deterministic actions exactly."""
    from stable_baselines3 import PPO  # noqa: PLC0415

    assert (trained_out / "ppo_lift_hold.zip").exists()
    obs = np.random.default_rng(0).uniform(-1, 1, (8, OBS_DIM)).astype(np.float32)

    pre = PPO.load(trained_out / "ppo_lift_hold", device="cpu")
    a_pre, _ = pre.predict(obs, deterministic=True)
    a_pre2, _ = pre.predict(obs, deterministic=True)
    assert np.array_equal(a_pre, a_pre2)  # deterministic predict is deterministic

    post = PPO.load(trained_out / "ppo_lift_hold", device="cpu")
    a_post, _ = post.predict(obs, deterministic=True)
    assert np.array_equal(a_pre, a_post), "checkpoint round trip changed actions"

    pre_sd = pre.policy.state_dict()
    post_sd = post.policy.state_dict()
    assert set(pre_sd) == set(post_sd)
    for k in pre_sd:
        assert torch_equal(pre_sd[k], post_sd[k]), f"weight {k} changed on reload"


def torch_equal(a, b) -> bool:
    import torch  # noqa: PLC0415

    return torch.equal(a.detach().cpu(), b.detach().cpu())


# -------------------------------------------------------------- P4: eval schema
def test_p4_eval_writes_formal_schema(trained_out: pathlib.Path):
    """Deterministic evaluation must produce the formal eval_episodes.json schema."""
    res = train_ppo.evaluate(n_episodes=2, seed=123, episode_len=40,
                             out_dir=trained_out)
    path = trained_out / "eval_episodes.json"
    assert path.exists()
    on_disk = json.loads(path.read_text())
    assert on_disk.keys() == res.keys()

    assert set(res) == EVAL_TOP_KEYS
    assert res["n_episodes"] == len(res["episodes"])
    assert 0 <= res["best_episode"] < len(res["episodes"])
    assert set(res["control_out_of_reach"]) == CONTROL_KEYS
    assert isinstance(res["control_out_of_reach"]["pass"], bool)
    assert set(res["evaluator"]) >= {"lift_threshold_m", "min_hold_steps"}

    for ep in res["episodes"]:
        assert set(ep) == EVAL_EPISODE_KEYS
        assert set(ep["trace"]) == EVAL_TRACE_KEYS
        assert len(ep["trace"]["lift"]) == len(ep["trace"]["contact_pairs"])
        for p in ep["contact_pairs_at_peak"]:
            assert len(p) == 2 and all(isinstance(x, str) for x in p)


def test_p4_env_matches_pipeline_dims():
    """The dims the whole pipeline assumes come from the env, not from constants."""
    env = _make_env(episode_len=20)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (OBS_DIM,) and obs.dtype == np.float32
    assert env.action_space.shape == (ACT_DIM,)
    env.close()
