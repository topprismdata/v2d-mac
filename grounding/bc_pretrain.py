"""P0 plumbing - BC warm start for the SB3 PPO policy (scene-agnostic validation).

Trains a behaviour-cloning MLP on a collected demo set and saves its weights so
``grounding/train_ppo.py --init-from-bc`` can load them into a fresh PPO policy
before training. The BC network is EXACTLY the SB3 ``MlpPolicy`` policy stack it
will be copied into (``policy_kwargs={"net_arch": [256, 256]}``, SB3-default
Tanh activations):

    policy_net = Linear(obs_dim, 256) -> Tanh -> Linear(256, 256) -> Tanh
    action_net = Linear(256, act_dim)

so the state dict transfers with no shape surgery. The value head and log_std
are left at SB3 init: warm-starting the critic would need returns labelled per
demo step, which the demo collector does not record (and this module does not
pretend to).

Honesty note: BC on the plumbing demo set is NOT expected to produce a competent
policy. The sets collected on MJCF 4508421 are best-available-not-clean (see
``collect_clean_demos.bc_demo_set``), and this module exists so the
BC -> PPO -> eval pipeline is proven end to end before the scene revision lands.
No performance claim may be built on it.

    python grounding/bc_pretrain.py --demos runs/p0/plumb_check/bc_demos.npz \
        --out runs/p0/plumb_check/bc_policy.pt --epochs 30
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from grounding.collect_clean_demos import MJCF_COMMIT  # noqa: E402

def _env_dims() -> tuple[int, int]:
    """W6a: obs/act dims come from the env, not a constant - the dual-hand
    scene changed them (107 / 44) and a hardcoded 64/22 would silently
    mis-shape every downstream net. Same import the env itself uses."""
    from grounding.envs.lift_hold_env import N_ACT, N_OBS  # noqa: PLC0415
    return N_OBS, N_ACT


OBS_DIM, ACT_DIM = _env_dims()
NET_ARCH = (256, 256)

REQUIRED_NPZ_KEYS = ("bc_obs", "bc_act", "bc_trajectory_len", "bc_pose", "mjcf_commit")


def _committed_xml_sha256() -> str:
    """sha256 of the allegro_floating.xml blob this git tree actually ships.

    The authoritative scene identity: MJCF_COMMIT-style constants go stale
    whenever a scene commit forgets to bump them (c7c6dac did), but the blob
    cannot lie.
    """
    import hashlib
    import subprocess

    raw = subprocess.run(
        ["git", "show", "HEAD:grounding/assets/allegro_floating.xml"],
        cwd=REPO, check=True, capture_output=True).stdout
    return hashlib.sha256(raw).hexdigest()


def load_bc_dataset(path: str | pathlib.Path) -> dict:
    """Load and validate a D4-layout demo npz; raise on any schema drift."""
    import torch  # noqa: PLC0415

    path = pathlib.Path(path)
    z = np.load(path, allow_pickle=False)
    missing = [k for k in REQUIRED_NPZ_KEYS if k not in z]
    if missing:
        raise ValueError(f"{path}: missing keys {missing} (D4 layout requires "
                         f"{REQUIRED_NPZ_KEYS})")
    obs = np.asarray(z["bc_obs"], dtype=np.float32)
    act = np.asarray(z["bc_act"], dtype=np.float32)
    lens = np.asarray(z["bc_trajectory_len"], dtype=np.int64)
    if obs.ndim != 2 or obs.shape[1] != OBS_DIM:
        raise ValueError(f"bc_obs must be (N, {OBS_DIM}), got {obs.shape}")
    if act.ndim != 2 or act.shape[1] != ACT_DIM:
        raise ValueError(f"bc_act must be (N, {ACT_DIM}), got {act.shape}")
    if int(lens.sum()) != obs.shape[0]:
        raise ValueError(f"bc_trajectory_len sums to {int(lens.sum())} but "
                         f"bc_obs has {obs.shape[0]} frames")
    if len(lens) != len(np.asarray(z["bc_pose"])):
        raise ValueError("one bc_pose row per trajectory expected")
    return {
        "obs": torch.from_numpy(obs),
        "act": torch.from_numpy(act),
        "lens": lens,
        "mjcf_commit": str(z["mjcf_commit"]),
        "mjcf_sha256": str(z["mjcf_sha256"]) if "mjcf_sha256" in z else None,
        "n_trajectories": int(len(lens)),
    }


class BCPolicyMLP:
    """The SB3-transferrable policy stack, kept as functions over torch modules."""

    @staticmethod
    def build(obs_dim: int = OBS_DIM, act_dim: int = ACT_DIM):
        import torch  # noqa: PLC0415
        import torch.nn as nn  # noqa: PLC0415

        class _Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.policy_net = nn.Sequential(
                    nn.Linear(obs_dim, NET_ARCH[0]), nn.Tanh(),
                    nn.Linear(NET_ARCH[0], NET_ARCH[1]), nn.Tanh(),
                )
                self.action_net = nn.Linear(NET_ARCH[1], act_dim)

            def forward(self, x):
                return self.action_net(self.policy_net(x))

        return _Net()

    @staticmethod
    def state_dict_of(net) -> dict:
        return {"policy_net": net.policy_net.state_dict(),
                "action_net": net.action_net.state_dict()}


def train_bc(demos: str | pathlib.Path, out: str | pathlib.Path, *, epochs: int = 30,
             batch_size: int = 256, lr: float = 1e-3, device: str = "cpu",
             seed: int = 0) -> dict:
    """Fit MSE(obs -> act) on the demo set; save an SB3-loadable state dict."""
    import torch  # noqa: PLC0415

    data = load_bc_dataset(demos)
    # Provenance against the scene this tree ACTUALLY ships. The old constant
    # import (collect_clean_demos.MJCF_COMMIT == "1b0f13f") went stale when
    # c7c6dac committed the chamfered XML without bumping it, which would
    # reject every demo set recorded on the current scene. The committed
    # blob's sha256 is the authority (same caliber as the assignment: git show
    # HEAD:grounding/assets/allegro_floating.xml | shasum -a 256); the npz
    # stamps the same digest at collection time. Sets recorded before that
    # stamp existed (no mjcf_sha256 key) keep the legacy constant check.
    xml_sha = _committed_xml_sha256()
    if data["mjcf_sha256"] is not None:
        if str(data["mjcf_sha256"]) != xml_sha:
            raise ValueError(
                f"demo set recorded against XML sha {data['mjcf_sha256']}, "
                f"this tree ships {xml_sha}")
    elif data["mjcf_commit"] != MJCF_COMMIT:
        raise ValueError(f"demo set recorded against MJCF {data['mjcf_commit']}, "
                         f"this tree is {MJCF_COMMIT}")
    torch.manual_seed(seed)
    dev = torch.device(device)
    net = BCPolicyMLP.build().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    obs, act = data["obs"].to(dev), data["act"].to(dev)

    losses = []
    gen = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        perm = torch.randperm(obs.shape[0], generator=gen).to(dev)
        epoch_loss = 0.0
        for i in range(0, obs.shape[0], batch_size):
            idx = perm[i:i + batch_size]
            pred = net(obs[idx])
            loss = torch.nn.functional.mse_loss(pred, act[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += float(loss.detach()) * len(idx)
        losses.append(epoch_loss / obs.shape[0])

    out = pathlib.Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = BCPolicyMLP.state_dict_of(net.cpu())
    meta = {
        "kind": "bc_warm_start",
        "obs_dim": OBS_DIM,
        "act_dim": ACT_DIM,
        "net_arch": list(NET_ARCH),
        "activation": "tanh",
        "target": "sb3 MlpPolicy mlp_extractor.policy_net + action_net",
        "mjcf_commit": MJCF_COMMIT,
        "mjcf_commit_live": subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO, check=True,
            capture_output=True).stdout.decode().strip(),
        "xml_sha256": xml_sha,
        "demo_path": str(demos),
        "n_frames": int(data["obs"].shape[0]),
        "n_trajectories": data["n_trajectories"],
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "device": device,
        "seed": seed,
        "final_loss": losses[-1] if losses else None,
        "losses_per_epoch": losses,
    }
    payload["meta"] = meta
    torch.save(payload, out)
    return meta


def load_into_sb3(model, bc_path: str | pathlib.Path) -> dict:
    """Copy BC weights into a fresh SB3 PPO policy, in place.

    Verifies every copied tensor actually moved the policy (a state dict that
    loads into a fresh net with an all-zero delta means the BC file was saved
    from an untrained net or the keys silently matched nothing).
    """
    import torch  # noqa: PLC0415

    payload = torch.load(bc_path, map_location="cpu", weights_only=False)
    meta = payload.get("meta", {})
    if meta.get("kind") != "bc_warm_start":
        raise ValueError(f"{bc_path} is not a bc_warm_start payload")
    policy = model.policy
    deltas = {}

    def _copy(sb3_module, bc_state: dict, tag: str) -> None:
        before = {k: v.detach().clone() for k, v in sb3_module.state_dict().items()}
        # strict=True: raises on any key or shape mismatch with the SB3 module
        sb3_module.load_state_dict(bc_state)
        after = sb3_module.state_dict()
        deltas[tag] = max(float((after[k].detach() - before[k]).abs().max())
                          for k in before)

    _copy(policy.mlp_extractor.policy_net, payload["policy_net"], "policy_net")
    _copy(policy.action_net, payload["action_net"], "action_net")
    if max(deltas.values()) == 0.0:
        raise ValueError("BC load was a no-op: every weight identical to SB3 init")
    return {"max_abs_delta": deltas, "meta": {k: v for k, v in meta.items()
                                              if k != "losses_per_epoch"}}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--demos", required=True, help="npz from collect_clean_demos")
    p.add_argument("--out", required=True, help="where to save the .pt warm start")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default="cpu", choices=("cpu", "mps"))
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    meta = train_bc(args.demos, args.out, epochs=args.epochs,
                    batch_size=args.batch_size, lr=args.lr, device=args.device,
                    seed=args.seed)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
