"""Task 4(e) evidence: what the new pedestal-borne-load term reads on real traces.

The reward change in :mod:`grounding.envs.lift_hold_env` (W_PEDESTAL_LOAD) is
only worth anything if the quantity it prices is actually the thing that
separates the f210f45 corner wedge from a genuine lift. This module measures it
on both, on the CURRENT scene, and writes the numbers into the run's report:

  * the round-p0(4d) PPO policy (archived as ``ppo_wedge_f210f45.zip``), which
    A1/A2/A3a/A3c/A4 pass and A3b fails 10/10 with a 42-degree nose-down wedge;
  * the round-p0(4d) scripted thumb rim-hook demonstrations, archived as
    ``reachable_demo_set_f210f45.npz``. They cleared A3b on the scene that
    recorded them; replayed here, on the current scene, they no longer do
    (``hook_scene_ab.json``), which is itself part of the evidence - they are
    the closest thing to a clean lift the scene offers, at ~0.67 box weights of
    pedestal load against the wedge's ~18.

It reports, per trace: the per-step pedestal-borne normal force over the lift
window (the new reward term's input), the box weight, the resulting share, and
the load decomposition of the FINGER forces by box face, so the three bearing
paths (top-face hook / side-wall friction / pedestal) are comparable between
the two mechanisms in one table.

    python grounding/pedestal_load_probe.py --policy <zip> --demos <npz> \
        --out runs/p0/smoke4_mujoco_ppo/pedestal_load_evidence.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

import mujoco

REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

RUN_DIR = REPO / "runs/p0/smoke4_mujoco_ppo"
WEDGE_POLICY = RUN_DIR / "ppo_wedge_f210f45.zip"
DEMOS = RUN_DIR / "reachable_demo_set_f210f45.npz"
OUT = RUN_DIR / "pedestal_load_evidence.json"
EPISODE_LEN = 400


def bearing_decomposition(env, peak_i: int) -> dict:
    """Split the load carrying the box over the lift window by mechanism.

    Three mutually exclusive bearing paths, as the adjudication named them:

      hook     upward force transmitted through contacts on the box's TOP face
      wall     upward force transmitted through contacts on the SIDE faces
               (friction along the wall - a real grasp's second bearing path)
      pedestal upward force the static support geometry applies to the box -
               the path A3b forbids, and the one W_PEDESTAL_LOAD now prices

    ``top``/``wall`` are measured from the FINGER contacts only, so the sum of
    the three is a genuine decomposition of what is holding the box up, not a
    restatement of the total contact magnitude. Forces are read from the live
    ``mj_contactForce`` at the end state, so this must be called with the env
    rolled to the peak step.
    """
    import mujoco
    m, d = env.model, env.data
    half = np.asarray(m.geom_size[env.obj_geom])
    bp = np.asarray(d.xpos[env.obj_body]).copy()
    bxm = np.asarray(d.xmat[env.obj_body]).reshape(3, 3).copy()
    acc = {"hook_up_N": 0.0, "wall_up_N": 0.0, "pedestal_up_N": 0.0}
    mag = {"hook_N": 0.0, "wall_N": 0.0, "pedestal_N": 0.0}
    for i in range(d.ncon):
        con = d.contact[i]
        g1, g2 = int(con.geom1), int(con.geom2)
        if env.obj_geom not in (g1, g2):
            continue
        other = g2 if g1 == env.obj_geom else g1
        f6 = np.zeros(6)
        mujoco.mj_contactForce(m, d, i, f6)
        total = float(np.linalg.norm(f6[:3]))
        if total <= 0.0:
            continue
        # world-frame force transmitted by this contact
        normal_world = _contact_normal_world(env, i)
        force_world = normal_world * float(f6[0])
        up = float(force_world[2])
        label = env.geom_label[other]
        if label in env.support_geoms:
            acc["pedestal_up_N"] += up
            mag["pedestal_N"] += total
            continue
        if other not in env.finger_geom_ids:
            continue
        rel = bxm.T @ (np.asarray(con.pos) - bp)
        on_top = (abs(rel[2] - half[2]) <= max(half[2] * 0.5, 1e-3)
                  and abs(rel[0]) <= half[0] + 1e-9 and abs(rel[1]) <= half[1] + 1e-9
                  and rel[2] > 0.0)
        if on_top:
            acc["hook_up_N"] += up
            mag["hook_N"] += total
        else:
            acc["wall_up_N"] += up
            mag["wall_N"] += total
    up_total = acc["hook_up_N"] + acc["wall_up_N"] + acc["pedestal_up_N"]
    out = {k: round(v, 4) for k, v in acc.items()}
    out |= {k: round(v, 4) for k, v in mag.items()}
    out["upward_total_N"] = round(up_total, 4)
    weight = float(env.model.body_mass[env.obj_body]) * 9.81
    out["box_weight_N"] = round(weight, 4)
    # Normalised by the box's OWN weight, not by the sum of the three paths.
    # Normalising by the sum is unstable exactly where the mechanism is
    # interesting: on a corner wedge the fingers press DOWN while the pedestal
    # presses UP, so the signed sum cancels and every "share" blows up. In units
    # of box weight each path reads as what it physically carries, and the sign
    # of a path is meaningful (negative = that path pushes the box down).
    out["bearing_box_weights"] = {
        k: round(v / weight, 4) if weight else 0.0
        for k, v in (("hook", acc["hook_up_N"]), ("wall", acc["wall_up_N"]),
                     ("pedestal", acc["pedestal_up_N"]))
    }
    out["bearing_box_weights_total"] = round(up_total / weight, 4) if weight else 0.0
    return out


def _contact_normal_world(env, i: int) -> np.ndarray:
    """Unit world-frame normal of contact i, oriented INTO the box from the other geom.

    MuJoCo stores ``mjContact.frame`` as a 3x3 row-major rotation whose first ROW
    (``frame[0:3]``) is the normal, directed from geom1 toward geom2. Verified
    against this scene: a box resting on the pedestal gives frame[0] == (0,0,1)
    when the box is geom1 and the pedestal geom2, i.e. pointing OUT of the box.
    Reading the first COLUMN instead (the obvious mistake) yields (0,0,-1) and
    silently inverts every bearing share, so the flip below is explicit.
    ``mj_contactForce`` returns the positive-compression normal magnitude; with
    the normal oriented into the box, ``fn * n`` is the force the OTHER geom
    applies TO the box.
    """
    d = env.data
    frame = np.asarray(d.contact[i].frame).reshape(3, 3)
    n = frame[0].copy()
    n /= max(float(np.linalg.norm(n)), 1e-12)
    if int(d.contact[i].geom1) == env.obj_geom:
        n = -n
    return n


def _peak(env) -> int:
    return int(np.argmax(env.trace["lift"]))


def make_env_shaper_weight(weight: float) -> float:
    """Round-trip the ablation weight through :class:`LiftHoldEnv`.

    Exists so the tests can assert that the knob a subprocess env is built with
    is the shipped one, without constructing a full gym env in a reward test.
    """
    from grounding.envs.lift_hold_env import LiftHoldEnv

    env = LiftHoldEnv(episode_len=1, domain_rand=False, pedestal_load_weight=weight)
    try:
        return float(env._shaper.pedestal_load_weight)
    finally:
        env.close()


def resting_support_sign_check(env) -> dict:
    """Sanity calibration for the normal orientation: a box AT REST on the
    pedestal must read a positive upward bearing close to its own weight.

    Without this the whole decomposition is a sign lottery - the first COLUMN of
    ``contact.frame`` also looks like a plausible normal and inverts every share
    silently. Asserted in tests/test_smoke4_pedestal_load.py.
    """
    d = env.data
    up = 0.0
    mag = 0.0
    for i in range(d.ncon):
        con = d.contact[i]
        g1, g2 = int(con.geom1), int(con.geom2)
        if env.obj_geom not in (g1, g2):
            continue
        other = g2 if g1 == env.obj_geom else g1
        if env.geom_label[other] not in env.support_geoms:
            continue
        f6 = np.zeros(6)
        mujoco.mj_contactForce(env.model, d, i, f6)
        up += float((_contact_normal_world(env, i) * float(f6[0]))[2])
        mag += float(np.linalg.norm(f6[:3]))
    weight = float(env.model.body_mass[env.obj_body]) * 9.81
    return {"upward_N": round(up, 4), "magnitude_N": round(mag, 4),
            "box_weight_N": round(weight, 4),
            "ratio_up_to_weight": round(up / weight, 4) if weight else 0.0}


def _summarise(name: str, env, ret: float, extra: dict) -> dict:
    tr = env.trace
    lift = np.asarray(tr["lift"], dtype=float)
    sup = np.asarray(tr["support_force"], dtype=float)
    fing = np.asarray(tr["finger_force"], dtype=float)
    pi = _peak(env)
    win = slice(0, pi + 1)
    weight = float(env.model.body_mass[env.obj_body]) * 9.81
    return {
        "name": name,
        "return_new_reward": round(ret, 3),
        "peak_lift_m": round(float(lift[pi]), 5),
        "peak_step": pi,
        "table_touching_at_peak": bool(any(
            p[0] in env.support_geoms for p in tr["contact_pairs"][pi])),
        "A3b_pass": bool(not any(p[0] in env.support_geoms
                                 for p in tr["contact_pairs"][pi])),
        "support_force_N_at_peak": round(float(sup[pi]), 4),
        "support_force_N_mean_over_lift_window": round(float(np.mean(sup[win])), 4),
        "support_force_N_max": round(float(np.max(sup)), 4),
        "support_share_of_box_weight_at_peak": round(float(sup[pi]) / weight, 4),
        "support_share_of_box_weight_mean_window": round(
            float(np.mean(sup[win])) / weight, 4),
        "finger_force_N_at_peak": round(float(fing[pi]), 4),
        "bearing_at_peak": bearing_decomposition(env, pi),
        **extra,
    }


def run_wedge(policy: pathlib.Path, seed: int = 10_000, n_ep: int = 3) -> list[dict]:
    from stable_baselines3 import PPO

    from grounding.envs.lift_hold_env import LiftHoldEnv

    model = PPO.load(str(policy), device="cpu")
    env = LiftHoldEnv(episode_len=EPISODE_LEN)
    rows = []
    for ep in range(n_ep):
        obs, _ = env.reset(seed=seed + ep)
        done, total = False, 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, rew, term, trunc, _ = env.step(action)
            total += rew
            done = term or trunc
        rows.append(_summarise(f"wedge_ppo_ep{ep}", env, total, {"seed": seed + ep}))
    env.close()
    return rows


def run_demos(demos: pathlib.Path) -> list[dict]:
    from grounding.envs.lift_hold_env import LiftHoldEnv

    z = np.load(demos, allow_pickle=False)
    lens = np.asarray(z["bc_trajectory_len"], dtype=int)
    rows = []
    ofs = 0
    for i, L in enumerate(lens):
        act = z["bc_act"][ofs: ofs + int(L)]
        env = LiftHoldEnv(episode_len=int(L) + 5, domain_rand=False)
        env.reset(seed=0)
        total = 0.0
        for a in act:
            _, rew, _, _, _ = env.step(np.asarray(a, dtype=np.float64))
            total += rew
        rows.append(_summarise(f"rim_hook_demo{i}", env, total,
                               {"frames": int(L),
                                "demo_mjcf_commit": str(z["mjcf_commit"])}))
        env.close()
        ofs += int(L)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", type=pathlib.Path, default=WEDGE_POLICY)
    ap.add_argument("--demos", type=pathlib.Path, default=DEMOS)
    ap.add_argument("--out", type=pathlib.Path, default=OUT)
    ap.add_argument("--episodes", type=int, default=3)
    args = ap.parse_args()

    rows = []
    if args.demos.exists():
        rows += run_demos(args.demos)
    if args.policy.exists():
        rows += run_wedge(args.policy, n_ep=args.episodes)

    def agg(prefix: str, key: str) -> dict:
        vals = [r[key] for r in rows if r["name"].startswith(prefix) and r.get(key) is not None]
        if not vals:
            return {}
        return {"min": min(vals), "median": float(np.median(vals)), "max": max(vals)}

    out = {
        "purpose": "evidence that W_PEDESTAL_LOAD prices A3b's criterion: the "
                   "f210f45 corner wedge and the clean rim-hook are separated "
                   "by the pedestal-borne normal force, on this scene",
        "mjcf_sha256": "1ea0fdf5c87ce5a161efcda7c597b0fc6647c9fe30bcf2f5d0bcb52402d24b89",
        "episode_len": EPISODE_LEN,
        "rows": rows,
        "summary": {
            "wedge": {
                "support_share_of_box_weight_at_peak":
                    agg("wedge", "support_share_of_box_weight_at_peak"),
                "A3b_pass_any": any(r["A3b_pass"] for r in rows
                                    if r["name"].startswith("wedge")),
                "return_new_reward": agg("wedge", "return_new_reward"),
            },
            "hook": {
                "support_share_of_box_weight_at_peak":
                    agg("rim_hook", "support_share_of_box_weight_at_peak"),
                "A3b_pass_all": all(r["A3b_pass"] for r in rows
                                    if r["name"].startswith("rim_hook")),
                "return_new_reward": agg("rim_hook", "return_new_reward"),
            },
        },
    }
    args.out.write_text(json.dumps(out, indent=2))
    for r in rows:
        b = r["bearing_at_peak"]["bearing_box_weights"]
        print(f"{r['name']:>18}  lift {r['peak_lift_m']*1000:7.1f}mm  "
              f"A3b={r['A3b_pass']}  pedestal {r['support_force_N_at_peak']:8.2f}N "
              f"({r['support_share_of_box_weight_at_peak']:6.2f} bw)  "
              f"bearing bw hook/wall/ped = {b['hook']:+.2f}/{b['wall']:+.2f}/"
              f"{b['pedestal']:+.2f}  R={r['return_new_reward']:.1f}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
