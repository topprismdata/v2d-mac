"""v21 coupled-closure PPO - the final card of the W6c lift chain.

Episodes start at the v21 placement-solved ENGAGE state (right ff/mf
rotation-aligned placement pressing 4-7 N, left rf placement pressing
1.6-3.1 N, box self-centered): the policy learns the COUPLED CLOSURE from
the engaged state instead of exploring from zero. A baseline wrist climb
(0.1 mm/frame while the vise sum criterion holds) is composed UNDER the
policy action (residual form): the policy owns the coupled corrections
(wrist x/y + curls) that keep the pads on the faces while the box rises.

Reward (per Main, slip/yaw as first-class citizens):
  +6 x min(clear/5mm, 1)            main lift
  +0.3 per frame at clear >= 5 mm   sustain
  -3 x max(0, slip_mm - 1)/10       cumulative xy slip (soft-thresholded)
  -0.5 x |yaw|/10                   box yaw
  +0.3 x min(N_r x N_l / 9, 1)      bilateral coupling (N in N)
  -1 per hand-pad pressing with the pad above the box top (the f40
    drape-over-the-top failure signature, measured via contact vectors)
All scaled by REWARD_SCALE (0.1) like every other W6c round.

Eval: pre-registered frozen gate (eval_stream_against_gate verbatim),
3 seeds, pass >= 2/3; on pass - replay_frames (65-wide qpos) + standard
camera evidence frames.
"""
import argparse
import json
import os
import pathlib
import sys

import mujoco
import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import grounding.train_w6c as W  # noqa: E402
from grounding.envs.lift_hold_env import LiftHoldEnv  # noqa: E402
from grounding.envs.lift_hold_env import N_ACT, N_OBS  # noqa: E402

OUT_DIR = REPO / "runs/w6c"
ENGAGE_PATH = OUT_DIR / "v21_engage_checkpoint.npz"
REPLAY_PATH = OUT_DIR / "v21_replay_frames.json"
FS = 25
EP_LEN = 160
BASE_Z = 0.0125          # action-units of baseline climb (0.1 mm/frame)
W_BOX = 1.4715


# ---------------------------------------------------------------- engage
def _jadr(m, name):
    return int(m.jnt_qposadr[mujoco.mj_name2id(
        m, mujoco.mjtObj.mjOBJ_JOINT, name)])


def _padp(env, gname):
    return env.data.geom_xpos[mujoco.mj_name2id(
        env.model, mujoco.mjtObj.mjOBJ_GEOM, gname)] - \
        env.data.xpos[env.obj_body]


def capture_engage() -> dict:
    """Yaw-normalized v21 engage snapshot: settle -> box yaw -> 0 ->
    sequenced placement (right ff/mf rotated solve, then left rf+y solve)
    -> closed-loop seats -> press/engage settle -> yaw re-normalize ->
    short re-seat -> snapshot (qpos 65 + ctrl 50)."""
    env = W.make_trigger_env(episode_len=400, noise=False)
    m = env.model
    env.reset(seed=0)
    bid = env.obj_body
    d = env.data
    oa = env.obj_qpos_adr
    def settle(n):
        for _ in range(n):
            mujoco.mj_step(m, d)

    def yaw_deg():
        q = d.qpos[oa + 3: oa + 7]
        return float(np.degrees(2.0 * np.arctan2(q[3], q[0])))

    # straight-box initialization (video-semantic box is square): the
    # inherited 6.2 deg trigger yaw was a retarget artifact and the source
    # of the 3-point wedge bind - zero it BEFORE the placement, then
    # re-settle so the placement solve (re-run for the straight box)
    # seats against the true geometry
    d.qpos[oa + 3: oa + 7] = [1.0, 0.0, 0.0, 0.0]
    settle(120)

    def normalize_yaw():
        d.qpos[oa + 3: oa + 7] = [1.0, 0.0, 0.0, 0.0]
        settle(120)

    rest_z = float(d.xpos[bid][2])
    # NOTE: no pre-placement yaw normalization - the trigger geometry
    # (box yaw ~6.2 deg inherited from the retarget trigger) is the frame
    # every placement solve was measured in; absolute-zero normalization
    # de-seats both hands (measured N 0/0). Training episodes randomize
    # yaw +-2 deg around the engage state instead (start_noise), and the
    # frozen yaw clause judges absolute yaw <= 10 deg.
    start_ctrl = d.ctrl.copy()
    clear = start_ctrl.copy()
    for i_ in (7, 8, 11, 12, 37, 38):
        clear[i_] = 0.40
    clear[2] += 0.050
    transit = clear.copy()
    place_r = transit.copy()
    place_r[0] += 0.040; place_r[1] += 0.060; place_r[2] += 0.050
    place_r[3] += 0.00; place_r[4] += 0.75; place_r[5] += -0.10
    place_r[11] = 0.45; place_r[12] = 0.35
    for a, b, n in ((start_ctrl, clear, 12), (clear, transit, 10),
                    (transit, place_r, 15)):
        for k in range(n):
            d.ctrl[:] = a + (b - a) * (k + 1) / n
            settle(FS)
    settle(15 * FS)
    place_l = d.ctrl.copy()
    place_l[23] += 0.120; place_l[22] += 0.070
    place_l[24] += -0.015
    place_l[25] += 0.00; place_l[26] += 0.60; place_l[27] += 0.00
    place_l[37] = 0.40; place_l[38] = 0.30
    for k in range(15):
        d.ctrl[:] = place_r + (place_l - place_r) * (k + 1) / 15
        settle(FS)
    settle(10 * FS)

    # curl press (Main cut-b: pure normal loading AFTER the seat)
    for k in range(12):
        f = (k + 1) / 12.0
        d.ctrl[11] += (0.20 * f) / 12
        d.ctrl[12] += (0.20 * f) / 12
        d.ctrl[37] += (0.15 * f) / 12
        d.ctrl[38] += (0.15 * f) / 12
        settle(FS)
    settle(20 * FS)

    def padp(g):
        return d.geom_xpos[mujoco.mj_name2id(
            m, mujoco.mjtObj.mjOBJ_GEOM, g)] - d.xpos[bid]

    cr_lo, cr_hi = m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1]
    # seat-correction budget: the loop below subtracts the box-relative tip
    # height from the ABSOLUTE ctrl z each iteration; without a budget the
    # left seat ratcheted 8.9 cm down (0.0309 -> -0.0577) = a ~955 N
    # down-press through the kp-15000 z actuator that pinned the box to the
    # table for every prior run. Bound total correction to +-8 mm.
    z_seat_r, z_seat_l = float(d.ctrl[2]), float(d.ctrl[24])
    for _ in range(6):
        bl = padp("rf_tip_left_collision")
        y_err = bl[1] - (-0.095)
        d.ctrl[23] = float(np.clip(d.ctrl[23] - y_err * 2.0,
                                   cr_lo[23], cr_hi[23]))
        z_r = (padp("ff_tip_collision")[2] + padp("mf_tip_collision")[2]) / 2
        x_r = (padp("ff_tip_collision")[0] + padp("mf_tip_collision")[0]) / 2 \
            - 0.066
        x_l = -bl[0] + 0.066
        d.ctrl[2] = float(np.clip(d.ctrl[2] - z_r,
                                  max(cr_lo[2], z_seat_r - 0.008),
                                  min(cr_hi[2], z_seat_r + 0.008)))
        d.ctrl[24] = float(np.clip(d.ctrl[24] - bl[2],
                                   max(cr_lo[24], z_seat_l - 0.008),
                                   min(cr_hi[24], z_seat_l + 0.008)))
        d.ctrl[0] = float(np.clip(d.ctrl[0] + float(np.clip(x_r, -0.012, 0.012)),
                                  cr_lo[0], cr_hi[0]))
        if abs(y_err) < 0.004:
            d.ctrl[22] = float(np.clip(
                d.ctrl[22] - float(np.clip(x_l, -0.015, 0.015)),
                cr_lo[22], cr_hi[22]))
        settle(10 * FS)
    # force-source onset on both sides until the vise sum stabilizes
    lo_c, hi_c = m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1]
    for f in range(60):
        # right demand halved: the yaw equilibrium scales with the right
        # press (the press line sits y +64..92 mm off the face center on
        # this mount) - a gentler seat keeps the engage yaw near ~6-7 deg
        for idx, jn, dem in ((7, "ffj1", 0.10), (8, "ffj2", 0.10),
                             (11, "mfj1", 0.10), (12, "mfj2", 0.10),
                             (37, "rfj1_left", 0.30),
                             (38, "rfj2_left", 0.30)):
            d.ctrl[idx] = float(np.clip(d.qpos[_jadr(m, jn)] + dem,
                                        lo_c[idx], hi_c[idx]))
        settle(FS)
    settle(20 * FS)
    # wedge release: the force-onset press wedges the box ~1 cm UP on the
    # top edge (box_z 0.75 vs the 0.7399 rest) - descend the right wrist
    # until the box settles back onto the table plane, so the engage state
    # starts at TRUE rest with the vise closed (a wedged start would pass
    # the window/support clauses as a corner-wedge false lift)
    for _ in range(40):
        if d.xpos[bid][2] < 0.7415:
            break
        d.ctrl[2] -= 0.0005
        settle(FS)
    settle(20 * FS)
    n_r, n_l = W.pad_normal_forces(env)
    er, el = W.endface_contact_per_hand(env)
    meta = {
        "yaw_engage_deg": yaw_deg(),
        "n_r": n_r, "n_l": n_l, "endface": [er, el],
        "rf_tip": [float(v) for v in padp("rf_tip_left_collision")],
        "ff_tip_r": [float(v) for v in padp("ff_tip_collision")],
        "box_z": float(d.xpos[bid][2]),
    }
    print(f"[capture] engage state: N=({n_r:.2f},{n_l:.2f}) "
          f"endface=({er},{el}) yaw={meta['yaw_engage_deg']:+.2f} deg")
    np.savez(ENGAGE_PATH, qpos=d.qpos.copy(), ctrl=d.ctrl.copy(),
             obj_xpos=d.xpos[bid].copy(), obj_xquat=d.xquat[bid].copy(),
             rest_z=rest_z, n_approach_frames=0)
    (OUT_DIR / "v21_engage_meta.json").write_text(
        json.dumps(meta, indent=1, default=float))
    return meta


def make_closure_env(episode_len: int = EP_LEN, seed: int = 0):
    """Env at the v21 engage state. The env's start_noise branch REPLACES
    the box quaternion absolutely, which would wipe the captured engage
    yaw (7.9 deg) and de-seat the placed hands - so start_noise is off and
    the yaw jitter (+-2 deg AROUND the engage orientation) is applied here
    as part of the start-state protocol (reset-adjacent initialization;
    step() never writes qpos)."""
    z = np.load(ENGAGE_PATH)
    env = LiftHoldEnv(episode_len=episode_len, ctrl_hz=W.CTRL_HZ,
                      domain_rand=False, start_state=(z["qpos"], z["ctrl"]),
                      start_noise=None, grasp_bias_nm=0.0)
    env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    yaw = float(rng.normal(0.0, 0.035))
    half = yaw / 2.0
    oa = env.obj_qpos_adr
    q0 = env.data.qpos[oa + 3: oa + 7].copy()
    dq = np.array([np.cos(half), 0.0, 0.0, np.sin(half)])
    q_new = np.zeros(4)
    mujoco.mju_mulQuat(q_new, dq, q0)
    env.data.qpos[oa + 3: oa + 7] = q_new
    for _ in range(100):
        mujoco.mj_step(env.model, env.data)
    # the gate baseline is the TRUE table plane (captured pre-placement),
    # not the wedged engage height - a wedged start reads as initial
    # clearance and is disclosed/judged as such (honesty clause)
    env._z_rest = float(z["rest_z"]) - env.box_half_z
    env._z0 = float(env.data.xpos[env.obj_body][2])
    env._z_peak = env._z0
    env._v21_xy0 = env.data.xpos[env.obj_body][:2].copy()
    return env


# ---------------------------------------------------------------- reward
def closure_reward(env, info, prev=None, rate_w=2.0):
    """prev: {'yaw': deg, 'slip': m} from the previous frame - the
    stabilization continuation adds yaw-RATE and slip-RATE penalties
    (weight rate_w) that damp swing kinetic energy directly."""
    bp = W.box_pose(env)
    clear = bp["bottom_clearance_m"]
    n_r, n_l = W.pad_normal_forces(env)
    slip = float(np.linalg.norm(
        env.data.xpos[env.obj_body][:2] - env._v21_xy0))
    yaw = abs(bp["yaw_deg"])
    r = 6.0 * min(clear / 0.005, 1.0)
    if clear >= 0.005:
        r += 0.3
    r -= 3.0 * max(0.0, slip * 1e3 - 1.0) / 10.0
    r -= 0.5 * yaw / 10.0
    r += 0.3 * min(n_r * n_l / 9.0, 1.0)
    half = float(env.model.geom_size[env.obj_geom][2])
    for g in ("ff_tip_collision", "mf_tip_collision",
              "rf_tip_left_collision"):
        p = _padp(env, g)
        if p[2] > half + 0.002:
            r -= 1.0
    if prev is not None:
        r -= rate_w * abs(bp["yaw_deg"] - prev["yaw"]) / 1.0
        r -= rate_w * abs(slip - prev["slip"]) / 0.001
    new_prev = {"yaw": bp["yaw_deg"], "slip": slip}
    return r, new_prev


# ------------------------------------------------------------------ ppo
GRIP_FLOOR_ACTS = (7, 8, 11, 12, 37, 38)


N_TARGET_PER_SIDE = 3.9   # the lift friction threshold 2.16 x 1.8
                          # safety (Main r3 authorization)
SERVO_KP = 0.05           # the action units per N of N-error
SERVO_DEADBAND = 0.5      # N
SERVO_MAX = 0.05          # the per-frame action cap


def force_servo(env, a: np.ndarray) -> np.ndarray:
    """r3 authorized: the normal-force inner loop on the curl channels -
    the curl increment toward the target N per side (the lift friction
    threshold 2.16 x 1.8 safety), +- deadband, capped per frame. Training
    AND the deterministic eval share the servo (the deployment-gap fix:
    the mean action + the servo = the vise never relaxes). The servo
    holds the NORMAL force only - slip/yaw/window remain the policy's to
    earn."""
    n_r, n_l = env._pad_normals_obs()
    for side, n_now in ((0, n_r), (1, n_l)):
        err = N_TARGET_PER_SIDE - n_now
        if abs(err) <= SERVO_DEADBAND:
            continue
        delta = float(np.clip(SERVO_KP * err, -SERVO_MAX, SERVO_MAX))
        base = 11 + side * 22          # mfa1: ctrl 11 (right), 33 (left)
        a[base] = max(a[base], 0.0) + delta
        a[base + 1] = max(a[base + 1], 0.0) + delta
    return a


def inner_servo(env, st: dict) -> None:
    """Track C: the 1 kHz-class low-level servo, called per substep.

    (1) the normal-force N loop: the curl ctrl increments toward the
    target N per side; (2) the box-pose damping: the tilt and the yaw
    rate feed the wrist differential channels - the constructive
    suppression of the carry drift."""
    m, d = env.model, env.data

    n_r, n_l = env._pad_normals_obs()
    for lo_off in (11, 37):
        n_now = n_r if lo_off == 11 else n_l
        err = N_TARGET_PER_SIDE - n_now
        if abs(err) > 0.3:
            inc = float(np.clip(st["kp"] * err / 25.0, -0.002, 0.002))
            d.ctrl[lo_off] = float(np.clip(d.ctrl[lo_off] + inc,
                                           st["lo"][lo_off], st["hi"][lo_off]))
            d.ctrl[lo_off + 1] = float(np.clip(
                d.ctrl[lo_off + 1] + inc, st["lo"][lo_off + 1],
                st["hi"][lo_off + 1]))

    q = d.qpos[env.obj_qpos_adr + 3: env.obj_qpos_adr + 7]
    roll = float(np.arctan2(2.0 * (q[0] * q[1] + q[2] * q[3]),
                            1.0 - 2.0 * (q[1] ** 2 + q[2] ** 2)))
    yaw = float(np.arctan2(2.0 * (q[0] * q[3] + q[1] * q[2]),
                           1.0 - 2.0 * (q[2] ** 2 + q[3] ** 2)))
    wz = float(d.qvel[env.obj_qvel_adr + 5])
    if st.get("z_servo", True):
        d_ctrl = -st["k_tilt"] * roll - st["k_yaw"] * wz
        d.ctrl[24] = float(np.clip(d.ctrl[24] + d_ctrl, st["lo"][24], st["hi"][24]))
        d.ctrl[2] = float(np.clip(d.ctrl[2] - d_ctrl, st["lo"][2], st["hi"][2]))
    st["roll"], st["yaw"] = roll, yaw


def cr_hi(env):
    return env.model.actuator_ctrlrange[:, 1]


def grip_floor(a: np.ndarray) -> np.ndarray:
    """Deployment toolkit (a): the press/curl channels cannot go below
    the placement curl - the policy modulates grip upward only, so a
    fading mean action can never relax the vise."""
    for idx in GRIP_FLOOR_ACTS:
        a[idx] = max(a[idx], 0.0)
    return a


def train(iters: int, steps_per_iter: int, eval_every: int, seed: int = 0,
          resume: str | None = None, rate_w: float = 2.0,
          act_rate_w: float = 0.5):
    import torch
    import torch.nn as nn

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    def mlp(sizes):
        layers = []
        for i in range(len(sizes) - 2):
            layers += [nn.Linear(sizes[i], sizes[i + 1]), nn.Tanh()]
        layers += [nn.Linear(sizes[-2], sizes[-1])]
        return nn.Sequential(*layers)

    actor = mlp([N_OBS, 256, 256, N_ACT]).to(dev)
    critic = mlp([N_OBS, 256, 256, 1]).to(dev)
    log_std = nn.Parameter(torch.full((N_ACT,), -1.0, device=dev))

    class Norm:
        def __init__(self):
            self.mean = np.zeros(N_OBS)
            self.var = np.ones(N_OBS)
            self.n = 1e-4

        def update(self, x):
            self.n += 1
            delta = x - self.mean
            self.mean += delta / self.n
            self.var += delta * (x - self.mean)

        def norm(self, x):
            return np.clip((x - self.mean) / np.sqrt(self.var + 1e-8),
                           -10, 10)

    norm = Norm()
    opt = torch.optim.Adam(list(actor.parameters()) + [log_std]
                           + list(critic.parameters()), lr=3e-4)
    if resume and (OUT_DIR / resume).exists():
        ck = torch.load(OUT_DIR / resume, map_location="cpu",
                        weights_only=False)
        actor.load_state_dict(ck["actor"])
        critic.load_state_dict(ck["critic"])
        with torch.no_grad():
            log_std.copy_(ck["log_std"].to(dev))
        norm.mean, norm.var, norm.n = (ck["norm"]["mean"],
                                       ck["norm"]["var"], ck["norm"]["n"])
        print(f"[v21] resumed from {resume} (iter {ck.get('iter')})")

    def to_t(x):
        return torch.tensor(x, dtype=torch.float32, device=dev)

    def act_fn(o):
        return actor_forward(actor, norm, o, dev)

    curve = []
    best = {"score": -np.inf, "path": None}
    consec_pass = 0
    early_stop = False
    env = make_closure_env(EP_LEN, seed)
    for it in range(iters):
        ep_frames, ep_returns, ep_peaks = [], [], []
        prev = None
        prev_a = None
        obs_raw, _ = env.reset(seed=int(rng.integers(1 << 31)))
        norm.update(obs_raw)
        done = False
        ep_rew = 0.0
        ep_peak = 0.0
        while len(ep_frames) < steps_per_iter:
            o_n = norm.norm(obs_raw)
            with torch.no_grad():
                mu = actor(to_t(o_n))
                std = log_std.exp().clamp(0.05, 0.05)  # continuation: sharpened
                dist = torch.distributions.Normal(mu, std)
                a_t = dist.sample()
                logp = dist.log_prob(a_t).sum(-1)
                v = critic(to_t(o_n)).squeeze(-1)
            a_np = a_t.cpu().numpy()
            a_np = grip_floor(a_np)
            a_np[2] = a_np[2] + BASE_Z
            a_np[24] = a_np[24] + BASE_Z
            obs_raw, r_env, term, trunc, info = env.step(
                np.clip(a_np, -1.0, 1.0))
            shaped, prev = closure_reward(env, info, prev, rate_w)
            if act_rate_w > 0.0 and prev_a is not None:
                da = np.abs(a_np[[0, 1, 22, 23]] - prev_a[[0, 1, 22, 23]])
                shaped -= act_rate_w * float(da.mean())
            prev_a = a_np.copy()
            r_shaped = (r_env + shaped) * W.REWARD_SCALE
            done = term or trunc
            ep_frames.append((o_n, a_np, r_shaped, float(done),
                              float(v), float(logp)))
            ep_rew += r_shaped
            ep_peak = max(ep_peak, info["lift"])
            if len(ep_frames) >= steps_per_iter:
                break
            if done:
                obs_raw, _ = env.reset(seed=int(rng.integers(1 << 31)))
                norm.update(obs_raw)
                done = False
                prev = None
                prev_a = None
                ep_rew = 0.0
        ep_returns.append(ep_rew)
        ep_peaks.append(ep_peak)

        obs_b, act_b, rew_b, done_b, val_b, logp_b = (
            np.array(x, dtype=np.float32) for x in zip(*ep_frames))
        with torch.no_grad():
            last_val = critic(to_t(norm.norm(obs_raw))).item()
        adv = np.zeros_like(rew_b)
        gae = 0.0
        for k in reversed(range(len(rew_b))):
            nxt = last_val if k == len(rew_b) - 1 else val_b[k + 1]
            delta = rew_b[k] + 0.99 * nxt * (1.0 - done_b[k]) - val_b[k]
            gae = delta + 0.99 * 0.95 * (1.0 - done_b[k]) * gae
            adv[k] = gae
        ret_b = adv + val_b
        adv_t = to_t((adv - adv.mean()) / (adv.std() + 1e-8))
        obs_t, act_t = to_t(obs_b), to_t(act_b)
        ret_t, val_old, logp_old = to_t(ret_b), to_t(val_b), to_t(logp_b)
        idx = np.arange(len(obs_b))
        stats = {"pi": 0.0, "v": 0.0, "ent": 0.0}
        for _ in range(4):
            rng.shuffle(idx)
            for s in range(0, len(idx), 256):
                mb = idx[s:s + 256]
                mu = actor(obs_t[mb])
                std = log_std.exp().clamp(0.05, 2.0)
                dist = torch.distributions.Normal(mu, std)
                logp = dist.log_prob(act_t[mb]).sum(-1)
                ratio = (logp - logp_old[mb]).exp()
                pi_loss = -torch.min(ratio * adv_t[mb],
                                     ratio.clamp(0.8, 1.2) * adv_t[mb]).mean()
                v = critic(obs_t[mb]).squeeze(-1)
                v_loss = 0.5 * ((v - ret_t[mb]) ** 2).mean()
                ent = dist.entropy().sum(-1).mean()
                loss = pi_loss + 0.5 * v_loss - 1e-3 * ent
                opt.zero_grad()
                loss.backward()
                opt.step()
                stats = {"pi": float(pi_loss), "v": float(v_loss),
                         "ent": float(ent)}
        curve.append({"iter": it, "return": float(np.mean(ep_returns)),
                      "peak_mm": float(np.mean(ep_peaks) * 1e3), **stats})
        print(f"[v21 {it:3d}] ret={np.mean(ep_returns):7.2f} "
              f"peak={np.mean(ep_peaks)*1e3:6.2f}mm pi={stats['pi']:+.4f} "
              f"v={stats['v']:.4f}")
        if (it + 1) % eval_every == 0 or it == iters - 1:
            m = evaluate(act_fn, seeds=(0, 1, 2))
            score = (min(m["window_frames_max"], 20) * 20.0
                     + min(m["peak_mm_max"], 20.0) / 20.0 * 300.0
                     - max(0.0, m["slip_mm_max"] - 6.0) * 50.0
                     - max(0.0, m["yaw_max"] - 8.0) * 40.0
                     + (800.0 if m["passed"] else 0.0))
            ck = {"actor": actor.state_dict(), "critic": critic.state_dict(),
                  "log_std": log_std.detach().cpu(),
                  "norm": {"mean": norm.mean, "var": norm.var, "n": norm.n},
                  "eval": m, "iter": it}
            torch.save(ck, OUT_DIR / f"v21_ppo_eval{it+1}.pt")
            print(f"[v21 {it:3d}] eval pass={m['passed']} "
                  f"window={m['window_frames_max']} "
                  f"peak={m['peak_mm_max']:.2f}mm score={score:.1f}")
            consec_pass = consec_pass + 1 if m["passed"] else 0
            if consec_pass >= 2:
                early_stop = True
            if score > best["score"]:
                best = {"score": score, "path": f"v21_ppo_eval{it+1}.pt"}
                torch.save(ck, OUT_DIR / "v21_ppo_best.pt")
    (OUT_DIR / "v21_ppo_curve.json").write_text(
        json.dumps({"curve": curve, "best": best, "early_stop": early_stop},
                   indent=1, default=float))
    print(f"v21 ppo done; best {best}; early_stop={early_stop}")


def table_support_n(env) -> float:
    """Summed |contact normal| on the box from the static support geoms."""
    m, d = env.model, env.data
    total = 0.0
    for i in range(d.ncon):
        c = d.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        if env.obj_geom not in (g1, g2):
            continue
        other = g2 if g1 == env.obj_geom else g1
        if env.geom_label[other] in env.support_geoms:
            f = np.zeros(6)
            mujoco.mj_contactForce(m, d, i, f)
            total += float(np.linalg.norm(f[:3]))
    return total


def actor_forward(actor, norm, obs, dev):
    import torch
    with torch.no_grad():
        mu = actor(torch.tensor(norm.norm(obs), dtype=torch.float32,
                                device=dev))
        return mu.cpu().numpy()


# ------------------------------------------------------------------ eval
def evaluate(act_fn, seeds=(0, 1, 2), export_replay: bool = False,
             evidence: bool = False, n_roll: int = 5, eval_std: float = 0.1):
    """Deployment toolkit (c): per seed, n_roll rollouts with sigma-0.1
    action noise (deployment = training distribution, killing the
    deterministic-mean collapse) and the gate runs on the MEAN frame
    series. The first rollout of seed seeds[0] provides the replay."""
    import torch
    env = make_closure_env(EP_LEN, seeds[0])
    episodes = []
    replay_dumped = False
    for ep, s in enumerate(seeds):
        rollouts = []
        for roll in range(n_roll):
            obs, _ = env.reset(seed=s * 10 + roll)
            env._z_rest = float(env.data.xpos[env.obj_body][2]) -                 env.box_half_z
            g = torch.Generator(device="cpu").manual_seed(s * 100 + roll)
            frames = []
            for k in range(env.max_episode_steps):
                mu_v = act_fn(obs)
                with torch.no_grad():
                    mu = torch.tensor(mu_v, dtype=torch.float32)
                std = eval_std
                noise = torch.normal(0.0, std, generator=g,
                                     size=mu.shape).numpy()
                a = np.clip(mu.numpy() + noise, -1.0, 1.0)
                a = grip_floor(a)
                a[2] = a[2] + BASE_Z
                a[24] = a[24] + BASE_Z
                obs, _r, term, trunc, info = env.step(np.clip(a, -1.0, 1.0))
                er, el = W.endface_contact_per_hand(env)
                bp = W.box_pose(env)
                n_r, n_l = env._pad_normals_obs()
                frames.append({"lift": info["lift"], "endface": (er, el),
                               "sup": table_support_n(env),
                               "xpos": bp["xpos"][:2].copy(),
                               "yaw": bp["yaw_deg"], "tilt": bp["tilt_deg"],
                               "clear": bp["bottom_clearance_m"],
                               "n_r": n_r, "n_l": n_l,
                               "qpos": env.data.qpos.copy(),
                               "obj7": np.concatenate([
                                   env.data.xpos[env.obj_body],
                                   env.data.xquat[env.obj_body]]).copy()})
                if term or trunc:
                    break
            rollouts.append(frames)
        n_f = min(len(fr) for fr in rollouts)
        mean_frames = []
        for k in range(n_f):
            kf = [fr[k] for fr in rollouts]
            mean_frames.append({
                "lift": float(np.mean([f["lift"] for f in kf])),
                "endface": (float(np.mean([f["endface"][0] for f in kf])),
                            float(np.mean([f["endface"][1] for f in kf]))),
                "sup": float(np.mean([f["sup"] for f in kf])),
                "xpos": np.mean([f["xpos"] for f in kf], axis=0),
                "yaw": float(np.mean([f["yaw"] for f in kf])),
                "tilt": float(np.mean([abs(f["tilt"]) for f in kf])),
                "clear": float(np.mean([f["clear"] for f in kf]))})
        m = W.eval_stream_against_gate(env, mean_frames)
        m["seed"] = s
        m["slip_mm"] = m["slip_m"] * 1e3
        episodes.append({"metrics": m, "frames": rollouts[0],
                         "mean_frames": mean_frames})
        print(f"[eval seed {s}] pass={m['pass']} window={m['window_frames']} "
              f"peak={m['peak_lift_m']*1e3:.2f}mm slip={m['slip_mm']:.1f}mm "
              f"yaw={m['yaw_abs_deg']:.1f} final={m['final_lift_m']*1e3:.2f}mm "
              f"({n_roll} rollouts x sigma {eval_std})")
    n_pass = sum(e["metrics"]["pass"] for e in episodes)
    out = {
        "episodes": [{"seed": e["metrics"]["seed"],
                      "pass": e["metrics"]["pass"],
                      "window_frames": e["metrics"]["window_frames"],
                      "peak_mm": e["metrics"]["peak_lift_m"] * 1e3,
                      "final_mm": e["metrics"]["final_lift_m"] * 1e3,
                      "slip_mm": e["metrics"]["slip_mm"],
                      "yaw_abs_deg": e["metrics"]["yaw_abs_deg"],
                      "endface_ratios": [e["metrics"]["endface_ratio_right"],
                                         e["metrics"]["endface_ratio_left"]],
                      "support_mean_n": e["metrics"]["support_mean_n"]}
                     for e in episodes],
        "passed": n_pass >= 2,
        "n_pass": n_pass,
        "window_frames_max": max(e["metrics"]["window_frames"]
                                 for e in episodes),
        "peak_mm_max": max(e["metrics"]["peak_lift_m"] * 1e3
                           for e in episodes),
        "slip_mm_max": max(e["metrics"]["slip_mm"] for e in episodes),
        "yaw_max": max(e["metrics"]["yaw_abs_deg"] for e in episodes),
    }
    if export_replay and n_pass >= 2:
        best_ep = max(episodes, key=lambda e: e["metrics"]["window_frames"])
        seg = best_ep["frames"]
        json.dump({"fps": W.CTRL_HZ,
                   "fps_scope": "control-rate export: one frame per 20 Hz "
                                "control frame of the v21 closure policy "
                                "episode; qpos 65-dof; window "
                                f"{best_ep['metrics']['window_frames']} frames",
                   "frames": [{"t": round(k / W.CTRL_HZ, 4),
                               "qpos": [float(v) for v in f["qpos"]],
                               "obj": [float(v) for v in f["obj7"]],
                               "control_frame": k} for k, f in enumerate(seg)]},
                  open(REPLAY_PATH, "w"), indent=1, default=float)
        _evidence(env, best_ep["frames"])
    return out


def _evidence(env, frames) -> None:
    import PIL.Image as Image
    m = env.model
    m.vis.global_.offwidth = 960
    m.vis.global_.offheight = 540
    renderer = mujoco.Renderer(m, height=480, width=854)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    presets = {"front": (105.0, -22.0, 0.80), "side": (-20.0, -30.0, 0.72),
               "close": (-20.0, -24.0, 0.46)}
    lifts = [f["lift"] for f in frames]
    w0 = int(np.argmax(np.array(lifts) >= 0.005)) if max(lifts) >= 0.005 else 0
    picks = [(max(0, w0 - 5), "front"),
             (min(len(frames) - 1, w0 + 10), "side"),
             (min(len(frames) - 1, w0 + 19), "close")]
    out = OUT_DIR / "evidence"
    out.mkdir(parents=True, exist_ok=True)
    for k, preset in picks:
        f = frames[k]
        env.data.qpos[:] = f["qpos"]
        oa = env.obj_qpos_adr
        env.data.qpos[oa:oa + 7] = f["obj7"]
        mujoco.mj_forward(m, env.data)
        cam.lookat[:] = [0.0, 0.0, 0.74]
        cam.azimuth, cam.elevation, cam.distance = presets[preset]
        renderer.update_scene(env.data, camera=cam)
        Image.fromarray(renderer.render()).save(
            out / f"v21_f{k:03d}_lift{f['lift']*1e3:.1f}mm_{preset}.png")
    renderer.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("capture", "train", "eval"))
    ap.add_argument("--iters", type=int, default=48)
    ap.add_argument("--steps", type=int, default=1536)
    ap.add_argument("--eval-every", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt", default="v21_ppo_best.pt")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--rate-w", type=float, default=2.0)
    ap.add_argument("--act-rate-w", type=float, default=0.5)
    ap.add_argument("--smooth-w", type=float, default=1.0)
    ap.add_argument("--curriculum", default="40,80,160")
    ap.add_argument("--round-tag", default="r1mlp")
    ap.add_argument("--lift-cap-m", type=float, default=0.050)
    ap.add_argument("--pen-cap", type=float, default=3.0)
    ap.add_argument("--anneal", action="store_true")
    args = ap.parse_args()
    if args.cmd == "capture":
        print(json.dumps(capture_engage(), indent=1, default=float))
    elif args.cmd == "train":
        train(args.iters, args.steps, args.eval_every, args.seed,
              resume=args.resume, rate_w=args.rate_w,
              act_rate_w=args.act_rate_w)
    elif args.cmd == "eval":
        import torch
        dev = "mps" if torch.backends.mps.is_available() else "cpu"
        ck = torch.load(OUT_DIR / args.ckpt, map_location="cpu",
                        weights_only=False)
        import torch.nn as nn

        def mlp(sizes):
            layers = []
            for i in range(len(sizes) - 2):
                layers += [nn.Linear(sizes[i], sizes[i + 1]), nn.Tanh()]
            layers += [nn.Linear(sizes[-2], sizes[-1])]
            return nn.Sequential(*layers)

        actor = mlp([N_OBS, 256, 256, N_ACT]).to(dev)
        actor.load_state_dict(ck["actor"])
        norm_mean = ck["norm"]["mean"]
        norm_var = ck["norm"]["var"]

        wrap = _WrapNorm(norm_mean, norm_var)

        def act_fn(o):
            return actor_forward(actor, wrap, o, dev)
        m = evaluate(act_fn, seeds=(0, 1, 2), export_replay=True,
                     evidence=True)
        print(json.dumps({k: v for k, v in m.items() if k != "episodes"},
                         indent=1, default=float))


class _WrapNorm:
    def __init__(self, mean, var):
        self.mean = mean
        self.var = var
        self.n = 1e4

    def norm(self, x):
        return np.clip((x - self.mean) / np.sqrt(self.var + 1e-8), -10, 10)


if __name__ == "__main__":
    main()
