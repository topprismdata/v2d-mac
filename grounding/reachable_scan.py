"""Reachable-class clean-lift scan (smoke4 round p0(4)).

The K1 dual-wall pinch is class-B infeasible (w35d, ~19M poses). The acceptance
gates never required dual-wall, so this scan searches ONLY the pose families
w35d proved kinematically reachable under the strict wall-contact criterion
(runs/w35/w35d_diag.json) for a COMPLETE scripted trajectory that honestly
passes the frozen smoke4 acceptance gates:

  R1  ff+th double pad on the SAME (-x) wall  (xyfull scan, 0.31 mm, in-range)
  R2  single long finger on -x wall + thumb   (xyfull scan variants)
  R3  long fingers on +x wall below the rim   (pinch-scan binding family ry~1.2)

Wrist control is the world-frame P-loop of grounding/probe_clean_lift.py
(polarity/sign fix); every trial is judged with
tests/test_smoke4_accept._analyse_episode - the exact code the frozen test
runs - so a scan pass is a test pass by construction.

    python grounding/reachable_scan.py --pilot            # mechanics check
    python grounding/reachable_scan.py                    # full structured grid
    python grounding/reachable_scan.py --record BEST_IDX  # D4 npz for BC

Outputs: runs/p0/smoke4_mujoco_ppo/reachable_scan.json (+ reachable_demo_set.npz
when --record finds a gate-passing trajectory).
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

RUN_DIR = REPO / "runs/p0/smoke4_mujoco_ppo"
OUT_JSON = RUN_DIR / "reachable_scan.json"
OUT_NPZ = RUN_DIR / "reachable_demo_set.npz"
MJCF_COMMIT = "4ee6f1d"

N_WRIST = 6
WRIST_ROT_STEP = 0.10           # rad per unit action (env _apply_action)
WORLD_GAIN = np.array([0.012, 0.012, 0.00436])   # measured, probe_clean_lift
WRIST_Z_BASE = 0.92             # hand_root world z = WRIST_Z_BASE + ctrl_z

# episode budget must fit the formal eval episode (250 steps) so a PPO policy
# can physically reproduce the recorded timing
N_TRANSIT = 50
N_DESC = 50
N_CLOSE = 70
EPISODE_LEN = 260


# --------------------------------------------------------------------- configs
def close_amp(cl: float, ct: float, t0: float = 0.0) -> np.ndarray:
    """16 finger action units, cmd16 convention: long fingers tied (j0=cl*0.3),
    thumb tied (thj0=t0). a=-1 extended, a=+1 curled."""
    a = np.empty(16)
    for k in range(3):
        a[4 * k: 4 * k + 4] = cl
        a[4 * k] = cl * 0.3
    a[12:16] = ct
    a[12] = t0
    return a


def _seed_world(base, rot):
    """Scan-pose (wrist joint qpos) -> world hand_root target.
    The w35d scans wrote qpos directly on the wrist slides with the box at the
    model qpos0 rest pose; with domain_rand=False the env reproduces that rest
    pose exactly, so world = (qx, qy, WRIST_Z_BASE + qz)."""
    return [float(base[0]), float(base[1]), WRIST_Z_BASE + float(base[2])], [float(v) for v in rot]


# seed families from runs/w35 evidence (all joint targets within ctrlrange),
# refined by runs/w35/rscan_mechanism_probe.py: real contact happens near the
# static families but a straight full-curl jump knocks the box away, so the
# grid is z/y/x placement x curl magnitude x RAMPED close x settle x lift.
_SEED_DEFS = {
    # ff+th pads on/near -x wall (xyfull family, rot 1.5/0.9)
    "R1": {"base": (-0.12, 0.06, -0.003), "rot": (1.5, 0.9, 0.0),
           "y": (0.06, 0.12, 0.18), "z": (-0.023, -0.003, 0.017),
           "cl": (0.85, 1.0), "ct": (-0.5, -1.0), "t0": (0.0,)},
    # long fingers +x wall below rim (pinch-scan binding family, rot 1.2/1.2)
    "R3": {"base": (0.06, 0.06, 0.0), "rot": (1.2, 1.2, 0.0),
           "x": (0.02, 0.06, 0.10), "z": (-0.02, 0.0, 0.03),
           "cl": (1.0,), "ct": (0.0, 0.5, 1.0), "t0": (0.0, 0.5)},
}


def build_grid(pilot: bool = False) -> list[dict]:
    """Structured grid: placement x curl x close profile x lift profile."""
    cfgs = []
    for name, sd in _SEED_DEFS.items():
        xs = sd.get("x", (sd["base"][0],))
        ys = sd.get("y", (sd["base"][1],))
        zs = sd.get("z", (sd["base"][2],)) if not pilot else (sd["base"][2],)
        cl_opts = sd["cl"] if not pilot else (sd["cl"][-1],)
        ct_opts = sd["ct"] if not pilot else (sd["ct"][-1],)
        t0_opts = sd["t0"] if not pilot else (0.0,)
        lifts = [(0.12, 0.008)] if pilot else [(0.12, 0.008), (0.16, 0.004)]
        for x in xs:
            for y in ys:
                for z in zs:
                    for cl in cl_opts:
                        for ct in ct_opts:
                            for t0 in t0_opts:
                                for lift_z, lift_rate in lifts:
                                    w, r = _seed_world(sd["base"], sd["rot"])
                                    w[0], w[1], w[2] = (WRIST_Z_XYZ(sd, x, y, z))
                                    cfgs.append({
                                        "family": name,
                                        "wrist_world": [w[0], w[1], w[2]],
                                        "rot": r,
                                        "close": close_amp(cl, ct, t0).tolist(),
                                        "close_params": [cl, ct, t0],
                                        "sequence": "descend_then_close",
                                        "close_ramp": True,
                                        "settle": 25,
                                        "lift_z": lift_z,
                                        "lift_rate": lift_rate,
                                    })
    return cfgs


def WRIST_Z_XYZ(sd, x, y, z):
    return [float(x), float(y), WRIST_Z_BASE + float(z)]


# ------------------------------------------------------------------- rollout
_ENV: dict = {}


def _init_worker():
    from grounding.envs.lift_hold_env import LiftHoldEnv
    _ENV["env"] = LiftHoldEnv(episode_len=EPISODE_LEN, domain_rand=False)


def _action(env, wrist_tgt, rot_tgt, fingers):
    """World-frame wrist P-loop (probe_clean_lift polarity fix) + absolute
    finger setpoints in action units."""
    err = np.asarray(wrist_tgt) - env.data.xpos[env.hand_root]
    a = np.zeros(22)
    a[0:3] = np.clip(err / WORLD_GAIN, -1.0, 1.0)
    a[3:6] = np.clip((np.asarray(rot_tgt) - env.data.ctrl[3:6]) / WRIST_ROT_STEP,
                     -1.0, 1.0)
    a[N_WRIST:] = np.clip(np.asarray(fingers), -1.0, 1.0)
    return a


def run_traj(env, cfg: dict, record: bool = False):
    """One scripted attempt. Deterministic: reset(seed=0), pure function of cfg."""
    env.reset(seed=0)
    w = cfg["wrist_world"]
    rot = cfg["rot"]
    close = np.asarray(cfg["close"])
    open_a = np.full(16, 0.6)
    seq = cfg.get("sequence", "descend_then_close")
    lift_z, lift_rate = cfg["lift_z"], cfg["lift_rate"]
    ramp = bool(cfg.get("close_ramp", False))
    settle = int(cfg.get("settle", 0))
    n_transit = int(cfg.get("n_transit", N_TRANSIT))
    n_desc = int(cfg.get("n_desc", N_DESC))
    n_close = int(cfg.get("n_close", N_CLOSE))
    obs_all, act_all = [], []

    def step_to(tgt, fingers_cmd):
        a = _action(env, tgt, rot, fingers_cmd)
        o, *_ = env.step(a)
        if record:
            obs_all.append(o.copy())
            act_all.append(a.astype(np.float32))

    # safe transit: lift the hand first, move xy, so the palm never sweeps the box
    high = [w[0], w[1], w[2] + 0.10]
    for _ in range(n_transit):
        step_to(high, open_a)

    if seq == "close_then_descend":
        for _ in range(n_close):
            step_to(high, close)
        for _ in range(n_desc):
            step_to(w, close)
    elif seq == "simultaneous":
        for i in range(max(n_desc, n_close)):
            f = -1.0 + (close[0] + 1.0) * min(1.0, (i + 1) / max(n_desc, n_close))
            step_to(w, np.full(16, f))
        for _ in range(n_close):
            step_to(w, close)
    else:  # descend_then_close
        for _ in range(n_desc):
            step_to(w, open_a)
        if ramp:
            # gradual curl: the pad bites instead of impacting (a straight jump
            # to full curl knocked the box off every probed pose)
            for i in range(n_close):
                f = open_a + (close - open_a) * min(1.0, (i + 1) / (n_close * 0.7))
                step_to(w, f)
            for _ in range(n_close - int(n_close * 0.7)):
                step_to(w, close)
        else:
            for _ in range(n_close):
                step_to(w, close)
    for _ in range(settle):
        step_to(w, close)

    # pull straight up at a rate-limited world z target, fingers held
    hold = int(cfg.get("lift_hold", 60))
    n_lift = int(math.ceil(lift_z / lift_rate)) + hold
    cur = float(env.data.xpos[env.hand_root][2])
    for i in range(n_lift):
        goal = min(w[2] + lift_z, cur + lift_rate)
        step_to([w[0], w[1], goal], close)
        cur = float(env.data.xpos[env.hand_root][2])

    extra = None
    if record:
        extra = (np.asarray(obs_all, dtype=np.float32),
                 np.asarray(act_all, dtype=np.float32))
    return extra


# ------------------------------------------------------------- gate evaluation
def gate_eval(env, trace=None) -> dict:
    """Frozen acceptance gate, verbatim machinery from tests/test_smoke4_accept."""
    sys.path.insert(0, str(REPO / "tests"))
    from test_smoke4_accept import _analyse_episode
    tr = (trace if trace is not None else env.trace).copy()
    tr.setdefault("dt", float(env.dt))
    ev = _analyse_episode({"trace": tr})
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
    return ev


# ------------------------------------------- strict wall/top classification
def _tip_pads(env) -> dict[str, int]:
    import mujoco
    m = env.model
    pads = {}
    for pref in ("ff", "mf", "rf", "th"):
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{pref}_tip")
        gids = [g for g in range(m.ngeom)
                if int(m.geom_bodyid[g]) == bid
                and int(m.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_CAPSULE)]
        if gids:
            pads[pref] = gids[0]
    return pads


def _penetration_tol(env) -> float:
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "baf_rscan", REPO / "grounding/assets/build_allegro_floating.py")
    baf = importlib.util.module_from_spec(spec)
    sys.modules["baf_rscan"] = baf
    spec.loader.exec_module(baf)
    return float(baf.penetration_tolerance(mjcf_path=REPO / "grounding/assets/allegro_floating.xml"))


def strict_contacts(env, pads: dict[str, int], tol: float) -> dict:
    """w35d strict criterion: a pad touches a box face iff mj_geomDistance's
    nearest point on the BOX lies inside that face rectangle. Returns per-pad
    face/gap plus aggregate class tags."""
    import mujoco
    m, d = env.model, env.data
    bp = d.xpos[env.obj_body]
    half = m.geom_size[env.obj_geom]
    top = float(bp[2] + half[2])
    ft = np.zeros(6)
    out = {}
    for f, g in pads.items():
        dist = float(mujoco.mj_geomDistance(m, d, g, env.obj_geom, 0.06, ft))
        pb = ft[3:6]
        rel = pb - bp
        on_x = abs(abs(rel[0]) - half[0]) <= tol and abs(rel[1]) <= half[1] + tol
        on_top = (abs(rel[2] - half[2]) <= tol and abs(rel[0]) <= half[0] + tol
                  and abs(rel[1]) <= half[1] + tol)
        on_endy = (abs(abs(rel[1]) - half[1]) <= tol and abs(rel[0]) <= half[0] + tol
                   and rel[2] <= half[2] + tol)
        below = float(d.geom_xpos[g][2]) < top
        face = "none"
        if on_top:
            face = "top"
        elif on_x:
            face = "x_pos" if rel[0] > 0 else "x_neg"
        elif on_endy:
            face = "y_pos" if rel[1] > 0 else "y_neg"
        touching = dist <= tol
        out[f] = {"gap_m": round(dist, 6), "face": face if touching or dist <= 0.008 else "none",
                  "below_top": bool(below), "strict_touch": bool(touching)}
    # class tags
    x_faces = [f for f, st in out.items() if st["strict_touch"] and st["face"].startswith("x")]
    sides = {st["face"] for st in out.values() if st["strict_touch"]}
    top_touch = any(st["strict_touch"] and st["face"] == "top" for st in out.values())
    cls = "none"
    if len(x_faces) >= 1:
        both = {"x_pos", "x_neg"} <= sides
        cls = "dual_wall" if both else ("single_wall+top" if top_touch else "single_wall")
    elif top_touch:
        cls = "top_only"
    return {"pads": out, "class": cls, "n_strict_x": len(x_faces), "top_touch": top_touch}


def contact_load_decomposition(env, actions: np.ndarray, peak_i: int) -> dict:
    """Replay recorded actions and split finger contact force on the box by
    face over the lift window (top vs side walls vs end faces)."""
    import mujoco
    m, d = env.model, env.data
    env.reset(seed=0)
    half = m.geom_size[env.obj_geom]
    faces = ("top", "x_pos", "x_neg", "y_pos", "y_neg", "other")
    acc = {f: 0.0 for f in faces}
    for t, a in enumerate(actions):
        env.step(a)
        if t > peak_i:
            break
        bp = d.xpos[env.obj_body].copy()
        bxm = d.xmat[env.obj_body].reshape(3, 3).copy()
        for i in range(d.ncon):
            con = d.contact[i]
            g1, g2 = int(con.geom1), int(con.geom2)
            if env.obj_geom not in (g1, g2):
                continue
            other = g2 if g1 == env.obj_geom else g1
            if other not in env.finger_geom_ids:
                continue
            f6 = np.zeros(6)
            mujoco.mj_contactForce(m, d, i, f6)
            mag = float(np.linalg.norm(f6[:3]))
            if mag <= 0.0:
                continue
            rel = bxm.T @ (np.asarray(con.pos) - bp)
            if abs(rel[2] - half[2]) <= half[2] * 0.5 and abs(rel[0]) <= half[0] + 1e-9 \
                    and abs(rel[1]) <= half[1] + 1e-9 and rel[2] > 0:
                face = "top"
            elif abs(abs(rel[0]) - half[0]) <= half[0] * 0.5:
                face = "x_pos" if rel[0] > 0 else "x_neg"
            elif abs(abs(rel[1]) - half[1]) <= half[1] * 0.5:
                face = "y_pos" if rel[1] > 0 else "y_neg"
            else:
                face = "other"
            acc[face] += mag
    tot = sum(acc.values())
    return {f: (acc[f] / tot if tot > 0 else 0.0) for f in faces} | {"total_N": round(tot, 3)}


# ---------------------------------------------------------------- scan driver
def score_one(cfg: dict) -> dict:
    from grounding.envs.lift_hold_env import LiftHoldEnv
    env = LiftHoldEnv(episode_len=EPISODE_LEN, domain_rand=False)
    run_traj(env, cfg)
    ev = gate_eval(env)
    pads = _tip_pads(env)
    tol = _penetration_tol(env)
    row = {"cfg": cfg, "gate": {k: v for k, v in ev.items() if k != "finger_pairs"},
           "finger_pairs": ev["finger_pairs"], "strict": None}
    if ev["peak_lift_m"] > 0.02 or ev["finger_force_mean_N"] > 2.0:
        # interesting: replay with recording and classify the grasp at the
        # peak-force step (the actual hold state, not the rollout end)
        obs, act = run_traj(env, cfg, record=True)
        env.reset(seed=0)
        fmax_i = int(np.argmax(env.trace["finger_force"])) if env.trace["finger_force"] else 0
        env.reset(seed=0)
        for a in act[: fmax_i + 1]:
            env.step(a)
        row["strict"] = strict_contacts(env, pads, tol)
        row["strict_at_step"] = fmax_i
        row["force_at_step"] = float(np.max(env.trace["finger_force"]))
    env.close()
    return row


def _work(cfg):
    return score_one(cfg)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="cap grid size (0 = all)")
    ap.add_argument("--record", type=int, default=-1,
                    help="index into the scan rows: record D4 npz for that config")
    args = ap.parse_args()

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    cfgs = build_grid(pilot=args.pilot)
    if args.limit:
        cfgs = cfgs[: args.limit]

    if args.record >= 0:
        data = json.loads(OUT_JSON.read_text())
        rows = data["rows"]
        passing = [i for i, r in enumerate(rows) if r["gate"]["gate_pass"]]
        idx = args.record if args.record in passing else passing[0]
        cfg = rows[idx]["cfg"]
        from grounding.envs.lift_hold_env import LiftHoldEnv
        env = LiftHoldEnv(episode_len=EPISODE_LEN, domain_rand=False)
        obs, act = run_traj(env, cfg, record=True)
        ev = gate_eval(env)
        assert ev["gate_pass"], "recorded trajectory does not pass the frozen gate"
        decomp = contact_load_decomposition(env, act, ev["peak_step"])
        lens = np.array([len(obs)], dtype=np.int32)
        cl, ct, t0 = cfg["close_params"]
        # D4 pose row: wx wy wz rx ry rz a_long a_thumb (test_p0_plumbing P1);
        # the full generator config rides along as JSON for the replay test.
        pose = np.asarray([[cfg["wrist_world"][0], cfg["wrist_world"][1],
                            cfg["wrist_world"][2], cfg["rot"][0], cfg["rot"][1],
                            cfg["rot"][2], cl, ct]], dtype=np.float32)
        np.savez_compressed(
            OUT_NPZ,
            bc_obs=obs.astype(np.float32), bc_act=act.astype(np.float32),
            bc_pose=pose, bc_trajectory_len=lens,
            mjcf_commit=np.array(MJCF_COMMIT),
            selection=np.array("reachable_class_clean"),
            cfg_json=np.array(json.dumps(cfg)),
            gate_metrics=np.array(json.dumps(
                {k: v for k, v in ev.items() if k != "finger_pairs"})),
            load_decomposition=np.array(json.dumps(decomp)),
        )
        print(json.dumps({"saved": str(OUT_NPZ), "frames": int(lens[0]),
                          "peak_lift_m": ev["peak_lift_m"],
                          "load_decomposition": decomp}, indent=2))
        return

    if args.pilot:
        rows = [_work(c) for c in cfgs]
    else:
        with mp.Pool(args.workers, initializer=_init_worker) as pool:
            rows = pool.map(_work, cfgs, chunksize=1)

    passing = [r for r in rows if r["gate"]["gate_pass"]]
    rows.sort(key=lambda r: (not r["gate"]["gate_pass"], -r["gate"]["peak_lift_m"]))
    by_family: dict[str, dict] = {}
    for r in rows:
        f = r["cfg"]["family"]
        s = by_family.setdefault(f, {"n": 0, "n_pass": 0, "best_lift": 0.0})
        s["n"] += 1
        s["n_pass"] += int(r["gate"]["gate_pass"])
        s["best_lift"] = max(s["best_lift"], r["gate"]["peak_lift_m"])

    OUT_JSON.write_text(json.dumps({
        "mjcf_commit": MJCF_COMMIT,
        "episode_len": EPISODE_LEN,
        "n_trials": len(rows),
        "n_pass": len(passing),
        "by_family": by_family,
        "rows": rows,
    }, indent=2))
    print(f"pass {len(passing)}/{len(rows)} -> {OUT_JSON}")
    for f, s in by_family.items():
        print(f"  {f}: {s['n_pass']}/{s['n']} pass, best lift {s['best_lift']*1000:.1f}mm")


if __name__ == "__main__":
    main()
