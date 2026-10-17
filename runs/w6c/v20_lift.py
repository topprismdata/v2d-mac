"""v20 full-stack re-run: follower pads + bilateral press + force lift.

Name-mapped checkpoint adapter (v14 51-dof -> v20 65-dof), bilateral-first
press, force-source hold, climb with N-hold, gate + morphology + mu_eff.
"""
import hashlib
import itertools
import json
import pathlib
import sys

import mujoco
import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import grounding.train_w6c as W  # noqa: E402

FS = 25


def table_support_n(env) -> float:
    """Summed |contact normal| on the box from the static support geoms (N).

    Same partition as the env's _read_contacts support clause - the frozen
    gate's support clause (<= 0.20 N inside the window) is unfalsifiable
    without it (a corner wedge reads tens of N here)."""
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


def main() -> None:
    env = W.make_start_env(episode_len=400, domain_rand=False, noise=False)
    m = env.model
    base = json.loads(open(REPO / "runs/w6c/teacher.json").read())["best_params"]
    env.reset(seed=W.SEED)
    bid = env.obj_body

    def start_snap(seed=None, noise=False):
        """Sanctioned trigger start state: make_trigger_env's name-mapped
        checkpoint adapter + the env's own settle (+ the env's start noise
        when noise=True, drawn from seed). Replaces the hand-rolled v14
        adapter, which zip-mapped the 7-wide object_free joint against ONE
        name slot and shifted every later v14 value six slots (object
        quaternion into the right wrist, wrist values into the fingers) -
        the wedge signature (settle 0.753) in the blocked-session grid."""
        e = W.make_trigger_env(episode_len=400, noise=noise)
        e.reset(seed=W.SEED if seed is None else seed)
        s = mujoco.MjData(m)
        mujoco.mj_copyData(s, m, e.data)
        print(f"trigger settle (seed={seed}, noise={noise}): obj z",
              round(float(s.xpos[bid][2]), 4))
        return s

    d0_base = start_snap()

    jn = [[f"{f}j{i}{s}" for f in ("ff", "mf") for i in (1, 2)]
          for s in ("", "_left")]
    JQ = [[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)]
           for j in jn[si]] for si in range(2)]
    AK = [[m.actuator_gainprm[mujoco.mj_name2id(
        m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{f}a{i}{s}")][0]
        for f in ("ff", "mf") for i in (1, 2)] for s in ("", "_left")]
    AC = [[m.actuator_ctrlrange[mujoco.mj_name2id(
        m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{f}a{i}{s}")].tolist()
        for f in ("ff", "mf") for i in (1, 2)] for s in ("", "_left")]
    padg = [[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM,
                               f"{f}_tip{s}_collision")
             for f in ("ff", "mf")] for s in ("", "_left")]
    lo_c = m.actuator_ctrlrange[:, 0]
    hi_c = m.actuator_ctrlrange[:, 1]
    press_idx = [[7, 8, 11, 12], [29, 30, 33, 34]]

    def run(timing, ff_r, mf_r, dz_mm, nr_star=4.0, nl_star=4.0, rate=1.0,
            mode_pin=None, capture=None, d0=None):
        """mode_pin: None = the auto force/pos hybrid per side; else a pinned
        ("force"|"pos", "force"|"pos") pair with no auto-flip. capture: list
        to receive per-control-frame qpos snapshots of the climb segment.
        d0: start data (defaults to the settled noiseless d0_base)."""
        d = env.data
        mujoco.mj_copyData(d, m, d0_base if d0 is None else d0)
        env._z0 = float(d.xpos[bid][2])
        env._z_rest = env._z0 - env.box_half_z
        env._z_peak = env._z0
        env._grasp_bias_sides = (0.0, 0.0)
        xy0 = d.xpos[bid][:2].copy()
        tgt = d.ctrl.copy()
        for h in range(2):
            for j, depth in ((0, ff_r), (1, ff_r), (2, mf_r), (3, mf_r)):
                idx = press_idx[h][j]
                tgt[idx] = d.ctrl[idx] + depth
        start = {i_: d.ctrl[i_] for i_ in press_idx[0] + press_idx[1]}
        for _ in range(20):
            d.ctrl[2] += dz_mm * 1e-3 / 20
            d.ctrl[24] += dz_mm * 1e-3 / 20
            for _ in range(FS):
                mujoco.mj_step(m, d)
        xy_press0 = d.xpos[bid][:2].copy()
        for k in range(20 + timing):
            fl = min(1.0, (k + 1) / 20.0)
            fr = min(1.0, max(0.0, (k + 1 - timing) / 20.0))
            for h, frac in ((0, fl), (1, fr)):
                for i_, idx in enumerate(press_idx[h]):
                    s = start[press_idx[h][i_]]
                    d.ctrl[idx] = s + (tgt[idx] - s) * frac
            for _ in range(FS):
                mujoco.mj_step(m, d)
        press_drift = float(np.linalg.norm(d.xpos[bid][:2] - xy_press0)) * 1e3
        anchor = d.ctrl.copy()
        mode = ["force", "force"] if mode_pin is None else list(mode_pin)
        n_star = (nr_star, nl_star)
        stable = 0
        lifting = False
        z_ramped = 0.0
        rows = []
        for i in range(80):
            n_r, n_l = W.pad_normal_forces(env)
            if mode_pin is None:
                for si, (nn, nstar) in enumerate(((n_r, n_star[0]),
                                                  (n_l, n_star[1]))):
                    if nn <= 0.05 and mode[si] == "force":
                        mode[si] = "pos"
                    elif nn > 0.1 and mode[si] == "pos":
                        mode[si] = "force"
            ok = (n_l >= n_star[1] * 0.9 and mode[1] == "force") and \
                 (n_r >= n_star[0] * 0.9 and mode[0] == "force")
            stable = stable + 1 if ok else 0
            if not lifting and stable >= 5 and \
                    1.8 * (n_r + n_l) >= 1.2 * 1.4715:
                lifting = True
            if lifting and z_ramped < float(base["z_lift"]):
                # 0.1 mm/frame at rate=1.0 (the l2e twin's climb speed
                # 0.002*1e-3/0.02); the pre-fix line dropped the 0.002 scale
                # and jumped the wrist target 50 mm per frame, shearing the
                # pads off the faces before any friction chain could form
                d.ctrl[2] += rate * 0.002 * 1e-3 / 0.02
                d.ctrl[24] += rate * 0.002 * 1e-3 / 0.02
                z_ramped += rate * 0.002 * 1e-3 / 0.02
            for si, nstar in enumerate(n_star):
                # the four demand joints are ffj1,ffj2,mfj1,mfj2 whose
                # actuators sit at press_idx[si] = [7,8,11,12] per hand - the
                # pre-fix si*22+6+j write hit ffa0..ffa3 instead, so the mf
                # curl never got a force demand and ffj0 (spread) was driven
                # against the press: the pinch stayed at the settle N (~1.5)
                # instead of the r7e +0.2 rad demand level (~3.6-4.3 N)
                for j in range(4):
                    idx = press_idx[si][j]
                    if mode[si] == "force":
                        # constant-torque force source expressed as a per-
                        # joint angular demand. Calibration: on v20.1 the
                        # +0.20 rad curl demand holds N 3.62/4.33 N (r7e
                        # re-check) - the only pinch level whose friction
                        # (N x mu_eff 0.34 ~ 1.2 N/side) exceeds the half
                        # box weight 0.736 N. n_star=4.0 -> +0.20 rad; the
                        # pre-fix tau/kp form (0.034 N m at kp 0.84-1.46)
                        # demanded 0.02-0.04 rad and never left the settle
                        # pinch (~1.5 N), whatever actuator it wrote.
                        # MEASURED (v20p1 grid): the demand does NOT convert
                        # on this chain - N stays 1.66 +-0.03 (the rf-distal
                        # kiss) for press depth -0.20..+0.80, descend
                        # -24..-55 mm and demand +-0.20..0.25 rad; the ff/mf
                        # pad tip path is tangential/away from the face at
                        # the trigger height (dx/dq +134 mm/rad) and 30-50
                        # mm short in x at the seed height. The r7e point
                        # came from a hand-built mid-face pad pose the
                        # checkpoint chain does not reach.
                        demand = min(nstar * 0.05, 0.30)
                        q = d.qpos[JQ[si][j]]
                        d.ctrl[idx] = float(np.clip(q + demand,
                                                    lo_c[idx], hi_c[idx]))
                    else:
                        d.ctrl[idx] = anchor[idx]
            for _ in range(FS):
                mujoco.mj_step(m, d)
            er, el = W.endface_contact_per_hand(env)
            bp = W.box_pose(env)
            n_r, n_l = W.pad_normal_forces(env)
            pad_z = float(np.mean([d.geom_xpos[g][2] for pair in padg
                                   for g in pair]))
            rows.append({"lift": float(d.xpos[bid][2] - env._z0),
                         "endface": (er, el), "sup": table_support_n(env),
                         "xpos": bp["xpos"][:2].copy(),
                         "yaw": bp["yaw_deg"], "tilt": bp["tilt_deg"],
                         "clear": bp["bottom_clearance_m"],
                         "n_r": n_r, "n_l": n_l, "pad_z": pad_z})
            if capture is not None:
                capture.append({"k": i, "qpos": d.qpos.copy(),
                                "obj": np.concatenate(
                                    [d.xpos[bid], d.xquat[bid]]).copy()})
        mres = W.eval_stream_against_gate(env, [
            {"lift": f["lift"], "endface": f["endface"], "sup": f["sup"],
             "xpos": f["xpos"], "yaw": f["yaw"], "tilt": f["tilt"],
             "clear": f["clear"]} for f in rows])
        mres["tilt_abs_deg"] = max((f["tilt"] for f in rows), default=0)
        mres["press_drift_mm"] = press_drift
        # morphology audit: window-relative
        lifts = np.array([r["clear"] for r in rows])
        w0 = int(np.argmax(lifts >= 0.005)) if (lifts >= 0.005).any() else -1
        win = rows[w0:w0 + 20] if w0 >= 0 else []
        audit = {"window": len(win),
                 "support_max": max((f["sup"] for f in win), default=None),
                 "endface_break": any(not (f["endface"][0] and f["endface"][1])
                                      for f in win) if win else None,
                 "both_N_frames": sum(1 for f in win
                                      if f["n_r"] > 0.05 and f["n_l"] > 0.05),
                 "pad_climb_mm": (max(f["pad_z"] for f in win) -
                                  min(f["pad_z"] for f in win)) if win else None,
                 "box_rise_mm": (max(f["clear"] for f in win) * 1e3) if win else None,
                 "slip_mm": (max(np.linalg.norm(np.asarray(f["xpos"]) -
                                                np.asarray(win[0]["xpos"]))
                                 for f in win) * 1e3) if win else None,
                 "yaw_max": max((f["yaw"] for f in win), default=None)}
        if win and audit["box_rise_mm"]:
            ratio = (audit["pad_climb_mm"] or 0.0) / max(audit["box_rise_mm"], 1e-6)
            both = audit["both_N_frames"] >= len(win) * 0.8
            audit["morphology"] = ("pinch-carry" if both and ratio < 3.0
                                   else "hook/wedge-dominant")
        mres["audit"] = audit
        return mres, rows

    def score_of(mres):
        return (1000.0 if mres["pass"] else
                min(mres["window_frames"], 20) * 12 +
                min(mres["peak_lift_m"], 0.005) * 1500)

    xml_sha = hashlib.sha256(
        (REPO / "grounding/assets/allegro_floating.xml").read_bytes()
    ).hexdigest()

    combos = []
    passing = []   # (sort_key, combo_rec, rows, capture)

    def do_run(leg, timing, ff_r, mf_r, dz, mode_pin=None, d0=None):
        cap: list = []
        mres, rows = run(timing, ff_r, mf_r, dz, mode_pin=mode_pin,
                         capture=cap, d0=d0)
        rec = {"leg": leg, "timing": timing, "ff_r": ff_r, "mf_r": mf_r,
               "dz": dz, "mode_pin": (None if mode_pin is None
                                      else list(mode_pin)),
               "window": mres["window_frames"],
               "window_start": mres["window_start"],
               "peak_mm": mres["peak_lift_m"] * 1e3,
               "final_mm": mres["final_lift_m"] * 1e3,
               "support_n": mres["support_mean_n"],
               "slip_mm": mres["slip_m"] * 1e3,
               "yaw_deg": mres["yaw_abs_deg"],
               "tilt_deg": mres["tilt_abs_deg_window"],
               "endface_ratios": [mres["endface_ratio_right"],
                                  mres["endface_ratio_left"]],
               "press_drift_mm": mres["press_drift_mm"],
               "audit": mres["audit"],
               "pass": bool(mres["pass"]),
               "score": round(score_of(mres), 1)}
        combos.append(rec)
        if mres["pass"]:
            passing.append(((-mres["peak_lift_m"], -mres["window_frames"]),
                            rec, rows, cap))
        return rec, mres, rows, cap

    # ---- leg 1: the 24-combo bilateral-first press grid (auto hybrid) ----
    for timing, ff_r, mf_r, dz in itertools.product(
            (0, 1, 2), (0.30, 0.45), (0.20, 0.35), (-0.024, -0.036)):
        rec, mres, rows, cap = do_run("press24", timing, ff_r, mf_r, dz)
        print(f"[{len(combos):2d}/32] {rec['leg']} timing={timing} "
              f"ff_r={ff_r} mf_r={mf_r} dz={dz}: "
              f"win={rec['window']:3d} peak={rec['peak_mm']:7.2f}mm "
              f"sup={rec['support_n']} slip={rec['slip_mm']:.1f}mm "
              f"yaw={rec['yaw_deg']:.1f}deg "
              f"morph={rec['audit'].get('morphology')} "
              f"pass={rec['pass']}")

    # ---- leg 2: +8 force-source / mixed-mode combos at the best press ----
    press = [c for c in combos if c["leg"] == "press24"]
    bp = max(press, key=lambda c: (c["score"], c["peak_mm"]))
    bt, bff, bmf = bp["timing"], bp["ff_r"], bp["mf_r"]
    print(f"mode leg at best press combo: timing={bt} ff_r={bff} mf_r={bmf}")
    for mode_pin, dz in itertools.product(
            (("force", "force"), ("force", "pos"),
             ("pos", "force"), ("pos", "pos")), (-0.024, -0.036)):
        rec, mres, rows, cap = do_run("mode8", bt, bff, bmf, dz,
                                      mode_pin=mode_pin)
        print(f"[{len(combos):2d}/32] {rec['leg']} mode={mode_pin} "
              f"dz={dz}: win={rec['window']:3d} "
              f"peak={rec['peak_mm']:7.2f}mm sup={rec['support_n']} "
              f"slip={rec['slip_mm']:.1f}mm yaw={rec['yaw_deg']:.1f}deg "
              f"morph={rec['audit'].get('morphology')} pass={rec['pass']}")

    # ---- verdict + archive ------------------------------------------------
    passing.sort(key=lambda p: p[0])
    combos.sort(key=lambda c: -c["score"])
    meta = {"xml_sha256": xml_sha, "seed": W.SEED,
            "gate": {k: W.GATE[k] for k in
                     ("lift_min_m", "lift_window_frames", "slip_max_m",
                      "yaw_max_deg", "support_force_max_n")},
            "n_combos": len(combos), "n_pass": len(passing)}
    json.dump({"meta": meta, "combos": combos},
              open(REPO / "runs/w6c/v20_grid.json", "w"), indent=1,
              default=float)

    if not passing:
        print("GRID VERDICT: no gate-passing candidate in 24+8")
        return

    _, rec, rows, cap = passing[0]
    print(f"GRID VERDICT: PASS candidate {rec['leg']} timing={rec['timing']} "
          f"ff_r={rec['ff_r']} mf_r={rec['mf_r']} dz={rec['dz']} "
          f"mode={rec['mode_pin']} window={rec['window']} "
          f"peak={rec['peak_mm']:.2f}mm")

    # ---- 3-seed reproduction (env start-noise protocol, >=2/3) ------------
    repro = []
    for seed in (0, 1, 2):
        d0s = start_snap(seed=seed, noise=True)
        mres, rows_s = run(rec["timing"], rec["ff_r"], rec["mf_r"],
                           rec["dz"], mode_pin=(None if rec["mode_pin"] is None
                                                else tuple(rec["mode_pin"])),
                           d0=d0s)
        repro.append({"seed": seed,
                      "window": mres["window_frames"],
                      "peak_mm": mres["peak_lift_m"] * 1e3,
                      "final_mm": mres["final_lift_m"] * 1e3,
                      "support_n": mres["support_mean_n"],
                      "slip_mm": mres["slip_m"] * 1e3,
                      "yaw_deg": mres["yaw_abs_deg"],
                      "endface_ratios": [mres["endface_ratio_right"],
                                         mres["endface_ratio_left"]],
                      "pass": bool(mres["pass"])})
        print(f"[repro seed {seed}] win={repro[-1]['window']:3d} "
              f"peak={repro[-1]['peak_mm']:7.2f}mm "
              f"slip={repro[-1]['slip_mm']:.1f}mm "
              f"pass={repro[-1]['pass']}")
    n_ok = sum(r["pass"] for r in repro)
    json.dump({"combo": rec, "seeds": repro, "passed": n_ok,
               "verdict": ("reproduced (>=2/3)" if n_ok >= 2
                           else "NOT reproduced (<2/3)")},
              open(REPO / "runs/w6c/v20_repro.json", "w"), indent=1,
              default=float)

    # ---- replay_frames export (qpos 65-wide, lift segment) ----------------
    ws = rec["window_start"]
    seg = cap[max(0, ws - 10):]
    json.dump({"fps": W.CTRL_HZ,
               "fps_scope": "control-rate export: one frame per 20 Hz "
                            "control frame of the v20_lift climb segment; "
                            "qpos is the full 65-dof v20.1 state; obj = "
                            "object_free [x y z qw qx qy qz]; window "
                            f"starts at control frame {ws}",
               "frames": [{"t": round(f["k"] / W.CTRL_HZ, 4),
                           "qpos": [float(v) for v in f["qpos"]],
                           "obj": [float(v) for v in f["obj"]],
                           "control_frame": int(f["k"])} for f in seg]},
              open(REPO / "runs/w6c/v20p1_replay_frames.json", "w"),
              indent=1, default=float)

    # ---- evidence frames (standard camera presets) -------------------------
    _evidence_frames(env, rec, cap, ws)


def _evidence_frames(env, rec, cap, ws) -> None:
    import PIL.Image as Image
    m = env.model
    m.vis.global_.offwidth = 960
    m.vis.global_.offheight = 540
    renderer = mujoco.Renderer(m, height=480, width=854)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    presets = {"front": (105.0, -22.0, 0.80), "side": (-20.0, -30.0, 0.72),
               "close": (-20.0, -24.0, 0.46)}
    picks = [(max(0, ws - 5), "front"), (min(len(cap) - 1, ws + 10), "side"),
             (min(len(cap) - 1, ws + 19), "close")]
    out = REPO / "runs/w6c/evidence"
    out.mkdir(parents=True, exist_ok=True)
    tag = f"v20p1_t{rec['timing']}_ff{rec['ff_r']}_mf{rec['mf_r']}_dz{rec['dz']}"
    for k, preset in picks:
        f = cap[k]
        env.data.qpos[:] = f["qpos"]
        oa = env.obj_qpos_adr
        env.data.qpos[oa:oa + 7] = f["obj"]
        mujoco.mj_forward(m, env.data)
        clear = float(f["obj"][2] - env._z_rest)
        cam.lookat[:] = [0.0, 0.0, 0.74]
        cam.azimuth, cam.elevation, cam.distance = presets[preset]
        renderer.update_scene(env.data, camera=cam)
        Image.fromarray(renderer.render()).save(
            out / f"{tag}_f{k:03d}_clear{clear*1e3:.1f}mm_{preset}.png")
    renderer.close()


if __name__ == "__main__":
    main()
