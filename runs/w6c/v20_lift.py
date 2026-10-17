"""v20 full-stack re-run: follower pads + bilateral press + force lift.

Name-mapped checkpoint adapter (v14 51-dof -> v20 65-dof), bilateral-first
press, force-source hold, climb with N-hold, gate + morphology + mu_eff.
"""
import hashlib
import itertools
import os
import json
import pathlib
import sys

import mujoco
import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import grounding.train_w6c as W  # noqa: E402

FS = 25

# Placement solve v2 (FK + tip-path Jacobian scan + closed-loop seat,
# 2026-10-17). THE physics criterion is the curl tip-path ANGLE to the
# face normal, not just pad position: N_max = forcerange/arm x cos(theta)
# (measured: the pre-rotation placement had cos 0.19 -> N 1.41 regardless
# of command; the rotated placement drives N 2.4 -> 6.2 N on demand).
# RIGHT (solved): rotate the wrist (rx +0.15, ry +0.60, rz -0.10) to align
# the curl plane with the face normal, wx +20 mm, wz +50 mm, curls
# (0.40, 0.30); pads seat x +0.0663 z -0.003; N 4.2-6.2 N on +0.25 rad
# demand (>= the 2.16 N/side liftable threshold).
# LEFT (v21, rf set + y): the y-augmented rescan found NO ff/mf solution
# (in-band poses sit 24-32 mm proud, arc-structural), but the RF set
# SOLVES: wrist y +30 mm, x +10 mm, rot (rx +0.20, ry +0.60, rz -0.15),
# rf curls (0.40, 0.45) -> rf tip lands (-0.0573, -0.0908, -0.0081)
# in-band at mid-height. The left ff/mf fingers park open at 0.35.
PL_RIGHT = {"wx": 0.020, "wz": 0.050, "drx": 0.15, "dry": 0.60,
            "drz": -0.10, "curl1": 0.40, "curl2": 0.30}
PL_LEFT = {"wy": 0.030, "wx": 0.010, "wz": -0.045, "drx": 0.20, "dry": 0.60,
           "drz": -0.15, "rf1": 0.40, "rf2": 0.45, "park": 0.35}
# v21: the left press/demand set is RF (rfj1/rfj2 left, actuators
# rfa1/rfa2_left at ctrl 37/38; W6a 98c498a raised their forceranges to
# 0.180/0.159). The right set stays ff/mf (actuators 7/8/11/12).
LEFT_PRESS_ACTS = (37, 38)
LEFT_PRESS_JOINTS = ("rfj1_left", "rfj2_left")
# Climb-engage criterion (runner machinery, decoupled from the frozen
# GATE clauses): the vise is a COUPLED system - the box is held when
# mu_pad x (N_r + N_l) > W (1.4715 N at mu 0.34 -> sum >= 4.5 N) and each
# side keeps contact. The earlier per-side 2.4 N threshold blocked the
# climb even when the sum capacity was 2.0-2.4 N (right 4.3-5.5 + left
# 1.6-1.85): the sum form is the physically-correct statement.
ENGAGE_SUM_N = 4.5
ENGAGE_MIN_SIDE = 1.0


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

    def jadr_local(name):
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
        return int(m.jnt_qposadr[jid])

    padg = [[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM,
                               f"{f}_tip{s}_collision")
             for f in ("ff", "mf")] for s in ("", "_left")]
    lo_c = m.actuator_ctrlrange[:, 0]
    hi_c = m.actuator_ctrlrange[:, 1]

    def run(timing, ff_r, mf_r, dz_mm, nr_star=4.0, nl_star=6.8, rate=1.0,
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
        # ---- placement phase: seat the hands' ff/mf pads on the end faces
        # per the placement solve v2 (right: rotation-aligned curl plane;
        # left: best in-band pose - see the PL_* block above). The
        # bilateral press then ramps ADDITIONAL curl on top. Replaces the
        # legacy blind descend, which pressed the pads over the box TOP
        # edge where the curl path is tangential.
        seat = start_ctrl = d.ctrl.copy()
        clear = seat.copy()
        for i_ in (7, 8, 11, 12) + LEFT_PRESS_ACTS:
            clear[i_] = 0.40
        clear[2] += 0.050
        clear[24] += 0.045
        transit = clear.copy()
        # right placement first (rotation-aligned ff/mf), LEFT channels
        # parked open - the right ramp moves the box, so the left rf seat
        # must happen AFTER the box stops moving
        place_r = transit.copy()
        place_r[0] += PL_RIGHT["wx"]; place_r[2] += PL_RIGHT["wz"]
        place_r[3] += PL_RIGHT["drx"]; place_r[4] += PL_RIGHT["dry"]
        place_r[5] += PL_RIGHT["drz"]
        place_r[7] = PL_RIGHT["curl1"]; place_r[8] = PL_RIGHT["curl2"]
        place_r[11] = PL_RIGHT["curl1"]; place_r[12] = PL_RIGHT["curl2"]
        for a, b, n in ((seat, clear, 12), (clear, transit, 10),
                        (transit, place_r, 15)):
            for k in range(n):
                f = (k + 1) / n
                d.ctrl[:] = a + (b - a) * f
                for _ in range(FS):
                    mujoco.mj_step(m, d)
        for _ in range(15 * FS):
            mujoco.mj_step(m, d)
        # left rf placement (v21): wrist y/x + rotation, rf curls; the
        # left ff/mf fingers stay parked open so their pads stay clear
        place_l = d.ctrl.copy()
        place_l[23] += PL_LEFT["wy"]; place_l[22] += PL_LEFT["wx"]
        place_l[25] += PL_LEFT["drx"]; place_l[26] += PL_LEFT["dry"]
        place_l[27] += PL_LEFT["drz"]
        place_l[37] = PL_LEFT["rf1"]; place_l[38] = PL_LEFT["rf2"]
        for k in range(15):
            f = (k + 1) / 15
            d.ctrl[:] = place_r + (place_l - place_r) * f
            for _ in range(FS):
                mujoco.mj_step(m, d)
        for _ in range(10 * FS):
            mujoco.mj_step(m, d)
        face_x = float(m.geom_size[env.obj_geom][0])
        pad_r = float(m.geom_size[mujoco.mj_name2id(
            m, mujoco.mjtObj.mjOBJ_GEOM, "ff_tip_collision")][0])

        def pad_local(gname):
            g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, gname)
            return d.geom_xpos[g] - d.xpos[bid]

        # closed-loop seat: 4 iterations of measured wrist correction
        # (signs probed: ctrl + seats right, ctrl - seats left; ctrl z
        # down seats down; corrections clipped to the wrist ctrl ranges)
        cr_lo, cr_hi = m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1]
        # closed-loop seat: the wrist y slide has a measured ~0.5 pad gain
        # (ctrl +30 mm -> pad +15 mm), so the y correction runs at x2 gain
        # and MUST converge before the x correction - at the wrong y the
        # rf tip sits on a different arc segment where the x slide moves
        # the tip tangentially (no net seat). z gain is ~1.
        for _ in range(6):
            bl_l = pad_local("rf_tip_left_collision")
            y_err = bl_l[1] - (-0.095)
            d.ctrl[23] = float(np.clip(d.ctrl[23] - y_err * 2.0,
                                       cr_lo[23], cr_hi[23]))
            z_r = (pad_local("ff_tip_collision")[2]
                   + pad_local("mf_tip_collision")[2]) / 2.0
            z_l = bl_l[2]
            x_r = (pad_local("ff_tip_collision")[0]
                   + pad_local("mf_tip_collision")[0]) / 2.0 - (face_x + pad_r)
            x_l = -(bl_l[0]) + (face_x + pad_r)
            d.ctrl[2] = float(np.clip(d.ctrl[2] - z_r, cr_lo[2], cr_hi[2]))
            d.ctrl[24] = float(np.clip(d.ctrl[24] - z_l, cr_lo[24], cr_hi[24]))
            d.ctrl[0] = float(np.clip(
                d.ctrl[0] + float(np.clip(x_r, -0.012, 0.012)),
                cr_lo[0], cr_hi[0]))
            if abs(y_err) < 0.004:
                d.ctrl[22] = float(np.clip(
                    d.ctrl[22] - float(np.clip(x_l, -0.015, 0.015)),
                    cr_lo[22], cr_hi[22]))
            for _ in range(10 * FS):
                mujoco.mj_step(m, d)
        for _ in range(20 * FS):
            mujoco.mj_step(m, d)
        if os.environ.get("V20_DEBUG"):
            bl = pad_local("rf_tip_left_collision")
            n_r, n_l = W.pad_normal_forces(env)
            print(f"[dbg] post-seat: rf_tip=({bl[0]:+.4f},{bl[1]:+.4f},{bl[2]:+.4f}) "
                  f"N=({n_r:.2f},{n_l:.2f}) box=({d.xpos[bid][0]:+.4f},{d.xpos[bid][1]:+.4f}) "
                  f"ff_tip_r=({pad_local('ff_tip_collision')[0]:+.4f},{pad_local('ff_tip_collision')[2]:+.4f}) "
                  f"q37/38=({d.qpos[jadr_local('rfj1_left')]:.2f},{d.qpos[jadr_local('rfj2_left')]:.2f})")
        xy_press0 = d.xpos[bid][:2].copy()
        tgt = d.ctrl.copy()
        # per-hand press actuators and depth mapping (v21: the left set is
        # rf - ff_r drives rfj1, mf_r drives rfj2)
        press_acts = ((7, 8, 11, 12), LEFT_PRESS_ACTS)
        press_depths = ((ff_r, ff_r, mf_r, mf_r), (ff_r, mf_r))
        all_press = tuple(press_acts[0]) + tuple(press_acts[1])
        for h in range(2):
            for idx, depth in zip(press_acts[h], press_depths[h]):
                tgt[idx] = d.ctrl[idx] + depth
        start = {i_: d.ctrl[i_] for i_ in all_press}
        for k in range(20 + timing):
            fl = min(1.0, (k + 1) / 20.0)
            fr = min(1.0, max(0.0, (k + 1 - timing) / 20.0))
            for h, frac in ((0, fl), (1, fr)):
                for idx in press_acts[h]:
                    s = start[idx]
                    d.ctrl[idx] = s + (tgt[idx] - s) * frac
            for _ in range(FS):
                mujoco.mj_step(m, d)
        press_drift = float(np.linalg.norm(d.xpos[bid][:2] - xy_press0)) * 1e3
        if os.environ.get("V20_DEBUG"):
            bl = pad_local("rf_tip_left_collision")
            n_r, n_l = W.pad_normal_forces(env)
            print(f"[dbg] post-press: rf_tip=({bl[0]:+.4f},{bl[1]:+.4f},{bl[2]:+.4f}) "
                  f"N=({n_r:.2f},{n_l:.2f}) ctrl37/38=({d.ctrl[37]:.2f},{d.ctrl[38]:.2f}) "
                  f"q37/38=({d.qpos[jadr_local('rfj1_left')]:.2f},{d.qpos[jadr_local('rfj2_left')]:.2f})")
        anchor = d.ctrl.copy()
        mode = ["force", "force"] if mode_pin is None else list(mode_pin)
        n_star = (nr_star, nl_star)
        # v21 per-hand demand joints: (joint name, actuator ctrl index).
        # Right stays ff/mf (4 joints); the left set is RF (2 joints,
        # forceranges 0.180/0.159 per W6a 98c498a)
        demand_sets = (
            tuple(zip(("ffj1", "ffj2", "mfj1", "mfj2"), press_acts[0])),
            tuple(zip(LEFT_PRESS_JOINTS, LEFT_PRESS_ACTS)),
        )
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
            ok = (n_r + n_l >= ENGAGE_SUM_N and n_r >= ENGAGE_MIN_SIDE
                  and n_l >= ENGAGE_MIN_SIDE)
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
                # constant-torque force source: per-joint angular demand
                # ctrl = q + demand (torque saturates at the actuator
                # forcerange). Right ff/mf: n_star 4.0 -> +0.20 rad (the
                # r7e-calibrated level, realized 4.15-4.98 N in the
                # placement grid). Left rf: v21 forceranges 0.180/0.159
                # raise the same +0.30 rad demand from a 1.85 N cap to a
                # ~3.1 N capacity (W6a 98c498a), so nl_star runs at the
                # scaled setpoint 6.8.
                for jname, idx in demand_sets[si]:
                    if mode[si] == "force":
                        demand = min(nstar * 0.05, 0.30)
                        q = d.qpos[jadr_local(jname)]
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
            if os.environ.get("V20_DEBUG") and i == 40:
                for ci in range(d.ncon):
                    c = d.contact[ci]
                    g1, g2 = int(c.geom1), int(c.geom2)
                    og1 = m.geom_bodyid[g1] == m.geom_bodyid[env.obj_geom]
                    og2 = m.geom_bodyid[g2] == m.geom_bodyid[env.obj_geom]
                    if not (og1 or og2):
                        continue
                    other = g2 if og1 else g1
                    f6 = np.zeros(6)
                    mujoco.mj_contactForce(m, d, ci, f6)
                    fr = d.xmat[env.obj_body].reshape(3, 3).T @ f6[:3]
                    print(f"[dbg] f40 CONTACT {env.geom_label[other]:24s} "
                          f"F_world=({fr[0]:+.2f},{fr[1]:+.2f},{fr[2]:+.2f})N "
                          f"pos=({c.pos[0]-d.xpos[bid][0]:+.3f},{c.pos[1]-d.xpos[bid][1]:+.3f},{c.pos[2]-d.xpos[bid][2]:+.3f})")
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
               "n_r_max": max((f["n_r"] for f in rows), default=0.0),
               "n_l_max": max((f["n_l"] for f in rows), default=0.0),
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
