#!/usr/bin/env python3
"""v14 clean-room verification: the v12 load-bearing chain PLUS the
fidelity optimization (the review-authorized iterative scheme).

The v14 sections extend v13: the optimized recipe's lift is raised to
9 mm (its 3.4-deg grasp tilt brings a box corner ~6.6 mm low, so the 6 mm
lift left a 0.173-0.185 N corner-table contact; 9 mm clears it to 0.0 N
with the fidelity unchanged) and the r196 middle-finger clearance bias
(0.8 rad curl on the left mfj1-3 over the last 40 approach rows), which
retracts the tip that r178 pinned as the pre-displacement pusher and, as
a side effect, removes the g4 limit event entirely (36 -> 0 frames: the
finger is no longer wedged by the box at the lift) and the section asserts the table-free lift
end explicitly.  Unchanged: the v12 sections (the delivered recipe's
gates, the force
account, the frozen envelope's corners, the trajectory closure).  The new
section [5] reproduces the fidelity campaign from scratch: the baseline
24.5% improvement is re-earned by two converging rounds of the damped
accumulating fixed-point injection plus the left-only home seek.

    runs/.venv-retarget/bin/python tools/w6b/verify_v14.py

The v11 review's demands this package answers:
  P0  a per-frame force ACCOUNT (not a mu*N proxy) with a Newton-Euler
      z residual, attributing the box's weight to specific contacts;
  P1  the same control sequence executed WITH and WITHOUT the auxiliary
      constraints (yaw stops + vise corners): the lift must not need them;
  P2  a FROZEN perturbation envelope, run in full, reported in full.

Closedness discipline (this verifier):
  * every expected number is pinned; a missing artifact is a FAIL, not a
    skip (fail-closed);
  * the scene XMLs, the plate references and the state seeds are pinned
    by SHA-256 in runs/w6b/v13_asset_manifest.json (committed); any drift
    fails the run before a number is read;
  * the executed trajectory is compared FRAME BY FRAME against the
    committed evidence trace (max |dx|, |dy|, |dz| over all 421 frames);
  * determinism: two fresh runs of the same recipe must agree to the
    printed precision.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RESULTS: list[tuple[bool, str, str]] = []
T0 = time.time()


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((bool(ok), name, detail))
    print(f"  {'OK ' if ok else 'FAIL'}  {name}"
          + (f"  [{detail}]" if detail else ""))


def run_tool(script: str, env_extra: dict | None = None,
             timeout: int = 900) -> tuple[int, str]:
    env = dict(os.environ)
    env.update(env_extra or {})
    p = subprocess.run(
        [sys.executable, str(REPO / "tools/w6b" / script)],
        capture_output=True, text=True, timeout=timeout, env=env, cwd=REPO)
    return p.returncode, p.stdout + p.stderr


def fresh(path: Path, since: float) -> bool:
    return path.exists() and path.stat().st_mtime >= since


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# the delivered clean-variant recipe (r166/r167)
RECIPE = {
    "W6B_SCENE": "ur5e",
    "W6B_SCENE_XML":
        "grounding/assets/allegro_floating_ur5e_side_r45_ablation.xml",
    "W6B_R63_REFERENCE": "runs/w6b/plate_refs/plate_2.json",
    "W6B_R155_TRIM_MM_LEFT": "-0.01",
    "W6B_R109_SQUEEZE_MM": "0.4",
    "W6B_LIFT_MM": "6",
    "W6B_LIFT_FRAMES": "90",
    "W6B_R127_LEFT_CURL_DELAY": "80",
    "W6B_R127_CURL_DELAY_RIGHT": "64",
    "W6B_R144_FINGER_LEAD": "100",
    "W6B_R38_SEAT": "left",
    "W6B_R23_FLYOVER_MM": "250",
    "W6B_R22_MARGIN_MM": "8",
    "W6B_R75_SIDE_APPROACH": "1",
    "W6B_R63_TARGET": "1",
    "W6B_R18_NATIVE_TARGET": "1",
    "W6B_R50_CURL": "1.25",
    "W6B_R39_TGC": "0",
    "W6B_R36_GSTATIC": "1",
    "W6B_R36_I": "0",
    "W6B_R39_HOLD": "60",
    "W6B_R126_BOX_TRACE": "1",
}

EXPECTED = {
    "rise_mm": (7.814, 0.5),          # peak == final, the r166 anchor
    "force_right_N": (24.00, 0.15),
    "force_left_N": (24.04, 0.15),
    "weight_N": 3.9279,
    "account_resid_N": (0.0036, 0.02),
    "account_approach_table_N": (3.928, 0.01),
}


def main() -> int:
    print("=== v12 clean-room verification (fresh processes)")

    # 0. asset-hash closure, BEFORE any number is read.
    # r176: fail-closed ORDER fixed per the v12 review (P3-1): check
    # existence FIRST (a missing file must not raise inside sha()), then
    # compare hashes, and ABORT the run immediately on any drift -- a
    # drifted asset must not be followed by any number-producing step.
    print("[1] asset manifest (fail-closed)")
    man_p = REPO / "runs/w6b/v13_asset_manifest.json"
    if not man_p.exists():
        check("the asset manifest exists", False, str(man_p))
        return _fin()
    man = json.loads(man_p.read_text())
    missing = [rel for rel in man if not (REPO / rel).exists()]
    changed = sorted(rel for rel in man
                     if rel not in missing
                     and sha(REPO / rel) != man[rel])
    assets_ok = not missing and not changed
    check("every pinned asset matches its committed SHA-256",
          assets_ok, f"changed={changed} missing={missing}")
    if not assets_ok:
        return _fin()

    # 1. the delivered recipe, fresh process
    print("[2] the delivered r45 recipe (fresh process)")
    t_run = time.time()
    trace_p = REPO / "runs/w6b/box_trace.json"
    report_p = REPO / "runs/w6b/report.json"
    rc, out = run_tool("dual_recipe_run.py", RECIPE, timeout=900) \
        if (REPO / "tools/w6b/dual_recipe_run.py").exists() \
        else _pipeline(RECIPE, timeout=900)
    check("the pipeline ran clean", rc == 0, "exit " + str(rc))
    check("the artifacts are fresh (fail-closed)",
          fresh(trace_p, t_run) and fresh(report_p, t_run),
          "stale or missing trace/report")
    if not (fresh(trace_p, t_run) and fresh(report_p, t_run)):
        return _fin()

    tr = json.loads(trace_p.read_text())
    fr = tr["frames"] if isinstance(tr, dict) else tr
    rep = json.loads(report_p.read_text())
    z = [f["pos"][2] for f in fr]
    rise = (max(z) - z[0]) * 1e3
    final = (z[-1] - z[0]) * 1e3
    exp_r, tol_r = EXPECTED["rise_mm"]
    check(f"the box rises to the hold and stays ({exp_r} mm)",
          abs(rise - exp_r) <= tol_r and abs(final - rise) < 1e-9,
          f"peak {rise:.3f} final {final:.3f} mm")
    fs = {s: rep["per_side"][s]["gate6_executed_contact"]
          ["normal_force_sum_N"] for s in ("right", "left")}
    g6ok = all(rep["per_side"][s]["gate6_executed_contact"]["pass"]
               for s in ("right", "left"))
    check("gate6 double-side PASS with real plate contact", g6ok,
          f"{fs['right']:.2f} / {fs['left']:.2f} N")
    check("the plate forces match the anchors",
          abs(fs["right"] - EXPECTED["force_right_N"][0])
          <= EXPECTED["force_right_N"][1]
          and abs(fs["left"] - EXPECTED["force_left_N"][0])
          <= EXPECTED["force_left_N"][1],
          f"{fs['right']:.4f} / {fs['left']:.4f} N")
    aux = sorted({str(c.get("other", "")) for f in fr
                  for c in f.get("contacts") or []
                  if "yaw_stop" in str(c.get("other", ""))
                  or "vise" in str(c.get("other", ""))})
    check("zero auxiliary-constraint contacts in the whole trace",
          not aux, str(aux))
    tbl_last = sum(c.get("fn", 0.0) for c in fr[-1].get("contacts") or []
                   if "table" in str(c.get("other", "")))
    check("the final table support is zero", tbl_last < 1e-2,
          f"{tbl_last:.4f} N")

    # 2. the force account closes (on the delivered recipe's fresh trace)
    print("[3] the per-frame force account (P0)")
    rc, out = run_tool("lift_force_account.py", {}, timeout=300)
    check("lift_force_account ran", rc == 0, "exit " + str(rc))
    rows = {}
    for line in out.splitlines():
        parts = line.split()
        if parts and parts[0] in ("approach", "grasp", "lift_accel",
                                  "lift_hold") and "=" in line:
            d = {}
            for tok in parts[1:]:
                if "=" in tok:
                    k, v = tok.split("=", 1)
                    try:
                        d[k] = float(v)
                    except ValueError:
                        pass
            rows[parts[0]] = d
    check("the account table parsed (4 phases, fresh process)",
          len(rows) == 4, str(sorted(rows)))
    if len(rows) == 4:
        app, hold = rows["approach"], rows["lift_hold"]
        exp_t, tol_t = EXPECTED["account_approach_table_N"]
        check("approach: the table carries the full weight",
              abs(app.get("table", 0.0) - exp_t) <= tol_t
              and app.get("plate", 0.0) == 0.0
              and app.get("other", 0.0) == 0.0,
              f"table {app.get('table', 0.0):+.3f} N")
        # hold: EVERY contact class must be shown; the plates alone carry
        # the box (their world-z shares may be asymmetric -- a negative
        # share is a real downward friction, required for closure)
        pl = hold.get("plate", 0.0)
        # per-frame uniqueness (the review rejected the phase-aggregate
        # check: same-class contacts could cancel inside an average):
        # every hold frame's contact set must contain ONLY plate geoms
        bad_frames = []
        for i in hold:
            # r187: only frame-index keys are per-frame entries; the
            # account row also carries aggregate metric keys (plate,
            # SUM-mg, resid_z) which are not frame indices (the v12
            # verifier crashed on them once the account's format gained
            # the per-frame residual column).
            try:
                fi = int(i)
            except (TypeError, ValueError):
                continue
            if not (0 <= fi < len(fr)):
                continue
            names = {str(c.get("other", "")) for c in
                     (fr[fi].get("contacts") or []) if "f6" in c}
            bad = [n for n in names
                   if "palm_collision" not in n
                   and "palm_left_collision" not in n]
            if bad:
                bad_frames.append((i, sorted(bad)))
        check("hold: every frame's contact set is plates-only (per-frame)",
              not bad_frames, str(bad_frames[:3]))
        check("hold: the two plates' world-z shares sum to the weight",
              abs(pl - EXPECTED["weight_N"]) <= 0.02,
              f"plate share {hold.get('plate', 0.0):+.3f} N "
              f"(sum {pl:+.3f} vs weight {-EXPECTED['weight_N']:+.3f})")
        check("hold: the account closes (SUM - mg within the tolerance)",
              abs(hold.get("SUM-mg", 1e9)) <= 0.02,
              f"SUM-mg {hold.get('SUM-mg', 0.0):+.4f} N")
        er, et = EXPECTED["account_resid_N"]
        check("the Newton-Euler z residual closes",
              abs(hold.get("resid_z", 1e9)) <= abs(et) + abs(er),
              f"resid {hold.get('resid_z', 0.0):+.4f} N (anchor {er:+.4f})")
        # r176 (P3-5): the review rejected phase-MEAN-only residuals --
        # a mean can hide per-frame cancellations.  The account now also
        # prints the per-frame |resid_z| maximum for every phase; the
        # HOLD phase (the load-bearing proof, quasi-static) must close
        # PER FRAME, not just on average.
        rzmax = hold.get("resid_z|max")
        check("hold: the z account closes on EVERY frame (per-frame max)",
              rzmax is not None and rzmax <= 0.01,
              f"per-frame |resid_z| max {rzmax if rzmax is not None else 'n/a'} N "
              f"(anchor 0.0050, tol 0.01)")

    # 3. the frozen envelope's corner cells (P2)
    print("[4] the frozen envelope's corner cells (P2)")
    # r195: the envelope measured in FULL before freezing -- six new
    # corners scanned (all exit 0, g6 double-green, rise 7.636-7.766 mm)
    # and pinned at their measured anchors; the delivered recipe's
    # perturbation envelope is now 8 cells (L in [-0.035,-0.005],
    # S in [0.2, 1.0]).
    corners = {"L-0.02_S0.2": 8.066, "L-0.005_S0.8": 7.805,
               "L-0.03_S0.2": 7.644, "L-0.03_S0.8": 7.719,
               "L-0.02_S0.8": 7.674, "L-0.005_S0.2": 7.766,
               "L-0.035_S0.8": 7.743, "L-0.02_S1.0": 7.636}
    for cell, exp in corners.items():
        lt, sq = cell.split("_")
        env2 = dict(RECIPE)
        env2["W6B_R155_TRIM_MM_LEFT"] = lt[1:]
        env2["W6B_R109_SQUEEZE_MM"] = sq[1:]
        rc, _ = _pipeline(env2, timeout=900)
        tr2 = json.loads(trace_p.read_text())
        fr2 = tr2["frames"] if isinstance(tr2, dict) else tr2
        z2 = [f["pos"][2] for f in fr2]
        r2 = (max(z2) - z2[0]) * 1e3
        check(f"the corner {cell} reproduces ({exp} mm)",
              rc == 0 and abs(r2 - exp) <= 0.02,
              f"rise {r2:.3f} mm, exit {rc}")

    # 4. frame-by-frame trajectory closure vs the committed evidence.
    # r176 (P3-2): the rerun's return code and artifact freshness are
    # asserted BEFORE any comparison, not just at the P2 corners.
    print("[5] full-trajectory comparison (committed evidence)")
    ev_p = REPO / "runs/w6b/box_trace_r45_p2_base.json"
    if not ev_p.exists():
        check("the committed evidence trace exists", False, str(ev_p))
    else:
        t5 = time.time()
        rc5, _ = _pipeline(RECIPE, timeout=900)
        fresh5 = (fresh(trace_p, t5) and fresh(report_p, t5))
        check("the evidence rerun exits clean and is fresh",
              rc5 == 0 and fresh5, f"exit {rc5}, fresh {fresh5}")
        if rc5 != 0 or not fresh5:
            return _fin()
        ev = json.loads(ev_p.read_text())
        evf = ev["frames"] if isinstance(ev, dict) else ev
        tr3 = json.loads(trace_p.read_text())
        fr3 = tr3["frames"] if isinstance(tr3, dict) else tr3
        if len(evf) != len(fr3):
            check("the trajectory lengths match", False,
                  f"{len(evf)} vs {len(fr3)}")
        else:
            # r176 (P3-3/P3-4): the state contract covers position AND
            # orientation AND the per-frame contact-geom SET -- position
            # equality alone cannot exclude a rotated or re-contacted
            # state.  The wording is also fixed: the contract is a
            # 1e-6 m / 1e-6 rad / set-equality tolerance, NOT strict
            # bit-level identity (the observed maxima happen to be 0.0,
            # which is a stronger observation, not the contract).
            mx = max(abs(a["pos"][k] - b["pos"][k])
                     for a, b in zip(evf, fr3) for k in range(3))

            def _qang(q0, q1):
                d = abs(sum(a * b for a, b in zip(q0, q1)))
                return 2.0 * math.degrees(math.acos(min(1.0, d)))

            mq = max(_qang(a["quat"], b["quat"])
                     for a, b in zip(evf, fr3))
            mc = [(i, sorted({str(c.get("other", ""))
                              for c in (a.get("contacts") or [])
                              if c.get("fn", 0.0) > 0.0}
                             ^ {str(c.get("other", ""))
                                for c in (b.get("contacts") or [])
                                if c.get("fn", 0.0) > 0.0}))
                  for i, (a, b) in enumerate(zip(evf, fr3))
                  if {str(c.get("other", "")) for c in (a.get("contacts") or []) if c.get("fn", 0.0) > 0.0}
                  != {str(c.get("other", "")) for c in (b.get("contacts") or []) if c.get("fn", 0.0) > 0.0}]
            check("the executed state is consistent with the evidence "
                  "within the contract (pos <1e-6 m, quat <1e-5 deg, "
                  "active-contact geom sets equal)",
                  mx < 1e-6 and mq < 1e-5 and not mc,
                  f"max |dpos| {mx:.3e} m, max |dq| {mq:.3e} deg, "
                  f"contact-set diffs {len(mc)} over {len(fr3)} frames"
                  + (f" first {mc[0]}" if mc else ""))

    # 5. the FIDELITY OPTIMIZATION (v13): the review's authorized
    # iterative scheme reproduces from scratch -- delete the sidecar,
    # run the baseline, then two converging rounds of the damped
    # accumulating fixed point + the left-only home seek; the numbers
    # and the green seat must come back.
    print("[5] the fidelity optimization reproduces from scratch")
    sidecar = REPO / "runs/w6b/r185_offset.npz"
    if sidecar.exists():
        sidecar.unlink()
    opt = dict(RECIPE)
    opt.update({
        "W6B_LIFT_MM": "9",
        "W6B_R178_MF_BIAS_LEFT": "0.8",
        "W6B_R199_SOFTSEAT_RIGHT": "48",
        "W6B_R199_CLEAR_MM_RIGHT": "1.0",
        "W6B_R185_LEARN": "2",
        "W6B_R185_DAMP": "0.15",
        "W6B_R185_CAP_MM": "120",
        "W6B_R185_HOME_SEEK": "1",
        "W6B_R185_HOME_SEEK_SIDES": "left",
        "W6B_R185_HOME_SEEK_MM": "150",
    })
    rc, _ = _pipeline(RECIPE, timeout=900)
    rep0 = json.loads(report_p.read_text())
    base = {s: rep0["trajectory_error"]["wrist_referenced"]["per_side"][s]
            ["conf_weighted_median_err_span_frac"]
            for s in ("right", "left")}
    check("the baseline fidelity reproduces (31.8/44.6%)",
          abs(base["right"] - 0.318) <= 2e-3
          and abs(base["left"] - 0.446) <= 2e-3,
          f"{base['right']:.3f} / {base['left']:.3f}")
    for rnd in (1, 2):
        rc, out = _pipeline(opt, timeout=900)
        check(f"the optimization round {rnd} ran clean", rc == 0,
              "exit " + str(rc))
        # r200 (review r199, the P3 execution-chain item): a knob that
        # never reaches the final execution stream must not be counted.
        # The r199 shaping is applied AFTER the r63-reorient/r38-seat
        # stages (which silently wiped an earlier edit location), so the
        # pipeline's post-seat print is the "actually applied" marker;
        # the executed consequence is asserted separately below.
        check(f"the r199 shaping reached the final stream (round {rnd})",
              "r199-softseat] right" in out,
              "marker " + str("r199-softseat] right" in out))
    repo = json.loads(report_p.read_text())
    fin = {s: repo["trajectory_error"]["wrist_referenced"]["per_side"][s]
           ["conf_weighted_median_err_span_frac"]
           for s in ("right", "left")}
    check("the optimized fidelity reproduces (24.5/34.0%)",
          abs(fin["right"] - 0.245) <= 3e-3
          and abs(fin["left"] - 0.340) <= 3e-3,
          f"{fin['right']:.3f} / {fin['left']:.3f}")
    check("the optimized recipe keeps the seat green (gate6 both sides)",
          all(repo["per_side"][s]["gate6_executed_contact"]["pass"]
              for s in ("right", "left")),
          "gate6 pass " + str([repo["per_side"][s]
                               ["gate6_executed_contact"]["pass"]
                               for s in ("right", "left")]))
    # r190: the lifted box must be OFF the table at the lift's end.
    # The baseline's 0.71-deg grasp tilt clears the 6 mm lift; the
    # optimized recipe's injection leaves a 3.4-deg tilt, i.e. a corner
    # ~6.6 mm low -- its 6 mm lift kept 0.173-0.185 N of corner-table
    # contact (measured, deterministic), so the optimized recipe lifts
    # 9 mm: the end is 10.01 mm up with 0.0 N table contact.
    trace = json.loads((REPO / "runs/w6b/box_trace.json").read_text())
    frames = trace["frames"] if isinstance(trace, dict) else trace
    zs = [f_["pos"][2] for f_ in frames]
    rise_mm = (zs[-1] - zs[0]) * 1e3
    tbl_end = sum(float(c.get("fn", 0.0))
                  for c in (frames[-1].get("contacts") or [])
                  if "table" in str(c.get("other", "")))
    q = frames[-1]["quat"]
    tilt = math.degrees(math.acos(max(
        -1.0, min(1.0, 1.0 - 2.0 * (q[1] ** 2 + q[2] ** 2)))))
    check("the optimized lift ends OFF the table (rise, table 0 N)",
          rise_mm >= 9.5 and tbl_end < 1e-2 and tilt < 6.0,
          f"rise {rise_mm:.2f} mm, table {tbl_end:.3f} N, tilt {tilt:.2f} deg")
    # r192 (review v14, the final table-off criterion): after the first
    # confirmation frame t_c (table force < eps_F AND clearance > 0),
    # EVERY later frame must keep the minimum signed corner-table
    # distance above eps_d and the table force at or below eps_F.
    # Tolerances frozen at 0.05 mm / 1e-3 N (the contact solver's
    # penetration scale is ~1e-5 m); the box half-extents are the
    # model's own (0.06, 0.12, 0.04) and the table top is z=0.70.
    half = (0.06, 0.12, 0.04)
    def _min_clear(f_):
        q = f_["quat"]
        w, x, y, z = q
        r20 = 2*(x*z - w*y); r21 = 2*(y*z + w*x); r22 = 1 - 2*(x*x + y*y)
        px, py, pz = f_["pos"]
        best = None
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                for sz in (-1.0, 1.0):
                    wz = (r20*sx*half[0] + r21*sy*half[1]
                          + r22*sz*half[2] + pz)
                    d = wz - 0.70
                    if best is None or d < best:
                        best = d
        return best
    def _tbl_f(f_):
        return sum(float(c.get("fn", 0.0))
                   for c in (f_.get("contacts") or [])
                   if "table" in str(c.get("other", "")))
    eps_d, eps_F = 5e-5, 1e-3
    clears = [_min_clear(f_) for f_ in frames]
    tf = [_tbl_f(f_) for f_ in frames]
    tc = next(i for i in range(len(frames))
              if tf[i] < eps_F and clears[i] > 0.0)
    min_c = min(clears[tc:])
    max_f = max(tf[tc:])
    check("the lift stays clear PER FRAME after t_c (d_min > eps, F<=eps)",
          min_c >= eps_d and max_f <= eps_F,
          f"t_c {tc}, min clearance {min_c*1e3:.3f} mm >= 0.05 mm, "
          f"max table {max_f:.4f} N")
    # the accepted reds of the optimized recipe, disclosed in the
    # package: g1/g5 are the delivered recipe's own accepted reds; g4 is
    # the left mfa0's limit-boundary event (36 lift/hold frames,
    # 3.1e-5 rad = 0.0018 deg, deterministic -- measured twice; command
    # deltas of +0.3/-0.2/+0.02/+0.04 leave it at exactly 36 frames
    # while the command column changes: the middle finger is wedged by
    # the box, so the fix is geometric, see the ledger) and the wrist
    # gate is the very metric under optimization (still FAIL at 10%,
    # improved by 24%)
    # r196: with the clearance bias the g4 limit event is gone
    # (0 frames) and gate4_joint_limits PASSES -- the failing set is the
    # delivered recipe's own accepted reds plus the wrist gate.
    check("the optimized failing set is the disclosed one",
          set(k for k, v in repo["verdict_checks"].items() if not v)
          == {"gate1_final_touch",
              "gate5_palm_deviation", "wrist_referenced_error"},
          str(sorted(k for k, v in repo["verdict_checks"].items()
                     if not v)))
    # r199: the residual pre-displacement was a right-palm chamfer
    # graze (87.42 N lateral impulse at frame 281); the soft-seat +
    # chamfer-clear profile kills it.  Pin: the approach displacement
    # (frames 0..300, xy) under 0.5 mm and no palm contact above 1 N
    # before the lift.
    trace_p2 = REPO / "runs/w6b/box_trace.json"
    _fr = json.loads(trace_p2.read_text())
    _fr = _fr["frames"] if isinstance(_fr, dict) else _fr
    _p0 = _fr[0]["pos"]
    _pre = ((sum((_fr[300]["pos"][i] - _p0[i]) ** 2 for i in (0, 1)))
            ** 0.5) * 1e3
    _imp = max((float(c.get("fn", 0.0)) for f_ in _fr[:300]
                for c in (f_.get("contacts") or [])
                if "palm" in str(c.get("other", ""))), default=0.0)
    check("the approach is contact-free (pre-disp, palm impact)",
          _pre <= 0.5 and _imp <= 1.0,
          f"pre-disp {_pre:.3f} mm, palm impact {_imp:.2f} N")
    check("the g4 limit event stays cleared (0 frames)",
          len(repo["per_side"]["left"]["gate4_joint_limits"]
              .get("finger_qpos_violation_frames") or []) == 0,
          str(len(repo["per_side"]["left"]["gate4_joint_limits"]
                  .get("finger_qpos_violation_frames") or [])) + " frames")

    # r201: the OPTIMIZED recipe's perturbation sweep (the review's P2
    # risk item: the r199 corner clearance is 0.137 mm at nominal, so a
    # clean nominal is NOT enough -- the whole frozen envelope must hold
    # with the bias + soft-seat active).  Six corners, all asserted:
    # exit 0, g4 stays 0, the approach stays contact-free (pre <= 0.5
    # mm, palm impact <= 1 N), the lift ends table-free.
    print("[6] the optimized recipe's perturbation sweep (P2)")
    states = {}

    def np_norm(a, b):
        import math
        return math.sqrt(sum((x - y) ** 2
                             for x, y in zip(a, b)))
    opt_corners = {"L-0.02_S0.2", "L-0.005_S0.8", "L-0.03_S0.2",
                   "L-0.035_S0.8", "L-0.02_S1.0", "L-0.005_S0.2"}
    for cell in sorted(opt_corners):
        lt, sq = cell.split("_")
        env3 = dict(opt)
        env3["W6B_R155_TRIM_MM_LEFT"] = lt[1:]
        env3["W6B_R109_SQUEEZE_MM"] = sq[1:]
        rc3, _ = _pipeline(env3, timeout=900)
        rep3 = json.loads(report_p.read_text())
        g4l3 = len(rep3["per_side"]["left"]["gate4_joint_limits"]
                   .get("finger_qpos_violation_frames") or [])
        tr3 = json.loads(trace_p.read_text())
        fr3 = tr3["frames"] if isinstance(tr3, dict) else tr3
        p03 = fr3[0]["pos"]
        pre3 = ((sum((fr3[300]["pos"][i] - p03[i]) ** 2
                     for i in (0, 1))) ** 0.5) * 1e3
        tbl3 = sum(float(c.get("fn", 0.0))
                   for c in (fr3[-1].get("contacts") or [])
                   if "table" in str(c.get("other", "")))
        check(f"the optimized corner {cell} holds (g4, pre, table)",
              rc3 == 0 and g4l3 == 0 and pre3 <= 0.5 and tbl3 < 1e-2,
              f"exit {rc3}, g4 {g4l3}, pre {pre3:.3f} mm, "
              f"table {tbl3:.3f} N")
        _rfp = REPO / "runs/w6b/replay_frames.json"
        if _rfp.exists():
            _rf = json.loads(_rfp.read_text())
            _ff = _rf["frames"] if isinstance(_rf, dict) else _rf
            if _ff:
                states[cell] = [float(v) for v in
                                _ff[min(300, len(_ff) - 1)]["qpos"]]

    # r202 (review r201, the P2 closure asks):
    # (a) the perturbations must actually reach the executed stream --
    #     the six corners' executed states at frame 300 are asserted
    #     pairwise DISTINCT (the L knob's effect is small -- measured
    #     ~3e-5 rad -- but far above the 1e-6 distinctness floor), so
    #     no two corners silently test the same trajectory;
    # (b) the TWO WORST corners (0.096 mm clearance: L-0.03_S0.2 and
    #     L-0.005_S0.2) get a full per-phase force account: the lift
    #     hold must close to the weight within 0.02 N, so the thin-
    #     corner success is not riding an unclosed force account.
    def _min_pair():
        vals = [np_norm(states[a], states[b])
                for a in states for b in states if a < b]
        return min(vals) if vals else float("nan")
    check("the six corners execute DISTINCT states (pairwise)",
          len(states) == 6 and _min_pair() > 1e-6,
          f"{len(states)} states, min pairwise {_min_pair():.2e}")
    for _wc in ("L-0.03_S0.2", "L-0.005_S0.2"):
        lt, sq = _wc.split("_")
        env4 = dict(opt)
        env4["W6B_R155_TRIM_MM_LEFT"] = lt[1:]
        env4["W6B_R109_SQUEEZE_MM"] = sq[1:]
        env4["W6B_DBG_G4"] = "1"
        rc4, out4 = _pipeline(env4, timeout=900)
        rcA, outA = run_tool("lift_force_account.py", {}, timeout=300)
        _hold = None
        for line in outA.splitlines():
            parts = line.split()
            if parts and parts[0] == "lift_hold" and "=" in line:
                for tok in parts[1:]:
                    if tok.startswith("SUM-mg="):
                        try:
                            _hold = float(tok.split("=", 1)[1])
                        except ValueError:
                            pass
        check(f"the worst corner {_wc} closes its lift account",
              rc4 == 0 and rcA == 0 and _hold is not None
              and abs(_hold) <= 0.02,
              f"exit {rc4}/{rcA}, lift_hold SUM-mg {_hold} N")
        # r203 (review r202): the contact-stability and control-margin
        # metrics.  slip_mm in the trace IS the accumulated true
        # relative slip (the tangential contact-point velocity
        # integrated per side); ft_ratio = |f_t| / (mu * f_n) per
        # contact -- a ratio above 1 means the friction cone is
        # violated (true sliding loss).  The control margin line
        # carries the per-actuator peak |force|/forcerange, the
        # saturated frame counts, and the smallest joint-limit margin;
        # the design's proximal 0.015 N.m saturation is known and
        # disclosed (r193) -- what must NOT appear is a negative joint
        # margin (a limit violation).
        _frt = json.loads((REPO / "runs/w6b/box_trace.json").read_text())
        _frt = _frt["frames"] if isinstance(_frt, dict) else _frt
        _rt = [float(c["ft_ratio"]) for f_ in _frt
               for c in (f_.get("contacts") or [])
               if c.get("ft_ratio") is not None]
        check(f"the worst corner {_wc} stays in the friction cone",
              bool(_rt) and max(_rt) <= 1.0,
              f"{len(_rt)} contact samples, ft_ratio max "
              f"{max(_rt) if _rt else float('nan'):.3f}")
        _marg = [ln for ln in out4.splitlines()
                 if "min joint margin" in ln]
        _vals = []
        for ln in _marg:
            try:
                _vals.append(float(ln.rstrip().rsplit(" ", 2)[-2]))
            except (IndexError, ValueError):
                pass
        check(f"the worst corner {_wc} keeps positive joint margins",
              len(_vals) == 2 and min(_vals) > 0.0,
              f"margins {[round(v, 4) for v in _vals]} rad")

    return _fin()


def _pipeline(env_extra: dict, timeout: int) -> tuple[int, str]:
    env = dict(os.environ)
    env.update(env_extra)
    p = subprocess.run(
        [sys.executable, str(REPO / "retarget/retarget_dual.py")],
        capture_output=True, text=True, timeout=timeout, env=env, cwd=REPO)
    return p.returncode, p.stdout + p.stderr


def _fin() -> int:
    bad = [n for ok, n, _ in RESULTS if not ok]
    print(f"\n{len(RESULTS) - len(bad)}/{len(RESULTS)} checks passed "
          f"({time.time() - T0:.0f} s)")
    if bad:
        print(f"FAIL: {len(bad)} checks failed:")
        for n in bad:
            print(f"  - {n}")
        return 1
    print("PASS: the v12 load-bearing chain is closed and reproducible")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
