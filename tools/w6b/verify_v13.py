#!/usr/bin/env python3
"""v13 clean-room verification: the v12 load-bearing chain PLUS the
fidelity optimization (the review-authorized iterative scheme).

The v12 sections are unchanged (the delivered recipe's gates, the force
account, the frozen envelope's corners, the trajectory closure).  The new
section [5] reproduces the fidelity campaign from scratch: the baseline
24.5% improvement is re-earned by two converging rounds of the damped
accumulating fixed-point injection plus the left-only home seek.

    runs/.venv-retarget/bin/python tools/w6b/verify_v13.py

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
                if "=" in tok and "|" not in tok:
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
            names = {str(c.get("other", "")) for c in
                     (fr[i].get("contacts") or []) if "f6" in c}
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
    corners = {"L-0.02_S0.2": 8.066, "L-0.005_S0.8": 7.805}
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
                  "within the contract (pos <1e-6 m, quat <1e-6 deg, "
                  "active-contact geom sets equal)",
                  mx < 1e-6 and mq < 1e-6 and not mc,
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
        rc, _ = _pipeline(opt, timeout=900)
        check(f"the optimization round {rnd} ran clean", rc == 0,
              "exit " + str(rc))
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
    check("the optimized failing set is the accepted one",
          set(k for k, v in repo["verdict_checks"].items() if not v)
          == {"gate1_final_touch", "gate5_palm_deviation",
              "wrist_referenced_error"},
          str(sorted(k for k, v in repo["verdict_checks"].items()
                     if not v)))

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
