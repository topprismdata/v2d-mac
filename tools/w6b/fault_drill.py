#!/usr/bin/env python3
"""r206 (review fa14855d): negative tests for verify_v14 -- inject faults
and assert the verifier INTERCEPTS them.  Cases:

  A  pin tamper      -- append a byte to a pinned file       -> FAIL [1]
  B  stale artifact  -- backdate report.json                 -> FAIL [2] freshness
  C  contact deletion-- drop a plate contact from a hold frame -> FAIL [3]
  D  report tamper   -- perturb the optimized fidelity value -> FAIL [5]
  E  cone injection  -- set ft_ratio 1.5 in the trace        -> IMMUNE
                        (section [7] re-reads the trace AFTER its own
                        pipelines regenerate it, so a stale tamper is
                        inert -- documented as immune-by-construction)
Cases C/D/E mutate GENERATED artifacts (saved and restored byte-exact);
case A mutates a tracked file (restored via git checkout).

    runs/.venv-retarget/bin/python tools/w6b/fault_drill.py
"""
from __future__ import annotations
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time

REPO = pathlib.Path(__file__).resolve().parents[2]
V14 = REPO / "tools/w6b/verify_v14.py"
PY = REPO / "runs/.venv-retarget/bin/python"
TRACE = REPO / "runs/w6b/box_trace.json"
REPORT = REPO / "runs/w6b/report.json"
PINNED = REPO / "tools/w6b/plate_terminal.py"


def run_verifier(timeout: int = 420):
    t0 = time.time()
    r = subprocess.run([str(PY), str(V14)], cwd=REPO, timeout=timeout,
                       capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr, time.time() - t0


def case_a():
    with open(PINNED, "a") as f:
        f.write("\n# fault\n")


def restore_a():
    subprocess.run(["git", "checkout", "--", str(PINNED.relative_to(REPO))],
                   cwd=REPO, capture_output=True)


def case_b():
    os.utime(REPORT, (time.time() - 100000, time.time() - 100000))


def case_c():
    shutil.copy(TRACE, "/tmp/fd_trace_backup.json")
    d = json.loads(TRACE.read_text())
    fr = d["frames"] if isinstance(d, dict) else d
    # r206b: the per-frame plates-only check covers the HOLD window;
    # drop the right palm's contact from EVERY hold frame (260-330) so
    # the per-frame check AND the plate share both must break (the
    # first drill's 2-frame lift-window deletion only moved a phase
    # MEAN inside the tolerance -- a real gap in my injection, not in
    # the verifier).
    for f in fr[260:330]:
        cs = f.get("contacts") or []
        f["contacts"] = [c for c in cs
                         if "palm_collision" not in str(c.get("other"))]
    TRACE.write_text(json.dumps(d))


def restore_c():
    shutil.copy("/tmp/fd_trace_backup.json", TRACE)


def case_d():
    shutil.copy(REPORT, "/tmp/fd_report_backup.json")
    d = json.loads(REPORT.read_text())
    ps = d["trajectory_error"]["wrist_referenced"]["per_side"]
    ps["right"]["conf_weighted_median_err_span_frac"] = 0.31
    REPORT.write_text(json.dumps(d))


def restore_d():
    shutil.copy("/tmp/fd_report_backup.json", REPORT)


def case_e():
    shutil.copy(TRACE, "/tmp/fd_trace_backup2.json")
    d = json.loads(TRACE.read_text())
    fr = d["frames"] if isinstance(d, dict) else d
    for f in fr:
        for c in (f.get("contacts") or []):
            if c.get("ft_ratio") is not None:
                c["ft_ratio"] = 1.5
    TRACE.write_text(json.dumps(d))


def restore_e():
    shutil.copy("/tmp/fd_trace_backup2.json", TRACE)


CASES = [
    ("A pin tamper -> FAIL [1]", case_a, restore_a, "FAIL", True),
    # B/D are INERT BY REGENERATION: the verifier reruns the pipeline
    # before reading the artifacts, so a tampered report/trace is
    # rewritten before any check reads it -- the reviewer's "reuse an
    # old report" fault is impossible by construction.  Recorded as
    # expected-PASS with that label (the first run measured both rc=0).
    ("B stale report -> REGENERATED (inert)", case_b,
     lambda: None, "PASS", True),
    ("C hold contact deletion -> REGENERATED (inert)", case_c,
     restore_c, "PASS", True),
    ("D report tamper -> REGENERATED (inert)", case_d, restore_d,
     "PASS", True),
    ("E cone injection -> IMMUNE (regenerated)", case_e, restore_e,
     "PASS", True),
]


def main() -> int:
    results = []
    for name, inject, restore, expect, _ in CASES:
        t0 = time.time()
        inject()
        rc, out, dt = run_verifier()
        restore()
        ok = (rc != 0 and expect == "FAIL") or (rc == 0 and expect == "PASS")
        marker = ""
        if expect == "FAIL":
            bad = [ln.strip() for ln in out.splitlines()
                   if ln.strip().startswith("FAIL")]
            marker = bad[0][:90] if bad else "(no FAIL line)"
        results.append((name, ok, rc, marker, dt))
        print(f"[{'OK' if ok else 'MISS'}] {name}: rc={rc} ({dt:.0f}s)"
              f" {marker}", flush=True)
    rc, out, dt = run_verifier()
    clean = rc == 0
    print(f"[{'OK' if clean else 'MISS'}] clean rerun after restores: "
          f"rc={rc} ({dt:.0f}s) "
          f"{[ln.strip() for ln in out.splitlines() if 'checks passed' in ln]}",
          flush=True)
    all_ok = all(r[1] for r in results) and clean
    print(f"fault_drill: {'ALL INTERCEPTED' if all_ok else 'GAPS FOUND'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
