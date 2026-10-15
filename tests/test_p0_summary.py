"""Pins for scripts/p0_summary.py — the P0 SUMMARY / calibration generator.

The generator is the single writer of runs/p0/SUMMARY.md and
runs/p0/p1_threshold_calibration.md. Two properties are load-bearing and are
pinned here:

  1. A missing or malformed report.json is a HARD ERROR (exit 2) that names
     the offending file — never a silent skip that would render an
     incomplete SUMMARY as if it were complete.
  2. FAIL verdicts stay FAIL in the rendered markdown, and the dry-run on the
     committed reports renders all six smokes with the FAIL reasons intact.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "p0_summary.py"
REAL_RUNS = ROOT / "runs" / "p0"

SMOKE_IDS = [
    "smoke1_sam2",
    "smoke2_vggt",
    "smoke3_foundpose",
    "smoke4_mujoco_ppo",
    "smoke5_retarget",
    "smoke6_hamer",
]


def run_gen(runs_dir: Path, out_dir: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT),
         "--runs-dir", str(runs_dir), "--out-dir", str(out_dir)],
        capture_output=True, text=True, timeout=180,
    )


def make_stub_runs(tmp: Path, *, missing=None, corrupt=None,
                   drop_verdict=None) -> Path:
    for sid in SMOKE_IDS:
        d = tmp / sid
        d.mkdir(parents=True)
        p = d / "report.json"
        if sid == missing:
            continue
        if sid == corrupt:
            p.write_text("{ this is not json", encoding="utf-8")
        elif sid == drop_verdict:
            p.write_text(json.dumps({"smoke": sid}), encoding="utf-8")
        else:
            p.write_text(json.dumps({"smoke": sid, "verdict": "pass"}),
                         encoding="utf-8")
    return tmp


def test_missing_report_is_a_hard_error(tmp_path):
    runs = make_stub_runs(tmp_path / "runs", missing="smoke3_foundpose")
    r = run_gen(runs, tmp_path / "out")
    assert r.returncode != 0, "missing report must not be silently skipped"
    assert "smoke3_foundpose/report.json" in r.stderr
    assert "missing" in r.stderr.lower()


def test_malformed_json_is_a_hard_error(tmp_path):
    runs = make_stub_runs(tmp_path / "runs", corrupt="smoke3_foundpose")
    r = run_gen(runs, tmp_path / "out")
    assert r.returncode != 0, "malformed JSON must not be silently skipped"
    assert "smoke3_foundpose/report.json" in r.stderr


def test_report_without_verdict_is_a_hard_error(tmp_path):
    runs = make_stub_runs(tmp_path / "runs", drop_verdict="smoke2_vggt")
    r = run_gen(runs, tmp_path / "out")
    assert r.returncode != 0
    assert "smoke2_vggt/report.json" in r.stderr
    assert "verdict" in r.stderr


def test_empty_runs_dir_errors_on_first_smoke(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    r = run_gen(runs, tmp_path / "out")
    assert r.returncode != 0
    assert "smoke1_sam2/report.json" in r.stderr


def test_dry_run_on_real_reports(tmp_path):
    """Scaffold check: all six smokes render, FAILs stay FAIL, calibration
    flags the MJCF-dependent gates for regeneration and marks N/A honestly."""
    r = run_gen(REAL_RUNS, tmp_path / "out")
    assert r.returncode == 0, r.stderr
    summary = (tmp_path / "out" / "SUMMARY.md").read_text(encoding="utf-8")

    rows = [ln for ln in summary.splitlines()
            if ln.startswith("| smoke") and not ln.startswith("| smoke |")]
    assert len(rows) == 6, "overview table must list all six smokes"
    for sid in SMOKE_IDS:
        assert f"| {sid} |" in summary

    assert "**FAIL**" in summary, "smoke4/smoke5 FAILs must stay FAIL"
    assert "A3b_airborne_at_peak" in summary
    assert "hand_world_pose_deviation_within_cap" in summary
    assert "p0-green" in summary, "must state the tag is withheld"
    assert "4508421" in summary, "MJCF basis must be recorded"

    calib = (tmp_path / "out" / "p1_threshold_calibration.md").read_text(
        encoding="utf-8")
    assert calib.count("须重生成") >= 10, "smoke4/5 gates flagged for regen"
    assert "4508421" in calib
    assert "N/A" in calib, "missing distribution data must read N/A"
    assert "实测分布" in calib and "低样本" in calib, "data-grade legend in use"
    assert "自校准" in calib, "smoke6 circular thresholds must be flagged"
    # the currently failing gate carries its measured worst case
    assert "0.0859" in calib or "0.086" in calib, (
        "smoke5 palm deviation worst case must appear verbatim")
