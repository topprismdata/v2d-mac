"""Stamp report.json with the scene this tree ACTUALLY ships.

grounding/envs/make_report.py imports (mjcf_commit, xml_sha256_declared) from
grounding/collect_clean_demos.py, whose constants went stale at c7c6dac (the
chamfer commit re-derived the XML without bumping them).  make_report itself
hashes the live file into xml_sha256 - that key is correct - but the two
constant-derived keys would sit beside it claiming 1b0f13f / 1ea0fdf..., i.e.
false provenance on the round's headline artifact.

This script runs make_report and then re-stamps the stale keys from the
authoritative source (git HEAD blob sha256, same caliber as
`git show HEAD:grounding/assets/allegro_floating.xml | shasum -a 256`),
recording what was corrected so the stamp is auditable.

    runs/.venv-mj/bin/python runs/p0/smoke4_mujoco_ppo/stamp_report_provenance.py
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[3]
RUN_DIR = REPO / "runs/p0/smoke4_mujoco_ppo"
REPORT = RUN_DIR / "report.json"


def main() -> None:
    subprocess.run([sys.executable, str(REPO / "grounding/envs/make_report.py")],
                   cwd=REPO, check=True)
    xml_sha = hashlib.sha256(subprocess.run(
        ["git", "show", "HEAD:grounding/assets/allegro_floating.xml"],
        cwd=REPO, check=True, capture_output=True).stdout).hexdigest()
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                            cwd=REPO, check=True,
                            capture_output=True).stdout.decode().strip()
    rep = json.loads(REPORT.read_text())
    corrected = {
        "mjcf_commit": {"from": rep.get("mjcf_commit"), "to": commit},
        "xml_sha256_declared": {"from": rep.get("xml_sha256_declared"),
                                "to": xml_sha},
    }
    rep["mjcf_commit"] = commit
    rep["xml_sha256_declared"] = xml_sha
    rep["provenance_stamp"] = {
        "corrected_by": "runs/p0/smoke4_mujoco_ppo/stamp_report_provenance.py",
        "corrections": corrected,
        "reason": ("collect_clean_demos.MJCF_COMMIT/xml_sha256_declared went "
                   "stale at c7c6dac (chamfer + 9dp masses); the live blob "
                   "hash is the authority"),
        "xml_sha256_matches_working_tree": xml_sha == hashlib.sha256(
            (REPO / "grounding/assets/allegro_floating.xml").read_bytes()).hexdigest(),
    }
    if rep["verdict"] != "pass":
        red = [k for k, v in rep["criteria"].items() if not v["pass"]]
        rep["failure_mode"] = {
            "red_criteria": red,
            "classification": (
                "palm-launch separation. The policy opens every episode by "
                "pressing the palm on the box's -x top-edge chamfer ramp "
                "(object_chamfer_negx|palm_collision 17-25 N), which converts "
                "the push into a +x slide (~11 cm across the pedestal) and a "
                "~30 mm vertical flick; the box's first supported->free "
                "transition (t~=16) therefore happens under PALM-only contact, "
                "so A3c(iii)'s finger-contact precondition reads False. The "
                "thumb then catches the box and carries it to 53.5-58.0 mm "
                "(A3b clean 10/10, pedestal 0 at peak). Bearing at peak: wall "
                "-11.1 box-weights (down-press), hook 0, pedestal 0 - the box "
                "coasts ballistically at its peak. For calibration, the "
                "gate-clean demo trajectories themselves read wall -1.2..-1.7 "
                "box-weights at peak (also ballistic, mildly), so the "
                "mechanism family (-x wall, no pedestal) matches the demos; "
                "the deviation is the palm-launch preamble and a 7x harder "
                "down-press at peak. The reward cannot see A3c(iii): the "
                "flick front-loads held-altitude reward, so PPO converges to "
                "it from the BC warm start regardless of exploration noise "
                "(run ledger: runs/p0/smoke4_mujoco_ppo/p0_4_run_ledger.json)."
            ),
        }
    REPORT.write_text(json.dumps(rep, indent=2))
    print(json.dumps({"mjcf_commit": commit, "xml_sha256": xml_sha,
                      "verdict": rep["verdict"]}, indent=2))


if __name__ == "__main__":
    main()
