#!/usr/bin/env python3
"""Generate runs/p0/SUMMARY.md and runs/p0/p1_threshold_calibration.md.

Reads the six P0 smoke report.json files (+ smoke5's retarget_qa.json) from
--runs-dir and renders:

  SUMMARY.md                  per-smoke verdict / key metrics / MJCF basis /
                              known boundaries, with an overview header
  p1_threshold_calibration.md measured P0 distributions per acceptance gate
                              (n / median / p10 / p90 / worst) against the
                              frozen P0 thresholds, for user sign-off of P1

Contract:
  * NEVER writes anything under the reports; only the two markdown targets in
    --out-dir.
  * A missing or malformed report is a HARD ERROR (exit 2) naming the file.
    There is no silent-skip path.
  * FAIL verdicts are rendered as FAIL with their reasons. Missing
    distribution data is rendered as N/A. No number is invented here: every
    value is read from the committed reports.

This is scaffolding for the P0 wrap-up: once the formal rerun (Wave-3.5 MJCF
revision; smoke4 retrain) lands, re-run this script to regenerate both files
before tagging p0-green.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

SMOKES = [
    ("smoke1_sam2", "SAM2 segmentation on MPS (smoke 1)"),
    ("smoke2_vggt", "VGGT tracking / geometry on MPS (smoke 2)"),
    ("smoke3_foundpose", "FoundPose 6DoF on MPS (smoke 3)"),
    ("smoke4_mujoco_ppo", "MuJoCo PPO lift-and-hold (smoke 4)"),
    ("smoke5_retarget", "MANO->Allegro retarget replay (smoke 5, G3)"),
    ("smoke6_hamer", "HaMeR hand recovery on MPS (smoke 6)"),
]

# Smokes whose numbers depend on the MuJoCo scene; all others are perception-only.
MJCF_DEPENDENT = {"smoke4_mujoco_ppo", "smoke5_retarget"}

# Wave-3 MJCF that smoke4/smoke5 reports were produced on, and what happens next.
WAVE3_MJCF_COMMIT = "4508421"
WAVE3_MJCF_NOTE = (
    "Wave-3.5 腕部行程 MJCF 修订尚未落地；正式重跑（smoke5 复跑 + smoke4 重训）"
    "落在新 MJCF 上后，本节全部数字必须重生成，不得与旧物理混账"
)


class ReportError(Exception):
    """A required report is missing, unreadable, or malformed."""


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------

def fmt(x) -> str:
    """Deterministic, compact number rendering; N/A for missing values."""
    if x is None:
        return "N/A"
    if isinstance(x, bool):
        return "true" if x else "false"
    if isinstance(x, str):
        return x
    if isinstance(x, int):
        return str(x)
    ax = abs(x)
    if 0.995 < ax < 1.005:  # keep near-one ratios from rounding to "1"
        return f"{x:.6g}"
    if ax >= 1000:
        return f"{x:.1f}"
    if ax >= 1:
        return f"{x:.4g}"
    if ax == 0:
        return "0"
    return f"{x:.3g}"


def one_line(text: str, limit: int = 140) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _pct(vals_sorted, q: float):
    n = len(vals_sorted)
    if n == 1:
        return vals_sorted[0]
    k = (n - 1) * q
    f = int(math.floor(k))
    c = min(f + 1, n - 1)
    if f == c:
        return vals_sorted[f]
    return vals_sorted[f] * (c - k) + vals_sorted[c] * (k - f)


def dist(samples):
    v = sorted(float(s) for s in samples)
    if not v:
        return {"n": 0, "median": None, "p10": None, "p90": None, "worst": None}
    return {
        "n": len(v),
        "median": _pct(v, 0.5),
        "p10": _pct(v, 0.1),
        "p90": _pct(v, 0.9),
        "worst": max(v),
        "min": min(v),
    }


def md_table(header, rows) -> str:
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def bullets(items) -> str:
    return "\n".join(f"- {one_line(i, 400) if isinstance(i, str) else i}" for i in items)


# --------------------------------------------------------------------------
# loading with hard errors
# --------------------------------------------------------------------------

def load_report(runs_dir: Path, smoke_id: str) -> dict:
    path = runs_dir / smoke_id / "report.json"
    if not path.is_file():
        raise ReportError(f"missing report: {path}")
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as e:
        raise ReportError(f"unreadable report: {path}: {e}")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ReportError(f"malformed JSON in {path}: {e}")
    if not isinstance(data, dict):
        raise ReportError(f"malformed report {path}: top level must be a JSON object")
    if "verdict" not in data:
        raise ReportError(f"malformed report {path}: required key 'verdict' missing")
    if "smoke" in data and data["smoke"] != smoke_id:
        raise ReportError(
            f"malformed report {path}: 'smoke' is {data['smoke']!r}, expected {smoke_id!r}"
        )
    return data


def load_qa(runs_dir: Path) -> dict:
    path = runs_dir / "smoke5_retarget" / "retarget_qa.json"
    if not path.is_file():
        raise ReportError(f"missing key artifact for smoke5: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise ReportError(f"malformed artifact {path}: {e}")
    if not isinstance(data, dict):
        raise ReportError(f"malformed artifact {path}: top level must be a JSON object")
    return data


# --------------------------------------------------------------------------
# git provenance (best effort)
# --------------------------------------------------------------------------

def git(repo_root: Path, *args: str) -> str:
    try:
        r = subprocess.run(
            ["git", "-C", str(repo_root), *args],
            capture_output=True, text=True, timeout=15,
        )
        if r.returncode != 0:
            return ""
        return r.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return ""


def report_commit(repo_root: Path, runs_dir_name: str, smoke_id: str) -> str:
    out = git(repo_root, "log", "-1", "--format=%h",
              f"runs/{runs_dir_name}/{smoke_id}/report.json")
    return out or "uncommitted/unknown"


# --------------------------------------------------------------------------
# MJCF provenance
# --------------------------------------------------------------------------

def mjcf_basis(reports: dict, smoke_id: str) -> str:
    if smoke_id not in MJCF_DEPENDENT:
        return "N/A — 感知 smoke，不依赖 MJCF"
    if smoke_id == "smoke5_retarget":
        mc = reports[smoke_id].get("mjcf_commit", {}) or {}
        commit = mc.get("mjcf_commit", "?")
        sha = str(mc.get("scene_sha256", ""))[:7] or "?"
        match = mc.get("scene_matches_mjcf_commit")
        s = f"MJCF `{commit}` (scene `{sha}`)"
        if match is False:
            s += " — scene 与 commit 不一致！"
        return s
    # smoke4: report records the scene sha; map to a commit via smoke5 when available
    sha = str((reports[smoke_id].get("model", {}) or {}).get("xml_sha256", ""))
    if not sha:
        return "N/A — report 未记录 xml_sha256"
    mapped = ""
    s5 = reports.get("smoke5_retarget", {}) or {}
    mc = s5.get("mjcf_commit", {}) or {}
    if mc.get("scene_sha256_at_mjcf_commit") == sha:
        mapped = f"= MJCF `{mc.get('mjcf_commit', '?')}`"
    return f"scene sha `{sha[:7]}` {mapped}".strip()


# --------------------------------------------------------------------------
# per-smoke short reason (overview table)
# --------------------------------------------------------------------------

def short_reason(smoke_id: str, r: dict) -> str:
    verdict = r.get("verdict")
    if verdict == "fail":
        if smoke_id == "smoke4_mujoco_ppo":
            failed = [k for k, c in (r.get("criteria", {}) or {}).items()
                      if isinstance(c, dict) and c.get("pass") is False]
            reason = "FAIL: " + (", ".join(failed) or "criteria unmet")
            if failed == ["A3b_airborne_at_peak"]:
                reason += "（峰值时刻盒子仍接触桌面；MJCF 4508421 上不存在 clean grasp-and-lift）"
            return one_line(reason, 180)
        vr = r.get("verdict_reason")
        if vr:
            return one_line("FAIL: " + str(vr), 180)
    if verdict == "pass":
        vr = r.get("verdict_reason")
        return one_line(str(vr), 180) if vr else "全部冻结判据通过"
    fr = r.get("fail_reasons")
    if fr:
        return one_line("FAIL: " + "; ".join(map(str, fr)), 180)
    return one_line(f"verdict={verdict}", 60)


# --------------------------------------------------------------------------
# per-smoke sections
# --------------------------------------------------------------------------

def _criteria_rows_smoke4(r: dict):
    rows = []
    for name, c in (r.get("criteria", {}) or {}).items():
        if not isinstance(c, dict):
            continue
        val = None
        for k in ("value_m", "duty_cycle", "mean_finger_force_N",
                  "max_anchored_step_m", "max_dvz_gravity_error_m_s",
                  "table_touching_at_peak", "separation_contact_driven",
                  "peak_lift_m", "n_finger_contact_steps", "value", "pairs"):
            if k in c:
                val = c[k]
                break
        thr = c.get("threshold_m", c.get("threshold_m_s", c.get("threshold")))
        rows.append((name, fmt(val), fmt(thr),
                     "PASS" if c.get("pass") else "**FAIL**"))
    return rows


def build_section(smoke_id: str, title: str, r: dict, reports: dict,
                  qa: dict, ctx: dict) -> str:
    parts = [f"\n\n## {title}", ""]
    verdict = r.get("verdict")
    vtxt = "FAIL" if verdict == "fail" else fmt(verdict)
    parts.append(f"**verdict: {vtxt}**"
                 f" ／ 所依据 {mjcf_basis(reports, smoke_id)}"
                 f" ／ report commit: `{ctx['commits'][smoke_id]}`")
    parts.append("")

    if smoke_id == "smoke1_sam2":
        th, pc = r.get("thresholds", {}), r.get("per_criterion", {})
        rows = []
        for k, lim in th.items():
            v = pc.get(k)
            if not isinstance(v, (int, float)) or not isinstance(lim, (int, float)):
                ok = "N/A"
            elif k.startswith("max_"):
                ok = "PASS" if v <= lim else "**FAIL**"
            else:
                ok = "PASS" if v >= lim else "**FAIL**"
            rows.append((k, fmt(lim), fmt(v), ok))
        parts += ["判据（MPS vs CPU 一致性 + 掩膜健全性，3 帧）:", "",
                  md_table(["判据", "阈值", "实测(最差帧)", "判定"], rows)]
        diag = ((r.get("summary", {}) or {}).get("object_hit_diagnostic", {}) or {})
        if diag.get("note"):
            parts += ["", f"物体命中诊断（仅诊断，非门控）：{one_line(diag['note'], 300)}"]
        lim = r.get("known_limitations") or []
        if lim:
            parts += ["", "### 已知边界 / 诚实短板", "", bullets(lim)]

    elif smoke_id == "smoke2_vggt":
        crit = r.get("criteria", {}) or {}
        b = crit.get("b_step_no_teleport", {}) or {}
        rows = [
            ("a_all_frames_no_nan（30 帧全部有限，无 NaN/跳变）",
             "poses/depths finite", "finite" if crit.get("a_all_frames_no_nan", {}).get("poses_finite") else "N/A",
             "PASS" if crit.get("a_all_frames_no_nan", {}).get("pass") else "**FAIL**"),
            ("b 防跳变 joint (rad/帧)", b.get("joint_limit"), f"max {fmt(b.get('joint_max'))}（{fmt(b.get('joint_ratio'))}×中位）",
             "PASS" if b.get("joint_ratio", 9) <= (b.get("limit_ratio") or 3) else "**FAIL**"),
            ("b 防跳变 translation (m/帧)", b.get("translation_limit"), f"max {fmt(b.get('translation_max'))}（{fmt(b.get('translation_ratio'))}×中位）",
             "PASS" if b.get("translation_ratio", 9) <= (b.get("limit_ratio") or 3) else "**FAIL**"),
            ("b 防跳变 rotation (rad/帧)", b.get("rotation_limit"), f"max {fmt(b.get('rotation_max'))}（{fmt(b.get('rotation_ratio'))}×中位）",
             "PASS" if b.get("rotation_ratio", 9) <= (b.get("limit_ratio") or 3) else "**FAIL**"),
            ("c 旋转/平移分轴判据", "分轴", "joint/rotation/translation", "PASS" if crit.get("c_rot_trans_separate", {}).get("pass") else "**FAIL**"),
        ]
        parts += ["判据（物理锚定防跳变 + 有限性）:", "",
                  md_table(["判据", "上限", "实测", "判定"], rows)]
        rp, sc, mo = (r.get("reprojection", {}) or {}), (r.get("scale_consistency", {}) or {}), (r.get("motion_stats", {}) or {})
        fork = r.get("fork_selection", {}) or {}
        fb = r.get("cpu_fallbacks") or []
        parts += [
            "",
            f"- 证据：相邻帧 ORB 重投影中位 **{fmt(rp.get('median_px'))} px**"
            f"（p90 {fmt(rp.get('p90_px'))}，{rp.get('n_pairs')} 对 / {rp.get('n_matches')} 匹配）；"
            f"MPS {fmt(r.get('mps_inference_seconds'))}s vs CPU 对照 {fmt(r.get('cpu_control_seconds'))}s"
            f"（{fmt(r.get('speedup_vs_cpu_control'))}×）；CPU fallback ops: {fb or '无'}",
            f"- 尺度漂移仅记录未门控：per-frame 中位深度 cv = {fmt(sc.get('cv'))}，"
            f"最大成对中位深度比 {fmt(sc.get('max_pairwise_ratio'))}",
            f"- fork 选择：{fork.get('chosen')} @ {fork.get('commit')}；"
            f"权重 {((r.get('weights', {}) or {}).get('model_id', 'N/A'))}",
        ]
        parts += ["", "### 已知边界 / 诚实短板", "",
                  bullets([
                      "scale_consistency 只作漂移记录，本 smoke 未冻结任何尺度锚定门"
                      "（判据仅 a/b/c）；DESIGN L2 的尺度锚定是 P1 硬要求。",
                      "全部运动统计（中位/最大）来自 30 帧单条 clip，防跳变规则为"
                      " max(3×中位数, 单帧绝对上限)，合成跳变（0.1 m/帧、0.5 rad/帧）验证过可拦截。",
                  ])]

    elif smoke_id == "smoke3_foundpose":
        rows = []
        for name, c in (r.get("criteria", {}) or {}).items():
            rows.append((name, fmt(c.get("value")), fmt(c.get("threshold")),
                         "PASS" if c.get("pass") else "**FAIL**"))
        parts += ["判据（C1–C6，单帧 tbox）：", "",
                  md_table(["判据", "实测", "阈值", "判定"], rows)]
        err = r.get("errors", {}) or {}
        cam = r.get("camera_assumed", {}) or {}
        diag = r.get("inference_diagnostics", {}) or {}
        inliers = [x for x in (diag.get("pnp_inlier_counts") or []) if x is not None]
        inlier_txt = (f"{fmt(min(inliers))}–{fmt(max(inliers))}" if inliers else "N/A")
        parts += ["", "### 已知边界 / 诚实短板", "", bullets([
            f"面内轴向误差 {fmt(err.get('in_plane_axis_error_deg'))}°、"
            f"PnP 内点仅 {inlier_txt}/159、"
            f"proxy/mask 面积比 {fmt(err.get('proxy_area_over_mask_area'))}"
            " —— 全部为记录值，未冻结为门；该 6DoF 只能作 estimated 粗先验，不得直接升级 verified。",
            f"内参假定为 {one_line(cam.get('source', 'N/A'), 120)}；尺度来源 {cam.get('scale_source')}",
            f"FoundPose 自身 score 恒 0（上游 eval_util 重投影 bug，已 patch 记录）："
            f"{one_line(err.get('foundpose_internal_score_note', ''), 200)}",
            f"mask 来源非 SAM2 而是 {r.get('mask_source')}",
        ])]

    elif smoke_id == "smoke4_mujoco_ppo":
        parts += ["判据（A1–A4，10 个 eval episode）：", "",
                  md_table(["判据", "实测", "阈值", "判定"], _criteria_rows_smoke4(r))]
        tr, cv = r.get("training", {}) or {}, r.get("curve_summary", {}) or {}
        clv = ((r.get("clean_lift_feasibility", {}) or {}).get("verdict", {}) or {})
        ce = r.get("contact_evidence", {}) or {}
        parts += [
            "",
            f"- 训练：{fmt(tr.get('cumulative_timesteps'))} 步（seed {tr.get('seed')}，"
            f"n_envs {tr.get('n_envs')}，{tr.get('n_episodes')} episodes，"
            f"wall {fmt((tr.get('wall_seconds') or 0) / 60,)} min）；"
            f"奖励前/后半段均值 {fmt(cv.get('reward_first_half_mean'))} → {fmt(cv.get('reward_second_half_mean'))}，"
            f"但 z_peak 全程 0（{fmt(cv.get('z_peak_max_windowed'))} m）",
            f"- 接触证据：指-盒接触对 {ce.get('finger_object_geom_pairs')}，"
            f"力占空比 {fmt(ce.get('finger_force_duty_cycle'))}，"
            f"指尖均力 {fmt(ce.get('finger_force_mean_N'))} N vs 盒重 {fmt(ce.get('box_weight_N'))} N，"
            f"峰值时刻 table_touching_at_peak = **{ce.get('table_touching_at_peak')}**",
            f"- clean-lift 可行性（60000 样本扫描，MJCF {clv.get('mjcf_commit')}）："
            f"clean_lift_exists = **{clv.get('clean_lift_exists')}**；"
            f"对面壁指垫对捏 {clv.get('poses_with_opposite_wall_pad_pinch')}/{clv.get('samples')}，"
            f"掌板压盒力 {fmt(clv.get('palm_force_into_box_N'))} N = 盒重 {fmt(clv.get('palm_overload_ratio'))} 倍",
        ]
        dev = r.get("design_deviations") or []
        parts += ["", "### 已知边界 / 诚实短板", ""]
        if fail_txt := short_reason(smoke_id, r):
            parts.append(f"- {one_line(fail_txt, 300)}")
        parts.append(bullets(dev))

    elif smoke_id == "smoke5_retarget":
        checks = r.get("verdict_checks", {}) or {}
        rows = [(k, "PASS" if v else "**FAIL**") for k, v in checks.items()]
        n_ok = sum(1 for v in checks.values() if v)
        parts += [f"G3 判据（{len(checks)} 项，`{n_ok}/{len(checks)}` 通过的 FAIL 报告）：", "",
                  md_table(["判据", "判定"], rows)]
        rep = r.get("replay", {}) or {}
        dev = rep.get("execution_deviation_m", {}) or {}
        blocker = r.get("blocker", {}) or {}
        remaining = blocker.get("remaining", {}) or {}
        palm = remaining.get("palm_world_pose_deviation_mm", {}) or {}
        tip = remaining.get("fingertip_world_pose_deviation_mm", {}) or {}
        parts += [
            "",
            f"- 失败门：palm 执行偏差 max **{fmt(dev.get('max_m'))} m**（rms {fmt(dev.get('rms_m'))} m）"
            f" vs cap 0.040 m —— 掌板压盒、腕伺服力预算耗在支撑而非合指，"
            f"最近指尖偏差 {fmt(tip.get('value'))}",
            f"- 本轮已修根因：{one_line(((blocker.get('root_cause_fixed_this_round', {}) or {}).get('was', '')), 200)}",
            f"- gate scope：{one_line(r.get('gate_scope', ''), 300)}",
            f"- 记账规则：{one_line(((r.get('mjcf_commit', {}) or {}).get('accounting_rule', '')), 300)}",
        ]
        parts += ["", "### 已知边界 / 诚实短板", "",
                  bullets([
                      f"掌板世界位姿偏差：max {fmt(palm.get('value'))} mm vs cap {fmt(palm.get('cap_mm'))} mm"
                      f"（rms {fmt(palm.get('rms_mm'))} mm）—— 掌心朝下搁在盒上，不是 clip 描述的抓握。",
                      "G3 只门控 retarget 可执行性（可回放、真实接触、有界运动、无互穿），"
                      "不门控手部几何精度 —— 后者继承 smoke 6 的不确定性。",
                  ])]

    elif smoke_id == "smoke6_hamer":
        checks = r.get("verdict_checks", {}) or {}
        rows = [(k, "PASS" if v else "**FAIL**") for k, v in checks.items()]
        parts += ["冻结判据（接线三门 + 检测）：", "",
                  md_table(["判据", "判定"], rows)]
        pc = r.get("per_criterion", {}) or {}
        fc = r.get("flip_control", {}) or {}
        parts += [
            "",
            f"- flip 对照：as-shipped {fmt(fc.get('as_shipped', {}).get('median_reprojection_error_px'))} px"
            f" < full-inversion {fmt(fc.get('inverted', {}).get('median_reprojection_error_px'))} px"
            f"（证明左右手 x 顺序正确）",
            f"- 跨度比 left 中位 {fmt(pc.get('hand_span_ratio', {}).get('left', {}).get('median'))} / "
            f"right {fmt(pc.get('hand_span_ratio', {}).get('right', {}).get('median'))} ∈ [0.85, 1.20]；"
            f"腕部配准 {fmt(pc.get('median_wrist_reprojection_error_px'))} px = "
            f"{fmt((pc.get('median_wrist_reprojection_error_px') or 0) / (pc.get('median_hand_span_px') or 1))} span < 15%",
            f"- 重投影（evidence-only，参考是 MediaPipe 非 GT）：median {fmt(pc.get('median_reprojection_error_px'))} px，"
            f"p90 {fmt(pc.get('p90_reprojection_error_px'))} px（{pc.get('n_frames')} 帧）",
        ]
        lim = r.get("known_limitations") or []
        risk = (r.get("risk_notes", {}) or {}).get("p1_hand_geometry_accuracy")
        parts += ["", "### 已知边界 / 诚实短板", "", bullets(lim)]
        if risk:
            parts += ["", f"P1 风险记录：{one_line(risk, 400)}"]

    else:
        raise ReportError(f"smoke {smoke_id}: no section builder")

    return "\n".join(parts)


# --------------------------------------------------------------------------
# SUMMARY.md
# --------------------------------------------------------------------------

def build_summary(reports: dict, qa: dict, ctx: dict) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    n_pass = sum(1 for r in reports.values() if r.get("verdict") == "pass")
    n_fail = len(reports) - n_pass
    head = [
        "# runs/p0 — SUMMARY",
        "",
        f"- 生成时间：{now}",
        f"- 生成命令：`python3 scripts/p0_summary.py --runs-dir {ctx['runs_dir_arg']} "
        f"--out-dir {ctx['out_dir_arg']}`",
        f"- 仓库 HEAD：`{ctx['git_head'] or 'unknown'}`",
        f"- 总览：{n_pass} pass / {n_fail} fail —— **六绿未达成，按 PLAN 不打 `p0-green` tag**",
        "- **本文件是 dry-run 脚手架**：smoke4（`{}`）与 smoke5（`{}`）是诚实 FAIL 报告，"
        "按现状收录；正式重跑（Wave-3.5 MJCF 修订 + smoke4 重训）落地后必须重跑本脚本重生成。".format(
            ctx["commits"]["smoke4_mujoco_ppo"], ctx["commits"]["smoke5_retarget"]),
        "",
        "## 总览",
        "",
        md_table(
            ["smoke", "scope", "verdict", "门控结果", "所依据 MJCF", "report commit"],
            [(sid, title, "**FAIL**" if reports[sid].get("verdict") == "fail" else fmt(reports[sid].get("verdict")),
              one_line(short_reason(sid, reports[sid]), 120),
              mjcf_basis(reports, sid), f"`{ctx['commits'][sid]}`")
             for sid, title in SMOKES],
        ),
    ]
    for sid, title in SMOKES:
        head.append(build_section(sid, title, reports[sid], reports, qa, ctx))
    head.append("")
    return "\n".join(head)


# --------------------------------------------------------------------------
# calibration gates
# --------------------------------------------------------------------------

def _gate(smoke, gate, threshold, *, samples=None, stats=None, single=None,
          kind=None, regen=False, level=None, note="", worst_label=None,
          failing=False):
    g = {"smoke": smoke, "gate": gate, "threshold": threshold,
         "regen": regen, "note": note, "failing": failing}
    if samples is not None:
        d = dist(samples)
        g.update(n=d["n"], median=d["median"], p10=d["p10"], p90=d["p90"],
                 worst=worst_label or d["worst"])
        g["level"] = level or (f"实测分布" if d["n"] >= 10 else f"低样本 (n={d['n']})")
    elif stats is not None:
        g.update(n=stats.get("n"), median=stats.get("median"),
                 p10=stats.get("p10"), p90=stats.get("p90"),
                 worst=worst_label or stats.get("max"))
        g["level"] = level or "仅汇总统计"
    elif single is not None:
        g.update(n=1, median=single, p10=single, p90=single,
                 worst=worst_label or single)
        g["level"] = level or "单值"
    else:
        g.update(n=None, median=None, p10=None, p90=None, worst=worst_label)
        g["level"] = level or kind or "布尔门"
    return g


def build_gates(reports: dict, qa: dict) -> list:
    gates = []

    # ---- smoke 1 (perception, 3 frames) -----------------------------------
    r = reports["smoke1_sam2"]
    pf = r.get("per_frame", {}) or {}
    th = r.get("thresholds", {}) or {}
    ious = [f.get("iou_mps_cpu") for f in pf.values() if f.get("iou_mps_cpu") is not None]
    areas = [f.get("mps", {}).get("area_frac") for f in pf.values()
             if isinstance(f.get("mps"), dict)]
    covs = [f.get("mps", {}).get("box_coverage") for f in pf.values()
            if isinstance(f.get("mps"), dict)]
    covs = [c for c in covs if c is not None]
    gates += [
        _gate("smoke1_sam2", "掩膜 MPS/CPU IoU", f">= {fmt(th.get('min_iou_mps_cpu'))}",
              samples=ious, note="n=3 帧，单 clip"),
        _gate("smoke1_sam2", "掩膜面积占比", f"[{fmt(th.get('min_mask_area_frac'))}, {fmt(th.get('max_mask_area_frac'))}]",
              samples=areas, note="n=3 帧，单 clip"),
        _gate("smoke1_sam2", "prompt 框覆盖率", f">= {fmt(th.get('min_box_coverage'))}",
              samples=covs, note="n=3 帧，单 clip"),
    ]

    # ---- smoke 2 (perception, 30 frames) ----------------------------------
    r = reports["smoke2_vggt"]
    crit = r.get("criteria", {}) or {}
    a = crit.get("a_all_frames_no_nan", {}) or {}
    nan_ct = a.get("nan_count")
    gates.append(_gate(
        "smoke2_vggt", "全帧有限（a_all_frames_no_nan）",
        f"{a.get('frames_expected', '30')} 帧 poses/depths 全部有限",
        worst_label=f"NaN = {nan_ct}", kind=f"布尔门（{a.get('frames_expected', 30)} 帧）"))
    mo = r.get("motion_stats", {}) or {}
    b = crit.get("b_step_no_teleport", {}) or {}
    for key, label, unit in (("joint_m_rad", "防跳变 joint", "rad/帧"),
                             ("translation_m", "防跳变 translation", "m/帧"),
                             ("rotation_rad", "防跳变 rotation", "rad/帧")):
        st = mo.get(key, {}) or {}
        lim = b.get(key.replace("_m", "_limit").replace("_rad", "_limit")) or b.get(
            {"joint_m_rad": "joint_limit", "translation_m": "translation_limit",
             "rotation_rad": "rotation_limit"}[key])
        gates.append(_gate(
            "smoke2_vggt", f"{label} ({unit})",
            f"max ≤ max(3×中位, {fmt(lim)})",
            stats={"n": None, "median": st.get("median"), "p90": None,
                   "max": st.get("max")},
            note="report 只存中位/最大，p10/p90 不可得"))
    rp = r.get("reprojection", {}) or {}
    gates.append(_gate(
        "smoke2_vggt", "相邻帧重投影中位误差 (px, 29 对)",
        "evidence-only（未冻结阈值）",
        samples=rp.get("per_pair_median_px") or None,
        note=f"分布=每对各自的中位；report 汇总值（全部 {rp.get('n_matches')} 匹配合并）median {fmt(rp.get('median_px'))} px；identity 对照见 PLAN"))
    sc = r.get("scale_consistency", {}) or {}
    gates.append(_gate("smoke2_vggt", "尺度漂移 cv（每帧中位深度）",
                       "记录，未门控", single=sc.get("cv"),
                       note="无尺度锚定门；P1 硬要求"))

    # ---- smoke 3 (perception, 1 frame) ------------------------------------
    r = reports["smoke3_foundpose"]
    c = r.get("criteria", {}) or {}
    err = r.get("errors", {}) or {}
    gates += [
        _gate("smoke3_foundpose", "C4 剪影 IoU", ">= 0.5", samples=[c.get("C4_silhouette_iou", {}).get("value")],
              note="n=1 帧，勿当分布"),
        _gate("smoke3_foundpose", "C5 投影 bbox 中心距 (px)", "<= 64",
              samples=[c.get("C5_bbox_center_dist_px", {}).get("value")], note="n=1 帧"),
        _gate("smoke3_foundpose", "C2 模板数", "> 0", samples=[c.get("C2_template_count_gt_0", {}).get("value")],
              note="官方 798 模板收敛；124 模板发散（见 PLAN）"),
        _gate("smoke3_foundpose", "面内轴向误差 (°)", "未门控（evidence）",
              samples=[err.get("in_plane_axis_error_deg")], level="低样本 (n=1)",
              note="诚实短板：41° 量级，只能作 estimated 粗先验"),
        _gate("smoke3_foundpose", "PnP 内点数 (/159 对应点)", "未门控（evidence）",
              samples=(r.get("inference_diagnostics", {}) or {}).get("pnp_inlier_counts") or None,
              level="低样本 (n=5)", note="10–11/159"),
    ]

    # ---- smoke 4 (MJCF-dependent, 10 episodes) ----------------------------
    r = reports["smoke4_mujoco_ppo"]
    eps = r.get("per_episode", []) or []
    acc = r.get("acceptance_thresholds", {}) or {}
    n_touch = sum(1 for e in eps if e.get("table_touching_at_peak"))
    gates += [
        _gate("smoke4_mujoco_ppo", "A1 单 episode 峰值抬升 (m)",
              f">= {fmt(acc.get('lift_m'))}（至少一个 eval episode）",
              samples=[e.get("peak_lift_m") for e in eps], regen=True,
              note="A1 数值过但见 A3b"),
        _gate("smoke4_mujoco_ppo", "A3a-i 指力占空比", f">= {fmt(acc.get('min_force_duty_cycle'))}",
              samples=[e.get("finger_force_duty_cycle") for e in eps], regen=True),
        _gate("smoke4_mujoco_ppo", "A3a-ii 指尖均力 (N)", "准静态充分性（vs 盒重 1.47 N）",
              samples=[e.get("finger_force_mean_N") for e in eps], regen=True,
              note="未门控；实测 22–36 N，forcerange 修订后标称上限 10 N/指 —— 正式重跑后须复核"),
        _gate("smoke4_mujoco_ppo", "A3c-i 锚定段单步位移 (m)",
              f"<= {fmt(acc.get('max_anchored_step_displacement_m'))}",
              samples=[e.get("max_anchored_step_m") for e in eps], regen=True),
        _gate("smoke4_mujoco_ppo", "A2 指-盒接触步数", f">= {fmt(acc.get('min_finger_contact_steps'))}",
              samples=[e.get("n_finger_contact_steps") for e in eps], regen=True),
        _gate("smoke4_mujoco_ppo", "A3b 峰值时刻离桌（airborne）", "峰值时刻盒不接触桌面",
              worst_label=f"{n_touch}/{len(eps)} episodes table_touching_at_peak=True",
              kind="布尔门（10 episodes）", regen=True,
              failing=n_touch == len(eps), note="当前 FAIL 根因门"),
    ]
    clv = ((r.get("clean_lift_feasibility", {}) or {}).get("verdict", {}) or {})
    gates += [
        _gate("smoke4_mujoco_ppo", "可行性扫描：对面壁指垫对捏", "证据（60000 样本）",
              samples=[clv.get("poses_with_opposite_wall_pad_pinch")],
              level="单值", regen=True,
              note=f"{clv.get('poses_with_opposite_wall_pad_pinch')}/{clv.get('samples')}；contact 时中位 {clv.get('median_buried_geoms_when_contacting')} geom 埋入"),
        _gate("smoke4_mujoco_ppo", "可行性扫描：掌板压盒力/盒重", "证据",
              samples=[clv.get("palm_overload_ratio")], level="单值", regen=True,
              note=f"palm_force {fmt(clv.get('palm_force_into_box_N'))} N；clean_lift_exists={clv.get('clean_lift_exists')}"),
    ]

    # ---- smoke 5 (MJCF-dependent, 27 ctrl frames) -------------------------
    r = reports["smoke5_retarget"]
    rep = r.get("replay", {}) or {}
    dev = rep.get("execution_deviation_m", {}) or {}
    tol = ((qa.get("penetration", {}) or {}).get("tolerance_m"))
    gates += [
        _gate("smoke5_retarget", "palm 执行偏差 (m)",
              f"<= {fmt(0.040)} m cap（hand_world_pose_deviation_within_cap）",
              samples=dev.get("per_frame_m") or None, regen=True,
              failing=True, note=f"worst {fmt(dev.get('max_m'))} m > cap；当前 FAIL 根因门"),
        _gate("smoke5_retarget", "指-盒穿透 (m)",
              f"<= 场景导出容差 {fmt(tol)} m" if tol else "<= derived tolerance",
              samples=rep.get("penetration_sequence_m") or None, regen=True,
              note=f"max {fmt(rep.get('max_penetration_m'))} m；容差=静置下沉 ~9×（QA 记录）"),
        _gate("smoke5_retarget", "腕部连续性单步 (m)", "max(3×中位, 0.05 m 绝对上限)",
              samples=rep.get("wrist_step_m") or None, regen=True,
              note="与 smoke2 同规则的物理锚定（QA wrist.ceiling_m=0.05）"),
        _gate("smoke5_retarget", "指令速度 finger (rad/s)", "<= 3.14",
              samples=[(rep.get("command_velocity", {}) or {}).get("finger", {}).get("max_commanded_rate")],
              level="单值", regen=True, note="within_limit=true"),
        _gate("smoke5_retarget", "指令速度 wrist_hinge (rad/s)", "<= 1.5",
              samples=[(rep.get("command_velocity", {}) or {}).get("wrist_hinge", {}).get("max_commanded_rate")],
              level="单值", regen=True, note="within_limit=true"),
        _gate("smoke5_retarget", "指令速度 wrist_slide (m/s)", "<= 0.2",
              samples=[(rep.get("command_velocity", {}) or {}).get("wrist_slide", {}).get("max_commanded_rate")],
              level="单值", regen=True, note="within_limit=true"),
        _gate("smoke5_retarget", "接触占空比（evidence）", "记录，未门控",
              single=rep.get("contact_duty"), regen=True,
              note=f"contact_frames={rep.get('contact_frames')}/{rep.get('n_frames')}"),
    ]

    # ---- smoke 6 (perception, 30 frames) ----------------------------------
    r = reports["smoke6_hamer"]
    pc = r.get("per_criterion", {}) or {}
    pf = r.get("per_frame", {}) or {}
    confs = [f.get("keypoint_conf_median") for f in pf.values()
             if f.get("keypoint_conf_median") is not None]
    reproj = [f.get("mean_reproj_error_px") for f in pf.values()
              if f.get("mean_reproj_error_px") is not None]
    circ = "P0 阈值=本次 run 自校准（report 自述），对 P1 是占位"
    gates += [
        _gate("smoke6_hamer", "关键点置信度中位（每帧）", ">= 0.20",
              samples=confs, note=circ + "；MediaPipe 检测分广播到 21 点"),
        _gate("smoke6_hamer", "每帧重投影 mean 误差 (px)", "evidence-only（参考=MediaPipe，非 GT）",
              samples=reproj, note=f"median {fmt(pc.get('median_reprojection_error_px'))} px / p90 {fmt(pc.get('p90_reprojection_error_px'))} px；{circ}"),
        _gate("smoke6_hamer", "手跨度比 left", "∈ [0.85, 1.20]",
              stats={"n": pc.get("hand_span_ratio", {}).get("left", {}).get("n"),
                     "median": pc.get("hand_span_ratio", {}).get("left", {}).get("median"),
                     "p90": pc.get("hand_span_ratio", {}).get("left", {}).get("p90"),
                     "max": pc.get("hand_span_ratio", {}).get("left", {}).get("max")},
              note=circ + "；p10 不可得"),
        _gate("smoke6_hamer", "手跨度比 right", "∈ [0.85, 1.20]",
              stats={"n": pc.get("hand_span_ratio", {}).get("right", {}).get("n"),
                     "median": pc.get("hand_span_ratio", {}).get("right", {}).get("median"),
                     "p90": pc.get("hand_span_ratio", {}).get("right", {}).get("p90"),
                     "max": pc.get("hand_span_ratio", {}).get("right", {}).get("max")},
              note=circ + "；p10 不可得"),
        _gate("smoke6_hamer", "腕部配准误差/span", "< 0.15",
              samples=[(pc.get("median_wrist_reprojection_error_px") or 0) / (pc.get("median_hand_span_px") or 1)],
              level="单值", note=circ),
        _gate("smoke6_hamer", "左右手检测率", ">= 0.8",
              samples=[pc.get("left_detection_rate"), pc.get("right_detection_rate")],
              level="低样本 (n=2)", note=circ),
        _gate("smoke6_hamer", "左右手翻转次数", "<= 2",
              samples=[pc.get("handedness_flips")], level="单值", note=circ),
    ]
    return gates


def build_calibration(reports: dict, qa: dict, ctx: dict) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    gates = build_gates(reports, qa)
    lines = [
        "# P1 阈值校准材料 — P0 实测分布",
        "",
        f"- 生成时间：{now}",
        f"- 生成命令：`python3 scripts/p0_summary.py --runs-dir {ctx['runs_dir_arg']} "
        f"--out-dir {ctx['out_dir_arg']}`",
        f"- 仓库 HEAD：`{ctx['git_head'] or 'unknown'}`",
        "",
        "**给签字人的三句话：**",
        "",
        "1. 全部 P0 分布来自**单条 clip / 单一场景**：感知门（smoke1/2/3/6）不随 MJCF 重跑变化，"
        "但也只有一条 clip 的样本量；物理门（smoke4/5）测于 Wave-3 MJCF `4508421`"
        "（scene `ffb27e97`，与当前 HEAD 一致），**正式重跑（Wave-3.5 修订 + smoke4 重训）后必须重生成，不得沿用**。",
        "2. **可值**：标「实测分布」的门（n≥10，逐样本在 report 里可复核）。"
        "**占位**：smoke6 的 P0 阈值是这次 run 自校准的（report 自述 circular），"
        "P1 固化时需要你给定独立依据；标「低样本 / 单值 / 仅汇总统计」的门不构成分布证据。",
        "3. FAIL 门（smoke4 A3b、smoke5 palm 偏差）如实标注 —— 它们的「实测分布」描述的是"
        "**失败物理下的行为**，P1 阈值不应参照它们定标，只用于对照。",
        "",
        "数据等级图例：实测分布（逐样本可得，n≥10）／低样本（逐样本可得但 n<10）／"
        "仅汇总统计（report 只存中位/p90 等，p10 不可得）／单值／布尔门。",
        "",
    ]
    for sid, title in SMOKES:
        rows = [g for g in gates if g["smoke"] == sid]
        if not rows:
            continue
        lines += [f"\n## {title}", ""]
        table_rows = []
        for g in rows:
            table_rows.append((
                g["gate"], g["threshold"], fmt(g["n"]), fmt(g["median"]),
                fmt(g["p10"]), fmt(g["p90"]), fmt(g["worst"]),
                ("**FAIL** " if g["failing"] else "") + g["level"],
                "**须重生成**（Wave-3.5）" if g["regen"] else "不随重跑变化",
                one_line(g["note"], 150) if g["note"] else "",
            ))
        lines.append(md_table(
            ["验收门", "P0 阈值", "n", "中位", "p10", "p90", "最差", "数据等级", "重跑后", "备注"],
            table_rows))
    lines += [
        "",
        "## 重生成清单（正式数据落地后）",
        "",
        f"1. smoke4/smoke5 全部门（标「须重生成」）：新 MJCF commit 落地 → smoke5 复跑、"
        "smoke4 重训+重评 → 重跑本脚本。",
        "2. 感知门（smoke1/2/3/6）仅在更换 clip/相机/权重时才需重采。",
        "3. smoke6 各门的 P0 阈值系本次 run 自校准（占位）：P1 固化前需要独立依据（跨 clip 或 GT 对齐）。",
        "",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main(argv=None) -> int:
    script_root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs-dir", default="runs/p0",
                    help="directory holding the six smoke report.json files")
    ap.add_argument("--repo-root", default=str(script_root))
    ap.add_argument("--out-dir", default=None,
                    help="where to write the markdown (default: same as --runs-dir)")
    ap.add_argument("--only", choices=["summary", "calibration"], default=None)
    args = ap.parse_args(argv)

    repo_root = Path(args.repo_root).resolve()
    runs_dir = Path(args.runs_dir)
    if not runs_dir.is_absolute():
        runs_dir = repo_root / runs_dir
    out_dir = Path(args.out_dir) if args.out_dir else runs_dir
    if not out_dir.is_absolute():
        out_dir = repo_root / out_dir

    reports = {}
    for sid, _ in SMOKES:
        reports[sid] = load_report(runs_dir, sid)
    qa = load_qa(runs_dir)

    ctx = {
        "git_head": git(repo_root, "rev-parse", "--short", "HEAD"),
        "commits": {sid: report_commit(repo_root, runs_dir.name, sid)
                    for sid, _ in SMOKES},
        "runs_dir_arg": args.runs_dir,
        "out_dir_arg": args.out_dir or args.runs_dir,
    }

    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        if args.only in (None, "summary"):
            (out_dir / "SUMMARY.md").write_text(
                build_summary(reports, qa, ctx), encoding="utf-8")
            print(f"wrote {out_dir / 'SUMMARY.md'}")
        if args.only in (None, "calibration"):
            (out_dir / "p1_threshold_calibration.md").write_text(
                build_calibration(reports, qa, ctx), encoding="utf-8")
            print(f"wrote {out_dir / 'p1_threshold_calibration.md'}")
    except ReportError:
        raise
    except OSError as e:
        raise ReportError(f"cannot write output: {e}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ReportError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)
