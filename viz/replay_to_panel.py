"""Build the W6d panel data (schema v2) from a robot replay.

Canonical output ``viz/panel_data_v2.json`` -- the v2 schema pinned with W6b:

    {
      "fps": 30.0,
      "video_frames": ["docs/video_frames/f_0001.jpg", ...],
      "replay": {"fps": 30.0, "frames": [
          {"t": 0.0, "qpos": [51], "obj": [x, y, z, qw, qx, qy, qz]}, ...]},
      "gates": [{"id", "name", "verdict", "measured", "scope"}, ...]
    }

``qpos`` is the full 51-vector of grounding/assets/allegro_floating.xml
(nq=51: [0:7] object free joint, [7:29] right hand, [29:51] left hand).
``obj`` repeats the object free joint separately because composed replay
qpos can carry it zeroed (the W6a gate-3 configuration does exactly that);
``viz/render_replay.py`` must place the object from ``obj``, never from qpos.

Inputs, in priority order:
  1. ``runs/w6b/replay_frames.json`` (W6b trajectory; fps + frames[{t,qpos,obj}])
     + gates from ``runs/w6b/report.json`` (verdict_checks + per_side), or a
     manual override in ``viz/replay_gates.json``
  2. Placeholder fallback while W6b data has not landed: the two VERIFIED
     configurations we already hold evidence for -- the settled reset and
     the W6a gate-3 both-hands-on-end-faces pose (runs/w6/gate3_qpos.json) --
     plus the live-measured W6a gate badges.  ``meta.placeholder`` marks this
     honestly; rerunning this script once W6b's replay_frames.json exists
     upgrades the panel to the real trajectory with no other change.

Also writes ``viz/panel_data_v2.js`` (``window.PANEL_DATA_V2 = {...}``) so the
panel works from file:// where fetch() of a sibling JSON is blocked.

    runs/.venv-mj/bin/python viz/replay_to_panel.py
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

REPLAY_IN = REPO / "runs/w6b/replay_frames.json"
REPORT_IN = REPO / "runs/w6b/report.json"
GATES_OVERRIDE = REPO / "viz/replay_gates.json"
GATE3 = REPO / "runs/w6/gate3_qpos.json"
VIDEO_DIR = REPO / "docs/video_frames"
OUT_JSON = REPO / "viz/panel_data_v2.json"
OUT_JS = REPO / "viz/panel_data_v2.js"

VIDEO_FPS = 30.0          # source clip; keep in step with extract_video_frames.py
SOURCE_TOTAL_FRAMES = 390  # tissue_box.mp4: 13.0 s at 30 fps (PLAN.md pinned)
REPLAY_FPS = 20.0         # W6b retarget replay rate (replay_frames.json "fps")
BASE_QPOS = 51            # v19-era base: [0:7] object free joint, then hands
NQ_MIN = 51               # frames must carry at least the base; v20.1 carries
                          # the full model nq (65 = base + arm/pad angles)


SCENES = {"ur5e": REPO / "grounding/assets/allegro_floating_ur5e.xml",
          "ur5e_vise": REPO / "grounding/assets/allegro_floating_ur5e_r24_vise.xml",
          "floating": REPO / "grounding/assets/allegro_floating.xml"}


def _scene_for_width(width: int) -> str:
    """Pick the scene whose model nq matches the replay qpos width."""
    import mujoco
    for kind, xml in SCENES.items():
        nq = int(mujoco.MjModel.from_xml_path(str(xml)).nq)
        if nq == width:
            return kind
    raise SystemExit(
        f"replay qpos width {width} matches no known scene "
        f"({ {k: int(mujoco.MjModel.from_xml_path(str(v)).nq) for k, v in SCENES.items()} })")

# Placeholder frames: t=2.233 s is source frame 67 (pads seat on the end
# faces in the human clip), so the two replay frames line up with the two
# video frames the panel will show.
PLACEHOLDER = [
    ("静置复位（400 步沉降）", 0.0),
    ("双手指垫触及端面（W6a 门 3 配置）", 67 / VIDEO_FPS),
]


def _fmt_mm(x_m: float) -> str:
    return f"{x_m:.1e} m" if abs(x_m) < 5e-7 else f"{x_m * 1e3:.3f} mm"


def _measure_w6a_gates() -> list[dict]:
    """Measure the three W6a gates live, mirroring tests/test_scene_w6.py's
    own protocol and constants (GATE 1 birth contacts, GATE 2 400-step
    settle, GATE 3 persisted reach), so the placeholder badges always carry
    the currently committed scene's numbers."""
    import mujoco

    import grounding.envs.lift_hold_env as lhe

    birth_tol_m, move_tol_m, settle_steps = 1e-9, 1e-3, 400  # test constants
    env = lhe.LiftHoldEnv(episode_len=50, domain_rand=False)
    env.reset(seed=0)
    m = env.model

    # GATE 1: fresh data + mj_forward = the birth state the test fixtures use
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    birth = min((float(d.contact[i].dist) for i in range(d.ncon)), default=0.0)

    # GATE 2: settle THAT SAME data 400 steps, drift of box + both hand roots
    obj = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "object")
    rr = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "hand_root")
    rl = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "hand_root_left")
    p0, r0, l0 = d.xpos[obj].copy(), d.xpos[rr].copy(), d.xpos[rl].copy()
    for _ in range(settle_steps):
        mujoco.mj_step(m, d)
    box_move = float(np.linalg.norm(d.xpos[obj] - p0))
    hand_move = max(float(np.linalg.norm(d.xpos[rr] - r0)),
                    float(np.linalg.norm(d.xpos[rl] - l0)))

    # GATE 3: the composed reach persisted by the test itself
    g3 = json.loads(GATE3.read_text())
    gaps = g3["gaps_mm"]
    tol_mm = g3["tol_m"] * 1e3

    return [
        {"id": "w6a_birth", "name": "出生穿透",
         "verdict": "pass" if birth >= -birth_tol_m else "fail",
         "measured": f"{_fmt_mm(birth)}（出生接触最小间距）/ 帽 1e-9 m",
         "scope": "复位瞬间手指与盒面的最大嵌入深度：双手初始位姿是否零埋入"},
        {"id": "w6a_settle", "name": "静置 400 步",
         "verdict": "pass" if box_move < move_tol_m and hand_move < move_tol_m
         else "fail",
         "measured": f"盒 {_fmt_mm(box_move)} / 手 {_fmt_mm(hand_move)} / 帽 1 mm",
         "scope": "无策略空置 400 控制步后盒与手的最大漂移：场景自稳"},
        {"id": "w6a_reach", "name": "双手可及",
         "verdict": "pass" if max(gaps.values()) <= tol_mm else "fail",
         "measured": (f"右 {abs(gaps['right']):.3f} mm / "
                      f"左 {abs(gaps['left']):.3f} mm / 帽 {tol_mm:.3f} mm"),
         "scope": "门 3：双手指垫到各自端面的最小距离（负=轻微压入，容差内即过）"},
    ]


def _settled_reset_frame() -> tuple[list[float], list[float]]:
    """Reset the dual-hand env (seed=0) and return (qpos, obj) after settling."""
    import mujoco
    import grounding.envs.lift_hold_env as lhe

    env = lhe.LiftHoldEnv(episode_len=50, domain_rand=False)
    env.reset(seed=0)
    adr = int(env.model.jnt_qposadr[mujoco.mj_name2id(
        env.model, mujoco.mjtObj.mjOBJ_JOINT, "object_free")])
    qpos = env.data.qpos.copy()
    return qpos.tolist(), qpos[adr:adr + 7].tolist()


def _placeholder_frames() -> list[dict]:
    if not GATE3.exists():
        raise SystemExit(
            f"missing {GATE3} -- run pytest "
            "tests/test_scene_w6.py::"
            "test_w6_gate3_both_hands_reach_their_end_faces first")
    q_reset, obj = _settled_reset_frame()
    g3 = json.loads(GATE3.read_text())
    q_reach = [float(v) for v in g3["qpos"]]
    # legacy gate3 evidence carries the 51-dim base; pad the v20.1 arm/pad
    # angle head with zeros (placeholder-only path, cosmetically straight arm)
    q_reach = [0.0] * (len(q_reset) - len(q_reach)) + q_reach
    return [{"t": t, "qpos": q, "obj": obj}
            for (_, t), q in zip(PLACEHOLDER, [q_reset, q_reach])]


def _convert_replay(raw: dict | list) -> tuple[list[dict], float, str]:
    """Normalize W6b's replay_frames.json into schema frames; returns
    (frames, fps, scene kind inferred from the qpos width)."""
    frames = raw["frames"] if isinstance(raw, dict) else raw
    fps = float(raw.get("fps", REPLAY_FPS)) if isinstance(raw, dict) else REPLAY_FPS
    if isinstance(raw, dict) and "fps" in raw:
        print(f"replay fps from file: {raw['fps']}")
    out = []
    for i, fr in enumerate(frames):
        if "qpos" not in fr:
            raise SystemExit(f"replay frame {i}: missing 'qpos'")
        q = [float(v) for v in fr["qpos"]]
        scene_kind = _scene_for_width(len(q))   # validates width, names scene
        if i == 0:
            print(f"scene: {scene_kind} (qpos width {len(q)})")
        obj = fr.get("obj")
        if obj is None:
            # no separate object pose -> the free-joint slice of qpos is it
            obj = q[0:7]
        else:
            obj = [float(v) for v in obj]
            if len(obj) != 7:
                raise SystemExit(f"replay frame {i}: obj must be [x,y,z,qw,qx,qy,qz]")
        out.append({"t": float(fr.get("t", i / fps)),
                    "qpos": q, "obj": obj})
    if not out:
        raise SystemExit("replay_frames.json carries zero frames")
    return out, fps, _scene_for_width(len(out[0]["qpos"]))


def _w6b_gates() -> list[dict]:
    """Derive the gate badges from runs/w6b/report.json -- every number
    verbatim from the committed r20 report, green/red exactly as measured."""
    if not REPORT_IN.exists():
        raise SystemExit(
            f"real replay ({REPLAY_IN}) without {REPORT_IN} -- supply the W6b "
            "report or a manual override in viz/replay_gates.json")
    rep = json.loads(REPORT_IN.read_text())
    checks = rep["verdict_checks"]
    ps = rep["per_side"]
    r, l = ps["right"], ps["left"]

    g1r, g1l = r["gate1_final_touch"], l["gate1_final_touch"]
    g2r, g2l = r["gate2_contact"], l["gate2_contact"]
    g3r, g3l = r["gate3_approach_clean"], l["gate3_approach_clean"]
    g4r, g4l = r["gate4_joint_limits"], l["gate4_joint_limits"]
    g5r, g5l = r["gate5_palm_deviation"], l["gate5_palm_deviation"]
    wr_r = rep["trajectory_error"]["wrist_referenced"]["per_side"]["right"]
    wr_l = rep["trajectory_error"]["wrist_referenced"]["per_side"]["left"]

    def wrist_side(w):
        ok = w.get("pass_4c_limit_15pct")
        frac = w["conf_weighted_median_err_span_frac"] * 100
        return (f"{w['conf_weighted_median_err_m'] * 1e3:.1f} mm"
                f"（{frac:.1f}%，15% 线{'达标' if ok else '超限'}）")

    def g4_txt(g):
        n = len(g.get("finger_qpos_violation_frames", []))
        peak = (g.get("finger_qpos_max_violation_rad") or 0.0) * 1e3
        return f"{n} 帧·峰 {peak:.0f} mrad"

    def reg(g5):
        reg = g5.get("registered", {})
        return {"max": reg.get("registered_palm_max_m", 0.0) * 1e3,
                "pass": bool(reg.get("registered_pass_40mm"))}

    reg_r, reg_l = reg(g5r), reg(g5l)
    gate5_pass = reg_r["pass"] and reg_l["pass"]

    return [
        {"id": "w6b_gate1", "name": "门 1 终触（双列）",
         "verdict": "pass" if checks["gate1_final_touch"] else "fail",
         "measured": (f"R5 距离腿 ✓ 右 {g1r['rect_dist_mm']:.4f} / "
                      f"左 {g1l['rect_dist_mm']:.4f} mm；"
                      f"2 垫腿 ✗ 右 {g1r['n_pads_in_rect']}/2、"
                      f"左 {g1l['n_pads_in_rect']}/2"
                      f"（前向 {g1r['forward_axis_deg']:.1f}°/"
                      f"{g1l['forward_axis_deg']:.1f}° 达标）"),
         "scope": ("v5.1 双列语义：R5 距离腿=单垫矩形距离 ≤0.116 mm（已达标）；"
                   "2 垫腿=≥2 指垫同入带内（当前 1/2，红）——两腿须同时过")},
        {"id": "w6b_gate2", "name": "门 2 端面接触保持",
         "verdict": "pass" if checks["gate2_contact"] else "fail",
         "measured": (f"右 {g2r['endface_contact_frames']} 帧（首触 第 "
                      f"{g2r['first_endface_control_frame']} 控制帧）/ "
                      f"左 {g2l['endface_contact_frames']} 帧（首触 第 "
                      f"{g2l['first_endface_control_frame']}）"),
         "scope": ("接触物理执行下指垫贴住端面的控制帧数。r26 低帧数归因："
                  "上摆段盒-垫接触瞬态（右 2 帧瞬时触面即离，非持续保持）")},
        {"id": "w6b_gate3", "name": "门 3 接近段干净",
         "verdict": "pass" if checks["gate3_approach_clean"] else "fail",
         "measured": (f"右压支撑 {g3r.get('support_contact_frames', 0)} 帧 / "
                      f"左 {g3l.get('support_contact_frames', 0)} 帧"),
         "scope": "接近段手指不得触桌或支撑（r25 终局：中程倾斜过侧 ~95° 后回正，"
                  "夹具域已尽——vise 锁漂移、pre-roll 触发保险丝被否）"},
        {"id": "w6b_gate4", "name": "门 4 关节限位",
         "verdict": "pass" if checks["gate4_joint_limits"] else "fail",
         "measured": f"右 {g4_txt(g4r)} / 左 {g4_txt(g4l)}",
         "scope": "手指与垫铰 qpos 不越关节限、指令全部落在 ctrlrange 内"},
        {"id": "w6b_gate5", "name": "门 5 掌偏差（配准）",
         "verdict": "pass" if gate5_pass else "fail",
         "measured": (f"右 {reg_r['max']:.2f} / 左 {reg_l['max']:.2f} mm"
                      f"（配准后 max，≤40 mm 达标）；"
                      f"求解收敛 右 {g5r['constrained_solve_palm_max_m'] * 1e3:.2f} / "
                      f"左 {g5l['constrained_solve_palm_max_m'] * 1e3:.2f} mm"),
         "scope": ("解剖学配准（Main Front B）：扣掉抓握窗常量偏置后的掌心残差——"
                   "raw 未配准腿（含工具系偏置）如实留在 report")},
        {"id": "w6b_wrist", "name": "腕轨迹误差",
         "verdict": "pass" if checks.get("wrist_referenced_error") else "fail",
         "measured": f"左 {wrist_side(wr_l)} / 右 {wrist_side(wr_r)}",
         "scope": "MediaPipe 腕点 vs 回放腕的置信加权中位误差，占手跨度百分比"
                  "（4c 同尺判据，逐手机判定）"},
        {"id": "w6b_align", "name": "1:1 帧对齐",
         "verdict": "pass" if checks.get("frame_alignment") else "fail",
         "measured": (f"{len(json.loads(REPLAY_IN.read_text())['frames'])} 帧 @30 fps"
                      " ↔ 原片逐帧"
                      + ("（对齐断言 PASS）" if checks.get("frame_alignment")
                         else "（对齐断言未过）")),
         "scope": "仿真帧 i ↔ 原片帧 i+1（t=i/30 s）的对齐断言——本面板滑块的时基依据"},
    ]


def build() -> dict:
    video_manifest = VIDEO_DIR / "manifest.json"
    if video_manifest.exists():
        man = json.loads(video_manifest.read_text())
        video_frames, segments = man["video_frames"], man["segments"]
    else:
        found = sorted(p.name for p in VIDEO_DIR.glob("f_*.jpg"))
        if not found:
            raise SystemExit(f"no video frames in {VIDEO_DIR} -- run "
                             "viz/extract_video_frames.py first")
        video_frames = [f"docs/video_frames/{n}" for n in found]
        segments = []

    replay_fps = REPLAY_FPS
    scene_kind = "floating"
    if REPLAY_IN.exists():
        frames, replay_fps, scene_kind = _convert_replay(
            json.loads(REPLAY_IN.read_text()))
        source = "runs/w6b/replay_frames.json (W6b)"
        placeholder, notes = False, []
    else:
        frames = _placeholder_frames()
        source = "占位：env reset + runs/w6/gate3_qpos.json（均为已验证配置）"
        placeholder = True
        notes = [n for n, _ in PLACEHOLDER]

    # playback timebase: round-4 replays ship the full 390-frame clip at
    # 30 fps, so sim frame i IS source frame i+1 -- the panel then drives
    # both sides 1:1 off the slider.  Older replays (163 @ 20 Hz) keep the
    # nearest-time video alignment and are labelled as the legacy timebase.
    if len(frames) == SOURCE_TOTAL_FRAMES and abs(replay_fps - VIDEO_FPS) < 1e-6:
        playback = {"mode": "one_to_one", "replay_frames": len(frames),
                    "note": "1:1：仿真帧 i ↔ 原片帧 i+1，t=i/30 s"}
    else:
        playback = {"mode": "legacy", "replay_frames": len(frames),
                    "note": (f"旧版时基：{len(frames)} 帧 @{replay_fps:g} Hz，"
                             "视频侧按时间就近对齐")}

    if GATES_OVERRIDE.exists():
        gates = json.loads(GATES_OVERRIDE.read_text())["gates"]
    elif placeholder:
        gates = _measure_w6a_gates()
    else:
        gates = _w6b_gates()

    payload = {
        "fps": VIDEO_FPS,
        "video_frames": video_frames,
        "replay": {"fps": replay_fps, "frames": frames},
        "gates": gates,
        "meta": {
            "generated_by": "viz/replay_to_panel.py",
            "source": source,
            "placeholder": placeholder,
            "placeholder_note": (
                "两个已验证场景帧（静置复位 / W6a 门 3 双手触面）。"
                "W6b 的 replay_frames.json 落地后，重跑 "
                "runs/.venv-mj/bin/python viz/replay_to_panel.py && "
                "runs/.venv-mj/bin/python viz/render_replay.py 即为成片。"),
            "notes_per_frame": notes,
            "playback": playback,
            "scene": scene_kind,
            "phase_notes": {
                # C-STABILITY 终局 (Main, final)
                "抬离": ("抬离：C-STABILITY 终局——8 类控制穷尽 + 发现墙 "
                         "rate_w<0.0067 量化 + 力伺服未闭合 + yaw-clean "
                         "t=0 1.05° + lift 三元组 127 帧/105mm/100mm"
                         "=5/7 条款。"),
                # r26 Track A：翻转消除；上摆段盒-垫接触瞬态如实注（g2 归因）
                "接近": ("上摆段盒-垫接触瞬态：右 2 帧瞬时触面即离——"
                         "g2 低帧数归因（非持续保持失效）"),
            },
            "video_fps": VIDEO_FPS,
            "total_source_frames": SOURCE_TOTAL_FRAMES,
            "video_segments": segments,
            "src_video": "runs/p0/assets/tissue_box.mp4",
        },
    }
    # viz/render_replay.py backfills meta.renders after offscreen rendering;
    # keep it when the frame set is unchanged so rerunning only this converter
    # does not blank the panel's sim panes (rerun render_replay.py if frames
    # changed)
    if OUT_JSON.exists():
        try:
            old = json.loads(OUT_JSON.read_text()).get("meta", {}).get("renders")
            if old and len(old.get("frames", {}).get("front", [])) == len(frames):
                payload["meta"]["renders"] = old
        except (json.JSONDecodeError, KeyError, TypeError, OSError):
            pass
    return payload


def emit(payload: dict) -> None:
    # preserve the render indexes (mujoco raster + blender cycles) across
    # rebuilds: they are regenerated by their own scripts and the frame
    # counts only change when the replay does
    prev = {}
    if OUT_JSON.exists():
        try:
            prev = json.loads(OUT_JSON.read_text()).get("meta", {})
        except (json.JSONDecodeError, OSError):
            prev = {}
    n = len(payload["replay"]["frames"])
    for key in ("renders_blender", "renders"):
        old = prev.get(key)
        if not old:
            continue
        counts = {k: len(v) for k, v in old.get("frames", {}).items()}
        if counts and all(c == n for c in counts.values()):
            payload["meta"][key] = old
    OUT_JSON.write_text(json.dumps(payload, indent=1, ensure_ascii=False))
    OUT_JS.write_text("window.PANEL_DATA_V2 = "
                      + json.dumps(payload, ensure_ascii=False) + ";\n")
    print(f"wrote {OUT_JSON.relative_to(REPO)} "
          f"({OUT_JSON.stat().st_size // 1024} KB) + "
          f"{OUT_JS.relative_to(REPO)} "
          f"(replay frames: {len(payload['replay']['frames'])}, "
          f"placeholder: {payload['meta']['placeholder']})")


if __name__ == "__main__":
    emit(build())
