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
  1. ``viz/replay_frames.json``  (W6b trajectory; fps + frames[{t,qpos[,obj]}])
     + ``viz/replay_gates.json``  (W6b gate verdicts, same shape as schema)
  2. Placeholder fallback while W6b data has not landed: the two VERIFIED
     configurations we already hold evidence for -- the settled reset and
     the W6a gate-3 both-hands-on-end-faces pose (runs/w6/gate3_qpos.json) --
     plus the W6a gate badges.  ``meta.placeholder`` marks this honestly;
     rerunning this script once W6b's replay_frames.json exists upgrades the
     panel to the real trajectory with no other change.

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

REPLAY_IN = REPO / "viz/replay_frames.json"
GATES_IN = REPO / "viz/replay_gates.json"
GATE3 = REPO / "runs/w6/gate3_qpos.json"
VIDEO_DIR = REPO / "docs/video_frames"
OUT_JSON = REPO / "viz/panel_data_v2.json"
OUT_JS = REPO / "viz/panel_data_v2.js"

VIDEO_FPS = 30.0          # source clip; keep in step with extract_video_frames.py
REPLAY_FPS = 30.0         # W6b retarget replay runs at the clip's frame rate
NQ = 51                   # allegro_floating.xml nq: [0:7] object free joint,
                          # [7:29] right (6 wrist slides + 16 fingers),
                          # [29:51] left (same, *_left joints)

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
    m = env.model
    nq = int(m.nq)
    if nq != NQ:
        raise SystemExit(f"model nq={nq} but converter pins NQ={NQ} -- "
                         "update the pinned schema before converting")
    adr = int(m.jnt_qposadr[mujoco.mj_name2id(
        m, mujoco.mjtObj.mjOBJ_JOINT, "object_free")])
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
    if len(q_reach) != NQ:
        raise SystemExit(f"gate3 qpos length {len(q_reach)} != {NQ}")
    return [{"t": t, "qpos": q, "obj": obj}
            for (_, t), q in zip(PLACEHOLDER, [q_reset, q_reach])]


def _convert_replay(raw: dict | list) -> list[dict]:
    """Normalize W6b's replay_frames.json into schema frames, validating."""
    frames = raw["frames"] if isinstance(raw, dict) else raw
    if isinstance(raw, dict) and "fps" in raw and abs(raw["fps"] - REPLAY_FPS) > 1e-6:
        print(f"[warn] replay fps {raw['fps']} != expected {REPLAY_FPS}; "
              "keeping the file's value")
    out = []
    for i, fr in enumerate(frames):
        if "qpos" not in fr:
            raise SystemExit(f"replay frame {i}: missing 'qpos'")
        q = [float(v) for v in fr["qpos"]]
        if len(q) != NQ:
            raise SystemExit(f"replay frame {i}: qpos length {len(q)} != {NQ} "
                             f"({pathlib.Path(str(REPLAY_IN)).name} must carry the "
                             "full allegro_floating.xml qpos)")
        obj = fr.get("obj")
        if obj is None:
            # no separate object pose -> the free-joint slice of qpos is it
            obj = q[0:7]
        else:
            obj = [float(v) for v in obj]
            if len(obj) != 7:
                raise SystemExit(f"replay frame {i}: obj must be [x,y,z,qw,qx,qy,qz]")
        out.append({"t": float(fr.get("t", i / REPLAY_FPS)),
                    "qpos": q, "obj": obj})
    if not out:
        raise SystemExit("replay_frames.json carries zero frames")
    return out


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

    if REPLAY_IN.exists():
        frames = _convert_replay(json.loads(REPLAY_IN.read_text()))
        source = "viz/replay_frames.json (W6b)"
        placeholder, notes = False, []
    else:
        frames = _placeholder_frames()
        source = "占位：env reset + runs/w6/gate3_qpos.json（均为已验证配置）"
        placeholder = True
        notes = [n for n, _ in PLACEHOLDER]

    gates = (json.loads(GATES_IN.read_text())["gates"]
             if GATES_IN.exists() else _measure_w6a_gates())

    payload = {
        "fps": VIDEO_FPS,
        "video_frames": video_frames,
        "replay": {"fps": REPLAY_FPS, "frames": frames},
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
            "video_fps": VIDEO_FPS,
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
