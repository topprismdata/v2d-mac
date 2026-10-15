"""Mechanism probe: how does pad-box contact evolve through the scripted
phases for the R1/R3 families? Prints per-phase strict pad state so the wedge
topology (wall + top) can be placed with evidence instead of guesses."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path("/Users/guohongbin/v2d-mac")
sys.path.insert(0, str(REPO))

from grounding.envs.lift_hold_env import LiftHoldEnv  # noqa: E402
from grounding.reachable_scan import (  # noqa: E402
    N_CLOSE, N_DESC, N_TRANSIT, _action, _penetration_tol, _tip_pads,
    close_amp, strict_contacts,
)


def probe(cfg: dict, label: str) -> None:
    env = LiftHoldEnv(episode_len=400, domain_rand=False)
    env.reset(seed=0)
    pads = _tip_pads(env)
    tol = _penetration_tol(env)
    rot = cfg["rot"]
    close = np.asarray(cfg["close"])
    open_a = np.full(16, 0.6)
    w = cfg["wrist_world"]

    def phase(name, n, tgt, fingers, snap_at):
        for i in range(n):
            a = _action(env, tgt, rot, fingers)
            env.step(a)
            if (i + 1) in snap_at:
                sc = strict_contacts(env, pads, tol)
                ff = float(env.trace["finger_force"][-1])
                lift = float(env.trace["lift"][-1])
                pd = {k: f"{v['face']}:{v['gap_m']*1000:.2f}mm"
                      for k, v in sc["pads"].items() if v["gap_m"] < 0.02 or v["strict_touch"]}
                ncon = sum(1 for p in env.trace["contact_pairs"][-1]
                           if p[0].split("#")[0].startswith(("ff", "mf", "rf", "th")))
                print(f"  {label} {name}+{i+1}: F={ff:5.2f}N lift={lift*1000:6.1f}mm "
                      f"ncon={ncon} cls={sc['class']} {pd}")

    high = [w[0], w[1], w[2] + 0.10]
    phase("transit", N_TRANSIT, high, open_a, {N_TRANSIT})
    phase("desc", N_DESC, w, open_a, {N_DESC})
    phase("close", N_CLOSE, w, close, {20, 45, N_CLOSE})
    phase("lift", 140, [w[0], w[1], w[2] + 0.14], close, {20, 60, 140})
    env.close()


if __name__ == "__main__":
    base_y018 = [-0.12, 0.18, 0.92 - 0.043]
    base_y000 = [-0.12, 0.0, 0.92 - 0.043]
    rot_r1 = [1.5, 0.9, 0.0]
    rot_r3 = [1.2, 1.2, 0.0]
    for label, w, rot, cl, ct in [
        ("R1.y018 z-0.043 cl0.85 ct-0.75", base_y018, rot_r1, 0.85, -0.75),
        ("R1.y018 z-0.003 cl1.0 ct-0.5", [-0.12, 0.18, 0.92 - 0.003], rot_r1, 1.0, -0.5),
        ("R1.y018 z+0.03 cl1.0 ct-1.0", [-0.12, 0.18, 0.92 + 0.03], rot_r1, 1.0, -1.0),
        ("R1.y018 z+0.06 cl1.0 ct-1.0", [-0.12, 0.18, 0.92 + 0.06], rot_r1, 1.0, -1.0),
        ("R1.y000 z-0.043 cl1.0 ct-1.0", base_y000, rot_r1, 1.0, -1.0),
        ("R3 z-0.003 x+0.06 cl1.0 ct+1.0", [0.06, 0.06, 0.92 - 0.003], rot_r3, 1.0, 1.0),
        ("R3 z+0.03 x+0.08 cl1.0 ct+1.0", [0.08, 0.06, 0.92 + 0.03], rot_r3, 1.0, 1.0),
        ("R3 z+0.06 x+0.04 cl1.0 ct+0.5", [0.04, 0.06, 0.92 + 0.06], rot_r3, 1.0, 0.5),
    ]:
        cfg = {"wrist_world": w, "rot": rot, "close": close_amp(cl, ct).tolist()}
        probe(cfg, label)
        print()
