"""Temporally coupled, constrained solve of the retargeted control trajectory.

Frame-by-frame IK cannot answer the question this module asks, because the
binding constraints -- actuator ctrlrange, joint range, and the signed velocity
limits -- are all *coupled in time*.  A frame whose command is 0.9 rad outside
the ring finger's ctrlrange is not repaired by fixing that frame alone; the
whole clip has to move.

So the clip is solved jointly:

    minimise  sum_t || hand_world(x_t) - hand_world(target_t) ||^2
    s.t.      lo_t <= x_t <= hi_t                     (actuator ctrlrange
                                                      INTERSECT joint range)
              |x_t - x_{t-1}| <= v_class * dt          (signed velocity gate)

The objective is stated in the world, because that is where "did the retargeted
hand end up where the video says it should be" is answerable.  "The hand" is
the palm frame origin plus all four fingertips: the palm alone would ignore the
fingers entirely, which are where most of the constraint violation lives.  The
objective is optimised by damped Gauss-Newton on MuJoCo's analytic body
Jacobians, and the constraints are enforced by alternating projection onto the
box and onto the rate band.  Both sets are convex, so the alternation
converges; the residual at convergence is recorded, never assumed.

Nothing here loosens a threshold.  Where the clip is infeasible the solve says
so by returning a larger residual, and the caller gates on it.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Alternation between the objective step and the constraint projection.  Each
# sweep is O(T * n_bodies); convergence is measured, not assumed, and the loop
# stops early when the world-pose residual stops moving.
MAX_SWEEPS = 60
RESIDUAL_TOL_M = 1e-5
# Damped least squares, so a frame whose hand-pose Jacobian is rank deficient --
# a pure wrist yaw moves no fingertip -- does not produce an unbounded step.
DAMPING = 1e-3
# Commands are generated strictly under the signed rate ceiling; see
# retarget_smoke.RATE_MARGIN.
RATE_MARGIN = 1.0 - 1e-9


def _body_jacobian(model, data, body: int, dofadr: np.ndarray) -> np.ndarray:
    """d(body world position)/d(command) at the current qpos, 3 x nu.

    ``dofadr`` indexes MuJoCo's *velocity* DoFs, which is what mj_jac returns;
    the hand's qpos addresses are not the same numbers once the box's free
    joint is in the model.
    """
    import mujoco

    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jac(model, data, jacp, jacr, data.xpos[body].copy(), body)
    return jacp[:, dofadr]


def _apply(model, data, ctx, x: np.ndarray, t: int) -> None:
    import mujoco

    data.qpos[:] = model.qpos0
    data.qpos[ctx["finger_qadr"]] = x[t, 6:]
    data.qpos[ctx["wrist_qadr"]] = x[t, 0:6]
    mujoco.mj_forward(model, data)


def _hand_world(model, data, ctx, x: np.ndarray) -> np.ndarray:
    """(T, 3*n_bodies) world positions of the palm frame and the four fingertips."""
    out = np.empty((x.shape[0], 3 * len(ctx["pose_bodies"])))
    for t in range(x.shape[0]):
        _apply(model, data, ctx, x, t)
        out[t] = np.concatenate([data.xpos[b].copy() for b in ctx["pose_bodies"]])
    return out


def _hand_residual_jac(model, data, ctx, x: np.ndarray, target: np.ndarray):
    res = np.empty((x.shape[0], 3 * len(ctx["pose_bodies"])))
    jac = np.empty((x.shape[0], 3 * len(ctx["pose_bodies"]), x.shape[1]))
    for t in range(x.shape[0]):
        _apply(model, data, ctx, x, t)
        res[t] = np.concatenate([data.xpos[b].copy() for b in ctx["pose_bodies"]]) - target[t]
        jac[t] = np.concatenate(
            [_body_jacobian(model, data, b, ctx["dofadr"]) for b in ctx["pose_bodies"]]
        )
    return res, jac


def _project_box(x: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Clip into the box. ``lo``/``hi`` may be per-DoF (nu,) or per-frame (T, nu)."""
    return np.clip(x, lo, hi)


def _project_rate(x: np.ndarray, step: np.ndarray, lo: np.ndarray, hi: np.ndarray,
                  sweeps: int = 24) -> np.ndarray:
    """Slew-limited projection: clip each step into the rate band, both ways.

    The forward pass alone guarantees |x_t - x_{t-1}| <= step by induction, but
    it biases the trajectory forward; alternating it with a backward pass settles
    onto the projection of the two-point boundary.  Both passes clamp into the
    same frame's box at the same time, so the result stays box-feasible even when
    the box is per-frame rather than per-DoF.
    """
    x = x.copy()
    for _ in range(sweeps):
        moved = 0.0
        for t in range(1, x.shape[0]):
            prev = x[t - 1]
            target = np.clip(x[t], np.maximum(prev - step, lo[t]), np.minimum(prev + step, hi[t]))
            moved = max(moved, float(np.abs(target - x[t]).max()))
            x[t] = target
        for t in range(x.shape[0] - 2, -1, -1):
            nxt = x[t + 1]
            target = np.clip(x[t], np.maximum(nxt - step, lo[t]), np.minimum(nxt + step, hi[t]))
            moved = max(moved, float(np.abs(target - x[t]).max()))
            x[t] = target
        if moved <= 1e-12:
            break
    return x


def _split_deviation(dev: np.ndarray, n_bodies: int) -> dict[str, Any]:
    d = dev.reshape(dev.shape[0], n_bodies, 3)
    palm = np.linalg.norm(d[:, 0, :], axis=1)
    tips = np.linalg.norm(d[:, 1:, :], axis=2).max(axis=1)
    return {
        "palm_frame": {
            "definition": "||palm world position(solved) - palm world position(retargeted)||",
            "rms_m": float(np.sqrt(np.mean(palm ** 2))),
            "max_m": float(palm.max()),
            "per_frame_m": palm.tolist(),
        },
        "fingertips": {
            "definition": (
                "worst of the four fingertips' ||solved - retargeted|| world "
                "positions, per frame"
            ),
            "rms_m": float(np.sqrt(np.mean(tips ** 2))),
            "max_m": float(tips.max()),
            "per_frame_m": tips.tolist(),
        },
    }


def solve_constrained_traj(model, data, ctx, x0: np.ndarray,
                           lo: np.ndarray, hi: np.ndarray,
                           step: np.ndarray,
                           slide_move_cap_m: float | None = None
                           ) -> tuple[np.ndarray, dict[str, Any]]:
    """Constrained solve. Returns (x, record).

    ``lo``/``hi``/``step`` are per-DoF: the box is the intersection of the
    actuator ctrlrange with the joint range (a command inside the ctrlrange can
    still sit outside the joint range), and ``step`` is the signed velocity
    limit times the control period.

    ``slide_move_cap_m`` bounds how far each frame's wrist-slide command may move
    from the retargeted one.  The palm frame origin sits exactly at
    ``hand_root + slides``, so this *is* the palm's world-pose deviation, and it
    has to be a constraint of the solve rather than an outcome of it: the palm's
    world position does not depend on any wrist angle, so an objective that
    cannot reach a rate-limited fingertip will otherwise satisfy itself by
    sliding the whole hand toward the box -- trading a bounded grasp error for an
    unbounded one.  The per-axis share is cap/sqrt(3), so the *norm* of the slide
    move is inside the cap, not three times it.
    """
    step = step * RATE_MARGIN
    n_bodies = len(ctx["pose_bodies"])
    x0 = np.asarray(x0, float)
    target = _hand_world(model, data, ctx, x0)

    n_t = x0.shape[0]
    box_lo = np.broadcast_to(lo, (n_t, lo.shape[0])).copy()
    box_hi = np.broadcast_to(hi, (n_t, hi.shape[0])).copy()
    if slide_move_cap_m is not None:
        share = slide_move_cap_m / np.sqrt(3.0)
        box_lo[:, :3] = np.maximum(box_lo[:, :3], x0[:, :3] - share)
        box_hi[:, :3] = np.minimum(box_hi[:, :3], x0[:, :3] + share)

    x = _project_box(x0.copy(), box_lo, box_hi)
    x = _project_rate(x, step, box_lo, box_hi)

    history: list[float] = []
    prev_rms = np.inf
    sweeps = 0
    for sweep in range(MAX_SWEEPS):
        res, jac = _hand_residual_jac(model, data, ctx, x, target)
        rms = float(np.sqrt(np.mean(np.sum(res ** 2, axis=1))))
        history.append(rms)
        grad = np.einsum("tij,ti->tj", jac, res)
        hess = np.einsum("tij,tik->jk", jac, jac) / max(1, x.shape[0])
        hess[np.diag_indices_from(hess)] += DAMPING
        try:
            delta = -np.linalg.solve(hess, grad.T).T
        except np.linalg.LinAlgError:  # pragma: no cover - damped solve rarely fails
            delta = -grad
        x = _project_box(x + delta, box_lo, box_hi)
        x = _project_rate(x, step, box_lo, box_hi)
        sweeps = sweep + 1
        if abs(prev_rms - rms) < RESIDUAL_TOL_M:
            break
        prev_rms = rms

    res, _ = _hand_residual_jac(model, data, ctx, x, target)
    per_frame_residual = np.linalg.norm(
        res.reshape(res.shape[0], n_bodies, 3), axis=2
    ).max(axis=1)
    deviation = _split_deviation(
        _hand_world(model, data, ctx, x) - _hand_world(model, data, ctx, x0), n_bodies
    )
    record = {
        "method": (
            "temporally coupled solve: alternating projection onto "
            "{actuator ctrlrange INTERSECT joint range} and onto the signed "
            "velocity band, with a damped Gauss-Newton step between them on the "
            "hand world-pose objective (palm frame origin + all four fingertips) "
            "using MuJoCo's analytic body Jacobians. No per-frame independent IK."
        ),
        "objective": (
            "sum_t ||hand_world(x_t) - hand_world(x_retargeted)||^2 over the palm "
            "frame origin and the four fingertips"
        ),
        "sweeps": sweeps,
        "converged": bool(len(history) > 1 and abs(history[-1] - history[-2]) < RESIDUAL_TOL_M),
        "world_pose_residual_rms_m": float(history[-1]),
        "world_pose_residual_history_first5": [float(v) for v in history[:5]],
        "world_pose_residual_per_frame_m": per_frame_residual.tolist(),
        "hand_world_pose_deviation": {
            "definition": (
                "how far the constrained solve moved the hand in the world away "
                "from the pose the retargeted clip asked for, split into the palm "
                "frame (the hand's base pose) and the worst fingertip"
            ),
            **deviation,
        },
        "command_deviation_from_retarget": {
            "rms_rad_or_m": float(np.sqrt(np.mean((x - x0) ** 2))),
            "max_rad_or_m": float(np.abs(x - x0).max()),
        },
        "constraints": {
            "box_lower": box_lo.tolist(),
            "box_upper": box_hi.tolist(),
            "max_step_per_frame": step.tolist(),
            "slide_move_cap_m": slide_move_cap_m,
            "slide_move_cap_note": (
                "per-frame wrist-slide movement from the retargeted command is "
                "bounded by cap/sqrt(3) per axis, so the palm frame's world "
                "position cannot leave a ball of that radius; the palm frame "
                "origin is hand_root + slides exactly"
            ),
        },
    }
    return x, record
