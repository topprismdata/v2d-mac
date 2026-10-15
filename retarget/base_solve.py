"""Floating-base xyz solve for the smoke-5 retarget replay.

The retargeted clip fixes the hand's *relative* geometry frame by frame -- the
fingertip arrangement, the wrist orientation, and the wrist's own slide motion
come out of the video reconstruction and are never touched here.  What is left
under-determined is the floating base: three degrees of freedom (the MJCF
``wrist_x``/``wrist_y``/``wrist_z`` slides) whose only physical meaning is *where
in the world the arm-less hand is*.  This module solves them.

The objective is the scene's own grasp geometry, read off the frozen MJCF and
never altered: the tissue-box proxy's grasp axis is its 120 mm x span, so the
retargeted fingertips must reach that pair of walls from outside without any
hand geom interpenetrating the box.  The box's pose is a scene fact and is not
moved -- only the hand's base is solved.

Scoring uses MuJoCo's own ``mj_geomDistance`` and contact list, so the gap and
the penetration this module reports are the same quantities the physics would
report, not a re-implementation of them.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Coarse search resolution and refinement halvings.  20 mm is a quarter of the
# wrist slide travel, so the coarse pass cannot skip a basin; each refinement
# halves it.
COARSE_STEP_M = 0.02
REFINE_HALVINGS = 6
# Frames sampled during the coarse pass.  Every frame is scored: feasibility is
# a property of the whole clip, and a base the coarse pass likes because it only
# looked at nine frames can be rejected outright by the frame it skipped -- which
# is exactly how the first version of this search ended up refining into a
# configuration the full clip does not admit.
COARSE_FRAME_STRIDE = 1
# A grasp needs at least this many fingertips at the closest-approach frame.
MIN_GRASP_TIPS = 2
# ... each within this distance of the box.
GRASP_TIP_RADIUS_M = 0.015
# The palm collision plate must stay at least this far off the box.  A grasp
# closes on the object with the *fingers*: the palm is the back of the hand and
# rests against nothing.  A configuration that reaches the box with the palm is
# not a grasp, and it is also a bad base to place the servos in -- the position
# actuators would spend tens of newtons holding the palm against the box, which
# shows up as a hand that sits centimetres away from its own commanded pose.  The
# scale is one fingertip pad radius, the same 12 mm capsule the pads are made of.
PLATE_CLEARANCE_M = 0.012

_FROMTO = np.zeros(6)


def _geom_distance(model, data, g1: int, g2: int) -> float:
    """Signed distance between two geoms (negative = overlapping).

    MuJoCo 3.16 declares the ``fromto`` scratch buffer without a usable default,
    so it has to be passed even though we never read it.
    """
    import mujoco

    return float(
        mujoco.mj_geomDistance(model, data, int(g1), int(g2), 1.0, _FROMTO)
    )


def classify_hand_geoms(model, hand_geom_ids, tip_body_ids) -> tuple[list[int], list[int]]:
    """(every hand geom, just the fingertip geoms).

    Keyed on geom->body ancestry, never on geom names: every finger geom in this
    scene comes from an unnamed <default> class, so a name-keyed lookup would
    silently return nothing.
    """
    return (
        sorted(hand_geom_ids),
        sorted(g for g in hand_geom_ids if model.geom_bodyid[g] in tip_body_ids),
    )


def _place(model, data, wrist_qadr, finger_qadr, base, slide_cmd, hinge, finger_cmd, t):
    data.qpos[:] = model.qpos0
    data.qpos[finger_qadr] = finger_cmd[t]
    data.qpos[wrist_qadr] = np.concatenate([base + slide_cmd[t], hinge[t]])
    import mujoco

    mujoco.mj_forward(model, data)


def _score(model, data, ctx, base: np.ndarray, frames) -> dict[str, Any]:
    """Closest fingertip approach to the box, and the deepest hand interpenetration."""
    import mujoco

    hand = ctx["hand_geoms"]
    tips = ctx["tip_geoms"]
    box = ctx["box_geom"]
    obj = ctx["obj_body"]
    plate = ctx["plate_geoms"]
    best_gap = np.inf
    best_t = -1
    best_tips_within = 0
    plate_gap = np.inf
    pen = 0.0
    for t in frames:
        _place(model, data, ctx["wrist_qadr"], ctx["finger_qadr"],
               base, ctx["slide_cmd"], ctx["hinge"], ctx["finger_cmd"], t)
        # penetration straight off MuJoCo's contact list: a hand geom buried in
        # the box shows up as a negative contact distance, which is exactly the
        # quantity the penetration criterion is stated in.
        for i in range(data.ncon):
            c = data.contact[i]
            if box in (int(c.geom1), int(c.geom2)) and -c.dist > pen:
                pen = -float(c.dist)
        for g in plate:
            d = _geom_distance(model, data, g, box)
            if d < plate_gap:
                plate_gap = d
        gaps = [_geom_distance(model, data, g, box) for g in tips]
        gap = min(gaps)
        if gap < best_gap:
            best_gap = gap
            best_t = t
            best_tips_within = int(sum(1 for g in gaps if g <= GRASP_TIP_RADIUS_M))
    # geometry of the winning frame, for the report
    detail: dict[str, Any] = {}
    if best_t >= 0:
        _place(model, data, ctx["wrist_qadr"], ctx["finger_qadr"],
               base, ctx["slide_cmd"], ctx["hinge"], ctx["finger_cmd"], best_t)
        obj_p = data.xpos[obj].copy()
        detail = {
            "closest_approach_frame": int(best_t),
            "fingertip_centroid_relative_to_box_m": (
                np.array([data.xpos[b].copy() for b in ctx["tip_bodies"]]).mean(axis=0)
                - obj_p
            ).tolist(),
            "fingertip_within_grasp_radius": best_tips_within,
            "box_half_size_m": ctx["box_half"].tolist(),
        }
    del mujoco
    return {
        "gap_m": float(best_gap),
        "penetration_m": float(pen),
        "palm_plate_gap_m": float(plate_gap),
        "tips_within_radius": best_tips_within,
        "rank": _rank(best_gap, best_tips_within),
        **detail,
    }


def _rank(gap: float, tips_within: int) -> tuple[int, float]:
    """Ordering key: a grasp is more than one fingertip, so count first.

    A single fingertip grazing the box minimises the gap but is not a grasp, and
    it contacts with a fraction of the fingers a real grasp does.  So the search
    maximises the number of fingertips inside the grasp radius (capped at
    MIN_GRASP_TIPS, past which more is not better), and only then minimises how
    far away that configuration still is.
    """
    return (0 if tips_within >= MIN_GRASP_TIPS else 1, gap)


def solve_base_xyz(model, data, ctx, lo: np.ndarray, hi: np.ndarray) -> dict[str, Any]:
    """Solve the constant floating-base translation.

    ``ctx`` carries the kinematic context (slide command, hinge command, finger
    command, joint/geom ids); ``lo``/``hi`` are the wrist-slide ctrlrange.  The
    returned base is the *mean* commanded slide, so the per-frame slide motion
    the retargeter produced is preserved exactly and only the base is solved.
    """
    t_all = range(ctx["slide_cmd"].shape[0])
    coarse_frames = list(range(0, ctx["slide_cmd"].shape[0], COARSE_FRAME_STRIDE))

    grid = [np.arange(lo[i], hi[i] + 1e-9, COARSE_STEP_M) for i in range(3)]
    best_rank: tuple | None = None
    best_base: np.ndarray | None = None
    n_feasible = 0
    n_grid = 0
    for x in grid[0]:
        for y in grid[1]:
            for z in grid[2]:
                cand = np.array([x, y, z])
                n_grid += 1
                sc = _score(model, data, ctx, cand, coarse_frames)
                if sc["penetration_m"] > 0.0 or sc["palm_plate_gap_m"] < PLATE_CLEARANCE_M:
                    continue
                n_feasible += 1
                if best_rank is None or sc["rank"] < best_rank:
                    best_rank, best_base = sc["rank"], cand

    result: dict[str, Any] = {
        "search": {
            "objective": (
                "minimise the closest fingertip-to-box gap over the whole clip, "
                "subject to zero hand-geom interpenetration of the box at every "
                "frame. The box is not moved; only the hand's base is solved."
            ),
            "grasp_axis": "x (the box's 120 mm span, frozen in the MJCF)",
            "score_source": "MuJoCo mj_geomDistance + mj_forward contact list",
            "coarse_step_m": COARSE_STEP_M,
            "coarse_frames": coarse_frames,
            "grid_points": n_grid,
            "collision_free_grid_points": n_feasible,
            "min_grasp_tips": MIN_GRASP_TIPS,
            "grasp_tip_radius_m": GRASP_TIP_RADIUS_M,
            "palm_plate_clearance_m": PLATE_CLEARANCE_M,
            "feasibility": (
                "no hand geom may interpenetrate the box, and the palm collision "
                "plate must stay a full fingertip-pad radius off it, at every frame"
            ),
        }
    }
    if best_base is None:
        result["status"] = "infeasible"
        result["reason"] = (
            "no base inside the wrist-slide ctrlrange keeps every hand geom out of "
            "the box, so no alignment of this clip can be executed here"
        )
        return result

    # Re-rank the coarse winner on the *full* clip before refining.  Carrying the
    # coarse subset's score into the refinement compares two different numbers
    # and silently stops the search at whatever the coarse pass happened to like.
    cand = best_base.copy()
    coarse_full = _score(model, data, ctx, cand, t_all)
    if coarse_full["penetration_m"] > 0.0 or coarse_full["palm_plate_gap_m"] < PLATE_CLEARANCE_M:
        # The coarse winner is not admissible on the full clip; there is no
        # second chance here, so say so rather than refining into it.
        result["status"] = "infeasible"
        result["reason"] = (
            "the best base the coarse pass found is not admissible over the full "
            "clip (hand-geom penetration "
            f"{coarse_full['penetration_m'] * 1000:.2f} mm, palm-plate gap "
            f"{coarse_full['palm_plate_gap_m'] * 1000:.2f} mm)"
        )
        result["coarse_winner_full_clip"] = coarse_full
        return result
    best_rank = coarse_full["rank"]
    step = COARSE_STEP_M
    for _ in range(REFINE_HALVINGS):
        improved = True
        while improved:
            improved = False
            for off in np.ndindex(3, 3, 3):
                trial = cand + (np.array(off) - 1) * step
                if np.any(trial < lo) or np.any(trial > hi):
                    continue
                sc = _score(model, data, ctx, trial, t_all)
                if (sc["penetration_m"] > 0.0
                        or sc["palm_plate_gap_m"] < PLATE_CLEARANCE_M
                        or sc["rank"] >= best_rank):
                    continue
                cand, best_rank, improved = trial, sc["rank"], True
        step /= 2.0

    final = _score(model, data, ctx, cand, t_all)
    result.update(
        {
            "status": ("solved"
                       if final["penetration_m"] <= 0.0
                       and final["palm_plate_gap_m"] >= PLATE_CLEARANCE_M
                       else "infeasible"),
            "grasp_tips_satisfied": bool(final["tips_within_radius"] >= MIN_GRASP_TIPS),
            "grasp_tips_note": (
                f"at the closest-approach frame {final['tips_within_radius']} of 4 "
                f"fingertips are within {GRASP_TIP_RADIUS_M * 1000:.0f} mm of the box"
            ),
            "base_slide_command_m": cand.tolist(),
            "palm_frame_world_position_m": (
                cand + ctx["hand_root_home"]
            ).tolist(),
            "refinement_step_final_m": step,
            **final,
        }
    )
    return result
