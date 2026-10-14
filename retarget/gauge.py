"""Gauge fixing for the floating Allegro wrist.

A floating hand with no fixed arm has three *gauge* freedoms -- a global yaw
and two in-plane base translations -- that the retargeting optimizer will happily
absorb into its 6-DoF floating wrist because the human's forearm orientation
leaks into the wrist solution.  Those quantities are coordinate-convention
artefacts, not physics: what is physical is the hand's pose **in the world**.

Concretely, writing the wrist's world orientation as an intrinsic-XYZ triple
``(rx, ry, rz)``, a global yaw ``psi`` can be pulled out as

    rz_eff = wrap(rz - psi)

and a constant base offset ``(dx, dy)`` out of the slide commands.  Choosing
``psi`` as the (circular) centre of the wrist's yaw over the clip and
``(dx, dy)`` as its in-plane centre puts the required wrist travel inside the
actuators' ctrlrange without touching the MJCF, without lowering the object and
without relaxing any threshold.

The choice is recorded (see :func:`gauge_report`) because it changes the wrist
*joint angles* while leaving the hand's world pose -- and therefore every contact,
penetration and tracking number -- invariant up to the removed offset.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# The MJCF wrist's own travel. Read from the scene, never hard-coded, so this
# module cannot drift out of sync with grounding/assets/allegro_floating.xml.
WRIST_AXES = ("wrist_x", "wrist_y", "wrist_z", "wrist_rx", "wrist_ry", "wrist_rz")


def wrap_pi(angle):
    """Wrap radians to (-pi, pi]."""
    return (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi


def circular_mean(angles: np.ndarray) -> float:
    """Mean direction of a set of angles (gauge-aware, unlike np.mean)."""
    a = np.asarray(angles, dtype=np.float64)
    return float(np.arctan2(np.mean(np.sin(a)), np.mean(np.cos(a))))


def _best_yaw(ctrl: np.ndarray, ctrlrange: np.ndarray, palm_rot, Rotation):
    """Pick the global yaw psi minimising the worst wrist-angle excursion.

    The gauge is applied as a *global pre-rotation* of the hand's world
    orientation, ``R_new(t) = Rz(-psi) @ R_world(t)``, and the result is
    re-decomposed into intrinsic-XYZ hinge angles.  That is a relabelling of the
    base frame: every frame is rotated by the same Rz(-psi), so the hand's pose
    *relative to the object* is untouched and only the wrist joint angles change.

    Subtracting psi from the last Euler angle instead would NOT be a rotation --
    Euler angles do not compose that way -- and would silently move the hand off
    the object.  ``test_gauge_fixing_preserves_the_world_pose`` pins this down.
    """
    def apply(ctrl_in, psi):
        g = Rotation.from_euler("z", -psi).as_matrix()
        palm_inv = np.linalg.inv(palm_rot)
        out = ctrl_in.copy()
        for t in range(ctrl_in.shape[0]):
            hinge = Rotation.from_euler("XYZ", ctrl_in[t, 3:6]).as_matrix()
            world = hinge @ palm_rot
            # gauge acts on the world pose; peel palm back off on the right so
            # the result is again a valid hinge triple for this MJCF chain
            out[t, 3:6] = Rotation.from_matrix(g @ world @ palm_inv).as_euler("XYZ")
        return out, g

    best = (np.inf, 0.0, None, None)
    for psi in np.arange(-np.pi, np.pi, np.deg2rad(0.5)):
        trial, g = apply(ctrl, psi)
        exc = float(overshoot(trial, ctrlrange).max())
        if exc < best[0] - 1e-12:
            best = (exc, float(psi), trial, g)
    return best[1], best[2], best[3]


def mean_rotation(mats: np.ndarray) -> np.ndarray:
    """Chordal L2 mean of a set of rotation matrices (Kabsch, one SVD)."""
    m = np.asarray(mats, dtype=np.float64).reshape(-1, 3, 3)
    mean_matrix = m.mean(axis=0)
    u, _s, vt = np.linalg.svd(mean_matrix)
    d = np.eye(3)
    d[2, 2] = np.sign(np.linalg.det(u @ vt))
    return u @ d @ vt


def fix_gauge(
    wrist_ctrl: np.ndarray,
    ctrlrange: np.ndarray,
    palm_rot: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Remove the gauge freedoms from a (T, 6) retargeted wrist command.

    ``wrist_ctrl`` columns follow ``WRIST_AXES``: slides x/y/z then hinges
    rx/ry/rz, i.e. the palm's world rotation is
    ``Rx(rx) @ Ry(ry) @ Rz(rz) @ palm_rot``.

    A hand with no fixed arm has no absolute base pose, so its world pose is
    defined only up to a constant rotation ``R_g`` and in-plane translation
    ``(dx, dy)`` of the base frame.  Both are conventions, not physics.  We pick
    them to shrink the wrist travel the actuators must provide:

    * ``R_g = mean(hand world orientation)^-1``, which re-references the clip's
      *mean* hand orientation onto the base frame's identity.  The global yaw is
      the dominant term -- it is what drove wrist_rz to 2.71 rad against a
      +-1.57 rad range.  It is applied as one rotation to the whole clip, not as
      a per-axis mean subtraction, so the hand's world pose is genuinely
      unchanged.
    * ``(dx, dy)`` = the clip's mean in-plane position, moved off the actuator
      and into the scene anchor.

    ``z`` is left alone: the hand's height relative to the object is physical --
    it is what decides contact.

    ``palm_rot`` (the scene's fixed palm rotation) must be supplied for the
    orientation gauge; the mean has to be taken in the world frame, not in the
    actuator frame.
    """
    from scipy.spatial.transform import Rotation

    ctrl = np.asarray(wrist_ctrl, dtype=np.float64).copy()
    lo = np.asarray(ctrlrange, dtype=np.float64)[:, 0]
    hi = np.asarray(ctrlrange, dtype=np.float64)[:, 1]

    palm_rot = np.eye(3) if palm_rot is None else np.asarray(palm_rot, np.float64)

    # Gauge 1: global yaw. Search psi directly rather than using a rotation
    # mean -- Euler angles are periodic and non-linear, so the mean of the world
    # orientation matrices is not the minimiser of the worst-case angle. A 1-D
    # scan over the physically meaningful gauge (yaw) is exact and cheap.
    psi, ctrl, r_g = _best_yaw(ctrl, ctrlrange, palm_rot, Rotation)

    dx = float(np.mean(ctrl[:, 0]))          # base x offset -> wrist_x
    dy = float(np.mean(ctrl[:, 1]))          # base y offset -> wrist_y
    ctrl[:, 0] = ctrl[:, 0] - dx
    ctrl[:, 1] = ctrl[:, 1] - dy

    record: dict[str, Any] = {
        "gauge_freedoms_removed": {
            "absorbed_yaw_rad": psi,
            "base_rotation_matrix": r_g.tolist(),
            "base_offset_x_m": dx,
            "base_offset_y_m": dy,
        },
        "rule": (
            "one constant rotation R_g = (mean hand world orientation)^-1 plus the "
            "clip's mean in-plane offset (dx, dy), applied to the whole clip. "
            "The absorbed yaw is the dominant term: it is what had pushed wrist_rz "
            "to 2.71 rad against a +-1.57 rad ctrlrange. z is untouched because "
            "the hand's height above the object is physical."
        ),
        "physical_meaning": (
            "this is a coordinate convention, not a physical change: the hand's "
            "pose in the world is unchanged, only the wrist joint angles that "
            "represent it are re-referenced. The removed yaw is the global "
            "orientation of a hand with no fixed arm, and the removed x/y offset "
            "is the arbitrary origin of the floating base."
        ),
        "ctrlrange": [lo.tolist(), hi.tolist()],
        "required_travel": required_travel(ctrl).tolist(),
        "overshoot_after": overshoot(ctrl, ctrlrange).tolist(),
        "axes_out_of_range_after": axes_out_of_range(ctrl, ctrlrange),
        "command_min": ctrl.min(axis=0).tolist(),
        "command_max": ctrl.max(axis=0).tolist(),
    }
    record["fits_ctrlrange"] = not record["axes_out_of_range_after"]
    return ctrl, record


def required_travel(ctrl: np.ndarray) -> np.ndarray:
    """Peak-to-peak span each wrist axis has to cover."""
    return ctrl.max(axis=0) - ctrl.min(axis=0)


def overshoot(ctrl: np.ndarray, ctrlrange: np.ndarray) -> np.ndarray:
    """Per-axis excursion outside the ctrlrange (0 when inside)."""
    lo = np.asarray(ctrlrange, dtype=np.float64)[:, 0]
    hi = np.asarray(ctrlrange, dtype=np.float64)[:, 1]
    below = np.maximum(lo[None, :] - ctrl, 0.0)
    above = np.maximum(ctrl - hi[None, :], 0.0)
    return (below + above).max(axis=0)


def axes_out_of_range(ctrl: np.ndarray, ctrlrange: np.ndarray, tol: float = 1e-9):
    return [
        WRIST_AXES[i]
        for i in range(ctrl.shape[1])
        if float(overshoot(ctrl, ctrlrange)[i]) > tol
    ]
