# r64–r85 ledger — the side cell's flat-pad grasp (W6b)

Campaign: the `v2d-mac` dual-arm MANO→UR5e retarget, the **side-mount cell**
(`grounding/assets/allegro_floating_ur5e_side_r37.xml`) — the layout the user
asked for: two forearms entering horizontally from the left/right frame edges,
palms facing each other around the box.

All numbers below are measured by the committed scripts and the committed
pipeline; nothing is estimated. The reproduction command is at the end.

---

## r65 — THE ROTATION-ERROR TRAP (this invalidated every earlier search)

The searches' IK used the rotation error

```python
e_r = R0 @ (0.5 * np.array([Rrel[2,1]-Rrel[1,2], Rrel[0,2]-Rrel[2,0],
                            Rrel[1,0]-Rrel[0,1]]))
```

which is `sin(theta) * axis` — it **vanishes at a 180° error**. The IK
therefore "converged" at a completely flipped orientation:

| reported | measured |
|---|---|
| `\|e_r\| = 0.000 deg` | `max\|R_achieved − R_target\| = 1.588` (a 180° flip) |

Every search result before r65 was invalid. The fix is
`scipy.spatial.transform.Rotation.from_matrix(R_target @ R_achieved.T).as_rotvec()`,
which is what `retarget/retarget_dual.py`'s own terminal IK has always used
(r18/r33). Affected: `tools/w6b/palm_pose_reach.py`,
`tools/w6b/palm_pose_reach2.py`, `tools/w6b/palm_pose_reach2.py`.

**Diagnostic rule**: an IK that reports `|e_p| = 0.000 mm` and
`|e_r| = 0.00 deg` must still print `max|R_achieved − R_target|`; > 1.0 is a flip.

---

## r63/r64 — WHAT THE REFERENCE GRASP ACTUALLY IS

Measured on the gate-3 seed (the known-good floating grasp, the one that
passes gate1):

* the index and middle pads sit **exactly on the box's end face**
  (x = 60.0 mm); their axes are **5.4° / 28°** from the face normal — the
  index pad presses **flat** with its 12 × 12 mm face;
* the ring pad is **7.5 mm past the face plane** and 20 mm above the box's
  top — it **wraps the top edge**, it does not press the face;
* the thumb is **115 mm clear** of the box;
* the reference press is **not** a zero-penetration touch: the pads
  penetrate 2–6 mm (the index tip −2.13 mm, the middle tip −5.77 mm, the
  middle distal link −5.94 mm). The pads are compliant; **that** geometry
  is what passes gate1.

Consequences that invalidated two earlier search designs:

* the palm-frame offsets are **scene-independent** (the hand's local tree is
  byte-identical in both scenes — measured), so the palm's world rotation is
  pinned by the finger joints, not free;
* the **ring finger must keep the reference posture**. Re-posturing it moves
  the four tips' centroid, and the gate's forward axis *is* that centroid's
  direction: with the ring re-postured the fwd reads 38.7°, with the
  reference's ring it reads **26.33°** — the reference's own value is 26.35°.

---

## r66 — THE SEARCH THAT SOLVES THE TERMINAL POSE

`tools/w6b/palm_pose_refine.py` (committed). For each candidate palm
orientation (the reference rotation rolled about the face normal) and palm
position:

1. the arm IK solves the 6 arm joints (the **proper** rotation error);
2. the index and middle fingers are IK'd (a real Jacobian IK on the tip geom)
   so their pads land **on the face plane**; the ring and thumb keep the
   reference posture;
3. everything is **measured with the gate's own metric**: the pad centres
   against the end-face rectangle, the four tips' centroid direction (the
   forward axis), the table clearance, the joint margins, the IK residuals.

**Result** (`runs/w6b/palm_pose_refine.json`):

| | right | left |
|---|---|---|
| the arm IK residual | 0.076 mm | 0.107 mm |
| the pad centres vs the face plane | index **0.0015 mm**, middle **0.0006 mm** | index **0.0016 mm**, middle **0.0008 mm** |
| the forward axis | **26.33°** (the reference 26.35°) | **27.58°** (the reference 27.61°) |
| the table clearance | +31.8 mm | +27.7 mm |
| the arm branch | no joint beyond ±3 rad | no joint beyond ±3 rad |

The R5 tolerance is **0.1164 mm**; both pads are inside it by two orders of
magnitude.

---

## r73–r85 — WHY THE EXECUTED STATE FAILED, AND THE FIXES

The command state was correct while the executed state was 350–450 mm away.
Five distinct causes, each measured:

1. **the joint-space terminal blend is not a safe path.** The straight line
   in joint space swung the hand out to x = 594 mm and **down through the
   table** — the palm reached z = 589 mm, 111 mm below the table's top, with
   the finger links 59 mm inside it.
2. **the scene's zero pose has the arms CROSSED** — the right arm's wrist
   sits at x = −117 mm and the left's at +119 mm. The right's path therefore
   ran through the left hand (the right's path hit the left's `ff_medial`),
   and the replay started 540 mm from the commanded path's first frame.
3. **the terminal IK flips branch by 2π.** The descent's IK changed the
   wrist_1 from 4.98 to −1.25 between two adjacent frames — the same
   physical pose, a different revolution. The servos cannot follow a 6.2 rad
   step: the executed state blew up by 5.4–8.3 rad.
4. **`model.qpos0` is the arm joints' reference configuration.** Writing the
   home pose into it changes the **forward kinematics** (the same joint
   vector then placed the palm 540 mm from the terminal's position). The
   home is therefore applied by `_reset_home`, not by touching `qpos0`.
5. **the vertical descent swept the fingers into the box's top edge** and
   pushed the box 27.9 mm. The staging pose is now **outboard along the
   face normal**, so the final phase is a straight inward approach — the
   horizontal entry the reference video shows.

The pipeline now builds the side cell's approach as a **Cartesian,
two-phase path**: home (the palm on its own side, high) → staging (the
terminal's position offset outboard along the face normal) → the terminal,
with the arm solved for each frame, the solution kept on one 2π branch, and
the descent tracking the terminal's arm configuration (an augmented IK
objective, weight 0.8).

---

## The result (measured, reproduced)

```
right: cmd rect 0.001 mm fwd 26.3 deg buried=False pass=True | executed rect 7.75 mm fwd 45.6 deg
left:  cmd rect 0.001 mm fwd 27.6 deg buried=False pass=True | executed rect 4.22 mm fwd 4.8 deg
the box's displacement over the whole replay: 16.2 mm
```

* **gate1 passes in the command state on both hands** — two pads inside the
  end-face rectangle (0.0006/0.0015 mm right, 0.0008/0.0016 mm left), the
  forward axis 26.33°/27.58° (the reference's own values), no burial;
* the right hand makes **16 end-face contact frames**, the first at control
  frame 338 (the terminal window);
* the executed pads land **7.75 mm (right) / 4.22 mm (left)** from the face.

### Open defects (not claimed as solved) — SUPERSEDED by r86 below

| # | defect | measured (r84) | measured (r87) |
|---|---|---|---|
| 1 | the right's executed forward axis | **45.61°** (the limit is 35°) | **24.37°** (the command is 26.33°) |
| 2 | the box's displacement over the replay | **16.2 mm** | **0.54 mm** (max 1.11 mm) |
| 3 | the executed pad distance | 7.75 / 4.22 mm | 9.38 / 10.44 mm |
| 4 | the arm's collision proxy inside the table | 7.4 / 13.6 mm | **0** (no support contact on the right) |
| 5 | the path's Cartesian tracking | not measured | **up to 947 / 1449 mm** (OPEN) |

Defect 2 is the known one: the pads are commanded with their **centres on
the face plane** (which is what gate1 measures and what the reference grasp
does — the pads penetrate 1.5–3 mm), so the compliant pads press the box.
The fix is a lighter touch, which needs the gate's criterion and the
reference's posture reconciled; it is not done.

---

## Reproduction

```bash
cd ~/v2d-mac
# 1) the terminal pose search (writes runs/w6b/palm_pose_refine.json)
runs/.venv-mj/bin/python tools/w6b/palm_pose_refine.py

# 2) the pipeline on the side cell
W6B_R75_SIDE_APPROACH=1 W6B_R63_TARGET=1 W6B_R18_NATIVE_TARGET=1 \
W6B_R50_PRELOAD_MM=0.05 W6B_R52_FINGER_TGC=1 W6B_R50_CURL=1.25 \
W6B_R39_TGC=0 W6B_R36_GSTATIC=0 W6B_R36_I=0 W6B_R39_HOLD=60 \
W6B_SCENE=ur5e \
W6B_SCENE_XML=grounding/assets/allegro_floating_ur5e_side_r37.xml \
W6B_R23_FLYOVER_MM=250 W6B_R22_MARGIN_MM=8 \
runs/.venv-retarget/bin/python retarget/retarget_dual.py
```

The knobs (all with the winning value as the default):

| variable | value | meaning |
|---|---|---|
| `W6B_R75_SIDE_APPROACH` | 1 | the side cell's Cartesian two-phase approach |
| `W6B_R75_STAGE_MM` | 100 | the staging offset outboard along the face normal |
| `W6B_R82_DESCENT_LAM` | 0.8 | the descent's arm-configuration weight |
| `W6B_R84_STAGE_MODE` | normal | the staging is outboard, not above |
| `W6B_R63_TARGET` | 1 | the terminal aims at the searched pose |
| `W6B_R63_REFERENCE` | `runs/w6b/palm_pose_refine.json` | the search's output |

The debug hooks (all off by default) name the evidence:
`W6B_DBG_TRACK` (the commanded vs the executed arm joints),
`W6B_DBG_CONTACT` (the final frame's contacts),
`W6B_DBG_FORCE` (the actuator and constraint forces),
`W6B_DBG_PATH` (the Cartesian path's per-step clearance),
`W6B_DBG_GATE` (the gate's command-state inputs),
`W6B_DBG_Q` (the terminal arm vector).


---

## r86 — THE ARM WAS INSIDE THE TABLE (the one cause of both r84 defects)

The r84 executed state (the right's forward axis 45.6°, the box flying
16.2 mm) had **one** cause, and it was not the servo stiffness.

**The measurement.** The replay's contact list at the terminal frame:

```
('ur5e_r_wrist_2_link', 'world', -2.41)
```

The right's wrist_2 mesh was 2.30 mm INSIDE the table slab, the table
pushed back with **30.0 N**, and the wrist_2's actuator force read
**-28.0** — exactly the actuator's ±28 N·m limit. The wrist was
saturated, the hand was twisted off the face, the right's pads lost the
face, and the box (pushed by a hand that was in the wrong place) was
shoved 16.2 mm into the vise's corner.

**The static check that missed it.** The search measured the table
clearance on the **HAND's** geoms only, and the arm's clearance on its
**MESH** geometry. Two errors:

* the UR5e's mesh geometry extends 40–50 mm below its collision
  capsules, so the mesh-based measure read **-44 mm** for a pose whose
  collision proxy was **-7 mm**;
* the arm was never checked at all.

The corrected measure — `mj_geomDistance` from each of the arm's
**collision** geoms (contype ≠ 0) to the table slab — gives the old
path-verified solution **-7.36 mm (right) / -13.55 mm (left)**: the arm's
collision proxy was inside the table in the pose that "worked".

**The fix.** `tools/w6b/palm_pose_refine.py` now treats the arm's
collision-proxy clearance as a HARD constraint (≥ 12 mm, so the measured
±0.13 rad tracking error cannot push the wrist in). The search returns:

| | right | left |
|---|---|---|
| the arm's collision clearance | **+13.3 mm** | **+16.8 mm** |
| the hand's clearance | +34.6 mm | +40.6 mm |
| the pad centres vs the face | 0.0001 / 0.0001 mm | 0.0076 / 0.0003 mm |
| the forward axis | 26.34° | 27.61° |

**The measured effect** (the same command stream otherwise):

| | r84 | r87 |
|---|---|---|
| the right's executed forward axis | 45.61° | **24.37°** (the command 26.33°) |
| the left's executed forward axis | 4.82° | **26.98°** (the command 27.58°) |
| the box's displacement | 16.2 mm | **0.54 mm** |
| the arm's tracking error at the terminal | 0.116–0.132 rad | **0.007–0.015 rad** |
| the support contacts (the right) | the wrist in the table | **0** |
| the executed pad distance | 7.75 / 4.22 mm | 9.38 / 10.44 mm |

## r87 — FOUR MORE MEASURED DEFECTS IN THE PATH

1. **The regularizer's sign was inverted.** The descent's IK adds a term
   pulling the arm toward the terminal's configuration; the residual was
   `λ·(dq + q_ref − q)`, i.e. it asked the step to move the arm **away**
   from the reference. The range clip then pinned every joint at its
   limit. Fixed to `λ·(dq + q − q_ref)`.
2. **The path's last frame was garbage.** Measured: the path's final
   frame was `[6.069, 0, 3.142, 6.283, 0, 0]` — the hand **1.6 m in the
   air** — while the pipeline's print showed the terminal's arm (it
   printed a different array). The replay therefore swung the arm wildly
   and then snapped to the hold, which is why the terminal state looked
   good while the approach did not.
3. **The IK's step was unbounded**, so one ill-conditioned iteration
   slammed the joints into their limits. A projected-gradient term now
   refuses to push a joint that already sits at its limit.
4. **The gate5's "constrained solve" leg measured the wrong target.** It
   compares the side path against the **floating cell's** registered task
   trajectory — a different trajectory in a different cell — and reported
   848 mm of "deviation" for a path that is on its own target. The side
   path's own achieved-vs-target deviation is now recorded separately
   (`gate5_palm_deviation.side_path_palm_deviation`).

### The KNOWN-BAD numbers that did NOT survive (recorded so they are not re-tried)

| tried | result |
|---|---|
| the finger servo stiffness ×3 … ×30 | the executed error 81–338 mm: the Allegro's finger servos are unstable at the scene's 2 ms timestep (the distal links need dt < 0.5 ms) |
| the physics timestep 0.5 ms / 0.25 ms | the executed error 18–228 mm: the 2 ms integration error was accidentally helping |
| the IK's step bounded to 0.15 / 0.25 / 0.6 rad | the IK cannot converge in 150 iterations: 188–373 mm |
| the phase-A non-convergence guard | freezes the path (the IK legitimately lags a fast Cartesian move): 188–373 mm |
| the finger/box mass scan | not run in r87 |

### The configuration that reproduces (the defaults in the code)

`W6B_R86_LAM_A_RIGHT = 1.5`, `W6B_R86_LAM_A_LEFT = 0.6`,
`W6B_R82_DESCENT_LAM = 0.8`, `W6B_R75_STAGE_MM = 100`.

**The phase-A tracking weight is a narrow, fragile knob** — the left's
region is 0.5–0.6 (0.45 sends the box 2018 mm). It is disclosed here
rather than hidden: the path's IK basin is knife-edged, and that is the
same root cause as the open defect below.

### The remaining open defect (r87)

**The path's Cartesian tracking is broken.** The side path's own
achieved-vs-target palm deviation reaches **947 mm (right) / 1449 mm
(left)** — the arm does not follow the intended home → staging →
terminal line; it flails and only the final HOLD lands the arm on the
terminal. The terminal state is correct (the arm tracks to 0.007 rad,
the pads 9–11 mm from the face, the fwd within 2° of the command, the
box 0.54 mm), but the **approach** in the render is wrong, and gate5
fails for that reason.

The cause is now localised: the path's IK diverges in phase A (the
Cartesian move from the home to the staging), the divergence is not
detected as such (the guard's absolute threshold freezes the path, the
relative criterion does not catch it), and the λ regulariser — which
exists only to pick the IK branch — fights the Cartesian task instead of
resolving it. The next step is a path construction that keeps the
Cartesian task exact and selects the branch by the ±2π wrap alone
(a null-space-free 6-DOF arm has no null space, so the branch must come
from the wrap, not from a competing objective).
