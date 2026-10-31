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
| the joint-space path (`W6B_R88_JOINT_PATH=1`) | it does not diverge (it ends exactly on the terminal) but the interpolation folds the arm into its own hand: the worst contact **-40.0 mm** (`ur5e_r_wrist_1_link<->palm`, right) and **-39.9 mm** (`world<->rf_medial_left`, left) — the r12 joint-space-blend failure, reproduced and localised. It needs a waypoint path (home → a clear intermediate → the terminal), not a straight interpolation. |
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
resolving it.

**The next step** (measured, not guessed): the joint-space path proves
the Cartesian one is not the only option — it does not diverge — but a
straight interpolation folds the wrist into the hand (-40.0 mm). So the
path needs **waypoints**: home → a clear intermediate pose → the
terminal, with the clearance checked per segment, and the intermediate
pose searched for (not guessed). The search machinery already exists
(`tools/w6b/palm_pose_refine.py` scores a candidate with the table
clearance, the joint margin and the natural branch); it needs a second
candidate set — the intermediate poses, scored for the clearance of the
WHOLE segment home→intermediate→terminal, not just the endpoint.


---

## r89-r100: the approach path, rebuilt

### What the r88 state actually was

The r87 numbers (the right 9.38 mm / the left 10.44 mm) were measured with
**two latent bugs** that had been in the pipeline for many rounds, and the
"the path's own palm deviation 1421 mm" was the visible symptom of the
second one.

| # | the defect | how it was found | the effect |
|---|---|---|---|
| 1 | the replay's reset wrote the home into joints named `ur5e_right_*` / `ur5e_left_*`, but the scene's joints are `ur5e_r_*` / `ur5e_l_*` -- so `mj_name2id` returned -1 and **the home was never applied** | the executed palm at frame 0 sat at the scene's crossed zero pose while the path's first frame was the home | the servos were asked to make a 600 mm jump on frame 1; the executed palm left the home by 190-300 mm in one frame |
| 2 | the path's frames were produced by solvers that each chose their 2*pi branch independently, so consecutive frames could differ by exactly 2*pi | the dbg-track error read 6.283 rad at t=0 while the physical pose matched | the position servo was told to turn a joint a full revolution -> the arm swung across the cell |
| 3 | the path's LAST frame converged to a *different arm configuration* than the searched terminal (the palm pose has several solutions; the elbow-up one differs by 2.56 rad in the shoulder_pan and 3.56 rad in the wrist_2) | the dbg-track at t=320: the command jumped from the path's last frame to the hold | the HOLD was a multi-radian jump; the executed pads ended 556 mm from the face even though the commanded path tracked the line perfectly (deviation 0 mm) |
| 4 | the descent's IK had no trust region and returned its last iterate, not its best | the phase-A residual probe: the residual stayed at 350 mm for 100 frames | a near-singular Jacobian made the damped step 1e4 x the residual; one iteration threw the arm a metre away and every later step inherited that state |

### The four fixes (all measured)

| fix | what it does | the measured effect |
|---|---|---|
| **r90** the path is unwrapped frame-to-frame in the 2*pi branch, and the reset state is the path's OWN first frame | the branch is continuous and the servos start exactly where the commands start | the executed-vs-commanded tracking: 0.000 rad at t=0 (was 6.283) |
| **r92** the IK is trust-region limited (0.05 rad/iteration) and returns its BEST iterate | a diverging step can no longer poison the rest of the path | the path's own palm deviation 1421 -> 364 mm |
| **r94** the path is solved BACKWARD from the terminal (the default) | the path's last frame IS the terminal's configuration by construction, and the whole path is in the terminal's branch -- no posture regulariser is needed, so the Cartesian task is not fought | the path's own palm deviation 364 -> **0 mm**, the executed terminal 9.90/10.04 mm |
| **r96/r97/r98** the reset also sets the fingers, and the finger command is clipped to the joints' ranges with a 5 mrad margin | the scene's zero pose puts the thumb's joint 0 at 0 while that joint's range starts at 0.263 rad | the gate4 finger residual 0.130 -> 0.0 (right) / 8.6e-07 rad (left) |

### The measured result (the final command, no env overrides)

```
W6B_R75_SIDE_APPROACH=1 W6B_R63_TARGET=1 W6B_R18_NATIVE_TARGET=1
W6B_R50_PRELOAD_MM=0.05 W6B_R50_CURL=1.25 W6B_R39_TGC=0 W6B_R36_GSTATIC=0
W6B_R36_I=0 W6B_R39_HOLD=60 W6B_SCENE=ur5e
W6B_SCENE_XML=grounding/assets/allegro_floating_ur5e_side_r37.xml
W6B_R23_FLYOVER_MM=250 W6B_R22_MARGIN_MM=8
runs/.venv-retarget/bin/python retarget/retarget_dual.py
```

| metric | right | left | the baseline (the committed floating cell) |
|---|---|---|---|
| the command's pad distance | 0.000 mm | 0.000 mm | - |
| **the executed pad distance** | **9.90 mm** | **10.04 mm** | the gate1 cmd 15.9/1.6 mm; the executed 320-384 mm |
| the executed forward axis | 33.8 deg (the cmd 26.3) | 33.1 deg (the cmd 27.6) | - |
| the path's own palm deviation | **2.60 mm** | **2.59 mm** | 1421 mm (the side path) / 163.8 mm (the registered leg) |
| the executed-vs-commanded deviation | 17.0 mm | 11.0 mm (cap 40) | - |
| the box's displacement | **0.24 mm** | | 16.2 mm (the r84 side cell) |
| the gate4 finger residual | 0.0 rad | 8.6e-07 rad | 1.63 / 1.52 rad |
| gate1 / gate2 / gate3 / gate5 | pass / pass / pass / pass | | fail / fail / fail / fail |

The executed trajectory is the reference motion: the palm descends vertically
(272, -30, 1148) -> (278, -25, 806) and then enters horizontally to
(181, -25, 792); the left mirrors it.

### The two red gates, and why they are red

| gate | the measurement | why |
|---|---|---|
| **gate4_joint_limits** | the left's thumb joint 0 is 8.6e-07 rad (0.86 um) outside its own stop for 16 frames | MuJoCo's joint-limit and contact constraints are SOFT: the contact on the thumb pushes it a micrometre past the stop. Stiffening the limit (jnt_solref 0.02 -> 0.0005 s) and the solver (100 -> 200 iterations) did not remove it, so the residual is the contact's compliance, not our command (the command keeps a 5 mrad margin). The gate demands exactly 0.0, so it is reported as FAIL. The gate is NOT changed: the committed baseline's residual on the same leg is 1.63 rad, i.e. 1.9 million times larger. |
| **wrist_referenced_error** | the confidence-weighted median 30.1 mm (right) / 40.4 mm (left), 0.318 / 0.427 of the span (the limit 0.15) | this is a VIDEO-FIDELITY metric: the executed palm's lateral vs the MediaPipe wrist's lateral over the video's approach window. The side cell does not use the video's trajectory (it is a constructed descent + horizontal entry), so the metric compares two different motions. The committed floating cell FAILS the same metric (the MediaPipe wrist signal is partial: the detected px jump between frames), so this is a known-red leg of the baseline, not a regression of the side cell. |

### The open defect

The **executed** pad distance is 9.90/10.04 mm, not the < 5 mm the r84 goal
asked for. The command is exact (0.000 mm) and the path tracks the line to
2.6 mm; the residual is the servo's own tracking (the executed-vs-commanded
deviation is 17.0/11.0 mm) plus the pads' contact compliance. The arm's
gravity compensation was measured and is NOT the fix: with it the right
regresses to 19.96 mm and gate2 breaks (the finger gravity cancel behaves the
same way: off 9.90/10.04, hold-only 10.43/10.65, whole replay 10.42/10.73).

### Tried and failed (so they are not re-tried)

| the attempt | the measurement |
|---|---|
| the joint-space path (r88) | the wrist folds into the hand: -40.0 mm (`ur5e_r_wrist_1_link<->palm`) |
| the waypoint search (r89), the candidates scored by the path's clearance | the joint interpolation swings the palm in a 2.3 m arc anyway - the palm's path is NOT controlled by the joint path |
| the posture regulariser lambda (the terminal's configuration as the reference) | it fights the Cartesian task: the path's deviation 0 -> 303/352 mm, and it is what produced the 1421 mm divergence in the first place |
| the terminal-branch +-2*pi wrap (r87) | neutral for the working configuration, kept as a switch |
| the blend of the path's last frames into the terminal (r93) | the two configurations are at the same pose but the interpolation sweeps the palm 528 mm |
| the arm's gravity compensation (r39/r36) | the right 9.90 -> 19.96 mm and gate2 breaks |
| the finger gravity compensation (r52, three modes) | off is the best of the three (see above) |
| the phase split (A/B from 40/261 to 260/41) | the executed terminal moves 9.6-9.9 mm; the wrist metric 0.308-0.319 -- insensitive, so the split is a shape choice, not a tuning knob |
| the stiffened joint limit + 200 solver iterations (r99) | the 0.86 um residual is unchanged -> it is the contact's compliance |


---

## r101-r102: the gravity compensations' sign, and the mass scan

### r101: every gravity compensation in the pipeline had the sign inverted

MuJoCo's equation of motion is

```
M qdd = tau_actuator + tau_applied + tau_constraint - qfrc_bias
```

so at rest (qdd = 0) with a P-servo at equilibrium, the applied torque
must EQUAL **+qfrc_bias**. Every compensation in this file applied
**-qfrc_bias** -- it pushed the arm AGAINST gravity and doubled the
droop. That is why every earlier attempt to use the compensation
regressed and was written off as "the compensation destabilises the
replay" (the r28 note: 687 mm; the r36/r39/r52 notes: the executed
terminal 9.90 -> 19.96 mm).

Measured at the terminal hold (the dbg-force probe): the right's
shoulder_lift carries 30.7 N.m of gravity with kp 2200 N.m/rad, so the
P-servo's droop is 14 mrad -- about 5.6 mm at the tool, which IS the
9.9 mm executed distance.

| the compensation | before (the sign was wrong) | after (the sign is right) |
|---|---|---|
| the r36 static gravity bias | the executed terminal 19.72/10.21 mm | **4.58/6.06 mm** |
| the r39 terminal gravity cancel | 19.96/10.53 mm, gate2 broke | (not used: r36 is better) |
| the r52 finger gravity cancel | the right 11.94 mm | (not used: the right 12.25 mm) |

The r36 static bias is now ON by default; the r39 and r52 cancels stay
off (measured worse).

### The final measured result

```
W6B_R75_SIDE_APPROACH=1 W6B_R63_TARGET=1 W6B_R18_NATIVE_TARGET=1
W6B_R50_PRELOAD_MM=0.05 W6B_R50_CURL=1.25 W6B_R39_TGC=0 W6B_R36_GSTATIC=1
W6B_R36_I=0 W6B_R39_HOLD=60 W6B_SCENE=ur5e
W6B_SCENE_XML=grounding/assets/allegro_floating_ur5e_side_r37.xml
W6B_R23_FLYOVER_MM=250 W6B_R22_MARGIN_MM=8
runs/.venv-retarget/bin/python retarget/retarget_dual.py
```

| metric | right | left |
|---|---|---|
| the command's pad distance | 0.000 mm | 0.000 mm |
| **the executed pad distance** | **4.58 mm** | **6.06 mm** |
| the executed forward axis | 33.3 deg (the cmd 26.3) | 32.9 deg (the cmd 27.6) |
| the path's own palm deviation | 2.60 mm | 2.59 mm |
| the executed-vs-commanded deviation | 16.70 mm (cap 40) | 11.02 mm |
| the box's displacement | 1.98 mm | |
| gate1 / gate2 / gate3 / gate5 | **pass / pass / pass / pass** | |

Reproduced twice, byte-identical (the pipeline is deterministic: the
same command gives the same numbers).

### r102: the mass robustness scan

`W6B_R102_BOX_MASS_KG` overrides the box's mass (and scales its inertia
with it, so the override is a consistent rigid body). The scene's box is
0.4004 kg.

| the box's mass | the right's executed | the left's executed | the box's displacement | the gates |
|---|---|---|---|---|
| 0.15 kg | 4.09 mm | 6.12 mm | 2.90 mm | gate1/2/3/5 pass |
| 0.25 kg | 4.17 mm | 6.16 mm | 2.65 mm | gate1/2/3/5 pass |
| 0.40 kg (the scene's) | 4.56 mm | 6.06 mm | 1.98 mm | gate1/2/3/5 pass |

The terminal's accuracy is flat across a 2.7x mass range: the residual is
not the box's weight but the arm's own compliance under the contact.

### The open defects (unchanged, stated)

1. the left's executed pad distance is 6.06 mm (the goal was < 5 mm; the
   right is at 4.58 mm) -- the same mechanism as the right's, a factor
   1.3x, from the left's mount (its shoulder_lift carries 21.9 N.m of
   gravity against the right's 30.7 N.m but its lever arm is longer).
2. gate4 fails on 8.6e-07 rad (0.86 um) of the left's thumb past its own
   stop -- MuJoCo's soft contact/limit compliance. Not changed: the gate
   is not touched and the committed baseline's residual on the same leg is
   1.63 rad.
3. wrist_referenced_error is a video-fidelity metric that the side cell's
   constructed approach does not reproduce; the committed baseline fails
   the same leg.


---

## r103-r104: the evidence camera and the evidence frame

**r103 -- the side cell's camera.** The triptych's side-cell preset was
`az 90 / el -14 / distance 0.40` with the lookat at z = 0.68: it framed
the box alone and cut both arms out of the picture, so the evidence image
showed a scene with no hands in it. The camera now follows the measured
composition target (`viz/composition_target.json`): the lookat at
z = 0.86, elevation -16 deg, distance 0.95 m. The rendered frame is now
the reference layout -- the box at the frame's centre, the two forearms
entering horizontally from the left and right edges at mid-height.

**r104 -- the contact evidence frame.** The triptych's "two pads in the
face rectangle" scan keeps the frame with the maximum pad count, and the
rect tolerance is the gate3 seed's 3 mm while the executed terminal sits
4.58/6.06 mm from the face -- so NO frame scored, and the scan kept the
FIRST frame with the maximum score, i.e. a descent frame with the hands
a metre above the box. The evidence image showed a scene with no contact
at all. The scan now falls back to the terminal frame (the honest one:
the hands ARE on the face there) and the pad counts are reported beside
it either way.


---

## r105-r106 + v10 的准备证据（来自 v9 评审 9.3/10 的具体要求）

v9 裁决 **9.3/10 APPROVE with conditions**（`docs/gpt_v9_review.md`）。评审指定的
下一轮唯一目标是**抬离可行性门**；本节记录评审在本轮点名要求、且本轮已补上的
三件事。

### r105：wrist 指标的缺失观测被掩膜掉（评审第 4 问）

评审不接受"wrist 门不适用"的归因：**MediaPipe 的信号缺失要用置信度、缺失掩码与
时间区间处理，不能把 0 当成真实腕坐标**。实测确认了这一点：`(0,0)` 的腕部像素
是检测器的缺失哨兵，而它此前被算进了配准参考的均值（把参考点拉向原点）并当成
真实坐标参与误差。

修正：缺失帧（present 标志为假 **或** 像素为 (0,0)）现在同时从**配准参考**与
**误差**里掩膜掉，掩膜帧数一并发布。

| 侧 | 有效观测帧 | 掩膜掉的缺失帧 | 修正前 | 修正后 |
|---|---:|---:|---:|---:|
| 右 | 130 | 21 | 0.318 | **0.318** |
| 左 | 144 | 7 | 0.427 | **0.427** |

结论不变：**wrist 门仍然 FAIL**，而且失败是真实的（不是缺失帧造成的假象）。

### r106：分支连续性指标

命令态腕关节流的逐帧最大 |Δq| 现在发布在报告里
（`per_side.<side>.commanded_wrist_max_joint_step_rad`）。它就是分支切换的探针：
2π 换支会在这里读到 ~6.283 rad。

### 泛化扫描（评审第 1 问：路径求解不能只对一条构造路径成立）

`tools/w6b/generalization_scan.py`：五种接近几何（staging 沿端面法向 60 / 100 /
160 mm × 下降高度 220 / 350 / 550 mm），每种跑一遍已提交配方，检查路径自身偏差、
分支连续性、接近段支撑接触与执行态终端。

| 几何 | stage | lift | 侧 | 路径自身偏差 mm | 分支步长 rad | 支撑接触帧 | 执行态 mm | 前向轴 |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| 交付几何 | 100 | 350 | 右 | 2.60 | 0.0099 | 0 | 4.58 | 33.3 |
| 交付几何 | 100 | 350 | 左 | 2.59 | 0.0095 | 0 | 6.06 | 32.9 |
| 更近的接近 | 60 | 350 | 右 | 1.81 | 0.0101 | 0 | 4.69 | 33.4 |
| 更近的接近 | 60 | 350 | 左 | 1.81 | 0.0097 | 0 | 6.01 | 33.0 |
| 更远的接近 | 160 | 350 | 右 | 4.16 | 0.0097 | 0 | 4.42 | 33.2 |
| 更远的接近 | 160 | 350 | 左 | 4.16 | 0.0094 | 0 | 6.15 | 32.9 |
| 更高的下降 | 100 | 550 | 右 | 2.60 | 0.0134 | 0 | 4.57 | 33.3 |
| 更高的下降 | 100 | 550 | 左 | 2.59 | 0.0130 | 0 | 6.06 | 32.9 |
| 更低的下降 | 100 | 220 | 右 | 2.60 | 0.0071 | 0 | 4.58 | 33.3 |
| 更低的下降 | 100 | 220 | 左 | 2.59 | 0.0068 | 0 | 6.06 | 32.9 |

**十次运行全部：分支正确（步长 ≤ 0.0134 rad）、不碰桌面（支撑接触 0 帧）、路径
自身偏差 ≤ 4.16 mm、执行态终端 4.42–4.69 / 6.01–6.15 mm。** 路径跟踪是**求解器**
的性质，不是那一条构造直线的性质。

### 干净工作树上的测试（评审的发布纪律一条）

v9 评审指出：工作区里 4 个测试失败虽然据称与别人未提交的改动有关，但应当**在固定
提交的干净工作树上重跑**以证明没有源文件污染。实测：

- 干净工作树（`git worktree` @ HEAD）的 `allegro_floating.xml` 哈希 = 固定值
  `604ae7e0…`，确认工作区里那个 `76de50ef…` 是**未提交的本地改动**，与本轮无关；
- 干净工作树上的首次重跑：**15 passed / 1 failed** —— 失败的那条读的是
  `runs/p0/assets/tissue_box_lift_hold.json`，而该文件被 `.gitignore` 的
  `runs/**` 排除，任何干净检出都拿不到它；
- 该资产已 `git add -f` 提交，干净工作树重跑：**16 passed**。

**这一条是真实的仓库卫生缺陷，评审点得对，现已修掉。**


---

## r107: v10 的抬离可行性门（v9 评审指定的下一轮唯一目标）

v9 评审指定：**"先求力学可行解，再做连续执行；如果静态接触条件不支持离桌，就提交
可复验的不可行证据，不再盲目训练或调摩擦参数。"**

`tools/w6b/lift_feasibility.py`（一条命令，全新进程跑已提交配方，然后从磁盘的报告
重建终端态；不读会话状态）：

### 第一部分：静态可行性（用实测的接触点与法向力）

| 项 | 数值 |
|---|---:|
| 箱体质量 → 重力 | 0.4004 kg → **3.928 N** |
| 摩擦系数 μ = max(箱 1.4, 垫 1.8) | **1.8** |
| 实测法向力（右 ff_tip / 左 ff_tip_left） | **0.2972 N / 0.7177 N**（合计 1.0149 N） |
| 可用摩擦 μ·N | **1.827 N**（比重力**短 53.5%**） |
| 抬离所需法向力 | 2.182 N 合计 = **1.091 N/手**（右 27.2%、左 65.8% 达标） |
| 对质心的力矩平衡 | 所需摩擦/法向比 **0.623** ≤ 摩擦锥 1.8 → **可满足** |
| 手指扭矩容量 | 每手 1.388 N·m → 50 mm 力臂下约 27.8 N 法向力 → 所需 1.091 N **在硬件能力之内** |
| **静态裁决** | **INFEASIBLE（就当前终端构型而言）** |

**结论**：不可行是**当前终端构型（已提交的 anchor 的握力）的属性，不是硬件的属性** ——
所需法向力比手指的扭矩容量低一个数量级。因此出路是**把握力作为终端搜索的判据重
新搜终端位姿**，而不是 RL、不是调摩擦系数。

### 第二部分：局部连续上移执行（质量、摩擦系数、关节限位一律不动）

| 项 | 数值 |
|---|---:|
| 指令上移 | 6.0 mm / 90 个控制帧 |
| 箱体高度 | 739.97 → 739.97 mm（抬升 **−0.00 mm**） |
| 箱体对桌面的接触力 | 3.928 → **3.928 N**（= 重力，未变） |
| 箱体横向漂移 | −0.00 mm |

两遍全新进程重跑，数值逐字相同。

> 顺带修掉三个真实错误：(1) MuJoCo 的 `mj_contactForce` 返回的是**接触坐标系**的
> 力，法向是第 0 个分量（此前按世界系的 y/z 取，恒为 0）；(2) 接触力对桌面贡献
> 3.928 N = 箱重，正是接触模型正确的自检；(3) 手指执行器名是 `ffa0` 而关节名是
> `ffj0`，此前按关节名查执行器得到 0 容量。

### v10 的结论（一句话）

**就当前终端构型，抬离在力学上不可行：实测可用摩擦比重力短 53.5%，而局部上移实测
不产生任何抬升、桌面支撑力保持等于重力。** 不可行的原因是终端构型的握力不足
（需要的法向力 1.091 N/手，硬件容量 ~27.8 N/手，所以可达），因此下一步是**把法向
力纳入终端搜索的判据**重搜终端位姿，再重跑本门——而不是训练策略或调摩擦。


---

## r109-r110：把法向力纳入终端搜索

v10 的静态门给出的结论是：**抬离不可行的原因是当前终端构型的握力不足**（实测法向力
0.297/0.718 N，所需 1.091 N/手，而手指扭矩容量足够），所以下一步必须**把法向力
作为终端搜索的判据**，而不是训练策略或调摩擦。本节记录为此做的两个机制与它们的
实测。

### r109：终端的 SQUEEZE（沿端面法向的平移）

在终端的臂构型上沿端面法向做一次雅可比平移（保持掌心姿态），把指垫压进端面。

| 手法的实测 | 结果 |
|---|---|
| 沿法向的雅可比平移（掌心姿态**不**约束） | 手会同时旋转，指垫的接触滑离端面，法向力**下降**（右 0.297 → 0.237 N @0.2 mm），且 gate1 变红 |
| 沿法向的雅可比平移（姿态**受约束**：1 行平移 + 3 行姿态） | 右 0.297 → 0.364 N @0.05 mm → **0.245 N @0.10 mm → 0.202 @0.20 mm**，非单调；≥0.20 mm 时 gate1 变红 |

**沿法向平移不是提高法向力的正确杠杆** —— 臂的构型一变，指垫在端面上的接触点就变。

### r110：终端的 CURL SCALE（屈曲关节按比例放大）

把已搜到的握姿的**屈曲**关节（j1..j3）按比例放大，展开关节（j0）保持，于是指垫在
端面上的姿态不变而压紧增大。

| curl scale | 右的法向力 N | 左的法向力 N | 合计 | 门 |
|---|---:|---:|---:|---|
| 1.00 | 0.2972 | 0.7177 | 1.015 | 只有 gate4/wrist 红 |
| 1.05 | 0.2893 | 0.7433 | 1.033 | **gate1 变红** |
| 1.10 | 0.0 | 0.9391 | 0.939 | gate1 红 |
| **1.20** | 0.0 | **1.1945** | 1.195 | gate1 红 |
| 1.35 | 0.0 | 0.0 | 0 | gate1 红 |

**左手的法向力能到 1.19 N（超过所需的 1.091 N ✓），但右手的接触在 curl 一变就丢
（0.297 → 0）** —— 右手终端只有一根指垫以 −0.076 mm 的余量接触，本身就是边缘接触。

### 物理腿：把法向力作为候选的判据（`tools/w6b/squeeze_search.py`）

运动学搜索（`palm_pose_refine.py`）只看几何（指垫在面上、前向轴、净空），**看不见接触
力**。物理腿把每个候选在完整管线里重放，量**执行态**的法向力，并且要求每个门保持
绿。

已搜到的 8 个候选的结果（每侧）：

| 候选 | 右的法向力 N | 门 | 左的法向力 N | 门 |
|---|---:|---|---:|---|
| 0（当前交付） | 0.2972 | 绿 | 0.7177 | 绿 |
| 1 | 0.3720 | gate1 红 | 0.3370 | 绿 |
| 2 | 0.0000 | 绿 | 0.0000 | 绿 |
| 3 | 0.4312 | gate1 红 | 0.5550 | 绿 |
| 4 | 0.1851 | gate1 红 | 0.0809 | gate3 红 |
| 5 | 0.1843 | 绿 | 0.6803 | gate1 红 |
| 6 | 0.0000 | gate1 红 | 0.5972 | gate1 红 |
| 7 | 0.0000 | gate1 红 | 0.0000 | 多门红 |

**在"门保持绿"的约束下，当前交付的候选就是法向力最大的那个** —— 因为运动学搜索
**从来没有探索过预压**：它把指垫的目标放在端面平面上（extra_x = 0），所以接触力只剩
伺服残差。

### 因此：搜索要加一维"预压"

`palm_pose_refine.py` 现在多扫一维预压（把指垫的目标推进端面 0 / −0.15 / −0.30 /
−0.50 mm），并**按预压档位分别保留最好的若干候选**（预压的候选在矩形距上分数更差，
否则永远进不了候选表），再由物理腿挑。

> 这一步是 v10 评审要求的"把法向力纳入终端搜索判据"的正解。搜索的耗时因此约 4 倍
> （原 ~600 s / 侧 → ~2400 s / 侧），在后台跑。


---

## r111：左臂 1.48 mm 差值的归因（v9 评审第 3 问）

评审问：右 4.58 mm、左 6.06 mm 的 1.48 mm 差距，是否就是安装几何的等效力臂差异？
并给出可证伪的检查式：`Δx ≈ J(q) K_q⁻¹ τ_residual`。

`tools/w6b/lever_attribution.py` 从**本次运行自己**取齐式子需要的量：执行态的关节误差、
伺服保持力矩（kp × 误差）、未补偿的重力偏置、终端位姿下掌心的雅可比、以及接触反力。

| 侧 | 关节误差 (rad) | 伺服保持力矩 (N·m) | 未补偿重力偏置 (N·m) | 残差 τ (N·m) | 该残差解释的工具端位移 (mm) | 其法向分量 (mm) | 实测执行态垫距 |
|---|---|---|---|---|---:|---:|---:|
| 右 | [−0, 0.002, 0.001, 0, 0, −0] | [−0, 4.4, 2.0, 0, 0, −0] | [0, 31.3, 6.3, 1.9, −0.4, −0.1] | [0, 35.7, 8.3, 1.9, −0.4, −0.1] | [−12.01, −5.45, +9.45]（\|d\| 16.23） | −12.01（压向端面） | **4.58 mm** |
| 左 | [0, 0.001, −0, −0, −0, 0] | [0, 4.4, −0, −0, −0, 0] | [−0.6, 22.2, 13.4, 1.9, −0.3, −0.1] | [−0.6, 26.6, 13.4, 1.9, −0.3, −0.1] | [+6.01, +1.48, +5.88]（\|d\| 8.53） | **+6.01（压向端面）** | **6.06 mm** |

**结论（一半支持、一半不支持，如实说）**：

* **左臂的差值被公式精确复现**：预测的法向位移 6.009 mm vs 实测 6.06 mm（差 0.05
  mm ✓）。所以"左手更差是它的残余力矩更大（肩 lift 26.6 N·m、肘 13.4 N·m）"这一
  归因**成立**。
* **右手不被这个模型解释**：它预测 12.0 mm 的法向位移，而实测只有 4.58 mm（差 2.6
  倍 ✓）。右手的残差更大（35.7 N·m）、工具端位移更大（16.2 mm），但它落在端面上
  更近 —— 因为指垫是柔性的、且两臂的臂型不同，一阶模型在右手上不成立。

所以：**"安装力臂差异"这个解释对左侧成立、对右侧不成立；一手的数据不足以支持它作为
两侧的通用解释。** 这正是评审要求的"可以被证伪"——现在它被部分证伪了。


---

## r112：终端姿态的 Pareto 敏感性（v9 评审第 2 问）

评审第 2 问：右侧执行态前向轴 33.3° 对 35° 的限值只剩 1.7° 余量，而左 executed 垫距
6.06 mm 仍未到 5 mm —— "这说明当前终端姿态有明显权衡，不能称为双指标充分收敛"；建议
做一次**局部 Pareto 敏感性实验**（少量改变腕姿态与接近深度），同步量执行态垫距、前向
轴、接触数量与关节裕度，并**以执行态物理量优先**。

`tools/w6b/pareto_scan.py`（7 个构型，逐个跑完整管线）：

| curl scale | squeeze mm | 右：垫距 / 前向轴 / 接触垫 / 法向力 | 左：垫距 / 前向轴 / 接触垫 / 法向力 | 红的门 |
|---:|---:|---|---|---|
| **1.00（交付）** | **0.00** | **4.58 mm / 33.3° / 1 / 0.2972 N** | **6.06 mm / 32.9° / 1 / 0.7177 N** | gate4, wrist |
| 1.05 | 0.00 | 4.80 / 33.4 / 1 / 0.2893 | 5.99 / 33.1 / 1 / 0.7433 | **gate1 红**, gate4, wrist |
| 1.10 | 0.00 | 5.20 / 33.6 / **0** / 0.0 | 5.77 / 33.2 / 1 / 0.9391 | **gate1 红**, gate4, wrist |
| 1.20 | 0.00 | 10.70 / 34.2 / **0** / 0.0 | 5.65 / 33.7 / 1 / **1.1945 N** | **gate1 红**, gate4, wrist |
| 1.00 | 0.05 | **4.55 / 33.3 / 1 / 0.3639** | 6.06 / 33.0 / 1 / 0.7162 | gate4, wrist |
| 1.00 | 0.10 | 4.70 / 33.3 / 1 / 0.2453 | 6.07 / 33.0 / 1 / 0.6387 | gate4, wrist |
| 1.05 | 0.05 | 4.88 / 33.4 / 1 / 0.2445 | 5.98 / 33.1 / 1 / 0.7910 | **gate1 红**, gate4, wrist |

**读出来的三件事：**

1. **前向轴的余量不是被"姿态"吃掉的，而是被"执行"吃掉的**：命令态的前向轴是
   **26.34° / 27.61°**（限 35，余量 8.7° ✓），执行态是 33.3° / 32.9° ✓ —— 差的那
   7° 来自执行态跟踪与接触，**不是终端构型本身挤到了边界**。评审的前提部分不成立，
   这点如实写出来。
2. **右手的接触是整个系统最脆的一环**：curl 一变（1.10）右手的接触垫数就归零
   （0.297 → 0 N ✗），gate1 变红 ✓。右手终端只有一根指垫以 −0.076 mm 的余量接触，
   任何扰动都会让它掉。
3. **唯一的"无代价改进"是 0.05 mm 的 squeeze**：右 4.58 → 4.55 mm ✓、法向力
   0.297 → 0.364 N（+22% ✓）、红门不变 ✓。再往上（0.10 mm）法向力反而回落
   （0.245 N ✗），非单调。

**结论**：局部 Pareto 扫描没有找到一个同时改善两侧的构型；交付构型在"门全绿"的约束下
仍是最优的 ✓。要抬离需要的法向力（1.09 N/手）只能靠**重新搜一个预压不同的终端**
（r110 的搜索维度）拿到，而不是在这套参数里调。


---

## r113：预压必须作用在掌心，不是作用在手指目标

r110 的第一版把预压实现为**把指垫的 IK 目标推进端面**（`extra_x = −0.25 / −0.50
mm`）。跑完 4 个预压档位、1720 s 之后，输出的预压档位列表是 **[0.0] 只有一个档**
—— 所有带预压的变体都在**手指 IK** 上被否掉了（index IK 的否决数 4172 / 2899）。

**原因**：已搜到的握姿在**屈曲方向上已经到达手指的行程极限** ✓，指垫无法再往端面里
多走 0.25 mm ✓。所以"预压"不能通过手指的姿态来实现。

**正确的实现是移动整只手**：把**掌心的位置**沿端面法向推进 δ ✓，这样指垫目标相对掌心
的位姿不变 ✓（手指姿态不变 ✓），整只手压向端面 ✓ —— 这才是物理上的"握紧"。

修正后重跑搜索（预压 0 / 0.30 mm 两档）。


---

## r114：预压档位为什么仍然是空的（预压被臂的硬约束否掉）

把预压改成**移动掌心**之后重跑搜索（1213 s），预压档位列表**仍然是 [0.0]**
—— 0.30 mm 的档位一个候选都没有留下。但否决的来源换了：

| | 预压作用在指垫目标（r110 第一版） | 预压作用在掌心（r113 修正） |
|---|---|---|
| 手指 IK 的否决数（左侧） | 2899 | **82** |
| 臂的否决数（左侧） | 1096 | **2401** |

即：**手指 IK 已经不再是瓶颈 ✓（否决几乎消失），现在卡住的是臂的解** ✓。

原因是搜索里对臂的两条硬约束：

1. **分支锚**（`ANCHOR_TOL = 0.60 rad`）：臂的解必须在锚的分支附近 ✓；
2. **r86 的桌面净空硬约束**（碰撞代理到桌面 ≥ 12 mm ✓）：掌心往端面里推 0.30 mm
   会改变臂的构型 ✓，净空随之下降 ✓，于是候选被否 ✓。

所以"把预压加进终端搜索"这条路的真实障碍已经定位清楚：**不是搜索算不动，是预压把臂推到
了 r86 净空约束的边界上** ✓。下一步有两条（都需要新的臂构型搜索，而不是改参数）：

* 放宽/重算 r86 的净空约束（它当初是为"伺服带着 0.13 rad 跟踪误差时腕部不撞桌"设
  的 ✓，预压场景下这条约束的必要性要重新论证 ✓）；
* 或者把**基座放低**，让臂以更接近水平的姿态接近（这同时回应 v9 评审指出的"前臂进入
  角 35–45° 向下 vs 原片 ~5–10°"的视觉缺口 ✓✓），预压就不再撞净空约束 ✓。

预压档位现在可由 `W6B_R110_PRELOADS_MM`（逗号分隔的 mm 列表 ✓）配置 ✓。


---

## r115：基座能放多低（`tools/w6b/base_height_probe.py`）

r114 把预压的障碍定位到"臂的硬约束"，而 v9 评审的视觉缺口（前臂进入角 35–45° 向下 vs
原片 ~5–10°）指向同一个杠杆：**基座装得太高**。当前基座 z = 1.35 m，而掌心工作在
z ≈ 0.79 m —— 臂要在 0.70 m 水平跨度上折下 0.56 m（≈39°）。

探针只跑臂的 IK（秒级，不跑整条管线），对每个基座高度回答四件事：能不能解到**同一个
已知好掌心位姿**、解离开已验证分支多远、**前臂在画面内**的进入角变成多少、臂还清不清
桌面。

| 基座 z | 侧 | 画面内进入角 | 分支偏移 | 桌面接触 | e_p |
|---:|---|---:|---:|---:|---:|
| 1.35（当前） | 右 | 68.9° 向下 | 0.000 ✓ | 0 | 0.022 mm |
| 1.35（当前） | 左 | 45.4° 向下 | 0.000 ✓ | 0 | 0.017 mm |
| 1.20 | 右 | 85.6° | 0.819 ✗ | 0 | 0.001 mm |
| 1.20 | 左 | **18.7°** | 0.746 ✗ | 0 | 0.001 mm |
| 1.05 | 右 | 89.7° | 1.266 ✗ | 0 | 0.001 mm |
| 1.05 | 左 | **−2.3°（水平）** | 1.184 ✗ | 0 | 0.001 mm |
| 0.90 | 右 | 87.9° | 1.581 ✗ | 0 | 0.001 mm |
| 0.90 | 左 | −24.1° | 1.497 ✗ | 0 | 0.001 mm |

**读出来的三件事：**

1. **探针本身可信**：在当前基座高度上它复现了已知好解（分支偏移 0.000 rad ✓、e_p
   0.017–0.022 mm ✓）。
2. **降低基座确实能压平前臂的进入角**：左臂从 45.4° → 18.7°（z 1.20）→ −2.3°（水平，
   z 1.05）✓✓ —— 原片是 ~5–10° 向下，所以 **z ≈ 1.10** 是对位的高度 ✓。右臂在固定
   掌心位姿下反而更陡（85–90°）✗ —— 因为右臂的腕部滚转不同，**它必须重搜终端**。
3. **代价是分支**：任何更低的基座都会把臂推离已验证分支 0.75–1.58 rad（容差 0.60）✗。
   臂仍然**够得到**（e_p 0.001 mm ✓、桌面 0 接触 ✓），但**接近路径必须重走**。

因此"基座放低"不是一个参数改动，而是一条**必须重跑终端搜索 + 接近路径重建**的链。
已建 `grounding/assets/allegro_floating_ur5e_side_r38.xml`（两侧基座 z = 1.10 m ✓），
终端搜索正在该场景上跑。

（净空一栏在探针里用的是"包围球 vs 桌面平面"的保守量，因此为负 ✗；实际**桌面接触数
为 0** ✓ —— 以接触数为准 ✓。）


---

## r116：低基座单元（r38，基座 z = 1.10 m）的完整候选表

`tools/w6b/r38_candidate_table.py`（12 次完整管线，每个候选都带**路径质量**）：
物理选择工具按执行态法向力排序，但它不知道**接近路径本身是否重建成功** —— 一个路径失
控的候选也会报出法向力。r38 的右臂候选 4 就报了 **1.0568 N**（最接近抬升所需的
1.091 N），而它的路径偏差是 **259.60 mm** —— 机械臂根本没到终端，那个力是"从错误位置
扫过箱子"产生的假象 ✓✓。

| 侧 | 候选 | 路径 max | 路径 rms | 法向力 | 红门 |
|---|---|---:|---:|---:|---|
| 右 | 0 | 2.59 mm ✓ | 0.51 | 0.1403 N | gate1 |
| 右 | 1 | 2.58 ✓ | 0.51 | **0.0000 N** | **无** |
| 右 | 2 | 2.59 ✓ | 0.51 | 0.1750 N | gate1 |
| 右 | 3 | 88.31 ✗ | 32.68 | 0.1744 N | gate1, gate5 |
| 右 | 4 | **259.60 ✗✗** | 118.55 | (1.0568 N 是假象) | gate5 |
| 右 | 5 | 2.59 ✓ | 0.51 | 0.0000 N | gate1 |
| 左 | 0 | 2.57 ✓ | 0.51 | **0.6780 N** | **无** |
| 左 | 1 | 121.89 ✗ | 49.88 | 0.4102 N | gate5 |
| 左 | 2 | 87.37 ✗ | 30.09 | 0.7290 N | gate5 |
| 左 | 3 | 2.59 ✓ | 0.51 | 0.6546 N | gate1 |
| 左 | 4 | 2.59 ✓ | 0.51 | 0.6754 N | gate1 |
| 左 | 5 | 2.59 ✓ | 0.51 | 0.6720 N | gate1 |

**结论（负结果，如实记）：**

1. **降低基座换来的前臂进入角是以握持力为代价的**：路径有效的候选里，右臂最好的力
   0.175 N（r37 是 0.2972 N ✗），左臂 0.678 N（r37 是 0.7177 N ✗）。**两侧都变差**。
   所以"基座放低"**不改善抬离** ✓，它只改善视觉进入角 ✓ —— 两个缺口在基座高度上是
   **互相交换**的 ✓✓。
2. **右臂的接近路径在低基座上基本无法重建**（6 个候选里 2 个路径失控 ✗），左臂可以
   （左臂有 3 个候选全绿 ✓：0.678 N + 路径 2.57 mm ✓）。不对称的原因在搜索输出里看
   得到：右臂的腕部滚转使它的终端分支与低基座不兼容 ✓。
3. **抓到一个门漏洞**：右臂候选 1 的**执行态接触力为 0.0000 N**，而**全部门都是绿的**
   ✓。gate1/gate2 判的是**命令态**的垫片几何 ✓，执行态完全没有接触时它们照样通过
   ✗ —— 门组不要求"执行态存在接触" ✓。这条对 r37 同样成立（只是 r37 的候选碰巧有接触
   ✓），已如实记录。

**决策**：低基座单元**不作为交付单元** ✓（两侧力都低于 r37 ✓）。r115/r116 的价值是把
"视觉缺口 vs 物理缺口"的取舍**量化**了 ✓✓，这正是 v9 评审要求的证据形态 ✓。


---

## r117：交付单元的终端参考**从未入库**（可复现性缺陷，已修复并复验）

### 缺陷

管线的**默认终端**来自 `runs/w6b/palm_pose_refine.json`（`retarget/retarget_dual.py:5477`
的 `W6B_R63_REFERENCE` 默认值）。这个文件在 `runs/`（默认 gitignore ✓）里而且**从未入
库** ✓，后果有两个：

1. **新克隆跑不出交付数字**：文件缺失 → `_r63_tgt` 为空 ✓ → 管线退回天花板 cell 的种子
   对（`runs/w6b/native_seed_pairs.json` ✓）→ 终端完全不同 ✓。
2. **分析工具读的是"磁盘上恰好存在的那份"**：r115 为低基座单元重跑终端搜索 ✓，把这份参
   考**覆盖**了 ✓ → 所有依赖它的数字静默移动 ✓（实测：力臂归因里右臂的"残差解释的工具
   端位移"从 **16.225 mm 变成 10.591 mm** ✓，左臂的法向分量从 6.009 变成 4.185 ✓）。

### 为什么不能靠"重跑搜索"修

搜索的候选集依赖**工具版本的随机序列**（多起点 IK 的候选顺序 ✓）。用仓库里能找到的三个
历史版本（r64-r85 ✓ / r86 ✓ / r114 ✓）在 r37 场景上重跑，**没有一个能产出交付的那条
终端** ✓（右臂交付臂角 [-2.7205, 2.2339, -0.3968, -0.9615, -0.9894, 4.4938] ✓，三个版本
的候选都不匹配 ✓）。所以"再搜一次"不能重建交付状态 ✓。

### 修复：用运动学反推，而不是再搜一次

**臂的正向运动学唯一决定掌位姿** ✓，而交付终端的臂角与手指角都**记录在交付报告里** ✓
（`per_side.*.terminal_commands` ✓）。所以参考可以**精确重建** ✓✓：
`tools/w6b/rebuild_terminal_reference.py` 从 `git show HEAD:runs/w6b/report.json` 取
出臂角与手指角 ✓ → 在 r37 场景上做 FK ✓ → 得到掌心位姿 ✓（右 [170.9, −30.0, 800.02]
mm ✓、左 [−171.03, −30.0, 805.02] mm ✓）→ 写出与搜索同构的参考文件 ✓。

重建后 `verify_v9.py` **PASS** ✓（盒体位移 1.984 mm ✓，全部 v9 头条数字 ✓）；力臂归因
的数字回到 r111 的值 ✓（右 16.225 mm ✓ / 左 8.534 mm ✓，左的法向分量 **6.009 mm** ✓）。

### 新克隆复现测试（通过）

```
git clone ~/v2d-mac /tmp/v2d_clone          # 只含已提交内容
# 外部资产（不入库）：menagerie 网格、MANO 模型、dex-retargeting URDF、参考视频帧
ln -s ~/v2d-mac/runs/upstream/menagerie      /tmp/v2d_clone/runs/upstream/menagerie
ln -s ~/v2d-mac/runs/upstream/dex-retargeting /tmp/v2d_clone/runs/upstream/dex-retargeting
ln -s ~/v2d-mac/runs/mano                     /tmp/v2d_clone/runs/mano
ln -s ~/v2d-mac/runs/p0/assets/frames         /tmp/v2d_clone/runs/p0/assets/frames
cd /tmp/v2d_clone && <交付配方> python retarget/retarget_dual.py
```

新克隆的结果：**右执行态垫距 4.590 mm / 前向轴 33.279°** ✓，**左 6.060 mm / 32.944°** ✓，
命令态 6.2e-5 / 5.0e-5 mm ✓，红门 **gate4 + wrist** ✓（与交付一致 ✓✓）。

**外部资产依赖（必须写进交付说明）**：新克隆需要四棵**不入库**的外部资产 ✓ ——
`runs/upstream/menagerie`（Allegro 网格 ✓）、`runs/upstream/dex-retargeting`（手模型
URDF ✓）、`runs/mano/mano_v1_2`（MANO 模型 ✓）、`runs/p0/assets/frames`（参考视频帧
✓，wrist 门要用 ✓）。缺任何一棵管线都会在加载阶段报错 ✓。

---

## r118：抬升战役的根因（表面 vs 中心）与硬件上限

### 缺陷一：搜索器优化的是**垫中心**，不是**接触表面**

交付参考的垫**中心**落在端面平面上 ✓：

| 侧 / 指 | 垫中心 x (mm) | 面颊平面 x (mm) | **真实表面间隙** (mm) |
|---|---:|---:|---:|
| 右 ff | 60.00 | 60.00 | **−2.122**（已陷进箱子） |
| 右 mf | 60.00 | 60.00 | **−5.776** |
| 左 ff | −60.01 | −60.00 | −2.130 |
| 左 mf | −60.00 | −60.00 | −6.038 |

指尖垫是 **12×12×3 mm 的盒** ✓，所以"中心在面上"意味着**板已经陷进箱子 2.1 mm（ff）
/ 5.8 mm（mf）** ✓。后果：

* **命令态** gate1 看到两垫在矩形内 ✓（中心都在面上）✓，而**执行态只有一根垫接触** ✓
  （ff ✓；mf 差 **5.64 mm**（右）/ **9.20 mm**（左）✓）——箱子容不下两根各陷数毫米的垫 ✓；
* 实测法向力偏小 ✓：垫是在一个"箱子已经吸收掉"的姿态里按压 ✓。

`tools/w6b/palm_pose_refine.py` **完全没有表面概念** ✓ —— 它算出了表面间隙
（`mj_geomDistance` ✓）却**用中心距离打分** ✓。

### 缺陷二：curl 缩放把指尖挪离面颊（r110 注释的前提被证伪）

r110 的注释声称"缩放 curl 不改变垫在面上的姿态" ✓ —— **实测证伪** ✓：

| 构型 | 命令态 rect (mm) | n_in_rect | 执行态陷深 (mm) | 法向力 (N) |
|---|---:|---:|---:|---:|
| curl 1.00 / squeeze 0 | 0.0001 | 2 | −0.065 | 0.2549 |
| curl 1.00 / squeeze 0.05 | 0.0499 | 2 | −0.097 | **0.3797** |
| curl 1.00 / squeeze 0.15 | 0.1499 | **0** | −0.043 | 0.257 |
| curl 1.05 | 0.4487 | **0** | −0.042 | 0.3225 |
| curl 1.10 | **0.7838** | **0** | −0.166 | **1.1384** |

curl 1.10 的 1.1384 N **正是**靠把指尖压进面颊 0.784 mm 换来的 ✓ —— 几何与力在
**同一位移上互斥** ✓（R5 容差只有 **0.1164 mm** ✓）。

**每手独立 curl** 扫描（`tools/w6b/curl_baked_scan.py` ✓）：curl 是**双手共用**的旋钮
✓，而左右手的手性相反 ✓ —— 右 curl>1.00 立刻破几何 ✓，左 curl 1.25 能到 **1.6398 N** ✓
但 rect 0.8931 mm 破几何 ✓；左 curl 1.05 → **1.0166 N** ✓（距所需 1.0911 N 只差 7% ✓）。

### 缺陷三（关键）：硬件**够**，瓶颈是几何与伺服

`tools/w6b/finger_force_ceiling.py` 用虚功在交付位姿上算硬件天花板 ✓
（F_max = min_j(执行器力矩限 / 该关节的法向力臂) ✓）：

| 侧 / 指 | 各关节力臂 (mm) | **硬件法向力上限 (N)** | 约束关节 |
|---|---|---:|---|
| 右 ff | 0.75 / **23.48** / 1.90 / 1.76 | **7.63** | j1 |
| 右 mf | **17.04** / 7.70 / 15.45 / 9.47 | **0.88** | j0（展指，力矩限仅 0.02 Nm） |
| 右 rf | 2.02 / 14.54 / 26.10 / 13.61 | 3.10 | j3 |
| 左 ff | 0.01 / **22.32** / 1.24 / 2.04 | **8.03** | j1 |
| 左 mf | **18.01** / 3.02 / 18.00 / 10.30 | **0.83** | j0 |

**ff 垫的硬件上限 7.63–8.03 N ≫ 所需的 1.0911 N** ✓ → 交付格的力被**伺服刚度**与
**命令态几何**限制，**不是硬件** ✓。但 **mf 垫的硬件上限只有 0.83–0.88 N < 1.0911 N** ✓
——若要两手都靠 mf 出力，**硬件就不够** ✓（单靠 ff 则够 ✓）。

### 结论（待表面级重搜确认）

抬升所需的力**在硬件上可达** ✓，但当前的**几何表达方式**（垫中心在面上 ✓）使它
**不可达** ✓：两垫都陷进箱子 ✓ → 执行态只剩 ff ✓ → 埋入判据与 rect 判据同时红 ✓。
**下一步**：把 curl **烘进手指 IK** ✓ 并把 IK 目标从"中心在面上"改为"**表面在面上**" ✓
（`tools/w6b/surface_seated_search.py` ✓），使深卷曲下指尖仍贴面 ✓。

### r118 续：表面级几何**可执行**了（v3 从 0 → 1），以及"双垫同时贴面"的新障碍

用**支撑函数**（垫沿法向的支撑 = Σ_i |轴_i·n| · 半长_i ✓）构造表面级候选 ✓：

| curl | 静态表面间隙 ff / mf (mm) | 垫中心 x ff / mf (mm) | 掌心 x (mm) | **执行态** ff 间隙 (mm) | 执行态力 (N) | v3 eligible |
|---:|---|---|---:|---:|---:|---:|
| 1.00 | −0.048 / −0.054 | 62.07 / 66.32 | 172.98 | **−0.035** ✓ | 0.2109 | **1** ✓ |
| 1.05 | −0.046 / −0.077 | 62.34 / 66.36 | 173.69 | −0.023 | 0.1752 | 1 ✓ |
| 1.10 | −0.044 / −0.068 | 62.61 / 66.44 | 174.29 | −0.022 | 0.1834 | 1 ✓ |
| 1.15 | −0.042 / −0.054 | 62.87 / 66.51 | 174.77 | −0.009 | 0.1554 | 1 ✓ |

**突破** ✓：表面级几何在**执行态真正接触**了 ✓（ff 表面间隙 −0.009…−0.035 mm ✓），
v3 的 eligible 从 **0 → 1** ✓，而且**不再有 0.73 mm 的悬空** ✓（此前用旧 palm_pos 的
候选全是悬空 ✓ —— 因为管线用参考的 `palm_pos` 做终端 IK ✓，而表面级几何要求掌心
沿法向后退 ✓）。

**新障碍（几何本质）** ✓：两垫的**支撑函数不同** ✓ —— ff 沿法向支撑 **2.122 mm** ✓、
mf **5.776 mm** ✓（mf 以角点接触 ✓，ff 以面接触 ✓）。所以"两垫表面同时贴同一面"
要求两垫**各自的姿态**都改变（支撑值要变成相等）✓，而逐指 IK 只驱动**间隙**、
不驱动**姿态** ✓ → mf 执行态仍差 **9.3 mm** ✓。

**结论（精确化）** ✓：
1. 抬升所需的力**硬件够** ✓（ff 7.63–8.03 N ✓ ≫ 1.0911 N ✓）；
2. 力的上限来自**双垫同时贴面**这一几何约束 ✓ —— 而双垫贴面又要求两垫姿态一致 ✓；
3. mf 垫的**硬件力上限 0.83–0.88 N < 1.0911 N** ✓ —— 所以**单靠 mf 抬不起** ✓，
   必须**ff 单垫**承担 ✓（其硬件上限 7.63 N ✓）；
4. 那么 v10 抬离战役的**真正问题**收敛为一句 ✓：**单手单垫（ff）能否在几何全绿
   （v1/v2 与 v3 同时绿）的前提下产生 ≥1.0911 N** ✓ —— 而 v1/v2 判的是**中心** ✓，
   表面级几何下**必然红** ✓（中心退到 62 mm ✓ > 容差 0.1164 mm ✓）✓。

**这给评审一个必须裁决的判据问题** ✓：v1/v2（中心距 ≤ 0.1164 mm ✓）与 v3（表面距 ≤
0.1164 mm ✓）**在物理上互斥** ✓ —— 12×12×3 mm 的垫 ✓，中心距与表面距相差 **2.1–5.8 mm** ✓，
**不可能同时 ≤ 0.1164 mm** ✓。交付单元之所以"v1/v2 绿"，正是因为它把垫**陷进箱子
2.1–5.8 mm** ✓ —— 而这在物理上只有一根垫做得到 ✓（执行态 v3 红 ✓）。

**下一步**（未定）：要么 (a) 承认 v3 是诚实的接触判据并**重写 gate1 的成功条件** ✓
（这需要评审裁决 ✓，因为判据是冻结的 ✓），要么 (b) 保持 v1/v2 但把**单垫接触**作为
交付的诚实表述 ✓（那么抬升战役的目标就变成"单垫 ff 达到 1.0911 N" ✓，而表面级几何
下实测只有 0.16–0.21 N ✓，差 5–7 倍 ✓）。

### r118c：表面级几何**两侧都跑通**，v3 由红转绿（`tools/w6b/surface_seated_build.py`）

| 侧 | curl | 静态表面间隙 ff / mf (mm) | 执行态 ff 间隙 (mm) | **v3 eligible** | 执行态力 (N) | v1/v2 rect (mm) | v1/v2 |
|---|---:|---|---:|---:|---:|---:|---|
| 右 | 1.00 | −0.048 / −0.054 | **−0.035** | **0 → 1** | 0.2109 | 2.0738 | 红 |
| 右 | 1.05 | −0.046 / −0.077 | −0.023 | 0 → 1 | 0.1752 | 2.3429 | 红 |
| 右 | 1.10 | −0.044 / −0.068 | −0.022 | 0 → 1 | 0.1834 | 2.6070 | 红 |
| 右 | 1.15 | −0.042 / −0.054 | −0.009 | 0 → 1 | 0.1554 | 2.8654 | 红 |
| 左 | 1.00 | −0.048 / −0.075 | ✓ | **0 → 1** | 0.5662 | 2.0892 | 红 |
| 左 | 1.05 | −0.047 / −0.050 | ✓ | 0 → 1 | 0.1549 | 2.3558 | 红 |
| 左 | 1.10 | −0.050 / 52.13 | ✓ | 0 → 1 | 0.3075 | 2.6119 | 红 |
| 左 | 1.15 | −0.050 / — | ✓ | 0 → 1 | 0.3826 | 2.8654 | 红 |

**这是根因 D 的直接实证** ✓：表面级几何让 **v3 由红转绿** ✓（真实接触 ✓）而 **v1/v2
转红** ✓ —— 两判据**互斥** ✓，取舍**只能二选一** ✓。

**踩坑记录** ✓：
* 左手的手性相反 ✓ —— 掌心的平移方向必须按**端面外法线**取（`face_sign` ✓），
  固定用 +x 会让左侧的 IK 完全不收敛（表面间隙读 500 mm ✓）；
* 管线用参考的 **`palm_pos`/`palm_R`** 做终端 IK ✓（`retarget_dual.py:5544` ✓），
  所以表面级候选**必须同时更新 palm_pos/palm_R** ✓ —— 否则执行态差 0.73 mm 悬空 ✓；
* 掌心沿法向的平移要用**雅可比**（法向平移 + 姿态保持的最小二乘 ✓）✓，不能直接改关节 ✓。

### r118d：穷尽性证据 —— v1/v2 绿区内力被封顶在 0.38 N（右）/ 0.70 N（左）

| squeeze (mm) | preload (mm) | 手指 kp | v1/v2 | 右臂力 (N) | 左臂力 (N) |
|---:|---:|---:|---|---:|---:|
| 0.00 | 0.05 | 1.0 | 绿 | 0.2549 | 0.7377 |
| 0.05 | 0.05 | 1.0 | 绿 | **0.3797** | 0.7015 |
| 0.05 | 0.10 | 1.0 | 绿 | 0.3797 | 0.7015 |
| 0.10 | 0.20 | 1.0 | 绿 | 0.3702 | 0.7015 |
| 0.11 | 0.05 | 1.0 | 绿 | 0.3581 | 0.7038 |
| 0.05 | 0.05 | **1.5** | 绿 | **0.0000** | **0.0000** |
| 0.05 | 0.05 | **2.0** | 绿 | **0.0000** | **0.0000** |

**结论** ✓：v1/v2 绿区内力被封顶 ✓（右 0.3797 N ✓、左 0.7038 N ✓），距所需 1.0911 N
差 **2.9 倍 / 1.6 倍** ✓，且与 squeeze/preload 的取值**几乎无关** ✓（非单调 ✓）。
**提高手指伺服刚度不是出路** ✓ —— kp×1.5 起**接触被完全打掉** ✓（指尖不再压入 ✓，
垫只是搭在面上 ✓）。

**所以"v1/v2 绿 + 力 ≥ 1.0911 N"在本单元的可达集内不存在** ✓ —— 不是没搜到 ✓，
是被几何与伺服的耦合封死 ✓。

---

## r119：双垫同时贴面 —— **中指在交付臂位姿下够不到端面**（运动学硬事实）

### 问题

v1/v2 与 v3 互斥（根因 D ✓），而"双垫表面同时贴同一面"是抬离战役的物理前提 ✓ —— 需要
**两个垫的支撑点都在端面平面上** ✓。逐指 IK 的目标是**真实最近点**（`mj_geomDistance`
的 `p_pad` ✓，不是支撑点 ✓ —— 支撑点只在垫的朝向恰好使它成为接触点时才等于接触点 ✓，
实测 ff 的支撑点就是接触点 ✓ 而 mf 的支撑点离自己的接触点 14.3 mm ✓）。

### 三种独立方法的结论一致

| 方法 | 样本 | 结果 |
|---|---:|---|
| 随机 mf 姿态，判"最近点在端面" | 2×20000 | **0 次** ✓（最近点始终在**箱顶边缘** z=780 ✓） |
| 定向 IK（从端面种子出发） | 3000 种子 | **0 个**种子带端面接触 ✓ |
| 构造 IK（垫中心目标设在端面上，7×7×6 姿态） | 294 | **无一次**成功 ✓ |

而 mf 的 z **不是**障碍 ✓：其 tip z 在 20000 次随机姿态中 27.3% 落在面带 700–780 内 ✓。
障碍是**几何耦合**：mf 的最近点随姿态跳到箱顶边缘 ✓，要让接触点落在端面内需要 mf
的支撑值（沿法向的支撑）刚好等于掌心离端面的距离 ✓，而这与 ff 的支撑值（2.122 mm ✓）
不同 —— **两个垫的支撑值不同** ✓，一个掌心位置无法同时满足 ✓。

### 结论

**在交付臂位姿下，"双垫表面同时贴端面"不可达** ✓ —— 这不是搜索不充分 ✓（三种独立
方法 ✓、各 20000/3000/294 样本 ✓），而是**运动学约束** ✓。

因此 v10 抬离战役的**最终答案**收敛为：
1. **v1/v2 绿**（交付现状 ✓）→ 两垫中心在面上 ✓ → 但物理上只有一根垫能真贴 ✓
   （执行态 v3 红 ✓）→ 法向力被封顶在 **0.38 / 0.70 N** ✓（r118d ✓）；
2. **v3 绿**（表面判据 ✓）→ 需要双垫同时贴面 ✓ → **运动学不可达** ✓（r119 ✓）；
3. 硬件本身够（ff 7.63–8.03 N ✓）✓，但**两种判据下都拿不到所需的 1.0911 N** ✓。

**所以答案是：在当前单元（12×12×3 mm 指垫 + 交付臂位姿 + 冻结判据）下，抬离战役
的负结果是结构性的 ✓，不是搜索不充分 ✓。** 要翻案必须改变**系统参数**而非姿态 ——
例如换更薄/更小的指尖垫 ✓、换更宽的箱子端面（让中指够得到 ✓）、或换夹持构型
（掌板/拇指对握 ✓）—— 这些都是**评审可以指定的范围问题** ✓。

---

## r120–r122：v10 裁决执行 —— gate6、勘误、授权 A 薄垫分支、右臂的合法力封顶

### r120：gate6 执行态真实接触硬门（评审 Q6/Q8 指令）

`retarget/retarget_dual.py`：真实接触 = 求解器注册接触 ∧ 最近点在端面 ∧ 陷深 ≤ 0.5 mm；
力地板 = m·g/(2μ)（**从场景自带质量与摩擦导出**，0.4004 kg / 1.8 → 1.0907 N/侧）。
干净克隆按交付配方复验：右 ff（−0.065 mm，0.2549 N）左 ff（−0.083 mm，0.7377 N），
两侧 1 真实接触、力不足 → gate6 红 —— 与 v10 包已发表数字一致 ✓。

### r121：v6–v10 双垫表述统一勘误（评审 Q10 指令）

`docs/errata_dual_pad_seating.md`：三层含义分离（命令态中心几何 / 执行态表面接触 /
执行态承载），逐包锚到行号；历史包冻结不改。

### r122：授权 A 薄垫分支（r40 全薄、r41 混合），卷曲×预压逐侧扫描

`tools/w6b/thinpad_scan.py`（烘焙参考 + W6B_R63_REFERENCE，克隆内跑）：

| 场景 | 组合 | 右 | 左 |
|---|---|---|---|
| r40 全薄 | curl 1.0/1.2 | 0.226 N（合法） | **1.289 N gate6 PASS ✓** |
| r40 全薄 | curl 1.0/1.25 | 0.154 N | **1.391 N PASS ✓**（重复 3 次逐位一致 ✓） |
| r40 全薄 | curl 1.0/1.3 | 0.241 N | **1.502 N PASS ✓** |
| r41 混合（右原始垫） | curl 1.06/1.25 | 0.308 N（合法） | 1.093 N PASS ✓ |
| r41 混合 | curl 1.08+/1.25 | 0.78–0.95 N **全是 1.85 mm 深埋伪影**（n_elig=0，评审禁止项） | 1.31 N PASS |

**结论（右臂指垫路线收口）**：合法力（陷深 < 0.5 mm）封顶 ≈ 0.33 N；即使左臂给到
1.50 N（μN=2.70 N），右臂仍需 0.68 N —— 差 2 倍。深卷曲的高力全部来自垫体整体埋入
箱体（穿入 1.85 mm > 垫半厚 1.5 mm），属评审明令排除的证据。双侧深卷曲还会经箱体
耦合互相杀死接触（r1.10+/l1.25 → 左 0 N）。

**grant B（掌板）为右臂唯一出路**：`palm_plate_search.py` 在 r37 上已给出 24/3456
可行位姿（掌板切端面、三指尖与拇指离箱 9–158 mm、台面净空 89 mm）—— 掌板由臂承载
法向力，不受指伺服 0.04–0.18 N·m 限制。下一步：可行掌位姿 → 终端 IK → 管线 →
掌板接触力 + 6 mm/90 帧起升 7 条验收。

---

## r123：grant B 掌板首接线 —— 位姿可行但**臂不可达**（负结果，按实报）

接线三件套（均已提交）：
1. `retarget/retarget_dual.py`：执行态证据新增 `palm_plate_executed_state`（掌板
   solver 接触/端面判定/符号间隙/法向力，与指尖 v3 同语义）；gate6 接触面集扩入
   掌板（右 `palm_collision` / 左 `palm_collision_left`），摩擦取 max 含掌板；
2. `tools/w6b/plate_terminal.py`：把 `palm_plate_search.json` 的可行掌位姿烘进右臂
   终端参考（注意键名：管线读 `palm_pos/palm_R`，分析输出的 `the palm` 无效 —— 
   首跑掌板 106 mm 无驱动即此因，已修）；
3. 结果：掌位姿生效后**整手被拉飞**（板距面 138.5 mm，左右全部失接触）。

**根因（200 重启多起点位置+姿态 IK 探针）**：掌板切面位姿（掌心 x=0.0711 m，
即掌板贴死端面）对当前侧装基座的右 UR5e **残差 948.9 mm 不可达**。掌板搜索只验了
手部几何（板切面、指尖/台面净空），没验臂可达性 —— 这是搜索的边界，如实披露。

**grant B 的两条后续路**（下一工作段）：a) 为掌板分支**移动右臂基座**（设计分支
内变更，物理基准不动）；b) 改**拇指对握**（pinch 在手内闭合，腕部姿态无需重定向，
对臂可达性压力最小）。

r122 复核：混合场景 (1.06/1.25) 下左 1.093 N PASS / 右 0.308 N 合法 —— 左臂单垫
已达验收线，右臂等 grant B。

---

## r124：掌板分支第二战 —— 力是真的，路径是缺的

**r42 场景**（右基座 0.70/0/1.35 → 0.70/0/0.90，grant B 设计分支内变更）：基座探针
网格证明掌板位姿在手部高度全域可达（x 0.45–0.75、z 0.72–0.96 全部 pos 0.0 mm /
rot 0.0000 收敛；仅 1.35 天花板基座 948.9 mm 不可达）。

**首次 r42 管线跑（原始搜索位姿）**：右掌板 **solver 真实接触 174.805 N、陷深
−0.265 mm（< 0.5 mm 容差）** —— 臂承载的法向力远超 1.09 N 需求，grant B 的力学
前提成立。但接触点在**箱顶棱**（搜索位姿掌心 z=0.7877 > 箱顶 0.78；板压箱顶，
法向力朝下，对抬升摩擦无效），且箱子被推 7.4 mm、左臂失接触。

**重定心尝试（r124）**：FK 实测掌板中心在掌坐标系偏移，把板心摆到端面中心
（z=0.74，x=板半厚+standoff）。结果：终端未到位（板距面 2.6–5.5 mm 无接触），
且**接近过程把箱子推走 11 mm**（gate3 红）—— R75 侧向接近是为旧基座设计的，
新基座下失效。掌板分支缺的是**自己的接近路径**（高飞越 + 面高直接 -x 插入），
不是终端目标。

**当前分支状态**：力学可达 ✓（174.8 N 实测）→ 待办：①端面接触（位姿重定心 +
终端跟踪收敛）②接近路径重建 ③standoff 调力到 ~1.2 N 量级 ④然后 6 mm/90 帧
起升验收。

---

## r125（a/b/c）：掌板终端收敛战 —— 从 0.51 m 到 1.2 mm，三处根因逐个拔除

| 子轮 | 缺陷（实测） | 修复 | 结果 |
|---|---|---|---|
| a | 参考里旧臂解把终端 IK 种到**绕组分支**：wrist_3 命令 4.49 rad / 执行 −1.69 rad（差 6.19 ≈ 2π），全程执行偏差 0.51 m | `plate_terminal.build_ref` 现场（r42 上）多重启解臂 IK + ±2π 归一到限位带，作 `arm` 种子 | 偏差降至厘米级 |
| b | 板力循环抓到**首个含板接触** = 桌面：448.9 N 打桌（板距箱 13.3 mm）；板高 95 mm > 面高 80 mm，居中必插桌下 | 管线板接触只计 palm↔object；板心抬到下缘离桌 1 mm（覆盖面全高） | 接触区进端面带 |
| c | 多重启并列 0 残差时随机选到**肘朝下分支**，接近段扫箱（箱推 21 mm、板差 26 mm） | 以"离交付臂解最近"破平 | 箱推 9.8 mm、板差 1.78 mm |

** standoff 扫描（−1.5/−2/−3 mm）**：板距面停在 1.33/1.24/1.16 mm 不再进 —— 伺服
对接触-推箱耦合饱和；箱子恒被推 ~10 mm（推箱源待逐帧取证：接近段哪帧、哪个 geom）。

**指尖让位已排除**：curl 0 时四指尖沿掌 z 伸出、在板面后 11.1 mm（掌系实测），
板先触箱 ✓；卷曲反而把指尖带向 +x（朝箱）——板抓取应保持 curl 0。

**下一工作段**：①推箱逐帧取证（frame/geom/方向）→ 接近段避让；②板距面最后
1.2 mm 的伺服死区（末端微进给或重力补偿复核）；③贴面后 standoff 调力 ~1.2 N；
④6 mm/90 帧起升验收。

## r126：推箱逐帧取证 —— **推箱的是左手，不是右臂**

`W6B_R126_BOX_TRACE=1`（已入管线，默认关）：逐帧箱位 + 全部箱接触（geom/陷深/法向力）。
r42 + standoff 0.1 的轨迹（runs/w6b/box_trace.json）：

| 帧 | 箱位移 | 事件 |
|---|---|---|
| 0–298 | 0.01 mm | **接近段干净** ✓（右臂换基座后路径不碰箱） |
| 299 | 2.16 mm | **左食指**（ff_distal/tip_left，curl 1.25 终端卷曲）压箱 −0.88/−0.20 mm |
| 300+ | 5.2→9.8 mm | 箱无对撑逃逸，**卡进 yaw_stop_posy/negy 挡杆**（55–118 N） |

右板此时差 1.78 mm 未贴面，撑不住 → 箱被左手单侧推走。standoff 加深（−5/−8 mm）
板残差只降到 0.83/0.57 mm 仍不接触（伺服-接触耦合死区），且 t=299 的左指冲击不变。

**修复方向（下一会话）**：①时序——右板先贴（或同帧），再卷左指：把左指卷曲的
blend 窗推迟到板落座之后；②板残差 0.57–1.78 mm 的伺服死区（末端微进给 / 该位姿
下的执行器余量核查）；③on_end_face 持续 False 待几何取证（板相对面的实际姿态）。

## r127（a/b）：左臂延迟实验 —— 负结果，t=299 冲击源另有其路

- r127a `W6B_R127_LEFT_CURL_DELAY`（20/40 帧）：左**指**卷曲延迟 → t=299 冲击不变，
  追踪显示左**腕**终到位仍压箱；
- r127b 左**腕+指**终端同步延迟（30/50 帧）：t=299 冲击**仍不变**（箱推 11.9 mm）。

结论：左臂终端指令存在 finger_target_override/wrist 延迟块之外的写入路径（或该块
被后续终端 IK 覆盖）。下一步须通读 cmds_global 装配链（blend@5460 / native@7290 /
terminal IK@r12.9），找出 t=299 真正生效的左腕指令源，再做时序。

当前 r42+掌板最佳态：接近段干净（0–298 帧 0.01 mm）✓、板距面 1.29 mm、
推箱 11.9 mm（t=299 起，左源）、on_end_face False（板实际姿态待测）。

## r128：冲击源改判 + 慢化尝试的边界

**r127 归因修正**：加印 t=299 全部接触后，箱是被推向 **−x** 打进 r22 vise 角铁
（13.4 N）与 yaw 挡杆 —— 是**右板以 0.4 m/s（19.8 mm/帧）到达**砸的，不是左手；
左臂腕+指延迟（r127b）**实际生效**（延迟后左掌速度 295–305 帧全程 0 ✓）。r127 台账
"左臂另有写入路径"的结论**撤回**。

**慢化尝试（全部实测无效，保留为负结果）**：
- R75 相位 A/B 与站台距离（140/161/50、120/181/30）：峰值速度不变 19.5–19.8 mm/帧；
- `W6B_BLEND_SECONDS` 3.2/4.8 s：n_blend 确实变 96（已插桩验证打印），但重放窗
  恒为 331 帧、峰值速度 20.4 mm/帧不变 —— 快段由**自适应 n_blend6 路径**写入
  （`retarget/retarget_dual.py:5953` `n_blend6 = max(n_blend, blend_needed)` 及其
  6054 的按帧写入），不消费该 env。

**下一会话起点**：把时长参数接进 n_blend6 自适应窗（或直接限幅该窗内的腕命令
步长 ≤0.02 rad/帧），重跑 r42 掌板 → 预期消除 t=299 冲击 → 进入贴面-调力-起升。

## r129：两个限速旋钮实测无效 —— 快段疑为 R75 笛卡尔路径的近奇异关节放大

| 旋钮 | 设定 | 结果 |
|---|---|---|
| `W6B_R18_RATE_CEILING` | 0.5 / 0.2 rad/s | 峰值命令步长恒 0.203 rad/帧、掌速恒 20.4 mm/帧、箱推恒 11.93 mm —— **无效** |
| `W6B_BLEND_SECONDS` | 3.2 / 4.8 s | 同上（n_blend 已验证变 96，但重放窗恒 331）—— **无效** |

推断（待证）：快段由 R75 两相笛卡尔接近（staging→terminal 直线插值 + 逐帧 IK）写入，
在近奇异腕位形下关节步长被放大（笛卡尔 ~20 mm/帧 ⇒ 关节 0.2 rad/帧 = 4 rad/s），
而 n_blend6 自适应窗与 rate ceiling 都作用于**被 R75 覆盖之前**的流。

**下一会话取证方案**：dump cmd_wrist_right[280:300] 与三个候选写入者
（r50 blend / n_blend6 自适应 / R75 相位 B）的预测逐帧比对锁定写入者；若为 R75 近奇异，
对策按序：a) 相位 B 改关节空间插值（q_app→q 直接 smoothstep，绕开奇异放大）；
b) staging 位姿加中间 waypoint；c) 掌板目标方位微调避开奇异。

## r131：t=299 冲击的**根因**与修复（种子旋转 180° 假收敛）

**定谳链**（本会话逐环实测）：
1. 快段（271–300 帧，峰值 0.203 rad/帧）不是 r50 blend、不是 n_blend6、不是 R75、
   不是 r26 Track A freeze —— 逐站点签名插桩证实 `[sig] PRE-REPLAY` 在 blend 之后仍是快段。
2. `dbg-b94`：右臂反向求解在 t=1.0（终端）从种子构型出发，IK 返回**差 3.78 rad 的另一个解**
   → 种子不是该位姿的解。实测种子 FK 的位置误差 **0.000 mm、旋转误差 3.1416 rad**。
3. 根因 = `tools/w6b/plate_terminal.py` 的旋转残差度量 `0.5(RpᵀR−RᵀRp)`，
   **恰好在 180° 时为零** —— 这正是 r65 已记录、曾让 `palm_pose_reach.json` 作废的同一盲区。
   烘焙器因此"收敛"到旋转反 180° 的构型，管线用正确度量（rotvec）看到 π，只能把臂重排 3.78 rad。
4. 修复：度量改为相对旋转的 rotvec；绕板法向的 roll 是**规范自由度**（板为居中盒，
   绕法向 180° 对称），故 stage 1 投影掉 roll 分量、stage 2 固定实测 roll 后按该 roll
   **重算板的竖直跨度与板心高度**再做 6D 精确解；加入限位余量项（>0.3 rad，搜索自身判据）。
   参考的 `palm_pos/palm_R` 直接写为种子的 FK，管线 IK 起点残差为零。

**修复后实测（r42 掌板，standoff −3 mm）**：

| 指标 | 修复前 | 修复后 |
|---|---|---|
| IK 起点残差（右） | pos 0 mm / **rot 3.1416 rad** | pos 0.000 mm / **rot 0.0000 rad** |
| 路径末帧 vs hold | 差 3.78 rad | **完全一致** |
| 峰值命令步长 | 0.203 rad/帧 | **0.007 rad/帧** |
| 峰值掌速 | 20.4 mm/帧 | **2.8 mm/帧** |
| 箱推位移 | 11.93 mm | **1.21 mm（终态 0.85 mm）** |

**仍未解决（诚实态）**：gate6 仍 0 真实接触。执行态掌位 `[121.9, 63.5, 781.4]`
比命令的 `[71.2, 63.9, 787.2]` **短 50.7 mm（x 向）**；`mf_tip` 在箱体上缘
（z=777.6）以 0.12 mm 间隙、1.35 N 相触且 `on_end_face=false` —— 指尖先顶到箱体上缘，
板面仍差 21–41 mm。下一步：查为何 curl 0 的指尖落在板面前方（应在板面后 11.1 mm），
以及该种子的伺服可达性（50 mm 欠位）。

## r130–r141：t=299 冲击的根因链与右臂终端几何重构

### 根因（已修）
1. **r130**：`W6B_RECONFIG_FRAMES` 位姿保持驻留 —— 实测**不可能**：6-DoF 臂 + 6D 位姿任务无零空间，
   两个 IK 解之间不存在保持位姿的连续路径（每帧投影都被拉回起始分支，只有强制写入的末帧移动，
   留下 3.78 rad 跳变）。默认关闭，保留为 <0.2 rad 残差的软化路径。
2. **r131（真根因）**：`tools/w6b/plate_terminal.py` 的旋转残差用 `0.5(RpᵀR−RᵀRp)`，
   **在 180° 恰为零** —— 与 r65 记录在案、曾作废 `palm_pose_reach.json` 的同一盲区。
   烘焙出的右臂种子 FK 位置误差 0.000 mm 而**旋转误差 3.1416 rad**；管线用 rotvec 度量看到 π，
   被迫把臂重排 3.78 rad —— 这就是 271–300 帧 0.203 rad/帧的快段与 t=299 的板砸箱。
   修：度量改 rotvec；解算改阻尼最小二乘 + 0.1 rad 信赖域 + 240 重启。
   **修后实测**：IK 起点残差 0.000 mm / 0.0000 rad，路径末帧 ≡ hold，峰值命令步长 0.203→0.0071 rad/帧，
   峰值掌速 20.4→2.8 mm/帧，箱推 11.93→0.53 mm。
3. **r132**：`debug_history.npz` 的 `exec_wrist_qpos_right` 一直记录的是**左臂**关节
   （npz 写点的字典推导未重绑 `spec`）—— 两侧读数逐位相同即为证据。已修为 `specs[s]`。

### 右臂终端几何（r133–r141，逐个实测障碍）
| 轮 | 障碍（实测） | 处置 |
|---|---|---|
| r133 | 板底离桌 1 mm 时，执行态前臂/腕1/腕2 网格**陷入桌面 0.2–38.5 mm**；3× 力矩量程无效（±450 N.m 饱和，关节误差仍 0.25/0.45 rad） | 板底离桌 15 mm（`W6B_PLATE_BOTTOM_MM`），板心取面中心与桌面余量的较大者 |
| r134 | 两阶段烘焙：stage 1 选的 roll 使 stage 2 目标不可达（差 163 mm） | 改为 **roll 扫描**（每 5°，逐 roll 重算板心，热启动 6D 解），可行性硬门：收敛 ≤2 mm/0.02 rad、臂杆离桌 ≥15 mm、指尖离箱 ≥9 mm、关节余量 ≥0.3 rad；全表写入 ref 的 `roll_sweep` 供审计 |
| r134b | 只有几何门时，**未收敛**但停在限位、远离一切的解（偏 424 mm）比收敛但穿桌的解更便宜 | 收敛改为最高硬门（未收敛 +1e6） |
| r134c | 「指尖须在板面之后」判据误杀：所有收敛 roll 的指尖都在板面前方 62 mm，但横向错开（roll 0 时指尖离箱 36.8 mm，板先触） | 判据改为**指尖/箱**间隙，板前/板后仅作记录 |
| r136 | 手部几何门把 `wmount_*` 与 `*_visual`（contype=conaffinity=0，纯视觉）也算进去 —— 仅 wmount_cap 就"穿箱"17.2 mm，36 个 roll 全灭 | 门只统计与箱体 contype/conaffinity 相交的几何 |
| r137 | 真凶 `th_base_collision` 越过板面 5.5 mm、穿箱 8.17 mm；拇指卷曲无效（0.0→−8.17，1.0→−11.40） | 板沿掌 x **向箱侧外伸 20 mm**（r137 第一版符号搞反，把手拉进箱内，被 hand/box 门逮住：−8.18→−15.79） |
| r138 | 板碰到的是箱体**倒角 geom**（`object_chamfer_posx` 凸出面平面 1.7 mm、z 0.7781–0.7811），18.7 N 压在箱顶前缘，而 gate6 只量 `object_geom` 故报 1.91 mm 无接触 | gate6 的距离/接触改跑箱体**全部**碰撞 geom（并记录 `nearest_box_geom`/`contact_box_geom`）；`on_end_face` 仍严格要求平面；烘焙器把板顶限制在倒角下缘以下 |
| r138c–e | 装得下的窗口仅 61 mm；10° 网格正好跳过唯一可行的 135/315°，而这两个 roll 会把掌体翻到 z=698.6 mm（桌面以下） | 网格改 5°；板几何加 **−45° 安装四元数**抵消掌自身 45° roll（roll 0 时板轴与世界对齐，竖向跨度=板厚 60 mm）；烘焙器的跨度计算补上安装四元数 |
| r140 | 135/315 两个 roll 下手指扎进箱体 47–52 mm | 卷曲度纳入搜索（`W6B_PLATE_CURLS`） |
| r141 | 113×60 mm 板下**手部自身几何**必进箱：拇指中节越过面平面 27.5 mm、环指近节 7.9 mm；81 种拇指关节组合最好仍 −7.89 mm；卷曲越深越糟（0.0→−27.5，2.0→−41.6，因 Allegro 是朝掌心接触侧卷） | 若靠加厚板把掌推后需 64 mm 厚 —— 那是夹具不是手。改为 **41×40×40 mm 小板**（竖向跨度 ≤28.3 mm，任何 roll 都装得下），让扫描去找手指让开的 roll |

### 左臂（已通）
`W6B_R38_SEAT=left` 打开落座精修后，左 `ff_tip_left_collision` 在端面真实接触：
陷深 −0.13 ~ −0.18 mm、法向力 **1.42–1.70 N ≥ 1.0907 N**、gate6 **PASS**（三种 standoff 复现）。
此前 `W6B_R38_SEAT` 默认 "0" ⇒ 落座精修从未运行，左指尖以棱边先碰箱。
左食指**远节连杆**（非垫片）会以 6.5–11.8 N 推箱使箱滑走 −3.6~−5.1 mm（超过 μ·mg≈7 N 静摩擦），
这是右板永远差 1.9 mm 的原因；`W6B_R127_LEFT_CURL_DELAY=80` 可把箱移降到 −1.17 mm。

### 当前诚实态
右板尚未取得平面真实接触（小板 + 全 roll 扫描正在进行）；左臂 gate6 已 PASS；
箱体位移 ≤2.8 mm；轨迹平滑（峰值命令步长 0.0071 rad/帧）。

## r144–r146：右臂取得**合法端面真实接触**（评审核心问题已回答：能）

### 净室复验配方（clone @ c04c0426，场景 r43）
烘焙：`W6B_PLATE_ROLL_DEG=10 W6B_PLATE_CURLS=-1.0 W6B_PLATE_STANDOFF_MM=-0.2
W6B_PLATE_TABLE_CLEAR_MM=5 W6B_TIP_CLEAR_MM=5.0 W6B_HAND_CLEAR_MM=2.0 plate_terminal.py 0 <r43>`
运行：`W6B_R109_SQUEEZE_MM=0.4 W6B_R38_SEAT=none W6B_R144_FINGER_LEAD=100
W6B_R127_LEFT_CURL_DELAY=80 W6B_R63_TARGET=1 W6B_R18_NATIVE_TARGET=1 W6B_R75_SIDE_APPROACH=1 ...`

### 右臂 gate6：**PASS**
| 指标 | 值 | 判据 |
|---|---|---|
| 真实接触数 | 1（`palm_collision`） | ≥1 |
| 法向力 | **2.697 N** | ≥ **1.090711 N**（= m·g/(2μ)，由场景 0.400398 kg / μ=1.8 导出） |
| 接触点是否在端面 | **True**（平面，非倒角） | True |
| 陷入深度 | **−0.006 mm** | ∈[−0.5, +0.2] mm |
| 终端跟踪误差 | ≤0.0012 rad（六关节） | — |
| 箱体峰值位移 | **0.323 mm**（终态 0.166） | 无滑移/无推走 |
| 峰值命令步长 | **0.0084 rad/帧**（修复前 0.203） | — |
| 峰值掌速 | **3.05 mm/帧**（修复前 20.4） | — |

挤压量 `W6B_R109_SQUEEZE_MM` 对右板单调可控（沿面法向 Jacobian 平移、姿态保持）：

| squeeze (mm) | 右板力 (N) | 右板陷入 (mm) | 箱峰移 (mm) |
|---|---|---|---|
| 0.2 | 1.417 ✓ | −0.003 | 0.27 |
| 0.4 | 2.697 ✓ | −0.006 | 0.32 |
| 0.6 | 4.035 ✓ | −0.010 | 0.39 |
| 0.9 | 5.807 ✓ | −0.014 | 0.44 |
| 2.0 | 12.42 ✓ | −0.029 | — |

### 左臂 gate6：**FAIL（92%）**
`ff_tip_left_collision`：力 **1.004 N**、on_end_face True、但垫**中心**离面 **+0.315 mm**
⇒ 落在真实接触窗 [−0.5, +0.2] 之外（垫片倾斜，角部承载）。
挤压加到 0.7–0.8 mm 时垫心进入窗内（−0.010~−0.018 mm）但力**塌到 0.09–0.15 N** —— 不单调、不稳。

已排除的左臂旋钮（实测无效，负结果入账）：
- `W6B_R50_PRELOAD_MM` 0.05→0.25：穿透恒 −0.026~−0.030 mm；
- `W6B_R50_CURL` 1.25→1.85：逐位无变化（终端手指来自参考，非该旋钮）；
- 参考 `palm_pos` 进给 0→2.0 mm：执行掌位只动 0.05 mm（左终端不跟随参考 palm_pos）；
- `W6B_R38_SEAT=left` 落座精修在本场景**测距错误**：报左垫离面 58–65 mm、sv=−4.54 mm
  （`retarget_dual.py:3058` 的 `face_x * spec.face_sign` 对左侧双重取反：`face_x` 已含符号
  −0.06，再乘 −1 得 +0.06；后续 r48/r49 迭代用的是带符号 `sv` 目标，故此式只污染初值 ——
  但初值 125 mm 量级已足以让迭代跑飞）。

有效杠杆是**垫厚**（授权 A）：左薄垫半厚 0.75→1.45 mm 后，左指尖力 0.26→0.98–1.01 N。

### 下一步（左臂 0.09 N 缺口）
1. 修 `:3058` 的符号（`face_x + spec.face_sign * PRELOAD_M`，深度取 `min(xs*face_sign)`），
   重开 `W6B_R38_SEAT=left`，验证左垫能否既平贴又 ≥1.09 N；
2. 若仍塌力：查 `ff_padrx/ry_left` 铰接刚度（0.02）在平贴时的卸载路径；
3. 左臂过门后再做双手 6 mm/90 帧起升的 7 条验收。

## r147–r148：双侧 gate6 同跑同过；起升相接通但箱体未真正离桌

### r147（左臂解锁）
`refine_terminal_seat` 的首个位移量把「最深垫」取成 `min(xs)`，但 `s` 是**带符号面坐标**
（下游按 `s*face_sign` 施加）：右臂两式相同，左臂的最深垫是 **最大 x**，
初值因此偏 ~125 mm，把 r48/r49 迭代带跑（报左垫离面 58–65 mm、sv=−4.54 mm）。
修为 `min(xs*face_sign)` 后，同一配方左臂落座正常：
`ff_tip_left` 间隙 **+0.102 mm**（落在真实接触窗 [−0.5,+0.2] 内）、力 **2.035 N**。

**双侧同跑同过（净室 3 次逐位一致）**：右板 4.1578 N、左指尖 2.0349 N，
均 ≥ 地板 1.090711 N；箱体峰移 0.32 mm；右臂终端跟踪 ≤0.0012 rad。

### r148（起升相）
`W6B_LIFT_MM/W6B_LIFT_FRAMES` 在双流末尾追加起升：姿态保持 + 手指保持终端卷曲，
逐帧 6×6 阻尼 Jacobian 步进（0.067 mm/帧、0.05 rad 信赖域）。
实测双掌分别上升 **5.9999 / 6.0027 mm**（命令 6 mm / 90 帧）✓。

**但箱体未真正离桌**：起升中桌面支撑 3.79 → 0.39–0.59 N（手确实接过了重量），
箱峰值只到 **+1.53 mm**，随后回落至 +0.10 mm。原因：
- **左垫接触断续**（有接触时 0.51–0.53 N，多数帧缺失）—— 左的压紧来自**手指伺服**
  （0.04–0.18 N·m）而非臂，手一升指接触即断；右板全程 2.27–2.86 N 稳定；
- 箱还靠在 `yaw_stop_negy_bar` 上（0.6–1.5 N）—— 寄生支撑，起升判据前必须消除。

### 下一步（按序）
1. 左臂压紧改由**臂**承担（与右板同理）：或加大左终端挤压使指伺服进入自锁区，
   或给左也加小掌板（授权 B 已涵盖），目标是起升全程左法向力 ≥1.09 N 不断续；
2. 消除 yaw 挡杆寄生支撑（查箱为何横向靠上挡杆：起升前箱已被左推 −0.2~−0.5 mm）；
3. 判据：箱升 6 mm 且**桌面支撑力归零**、无滑移/穿透/腕撞；
4. 7 条验收 + 净室验证器 verify_v11 + 组包 v11。

### 视频保真度（用户关注点）
gate5（掌位偏差）仍红：侧装单元的接近段是 R75 笛卡尔 staging（升空→转姿→垂直落下），
是机器人专属路径、刻意不同于视频人手弧线。评审授权 B 的验收顺序是「先证明真实接触+法向力+起升」，
保真度回贴排在其后：届时把 staging 段与视频驱动轨迹混合，用 gate5 与三联图量化。

## r149：起升挤压 —— 提升峰值但暴露左手单点接触

`W6B_LIFT_SQUEEZE_MM`（沿面法向、随升程斜坡的 inward 平移，默认 0）实测：

| 起升挤压 | 箱峰升 | 箱终升 | 终态桌面支撑 | 终态挡杆力 |
|---|---|---|---|---|
| 0.0 mm | +1.528 mm | +0.104 mm | 0.94 N | 0.55+0.30 N |
| 0.5 mm | **+3.071 mm** | +0.439 mm | **28.07 N** | 17.8–22.1 N |
| 1.0 mm | +2.573 mm | −0.754 mm | **251.49 N** | 11.8 N |

升程中桌面支撑确实一度归零（min 0.00 N）—— 手接过了重量 —— 但终态把箱**楔进**桌面与挡杆。
机理：左手只在**一个高位指尖**（z≈766 mm）接触，右板居中（z≈740 mm），
两个法向力构成力偶，把箱往下/往挡杆里压。

**结论**：出路不是加大挤压，而是让左手也取得**臂承力、居中**的接触
（授权 B 对右臂已允许的掌板，同样适用于左）—— 两侧法向力共线后，
6 mm 起升才可能既离桌又不楔死。

### 本轮已交付（全部净室可复现、已推送）
| 项 | 状态 |
|---|---|
| t=299 板砸箱根因（烘焙器 180° 旋转盲区） | **已修**：峰值命令步长 0.203→0.0084 rad/帧、掌速 20.4→3.05 mm/帧、箱推 11.93→0.32 mm |
| 右臂能否合法提供 1.0907 N（评审核心问题） | **能**：真实 solver 接触、平面端面、力 1.42–12.42 N 可调、陷入 ≤0.03 mm |
| 双侧 gate6 同跑同过 | **是**：右 4.1578 N + 左 2.0349 N，净室 3 次逐位一致 |
| gate6 测量正确性 | **已修**：纳入箱体倒角 geom（此前 18.7 N 的倒角接触被漏报为 1.91 mm 无接触） |
| `debug_history.npz` 右臂关节记录 | **已修**（此前记录的是左臂） |
| 左臂落座精修 | **已修**（`min(xs)`→`min(xs·face_sign)`，左力 0.26→2.035 N） |
| 起升相（6 mm/90 帧） | **已接通**：双掌实升 5.9999/6.0027 mm；箱未真正离桌（见上） |
| 双手 6 mm 起升 7 条验收 | **未完成**：阻塞于左手单点接触 |
| 视频保真（gate5） | **仍红**：接近段是 R75 staging，按评审顺序排在接触/起升之后 |

## r150：左手掌板不可直接用作压面（负结果，关闭一条捷径）

曾设想「左手沿用已有 `palm_left_collision`、只前伸一段即可臂承力」——**实测否定**：
在左终端位姿下（左臂执行/命令构型 + 终端手指），

| 量 | 实测 |
|---|---|
| 左掌心局部 x 轴在世界系 | `(0.697, −0.211, 0.685)` —— 与世界 x 成 **≈45°**，即板法向**不朝箱面** |
| 左板中心 | `(−162.85, −21.40, 842.91)` mm（在掌心 `−122.84` 之后） |
| 左板 ↔ 箱 `object_geom` | **88.69 mm** |
| 左板 ↔ 桌面 | 94.13 mm |

因此 gate6 记录里 `palm_collision_left` 的 `surface_gap_mm = 0.0` 是**退化读数**（相距 88.7 mm
却报 0.0），左终端位姿是为**垫片**解出的（`palm_pose_refine.json` 的左侧），不是为板解的。

**结论**：要让左手取得臂承力、居中、与右板共线的接触，必须像右侧一样为左**重解终端**
（把掌心法向对到 −x 面法向 + roll 扫描 + 间隙硬门 + 压柱前伸），即把 `plate_terminal.build_ref`
泛化到左侧（当前写死 `ur5e_r_*`、`palm_collision`、+x 面、右手指尖）。这是下一步的主工作项。

顺带记一笔测量缺陷：gate6 对 `palm_collision_left` 的 `surface_gap_mm` 会返回 0.0 而非真实距离
（88.69 mm），需在 v11 前修（`mj_geomDistance` 的 distmax 截断被当成 0 记录）。

## r151–r152：gate6 测量缺陷两处修毕，gate6 退出失败门列表

**r151（名称错误）**：gate6 查左板用名 `palm_collision_left`，而三个侧装场景
（r41 mixed / r42 / r43）实名均为 `palm_left_collision` ⇒ `mj_name2id` 返回 **−1**，
左板记录整体是伪影：`surface_gap_mm` 读到 **0.0**（对 geom −1 做距离查询的退化返回），
真实距离 **117.4536 mm**；`in_contact` 永假；箱-手摩擦取 max 时也漏掉了左板。
两种拼法现在都尝试。

**r152（判据过宽）**：`on_end_face` 只查 x 平面，于是左板最近点 —— 箱体**上棱边**
（z_rel = +40.0 mm 恰等于 half_z，x_rel = −60 mm，距板 117 mm）—— 被记为「在端面上」。
现要求最近点落在面矩形内且离边 ≥1 mm。`real_contact` 还需 `in_contact`，
所以此前**没有**造成假 PASS，但记录会误导评审阅读。

**修后（squeeze 0.4 / preload 0.05 / seat=left / lead 100 / left delay 80，r43）**：

| 侧 | 接触几何 | 间隙 | on_end_face | 法向力 | 地板 | 判定 |
|---|---|---|---|---|---|---|
| 右 | `palm_collision`（掌板） | **−0.0099 mm** | True | **4.1578 N** | 1.090711 N | **PASS** |
| 左 | `ff_tip_left_collision` | **+0.1024 mm** | True | **2.0349 N** | 1.090711 N | **PASS** |
| 左 | `palm_left_collision` | 117.4536 mm | **False**（修正后） | 0 | — | 不参与 |

`verdict` 失败门从 4 项降到 3 项：**gate6_executed_contact 已退出**。

## 剩余 3 门的确切失败原因（不事后调阈值，如实记录）
| 门 | 失败原因（实测） |
|---|---|
| `gate1_final_touch` | 判据是**垫片矩形**（每手 ≥2 垫落在中带矩形内）。右臂现在用**掌板**（0 垫入矩形，rect_dist 46.92 mm），左臂只有 1 垫接触（rect_dist 7.34 mm）。授权 B 的板分支下该判据按构造不可能过 —— 等价判据是 gate6（真实接触 + 力地板）。v11 需向评审说明判据适用域。 |
| `gate4_joint_limits` | 仅因左手指 qpos 越界 **8.085628e-07 rad**（≈0.00005°，第 172–207 帧中 15 帧）；判据是严格 `== 0.0`。属浮点噪声，但**不放宽阈值**，如实留红。 |
| `wrist_referenced_error` | 视频保真门（4c：腕参考轨迹误差 ≤ 手跨距 10%）。接近段（第 1–151 帧）实测：右 **28.58 mm = 30.2%**、左 **18.53 mm = 19.6%**，均超 10% 限。这就是用户在页面上看到的「与视频差异大」的量化来源。 |

## 保真度（用户关注点）的量化现状与成因
- 手跨距中位数 195.44 px = 94.61 mm；误差以「占跨距百分比」计，限 10%。
- 右 30.2% / 左 19.6% —— 侧装单元的既有状态：r30 台账已判「侧装场景需要整条管线重设计：
  retarget 的世界坐标目标（face anchor ±0.11/0/0.76、flyover、margin、approach）都是
  **吊顶安装专属**，侧装基座够不到（换场景后 gate1 1044/685 mm）」。
- 因此保真度回贴需要把接近段目标从「吊顶坐标系常量」改为**侧装可实现的视频跟随目标**，
  这是评审顺序上排在接触/起升之后的一项独立工作。

## r153：左臂掌板分支开工 —— 场景 r44 + sweep/baker 双侧参数化 + 左侧可行性

**为什么必须做**：起升时左垫接触断续（力来自 0.04–0.18 N·m 的手指伺服），
且推进左终端会使指垫屈曲（squeeze 0.7→1.5 mm 时左力 2.035→0.07/0.28/0.48 N 崩塌），
而右板（刚性装在掌上）同期 4.16→9.94 N 单调上升 ⇒ 只有**掌刚性元件**能提供臂承力，
起升需要双侧臂承力且共线。

**场景 r44**（`allegro_floating_ur5e_side_r44_dualplate.xml`）= r43 + 左掌板：
`palm_left_collision` 取与右逐字相同的 `pos="0.0400 0 -0.0475"`（尺寸/挂载四元数来自
class 默认）。左掌 body 自带镜像 quat `0 1 0 1`、childclass 同为 allegro_right，
故同局部 pos/quat 即镜像板。r43 冻结，右臂单侧配方仍可复现。

**sweep 参数化**（`W6B_PLATE_SIDE`）：body/geom/joint 名加 `_left` 后缀、
面法向与板站立侧随 FS=±1 翻转。**修了一个右专属符号 bug**：切面偏移原为
`centre − half`，对左应为 `centre − FS*half` —— 旧式使每个左候选读 surf=−84 mm，
左 sweep 恒 0 可行。

**左侧可行性（r44，3456 网格）**：**688 个可行位姿**。最优行示例：
curl 0.0 / roll 60° / lat −20 mm：surf +0.00 mm、箱隙 +8.38 mm、桌隙 +9.94 mm，
四指尖离箱 8.4–81.4 mm、离桌 9.9–161.0 mm。左手**有**臂承力压面的几何解。

**baker 参数化**：同一 spec 贯通 build_ref 的全部右专属字面量（palm/plate/hand_root/
ur5e_r↔ur5e_l/指尖名/chamfer_posx↔negx/面符号/板 y 偏移镜像），并加
`W6B_PLATE_BASE_REF` 使左遍以右板 ref 为基（两遍合成双板 ref，不丢右板）。
左烘焙（idx=1）在净室运行中，结果另记。

## r154：保真度（4c 腕参考门）的偏差来源与帕累托冲突（本地 A/B，方向性）

用户要求「和原视频高度一致」。量化现状（净室主配方）：右 **30.2%**、左 **19.6%**
（限 10%，手跨距 94.61 mm）。来源分解：

1. **r59 侧装重注册（设计使然）**：原视频是 ego 视角吊顶单元，手的接近路径在侧装
   单元里**物理不可行**（掌会穿越箱中面并插进桌面；r53 基线实测左拇指 228 接触帧、
   右指 66 帧、箱被推 55 mm）。r59 保留**时序**、重注册**几何** —— 偏差是可行性代价。
2. **flyover 钟形抬升（主导旋钮）**：主配方 `W6B_R23_FLYOVER_MM=250`，即接近段中段
   抬高 250 mm ≈ 2.6 倍手跨距 —— 单此项就远超 10% 限。

本地 A/B（同配方仅改 flyover；本地 runs/w6b 被验证器重建过，绝对值不作母版数，方向可信）：

| flyover | 4c 误差 frac 右/左 | 手-桌接触帧 右/左 | 箱峰移/终移 mm |
|---|---|---|---|
| 250（主配方） | 0.668 / 1.004 | 接近段 0 | 小 |
| 120 | 0.298 / 0.631 | 158 / 88（拖桌） | 17.8 / 13.3 |
| 0 | 0.282 / 0.877 | 30 / 4（右 29 帧在**终端窗**：拇指贴桌） | 12.0 / 8.0 |

**结论**：10% 保真限与「零手-桌接触」在侧装单元**不可同时满足**；flyover 是两者间的
帕累托旋钮且非单调（左在 0 时反而比 120 差）。要真正回贴视频，需要把接近段从
「锚点+钟形抬升」改为「视频路径在可达且无碰撞走廊内的投影/裁剪」—— 独立设计任务，
排在双板起升之后。任何阈值或判据重定义都须向评审显式披露。

## r155–r156：双板终端 —— 双侧同时真实接触达成；温和双压的平衡问题定界

**r155**：管线加 `W6B_R155_TRIM_MM_RIGHT/LEFT`（运行期、沿面法向回撤终端目标，
只动目标不动烘焙臂种子）。动机：双板把箱夹在两个刚性臂伺服之间，需逐侧配平穿透量；
重烘焙 12 min/次，运行期旋钮 10 s/点。

**r156（双板首跑，r44 + plate_1 ref）**：baker 的 BASE 缺主配方旋钮
（R127 left curl delay 80 / R144 finger lead 100 / R38 seat left）时，**右指尖**在
278–279 帧插进箱 +x 面（−0.14→−0.35 mm、1.87→4.21 N）把箱推 −1.53 mm 顶进 yaw 挡杆，
右板差 2.19 mm 够不着（F=0）。补上三旋钮后：

| 侧 | 接触几何 | 间隙 | 法向力 | 地板 | 判定 |
|---|---|---|---|---|---|
| 右 | `palm_collision` | −0.063 mm | **26.95 N** | 1.0907 | **PASS** |
| 左 | `palm_left_collision` | −0.012 mm | **21.44 N** | 1.0907 | **PASS** |

**双侧掌板同时真实接触、同时过 gate6 —— 首次达成。** 但箱终移 4.0–6.8 mm（滑进挡杆），
力 21–27 N 远超所需。

**平衡问题定界（trim/squeeze 扫描）**：力对 trim 呈 **bang-bang**（左 21 N 或 0，
右 23–27 N 几乎不变）—— 接触刚度极大，0→21 N 之间仅 ≈2 µm 穿透窗，0.1 mm 级旋钮跨不过。
力的量级由**伺服停驻平衡**决定：同一管线右构型停驻在 4.16 N（r43 主配方）、
左烘焙构型（roll 80°）停驻在 21 N ⇒ 左臂在该构型的 x 向等效刚度高一个量级。

**下一步（已定界未做）**：温和双压（双侧 ≤3 N、箱移 <1 mm）需要
(a) 换左 sweep 的 roll 重烘焙，选 x 向更软的构型（48 个可行 roll 可选），或
(b) µm 级 standoff 扫描（把烘焙 standoff 当连续参数，二分至左力 ≤3 N）。
起升验收必须在温和双压下做 —— 箱被挡杆楔住时不可能离桌。

## r157：双板力量级的确切机制（µm 扫描 + 与主配方对照）

µm 级左 trim（0.002–0.06 mm）仍 bang-bang：左 20.1–21.2 N 或 0，右 23.7–26.2 N 几乎不变。
⇒ 力**不由目标位置**设定，而由伺服停驻平衡设定。

对照实验揭示楔箱放大效应：r43 主配方（箱自由，终移 0.32 mm）右板停驻 **4.16 N**；
r44 双板（箱被挡杆楔死，终移 3.9–4.8 mm）右板 **26 N** —— 同一右 ref、同一 standoff −3.0 mm。
箱自由时会退让、把力封顶在 4 N；楔死后两侧压力升到各自停驻值。

左停驻力高的直接原因：**终端插入过冲**。左烘焙 standoff = +0.1 mm（目标本应在面外），
执行态却穿透 −0.002…−0.012 mm 并停驻 20 N ⇒ 左臂 x 向等效刚度 ≈70–200 N/mm
（右 ≈1.4 N/mm，由 4.16 N/3 mm 估）。过冲来自左终端插入的速度/伺服滞后
（左终端相对其视频末位姿是大关节重排）。

**温和双压的两条杠杆（已定界）**：
(a) 减慢左终端插入（逐侧 blend 窗或左终端步长限幅），把过冲压到 <5 µm；
(b) 换左 sweep 的 roll 重烘焙（48 个可行 roll），选 x 向更软/插入行程更小的构型。
两者都须先解除楔箱：双侧力 ≤ 桌面摩擦阈值（μ_t×3.93 N）时箱不会滑向挡杆。

## r158：温和双压卡点的最终定谳 —— 左插入路径扫掠碰撞（非力度整定问题）

排除链（全部净室实测）：
- `W6B_BLEND_SECONDS` 1.6/4.8/9.6：**逐位惰性**（力/箱移/时间线三处全同），
  插桩证实自适应窗**确实生效**（n_blend6 40/74 → 96/96）但结果不变 ⇒
  窗长不是力的因。
- `W6B_R155_TRIM_MM_RIGHT` 1.0–2.9 mm：右力仅 26.1→24.5 N ——
  力**不由终端目标位置**设定（trim 后右目标已在面外，仍 −0.06 mm/25 N）。
- `W6B_R75_STAGE_MM` 0/30/60：力 17–49 N 双稳震荡，楔箱不解除。

**根因**：r73 日志 `path clearance −12.204 mm [object<->rf_distal_left]`。
左烘焙行 curl 0.0（左可行行仅 curl 0，40 个），张开的手指链虽在**终端位姿**
净（手/箱 +50 mm、指尖在板后 −82…−95 mm），但 r74 笛卡尔直线 lerp + 姿态
slerp 的**插入途中**环指中节扫进箱面 12 mm，把箱横向推进对面板 ⇒ 双臂对顶、
箱楔死、两侧力升到各自伺服停驻值（右 17–49 N 双稳）。
右板无此问题：curl 1.25 折叠指链 + 单侧受压箱可退让（r43 主配方 4.16 N）。

**结论**：温和双压不是整定问题，是**左插入路径规划**问题。修法（已定界，下一步）：
左侧插入改两相 —— 先在箱外（−x）完成姿态旋转（掌保持 x ≪ 面平面），
再沿 +x 短程平移进场；即把 r74 的单段 lerp 换成「旋转在远处、平移贴面」的两段路径，
并让 r73 的候选搜索在此路径族内重开（现仅 1 候选、无从规避）。

## r162：v12 P0 —— 逐帧摩擦承重账目闭合（评审 §六 P0 的交付）

管线 trace 扩展（`W6B_R126_BOX_TRACE=1` 时）：每接触记录 `f6`（接触系
[fn, ft1, ft2, tq1..3]）、`frame`（接触系→世界系）、`box_is_g1`；箱体加记 `quat`。
新分析器 `tools/w6b/lift_force_account.py`：按类汇总世界系力、有限差分箱加速度、
牛顿-欧拉线性残差分相检查。

**符号约定（实测定谳，5 方向同时校验）**：`mj_contactForce` 返回的是 **geom2 受力**，
对箱体恒取 `−fw`。同一保持帧上同时满足：桌上静置 +mg、右板推 −x、左板推 +x、
negy 挡杆推 +y、滑动摩擦向下。（曾按 box_is_g1 分侧翻号 → 板力方向反、残差 10.8 N；
统一取负后全暗合。）

| 相 | 板 Fz | 桌 Fz | 挡杆 | 角铁 | ΣFz | m·az | z 残差 |
|---|---|---|---|---|---|---|---|
| approach | 0 | +3.928 | 0 | 0 | +3.928 | 0 | **0.0000** |
| grasp | 0 | +5.714 | −2.377 | 0 | +2.807 | 0 | +1.12（插入瞬态） |
| lift_accel | +1.667 | +2.706 | −0.252 | 0 | +4.020 | ~0 | −0.092 |
| **lift_hold** | **+5.776** | **0.000** | −1.427 | −0.822 | +4.032 | ~0 | **−0.105（=2.7% 重量）** |

- **承重归因**：保持相双板竖直摩擦 +5.776 N = **147% 重量**；桌面支撑恰 0。
  多出的 2.25 N 用于克服挡杆（−1.43）与角铁（−0.82）的下拉摩擦 —— 这正是
  P1 消融要隔离的旁载；若消融后仍成立，147% 将降回 ≈100% 且更干净。
- 3D 残差范数 hold ≤2.0 N（44 N 板压的 ≈5%）：x/y 向板压 22.24/22.56 N 的
  不平衡与求解器单点记录噪声；z 向（承重方向）残差 0.10 N。事前登记容差建议：
  z ≤ 5% mg（实测 2.7% PASS）。

## r163–r165：干净抓取变体的排除链（v12，"不接受失败"方向下的如实记录）

**取证升级（r162 trace + 力矢量）**：抓握前箱移的真肇事者是**左中指中节**
（`mf_distal_left_collision`，第 276–287 帧 1.7 → 22.4 N，箱 +x 2.6–2.7 mm 顶进
yaw 挡杆）。此前 r158 引 r73 的 `rf_distal −12.2 mm` 为路径扫掠——实际是进入段
本身：左臂进入几何下，中节横穿端面平面，推进的手把它沿面压滑。

**排除链（每个都有实测锚）**：
| 尝试 | 结果 |
|---|---|
| 运行期折指 curl 0.6/1.0（改 ref fingers） | 左板失接触（F=0）：落座精修按垫片量测、终端随指姿态偏移 |
| 折指 + 关落座精修（R38_SEAT=0） | 左板仍 F=0；顶箱 20.8 N 依旧 |
| 推台 150/200 mm | 顶箱 23.3–23.6 N 不变 → 非前伸清障问题，是进入段几何 |
| **r163 折指重烘焙（plate_2，curl 1.0）** | 右 24.18 N ✓；左板 +0.0102 mm 差 10 µm |
| **plate_2 + 左 trim −0.01 mm** | **双侧板真实接触**：右 28.09 / 左 22.48 N ✓ |
| plate_2 起升全跑 | 箱升 6.171 mm 保持、末帧桌支 0 ✓；**但顶箱 22.4 N 依旧、箱移 4.7 mm** |

**结论（假说修正）**：顶箱不是"开链前伸"，而是 **Allegro 指链几何 × 左 roll 60° 进入
方向**的必然——折指与开链都撞（curl 朝接触侧卷，r141 对右手已实测；右臂不撞是
其 roll 解恰好让指链走面上方）。折指变体（plate_2）本身有效：双侧板接触 + 起升成立，
且左终端指链收拢更接近真实双板夹持语义，**保留为干净变体候选**。

**下一步设计（P1 前置，已定界）**：改 r75 相 B 末段路径——以 +20 mm z 偏移进入、
末 20 mm 沿面下滑落座，让指链从面上方掠过（右臂不撞的机理移植到左臂）。
A/B：v11 基线（plate_1）不动，干净变体（plate_2）各自跑有无挡杆。

## r166：v12 P1 —— 挡杆约束消融 **PASS**：无辅助约束下双板独立摩擦起升成立

场景 `r45_ablation` = r44 仅移走辅助约束（两条 yaw 挡杆 + 四个 vise 角铁 →
y=±0.35 m），其余逐字相同。同控制序列、同参考（plate_2 + 左 trim −0.01 mm），
A/B 两条件：

| 条件 | 辅助接触 | 双板力 (N) | 箱升 (mm，峰=终) | 末帧桌支 | z 残差 |
|---|---|---|---|---|---|
| r44 保留约束 | 挡杆 −1.43 / 角铁 −0.82（下拉摩擦） | 22.24 / 22.56 | 6.080 | 0 | −0.105 |
| **r45 解除约束** | **全程 0 次接触** | **24.00 / 24.04** | **7.814** | **0** | **+0.0036** |

- **独立承重因果链闭合**：解除辅助约束后，双板竖直摩擦 **+9.603 N（244% 重量）**
  独自承担全部重量，无任何其他承重接触；牛顿-欧拉 z 残差 **0.0036 N（0.09% 重量）**
  —— 无旁载后残差反而缩小 29 倍，反证 r44 的 0.105 N 残差即旁载记录噪声。
- 复跑逐位一致（7.8136 mm / 23.9995 / 24.0448 N）。
- 如实披露：解除约束后抓握前箱移更大（x 6.3 mm / y 2.4 mm，无约束可挡）——
  预位移是接近段指链过面几何的独立问题（r165 已定界），不影响承重归因。
- 评审原文 PASS 条款对照：「解除辅助约束后仍能独立起升且无其他承重」——**满足**。

## r167：v12 P2 —— 冻结扰动包络 **9/9 全 PASS**（r45 消融场景，净室克隆逐位复现）

**事前冻结集**：左 trim {−0.005, −0.01, −0.02} mm × squeeze {0.2, 0.4, 0.8} mm，
其余旋钮全部冻结在 r166 配方（plate_2 + 折指 + 主控旋钮组）。

| L\S | 0.2 | 0.4 | 0.8 |
|---|---|---|---|
| −0.005 | 7.766 / 23.5 N | 7.818 / 24.1 N | 7.805 / 28.5 N |
| −0.01 | 8.064 / 23.4 N | **7.814 / 24.0 N** | 7.736 / 28.6 N |
| −0.02 | 8.066 / 23.4 N | 7.831 / 24.0 N | 7.674 / 28.6 N |

（格值 = 箱升 mm（峰=终，无滑移）/ 双板合力 N；9 格全部：末帧桌支 0、辅助接触 0、
gate6 双侧 PASS、x 漂移 6.3–6.8 mm、y 漂移 2.4–2.6 mm。）

- **包络结论**：升幅 7.67–8.07 mm、力 23.4–28.6 N，无一格失稳/滑移/失接触；
  squeeze 主导板力（+4.5 N/0.6mm），trim 对升幅扰动 <±0.35 mm —— 终端微
  扰动下起升因果稳健。
- **净室复验锚**：基线格（−0.01/0.4）与 r166 P1 逐位一致（7.814 mm，
  24.00/24.04 N，x 6.32）；陈旧提交 trace 曾造成 3.746 mm 鬼影读数，已从
  工作流排除（跑前必删 trace，读数必配新鲜度校验）—— 作为流程教训入账。

## r168：v12 P3 —— verify_v12 封闭性验证器 **16/16 PASS**（净室）

`tools/w6b/verify_v12.py` + `runs/w6b/v12_asset_manifest.json`（8 项 SHA-256，
从 git HEAD 内容取哈希）：

| 段 | 检查 | 结果 |
|---|---|---|
| 1 | 资产清单逐项匹配（场景 XML×2、板参考×2、gate3 种子、管线/工具×3） | ✓ fail-closed |
| 2 | 交付配方新进程复现：rise 7.814、gate6 24.00/24.04 N、辅助接触 0、末帧桌支 0 | ✓ |
| 3 | 力账目：approach 桌 3.928 N、hold 板 9.603 N（244%）、残差 +0.0036 N | ✓ |
| 4 | P2 角点复现（8.066 / 7.805 mm，容差 0.02） | ✓ |
| 5 | 逐帧全轨迹对账（421 帧 ×3 维）：**max |dpos| = 0.0** | ✓ 位级一致 |

流程教训（入账）：证据 trace 的复制曾与网格循环竞态（存成最后一格）；验证器
把这类错误变成了 FAIL 而不是静默通过——这正是封闭性的意义。

## r169：v12 —— 双段进入路径落地：左臂指链顶箱**消除**，漂移 −25%

`W6B_R169_ENTRY_Z_MM_LEFT`（r169/r169b）：相 B 改两段——staging → 终端上方
+30 mm 航点（净空内）→ 末 30% 垂直下落进靶。R94 反向解与命令目标表同步改走
`_b_tp`。

| 配置 | 箱 x 漂移 | mf_distal 顶箱 | rise（峰=终） | 双板 F | g6 |
|---|---|---|---|---|---|
| 直进（r167 交付） | 6.32 mm | 22.4 N @287 | 7.814 mm | 24.0/24.0 | ✓ |
| **Z=30 进入** | **4.73 mm** | **无** | 7.153 mm | 24.97/24.95 | ✓ |
| Z=20 | 7.85 | — | 14.8（异常） | — | ✗ |

- Z 高度双峰：20 mm 破坏 IK 分支收敛（rise 翻倍、wrist_ref 红），30 mm 良好
  —— 分支选择对航点几何敏感，30 mm 为选定工作点。
- 漂移分解（Z=30）：~1.6 mm 在早期进入（无接触记录，待查），~3.1 mm 在双板
  落座挤压段（320–380 帧）—— 对 µm 级左右 trim 不敏感（右侧 trim 在此路径
  未接线，已查证逐位无效），是双板无约束挤压的内在平衡，作为披露项保留。
- 判据对照：评审 P1 要求的是「独立起升」✓（无辅助约束 7.15 mm 保持）；
  预抓握横移从「指链扫面事故」降级为「挤压平衡的连续滑移」，性质完全不同。

## r170：评审 v12（8.9/10 REVISE）的「消失的 5.675 N」—— **已解：左板分类 bug，物理从未破**

评审 P0 裁决 FAIL — ACCOUNT INCOMPLETE：hold 相 plate +9.603 N vs 重量 3.928 N，
差 5.675 N（144.5% 重量）去向不明。

**根因（`classify()` 子串匹配）**：左板 geom 名 `palm_left_collision` **不含**
子串 `palm_collision` → 被归入 `other` 列。SUM 一直含左板（闭合从未破：
SUM−mg = −0.0040），但打印把左板的 −5.668 N 藏进了未显示列；包里又把
+9.603 误标为"双板合力"（实为右板单板份额）。

**修正后的真实账目（r45 基线格 hold 相，世界系，全类显示）**：
| 类 | mean Fz |
|---|---|
| 右板 palm_collision | +9.592 N（向上摩擦） |
| 左板 palm_left_collision | **−5.668 N（向下摩擦）** |
| 其他所有 geom | 0 |
| **Σ** | **+3.924 = mg（闭合到 4 mN）** |

物理读法：箱子相对左板向上滑 → 左板摩擦向下（牛顿-欧拉要求，非缺接触）；
右板承担 244% 重量的竖直摩擦，左板反向拖拽。**双板独立承重成立**——无任何
非板接触，闭合 0.004 N。

**r169 Z=30 变体的附加优势**：hold 相右板 +3.247 / 左板 +0.678 —— 两板**都向上
托**，无反向滑动（更健康的抓取，远离滑移边界）。

修复：classify 双拼写匹配 + 账目逐类全打印 + verify_v12 断言
"hold 相除双板外无任何接触类 + plate 份额 = mg + SUM−mg ≤ 0.02 N"。
净室 verify_v12 **18/18 PASS**（含位级轨迹对账 0.0）。

## r171：P0 整改复核 —— **FAIL → PASS WITH EDITS，9.1/10 REVISE**

评审接受 5.675 N 根因解释、撤销 ACCOUNT INCOMPLETE。新揭示的研究事实：
"相同净承重可来自截然不同的左右力分配" —— 左右份额应成为抓取质量指标
（r169 Z=30 的 +3.25/+0.68 双板同向 vs 基线 +9.59/−5.67 反向拖）。

三项验证层整改（不重开实验）：
1. verify_v12 hold 相**逐帧**检查接触集合（防同类相消）→ 每帧 plates-only；
2. 账目尾部闭合公式 `abs(tot + m·g)` → `abs(tot − m·g)`（现输出闭合 0.0040 N）；
3. 删除 EXPECTED['hold_plates_N']=9.603 旧锚（防再误读右板份额为双板合力）。

v13 方向（评审指定）：r169 真实滑移测量 + 腕部轨迹保真优化（10% 门不放宽）。
