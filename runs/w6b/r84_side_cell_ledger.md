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
