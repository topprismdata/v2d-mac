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

### Open defects (not claimed as solved)

| # | defect | measured |
|---|---|---|
| 1 | the right's executed forward axis | **45.61°** (the gate's limit is 35°) |
| 2 | the box's displacement over the replay | **16.2 mm** (the r63 item wants ~0) |
| 3 | the executed pad distance | 7.75 mm right / 4.22 mm left (the target is < 5 mm) |
| 4 | the home pose's arm self-collision | the upper arm vs the wrist_1 interpenetrate 79 mm at the home |

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
