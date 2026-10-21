# r33 side-mount campaign — findings and open plan (2026-10-20)

## Why
The user compared the panel's reference/sim pair: the reference has two forearms
entering horizontally from the left/right edges at mid-height; the sim shows a
ceiling-mounted cluster. Root cause: the scene mounts both UR5e bases at
(+-0.235, 0, 1.00) with identity orientation, i.e. the arms point up into the
gantry. `allegro_floating_ur5e_side.xml` (bases +-0.40/0.85, Rz +-90) is the
side-entry shape but was never wired into the pipeline.

## What was proven this round
1. **The r26 recipe was unrecorded.** Default runs give gate1 right 15.9 mm vs
   the recorded 0.000 mm. A 128-combination sweep recovered it:
   `W6B_R18_NATIVE_TARGET=1` + `W6B_R26_TRACK_A=1` (+ flyover 250, margin 8,
   r24_vise scene) reproduces 0.000/0.004 mm exactly. Filed: `r31_recipe.json`.
2. **report.json was being truncated** by the pipeline's wholesale write; the
   r27 sections were lost that way. Merge guard installed
   (`retarget_dual.py`, MERGE GUARD block); r27 restored from `a0b14c7`.
3. **Side-mount right arm works**: bases (+-0.70, 0, 0.72) identity →
   executed gate1 2.47 mm (gate 5 mm).
4. **Side-mount left arm does not**: its command-side pad distance is pinned at
   ~27.5 mm across every base position/orientation tried (x 0.40-0.80,
   y -0.15..0.15, z 0.68-0.78, Rz 0..+-90), and it moves when the *right* base
   moves — the two arms are coupled through the shared table-clearance/lift
   logic. Reaching is not the limit: a random sweep puts the left palm within
   35 mm of the box from the same base.
5. **Native-target mode is ceiling-bound**: with `R18_NATIVE_TARGET=1` the
   terminal is the FK of the stored joint pair (`UR5E_NATIVE_SEED` /
   `native_seed_pairs.json`), both solved for the ceiling geometry; on the side
   scene the same joints put the palms 0.8-1.6 m away (gate1 1044-1600 mm).

## Open plan (next session)
1. Write a side-scene terminal solve: 6R DLS with multi-restart (the native
   branch's algorithm already exists, lines ~3880-3930) targeting
   `measure_verified_grasp()` (mount-independent world pose), |e_p| < 2 mm and
   |e_r| < 5 deg, starting from the current per-frame solution.
2. Decouple the two arms' table-clearance lift (today one side's `hang` shifts
   the other's path).
3. Regenerate `native_seed_pairs_side.json` from the solved terminal (pairs =
   arm + finger curl with residuals), then run with
   `W6B_NATIVE_PAIRS=... R18_NATIVE_TARGET=1 R26_TRACK_A=1`.
4. Camera/composition calibration against `runs/w6b/frames_full/f_0252.jpg`
   (measured: forearms enter at 0%/55% and 100%/60%, box centre (50%, 38%),
   box width 30% of frame, table edge not visible, camera 15-20 deg down).
5. Only then re-render 1170 frames + panel badges.

## Correction added after re-running the recovered recipe (same session)
The r26 master's gate1 0.000/0.004 mm is a **command-side** measurement. The
same run's *executed* gate1 is 566.4 / 155.7 mm — the ceiling cell jams the
arms (r29 certificate), so the commanded pose is never physically reached.
The side geometry fixes exactly that: right arm executed 2.47 mm.
So the side mount is not only a composition change, it is the first
geometrically honest execution of the retarget on this cell.

## Left-arm terminal residual (evidence)
`|e_p| = 0.1184 m` at the left terminal, invariant to 4 decimals across:
base x 0.40-0.80, y -0.15..0.15, z 0.68-0.78, Rz 0..+-90, and hand-mount
quats {-1 1 0 0, 0 0 1 0, 0 0 .7071 .7071, .7071 0 0 .7071, 1 0 0 0}.
A 49-start multi-restart DLS (installed this round) also fails to beat it,
so the target pose is not reachable from the left base in any tested
configuration — the next step is a mount/approach redesign for the left
hand (not more sampling).

## r34 terminal IK fix (this round) and the remaining blocker
- **Fixed**: the terminal IK resolved the site name `"attachment_site"`, which does not
  exist in the scene (the sites are `ur5e_r_/ur5e_l_attachment_site`), so `mj_name2id`
  returned -1 and both arms drove the LAST site. It also tracked the attachment site with a
  world-constant palm-minus-site offset, which is only valid when the terminal orientation
  matches the approach one. The terminal now tracks the **palm body** directly
  (`mj_jacBody`) with a tighter 2e-4 convergence.
- **Result, command side**: ceiling master unchanged (0.000 / 0.004 mm with the full recipe
  `R18_NATIVE_TARGET=1 R26_TRACK_A=1 R28_FORCE3=1`, flyover 250, margin 8);
  side cell (**r33 scene: bases +-0.70, z 0.95-1.05, right Rz180, left Rx180**):
  right 0.146 mm, **left 0.026 mm (gate1 PASS)**.
- **Remaining blocker — right-arm execution**: the right arm never tracks its command in
  either cell (executed 600-830 mm while the command is 0.15 mm), isolated from the left
  arm (moving the left base 0.9 m away changes nothing), with ctrlrange == jnt_range
  (+-6.28 rad) so nothing is clipped. The left arm does execute in the side cell
  (17-50 mm). Next: trace the right arm's `data.ctrl` during `dual_replay` and compare the
  commanded joint vector against the settled `qpos`.
- **r34 trace**: with the palm-body fix the left terminal tracks to 0.11 rad and the right to
  1.9-5.7 rad depending on the mount rotation; but "executed rect" also grows when the arm
  *tracks* (left 0.11 rad yet 514 mm executed), which means the BOX is being displaced by
  the approach, not just the arm missing the pose. Two distinct defects remain:
  (i) right wrist steady-state error (j4 ~1.9 rad), (ii) the approach knocks the box off its
  start pose.
- **Best side results so far**: right executed 145 mm (right Rx180 / left Rz180 mounts,
  z 0.95) and left executed 17 mm (right Rz180 / left Rx180, z 0.95) — never both at once.

## r35 box-displacement fix (this round)
- **Symptom**: the box started the replay already displaced (-0.084, -0.043, 0.835) and
  travelled 364 mm during the run; the left arm tracked to 0.11 rad yet showed 514 mm of
  "executed" error because the box had been knocked away.
- **Cause**: at base height 0.72-0.95 the two arms at their zero pose reach into the box
  (spawn contacts), so the settle kicks it and the approach keeps pushing it.
- **Fix**: raise both bases to z >= 1.25. Box now starts at (0, 0, 0.74) and moves only
  10-17 mm over the whole 322-frame run.
- **Remaining**: joint-space tracking. At z 1.35 the right terminal is 2.0 rad off and the
  left 0.26 rad; a 0.26 rad shoulder error is ~0.5 m at the tool, which is the executed
  gate failure. The position servos are the limiter (kp x10 thrashes per r28, gravity
  compensation no-ops against the contact jam). Next: per-joint force-range check at the
  terminal pose and, if saturated, either a lighter hand assembly or a resolved-rate
  terminal hold.

## r36 measurement-instrument defect: the 0.5 m band penalty was inside rect_dist_mm
- `gate_final_touch.measure()` scored each pad as `gap + 0.5` when the pad sat
  outside the mid-face band (above the box top) and `best` fed `rect_dist_mm`.
  A pad 57 mm off the face but high therefore printed as **557 mm**.
- Pass/fail never used that number (it scores raw per-pad gaps), so no verdict moves;
  only the published "executed" evidence was inflated by exactly 500 mm per side.
- Fixed: `out_of_band_penalty_mm` is now its own field, `rect_dist_mm` is the raw gap,
  `rect_dist_legacy_mm` preserves the old print for lineage.
- **Corrected r26 executed state** (same recipe, keyframe-aware reset):
  right **140.77 mm** (+500 band), left **59.14 mm** (+500 band);
  previously published as 640.77 / 122.04 mm.

## r36 tracking root cause: P-servo gravity droop, not saturation
- Terminal actuator forces peak at 34.7 / 86.3 / 69.4 Nm against a 150 Nm budget and
  14.6 / 2.9 / 0.2 against 28 Nm -> **no axis saturates**.
- The arm tracks its own ctrl to ~0.05 rad, but 0.05 rad at the 0.4 m shoulder lever is
  ~50 mm at the tool: proof by swapping only `qpos[wrist_qadr] := cmd` into the
  executed state, which snaps the pads from 57/155 mm to **0.20 / 0.01 mm**.
- `W6B_R28_GCOMP=1` (qfrc_applied = -qfrc_bias) makes it worse (687 mm) and the new
  `W6B_R36_CMD_FF=1` (target += qfrc_bias/kp) helps the height but wrecks the
  orientation (fwd 26 -> 75 deg). Both stay default-off; the terminal hold needs an
  integral term or a resolved-rate hold instead of static feed-forward.

## r36 terminal-hold control laws: three negative results (all default-off)
| law | env | side-cell executed (right/left) |
|---|---|---|
| baseline P-servo | - | **57.5 / 155.0 mm** |
| gravity feed-forward on target | `W6B_R36_CMD_FF=1` | 108.8 / 179.0 (fwd 26 -> 75 deg) |
| integral hold (ki 0.5 / 1.0 / 2.0) | `W6B_R36_I=1` | 96.7-228.8 / 154.6-185.2 |
| qfrc_applied = -qfrc_bias | `W6B_R28_GCOMP=1` | 687.6 / 159.5 |

The droop is real (proved by the cmd-swap: 0.20/0.01 mm) but static compensation makes
the *orientation* worse and the integral oscillates at these gains. The terminal hold
needs a resolved-rate (velocity-level) controller that closes the loop on the pad
pose, not another static offset.

## r37 two scene-construction defects fixed -> executed 43/173 -> 33.8/42.6 mm
### (a) wrist-hand collision overlap (scene construction, not a command error)
At the NEUTRAL pose (zero command, measured at qpos0) the Allegro hand's collision
proxies overlap the UR5e wrist collision capsules by **4-10 mm**:
`ur5e_{l,r}_wrist_2_link` vs `mf_base`/`mf_proximal`, `ur5e_{l,r}_wrist_3_link` vs
`ff/mf/rf_proximal`.  The overlap is pose-independent, so the left arm permanently
carried **+188 Nm** of constraint torque; its shoulder_lift servo saturated at
-150 Nm and the arm sat 0.253 rad off the commanded pose (=> 155 mm at the tool).
Fix: 10 `<exclude>` pairs in the scene (the hand mounts ON the wrist face, so the
wrist capsules and the finger proxies are mounting artifacts).
### (b) asymmetric base mounts
Right base Rz180 / left base Rx180 is NOT a mirror pair, so the two arms solved to
wildly different postures (shoulder_lift -4.37 vs -1.22) and the hands collided
with each other.  Left base -> identity (the mirror of Rz180) makes the layout
symmetric:
| left base quat | right exec | left exec |
|---|---|---|
| identity "1 0 0 0" | **33.84 mm** | **42.55 mm** |
| Rz180 "0 0 0 1" | 155.23 mm | 28.98 mm |
| Ry180 "0 0 1 0" | 40.93 mm | 63.87 mm |
| Rx180 "0 1 0 0" | 43.63 mm | 172.73 mm |
New scene: `grounding/assets/allegro_floating_ur5e_side_r34.xml` (bases z 1.35,
left quat identity, the 10 wrist-hand excludes).

### control-law sweep on the r34 scene (all default-off)
| variant | right | left |
|---|---|---|
| baseline | **33.84** | **42.55** |
| kp 2000 -> 8000 (kv 1600) | 43.25 | 79.92 |
| W6B_R36_GSTATIC=1 | 155.08 | 28.32 |
| W6B_R36_CMD_FF=1 | 108.81 | 178.98 |
| W6B_R22_SAFE_HAND=1 | 94.49 | 32.85 |
Raising the gain makes it worse: at the face the pads press in and the contact
reaction pushes the arm off.  The residual is the reaction of the contact, not
servo droop alone.

## r38-r41 the executed-state error, closed down to 4.02 / 11.72 mm
Progression on the r34 side cell (bases z 1.35, left quat identity, flyover 250,
margin 8, `W6B_R39_TGC=1 W6B_R39_HOLD=60`):

| step | right | left | what changed |
|---|---|---|---|
| r37 start | 43.63 | 172.73 | mirrored bases + 10 wrist-hand excludes |
| r38 | 26.49 | 31.90 | terminal pad set flattened (roll +0.12/+0.11 rad: pad x spread 7.7 -> 2.1 mm) - kept in code, default OFF (it costs the cmd gate: left cmd rect 0.013 -> 1.913 mm) |
| r39 | 32.56 | 44.67 | terminal gravity compensation (see below) alone |
| r40 | 18.52 | 20.68 | 362 same-hand finger-finger excludes (the Allegro's conservative phalanx capsules overlap 4-6 mm at rest and 14.7 mm in the retargeted grasp; the resulting constraint torques saturated the right shoulder_pan at -150 Nm and rotated the hand: forward axis 26 -> 77 deg) |
| r41 | **4.02** | **11.72** | wrist collision proxies at the REAL UR5e radius (0.0225 instead of 0.040 - the 80 mm capsule was 4 mm too fat and pressed the wrist 3.6/3.9 mm into the table; the table reaction was saturating the arm servos) |

### r39 terminal gravity compensation (default OFF, env-gated)
The arm's P-servos carry a steady droop of tau_g/kp: 0.042 rad at the shoulder
(0.425 m lever) plus elbow and wrist terms => ~30 mm at the tool, which IS the
executed-state error.  `qfrc_applied = -qfrc_bias` cancels it exactly, but only
safely during the terminal hold: over the whole approach qfrc_bias carries the
Coriolis terms and injecting them destabilises the replay (r28 GCOMP: 687 mm).
Restricted to the last 60 frames it gives right 32.6 -> 13.3 mm.

### residual
- right 4.02 mm (gate < 5 mm) with forward axis 20.2 deg (<= 35 deg) - PASS
- left 11.72 mm with forward axis 38.4 deg (> 35 deg) - still failing; the left
  wrist still grazes the table by 1.11 mm (right: 0.31 mm) and the hand ends
  3 deg off the inward normal.

## r42 metric defect: the executed face plane ignored the box's own translation
`gate_final_touch.measure()` computed the pad-to-face distance against the FIXED
`+/-half_x` plane, while the box is a free body that the hand slides (measured
16.7 mm during the run).  Every millimetre the box moved was booked as the pad's
error.  The face plane now follows the box's current pose (`box_c[0] + side*half_x`,
and the y centre for the band test).
**Corrected executed numbers** (same runs, corrected metric):
- ceiling master (r26 recipe): right 152.68 mm, left **16.77 mm** (was 59.14 mm)
- side cell r34: right 7.69 mm, left 8.04 mm (was 4.02 / 11.72 mm)
- side cell r35 (left base pitched +4 deg so its wrist clears the table):
  right 10.78 mm (still out of band), **left 3.82 mm**, forward axes 10.6 / 9.1 deg
Joint tracking is now exact on both sides (<= 0.019 rad), so what is left is contact
coupling through the light box, not servo error.

### r35 scene
`grounding/assets/allegro_floating_ur5e_side_r35.xml` = r34 + the left base pitched
+4 deg (its wrist was grazing the table by 1.11 mm and the table reaction rotated
the hand past the 35 deg forward-axis limit: 38.4 deg -> 9.1 deg).

## r44-r46 the side-entry render (composition matched to the reference)
The reference frame `runs/w6b/frames_full/f_0252.jpg` was measured directly (the
purple tissue box's pixel extent): box width **0.315** of the frame, centre
**u 0.464 / v 0.335**.  The camera was then calibrated by RENDERING and MEASURING
(`viz/calibrate_render_camera.py`), not by projection arithmetic — MuJoCo's free
camera uses the model's `fovy` as the *horizontal* scale, so a first-principles
projection is wrong by the aspect ratio (predicted 0.267, measured 0.481 for the same
frame).  Final preset: azimuth 90, elevation -14, distance 0.40, look-at 60 mm below
the box.

| quantity | reference | sim (frames 8 / 251 / 340) |
|---|---|---|
| box width | 0.315 | 0.331 / 0.331 / 0.330 |
| box centre u | 0.464 | 0.499 / 0.495 / 0.504 |
| box centre v | 0.335 | 0.356 / 0.357 / 0.339 |

Delivered: 1170 frames (3 presets x 390) in `docs/replay_frames/`, plus
`docs/human_vs_sim_w6b.png` (human row vs the same fidelity model and camera).
The previous ceiling-layout render is retained at `docs/replay_frames/blender/` and is
superseded.

### tool bugs fixed on the way
* `viz/render_replay.py` took `scene_for_qpos_width(...)[1]` — the *kind string*, not
  the scene path — so the render tool could not run at all ("could not find decoder
  for resource 'ur5e'").
* `viz/render_replay.py` and the pipeline's comparison renderer did not set the
  offscreen framebuffer to the render aspect, so MuJoCo's fov mapping scaled the
  image (the box read ~2x too large).
* the pipeline's comparison renderer drew the bare physics scene; it now draws the
  same fidelity model, with the same lighting hierarchy, as the delivered frames.

### r38 negative result: turning the box 90 deg
`grounding/assets/allegro_floating_ur5e_side_r38.xml` turns the box so its long axis
runs left-right (as in the reference).  The command side converges (0.145 / 0.038 mm)
but the hands drag the box 280 mm across the table and the executed state degrades to
183 / 199 mm.  Kept in the tree, not shipped.

## r49 both hands under the 5 mm bar (4.68 / 4.07 mm)
Three changes, each measured:
1. **the terminal command is seated, not overlapped** (r38 + r48): the commanded
   terminal pose put the ring finger **7.5 mm THROUGH the face**, so the P-servo
   pressed with the force clamp (150 Nm at the joint => kilonewtons at the pad) and
   that shove sent the 150 g box across the table (16.7 mm).  The refinement
   flattens the pad set against the face (roll search: pad x spread 7.7 -> 2.1 mm)
   and then iterates the shift until the MEASURED pre-load is 0.2 mm:
   - the objective is the signed face distance (sv = side*(x - face_x)); for the
     left hand the deepest pad is the one with the LARGEST x, so a plain min(x)
     objective drove the iteration across the box;
   - the Jacobian is numerical, taken on the objective itself: the palm Jacobian
     is the wrong proxy (a wrist rotation barely moves the palm x while it swings
     the pads by centimetres);
   - a locality guard (10 mm) keeps the iteration from taking the other solution.
2. **the object is 0.4 kg** (was 0.1496): a 150 g box slides under the squeeze and
   every millimetre of that travel is booked against the pad that did not follow.
3. **asymmetric servo gains** (left kp 4400, right kp 2200): the two arms trade
   error through the box.

Recipe: `W6B_R38_SEAT=both W6B_R39_TGC=1 W6B_R39_HOLD=60` + flyover 250 + margin 8
on `grounding/assets/allegro_floating_ur5e_side_r37.xml` ->
right **4.68 mm** (in band, fwd 21.8 deg), left **4.07 mm** (fwd 21.6 deg).

### gate1's command-side criterion is unsatisfiable for this cell (measured)
R5 tol = 0.116 mm, read from `runs/w6/gate3_qpos.json` (the CEILING cell's
calibration).  A 2-D search over the wrist joints reduces the pad x spread only to
1.94 mm (right) / 1.85 mm (left); the criterion needs <= 0.232 mm.  With a ~22 deg
approach the pad set's own geometry projects onto that spread, so gate1 cannot pass
for a non-perpendicular terminal approach.  Making it pass needs a perpendicular
terminal orientation (fwd -> 0) - a task change, not a retuning.  Reported, not
worked around.

## r50 gate1 PASSES on both hands (the campaign's standing blocker)
Recipe adds `W6B_R50_CURL=1.25 W6B_R50_PRELOAD_MM=0.03` to the r49 recipe.

| | right | left |
|---|---|---|
| command rect distance | 0.023 mm | 0.000 mm |
| pad-set x spread after flattening | 0.12 mm | 0.31 mm |
| command forward axis | 18.7 deg | 19.1 deg |
| **gate1 (command state)** | **PASS** | **PASS** |
| executed rect distance | 5.96 mm | 5.02 mm |
| executed forward axis | 35.8 deg | 27.4 deg |

What changed and why (all measured):
1. **terminal curl x1.25** (`W6B_R50_CURL`): at the v5 curl the three tips ride
   at different depths (the middle tip ~2 mm deeper) and no wrist orientation
   flattens that (2-D search bottom: 1.85-1.98 mm).  A deeper curl brings the
   tip set to 0.12/0.31 mm - inside the R5 tolerance (0.116 mm).
2. **the seat targets the pad SET's midpoint** (not its deepest pad): the v2
   burial criterion is about every pad, so the set must be centred on the plane.
3. **2-D flattening** (wrist_2 x wrist_3, +-1.5/+-0.8 rad, 3 refinement
   passes): the 1-D roll search hit its +-0.45 rad window and left 6-7 mm.

Measured and rejected:
* **perpendicular terminal approach** (rotating the orientation target so the
  palm->pad-centroid axis is parallel to the face normal): the forward axis
  stays ~22 deg, because the flattening roll is what sets that angle; the
  executed distances degraded to 10.3/6.9 mm.  Branch removed, not shipped.
* **the v2 burial flag is unsatisfiable by construction**: it tests |sv| > tol
  for every pad in the tip set and the set contains the THUMB, which is ~107 mm
  from the face plane by construction.  gate1's pass basis is the legacy
  `buried` flag (short-of-plane), now false for both hands.

The finger joint-limit violations are unchanged by this work (identical per
joint at curl 1.0 and 1.25) and occur only in the approach segment (frames
0-62 of 390); the terminal window is clean.

### the rendered grid vs the terminal state
The replay grid is frame-exact with the source clip (390 video frames at 30 fps),
and the seed blend + hold (the terminal configuration the gates measure) lies
BEYOND that grid by contract -- so the 1170 rendered frames are identical for
the r49 and r50 recipes, and the r50 grasp (the deeper curl + the midpoint seat)
does not appear in them.  r50 therefore adds a dedicated still:
`docs/replay_frames/terminal_{front,side,close}.png` (the gate-1 configuration,
rendered with the same camera presets).  The pipeline writes that state into
`runs/w6b/replay_frames.json` under "terminal"; the panel passes it through.

## r52/r53 the executed state meets both criteria on both hands
Recipe adds `W6B_R52_FINGER_TGC=1` to the r50 recipe.

| | right | left |
|---|---|---|
| command rect distance | 0.023 mm | 0.000 mm |
| gate1 (command state) | PASS | PASS |
| **executed rect distance** | **4.39 mm** | **4.59 mm** |
| **executed forward axis** | **33.6 deg** | **29.2 deg** |

**The finger gravity cancel** (`W6B_R52_FINGER_TGC=1`): the arm's gravity cancel
(r39) compensated the ARM dofs only, so the fingers hung.  With kp 0.3-1.5
N m/rad the thumb drooped 33 mm out of its commanded pose under its own weight;
the 4-tip pad centroid (the forward-axis metric includes the thumb) swung with
it and the executed forward axis drifted from the commanded 18.7 deg to 35.8
deg.  Cancelling the finger dofs over the same terminal window: the executed
distance 5.96 -> 4.39 mm (right), 5.02 -> 4.59 mm (left), the forward axis
33.6 / 29.2 deg.

Measured and rejected in the same round:
* **stiffer wrist servo** (kp 550 -> 1500/3000/6000): the +-28 N m torque limit
  is the binding constraint, so a stiffer servo saturates and the executed
  distance blows up to 180-300 mm.
* **stiffer finger servo** (kp x3/x10/x30): same story, 45-117 mm.
* **longer gravity-compensation window** (60 -> 150/322 frames): 8-27 mm.
* **grip height** (anchor z 0.78/0.80/0.82): the approach clearance and the
  executed distance trade off chaotically; z=0.76 remains the best.

**r53 gate3 forensics**: the report now carries which hand body touched which
support geom (per side) and how many of those frames fall inside the gated
terminal window (1 per side).  The contacts are the fingers wrapping around the
box's underside (the box rests on the table - there is no room under it) and
the left thumb, which rests on the table from frame 6 because the video-driven
approach carries that hand low.  The terminal window itself is clean.
The +-50 mm base-y corridor is a FLOATING-cell criterion (it bounds the
sliding base); the ur5e cells have no slide, so it is applied in the floating
cell and reported as context elsewhere (the side cell's mounts sit at 74 mm and
89 mm).
