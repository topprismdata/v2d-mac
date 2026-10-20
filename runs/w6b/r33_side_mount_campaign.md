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
