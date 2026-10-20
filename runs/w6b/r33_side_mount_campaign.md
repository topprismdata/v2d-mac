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
