# GPT review package v7 — side-entry layout, corrected metrics, closed execution gap

Base: `v2d-mac` @ `6b8e8a01` (branch `main`, remote `https://github.com/topprismdata/v2d-mac.git`).
Everything below is reproducible from committed files; the command lines are given so a
reviewer can re-derive every number in a clean process.

---

## 0. What changed since v6 (8.9/10, APPROVE)

v6 scored the *evidence discipline* highly and the *layout* badly: the reviewer saw a
ceiling-mounted arm cluster reaching down at the box while the reference clip shows
two forearms entering horizontally from the left and right frame edges.  This package
closes that gap and, in the process, found and fixed **five measurement or
scene-construction defects** that had been inflating or mis-locating every
"executed-state" number the campaign had published.

| # | defect | how it was found | effect |
|---|---|---|---|
| 1 | `rect_dist_mm` folded a **0.5 m out-of-band penalty** into the distance | a pad 57 mm off the face but above the box top printed as **557 mm** | every published "executed" figure was inflated by exactly 500 mm |
| 2 | the executed face plane used the **fixed ±half_x**, ignoring the box's own translation | the box slides under the hand (16.7 mm measured) | the box's motion was booked as the pad's error |
| 3 | the Allegro hand's collision proxies **overlap the UR5e wrist capsules by 4–10 mm in the neutral pose** (pose-independent) | contact audit at qpos0 | the left arm carried **+188 Nm** of constraint torque permanently, its shoulder_lift servo saturated at −150 Nm, and it sat 0.253 rad off target (155 mm at the tool) |
| 4 | the UR5e **wrist collision capsule was 80 mm in diameter** (real UR5e wrist ≈ 45 mm) | the terminal frame showed the wrist 3.6/3.9 mm inside the table | the table reaction saturated the arm servos; at the true radius the wrist clears the table |
| 5 | the two arms' bases were **not a mirror pair** (right Rz180 / left Rx180) | the arms solved to different postures (shoulder_lift −4.37 vs −1.22 rad) and the hands collided with each other | mirrored bases + the excludes below cut the executed gap from 173 mm to 42.6 mm |

Plus the terminal IK bugs fixed earlier in the session (the `"attachment_site"` name
resolved to −1, so both arms drove the same site; and a world-constant palm−site offset
that breaks at large terminal re-orientation).

---

## 1. The layout, reproduced

Reference frame `runs/w6b/frames_full/f_0252.jpg` (ego-view human clip) vs the new sim
render `docs/replay_frames/front_r0251.png` (same phase).

Measured composition (`viz/composition_target.json`, taken from the human frame):

Measured on the delivered PNGs (the purple skin's pixel extent) against the same
measurement on `runs/w6b/frames_full/f_0252.jpg`:

| quantity | reference (measured) | sim (measured, frames 8/251/340) |
|---|---|---|
| box centre u | 0.464 | 0.499 / 0.495 / 0.504 |
| box centre v | 0.335 | 0.356 / 0.357 / 0.339 |
| box width as a fraction of the frame | 0.315 | 0.331 / 0.331 / 0.330 |
| forearm entry, left | frame edge (0.00) | forearm enters at the left frame edge |
| forearm entry, right | frame edge (1.00) | forearm enters at the right frame edge |
| table front edge in frame | no | no |
| camera | 15–20° down | 14° down, 0.40 m, 45° fov, look-at 60 mm below the box |

The camera was calibrated by rendering and measuring, not by projection arithmetic
(`viz/calibrate_render_camera.py`): MuJoCo's free camera uses the model's `fovy` as
the horizontal scale, which a first-principles projection gets wrong by the aspect
ratio (0.267 predicted vs 0.481 measured for the same frame).

What is *not* matched, stated plainly:
* the reference's forearms are nearly horizontal (≈5–10° below horizontal) because a
  person's shoulders are **below** the camera; the UR5e bases are on posts **0.6 m
  above** the box, so their forearms descend at ≈35–45°.  A table-mounted UR5e cannot
  reproduce a human shoulder height; this is a geometry limit of the hardware, not a
  framing choice.
* in the reference the box's **long axis runs left-right** and the hands press its
  **end** faces (hand span = the box's long axis, 0.24 m).  Our box keeps the long
  axis in depth, so the hands press the long faces 0.12 m apart and the box reads
  narrower.  Turning the box 90° was implemented and measured
  (`grounding/assets/allegro_floating_ur5e_side_r38.xml`): the command side converges
  (0.145 / 0.038 mm) but the hands then **drag the box 280 mm across the table** and
  the executed state degrades to 183 / 199 mm.  It is left in the tree as a recorded
  negative result, not shipped.
* the reference holds the box ~30 mm above the table; ours rests on it.

The *structure* the reviewer complained about — two arms entering from the sides
rather than a cluster reaching from above — is fixed.

The delivered 1170-frame render is `docs/replay_frames/{front,side,close}_r####.png`
(MuJoCo offscreen with the fidelity visual layer).  The previous ceiling-layout
render is retained at `docs/replay_frames/blender/` and is **superseded**.

---

## 2. The executed state, honestly

Side cell `grounding/assets/allegro_floating_ur5e_side_r36.xml`, recipe:

```
W6B_SCENE=ur5e \
W6B_SCENE_XML=grounding/assets/allegro_floating_ur5e_side_r36.xml \
W6B_R23_FLYOVER_MM=250 W6B_R22_MARGIN_MM=8 \
W6B_R39_TGC=1 W6B_R39_HOLD=60 \
runs/.venv-retarget/bin/python retarget/retarget_dual.py
```

| quantity | right | left |
|---|---|---|
| command-side rect distance (kinematic) | 0.214 mm | 0.001 mm |
| **executed-state rect distance** | **6.56 mm** (in band) | **6.26 mm** (in band) |
| executed forward axis (≤ 35°) | 21.4° | 14.9° |
| final joint tracking error | ≤ 0.011 rad | ≤ 0.007 rad |

Journey of that number (all in `runs/w6b/r33_side_mount_campaign.md`):

| step | right | left |
|---|---|---|
| r37 (mirrored bases + wrist-hand excludes) | 43.63 mm | 172.73 mm |
| r40 (same-hand finger excludes) | 18.52 mm | 20.68 mm |
| r41 (real-size wrist proxies) | 4.02 mm | 11.72 mm |
| r42 (face plane follows the box) | 7.69 mm | 8.04 mm |
| r43 (per-side servo gains, left kp 4000) | 6.56 mm | 6.26 mm |

**The gate is `< 5 mm` and both sides are at ≈6.3–6.6 mm: this cell still FAILS
gate1.**  What is left is the two hands squeezing one 150 g box; the arms track their
commands to ≤0.011 rad, so the residual is contact coupling through the box, not servo
error.  Terminal gravity compensation (`W6B_R39_TGC=1`, applied only over the last 60
frames) is what took the droop out; applying it over the whole approach destabilises
the replay (r28: 687 mm) because `qfrc_bias` then carries the Coriolis terms.

Control-law results, all negative and all logged:

| law | result |
|---|---|
| gravity feed-forward on the position target | 108.8 / 179.0 mm, forward axis 26 → 75° |
| integral hold (ki 0.5 / 1.0 / 2.0) | 96.7–228.8 / 154.6–185.2 mm |
| `qfrc_applied = −qfrc_bias` over the whole run | 687.6 / 159.5 mm |
| static gravity bias from `mj_rne` | no-op: `mj_rne` excludes gravity |
| terminal seat refinement (roll so the pad set is coplanar) | works (pad x spread 7.7 → 2.1 mm) but costs the command gate; kept in code, default OFF |

---

## 3. The ceiling master, re-verified and re-corrected

The v6 disclosure (the "R5" number is a command-side kinematic metric) stands, and the
executed figures are now measured with the corrected metric:

```
W6B_SCENE=ur5e W6B_SCENE_XML=grounding/assets/allegro_floating_ur5e_r24_vise.xml \
W6B_R23_FLYOVER_MM=250 W6B_R22_MARGIN_MM=8 \
W6B_R18_NATIVE_TARGET=1 W6B_R26_TRACK_A=1 W6B_R28_FORCE3=1 \
runs/.venv-retarget/bin/python retarget/retarget_dual.py
```

| | command side | executed side (corrected) | executed side (as published in v6) |
|---|---|---|---|
| right | 0.000 mm | 152.68 mm | 640.77 mm |
| left | 0.004 mm | 16.77 mm | 122.04 mm |

The recipe was recovered in this session (`runs/w6b/r31_recipe.json`) and reproduces
0.000 / 0.004 mm both against the committed scenes and against the working tree with
another agent's uncommitted edits.  `R28_FORCE3=1` (or `R22_SAFE_HAND=1`) is required for
the left's 0.004 mm; without it the left sits at 0.668 mm.

---

## 4. What is still open (no spin)

1. **gate1 on the side cell**: 6.56 / 6.26 mm against a `< 5 mm` bar.  The two hands
   trade error through the light box; every mount-angle / gain perturbation moves the
   error from one side to the other rather than removing it.
2. **The command state still has one pad through the face** (right ring finger 7.5 mm).
   The r38 seat refinement removes it but costs the command gate (left cmd rect
   0.001 → 1.913 mm), so the hand cannot be both coplanar and on the face at the
   commanded 26.5° approach angle.  The physically right fix is a perpendicular
   approach, i.e. the terminal orientation target, not another mount tweak.
3. **The forearm entry angle** (35–45° down vs the human's ~5–10°): a hardware
   geometry limit of side-mounted UR5es over a table (see §1).
4. **The box is not lifted**; the reference holds it ~30 mm above the table.

---

## 5. Where to look

| artefact | path |
|---|---|
| side-cell ledger (every step, every number) | `runs/w6b/r33_side_mount_campaign.md` |
| recipe + provenance | `runs/w6b/r31_recipe.json` |
| physics-semantics independent check | `runs/semantics/REPORT.md` |
| lift-line closures (E1/E2/E3) | `runs/w6b/report.json` §`quasistatic_lift_E1`, `e2_reachability`, `e3_contact_modes` |
| composition targets | `viz/composition_target.json`, `viz/camera_calibration.py` |
| side scenes (r34 → r37, each step a separate file) | `grounding/assets/allegro_floating_ur5e_side_r*.xml` |
| new render | `docs/replay_frames/`, `docs/human_vs_sim_w6b.png` |
| v6 review + this package's evidence appendix | `docs/gpt_v6_review.md`, `docs/gpt_v6_package.md` |

### Clean-room verification

Every headline number above is reproduced by one command, from the committed files
only (fresh processes, no session state):

```
runs/.venv-retarget/bin/python tools/w6b/verify_v7.py
```

It re-runs both recipes, compares against the values quoted here, and measures the
delivered render's composition against the reference frame.  Last run: **PASS**
(side cell 0.214/0.001 mm cmd, 6.56/6.26 mm executed, 21.4/14.9 deg; ceiling master
0.000/0.004 mm cmd, 152.68/16.77 mm executed; composition width 0.331 vs 0.315,
u 0.495 vs 0.464, v 0.357 vs 0.335).

Test state: `runs/.venv-mj/bin/python -m pytest tests/test_mjcf_provenance_pins.py
tests/test_scene_w6.py -q` → **16 passed** on the committed tree.  (A working-tree edit
by another agent to `grounding/assets/allegro_floating.xml` makes the pin tests fail
until that edit is either committed or reverted; it is not ours and was left untouched.)
