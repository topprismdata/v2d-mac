# Wave 3 MJCF revision — accounting

Every number here is measured from `grounding/assets/allegro_floating.xml` by
`grounding/assets/model_stats.py`, or by a function in
`grounding/assets/build_allegro_floating.py` that simulates the scene. Nothing is
typed in by hand: the generator recomputes the tables on every run and the XML is
rebuilt from them, so this document and the XML cannot drift apart silently.

Rebuild + re-verify:

```
python grounding/assets/build_allegro_floating.py     # regenerates the XML
python grounding/assets/model_stats.py                # dumps the numbers below
pytest tests/test_smoke6_mjcf.py                      # the gates
```

---

## R1 — finger `forcerange` faithful to the Allegro V5's 10 N/fingertip

**Defect.** Every one of the 16 finger actuators carried a flat
`forcerange="-2.0 2.0"`, a literal in the generator. smoke4b measured the
consequence: a **37.7 N mean fingertip force**. The real Wonik Allegro V5 is
spec'd at **10 N per fingertip**, so the sim hand was ~3.8× too strong and the
policy was being rewarded for a capability the hardware does not have — it learned
to win by squeezing rather than by grasping.

**Derivation.** The torque limit of a joint is whatever converts into 10 N through
that joint's own **moment arm** — the perpendicular distance from the joint axis to
the fingertip:

```
τ_j = share_j × 10 N × arm_j
```

`arm_j` is the **minimum** over the reachable joint space, because `τ/arm` grows as
the arm shrinks and the smallest arm is where a given torque does the most damage.
It is measured from the MJCF's own geometry (12 joint axes × the four fingertip
capsule centres, over 120 000 sampled configurations, fixed seed `20251015`).

**Why `share_j = 1/4`.** 10 N is a budget for a whole *fingertip*, but all four
joints of a finger press the same pad and their contributions add:

```
F_tip = |Σ_j τ_j / arm_j|  ≤  Σ_j τ_j / arm_j
```

An early pass of this revision gave each joint the full 10 N. That passed a naive
per-joint check — each joint's own `τ/arm` is 10 N — while letting one fingertip
reach **40 N** in aggregate. It was caught by summing the four contributions over
the reachable space: **11.0 / 10.8 / 10.6 / 12.1 N** for ff / mf / rf / th against a
10 N spec. Any weights summing to 1 satisfy the bound; equal is the simplest and
gives no joint a privileged share of the real hand's budget.

**Poses excluded from the arm minimum** (both are not grasp poses):

* any configuration where a finger link is already touching the palm — the finger is
  folded into itself and its tip is not pressing anything external;
* any configuration where a joint axis passes within the fingertip capsule's own
  radius (12 mm) of the fingertip centre — the axis pierces the pad, so the pad
  cannot be transmitting torque about that axis at all. Without this exclusion the
  minimum arm collapses toward zero and the table is meaningless.

### Per-joint table

| joint | moment arm (mm) | forcerange (N·m) | own share of tip force (N) | kp | gravity torque (N·m) | I_eff (kg·m²) | ω·dt |
|---|---|---|---|---|---|---|---|
| `ffj0` | 12.00 | ±0.0300 | 2.50 | 0.3261 | 0.0163 | 5.85e-06 | 0.472 |
| `ffj1` | 42.16 | ±0.1054 | 2.50 | 0.7269 | 0.0363 | 1.28e-04 | 0.151 |
| `ffj2` | 37.48 | ±0.0937 | 2.50 | 0.2348 | 0.0117 | 3.93e-05 | 0.155 |
| `ffj3` | 16.88 | ±0.0422 | 2.50 | 0.0449 | 0.0022 | 5.65e-06 | 0.178 |
| `mfj0` | 12.00 | ±0.0300 | 2.50 | 0.3278 | 0.0164 | 5.86e-06 | 0.473 |
| `mfj1` | 41.55 | ±0.1039 | 2.50 | 0.7269 | 0.0363 | 1.28e-04 | 0.151 |
| `mfj2` | 37.60 | ±0.0940 | 2.50 | 0.2348 | 0.0117 | 3.93e-05 | 0.155 |
| `mfj3` | 16.94 | ±0.0423 | 2.50 | 0.0449 | 0.0022 | 5.65e-06 | 0.178 |
| `rfj0` | 12.00 | ±0.0300 | 2.50 | 0.3285 | 0.0164 | 5.86e-06 | 0.473 |
| `rfj1` | 42.35 | ±0.1059 | 2.50 | 0.7266 | 0.0363 | 1.28e-04 | 0.151 |
| `rfj2` | 37.53 | ±0.0938 | 2.50 | 0.2348 | 0.0117 | 3.93e-05 | 0.155 |
| `rfj3` | 16.88 | ±0.0422 | 2.50 | 0.0449 | 0.0022 | 5.65e-06 | 0.178 |
| `thj0` | 39.73 | ±0.0993 | 2.50 | 1.0423 | 0.0521 | 1.41e-04 | 0.172 |
| `thj1` | 12.07 | ±0.0302 | 2.50 | 0.4073 | 0.0204 | 4.66e-06 | 0.591 |
| `thj2` | 20.80 | ±0.0520 | 2.50 | 0.4186 | 0.0209 | 8.77e-05 | 0.138 |
| `thj3` | 12.01 | ±0.0300 | 2.50 | 0.0968 | 0.0048 | 1.73e-05 | 0.149 |

Four joints × 2.50 N = **10.00 N at each fingertip**, exactly the spec.

### Measured, before vs after — and the two regimes that have to be separated

Same probe, same 200 sampled grasp poses, same seed, old MJCF vs new:

| | old (±2 N·m, default solref) | new (10 N/finger, 4 ms solref) |
|---|---|---|
| **surface contact** (pen ≤ R5 tolerance) — the fingertip force | max **4.22 N**, mean 0.69 N | max **7.68 N**, mean 1.42 N |
| buried geom (pen > tolerance) | max 10.5 N, max pen 2.5 mm | max 947 N, max pen 53.4 mm |

The surface-contact row is the R1 claim, and it is met. But the honest reading is
that **this probe does not reproduce smoke4b's 37.7 N at all** — the old model also
tops out around 4 N here. The probe therefore confirms the fingertip force is under
the spec; it does **not** independently re-derive the 37.7 N evidence, and no claim
is made that it does. What it does establish is that the new limits are not looser
than the old ones anywhere.

The buried row is a **finding, not a regression to paper over**: R2's extra 63 mm of
`wrist_z` travel makes it reachable to command a finger *link* into a pose that
intersects the box by tens of millimetres. MuJoCo resolves that overlap with the
constraint solver, whose force follows the overlap depth and not any actuator's
authority — at 53 mm it reports ~950 N. Every actuator is inside its forcerange
throughout (verified: the finger forces sit at their ±τ limits, and the wrist is at
6.3 N of its 45 N). So this number says nothing about hand strength, which is why
the test reports it instead of gating on it, and why it is listed under known
limits. It is a property of the scene's reachability, reachable only by commanding
an interpenetrating pose; a policy rewarded for lifting has no incentive to find it,
but the re-verification round should watch for it.

### Why kp is set where it is

`kp` is not a tuning constant; it is pinned between two measured lower bounds and
one measured upper bound, and the generator picks the value in that window:

| constraint | formula | why |
|---|---|---|
| **reach** | `kp ≥ τ_j / joint_span` | servo torque is `kp·(ctrl − qpos)`; the largest error the ctrlrange can command is the joint's own stroke. Below this the actuator saturates *below* its declared `forcerange` and the forcerange is fiction the policy can never use. |
| **hold** | `kp ≥ gravity_torque / 0.05 rad` | otherwise the finger cannot hold its own weight within a 0.05 rad (2.9°) sag budget. |
| **stable** | `kp ≤ I_eff / dt²` | `sqrt(kp/I)` is the servo's natural frequency; above ~1 rad per timestep the 2 ms integration rings. `I_eff` is the **minimum** over the workspace, so this is the worst case. |

The binding column per joint: `reach` for `ffj1`/`mfj1`/`rfj1` and `thj0`
(I_eff ≈ 1.3e-4, gravity 0.036–0.052 N·m), `hold` for the `j0` bases, `reach` again
for the light `j3` links. Every resulting `ω·dt` lands between 0.14 and 0.59, i.e.
the servos are stable with margin and none is pinned at the stability edge.

`kp` is rounded **up** to the written 4-decimal precision, never to nearest. On most
joints `kp_hold` is the binding bound, and round-to-nearest lands a few times 1e-5
*below* it — enough to fail the very gravity-holding guarantee the table above
claims, by a margin too small to notice in the printed number.

`wrist_z` / `wrist_ry` travel is likewise rounded **up** past the measured residual
(0.062088 m → 0.063 m, 0.058449 rad → 0.059 rad). A low end at exactly
`-0.162000` still clips smoke5's recorded command minimum of `-0.162088`, so the
axis would read as out-of-range on replay even though the nominal 0.062 had been
"added".

Upstream's `<position kp="1"/>` default is deliberately **left alone**: every finger
actuator now writes its own `kp`, so the class default never applies to the hand.
Only the joint `damping` default is still patched (0.1 → 0.5), as before.

### Bug found and fixed while writing this: actuator name aliasing

Upstream names actuators after the joint's *role*, not the joint — `ffj0`'s
actuator is `ffa0`. `mj_name2id(model, ACTUATOR, "ffj0")` returns **-1**, and
indexing a NumPy array with -1 returns the **last** element. A forcerange check
written that way reads `tha3`'s numbers sixteen times and passes regardless of what
the other fifteen joints are set to — the R1 gate would have been vacuous. Fixed via
an explicit `FINGER_ACTUATOR_OF` map plus
`test_r1_actuator_lookup_is_not_silently_aliasing`, which fails if two joints ever
resolve to the same actuator again.

---

## R2 — wrist travel, widened only where smoke5 measured a residual

| axis | before | after | added | source |
|---|---|---|---|---|
| `wrist_z` | `-0.10 … 0.16` | `-0.163 … 0.16` | **+0.063 m** | smoke5 `overshoot_after[2]` = 0.062088 m |
| `wrist_ry` | `-1.20 … 1.20` | `-1.259 … 1.20` | **+0.059 rad** | smoke5 `overshoot_after[4]` = 0.058449 rad |
| `wrist_x` | `-0.18 … 0.18` | unchanged | 0 | residual was 0 |
| `wrist_y` | `-0.18 … 0.18` | unchanged | 0 | residual was 0 |
| `wrist_rx` | `-1.20 … 1.20` | unchanged | 0 | residual was 0 |
| `wrist_rz` | `-1.57 … 1.57` | unchanged | 0 | residual was 0 |

Both additions are at the **low** end, which is the physical direction: `wrist_z`
descends toward the box, `wrist_ry` pitches. The residual is what smoke5's gauge
fixing left over after it absorbed the hand's arbitrary yaw and in-plane offset —
those are coordinate conventions, whereas descent and pitch are real motion, so
widening travel is the honest fix. The high ends are untouched.

`test_r2_commanded_smoke5_trajectory_now_fits` replays smoke5's actual recorded
command minima against the new ctrlrange: every axis now lands inside.

---

## R3 — contact stiffness

**Defect.** MuJoCo's default `solref` (timeconst 20 ms) let a fingertip press the
rigid paper box **2.7 mm** deep — smoke5's `penetration/max_m` = 0.002724 m,
which failed the gate.

**Change.** `solref="0.004 1"` (4 ms, critically damped) and
`solimp="0.95 0.99 0.001 0.5 2"` on the box and the four fingertip pads. Measured
under a 10 N load — the most the revised hand can ever apply, i.e. the worst case
the R1 budget permits — the deepest box-vs-support penetration drops from
**0.556 mm → 0.213 mm**, inside the 1 mm requirement, and the settled box's
position jitter is **0** (σ < 1e-6 m).

**Scope, and a defect this revision's own first pass introduced.** The stiffness was
initially stamped on *every* colliding hand geom. That governed the box↔finger
pairs as intended, but it also stiffened the **palm-vs-pedestal** contact — a
pre-existing near-tangential graze in this scene, where the palm's collision box
comes within 0.1 mm of the pedestal top at the bottom of the new `wrist_z` range.
Turning that graze into an incompressible constraint raised its peak force from
**3.2 kN to 77 kN, a 24× amplification**, on a contact R3 says nothing about. The
stiffening is now written to the box and the four fingertip pads only; MuJoCo takes
the stiffer of a pair, so every box↔finger contact is still governed.
`test_r3_did_not_stiffen_the_palm_into_a_force_spike` pins that scope.

---

## R4 — every finger and palm geom is named

**Defect.** All **42** hand geoms were produced by unnamed `<default>` classes, so
`mj_id2name` returned `None` for every one of them. smoke5's contact detector
classifies by name, so it skipped all 42 and reported a structural **zero contact**
— a false conclusion about the physics, produced by a naming gap.

Now `palm_visual`, `palm_collision`, `ff_base_visual`, `ff_base_collision`,
`ff_tip_visual`, `ff_tip_collision`, … all 42, unique and resolvable by name with
no ancestry fallback. `model_stats.py` reports `hand_geoms: 42`,
`hand_geoms_unnamed: 0`.

Naming lives in the generator (`_name_hand_geoms`), not in the XML, so a rebuild
cannot silently drop it.

---

## R5 — penetration tolerance derived from the measured resting sink

**Defect.** The tolerance was a hardcoded `PENETRATION_TOLERANCE_M = 1e-3`,
written next to a measurement and then left alone through two rounds of contact
changes that moved the baseline underneath it.

**Change.** `measure_resting_sink()` settles the box on the pedestal for 900 steps
and returns the deepest box-vs-support penetration — MuJoCo's own numerical
compliance *in this scene*. The tolerance is then

```
tolerance = PENETRATION_TOLERANCE_K × resting_sink
```

| quantity | before | after |
|---|---|---|
| resting sink | 2.39e-4 m | **2.909e-05 m** (R3 made it stiffer) |
| tolerance | 1e-3 m, hardcoded | **1.164e-04 m**, `k = 4` |

`k = 4 ≥ 3` by contract: the tolerance sits clear of the solver's own resting
tolerance, and at 1.16e-4 m is still ~2.7 orders of magnitude below the box's 40 mm
half-height, so a real interpenetration cannot hide under it.

The two tests that matter are the negative ones:
`test_r5_resting_sink_is_measured_not_assumed` and
`test_r5_tolerance_follows_a_stiffness_change` both perturb `solref` and require the
number to move — a function that returned a constant would fail both.

---

## Frozen, and verified unchanged

Box 0.24 × 0.12 × 0.08 m and 0.15 kg; pedestal 60 mm; 20 Hz control; 22-D action
space of `[Δpose(6) + targets(16)]`; `wrist_rz` / `x` / `y` travel; all A1–A6
thresholds in `tests/test_smoke4_accept.py` and `tests/test_smoke4_reward.py`; and
`retarget/` + `tests/test_smoke5_retarget.py`, which the re-verification round
re-runs untouched.

**No training, no PPO run, and no acceptance-gate number changed in this round.** The
MJCF's physics changed, so smoke4 must retrain from scratch — the old checkpoint is
void — and smoke5 must re-run. Both are the next round's job.

## Known limits of this revision

* The palm-vs-pedestal graze at the bottom of the new `wrist_z` range (0.1 mm
  clearance) is pre-existing and left as-is; R3 deliberately does not stiffen it,
  but a policy that drives the hand fully down will feel it.
* R2's deeper `wrist_z` travel makes a commanded interpenetrating finger pose
  reachable, and MuJoCo resolves the resulting overlap with solver forces up to
  ~950 N at 53 mm of burial. Not actuator-driven, so it is not a hand-strength
  statement, and the R1 test gates the surface-contact regime instead. If the
  re-verification round sees the box launched, this is the first thing to check.
* The 12 mm moment-arm floor for `ffj0`/`mfj0`/`rfj0`/`thj1`/`thj3` is set by the
  fingertip capsule radius exclusion, not by the mechanism reaching a true
  singularity. Those five joints therefore sit at the conservative end of their
  range; the real hand's own tip-pad radius would set the same floor.
* kp is checked against `I_eff` minima sampled from the joint space, not against a
  contact-loaded configuration, where the effective inertia of a pressed finger is
  lower and the true bandwidth requirement slightly higher.
