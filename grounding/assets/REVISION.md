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

## Wave 3.5-B — explicit palm↔box contact pair

**Defect.** R3's scope fix deliberately left the palm's collision box unstamped,
so an implicit palm-vs-box pair *mixed* two sides: MuJoCo averages positive
`solref` across a pair, so the palm's default 20 ms and the box's 4 ms became a
12 ms blend. Measured (`runs/w35/palm_probe.py` → `runs/w35/palm_probe.json`): a
10 N palm press sank **0.359 mm** into the box — 1.7× the fingertip budget at the
same load, and 5× the R5 tolerance. The box's floor under the palm was softer than
the same floor under a fingertip.

**Change.** One explicit pair in the generated XML, emitted by the generator
(`PALM_BOX_PAIR`), not hand-written:

```
<pair geom1="palm_collision" geom2="object_geom"
  solref="0.004 1" solimp="0.95 0.99 0.001 0.5 2" condim="4"/>
```

Measured on the regenerated XML, 10 N palm press: **0.359 mm → 0.035 mm (10.4×)**,
now 3.4× *inside* the R5 tolerance.

**Why this cannot repeat R3's 77 kN blow-up — scope, by construction.** R3's
amplification came from stamping stiffness on a *side* (every colliding hand
geom), which silently re-governed every pair that side belonged to — including the
palm-vs-pedestal graze nobody was measuring. An explicit `<pair>` names exactly
one geom pair; MuJoCo uses it for that pair alone and every other pair keeps its
implicit mixing. The probe's P2 guard drives the palm onto the bare pedestal with
the box disabled: peak force **14.694 N before and after, bit-identical**
(`palm_probe.json`, both scenes) — vs 77 kN when R3 first overreached. The stiff
side of the new pair (`palm_collision`) is also the same geom the graze touches,
which is precisely why the pair — not a side-wide stamp — is the correct tool:
the graze pair (palm↔pedestal) is untouched.

The probe script itself moved with the baseline: its "stiff" variant is now the
committed XML verbatim, and its "soft" variant strips the pair line — so re-running
it against any future rebuild still measures both scenes.

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
  **Wave 3.5-E moved this number on the proximal links**: a *commanded* burial of
  one of the now-stiffened proximal links resolves at **9.93 kN at 20.8 mm**
  (`ff_proximal_collision`, measured by the same R1 probe that measures the grip
  force), against 9 N at 0.6 mm for the soft pair. Still not actuator-driven — a
  finger's entire budget is 0.27 N·m — so it remains a solver-conditioning fact
  about commanded interpenetration and not a strength claim, but a policy that
  commands a link into the box now gets ~1000x harder push-back than before. The
  `tests/test_smoke6_mjcf.py` sanity bound moved 5e3 → 2e4 N to record it.
* The 12 mm moment-arm floor for `ffj0`/`mfj0`/`rfj0`/`thj1`/`thj3` is set by the
  fingertip capsule radius exclusion, not by the mechanism reaching a true
  singularity. Those five joints therefore sit at the conservative end of their
  range; the real hand's own tip-pad radius would set the same floor.
* kp is checked against `I_eff` minima sampled from the joint space, not against a
  contact-loaded configuration, where the effective inertia of a pressed finger is
  lower and the true bandwidth requirement slightly higher.

## Wave 3.5-C（2026-10-15）：腕部俯仰行程 ±1.2 → ±1.8 rad——必要不充分

- 依据：rx 降指必翻（三长指高度差 47/65/78mm）、rz 保平不能降（runs/w35/verdict.json）；R2 同原则（行程=建模选择，须覆盖任务工作空间）
- **效果（诚实记账）**：拇指下探可行位姿 0% → 1279/53,624（0.70%，随 +rx 单调上升）；联合指标 13.702mm → **0.089mm**（154×）
- **K1 双侧接触仍为 0**（263 万位姿全局 + 50 万局部致密 + rx=±2.2/2.6 外推）：拇指入盒顶下的位姿族与长指触壁位姿族**互斥**（互斥间隙 ≥14.5mm 局部 / ≥27.6mm 全局）；外推 rx 恶化（joint 指标 14.3→20.2mm）→ 残余是**转轴几何耦合**，非行程量问题
- 新 XML sha256 `dbba3d70…`；证据 runs/w35/rx_widen_scan / rx_coupling_diag / rx_local_dense；下一杠杆：拇指对向锥建模保真度（对照 Allegro URDF 实有范围）、手姿态或盒-手相对滚转（rz）解耦

### 勘误（2026-10-15）
- 4ee6f1d 的 commit message 误写"wrist pitch ±1.2→±1.8"，实际改动为 **wrist_rx（roll）**；真 pitch（wrist_ry）保持 -1.259..1.2 未动。诊断（runs/w35/w35d_diag.json）证明放宽真 pitch 亦无法开启 K1，无需后续修订。
- 波段 3.5-C 报告的"联合指标 0.089mm"经复核为**端面/平面延拓度量伪影**（ff 垫距 +y 端面、拇指距壁外 82mm 的顶平面延拓），非壁接触。后续轮次采用 w35d_diag.json 的严格壁接触判据（最近点须落在侧面矩形内）。
- Wave 3.5-D 结论：K1 双侧壁捏为 **B 类（真实形态×场景）结构性不可行**（~19M 位姿穷举 + 100 次 descent；正 pitch 拇指锥轴离 -x 143.6° / 负 pitch 手指扇出通道，全俯仰圆反相关）；MJCF 无可修目标。

## Wave 3.5-E / R6（2026-10-15）：近节↔盒显式 pair——把 3.5-B 的修法补到真正承力的连杆上

**缺陷。** R3 只在四个指尖垫上盖了刚度戳，Wave 3.5-B 又单独补了掌板一条 pair；**其余每根手指连杆的盒接触仍走隐式配对**——MuJoCo 对隐式 pair 取两侧 solref 的平均（取正值较大者语义），连杆自己的默认 20 ms 时间常数因此赢下这一侧，盒的地板在连杆下面比在指尖下面软。4ee6f1d 上的 smoke5 复跑正是这样把**最深穿透从掌板转移到未加硬的 `rf_proximal_collision`**（端面采样 1.188 mm @3.136 N f17，帧内峰值 1.282 mm @7.258 N），穿透门从 12/13 掉到 10/13。

**改动。** 生成器里四条显式 pair（`PROXIMAL_BOX_PAIRS`），与掌板同字段；重新生成后 XML 只多 4 行 `<contact>` 内容 + 一行头注释，byte 可复现（两次生成 `cmp` 一致）：

```
<pair geom1="ff_proximal_collision" geom2="object_geom" solref="0.004 1" solimp="0.95 0.99 0.001 0.5 2" condim="4"/>
<pair geom1="mf_proximal_collision" geom2="object_geom" solref="0.004 1" solimp="0.95 0.99 0.001 0.5 2" condim="4"/>
<pair geom1="rf_proximal_collision" geom2="object_geom" solref="0.004 1" solimp="0.95 0.99 0.001 0.5 2" condim="4"/>
<pair geom1="th_proximal_collision" geom2="object_geom" solref="0.004 1" solimp="0.95 0.99 0.001 0.5 2" condim="4"/>
```

### 穿透前后对照（`runs/w35/w35e_proximal.json`，冻结的 smoke5 replay，逐物理子步记录接触）

口径＝控制帧末尾（`retarget_smoke._finger_box_contact` 的采样点）：

| geom | 加硬前 | 加硬后 | 倍数 |
|---|---|---|---|
| `ff_proximal_collision` | 0.4812 mm @1.929 N (f15)；帧内峰值 0.7712 mm @9.805 N | **0.0471 mm** @6.912 N (f18)；帧内峰值 0.1149 mm @29.108 N | 10.2× |
| `mf_proximal_collision` | 0.4025 mm @4.581 N (f16)；帧内峰值 0.4811 mm @6.050 N | **0.0934 mm** @17.305 N (f25)；帧内峰值 0.0934 mm @18.740 N | 4.3× |
| `rf_proximal_collision` | **1.188 mm** @3.136 N (f17)；帧内峰值 1.282 mm @7.258 N | **0.0950 mm** @8.986 N (f25)；帧内峰值 0.2911 mm @20.522 N | 12.5× |
| `palm_collision` | 0.1067 mm @0.129 N | 不变 | — |
| 中节/远节/拇指各节 | 盒受力 0.000 N、穿透 0.0000 mm | 同左 | — |

**10 N 读数**（在实测的逐帧力–穿透曲线上插值，加硬后曲线实测到 11.0–20.1 N，10 N 落在实测区间内）：ff 0.0361 mm、mf 0.0666 mm、rf 0.0344 mm——落在 R3 指尖预算带（3.5-B 掌板同载荷 0.035 mm；R5 容差 0.213 mm）内。加硬前的 10 N 值由放大指令扫描夹逼：0.773–0.791 mm / 0.634–0.896 mm / 1.224–1.229 mm。smoke5 门数 **1.188 mm → 0.107 mm**（11.1×），已回到 1.0 mm 容差内。

### 跨对影响矩阵：非目标对逐位不变（R3 的 77 kN 教训未复现）

整段 replay 的逐物理子步峰值法向力，加硬前后逐位比较：

| 接触对 | 前 | 后 | 判定 |
|---|---|---|---|
| `pedestal\|object_geom` | 54.56006 N | 54.56006 N | **逐位相同** |
| `pedestal\|palm_collision` | 21.64077 N | 21.64077 N | **逐位相同** |
| `object_geom\|palm_collision` | 55.72984 N | 55.72984 N | **逐位相同** |
| `object_geom\|th_base_collision` | 1.38582 N | 1.38582 N | **逐位相同** |
| `ff_distal\|mf_distal` | 0.25175 N | 0.25175 N | **逐位相同** |
| `object_geom\|{ff,mf,rf}_proximal` | 9.80 / 6.05 / 7.26 N | 29.11 / 18.74 / 20.52 N | 目标对 |
| `ff_distal\|mf_medial` | 0.59239 N | 1.98174 N | 连带，+1.39 N |

R3 当年把刚度盖在"一侧"上，连掌板-基座擦碰都被顺带治理，从 3.2 kN 放大到 77 kN；显式 `<pair>` 只命名一个 geom 对，其余配对保留隐式混合——上表里唯一非逐位的非目标项是 +1.39 N 的指–指接触，绝对量级可忽略。掌板↔基座（当年最危险的那对）与掌板↔盒均逐位不变。

### ⚠️ 峰值列 vs 保持列：3× 反向效应，供下游力判据复核

加硬后接触把伺服器的出力在接触瞬间原样传给盒，**帧内峰值反力升约 3 倍**（rf_proximal 7.26 → 20.52 N），而**帧末保持力只从 4.91 升到 8.99 N**。任何下游的力阈值必须对照**保持列**复核；若拿帧内峰值当握力指标，会把加硬误读成"手变强 3 倍"。穿透指标则相反地受益于峰值列（更早、更硬地建立接触）。

### 最小充分范围＝仅近节（这是测量结论，不是偏好）

中节、远节、base 各连杆在本场景**对盒零受力**（0.000 N / 0.000 mm，见上表）。探针因此跑了三个变体：仅近节（4 条 pair）、近节+中节（8 条）、全部指骨（12 条）——**三者逐位相同**：穿透、每一个接触对的峰值力、每一项门指标全部 bit-for-bit 一致。给零受力的连杆加 pair 在此是无证据的改动，故不落。扩大范围必须由"确有片段加载这些连杆"的证据重新论证，`tests/test_smoke6_mjcf.py::test_w35e_scope_is_proximal_only` 已把该义务钉在测试里。

### 影响面（诚实记账）

- 接触 duty 0.667 不变；掌板执行偏差 102.24 mm **不变**——本改动不触碰 smoke5 的主 blocker（掌板压盒、腕部力预算耗尽），只解穿透。
- **执行器力饱和帧 261 → 259**（近见饱和减少）。
- `mfj1` 物理态越界 7.173 mrad **逐位不变**（取证见 `runs/w35/w35e_mfj1_forensics.json`）：该越界由 0.347 N 的指–指接触 `ff_distal|mf_medial` 驱动的位置伺服弹性挠曲引起，与盒接触无关，加硬与否都不改变它——需单独裁决门容差，本轮未改任何门阈值。
- R5 派生的穿透容差**不变**（实测静止下沉 1.163697055107007e-04 m 加硬前后逐位相同，因为 `pedestal|object_geom` 逐位相同）。
- 新 MJCF sha256 `1ea0fdf5c87ce5a161efcda7c597b0fc6647c9fe30bcf2f5d0bcb52402d24b89`（旧 `dbba3d70…`）。`MJCF_COMMIT` 钉在旧 sha 的示教集与下游产物由重跑轮处理；本轮**未改其任何文件**。
- R1 的 10 N 门语义随之明确：V5 的 10 N 是**指尖**规格，指尖垫实测峰值 1.48 N；连杆接触另按 R1 自身推导的预算 `Στ_j / min(arm_j)` 门（rf 16.1 N，实测 10.52 N）。见 `tests/test_smoke6_mjcf.py` 的 R1 段注释。

## Wave 3.5-F / R7（2026-10-15）：伺服绕组耐受表——把「命令侧上限衡量物理侧瞬态」换成三段式门

**缺陷。** 两条门拿**命令侧**的上限去卡**物理侧**的瞬态，而这两类量在本 MJCF 上根本不同源：

| 门 | 命令侧上限 | 物理侧被卡量 | 实测 |
|---|---|---|---|
| C3 `joint_limits` | `jnt_range` 零容差 | 手指关节实到 qpos 越界 | `mfj1` 7.173 mrad |
| W1B 腕铰实到速率 | 1.5 rad/s（自由空间实测可跟踪命令率） | 腕铰实到速率 | 1.7234 rad/s |

零容差不是力有限的位置伺服能被约束的东西：本 MJCF 每个手指关节都是 `limited=true` 且 `stiffness=0`，`jnt_range` 是**断言**而非求解器执行的机械止挡，接触下的弹性挠曲必然把 qpos 推过去。1.5 rad/s 则是**自由空间**实测的**命令**跟踪率，与加载工况下释放绕组储能的物理速率不是一回事。

**推导（用户 2026-10-15 批准）。** 允许量 = 该自由度的伺服力上限，经该自由度的**被动柔度**映射到被测量。两条都是「同一 dof 上两个 MJCF 属性之比」，逐自由度成表，**无自由参数**：

```
位置侧   |qpos − jnt_range|  ≤  |forcerange| / kp           [rad]
速率侧   实到速率            ≤  |forcerange| / dof_damping  [rad/s]
```

### 逐关节位置耐受表（16 指关节，`|forcerange| / kp`）

| 关节 | kp (N·m/rad) | forcerange (N·m) | **允许量 (rad)** | 实测越界 |
|---|---|---|---|---|
| ffj0 / mfj0 / rfj0 | 0.3261 / 0.3278 / 0.3285 | 0.03000 | 0.091996 / 0.091519 / 0.091324 | 0 |
| ffj1 / **mfj1** / rfj1 | 0.7269 / **0.7269** / 0.7266 | 0.10541 / **0.10388** / 0.10586 | 0.145013 / **0.142908** / 0.145692 | 0 / **0.007173** / 0 |
| ffj2 / mfj2 / rfj2 | 0.2348 | 0.09369 / 0.09400 / 0.09382 | 0.399020 / 0.400341 / 0.399574 | 0 |
| ffj3 / mfj3 / rfj3 | 0.0449 | 0.04219 / 0.04235 / 0.04219 | 0.939644 / 0.943207 / 0.939644 | 0 |
| thj0 / thj1 / thj2 / thj3 | 1.0423 / 0.4073 / 0.4186 / 0.0968 | 0.09933 / 0.03018 / 0.05201 / 0.03001 | 0.095299 / 0.074098 / 0.124247 / 0.310021 | 0 |

**`mfj1` 余量 = 0.142908 / 0.007173 = 19.92×**。表被 `tests/test_smoke5_winding_tolerance.py` 逐项从 model 重算比对，任何漂移即红。

### 逐腕铰速率耐受表（3 铰，`|forcerange| / dof_damping`）

| 铰 | kp | forcerange (N·m) | dof_damping | **允许量 (rad/s)** | 实测 (w1b 口径) | **余量** |
|---|---|---|---|---|---|---|
| wrist_rx | 14.0 | ±6.0 | 1.2 | **5.000** | 1.7234 | **2.90×** |
| wrist_ry | 14.0 | ±6.0 | 1.2 | **5.000** | 1.6171 | 3.09× |
| wrist_rz | 14.0 | ±6.0 | 1.2 | **5.000** | 1.6033 | 3.12× |

三铰数值相同，是因为 kp / forcerange / dof_damping 在本 MJCF 上**逐项核对出来确实相同**，不是借用。

### 三段式门（三条全过才绿，任一不满足即红）

1. **幅值** ≤ 上表逐自由度允许量；
2. **时长** ≤ **3 个连续控制帧**（20 Hz 下 150 ms）；
3. **解释项**——瞬态必须是弹性绕组挠曲，不是故障：
   - 该自由度**未饱和**（`|τ_act| < forcerange`）。**饱和期间记录的超限一律不予豁免**：过了饱和点绕组量按定义自封顶在 `|forcerange|/kp`，再大的越界不可能是绕组；
   - 且绕组量与载荷吻合。**饱和期间的超限、无接触力矩的自由漂移，两个反例都由测试钉死。**

位置侧的数值形式：`τ_other = −qfrc_bias + qfrc_passive + qfrc_constraint`（该 dof 上全部非执行器广义力），要求 `|kp·(ctrl−qpos)|` 与 `|τ_other|` 吻合在 **5%** 内，且 `|τ_contact| > 0`。实测 `mfj1` frame 0：`|τ_servo| = 0.0131270`、`|τ_other| = 0.0131639`，残差 **0.28%**；峰值扭矩 0.0131 N·m 远在 forcerange 0.10388 N·m 之下。

> ⚠️ **符号勘误（实测，非照抄文档）**：本 MuJoCo 构建里 DoF 方程闭合为
> `M q̈ = −qfrc_bias + qfrc_passive + qfrc_constraint + qfrc_applied + qfrc_actuator`，
> 即 `qfrc_bias` 与经典文档**反号**。这是五项符号全枚举里唯一能把 `mj_fullM(model, data) @ data.qacc` 复现到 **4e-16** 的组合（`retarget_smoke._qfrc_other`）。照抄经典符号会把绕组量与一个**不存在的载荷**比对——那正好是一个看着能过、其实什么都没卡的解释项。

速率侧的数值形式（逐控制区间 t、逐腕铰 j）：

```
excess_j(t) = |Δq_j/Δt| − v_cmd_j(t)                     # 要求 excess > 0
windup_j(t) = ctrl_j(t) − qpos_j(t)
release_j(t) = √( kp_j · windup_j(t)² / (dof_M0_j + armature_j) )
可容许 ⟺ excess_j(t) ≤ release_j(t)  且  |τ_act| < forcerange
```

`release` 取无阻尼上界（偏严：阻尼只会让实际释放更小，所以它是**允许的最大解释量**）。实测峰值区间（rx，interval 11）：`excess = 0.2234`、`release = 1.2932 rad/s`，余量 5.8×（ry 16.9×，rz 15.5×）。峰值**全部落在斜坡中段**——命令侧速率是平的 1.5 rad/s，而绕组已累积 0.086–0.132 rad；机理是**斜坡期间累积的弹性绕组在伺服追平时释放，叠加到命令速率之上**，与 `mfj1` 同构。

**两侧的时长子句都切在各自的推导允许量上**（位置侧：越界量；速率侧：5.00 rad/s），这与用户批注的原话「幅值 ≤ 5.00 + 时长 ≤3 帧」成对。代价是速率侧第 2 条被第 1 条蕴含：对一个有界量，单个区间越界就已经是幅值违规，所以不存在「幅值过、时长不过」的轨迹（`test_rate_duration_clause_is_implied_by_the_amplitude_clause` 钉住这条性质）。位置侧不存在这个退化——那里越界量以 0 为基准，`excess > 0` 的连续帧数与 `excess ≤ cap` 是两条独立约束。

因此，「持续高于命令侧 1.5 rad/s」这一量**不参与判定，但被逐铰记录**为 `max_consecutive_intervals_over_command_cap`：

| 铰 | 实到峰值 | 峰值处 excess | **连续高于 1.5 rad/s 的区间数** | 3 帧窗口？ |
|---|---|---|---|---|
| wrist_rx | 1.7234 | 0.2234 | **10** | 否（超出 3.3×） |
| wrist_ry | 1.6171 | 0.1759 | 2 | 是 |
| wrist_rz | 1.6033 | 0.1033 | **4** | 否（超出 1.3×） |

这是本轮的一条**真实发现，不是被埋掉的数字**：斜坡期间 rx 被以约 1.15 倍命令率跟踪了整整 500 ms（10 个控制区间），而 3 帧窗口是 150 ms。

**裁决（Main，2026-10-15）：时长子句切在推导允许量 5.00 rad/s，不切在 1.5 rad/s；上面那个 1.15× 记为跟踪保真度指标，本轮不设为门。** 理由：已证明该超量全程是绕组释放（最紧格 rx 区间 11：`excess 0.2234 ≤ release 1.2925`），即约 1.15× 的**斜坡跟踪比**，属伺服跟踪特性而非物理故障；为它设门等于用跟踪误差冒充物理故障。P1 再决定是否把「加载工况下持续实到/命令 ≤1.25×（自由空间实测包络 0.8%，加载 15%）」提升为**独立的跟踪保真度门**——那需要它自己的证据链，不在本轮。数字由 `test_rate_over_command_cap_is_recorded` 钉住，不会消失。

### 被否决的界：`Δv ≤ a_max · τ_s`（留档）

```
a_max = forcerange / I = 93.8 … 97.6 rad/s²
τ_s   = e_max / v_cap = 0.4286 / 1.5 = 0.2857 s
Δv    = a_max · τ_s   = 27.886 / 27.330 / 26.787 rad/s   (rx / ry / rz)
```

比实测 1.7234 高一个数量级以上，**空界**：腕铰又硬（kp=14）又轻（I≈0.062），加速度上限 `forcerange/I ≈ 97 rad/s²` 极大；而 wind-up 穿越窗口 0.286 s 是**命令速率给的时间**，不是伺服给的。写进来是为了让「被检查过的界」留在记录上，而不是被悄悄丢掉。留在 `retarget_smoke.SERVO_WINDING_REJECTED_RATE_FORMS`，并由测试钉住它必须同时高于实测一个数量级、高于替代界 5 倍。

### 紧度：界是紧的（按本轮复测记账）

自由空间阶跃（盒沿自身 freejoint 抬 3 m、质量置 1e-6 kg，1.2 rad 阶跃，两向，控制周期平均速率）：

| 铰 | −1.2 rad | +1.2 rad | 与推导 5.0 的偏差 |
|---|---|---|---|
| wrist_rx | 4.9076 | 4.9664 | 1.85% / 0.67% |
| wrist_ry | 4.9732 | 4.9011 | 0.54% / 1.98% |
| wrist_rz | 4.9569 | 4.9546 | 0.86% / 0.91% |

**平台 4.90–4.97 rad/s，与推导差 ≤2.0%**——这不是巧合：界里的 `forcerange` 就是让平台出现的那个量。`rx`/`ry` 的两向不对称是重力偏置。取证报告记录的 4.95–5.05、差 <1% 在本轮复现中**没有复现出来**，按实测值记账（测试阈值相应取 3%）。

### `kv = 0` 不影响界的成立

腕铰执行器是纯增益（`kv = 0`，`actuator_biasprm[a,2] = 0`），所以 dof 方程里唯一的速度正比项是关节自身的 `damping` 属性。**反事实实测**：把三个铰的 `dof_damping` 置 0（其余全同）重跑同一套自由空间阶跃，渐近平台**彻底消失**，峰速变成弹性直线 `A·√(kp/I)`（1.2 rad 时 ≈18 rad/s，接近 ctrlrange 允许的极限）。所以 **`dof_damping = 1.2` 才是有限界存在的唯一原因，`kv = 0` 与之无关**。该性质由 `test_rate_bound_does_not_depend_on_kv_but_does_on_dof_damping` 钉住。

### 加载工况比自由空间更高——如实记录

同一份 w1b 斜坡指令跑两遍（自由空间 vs 真实 clip 加载）：

| 铰 | 自由空间 | 加载 | 差值 | 加载/自由 |
|---|---|---|---|---|
| wrist_rx | 1.6853 | 1.7234 | +0.0381 | **+2.3%** |
| wrist_ry | 1.4917 | 1.6171 | +0.1255 | **+8.4%** |
| wrist_rz | 1.5099 | 1.6033 | +0.0934 | **+6.2%** |

**自由空间不是保守包络**：接触力矩与驱动力矩同号叠加时抬高瞬时速率，接触同时锁死三个 slide 自由度、改变腕铰的有效惯量。量级上完全可忽略（最大差 0.125 rad/s，对 5.0 的余量是 2.90×），两种工况都被 A 覆盖，接触含入上限 `(fr + τ_c)/b` = 5.72 / 5.75 / 6.15 rad/s 也仍高于实测。

### C9 反空洞探针几何无关化（同轮）

`test_c9_contact_detector_is_not_vacuous` 的正向半边原是「最大卷曲 + 腕最低 → 必须触到盒」，这实际上是对**本手几何**的一个断言：链长、关节范围或盒的加硬 pair 一改它就红，而检测器并无过错——MJCF `1b0f13f` 正是这样把它打破的。被测性质从来不是「这只手够得到盒」，而是「检测器能分辨接触与无接触」。故改为**主动构造接触**：把盒沿自身 freejoint 驱到某个指尖体原点上（该重叠与手形、关节范围、盒落点全无关），要求检测器在求解器的接触表里报告它，且在再走 10 个物理子步后仍然报告。负向对照（手停在够不到处）不变。

### 影响面（诚实记账）

- **无任何 MJCF 改动。** XML sha256 仍 `1ea0fdf5c87ce5a161efcda7c597b0fc6647c9fe30bcf2f5d0bcb52402d24b89`，新 MJCF commit 仍 **`1b0f13f`**。本条只改门定义与其判据代码（`retarget/retarget_smoke.py`、`tests/`）。
- **顺带修掉的 provenance 缺陷（否则本轮会签出错误的账）**：`retarget/mjcf_provenance.py` 的 `MJCF_COMMIT` 一直钉在 `4508421`，而工作树的 XML 早在 Wave 3.5-E 就是 `1b0f13f` 的 blob。因此**每一次重跑 smoke5 都会把产物重新盖章成 4508421，并附一条假的「工作树 XML 与 4508421 不匹配」警告**——已提交的 report.json 之所以写着 1b0f13f，是因为它不是被这条路径重写的。两个常量已改为 `MJCF_COMMIT = "1b0f13f"` / `MJCF_COMMIT_BASELINE = "4ee6f1d"`，与已提交产物逐字一致，且 `scene_matches_mjcf_commit` 恢复为 true。核对：`blob(1b0f13f) == sha256(工作树 XML) == 1ea0fdf5…`，`blob(4ee6f1d) == dbba3d70…`。
- smoke5 的 13 项 `checks`：**11 → 12 绿**（`joint_limits_ok` 由 `mfj1` 7.173 mrad 转绿）。**仍红 1 项**：掌板执行偏差 102.24 mm —— 属已批准的 P1 接触一致 retarget，不是门容差问题，本轮不碰。
- `test_w1b_realised_rates_within_caps` 由红转绿（峰值 1.7234 ≤ 5.000；时长子句在 5.00 线上为 0 区间；解释项逐区间成立，最紧的一格是 rx interval 11 的 `excess 0.2234 ≤ release 1.2925`；峰值 τ 未饱和）。**注意其中的口径决定**：若时长子句改切在 1.5 rad/s 上，该测试仍为红——见上文「持续高于命令侧 1.5 rad/s」一节。
- 1.5 rad/s **仍是**命令侧上限（W1 / W1c 未动）。本轮只把**实到速率**一侧改判；腕 slides（dof_damping = 25）与手指（URDF 3.14 rad/s）两侧口径均未动。

## Wave 4 / R8（2026-10-15）：盒上缘唇边——**证据无效，结论待重扫**（原"运动学否决"已撤回）

**裁决变更（2026-10-15，Main 裁定）：本节的"运动学否决"作废。** 几何仍**不在**仓库里
（XML 与 `1b0f13f` 逐字节一致，sha256 `1ea0fdf5…`，`git diff` 为空），但**撤回几何的理由不成立**：
原判据读错了 MuJoCo API 字段，"15 格全 0"不是可达性证据。唇边**尚未被否**。

### 设想（仍然成立，与判据无关）

纸巾盒有一圈折边，加到代理体上缘：每面 ±x 壁顶部 `lip_h` 高的凸缘，外伸 `overhang`。
凸缘**下沿**是一片悬空水平面——指尖若能钻到它下面向上顶，就得到一个**纯法向**承力路径，
而裸的竖直侧壁只能靠壁面摩擦撑着盒子。

### 判据写法（不变）

最近点须落在**唇边下沿矩形**内（z = 盒顶 − lip_h，bx ≤ |x| ≤ bx + overhang，|y| ≤ by），
间隙 ≤ R5 容差，且：任一手指 geom 不得埋入物体；连杆顶在唇下是楔不是抓，故必须是**指尖垫**。

### 原扫描为什么无效：`mj_geomDistance` 的字段误用

`runs/w4/lip_kin_scan.py` 取 `dist = float(ft[0])`，其中 `ft` 是该函数的 `fromto` 缓冲区。
**`fromto` 里没有距离**：该函数的**返回值**才是有符号距离，而
`fromto = [geom1 上的最近点(3), geom2 上的最近点(3)]`。对照实验（两个 box 相距 100 mm）：

```
ret = mj_geomDistance(m, d, a, b, 1.0, ft)   ->  0.100000
ft                                          ->  [0.1  0.  0. | 0.2  0.  0.]
```

即 `ft[0]` 是**手指侧最近点的世界 x 坐标**。于是原判据 `0.0 <= dist <= 1.16e-4`
实际在问"手指最近点的 x 落在 0 到 0.116 mm 之间"——与唇下沿是否可达**无关**，
是一条随机性极强的条件。**15 格全 0 因此不能作为唇边不可达的证据**，原节保留在此仅作留痕。

（`ft[3:6]` 取的是 geom2 上的点，即唇上的点，那部分用法是对的；错的只有距离那一项。）

### 重扫

脚本：`runs/w4/lip_kin_recheck.py`——同一严格判据、同一粗网格 + 400 次随机重启的局部精修
（与 W4-B 的 K1 扫描同一套搜索），场景走本节留下的参数化入口 `build(lip_overhang_m, lip_h_m)`，
XML 在内存里编译、临时文件用完即删。结果见该脚本写出的 `runs/w4/lip_kin_recheck.json`。

### 几何状态（未变）

几何**不在**仓库里。参数化与质量记账**留着**：`build(0, 0)` 逐字节复现裸盒 XML（实测 `cmp` 通过），
新增质量从 0.15 kg 里扣而不叠加（实测 145 g + 2×2.5 g = 150.000 g，`body_mass == 0.15` 精确）。
**在重扫出结论之前，不得引用"唇边已被否决"这句话。**

### 本轮零改动声明（不变）

- **验收门一字未改**（A1–A5、`test_smoke4_reward.py`、`test_smoke5_*`、`DESIGN.md`）。
- **`W_PEDESTAL_LOAD = 4` 未动**；**盒重仍 0.15 kg**。
- 撤回几何时**没有**任何门转绿，本轮**依然没有**任何门因唇边而转绿。

## Wave 4 / R9（2026-10-15）：落位取足迹而非取中心——`test_c6` 的假绿取证与修复

**改动面：只有 `grounding/envs/lift_hold_env.py`（落位与静置）+ 新测试
`tests/test_scene_init.py`。MJCF 零改动**（XML sha256 仍
`1ea0fdf5c87ce5a161efcda7c597b0fc6647c9fe30bcf2f5d0bcb52402d24b89`），
验收门一字未改，`W_PEDESTAL_LOAD = 4` 未动，盒重仍 0.15 kg。

### 一、`test_c6` 的绿灯是假的（完整取证链）

C6 的意图是"物体只能通过接触移动"：把手够不到的盒子停在桌上，随机扑腾 100 步，
位移必须 < 1e-3 m。**这条门本身是对的，错的是场景在骗它。**

| 步骤 | 量 | 值 | 测量方法 |
|---|---|---|---|
| ① 命令落位 | `out_of_reach` 把盒放在 | `(x, y) = (0, 0.35) m` | `lift_hold_env.py` reset 原文 |
| ② 盒足迹 | `half = (0.06, 0.12)` | `y ∈ [0.23, 0.47] m` | `model.geom_size["object_geom"]` |
| ③ 旧 `_support_height` | **只采样盒中心** `(0, 0.35)` | 返回桌面 **0.700 m** | 中心在 pedestal 之外（pedestal 只到 `y = 0.30`） |
| ④ 落位高度 | `z = 0.700 + 0.040` | **0.740 m** | 即盒底 0.700、盒顶 0.780 |
| ⑤ **出生接触表** | `mj_forward` 后的 `d.contact` | **4 × `pedestal\|object −0.0600 m`** | 足迹 y∈[0.23,0.30] 那 70 mm 盒底整段陷在 pedestal 里 |
| ⑥ 400 步静置后 | 盒位姿 | `(0, 0.7557, 0.0394) m` | `z0 = 0.0394` —— **已经掉到地板上**（盒半高 0.040，平躺） |
| ⑦ 随机扑腾 100 步 | 位移 | **0.571 mm < 1e-3 m** | 门判 pass |

也就是说，**reset 的 400 步静置里盒子已经被弹飞了 0.41 m 并落到地面**，这段运动发生在
`z0` 记录之前、对门完全不可见；门随后测的是"一块躺在地板上的静止物"。

**对照实验（排除唇边等其它变量）**：把盒子挪到完全落在桌面内的 `y = 0.25 m`（足迹
`y ∈ [0.13, 0.37]` 全在桌面内），**HEAD 侧同样失败，位移 22.7 mm**。位移来自"出生重叠
被弹飞"，与 R8 的唇边几何无关。

### 二、修法

1. **`_support_height` 取足迹**：对所有与盒足迹**相交**的支承面取最高顶面，而不是问中心
   一个点。这一步就把 ⑤ 的负向穿透变成不可表示——盒子永远出生在它真正压到的最高面上。
2. **`_face_fit` 要求整足迹落在**同一块**支承面上**（含 `SUPPORT_EDGE_MARGIN_M = 5 mm` 边距），
   且该面必须**就是**足迹相交到的最高面。第二条不是冗余：pedestal 嵌在 slab 里，一个跨过
   pedestal 边缘的足迹在 slab 上"看起来留有余量"，按 slab 高度落下去反而把悬空那截埋进
   pedestal——同一个 bug 的另一副面孔（实测命令落位 `(0.30, 0)` 即如此）。
3. **落位不满足条件就搬走，并记账**：`_search_placement` 先单轴滑动（够覆盖 demo collector
   的 ±6 mm 网格），单轴无解再二维扫（控制位的 +y 请求在 y 方向**任何距离都无解**，单轴滑动
   救不了被钉死的另一轴）。评分 = (边距, 离手距离)，取最稳的合法位姿。找不到就
   `RuntimeError`，**不静默降级**。落位结果写进 `reset` 的 info（`placement`），
   包含 `requested_xy_m` / `placed_xy_m` / `resting_z_m` / `edge_margin_m` / `clearance_m` /
   `moved`——**这次修正必须看得见**，当初的 bug 就是因为看不见才活下来的。

### 三、控制位的 +y 请求为什么必须被否决（不是"挪一挪"）

- 几何：pedestal 到 `y = 0.30`，slab 到 `y = 0.45` → 基座外只有 **150 mm** 宽的裸桌带；
  盒子 y 向 **240 mm** 长。**沿 y 轴不存在"完全落在单一支承面上"的位姿**（y ≥ 0.42 才
  躲开 pedestal，但 y ≤ 0.33 才不下桌，两条互斥）。
- 而唯一勉强合法的 y 位姿是 `y = 0.18`（坐在 pedestal 上）：**实测随机扑腾 4 种子 × 100 步
  出现 24 次手指-盒接触**，C6 的"够不到"前提直接不成立（该位姿扑腾位移 19.9 mm）。
- 沿 x 轴则成立：盒 x 向只有 120 mm，150 mm 的带子装得下，合法窗口 `x ∈ [0.36, 0.39] m`。

因此 `CONTROL_CLEARANCE_M = 0.28 m`：盒足迹必须整体落在手的可及圆盘之外。依据是实测的
**腕部行程 ±0.18 m**（`actuator_ctrlrange` 前两路）与**扑腾中指尖最远 0.2181 m**
（4 种子 × 100 步的最大 `|tip_xy|`）。0.28 m 对两者各留 ~60 mm 余量，"够不到"从此是
**落位本身的性质**，而不是对某次扑腾的祈愿。

### 四、改后实测

| 量 | 改前（HEAD） | 改后 | 测量方法 |
|---|---|---|---|
| 控制落位 | `(0, 0.35, 0.740)`，足迹跨 pedestal | `(-0.36, 0.30, 0.740)`，平躺裸桌 | `reset` info `placement` |
| 边距 / 离手距离 | 无（不可测） | 30 mm / **0.350 m** | `edge_margin_m` / `clearance_m` |
| **出生接触表** | **4 × −0.0600 m** | 全部 `≥ 0`（仅相切） | 落位 qpos → `mj_forward` → `d.contact` |
| 静置 399 步位移 | — | **0.000000 mm**（最大同值） | 冻结 ctrl 于 reset 位姿，399 × `mj_step`，逐帧取 `‖xpos−p0‖` 最大 |
| 静置末速度 | — | `|v| = 1.03e−11 m/s`，`|ω| = 2.40e−10 rad/s` | `qvel[obj_dof : +3] / [+3 : +6]` |
| 扑腾 100 步位移 | 0.571 mm（假绿） | 0.000 mm，盒仍在桌面 `z > 0.70` | C6 原协议 + 新增在桌断言 |
| 指-盒接触 | 0 | **0** | C6 原协议 |

**对照（仅取足迹高度、不搬位）**：`y = 0.35` 处足迹高度给出 `z = 0.800`，不再互穿，但盒子
骑在 pedestal 边缘**歪着**落到 `y = 0.3618, z = 0.7783`，且残留 `|v| = 4.3e−05 m/s`
仍在蠕变。**"不互穿"不等于"躺得住"**，两条门都钉上才作数。

### 五、影响面与回归

- **验收门未动**：`tests/test_smoke4_accept.py` A1–A5、`test_smoke4_reward.py`、
  `test_smoke5_*`、`DESIGN.md` 全部原样。C6 门仍是 1e-3 m，**没有为通过而放宽**。
- **新测试 `tests/test_scene_init.py`（42 项全绿）**：足迹高度回归、10 组命令落位零负向穿透
  （含跨 pedestal 边缘与带 yaw 的）、24 次真实 `reset` 端到端零互穿且静置不动 < 1 mm、
  静置 399 步位移/线速度/角速度/四元数四重断言、控制盒扑腾后仍在桌面、控制位姿离手距离
  > 腕部行程、以及"yaw 后无合法位姿必须大声报错而不是悄悄降级"。
- 回归：`test_smoke4_env.py` **7/7 绿（含 `test_c6`）**、`test_smoke4_reward.py` 绿、
  `test_scene_init.py` 42/42。
- **两项与本轮无关的既有红灯**（在 `HEAD` 的干净 worktree 上逐一复现，非本轮引入）：
  `test_p0_plumbing.py::test_p4_eval_writes_formal_schema`（`b252081` 引入的
  `*_peak` 轨迹键与评估落盘键集不同步）、`test_p0_summary.py` /
  `test_smoke4_demos.py`（runs/p0 记账里的 MJCF basis 仍写 `1b0f13f`，实测报 `4ee6f1d`）。

### R8 终版（2026-10-15）：唇下沿钩——运动学不可达（审计定案）
- 可信度证据保留：R5 重导值（静置下沉 `2.90924263123702644e-05` → 容差 `1.16369705249481058e-04`）、唇边质量记账（145 g + 2×2.5 g = 150.000 g）、审计运行所依据的 XML = `1b0f13f`。
- 重扫（修正判据 v1）报 3/3 clean，但正反对照审计（`runs/w4/hook_direction_audit.json`）逐垫核查揭穿：三格的"命中"全是垫摸**唇顶棱/盒顶平面**（z=0.840），非唇下沿（z=0.786–0.794）；唯一到唇下沿以下的 th_tip 距唇 15.7cm。`any_true_hook=false`。
- 机理同前：够到唇下沿需在盒 x 面外侧从顶棱上方卷下，掌板底 0.839 与盒顶 0.840 齐平，工作空间无此路径。
- 本节经历三次测量：`fromto[0]` 误用（过严，0/15）→ z 带过宽（过松，3/3 假阳性）→ 审计定案（不可达）。三次里两次给出"漂亮"的相反数字，只有落点/朝向核验可靠。
- 首扫网格（错误字段下）：overhang 4–12 mm × lip_h 6–14 mm，15/15 格 0 命中；重扫（z 带过宽）3/3 假阳性，命中点全在盒顶面 rel_z=+0.040（唇下沿在 0.026–0.034）。审计口径：最近点须落在唇下沿矩形内、仅指尖垫、任一手部 geom 不得埋入。
- 唇边几何不入库；参数化 `build(lip_overhang_m, lip_h_m, out)` 保留为资产，`build(0,0)` 逐字节复现裸盒。
