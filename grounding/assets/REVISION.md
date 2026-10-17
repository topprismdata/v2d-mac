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

## W6a / R10（2026-10-15）：双手机场景——盒子直接落桌，两只手各据一个短端面（三关验证通过）

任务（签字于 8a18651）：按源视频把场景实例化为**双 Allegro 手 + 盒子直接落在桌面上（去 pedestal）**。
本节记录 cj 设计、依据与三关验证数字。验收门阈值一字未动（A3a 的 1.4715 N 总力仍由双手合计供给，
`W_PEDESTAL_LOAD = 4.0` 未动，盒重恒 0.15 kg，mass 9 位精度保持）。

### 一、命名约定：右手保持全名，左手统一 `_left` 后缀

- **右手逐字保留上游名**（`hand_root` / `palm` / `ffj0..thj3` / `ffa0..tha3` / `ff_tip` / `wrist_*`）：
  `retarget/` 与一切既有 trace 里的名字继续解析到同一只手，零迁移成本。
- **左手每个 body / joint / actuator / geom 名追加 `_left`**（`hand_root_left` / `palm_left` /
  `ffj0_left` / `ff_tip_left_collision`）。后缀而非前缀，是**被冻结的验收分类器倒逼的**：
  `tests/test_smoke4_accept.py::_finger_geom_ids` 按 body 名**前缀** `ff_/mf_/rf_/th_` 认指尖，
  `ff_tip_left` 命中，而 `l_ff_tip` 会让左手整个从 A3a 力求和中**静默消失**。
  `tests/test_scene_w6.py::test_w6_finger_geoms_carry_the_frozen_prefix_rule` 把这条钉死。
- 左子树 = 原始 palm 子树的深拷贝改名，走**同一套** R4 命名 / R3 刚度代码路径
  （`_left_palm_from` 在命名**前**拷贝，无重复 geom 名风险）；显式 pair 与 exclude 同步翻倍
  （30 对 = 右 15 + 左 15；exclude 5 + 5）。力矩臂/kp/forcerange 表在右手上测量一次，
  逐字写入左手执行器（几何逐字节相同，对称性由测试钉住）。

### 二、pedestal 移除依据（源视频）与参数化

pedestal（60 mm）存在的唯一理由是给**单手**让出盒下空间（R4-B 测量：贴桌时单手只能钩顶缘，
滑脱 3.2–3.6 cm，低于 5 cm 线）。源视频的抓取是**两手各捏一个 120×80 mm 短端面**，盒子直接
落在桌面上——清障功能被第二只手取代，`PEDESTAL_H_M = 0.0`，geom 不再发射。参数保留：
调回 0.06 即重生旧几何场景（`test_r3_revised_scene_still_holds_the_box_up` 等读该常量的公式全部自适应）。

### 三、腕部俯仰扩程与"掌面垂直"构型的实测否决

- **`wrist_ry` 扩到 ±1.571 rad（整个四分之一转）**。R2 时代 (−1.259, 1.20) = ±68.7°；端面抓取
  要求掌面转到水平，边界网格求解 `|R@(0,0,−1) − 目标|` 恰好卡在旧 ry 上限（实测，非假设）。
  新范围严格包含旧范围，smoke5 全部记录命令（最大 |0.058| rad）仍在其中。
- **掌面垂直（±90°）构型被运动学否决**：腕链复合 R = Rx@Ry@Rz，对已建模型对 `ffj1` 有限差分实测，
  屈曲正方向永远把指尖沿 **+R@e3（掌法向的反向）**卷——掌面垂直时屈曲轴保持世界竖直，
  所有关节范围的合成都把指尖卷向**盒外**，指垫到不了端面（探针实测 +93 mm）。45° 倾斜掌
  （euler `0 ∓45° 0`）保住"掌面朝向端面"的任务语义，同时给屈曲留出可用的绕卷路径。
  这是本节三次测量之一：掌垂直两次给出"看似可行"的解析，只有有限差分证据算数。
- **滑台伺服重整**：45° 倾斜使 wrist_x / wrist_z 各承载 12.645 N 手重中的 8.9 N 分量
  （单手时代只有 wrist_z 承重，且由掌-盒接触结构支撑）。保持 kp=180 时 z 滑台将下沉 ~50 mm
  把手拖进盒里（实测：旧 kp 下 400 步静置滑移 30–35 mm 后指尖磨桌 12.6 N）。
  两个滑台 kp → 15000（静沉 0.6–0.8 mm，ω·dt = 0.216 < SERVO_BANDWIDTH_GUARD，阻尼 → 280 ≈ 临界，
  c_crit = 278）。力上限实际不变：8.9–12.6 N 偏置要到 ~1 mm 误差才触及 ±45 N 限幅。
- 出生位姿（`HAND_HOME_RIGHT = 0.25 0 0.95`，左镜像；euler 右 `0 -0.785398163397 0` / 左 `0 +0.785398163397 0`）：
  右手指尖出生在 (+0.326, y, 1.026)，左在 (−0.174, y, 0.874)，全部离开盒与桌。

### 四、模型统计与逐字节数

| 量 | 单手（HEAD） | W6a 双手 |
|---|---|---|
| nq / nv / nu | 29 / 28 / 22 | **51 / 50 / 44** |
| ngeom / nbody | 48 / 24 | **89 / 46**（42 手 geom ×2 + 3 盒 + 地板 + 桌板） |
| 显式 box pair / exclude | 15 / 5 | **30 / 10** |
| 盒质量 | 0.15 kg | 0.15 kg（149.602 g 盒体 + 2×2.5 g 倒角板，记账不变） |
| XML sha256 | d61dc0de…（c7c6dac） | `ff4d4355337cb2d9e59f70ba67bac865576ef36a9c5158f51e53eeff771ee691` |
| 逐字节可复现 | 是 | 是（连续两次 build 相同） |

R1 力矩臂表（双手机逐字相同，右手测量；单位 mm / N·m）：

| joint | arm | τ | joint | arm | τ |
|---|---|---|---|---|---|
| ffj0 | 12.00 | 0.0300 | mfj0 | 12.00 | 0.0300 |
| ffj1 | 42.16 | 0.1054 | mfj1 | 41.55 | 0.1039 |
| ffj2 | 37.48 | 0.0937 | mfj2 | 37.60 | 0.0940 |
| ffj3 | 16.88 | 0.0422 | mfj3 | 16.94 | 0.0424 |
| rfj0 | 12.00 | 0.0300 | thj0 | 31.57 | 0.0789 |
| rfj1 | 42.35 | 0.1059 | thj1 | 12.00 | 0.0300 |
| rfj2 | 37.53 | 0.0938 | thj2 | 20.80 | 0.0520 |
| rfj3 | 16.88 | 0.0422 | thj3 | 12.01 | 0.0300 |

### 五、三关验证（决定性，全部通过后才 commit）

1. **出生穿透零**：qpos0 `mj_forward` 全场景接触表 4 条，全部 `table_slab ↔ object_geom`，
   最深 −6.245e−17 m（数值零，精确相切）。两手无任何出生接触。
2. **静置稳定**（ctrl=0，400 步）：盒位移 **0.029 mm**，两手根位移各 **0.422 mm**，
   `max|qvel|` 2.0e−2（初始瞬态），终态接触仅盒-桌 4 条（沉陷 2.91e−5 m = R5 静置下沉，与单手时代同值）。
   无弹飞、无振荡、无指尖磨桌（伺服重整前后对比见第三节）。
3. **双手可及**（运动学探针，pedestal_scan 方法：结构化粗网格 + 收缩随机精修 + 随机重启，
   22 手部自由度/手，任意手部 geom 埋入盒内 +0.5 m 惩罚，指垫球过面平面超容差按埋入罚）：
   一个同时配置下，右手**拇指垫**触 +x 端面 (+0.0703, −0.1200, +0.7860)（顶缘棱区），
   左手**食指垫**触 −x 端面 (−0.0719, +0.0535, +0.7588)（端面中部），
   间隙 **−0.115 mm / −0.115 mm**，R5 触及容差 ±0.1164 mm，判据 |gap| ≤ tol → **PASS**，零埋入。
   手性差异（右拇指/左食指）是两只同构右手模型的必然，镜像语法上无解，物理上各自成立。
   探针固化在 `tests/test_scene_w6.py::test_w6_gate3_both_hands_reach_their_end_faces`。

### 六、env 适配与影响面

- `N_ACT = 44`（动作 = 右手 22 维块在前、左手在后，块内语义与单手版逐维相同）；
  观测 107 维（13 盒 + 每手 43 + 8 历史）；`FINGER_BODY_PREFIXES` 不变即覆盖双手。
- 奖励语义零改动：掌距/对准仍测**右掌**，`W_PEDESTAL_LOAD = 4.0`、cap 12 不动；
  支撑分解/承载通道自动覆盖双手（前缀分类），`pedestal_up_N` 键名保留（现为桌板支撑，schema 稳定）。
- `tests/`：新增 `test_scene_w6.py`（12 项）；`test_smoke6_mjcf.py` 场景钉子按 W6a 更新
  （计数/范围/pair，冻结阈值未动）；`test_smoke4_env.py` C1–C4 维度适配（C4 全行程采样增强为
  整段 episode 统计，语义加强未弱化）；`test_scene_init.py` 42/42 绿（足迹回归改钉桌板边缘）；
  `test_smoke4_pedestal_load.py` P2/P4 标注 superseded（楔形策略回放的是已不存在的 pedestal 场景）。
- **回归全绿**：`test_smoke4_env` / `test_smoke4_reward` / `test_p0_plumbing` 8/8 /
  `test_smoke6_mjcf` / `test_scene_init` / `test_scene_w6` / `test_smoke4_pedestal_load` 11 过 2 跳。
- 已知外围（W6b 处理）：`collect_clean_demos.py` 已做最小双手形状适配（脚本驱动右手、左手驻停），
  但**双手协同示教的采集编排**属后续波次；旧 64 维观测的策略制品（含 f210f45 wedge）按证据归档，
  不予重盖。

## W6a-v2（2026-10-15）：立姿门 3——姿态约束重扫 + 左掌符号修正 + 两遍构建卫生

主代理审渲染后否决了 v1 的门 3 解：判据（指垫到端面距离）被"手趴平、指尖贴底缘"的解满足，
**测的东西比名字窄**。v2 给门 3 加三条姿态约束并重扫；同时查出两处真 bug。

### 一、两处测量修正（都是"亲测模型"抓出来的）

1. **左掌 euler 符号**：v1 给左手 euler `0 +45° 0`，实测（`palm_collision` 世界法向 vs 端面内向法向）
   左掌法向偏 **135°**——掌心朝桌！右掌同式为 45°。v2 统一 `0 -45° 0` 后再复核，
   发现更根本的事实（下条）。
2. **palm body 自带 `quat="0 1 0 1"`**（绕 (1,0,1)/√2 转 180°）：掌碰撞板在手的自然系里是
   **竖直板**，其面法向 = 手系 (−1,0,0)——与伸展的手指反向。此前所有"掌面朝下"的分析都建在
   错误前提上（v0 场景里掌板是竖直的，"palm floor 0.839" 是板底**缘**）。

### 二、立姿设计（v2 定稿）

- **右手 euler `0 0 0`**：掌法向恰好 = (−1,0,0) = +x 端面内向法向，**0° 偏差**；
  手指水平伸过端面上方，屈曲即沿端面外侧下卷钩住端面（拇指空出朝顶面，与视频一致）。
- **左手 euler `0 0 π`**：镜像（掌法向 = +x̂ = 其端面内向法向，0°）。
- **初始位姿** `HAND_HOME = ±0.185, 0, 0.795`：掌板立在端面外侧 30 mm（板 x 跨 0.09–0.185，
  z 跨 0.765–0.806 覆盖端面高度段），指尖伸离盒子，出生零接触。
- 出生穿透 −6.2e−17（精确相切）；静置 400 步：盒 0.029 mm、双手根 0.422 mm、
  **手部 geom 零桌面接触**（实测接触表）。
- 伺服：identity 朝向下 wrist_z 滑轴竖直承重 12.645 N（kp 15000 + 阻尼 280 维持 v1 的整定）。

### 三、门 3 姿态约束（全部可测，写入 `test_scene_w6.py`）

1. **P1 零桌面接触**：任何手部 collision geom 的半径扩展 AABB 不得触及桌板盒体或地面
   （运动学代理；触碰罚 +0.5）。
2. **P2 触点中段**：指垫命中点 z ∈ [盒底 + 0.4×盒高, 盒顶 − 10 mm] = [0.732, 0.770]；
   带外命中罚 +0.5（v1 的顶缘解 0.786 被此条排除）。
3. **P3 掌法向朝端面**：掌板世界法向与其端面内向法向夹角 < 60°（罚 = 0.5 + 角度）。
   深穿透（垫球过面平面超容差）仍按埋入罚。

### 四、门 3-v2 实测数字（离线重扫 + 测试内精修）

| 手 | 触碰垫 | 命中点 (x, y, z) | 间隙 | 掌法向偏差 | 桌面接触 |
|---|---|---|---|---|---|
| 右 | ff_tip | (+0.0599, −0.1319, +0.7648) | −0.116 mm（容差 ±0.1164） | 53.1° | 0 |
| 左 | rf_tip_left | (−0.0719, +0.0421, +0.7656) | −0.116 mm | 57.9° | 0 |

同时配置、同时触碰、零埋入、零桌面接触 → **PASS**。离线扫描解作为精修 SEED 写入测试
（`GATE3_SEEDS`，仅加速收敛——目标函数与全部断言与种子无关，测试文档已注明）。
XML sha256 = `6449871e5dea2dc412d04674d8de9d39cf8a3efbc98ac30f5cf23377d2acadec`（逐字节可复现）。

### 五、两遍构建卫生

`build()` 的 pass-1 占位态（`forcerange="-0 0" kp="1"`）此前直接写 canonical 路径再覆盖，
进程若死于两遍之间就把占位态留在仓库（W6ServoTol 的验证正因此被打断）。v2 起 pass-1 写
**scratch 兄弟路径**（`*.placeholder.xml`），测量在 scratch 上做，只有 pass-2 终稿写 `out`，
写后即删 scratch。字节输出不变（重跑验证 byte-stable）。

### 六、影响面

- `test_scene_w6.py`：门 2 增加手-桌接触零断言；门 3 换姿态约束目标 + 三条新断言
  （中段触碰 / 掌法向 < 60° / 桌面零接触）+ seed 机制；11+1 项全绿。
- 核心回归复跑全绿（scene_w6 12、smoke4_env、smoke4_reward、scene_init 42、p0_plumbing 8、
  pedestal_load 11+2skip、accept 5+2skip、smoke4_demos 14+6skip）。
- 渲染对比：`docs/human_vs_sim_w6a.png`（左列人类视频帧，右列 W6a-v2 三面板：
  静置立姿 / 门 3 触碰姿态 / 右手特写）。

## W6a-v3（2026-10-15）：触面判据改矩形距离 + 前臂 cuff + P4 前向轴 blocked（结构性不可达）

用户亲验再次打回：v2 的"触碰"解里右 ff 垫在 (0.0599, **−0.1319**, 0.7648)——盒子 y 半宽 0.12，
垫心在端面矩形**外 11.9 mm**，是垫球擦过 y=−0.12 侧棱的掠射，不是端面接触；视觉上手指搭过盒顶、
指尖悬在角外。这是同一判据**第三轮窄化**：v1 只测 x 平面距离 → v2 加了半径项但半径吸收了出界量
→ v3 改为**垫心的真·点到矩形距离**（x/y/z 全受矩形约束，出界按 3D 出界距离计，不再有半径项可借）。

### 一、v3 触面判据（实测）

| 手 | 触碰垫 | 垫心 (x, y, z) | 心-矩形距离 | 掌法向 | 桌面接触 |
|---|---|---|---|---|---|
| 右 | th_tip | (+0.06, −0.1127, +0.7423) | **0.000 mm**（容差 ±0.1164） | 10.0° | 0 |
| 左 | mf_tip_left | (−0.06, −0.1191, +0.7669) | **0.046 mm** | 59.8° | 0 |

垫心**落在端面矩形内**、z 在中带、掌法向立着、零桌面接触、零埋入 → PASS。
右手解是**拇指扣端面中段**（掌法向仅偏 10°——几乎正对端面），与视频"拇指搭顶缘"同族。

### 二、P4 前向轴 ≤35°：结构性不可达（blocked，按"宁可报不可达"如实上报）

- 实测：满足 P1–P3 + 矩形判据的全部解里，前向轴（掌心→指尖质心）与端面内向法向夹角
  **右 108.3° / 左 87.7°**（800 重启全空间扫描）——不是差一点，是**差 53–73°**。
- 结构根因：wonik_allegro 模型里 palm body 自带 `quat="0 1 0 1"`，掌板法向与手指伸展轴
  **反平行**（掌法向 = 手系 −x̂，伸展 = +x̂）。于是 P3（掌法向 <60° 偏内向）**恒等蕴含**
  伸展方向 >120° 偏内向——与 P4（≤35°）合并就是要求 180° ≤ 95°，对任何搜索预算都无解。
  人手满足它是因为人手掌法向**垂直**于手指伸展方向；此模型不是。
- 可达解族的量化：垫心在端面矩形内 ⟹ 掌板必须悬在盒顶上方（板内缘伸入盒 x 域但 z 高于盒顶），
  指尖从上方下卷触碰——前向轴必然指向外下（≈108°）。指尖要指向面内（≤35°）必须收拳，
  拳面贴掌 → 指尖落在掌板与盒面之间 ≤5 mm 的缝里，垫心距面 ≥ 2–4 cm，且 95 mm 深的掌板
  必然埋入盒体（埋入判据 0.5 罚）。
- 处置：P4 **未入断言**（不放松、不假绿）；v3 门 = 矩形判据 + P1/P2/P3 + 埋入判据，
  前向角作为证据量随 gate3_qpos.json 落盘。若 W6c/W6d 需要真前向抓，需换手模型或允许
  前臂/腕部偏置安装——场景设计级决定，报主代理裁决。

### 三、其余 v3 项

- **前臂 cuff**（双手 root 各一根视觉-only capsule，`fromto 0 0 0 -0.13 0 0`、半径 24 mm、
  `contype=0 conaffinity=0 mass=0`）：物理零影响——出生 −6.2e−17、静置盒 0.029 mm /
  双手根 0.422 mm / ncon 4，与 v2 位级一致。ngeom 48→89→**91**。
- **ngeom 断言改具名常量 + 加法记账**（`EXPECTED_NGEOM = 91`：84 手 + 2 cuff + 3 盒族 +
  地板/桌板 2，三族独立计数求和必须相等），消灭硬编码地雷。
- **reset 预屈曲 v3**：`[0, 0.7, 0.55, 0.35]×3 + [0.7, 0.45, 0.35, 0.3]`——并拢指向端面，
  静置复验稳定。XML sha256 = `3836ab89789d5d8bd1755c1c0fc42095335b23bd885d42d027c6c1f1a665ae60`。
- 渲染：`docs/human_vs_sim_w6a.png` 已再生（v3 姿态 + v3 门 3 解）。

## W6a-v4（2026-10-15）：机械臂挂架 + 龙门架（用户插队需求）+ P4 裁决落地 + 渲染器修正

### 一、前臂 + 法兰 + 龙门架（全视觉，零物理）

- **前臂组件**（每手一个焊接子 body `forearm_right/left`，随手移动）：圆筒（长 120 mm、
  半径 25 mm，从腕部沿局部 (−1,0,1)/√2 方向即"后上方 45°"伸出）+ 末端法兰盘
  （半径 35 mm、厚 15 mm，同轴）。全部 `mass=0 contype=0 conaffinity=0`。
  替代 v3 的直 cuff（"吸收"）。
- **龙门架**（静态视觉）：x 向顶梁（跨 ±0.5 m，z 1.05）+ y 向吊轨（跨 ±0.55，z 1.015）
  + 两立柱（x=±0.5，落地面）+ 两配重底座。位置按"法兰在滑轴中点时大致指向顶梁"取值。
- **零物理验证**：全部新 geom `mass=0`（body 总质量不含它们，仍 1.289 kg），
  出生 −6.2e−17、静置盒 0.029 mm / 双手根 0.422 mm / ncon 4 —— 与 v3 位级一致。
- **重要教训（回归抓出）**：龙门 geom 最初放在 worldbody 上——env 的落位解析器把
  world-body 盒体一律当支撑面，盒子被"放"到 1.045 m 的顶梁上再摔落，
  `test_scene_init` 16 项 + pedestal P1/g1 连环红。修法：geom 挂在**无关节静态 body**
  （`gantry`）上，支撑检测（world-body 过滤）自动排除，无任何名字过滤 hack。
  "视觉 geom 永远不要直接放 worldbody"入册。

### 二、P4 裁决落地（前向轴 ≤35°）

- P3（掌法向 <60°）断言**删除**（主代理裁决：P3∧P4 由模型解剖决定不可和解，选 P4 弃 P3）。
- P4 **实测后仍 blocked**（未入断言，不假绿）：腕铰链全范围内手指方向最近可达
  **(0, ±0.012, −1)**（竖直向下）——距"35° 下插"目标 (−cos35°, 0, −sin35°) **差 55.2° 腕俯仰**；
  叠加掌板腕-指节 105 mm 阻挡（滑台贴面必埋掌板 83 mm）。
  矩形判据可达解的前向轴实测 **右 108.3° / 左 87.7°**（差 73.3°/52.7°）。
  前向角随 `runs/w6/gate3_qpos.json` 落盘为证据；掌法向（已降级为证据量，
  实测 10.0°/59.8°——直立解族的掌法向其实很好）同样落盘。
- 取舍说明：全关节机械臂 = 场景换代，本轮不做（已向用户说明）；本模型的"端面抓取"
  机器人等价物 = 立姿 + 掌侧向 + 手指从上方下卷钩面（v3/v4 解族），已固化。

### 三、其余 v4 项

- **reset 预屈曲生效性复核**：env reset + 400 步 settle 后最低指/中节 geom 清零高度
  **133–147 mm**（≥15 mm 要求的 9 倍），手-桌接触 0。（主代理看到的"趴桌"渲染来自
  v3 面板脚本的两个 bug：物体关节置零陷阱 + v2 图注——均已修，见下。）
- **渲染器修正**：对比图脚本每帧显式放置物体 freejoint（v3 的中行面板就是栽在
  置零物体上），图注更新 W6a-v4。`docs/human_vs_sim_w6a.png` 已再生并自查。
- ngeom **99**（84 手 + 4 前臂/法兰 + 3 盒族 + 2 支撑 + 6 龙门）、nbody 49；
  `EXPECTED_NGEOM = 99` + 三族加法记账；smoke6 计数同步。
- XML sha256 = `88f2d8db42483aa45000051df43c55a4d8d752baff258b96a04eee69e601127e`（逐字节可复现）。
- 电池：核心 9 套件 **120 passed / 8 skipped / 2 failed**（2 failed = R9 既有
  MJCF-basis 红灯，记录值 4ee6f1d vs 现场 commit 号，随 commit 漂移，非本轮引入）。

## W6a-v5（2026-10-15）：安装朝向重开——双手 180° yaw、P4 升门、统一 reset 路径

主代理裁决：三轮被冻结的变量是**手的安装朝向**——v2/v3/v4 全在"掌贴面"朝向下搜关节空间，
而模型解剖（掌法向反平行于手指伸展轴）注定该族不可达 P4。v5 重开朝向：

### 一、安装朝向 180° yaw（场景级一行/手）

- **右手 euler `0 0 π`、左手 `0 0 0`**：手指伸展轴指向盒心；**手背朝相机**——与视频同构。
- **P4 随之自然满足**：掌心→指尖质心的前向轴在 180° yaw 下指向面内。
  实测门 3（P4 ≤35° 升为断言）：右 rf 垫 (0.06, 0.0667, 0.7509) 矩形距离 **0.000 mm**、
  前向 **28.8°**；左 mf 垫 (−0.06, −0.074, 0.7631) **0.000 mm**、前向 **17.2°**——
  双手同时、中段带内、零桌面接触、零埋入 → **PASS**。
- 掌向完全自由（v5 裁决覆盖 v4 的 <120°）：实测掌法向 104.6°/141.7°（背对相机，
  恰是视频正确的样子），纯证据量落盘 gate3_qpos.json。

- **统一 reset 路径（裸 settle == env.reset）**

v4 报告的"净空 133-147 mm"是 env.reset 路径；裸 MjData settle（qpos0 + ctrl 0）仍是直指姿态——
两条路径不同。修正机制用 **关节弹簧**（`stiffness=0.4` + `springref`=各指屈曲角）而非 joint `ref`：
`ref` 只平移关节值（编译姿态仍是直指），表达不了"默认即屈曲"；弹簧让**两条 settle 路径
都收敛到同一屈曲家位**（实测裸 settle 屈曲 0.15-0.27 rad；env.reset 经 ctrl 镜像收敛到
springref 附近）。位置伺服仍是主动控制（reset 先 settle 后镜像，弹簧无对抗）；
执行器 kp/forcerange 未动。指弹簧（0.4 N·m/rad ≈ 指重力矩的 5-20 倍）是新的被动常量，
已记入本节、知会 W6ServoTol。

### 三、home/净空（v5 朝向 + 终版 home ±0.235 上实测）

- `HAND_HOME = ±0.235, 0, 0.755`：直指位指尖距端面矩形 ~23 mm（出生零接触，仅盒 4 条精确相切）。
- **裸 settle**：指尖 z 0.782–0.789，离桌 **82–89 mm**；弹簧卷起 0.15–0.27 rad（弹簧-重力平衡，
  指尖上卷背离桌面）。
- **env reset**：指尖 z 0.800–0.839，离桌 **100–139 mm**。两条路径视觉同族（立姿、指卷向面）。
- 门 2 新增**双手 8 指尖净空 ≥ 50 mm 断言** + 手-桌零接触断言（v2 起）。
- 流程更正：本节初稿误记 sha `b17544ab…`（home 终版 ±0.235 定稿**前**的中间态构建）。
  正确值 = 提交树内 blob：**`7b7a1adf…`**（git cat-file 复核）。流程新规已执行：
  落地 sha 必须对象库复核后报告。

### 四、C6 控制实验 superseded（结构性前提消失）

`test_c6`（"停车位盒子在扑腾下不动"）实测被左**掌**横扫 15 mm（seed 3/7/11/13，零手指接触）：
双手（根 ±0.235 + 滑轴 ±0.18 + 指伸 0.13）的扫掠体积**覆盖整个桌面**——
"无手区域"在双手场景里结构性不存在，C6 的落位式前提失效。
其守护的性质（盒子运动必须接触驱动）由冻结的验收门接触证据判据（A2/A3a）与
`W_OFF_TABLE` 奖励项继续承担；按 W6c 方向改造为逐手接触归因检查。已注记 superseded。

### 五、W6a-v17 追加（per-body 分布：手侧臂段 + 龙门侧球壳 junction）

- **审计驱动重设计**：390 帧 replay 包络实测——right 手 max |Δxyz| = 0.2157 m、left = 0.1805 m。
  v16 静态 Z 台/肩/前臂被替换为**龙门 body 上的包络尺寸球壳 junction**
  （right R=0.220 m > 0.2157、left R=0.190 m > 0.1805）。
- **手侧臂段**（挂 hand_root，跟手动）：forearm → elbow → upper arm 各手 3 geom。
  upper 臂尖在每帧 replay 都落在龙门球壳内——**junction 覆盖：right worst 207.6 mm < 220 mm（余 12.4 mm）、
  left worst 145.2 mm < 190 mm（余 44.8 mm）**，ALL PASS。
- **零物理**：所有新 geom mass=0/contype=0。出生/静置/门位级一致。
- ngeom 143 / nbody 51 / gantry 族 24 / 手侧臂段族 6。

### 六、W6a-v17.1（视觉定版：球壳覆盖任务带验证）

- **任务带覆盖程序化验证**（390 帧 replay，臂尖-球壳距）：
  - right f0-189 主带 worst **207.6 mm** < R 220 mm（余 12.4 mm）✓ 全覆盖
  - left  f0-189 主带 worst **127.5 mm** < R 190 mm（余 62.5 mm）✓ 全覆盖
  - 尾段 f190-389 也全覆盖（right 179.5 mm、left 145.2 mm）——允许出壳但实际未出，保守覆盖
- **无需调参**：R=0.220/0.190 已满足"主带全覆盖 + 尾段允许出壳"双条件；W6d 壳体样式（接缝环+顶盖螺栓+金属臂段）未动。
- **电池**：83 passed / 1 skipped（scene_w6 13/13 + smoke4_env + reward + scene_init）。
- **视觉线 16 轮定版**：v17 结构 + 任务带全覆盖 + 尾段诚实注记。

### 七、W6a-v19（3-DoF env 驱动铰接臂 — 解析 IK 伺服，checkpoint 全兼容）

- **结构**：每臂 3 铰（yaw + pitch + elbow），肩锚在龙门 (±0.235, 0, 1.000)。
  连杆 2 段 capsule（L1=L2=0.16 m，10 g/段 + 2 个 5 g 关节盘），臂尖 22 mm 球。
  6 个 position servo（kp=60, forcerange ±8 N·m, ctrlrange ±3.2）。
- **IK**：env 侧每 step 解析 3-DoF IK（`yaw=atan2(Δy,Δx)` + 平面内二连杆余弦定理），
  写 ctrl[44:50]。策略 action 仍 44 维 / obs 109 维零变化——**已训 checkpoint 全部有效**。
- **残差**：390 帧 replay，right/left worst tip-to-wrist = **0.0000 mm**（运动学直写）。
  闭环伺服稳态 0.52 mm（reset 后 ~100 ms 摆入暂态，稳态紧跟踪）。
- **涟漪记账**：nq 51→57、nv 50→56、nu 44→50、ngeom 143→145（v17 手侧臂 6 geom 移除，
  v19 臂 10 geom + junction 球 2 移除 + gantry 网格不变）、nbody 51→55。
  gate3_qpos.json 布局 51→57 wide（reader 偏移 13→19、probe 偏移 7→13 全部同步）。
  test_scene_init 的 qpos[:3] 硬编码改 obj_qpos_adr 寻址。
- **y 包络前置测量**：right 180.8 mm / left 122.9 mm ≫ 40 mm → 2 铰不可行，
  3 铰方案经 Main 批准（A 裁决）。reach 需求 right 0.3175 m / left 0.2547 m → L=0.16 m。
- **遗留（预存，HEAD 即挂，非 v19 引入）**：
  1. `test_r1_forcerange_is_the_shared_budget_times_the_moment_arm`——v12 引入 1.7
     forcerange 缩放后恒等式测试未同步（模型值 0.17919 = 1.7 × 0.10541 正确，测试期望未缩放值）。
  2. `test_revision_md_records_the_forcerange_table`——本文档 v17.1 中文追加后
     力矩臂表匹配失败。

### 八、W6a-v20（被动随动垫 — 接触下自对中，checkpoint 兼容）

- **机制**：ff/mf 四垫（rf/th 保持刚性）从 tip 刚性挂载改为独立 pad body +
  2 正交 hinge（stiffness 0.02 N·m/rad、damping 0.002、range ±0.6、springref 0，
  垫 geom 5 g）。弹簧rest=挂载系——gate3 工作位形下即已证面平行姿态；
  接触法向力矩 ≫ 轻弹簧（1.5N×3mm=0.0045 ≫ 0.02×偏差），卷曲旋转被被动铰吸收。
- **机制验收（r7e 复测条件）**：压紧 N 3.8/4.8 N（v19 同 harness 的 3 倍——自对中贴合），
  +0.20 rad 卷曲需求下 N 持 2.58/4.44 N **不塌零**（v20 前 r7e 力模式 N 崩 0.00），
  pad 铰偏转 38–49 mrad 吸收旋转 ✓。抬升传递由 W6c 力控机器复跑（N* 1.5N/垫 规格已定）。
- **零策略涟漪**：ctrl 50 / obs 109 不变，checkpoint 有效（v19 先例）。
- **涟漪记账**：nq 57→65（4 pad × 2 hinge）、nv 56→64、nbody 55→59（ngeom 145 不变——
  geom 改挂不改名）。gate3_qpos.json 65-wide（reader 全部切 jnt_qposadr 名字寻址，
  scene_w6 gate3b + smoke6 probe）；
  `_pad_checkpoint` 重写为名字映射（51-wide checkpoint → 65-wide，pad 槽补零）。
- **已知布局隐患（W6c 已修其两处，此处重申）**：任何绝对 `qpos[0:7]` 写在 65-wide 上
  错写臂铰——统一走 `obj_qpos_adr`。

### 九、W6a-v20.1（随动垫 rest-pose 预紧 — settle 楔高修复）

- **现象**：随动垫重力下垂 −46.6 mrad（四垫一致，pad_rx 轴）在 settle 楔入盒足迹下方，
  W6c 训练位姿 settle 盒心读到 0.753（高于静置面 13mm），dz 参考面失准。
- **修复**：pad_rx springref 预紧 **+0.047 rad**——重力 sag 恰好归零（rest = 挂载位形 =
  已证刚垫几何）。ry 不受重力（实测 ±0.2 mrad），预紧不变。
- **验证**：XML settle 盒 z 0.7400/漂移 0.029mm；**训练位姿 settle 盒心 0.7400**（楔高判据
  ±0.1mm 过，修前 0.753）；r7e 机制复测：压紧 N 3.80/4.72 → +0.20 rad 卷曲需求 3.62/4.33 N
  不塌——**预紧未吃自对中裕度，工作点 pad 偏角 38-49 → 17-18 mrad 反而改善**。
- 自对中裕度算术：预紧只移 rest 点不移 authority（接触力矩 0.0045 / 0.02 = 225 mrad 偏转
  预算不变）。

### 十、状态

- ngeom 99 / nbody 49；XML sha256 =
  XML sha256 = `7b7a1adffab68f0156297f5e654d72b9ef01bc4036874e34870027ffc11563dc`（git cat-file 对象库复核；初稿误记中间态 b17544ab，见 §三流程更正）。
- 核心电池：scene_w6 12、smoke4_env（C6 superseded 外全绿）、smoke4_reward、scene_init 42、
  p0_plumbing 8、pedestal_load 11+2skip、accept 5+2skip、smoke4_demos 14+6skip+2 既有红灯。
- 渲染：`docs/human_vs_sim_w6a.png`（v5 图注、每帧置物、龙门入画、指尖离桌可见余量）。

## W6a-v6（2026-10-15）：机械结构可信度返工（GPT 外部评审 4.3/10 REWORK 的采纳清单）

视觉-only 轮：物理零变化红线不变（出生 −6.2e−17、静置盒 0.029 mm / 根 0.42 mm / ncon 4，
与 v5 位级一致；34 个挂架/龙门 geom 全部 mass=0 无碰撞，逐 geom 断言）。
XML sha256 = `c0114a640011b160900e2c38354c8c93475535924e81110172c8c02079f58928`
（git cat-file 对象库复核后记录）。

### 一、采纳清单

- **P0-1 杀 45° 圆筒**：v4 的长 capsule+法兰读作"炮管"，暗示不存在的手腕自由度。
  替换为**短腕部安装模块**（每手焊接子 body `wmount_right/left`）：安装板（90×70×35 mm）
  + 法兰盘（**直径 60 mm = 掌宽 113 mm 的 0.53 倍**，带 6 螺栓圈）+ 端盖 + 出线口，
  总长 55.5 mm ≤ 60 mm，水平直连手背侧——读作"末端执行器法兰"。每手 10 geom ×2。
- **P0-3 承力链可读**：龙门加**滑块小车**（骑 y 吊轨）+ **Z 向升降台**（正对右手腕模块上方，
  与随身模块行程内贴合不脱接）+ **拖链**（4 节，从升降台到横梁）+ **限位挡块**（吊轨两端）。
  观察者可沿"手→法兰→腕部模块→Z 台→滑块→吊轨→横梁→立柱→底座"追踪承力链。
- **P1-4 比例**：外挂件以真实 Allegro 尺寸（掌宽 ~113 mm）为基准取 0.53×，手不缩。
- **gate3b（P0-2 运动学部分）**：矩形判据下实测每手垫分布——**左手演示了 2 垫同入矩形**
  （mf 0.033 mm + ff 0.051 mm，中段带内，前向 23.4°，零桌触）；右手本轮搜索收敛到 1 垫
  （mf 0.049 mm ON-face），其 2 垫镜像种子映射的欧拉符号在首次尝试中用错（已撤）。
  处置：门 3 断言保持"每手 ≥1 垫"（双手 PASS）；左手 2 垫证明该运动学族存在（同构手镜像可达），
  右手的 ≥2 收敛留给 W6b 轨迹（MANO 动态闭合不必静态强行）。**矩形判据一字未放松。**

### 二、驳回/缓议清单（记档）

- 主动腕部姿态模块（新 DoF = 场景换代；GPT 自标注固定朝向任务可不用）。
- 6 轴双臂（长期选项）。
- 工业材质/盒子贴图精修（GPT 排序"结构过了再精修"；盒子贴图归 W6d）。

### 三、回归与记账

- `EXPECTED_NGEOM = 123`（84 手 + 20 腕部模块 + 3 盒族 + 2 支撑 + 14 龙门），三族加法记账；
  smoke6 计数同步（ngeom 99→123、nbody 49）。
- 核心电池（scene_w6 12 / smoke4_env / smoke4_reward / scene_init 42）全绿；
  既有红灯 2（R9 MJCF-basis）不变。
- 对比图 `docs/human_vs_sim_w6a.png` 已再生并自查（承力链可读、无炮管、比例正常）。

## W6a-v7（2026-10-15）：GPT 二轮 6.2/10 的采纳批（纯视觉，零物理）

物理零变化红线照旧：出生 −6.2e−17、静置盒 0.029 mm / 根 0.42 mm / ncon 4，与 v6 位级一致。

### 一、采纳（本批）

- **龙门设备轮廓完整化（GPT 二轮 P0-2）**：三层一眼可读——立柱+底座 / X 主横梁+y 吊轨 /
  **左右各自 Y/Z 末端模组**。"共梁双滑块"的物理事实已建成视觉答案：**一条 y 吊轨上两只滑块小车**
  （±0.20），各带端盖、Z 升降台、桥臂下探到各自法兰区、左拖链 2 节；吊轨两端限位挡块。
  "滑轨+滑块+端盖+拖链"四件套在侧视一位可见。
- **掌根过渡连接板（P0-1 残余）**：腕部模块与掌根之间加转接板（38×56×20 mm 视觉盒），
  手不再"从黑块里长出来"。
- **法兰细节辨识度**：法兰基座加**台阶环**（ø68 薄环），螺栓加大加深（r 5 mm、深色），
  螺帽/台阶阴影一眼可辨。
- **指尖白球降调（P1-4 残余）**：8 个指尖碰撞胶囊 rgba → 灰白 (0.72, 0.72, 0.76)。
  **只改 rgba，尺寸/位置未动**（碰撞物理冻结）。

### 二、押后（等 W6b 落地后第二批，理由随附）

- 捏持姿态精修（主夹持三角、无名指/小指收拢、接触抬高）——动 springref/家位 = 物理变化，
  会作废 W6b 在跑链；W6b 交付后与其发现合并一批。
- 右手 2 垫入矩形 seed——同上（左手 2 垫已演示可达，见 R10-v6 gate3b）。
- 接触中高位（GPT 观感"偏低"）——中带 [0.732, 0.770] 判据先于观感，等轨迹段同看。

### 三、驳回（记档，含依据）

- 腕部 yaw+pitch 自由度：场景换代（表/retarget/IK 全重建）；GPT 自写明"只做这个盒体任务"，
  本任务 = 固定朝向盒。记为任务多样化后的代际选项。
- 掌心正对端面：**结构性不可能**——v3 已证此模型掌法向（palm quat "0 1 0 1" ⟹ 手系 (−1,0,0)）
  与手指伸展轴**反平行**；且视频里人手本就手背朝相机。GPT 该条与解剖证据冲突，以证据为准。

### 三-b、gate3_qpos.json 的生成方式与入库（W6b 锚定件）

- **生成**：`tests/test_scene_w6.py::test_w6_gate3_both_hands_reach_their_end_faces`
  每次运行用其自带的 gate3b 搜索（第二小矩形距离目标 + P4/桌触/埋入约束 + 离线种子）
  重搜并**原子落盘**该文件；本次入库版本 = 2026-10-15 该测试的验证通过态
  （双手各 2 垫：右 ff 4.7 µm + mf 17.0 µm；左 ff 50.0 µm + mf 16.4 µm，
  皆在 ±0.06 面矩形、z 中带、容差 116 µm 内）。
- **入库**：`.gitignore` 白名单 `!runs/w6/gate3_qpos.json`（先例 mano_params.npz）。
- **消费者**：W6b 三轮终态 blend 直接锚此文件；重渲面板亦读它。

### 四、W6a-v16 追加（臂段重分布 + 世界体 Z 台/肩/前臂）

- **旧 v15 臂体（arm_right/arm_left，5 段焊在手 root）全部移除**——手动臂整体动、
  导轨不动 → 滑车离轨 0.3–0.5 m（W6d 程序化审计）。
- **替换**：龙门 body 上加每手 3 段静态 Z 台+肩+前臂（从 x-beam 下垂到腕部上方），
  与手侧 wmount 在回放位形重叠——接触链：手 → wmount → 静态前臂 → 肩 → Z 台 → 横梁。
- ngeom 141 / nbody 49 / gantry 族 28 geom。

### 五、W6a-v17 追加（per-body 分布：手侧臂段 + 龙门侧球壳 junction）

- **审计驱动重设计**：390 帧 replay 包络实测——right 手 max |Δxyz| = 0.2157 m、left = 0.1805 m。
  v16 静态 Z 台/肩/前臂被替换为**龙门 body 上的包络尺寸球壳 junction**
  （right R=0.220 m > 0.2157、left R=0.190 m > 0.1805）。
- **手侧臂段**（挂 hand_root，跟手动）：forearm → elbow → upper arm 各手 3 geom。
  upper 臂尖在每帧 replay 都落在龙门球壳内——**junction 覆盖：right worst 207.6 mm < 220 mm（余 12.4 mm）、
  left worst 145.2 mm < 190 mm（余 44.8 mm）**，ALL PASS。
- **零物理**：所有新 geom mass=0/contype=0。出生/静置/门位级一致。
- ngeom 143 / nbody 51 / gantry 族 24 / 手侧臂段族 6。

### 六、W6a-v17.1（视觉定版：球壳覆盖任务带验证）

- **任务带覆盖程序化验证**（390 帧 replay，臂尖-球壳距）：
  - right f0-189 主带 worst **207.6 mm** < R 220 mm（余 12.4 mm）✓ 全覆盖
  - left  f0-189 主带 worst **127.5 mm** < R 190 mm（余 62.5 mm）✓ 全覆盖
  - 尾段 f190-389 也全覆盖（right 179.5 mm、left 145.2 mm）——允许出壳但实际未出，保守覆盖
- **无需调参**：R=0.220/0.190 已满足"主带全覆盖 + 尾段允许出壳"双条件；W6d 壳体样式（接缝环+顶盖螺栓+金属臂段）未动。
- **电池**：83 passed / 1 skipped（scene_w6 13/13 + smoke4_env + reward + scene_init）。
- **视觉线 16 轮定版**：v17 结构 + 任务带全覆盖 + 尾段诚实注记。

### 七、W6a-v19（3-DoF env 驱动铰接臂 — 解析 IK 伺服，checkpoint 全兼容）

- **结构**：每臂 3 铰（yaw + pitch + elbow），肩锚在龙门 (±0.235, 0, 1.000)。
  连杆 2 段 capsule（L1=L2=0.16 m，10 g/段 + 2 个 5 g 关节盘），臂尖 22 mm 球。
  6 个 position servo（kp=60, forcerange ±8 N·m, ctrlrange ±3.2）。
- **IK**：env 侧每 step 解析 3-DoF IK（`yaw=atan2(Δy,Δx)` + 平面内二连杆余弦定理），
  写 ctrl[44:50]。策略 action 仍 44 维 / obs 109 维零变化——**已训 checkpoint 全部有效**。
- **残差**：390 帧 replay，right/left worst tip-to-wrist = **0.0000 mm**（运动学直写）。
  闭环伺服稳态 0.52 mm（reset 后 ~100 ms 摆入暂态，稳态紧跟踪）。
- **涟漪记账**：nq 51→57、nv 50→56、nu 44→50、ngeom 143→145（v17 手侧臂 6 geom 移除，
  v19 臂 10 geom + junction 球 2 移除 + gantry 网格不变）、nbody 51→55。
  gate3_qpos.json 布局 51→57 wide（reader 偏移 13→19、probe 偏移 7→13 全部同步）。
  test_scene_init 的 qpos[:3] 硬编码改 obj_qpos_adr 寻址。
- **y 包络前置测量**：right 180.8 mm / left 122.9 mm ≫ 40 mm → 2 铰不可行，
  3 铰方案经 Main 批准（A 裁决）。reach 需求 right 0.3175 m / left 0.2547 m → L=0.16 m。
- **遗留（预存，HEAD 即挂，非 v19 引入）**：
  1. `test_r1_forcerange_is_the_shared_budget_times_the_moment_arm`——v12 引入 1.7
     forcerange 缩放后恒等式测试未同步（模型值 0.17919 = 1.7 × 0.10541 正确，测试期望未缩放值）。
  2. `test_revision_md_records_the_forcerange_table`——本文档 v17.1 中文追加后
     力矩臂表匹配失败。

### 八、W6a-v20（被动随动垫 — 接触下自对中，checkpoint 兼容）

- **机制**：ff/mf 四垫（rf/th 保持刚性）从 tip 刚性挂载改为独立 pad body +
  2 正交 hinge（stiffness 0.02 N·m/rad、damping 0.002、range ±0.6、springref 0，
  垫 geom 5 g）。弹簧rest=挂载系——gate3 工作位形下即已证面平行姿态；
  接触法向力矩 ≫ 轻弹簧（1.5N×3mm=0.0045 ≫ 0.02×偏差），卷曲旋转被被动铰吸收。
- **机制验收（r7e 复测条件）**：压紧 N 3.8/4.8 N（v19 同 harness 的 3 倍——自对中贴合），
  +0.20 rad 卷曲需求下 N 持 2.58/4.44 N **不塌零**（v20 前 r7e 力模式 N 崩 0.00），
  pad 铰偏转 38–49 mrad 吸收旋转 ✓。抬升传递由 W6c 力控机器复跑（N* 1.5N/垫 规格已定）。
- **零策略涟漪**：ctrl 50 / obs 109 不变，checkpoint 有效（v19 先例）。
- **涟漪记账**：nq 57→65（4 pad × 2 hinge）、nv 56→64、nbody 55→59（ngeom 145 不变——
  geom 改挂不改名）。gate3_qpos.json 65-wide（reader 全部切 jnt_qposadr 名字寻址，
  scene_w6 gate3b + smoke6 probe）；
  `_pad_checkpoint` 重写为名字映射（51-wide checkpoint → 65-wide，pad 槽补零）。
- **已知布局隐患（W6c 已修其两处，此处重申）**：任何绝对 `qpos[0:7]` 写在 65-wide 上
  错写臂铰——统一走 `obj_qpos_adr`。

### 九、W6a-v20.1（随动垫 rest-pose 预紧 — settle 楔高修复）

- **现象**：随动垫重力下垂 −46.6 mrad（四垫一致，pad_rx 轴）在 settle 楔入盒足迹下方，
  W6c 训练位姿 settle 盒心读到 0.753（高于静置面 13mm），dz 参考面失准。
- **修复**：pad_rx springref 预紧 **+0.047 rad**——重力 sag 恰好归零（rest = 挂载位形 =
  已证刚垫几何）。ry 不受重力（实测 ±0.2 mrad），预紧不变。
- **验证**：XML settle 盒 z 0.7400/漂移 0.029mm；**训练位姿 settle 盒心 0.7400**（楔高判据
  ±0.1mm 过，修前 0.753）；r7e 机制复测：压紧 N 3.80/4.72 → +0.20 rad 卷曲需求 3.62/4.33 N
  不塌——**预紧未吃自对中裕度，工作点 pad 偏角 38-49 → 17-18 mrad 反而改善**。
- 自对中裕度算术：预紧只移 rest 点不移 authority（接触力矩 0.0045 / 0.02 = 225 mrad 偏转
  预算不变）。

### 十、状态

- ngeom **135**（84 手 + 24 腕部模块/过渡/台阶 + 3 盒族 + 2 支撑 + 22 龙门）、nbody 49；
  `EXPECTED_NGEOM = 135` + 族加法记账；smoke6 计数同步。
- 核心电池：scene_w6 12/12、smoke6 快子集、smoke4_env、scene_init 全绿。
- XML sha256 = `60ba2c3be593a5e183064c0fdf38154747de027c1cbc1856435ea1008e913a08`
  （**git cat-file 对象库复核后记录**，流程新规）。
- 渲染：`docs/human_vs_sim_w6a.png`（v7 图注、Main 实测机位）已再生并自查：
  承力链可读、无炮管、比例正常、指尖降调可见。

## W6a-v8（2026-10-15）：指弹簧静态可持性修正（W6b finding 1）+ "运动学可达 ≠ 动态可持"

W6b 轨迹执行带回发现 1（逐关节证据 runs/w6b/report.json）：v5 的指弹簧（k 0.4, ref 0.6）
在深卷曲 q=1.3 处产生 **0.279 N·m 回开力矩**——超过 ffj2 静态力帽 0.094 的 3 倍；
执行态双手都卷不到 seed 深度（左手垫差 44.9 mm 的根因）。门 3 的 qpos 是静态 compose
（mj_forward），从未验过"伺服+弹簧力平衡下可持"——**运动学可达 ≠ 动态可持，判据第三次窄化**。

### 一、修正（Main 选项 b+a 组合）

- **springref = 门 3 解的逐关节双手均值，clip 进各自关节行程**（初版均值未 clip，
  rfj0 均值 0.972 超出 ±0.47 行程 2 倍、把外展关节顶死在限位上——已修正）
  ——弹簧在文档位形处**零需求**（助握不抗握，选项 b）。
- **stiffness 逐关节化**（选项 a）：共享常数被双手 seed 分歧卡死
  （rfj0 双手 seed 差 0.651 rad，共享 ref 可行 k ≤ 0.019 且仍顶限位），
  改为每关节独立：k_j = min(0.09, max(0.01, (0.8×cap − 重力)/|seed 偏差|))。
  执行器 kp/forcerange 未动。
- **下表为定稿树 `eba5d7b` 程序化再生**（mj_model.jnt_stiffness / qpos_spring 逐关节读出，
  非手抄；k 范围 0.005–0.09，W6ServoTol 独立实测一致）：

```
ffj0 0.0318 -0.1440
ffj1 0.0900 +0.6300
ffj2 0.0900 +0.5800
ffj3 0.0251 -0.0140
mfj0 0.0050 +0.1320
mfj1 0.0739 +0.4660
mfj2 0.0850 +0.6400
mfj3 0.0315 +0.4490
rfj0 0.0104 +0.1010
rfj1 0.0445 +1.0570
rfj2 0.0900 +0.7300
rfj3 0.0552 +0.6520
thj0 0.0900 +0.6060
thj1 0.0900 +0.5730
thj2 0.0100 +0.9380
thj3 0.0100 +0.4070
```
- **流程更正（第三次"报非定稿态"）**：v2 的 nu 计数、v5 的 sha、v8 的本表——
  三次同病：报告数值从中间态手抄。已立规矩：**落地报文的数值一律从定稿树程序化再生**
  （脚本读树出表），禁止手工转抄。
- **统一 reset 保持**：两条 settle 路径仍收敛同一平衡（弱弹簧 + 伺服主导，指伸展限位附近，
  指尖直指端面）；**指尖净空实测 51.1 mm ≥ 50 mm 断言不回退**。
- 代价（如实）：家位屈曲从 v5 的视觉卷曲退为"直指端面"（弱弹簧压不过重力）——
  视频的"手指张开压面"由 W6b 动态闭合实现，静态家位是接近姿态。

### 二、新冻结门 static hold（入 test_scene_w6，永久）

在 gate3_qpos.json 构型逐指关节断言：**|k(q_seed−ref)| ≤ 0.8×力帽**（弹簧需求占帽 80% 以内，
20% 余量）；重力作为独立常量负载另行记录（evidence json）。Gravity-only 例外：
mfj0_left 在该位形的重力矩 0.0258 = 力帽的 86%（>80% 预算）——纯重力、与弹簧无关，
已记为该关节该位形的边缘案例归 W6c。实测全部 32 关节：最大弹簧需求/帽 ≈ 0.31 → **PASS**。

### 三、数字

- 出生 −6.2e−17；静置盒 0.029 mm（位级不变）；**指尖净空 56.7–59.8 mm**
  （弱弹簧下直指端面，≥50 断言 ✓）；settled 关节出程巡检：越程 = none。
- XML sha256 = `8b6a6ddd1b0df4f7d1b16df18263af6093a48eba31d7818630f44d1677dc11b0`
  （**git cat-file 对象库复核后记录**，流程新规；v8 首报的 `aba32203…` 为
  refs/stiffness 修正前的中间态，已作废重报——"先量后改再报"的教训入册）。
- 电池：核心 9 套件 120 passed / 11 skipped / 2 failed（R9 既有 MJCF-basis 红灯，
  记录值随 commit 号漂移）+ scene_w6 13/13（新增 static hold 门）。
- 渲染自查：直指位 + cuff/模块/龙门入画。

## W6a-v9（2026-10-15）：hold 门补重力项（第五次判据窄化修正）+ 左手 seed 重搜

W6b 二轮 finding F4：v8 的 static-hold 门**只验弹簧需求、重力"另列 evidence"**——
又一次判据窄化。held seed 下左掌精确在位、但左手指垂落压桌（393 mrad 越限）：
左 seed 腕 rx +0.37（右手 −1.67）使卷曲平面朝下，指伺服重力预算缺口。

### 一、hold 门补重力（判据修正）

`test_w6_gate3b_static_hold` 改为 **|k(q−ref)| + τ_gravity ≤ 0.8×cap**
（τ_gravity 用 qfrc_bias 在 seed 构型取）。修正后现有左手 seed **如实变红**
（F4 的量化表达——正是门该干的）。

### 二、左手 seed 重搜（右手 seed 不动）

- 左 seed 注入 rx 中性族偏置（−1.05，参照右手 −1.67 成功家族）为精修种子，
  在 **hold-含重力 + 矩形中带 + 前向 ≤35° + 零桌触 + 零埋入** 全约束下重搜。
- **结果：左手全过且 2 垫**——mf_tip_left (−0.06, −0.1105, 0.7596) 0.000 mm、
  ff_tip_left (−0.0601, −0.1016, 0.7419) 0.051 mm，前向 23.4°，零桌触零埋入。
- **右手 seed 不动**（Main 裁决：W6b 动态执行已证可持）；其 static-hold 含重力的
  超标（rfj0 弹簧+重力 0.0271 vs 0.0240）作为**已记录偏差**随 gate3_qpos.json 落盘
  ——W6b 的动态混合在执行态持有它，静态 20% 余量口径下的超标留给 W6c 复核。
- mfj0_left 86% duty 案例：新 seed 下**已顺带解决**（重搜位形的 mfj0_left 重力负载
  落回预算内），从 W6c 移交清单撤下。

### 三、复盘：第五次判据窄化

| 轮 | 判据 | 漏掉什么 |
|---|---|---|
| v1 | 平面 x 距离 | y 出界掠棱 |
| v2 | +半径 | 半径吸收出界量（角外 11.9 mm 仍"触碰"） |
| v3-v4 | 垫心矩形距离 | 静止可达 ≠ 可视正确（趴平解）→ 姿态约束 |
| v8 | 弹簧需求 ≤ 0.8cap | **重力项另列**——弹簧+重力合计才算伺服需求 |
| v9 | +重力项 | （本修正；左手 seed 重搜后全过） |

教训固化为门内的固定组成：**伺服需求 = 弹簧 + 重力，一次算清**。

### 三-b、gate3_qpos.json 的生成方式与入库（W6b 锚定件）

- **生成**：`tests/test_scene_w6.py::test_w6_gate3_both_hands_reach_their_end_faces`
  每次运行用其自带的 gate3b 搜索（第二小矩形距离目标 + P4/桌触/埋入约束 + 离线种子）
  重搜并**原子落盘**该文件；本次入库版本 = 2026-10-15 该测试的验证通过态
  （双手各 2 垫：右 ff 4.7 µm + mf 17.0 µm；左 ff 50.0 µm + mf 16.4 µm，
  皆在 ±0.06 面矩形、z 中带、容差 116 µm 内）。
- **入库**：`.gitignore` 白名单 `!runs/w6/gate3_qpos.json`（先例 mano_params.npz）。
- **消费者**：W6b 三轮终态 blend 直接锚此文件；重渲面板亦读它。

### 四、W6a-v16 追加（臂段重分布 + 世界体 Z 台/肩/前臂）

- **旧 v15 臂体（arm_right/arm_left，5 段焊在手 root）全部移除**——手动臂整体动、
  导轨不动 → 滑车离轨 0.3–0.5 m（W6d 程序化审计）。
- **替换**：龙门 body 上加每手 3 段静态 Z 台+肩+前臂（从 x-beam 下垂到腕部上方），
  与手侧 wmount 在回放位形重叠——接触链：手 → wmount → 静态前臂 → 肩 → Z 台 → 横梁。
- ngeom 141 / nbody 49 / gantry 族 28 geom。

### 五、W6a-v17 追加（per-body 分布：手侧臂段 + 龙门侧球壳 junction）

- **审计驱动重设计**：390 帧 replay 包络实测——right 手 max |Δxyz| = 0.2157 m、left = 0.1805 m。
  v16 静态 Z 台/肩/前臂被替换为**龙门 body 上的包络尺寸球壳 junction**
  （right R=0.220 m > 0.2157、left R=0.190 m > 0.1805）。
- **手侧臂段**（挂 hand_root，跟手动）：forearm → elbow → upper arm 各手 3 geom。
  upper 臂尖在每帧 replay 都落在龙门球壳内——**junction 覆盖：right worst 207.6 mm < 220 mm（余 12.4 mm）、
  left worst 145.2 mm < 190 mm（余 44.8 mm）**，ALL PASS。
- **零物理**：所有新 geom mass=0/contype=0。出生/静置/门位级一致。
- ngeom 143 / nbody 51 / gantry 族 24 / 手侧臂段族 6。

### 六、W6a-v17.1（视觉定版：球壳覆盖任务带验证）

- **任务带覆盖程序化验证**（390 帧 replay，臂尖-球壳距）：
  - right f0-189 主带 worst **207.6 mm** < R 220 mm（余 12.4 mm）✓ 全覆盖
  - left  f0-189 主带 worst **127.5 mm** < R 190 mm（余 62.5 mm）✓ 全覆盖
  - 尾段 f190-389 也全覆盖（right 179.5 mm、left 145.2 mm）——允许出壳但实际未出，保守覆盖
- **无需调参**：R=0.220/0.190 已满足"主带全覆盖 + 尾段允许出壳"双条件；W6d 壳体样式（接缝环+顶盖螺栓+金属臂段）未动。
- **电池**：83 passed / 1 skipped（scene_w6 13/13 + smoke4_env + reward + scene_init）。
- **视觉线 16 轮定版**：v17 结构 + 任务带全覆盖 + 尾段诚实注记。

### 七、W6a-v19（3-DoF env 驱动铰接臂 — 解析 IK 伺服，checkpoint 全兼容）

- **结构**：每臂 3 铰（yaw + pitch + elbow），肩锚在龙门 (±0.235, 0, 1.000)。
  连杆 2 段 capsule（L1=L2=0.16 m，10 g/段 + 2 个 5 g 关节盘），臂尖 22 mm 球。
  6 个 position servo（kp=60, forcerange ±8 N·m, ctrlrange ±3.2）。
- **IK**：env 侧每 step 解析 3-DoF IK（`yaw=atan2(Δy,Δx)` + 平面内二连杆余弦定理），
  写 ctrl[44:50]。策略 action 仍 44 维 / obs 109 维零变化——**已训 checkpoint 全部有效**。
- **残差**：390 帧 replay，right/left worst tip-to-wrist = **0.0000 mm**（运动学直写）。
  闭环伺服稳态 0.52 mm（reset 后 ~100 ms 摆入暂态，稳态紧跟踪）。
- **涟漪记账**：nq 51→57、nv 50→56、nu 44→50、ngeom 143→145（v17 手侧臂 6 geom 移除，
  v19 臂 10 geom + junction 球 2 移除 + gantry 网格不变）、nbody 51→55。
  gate3_qpos.json 布局 51→57 wide（reader 偏移 13→19、probe 偏移 7→13 全部同步）。
  test_scene_init 的 qpos[:3] 硬编码改 obj_qpos_adr 寻址。
- **y 包络前置测量**：right 180.8 mm / left 122.9 mm ≫ 40 mm → 2 铰不可行，
  3 铰方案经 Main 批准（A 裁决）。reach 需求 right 0.3175 m / left 0.2547 m → L=0.16 m。
- **遗留（预存，HEAD 即挂，非 v19 引入）**：
  1. `test_r1_forcerange_is_the_shared_budget_times_the_moment_arm`——v12 引入 1.7
     forcerange 缩放后恒等式测试未同步（模型值 0.17919 = 1.7 × 0.10541 正确，测试期望未缩放值）。
  2. `test_revision_md_records_the_forcerange_table`——本文档 v17.1 中文追加后
     力矩臂表匹配失败。

### 八、W6a-v20（被动随动垫 — 接触下自对中，checkpoint 兼容）

- **机制**：ff/mf 四垫（rf/th 保持刚性）从 tip 刚性挂载改为独立 pad body +
  2 正交 hinge（stiffness 0.02 N·m/rad、damping 0.002、range ±0.6、springref 0，
  垫 geom 5 g）。弹簧rest=挂载系——gate3 工作位形下即已证面平行姿态；
  接触法向力矩 ≫ 轻弹簧（1.5N×3mm=0.0045 ≫ 0.02×偏差），卷曲旋转被被动铰吸收。
- **机制验收（r7e 复测条件）**：压紧 N 3.8/4.8 N（v19 同 harness 的 3 倍——自对中贴合），
  +0.20 rad 卷曲需求下 N 持 2.58/4.44 N **不塌零**（v20 前 r7e 力模式 N 崩 0.00），
  pad 铰偏转 38–49 mrad 吸收旋转 ✓。抬升传递由 W6c 力控机器复跑（N* 1.5N/垫 规格已定）。
- **零策略涟漪**：ctrl 50 / obs 109 不变，checkpoint 有效（v19 先例）。
- **涟漪记账**：nq 57→65（4 pad × 2 hinge）、nv 56→64、nbody 55→59（ngeom 145 不变——
  geom 改挂不改名）。gate3_qpos.json 65-wide（reader 全部切 jnt_qposadr 名字寻址，
  scene_w6 gate3b + smoke6 probe）；
  `_pad_checkpoint` 重写为名字映射（51-wide checkpoint → 65-wide，pad 槽补零）。
- **已知布局隐患（W6c 已修其两处，此处重申）**：任何绝对 `qpos[0:7]` 写在 65-wide 上
  错写臂铰——统一走 `obj_qpos_adr`。

### 九、W6a-v20.1（随动垫 rest-pose 预紧 — settle 楔高修复）

- **现象**：随动垫重力下垂 −46.6 mrad（四垫一致，pad_rx 轴）在 settle 楔入盒足迹下方，
  W6c 训练位姿 settle 盒心读到 0.753（高于静置面 13mm），dz 参考面失准。
- **修复**：pad_rx springref 预紧 **+0.047 rad**——重力 sag 恰好归零（rest = 挂载位形 =
  已证刚垫几何）。ry 不受重力（实测 ±0.2 mrad），预紧不变。
- **验证**：XML settle 盒 z 0.7400/漂移 0.029mm；**训练位姿 settle 盒心 0.7400**（楔高判据
  ±0.1mm 过，修前 0.753）；r7e 机制复测：压紧 N 3.80/4.72 → +0.20 rad 卷曲需求 3.62/4.33 N
  不塌——**预紧未吃自对中裕度，工作点 pad 偏角 38-49 → 17-18 mrad 反而改善**。
- 自对中裕度算术：预紧只移 rest 点不移 authority（接触力矩 0.0045 / 0.02 = 225 mrad 偏转
  预算不变）。

### 十、状态

- 全套门：scene_w6 13/13（含重力版 static hold）；核心电池同 v8。
- gate3_qpos.json：双手完整 qpos（物体关节置零 + 本节文档标注）。
- XML sha256 = `8b6a6ddd1b0df4f7d1b16df18263af6093a48eba31d7818630f44d1677dc11b0`
  （本版无 XML 变化——判据与 seed 重搜为测试/证据层；git cat-file 复核）。


## W6a-v10（2026-10-15）：垫材质 + 垫形（W6c 二轮杠杆序第 1/2 条）

- **垫摩擦 1.0 → 1.8**（橡胶级；Allegro 真机即胶垫——物理正当）：8 个指尖碰撞 geom
  `friction = 1.8 0.005 0.0001`。W6c 实测杠杆：r5 位姿滑动容量 2.0 → 3.3 N = 2.2× 盒重，
  其 3.5 mm 停滞直接过门。
- **垫形：12 mm 球 → 11 mm 高圆柱平面垫**（r 9 mm × 高 11 mm，`type=cylinder`）：
  滑动中接触点不动——杀死"法向 1.09 → 0.52 N 衰减"的根因（球面接触点随滑动迁移）。
  碰撞 geom 变更的质料影响：**tip body 带显式惯性（6.845 g 原样）——质量分布零变化**，
  位级等价精确成立（出生/静置/门全数位级一致，实测见下）。
- **seed 无需重搜**：门 3 判据为垫心矩形距离，垫形/摩擦不移动垫心——现有 seed
  本就在中带（0.7423/0.7588/0.7596/0.7669 ∈ [0.732, 0.770] ✓），复核即证。
- 实测：出生 −6.2e−17；静置盒 0.029 mm；净空 58.4 mm；全门绿。
- XML sha256 = `133174371e884bf0b8ed6f3fc4228b89f54302893974ef4da9557fc6795d6273`。

### W6a-v10 附：垫形变更后的 R1 力臂/力帽重测（圆柱垫定稿树实测）

垫形（12 mm 球 → r9×11 mm 圆柱）改变了 R1 力臂测量的工作空间剔除带
（指尖中心 12→9 mm 排除半径），j0 类与 thj0/thj1/thj2 的最小力臂随之变化，
力帽按 R1 公式（10 N/4 × 力臂）重推导如下（定稿树实测，非手抄）：

| joint | arm (mm) | τ (N·m) |
|---|---|---|
| ffj0 | 9.00 | ±0.0225 |
| ffj1 | 42.16 | ±0.1054 |
| ffj2 | 37.48 | ±0.0937 |
| ffj3 | 16.88 | ±0.0422 |
| mfj0 | 9.00 | ±0.0225 |
| mfj1 | 41.55 | ±0.1039 |
| mfj2 | 37.60 | ±0.0940 |
| mfj3 | 16.94 | ±0.0424 |
| rfj0 | 9.00 | ±0.0225 |
| rfj1 | 42.34 | ±0.1059 |
| rfj2 | 37.53 | ±0.0938 |
| rfj3 | 16.88 | ±0.0422 |
| thj0 | 37.69 | ±0.0942 |
| thj1 | 9.02 | ±0.0226 |
| thj2 | 20.37 | ±0.0509 |
| thj3 | 9.01 | ±0.0225 |

变化关节：ffj0/mfj0/rfj0/thj1（12.0→9.0，工作空间剔除带变化）与 thj0/thj2（力臂随
朝向相关剔除带变化）。力帽变化随 R1 公式自动传播至执行器 forcerange（XML 已再生）。

## W6a-v12（2026-10-15）：抬升链 forcerange 对齐硬件规格（W6c 三轮判停裁决）

- **裁决**：ffj1/ffj2/mfj1/mfj2（双手 8 关节）forcerange ×1.7（R1 保守派生值 ×1.7）。
  现行 0.09–0.105 N·m 是 R1 公式的保守派生，非 Allegro 硬件真值——本轮是
  **硬件规格校准**，不是放松判据；抬离门 5 mm 不变。
- **实现**：`FORCERANGE_SCALE_LIFT_CHAIN = 1.7` 作用于
  `FORCERANGE_SCALE_JOINTS = {ffj1, ffj2, mfj1, mfj2}`（双手）。
  其余 32 关节不动；执行器 kp/forcerange 以外的参数不变。
- **数字**：ffj1 0.105→0.179 / ffj2 0.094→0.159 / mfj1 0.104→0.177 / mfj2 0.094→0.160
  （双手对称）。力帽放大后弹簧需求占帽比下降（裕度只会改善）——static hold 门重验 PASS。
- 静态可持/出生/静置：位级一致（力帽放大不影响被动弹簧/重力平衡——只放大伺服可用力矩上限）。
- XML sha256 = `f8925e275e601355606b84b711d848b4901ea6be2a55c1746ea0827ee1aad08a`
  （cat-file 对象库复核）。
- 电池：核心 9 套件 121 passed / 11 skipped / 2 R9 既有红灯。scene_w6 13/13 全绿。

## W6a-v11（2026-10-15）：支撑视觉连续化 + 物理理想化文档化 + 腕部 DoF 更正

用户面板复看打回"机械手悬空"。主代理物理核查：运动学接地 ✓（仅 object_free 是自由关节），
"悬空" = 视觉断裂——55.5 mm 短节 vs ±0.18 m 行程，动离静态龙门件。

- **伸缩支撑桅杆**：每手 hand_root 焊视觉圆柱桅杆（r 18 mm，长 280 mm，
  覆盖 z 名义行程 0.75–0.95 + 腕倾角），顶端**滑车领圈**（56×56×50 mm）骑在 y 吊轨上
  （吊轨 z 带 [0.985, 1.045]，家位桅杆顶 1.035 在带内——桅杆全程穿透吊轨平面）。
  读作"手挂在吊轨上的滑车上"。
- **验收断言**：静置家位桅杆 z 跨与吊轨 z 带相交（0 帧脱接的静态形式）；
  逐回放帧断言由 W6d 的渲染管线继承同一边界（滑轴 ±0.163 全行程下桅杆顶
  0.872–1.198，极端位短暂出带——如实记录，名义带内恒接）。
- **物理理想化文档化**：物理支撑 = 理想化直角坐标关节链（刚性滑台-stage 的标准抽象）；
  龙门/桅杆/领圈视觉件**无物理角色**（装饰性 stage furniture，mass=0 无碰撞）。
  非缺陷，但必须写明，防止后续轮次误把龙门当承力结构做物理推断。
- **腕部 DoF 更正**：wrist_rx/ry/rz 铰链（±1.8/±1.571/±1.57 rad）为**主动自由度**
  且在伺服表内（R1/R2 起在库）——此前"固定腕 = 结构边界"的表述过度；更正为
  "seed 选择固定朝向、动态重定向在 ±90° 内可用"。GPT"主动腕"建议实为在库能力。
  v7 驳回条目据此部分更正（驳回的是"新增 DoF"，非"使用在库 DoF"）。
- 零物理验证：出生/静置/门全数与 v9 位级一致（实测见提交）；ngeom 135 → 139
  （+4：桅杆 ×2 + 领圈 ×2），`EXPECTED_NGEOM = 139`（+4 具名 mast 族）。
- 渲染：`docs/human_vs_sim_w6a.png`（v8 家位重渲，桅杆/领圈/圆柱垫入画）。

## W6a-v14（2026-10-15）：卷曲 kp ×5（r7d 合规修复）

r7d 诊断闭环：μ_eff = 0.00 但钩挂拖起 7 mm = 摩擦有效、路径失控。
ffa2/mfa2 kp 0.235 → 1 N@50 mm 偏转 12°；kp ×5 → 偏转 ≤2.3°，
垫路径进 0.02 mm/mm 断言带。执行器参数对齐真实 Allegro 关节伺服刚度。

- `KP_CURL_SCALE = {"ffj2": 5.0, "mfj2": 5.0}` — 只动 r7d 点名的远端卷曲关节。
  ffj0/thj* 等不动。执行器 forcerange 不变（W6a-v12 硬件对齐独立于 kp）。
- **求解器稳定性验证**：400 步 settle max|qvel| 3.17（弹簧瞬态），末 100 步均值
  0.069 rad/s 收敛 ✓。timestep 0.002 不变（30 fps 控制率/物理 substeps 合同不变）。
- 出生 −6.2e−17；静置盒 0.029 mm；净空 57.0 mm；static hold 门 PASS。
- XML sha256 = `257ef61110d0c1c03dbbe9dc639440b01337ec2a0b8a1ba31a2109e5909da100`
  （cat-file 对象库复核）。
- 电池：核心 9 套件 121 passed / 11 skipped / 2 R9 既有红灯。
- **复盘**：从"控制模式"挖到"执行器物理参数"——八轮机制的最后一层。

## W6a-v13（2026-10-15）：平垫替代圆柱垫 + 指部金属质感（GPT 终验条件 #1）

### 一、平垫（GPT 终验条件 #1 的针对性修复）

r5 实测锚定机制：圆柱垫的卷曲面在闭合扭矩下产生接触点 ROLL/翻越顶缘——
闭合扭矩被转化为接触点迁移而非纯法向压力。平垫消除该自由度：

- **替换**：指尖碰撞 geom 从 r9×11 mm 圆柱 → **12×12×3 mm 薄 box**
  （`type=box, size="0.006 0.006 0.0015"`）。垫面（最大面积面）与端面平行
  对齐于终态捏持位姿：闭合扭矩 → 纯法向压力，零接触点迁移。
- **厚度 ≤3 mm** ✓（接近段防勾：垫薄、前缘不过指尖）。
- **摩擦 1.8 保持**（橡胶级，W6c 杠杆第 1 条不动）。
- **零物理**：tip body 带显式惯性（6.845 g 原样）——质量分布零变化；
  出生 −6.2e−17、静置盒 0.029 mm 位级一致。

### 二、指部金属质感（GPT 批"指部金属质感"的指部部分）

双手 palm 子树的 visual geoms 统一覆盖 rgba → `0.45 0.45 0.50 1`
（工业深灰金属质感）。Visual-only，零物理影响。

### 三、数字与状态

- 出生 −6.2e−17；静置盒 0.029 mm / 根 0.42 mm / ncon 4 —— 与 v12 位级一致。
- gate3_qpos.json 不变（垫心位置与判据不受垫形/材质影响）。
- static-hold 门 PASS（重力项不变——质量未变）。
- 电池：核心 9 套件 83+ passed / 全绿（既有红灯 2 不变）。
- XML sha256 = `6ec76dee6d85faa5bdd1496bcde529f4de14781ac4aa0edfef4e9591b97add79`
  （**git cat-file 对象库复核后记录**）。
- 渲染：`docs/human_vs_sim_w6a.png` 已再生（金属质感指部、平垫入画、龙门全件）。
