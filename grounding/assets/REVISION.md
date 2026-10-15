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
