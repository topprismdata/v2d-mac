# runs/p0 — SUMMARY

- 生成时间：2026-10-15 01:49 UTC
- 生成命令：`python3 scripts/p0_summary.py --runs-dir runs/p0 --out-dir runs/p0`
- 仓库 HEAD：`c92de1a`
- 总览：4 pass / 2 fail —— **六绿未达成，按 PLAN 不打 `p0-green` tag**
- **本文件是 dry-run 脚手架**：smoke4（`622d64b`）与 smoke5（`a6268c4`）是诚实 FAIL 报告，按现状收录；正式重跑（Wave-3.5 MJCF 修订 + smoke4 重训）落地后必须重跑本脚本重生成。

## 总览

| smoke | scope | verdict | 门控结果 | 所依据 MJCF | report commit |
|---|---|---|---|---|---|
| smoke1_sam2 | SAM2 segmentation on MPS (smoke 1) | pass | 全部冻结判据通过 | N/A — 感知 smoke，不依赖 MJCF | `827f002` |
| smoke2_vggt | VGGT tracking / geometry on MPS (smoke 2) | pass | 全部冻结判据通过 | N/A — 感知 smoke，不依赖 MJCF | `6dab1de` |
| smoke3_foundpose | FoundPose 6DoF on MPS (smoke 3) | pass | 全部冻结判据通过 | N/A — 感知 smoke，不依赖 MJCF | `8c1f4c5` |
| smoke4_mujoco_ppo | MuJoCo PPO lift-and-hold (smoke 4) | **FAIL** | FAIL: A3b_airborne_at_peak（峰值时刻盒子仍接触桌面；MJCF 4508421 上不存在 clean grasp-and-lift） | scene sha `ffb27e9` = MJCF `4508421` | `622d64b` |
| smoke5_retarget | MANO->Allegro retarget replay (smoke 5, G3) | **FAIL** | FAIL: failing G3 criteria: hand_world_pose_deviation_within_cap -- see retarget_qa.json | MJCF `4508421` (scene `ffb27e9`) | `a6268c4` |
| smoke6_hamer | HaMeR hand recovery on MPS (smoke 6) | pass | all frozen criteria met on MPS | N/A — 感知 smoke，不依赖 MJCF | `827f002` |


## SAM2 segmentation on MPS (smoke 1)

**verdict: pass** ／ 所依据 N/A — 感知 smoke，不依赖 MJCF ／ report commit: `827f002`

判据（MPS vs CPU 一致性 + 掩膜健全性，3 帧）:

| 判据 | 阈值 | 实测(最差帧) | 判定 |
|---|---|---|---|
| min_iou_mps_cpu | 0.8 | 0.999984 | PASS |
| min_mask_area_frac | 0.005 | 0.0548 | PASS |
| max_mask_area_frac | 0.9 | 0.0698 | PASS |
| min_box_coverage | 0.1 | 0.466 | PASS |

物体命中诊断（仅诊断，非门控）：DIAGNOSTIC ONLY, not a pass/fail gate. A bare non-empty + MPS/CPU-IoU criterion also passes for a mask that landed on the wooden table instead of the tissue box; median_mask_hue_deg near 30 deg means the mask is on the table.

### 已知边界 / 诚实短板

- Fixed centre-box prompt does not follow the object. By f_0020 the tissue box has been lifted out of the centre region, so SAM2 segments the wooden table under the prompt instead of the box (median_mask_hue_deg ~30 deg, violet_recall ~0). Single-image SAM2 has no temporal memory; downstream L2 should use the SAM2 video predictor (propagate_in_video) for tracking across frames. This does NOT affect…
- Op-level CPU fallback is enabled via PYTORCH_ENABLE_MPS_FALLBACK for a handful of ops (view/unsqueeze/clone/div/_to_copy). The bulk of compute (Hiera image encoder) runs natively on MPS -- see mps_evidence.mps_peak_allocated_bytes and the ~2.9x MPS speedup over the CPU control.


## VGGT tracking / geometry on MPS (smoke 2)

**verdict: pass** ／ 所依据 N/A — 感知 smoke，不依赖 MJCF ／ report commit: `6dab1de`

判据（物理锚定防跳变 + 有限性）:

| 判据 | 上限 | 实测 | 判定 |
|---|---|---|---|
| a_all_frames_no_nan（30 帧全部有限，无 NaN/跳变） | poses/depths finite | finite | PASS |
| b 防跳变 joint (rad/帧) | 0.03 | max 0.018（2.854×中位） | PASS |
| b 防跳变 translation (m/帧) | 0.03 | max 0.0169（2.994×中位） | PASS |
| b 防跳变 rotation (rad/帧) | 0.02 | max 0.00608（2.88×中位） | PASS |
| c 旋转/平移分轴判据 | 分轴 | joint/rotation/translation | PASS |

- 证据：相邻帧 ORB 重投影中位 **1.869 px**（p90 29.55，29 对 / 14061 匹配）；MPS 28.04s vs CPU 对照 21.61s（5.781×）；CPU fallback ops: 无
- 尺度漂移仅记录未门控：per-frame 中位深度 cv = 0.0897，最大成对中位深度比 1.287
- fork 选择：jmanhype/vggt-mps @ a81c9e5；权重 facebook/VGGT-1B

### 已知边界 / 诚实短板

- scale_consistency 只作漂移记录，本 smoke 未冻结任何尺度锚定门（判据仅 a/b/c）；DESIGN L2 的尺度锚定是 P1 硬要求。
- 全部运动统计（中位/最大）来自 30 帧单条 clip，防跳变规则为 max(3×中位数, 单帧绝对上限)，合成跳变（0.1 m/帧、0.5 rad/帧）验证过可拦截。


## FoundPose 6DoF on MPS (smoke 3)

**verdict: pass** ／ 所依据 N/A — 感知 smoke，不依赖 MJCF ／ report commit: `8c1f4c5`

判据（C1–C6，单帧 tbox）：

| 判据 | 实测 | 阈值 | 判定 |
|---|---|---|---|
| C1_T_finite_rigid_4x4 | true | finite, last row [0,0,0,1], |det R - 1| < 1e-3 | PASS |
| C2_template_count_gt_0 | 798 | > 0 | PASS |
| C3_corners_inside_image | true | all 8 corners in [0,1280) x [0,720) | PASS |
| C4_silhouette_iou | 0.559 | >= 0.5 | PASS |
| C5_bbox_center_dist_px | 33.6 | <= 64.0 | PASS |
| C6_inference_on_mps | mps | mps | PASS |

### 已知边界 / 诚实短板

- 面内轴向误差 41.14°、PnP 内点仅 10–11/159、proxy/mask 面积比 1.338 —— 全部为记录值，未冻结为门；该 6DoF 只能作 estimated 粗先验，不得直接升级 verified。
- 内参假定为 assumed 90 deg HFOV (no calibration available for this clip)；尺度来源 handmade_proxy_box
- FoundPose 自身 score 恒 0（上游 eval_util 重投影 bug，已 patch 记录）：FoundPose's own per-hypothesis score is 0.0 because upstream eval_util.update_without_anno is called with camera_c2w=orig_camera_c2w while the correspondences live in the crop camera, so it reproject…
- mask 来源非 SAM2 而是 opencv GrabCut (GC_INIT_WITH_RECT, seed rect (424, 271, 342, 222), 5 iters)


## MuJoCo PPO lift-and-hold (smoke 4)

**verdict: FAIL** ／ 所依据 scene sha `ffb27e9` = MJCF `4508421` ／ report commit: `622d64b`

判据（A1–A4，10 个 eval episode）：

| 判据 | 实测 | 阈值 | 判定 |
|---|---|---|---|
| A1_episode_lift_gt_5cm | 0.0946 | 0.05 | PASS |
| A2_finger_object_contact_pair | 585 | N/A | PASS |
| A3a_i_finger_force_duty_cycle | 0.899 | 0.5 | PASS |
| A3a_ii_quasi_static_force_sufficiency | 31.71 | N/A | PASS |
| A3b_airborne_at_peak | true | N/A | **FAIL** |
| A3c_i_anchored_step_displacement | 0.00273 | 0.01 | PASS |
| A3c_ii_free_fall_gravity_consistency | 0 | 0.6 | PASS |
| A3c_iii_separation_contact_driven | true | N/A | PASS |
| A4_control_out_of_reach | 0.000544 | 0.05 | PASS |

- 训练：2000896 步（seed 0，n_envs 8，3328 episodes，wall 15.88 min）；奖励前/后半段均值 13046.2 → 19011.2，但 z_peak 全程 0（0 m）
- 接触证据：指-盒接触对 [['ff_proximal_collision', 'object_geom'], ['mf_proximal_collision', 'object_geom'], ['rf_proximal_collision', 'object_geom']]，力占空比 0.899，指尖均力 31.71 N vs 盒重 1.472 N，峰值时刻 table_touching_at_peak = **True**
- clean-lift 可行性（60000 样本扫描，MJCF 4508421）：clean_lift_exists = **False**；对面壁指垫对捏 0/60000，掌板压盒力 26.42 N = 盒重 17.96 倍

### 已知边界 / 诚实短板

- FAIL: A3b_airborne_at_peak（峰值时刻盒子仍接触桌面；MJCF 4508421 上不存在 clean grasp-and-lift）
- MuJoCo forbids actuators on a <freejoint>, so the hand's floating base is expressed as the freejoint's own 3 slide + 3 hinge joints (wrist_x/y/z, wrist_rx/ry/rz), each with a written ctrlrange and forcerange. The object body keeps the literal <freejoint/>.
- Upstream menagerie hand uses <position kp="1"/> with no forcerange and joint damping 0.1, authored for a table-mounted hand. For a free-floating hand these are raised to kp=3, forcerange +-2 N*m and damping 0.5 so the 16-joint posture servo can hold the hand against gravity. Gravity itself is NOT compensated - the ~35 mm steady-state sag under zero ctrl is real physics.
- The tissue-box long axis (0.24 m) is aligned with the palm long axis (y) because all three fingers flex in the x-z plane; with the long axis on x the 0.24 m extent would point away from the fingers.
- PLAN task 4.1 cites github.com/mujoco/mujoco_menagerie, which does not exist; the real upstream is google-deepmind/mujoco_menagerie and the Allegro hand now lives under wonik_allegro/.
- The box rests on a 60 mm pedestal rather than flush on the table. Scripted open-loop grasp search (~5800 samples over three independent searches) showed that with the box flush on the slab EVERY grasp caps at 3.2-3.6 cm - below the 5 cm acceptance bar, i.e. the task was geometrically unsolvable rather than merely hard for PPO. With 60 mm of clearance the fingers slide underneath and form a real p…
- A3a RE-FROZEN (Main ruling): was Pearson r(force, lift) > 0.3, measured r = 0.089 on the best eval episode - a FAIL. That is the wrong statistic for a real grasp: the hand builds force FIRST, then lifts, so once the grip saturates the box rises roughly linearly while force plateaus and instantaneous correlation is low by construction. Re-frozen into two STRONGER gates: (i) finger-force duty cycle…
- A3c RE-FROZEN (Main ruling): was a flat max per-step displacement < 0.01 m, measured 0.0683 m - a FAIL. A box knocked off the pedestal is in LEGAL free fall, where a per-step displacement cap is meaningless; what must be forbidden is energy injection. Re-frozen by segment: (i) supported-or-finger-held segment keeps the 0.01 m/step cap, (ii) free-flight segment is checked for gravity consistency |…


## MANO->Allegro retarget replay (smoke 5, G3)

**verdict: FAIL** ／ 所依据 MJCF `4508421` (scene `ffb27e9`) ／ report commit: `a6268c4`

G3 判据（13 项，`12/13` 通过的 FAIL 报告）：

| 判据 | 判定 |
|---|---|
| frame_counts_consistent | PASS |
| all_finite | PASS |
| joint_limits_ok | PASS |
| joint_velocity_ok | PASS |
| joint_acceleration_ok | PASS |
| penetration_ok | PASS |
| contact_occurred | PASS |
| wrist_continuous | PASS |
| actuator_ctrl_not_saturated | PASS |
| urdf_mjcf_frames_agree | PASS |
| base_solve_feasible | PASS |
| base_displacement_within_cap | PASS |
| hand_world_pose_deviation_within_cap | **FAIL** |

- 失败门：palm 执行偏差 max **0.0859 m**（rms 0.0462 m） vs cap 0.040 m —— 掌板压盒、腕伺服力预算耗在支撑而非合指，最近指尖偏差 132.9
- 本轮已修根因：the superseded anchor rule placed the mid-clip fingertip centroid on the box's top face, which with the clip's wrist pitch asks for a palm frame below the box top: the commanded hand sat -76 mm insid…
- gate scope：G3 gates retarget *executability*: the MANO->Allegro sequence must be replayable through the scene's actuators with real finger-object contact, bounded motion and no interpenetration. It says nothing about hand-geometry accuracy, which stays inherited from HaMeR (smoke 6) and is recorded under fing…
- 记账规则：every number in this report was produced on scene_sha256 == ffb27e97070e, i.e. commit 4508421. Numbers produced on commit ff5ec17 (scene_sha256 a550c636c589) belong to the superseded physics and must not be mixed with these.

### 已知边界 / 诚实短板

- 掌板世界位姿偏差：max 85.9 mm vs cap 40 mm（rms 46.16 mm）—— 掌心朝下搁在盒上，不是 clip 描述的抓握。
- G3 只门控 retarget 可执行性（可回放、真实接触、有界运动、无互穿），不门控手部几何精度 —— 后者继承 smoke 6 的不确定性。


## HaMeR hand recovery on MPS (smoke 6)

**verdict: pass** ／ 所依据 N/A — 感知 smoke，不依赖 MJCF ／ report commit: `827f002`

冻结判据（接线三门 + 检测）：

| 判据 | 判定 |
|---|---|
| mano_params_finite | PASS |
| keypoint_conf_median>=0.20 | PASS |
| flip_control_shipped_beats_inversion | PASS |
| hand_span_ratio_in_0.85_1.2 | PASS |
| wrist_error_over_span<0.15 | PASS |
| left_detection_rate>=0.8 | PASS |
| right_detection_rate>=0.8 | PASS |
| handedness_flips<=2 | PASS |
| device_is_mps | PASS |

- flip 对照：as-shipped 18.4 px < full-inversion 45.11 px（证明左右手 x 顺序正确）
- 跨度比 left 中位 1.109 / right 1.019 ∈ [0.85, 1.20]；腕部配准 14.54 px = 0.111 span < 15%
- 重投影（evidence-only，参考是 MediaPipe 非 GT）：median 18.4 px，p90 70.67 px（30 帧）

### 已知边界 / 诚实短板

- The 2D hand-keypoint reference and the hand box both come from MediaPipe Hands (CPU, MediaPipe graph); reprojection error therefore measures agreement between HaMeR's MANO+camera and MediaPipe's landmarks, not ground-truth accuracy.
- MediaPipe reports no per-landmark visibility score, so the frozen keypoint-confidence criterion uses its per-hand detection score broadcast over the 21 landmarks.
- Hand box + handedness come from MediaPipe instead of HaMeR demo.py's detectron2/ViTPose stack, which has no darwin/arm64 build.
- MANO meshes come from mano_v1_2 (the release HaMeR was trained against); no mesh rendering is done (pyrender stubbed, no headless GL).
- Hand-geometry ACCURACY is reported, not gated: this smoke's verdict is a wiring verdict (mirror / scale / anchor). The accuracy residual against MediaPipe on curled grasp poses under occlusion is a P1 item; see reprojection_accuracy_evidence and risk_notes.

P1 风险记录：HaMeR fingertips are unreliable on hands gripping an occluding object, and the curled-finger PIP/DIP joints disagree most with the MediaPipe reference. Dex-retargeting contact fidelity inherits this uncertainty. P1: needs ground-truth-aligned measurement before it can be closed.
