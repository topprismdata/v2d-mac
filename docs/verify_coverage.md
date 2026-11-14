# verify_v14 覆盖治理图（P3）

51 项检查 → 所钉声明的映射（净室、缺工件 = FAIL 不是 SKIP）。

## [1] 资产哈希 fail-closed
13 项 pinned sha256（场景 XML ×2、retarget_dual.py、v6 时代 gate3、
基线 box_trace、plate_refs ×2、lift_force_account、plate_terminal、
verify_v12/v13、wrist_landmarks.npz、side_seed_pairs.json）。
任何漂移 = FAIL。

## [2] 交付配方（r45/r166，6 mm 抬升）
管线 exit 0、工件新鲜、7.814 mm 峰==终、gate6 双绿实接触
（24.00/24.04 N 钉锚）、全 trace 零辅助约束接触、终态桌反力 0。

## [3] 逐帧受力账目（P0）
4 相账目表、approach 桌承全重 +3.928 N、hold 逐帧 plate-only、
板世界 z 份额合重量、SUM−mg ≤ 0.02、Newton-Euler z 残差
+0.0036 N（钉锚）、逐帧 |resid_z| ≤ 0.01。

## [4] 冻结扰动包络（交付配方，8 角）
L∈[−0.035,−0.005] × S∈[0.2,1.0]，逐角钉实测 rise（tol 0.02 mm）。

## [5] 轨迹逐帧闭合 + 保真优化从零复现
证据重跑逐帧（pos <1e-6 m、quat <1e-5 deg、接触集全等 421 帧）；
删 sidecar → 基线 31.8/44.6 → 两轮 → 24.5/34.0；g6 双绿；
抬升终点离桌（rise ≥9.5、桌 <1e-2、倾角 <6°）；t_c 后逐帧
d_min ≥0.05 mm 且 F≤1e-3 N；失败集 = {g1,g5,wrist}；g4 = 0 帧；
接近段无接触（pre ≤0.5 mm、冲击 ≤1 N）；r199 整形标记进执行流。

## [6]-[7] 优化配方扰动域（P2，已 CLOSED）
6 角（exit/g4/pre/table）+ 6 态执行独立（两两 >1e-6）+
两最差角账目闭合（−0.0068 N）+ 摩擦锥（ft_ratio ≤1.0，实测
0.714）+ 正关节裕度（+0.0135/+0.0518 rad）。

## 已知不可检项（诚实边界）
- 时代提交（v6/v7）无直接字段的声明：gate3 定义漂移、g5 注册 mm、
  up_z —— 报 UNVERIFIABLE（verify_v6/v7 分轨）
- 连续参数域鲁棒性：只证 6 角 + 标称，不证插值状态
- mfa0_left 设计性饱和：已披露的冻结工作点，非裕度声明
