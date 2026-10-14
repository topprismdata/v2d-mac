# V2D-Mac 设计文档（DESIGN.md）

> 状态：GPT 终审 **APPROVE with comments**（2026-10-14，6 项必改已全部修订入文）；待用户签字（"执行"）后才进入 git init / PLAN.md / 子代理波次。
> 方法论：superpowers 七步流程（头脑风暴 → 工作区 → 计划 → 子代理波次 → TDD → 复验 → 收尾）。

## 0. 一句话目标

在 Apple Silicon（无 CUDA）上，用开源替代件重构 nvidia-isaac/video_to_data 的全链路：**tissue_box.mp4 → 感知重建 → MANO retarget 到悬浮 Allegro 灵巧手 → SB3 PPO 专家 → LeRobot 数据 → ACT + SmolVLA-LoRA 双线 → MuJoCo 闭环评测**，全程证据门控。

目标定义（GPT 评审裁决）：**"Mac 原生、可验证的 Video-to-Robot-Data 闭环"，不追求与 NVIDIA V2D 完全等价。**价值不在把 18 个 CUDA 容器逐个 MPS 化，而在建一条能证明"视频观测 → 物理可信动作 → 可训练策略 → 闭环任务成功"的证据链。

## 1. 文献与数据地基（锚点）

| 件 | 来源 | 状态 |
|---|---|---|
| 原链路参照 + 演示视频 | nvidia-isaac/video_to_data（tissue_box.mp4 LFS + task_profiles/tissue_box_lift_hold.json + lift_hold evaluator 语义：物体抬升 ≥0.1m，随后位置波动 ≤0.05m 保持 ≥20 控制步） | 官方现成 |
| Embodiment 契约格式 | V2D `.codex/skills/groot-finetune/references/embodiment-contract.md` + `floating-hand.md` | 官方文档，读格式自定契约 |
| 6DoF 物体位姿 | FoundPose（facebookresearch），**官方提供 `conda_foundpose_mps.yaml`**（GPT 检索发现）；Gen6D 对照 | MPS 环境官方有 |
| 相机位姿+深度+点图 | VGGT 社区 MPS 移植（jmanhype/zixingjiang/vaos-ai 等 fork）；MASt3R-SLAM 对照（CPU/MPS PyTorch）；mlx-mast3r | 社区移植，P0 验 |
| Mask | SAM2 Mac MPS 实测报告（ice-ice-bear 2026-04）；sam2-mlx | 可行已验证 |
| 手部 MANO | HaMeR（PyTorch，MANO 手） | MPS 需小适配 |
| Retarget | dex-retargeting（j96/DexPilot）：MANO→机器人手优化式，纯 NumPy CPU，内置 Allegro config | 主选 |
| 灵巧手资产 | MuJoCo Menagerie allegro_hand MJCF（16 DoF） | 官方现成 |
| RL | Stable-Baselines3 PPO + SubprocVecEnv（CPU 主线，官方建议小网络 PPO 留 CPU）；Genesis Metal = 支线 | 裁决已定 |
| VLA | LeRobot（官方支持 MPS 训练）+ ACT；SmolVLA-LoRA（PEFT）；mlx-smolvla（M5 Pro 0.873 updates/s）；langgraph-vla-agent（M4 Max 10K steps ~2.4h） | 双线 |
| 评审记录 | GPT 对抗评审：chatgpt.com/c/6acf8d21-8c64-83ee-9755-c669290a6d92，有条件 APPROVE | 2026-10-14 |

## 2. 架构（五层 + 契约 + 门）

```
L1 ingestion   LangGraph + mlx-vlm(OpenAI 兼容端点) + SigLIP2/MPS → graph.db(sqlite) + vector.db(sqlite-vec)
L2 perception  SAM2(MPS) mask · VGGT-mps 位姿+深度 · HaMeR MANO · FoundPose(MPS) 物体粗位姿 · 场景尺度锚定（manual/aruco 优先，scale_source 必填） · mesh：片段级 TSDF 提取 或 点云 Poisson（二选一，纹理投影独立步骤）
L3 retarget    dex-retargeting MANO→Allegro + 接触感知后处理 + QA 报告
L4 grounding   MuJoCo 悬浮 Allegro + SB3 PPO (SubprocVecEnv, CPU) · task: Allegro-V2D-Mac-v0
L5 data+VLA    专家轨迹 → LeRobot dataset → ACT(主线) + SmolVLA-LoRA(双线) → MuJoCo 闭环评测
```

### 类型契约（对齐 V2D v2d_common 哲学，包间禁传裸数组）

- 感知层：`DepthFrame` `CameraPose` `Mask` `ManoFrame` `Object6DoF{T_camera_object, scale_source, confidence, symmetry_class, validation_status: estimated|verified|rejected}`（GPT 契约原文）
- Retarget 层：`RetargetQA{frame_counts_consistent, all_finite, joint_limits_ok, penetration_ok, unresolved_warnings: 0}`（对齐 V2D vega_retarget_qa 报告形态）
- 数据层：`EpisodeContract{action_semantics, coordinate_frame, fps=20, timestamps, normalization, instruction, camera_ids}`（GPT 风险点 P1：固定 action 语义/坐标系/FPS/归一化；`instruction` 为 SmolVLA 线必需的语言指令标注，ACT 线忽略；`camera_ids` 绑定 `observation.images.*` 机位）
- 运行根：run-root 记录每阶段 command/inputs/outputs/provenance/status/logs（对齐 run_e2e.sh）；训练产物记 SHA256

### 门（gate）链 —— 硬门槛，未过不准进下一层

| 门 | 判据 | 失败处置 |
|---|---|---|
| G0 环境 | doctor：MPS 可用、依赖版本、磁盘、模型权重在位 | 停，不降级 |
| G1 感知产物 | 每帧深度/mask/mesh 非空且数值有限 | 该片段剔除并入账 |
| G2 姿态可信度 | validation_status == verified 才可作 RL 真值 | 人工校验后升级，否则 rejected |
| G3 retarget 可执行性 | RetargetQA 全绿（帧数一致/有限值/限位/穿透/warning=0） | 停，修 retarget |
| G4 专家资格 | 对齐 V2D 语义：rollout 达到 episode timeout 才算 eligible；资格/数据管道/闭环成功三 gate 独立 | 不混算 |
| G5 数据审计 | LeRobot 格式 + EpisodeContract 全字段一致 | 停，修转换 |
| G6 闭环评测 | 扰动成功率 + 接触安全 + 轨迹可追溯 | 出报告，不粉饰 |

**G2 判据细则（可复核，缺一不准升级 `verified`）**：多帧 mask 重投影 IoU；位置/旋转时序误差分开统计；尺度误差或不确定度；对称姿态按等价类判定；固定测试帧集合 + 明确阈值；人工审核者与依据入账。

**G3 判据细则（可复核）**：手指-物体接触状态序列；最大穿透深度（按 MuJoCo 数值容差判，不苛求字面零，同时防"碰撞体完全不接触却过 QA"）；关节速度/加速度界；执行器力限幅；腕部运动连续性；指尖跟踪误差。全部数值入 QA 报告。

### 关键语义保留（从 V2D 抄的账）

- 物体 = 动态刚体，**独立重建**，禁止混入静态场景 TSDF 融合（GPT 坑 A/问题三连）
- 单目尺度、遮挡面、对称物体等价姿态集合都入账，不硬拍一个旋转矩阵；尺度锚定 `scale_source: manual|aruco` 优先（度量真值），`mano_anchor` 仅作**估计尺度**且必须附不确定度（人手个体差异大，MANO 输出 ≠ 实测真值）；VGGT 输出是尺度歧义的，必须显式定标后才可进 FoundPose 与 sim
- collection 的 timeout 预估公式保留：`ceil(source_frames / source_fps / motion_speed * control_fps)`，motion_speed=0.5, control_fps=20
- 训练产物校验 SHA256 后才可进 collection

## 3. 任务与动作空间（悬浮 Allegro 先行，全臂后置）

- 本体：Menagerie Allegro（16 指关节执行器）+ **自建浮动腕**——MJCF 新增 freejoint 与受限 6D 执行器（力/速度限幅），动作 = `[Δpose(6), joint_targets(16)]` = 22 维，控制 20 Hz；**禁止绕过执行器直接写 qpos 产生"成功"**（非物理状态赋值 = 假成功，G6 抓）
- 任务：lift_hold（对齐 tissue_box_lift_hold.json evaluator：物体抬升 ≥0.1m，随后位置波动 ≤0.05m 保持 ≥20 控制步；同一 episode 内达成才算成功）
- 观测：物体位姿 + 手关节角 + 速度 + 历史窗口；reset 从参考首帧；reward 按 evaluator 语义塑形
- 全臂（Panda + 2F-85？→ 后置另议）只预留 embodiment contract 扩展位，本轮不实现

## 4. VLA 双线

- **ACT 主线**：PPO 专家轨迹 → LeRobot **v3.0** dataset（锁定安装版本，转换工具同版本验证）→ ACT 行为克隆 → MuJoCo 闭环。单任务抓取的现实路线（GPT 裁决）。
- **SmolVLA-LoRA 双线**：同一数据集，LoRA/冻结 VLM、micro-batch 1–2 + 梯度累积 effective 8–16、BF16 先验证数值稳定。本机 M1 Max/64GB 参考实测（M4 Max 10K steps ~2.4h；M5 Pro 0.873 updates/s）。容忍落后于 ACT，出对比数字即可。
- **观测公平性**：PPO 专家可用 privileged state；ACT/SmolVLA 只吃**固定仿真 RGB 相机**（`observation.images.*` 契约，≥2 机位，训练/推理同构）+ 本体状态；教师-学生观测差异必须在评测报告标注，不得计入算法优劣。
- 两线共用 G5 数据门与 G6 评测门，评测口径一致。

## 5. 验收（可验证）

### P0 —— 六绿 smoke（全绿才锁版本开工搭全链）

| # | Smoke | 过门判据 |
|---|---|---|
| 1 | SAM2 MPS | tissue_box 任一帧出非空 mask；固定 prompt + 模型版本入账；MPS/CPU 对照 ≥3 帧，IoU>0.8 |
| 2 | VGGT-mps | 30 帧位姿+深度全产出且无 NaN；相机重投影误差与尺度一致性入账，旋转/平移分开统计；帧间相对位姿单步范数 < 3× 序列中位数；抽 3 帧深度-边缘目检通过 |
| 3 | FoundPose MPS | **无 CAD 条件实测**：用 tissue-box 自建代理 mesh 生成渲染模板 → MPS 推理出粗位姿；记录设备/模板来源/误差。官方样例跑通不算过门（YAML 含 PyTorch/FAISS CPU 依赖，须实证） |
| 4 | MuJoCo+SB3 | 悬浮 Allegro **8 env 基线**，PPO 曲线上升；首个 lift>5cm episode 必须由真实接触产生——查接触日志，穿透/瞬移/初始位置即判无效 |
| 5 | dex-retargeting | 自带 MANO 样例 30 帧 → Allegro 关节映射 + **在含 tissue-box 代理的 MuJoCo 场景内**出接触/穿透 QA 报告（纯关节映射不算过门） |
| 6 | HaMeR MPS | tissue_box 30 帧 → MANO 非空有限；左右手一致性 + 关键点置信度 + 重投影检查入账 |

### P1 —— 全链 tissue-box 闭环

- L1 验收：自然语言检索（如"拿起纸巾盒的片段"）top-3 召回交互片段 + ≥2 条负样本查询不误召回 + 片段边界标注检查（单查询仅作烟测）
- L2–L3 跑通：RetargetQA 过 G3（含 G3 细则全部数值）；6DoF 出 verified 或 rejected 入账
- 数据规模与完整性：专家演示 ≥10 条 timeout-eligible（管道口径）**且逐条带独立 `task_success` 标签**——timeout 资格 ≠ 任务成功（对齐 V2D 原口径）；失败演示数、相机帧/动作帧对齐误差、首尾帧完整性入账；训练集与评测集隔离
- 统计口径：20 episodes + 95% CI 仅作 smoke；正式统计 = **≥100 独立评测 episodes × ≥3 训练 seeds，报告 Wilson CI**；成功率阈值在正式评测前锁定，禁止用同一测试集反向调阈值
- PPO 专家：lift_hold 成功率 ≥ 阈值（**占位 50%**，P0 后用实测分布校准，由用户签字固化——已确认的口径）
- ACT：同 evaluator 闭环成功率 ≥ 阈值（占位 30%，同上校准签字）
- SmolVLA-LoRA：出与 ACT 同口径对比数字（容忍更低）
- 全部产物 provenance + SHA256 入账，报告含失败率与 rejected 清单

### P2 ——（候选，不承诺）

全臂 embodiment、自拍视频泛化、Genesis Metal 吞吐对照、SmolVLA 全参

## 6. 非目标（本轮不做；用户未勾永久排除，以下为"本轮 out-of-scope"而非永久放弃）

- 云端 GPU（立项前提 = 纯本机 Apple Silicon）
- 多视图/rosbag 输入（只收单目 RGB 视频）
- 实机部署（sim-only）
- GR00T N1.7（CUDA 生态；VLA 双线 = ACT + SmolVLA-LoRA）
- **永久排除**：直接移植 CUDA 专有容器（FoundationPose 原版/cuVSLAM/DROID-SLAM/TensorRT/Docker GPU 编排）

## 7. 风险与降级预案

| 风险 | 等级 | 预案 |
|---|---|---|
| 人手→灵巧手接触动作复原（IK≠接触） | P0（GPT 点名） | dex-retargeting 优化式打底 + 接触感知后处理门 G3；不行则降维任务课程（先推→再抓→后抬） |
| 6DoF 动态物体不收敛 | P0 | 粗位姿+时序优化+人工校验；rejected 不进 RL 真值 |
| vggt-mps 精度不够 | P1 | MASt3R-SLAM / COLMAP CPU 慢速对照 |
| PPO 16 DoF 不收敛 | P1 | 课程分解 + reward shaping + env 数扫描 8/16/32；无硬时间预算（用户裁定），但不无限磨——每轮出指标报告 |
| SmolVLA OOM/慢 | P2 | micro-batch 1 + grad accum；mlx-smolvla 路线对照 |
| "假成功训练数据"（GPT 坑 C） | 全程 | 独立评测几何误差/接触/约束违规，G6 不粉饰 |

## 8. 硬件与运行基线

- Apple M1 Max，10 CPU 核，64GB 统一内存，MPS
- SubprocVecEnv 基线 **8 envs**（与 P0 #4 同口径）；性能扫描 10/16/32（10 = 物理核数参考上限）；PPO 更新留 CPU；训练过夜允许（无硬预算，用户裁定）
- 可复现性：每实验固定 seed，seed + 完整配置入 run-root provenance
- 仓库：单仓 monorepo `~/v2d-mac`（名字可改），分层包 = 层目录，typed contracts 集中一处

### 8.1 目录骨架（PLAN 的锚点）

```
v2d-mac/
  contracts/        # 全部 typed contracts 唯一定义处（禁裸数组跨包）
  ingestion/        # L1
  perception/       # L2
  retarget/         # L3
  grounding/        # L4
  vla/              # L5
  runs/             # run-root provenance（.gitignore）
  tests/            # 每层测试 + P0 smoke e2e
  DESIGN.md  PLAN.md  README.md
```
