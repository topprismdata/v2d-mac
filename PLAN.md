# PLAN.md — P0 六绿 smoke（设计来源：DESIGN.md §5，已签字）

## 共享上下文（所有子代理必读）

- 仓库：`/Users/guohongbin/v2d-mac`（branch `main`）
- 资产：`runs/p0/assets/tissue_box.mp4`（1280×720, 30fps, 13s, 390 帧，SHA256=5616a861…）；`runs/p0/assets/frames/f_0001.jpg … f_0039.jpg`（每 10 帧采样，39 张）；`runs/p0/assets/tissue_box_lift_hold.json`（官方 evaluator：lift≥0.1m，hold≤0.05m ≥20 步）
- 硬件：Apple M1 Max，10 核，64GB，MPS；conda 在 `/opt/homebrew/Caskroom/miniconda`；uv 在 PATH
- 产出纪律（每个 smoke 相同）：
  1. 代码放对应层包（`perception/`、`grounding/`、`retarget/`），测试放 `tests/`，证据放 `runs/p0/smokeN_<name>/`
  2. **先写验收脚本（RED：验收项失败）→ 实现 → GREEN → `git commit -m "p0(N): <name> GREEN"`**；禁止先写实现
  3. `report.json` 必含：设备（mps/cpu）、关键库版本、权重版本、逐项判据数值、结论 pass/fail
  4. 只动自己的目录；公共文件（DESIGN/PLAN/assets）只读
  5. 权重/大文件放 `runs/`（已 gitignore），只提交代码+测试+report
- 红线（GPT 终审 6 必改）：判据必须防假阳性——穿透/瞬移/初始位置/纯关节映射均不算通过

## Wave 1（五个子代理并行）

### Smoke 1：SAM2 MPS（包：perception/）
- 任务 1.1 `uv venv runs/.venv-sam2 && uv pip install torch torchvision sam2`（sam2 装 git+https://github.com/facebookresearch/sam2.git，若 pip 名冲突则 `pip install -e .` 克隆源）；权重 sam2.1_hiera_small（官方下载脚本或 HF 镜像，记录版本）
- 任务 1.2 写 `tests/test_smoke1_sam2.py`：对 f_0001/f_0010/f_0020 三帧，中心点 box prompt 出 mask；判据 MPS/CPU 同帧 mask IoU>0.8 且 mask 非空；先跑出 RED
- 任务 1.3 实现 `perception/sam2_smoke.py` → GREEN；证据帧叠加图存 `runs/p0/smoke1_sam2/evidence/`；`report.json` 记 prompt/版本/IoU 逐帧值

### Smoke 2：VGGT-mps（包：perception/）
- 任务 2.1 克隆 VGGT MPS 移植（首选 `zixingjiang/vggt-mps`，备选 `jmanhype/vggt-mps`、`2apanner/vggtmac`；选标准：能加载权重并 MPS 推理；fork 选择理由写入 report）到 `runs/upstream/vggt`；下载 VGGT-1B 权重（记录版本）
- 任务 2.2 写 `tests/test_smoke2_vggt.py`：取 30 帧（f_0001…f_0030）→ 相机位姿+深度；判据：全产出无 NaN；帧间相对位姿单步范数 < 3× 序列中位数；旋转/平移误差分开统计入账；RED 先行
- 任务 2.3 实现 `perception/vggt_smoke.py` → GREEN；抽 f_0001/f_0015/f_0030 深度-边缘叠加图入 `runs/p0/smoke2_vggt/evidence/`；重投影误差与尺度一致性数值入 report

### Smoke 3：FoundPose MPS（包：perception/）
- 任务 3.1 克隆 `facebookresearch/foundpose` 到 `runs/upstream/foundpose`；用 conda 从 `conda_foundpose_mps.yaml` 建环境 `foundpose-mps`（conda 已在 PATH）
- 任务 3.2 建 tissue-box 代理 mesh：`perception/assets/tissue_box_proxy.obj`（盒体 0.24×0.12×0.08 m，来源：消费级纸巾盒典型尺寸，写入 report `template_source` 字段）
- 任务 3.3 用代理 mesh 渲染参考模板（foundpose 模板流程）→ 对 f_0001 出粗位姿；判据：`T_camera_object` 有限、模板数>0、代理盒重投影轮廓与帧内物体目检对齐（叠加图入 evidence）
- 任务 3.4 `tests/test_smoke3_foundpose.py`（断言 T 有限 + 重投影角点在图像内 + report 完整）RED→GREEN；**官方样例跑通不算过门**

### Smoke 4：MuJoCo+SB3 悬浮 Allegro（包：grounding/）
- 任务 4.1 `grounding/assets/allegro_floating.xml`：以 MuJoCo Menagerie `allegro_hand` MJCF 为基底（`git clone --depth 1 mujoco/mujoco_menagerie runs/upstream/menagerie`），顶层加 freejoint 浮动体 + 6D 执行器（3 slide + 3 hinge，`ctrlrange` 写死、`forcerange` 限幅）+ tissue-box 代理盒（0.24×0.12×0.08 m，质量 0.15 kg）+ 2 个固定 RGB 相机（`cam_overhead`、`cam_wrist`，为 L5 `observation.images.*` 契约预留）
- 任务 4.2 `grounding/envs/lift_hold_env.py`：gymnasium.Env；action=22 维 `[Δpose(6), joint_targets(16)]`；obs=物体位姿+手关节角+速度；reward 按 evaluator 塑形（lift 0.1 / hold 0.05 / min_hold 20）；**动作只经执行器生效，env 内禁止任何直接 qpos 赋值移动手或物体**
- 任务 4.3 `tests/test_smoke4_env.py`：env 自检（随机 22 维动作 100 步：执行器满行程、无 NaN、无 qpos 越权）RED→GREEN
- 任务 4.4 `grounding/train_ppo.py`：SB3 PPO MlpPolicy + SubprocVecEnv(8)，seed=0，预算 2M steps（或先到首个合格 episode 即停）；训练曲线窗口均值入 `runs/p0/smoke4_mujoco_ppo/curve.csv`
- 任务 4.5 验收 `tests/test_smoke4_accept.py`：存在 ≥1 episode 满足 lift>5cm，且该 episode 内 mujoco contact 数据含 手指-物体 geom 对（查 `data.contact`），物体 z 上升由接触驱动（对照接触力时间序列）→ RED→GREEN；`report.json` 记曲线/contact 证据

### Smoke 6：HaMeR MPS（包：perception/）
- 任务 6.0 **预检 MANO**：`find ~ -name "MANO_RIGHT.pkl" -o -name "MANO_LEFT.pkl" 2>/dev/null`；两文件不全 → 立即停止并产出 `runs/p0/smoke6_hamer/BLOCKER.md`（写明需用户到 mano.is.tue.mpg.de 注册下载，License 门控，agent 不可代办），本 smoke 判 BLOCKED 而非 FAIL
- 任务 6.1 克隆 `otaheri/HaMeR` 到 `runs/upstream/hamer`，按其 README 装环境（uv venv；MANO 模型软链到其期望路径）
- 任务 6.2 `tests/test_smoke6_hamer.py`：f_0001…f_0030 推理 → MANO 参数非空且有限；左右手一致性、关键点置信度、重投影误差入账；RED→GREEN
- 任务 6.3 证据：2 帧手部叠加图入 `runs/p0/smoke6_hamer/evidence/`

## Wave 2（依赖 Wave 1 产物）

### Smoke 5：dex-retargeting（包：retarget/，依赖 Smoke 4 的 MJCF）
- 任务 5.1 `uv pip install dex-retargeting`；用其自带 MANO 样例 30 帧 → `AllegroHandPositionRetargeting` 出关节序列
- 任务 5.2 在 `grounding/assets/allegro_floating.xml` 场景（含代理盒）内回放关节序列：QA 数值全按 DESIGN G3 细则——接触状态序列、最大穿透深度（MuJoCo 容差判）、关节速度/加速度界、执行器力限幅、腕部连续性、指尖跟踪误差
- 任务 5.3 `tests/test_smoke5_retarget.py` 产出 `RetargetQA` 报告全绿 RED→GREEN；报告存 `runs/p0/smoke5_retarget/retarget_qa.json`

### 收尾任务（主代理亲做）
- 复跑六个验收脚本全绿；六个 report.json 汇总 `runs/p0/SUMMARY.md`；git tag `p0-green`
- 六绿后：阈值校准材料（P0 实测分布）呈给用户签字固化 P1 阈值

## P1 衔接备注（波次产出中累积，不改 DESIGN 签字稿）

- **Smoke 1 实证缺陷**：固定中心框 prompt 无法跟踪——f_0020 起 tissue box 离开画面中心后 SAM2 分割到木头桌面（mask hue 中位 29° vs 正确物体 206–222°），而"MPS vs CPU 一致性"判据会同错而静默通过。P1 的 L2 必须用 **SAM2 视频预测器 `propagate_in_video`** 做跟踪（注意其 `load_video_frames` 要求整数文件名，`f_####` 命名需转换/软链接），并保留 smoke1 已实现的物体命中诊断（色相代理）作为每帧断言。
- **Smoke 2 结论**：jmanhype/vggt-mps（92★，A81c9e5，vendor 上游 VGGT + MegaLoc-MPS 补丁），VGGT-1B 权重 5.03GB，MPS 2.03×/帧加速、0 CPU fallback，30 帧单次前向吃下 64GB 统一内存无需分批。重投影 **1.87px 中位**（identity-pose 对照 2.59px，胜出即有效几何证据）。防跳变判据已由主代理改为物理锚定 `max(3×中位数, 单帧上限)`（平移 0.03、旋转 0.02、joint 0.03），并用合成跳变（0.1m/帧、0.5rad/帧）验证仍可拦截。**遗留**：`scale_consistency` cv=0.0897 仅作漂移记录，**尚未施加 scale_source 锚定**（DESIGN L2 的尺度锚定是 P1 硬要求）；identity 对照显示该门在近静态片段最多只能证明 ~28% 改进，不构成度量精度证据。
