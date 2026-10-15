# TopPrism 集成蓝图：v2d-mac × agv-world-model × dc-twin

> 状态：方向已批准（2026-10-15）。Phase A 证据 schema 草案见 §4。
> 关联：`PLAN.md` P1 升级设计（contact-consistent retarget / residual RL）、`retarget/mjcf_provenance.py`（印章机制）。

## 1. 为什么能结合

三个仓库的 DNA 相同：**一切事实必须有证据**。

| 仓库 | 核心资产 | 在本体系中的角色 |
|---|---|---|
| agv-world-model | EvidenceVerdict 五门判据、原子事务（1200 runs 消融：Unsafe 提交 11.1%→0）、CP-SAT 调度、观测隔离 | 物理内核 + 执行可信层 |
| dc-twin | ESR-Loop 五拍（运行→结晶→沉淀→进化→消融）、技能结晶格式、进化门 | 场景与技能的持续进化层 |
| v2d-mac（本仓库） | 四段 MPS 感知（SAM2/VGGT/FoundPose/HaMeR）、接触一致 retarget（P1）、证据门控验收 | **视频→数据的前端**：现实的入口 |

对照 NVIDIA：V2D 是**死的单程管线**（视频进、数据出、场景不变），SimReady 是**死的资产规范**（规格冻结）。dc-twin README 的论断同样适用于 V2D：没有智商。三者结合后：

```text
现实视频 → 孪生场景增量（过进化门）→ 操作技能结晶 → 物理执行（过证据门）→ 账本
    ↑______________________________________运行经验回写______________________________________|
```

这是 V2D 与 SimReady 都不具备的：**从现实长出来、且越用越聪明的闭环**。

## 2. 结合点映射

| v2d-mac 组件 | 对方组件 | 产出 |
|---|---|---|
| L1-L2 感知（VGGT 重建 / FoundPose 6-DoF / SAM2 跟踪） | dc-twin L3 场景（程序化 MJCF + manifest + 标签注册表） | Real2Sim 更新器：现场视频→场景增量，过进化门才入册 |
| L3-L5 retarget + A 门证据 | ESR 技能结晶（前置→执行→证据→后置） | 工人操作视频 → 机器人技能条目，证据字段由门报告填充 |
| L6 策略动作 | EvidenceVerdict 五门 + 原子事务 | 策略抓取经五门裁决才 commit（A3b"盒重由手指承担" ≈ 承托率门，判据可合并） |
| Wave 修订实践（REVISION.md + 回归 pin + provenance 印章） | ESR-Loop 五拍 + 消融门 + DetFix 样本格式 | 手动修订流程形式化入册（Wave 3.5 standoff = 首个跨仓技能条目候选） |
| 观测隔离实践（HaMeR/FoundPose 仅作 estimated 先验） | 观测隔离红线（决策侧零真值，AST 强制） | 共享 AST 检查器 |

## 3. 分阶段落地

- **Phase A（纯赢，先行）**：统一证据 schema。`GateReport`/`mjcf_provenance` 印章 ↔ `EvidenceVerdict`/`incident_record` 适配器（草案见 §4）。首个跨仓样本：Wave 3.5 standoff 修订作为 DetFix 式技能条目入册。
- **Phase B**：Real2Sim 更新器。v2d-mac L1-L2 输出场景增量（几何 + 位姿 + 证据），走 dc-twin 进化门（附契约草案 + 消融子集 + 回归用例，缺一拒绝）才改场景。
- **Phase C**：技能摄入。操作工视频 → retarget → ESR 技能 → 仓库仿真回归 → :8400 面板可视化。
- **Phase D（随 P1）**：SmolVLA / DexMimicGen 扩增消费累积的技能库与数据——组织级飞轮。

## 4. Phase A 证据 schema 草案

目标：一个仓库无关的 `evidence_record.json`，两边都能读、能盖章、能进对方的账本/报告。

```jsonc
{
  "schema": "topprism.evidence_record/0.1",
  "id": "evt-<uuid>",                       // 幂等键，跨仓唯一
  "ts": "2026-10-15T03:09:00+08:00",
  "source": {
    "repo": "v2d-mac | agv-world-model | dc-twin",
    "component": "grounding.scan_feasible_poses",   // 产生证据的函数/模块
    "run_id": "scan4000"                             // 仓内运行标识
  },
  "subject": {                               // 证据关于什么
    "kind": "scene_revision | skill_entry | transfer_commit | gate_report",
    "ref": "grounding/assets/allegro_floating.xml",
    "provenance": { "mjcf_commit": "4508421", "sha256": "ffb27e97…" }
  },
  "verdict": {                               // 与 EvidenceVerdict 对齐
    "gates": [ { "name": "support_ratio", "value": 0.97, "threshold": 0.95, "pass": true } ],
    "pass": true,
    "schema_version": "5-gate/1.2"           // 判据版本化，口径变更必须升版
  },
  "evidence_artifacts": [                    // 落盘证据，相对仓库根
    "runs/p0/smoke4_mujoco_ppo/feasible_poses_meta.json"
  ],
  "crystallization": {                       // 可选： incident → 技能四件套
    "incident": "INC-…", "root_cause": "…", "fix": "…",
    "regression_tests": ["tests/test_smoke6_mjcf.py"]
  }
}
```

对齐约定：
- `verdict.gates[].name` 命名空间共享（首版清单：`support_ratio` / `unload_ratio` / `stability_window` / `at_rest` / `pose_error` / `force_duty` / `anchored_step` / `penetration`）。
- 阈值口径合并随 P1 阈值校准一次完成（A3b 力占空比 vs 承托率支撑比，基线都取 P0 重跑后的实测分布）。
- `mjcf_provenance.attach()` 增加一个 emitter：工作树漂移时写 `provenance_guard` 事件而非静默拒盖（沿用 smoke5 的守卫语义）。

## 5. 诚实差距（合并前要补）

1. **身体尺度**：v2d-mac 仅手部（HaMeR/MANO）；工人操作演示需全身（SMPL 扩展）。AGV 叉取先行（货叉↔托盘槽，比手指简单）。
2. **判据口径**：A3b（力占空比 ≥0.50）vs 承托率（≥95%）量纲不同；随 P1 阈值校准一次性对齐。
3. **资产格式**：对方 STEP/STL + collision.yaml + 标签注册表；本仓生成式 MJCF。`mjcf_provenance` 印章需接对方的 asset manifest。
4. **仓库依赖方向**：dc-twin 依赖 `agv-warehouse-core`（不复制内核）；v2d-mac 以文件契约接入（零 import 耦合，与 :8400 面板同模式），不新增 Python 依赖边。
