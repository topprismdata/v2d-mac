# GPT v14 · r193 最终复审签署

<row gap={2} wrap="wrap">
  <box border radius="lg" padding={3} flex="1" gap={1}>
    <text color="secondary" weight="medium" size="xs">g4 故障根因</text>
    <title size="lg" color="success">CLOSED</title>
    <text color="secondary" size="xs">归因证据通过</text>
  </box>
  <box border radius="lg" padding={3} flex="1" gap={1}>
    <text color="secondary" weight="medium" size="xs">g4 物理验收门</text>
    <title size="lg" color="danger">FAIL</title>
    <text color="secondary" size="xs">关节越界仍然存在</text>
  </box>
</row>

**正式裁决：APPROVE — g4-ROOT-CAUSE CLOSED；g4-PHYSICAL-GATE FAIL。**

r193 已满足我在上一轮指定的最后一项取证要求，无须继续扩大实验范围。

## 一、证据闭环

| 证据 | 结果 | 裁决 |
|---|---|---|
| 名义控制力矩需求 | 0.17719 N·m，约额定 11.8 倍 | PASS |
| 实际执行器力矩 | +0.015010 N·m | PASS |
| 全部越界帧 | 36/36 帧实际输出达到额定上限 | PASS |
| 中指接触隔离 | 越界 36→19 帧 | PASS |
| 原始物理参数 | forcerange、关节限位、摩擦、质量均未修改 | PASS（按提交记录） |

两类证据相互支持：

1. **执行器饱和已实测确认。** 不再只是根据控制增益推断，而是 `data.actuator_force` 在所有 36 帧持续达到上限。
2. **中指接触具有部分因果贡献。** 消融后减少 17 帧越界，证明箱体接触参与了问题，但不是唯一原因。

这里保留一个科学边界：执行器饱和是已证实的直接控制约束；至于重力、其余接触和被动关节约束各占多少负载，尚未做完整分解，不作为本次 CLOSED 的必要条件。

## 二、最终归因登记

<box border radius="lg" padding={3} gap={2}>
  <text color="secondary" weight="medium" size="xs">审计记录 · r193 · 提交 f7c53247</text>

  **g4-ROOT-CAUSE: CLOSED**

  执行器额定力矩限制导致实际输出持续饱和；左中指与箱体接触构成额外约束，并对部分越界帧具有因果贡献。

  **g4-PHYSICAL-GATE: FAIL**

  左中指近端关节仍存在 36 帧越界。故障解释完成不代表物理约束满足。

  **修复权限：未扩大**

  不授权修改执行器额定、关节限位、质量、摩擦或历史验收阈值。
</box>

## 三、评分与后续处置

**v14 评分维持 9.1/10，不因故障归因完成而取消实际越界扣分