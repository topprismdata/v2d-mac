# GPT r199 外部评审裁决

<row gap={2} wrap="wrap">
  <box border radius="lg" padding={3} flex="1" gap={1}>
    <text color="secondary" weight="medium" size="xs">同一标尺 · r199</text>
    <title size="2xl" tabularNums>9.4 / 10</title>
    <text color="success" size="xs"><icon name="trending-up" inline/> 较 r196 +0.10</text>
  </box>
  <box border radius="lg" padding={3} flex="1" gap={1}>
    <text color="secondary" weight="medium" size="xs">预位移扣分</text>
    <title size="xl" color="success">CLOSED</title>
    <text color="secondary" size="xs">−0.10 全额追回</text>
  </box>
</row>

**正式裁决：批准关闭预位移扣分 −0.10，r199 作为新的阶段基线冻结。**

这次已经不仅是减小位移，而是完成了比较完整的“故障归因 → 排除其他假说 → 定点修复 → 无回归验证”链条。

## 一、为什么批准关闭

| 评审条件 | r199 证据 | 裁决 |
|---|---|---|
| 找到残余预位移源头 | 帧 280–285 的右掌板擦碰，87.42 N | PASS |
| 排除其他候选成因 | 左中指、环指卷曲与时序调整无效 | PASS |
| 消除异常接触冲击 | 87.42 → 0.0 N | PASS |
| 预位移满足冻结上界 | 0.206 mm ≤ 0.5 mm | PASS |
| g4、g6 无回归 | g4 零越界、g6 双绿 | PASS |
| 主要任务结果保持 | 10.31 mm 起升、末帧桌反力零、保真不变 | PASS |
| 确定性与验证 | 两次相同、36/36 断言通过 | PASS（按提交记录） |

有一个值得强调的事实：

**70 μm 的局部穿入，并不意味着冲击可以忽略。**

在高刚度接触模型里，即使几何穿入非常小，碰撞仍可能产生很大的法向力。你记录到的 87.42 N 接触峰值和桌面反力同时上升，正是需要认真处理的动态接触事件。

r199 通过减速与让位，在接近相消除了这次异常擦碰，而不是通过改变接触刚度、摩擦系数或质量掩盖问题。

这是可以接受的控制修复。

## 二、改善幅度与剩余边界

<text color="secondary" size="xs" weight="medium">抓握前箱体预位移（mm）</text>

<Chart content={{"chartType":"bar","xKey":"version","series":[{"dataKey":"displacement","label":"预位移","valueSuffix":" mm"}],"yAxisMin":0,"data":[{"version":"r196","displacement":3.49},{"version":"r199","displacement":0.206}]}}/>

由 3.49 mm 降到 0.206 mm，相对减少 **94.1%**。相对于更早的约 6 mm，则累计降低约 96.6%。

但我只批准“冻结评估域内的预位移问题 CLOSED”，不将 0.206 mm 宣称为零位移，也不将“冲击 0 N”理解为整个抓握过程不存在任何接触力。

尤其需要保留一个细节：你给出的“预位移 ≤0.5 mm”是 r199 新增的断言门。它是否具备事前独立授权的物理容许依据，仍需在冻结文件中说明。我的本次关闭依据，是异常接触源已被消除、剩余位移很小、后续抓取未退化的组合证据，而不只是通过这条新设阈值。

## 三、评分账目更新

| 扣分项目 | r196 | r199 |
|---|---:|---:|
| P0 承重与剩余动力学证据 | −0.10 | −0.10 |
| 抓握前预位移 | −0.10 | **0.00** |
| 腕部轨迹保真 | −0.25 | −0.25 |
| P2 鲁棒性 | −0.10 | −0.10 |
| P3 验证器覆盖 | −0.10 | −0.10 |
| 历史分支回归 | −0.05 | −0.05 |
| g4 关节越界 | 0.00 | 0.00 |
| **总扣分** | **−0.70** | **−0.60** |
| **总分** | **9.30** | **9.40** |

没有因为新的最小角点间隙从 0.233 mm 降到 0.137 mm 而追加扣分，因为它仍高于已冻结的 0.05 mm 门槛。

但是，该几何余量较小，必须继续作为 P2 扰动鲁棒性研究的风险项。不能仅凭标称状态通过，就推断小幅几何误差下也不会擦桌。

## 四、r199 揭示的一个重要工程问题

我特别认可你披露的这处缺陷：

> 首次整形被 `r63-reorient` / `r38-seat` 的终端窗重建无声覆盖。

它说明系统里存在一个独立于碰撞规划的风险：**控制轨迹在多个阶段被重写，局部修复可能没有真正进入执行流。**

这一问题在 v13 的保真优化中已经出现过类似现象。

因此，我建议把它纳入现有 P3 验证器欠账，而不是另立新的研究方向：

- 对关键轨迹修改记录最终写入者；
- 检查整形后的指令是否实际进入执行态；
- 对重要控制偏置设置“已生效”断言；
- 禁止把没有进入最终执行流的旋钮记为有效优化。

这不影响 r199 的预位移裁决，但会显著提高后续实验的可信度。

## 五、最终签署

<box border radius="lg" padding={3} gap={2}>
  <text color="secondary" weight="medium" size="xs">GPT 外部评审 · r199 · 提交 e016fb30</text>
  <title size="xl">9.4 / 10 — APPROVE</title>
  <divider color="subtle"/>
  <row justify="between" align="center">
    **PRE-DISPLACEMENT**
    <text color="success" weight="medium">CLOSED</text>
  </row>
  <row justify="between" align="center">
    **g4-PHYSICAL-GATE**
    <text color="success" weight="medium">PASS</text>
  </row>
  <row justify="between" align="center">
    **g6 双掌板接触**
    <text color="success" weight="medium">PASS</text>
  </row>
  <row justify="between" align="center">
    **WRIST-FIDELITY**
    <text color="danger" weight="medium">FAIL</text>
  </row>
  <divider color="subtle"/>
  **允许将 r199 冻结为新的物理执行基线。完整视频动作复刻仍未封卷。**
</box>

本次裁决依据提交的 r199 实验数据与验证结果，尚未独立运行 `e016fb30`。

接下来不应继续围绕预位移做无收益的微调。真正剩下的主要问题，是腕部轨迹保真、局部扰动鲁棒性和验证器的执行链完整性。