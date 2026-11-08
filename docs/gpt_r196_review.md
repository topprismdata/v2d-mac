# GPT r196 外部评审终裁

<row gap={2} wrap="wrap">
  <box border radius="lg" padding={3} flex="1" gap={1}>
    <text color="secondary" size="xs" weight="medium">同一标尺 · 新评分</text>
    <title size="2xl" tabularNums>9.3 / 10</title>
    <text color="success" size="xs"><icon name="trending-up" inline/> 较 v14 +0.20</text>
  </box>
  <box border radius="lg" padding={3} flex="1" gap={1}>
    <text color="secondary" size="xs" weight="medium">本轮裁决</text>
    <title size="xl">APPROVE</title>
    <text color="secondary" size="xs">r196 定点整改通过</text>
  </box>
</row>

**正式裁决：三个问题均予以肯定，其中 g4 全额追回 −0.15，预位移部分追回 −0.05。**

这是一次真正的因果修复，而不只是再次找到一组成功参数。

此前已经证实执行器持续饱和，本轮没有增加额定力矩，也没有放宽限位，而是通过改变接触发生之前的手指姿态，避免进入引起约束冲突的构型。

但需要明确：本次 APPROVE 针对 r196 整改，不代表整条视频复刻任务最终通过。

## 一、三项正式裁决

| 评审问题 | 裁决 | 追回分数 |
|---|---|---:|
| ① gate4 物理门是否转绿 | **PASS** | — |
| ② 预位移下降约 42% | **PARTIAL CLOSE** | +0.05 |
| ③ g4 实际越界扣分能否关闭 | **CLOSED** | +0.15 |
| **合计** | | **+0.20** |

### 1. gate4：物理门转绿成立

原先的诊断链是：

接触与负载约束 → 左中指无法恢复目标姿态 → 执行器持续饱和 → 关节越界。

r196 在上游改变了条件：

<box border radius="lg" padding={3} gap={2}>
  <row align="center" justify="between">
    <box flex="1" gap={1}>
      <text color="secondary" size="xs">原构型</text>
      <title size="lg" color="danger">36 帧越界</title>
      <text color="secondary" size="xs">接触发生后再调指令</text>
    </box>
    <icon name="arrow-right" color="secondary" size="lg"/>
    <box flex="1" gap={1}>
      <text color="secondary" size="xs">r196 避让构型</text>
      <title size="lg" color="success">0 帧越界</title>
      <text color="secondary" size="xs">接触发生前调整姿态</text>
    </box>
  </row>
  <divider color="subtle"/>
  <text color="secondary" size="sm">控制干预：左 mfj1–3 在落座前 40 帧渐进卷曲，偏置 0.8 rad；执行器额定、关节限位和物理参数均保持不变。</text>
</box>

更重要的是，这个修复没有以牺牲主要任务结果为代价：

- g6 双侧保持 PASS；
- 箱体最终起升 11.81 mm，无回落；
- 桌面支撑力最终为零；
- t_c 后最小间隙从 0.110 mm 增至 0.233 mm；
- 腕部保真值保持原样；
- gate4 从 36 帧越界变为 0 帧。

在提交的同一评估协议下，**我接受 `g4-PHYSICAL-GATE PASS`。**

此前 r193 的 `g4-ROOT-CAUSE CLOSED` 也无需撤销。新的研究认识是：执行器饱和解释了旧构型为什么失败，接近相避让解释了如何避免进入该失败状态。

### 2. 预位移：追回 0.05，但不关闭

<text color="secondary" weight="medium" size="xs">抓握前箱体预位移</text>
<Chart content={{"chartType":"bar","xKey":"config","series":[{"dataKey":"displacement","label":"预位移","valueSuffix":" mm"}],"yAxisMin":0,"data":[{"config":"v14 冻结态","displacement":6},{"config":"r196 避让","displacement":3.49}]}}/>

减少约 2.51 mm，相对下降 41.8%。

我认可这一改善不仅有数值意义，还有因果意义：同一根左中指既参与此前的推箱，又与 g4 故障有关。一个上游干预同时改善两个不同结果，增强了共同接触机制的解释力度。

但是，剩余 3.49 mm 仍不可忽略。

在尚未明确箱体预位移的允许上限、其他接触源及完整接近轨迹之前，不能把此项关闭。

**裁决：预位移扣分 −0.15 → −0.10。**

## 二、评分账目

| 扣分项目 | v14 | r196 |
|---|---:|---:|
| P0 承重与剩余动力学证据 | −0.10 | −0.10 |
| 抓握前预位移 | −0.15 | **−0.10** |
| 腕部轨迹保真 | −0.25 | −0.25 |
| P2 鲁棒性 | −0.10 | −0.10 |
| P3 验证器覆盖 | −0.10 | −0.10 |
| 历史分支回归 | −0.05 | −0.05 |
| g4 关节越界 | −0.15 | **0.00** |
| **总扣分** | **−0.90** | **−0.70** |
| **总分** | **9.10** | **9.30** |

这是同一标尺的分数调整，没有追加新的验收条件。

## 三、对 r196 的一项重要技术评价

r196 使前面几轮形成了一条相当清楚的因果链：

<box border radius="lg" padding={3} gap={2} align="center">
  <box background="surface-secondary" radius="md" padding={3} width="100%" align="center">
    **接近相中指姿态不合适**
  </box>
  <icon name="arrow-down" color="secondary"/>
  <grid columns={2} gap={2}>
    <grid-item>
      <box border radius="md" padding={3} align="center" gap={1}>
        **箱体被推移**
        <text color="secondary" size="xs">预位移约 6 mm</text>
      </box>
    </grid-item>
    <grid-item>
      <box border radius="md" padding={3} align="center" gap={1}>
        **接触约束恶化**
        <text color="secondary" size="xs">执行器饱和与关节越界</text>
      </box>
    </grid-item>
  </grid>
  <icon name="arrow-down" color="secondary"/>
  <box background="rgba(37,126,95,0.07)" border radius="md" padding={3} width="100%" align="center" gap={1}>
    **接近相提前卷曲避让**
    <text color="secondary" size="xs">预位移减少约 42%，g4 越界归零</text>
  </box>
</box>

这说明此前“指令不可达”的说法必须带有明确的时间与构型限定。

**在已经发生接触约束的抬升相，单纯修改中指控制目标无法解除饱和；但通过接近相提前改变指姿，可以避免进入原来的失败构型。**

这比简单增加伺服驱动力更有研究价值，也是统一轨迹与接触约束设计的一个具体成功案例。

需要保留一个范围边界：现有 A/B 充分支持“提前避让改善了两个故障指标”，但还不能证明全部剩余预位移都由其他手指产生，也不能保证 0.8 rad 是全局最优避让量。

## 四、r196 是否可以冻结？

**可以。**

建议将 `cae449be` 冻结为新的优化分支基线，同时保留 v14 原配方作为历史对照。

冻结时应确保 `verify_v14` 的 35 项验收包含两个不同要求：g4 检查覆盖整条执行轨迹，而非只检查原先 385–420 帧；离桌检查覆盖新的 r196 轨迹，而非沿用旧轨迹缓存。

这些是现有验收含义的核对，不是追加新的研发任务。

对于后续研究，我不建议立即继续搜索新的中指偏置量。当前最值得做的是针对残余 3.49 mm 进行接触归因，看看能否在不破坏 g4 与 g6 的条件下继续减少预位移。

---

## 最终签署

<box border radius="lg" padding={3} gap={2}>
  <text color="secondary" weight="medium" size="xs">GPT 外部评审 · r196 · 提交 cae449be</text>
  <title size="xl">9.3 / 10 — APPROVE</title>
  <divider color="subtle"/>
  <row align="center" justify="between">
    **g4-ROOT-CAUSE**
    <text color="success" weight="medium">CLOSED</text>
  </row>
  <row align="center" justify="between">
    **g4-PHYSICAL-GATE**
    <text color="success" weight="medium">PASS</text>
  </row>
  <row align="center" justify="between">
    **PRE-DISPLACEMENT**
    <text weight="medium">PARTIAL CLOSE</text>
  </row>
  <row align="center" justify="between">
    **WRIST-FIDELITY**
    <text color="danger" weight="medium">FAIL</text>
  </row>
  <divider color="subtle"/>
  **授权冻结 r196 为新的阶段基线。完整视频动作复刻仍未封卷。**
</box>

本次裁决依据你提供的两次确定性运行、35/35 验证记录和 A/B 数据；我未独立运行提交代码。

**本轮最重要的成果：不是“把越界调没了”，而是找到了一条不改变硬件能力、通过预先避免不利接触构型来消除物理失败的控制路径。** 这足以成为本研究的一项独立方法论成果。