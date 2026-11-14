# GPT r209 外部评审裁决：P0 动力学证据

<row gap={2} wrap="wrap">
  <box border radius="lg" padding={3} flex="1" gap={1}>
    <text color="secondary" size="xs" weight="medium">更新评分</text>
    <title size="2xl" color="success" tabularNums>9.70 / 10</title>
    <text color="success" size="xs">较 r207 +0.05</text>
  </box>
  <box border radius="lg" padding={3} flex="1" gap={1}>
    <text color="secondary" size="xs" weight="medium">P0 扣分</text>
    <title size="xl">−0.05</title>
    <text color="secondary" size="xs">部分追回，尚未关闭</text>
  </box>
</row>

**正式裁决：批准追回 P0 的 0.05 分，保留剩余 0.05 分。**

r209 是一次实质性进展。它第一次把原先主要依靠静态受力平衡的证据，扩展到了完整刚体动力学的线动量和角动量层面。

但我不同意将本轮描述为完整 6D 动力学闭合。原因是：角动量在抬升阶段表现良好，线动量仍存在显著残差，而且 pre-lift 的角动量残差也不能忽略。

## 一、逐项裁决

| 证据 | 实测 | 裁决 |
|---|---:|---|
| approach+grasp 角动量 | 0.0000 N·m·s | PASS（需保留精度说明） |
| pre-lift 角动量 | 0.0861 N·m·s | OPEN |
| lift_accel 角动量 | 0.0085 N·m·s | PASS（条件性） |
| lift_hold 角动量 | 0.0099 N·m·s | PASS（条件性） |
| z 向均值受力闭合 | +0.4 mN | PASS |
| 全相线动量闭合 | 最大残差 76.47 N·s | OPEN |
| 采样限制及残差披露 | 已量化入账 | PASS |

### 1. 角动量：认可抬升阶段的闭合成果

<Chart content={{"chartType":"bar","meta":{"title":"各阶段角动量残差","description":"单位：N·m·s；数值越低越好。"},"xKey":"phase","series":[{"dataKey":"residual","label":"角动量残差","valueFormat":"raw"}],"yAxisMin":0,"data":[{"phase":"Approach+grasp","residual":0},{"phase":"Pre-lift","residual":0.0861},{"phase":"Lift accel","residual":0.0085},{"phase":"Lift hold","residual":0.0099}]}}/>

我特别认可 lift_accel 和 lift_hold 的结果。

这意味着，在最关键的实际承重阶段，接触力作用位置与刚体角动量变化之间建立了相当好的数量级一致性。

但有两个限定：

- pre-lift 的 0.0861 不能随抬升阶段的 0.01 量级一起被概括为闭合良好。
- 绝对残差需要结合对应阶段的实际角动量变化量和冻结容差解释。没有这些信息，不能单凭 0.0085 就宣称严格通过。

因此，我接受抬升阶段的角动量证据，但不接受整个执行过程已经完成角动量闭合。

### 2. 线动量：接受采样限制解释，但不接受闭合结论

这是本轮保留最后 0.05 的主要原因。

你提出的解释具有合理的物理基础：

20 Hz 帧末瞬时接触力，不等于整个 50 ms 帧间隔内的平均接触力。

真实冲量应为：

\[
J=\int_{t_0}^{t_1}F(t)\,dt
\]

而不是简单地把帧末采样值当作整个时间间隔的平均值。

<box border radius="lg" padding={3} gap={2}>
  **两种记录方式的区别**

  <box gap={1}>
    <text color="secondary" size="xs">现有方式：帧末瞬时采样</text>
    <row align="center" gap={1}>
      {#each [0.2,0.8,0.3,1,0.4,0.9,0.2,0.7] as v}
        <box flex="1" height={32} justify="end" background="surface-tertiary" radius="xs">
          <box height={`${v*100}%`} background="#D69A48" radius="xs"/>
        </box>
      {/each}
    </row>
    <caption>只取得离散时刻的力值，不能代表完整帧间冲量。</caption>
  </box>
  <divider color="subtle"/>
  <box gap={1}>
    <text color="secondary" size="xs">理想方式：子步力积分</text>
    <row align="center" gap={1}>
      {#each [0.3,0.5,0.8,0.9,0.7,0.4,0.3,0.2] as v}
        <box flex="1" height={32} justify="end" background="surface-tertiary" radius="xs">
          <box height={`${v*100}%`} background="#269B76" radius="xs"/>
        </box>
      {/each}
    </row>
    <caption>累积整个帧间隔的实际接触力冲量。</caption>
  </box>
  <caption>示意图，不是 r209 的实测力曲线。</caption>
</box>

但是，采样限制目前是一个有证据支持的解释，还不是已经完成独立验证的唯一原因。

其他可能因素仍包括：

- 接触力记录是否遗漏其他外部接触；
- 接触力坐标系及作用方向转换；
- 帧间速度差分与接触力采样时刻是否严格对齐；
- 约束接触力和其他外部力是否完整进入账目。

因此，76.47、20.62、4.40、22.06 N·s 这些残差必须继续作为正式结果保留。

**不能因为 z 向均值闭合，就推导出三维线动量也已经闭合。**

## 二、为什么追回 0.05 是合理的

r209 实际上完成了三件以前没有完成的事情：

1. 从逐帧接触力记录建立了独立的线动量账目。
2. 从接触点位置和刚体姿态建立了角动量账目。
3. 把采样限制造成的误差与静态受力闭合区分开来，没有为了得到漂亮数字而修改交付锚。

尤其是第三点。

我支持你不修改已经冻结的 24.00/24.04 N 交付锚。

但这里有一个更好的工程方案：**增加审计专用的子步冲量记录器，而不是修改交付控制链。**

例如：

- 原有 20 Hz 控制和输出完全不动；
- 在每个物理子步累计接触力积分；
- 同步累计接触力矩积分；
- 单独生成 audit sidecar；
- 与现有 20 Hz 报告并行比较。

这样可以改善动力学证据，而不改变正式交付配方和终审锚点。

这不是本次关闭的强制条件，但如果未来要追回剩余 0.05，这是最直接的路径。

## 三、评分账目

| 扣分项目 | r207 | r209 |
|---|---:|---:|
| P0 完整动力学证据 | −0.10 | **−0.05** |
| 腕部视频轨迹保真 | −0.25 | −0.25 |
| 其他已关闭项目 | 0 | 0 |
| **总扣分** | **−0.35** | **−0.30** |
| **总评分** | **9.65** | **9.70** |

## 四、最后 0.05 的关闭条件

我不要求重新做机器人起升实验，也不要求改变物理参数。

只需补充一项审计工作：

**子步级独立冲量账目。**

需要验证：

1. 每个物理子步的实际接触力和作用点均被纳入积分。
2. 线动量三轴分别闭合，而非只看 z 轴。
3. 角动量在 approach、pre-lift、lift_accel、lift_hold 四个阶段均给出残差。
4. 积分结果与原始 20 Hz 记录之间的差异能够定量解释。
5. 审计记录器与交付控制链隔离，不改变 24.00/24.04 N 锚点。

如果这一轮通过，我将不再要求追加 P0 动力学实验。

---

## 最终签署

<box border radius="lg" padding={3} gap={2}>
  <text color="secondary" size="xs" weight="medium">GPT 外部评审 · r209 · 提交 b826282f</text>
  <title size="2xl" color="success">9.70 / 10</title>
  <badge color="success">PARTIAL APPROVE</badge>
  <divider color="subtle"/>
  <row justify="between" align="center">
    **角动量 · 抬升阶段**
    <text color="success" weight="medium">PASS</text>
  </row>
  <row justify="between" align="center">
    **z 向均值受力**
    <text color="success" weight="medium">PASS</text>
  </row>
  <row justify="between" align="center">
    **全相线动量闭合**
    <text color="warning" weight="medium">OPEN</text>
  </row>
  <row justify="between" align="center">
    **P0 完整动力学证据**
    <text color="warning" weight="medium">PARTIAL CLOSE</text>
  </row>
  <row justify="between" align="center">
    **P3 验证器治理**
    <text color="success" weight="medium">CLOSED</text>
  </row>
  <row justify="between" align="center">
    **腕部视频保真**
    <text color="danger" weight="medium">FAIL</text>
  </row>
  <divider color="subtle"/>
  **批准追回 P0 的 0.05，保留最后 0.05。**
</box>

本次依据你提供的 r209 数据和提交说明裁决，未独立运行 `b826282f`。

**我的总体判断：研究已经从“证明箱子能够被抬起来”，进入“证明机器人究竟如何通过接触力和运动状态完成抬升”的阶段。这是证据质量上的重要升级。**

剩余 0.30 分中，腕部保真仍占 0.25。不要让动力学审计无限扩张，完成最后一轮独立冲量核验后，应把主要研发资源重新投入原始视频动作目标。