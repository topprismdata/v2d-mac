# v14 补充证据复审裁决（r191–r192）

<row gap={2} wrap="wrap">
  <box border radius="lg" padding={3} flex="1" gap={1}>
    <text color="secondary" weight="medium" size="xs">g4 故障归因</text>
    <title size="lg">PASS WITH EDITS</title>
    <text color="secondary" size="xs">主因高度可信，需一项定点确认</text>
  </box>
  <box border radius="lg" padding={3} flex="1" gap={1}>
    <text color="secondary" weight="medium" size="xs">持续离桌判据</text>
    <title size="lg" color="success">PASS</title>
    <text color="secondary" size="xs">接受 t_c 后 59 帧的证明</text>
  </box>
</row>

**裁决：接受撤回“机械楔死是唯一根因”的旧解释；认可执行器额定饱和是当前最有力的主因解释。但仅凭增益乘位置误差，尚不足以把完整的饱和因果归因无条件标为 CLOSED。**

## 一、g4：新证据改变了故障归因

本轮最关键的是两个独立证据：

| 证据 | 观察结果 | 可以支持的结论 |
|---|---|---|
| B：禁用中指碰撞 | 越界 36→19 帧 | 中指接触对 17 帧越界有因果贡献 |
| C：执行器额定检查 | 额定 ±0.015 N·m，名义位置控制输出 0.177 N·m | 位置控制的名义力矩需求超过额定约 11.8 倍 |

特别重要的是，**移除中指碰撞后仍然剩余 19 帧越界**，这直接否定了此前把故障完全归因于箱沿楔死的解释。

名义控制力矩：

\[
\tau_{\rm nominal}=0.377(0-(-0.470))
=0.17719\ \mathrm{N\,m}
\]

相对额定值：

\[
\frac{0.17719}{0.015}\approx11.81
\]

如果 `mfa0_left` 使用该位置控制关系，并且没有其他内部限幅或控制变换，这意味着输出应被限制在额定值附近。

但这里还差一个非常小的证据闭环：

**必须直接读取 MuJoCo 执行时的 `actuator_force` 或对应广义驱动力，而不能只计算名义控制需求。**

建议在第 385、401、402、420 帧记录实际输出、控制输入、位置误差和关节约束力。若实际输出确实持续钳制在约 ±0.015 N·m，就可以将“执行器饱和”标为 CONFIRMED。

另外，“重力下垂至限位”仍是更深一层的原因，需要通过重力与接触消融区分。当前数据尚不能排除其他接触负载或被动约束参与维持限位状态。

### g4 的正式状态

- **原机械楔死唯一根因：REJECTED。**
- **中指接触部分贡献：CONFIRMED，17 帧。**
- **执行器额定饱和：STRONGLY SUPPORTED，待实际输出核验。**
- **物理 g4 门：FAIL，不变。**

我不要求再做大规模搜索，也不授权修改 `forcerange`。补齐实际驱动力这一项就足够推进归因闭环。

## 二、持续离桌：同意关闭此前的缺口

这次给出的证据直接对应上轮指定的验收条件。

<grid columns={3} gap={2}>
  <grid-item>
    <box background="surface-secondary" padding={3} radius="lg" gap={1}>
      <text color="secondary" size="2xs">连续离桌</text>
      <title size="xl" tabularNums>59 帧</title>
      <text color="secondary" size="2xs">362–420</text>
    </box>
  </grid-item>
  <grid-item>
    <box background="surface-secondary" padding={3} radius="lg" gap={1}>
      <text color="secondary" size="2xs">最小角点间隙</text>
      <title size="xl" tabularNums>0.110 mm</title>
      <text color="secondary" size="2xs">门槛 0.050 mm</text>
    </box>
  </grid-item>
  <grid-item>
    <box background="surface-secondary" padding={3} radius="lg" gap={1}>
      <text color="secondary" size="2xs">最大桌面反力</text>
      <title size="xl" tabularNums>0 N</title>
      <text color="secondary" size="2xs">上限 0.001 N</text>
    </box>
  </grid-item>
</grid>

**裁决：P0 中“起升后持续离桌”的证据缺口 CLOSED。**

需要保留两个限定：

第一，这证明的是 t_c=362 后连续离桌，而不是起升动作从第一帧开始就无桌面接触。你已经正确披露了这一点。

第二，0.110 mm 间隙虽然超过冻结容差，但物理余量只有 0.060 mm。它证明的是标称仿真条件下的持续离桌，不构成几何扰动鲁棒性证明。

本次新增断言不必再扩大为全新实验，但要确保使用的是箱体完整碰撞几何；若箱体为规则长方体，八角点方法适用。

## 三、评分与封卷意见

我建议**暂时维持 v14 的 9.1/10，不因归因叙事完善就提前追回 g4 的 −0.15**。这是因为该扣分对应的是实际关节越界，仍然存在。

P0 的持续离桌子项则可以正式关闭，但 P0 剩余的完整力矩平衡范围限制仍要保留。

完成实际驱动力核验后，可直接签署：

> **g4-ROOT-CAUSE CLOSED；g4-PHYSICAL-GATE FAIL。**  
> 原因：执行器额定饱和为已证实的主要控制约束，中指接触贡献部分越界时间。保持原关节限位、驱动额定与历史阈值不变。

本次没有独立运行提交 `a40668f2` 的代码，以上是对提交证据的条件性技术裁决。

**下一步只需补实际执行器力矩记录，不需要为 g4 再启动一次完整的大版本研究。** 这能让故障机理完成闭环，同时保持物理任务失败记录的真实性。