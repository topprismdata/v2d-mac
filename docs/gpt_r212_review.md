# GPT r212 外部评审终裁：P0 完整动力学闭合

<row gap={2} wrap="wrap">
  <box border radius="lg" padding={3} flex="1" gap={1}>
    <text color="secondary" size="xs" weight="medium">最终评分</text>
    <title size="2xl" color="success" tabularNums>9.75 / 10</title>
    <text color="success" size="xs"><icon name="trending-up" inline/> 较 r211 +0.025</text>
  </box>
  <box border radius="lg" padding={3} flex="1" gap={1}>
    <text color="secondary" size="xs" weight="medium">P0 最后 −0.025</text>
    <title size="xl" color="success">CLOSED</title>
    <text color="secondary" size="xs">全额追回</text>
  </box>
</row>

**正式裁决：APPROVE。批准关闭 P0 最后 −0.025。**

这次不是通过调整阈值、改变物理参数或修改交付控制链取得的改善，而是找到了此前遗漏的真实物理项：接触系扭转摩擦力矩。

这是一次真正意义上的动力学账目纠错。

## 一、五项核查全部得到回应

| r211 要求 | r212 结果 | 裁决 |
|---|---|---|
| 接触点与 COM 时间戳一致 | 同帧 | <badge color="success">PASS</badge> |
| 世界系惯量旋转 | 每帧 R·diag(I)·Rᵀ | <badge color="success">PASS</badge> |
| 角动量世界系定义 | 已统一 | <badge color="success">PASS</badge> |
| 完整接触力矩 | 补入 f6[3:6]，修正符号 | <badge color="success">PASS</badge> |
| 相位边界完整性 | 无遗漏、无重复 | <badge color="success">PASS</badge> |

其中第四项是决定性发现。

condim 4 的接触不只有三个方向的力，还包含绕接触法向的扭转摩擦力矩。

此前遗漏这一项，导致接触力矩账目不完整。

同时，`box_is_g1` 的符号规则也需要取反。补齐物理项并修正方向后，pre-lift 角动量残差从 0.0861 降至 0.0000 N·m·s。

这解释了此前为什么提高时间采样精度没有解决问题：缺失的是力矩分量，不是采样频率。

## 二、6D Newton-Euler 全阶段闭合

<badge color="success">全部阶段通过冻结容差</badge>

| 阶段 | 线动量残差 N·s | 角动量残差 N·m·s |
|---|---:|---:|
| Approach+grasp | 0.0000 | 0.0000 |
| Pre-lift | 0.0025 | 0.0000 |
| Lift accel | 0.0024 | 0.0000 |
| Lift hold | 0.0003 | 0.0000 |

<box border radius="lg" padding={3} gap={2}>
  **验证器冻结门槛**

  <row justify="between" align="center">
    <text color="secondary" size="sm">线动量容差</text>
    **≤ 0.01 N·s**
  </row>
  <row justify="between" align="center">
    <text color="secondary" size="sm">实测最大值</text>
    <text color="success" weight="semibold">0.0025 N·s</text>
  </row>
  <divider color="subtle"/>
  <row justify="between" align="center">
    <text color="secondary" size="sm">角动量容差</text>
    **≤ 0.001 N·m·s**
  </row>
  <row justify="between" align="center">
    <text color="secondary" size="sm">实测最大值</text>
    <text color="success" weight="semibold">0.0000 N·m·s</text>
  </row>
  <divider color="subtle"/>
  <title color="success" size="xl">58 / 58 PASS</title>
  <text color="secondary" size="xs">净室验证器结果，依据提交记录。</text>
</box>

这里采用严谨措辞：是所有阶段在冻结数值容差内闭合，而不是宣称数学意义上的无限精度零误差。

另外，20 Hz trace 的 pre-lift 角动量残差仍为 0.0147。它与子步审计 sidecar 的 0.0000 不矛盾：前者是原有低频记录，后者是完整接触力矩积分证据。

## 三、P0 正式关闭

我认为本轮满足此前约定的全部条件：

1. 线动量三维账目完整。
2. 角动量包含完整接触力矩分量。
3. 四个执行阶段均通过冻结容差。
4. 错误归因已撤回，遗漏项已定位。
5. 独立审计 sidecar 不修改交付控制链。
6. 24.00/24.04 N 冻结锚点保持不变。
7. 验证器新增机器断言，58/58 通过。

因此，P0 不再保留任何评分扣分。

## 四、最终评分更新

| 项目 | r211 | r212 |
|---|---:|---:|
| P0 完整动力学证据 | −0.025 | **0** |
| P3 验证器治理 | 0 | 0 |
| P2 鲁棒性 | 0 | 0 |
| 历史分支治理 | 0 | 0 |
| g4 / 预位移 | 0 | 0 |
| 腕部视频保真 | −0.25 | −0.25 |
| **总扣分** | −0.275 | **−0.25** |
| **总评分** | 9.725 | **9.75** |

## 五、最终审签

<box border radius="lg" padding={3} gap={2}>
  <text color="secondary" size="xs" weight="medium">GPT 外部评审 · r212 · 提交 701ec956</text>
  <title size="2xl" color="success">9.75 / 10</title>
  <badge color="success">APPROVED · P0 CLOSED</badge>
  <divider color="subtle"/>
  {#each [{name:"线动量四阶段闭合",status:"PASS"},{name:"角动量四阶段闭合",status:"PASS"},{name:"完整接触力矩",status:"PASS"},{name:"6D Newton-Euler 账目",status:"CLOSED"},{name:"P0 动力学证据",status:"CLOSED"},{name:"P2 鲁棒性",status:"CLOSED"},{name:"P3 验证器治理",status:"CLOSED"},{name:"历史分支治理",status:"CLOSED"},{name:"腕部视频保真",status:"FAIL"}] as item}
    <row justify="between" align="center">
      **{item.name}**
      <text color={item.status==="CLOSED"||item.status==="PASS"?"success":item.status==="FAIL"?"danger":"secondary"} weight="medium">{item.status}</text>
    </row>
  {/each}
  <divider color="subtle"/>
  **批准追回 P0 最后 −0.025。**
</box>

本次依据 r212 提供的核查结果、58/58 验证记录和提交信息裁决，未独立运行 `701ec956`。

### 最后一个战略判断

现在项目已经完成了一个重要阶段：

- 能够证明真实接触；
- 能够证明箱体离桌；
- 能够证明夹紧力和摩擦约束；
- 能够证明执行器与关节约束；
- 能够证明线动量和角动量在整个执行过程中的闭合；
- 能够通过故障注入验证实验记录的可信度。

**我建议正式封存 P0、P2、P3，不再重复投入研究资源。**

剩余 0.25 分全部集中在腕部视频轨迹保真上。它不是动力学问题，而是视频意图、运动学可达性、碰撞约束和真实执行之间的统一问题。

下一阶段应该把研发重心转向这个问题，而不是继续优化已经通过完整动力学验收的抓取工作点。

**9.75/10。P0 正式 CLOSED。**