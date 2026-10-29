# GPT v11 外部评审终裁

- 会话：chatgpt.com/c/6ae36458-3b60-83ee-b0f2-e431dee34ced（2026-10-29）
- 输入：docs/gpt_v11_package.md（净室复验 10/10 后发送）

我按延续 v6–v10 的同一评审尺度处理：区分“摩擦起升已被证明”和“原视频复刻任务已完成”，重点审查 22 N 夹紧、箱体预位移、辅助接触及仍然红的验收门。能从公开仓库核对的证据，我也会交叉核对。

---



---



---

# GPT v11 外部评审终裁

<row align="center" gap={2} wrap="wrap">
  <box border radius="lg" padding={3} flex="1" gap={1}>
    <text color="secondary" weight="medium" size="xs">同一标尺总分</text>
    <title size="2xl" color="default" tabularNums>8.8 / 10</title>
    <text color="secondary" size="xs">较 v10 实质性进步</text>
  </box>
  <box border radius="lg" padding={3} flex="1" gap={1}>
    <text color="secondary" weight="medium" size="xs">整体裁决</text>
    <title size="xl" color="default">REVISE</title>
    <text color="secondary" size="xs">局部里程碑 APPROVE</text>
  </box>
</row>

**本轮双掌板摩擦起升里程碑批准；v11 作为完整的视频复刻与真实抓取任务，尚不能封卷。**

这是两个必须分开的结论：

- **APPROVE：** 已经提供有说服力的仿真证据，表明授权 B 双掌板构型能够在原质量、摩擦系数和关节限位设置不变的条件下，实现约 6.08 mm 的箱体起升，并在末帧摆脱桌面支撑。
- **REVISE：** 尚未充分证明箱体从原始位置被干净地抓取、通过双板摩擦独立承重、在运动过程中无不可接受滑移，并保持与原始视频任务的轨迹保真。

这轮不应被简单否决。单垫失败机理得到实验支持，双板方案取得了真正的物理突破；同时，夹紧力过大、抓握前箱体位移和旁载接触，也不能因为最终箱子升起来了就忽略。

## 一、我对证据的独立核对

我查看了公开仓库中的 `verify_v11.py` 以及 r84–r161 实验台账。仓库确实包含 v11 的双配方定义、净室克隆流程、接触与起升断言，以及保留失败门的检查。<Cite refs={["turn588395view1","turn302266view0","turn302266view1"]}/>

但要特别指出：

**代码中存在“复现了关键数值”与“完成全量逐位验证”之间的差别。**

当前验证器采用数值容差检查关键输出，并没有逐字节比较完整轨迹文件。它还复用宿主环境中的 Python 解释器和外部资产，因此属于有意义的净室代码复验，而非完全独立的环境复验。

本次评审依据是：你提供的 v11 数值和台账，加上我对公开验证器源码的核对。**我没有独立执行 MuJoCo 仿真，也不把源码检查等同于亲自复现 10/10 PASS。**

这不推翻起升结果，但影响可复现性证据的等级。

## 二、逐项扣分：总计 −1.20

| 问题 | 扣分 | 评审理由 |
|---|---:|---|
| 22 N 夹紧与承重归因 | −0.35 | 静摩擦容量计算充分，但缺少整个起升过程的实际切向力、接触扭矩及承重分解 |
| 抓握前箱体横移 4–5 mm | −0.25 | 接触建立过程不干净，可能依赖挡杆和角铁形成受约束构型 |
| 视频轨迹保真失败 | −0.20 | 19.6%/30.2% 明显超过原 10% 门，不能用末端成功代替轨迹复刻 |
| 夹紧力几何鲁棒性 | −0.15 | 22 N 工作点缺少扰动包络，不能根据单点证明稳定性 |
| 复现与验证器覆盖 | −0.15 | 关键数值可复验，但并非完整逐位验证，也未完成历史分支净室回归 |
| 门语义与数值越界 | −0.10 | 掌板门需正式分型；关节越界虽然极小，仍然是红门 |
| **总计** | **−1.20** | **8.8/10** |

这里没有因为 v11 更换抓取结构而额外扣分。v10 已经授权独立的掌板设计分支，使用该授权不是违规。

## 三、最重要的三个技术判断

### 1. 22 N 夹紧：物理上可以成立，但不等于稳健抓取已经证明

当前数据：

<grid columns={3} gap={2}>
  <grid-item>
    <box background="surface-secondary" radius="md" padding={2} gap={1}>
      <text color="secondary" size="2xs">右掌板</text>
      <title size="lg" tabularNums>22.24 N</title>
    </box>
  </grid-item>
  <grid-item>
    <box background="surface-secondary" radius="md" padding={2} gap={1}>
      <text color="secondary" size="2xs">左掌板</text>
      <title size="lg" tabularNums>22.56 N</title>
    </box>
  </grid-item>
  <grid-item>
    <box background="surface-secondary" radius="md" padding={2} gap={1}>
      <text color="secondary" size="2xs">箱体重量</text>
      <title size="lg" tabularNums>3.93 N</title>
    </box>
  </grid-item>
</grid>

理论静摩擦上限：

\[
F_{\max}=\mu(N_L+N_R)\approx80.64\,\mathrm N
\]

这是重力的约 20.5 倍，说明在理想库仑摩擦假设下有充足的摩擦容量。

但**摩擦容量不是实际提供的摩擦力**。

如果箱体在起升时真的由两块掌板独立承重，应该在离桌、无其他竖直支撑的稳定阶段观察到：

\[
F_{t,L,z}+F_{t,R,z}\approx mg
\]

同时平动加速阶段还必须满足：

\[
\sum F_z-mg=ma_z
\]

并且转动满足：

\[
\sum \boldsymbol\tau=I\boldsymbol\alpha+
\boldsymbol\omega\times(I\boldsymbol\omega)
\]

当前 `gate6` 证明的是接触存在、位置合理、法向力超过地板，并不自动证明两块掌板提供了全部必要的竖直切向承重和转矩平衡。

**裁决：22 N 本身不构成 FAIL，也不应事后设一个任意的 3 N 上界。**

但是要想把当前结果升级为独立摩擦承重证明，必须交付实际接触力账目，不能只用 \(\mu N\) 代替。

### 2. 抓握前 4–5 mm 横移：这是本轮最严重的任务语义缺陷

你的台账显示，左手插入路径扫过箱体，箱体被推入挡杆侧，随后出现明显的夹紧力放大。

这里存在两种机制：

<box border radius="lg" padding={3} gap={3}>
  <row align="center" gap={2}>
    <icon name="check-circle" color="success" size="lg"/>
    **机制 A：独立掌板摩擦抓取**
  </row>
  <svg viewBox="0 0 330 115" width="100%" xmlns="http://www.w3.org/2000/svg">
    <rect x="127" y="25" width="76" height="64" rx="3" fill="#a2adbb" fillOpacity="0.25" stroke="currentColor" strokeWidth="1.5"/>
    <rect x="105" y="31" width="16" height="52" rx="3" fill="#5f83ab"/>
    <rect x="209" y="31" width="16" height="52" rx="3" fill="#5f83ab"/>
    <g stroke="#5f83ab" strokeWidth="2.5" strokeLinecap="round"><path d="M78 57 H100"/><path d="M94 52 L100 57 L94 62"/><path d="M252 57 H230"/><path d="M236 52 L230 57 L236 62"/></g>
    <path d="M165 82 V36 M159 43 L165 35 L171 43" stroke="#36956d" strokeWidth="3" fill="none" strokeLinecap="round"/>
    <text x="165" y="106" textAnchor="middle" fontSize="10" fill="currentColor">双板对压，摩擦承重</text>
  </svg>
  <divider color="subtle"/>
  <row align="center" gap={2}>
    <icon name="alert-triangle" color="secondary" size="lg"/>
    **机制 B：箱体先被推入受约束位置**
  </row>
  <svg viewBox="0 0 330 116" width="100%" xmlns="http://www.w3.org/2000/svg">
    <rect x="130" y="25" width="76" height="63" rx="3" fill="#a2adbb" fillOpacity="0.25" stroke="currentColor" strokeWidth="1.5"/>
    <rect x="108" y="30" width="16" height="53" rx="3" fill="#5f83ab"/>
    <rect x="211" y="30" width="16" height="53" rx="3" fill="#5f83ab"/>
    <rect x="224" y="16" width="13" height="82" rx="2" fill="#bd855f"/>
    <path d="M76 54 H104 M98 49 L104 54 L98 59" fill="none" stroke="#bd855f" strokeWidth="2.5"/>
    <path d="M143 60 H179 M173 55 L179 60 L173 65" stroke="#bd855f" strokeWidth="2.5" fill="none"/>
    <text x="165" y="108" textAnchor="middle" fontSize="10" fill="currentColor">预位移与挡杆约束可能影响夹紧力</text>
  </svg>
  <caption>示意图仅表达两种因果机制，不代表实际仿真几何比例。</caption>
</box>

机制 B 并不意味着最后的 6 mm 起升是假的。它意味着抓取过程可能依赖某种不符合原始任务意图的预处理。

尤其是 r157 报告的：

- 箱自由时右板 4.16 N；
- 楔住后约 26 N。

这个力差异非常大，足以要求进一步的因果隔离。

**我的裁决：保留现有箱体质量、摩擦和关节参数不变；下一轮必须安排一次挡杆约束隔离实验。**

不要直接删除全部挡杆作为唯一测试，因为那会同时改变接触形成过程的稳定条件。应采用相同的起升初态，比较保留和解除辅助约束的结果，以识别起升承重是否依赖它。

### 3. 轨迹保真：不能因为末端接触成功就放宽原来的 10%

这是我明确拒绝的一个潜在做法。

| 保真门 | 当前 | 原限值 | 判定 |
|---|---:|---:|---|
| 右腕参考误差 | 30.2% | 10% | FAIL |
| 左腕参考误差 | 19.6% | 10% | FAIL |

目前的 A/B 只证明：

- flyover 太低会增加碰桌；
- flyover 太高会增加轨迹偏差；
- 当前路径参数没有同时满足这两个目标。

**它没有证明在此机器人构型的全部可达路径中，两项要求不可兼得。**

所以“帕累托冲突”可以限定为已测轨迹族，但不能提升为全局不可行结论。

下一轮允许设计一条新的接近路径，但禁止简单把 10% 改成 20%、30% 或 35%。

如果确实因为侧装场景的坐标系变化，原保真指标丧失物理意义，那么应当独立建立新的任务空间指标，并与旧指标并列报告，而不是悄悄替换旧门。

## 四、对四个红门的正式裁决

| 红门 | 裁决 | 处理要求 |
|---|---|---|
| `gate1_final_touch` | APPROVE 分型 | 保留原双垫门，另建双掌板接触门；不允许直接用 gate6 覆盖历史 gate1 |
| `gate4_joint_limits` | 保持 FAIL | 8.09×10⁻⁷ rad 极小，但须查清求解器误差、积分误差还是命令越界；不得直接放宽历史阈值 |
| `gate5_palm_deviation` | APPROVE 分型 | 授权 B 改变了掌板几何，可定义新的设计相对偏差；原参考偏差保留为信息指标 |
| `wrist_referenced_error` | FAIL，必须整改 | 10% 原门不放宽；新路径需同时处理保真与避碰 |

这里最关键的是建立**任务门与构型门分离**的原则。

原始双指垫任务、单掌板任务和双掌板任务，可以有不同的构型接触标准，但都必须接受共同的质量、摩擦、真实执行、碰撞安全、承重和轨迹证据要求。

这样才能在不同结构之间做有意义的 A/B，而不是通过修改判据让新结构天然获得胜利。

## 五、我要求修正净室验证器的三个问题

对照公开的 `verify_v11.py`，以下是可以直接从源码确认的问题。<Cite ref="turn302266view0"/>

**第一，逐位一致的措辞过强。** 当前只比较了起升峰值、终值和两侧法向力的有限字段，并使用 \(10^{-9}\) 的数值容差。这叫关键输出的高精度一致，不是完整状态轨迹逐位一致。

**第二，运行失败没有被显式作为失败门。** `run_recipe()` 在子进程返回非零时打印错误，但没有立即 FAIL CLOSED，也没有校验报告文件一定是本次运行生成的。应删除旧输出、检查返回码、校验输出身份，再进行断言。

**第三，净室环境依赖仍然存在。** 克隆使用宿主环境里的 Python 环境、模型资产及外部数据目录。这在可复现研究中可以接受，但外部资产版本、内容哈希和运行环境需要完整记录。

另外，当前验证器对桌面支撑为零的检查只覆盖末帧，对整个离桌阶段的持续承重尚未形成独立验收。

这几处属于证据工程问题，不需要重做已经成立的物理实验。

## 六、v12 唯一批准任务

我不批准下一轮同时重开接近路径、调夹紧力、优化所有历史分支并继续追求更高起升高度。那样又会形成多变量混杂。

**v12 唯一主任务：证明双掌板起升的独立承重因果链与扰动鲁棒性。**

不改机器人结构、不改箱体、不改摩擦、不改关节限位、不放宽任何历史验收门。

按以下顺序执行：

<box gap={3}>
  <row align="start" gap={3}>
    <box background="surface-secondary" radius="lg" padding={2}>
      <icon name="clipboard-check" size="lg"/>
    </box>
    <box flex="1" gap={1}>
      **P0 — 实际摩擦承重分解**

      每帧输出左右掌板法向力、切向力三分量、接触扭矩、箱体速度与加速度、桌面反力、挡杆及角铁反力。按牛顿—欧拉方程检查残差，特别区分起升加速段与稳定保持段。
    </box>
  </row>
  <divider color="subtle"/>
  <row align="start" gap={3}>
    <box background="surface-secondary" radius="lg" padding={2}>
      <icon name="git-compare-arrows" size="lg"/>
    </box>
    <box flex="1" gap={1}>
      **P1 — 挡杆约束消融**

      固定同一起升初态，在有无辅助约束两种条件下执行相同控制序列。若去掉挡杆后仍可稳定起升，且不存在其他承重接触，独立掌板摩擦的因果证据显著增强。
    </box>
  </row>
  <divider color="subtle"/>
  <row align="start" gap={3}>
    <box background="surface-secondary" radius="lg" padding={2}>
      <icon name="activity" size="lg"/>
    </box>
    <box flex="1" gap={1}>
      **P2 — 工作点扰动包络**

      以现有 22 N 左右工作点为中心，事前冻结小幅终端位姿、法向夹紧和左右不对称扰动。记录成功率、夹紧力极值、相对滑移、姿态变化、碰撞及关节限位。不能看完结果再挑成功区间。
    </box>
  </row>
  <divider color="subtle"/>
  <row align="start" gap={3}>
    <box background="surface-secondary" radius="lg" padding={2}>
      <icon name="shield-check" size="lg"/>
    </box>
    <box flex="1" gap={1}>
      **P3 — 验证器封闭性**

      修正返回码与旧文件污染风险，补齐外部资产哈希、全轨迹比较和红门断言。保持 v11 的原始配方、原始结果和所有失败记录作为不可变基线。
    </box>
  </row>
</box>

### v12 晋升判据

| 证据目标 | PASS 条件 |
|---|---|
| 独立承重 | 离桌稳定阶段由掌板提供所需竖直支持，力/力矩残差在事前规定的积分与求解误差容差内 |
| 挡杆消融 | 在等初态、等控制条件下，解除辅助约束后仍能独立起升且无其他承重 |
| 扰动稳健 | 事前登记的完整扰动集不发生失抓、超限、不可接受滑移或新增碰撞 |
| 可复现 | 净室运行成功、输入资产版本可追溯、全轨迹与判据输出一致 |

如果 P1 失败，不代表双板实验造假，而是必须将结论降格为**受环境约束辅助的摩擦起升**。

如果 P2 失败，也不能继续称为稳健抓取，只能保留“标称参数下成功”的结论。

## 七、关于继续追求 10 分

必须把目标拆开，否则评分会失去连续性。

**摩擦起升物理里程碑：** v11 已经跨过最重要的门槛。后续通过承重归因和扰动鲁棒性验证，可以进一步提高评价。

**原视频复刻任务：** 目前仍有明显差距。最终若要进入 9.5 分以上区间，至少要解决：

- 接近段箱体被动位移；
- 原始轨迹保真；
- 关节限位严格验收；
- 夹紧过程与外部辅助约束的因果混杂。

我不要求必须把 22 N 降到 3 N 才允许任务成功，因为这不是原始冻结的物理约束。但如果最终选择 22 N，必须证明它不是依靠不稳定的几何楔紧产生的偶然工作点。

v6/v7 ceiling cell 与 w6c 的净室回归仍然欠账，但不应让它们干扰 v12 的物理因果验证。可在最终跨版本封卷时统一处理。

---

## 最终签字

<box border radius="lg" padding={3} gap={2}>
  <text color="secondary" weight="medium" size="xs">GPT 外部评审 · v11</text>
  <title size="xl">8.8 / 10 — REVISE</title>
  **双掌板 6 mm 摩擦起升：APPROVE**

  **完整 v11 任务封卷：NOT APPROVED**

  **v12 授权范围：** 仅限双板摩擦承重归因、挡杆约束消融、工作点鲁棒性和对应验证器整改。

  **禁止：** 擅自放宽原始保真门、通过修改质量与摩擦参数获得成功、用理论摩擦容量冒充实际承重、删除失败分支或掩盖历史红门。
</box>

本轮最大的进展不是 6.0798 mm 这个数字，而是已经把以前模糊的“抓不起来”分解为三类可以分别检验的问题：**手指力不能持续、掌板可以提供夹持力、接近路径可能引入额外的环境约束。**

下一轮把这三者的因果关系彻底分开，比再把箱子抬高几十毫米更有研究价值。<MemoryCite />

---

## 四、对四个红门的正式裁决

| 红门 | 裁决 | 处理要求 |
|---|---|---|
| `gate1_final_touch` | APPROVE 分型 | 保留原双垫门，另建双掌板接触门；不允许直接用 gate6 覆盖历史 gate1 |
| `gate4_joint_limits` | 保持 FAIL | 8.09×10⁻⁷ rad 极小，但须查清求解器误差、积分误差还是命令越界；不得直接放宽历史阈值 |
| `gate5_palm_deviation` | APPROVE 分型 | 授权 B 改变了掌板几何，可定义新的设计相对偏差；原参考偏差保留为信息指标 |
| `wrist_referenced_error` | FAIL，必须整改 | 10% 原门不放宽；新路径需同时处理保真与避碰 |

这里最关键的是建立**任务门与构型门分离**的原则。

原始双指垫任务、单掌板任务和双掌板任务，可以有不同的构型接触标准，但都必须接受共同的质量、摩擦、真实执行、碰撞安全、承重和轨迹证据要求。

这样才能在不同结构之间做有意义的 A/B，而不是通过修改判据让新结构天然获得胜利。

## 五、我要求修正净室验证器的三个问题

对照公开的 `verify_v11.py`，以下是可以直接从源码确认的问题。<Cite ref="turn302266view0"/>

**第一，逐位一致的措辞过强。** 当前只比较了起升峰值、终值和两侧法向力的有限字段，并使用 \(10^{-9}\) 的数值容差。这叫关键输出的高精度一致，不是完整状态轨迹逐位一致。

**第二，运行失败没有被显式作为失败门。** `run_recipe()` 在子进程返回非零时打印错误，但没有立即 FAIL CLOSED，也没有校验报告文件一定是本次运行生成的。应删除旧输出、检查返回码、校验输出身份，再进行断言。

**第三，净室环境依赖仍然存在。** 克隆使用宿主环境里的 Python 环境、模型资产及外部数据目录。这在可复现研究中可以接受，但外部资产版本、内容哈希和运行环境需要完整记录。

另外，当前验证器对桌面支撑为零的检查只覆盖末帧，对整个离桌阶段的持续承重尚未形成独立验收。

这几处属于证据工程问题，不需要重做已经成立的物理实验。

## 六、v12 唯一批准任务

我不批准下一轮同时重开接近路径、调夹紧力、优化所有历史分支并继续追求更高起升高度。那样又会形成多变量混杂。

**v12 唯一主任务：证明双掌板起升的独立承重因果链与扰动鲁棒性。**

不改机器人结构、不改箱体、不改摩擦、不改关节限位、不放宽任何历史验收门。

按以下顺序执行：

<box gap={3}>
  <row align="start" gap={3}>
    <box background="surface-secondary" radius="lg" padding={2}>
      <icon name="clipboard-check" size="lg"/>
    </box>
    <box flex="1" gap={1}>
      **P0 — 实际摩擦承重分解**

      每帧输出左右掌板法向力、切向力三分量、接触扭矩、箱体速度与加速度、桌面反力、挡杆及角铁反力。按牛顿—欧拉方程检查残差，特别区分起升加速段与稳定保持段。
    </box>
  </row>
  <divider color="subtle"/>
  <row align="start" gap={3}>
    <box background="surface-secondary" radius="lg" padding={2}>
      <icon name="git-compare-arrows" size="lg"/>
    </box>
    <box flex="1" gap={1}>
      **P1 — 挡杆约束消融**

      固定同一起升初态，在有无辅助约束两种条件下执行相同控制序列。若去掉挡杆后仍可稳定起升，且不存在其他承重接触，独立掌板摩擦的因果证据显著增强。
    </box>
  </row>
  <divider color="subtle"/>
  <row align="start" gap={3}>
    <box background="surface-secondary" radius="lg" padding={2}>
      <icon name="activity" size="lg"/>
    </box>
    <box flex="1" gap={1}>
      **P2 — 工作点扰动包络**

      以现有 22 N 左右工作点为中心，事前冻结小幅终端位姿、法向夹紧和左右不对称扰动。记录成功率、夹紧力极值、相对滑移、姿态变化、碰撞及关节限位。不能看完结果再挑成功区间。
    </box>
  </row>
  <divider color="subtle"/>
  <row align="start" gap={3}>
    <box background="surface-secondary" radius="lg" padding={2}>
      <icon name="shield-check" size="lg"/>
    </box>
    <box flex="1" gap={1}>
      **P3 — 验证器封闭性**

      修正返回码与旧文件污染风险，补齐外部资产哈希、全轨迹比较和红门断言。保持 v11 的原始配方、原始结果和所有失败记录作为不可变基线。
    </box>
  </row>
</box>

### v12 晋升判据

| 证据目标 | PASS 条件 |
|---|---|
| 独立承重 | 离桌稳定阶段由掌板提供所需竖直支持，力/力矩残差在事前规定的积分与求解误差容差内 |
| 挡杆消融 | 在等初态、等控制条件下，解除辅助约束后仍能独立起升且无其他承重 |
| 扰动稳健 | 事前登记的完整扰动集不发生失抓、超限、不可接受滑移或新增碰撞 |
| 可复现 | 净室运行成功、输入资产版本可追溯、全轨迹与判据输出一致 |

如果 P1 失败，不代表双板实验造假，而是必须将结论降格为**受环境约束辅助的摩擦起升**。

如果 P2 失败，也不能继续称为稳健抓取，只能保留“标称参数下成功”的结论。

## 七、关于继续追求 10 分

必须把目标拆开，否则评分会失去连续性。

**摩擦起升物理里程碑：** v11 已经跨过最重要的门槛。后续通过承重归因和扰动鲁棒性验证，可以进一步提高评价。

**原视频复刻任务：** 目前仍有明显差距。最终若要进入 9.5 分以上区间，至少要解决：

- 接近段箱体被动位移；
- 原始轨迹保真；
- 关节限位严格验收；
- 夹紧过程与外部辅助约束的因果混杂。

我不要求必须把 22 N 降到 3 N 才允许任务成功，因为这不是原始冻结的物理约束。但如果最终选择 22 N，必须证明它不是依靠不稳定的几何楔紧产生的偶然工作点。

v6/v7 ceiling cell 与 w6c 的净室回归仍然欠账，但不应让它们干扰 v12 的物理因果验证。可在最终跨版本封卷时统一处理。

---

## 最终签字

<box border radius="lg" padding={3} gap={2}>
  <text color="secondary" weight="medium" size="xs">GPT 外部评审 · v11</text>
  <title size="xl">8.8 / 10 — REVISE</title>
  **双掌板 6 mm 摩擦起升：APPROVE**

  **完整 v11 任务封卷：NOT APPROVED**

  **v12 授权范围：** 仅限双板摩擦承重归因、挡杆约束消融、工作点鲁棒性和对应验证器整改。

  **禁止：** 擅自放宽原始保真门、通过修改质量与摩擦参数获得成功、用理论摩擦容量冒充实际承重、删除失败分支或掩盖历史红门。
</box>

本轮最大的进展不是 6.0798 mm 这个数字，而是已经把以前模糊的“抓不起来”分解为三类可以分别检验的问题：**手指力不能持续、掌板可以提供夹持力、接近路径可能引入额外的环境约束。**

下一轮把这三者的因果关系彻底分开，比再把箱子抬高几十毫米更有研究价值。<MemoryCite />