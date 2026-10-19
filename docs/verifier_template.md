# VERIFIER — v6 包传输性验证模板（ASPIRE verifier-prompt 移植）

> **What:** 提案者（W6c/W6b/W6d）写报告段；verifier 是**全新 agent，只读报告段文字**（不读任何提案者代码/脚本/中间产物），
> 凭文字独立复现头条声明。过了=报告**传输**了知识（GPT 面前这是"写下来就够别人重做"的证据）；不过=报告有隐性知识漏洞，gaps 列出。
> **它证明什么：** 传输性（writing carries the fix）。**不证明什么：** 泛化（声明本就从这个任务蒸馏）。verified=可提交评审，≠普适。

## 阅读规则（违反即 verdict: void + 自报 BREACHED）

MAY 读：
- `runs/*/report.json` 的对应声明段（唯一提案者产物，即"被测写述"）
- `PLAN.md` 对应条目、`docs/` 说明、`assets/` MJCF（共享环境基础设施，非答案）
- `$VERIFY_DIR/` 下自己产出的一切

MUST NOT 读（直接/间接/grep/glob 皆算）：
- 提案者代码：`runs/w6c/*.py`、`runs/w6b/*.py`、`grounding/*.py`、`tools/**`
- 提案者的探针/中间产物：`/tmp/probe*`、`runs/*/*.npz`、`runs/*/v2*_grid.json` 原始数据
- 其他 verifier 的工作区

## 预算（预算即测量）

| 项 | 值 |
|---|---|
| 复现尝试 | **≤2**（v01 按文字直写，v02 允许一次自行诊断） |
| 种子 | 声明冻结的种子/配置各 1 次 + 文字允许的扰动档 |
| 通过 | v01 或 v02 达到声明数字 ±10% 容差（确定性 MuJoCo 下应精确复现，容差只留给浮点/版本差） |

## 程序

1. 先读声明段，写下预期用到的机制与预测结果（报告里要对照）。
2. 只凭文字写 `$VERIFY_DIR/code/v01_from_report.py`——**不许改进它**：文字错了必须让复现失败暴露出来，悄悄修好=测量作废。
3. 失败→从自己的 run 产物诊断，写 v02，**记录"报告没告诉我的东西"（gaps 是本任务最有价值的产出）**。停。
4. 交 `$VERIFY_DIR/verification_report.json`：

```json
{
  "schema_version": 1,
  "claim": "<report 段名 + 头条数字>",
  "attempts_used": 1,
  "verdict": "verified | refuted | void",
  "entries": [
    {"title": "<声明条目>", "disposition": "sound | defective | untested", "note": "<复现数字或破点+修法>"}
  ],
  "gaps": ["<报告该写没写的，或 []>"],
  "reading_rules": "RESPECTED | BREACHED (<开了什么>)",
  "notes": "<Main 可直接行动的一两句>"
}
```

**verdict 给你的尝试打分，entries[] 给条目打分——后者更有用，refuted 也要填完**（尝试失败但条目 sound 很常见：缺的是报告的哪句话，不是声明本身）。refuted 是好结果——拦下一份不传输的报告就是这份工的意义。

## v6 包挂载点

- 头条声明送验清单（策略轮出数后定稿）：
  1. `w6c.lift_variant`：+43.46mm/120f/yaw5.0°（垫 z 滑柱 kp2000 变体）——verifier 只凭 report 文字重建变体 XML+协议
  2. `w6c.poison_chain`：955N 座位棘轮的**修复后诚实 vise**（N 8.94/7.58、yaw 2.28°）
  3. `w6b.r26_trackA`：盒全程正立 up_z≥0.962 + pad1 双手 R5（0.0002/0.0039mm）
- 三份 verification_report.json 与九宫格/对比图并列进 GPT v6 包；GPT 看到的不再是"我们说"，而是"我们说 + 独立 agent 只凭我们写的复现了/没复现（gaps 在此）"。
