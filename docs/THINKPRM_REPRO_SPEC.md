# ThinkPRM 复现规格 — 基于 TMLR 正式版逐节核实（2026-10-05）

论文：Khalifa et al., *Process Reward Models That Think*, TMLR 03/2026（arXiv:2504.16828 v5）。
代码/数据/模型：https://github.com/mukhal/thinkprm
以下事实全部来自论文正文 + 附录，行号为 OpenReview PDF 文本行。

## 1. 训练数据（§3.2）

- 用 **QwQ-32B-Preview** 对 PRM800K 的 (problem, step-by-step solution) 生成 verification CoT。
- 采样直到拿到 **1K 条 CoT = 8K step labels**（只用了 PRM800K 的 ~1%）。
- 过滤三标准：(i) 格式正确（每步有可提取的 `\boxed{}` 决策标签）；(ii) **step judgments 与 PRM800K gold labels 完全一致**（process-based filtering）；(iii) CoT 长度在预算内（防 overthinking）。
- §5.7 消融：process-based filtering 对性能至关重要（vs 只看 outcome）。
- 训练成本：finetune QwQ-32B-Preview 只需 **4.5 小时 @ 1×A100 80GB**。
- 另有 65K CoT 版本（附录 F.2），性能更高但 OOD 略降。
- F.5：用 Math-Shepherd **自动标注**替代人工标注，效果相当（OlympiadBench 75.8 vs 76.3）。→ 支持我们用 Math-Shepherd 做 GRPO 数据。

## 2. 基座模型

R1-Distill-Qwen-{1.5B, 7B, 14B} 和 QwQ-32B-Preview；主实验用 **ThinkPRM-1.5B / ThinkPRM-14B**。

## 3. 打分方式（§E.1）— 我们的评测必须对齐

- 生成 verification CoT（最多 **8192 tokens**），每步标签在 `\boxed{}` 内；
- 然后 **force-decode 字符串 "Is the solution correct?"**，取 **P("yes") / (P("yes")+P("no"))** 作为 solution score。
- DiscPRM：取 **last step** 的分数做聚合。
- ProcessBench 二分类：regex 提取最后的 `\boxed{}`，yes/correct/true → 正类；**解析失败直接计为 incorrect**。
- LLM-as-judge prompt（Figure 29）：steps 从 0 开始编号，`Step_0: <critique>...`，最后 `\boxed{yes/no}`。

## 4. 评测协议（§4.1, §E）

- **MATH-500**：只用 100 题子集（按难度分层抽样），generator Qwen-2.5-14B-Instruct（best-of-N）/ Llama-3.2-3B-Instruct（beam search）。
- **AIME 2024**：generator Qwen-2.5-32B-Instruct；best-of-N 直接用 verifier score 排序（不用 weighted voting）。
- **GPQA-Physics**：86 题；**LiveCodeBench**：v5 的 200 题。
- Best-of-N：**weighted majority voting**（verifier scores 加权），generator 温度 T=0.8（Llama-3.2-3B）/ T=0.4（Qwen-2.5-14B）。
- Beam search：每 beam 采样 M=4 个候选，T=0.6，step 分隔符双换行，最多 20 步。
- Parallel scaling：K 条 CoT（T=0.6）分数平均，记 `model@K`；K=8 ≈ K=4（快速 plateau）。
- Sequential scaling：trigger phrases "Let me double check" / "Let's verify again" / "Did I miss something?"，每轮最多 8192 tokens，最多 R=4 轮。
- 统计：bootstrapped 95% CI（n=100），Figure 23。→ 我们的 3 seeds + 95% CI 与之对齐。

## 5. 关键基线数字（我们的 Part A/B/C 要打败的）

**Calibration（§F.9, Figure 26）** — ECE：
| | MATH-500 | GPQA-Physics |
|---|---|---|
| LLM-as-Judge | 0.238 | 0.311 |
| ThinkPRM-14B | 0.168 | 0.336 |
| DiscPRM-14B | 0.142 | 0.142 |

→ Part A 目标：我们的 GRPO 模型在 MATH 上 **ECE < 0.084**（相对 ThinkPRM 减半），且 Brier 同步下降。

**Inference cost（§F.8, Table 7, 128 OlympiadBench pairs, A40, batch=4）**：
| | 时间 | FLOPs | 说明 |
|---|---|---|---|
| ThinkPRM-14B | 29.3s | 3.48×10¹³ | 自回归，最多 8192 tokens |
| DiscPRM-14B | 0.46s | 2.80×10¹⁰ | 单次前向 |
| LLM-as-Judge-14B | 87.6s | 1.02×10¹⁴ | |

FLOPs 公式：**FLOPs ≈ 2×T×P**（T=生成 tokens，P=参数量）→ Part C 的 FLOPs 对齐评估直接复用该公式。

**Best-of-N（full MATH-500, Table 4, Qwen-2.5-14B generator）**：ThinkPRM-14B 在 N=8/16/32 为 73.6 / 77.2 / 80.2。

## 6. 论文亲口承认的 open problems（§5.8 Limitations）— 正是我们的三个 Part

1. **Calibration**："generative verifiers such as ThinkPRM and LLM-as-Judge are more miscalibrated than DiscPRM… Future work should explore more reliable techniques to extract calibrated scores from generative reward models."→ **Part A**
2. **Step label interference**（论文原生命名）："we observe a phenomenon we term step label interference, where verification errors for earlier steps impact verification of later steps. For example, if the PRM judges a particular step as incorrect, it becomes more likely to label subsequent steps as incorrect even if they are not."→ **Part B**（论文只给了定性描述，我们做定量）
3. **Compute overhead**："they incur additional compute overhead from generating verification chains compared to scalar-valued discriminative PRMs"→ **Part C**

另：§3.2 明确说 "we refrain from more expensive training techniques, such as **reinforcement learning** or preference-based learning"→ RL 是作者亲口留下的空白，Part A 的 GRPO 正好填上。

## 7. 我们的偏离点（deliberate deviations，要在论文里写清楚）

1. **输出格式**：原论文用自由 CoT + `\boxed{}` per-step + force-decode "Is the solution correct?" 取 P("yes")。我们用结构化格式（`Step i: correct|incorrect / First error step / Final / Confidence`），因为 GRPO 需要可解析的 verifiable reward。评测时两种打分都报：我们的 Confidence（Brier/ECE）+ 原论文式 P("yes")/(P("yes")+P("no"))（与基线对齐）。
2. **数据**：原论文 1K synthetic CoTs（QwQ-32B 生成 + 过滤）。我们第一阶段用 PRM800K 全量 + Math-Shepherd（F.5 已证明 auto labels 可用），QwQ-32B 生成 synthetic 数据作为可选增强（需要大显存时再做）。
3. **训练范式**：SFT（rejection sampling / chosen-only）→ 我们的复现基线；GRPO（all chains + calibration reward）→ 我们的核心贡献。
