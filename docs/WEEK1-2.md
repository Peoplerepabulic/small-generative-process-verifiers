# Week 1–2 kickoff — 跑在 Kaggle 上（免费，不用等集群）

目标：环境跑通 + 数据管线验证 + 两个便宜基线（DiscPRM、SFT）出分。
GRPO 正式训练等 A6000 / WashU 集群，这两周先做 pilot（短序列）验证 reward 能学起来。

## Day 1: 环境 + 数据（~2–3 GPU 小时）

Kaggle 新建 Notebook，Accelerator = GPU T4 x2，Internet = ON，然后：

```bash
!pip install -q -r requirements.txt   # 上传本仓库为 Kaggle Dataset，或 git clone
!python scripts/data/prepare_prm800k.py --out data/prm800k_unified.jsonl
!python scripts/data/prepare_mathshepherd.py --out data/mathshepherd_unified.jsonl --max_records 20000
```

检查点：
- [ ] `data/prm800k_unified.jsonl` 约 931+ 条（phase1_train），`gold_0` 占比 ~40%
- [ ] Math-Shepherd 解析失败率 < 1%（脚本会打印 parse_failed）
- [ ] 跑 `python -c "from scripts.train.rewards import parse_verification; ..."` 确认 parser 正常

> Kaggle 磁盘只有 20GB：数据放 `/kaggle/working`，大的 checkpoint 及时转存为 Kaggle Dataset。

## Day 2–3: DiscPRM 基线（~4–6 GPU 小时）

```bash
!python scripts/train/train_discprm.py --config configs/discprm_1p5b.yaml
```

检查点：
- [ ] 训练 loss 下降、无 NaN
- [ ] 在 phase1_test 上 step-level accuracy 明显 > 随机（~60%+ 才算 pipeline 没 bug）

## Day 4–5: SFT 基线 = rejection sampling 代理（~4–6 GPU 小时）

```bash
!python scripts/train/train_sft.py --config configs/sft_1p5b.yaml
```

检查点：
- [ ] 模型能稳定输出目标格式（`parse_verification` 解析成功率 > 90%）
- [ ] 用 `scripts/evaluation/evaluate.py` 在 phase1_test（200 条）上出分：
  `!python scripts/evaluation/evaluate.py --model outputs/sft-1p5b --data data/prm800k_unified.jsonl --max_n 200`
- [ ] 记录：step accuracy / solution accuracy / **ECE / Brier** —— 这是 Part A 要打败的基线

## Day 6–7: GRPO pilot（短序列，验证 reward 学得起来）

```bash
!python scripts/train/train_grpo.py --config configs/grpo_1p5b.yaml
# pilot 先改小：num_generations=4, max_completion_length=512（直接改 yaml）
```

检查点（proposal 里的熔断机制）：
- [ ] `r_agree` / `r_cal` 随训练上升、`r_format` 快速收敛到 ~0.2
- [ ] 不发散 → 第 4 周全量 GRPO 可继续；发散 → 降级保底方案（SFT + temperature scaling + cascade）

## 并行：算力邮件

- [ ] 发邮件 1 → `eitsupport@email.wustl.edu`（McKelvey 集群）——草稿在 `~/workspace/phd-outreach/compute-access-emails.md`
- [ ] 等黄教授回复 10/4 邮件后再发邮件 2（GPU/sponsor 请求）

## 本周产出

`docs/baselines-week2.md`：三个基线（DiscPRM / SFT / SFT+temperature scaling）在 phase1_test 上的
accuracy + ECE + Brier 表格。这是整个项目的地基，也是 Part A 的对照组。
