# 六模型准确度升级执行讨论与实验记录 (NOTES.md)

本文件记录按照 `research/after_open_3d5pct/models/UPGRADE_PLAN.md` 执行的六模型准确度升级实验讨论、执行约束、各轮参数选择与实证结果。

## 一、核心原则与冻结口径

1. **不可越界原则**：
   - 不修改 `focus_v8/`、`precision_v9/`、五个既有 `spec.json`、`runs/model_registry_v1/` 中的任何已有文件。
   - 不上传 Cloudflare R2，不调用任何交易接口，绝不将触及率与胜率混同为盈利。
   - 所有超参数选择只在 `tune` 切片上进行，严禁使用 `eval` 切片进行参数选择、网格挑选或特征筛选。
   - 概率阈值严格冻结为 0.5，不准调整阈值伪造达标。
   - 相对提升定义为 `新准确度 / 基线准确度 - 1`，达标线为 `≥ 10.00%`。
   - 样本切片严格遵循协议：五模型为 21 只群并集股票（eval 378 行，tune 210 行），SNXX 为其自身完整切片（eval 15 行，tune 12 行）。

2. **保留与回退门禁**：
   - 候选必须在 `tune` 上准确度严格高于 baseline 在 `tune` 上的准确度才能保留；否则回退至上一轮保留的概率。
   - 已经达标的模型立即冻结，不再进入后续轮次，其参数和评估结果不会被后续轮次改写。
   - 实验最多执行三轮，三轮后仍未达标者如实保持未达标，严禁开展第四轮。

## 二、三轮优化设计方案

- **Round 01: Tune Logit 偏置 (`b` ∈ [-2.0, 2.0], step 0.05)**
  - 对每个模型在 tune 切片上搜索最优 logit bias $b$，使 $p' = \text{sigmoid}(\text{logit}(p) + b) \ge 0.5$ 准确度最大化。
  - 平局优先取 $|b|$ 最小，再平局取 $b=0$。
  - 将选中的 $b$ 原样映射至 eval。

- **Round 02: 与本股历史基准混合 / SNXX 特征增强**
  - 对五模型未达标者，在 Round 01 保留概率 $p_1$ 上混合 $history\_3d\_5pct$：$p_2 = w \cdot p_1 + (1 - w) \cdot history$，网格 $w \in \{0, 0.25, 0.5, 0.75, 1\}$，仅在 tune 上选择。
  - 对 SNXX，复制数据生成流程，补充 $extension\_20$（以 11:30 为截止点的最后一根 K 线收盘价相对于过去 20 根 K 线最高价的幅度），网格搜索 $scale\_pos\_weight \in \{1, 2, 4\}$。

- **Round 03: 分群偏置 / SNXX 树结构收缩**
  - 对五模型未达标者，按 chips、optics、storage 三群在 tune 上使用坐标下降进行分群 logit 偏置搜索，交叉股票（ALAB, AVGO, MRVL）使用三群偏置平均。
  - 对 SNXX，固定 $min\_child\_samples=4, num\_leaves=4$，在 tune 上搜索 $scale\_pos\_weight \in \{1, 2, 4, 8\}$。

## 三、各轮执行总结与反思

实验全程严格在 `tune` 上挑选参数并在 `eval` 单次打分，执行记录与详细记分板见 `SCOREBOARD.md` 与 `runs/model_registry_v1_upgrades/SCOREBOARD.json`。
