# 盘前尾期：三条训练改动的组合

对照是 `premarket_tail_v1` 第二轮，数字在 `runs/premarket_tail_v1/result.json` 的 `round2`。输入仍是那 6 列。买入线仍是拟合行分数的 80 分位。标签、purge、8 周窗口、21 只股票都不改。考试周不参与拟合，也不参与早停。

Pugh 开跑前高于 0 分的三条，按 `focus_v8/PUGH.md` 的权重：

| 开关 | 工程改动 |
|---|---|
| `reg` | `min_child_samples=150`，`reg_lambda=30`。对应 `focus_v8/run.py` 里 `recipe["regularization"]` 为真时的 B 配方 |
| `early` | 8 个训练周的前 7 周拟合，最后 1 周里已经成熟的行做 `eval_X` / `eval_y`，`early_stopping(20)`。分类器用 `auc`，排序器用 `ndcg`、`eval_at=[3]` |
| `rank` | `LGBMRanker(objective="lambdarank", label_gain=[0, 1])`，`group` 按 `symbol` 在拟合行里的行数。80 分位用的是排序分数，不是概率 |

其余超参保持 `n_estimators=40`、`num_leaves=7`、`max_depth=3`、`learning_rate=0.05`、`random_state=3566`。不开 `reg` 时，`min_child_samples=10`、`reg_lambda=10`。

7 个非空组合一次跑完，标识为 `R`、`E`、`K`、`RE`、`RK`、`EK`、`REK`。不看前一个组合的考试结果再决定下一个开不开。跑完不再加第 8 个组合。

早停周成熟行少于 20，或拟合行少于 40，或其中只有一种标签，该 triplet 跳过。跳过任何一个 triplet 的组合，不与对照比买入命中率：对照是 127 组、测试 13075 行、验证 13085 行。

一个组合胜过对照，只有测试块和验证块的 precision 都严格更高，并且两侧买入都不少于 15 笔。上界是这些胜过对照的组合里，两周 precision 较低的那一周最高的一个。并列时取两侧买入笔数之和较大的标识，再并列按标识字典序。没有组合胜过对照，上界仍是第二轮。验证周不单独挑组合。

80 分位只在真正 `fit` 的那些行上计算。早停周不进入这批行。
