# 六模型准确度升级计划

执行者按本文件编码并跑完。不要改计划里的指标、切分和停止线。讨论写在 `models/upgrades/NOTES.md`，代码和结果写在 `models/upgrades/` 与 `runs/model_registry_v1_upgrades/`。

## 不许做的事

- 不改 `focus_v8/`、`precision_v9/`、五个既有 `spec.json`、`runs/model_registry_v1/` 里已有文件。
- 不上传 R2，不调用交易接口，不把触及率写成盈利。
- 不用 eval 行选择参数、特征或阈值。eval 每个候选只评一次。
- 不把未达线的数字改成达标。某一轮没到 10%，记未达到，进入下一轮预写方案。
- 训练用 `research/after_open_3d5pct/.venv/bin/python`，工作目录是仓库根 `/Users/admin/Code/stock`。

## 指标（已冻结）

比较对象是 `runs/model_registry_v1/ACCURACY.json` 里的 `accuracy_at_0.5`。

五个既有模型的切片是三群并集 21 只股票的 eval 行（378 行）。群成员以 `focus_v8/protocol.json` 的 `groups` 为准，交叉股票只算一次。主列是 `y_3d_5pct` 与 `p_3d_5pct`。概率阈值固定 0.5，不准改成别的数再叫准确度。

`SNXX_3d3pct` 的切片是它自己的 eval 全部 15 行，列是 `y` 与 `p`。

相对提升 = `新准确度 / 基线准确度 - 1`。停止线是 **≥ 0.10**。

| 模型 | 基线准确度 | 停止线 |
|---|---:|---:|
| B_group | 0.5370370370370371 | 0.590741 |
| B_no_group | 0.5529100529100529 | 0.608201 |
| C_group | 0.5555555555555556 | 0.611111 |
| C_no_group | 0.58994708994709 | 0.648942 |
| C_no_daily | 0.4576719576719577 | 0.503439 |
| SNXX_3d3pct | 0.8666666666666667 | 0.953333 |

基线概率来自：

- 五模型：`runs/model_registry_v1/<id>/reserved/{tune,cal,eval}.parquet`
- SNXX：`runs/model_registry_v1/SNXX_3d3pct/{tune,cal,eval}.parquet`

tune 用来选参数。cal 不参与选择，只作为旁证写进结果。eval 只在参数冻结后打分。

一个候选要被保留，必须同时满足：tune 准确度高于该模型基线概率在 tune 上的准确度。否则这一轮回退到上一轮保留的概率，eval 仍要记下但不能当成升级。

已经达标的模型不再进入后面的轮次。最多三轮。三轮后仍未达标的模型保持未达标，不要再开第四轮。

每一轮先把本轮唯一改动写进 `models/upgrades/round_XX/PROPOSAL.md`，再跑代码。

## 第一轮：tune 上选择的 logit 偏置

新文件 `models/upgrades/round_01/apply.py`。

对每个模型的 tune 行，在网格 `b = -2.0, -1.95, ..., 2.0` 上计算

`p' = sigmoid(logit(clip(p, 1e-4, 1-1e-4)) + b)`

用该模型的切片（五模型用 21 只并集，SNXX 用全部 tune 行）选使 `p' >= 0.5` 准确度最高的 `b`。平局取绝对值更小的 `b`，再平就取 0。

把选中的 `b` 原样用到 eval。不要在 eval 上再搜。

输出 `runs/model_registry_v1_upgrades/round_01/result.json`，每个模型包含：`b`、tune 基线准确度、tune 新准确度、eval 基线准确度、eval 新准确度、相对提升、是否保留、是否达标。

## 第二轮：与本股历史基准混合

只处理第一轮之后仍未达标的模型。

五模型的 parquet 里已有因果列 `history_3d_5pct`（拟合时只用当时已成熟的本股历史）。在第一轮保留下来的 `p1` 上：

`p2 = w * p1 + (1 - w) * history_3d_5pct`

`w` 取 `{0, 0.25, 0.5, 0.75, 1}`，只在 tune 上选。历史列缺失或非有限的行不进这一轮的选择和评分，并单列删除数。若删除后 tune 或 eval 少于原切片的 90%，这一轮作废并回退。

SNXX 没有这列。第二轮改训练，不混历史列。在 `models/upgrades/round_02/snxx_train.py` 里复制 `SNXX_3d3pct/labels.py` 的行构造，额外加一列 `extension_20 = 最后一根已完成 K 的 close / 过去 20 根 high 的最大值 - 1`。这列只用 `end <= 11:30` 的 K。LightGBM 配方与原实验相同，只把 `scale_pos_weight` 在 `{1, 2, 4}` 里用 tune 准确度选择。种子仍是 3566。输出写到新目录，不覆盖 `runs/model_registry_v1/SNXX_3d3pct/`。

## 第三轮：分群偏置

只处理第二轮之后仍未达标的五模型。在第二轮保留的概率上，为 chips、optics、storage 各选一个偏置，交叉股票使用三个群偏置的平均。网格仍是 `-2` 到 `2`、步长 `0.1`。三个偏置要在 tune 上联合搜索；若全网格太大，用坐标下降，每群轮流搜一遍，顺序固定为 chips、optics、storage，只循环两遍。起点是 0。选择标准仍是并集 tune 准确度。

SNXX 若仍未达标，第三轮把 `min_child_samples` 固定为 4，`num_leaves` 固定为 4，`scale_pos_weight` 在 `{1, 2, 4, 8}` 里只按 tune 选。其他配方不变。这是最后一个候选。

## 交付

`models/upgrades/run_all.py` 顺序执行三轮，每轮读上一轮保留的概率。跑完后写：

- `runs/model_registry_v1_upgrades/SCOREBOARD.json`
- `models/upgrades/SCOREBOARD.md`

记分板每行一个模型：基线准确度、最终准确度、相对提升、停在第几轮、是否达标、保留下来的参数。另写三群各自的最终准确度，以及相对多数类准确度的差。多数类准确度是「永远猜 tune 里更常见的那一类」在 eval 上的准确度，用 eval 标签算，但它不是停止指标。

跑完用同一解释器执行：

```bash
research/after_open_3d5pct/.venv/bin/python -m research.after_open_3d5pct.models.upgrades.run_all
```

把记分板里的六个相对提升原样留在 `SCOREBOARD.md` 顶部。不要为了凑 10% 改阈值定义或改 eval 日期。
