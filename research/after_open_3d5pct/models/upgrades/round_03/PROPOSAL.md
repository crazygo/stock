# 第三轮升级提案：分群偏置 / SNXX 树结构收缩

## 唯一改动

仅对前两轮之后仍未达标的模型进行处理（第一轮已达标的 `C_no_daily` 保持冻结跳过）。

### 1. 五个既有模型（未达标标的）：分群 Logit 偏置

在第二轮保留的概率 $p_2$ 基础上，引入基于股票所属产业群（`chips`、`optics`、`storage`）的分群 logit 偏置：

$$p_3 = \sigma\left(\text{logit}(\text{clip}(p_2, 10^{-4}, 1 - 10^{-4})) + b_{\text{symbol}}\right)$$

- **交叉股票规则**：对于同时属于多个群的股票（`ALAB`, `AVGO`, `MRVL`），使用三个群偏置的算术平均值：
  $$b_{\text{inter}} = \frac{b_{\text{chips}} + b_{\text{optics}} + b_{\text{storage}}}{3}$$
  非交叉股票直接使用其所属群的偏置。
- **网格与搜索策略**：偏置网格为 $[-2.0, 2.0]$，步长 $0.1$。在 21 只并集股票的 `tune` 切片上采用坐标下降法进行搜索：
  - 起始点：全部群偏置设为 $0.0$。
  - 优化轮次：固定循环两遍（2 cycles）。
  - 优化顺序：严格按照 `chips` $\to$ `optics` $\to$ `storage` 依次单维度优化。
  - 目标函数与平局规则：最大化并集 `tune` 准确度；平局时优先选择绝对值更小者，再平局优先取 $0$。

### 2. SNXX_3d3pct：树结构收缩与正样本加权

- **脚本路径**：`models/upgrades/round_03/snxx_train.py`。
- **模型结构调整**：将 `min_child_samples` 固定为 4，`num_leaves` 固定为 4。
- **超参数选择**：在 $scale\_pos\_weight \in \{1, 2, 4, 8\}$ 中仅根据 `tune` 准确度选择最优权重。
- **输出隔离**：输出写入 `runs/model_registry_v1_upgrades/round_03/SNXX_3d3pct/`。

## 保留与达标规则

- **保留条件**：`tune` 准确度必须严格高于该模型基线概率在 `tune` 上的准确度，否则回退至第二轮保留的概率。
- **达标与停止**：在 `eval` 切片上计算相对提升 $\ge 0.10$ 则标记为达标。三轮结束后仍未达标者保持未达标，严禁开展第四轮。
