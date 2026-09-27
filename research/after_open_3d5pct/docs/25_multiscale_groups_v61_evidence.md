# v6.1 多尺度、动态群、多目标：交付证据（2026-09-26）

后续发现C群类型身份在本版被平均抹平；本页保留冻结的v6.1证据，最新的仅C结构修正及配对结果见 [v6.1.1证据](27_multiscale_groups_v611_typefix_evidence.md)。

**结论：C路线没有提升主目标。** 在完全相同的2026开发期外层1,140行、12个ET交易日、95只股票上，C有群的3日+5% Brier为 **0.162862**，同信息范围的LightGBM B有群为 **0.144071**。差额C−B=+0.018791，按5个连续交易日同步重采样的探索性95%区间为 **[+0.015094,+0.025069]**。这是已暴露2026历史，不能解释为独立失败概率或未来保证。没有模型达到上线/行动门禁；保留全部旧v3–v5.1与v6制品，不改看板、不自动交易。

## 数据、时间与来源

只读本地 `market_data/us_5m/**/2026.parquet` 的未复权NONE行情；对应Cloudflare R2 `us_5m/` 为109个对象且全是`2026.parquet`，本地也是109份（106从1月、2从6月、1从8月开始），全部最迟到9月。`us_60m_raw/` 本地/R2亦各109份，但部分股票（例如AAPL）只从8月起；本试验的60m和完整日级均从同一5m按ET官方时段聚合。旧QFQ 60m和2025–26已调整的全市场日归档没有混入。没有向OpenD请求或向R2上传；现有来源不足以检验多年/不同市场状态记忆。

冻结配置、逐文件SHA、源质量与schema在 [`multiscale_groups_v61.json`](../configs/multiscale_groups_v61.json) 和 `runs/multiscale_groups_v61_dataset_20260926_r2/manifest.json`。共有101只当前名单回溯候选×日期的9,696个11:30机会，9,222行有完整最长5日标签；396行因拆股回看排除、57行缺/迟到前缀、21行最长窗口未成熟或缺数据。`available_at`是归档假定bar结束+1秒，源无实测`received_at`；代码若有该列则取两个到达时刻较晚者。构建前校验5m真实起止，输入最长146日前源时刻都不晚于决策，标签可用时刻取完整390根bar真实最大可用时刻及窗口结束的较晚者。迟到历史源保守拒绝当前样本，不将其喂给相对量分母。未来追加不改过去特征的合成检查通过。矩阵周标签的valid time核对周生效区间和上一周`history_end`；knowledge time实际为2026-09-25重建，**不是历史现场PIT**。行业41群仅覆盖101股中的45股，且来源为当前快照回填，因此未进入主模型；ETF当前自选标签亦未输入。GOOG/GOOGL按单一发行人固定顺序去重。

三个时序分支保留前7个完整session和当日已完成前缀5m、前30日和当日已完成小时60m、前126个完整session日级；缺失/部分时段、实际柱长、盘前/常规/盘后、价格跨段连接、实际价格/股数/美元额与自身历史同类时段相对额都有显式通道或mask。C对每股编码一次，群分支包含近6个有序周的成员/同伴状态；B取同一原始信息范围的时序摘要和同一群历史摘要。九个共享目标是1/3/5日×+3/+5/+8%，按完整正常交易分钟从同一个11:35代理入场起算；输出经二维保序投影。3日期末Close仅为未触及时的估值，不是强制出售或止损。v6.1是在查看v6结果后登记的绝对价格与成交股数尺度工程补正；与v6的样本、九标签和原12个行情通道逐元素一致。另修复双股类无序集合导致的群状态不确定性。v6.1不能算新独立验证。

## 固定五候选的配对结果

训练/内层/外层分别5,606/1,521/1,140行，覆盖58/16/12个ET日期；按最长目标真实结束和可用时间purge，训练最后可用日期为7月24日，外层为9月1–17日。全部2026日期均标为exposed。固定单种子、C最多5 epoch、B各目标120树，无外层选参、无概率校准。表中Brier、logloss、AUC与十等距桶ECE均为3日+5%外层；实际率0.20965。

| 候选 | Brier ↓ | logloss ↓ | AUC ↑ | ECE10 ↓ | 内层Brier |
|---|---:|---:|---:|---:|---:|
| 训练期常数率 | 0.170554 | 0.526281 | 0.500 | — | — |
| B LightGBM 有群 | **0.144071** | **0.444989** | **0.7875** | **0.0610** | **0.159787** |
| B LightGBM 无群 | 0.139406 | 0.439708 | 0.7965 | 0.0703 | 0.163588 |
| C patch 有群、有日级 | 0.162862 | 0.506409 | 0.7071 | 0.0790 | 0.172919 |
| C patch 无群 | 0.157473 | 0.489810 | 0.7190 | 0.0773 | 0.171569 |
| C patch 无日级 | 0.159534 | 0.496002 | 0.7090 | 0.0786 | 0.172144 |

B无群虽在已暴露外层点估计最低，但内层逊于B有群，不能据外层反选并宣称验证成功。C有群比C无群的Brier高0.005389，5日日期块区间[+0.000560,+0.008604]；C有日级比无日级高0.003328，区间[+0.000624,+0.004973]，都未显示增量。B无群与B有群的外层差额区间[-0.014163,+0.009287]跨零。训练期完整126日仅1,316/5,606行（23.5%），内层1,489/1,521（97.9%），外层1,116/1,140（97.9%）；覆盖随日期跃迁，绝不能把日级消融解释为126日记忆的独立证明。过去严格>60%质量子组在外层只有27行，无法形成稳定子组结论；逐日期、股票、目标与子组明细分别见运行目录`diagnostics_*.json`及`predictions_*.parquet`。

原始B有群九概率有6.58%的行违反期限/涨幅偏序，投影后为0；C有群原始及投影后均为0。保序不等于校准，所有输出仍是未校准模型概率估计。外层未命中901行，3日期末gross均值−1.83%、10分位−6.39%、1分位−12.31%；按命中固定+5%、未命中期末估值和买卖各6bp的**描述性**净情景，全体均值−0.51%、ES95为−10.60%。没有中途止损、仓位管理或实际成交证明。与旧v5/v5.1质量池六时点Brier不是同样本、同信息范围，不作配对改善宣称。

## 可重放命令与边界

从仓库根目录运行，输出均使用新的空目录名；数据构建不联网。当前冻结目录为`runs/multiscale_groups_v61_dataset_20260926_r2/`和`runs/multiscale_groups_v61_fit_20260926/`，分别含行级特征/标签/源清单以及五组模型、逐行预测、评价、校准桶与风险观察。实测构建118秒、五候选训练37.5秒；第一次混用OpenMP的v6试跑停滞已作为失败记录保留，单线程重试成功。没有无界调参。

```bash
research/after_open_3d5pct/.venv/bin/python -c 'import json; from pathlib import Path; from research.after_open_3d5pct.v6_data import build; c=json.load(open("research/after_open_3d5pct/configs/multiscale_groups_v61.json")); build(c, Path("research/after_open_3d5pct/runs/<new-dataset>"))'
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 research/after_open_3d5pct/.venv/bin/python -m research.after_open_3d5pct.train_multiscale_v6 run --config research/after_open_3d5pct/configs/multiscale_groups_v61.json --dataset research/after_open_3d5pct/runs/<new-dataset> --output research/after_open_3d5pct/runs/<new-fit> --reuse-dataset
research/after_open_3d5pct/.venv/bin/python -m research.after_open_3d5pct.evaluate_multiscale_v6 --run research/after_open_3d5pct/runs/<new-fit> --output research/after_open_3d5pct/runs/<new-fit>/analysis.json
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 research/after_open_3d5pct/.venv/bin/python -m research.after_open_3d5pct.train_multiscale_v6 predict --run research/after_open_3d5pct/runs/<new-fit> --dataset research/after_open_3d5pct/runs/<new-dataset> --model B_lgbm_group --output research/after_open_3d5pct/runs/<new-fit>/replay_B.parquet
```

保存模型对冻结数据集的B与C重放预测均与原外层逐目标**完全一致（最大绝对误差0）**。79项工程测试、v1配置检查、合成smoke、Python编译及`git diff --check`通过。模型只支持此版11:30离线重放，尚未恢复旧v3六时点，也无真实前向`received_at`或原始历史PIT成员。当前仅建议把B家族保留为下一次前向比较的研究参照；最重要的下一步是从新交易日起冻结真实接收时间和当时成员快照，再做未暴露前向配对验证。
