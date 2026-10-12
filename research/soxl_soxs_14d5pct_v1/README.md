# SOXL / SOXS：14 自然日 +5% 研究迁移

> 当前要求已更新：用户不接受负收益月份，5、7、9 月只作回归用例，不得定向拟合；允许每天学习当时已成熟的数据。以下为历史版本记录；旧算术均值通过不再授予当前买入资格。最新每日训练研究与未达标结论见 [v4](../soxl_soxs_14d5pct_v4/README.md)。

2026-10-12 从 `SOXL_SOXS_14day_5pct_research_2026-10-12.zip` 迁入本项目，工程标识为 `soxl_soxs_14d5pct_v1`，原协议版本为 `soxl-soxs-14calendar-5pct-v1`。

用户澄清后的实现与新回测见 [v2 任意时点分析](../soxl_soxs_14d5pct_v2/README.md)。v2 固定 20 日趋势路线的 09:25 盘前分析已达到三个指定月净收益均值 +11.69%；本目录继续保留原日线模型及其失败结果。

**v1 迁移和离线复算已完成，原日线主模型收益未达目标；v1 未实现任意时点及盘前分析。** 28 份数值结果与原包一致；12 项检查通过，101 条跨策略已完成交易的小时事件一致。主模型 2026 年 5–9 月累计净收益 -34.73%，收盘净值最大回撤 66.05%。目的、进展、问题和后续工作见 [ANALYSIS.md](ANALYSIS.md)。

## 当前目标与验收口径（2026-10-12 用户澄清）

- **分析启动时间：任何时间。** 至少覆盖盘前的一个分析时点；每次分析明确实际启动时间、可用数据截止和可执行的入场时间。用户未限定唯一分析时点；v2 已登记并评价盘前 09:25:30 ET。不能将“前一常规盘收盘后分析、次日开盘买入”写成用户要求。
- **动作：买入 SOXL、买入 SOXS，或不行动。** 买入时设置以实际买入成交价的 +5% 为目标的自动卖出止盈单，通过操作争取在 14 天内实现 5% 收益；第 14 天仍未止盈的持仓强制卖出，并计入其实际盈亏。
- **回测通过标准：2026 年 5、7、9 月账户净收益的算术均值达到 10%。** 这是用户确认的“整体期望月收益率”验收口径。完整连续资金账本保留全部模拟操作收益、到期强制卖出的收益或亏损、成本和跨月持仓，不要求每个单月分别达到 +10%，也不以止盈触及率代替操作收益。

本目录代码仍是前一完整常规盘日线决策、次日开盘模拟买入的历史版本，没有实现上述任意时点及盘前分析。原包的“各单月 ≥10%”字段保留作历史证据；本节采用用户确认后的当前需求。完整账本和三个指定月净收益算术均值已在 [v2 协议](../soxl_soxs_14d5pct_v2/protocol.json)中固定；不能直接把旧版字段当作新版本完整验收结果。

## 保存位置

| 内容 | 位置 |
| --- | --- |
| 项目运行代码、说明、检查和摘要证据 | 本目录 |
| 原作者代码、协议、说明与清单的原字节副本 | `original/` |
| 完整的 52 文件原包，包括行情、旧结果和冻结模型 | 仓库根目录 `.cache/soxl_soxs_14d5pct_v1/imports/2026-10-12/SOXL_SOXS_research/` |
| 本次从源码重新训练及复算的结果 | 仓库根目录 `.cache/soxl_soxs_14d5pct_v1/runs/migration_2026-10-12/` |
| 独立 Python 环境 | 仓库根目录 `.cache/soxl_soxs_14d5pct_v1/venv/` |

原始数据、逐日特征/标签、模型二进制和生成结果沿用项目现有 `.cache/` 管理，不进入 Git；无需修改忽略规则。`original/STATUS.json` 中 R2 访问受阻描述是原运行的历史记录。本次仅使用 ZIP 与已有本地文件，没有访问 R2 或新取行情。新机器或清理缓存后，需要重新导入原 ZIP；仅克隆源码不会获得完整输入。

## 在仓库根目录运行

原包固定 NumPy 2.3.5、pandas 2.2.3、scikit-learn 1.8.0。本机使用独立 Python 3.12.13 环境，避免改变项目现有 Python 3.14 依赖。`requirements-reproduction.txt` 额外固定本次复算的全部依赖；`requirements.txt` 保留原包要求。

```bash
uv venv --python 3.12 .cache/soxl_soxs_14d5pct_v1/venv
uv pip install --python .cache/soxl_soxs_14d5pct_v1/venv/bin/python -r research/soxl_soxs_14d5pct_v1/requirements-reproduction.txt
python3 research/soxl_soxs_14d5pct_v1/import_package.py /Users/admin/Downloads/SOXL_SOXS_14day_5pct_research_2026-10-12.zip

.cache/soxl_soxs_14d5pct_v1/venv/bin/python research/soxl_soxs_14d5pct_v1/backtest.py --output .cache/soxl_soxs_14d5pct_v1/runs/reproduction_new
.cache/soxl_soxs_14d5pct_v1/venv/bin/python research/soxl_soxs_14d5pct_v1/verify.py --results .cache/soxl_soxs_14d5pct_v1/runs/reproduction_new
.cache/soxl_soxs_14d5pct_v1/venv/bin/python research/soxl_soxs_14d5pct_v1/compare_results.py --results .cache/soxl_soxs_14d5pct_v1/runs/reproduction_new
```

依赖安装完成后，导入、训练、回放和核对均可离线运行。新复算请选新输出目录；省略 `--output` 时自动使用 UTC 时间命名。`--raw-dir` 可以显式指定输入目录，但换数据后的研究结果不再代表原批次；与原包不同会使数值比较失败。模型代码重新生成自己的 pickle，不读取导入的 pickle。

## 迁移改动与验证

- 模型、60 个特征、标签、月前选模、成本、止盈和资金账本公式保留原版。
- 输入改为项目内冻结导入路径；新结果与原包分开，运行与验证均拒绝写入原包目录。
- `import_package.py` 校验 ZIP 路径、全部成员、51 个 SHA256 和本版固定清单，重复导入不会覆盖不同内容。
- 原小时检查只数已出现日期的柱数。迁移版按官方开闭市时刻逐日核对每根应有柱，加入整日缺失及半日市回归检查；小时事件不一致时返回非零状态。实际本批次全部 101 条路径仍通过。
- `compare_results.py` 对照 28 份 CSV/JSON，数值容差为相对 `1e-11`、绝对 `1e-12`。新增小时审计列单独验收；PNG 与 pickle 不做跨平台字节等同比较。

摘要证据见 [migration.json](evidence/migration.json)、[reproduction.json](evidence/reproduction.json)、[原结果摘要](evidence/imported_summary.json)。完整逐笔结果保存在上述缓存目录。

## 本地 5 分钟数据覆盖审查

可使用项目已有 pandas / pyarrow 环境运行：

```bash
python3 research/soxl_soxs_14d5pct_v1/audit_local_data.py --output .cache/soxl_soxs_14d5pct_v1/local_data_audit_new.json
```

本次只读审查见 [local_data_audit.json](evidence/local_data_audit.json)。这些 Futu 未复权数据尚未接入原 IBKR 拆股复权模型；检查了时间戳、时段、口径元数据和柱数，尚未验证价格、公司行动及订单执行。5 分钟扩展需另立版本与验证区间。
