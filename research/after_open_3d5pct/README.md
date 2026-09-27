# 开盘后 3 日 +5%：训练工程

2026-09-27 已完成用户授权的 [v7 五路线、各三轮真实迭代](iterations_v7/REPORT.md)，承接 [v6.1.1 多尺度与动态群组证据](docs/27_multiscale_groups_v611_typefix_evidence.md)。此处 B 表示 LightGBM、C 表示轻量序列模型，与旧方案的 H/A/B 输入模块分别命名。五路线为 B有群、B无群、C有群、C无群、C无日级；保持九目标和相同样本，分别诊断瓶颈、登记修改、真实拟合和比较。C无群与C无日级修复了原训练缺陷，但增加曲线特征、群差分或任务权重没有持续改善；全部版本和负结果均保留。所有2026日期仍是已暴露开发数据。完整边界见 [协议](iterations_v7/PROTOCOL.md)及[诊断说明](iterations_v7/SHAPE_AND_DATA_NOTES.md)。

状态：**基础工程及每小时一次运行研究系统已交付；v5 因果质量股票池的有限真实训练和评价已完成，但没有通过概率改善或行动证据开发目标，更无独立验证通过的高置信模型。** 旧训练登记见 [08](docs/08_training_launch.md)，每小时版本的真实结果和效果边界见 [16](docs/16_hourly_v3_delivery_evidence.md)；v5 的冻结口径与负结果见 [19](docs/19_quality_training_v5_registration.md)、[20](docs/20_quality_training_v5_evidence.md)。

已继续执行独立 v5.1 校准策略迭代：[21 · 预登记](docs/21_quality_calibration_v51_registration.md)、[22 · 实测与复现](docs/22_quality_calibration_v51_evidence.md)。只在内层选择原始输出或 sigmoid，复用 23 个基础模型；August 内层改善成立，但外层改善较小且区间跨零，September 保持原模型。仍无行动达标模型，不能把内层选后改善称为独立验证。

新增无中途止损的三日期末估值研究版本 `three_day_touch_terminal_valuation_v4`：目标仍是延迟入场后 1170 常规分钟内触及 +5%，未触及则用期末 Close 做可变现估值；成本与全体尾损另行评价，**不代表真实成交或三日强制卖出**。冻结候选、时间折和开发风险场景见 [17](docs/17_terminal_risk_v4_registration.md)，真实训练、负结果与只读预测命令见 [18](docs/18_terminal_risk_v4_evidence.md)。2026 外层已暴露、用户风险预算未定，当前没有正式行动模型。本版本不改研究看板。

新增只读 [模型研究看板](dashboard/index.html) 与 `manual_three_cutoffs_v1` 模拟操作回放。产品/指标口径见 [09](docs/09_dashboard_product_metrics.md)，[产物与回放契约](docs/10_artifact_and_replay_contract.md)、[启动手册](docs/11_training_and_dashboard_runbook.md)、[未启动研究登记](docs/12_future_research_register.md) 分别说明可复现接口、运行方法及后续边界。首轮制品仍是开发期探索，页面不能作为交易建议。新阶段的每小时一次运行系统采用独立的 `hourly_once_v3` 协议；五个实施里程碑见 [13](docs/13_hourly_once_implementation.md)。

核心问题：在预筛股票池中，用户开盘后实际会查看的时刻，从延迟后可成交价格出发，未来 3 个交易日内再上涨 5% 的概率是多少？用户手动决定是否下单，程序不自动交易。

本轮采用 **仅行情输入、LightGBM 与小型多尺度 TCN 双路线**。主方案为 H（历史背景）＋A（盘后/夜盘/盘前）＋B（开盘后路径），用相同时间窗口做四组输入 × 两种模型对照。设计详见 [07 · 行情双路线训练方案](docs/07_model_training_plan.md)；固定参数的首轮探索结果见 [08 · 训练记录](docs/08_training_launch.md)。

这是独立研究工程，目录为 `research/after_open_3d5pct`。旧的 Outcome Matrix、路径分段和热度研究继续作为历史研究保留，新工程不继承它们的实时预测能力结论。

## Coder 从这里开始

开始设计、编码、训练、评估前，按顺序完整阅读：

1. [AGENTS.md](AGENTS.md)：必须遵守的边界与交付要求。
2. [01 · 目标与研究思想](docs/01_mandate.md)：用户意图、目标演进与默认决策。
3. [02 · 数据和时间契约](docs/02_data_time.md)：当时可知、股票池、交易日历、价格与输入输出。
4. [03 · 样本、标签和特征](docs/03_features_labels.md)：精确计算口径和特征登记。
5. [04 · 训练和验证协议](docs/04_validation.md)：基准、purge、校准、独立测试和通过标准。
6. [05 · 实施路线和验收](docs/05_delivery.md)：下一位 coder 的具体任务与完成标准。
7. [06 · 数据采集可行性与交接单](docs/06_data_acquisition_feasibility.md)：有日期的现场证据、频率/完整性约束与本轮行情数据范围。
8. [07 · 行情双路线训练方案](docs/07_model_training_plan.md)：H/A/B 定义、LightGBM/TCN 输入表示、4×2 对照与选择规则。
9. [08 · 首轮训练启动登记](docs/08_training_launch.md)：本轮真实数据探索训练的固定配置、实际结果与局限。
10. [09 · 看板产品与指标验收](docs/09_dashboard_product_metrics.md)：三个操作时点、各比例分母、下钻和页面验收。
11. [10 · 产物与回放契约](docs/10_artifact_and_replay_contract.md)：ModelSpec/TrainingRun/PredictionSet/PolicyRun/Report、哈希和状态机。
12. [11 · 训练与看板启动手册](docs/11_training_and_dashboard_runbook.md)：检查、配置、启动、日志、失败和复现。
13. [12 · 后续研究登记](docs/12_future_research_register.md)：只登记，不自动训练。
14. [13 · 每小时一次运行实施与验收](docs/13_hourly_once_implementation.md)：五个线性里程碑、新版本协议与发布门禁。
15. [14 · 每小时训练前登记](docs/14_hourly_v3_training_registration.md)：新实验的有限候选、时间折、校准与阈值规则。
16. [15 · 每小时一次运行手册](docs/15_hourly_once_runbook.md)：数据检查、训练、一次运行、看板和前向成熟评估。
17. [16 · 每小时版本交付证据](docs/16_hourly_v3_delivery_evidence.md)：五项验收、真实开发期结果、独立效果门禁。
18. [17 · 无止损三日期末收益与风险登记](docs/17_terminal_risk_v4_registration.md)：触及、期末估值、成本、完整窗口和嵌套选择。
19. [18 · v4 实测与一次预测](docs/18_terminal_risk_v4_evidence.md)：完整窗口审计、真实开发期负结果和研究预测边界。
20. [19 · 因果质量股票池拟合前登记](docs/19_quality_training_v5_registration.md)：63 日、严格 >60%、六时点成熟、有限候选及双重开发门槛。
21. [20 · v5 真实训练和评价证据](docs/20_quality_training_v5_evidence.md)：23 个试验、逐日期/股票结果、失败门槛与复现命令。
22. [21 · 质量池校准策略预登记](docs/21_quality_calibration_v51_registration.md)：仅按内层选择原始概率或 sigmoid，保持既有质量、收益和风险门槛。
23. [22 · 校准迭代与最终复现](docs/22_quality_calibration_v51_evidence.md)：实际改善、未达标条件、检查点重放与完整验收证据。

原始双路线方案为 `market_dual_track_v2`，沿用 v1 的目标/入场/标签契约；后续每小时、期末收益风险和因果质量池分别升至 v3/v4/v5，按各自冻结文档与制品解释。事件信息留作后续研究，不进入这些主实验预测输入。旧讨论的 A/B/C/D 模型代号停用，统一使用 H/A/B 模块与 H/HA/HB/HAB 实验 ID。

## 已运行的探索训练

本地使用 2026 年未复权 5m 行情和当前名单回溯股票池，完成两个月份的 H/HA/HB/HAB × LightGBM/TCN 共 16 次训练；原始制品保存在 `runs/market_dual_track_pilot_20260925/`，不进 Git。配置在 [pilot_v2.json](configs/pilot_v2.json)，完整报告在 [首轮登记](docs/08_training_launch.md)。无夜盘记录、历史可用时间为假设，也未做正式概率校准。复跑需要新的输出目录，不能覆盖既有证据：

```bash
uv venv --python 3.12 research/after_open_3d5pct/.venv
uv pip install --python research/after_open_3d5pct/.venv/bin/python -r research/after_open_3d5pct/requirements-pilot-v2.txt
research/after_open_3d5pct/.venv/bin/python -m research.after_open_3d5pct.train_pilot_v2 --config research/after_open_3d5pct/configs/pilot_v2.json --output research/after_open_3d5pct/runs/<new-run-id>
```

`<new-run-id>` 是占位符，运行时换成尚不存在的目录名。训练器只读本地行情，不调用 OpenD 或交易接口。

## 合成脚手架仍能运行什么

从仓库根目录运行；脚手架只需要 Python 3.11+ 标准库，不联网，不访问 OpenD、不读取账户：

```bash
python3 -m research.after_open_3d5pct validate-config
python3 -m unittest discover -s research/after_open_3d5pct/tests -v
python3 -m research.after_open_3d5pct smoke --output research/after_open_3d5pct/runs/smoke-001
```

输出目录必须不存在。再次运行换新目录，避免覆盖研究证据。

`smoke` 用两个虚构股票的 5 分钟数据，走通 **特征 → 延迟入场 → 标签 → 两个前向时间折 → 训练期常数基准 → 评估 → 记录**。所有输出标注 `synthetic`；分数只说明流程可执行。它不是对真实股票的训练验收。

| 路径 | 当前职责 |
|---|---|
| `configs/v1.json` / `config.py` | 默认契约与版本约束 |
| `contracts.py` / `timeaxis.py` | 标准化 bar、标签、样本，显式交易日历运算 |
| `features.py` | 4 个已实现的盘中价格/量特征，含时间与覆盖检查 |
| `labeling.py` | 5 分钟延迟入场，+5%、MFE、MAE、首次触及时间区间 |
| `splits.py` / `baseline.py` | 按日期切分、标签隔离、B0 无特征基准与评分 |
| `smoke.py` / `__main__.py` | 可复现的合成集成流程与 CLI |
| `tests/` | 因果时间、标签成熟、日历与切分边界测试 |
| `templates/experiment.md` | 每轮真实实验开始前填写的登记单 |
| `runs/`、`datasets/`、`models/` | 本地生成物，Git 忽略；不自动上传 |

真实数据适配、固定窗口 H/A/B 探索训练以及后续 v3–v5 有限训练已运行；部分版本具有内层概率校准，但**历史 PIT 股票池、真实接收时间校验、独立前向验证和可执行策略仍待完成**。不要把本 README 的“能跑”理解为这些已经完成。

## 只读研究看板与模拟操作

在仓库根目录运行以下命令，输出目录必须不存在；导出器核对冻结运行和本地 5m 源行情哈希，不联网、不训练、不接券商：

```bash
research/after_open_3d5pct/.venv/bin/python -m research.after_open_3d5pct.export_dashboard_v1 \
  --output research/after_open_3d5pct/runs/research_dashboard_policy_v1_20260925
```

现有本地服务可直接访问 [研究台](http://127.0.0.1:8768/research/after_open_3d5pct/dashboard/index.html)；先检查 `lsof -nP -iTCP:8768 -sTCP:LISTEN`，不要重复启动。页面使用 11:30/13:30/14:30 ET 的独立操作协议；12:30 仍可在原训练评分对照中查看。阈值默认 0.35，只是演示配置，原始模型输出尚未校准。若换阈值或新运行目录，以 `--threshold` 和新的 `--output` 导出，再用页面 `?run=<目录名>` 查看，不改旧产物。详细规则和剩余局限见 09–12。
