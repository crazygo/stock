# 开盘后 3 日 +5%：训练工程

状态：**M0 基础工程已建立；已完成一轮真实行情的探索训练，尚无独立验证通过的预测模型。** 训练登记与结果见 [08 · 首轮训练启动登记](docs/08_training_launch.md)。

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

最新模型方案为 `market_dual_track_v2`，沿用 v1 的目标/入场/标签契约；新增特征需另建 v2 schema 和训练配置。事件信息留作后续研究，本轮不采集为训练硬依赖、不进入预测输入。旧讨论的 A/B/C/D 模型代号停用，统一使用本文的 H/A/B 模块与 H/HA/HB/HAB 实验 ID。

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

真实数据适配及固定窗口 H/A/B 的探索训练已有首版；历史 PIT 股票池、完整特征与超参搜索、概率校准、独立前向验证和可执行策略仍待完成。不要把本 README 的“能跑”理解为这些已经完成。
