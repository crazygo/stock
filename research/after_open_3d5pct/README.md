# 开盘后 3 日 +5%：训练工程

状态：**M0 基础工程已建立；没有真实数据训练结果，没有已验证的预测模型。**

核心问题：在预筛股票池中，用户开盘后实际会查看的时刻，从延迟后可成交价格出发，未来 3 个交易日内再上涨 5% 的概率是多少？用户手动决定是否下单，程序不自动交易。

这是独立研究工程，目录为 `research/after_open_3d5pct`。旧的 Outcome Matrix、路径分段和热度研究继续作为历史研究保留，新工程不继承它们的实时预测能力结论。

## Coder 从这里开始

开始设计、编码、训练、评估前，按顺序完整阅读：

1. [AGENTS.md](AGENTS.md)：必须遵守的边界与交付要求。
2. [01 · 目标与研究思想](docs/01_mandate.md)：用户意图、目标演进与默认决策。
3. [02 · 数据和时间契约](docs/02_data_time.md)：当时可知、股票池、交易日历、价格与输入输出。
4. [03 · 样本、标签和特征](docs/03_features_labels.md)：精确计算口径和特征登记。
5. [04 · 训练和验证协议](docs/04_validation.md)：基准、purge、校准、独立测试和通过标准。
6. [05 · 实施路线和验收](docs/05_delivery.md)：下一位 coder 的具体任务与完成标准。

## 现在能运行什么

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

真实数据适配、历史股票池、真实交易日历、完整特征管线、可学习模型、概率校准和前向跟踪均在 M1–M5。不要把本 README 的“能跑”理解为这些已经完成。
