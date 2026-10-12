# SOXL / SOXS · 任意时点分析与 14 天操作回放 v2

> 当前要求已更新：用户不接受负收益月份，5、7、9 月只作回归用例，不得定向拟合；允许每天学习当时已成熟的数据。以下为历史版本记录；旧算术均值通过不再授予当前买入资格。最新每日训练研究与未达标结论见 [v4](../soxl_soxs_14d5pct_v4/README.md)。

**用户确认的回测门槛已由固定 20 日趋势路线达到：2026 年 5、7、9 月账户净收益算术均值 +11.69%。** 09:25:30 ET 为已评价的盘前分析时点，之后 09:30 入场。ML 自适应路线仍未通过。完整结果、失败实验和风险见 [REPORT.md](REPORT.md)。

分析命令接受任意带时区的时间；省略时间时查询当下。买入意图带 +5% GTC 自动卖出止盈单模板，第 14 个自然日仍持仓则强制卖出。代码建立研究结果和模拟订单，实际券商交易仍由用户手动执行。

## 入口

- [独立调试页](http://127.0.0.1:8768/.cache/soxl_soxs_14d5pct_v2/current/review.html)：任意时点的快照分析、真实全时段分钟图、两年日级 OHLC 导航、逐笔成功/失败路径、账户收益与成本压力测试。
- `analyze.py`：任意时点/当下分析；数据过期、缺失、已有持仓或不在新入场时段时不行动。
- [基础协议](protocol.json)与[盘前时点后续协议](protocol_clock_v1.json)：明确目标、时间、成本、资金与验收口径，保留全部先前失败结果。
- `baseline_clock_study.py`：复算已达数值门槛的固定路线；`study.py` 保留 ML 的月前训练/选择研究。
- `verify_run.py`：从输出与真实分钟数据独立核对订单、首次触及、到期、亏损与净值。

## 安装与运行（仓库根目录）

```bash
uv venv --python 3.12 .cache/soxl_soxs_14d5pct_v2/venv
uv pip install --python .cache/soxl_soxs_14d5pct_v2/venv/bin/python -r research/soxl_soxs_14d5pct_v2/requirements-reproduction.txt

# 已知历史盘前时点；原始输入在本机缓存中，无须网络
.cache/soxl_soxs_14d5pct_v2/venv/bin/python research/soxl_soxs_14d5pct_v2/analyze.py --at 2026-09-15T09:25:30-04:00

# 任意历史时点，明确标注为该时点的研究参考
.cache/soxl_soxs_14d5pct_v2/venv/bin/python research/soxl_soxs_14d5pct_v2/analyze.py --at 2026-09-15T08:57:20-04:00

# 当下：先补当前快照，再固定实际分析时刻；默认补数据总预算 30 秒
.cache/soxl_soxs_14d5pct_v2/venv/bin/python research/soxl_soxs_14d5pct_v2/analyze.py
```

`--data-timeout 60` 可调整当前数据预算；`--offline` 跳过网络，只检查已有捕获。历史回放使用 `bar_end+1 秒` 的可用性假设，当前查询另检查实际观察收到时间。预算到期会终止获取进程、保留完整文件并继续输出缺失/不行动；原评价数据不覆盖。历史查询的 `--at` 必须有时区，不能把旧概率或旧价格冒充当下。

已有持仓可以用 `--position-json <文件>` 明确传入 `symbol`、带时区的 `entry_at`、实际 `entry_fill_raw` 和可选 `quantity`。它会单独返回止盈单与到期退出意图：期限前保持订单，到期后 `exit_action=FORCE_SELL_AT_EXPIRY`，先取消原止盈单且仅在仍持仓时卖出；确认已止盈时不重复卖出。该输入不是自动读取真实账户，已有持仓的期限管理也不会等待网络补数据。止盈模板的目标价在实际成交后按 `ceil(fill ×1.05×100)/100` 设置，避免在下单前假定未来成交价。

**任何时间可以启动分析，不等于任何时间都已经证明可实现 +11.69%。** 已评价的是每日 09:25:30 的固定日程；其他时点用相同因果数据函数生成参考，并明确标注其收益表现尚未评价。登记的新入场时段为 04:00–15:30 ET，半日市按官方收盘提前停止；其他时段仍可分析并输出不行动。

## 数据与复算

冻结输入位于 `.cache/soxl_soxs_14d5pct_v2/data/`；SOXL、SOXS、SOXX、QQQ 的未复权 5 分钟数据均有 499 个完整常规盘交易日，覆盖 2024-10-14–2026-10-09。来源为 Local → R2 → OpenD；补齐数据另存，不覆盖共享归档、不自动上传 R2。公司行动与支付日期核对沿用 v1 的 IBKR 捕获，并用已有 Futu 拆股元数据交叉检查；原 ZIP 保持不变。

新机器须先按 [v1 导入说明](../soxl_soxs_14d5pct_v1/README.md)恢复公司行动捕获，再获取本版数据：

```bash
.cache/soxl_soxs_14d5pct_v2/venv/bin/python research/soxl_soxs_14d5pct_v2/acquire.py
.cache/soxl_soxs_14d5pct_v2/venv/bin/python research/soxl_soxs_14d5pct_v2/test_contract.py
.cache/soxl_soxs_14d5pct_v2/venv/bin/python research/soxl_soxs_14d5pct_v2/baseline_clock_study.py --output .cache/soxl_soxs_14d5pct_v2/runs/new_clock_run
.cache/soxl_soxs_14d5pct_v2/venv/bin/python research/soxl_soxs_14d5pct_v2/verify_run.py --run .cache/soxl_soxs_14d5pct_v2/runs/new_clock_run
.cache/soxl_soxs_14d5pct_v2/venv/bin/python research/soxl_soxs_14d5pct_v2/render_review.py --run .cache/soxl_soxs_14d5pct_v2/runs/new_clock_run
```

读取新批次可显式传 `--run`，不会自动替换 current。源码、模板和摘要可随 Git；原始分钟数据、逐行特征/标签、模型二进制、交易明细和内嵌行情的 HTML 留在项目现有 `.cache/`。仅克隆源码不会自动取得冻结行情。

## 本次验证

17 项契约测试通过，包括任意时点、盘前、未来不变性、分析后的入场、真实拆股、不提前获胜、缺失/未成熟保留、股息应收、半日市、实时收到时刻及单月均值口径。独立检查覆盖 956 条因果截止记录、106 条跨策略订单与事件记录；主路线为 36 笔入场、35 笔在 9 月底前完成、1 笔保留跨月状态。所有亏损保留。

当前查询实测在 30 秒预算用完、QQQ 最新行情尚未齐全时返回 `NO_ACTION`，未发出陈旧操作；60 秒复测取得四个标的最新已完成柱，数据状态为 ready，并因当时为 00:37 ET、处于新入场时段之外而不行动。当日尚未收盘，整日覆盖仍可为 partial，这不代表已完成柱无法分析。历史 09:25:30 和任意 08:57:20 分析调用均已验证。页面桌面/窄屏、失败交易、日期控制、明确平移与事件跳转可检查。摘要证据见 `evidence/`，完整记录在 `.cache/soxl_soxs_14d5pct_v2/runs/phase3_clock_20261012/`。
