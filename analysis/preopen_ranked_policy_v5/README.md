# 盘前 + 盘中五分钟统一入口

## 新增：5 个交易日 +5% / 10 个交易日 +10%

一次给出全部目标、每个目标的三条路线各前三名：

```bash
python3 analysis/preopen_ranked_policy_v5/recommend.py --target all --top 3
```

每份报表明确写出路线与算法：① 基础风险 / **LogisticRegression**，② 相关股量价 / **LightGBM**，③ 回撤恢复 / **ExtraTrees**。三个目标各自有模型与校准器；不能把当天 +3% 概率当作跨日概率。`all` 共输出 27 个参考选项，先补齐三个目标需要的行情，后续计算复用本地；也可单独使用 `--target 5d5pct` 或 `--target 10d10pct`。默认不带 target 仍是原当天 +3%。

五月至九月的完整两个 **23 周表**、逐月/逐股/概率检查见 [`horizon_v1/REPORT.md`](horizon_v1/REPORT.md)、[`horizon_v1/weekly.csv`](horizon_v1/weekly.csv) 和 [`horizon_v1/results.json`](horizon_v1/results.json)。统一首页及三条独立页面增加了目标切换、成功/失败/待成熟筛选、跨日窗口阴影、真实 5m 路径和当时特征。

| 路线 / 算法 | 5 日 +5%：命中 / 成熟有效 | 达成率 | 待成熟/不可评分 | 10 日 +10%：命中 / 成熟有效 | 达成率 | 待成熟/不可评分 |
|---|---:|---:|---:|---:|---:|---:|
| 基础风险 / LogisticRegression | 193/283 | 68.20% | 2 | 269/488 | 55.12% | 8 |
| 相关股 / LightGBM | 594/854 | 69.56% | 33 | 292/446 | 65.47% | 17 |
| 恢复 / ExtraTrees | 578/865 | 66.82% | 25 | 160/240 | 66.67% | 1 |

共 3,262 个路线/目标内有效信号，86 个未知全部保留，六组整体都没有达到 70%。结果截止 2026-09-30；买入当日计第 1 个交易日，以决定后下一根 5m Open×1.001 评价，只看其后常规盘 High。完整窗口才计分，跨公司行动日期标为未知。周末/假日不计，半日市按官方实际收盘。窗口重叠、不同模型重复不能合并当独立样本，也未模拟有限资金跨日持仓。

三模型 × 两目标 × 五个月各自重训；训练/校准/门槛/评价之间均留 10 个交易日空档，并验证上一块所有标签在下一块开始前已成熟。训练最多200日（五月实际181日），盘前/盘中分别20日校准、20日选门槛；70%、样本数、日期和基准增量要求仍用月前数据确定。独立原始行情核对、第一名、去重、周表及 SQLite 记录见 `horizon_v1/verification.json`。

十月新目标六份冻结模型见 `horizon_v1/<target>/portable_models/`，仅研究参考，所有十月分时门槛均未通过；不改变原自动观察与 +3% 准入。每份 native 在20,092行（覆盖盘前/盘中）与原模型比较，最大差小于1e-10，这只证明移植一致。训练登记与云端补充登记分别见 [`HORIZON_PROTOCOL.md`](HORIZON_PROTOCOL.md)、[`HORIZON_CLOUD_PROTOCOL.md`](HORIZON_CLOUD_PROTOCOL.md)。原始行情、pickle 和 SQLite 均不进 Git。

复跑历史研究（需完整本地 raw、weekly_v1/panel.parquet 和训练依赖 scikit-learn/scipy）：

```bash
python3 analysis/preopen_ranked_policy_v5/horizon.py --prepare --fit --aggregate --old-signals --workers 2
python3 analysis/preopen_ranked_policy_v5/verify_horizon.py
python3 analysis/preopen_ranked_policy_v5/horizon.py --fit --october --workers 2
python3 analysis/preopen_ranked_policy_v5/export_horizon_models.py
```

云端日常推理仍只需下文 `requirements-cloud.txt`；不足数据、旧参考、实际窗口、门槛与状态会分别写明。

## 云端每天主动运行一次：每个模型各给前三名

在仓库根目录运行：

```bash
python3 analysis/preopen_ranked_policy_v5/recommend.py --top 3
```

**这就是明天主动调用的命令。** 默认只运行一次，按当前美东时间检查行情、计算全部三条模型路线，每条分别给出前三只股票的预测概率、分时置信度门槛、置信度状态、参考价、指示目标、准确的评价入场时点和获胜截止。不启动常驻服务，不自动训练或下单；前三候选报告不自动成为已发有效信号，也不进入主准确率分母。

三路线确实分别训练：基础风险是 LogisticRegression，相关股量价是 LightGBM，单股恢复是 ExtraTrees；共用数据和 +3% 标签，相关性和共用预处理意味着它们不是三份独立市场证据。

研究统一使用主干 `main`。`codex/preopen-weekly-cloud` 是历史分支，停留在旧交付版本，不再作为云端入口。先保留本地未提交工作，再更新主干；不要用 reset 丢弃扫描结果：

```bash
git fetch origin
git switch main
git pull --ff-only origin main
```

首次云端环境准备（Python 3.11 或以上）：

```bash
python3 -m venv .venv-preopen
. .venv-preopen/bin/activate
python3 -m pip install -r analysis/preopen_ranked_policy_v5/requirements-cloud.txt
python3 analysis/preopen_ranked_policy_v5/recommend.py --top 3
```

`portable_models/` 内随 Git 保存三份数据格式的十月冻结模型，合计约 1.4 MB。推理不依赖本机 pickle、scikit-learn 或未提交的 v2/v3 目录。SHA 不符立即失败。各模型已在 14,520 行上与原冻结模型对比，最大概率差异小于 1e-10；这种数值一致性仅验证移植实现，不证明预测效果。

### 最新行情接入和不足处理

命令默认按 **Local → R2 → 可访问的 OpenD** 只读补齐；缺失尾部和此前 30 个完整常规盘不足均触发获取。`config/r2_storage.json` **已经随 Git 提交**，包括连接配置和凭证，路径按代码所在仓库自动定位；克隆本仓库即可读取，不需要猜凭据或另外复制本机文件。历史提交 `6bfb9d4` 也包含此配置。环境变量 `R2_ENDPOINT`、`R2_BUCKET`、`R2_ACCESS_KEY_ID`、`R2_SECRET_ACCESS_KEY` 可显式覆盖文件；如果云端设置了旧环境变量，应检查是否覆盖了仓库配置。命令不打印凭证，也不自动上传行情。必须是明确 NONE 价格基准、明确开始和结束时间的 5m 行情，60m 或未知复权基准不会冒充模型输入。

先用只依赖 Python 标准库的探针检查配置与一次真实读取；它只请求一页、最多一个对象，不枚举整个桶，不打印 key：

```bash
python3 scripts/check_r2.py --timeout 5
python3 analysis/preopen_ranked_policy_v5/recommend.py --target all --top 3
```

探针 `configured=true` 且 `read_access=passed` 只说明配置和一次读取可用，不证明行情足够新。HTTP 403 应排查授权、环境覆盖及系统时钟；TimeoutError/URLError 应排查网络、DNS和代理；配置无法解析或字段不足明确报错，不再静默吞掉。

**2026-10-05 云端启动修复：** R2 单次请求最多5秒；整次 R2/OpenD 补数据默认30秒，使用可终止的子进程，网关或网络卡住时结束等待、保留已完成文件，再生成明确的参考/缺失报告。先下载常用 `us_5m/`，再检查训练历史目录。`--target all` 共用一次行情准备和因果特征计算，不再分别重算三次。Markdown 每个目标计算完成即输出，JSON 保持单个完整文档；stderr 显示下载、特征和算法阶段。预算只限制补数据，之后仍需读取本地数据和推理，不承诺任意机器30秒内完成全命令。

```bash
# 网络较慢时显式调整补数据总预算；不是每只股票各等60秒
python3 analysis/preopen_ranked_policy_v5/recommend.py --target all --top 3 --data-timeout 60

# 验证安装与全部九个模型，无需 R2/OpenD；结果为历史参考
python3 analysis/preopen_ranked_policy_v5/recommend.py --target all --top 3 --offline
```

未完成的下载留在本地供下次继续使用，未知和过期保持显式状态；R2 可访问不代表完整股票池已补齐。运行层修复不改变冻结模型、概率校准、阈值或有效信号资格。

启动修复证据见 [`cloud_launch_verification.json`](cloud_launch_verification.json)：55项测试通过；无本地行情、真实R2的隔离启动约32.6秒返回27个参考选项，网络不可达、配置缺失和离线也能完成。与旧提交的27个排序、概率、目标条件逐项相同。该耗时是本机检查记录，不保证云端机器的推理耗时；没有重新训练或升级效果通过状态。

云端一般无法连接这台 Mac 的 `127.0.0.1:11111`。若云端已有可访问的行情网关，配置其地址：

```bash
export PREOPEN_FUTU_HOST="your-accessible-opend-host"
export PREOPEN_FUTU_PORT="11111"
python3 analysis/preopen_ranked_policy_v5/recommend.py --top 3
```

也可用 `--data-dir /path/to/raw` 读取外部挂载的 `<SYMBOL>.parquet`。这个目录独立使用，不偷偷读取其他本机研究目录。R2 下载缓存会比较 ETag，远端更新时重新下载，原始文件均不进 Git。

2026-10-04 核查：R2 常用 `us_5m/` 为 113 个对象，抽查 AAPL 实际末根为 2026-09-25 20:00 ET；AAOI 无该目录对象，独立训练历史目录也无其对象。因此**只有 R2 的云端环境目前不能承诺最新全池行情**。不足会列出缺失 / 过期状态并尝试 OpenD；若仍拿不到，使用随代码保存的 2026-10-02 15:30 ET 因果特征参考快照提供各模型前三，明确标为旧参考，全部不构成当前有效信号。不会把九月底或周五的概率称作周一当前概率。未来的参考快照不会用于更早时点。

盘前和盘中决定窗口为官方交易日 04:05–15:30 ET，每五分钟边界 +30 秒；半日市提前。周末、夜盘、盘后、最后 30 分钟、错过下一根入场时限均是非当前参考。2026-10-05 周一首个可决策时点是 04:05:30 ET，即北京时间 16:05:30；在此之前主动调用会返回闭市 / 非决策参考。

### 输出与达标条件

```bash
# 默认中文表格，三个模型各三行
python3 analysis/preopen_ranked_policy_v5/recommend.py --top 3

# JSON + 本地参考报告审计记录（仍不产生操作信号）
python3 analysis/preopen_ranked_policy_v5/recommend.py --top 3 --format json --output .cache/preopen-report.json --record .cache/preopen-reports.sqlite

# 仅用现有文件，不连接 R2 / OpenD；缺失时明确使用参考快照
python3 analysis/preopen_ranked_policy_v5/recommend.py --top 3 --offline
```

**每个选项同一达标条件：** 特征截止 t，决定 t+30s，评价入场为 t+5 分钟的 5m Open×1.001；从该时点起至当天常规盘收盘，常规盘 High 至少触及评价价×1.03。盘前入场允许，但盘前触及不算成功。表中的目标用参考 Close×1.001×1.03 推算，尚未知的下一根 Open 到来后须重算；它不是已经确定的入场价格或实际成交承诺。

预测概率与实测信号 precision 分开。`probability` 是冻结分时校准后的估计；`threshold` 为空表示前置资格未通过；`current_probability=false` 表示不是当前时点；`current_qualified_candidate` 表示行情时效及置信度都合格的参考候选；`issued_signal=false` 表示本报告没有发出计分信号。十月三模型分时信号门槛均为空，当前参考排序不能变成合格操作，扩展回测不修改原十月配置。

## 2026 年五月至九月：周度有效信号回测

完整 **23 周表格** 见 [`weekly_v1/REPORT.md`](weekly_v1/REPORT.md)，可下载 [`weekly_v1/weekly.csv`](weekly_v1/weekly.csv)，结构化结果与全部前置门槛见 [`weekly_v1/results.json`](weekly_v1/results.json)。首页也有周表，可点击本周检查成功 / 失败的真实 K 线与当时特征；模型独立页面同样可用。

| 模型 | 命中 / 有效信号 | 达成率 | 95% Wilson | 日期数 | 同股同时点基准 |
|---|---:|---:|---:|---:|---:|
| 逻辑回归 | 9/12 | 75.00% | 46.77%–91.11% | 8 | 45.62% |
| LightGBM | 62/95 | 65.26% | 55.26%–74.08% | 16 | 43.03% |
| ExtraTrees | 64/90 | 71.11% | 61.04%–79.46% | 54 | 50.64% |

全部 197 个路线内有效信号已成熟，失败全部保留，低置信度弃权不计分。跨模型重复股票日不相加当独立样本。逻辑回归不足 10 个信号日期，相关股量价不足 70%；恢复路线通过登记的历史合计门禁，但其 90 个信号中 68 个来自 AAOI/AXTI，月份达成率从 44.44% 到 81.58% 波动。特别关注满足 n≥12、70%+、增量≥3pp 的只有 AAOI 16/20=80%，五股目标未达成；不能将历史合计通过升级为十月当前资格。

每月使用月前 240 个日期，200 日训练、20 日分时 Platt 校准、20 日门槛选择；新增五月前置训练补入 2025 年五月，六月 / 七月分别拟合，八月 / 九月原模型与门槛复用，十月冻结保持。ET 周一至周日，首尾截断；零信号为 —。三路线各月注册数分别为五月86、六月120、七/八/九月121；历史不足不能取得该月模型资格。当前成员回溯非 PIT，数据覆盖仍限既有 124 只特征 / 121 只十月注册，整个 ETF 请求池未完成。已暴露历史只称历史前向开发回测。

协议在新增训练之前登记：[`WEEKLY_PROTOCOL.md`](WEEKLY_PROTOCOL.md)。原始价格、因果特征、冻结概率、第一名、周计数和 SQLite 入库共核查 197 个信号，证据见 [`weekly_v1/verification.json`](weekly_v1/verification.json)。工程检查不能替代独立效果验证。

复算新增月份（需要本地原始 5m 档案，原始行情 / 大面板 / 训练 pickle 不随 Git）：

```bash
python3 analysis/preopen_ranked_policy_v5/weekly.py --prepare
python3 analysis/preopen_ranked_policy_v5/weekly.py --fit --workers 2
python3 analysis/preopen_ranked_policy_v5/weekly.py --aggregate
python3 analysis/preopen_ranked_policy_v5/verify_weekly.py
```

训练 / 全量调试服务还需要原研究环境的 scikit-learn、SciPy 与原冻结 pickle；云端日常调用只需要上面的 cloud requirements 和 Git 中的 portable models。历史大回放留在本地，Git 保留周度汇总、`weekly_v1/signals.json` 信号明细和验证结果。

---

## 原 8/9 月研究与常驻本地调试服务

**主评估已按用户澄清改为有效信号准确率。** 达到月前冻结置信度门槛才发出有效信号；准确率只统计这些信号的成熟达成率，弃权不计分，后来失败仍计FP。资金、持仓和盈亏不改变信号分母。新结果与原账户回放分别保存，见 `SIGNAL_REVIEW.md`。

信号结果：基础风险0个，不可评分；相关股3/7=42.86%；恢复35/47=74.47%（八月31/38、九月4/9）。AAOI11/12=91.67%，五股目标仍未完成。十月前置门槛仍未通过，弃权。

访问 **http://127.0.0.1:8770/**。这是需要本地 API 的调试应用；双击 HTML 文件不能获取行情或写入账本。

三条独立调试入口：

- `/hazard_linear.html`：剩余时间与单股基础风险。
- `/relative_flow.html`：QQQ / 固定相关组的分钟相对量价。
- `/recovery_forest.html`：单股回撤恢复与量价效率。

页面采用 `.agents/skills/html-wireframe/SKILL.md` 的灰阶 wireframe。桌面与 390px 手机宽度已检查日期快捷项、两年日 K mini、键盘导航、禁缩放、显式平移、失败路径、当时特征、排名、不操作记录及 CSV。证据见 `ui_verification.json` 与 `screenshots/`。

## 本次研究结论

评价月仅 **2026 年 8 月、9 月**，以前的数据用于训练和前置校准。下表是旧模拟账户结果，为分时校准 `ranked_policy_v5_phase_v1`；原始共享校准结果也保留，没有用新结果覆盖旧失败。页面的“回放版本”可切换另行登记的限量小仓位探索，实时排序继续使用原冻结分时版本。

| 路线 | 第一名实际模拟买入 | 命中 +3% | Precision | 连续账户收益 | 通过 |
|---|---:|---:|---:|---:|---|
| 单股基础风险 | 19 | 11 | 57.89% | -1.96% | 否 |
| 相关股分钟量价 | 0 | 0 | 无可评分买入 | 0 | 否 |
| 回撤恢复 | 26 | 16 | 61.54% | +0.53% | 否 |

**没有三条通过路线，也没有五只特别关注股票达到有样本约束的 70%。** 当前特别关注有效美股股票 32 只，分钟特征覆盖 32 只，冻结模型注册 31 只；CBRS 历史不够。特别关注另有 `HK.00100` 等非美股，保留原市场分类，不能误映射为 `US.00100`。

限量小仓位探索将整笔 20% 仓位改为按上一根 5m 成交量 1% 缩小、最低 $1,000；基础风险路线 12/19=63.16%、权益 -1.53%，恢复路线 15/26=57.69%、权益 -0.66%，相关股路线仍为零买入。全部失败，未切换十月模型。

所有前置盘前门槛均未通过，实际盘前买入为零。系统支持盘前入场，并不等于盘前低价优势已经验证。详见 `REPORT.md`。

## 五分钟计算与动作

美东 04:05–15:30 每 5 分钟，以完成 K 线结束时间 t 截止，t+30 秒作决定。半日市提前停止。入场代理为 t+5 分钟的 5m Open ×1.001，盘前允许入场。

获胜为 **从实际模拟买入价起，当天剩余常规盘 High 触及 +3%**；盘前 High 触及不替代主标签。首次触及时退出，未触及当日常规盘收盘退出，无止损；退出另计 10bp。页面在入场前显示最近完成 Close 推算的指示目标，入场后按买入价重定。

前置阶段资格、分钟成交量和当天买入限制先形成合格池，按概率降序、股票代码升序选择第一名。实时第一名再检查买卖价时间、价差与偏离；失败记录不操作，不替补。最多一笔持仓；同股每天最多一次；初始 $100,000，单笔最多权益 20%，整数股，卖出款下一官方交易日结算。

历史没有完整 bid/ask，因此历史合格池只有模型、分钟量、除权日和资金检查，实时报价门禁未能历史重放。OHLC 触及与 Open 代理不证明实际成交；现场资格和历史资格仍有这一实证差距。

## 运行与刷新

```bash
/opt/homebrew/bin/python3 analysis/preopen_ranked_policy_v5/server.py --port 8770
```

服务在 **2026-10-05–10-30** 的盘前与常规盘按 ET 五分钟边界 +30 秒自动评估、补齐数据、入库；检查持续到收盘前最后一个五分钟时点，模型预测 / 买入资格窗口截至收盘前 30 分钟，最后 30 分钟固定不新增买入，引用旧时点排序需标为非当前。收盘后 5 分钟另补齐一次，成熟预测标签和模拟结果，不新增买入。浏览器不必一直打开；进程需要保持运行，电脑睡眠 / 进程退出期间不补写实时观察。页面进入时若本地结果超过 90 秒，主动刷新；按钮随时刷新，复选框可让页面每五分钟更新显示。没有建立系统开机服务。

```bash
# 一次刷新并持久化观察
/opt/homebrew/bin/python3 analysis/preopen_ranked_policy_v5/server.py --once
```

当前三条路线都未取得 10 月资格，服务会记录不操作及探索概率，不预约买入。10 月模型、分时校准器和阈值已冻结，运行时校验模型哈希，不自动训练。10 月 1、2 日已暴露，不能列为独立验证；冻结后新交易日承担观察。

## 获取与覆盖

- 刷新当前“特别关注”、全部自选及 ETF 自选；保留发行人成分文件的原快照时间，不称历史 PIT。
- 已有分钟注册池缺尾部，按 Local → R2 → OpenD 补齐；使用 ALL session / NONE 价格基准。只使用行情上下文，不初始化交易上下文、不发送订单、不自动上传 R2。
- 124 只具有可用分钟历史，121 只达到冻结训练的历史约束。原始范围总包络 2024-08-01–2026-10-02；两年 mini 请求 2024-10-03–2026-10-02，IPO / 缺失保留。用于训练的特征面板从 2025-06-02 开始，月度训练使用前 240 个交易日。
- QQQ 已从 [Invesco 官方持仓 API](https://dng-api.invesco.com/cache/v1/accounts/en_US/shareclasses/QQQ/holdings/fund?idType=ticker&interval=monthly&productType=ETF) 获取 106 条全持仓，分类为 101 只股票 / ADR 和 5 条现金 / 期货，原始响应与 SHA 保留；该来源修正不改变冻结模型。已知成员池 1,860 只；1,736 只尚无本版特征覆盖。57 份 ETF 成分来源已有 21 份解析，36 份仍未解决；1,860 不是完整请求池的最终规模。未注册股票不制造概率，现有实时获取器也不等于全池两年历史已补齐。
- 历史 available=end+1s 是假设，实际获取另记 received_at；只读获取核对了 AAOI 2026-10-02 的提供方时间戳、开盘和收盘大成交量 K 线。

## 文件与账本

- `PROTOCOL.md`：原冻结协议；`PHASE_PROTOCOL.md`：共享校准失败后、分时实验运行前登记的补充；`CAPACITY_PROTOCOL.md`：限量小仓位运行前登记。
- `results.json` / `deployment.json`：原共享校准失败对照。
- `phase_results.json` / `phase_deployment.json`：当前分时结果和 10 月冻结模型注册；`capacity_results.json`：小仓位失败对照。
- `research.sqlite`：本地数据库，原始行情 / 训练 pickle / 大面板 / SQLite 被忽略，不进入 Git；`portable_models/` 的冻结参数随代码。
- `runs` / `candidates` / `decisions` / `trades` / `outcomes` / `equity`：历史模型、全部排序、唯一动作、入场、未来结果与权益。
- `observations` / `live_fills`：独立实时模拟账户；重复同一五分钟决定不重复预约。
- `observation_predictions` / `observation_outcomes`：当时已记录的 10 月概率及日终成熟标签；未买入仍能评分，缺失未来窗口继续待成熟。按股票日等权计算分时 Brier，预测行不计为交易。实时模拟退出要求从入场后常规盘到首次触及 / 收盘的 K 线前缀完整；缺失继续占用账户等待确认，不能凭最后一根收盘 K 判负。API 返回简要观察列表，完整特征留在数据库，不把整月大 payload 每次发送给页面。
- 连续运行 ID：`v5phase:<route>:2026-08_09`。容量对照为 `v5capacity:<route>:2026-08_09`。月度诊断独立账户另有 ID，不把月度与连续记录相加当作独立交易。
- `coverage.json` / `universe_coverage.json` / `raw_coverage_audit.json`：数据、成员及时间覆盖。
- `artifact_verification.json`：两版共 90 笔路线内买入的原始价格、因果特征、第一名、模型概率与 T+1 复算。

工程核验命令：

```bash
/opt/homebrew/bin/python3 -m unittest discover -s analysis/preopen_ranked_policy_v5 -p 'test_*.py' -v
/opt/homebrew/bin/python3 analysis/preopen_ranked_policy_v5/verify_artifacts.py
/opt/homebrew/bin/python3 analysis/preopen_ranked_policy_v5/verify_ui.py
```

28 项契约与成熟标签检查和页面交互通过只证明实现检查，**不证明模型有效或真实交易有效**。`v4` 盘中版本、`v3` 股票周期版本和旧研究保持隔离。

有效信号新增工程检查：`test_signals.py`；原始价格、冻结概率和第一名独立复算：`verify_signals.py`。新表 `signal_events` / `signal_labels` 在信号和标签成熟时分别写入，`signal_ticks` 的低置信度弃权不计主准确率。
