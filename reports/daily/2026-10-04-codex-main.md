# 股票量化研究日报 · 2026-10-04 · codex-main

报告日期按 Asia/Shanghai；交易日与回测时点按 America/New_York。范围为本次盘前/盘中研究、有效信号口径纠偏、五月至九月周回测、跨日目标、调试页面与云端命令交付，不覆盖其他 Agent 的工作。

工程交付已完成，模型效果目标仍未完成：三个路线分别使用 LogisticRegression、LightGBM、ExtraTrees；新训练的六个跨日组合整体达成率为 55.12%–69.56%，均不足 70%。十月分时门槛为空，命令输出研究参考，不能把概率排名当作有效买入信号。

## 一、任务层级树

```yaml
date: "2026-10-04"
agent: "codex-main"
tasks:
  - id: "20261004-C01"
    name: "统一盘前与盘中排名研究入口"
    status: "DONE"
    subtasks:
      - id: "20261004-C01-01"
        name: "每五分钟的因果特征、第一名与模拟账本"
        status: "DONE"
      - id: "20261004-C01-02"
        name: "八九月回放与十月配置隔离"
        status: "DONE"
  - id: "20261004-C02"
    name: "有效信号准确率与账户操作分离"
    status: "DONE"
    subtasks:
      - id: "20261004-C02-01"
        name: "冻结前置阈值、弃权、去重与未知标签规则"
        status: "DONE"
      - id: "20261004-C02-02"
        name: "有效信号独立入库和历史复算"
        status: "DONE"
  - id: "20261004-C03"
    name: "五月至九月当天加百分之三的周回测"
    status: "DONE"
    subtasks:
      - id: "20261004-C03-01"
        name: "补齐五月至七月模型并保留八九月冻结版本"
        status: "DONE"
      - id: "20261004-C03-02"
        name: "二十三周表、逐股基准与算法名称"
        status: "DONE"
  - id: "20261004-C04"
    name: "五日加百分之五与十日加百分之十独立训练"
    status: "DONE"
    subtasks:
      - id: "20261004-C04-01"
        name: "注册标签与十交易日隔离，完成三十份历史模型"
        status: "DONE"
      - id: "20261004-C04-02"
        name: "六组周回测、原始标签审计与概率复盘"
        status: "DONE"
  - id: "20261004-C05"
    name: "云端一次性全部模型前三参考报告"
    status: "DONE"
    subtasks:
      - id: "20261004-C05-01"
        name: "九份可移植模型、目标条件与行情时效"
        status: "DONE"
      - id: "20261004-C05-02"
        name: "启动命令、依赖与 README"
        status: "DONE"
  - id: "20261004-C06"
    name: "调试页面与交付验证"
    status: "DONE"
    subtasks:
      - id: "20261004-C06-01"
        name: "统一首页及三路线页面的全日 K 线与跨日检查"
        status: "DONE"
      - id: "20261004-C06-02"
        name: "单元、数值移植、标签、周计数与页面检查"
        status: "DONE"
  - id: "20261004-C07"
    name: "整理 main 发布内容与今日日报"
    status: "DONE"
    subtasks:
      - id: "20261004-C07-01"
        name: "合并最新 master 归档并保留未提交工作"
        status: "DONE"
      - id: "20261004-C07-02"
        name: "按日报技能归档成果、失败与后续工作"
        status: "DONE"
  - id: "20261004-C08"
    name: "达到可靠概率与特别关注五股目标"
    status: "IN_PROGRESS"
    subtasks:
      - id: "20261004-C08-01"
        name: "第一名群体单独校准实验，尚未拟合"
        status: "PAUSED"
      - id: "20261004-C08-02"
        name: "完整 ETF 成分与两年分钟历史覆盖"
        status: "PAUSED"
      - id: "20261004-C08-03"
        name: "冻结后的新数据独立观察"
        status: "PAUSED"
```

## 二、任务成果与交付

以下父任务与子任务均对应上面的唯一编号。工程通过、历史效果、未来准入分别核算。

| 任务编号 | 产出与量化证据 | 文件 |
|---|---|---|
| 20261004-C01 / 20261004-C01-01 / 20261004-C01-02 | 盘前和盘中每 5 分钟评估；已完成 bar_end 后 30 秒决策，回测采用随后下一根 5m Open 加成本。八九月模拟账户默认最高 16/26=61.54%，限量小仓探索最高 12/19=63.16%、权益 -1.53%；两者均未通过，原结果保留。 | [统一入口说明](/Users/admin/Code/stock/analysis/preopen_ranked_policy_v5/README.md)、[分时回放报告](/Users/admin/Code/stock/analysis/preopen_ranked_policy_v5/REPORT.md)、[模拟账本代码](/Users/admin/Code/stock/analysis/preopen_ranked_policy_v5/store.py) |
| 20261004-C02 / 20261004-C02-01 / 20261004-C02-02 | 发出时按前置阈值确定有效性；低置信度弃权不计分，有效但失败计 FP，未知保留。八九月恢复路线 35/47=74.47%，AAOI 11/12=91.67%；仅一条路线通过该历史合计门禁，五股目标未达成，十月两个时段阈值均未通过。 | [信号协议](/Users/admin/Code/stock/analysis/preopen_ranked_policy_v5/SIGNAL_PROTOCOL.md)、[信号复盘](/Users/admin/Code/stock/analysis/preopen_ranked_policy_v5/SIGNAL_REVIEW.md)、[信号运行代码](/Users/admin/Code/stock/analysis/preopen_ranked_policy_v5/signal_runtime.py) |
| 20261004-C03 / 20261004-C03-01 / 20261004-C03-02 | 105 个交易日、23 个 ET 周；197 个路线内有效信号全部成熟。逻辑回归 9/12，LightGBM 62/95，ExtraTrees 64/90；保留零信号周。逻辑回归只有 8 个信号日期；恢复路线 68/90 信号来自 AAOI/AXTI，供给集中。 | [周报](/Users/admin/Code/stock/analysis/preopen_ranked_policy_v5/weekly_v1/REPORT.md)、[周数据](/Users/admin/Code/stock/analysis/preopen_ranked_policy_v5/weekly_v1/weekly.csv)、[周度验证](/Users/admin/Code/stock/analysis/preopen_ranked_policy_v5/weekly_v1/verification.json) |
| 20261004-C04 / 20261004-C04-01 / 20261004-C04-02 | 两个新目标各自训练、校准和选阈值；30 份历史模型与 6 份十月只读模型完成。3262 个路线/目标内信号中 86 个未知保留；每组均有 23 周。原始标签和冻结概率逐个复算，并核对 86940 个第一名决策点。 | [跨日协议](/Users/admin/Code/stock/analysis/preopen_ranked_policy_v5/HORIZON_PROTOCOL.md)、[跨日报告](/Users/admin/Code/stock/analysis/preopen_ranked_policy_v5/horizon_v1/REPORT.md)、[周数据](/Users/admin/Code/stock/analysis/preopen_ranked_policy_v5/horizon_v1/weekly.csv)、[独立复算](/Users/admin/Code/stock/analysis/preopen_ranked_policy_v5/horizon_v1/verification.json) |
| 20261004-C05 / 20261004-C05-01 / 20261004-C05-02 | 原当天 +3% 的 3 个模型，加新目标的 6 个模型，命令各给前三，合计最多 27 个参考选项；包含概率、算法、阈值、目标价、截止与有效窗口。新模型每份验证 20092 行，覆盖盘前/盘中，移植误差小于 1e-10。十月全部门槛为空；参考排序不发有效操作信号。 | [云端命令](/Users/admin/Code/stock/analysis/preopen_ranked_policy_v5/recommend.py)、[根 README](/Users/admin/Code/stock/README.md)、[移植验证](/Users/admin/Code/stock/analysis/preopen_ranked_policy_v5/horizon_v1/native_verification.json) |
| 20261004-C06 / 20261004-C06-01 / 20261004-C06-02 | 统一首页与三条路线页面均为 wireframe；日期起止、30/60/90/180 日历天、真实两年日 K mini、全日时段、明确平移与禁鼠标缩放。新目标成功/失败/未知路径、整段常规盘阴影、CSV 和 390px 页面检查通过；48 个单元测试通过。 | [首页](/Users/admin/Code/stock/analysis/preopen_ranked_policy_v5/index.html)、[页面证据](/Users/admin/Code/stock/analysis/preopen_ranked_policy_v5/horizon_v1/ui_verification.json)、[交付清单](/Users/admin/Code/stock/analysis/preopen_ranked_policy_v5/horizon_v1/delivery_manifest.json)、[交互契约](/Users/admin/Code/stock/AGENTS.md) |
| 20261004-C07 / 20261004-C07-01 / 20261004-C07-02 | 研究提交 6720cea、6bfb9d4 已存在；远端原先没有 main，最新 master 为 894bf1b。本次 main 合并提交 0387b99 保留 master 的四份归档提交；对 152 个未跟踪文件建立发布前完整性快照，日报单独加入提交范围。远端最终交付 SHA 以 Git 核验为准。 | [本日报](/Users/admin/Code/stock/reports/daily/2026-10-04-codex-main.md)、[日报规范](/Users/admin/Code/stock/.agents/skills/daily-report/SKILL.md) |
| 20261004-C08 / 20261004-C08-01 / 20261004-C08-02 / 20261004-C08-03 | 效果未达成。下一轮三个假设已有文字登记，但没有新拟合结果；覆盖不足与未来验证保留为未竟任务。 | [研究复盘与下一轮假设](/Users/admin/Code/stock/analysis/preopen_ranked_policy_v5/horizon_v1/REVIEW.md)、[覆盖说明](/Users/admin/Code/stock/analysis/preopen_ranked_policy_v5/universe_coverage.json) |

### 三条路线与算法的关系

“路线”是要检验的思路及输入；“算法”是拟合方式；训练后每个目标、月份各有模型；概率校准器负责修正数值，信号门槛负责决定是否弃权。三条路线共用行情和目标，不能当成三份独立市场证据。

| 路线 | 算法 | 输入侧重点 |
|---|---|---|
| 剩余时间与单股基础风险 | LogisticRegression（逻辑回归） | 时间、单股波动与盘前/盘中状态 |
| 相关股分钟量价 | LightGBM | 增加固定相关股的当时分钟量价和相对强弱 |
| 回撤恢复与量价效率 | ExtraTrees | 增加价格回撤、恢复位置与量价效率；“恢复”指价格路径从回撤中恢复 |

### 新目标主结果：仅成熟有效信号计达成率

评价日期为 2026-05-01–09-30，结果行情截止 09-30。入场日算第 1 个交易日，评价入场价为延迟后的下一根 5m Open×1.001，只统计入场后的常规盘 High。盘前可以入场，盘前 High 不替代主目标；节假日与半日市用官方日历。训练/校准/选择/评价之间统一隔离 10 个交易日。

| 路线 / 算法 | 5 日 +5%：达成/成熟 | 未知 | 10 日 +10%：达成/成熟 | 未知 |
|---|---:|---:|---:|---:|
| 基础风险 / LogisticRegression | 193/283 = 68.20% | 2 | 269/488 = 55.12% | 8 |
| 相关股 / LightGBM | 594/854 = 69.56% | 33 | 292/446 = 65.47% | 17 |
| 回撤恢复 / ExtraTrees | 578/865 = 66.82% | 25 | 160/240 = 66.67% | 1 |

5 日相关股模型相对同股同时点前置基准 57.10% 高 12.45 个百分点，值得继续验证；但六组整体都未达到 70%，六组股票日等权 Brier 均未优于基准。10 日基础风险全部有效信号的平均预测概率 90.31%，成熟达成率却只有 55.12%，过度自信明显。

5 日相关股路线中，ALAB 37/41、AMD 13/17、MRVL 22/27、TER 14/16、TXG 12/14 各自达到 n≥12、70% 和基准增量≥3pp。这是看过评价结果后发现的特别关注子集，不能据此赋予未来五股或模型资格。旧当天 +3% 信号的跨日后续达成率另存，它们也不是新目标的预测概率。

### 云端使用与交付边界

```bash
python3 -m pip install -r analysis/preopen_ranked_policy_v5/requirements-cloud.txt
python3 analysis/preopen_ranked_policy_v5/recommend.py --target all --top 3
```

本地调试服务：

```bash
python3 analysis/preopen_ranked_policy_v5/server.py --port 8770
```

缺少最新行情按 Local → R2 → 可配置 OpenD 获取；云端的 127.0.0.1 不能自动访问用户 Mac 的 OpenD。过期、缺失、未注册和月份不适用必须显示，随代码的十月二日特征只能作历史参考。原始 Parquet、pickle、SQLite 不入 Git；没有真实下单，也没有自动上传 R2。

发布前在合并后的 main 上重新运行 48 个单元测试与 `--target all --top 3 --offline` 命令检查。离线检查验证命令交付，不验证明天的实时行情可达性。既有页面、原始标签与数值移植证据为今日研究交付记录，本次纯发布没有重新训练或重跑所有页面。

## 三、踩坑复盘与用户反馈

### 1. 用户要求当下可买，并覆盖盘前

**事实：** 用 09:30 Open 起算的涨幅，不能回答盘前或盘中当前价买入后的触及概率；只在盘中入场会遗漏盘前出现的机会。

**深层目的：** 信号必须在决定时可执行、可复算。未来完整盘前统计、未来成交量和当日最终达成率都会偷看结果；用户关心的是延迟后买入价对应的剩余机会。

**系统规则：** 每五分钟用当时已完成的数据，记录 bar_end、available_at、决策与评价入场。全日数据用于观察，预测特征只用截止前数据；夜盘归属、半日市和行情时效单独审计。

### 2. 用户纠正准确率分母

**事实：** 模拟账户是否买入会受现金、持仓和执行条件影响，旧账户 16/26 与新有效信号 35/47 不是同一个指标；全部概率行也不应计入有效信号。

**深层目的：** 用户要衡量模型主动推荐时是否可靠，允许信心不足时弃权，但不能在结果失败后把已经发出的信号改称无效。

**系统规则：** 前置阈值决定有效性，主指标 TP/(TP+FP)，同时列信号供给与日期；未知保留并给保守下界。信号表与账户表分别保存，成功失败都能检查实际路径。

### 3. 用户要求写清算法

**事实：** “相关股”“恢复”是路线和特征主题，不是与逻辑回归、LightGBM、ExtraTrees 同层级的概念。

**深层目的：** 可学习的研究需要知道哪个思路使用哪个拟合器，以及结果来自输入、算法、标签还是阈值变化。三个算法结果不同，不等于证明某个特征有效。

**系统规则：** 所有报表同时显示路线/算法/目标；新目标独立训练。若要归因，先登记固定算法的特征消融与固定输入的算法比较，保留失败组合。

### 4. 概率很高，真实达成率仍不足

**事实：** 10 日逻辑回归平均预测 90.31%，成熟有效信号仅 55.12%；六组 Brier 均未超过简单基准。全候选校准后再筛第一名，群体分布可能改变；这仍是待验证解释。

**深层目的：** 用户要求正置信度，指有实证支撑的达成概率，而不是模型数字看起来高。旧目标长窗口触及率高，也不能直接移植成新概率。

**系统规则：** 分开报告预测数值、实际达成、基准增量、校准分箱和机会供给；只用前置数据校准/选阈值。下一轮验证第一名群体校准，不根据已暴露结果反复调阈值追 70%。

### 5. 覆盖与研究过程也必须可审计

**事实：** 已知成员池 1860 只，而本版仅 124 只具有分钟特征覆盖、十月原模型注册 121 只；57 份 ETF 来源仅 21 份已解析。HONA 首日有 75 根零价占位，未进入任何新模型。一次训练元数据保留了附有只读云端说明的不同协议 SHA。

**深层目的：** 用户指定的股票池和两年范围不能被局部样本替代；研究登记与实际模型输入必须对得上，工程检查不能掩盖覆盖和效果缺口。

**系统规则：** 请求池、成员来源、分钟覆盖和注册池分别报告，当前成员回溯不冒充 PIT。无效 OHLC 和公司行动跨窗留未知；两个协议快照保留且验证核心条款一致，不事后改写模型 SHA。仅暂存发布清单，保留其他未提交研究。

## 四、未竟任务与后续路线

| 任务 | 状态 | 当前边界与下一步 | 预期交付 |
|---|---|---|---|
| 20261004-C08（可靠概率与特别关注五股） | IN_PROGRESS | 六组整体未通过；后验五股子集不能算独立达标。不得将当前结果发布为已验证买入建议。 | 新实验效果、供给、逐股基准与失败记录 |
| 20261004-C08-01（第一名校准） | PAUSED，待下一轮启动 | 当前仅登记研究假设，尚未拟合。先写具体协议，使用时间隔离的前置候选校准、另块选门槛，比较过度自信是否减少及供给是否保留。 | 新协议、概率分箱、Brier、23 周对照与独立页面 |
| 20261004-C08-02（全池覆盖） | PAUSED，待数据准备批次 | 36 份 ETF 来源未解决、1736 只尚无本版特征；逐源确认成员，再按 Local→R2→OpenD 补历史。获取器可用不代表全池补齐。 | 来源矩阵、缺失清单、实际历史与可评分覆盖 |
| 20261004-C08-03（新数据观察） | PAUSED，等待市场新数据 | 十月一二日已暴露；冻结后的新观察起点为十月五日。全部门槛为空时只记录参考和弃权，服务退出或电脑睡眠不得补成实时信号。 | 当时记录、标签成熟表、未来校准与供给报告 |

后续另有两个已登记但未执行的假设：用日线估计多日机会、分钟模型择时；固定算法/输入做消融。必须与第一名校准分开编号，不把多次尝试挑出的最好结果当独立验证。公司行动公告完整性与实际成交/价差仍未获验证，不由 OHLC 触及和工程测试推导保证。
