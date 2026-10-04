# AGENTS.md · 股票量化研究与交易系统开发者指南

> 本文档面向所有在此仓库中工作的 AI Coding Agents 与量化开发者，记录项目架构、核心实证结论、数据源接口与关键排查避坑经验。

## 新训练工程入口（2026-09-25）

开盘后预测“从延迟后可成交价出发，未来 3 个交易日内触及 +5%”的工程位于
[`research/after_open_3d5pct/`](research/after_open_3d5pct/README.md)。参与该工程前必须阅读其
[`AGENTS.md`](research/after_open_3d5pct/AGENTS.md) 及 README 列出的全部必读文档（01–08，含行情双路线训练方案与首轮训练记录）。

**“重大核心量化实证结论”中的首小时/累计达成率，是等待未来标签成熟后计算的历史结果。当天同一时刻无法知道，不能直接作为实时特征、交易门禁或置信度乘数。**
本工程已完成首轮真实行情探索训练，尚未完成独立验证；结果见 [`08 · 首轮训练启动登记`](research/after_open_3d5pct/docs/08_training_launch.md)。新工程规则不改变旧研究的原始结果。
只读研究看板与三时点模拟操作回放见 [`09 · 看板产品与指标验收`](research/after_open_3d5pct/docs/09_dashboard_product_metrics.md)及工程 README；操作回放另升版本，不接真实下单。
后续每小时一次运行的 `hourly_once_v3` 协议与五项验收见 [`13 · 每小时一次运行实施与验收`](research/after_open_3d5pct/docs/13_hourly_once_implementation.md)；其授权与时点配置优先于旧训练协议的固定采样时点，真实交易仍由用户手动执行。
每小时版本的开发期训练、校准退化及未通过的独立效果门禁见 [`16 · 交付证据`](research/after_open_3d5pct/docs/16_hourly_v3_delivery_evidence.md)。

---

## 一、富途 / Moomoo OpenD 接口与多账户架构避坑指南 (必读)

### 1. 核心踩坑机理：富途证券 vs Moomoo US 双券商实体隔离

- **现象**：
  直接使用默认参数初始化交易上下文时：
  ```python
  trd_ctx = ft.OpenSecTradeContext(filter_trdmarket=ft.TrdMarket.US, host='127.0.0.1', port=11111)
  ret, acc_df = trd_ctx.get_acc_list()
  ```
  只能查到**富途证券（香港实体 FUTUSECURITIES）**的账户，用户在 **Moomoo US（美股实体 FUTUINC）**下的现金账户（尾号 0086）及 18+ 只美股持仓会**完全不可见**！

- **根本原因**：
  `OpenSecTradeContext.__init__` 包含隐藏参数 `security_firm`：
  ```python
  def __init__(self, filter_trdmarket='HK', host='127.0.0.1', port=11111, is_encrypt=None, security_firm='N/A', ai_type=0):
  ```
  - 当 `security_firm` 为默认值 `'N/A'` 时，OpenD 会默认回落为 `FUTUSECURITIES`（富途证券）。
  - Moomoo US 独立受 SEC / FINRA 监管，必须显式传入 `security_firm = ft.SecurityFirm.FUTUINC` 才能解锁其名下账户。

---

### 2. 用户真实账户映射表（已实证核准）

| 券商实体 (`security_firm`) | 内部账户 ID (`acc_id`) | 综合卡号 (`uniCardNum`) | 账户类型 | 状态 | 持仓规模与特征 |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **`ft.SecurityFirm.FUTUINC`<br>(Moomoo US 实体)** | **`283445330641239202`** | **`1007292558030086`**<br>(**尾号 0086 主账户**) | **`CASH`<br>(现金账户)** | **`ACTIVE`** | **核心美股主账户 (18+ 只持仓)**<br>包含 CRDO, SOXS, MRVL, GOOG, INTC, AVGO, AMAT, HLTH, AIPO, CIEN, VRT, COHR 等 |
| **`ft.SecurityFirm.FUTUINC`<br>(Moomoo US 实体)** | `283445330187455846` | `1007262236326582` | `MARGIN`<br>(融资账户) | `ACTIVE` | 杠杆融资持仓 (如 LITX) |
| **`ft.SecurityFirm.FUTUSECURITIES`<br>(富途证券实体)** | `281756481392257983` | `1001200108960366` | `MARGIN` | `ACTIVE` | ETF 配置账户 (含 VOO, SOXX, SMH, IHE 等 4 只) |

---

### 3. 当找不到目标账户时的“神级排查命令”

如果未来遇到类似“账户号对不上”或“某些账户看不到”的情况，**不要盲目猜测**，直接执行以下命令检索 OpenD 网关登录时的全量握手日志：

```bash
# 1. 在本地 OpenD Gateway 日志中搜索卡号尾号 (例如 0086)
grep -r "0086" ~/.com.futunn.FutuOpenD/Log/

# 2. 从日志中精准查看该账号对应的 accID、securityFirm 与 accType
# 日志将输出类似如下的原生 JSON 报文：
# {"trdEnv":1,"accID":"283445330641239202","accType":1,"securityFirm":2,"uniCardNum":"1007292558030086","accStatus":0}
# 其中 securityFirm=2 即代表 FUTUINC (Moomoo US)
```

---

### 4. 即拷即用的核心 Python 脚本模板

#### ① 查询 Moomoo US 现金账户（尾号 0086）全部持仓
```python
import futu as ft

ft.SysConfig.enable_proto_encrypt(False)

# 必须指定 security_firm=ft.SecurityFirm.FUTUINC
trd_ctx = ft.OpenSecTradeContext(
    filter_trdmarket=ft.TrdMarket.US,
    host="127.0.0.1",
    port=11111,
    security_firm=ft.SecurityFirm.FUTUINC
)

TARGET_ACC_ID = 283445330641239202  # Moomoo US 现金账户 (UniCard 尾号 0086)
ret, pos_df = trd_ctx.position_list_query(acc_id=TARGET_ACC_ID)

if ret == ft.RET_OK:
    for _, row in pos_df.iterrows():
        print(f"[{row['code']}] {row['stock_name']} | 股数: {row['qty']} | 成本: ${row['cost_price']:.2f} | 盈亏: {row['pl_ratio']:+.2f}%")
else:
    print("查询失败:", pos_df)

trd_ctx.close()
```

#### ② 读取牛牛“特别关注 (Favorites)”自选股
```python
import futu as ft

ft.SysConfig.enable_proto_encrypt(False)
quote_ctx = ft.OpenQuoteContext(host="127.0.0.1", port=11111)

# 注意：富途自选股接口有频控（30秒最多10次），勿频繁并发请求
ret, fav_df = quote_ctx.get_user_security(group_name="Favorites")
if ret == ft.RET_OK:
    target_codes = [c for c in fav_df["code"] if c.startswith("US.")]
    print(f"特别关注美股池 ({len(target_codes)} 只):", target_codes)

quote_ctx.close()
```

#### ③ 免配额拉取最近 30 天 60 分钟 K 线
```python
# 30 天内的 60m K 线不消耗任何历史配额，可随时全量刷新
ret, df, _ = quote_ctx.request_history_kline(
    "US.AMD",
    start="2026-09-18",
    end="2026-09-24",
    ktype=ft.KLType.K_60M,
    autype=ft.AuType.QFQ,
    max_count=200
)
```

---

## 二、行情数据存储架构与多 Client 同步规范 (Parquet + Cloudflare R2)

> ⚠️ **所有 Coder / Agent 必读原则**：时序行情数据（Parquet）**坚决不进 Git 仓库**（已在 `.gitignore` 中声明），统一采用 **Cloudflare R2 对象存储** 作为远程集中归档介质。

### 1. 数据存放路径与格式规范
- **物理路径**：[`market_data/us_60m/<SYMBOL>/2026.parquet`](file:///Users/admin/Code/stock/market_data/us_60m/)
- **存储格式**：严格使用 **Apache Parquet (ZSTD 压缩，level 7)**，包含强类型字段：
  `['time_key', 'open', 'high', 'low', 'close', 'volume', 'turnover', 'pe_ratio', 'turnover_rate', 'change_rate', 'last_close']`
- **时段完整性**：必须包含 **全天 24 小时全部时段**（夜盘 20:00~04:00、盘前 04:00~09:30、常规盘 09:30~16:00、盘后 16:00~20:00）。

### 2. 初始化同步（新环境 / 新 Client 首次运行必读）
新开发者或新环境克隆本仓库后，本地默认没有 `.parquet` 行情文件，**第一步必须先执行全量初始化同步**：
```bash
# 从 Cloudflare R2 一键拉齐当前全量 60m Parquet 数据（多线程极速下载）
python3 scripts/r2_sync.py pull
```
- 凭证已配置于 [`config/r2_storage.json`](file:///Users/admin/Code/stock/config/r2_storage.json)，无需额外安装 `boto3` 或 `rclone`，纯 Python 标准库秒级同步。

### 3. 日常比对与更新缺失（如何检查并补齐）
#### ① 极速比对（Diff）
```bash
# 检查本地与云端 R2 的数据差异（基于文件大小与时间戳，2秒内出结果）
python3 scripts/r2_sync.py diff
```

#### ② 缺失数据补齐流程（三级分层获取原则）
当策略或回测需要某只个股的数据时，**绝对不要直接盲目调用 Futu OpenD**，必须使用统一下载器执行三级分层策略：
```bash
# 自动走：Local Hit -> R2 Hit -> Futu OpenD 兜底
python3 scripts/fetch_market_data.py --symbols AAPL NVDA CRDO
```
* **逻辑**：本地有则直接返回；本地缺失则优先从 R2 下载；R2 亦缺失才回退到 Futu OpenD 拉取。

### 4. 数据推送规范（手动执行，绝不自动推送）
- **核心规约**：为防止本地偶发网络中断或不完整抓取污染云端，从 Futu OpenD 抓取到的新数据**默认仅保存在本地**，**绝对不会自动推送到 R2**。
- **手动推送命令**：本地检查数据完整后，手动执行推送：
```bash
# 1. 预演检查（查看将要上传的文件）
python3 scripts/r2_sync.py push --dry-run

# 2. 全量推送本地新增/更新的 Parquet 文件到 R2
python3 scripts/r2_sync.py push

# 3. 仅推送特定标的
python3 scripts/r2_sync.py push --symbols AAPL NVDA
```

### 5. 批量成分股拉取规范（以 QQQ / 纳斯达克 100 为例）
针对指数全量成分股拉取，禁止使用高并发冲击 Futu 频控，使用专用稳健抓取脚本：
```bash
# 1. 先进行本地缓存断点扫描（不发请求）
python3 scripts/fetch_qqq_hourly_parquet.py --dry-run

# 2. 匀速推进拉取（内置 1.2s 首页停顿 + 0.25s 翻页停顿，安全防封）
python3 scripts/fetch_qqq_hourly_parquet.py
```

---

## 三、项目核心模块与目录导航

```
stock/
├── AGENTS.md                                           # 本开发者指南
├── config/r2_storage.json                              # Cloudflare R2 存储凭证与配置
├── scripts/
│   ├── r2_client.py                                    # 原生 S3 SigV4 客户端 (零外部依赖)
│   ├── r2_sync.py                                      # R2 多线程 diff / push / pull 同步工具
│   ├── fetch_market_data.py                            # 三级分层获取工具 (Local -> R2 -> Futu)
│   └── fetch_qqq_hourly_parquet.py                     # QQQ 成分股 60m 全量拉取脚本
├── market_data/us_60m/                                 # 美股 60m K 线 Parquet 归档 (<SYMBOL>/2026.parquet)
├── analysis/
│   ├── clustering_methods/
│   │   ├── method_1_hourly_by_week/                   # 聚类方法 1：周度 × 小时交叉热力图
│   │   │   ├── index.html                             # 交互页面 (端口 8768)
│   │   │   ├── generate_method1_hourly.py             # 生成脚本
│   │   │   └── STRATEGY_NOTES.md                      # 策略笔记
│   │   └── method_2_hourly_by_day/                    # 聚类方法 2：单月日度 × 小时微观热力图
│   │       ├── index.html                             # 交互页面 (默认展示 2026-09 最新月)
│   │       ├── data.json                              # 预计算全量微观矩阵数据
│   │       ├── generate_method2_daily.py              # 生成脚本
│   │       └── STRATEGY_NOTES.md                      # 首小时锚定效应实证与量化风控规则
│   └── today_intraday_watch/                          # 盘中实时监控看板 (0086 持仓 + 特别关注)
│       ├── index.html                                 # 实时看板页面
│       ├── today_data.json                            # 毫秒级盘中快照与达成率数据
│       ├── generate_today_monitor.py                  # 盘中实时同步生成脚本
│       └── STRATEGY_NOTES.md                          # 字段口径与盘中实操研判指引
```

---

## 四、重大核心量化实证结论（策略核心资产）

### 1. 开盘首小时锚定效应 (Opening Bell Gate)
- **实证样本**：全量 115 只股票、2026-04 至 2026-09 共 118 个完整闭环交易日。
- **核心规律**：**10:30（开盘首小时）群体达成率与当天后续时段（11:30~16:00）胜率均值相关系数高达 $r = 0.788$**。
  - 若 10:30 达成率 $< 30\%$：当日全天胜率仅 **25.3%**（系统性逆风日，坚决不追涨，一票否决）；
  - 若 10:30 达成率 $\ge 50\%$：当日全天胜率高达 **57.5%**（系统性起爆日，顺势买入）；
  - **日内分化极差高达 +32.2%**！

### 2. 盘中动态累积均值乘数 (Cumulative Momentum Multiplier)
- 随着交易进行，采用“开盘到时点 $t$ 的平均达成率”对剩余时段胜率的预测精度单调陡增：
  - **11:30 上午盘结束时（累计 10:30+11:30 均值）**：预测下午盘（12:30~16:00）胜率相关系数 **$r = 0.886$**；
  - **13:30 午后启动时（累计 10:30~13:30 均值）**：预测尾盘（14:30~16:00）胜率相关系数高达 **$r = 0.941$**！
- **适用边界**：上述相关性基于事后成熟的达成率，不能直接用作盘中“环境置信度乘数”；若使用当时可知数据另行预测热度，必须进行独立的时间样本外验证。

---

## 五、本地开发与服务运行约定

1. **Python 运行环境**：
   - 系统 Python 位于 `/opt/homebrew/Cellar/python@3.14/3.14.7/Frameworks/Python.framework/Versions/3.14/bin/python3`。
   - 依赖库：`futu-api` (10.11), `pandas` (3.0), `numpy`。
2. **本地 Web 服务**：
   - 静态 Web 服务器监听在 `127.0.0.1:8768`。
   - 不要重复启动多个端口占用进程，保持 PID 78233 正常常驻。
3. **Futu OpenD 本地守护**：
   - 运行在 PID 9353，监听端口 `127.0.0.1:11111`。
   - 连接时务必执行 `ft.SysConfig.enable_proto_encrypt(False)`，否则握手会一直阻塞。

## 六、量化研究调试页面交互契约（2026-10-03 用户要求）

适用于新增和重做的可回测调试页面；使用 `.agents/skills/html-wireframe/SKILL.md` 的 wireframe 风格：灰阶、系统字体、普通边框、原生表单，不做品牌视觉包装。

- 必须提供起点、终点日期选择，以及 30 / 60 / 90 / 180 天快捷选项；明确快捷项为日历天还是交易日。
- K 线区域禁用鼠标滚轮、触控板、双指和拖拽缩放，避免滚动误操作。日期输入、明确按钮和 mini 导航条负责调整观察范围。
- 必须提供最近两年固定范围的 mini 快速导航条，预览真实日级 OHLC，显示当前选择区间；支持点击跳转和键盘操作，缺失行情不得补成真实 K 线。
- 默认展示全日行情；明确标记常规盘、盘前、盘后、夜盘。美东日期与时区为准，按官方日历处理假日和半日市；非交易日空白不得占横向宽度，用粗日期分割线提示跨日，真实周日晚夜盘归属周一并保留。
- K 线不得挤成亚像素：默认至少 5px 间距，日期区间内全部行情可通过明确平移按钮 / 滑条查看；提供分钟粒度和可见范围说明，禁止以隐藏数据实现宽度要求。
- 将算法预测会触及目标的整个常规盘区间画成阴影，标明目标价、首次触及、未达标状态；点击失败日能检查真实分钟量价路径及该次预测当时的特征。成功和失败不得只显示信号点。
- 单股 / 固定相关股模型允许按 30 日历天周期评价；模型与阈值只能用该周期开始前的训练和校准数据确定。分钟滑窗、成交量、行业环境与缺失状态应有可复算定义；同股每天一次信号，不能用相邻分钟重复信号扩大样本数。
- 每个研究思路有独立调试页面；展示公式、预测截止、价格参考、未来标签、逐股样本数、命中/失败、基准、时间切分和数据覆盖。修改参数必须标为探索，不得沿用冻结回测的通过状态。
- “准确度”须明确分母；预测上涨信号的主指标为 precision=TP/(TP+FP)，同时报告信号数量、日期覆盖、召回、同股基准与增量。不能用大量“不涨”负例提高总体 accuracy，也不能用高波动基础触及率替代预测能力。
- 特别关注必须实时查询分组名称，当前为“特别关注”；`Favorites` 别名不可假定有效。自选 ETF 自身和其成分股分别建表，保留成员快照时间、来源、缺失与非美股/现金/衍生品分类。当前成员回溯不冒充历史 PIT 成分。
- 两年请求范围、实际行情范围与可评分覆盖分别展示；旧数据已经暴露时只能称历史前向回测，独立验证需新数据。不足五只特别关注股票达到有样本约束的 70% 时，明确未达标。

盘前预测当天涨幅的新研究入口为 `analysis/preopen_intraday_v2/`，独立于旧 `research/after_open_3d5pct/models/premarket_tail_v1/` 的三日 +5% 标签；不得覆盖旧实验。

单股 / 相关股 × 30 天周期补充入口为 `analysis/preopen_stock_cycle_v3/`，冻结协议与全部未通过周期一起保留。70% 为信号 precision；必须展示样本约束、同股基准与区间不确定性，不把后验发现的股票周期当作未来保证。

## 七、8 / 9 月排名买入与 10 月观察（2026-10-04）

新统一入口位于 `analysis/preopen_ranked_policy_v5/`，运行命令、实证结果与数据契约见其 README / REPORT / PROTOCOL / PHASE_PROTOCOL。当前版本为 `ranked_policy_v5_phase_v1`；盘中版本 v4 保留作对照。评价周期限定为 **2026 年 8 月、9 月**；允许此前历史用于训练和前置校准，10 月标签不得选模或调门槛。10 月 1、2 日已被旧研究查看，必须标为已暴露；新观察从本次冻结后首个交易日 10 月 5 日开始。

- “当下可购买胜率”必须说明预测截止、入场延迟、价格参考和剩余获胜窗口；旧 09:30 Open 起涨幅不得直接用于当前买入。用户要求**盘前和盘中均每 5 分钟**估计，检查范围 04:05–15:55 ET；本版模型预测 / 买入资格为 04:05–15:30，最后 30 分钟固定不新增买入、不能把旧时点概率标为当下（半日市提前停止）；决定为已完成 bar_end+30 秒，回测从之后下一根 5m Open 加入场成本出发，当天剩余常规盘触及 +3%。入场可以在盘前，盘前 High 触及不能替代常规盘主标签；同日本日盘前特征必须按当前截止重算，不能用完整盘前统计。
- 回测、最新推荐和 10 月模拟观察用同一 SQLite 契约，预测、排名、唯一动作、实际模拟入场和未来结果分别记录。模型与上一根分钟量先定义合格池，每次只买其中概率第一名；实时第一名另查报价时效、价差和偏离，失败不替补。无合格候选、已有持仓、现金不足或报价过期均记录不操作，不用第二名的事后获胜替代第一名。重复同一决定不得重复预约。
- 同股每日最多买一次；成交量 / 资金占用 / 成本 / 退出和结算明确建模。分钟信号不可充当独立买入样本。旧账户回放 precision 分母为全部实际模拟买入，未知标签仍保留；最新主信号口径见第八节；另报机会供给、每股样本、同股同时点基准、权益回撤与置信区间。
- 缺少已注册股票最新行情主动走 Local → R2 → OpenD；原始数据不入 Git，不自动上传 R2。在线实际 received_at 与历史 available_at 假设分开。新增当前自选不自动获得冻结模型资格；覆盖不足单列并进入下一版数据准备，不制造概率。必须区分发行人成分来源覆盖、已知成员池、分钟历史覆盖和冻结注册覆盖；当前成员回溯不是历史 PIT。
- 10 月只允许使用 8/9 月完整“第一名买入”策略结果确定部署资格，并冻结模型、门槛、成员和动作协议。无人通过时统一入口仍可检查探索排序，但不得标为合格买入。
- 统一首页和每条路线独立调试页延续第六节全部交互要求。新 API 服务用独立 loopback 端口 8770，保留已有静态服务 8768。服务运行期间在观察期按五分钟边界 +30 秒自动评估；电脑睡眠 / 服务退出不冒充实时记录。真实券商交易仍由用户手动执行，代码只建模拟账本。
- v5 默认分时三条路线未通过；该版最高为 16/26=61.54%，特别关注五股 70% 目标未达成，实际盘前买入为零。10 月没有获准模型。工程检查 / OHLC 价格触及不能提升为模型有效 / 实际成交有效；发现入场根量参与率超过约束时保留原回放并说明，不事后删除交易提高指标。
- 10 月当时记录的预测即使未买入，也在常规盘结果完整后单独成熟入库；预测标签表不制造买入。待成熟 / 缺失不能删去，按 stock-day 等权报告概率 Brier，不能把每五分钟预测行数当独立交易样本。收盘后一次补齐只负责结果，不作为新的买入时点。

- 限量小仓位探索见 v5 的 `CAPACITY_PROTOCOL.md` / `capacity_results.json`，提前登记后运行，最高 12/19=63.16% 且权益 -1.53%，盘前零买入；不改十月默认。调试页版本切换不得切换实时冻结模型；月度、连续、不同路线与仓位版本的重复交易不能累加当独立样本。
- 实时模拟持仓判定首次触及或日终未触及，必须检查从入场后常规盘起到该事件的完整分钟前缀；缺失保持待确认和资金占用，不能仅因存在收盘 K 线就判定失败。

## 八、有效信号准确率（2026-10-04 用户澄清，优先于第七节主指标）

用户要求评估有足够置信度、当时发出的有效操作信号，不是实际买入或账户盈亏。新协议与结果为 v5 的 `SIGNAL_PROTOCOL.md` / `SIGNAL_REVIEW.md` / `signals_results.json`。

- 有效性必须在发出时确定：概率达到月前确定的分时门槛才允许发出；信心不足则弃权。每五分钟最多发出一个置信度第一名有效信号，同股每天一次，盘前与盘中共用去重。未发出的候选不充当已发信号。
- 主达成率为成熟有效信号 TP/(TP+FP)。低置信度弃权不进准确率；当时有效但后来失败计入FP，不能事后改称无效。未成熟或行情缺失的已发有效信号保留、单列待确认，另给TP/全部有效信号的保守下界；空分母显示不可评分。
- 现金、持仓、成交量 / 股数、报价、收益、是否真的买入不改变信号分母。信号评估与模拟账户回放分别入库和展示；旧账户结果不得改名为新信号准确率。
- 模型 / 概率校准器沿用冻结版本。门槛只用评价月前的20日选择块确定，保留全部尝试、次数、日期、覆盖率和逐股基准；修改口径另升版本，不在未来标签出现后挑选有效信号。
- 新表 signal_runs / signal_ticks / signal_events / signal_labels；有效性表先写、未来结果成熟后写标签。十月配置独立冻结，旧模拟账户配置不覆盖。十月门槛未通过则弃权，参考概率排序不称作有效信号。
- 当前新历史结果：恢复模型35/47=74.47%，八月31/38、九月4/9；AAOI11/12=91.67%。仅一条信号路线达到该历史门禁，五只特别关注目标未达成；十月两时段门槛均未通过，不能因为两月平均高于70%就忽略最新校准退化。


## 九、五月至九月周表与云端一次性前三（2026-10-04）

用户将历史评价扩展至 2026-05-01–09-30，新增协议为 v5 `WEEKLY_PROTOCOL.md` / `weekly_v1/`；原 8/9 月、模拟账户与十月冻结配置保留。新增前三报告命令见 v5 README 与根 README。

- 三条路线分别是 LogisticRegression、LightGBM、ExtraTrees；分别拟合与分时校准，共用数据 / 目标，不能称三份独立市场证据。
- 新增五月 / 六月 / 七月按月前 240 日的 200/20/20 切分训练、分时校准、有效信号门槛选择；八 / 九月直接复用冻结模型与阈值。周度为 ET 周一至周日，首尾截断，保留零信号周，跨月周使用各自月模型。以有效信号 TP/(TP+FP) 统计，不能将低置信度行、前三候选或跨模型重复当独立信号。
- 周度合计：基础风险9/12=75%（仅8个日期）；相关股62/95=65.26%；恢复64/90=71.11%（54个日期，AAOI/AXTI占68/90）。特别关注只有AAOI16/20满足n≥12、70%与增量3pp，五股目标未达成。197个路线内信号的原始标签、因果特征、冻结概率、第一名、周计数和入库已复算；只称历史前向开发回测。
- 云端 `recommend.py --top 3` 一次运行，全部模型各给前三选项、概率、分时门槛、行情截止、评价入场与剩余常规盘+3%条件。参考报告不自动发信号、不计入主准确率、不下单、不重新训练。支持JSON与独立SQLite参考报告日志。
- `portable_models/` 的数据格式冻结模型随Git，SHA必须核验；不依赖本地pickle或未提交的v2/v3代码，数值移植验证不是效果验证。模型仅用于登记的十月范围，不能把旧月模型当作后续月获准模型。
- 缺行情主动只读Local→R2→可配置OpenD；云端通常不能访问用户Mac的127.0.0.1。不足和过期必须显示，不自动上传R2。随代码的10月2日因果特征参考仅在其截止已过去时允许使用，不能冒充当下；R2旧行情也不能标当前。前三报告不得改变既有十月准入，十月分时阈值为空继续弃权。
