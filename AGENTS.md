# AGENTS.md · 股票量化研究与交易系统开发者指南

> 本文档面向所有在此仓库中工作的 AI Coding Agents 与量化开发者，记录项目架构、核心实证结论、数据源接口与关键排查避坑经验。

## 新训练工程入口（2026-09-25）

开盘后预测“从延迟后可成交价出发，未来 3 个交易日内触及 +5%”的工程位于
[`research/after_open_3d5pct/`](research/after_open_3d5pct/README.md)。参与该工程前必须阅读其
[`AGENTS.md`](research/after_open_3d5pct/AGENTS.md) 及 README 列出的全部五份契约文档。

**“重大核心量化实证结论”中的首小时/累计达成率，是等待未来标签成熟后计算的历史结果。当天同一时刻无法知道，不能直接作为实时特征、交易门禁或置信度乘数。**
本次仅建立工程基础；新模型尚未完成真实训练或独立验证。新工程规则不改变旧研究的原始结果。

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
