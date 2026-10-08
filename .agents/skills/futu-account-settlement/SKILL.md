---
name: futu-account-settlement
description: >-
  Futu / Moomoo OpenD 账户资金、未交收清算 (T+1 / GFV)、多券商实体隔离与自选股导出的核心接口规范与工具集。
  用于排查美股账户不可见、解决现金账户未交收资金穿透与流水查询受限、处理跨时区清算及规避自选股频控。
---

# Futu / Moomoo OpenD 账户资金、清算交收与接口避坑指南 (futu-account-settlement)

本 Skill 汇总了在 Futu OpenD 网关下进行**多券商实体账户隔离**、**Moomoo US 现金账户未交收资金穿透分析（SEC T+1 与 GFV 规避）**、**时区换算（HKT vs EDT）** 以及 **自选股分组提取与频控保护** 的完整实证经验、接口规范与 CLI 工具。

---

## 一、双券商实体隔离与账户连接避坑 (Broker Entity Isolation)

### 1. 核心踩坑机理：富途证券 (HK) vs Moomoo US 实体隔离
- **问题现象**：
  若使用默认参数初始化交易上下文：
  ```python
  trd_ctx = ft.OpenSecTradeContext(filter_trdmarket=ft.TrdMarket.US, host='127.0.0.1', port=11111)
  ret, acc_df = trd_ctx.get_acc_list()
  ```
  只能查到**富途证券（香港实体 FUTUSECURITIES）**的账户，用户在 **Moomoo US（美股实体 FUTUINC）**下的现金账户（尾号 0086 / 1183）及美股持仓会**完全不可见**！

- **根本原因**：
  `OpenSecTradeContext.__init__` 的参数 `security_firm` 默认值为 `'N/A'`，此时 OpenD 会回落为香港实体 `ft.SecurityFirm.FUTUSECURITIES`。Moomoo US 独立受 SEC / FINRA 监管，必须显式传入：
  ```python
  security_firm = ft.SecurityFirm.FUTUINC
  ```

### 2. 用户真实账户映射表（已实证核准）

| 券商实体 (`security_firm`) | 内部账户 ID (`acc_id`) | 综合卡号 (`uniCardNum`) | 账户类型 | 状态 | 持仓规模与特征 |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **`ft.SecurityFirm.FUTUINC`<br>(Moomoo US 实体)** | **`283445330641239202`** | **`1007292558030086`**<br>(**尾号 0086 主账户**) | **`CASH`<br>(现金账户)** | **`ACTIVE`** | **核心美股主账户 (18+ 只持仓)**<br>CRDO, SOXS, MRVL, GOOG, INTC, AVGO 等 |
| **`ft.SecurityFirm.FUTUINC`<br>(Moomoo US 实体)** | `283445330187455846` | `1007262236326582` | `MARGIN`<br>(融资账户) | `ACTIVE` | 杠杆融资持仓 |
| **`ft.SecurityFirm.FUTUSECURITIES`<br>(富途证券香港实体)** | `281756481392257983` | `1001200108960366` | `MARGIN` | `ACTIVE` | ETF 配置账户 (含 VOO, SOXX, SMH, IHE 等) |

### 3. 神级排查技巧：OpenD 网关日志反查
若未来遇到账号对不上或资产不可见，**不要猜测**，直接在网关日志搜索卡号尾号：
```bash
grep -r "0086" ~/.com.futunn.FutuOpenD/Log/
```
日志将输出原始握手报文，精准获取 `accID`、`securityFirm` 与 `accType`：
```json
{"trdEnv":1,"accID":"283445330641239202","accType":1,"securityFirm":2,"uniCardNum":"1007292558030086","accStatus":0}
```
*(其中 `securityFirm=2` 即代表 `FUTUINC`)*

### 4. 协议加密铁律
本地连接必须显式关闭协议加密，否则握手将永久挂起：
```python
ft.SysConfig.enable_proto_encrypt(False)
```

---

## 二、Moomoo US 现金账户未交收资金穿透分析 (Unsettled Funds & T+1)

### 1. 核心痛点：`get_acc_cash_flow` 接口报错
直接调用 `trd_ctx.get_acc_cash_flow(...)` 查询流水时，OpenD 会直接返回错误：
```
code: -1, msg: "moomoo证券(美国)账户暂不支持查询现金流水"
```
Moomoo US 实体受美国清算所 (DTC) 规则约束，不向 OpenD 暴露通用现金流水查询端点。

### 2. 解决方案：三维穿透逆向重构
必须组合调用以下三个接口，逐笔重构清算账本：
1. `accinfo_query(acc_id)`：获取即时资金水位：
   - `us_cash`：账面总现金
   - `us_avl_withdrawal_cash`：可提现现金（已完全交收部分）
   - `usd_net_cash_power`：即时现金购买力
   - `frozen_cash`：冻结现金（通常以港币显示，如 HK$80.51 ≈ $10.26 USD，对应 SEC/FINRA 预扣交易规费）
2. `order_list_query(acc_id)`：获取挂单中锁定的购买力 (`SUBMITTED`, `WAITING_SUBMIT`)。
3. `history_deal_list_query(acc_id, start, end)`：拉取过去 N 天的真实成交明细（`qty`, `price`, `trd_side`, `create_time`）。

### 3. 关键换算：跨时区映射与 SEC T+1 清算日推算
- **时区转换陷阱**：
  OpenD 返回的成交时间 `create_time` 默认为 **北京时间 / 香港时间 (HKT, UTC+8)**。
  而美股交易日与 SEC 清算均以 **美东时间 (EDT, UTC-4 或 EST, UTC-5)** 为准。
  - 夏令时（EDT）：`create_time_edt = create_time_hkt - 12 hours`
  - 冬令时（EST）：`create_time_est = create_time_hkt - 13 hours`
  必须先换算为 EDT，提取出美股交易日 `us_trade_date`，否则跨午夜的交易会被错误归档到次日。
- **SEC T+1 清算日逻辑**：
  从 2024 年 5 月起，美股实行标准 **T+1** 交收：
  - 周一至周四交易：次日完成交收（Settled）。
  - 周五交易：由于周末闭市，顺延至下周一完成交收。
  - 遇到美股法定节假日需顺延一个工作日。

### 4. 现金账户风控备忘 (Good Faith Violation / GFV)
在 Moomoo US 现金账户中：
1. **可用性**：卖出股票所得的未交收资金，**可以立即用于买入新股票**（购买力即时释放），但**不可提现**。
2. **违规红线**：若使用**未交收资金**买入了股票 B，必须等待产生该资金的原卖单**完成 T+1 交收**后，才可以卖出股票 B。如果在此之前卖出股票 B，将构成一次 **善意交易违规 (Good Faith Violation, GFV)**。
3. **处罚**：12 个月内累计达到 3 次 GFV，账户将被限制为 90 天内只能使用已交收资金交易。

---

## 三、自选股提取与频控规避 (Watchlists & Rate Limiting)

### 1. Enum 名称踩坑
OpenD 的自选分组类型枚举为：
- ❌ 错误：`ft.SecurityGroupType.ALL`（报 `AttributeError`，该枚举不存在）
- ✅ 正确：`ft.UserSecurityGroupType.ALL`

### 2. 接口频控陷阱 (Rate Limiting)
- 富途 OpenD 的 `get_user_security` 存在严格频控：**每 30 秒最多允许 10 次请求**。
- 若遍历 20+ 个分组且无充足延迟，将必定触发错误：
  ```
  获取自选股分组频率太高，请求失败，每30秒最多10次。
  ```
- **避坑实践**：
  - 单次循环之间必须加入安全停顿（推荐 `time.sleep(3.2)`）。
  - 对返回失败并包含“频率太高”的结果，必须捕获并执行 30 秒休眠重试机制（Retry with Backoff）。

### 3. 分组名称动态获取原则
用户的默认分组名称在不同语言或客户端可能不同（例如中文客户端为 `"特别关注"`，海外可能为 `"Favorites"`）。
**坚决不要硬编码分组名称**，先调用 `get_user_security_group` 动态检索分组名称列表。

---

## 四、CLI 工具速查 (CLI Tooling)

### 1. 未交收资金穿透与清算明细分析工具
**脚本路径**：[`scripts/analyze_unsettled_funds.py`](file:///Users/admin/Code/stock/.agents/skills/futu-account-settlement/scripts/analyze_unsettled_funds.py)

```bash
# 默认分析 Moomoo US 现金主账户 (283445330641239202) 最近 7 天成交与 T+1 状态
python3 .agents/skills/futu-account-settlement/scripts/analyze_unsettled_funds.py

# 指定分析指定账户
python3 .agents/skills/futu-account-settlement/scripts/analyze_unsettled_funds.py 283445330187455846
```

**输出内容**：
- 账面现金、可提现金额、现金购买力、预扣冻结规费差额。
- 当前活跃挂单及占用。
- 逐个交易日重构成交列表，标注成交时刻 (EDT/HKT 双时区)、预计交收日及交收状态（✅ 已完成交收 / ⏳ 未交收）。

---

### 2. 自选股全量安全导出 CSV 工具
**脚本路径**：[`scripts/export_watchlists_csv.py`](file:///Users/admin/Code/stock/.agents/skills/futu-account-settlement/scripts/export_watchlists_csv.py)

```bash
# 1. 导出全部分组自选股 (内置 3.2s 频控保护与自动重试，安全防封)
python3 .agents/skills/futu-account-settlement/scripts/export_watchlists_csv.py --output futu_watchlists.csv

# 2. 仅快速导出指定关键分组 (如特别关注与美股，耗时极短)
python3 .agents/skills/futu-account-settlement/scripts/export_watchlists_csv.py -g 特别关注 美股 ETF --output core_watchlists.csv --delay 0.5
```

---

## 五、即拷即用 Python 代码模板 (Code Recipes)

### 模板 1：连接 Moomoo US 账户并获取现金与持仓
```python
import futu as ft

ft.SysConfig.enable_proto_encrypt(False)

trd_ctx = ft.OpenSecTradeContext(
    filter_trdmarket=ft.TrdMarket.US,
    host="127.0.0.1",
    port=11111,
    security_firm=ft.SecurityFirm.FUTUINC  # 必须显式指定美股实体
)

ACC_ID = 283445330641239202  # Moomoo US 现金主账户
ret, pos_df = trd_ctx.position_list_query(acc_id=ACC_ID)
if ret == ft.RET_OK:
    for _, r in pos_df.iterrows():
        print(f"[{r['code']}] {r['stock_name']} | 持仓: {r['qty']}股 | 市值: ${r['market_val']:.2f}")

trd_ctx.close()
```

### 模板 2：安全查询自选股（带重试与频控）
```python
import time
import futu as ft

ft.SysConfig.enable_proto_encrypt(False)
quote_ctx = ft.OpenQuoteContext(host="127.0.0.1", port=11111)

ret_g, group_df = quote_ctx.get_user_security_group(group_type=ft.UserSecurityGroupType.ALL)
if ret_g == ft.RET_OK:
    for _, g in group_df.iterrows():
        time.sleep(3.2)  # 严格遵守 30s <= 10 次限制
        ret_s, sec_df = quote_ctx.get_user_security(group_name=g['group_name'])
        if ret_s == ft.RET_OK:
            print(f"分组 [{g['group_name']}]: 包含 {len(sec_df)} 只标的")

quote_ctx.close()
```
