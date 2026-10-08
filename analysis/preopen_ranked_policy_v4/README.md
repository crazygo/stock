# 8/9 月验证、10 月观察的统一入口

入口：**http://127.0.0.1:8770/**。这是有本地数据 API 的 wireframe 调试页；需要用 HTTP 打开。

三个独立调试入口：`/hazard_linear.html`、`/relative_flow.html`、`/recovery_forest.html`。模型、数据覆盖、所有前置门槛、第一名买入和不操作均可查。点击失败买入可检查当时特征、排名与真实全日 5m K 线。

```bash
# 服务未运行时启动；已有 8768 静态服务保留。
python3 analysis/preopen_ranked_policy_v4/server.py --port 8770

# 获取不足的最新行情、计算一次并记录模拟观察。
python3 analysis/preopen_ranked_policy_v4/server.py --once

# 因果、资金与动作契约测试。
python3 -m unittest discover -s analysis/preopen_ranked_policy_v4 -p test_contract.py -v
```

“获取最新数据并计算”异步执行 Local → R2 → OpenD；页面打开期间可勾选每 5 分钟刷新。休市显示收盘参考，记录不操作。新增自选 / 尚无历史的成分在覆盖列表中说明；被冻结模型未注册的股票不会出现虚假概率。完整池 1,860 个成员中本轮有 124 个分钟历史标的，87 个标的实际补齐尾部；当前概率注册 121 个。

`research.sqlite` 为本机持久账本，Parquet / SQLite / 模型不进 Git，不自动上传 R2。`runs` 固定模型哈希；`candidates` 保存逐次排名；`decisions` 保存买入与不操作；`trades` 和 `outcomes` 分别保存模拟成交与后来成熟的结果；`equity` 保存现金与权益；`observations` / `live_fills` 属于独立 10 月观察账户。相同决策键幂等。

研究范围、固定参数和门禁见 [PROTOCOL.md](PROTOCOL.md)，当前问题见 [STATE.md](STATE.md)。10 月 1、2 日已被旧研究查看；10 月训练不使用这两日标签，新观察 10 月 5 日开始。当前三路线均未通过，`deployment.json` 的 `admitted=false`，因此不会预约买入。页面展示探索估计，不代表已验证的可买胜率。

离线流程为 acquire.py → data.py → train.py → train.py --october。原始来源 / panel / 模型 / 协议哈希都保留。已冻结的运行 ID 拒绝另一份模型覆盖；继续优化需要升版本。不要自动重跑训练以改变 10 月部署。

交易口径：每 5m 完成 bar 后估计，入场用之后下一根 5m Open 加 10bp，目标相对入场 +3%，失败日终退出；卖出再计 10bp。模拟 100,000 美元现金、单笔权益 20%、一笔持仓、T+1 结算、同股每日一次。触及 High 是成交代理；实盘价差、报价与实际延迟尚需 10 月在线验证，真实交易由用户手动执行。
