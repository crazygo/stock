# 五日 +5% 免费数据研究 v6

独立实验，保留v5模型、结果与十月冻结配置。R00只称已暴露历史开发回测，没有通过未来独立验收。

研究协议见 PROTOCOL.md；当前结果 REPORT.md / results.json；持续工作状态 PROGRESS.md。三思路调试页面：opportunity.html、sector.html、events.html，统一首页 index.html。真实行情通过本地只读API读取，原始分钟数据不嵌入Git。

```bash
python3 analysis/preopen_5d5pct_v6/free_data.py --r2 --sec --futu
python3 analysis/preopen_5d5pct_v6/acquire_history.py --limit 120
python3 analysis/preopen_5d5pct_v6/prepare.py
python3 analysis/preopen_5d5pct_v6/research.py --fit --workers 2
python3 analysis/preopen_5d5pct_v6/restore_reporting_prices.py
python3 analysis/preopen_5d5pct_v6/verify.py --round R00
python3 analysis/preopen_5d5pct_v6/calibration_audit.py --round R00
python3 -m unittest discover -s analysis/preopen_5d5pct_v6 -p 'test_*.py' -v
python3 analysis/preopen_5d5pct_v6/server.py --port 8771
python3 analysis/preopen_5d5pct_v6/recommend.py --top 3
```

数据准备必须完成、coverage SHA核对后才能训练。完整重算需先将本目录旧cache/runs和models归档至另一个本地实验快照，不能在已有月度元数据下改变面板或重跑覆盖；后续研究另升R01。历史下载可断点恢复，已有分钟注册池先Local，缺失查R2，再OpenD；只保存本地，不上传R2。`.pkl`、Parquet、SQLite、日志与原始响应在忽略目录。

R01真正两阶段实验：先读R01/PROTOCOL.md，然后运行staged_prepare.py与staged.py --fit --workers 2。上游分数只用每个20日块之前成熟的日线锚点标签，所有下游arms使用同一真实OOF覆盖样本。R00与R01结果分别保存。

统一前三命令以最新可用完整交易日截止生成真实因果行情前缀，输出各模型概率估计、目标、五日截止和弃权原因。闭市参考不是当前信号；旧月研究模型未取得当前冻结资格，effective_threshold为空。`--refresh`重算最新特征，`--format json`返回结构化报告，原十月配置保持。

R00固定LightGBM比较四种行情输入，再固定基础输入比较LR/ExtraTrees。每个模型分别比较候选校准与额外第一名校准，阈值只从月前选择块取得，无合格门槛则弃权。周、月、逐股、股票日Brier、第一名群体、有效信号和非重叠敏感性分开。当前ETF成员回溯不是历史PIT。

免费SEC接口说明：[EDGAR APIs](https://www.sec.gov/search-filings/edgar-application-programming-interfaces)。公司股票索引403，但官方公司申报API正常，已用现有官方CIK及返回ticker校验获取事件；Futu财报查询历史记录不提供历史共识修订版本，所以R00事件路线明确数据受阻，没有制造概率。免费事件审计、公布接收与历史可知性是不同证据。

12小时窗口截止2026-10-05 12:59:58北京时间。未来独立效果需实际冻结后连续至少12周及全部样本门禁，不在本次窗口内宣称完成。

R02：先读R02/PROTOCOL.md，再运行event_prepare.py、events_research.py --fit、verify.py --round R02、calibration_audit.py --round R02。公开申报确认不是共识超预期。R01复算/校准对应--round R01。统一前三默认--round all；所有有效门槛继续为空。

R04季度实际同比对照：facts_prepare.py、facts_research.py --fit、verify.py --round R04、calibration_audit.py --round R04。原文抽查见SEC_FACTS_SAMPLE.md；没有历史共识，不称超预期。favorite_report.py补充当前全部33只特别关注美股股票，含零信号/未注册。

portable_models/仅含最新月原生数据模型，SHA核验；export_models.py生成并对16模型及日级上游分别复算。统一命令优先原生加载，已用禁止pickle.loads的运行检验。数值移植不证明效果；原始行情/事件缓存仍需本地或数据准备，不附带伪造的当前行情。

R03完整历史/成熟状态：先读R03/PROTOCOL.md。backfill_history.py --wait逐股补过去行情，continue_history.py在完成的符号上准备特征，随后顺序运行history_prepare.py、history_research.py --fit、校准、特别关注、独立复算及原生导出。deadline前无法完成的历史明确保留部分覆盖，不把新取得股票自动授予模型资格。原始价格逐项保持，不覆盖旧raw。

旧+3%固定信号另作五日对照：old_control.py / OLD_CONTROL.md。它重新核对新行动与匹配五日基准，不重选旧信号，也不把当天+3%旧概率当五日概率。免费当前快照与盘口能力见execution_audit.json / book_probe.json；当前报价不证明历史成交。

本轮复核发现原第一名校准分时去重及波动分层边界与文字契约的差异；R03四臂统一修正，原结果保留。历史覆盖变化、规则修复和新增市场信息分别解释。第一批提交e3504fa只含本目录。
