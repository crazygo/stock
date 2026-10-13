# v85：半导体财报预告的时间证据

**本轮交付的是信息源审计和按公告版本查询的代码，尚未形成可训练的完整日历。没有新模型、资金回测或收益通过结论。** 原增长目标保持进行中，最近完成的策略结果仍是 [v84](../soxl_soxs_14d5pct_v84/README.md) 的开发期月均 −5.02%。

[来源审计](evidence/source_audit.json) · [经正文核实的样例](verified_examples.json) · [行业感知思路](RESEARCH_DIRECTION.md) · [继续工作清单](GOAL_PROGRESS.md)

## 已解决的时间语义问题

核实八家公司的九份公告，保存十六条事件版本。它们是查询规则的实证样例，不是完整训练集。

- 博通 2024 年一季度电话会最初安排在 3 月 7 日 17:00 ET；2 月 16 日公告改为 17:15 ET。两个版本都保留，较早的查询不能看到修订。财报仍只承诺收盘后，不能把电话会时间充当财报精确发布时刻。[初次预告](https://investors.broadcom.com/news-releases/news-release-details/broadcom-inc-announce-first-quarter-fiscal-year-2024-financial) · [修订](https://investors.broadcom.com/news-releases/news-release-details/broadcom-inc-announce-first-quarter-fiscal-year-2024-financial-0)
- 应用材料同一份公告包含 5 月 16 日财报电话会与 5 月 2 日设备市场总结；两场活动分别存储，后者不进入财报事件计数。[官方公告](https://ir.appliedmaterials.com/news-releases/news-release-details/applied-materials-announces-q2-fy2024-earnings-webcast-and)
- NVIDIA 区分约定的财报发布时间与电话会；Lam 只明确电话会；Micron 的 Mountain time 保留原精度，当前样例不擅自转换未核实的时区。AMD、KLA、Intel 的收盘后发布也保持区间描述。来源链接逐条保存在样例中。

`announcement_calendar.py` 按查询截止选择每个事件当时最新的公告版本。同公司同日的财报与电话会合计为一个财报相关活动日，行业市场总结不混入；只有电话会日期时，不能推断它就是结果披露日期。计划日期已过只标记“计划日已过”，不能据此确认结果已发布。

发布时间已知时，历史模拟约定下一美东日历日 00:00 才可用；只有出版日期且时区未核实时，延至该日期之后第二个美东日历日 00:00。这个保守延迟覆盖该日期在 UTC−12 下的结束时刻，**是待下一训练协议登记的模拟假设，不是实测历史接收时间**。原 `published_at` 的未知值保持为空。在线管线仍需另外记录真实收到时间。

## 当前覆盖

固定范围为 NVDA、AMD、AVGO、MU、AMAT、LRCX、KLAC、INTC；公告回查区间 2022-10-01 至 2026-04-30。这是既有固定参考组，不是完整历史指数成分。

| 来源 | 目录取得情况 | 完整正文及版本日历 |
| --- | --- | --- |
| KLAC、LRCX | HTML 分页已到回查起点 | 未完成逐条对账 |
| AVGO、MU、AMAT | 正常浏览器遍历年度目录，日期和链接已保存 | 未完成逐条对账 |
| NVDA | 取得前九页，后续遇到传输截断 | 未完成 |
| AMD | 缓存前十八页；另检查第十九、二十页，但持久化目录尚未贯通 | 未完成 |
| INTC | 取得第二页；其他请求遇到连接错误 | 未完成 |

自动正文获取尝试 44 条，32 条取得完整 HTML，12 条失败。完整 HTML 仍需检查标题、发行人、日期和事件含义；下载成功不等于事实已经核实。九份人工核实样例与上述获取次数是不同口径，不能相加为覆盖率。所有公司完整目录均未通过，因此真实样例的八公司查询返回未知计数，不能把少数已知事件以外的日期视为安全日。

`evidence/indexes/` 保存目录事实；其中三份明确标为浏览器转录，保留各年条数及来源，不冒称原始 HTML。正文、截断响应和工具返回留在忽略缓存，文件哈希见 [原始材料清单](evidence/raw_manifest.json)。本轮未获取新行情，没有修改旧模型或旧结果。

## 验证与复跑

20 项测试通过：包括修订前后版本、未来公告追加不改变较早查询、同日事件去重、两场活动拆分、缺失及过期覆盖、未知时区、计划日与实际发生的区别、HTTP 200 截断和无日期 URL 的目录解析。

```bash
.cache/soxl_soxs_14d5pct_v2/venv/bin/python -W error -m unittest discover -s research/soxl_soxs_14d5pct_v85 -p 'test_*.py' -v
.cache/soxl_soxs_14d5pct_v2/venv/bin/python research/soxl_soxs_14d5pct_v85/audit_sources.py
```

审计可离线复算，预期 `training_ready=false`。如需保存复算回执，`--output` 必须指定不存在的文件。真实公告采集脚本是数据准备工具；它们的缓存和失败状态不代表研究完成。没有启动周期调度、实盘下单或自动 R2 上传。
