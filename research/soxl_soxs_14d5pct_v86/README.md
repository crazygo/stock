# v86：可按当时信息查询的半导体财报日历

**已补齐并核对固定八家公司季度财报活动目录，形成 119 份正文核实公告、193 条事件版本。当前交付是新输入与查询代码，没有新模型或资金回测。** 收益目标仍未完成，最近完成的策略仍为 [v84](../soxl_soxs_14d5pct_v84/README.md) 的开发期月均 −5.02%。

[来源审计与季度对账](evidence/source_audit.json) · [公告事实](reviewed_announcements.json) · [覆盖约定](coverage.json) · [目标继续](GOAL_PROGRESS.md)

## 对策略的用途

用户提出的“半导体会阶段性热门”仍按当时可见信息研究：已有相对强弱、短期趋势、量价状态描述市场方向；本日历新增的是持仓窗口内提前公布的季度活动及集中程度。两者可能共同影响止盈和亏损概率，必须与没有新事件输入的同条件模型比较，不能直接规定临近财报就买入或空仓。

此前可见价格状态的研究已经失败，不能把相同输入改名“热度”便宣称能力提高。本轮也不使用财报业绩内容、后来上涨的阶段标签或五月／七月／九月收益来选择信息源。所有来源范围事先固定为 NVDA、AMD、AVGO、MU、AMAT、LRCX、KLAC、INTC；这是固定参考组，并非完整历史指数成分。

## 覆盖与时间语义

公告回查从 2022 年 9 月开始，截止 2026-04-30。可查询的完整目录截止范围为 **2023-01-01 至 2026-04-30**，此前保留一个季度作起点核对；范围外返回未知。115 条已到期季度结果目录记录全部匹配到正文核实的提前活动日期，另有 3 个已公告、尚未到来的活动；合计 118 个公司活动日。博通一份修订与初稿属于同一活动，因此公告数量为 119。

| 公司 | 正文核实公告 | 公司活动日 | 已到期结果目录记录 |
| --- | ---: | ---: | ---: |
| NVIDIA | 15 | 15 | 14 |
| AMD | 15 | 15 | 14 |
| Broadcom | 15 | 14 | 14 |
| Micron | 14 | 14 | 14 |
| Applied Materials | 15 | 15 | 14 |
| Lam Research | 15 | 15 | 15 |
| KLA | 15 | 15 | 15 |
| Intel | 15 | 15 | 15 |

`earnings_calendar.py` 沿用 v85 的公告可用约定：有明确出版时刻时延至下一个美东日历日零点；只有日期、未核实出版时区时延至第二个美东日历日零点。实际本次读取时间另外保存。这是历史模拟假设，不是历史系统收到消息的证据。

查询只取截止时点可见的最新公告版本。财报与电话会按公司及日期去重；计划日期已经过去不表示确认实际发生。未来窗口包含查询日及之后 13 个日历日，最近窗口为之前 14 日。事件当天采用整日粒度，保留源文分钟精度但不在盘中假定电话会已经举行。缺失、过期或超出覆盖时，四个特征均为未知；少数已知事件仍可查看。查询结果始终没有模型或交易资格。

目录完整仅指这次核对的**常规季度活动**：没有穷尽全部企业新闻，也无法证明已经收集所有未被官网归档的修订。目录结果标题仅用于覆盖审计，不是预测输入或实际事件发生证明。完整原文和失败响应保存在忽略缓存，不入 Git。

## 本轮解决的具体漏项

- AMD 三次活动日期分别藏在[初步业绩](https://ir.amd.com/news-events/press-releases/detail/1093/amd-announces-preliminary-third-quarter-2022-financial-results)、[CFO 交接](https://ir.amd.com/news-events/press-releases/detail/1112/amd-announces-cfo-transition)和[合作公告](https://ir.amd.com/news-events/press-releases/detail/1260/amd-and-openai-announce-strategic-partnership-to-deploy-6-gigawatts-of-amd-gpus)正文中。标题筛选漏掉的三次均已补入；同稿当天合作电话会不混入后来的财报电话会。最后一稿对十一月活动使用夏令时缩写，分钟转换保留未决，日期可用。
- Lam 2022-09-28 官网归档标题与正文不一致，正文实际描述另一场八月公告的投资者活动。保留冲突页，采用[公司发布的原始发行公告](https://www.globenewswire.com/news-release/2022/09/28/2524713/0/en/lam-research-corporation-announces-september-2022-quarter-financial-conference-call.html)核实十月十九日电话会。
- 保留博通 2024 年二月公告中的电话会改时、应用材料同稿中的两场不同活动，以及美光未转换的 Mountain time。只写明电话会的来源不补造精确财报发布时间。
- NVDA、AMD、Intel 分页已贯通；美光、博通年度目录及应用材料全部年度分页均保存了直接 DOM 观察。旧的截断记录和一次空加载目录保留并排除，未计作有效覆盖。

## 验证与复跑

18 项测试通过，涵盖真实修订、未来追加不改变过去、内嵌公告、错误正文拒绝、活动拆分、十四日边界、同日去重、缺失、覆盖过期和源文件复算。测试通过仅证明这些数据与时间规则，不能替代收益验收。

```bash
.cache/soxl_soxs_14d5pct_v2/venv/bin/python -W error -m unittest discover -s research/soxl_soxs_14d5pct_v86 -p 'test_*.py' -v
.cache/soxl_soxs_14d5pct_v2/venv/bin/python research/soxl_soxs_14d5pct_v86/earnings_calendar.py --cutoff 2026-04-15T09:25:30-04:00
.cache/soxl_soxs_14d5pct_v2/venv/bin/python research/soxl_soxs_14d5pct_v86/audit_calendar.py
```

事实查询只依赖随 Git 的公告与覆盖契约；完整来源审计需要本地原始缓存，缺缓存明确失败。`assemble_indexes.py`、`prepare_review.py` 仅生成忽略缓存中的待核实材料，不会自动改写已核实事实。`capture_receiver.py` 是手动临时 loopback 表单，用于把只读浏览器观察原样落盘；不联网抓取、不执行源文、不接受任意输出路径，用毕停止。
