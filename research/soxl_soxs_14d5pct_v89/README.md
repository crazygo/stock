# v89：公司收入指引与兑现的来源准备

**128 份目标公告正文已取得，候选解析和配对检查完成；尚未形成获准训练输入或新的收益结果。** 本轮把用户关于半导体行业升温的提示继续推进到基本面证据：公司实际收入是否兑现其此前对同一季度的指引，以及期间是否已预告或修订。原最新策略仍为 [v88](../soxl_soxs_14d5pct_v88/README.md)，其开发月均 −2.60%，收益目标未通过。

这不是分析师一致预期，也不把事后上涨的月份标为“热门”。未来若使用这些信息，只能从公告公开后的保守时点起改变判断，随后每日使用当时成熟历史训练。此次尚无新算法拟合，不根据五／七／九月结果选字段或公司。

## 完成了什么

固定八家公司，目录范围为 2022-07-01 至 2026-04-30。沿用已保存的官方目录定位，并保留五项其他主题的排除理由。所有 128 份正文均通过标题身份和实质内容检查。78 份通过普通 HTTP 取得，一份 AMD 预告复用已有浏览器缓存，49 项 HTTP 失败或不完整记录保留，并通过正常浏览器读取对应官方正文补齐。完整正文留在忽略缓存，Git 保存来源、哈希、数值候选和核验摘要。

| 公司 | 正文 | 正常季度业绩 | 其他公告 | 季度指引候选 | 年度指引候选 |
| --- | ---: | ---: | ---: | ---: | ---: |
| NVIDIA / NVDA | 16 | 15 | 1 | 15 | 0 |
| AMD | 16 | 15 | 1 | 15 | 0 |
| Broadcom / AVGO | 15 | 15 | 0 | 12 | 3 |
| Micron / MU | 17 | 15 | 2 | 15 | 0 |
| Applied Materials / AMAT | 16 | 15 | 1 | 15 | 0 |
| Lam Research / LRCX | 16 | 16 | 0 | 16 | 0 |
| KLA / KLAC | 17 | 16 | 1 | 16 | 0 |
| Intel / INTC | 15 | 15 | 0 | 15 | 0 |
| 合计 | 128 | 122 | 6 | 119 | 3 |

122 份季度业绩均抽取到当前季度收入候选，且附近存在对应的百万／千美元单位标记；119 项前瞻季度指引的财务期间与下一季度一致。八家各一份正文的单位、当季列和指引数值已单独阅读核对，另核对六份特殊公告与三份年度指引。这些抽样及结构检查不能代替其余 114 份季度结果的逐项核验，也没有完成全部业务范围与公开时间审查。

## 配对与已发现的问题

候选公式为 `本季实际收入 / 此前同一季度收入指引中点 − 1`。保留两个不同的参照：目录内较早的定期季度公告指引，以及考虑已取得修订公告后的最后已知参照。前者不称为全历史首次指引，后者不称为完整历史最后指引。

- 122 个季度实际结果中，111 个能匹配此前定期公告中的同季度指引；11 个保持缺失。其中八个属于目录开端，另三个是 Broadcom 的 FY2024 Q1–Q3，前期公告只给出全年指引。
- 110 个能形成最后已知数值参照的候选比值：108 个仍用定期公告，两个使用明确修订。剩余一个 Micron 修订没有精确中点，保留缺失。
- NVIDIA 与 AMD 的 2022 年业绩预告单列，不写成正式实际收入，也不悄悄替换原指引。正式季报里的差异可能早已部分公开，不能都当成当天新消息。
- AMAT 2022-10-12 将本季收入指引从 66.5 亿美元 ±4 亿改为 64 亿美元 ±2.5 亿。正文另有出口限制影响金额，解析时不能把它当收入总额。见[官方修订](https://ir.appliedmaterials.com/news-releases/news-release-details/applied-materials-revises-fourth-quarter-fiscal-2022-business)。
- Micron 2023-11-28 将原 44 亿美元 ±2 亿更新为“接近 47 亿美元”，未给出精确中点／上下界；候选配对拒绝为该修订制造精确比值。其 2025-08-11 修订则明确给出 112 亿美元 ±1 亿。见[2023 年更新](https://investors.micron.com/news/press-release/2023/Micron-Updates-First-Quarter-Fiscal-2024-Guidance-11-28-2023/default.aspx)及[2025 年更新](https://investors.micron.com/news/press-release/2025/Micron-Updates-Fourth-Quarter-Fiscal-2025-Guidance-08-11-2025/default.aspx)。
- KLA 2026-03-12 重申三月季度指引但未重列收入金额，保留对原季度公告的引用；回购金额不能被提取为收入。

业务范围也需审查。例如 Intel 2025 年三季度公告说明 Altera 在季度中途退出合并报表；Broadcom 的 FY2024 年度指引包含 VMware。当前的公司总收入不能直接等同于纯半导体收入或不变范围的行业需求。见[Intel 公告](https://www.intc.com/news-events/press-releases/detail/1753/intel-reports-third-quarter-2025-financial-results)及[Broadcom 公告](https://investors.broadcom.com/news-releases/news-release-details/broadcom-inc-announces-fourth-quarter-and-fiscal-year-2023)。

## 时点、验证与下一步

配对预览暂用目录公开日期加两个日历日的美东零点作为保守可用假设。它不是已经核实的历史接收时刻，也不证明网站正文未曾修改。2026 年 10 月取得正文的时间单独保存；不由财报计划日推定实际公开时间。精确时间、时区、日期一致性、修订史覆盖仍待审查，全部候选 `training_use_allowed=false`，训练入口守卫实际拒绝本版候选和预览。

15 项测试通过，覆盖年度与季度隔离、目录开端缺失、预告与正式结果隔离、模糊修订、重申、财年、单位、时间边界、未来值扰动不改变过去配对及未审核输入拒绝。全量来源哈希、128 份解析复现、122 个单位邻近标记、119 个下一季度关系、三份年度例外、候选比值算术核对通过。该审计没有认证全量财务口径、发布时间、修订完整性或策略效果。

后续按[继续研究](NEXT_RESEARCH.md)完成数值、公开时点和业务范围审查，再预先冻结新的输入表达与开发实验。最终五／七／九月净资金收益算术均值至少 10%、五月至九月各月非负的目标保持；本轮没有新增指定月份回归、R2 上传或券商交易。[持续目标](GOAL_PROGRESS.md)

## 文件与复跑

[协议](PROTOCOL.md) · [目录与排除项](catalogue.json) · [正文清单](evidence/source_inventory.json) · [收入候选](evidence/revenue_candidates.json) · [特殊情况与核对锚点](review_cases.json) · [配对候选](evidence/pair_candidates.json) · [待审清单](evidence/review_queue.json) · [来源核验](evidence/source_verification.json) · [交付清单](evidence/delivery_manifest.json)

以下命令只做本版来源处理，不训练模型。完整复跑需要清单所列的本地原始正文及 v86 目录缓存；仅克隆 Git 不含这些原始材料。`catalogue.py` 已冻结目录，不能覆盖重建；`collect.py` 可复用已记录获取结果，失败不自动重试。浏览器补充采用 `capture_receiver.py` 的临时 loopback 表单，采集完成后已停止，不接触既有 8768 服务。

```bash
python3 research/soxl_soxs_14d5pct_v89/extract_sources.py
python3 research/soxl_soxs_14d5pct_v89/parse_revenue.py
python3 research/soxl_soxs_14d5pct_v89/pair_candidates.py
python3 -W error -m unittest discover -s research/soxl_soxs_14d5pct_v89 -p 'test_*.py' -v
python3 research/soxl_soxs_14d5pct_v89/audit_sources.py
```

以上生成命令会刷新本版派生证据，不应把已发布清单的变化冒充旧版本。原始抓取及失败记录保留；验证后由 `package_delivery.py` 生成交付哈希清单。
