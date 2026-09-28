# 群组补充预测报告

[打开本机报告](http://127.0.0.1:8768/analysis/model_group_supplement_20260927/index.html)，也可直接在浏览器打开自包含 `index.html`。

100只证券，63个群组，5个冻结模型，9个触及目标。新增覆盖 CRDO、COHR、VRT、SOXL、SOXS；没有重训或改写原预测。

历史评估为2026-09-01至09-17的12个交易日，概率快照为09-24 11:30 ET。主目标为延迟入场后的3日内触及+5%。默认按50%阈值的历史分类正确率选各模型前三群，并展示最新成员去重并集。正确率包含正确预测“不达成”，不能当成上涨机会命中率。

使用顶部控件切换目标、排序和模型；按行业、趋势等筛选或输入证券代码；点击群名查看五模型对照。页面可下载完整JSON。

- [计算前登记](../../research/after_open_3d5pct/docs/29_group_supplement_registration.md)
- [结果与验收证据](../../research/after_open_3d5pct/docs/30_group_supplement_evidence.md)
- [归档结构化数据](../../research/after_open_3d5pct/runs/group_supplement_20260927_r2/report.json)
- [来源和冻结文件哈希](../../research/after_open_3d5pct/runs/group_supplement_20260927_r2/manifest.json)
- [数据复算验收](../../research/after_open_3d5pct/runs/group_supplement_20260927_r2/audit.json)

前三群来自同一历史窗口的事后选择，尚无独立前向证据；行业是当前分类回溯。SOXS合股前原价历史已屏蔽。完整解释见验收证据。
