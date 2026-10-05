# 广池证券类型与CIK覆盖修复 v4

本版修复旧名称分类误排普通ADS、注册/投票股，以及SEC frame名称关联缺失。全Nasdaq目录13,295条均保留去向：5,516明确股票、98待核验价格/财务观察证券、7,681排除；全观察池5,614证券，唯一精确CIK覆盖5,603。比旧注册池增加457条观察证券，排除104条明确基金；不能把基金当普通经营公司。当前SEC映射仍须submissions当前ticker再次确认，不是历史PIT。

[新旧目录逐项差异](COVERAGE_CHANGES.csv)及[差异复算](COVERAGE_CHANGES.json)保留新增、排除与CIK变化。457个新增观察证券包含360个明确股票和97个类型待核验证券；旧CATALOG_EVIDENCE的added_candidates计数仅指明确股票。范围为当前Nasdaq traded目录，OTC、退市历史与PIT成员尚未完整覆盖。

[获取协议](PROTOCOL.md)、[训练协议](TRAINING_PROTOCOL.md)、[全目录逐行证据](CATALOG_EVIDENCE.json)、[实际目录复算](CATALOG_VERIFICATION.json)。旧v3报告独立保留；本版完整源、固定4–9月回测、报告和桌面/320px实际检查已结束，默认根入口已切换到v4。没有通过>80%目标。证券类型未知不发主有效信号，不用未来涨幅决定类型。

```bash
# 单次完整扫描（等候已有完整源，不重复启动）；新源完成后生成回测、报告及调试页
python3 analysis/doubling_opportunity_v1/scan.py --version coverage-v4 --top 30 --debug

# 仅读实际来源/训练状态
python3 analysis/doubling_opportunity_v1/scan.py --version coverage-v4 --status

# 全新来源cohort，刷新目录和SEC；原始来源与旧模型不覆盖
python3 analysis/doubling_opportunity_v1/scan.py --version coverage-v4 --refresh --top 30 --debug

# 重算完整证券目录与唯一精确CIK（保持已有官方来源快照）
python3 analysis/doubling_opportunity_v1/coverage_v4/catalog.py
python3 analysis/doubling_opportunity_v1/coverage_v4/verify_catalog.py

# 完整发行人两种SEC来源，断点续取，最多3工作者，共用0.4秒请求间隔
python3 analysis/doubling_opportunity_v1/coverage_v4/acquire_sec.py

# 原生日线只读获取、失败/空/旧来源恢复、完整动作核对
python3 analysis/doubling_opportunity_v1/coverage_v4/acquire_daily.py
python3 analysis/doubling_opportunity_v1/coverage_v4/recover_sources.py
python3 analysis/doubling_opportunity_v1/coverage_v4/prepare_actions.py

# 全部源终止后，固定4–9月评估，不对下载中的子集训练
python3 analysis/doubling_opportunity_v1/coverage_v4/train.py --reuse-if-unchanged
```

不要重复启动已有获取进程；锁文件保护原生日线、SEC和动作审计。原始数据留在忽略缓存，不入Git、不上传R2、不调用交易接口。实际已完成范围、来源失败、校准退化和未知都保留。覆盖修复不等于找到已证明>80%的翻倍机会。

相同完整交易日/目录复用冻结源与模型，报告重新读取全池报价；日期推进或显式refresh会建独立source_cohorts缓存，重新获取SEC，不将旧财报快照称最新。SEC完整ticker网页的派生映射按保留的来源快照使用，当前ticker另由新submissions确认；映射外的新证券保留缺失，不能制造CIK。未来同一版本仍不消除PIT/RTH和独立验证限制。默认根入口为本版；明确--version raw-v3仍可复跑旧注册池。

2026-10-05实际新增原始来源核查：AMAT/NVDA可按原冻结经营必要条件解析；ASML为EUR等金额单位，TSM/BNS使用IFRS，现有USD US-GAAP解析器没有支持，不能将这些缺失当成经营差。原因和单位单列，模型特征与准入不因此改动，证据见FINANCIAL_SOURCE_REVIEW.json。

单次run.py报价补充由可终止子进程执行，默认30秒；每个已完成批次原子落盘，旧完整快照按SHA保留，超时仍输出日线参考和缺失。该预算适用于报告报价补充；完整历史/财报准备及训练是独立的注册批处理阶段，不冒充30秒即时推荐。研究在main继续，保留其他WIP，原始数据不入Git。

[实际30秒预算与中断保留核验](QUOTE_BUDGET_VERIFICATION.json)、[报告股票优先获取核验](QUOTE_PRIORITY_VERIFICATION.json)。后一次实测取得1,512/5,614个快照，4,102个缺失，30.02秒超时终止；MXL、AMAT、NVDA、ASML、TSM、BNS均在已返回批次。此处“完整快照文件”指完整写入的JSON文件，证券池仍可能仅部分覆盖。报告逐股保留latest_quote_status和latest_quote_missing_reason，不能称全池实时行情已取得；报价优先顺序不改变模型排序、训练成员或有效信号资格。

[首次拟合前登记的固定概率诊断](PROBABILITY_DIAGNOSTIC_PROTOCOL.md)复用原有>80%与60日冷却口径，单列去掉质量/热点门禁后的诊断机会。只读旧v3冻结预测的[实际回放核验](DIAGNOSTIC_REPLAY_VERIFICATION.json)复算六组TP/FP/未知，22个原生日线标签检查通过；重跑冻结产物逐字节不变。旧回放核验只证明诊断代码和旧标签可复算。新版另行执行并核验原生标签：月前选择60日有4次诊断机会、1TP/1FP/2未知，成熟precision=50%；小样本和未知不证明80%，不充当主策略有效信号。

完整来源和实际结果见[当前交付审计](../DELIVERY_AUDIT.md)、[CURRENT_EVIDENCE](CURRENT_EVIDENCE.json)、[UI核验](UI_VERIFICATION.json)。正式报告引用[latest_run.json](latest_run.json)。4–9月六组主策略零>80%信号，precision不可评分；当前最高P60探索估计30.88%，25个经营/个股热度候选最高3.02%，无合格股票。

[全无报价参考报告的实际核验](NO_QUOTE_REFERENCE_VERIFICATION.json)保留5,614证券与3,886冻结估计，全部报价缺失不阻止参考报告。首次内部报告的最新交易日缺失展示字段已[修正并留存](REPORT_FIELD_CORRECTION.json)，模型与原始标签不改。

[根入口默认版本的实际复跑](DEFAULT_REPEATABILITY_VERIFICATION.json)已正常退出，状态为complete_development_experiment_reused。相同数据与训练指纹、冻结文件SHA和5,614逐股概率/资格字段一致。本次30.01秒终止报价子进程，保留1,200个完整快照、4,414个缺失；最终页面与报告绑定重新核对，较早的桌面/320px交互实测以原始收据保留。
