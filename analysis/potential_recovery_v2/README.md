# AI恢复候选 v2

入口：`http://127.0.0.1:8768/analysis/potential_recovery_v2/index.html`。

本次重新推荐把持续下行与业务待修复排除在优先名单之外，沿用stock-data-backfill / stock-report-refresh的数据版本、独立标签、覆盖及实际运行思路。原potential_recovery_v1、原五维地图与质量评级均保留。

范围为原报告已明确AI++/+的149只经营候选，另保留MXL对照。重点刷新全部财务A、全部AI++及QQQ/SMH，未刷新股票保留原截止。501只AI待分类仍是覆盖缺口。最新完成行情目标为2026-10-06 ET，业务重点核查8家公司9条官方披露；标准化SEC财务沿用原缓存，其日期与重点新读披露分开。部分官方原文抓取失败，页面明确为浏览器查阅并保留URL，不能称全部原文已归档。

五属性是five_traits_v2的描述统计，核对公式后按同一窗口、256次5日联合移动块重采样复用。新探索门槛见PROTOCOL；没有训练新的收益模型，也未证明未来收益或操纵风险已排除。S较高可能表示稳定下跌，本次TTD即为真实反例。

本次价格引用统一使用完成日Close，年高点和峰后低点来自同一QFQ完整序列。FOCUS默认显示ON/POWL/ENS及等待/修复中观察的AVGO/AEIS；TTD单列反例。未满足条件的股票及未补齐输入均留在审计。当前价与旧年高点的算术空间不是合理价值，旧PE/FCF收益率没有重估为当前估值。

命令从仓库根目录运行：

```bash
python3 analysis/potential_recovery_v2/acquire.py --budget 120
python3 analysis/potential_recovery_v2/capture_evidence.py
python3 analysis/potential_recovery_v2/build.py
python3 analysis/potential_recovery_v2/verify.py
python3 analysis/potential_recovery_v2/render.py
```

acquire是可终止工作进程，使用Local→同粒度R2→限速OpenD；已发布行情在market_data/stock_data_v1/captures，每次capture与旧输入隔离，不上传R2。首次运行取得67份当前必要行情；发现两年导航的前段不足后，第二次复用43份不可变完整序列、补24份两年完整序列。两个运行清单与资源记录均保留于.cache/stock_data_v1/runs。整体149只范围仍是partial，不能以重点覆盖齐全冒充全池更新完成。

build只读取DATA_POINTER发布清单并校验哈希，输出不可覆盖快照与实际尝试账本；相同输入及参数重复执行复用原快照。时间谱为历史回算，不代表当日实时执行。render不联网、不新增标签点，仅从已保存输入原子发布单文件HTML。

index.html包含真实OHLC，忽略Git；原始Parquet/缓存不入Git。跟踪代码、公式、摘要快照和证据指针。重建需要指针所引用的本地不可变输入，缺失时须通过数据流程补齐，不把不存在的文件当作已交付行情。

验证覆盖独立复算G/V/M/J/S、未来行情截断、内部缺日、稳定下跌反例、全成员保留、输入哈希与真实页面筛选/日期/mini导航/窄屏。工程验证不是策略效果验证。
