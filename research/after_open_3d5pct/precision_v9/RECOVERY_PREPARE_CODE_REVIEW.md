# R03 恢复采集的 prepare 只读验收（2026-09-27）

状态：代码及离线夹具验收完成；恢复采集尚未采用或运行，本记录不证明真实 OpenD 空返回、完整行情覆盖或训练可用性。原 R03 采集目录、`acquire_batch.py`、`acquire_recovery.py`、数据构建器、标签和模型均未改动。

`prepare.validate_acquisition` 在创建输出目录前检查全部 109 个冻结证券 × 2025 年 8–12 月分片。原 R03 登记仍可读取；恢复登记只接受显式 `r03_acquisition_recovery_v1`，从 `identity.source_registration.config` 读取原范围，并核验原登记、原代码快照、ALL/NONE、`no_upload=true`、五个月份及与冻结 v8 候选池加 QQQ 完全相同的证券集合。不能按实际已完成分片缩小股票池。

每个预期分片均须有 `pagination_complete=true`、空 failure、canonical bars、正确 bars SHA、与日历重新计算一致的质量审计，以及 audit 文件。恢复版继承分片还须指向登记的原运行相同证券和月份，并匹配 `inherited/` 记录中的原 result、audit、bars SHA；缺失、变动或错误链接均拒绝。新增分片的成功空表须同时有 `empty_success=true`、未知上市状态标记和成功的零行原始页审计。空表不表示上市前、停牌、无成交或负标签。

来源清单记录采集 registration、calendar、原代码快照、每月 result、audit、bars、原始页及继承记录的 SHA。`coverage.json` 单列 `successful_empty_parts` 与 `unknown_coverage_gaps`；分页完成只表示请求完成，空月的 `rth_complete` 保持 false。五个 2025 月全为 canonical 空表而冻结 2026 有 bar 时允许拼接，再由既有 builder 判断样本；两年均空则显式失败。2026 配对行的时间、入场、终值及标签张量数值门禁保持原样。

离线验证：`research/after_open_3d5pct/.venv/bin/python -m unittest -q research.after_open_3d5pct.precision_v9.test_data research.after_open_3d5pct.precision_v9.test_acquire_recovery`，16 个测试通过。覆盖旧格式完整月、恢复继承和成功空月、来源 audit 变动、错误链接、bars 哈希损坏、未解决失败、未知版本、缺分片，以及跨年空月拼接。未运行完整 prepare、联网、下载或训练。
