# R03 行情采集恢复版本代码复核（2026-09-27）

状态：**仅完成独立代码和离线模拟验证；未启动恢复采集，未改变运行中的 R03 或既有 run。** 采用和运行须由 R03 负责人另行登记。数据目标仍是 R03 BACKLOG/MATRIX 已冻结的 109 证券 × 2025 年 8–12 月、未复权 5m ALL；空返回不推断上市日、停牌或负标签。

## 确定缺口与修复范围

`scripts/fetch_research_data.normalize_kline_dataframe` 在原始 DataFrame 为空时直接返回原表；现行 `acquire_batch.normalize` 随后访问 `start_at`，若将来收到 `RET_OK` 且无 bar，会触发 `KeyError`。这是静态代码路径，不代表当前运行已发生空返回。新 `acquire_recovery.py` 只在空原始页时生成与非空标准化结果同列的零行 canonical Parquet。成功原始页及请求/接收时刻仍保存在新输出的 `raw_*.parquet` 和 `audit.json`。结果标记 `provider_returned_no_bars_unknown_listing_state`；质量审计保留所有预期 RTH bar 缺失，`rth_complete=false`。

该独立版本显式读取旧 R03 登记和 `source_snapshot`，核验原采集器及 normalizer 的登记 SHA；自身登记记录新代码、被导入模块、旧登记 SHA。已完成旧分片只有在 result、bars SHA、canonical schema、质量审计及 audit 文件一致时，才在新输出以符号链接继承。`inherited/` 下逐月记录原结果、审计和 bars SHA；原目录只读，不复制或改写原始时间戳。每次恢复运行重新核验链接来源。

新分片沿用 Local → R2 → Futu、单进程文件锁、每月最多 30 页、每页最多 3 次、`--max-minutes`、ALL/NONE；不使用交易账户或上传。请求异常、额度错误和页数上限保留原始页与审计，写入 `pagination_complete=false` / `unresolved_failure`，继续处理其他月份。已失败的新分片不会在同一输出中隐式重试；旧失败分片须显式传 `--retry-source-incomplete` 才会在**新输出**重取。汇总分别统计完整分片、成功空分片、未解决失败及待处理分片；只有全部完成且没有失败时状态才是 `acquisition_complete`。

## 验证与采用边界

`python3 -m unittest -q research.after_open_3d5pct.precision_v9.test_acquire_recovery`：4 个离线用例通过，覆盖无列空成功、原始成功审计、正常月哈希继承、损坏/未完成 checkpoint 拒绝、分页失败和汇总状态。另对运行中旧 run 的一个**已完成**正常分片只读执行 `verified_complete`，bars、质量及审计匹配。未测试真实 OpenD 空回复或恢复运行结果。

现有 `prepare.py` 的 `pagination_complete`/`failure`/bars SHA 门禁会拒绝新输出的失败分片；canonical 空表可由其按月拼接。正式采用前，建议在 `prepare.py` 增加最小只读门禁：识别新恢复版 registration；对符号链接分片复核 `inherited/` 中的原 result、audit、bars SHA；在 coverage 中单列成功空月的未知来源缺口。该共享文件本次未修改。若 2025 全年空且 2026 来源也空，builder 的 price-basis 判断仍需单独审计，不能把无数据标成有效训练行。
