# R03 本地 raw 复用实现交付

2026-09-27。按已采用的 `REGISTRATION_DRAFT.md` 和 `ADOPTION.md` 实现独立版本 `r03_acquisition_local_reuse_v1`；本页是**代码与离线 fixture 交付**，不是545片采集、prepare 数据集或训练验收，也不批准启动新入口。

## 实际范围

- `local_reuse_source.py` 只读核外部登记、日历及源码身份；按请求复制完整 raw 页、result、audit、normalized、覆盖月 complete/imports/bars 和存在的 revision 文件，复制前后及副本核 SHA。失败保留 `.staging/failure.json`，不提交 bundle；既存 staging 拒绝隐式重试。
- raw 审计逐事件验证页/attempt 有序、有限重试、带时区 receipt、连续页/has_more、原始页 hash/行数、每行证券、5m 时间栅格及请求范围；使用 R03 normalizer 重放并逐月核外部已导入内容。外部 metadata-only、缺 result、合并修订、来源变化、坏 OHLC/重复/basis 均不能升格成功。
- `acquire_local_reuse.py` 新入口保留原 R03 已验真完成片优先，单 worker 和原锁；只在显式 `execute` 命令才可能进入 R2/OpenD，`offline-audit` 不写、不发请求。R2 异常记录后走 OpenD。新结果记录 source bundle、真实 provider receipt 与另行 imported_at，未决来源停止该 run。
- `prepare.py` 仅对新 identity 复验上述 bundle 和代码快照，强制原完成片继承；旧 original/recovery_v1 语义不变。成功空月记录未知覆盖、`rth_complete=false`；旧 v1 无法借新 tier 跳过原 raw 零页门禁。登记源码快照包含 R03 正规化链、prepare、evaluate、data、R2 client 和新入口。

## 离线验证与现时边界

定向命令：`research/after_open_3d5pct/.venv/bin/python -m unittest research.after_open_3d5pct.precision_v9.test_local_reuse research.after_open_3d5pct.precision_v9.test_data research.after_open_3d5pct.precision_v9.test_acquire_recovery -q`；**28 tests，2.255s，OK**。覆盖多月重放/成功空月、旧完整片替换拒绝、旧 v1 tier 伪装拒绝、导入 receipt 和源码篡改、分页/时间/证券/5m 栅格、源变更及失败 staging。测试只使用临时 fixture，没有 R2/OpenD 请求、真实锁或采集写入。

只读 `audit_inventory()` 本次返回545 scope：原完成305、外部待用240为 `missing`；这只是当时来源目录的状态，不代表未来执行结果。没有创建 `R03_recovery_local_v1`、没有下载、没有实际545来源验收或调用完整 prepare。根保留现有等待器及启动决定。

请求组**只复制一次 bundle**；目前为每个导入月及每次 prepare 复验重新完整 raw normalize。五个月组正常路径约14次重放，尚未满足草案“重放一次”的性能目标；未做真实全池耗时基准。正确性门禁不因此放宽，后续运行需观察本地验证耗时与180分钟网络预算，超限按 checkpoint 处理。

源码 SHA256 以本次交付末尾四文件输出为准；独立审查应复核该组字节，再决定是否批准启动。未更改旧 acquisition 入口、旧 source/run、2026 行情、群/标签/模型或其他任务进程。
