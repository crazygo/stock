# R03 恢复启动后的锁阻塞检查点

2026-09-27。原采用登记 [RECOVERY_ADOPTION.md](RECOVERY_ADOPTION.md) 已通过根代理审查，SHA-256仍为 `a512352c280d3a532c27916c00a5bc30b9486033a15b591ad0c3726b19441c6b`；本页补记后续状态，不修改冻结登记。

## 实际状态

- 根代理首次启动恢复器后，`_initialize`完成，随后 `flock(LOCK_EX|LOCK_NB)`报Errno35退出。锁获取位于网络连接及分片循环之前；根执行结果确认本次未发请求、未继承分片。
- 新目录 `runs/precision_v9_20260927/R03_recovery_v1` 当前只有 `registration.json`、`calendar.json`、`source_snapshot/`，没有parts/inherited/progress/summary。登记创建于 `2026-09-27T11:17:53.547172Z`；这是已初始化、等待共享锁，不是恢复采集完成。
- 根代理随后已启动统一exec等待会话 **22976**，当前输出 `Waiting for shared acquisition lock`。它只等待同一flock，当前没有网络请求或续采；锁释放后调用已登记 `acquire_recovery.run(..., 180, retry_source_incomplete=True)`。由根代理后续poll此会话，不需要另启等待器或用户回来提醒；旧71777已退出1，不再poll。
- 原R03仍为305/545已完成，HONA/2025-08原始零行页与audit保留；原progress遗留running不能当活跃任务证据。
- 只读 `ps`核对PID96498：Python3.14 `-u scripts/backfill_model_history.py acquire`，启动于2026-09-27 19:03:52 Asia/Shanghai。`lsof /tmp/stock_futu_acquisition.lock`显示该PID持有文件描述符；结合恢复器锁失败及该任务日志，确认为本次锁竞争来源。
- `.cache/model_history_acquire.log`记录等待锁、prepared 109证券/305 R03分片、19:13:15建立OpenD连接，随后补2025年1–7月、2024全年、2023年4–12月等历史。该任务属于另一项已运行工作，不终止、不修改、不删除其锁或文件。
- 根代理提供完整采购跨度2023-04至2026-09；本次只读日志确认上述已打印区间，不把它视作R03恢复结果或完整性认证。

## 源码快照核对

本次逐项重算并匹配7份快照：acquire_recovery.py、identity列出的5个imports（acquire_batch/acquire_pilot/fetch_research_data/r2_client/evaluate）及RECOVERY_ADOPTION.md。当前工作源码是否仍匹配需续跑前再检查；本页没有改源码。

| 产物 | SHA-256 |
|---|---|
| recovery registration.json | `5ecaa998cac57f5cf33918677232b2bd35ca6ce9351479bf217655cc90910001` |
| recovery calendar.json | `26ac90fc8b48156ebb34f869d9b24fb5e81e3e420948b32ee582f0e16459365d` |
| 7份 `{filename:sha256}` 映射，sort_keys、紧凑JSON的SHA | `c30697bccace56747922bd8316334c67f740aeb480417de9f80338c351d83d27` |

单文件值保存于registration.identity及source_snapshot，采用文档SHA见页首。本检查点只证明源字节保存与初始化，不证明恢复请求或数据完成。

## 继续条件与动作

1. 等现有任务自然释放共享锁；由已存在会话22976继续等待，根代理poll该会话并只读核对进程/日志。不要并发启动额外采集、重复等待器或删除锁绕过互斥。锁占用是协调检查点，不是HONA数据重试失败。
2. 续跑须核对原registration、恢复identity、7份快照及当前imports SHA。未变且新目录仍无未提交part时，会话22976复用同一个 `R03_recovery_v1`，按采用登记的同一参数继续；不另手动启动第二份命令，不需要另起开发或新采集版本。
3. 如果锁再次被合法任务抢先获得，保留本目录并继续等待；若代码/identity变化或出现未提交part，停止续跑，按采用登记另行审查，不覆盖旧证据。
4. 启动后确认新progress与实际part；期待重新核验后继承305片、处理余240片。另一任务新增本地缓存可能改变Local命中，仍须由恢复器检查范围/口径/质量并保存实际来源，不能只凭日志直接认定完成。
5. 全部请求结束且无失败后才进入prepare与配对验收；成功空月保留unknown覆盖。模型匹配基线尚未开始，90%目标当前仍为0路线通过。

本次只新增检查点并更新协调入口；未采集、未调度、未干预PID96498、未改旧R03或冻结采用登记。
