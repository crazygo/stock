# R03 本地复用执行采用

2026-09-27，根采用 `CODE_INDEPENDENT_REVIEW.md`（SHA `9a36d7fa7a3eecd9e27d456a623bee07f24bbd431833bbfb984b10ffc3834ad0`）及代码清单 `CODE_REVIEW_MANIFEST.json`（SHA `60166647c52805403d5d486acb2974b9686e8a7dc48499b58660a8f53c5eefc3`）。28项套件和3项独立控制流检查通过；十项源码组SHA `5678214636b3037505fabb659a92fb6759f46d4583e436603414957055c193e3`。根再次核十项实际源码与独立审查一致。

本轮显式接受一项实现偏差：每个provider请求bundle只复制一次，但逐月使用和prepare会重复完整raw重放（五个月组约14次）。保留全部来源/质量门禁；不宣称一次重放或全池性能。实际耗时另记，根每20分钟检查进度。`max_minutes=180`从首次网络阶段开始计墙钟，含后续穿插Local，在新网络分片前检查；首次网络前Local不计入。它不是纯累计网络计时或强制中断每个正在执行的请求。

输入固定原 `runs/precision_v9_20260927/R03`；新输出固定 `R03_recovery_local_v1`，新version `r03_acquisition_local_reuse_v1`。108 candidate＋QQQ、2025年8–12月、原305优先、冻结2026、单worker、Local→R2→OpenD、原分页/重试、no_upload与所有unknown/失败语义不变。545来源状态及prepare配对另验，当前不训练。

根启动前检查已保存 `ROOT_HANDOVER_PREFLIGHT.json`，SHA `88a290a1161fbf5ad0148366429d2bf0d1b5ae0cb42c66b45c67ea5eeb36bfe8`：旧v1仅registration/calendar/source_snapshot，无parts/progress；新目录尚不存在；自有waiter PID97894启动19:21:58上海，另一采集PID96498启动19:03:52。取消前必须再检查这些条件，仅向自有waiter发中断并poll会话22976；另一任务/共享锁文件/OpenD不操作。若旧v1已开始采集则本接管停止，另审checkpoint。

旧waiter退出后保存状态，再初始化新run冻结identity/十源码/日历。唯一新waiter等待同一共享锁自然释放，获得释放信号后调用 `acquire_local_reuse.run(source, output, max_minutes=180)`。release→非阻塞run之间若发生合法锁竞争，保存失败并另协调，不删除锁或并发补开采集器。实际会话/时间/哈希另保存在交接记录，不能把排队写成已采集。
