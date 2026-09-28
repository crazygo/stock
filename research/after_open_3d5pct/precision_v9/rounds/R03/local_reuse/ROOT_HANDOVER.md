# 根协调器：恢复等待器接管记录

2026-09-27，执行采用见 `RUN_ADOPTION.md`（SHA `8cee62b3d1b40730ca6dea6509e7e871dad74362868a9a7df9c31c43ccb6dcc1`）。独立审查28项＋3项通过，根实际再次核十项源码无变化。

- 13:36:49 UTC：再次核自有PID97894的启动时间/命令与初始证据一致、共享锁仍占用、旧v1仅registration/calendar/source_snapshot后，仅向它发送SIGINT。会话22976随后返回完成（工具报告exit_code=0），PID97894已不存在。原v1没有parts或新请求制品，registration SHA仍为`5ecaa998cac57f5cf33918677232b2bd35ca6ce9351479bf217655cc90910001`。
- 13:37:14 UTC：初始化独立`R03_recovery_local_v1`，version=`r03_acquisition_local_reuse_v1`，冻结十源码与日历。新registration SHA `568806614a877b3e3b94e051ac8b3a2eef3a12d72ceb6005556d6cb0e535d572`，calendar SHA `26ac90fc8b48156ebb34f869d9b24fb5e81e3e420948b32ee582f0e16459365d`；未联网。
- 13:37:55 UTC：启动唯一新统一exec会话 **74618**，自有PID **7064**，已输出`waiting_for_shared_lock`。等待前和获锁后重核采用文档、新registration及十源码身份；随后按固定`max_minutes=180`调用新run。当前仅排队，不是采集完成。
- 另一任务PID96498仍运行，未中断/修改它或OpenD；没有删除共享锁。不得再启动22976、原71777或额外waiter。

后续只poll74618，并结合新run的实际progress检查。锁释放后原305片重新验真继承，剩240按当时来源状态处理，不沿用旧盘点推定齐备。Local重复重放及墙钟预算边界见采用文档；20分钟为根观察检查点。全545来源、unknown覆盖、完整prepare和配对2026审计尚待执行，不开始训练或声明模型达标。

逐步机器记录：`ROOT_HANDOVER_PREFLIGHT.json`、`ROOT_WAITER_CANCEL.json`、`ROOT_NEW_RUN_INITIALIZED.json`、`ROOT_LOCAL_WAITER_EVENTS.jsonl`。旧R03/v1与全部首失败证据保留。
