# R03 原配方 runner 与冻结 v8 工程复放

`baseline.py` 每次只运行一条路线、一折、一臂。`old_common` 用旧快照共同键；`new_common` 用新快照共同键；`new_expanded_fit` 仅额外纳入共同键起点之前且不早于 `fit_date_floor` 的新 fit 行。四段资格受共同键约束，成熟本股先验始终从所选快照的**全部行**按当时已成熟时间计算。旧 `fit_b/fit_c`、九目标、校准、投影、阈值和冷却原样调用；无 R01 失败机制。`eval_full` 另存原始评价域，供给分母始终为完整官方 eval session。

计划 JSON 必填：`dataset_root`、`old_dataset_root`、`new_dataset_root`、`calendar`、`route`、`fold`、`arm`、`allowed_keys`、`fit_date_floor`、`incumbent_summary`、`actual_manifest`、`paired_audit`、`backlog`、`matrix`、`seed:3566`。`allowed_keys` 是旧新 `(symbol,session_date)` 全部交集的 CSV/Parquet。`actual_manifest` 须有 `status:"accepted_for_fit"`、`evidence_scope:"exposed_development"`、三文件 `dataset_sha256`、`calendar_sha256`、`paired_audit_sha256`，及当前池回溯、历史到达时间假设、公司行动历史覆盖未认证三项 `limitations`。先验登记和全部源哈希在拟合前落盘；源码快照、fit正权重行及先验历史ID另存，结束复核源哈希。

```json
{"dataset_root":"/abs/v9","old_dataset_root":"/abs/v8","new_dataset_root":"/abs/v9","calendar":"/abs/v9/inputs/market_data/calendars/multiyear.json","route":"B_no_group","fold":"dev1","arm":"new_common","allowed_keys":"/abs/common_keys.parquet","fit_date_floor":"2026-03-02","incumbent_summary":"/abs/v8/round_10/summary.json","actual_manifest":"/abs/actual_manifest.json","paired_audit":"/abs/v9/paired_2026_audit.json","backlog":"/abs/R03/BACKLOG.md","matrix":"/abs/R03/MATRIX.md","seed":3566}
```

示例日期只在共同键最早日确为该日时有效；路径均须换成实际不可变快照。第三臂需把 `arm` 改为 `new_expanded_fit`，并填已登记的更早拟合下界。

运行示例：`python -m research.after_open_3d5pct.precision_v9.baseline --plan /absolute/plan.json --output /absolute/new-run`。`fold:"diagnostic"` 还需 `--allow-diagnostic` 与同臂同源的 `dev1_result`、`dev2_result`，结果标 exposed。所有新输出使用新目录，不写源快照；不产生正式达标结论。

20 项 data/baseline 单测通过，含跨年官方无信号日分母、purge、先验与监督分离、配对拒绝、源哈希篡改、specialize 有效支持及旧 fit 接口替身的完整输出回放。源码清单补入 `focus_v8/prepare.py`（旧 `run.py` 实际导入的 `write`）；其余静态研究模块 import 链均在源码 hash/快照清单内。

冻结 v8 工程复放位于 `runs/precision_v9_baseline_engine_smoke_20260927/`。拟合前已写顶层和各路线 `registration.json`，`actual_manifest.json` 的状态仅为 `engineering_validation_only`，旧新数据均指向同一冻结 v8 快照，14,424 行全量共同键、自配对结果及日历和三份数据文件 SHA256 已登记。B_no_group/dev1 与 C_no_group/dev1 均按旧 `round_10/summary.json` 的 incumbent recipe、seed 3566 真实拟合；四段按 `sample_id` 与各自 `incumbent_paths[0]` 对齐，fit/tune/cal/eval 分别为 3064/1045/936/1975 行，九目标 raw 与校准后 p 的最大绝对误差均为 **0.0**（门限 1e-7）。B 的有效 fit 为 3064 行/29 日；C specialize 后为 **609 行/29 日**，不是全池 3064 行。两路线运行结束与顶层复核的登记源 SHA 均不变。逐段数值、哈希与支持量在 `comparison.json`。

这是旧输入自复放的工程验证，不是 R03 新历史数据验收或研究候选；R03 采集和实际 manifest 尚未完成，因此没有运行 R03 三臂，也不能据此评价 90% 精度及周供给目标。
