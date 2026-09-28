# R03 native artifact limited interface · code review

2026-09-27。实施依据：`REGISTRATION_DRAFT.md` SHA-256 `74aa72e1cd346fbb428d04790eb34147bdfa5860d5672f9704b9aeca653f3711`、`ADOPTION.md` SHA-256 `01bd76f0806883c8c52b87f2838261cb7046f17b3ff51cc2338b1429707f9647`。本轮仅完成接口、离线反例和两个**既有**无群开发模型的事后工程封装；没有 R03 数据采用、30 套新拟合、正式因果群模型、发布或 G1/G2/G3 验收。

## 实施与冻结源码

- `forward/schema.py` 为生产/测试共用的五路线 schema，按 route/recipe/peer 语义计算宽度、列序、support_rule。带群 old_common 为 `legacy_static_only`，新两臂只在真实预拟合来源合同成立时允许 `v9_causal_history_end_v1`；无群为 `excluded`。
- `native_artifact.py` 的 `register_prefit` 将完整有序 split keys、有效训练 IDs/权重身份、全候选成熟先验、old/new/selected dataset 三文件、实际接受的 `actual_manifest`、计划/配对/日历/采用文件与源码字节在 fit 前登记。带群新臂另外验证实际冻结 builder、输入 source_hashes、完整旧池 108 candidate 与 QQQ benchmark 对应、稀疏合法 membership 的类别和历史截止、官方 session/版本时钟。未分类不补群。
- `baseline.py` 只在 `r03_baseline` 上连接 pre-fit 合同及验真，原 fit/seed/recipe/阈值逻辑未改。`export_route` 在完成后只读原 run，验证完整 split 顺序、拟合监督、prior history、模型/校准/selection，再在新目录导出。`verify_native_manifest` 同时绑定 manifest、原 registration、完整身份、pre-fit causal 对象、实际 dataset、recipe/阈值/校准、训练快照及当前服务依赖；`inference.py` 在 load 和 predict 后重验，调用者拼接 loaded dict 或改内存 manifest 不能绕过。
- `verify_in_child` 独立进程先 import inference，再载入 torch/原模型；要求 sealed cache、所属 run 一致，保存命令、环境、stdout/stderr、终态、前后来源 SHA。原 eval 全序列按原批次重放，固定 AMD 单样本以同一 singleton batch 比较九 raw/p；不以跨批差作为同批门禁。当前历史 fixture 一律不可发布。

完成代码 SHA-256：`native_artifact.py` `10c8601628357611549035f284b53c905e0ea7c88ebe314c95ae520936d1d269`；`forward/schema.py` `a7d72b5ad2acffddd4af755d2592032086f7533e5eab3b133b2137d527eb23b2`；`forward/inference.py` `e5407c69a9fe1160d41769232e1791051774808b35ebdc78ba05b812c0f584d0`；`baseline.py` `a4923d9f4741f62448bd10834769f06465b821a6ca57fa082a9820850775e47b`；`test_native_artifact.py` `a5d6c120e28cd5245e2efafdb5c528f580d198207d3b98663bbd86d61e7f18fd`；`test_baseline.py` `c6a3b999f51d4134257101ed50a868a529c82abd891f63b6ab89516e22aab18d`。

## 回归和实际工程对照

运行环境为 `research/after_open_3d5pct/.venv/bin/python`。命令：

```bash
research/after_open_3d5pct/.venv/bin/python -m unittest research.after_open_3d5pct.precision_v9.test_native_artifact research.after_open_3d5pct.precision_v9.test_baseline research.after_open_3d5pct.precision_v9.forward.test_inference -q
```

结果 **29/29**。反例覆盖合法重算 SHA 后的不同 membership/身份、108 池替换、空日历/缺版本时钟/非交易日 history_end、recipe/阈值/校准/模型/源码变更、post-load manifest 篡改、错误 child run、preflight 失败终态、真实 Parquet `np.ndarray` prior 列表规范化。模拟 sparse causal 正例仅验证接口，不是 R03 已接受数据。

持久工程目录：`research/after_open_3d5pct/runs/precision_v9_20260927/R03_native_artifact_engineering_v1/`。真实执行入口：

```bash
research/after_open_3d5pct/.venv/bin/python -m research.after_open_3d5pct.runs.precision_v9_20260927.R03_native_artifact_engineering_v1.run_posthoc
```

脚本 SHA `43bda5f32173e970fbaa88e0a6a17473f1daca08843e19b0783dc3f5c4ad98e2`；`engineering_summary.json` SHA `8080b0632c324cdacd985fe355ff2bcfede13ca80fe89b86ab440d51387ddd0b`。首次以脚本文件路径直接运行时在 import 前因 `research` 模块搜索路径退出，没有产生导出；随后使用上述 `-m` 命令唯一成功执行。两条路线均是旧 `precision_v9_baseline_engine_smoke_20260927` 的 dev1 **posthoc engineering_only**：B_no_group、C_no_group 各 1975 条原 eval 的九 raw/p 最大误差均 0，固定 `AMD:2026-06-01:11:30:v6` singleton 最大误差均 0。B child `child_final_status.json` SHA `ae586db1313f245d3848e9075900eb8ff468578f439b6aabce2bf363d745a90a`，C child SHA `a29b0918754e3a175a9325fe3532055dc49b8470b9726fceffa8ca81c287ae23`；退出码均 0、source_drift 均空、G2/publication 均 false。总控 23 源前后相同；每个 child 的 `invocation.json` 保存完整 `--artifact/--run/--selection/--output` 命令、cwd/Python/platform 和 OMP/OPENBLAS/MKL=1，stdout/stderr 原字节及 SHA、68/76 个来源前后记录和阶段耗时均在各自 child 目录。

## 仍需另行登记

真实 R03 `actual_manifest=accepted_for_fit`、官方日历/来源与全部 30 身份的 pre-fit 计划尚未形成；本轮未运行 `export_route` 的真实正式分支，也未从真实新臂行情构建代表 feature-only 对照。新群模型只有真实训练 provenance、独立数值和代表输入通过后才能谈 G1；归档无真实 receipt 不构成 G2，亦无任何 G3 发布或前向效果证据。旧 engine 的两个零差只证明这两个既有无群权重的当前工程接口兼容。
