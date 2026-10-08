# 两个认可页面的统一日更

保留股票历史形态描述 v3 和 AI 趋势象限 v2 的完整页面、公式及交互，共用一个版本化日线清单。一次运行完成成员查询、Local → R2 → OpenD 补齐、两套计算、浏览器验收和同步发布。

```bash
python3 analysis/stock_review_pair_v1/run.py
```

仍使用原入口：

- `http://127.0.0.1:8768/analysis/stock_behavior_description_v3/index.html`
- `http://127.0.0.1:8768/analysis/ai_trend_quadrant_v2/index.html`

页面顶部显示共同数据批次、各市场完整交易日目标和实际到达比例，并可跳到另一页面。打开 HTML 本身不取新数据。此命令只执行一次，不建立调度。

## 数据与发布流程

1. 实际查询自选分组名、特别关注与两个券商实体的持仓成员，合并既有 AI A+/A 池。仅保存持仓成员标记；数量不写死。失败沿用原成员并保留其原时间。
2. 按官方日历选择已完成的交易日，分别登记美股与港股截止。盘中当日不能作为完整日线。日历超出已核对范围明确失败，不猜节假日。
3. 检查两年 QFQ 日线的尾部和内部缺口。依次读取本地、同口径 R2、OpenD；每次捕获使用完整一致复权序列，不拼接新旧 QFQ。原始行情保存在被忽略的缓存目录，不自动上传 R2。
4. 固定 `inputs/data_manifest.json`，两页从同一个标准 OHLC 输入派生。五标签窗口仍为 120/60/60/60/140 次日变化；历史形态规则仍为 v3。图的共同右端与逐市场可评分截止分开，避免香港当日收盘导致全部美股被判过期。
5. 校验输入哈希、两页 OHLC 一致、截止后数据排除、脚本语法，实际浏览器检查日周图、筛选、详情审查保存、地图时间谱、手机及离线阅读。通过后一次替换共同 `current` 指针；失败保存错误，旧页面继续使用。

缺日、短历史、日历未核验、零变化及过期按指标分别保留未知，不填零。两年行情覆盖不足不等于短窗口指标全部无法评分；逐股数据覆盖与五标签状态分别保存。

默认采集总预算 900 秒，OpenD 单请求 15 秒，R2 单请求最多 5 秒，串行限速；最多尝试 100 个新配额证券，保留 400 个剩余配额。预算结束停止新增请求并交付部分覆盖。OpenD 在可终止子进程中运行，并发更新使用本地文件锁。

## 批次目录

默认存储根 `.cache/stock_review_pair_v1/`，已由仓库原有规则忽略：

```text
runs/YYYY-MM-DD__run-<uuid>/
  inputs/                    # 数据清单；成员、日历、副证据的共用版本引用
  results/                   # 两套独立结果及幂等评估快照
  pages/                     # 两页和历史形态详情；数据与脚本全部内嵌
  validation/                # 接受状态、浏览器检查及截图
  repro/workspace/           # 本次实际使用的最小源码快照，无凭证
  repro/replay.sh             # 用包内源码与输入断网重算，写入另一个目录
  repro/environment.json
  run.json
  manifest.json
  checksums.json
  checksums.sha256
```

每次尝试用独立 UUID，已生成文件封存为只读。`data_run_id` 由输入内容与范围确定；`report_run_id` 由数据及算法版本确定，同输入重跑复用评估，不制造新的每日观察点。时间谱中的过去点仍明确是历史回算。

底层数据与 run 分开，存放在 `.cache/stock_review_pair_v1/data/`：

- `raw/<证券>/<文件SHA>.parquet`：原始 QFQ 捕获，同内容只保存一份。
- `daily/<证券>/<dataset_id>.json`：标准化 OHLCV 版本，计算时读取；run 清单仅记录引用与哈希，不再内嵌整套 K 线数组。
- `facts/<文件SHA>.json`：成员、日历、财务、评级、产业链及小时证据；run 中的小型相对链接引用这些不可变版本。

新 run 内不保存 Parquet；同版本重跑不会新增共用对象。历史数据修订新增版本，原版本与旧 run 不改写。首次复用旧 run 时会把其输入登记到共用库，不联网；旧目录里的历史副本暂时保留。页面与结果文件为了独立阅读仍内嵌展示数据，它们是派生产物。

原入口的三个 HTML 变为共同指针的别名，原文件精确备份到 `legacy-/`；源码、旧实验 JSON 和原始行情不覆盖。若别名被其他任务修改，发布会拒绝覆盖。显式历史候选不能使已发布市场截止倒退。

`latest_attempt.json` 表示最近尝试，`last_success.json` 和 `current` 表示最后成功发布，两者不能混用。缓存目录包含发布所需文件，清理它之前须保留运行包及原 HTML 备份。

## 离线、审查与复算

只生成候选，原页面保持当前版本：

```bash
python3 analysis/stock_review_pair_v1/run.py --no-publish
```

明确使用缓存、重建历史目标：

```bash
python3 analysis/stock_review_pair_v1/run.py --offline --cutoff 2026-10-06 --no-publish
```

从指定完整或失败运行的已完成输入重算，不再联网取数：

```bash
python3 analysis/stock_review_pair_v1/run.py --from-run /absolute/path/to/run --no-publish
```

纯计算审查：关闭五标签计算缓存，重新计算全部指标，与指定批次的 OHLCV、覆盖、五标签、历史时间谱、不确定性及完整形态证据逐股比较。计算结果与原缓存不一致会报错；缺失状态仍参与核对。

```bash
python3 analysis/stock_review_pair_v1/run.py --from-run .cache/stock_review_pair_v1/current --recompute --no-publish
```

查看本次 `validation/recompute_review.html`，可筛选特别关注、搜索证券、查看差异，并打开该次两套日周图页面。`recompute_review.json` 保存真实存储占用、共用对象新增/复用、实际新算与缓存计数。重算沿用原信息和行情截止，不查询成员或行情，不产生新评估点。

运行目录复制后，其内嵌页面仍可阅读；输入引用依赖共用库。要独立分享或离线复算，显式导出：

```bash
python3 analysis/stock_review_pair_v1/export.py --run .cache/stock_review_pair_v1/current --output /absolute/path/to/new-offline-package
```

导出才复制实际引用的数据；原数据清单保持原字节与哈希，另用 `data_location.json` 定位包内副本。导出的目录无目录外数据依赖。用导出包内源码复算：

```bash
sh /absolute/path/to/run/repro/replay.sh /absolute/path/to/new-output-store
```

依赖 Python 3（pandas、numpy、pyarrow）、Node.js、Playwright 和 Chromium；在线补齐另需 futu-api、OpenD，以及仓库既有 R2 配置。离线复算无需 OpenD/R2/券商配置。`--skip-browser` 仅供候选排查，强制搭配 `--no-publish`。

契约测试：

```bash
python3 -m unittest discover -s analysis/stock_review_pair_v1 -p 'test_*.py' -v
```

## 本次接入边界

本入口是 [一次更新 SOP](../ai_value_chain_map_v1/ONE_RUN_UPDATE_SOP.md) 的“双页面日线更新”配置。当前会刷新成员及日线；财务、业务评级、产业链和小时证据沿用实际截至日缓存，新闻明确未接入。它没有完成完整日报 SOP 的新闻核查、精华消息、事件跟踪、跨机器包合并或自动调度，也不生成未来胜率与操作建议。

首次在线运行仍接受现有成员、质量、业务及日历基础资料的依赖，不能宣称干净 Git 克隆已包含它们。显式导出包复制实际输入并可独立重算。美股官方日历覆盖到 2026-12-31；港股目前核对 2026 年，此前真实 K 线可显示但对应历史日历保持未知。
