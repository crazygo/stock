# stock_traits_daily_v1 — stock-data-backfill → stock-report-refresh 最小可用实现

依据：`stock-data-backfill` / `stock-report-refresh` 两个 skill、数据交接契约 v1、`five_traits_v2` 公式参考。
与旧实验目录完全隔离：只**读取**旧输入（`market_data/us_5m`、R06 特别关注快照、SEC 缓存、官方日历），
写入只发生在下面列出的新根目录。不上传 R2、不 push、不建定时任务、不读写交易或自选（OpenD 只读报价上下文）。

## 命令

```bash
# 所有命令在仓库根目录执行；路径均相对仓库根（common.REPO），不依赖机器特定目录。
PY=python3                                    # 任意装有 pandas/pyarrow（OpenD 兜底另需 futu-api）的解释器，如 box 的 venv 或 Mac 的 /opt/homebrew/bin/python3
# 1) 数据层：发布不可变 manifest（cutoff=最近已收盘美股交易日）
$PY analysis/stock_traits_daily_v1/backfill.py --cutoff auto --task init \
    [--opend-host <Mac 地址> --opend-port 11111] [--symbols AAPL NVDA] [--budget-seconds 240] [--no-r2] [--no-opend]
# 2) 报表层：五标签快照 + 账本 + 页面（默认读取 .cache/stock_data_v1/current.json）
$PY analysis/stock_traits_daily_v1/refresh.py [--manifest <path>] [--evaluation-cutoff 2026-10-06] \
    [--mode live|historical_reconstruction] [--stale-policy label|unknown]
# 3) 仅重新渲染（不增加标签点）
$PY analysis/stock_traits_daily_v1/render.py
# 验收 fixture（合成数据，无网络）
$PY -m unittest discover -s analysis/stock_traits_daily_v1/tests -v
# 无头 Chrome 交互检查（地图→点证券→时间图→日期详情→14/30/60→关闭；桌面+390px 窄屏）
node --experimental-websocket analysis/stock_traits_daily_v1/tests/browser_check.js "$(readlink -f .cache/stock_report_v1/current)/index.html" /tmp/out
```

## 与其它管线共用机器时（如 Ray 的 Mac）

默认根目录 `.cache/stock_data_v1` 可能已被别的管线占用（Mac 上已有 `recovery_*` 的 QFQ manifest）。
backfill/refresh 在写任何文件前检查 `current.json` 是否为本实现所写，不是则拒绝运行。此时显式指定独立根目录：

```bash
python3 analysis/stock_traits_daily_v1/backfill.py --cutoff auto --task init \
    --data-root .cache/stock_traits_daily_v1/data --capture-root market_data/stock_traits_daily_v1/captures
python3 analysis/stock_traits_daily_v1/refresh.py --data-root .cache/stock_traits_daily_v1/data \
    --report-root .cache/stock_traits_daily_v1/report
```
（原始 capture 目录仅通过 `.git/info/exclude` 本地忽略，不改 `.gitignore`。）

## 产物位置（全部被 .gitignore 忽略）

| 内容 | 路径 |
| --- | --- |
| 日线 capture（不可变，只读） | `market_data/stock_data_v1/captures/<capture_id>/daily/<US.SYM>.parquet`（ZSTD 7） |
| OpenD 兜底原始 5m | `market_data/stock_data_v1/captures/<capture_id>/opend_5m/` |
| 数据 manifest / plan | `.cache/stock_data_v1/runs/<data_run_id>/{manifest,plan}.json`（只读） |
| 数据 current 指针 / 尝试账本 | `.cache/stock_data_v1/current.json`（os.replace 原子替换）/ `ledger.jsonl` |
| R2 读缓存 | `.cache/stock_data_v1/r2_cache/` |
| 评估快照 | `.cache/stock_report_v1/snapshots/<report_run_id>/snapshot.json`（O_EXCL 写入、只读） |
| 报表账本 | `.cache/stock_report_v1/ledger.jsonl`（refresh computed / reused、render published / failed） |
| 页面 | `.cache/stock_report_v1/pages/<render_id>/index.html`；`current` 为原子替换的相对 symlink |

发布顺序：写 `.tmp-*` 目录 → 回读校验（内容哈希、引用文件 SHA-256、无未来数据）→ `rename` 成最终目录 → 原子更新 current。
失败不替换已有版本；预算耗尽的证券沿用上一版本（`carried_forward`）。数据与报表各有 `flock` 防重入锁。

## 口径

- 行情：Futu 5m、`autype=NONE`（未复权 USD）、仅常规盘；日收盘 = 结束时刻等于官方日历 `close_at_et` 的 5m bar 收盘。
  不与 QFQ 60m 或 Massive 日线归档拼接。读取顺序 本地 `market_data/us_5m` → R2 `us_5m/` → OpenD（可配置 host/port，可终止子进程）。
  R2 对象 `Last-Modified` 早于首个缺失交易日收盘时判定为过期（不可能包含缺口），不下载、继续兜底。
- 未复权的公司行动风险：R2 `corporate_actions/` 有数据时读取拆合股事件；另按收盘比值检测疑似拆股 / 单日 |r|>ln2，
  记为 `return_breaks`，五标签不跨这些日期构造收益。
- 完整性：按官方日历区分休市、上市前（`pre_history`）、内部缺口（`missing_intervals`）、尾部过期（`stale`）；
  `complete_through` 为连续完整截止。零成交量只标记不删除。
- 五标签：`traits.py` 按 five-traits-v2 实现（G120 / V,M,J 60 / S 21 个日点 + 256 次块重采样，种子规则见 `PARAMS`）。
  价格未到评估截止时默认 `--stale-policy label`：用最近完整收盘计算并显示原价格截止，同时生成 `backfill_requests`。
- 新闻 / 估值 / 业务评级：未接入（`not_connected`），不生成综合分数。SEC 年度财务缓存仅登记为“沿用旧缓存”。
- 幂等键 = 模式 + 评估截止 + 成员哈希 + 输入内容哈希（含 return_breaks）+ 算法参数；相同键只记 `reused` 尝试。
  同一截止的新输入版本生成新快照并 `supersedes` 旧快照，旧快照不变；页面每个（模式, 截止）只画最新版本。

## QFQ 日线模式（`--price-basis qfq_daily`）与 AI 价值链股票池

```bash
python analysis/stock_traits_daily_v1/backfill.py --cutoff 2026-10-06 --task init \
  --universe ai_basket --grades A+ A --price-basis qfq_daily \
  --qfq-reuse-dir market_data/trend_quadrant_v1/daily \
  --quota-max-new 260 --quota-min-remaining 400 --budget-seconds 3600 \
  --data-root .cache/stock_traits_daily_v1/ai_basket_AplusA/data \
  --capture-root market_data/stock_traits_daily_v1/ai_basket_AplusA/captures
python analysis/stock_traits_daily_v1/refresh.py \
  --data-root .cache/stock_traits_daily_v1/ai_basket_AplusA/data \
  --report-root .cache/stock_traits_daily_v1/ai_basket_AplusA/report
```

- 数据层：OpenD `request_history_kline` K_DAY、autype QFQ（`opend_worker.py kline_day`），一次请求覆盖整段窗口（end = 截止日）。
- **每只只用一个来源、一种复权口径**：要么本次新拉的一份 OpenD QFQ 响应，要么一份通过检查的已有文件；不同来源/响应绝不拼接（`finalize_dataset` 有硬校验）。
- 复用（只读）候选：`--qfq-reuse-dir` 下的 `<security_id>.parquet`。只在以下全部满足时可信：末日 = 截止日、文件写入时间晚于截止日收盘、窗口内无缺口、`last_close[t]` 与 `close[t-1]` 一致、没有拆股比例式跳变。
- 复权基准变化（rebase）：QFQ 在分红/拆股后会改写历史（Futu 可能是线性 `price*A+B`）。默认策略 `verify_free`：股票已在 7 天额度窗口内（重拉不再计数）就重拉，并与所有复用候选逐日比较 OHLC，差异记为 `rebase_detected_vs_reuse` 并采用新拉；不在窗口内且候选可信则复用（不耗额度）；否则新拉（受额度预算限制）。
- 额度预算：开跑前只读查询 `get_history_kl_quota(get_detail=True)`。计划新增 > `--quota-max-new` 或剩余会低于 `--quota-min-remaining` 时，默认 `--quota-on-exceed abort` 在任何拉取前停止（ledger 记 failed + plan_quota）；`cap` 则拉到上限后其余标 `skipped_quota_budget`。结束后再查一次额度写入 manifest `resources.history_kl_quota`。请求间隔 ≥ 1 秒（≤ 30 次/30 秒，低于 60 次/30 秒限制）。
- QFQ 序列上 `split_ratio` 跳变视为复权异常并成为 `return_break`；`extreme_move` 只做标记；不再套用 R2 拆股事件。
- 股票池 `--universe ai_basket`：`analysis/ai_value_chain_map_v1/ai_basket_members.json` × `analysis/ai_value_chain_map_v1/quality/screen_current.json` 按 `--grades` 筛选；成员带 `tags`（grade / verified_grade / tier / value_groups / primary_group）。manifest/快照记录两个文件的路径、sha256、built_at、screen run_id、as_of。港股暂无交易日历 → out_of_scope（理由写明）；日股同样 out_of_scope。
- 页面：标题取股票池标签；筛选支持类别、档位、16 个价值链分组（任一所属分组勾选即显示，含全选/全不选）、代码搜索、代码标签开关（>80 点默认关）。
- 断点续跑：`--reuse-opend-raw-dir <上次 .tmp-cap-*/opend_day_qfq>` 复制已拉的原始响应，不重复请求。

## 已知限制

- 旧 `analysis/ai_trend_quadrant_v1|v2/`、`ai_value_chain_map_v1/` 不在本 checkout：没有复用或验证原报表 / Pugh 模型。
- 当前只覆盖美股；HK 成员标为不支持的市场。官方日历文件只到 2026-10-15，需要延长后才能继续使用 `--cutoff auto`。
- OpenD 兜底已在 Ray 的 Mac（127.0.0.1:11111，futu 10.11）真实跑通（2026-10-07，data-20261007T083721Z-e51e56）：
  只用行情上下文（get_global_state / get_history_kl_quota / get_user_security_group / get_user_security / request_history_kline），
  常规时段 5m、页间 0.6 s、调用间 ≥1 s；161 个交易日一只约 13 页、30–45 s。futu 会往 stdout 打日志，worker 结果行带 `@@STDV1_RESULT@@` 前缀。
- 被打断的抓取可用 `--reuse-opend-raw-dir <.tmp-cap-*/opend_5m>` 续跑（原始文件复制进新 capture，不重复请求）。
- 本地 `market_data/us_5m` 中部分文件是 OpenD 原始列格式（无 session_date），按读取错误记录并转下一层，不猜其复权口径。
