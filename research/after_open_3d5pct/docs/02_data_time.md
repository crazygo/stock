# 02 · 数据和时间契约

## 现有数据与可复用边界（2026-09-25 本地检查）

| 资源 | 可以复用 | 不能据此宣称 |
|---|---|---|
| `market_data/us_60m/**`，Parquet / JSON.gz | 小时级价格路径、覆盖审计 | 已具备 5m 人工入场、精确 VWAP 或完整 PIT 数据 |
| `analysis/qqq_constituents.json` | 待核验的当前候选名单 | 历史 QQQ 成分；该文件只有 source/count/tickers，没有 announced/effective/available 时间，并混有 SPY/QQQ/DIA |
| `scripts/fetch_market_data.py` | 数据获取流程参考 | 可在训练时自动联网；已有读取器含拉取能力，应与训练 IO 分离 |
| `analysis/uptrend_path_research/spec_v1.md` | 1170 分钟期限、MFE/MAE、因果思想 | 旧实现可无条件直接导入；其时间戳/覆盖与本协议须重新验证 |
| clustering / today watch 的达成率 | 事后结果研究、发现假设 | 当天实时特征、可用模型概率 |

上述清单是输入盘点，不是逐股完整性认证。未更改既有 WIP。新训练数据应从显式 manifest 加载，不 glob 全目录后悄悄选最新版本。

## 三种时间分清

- `feature_cutoff_at`：如 11:30:00 ET，允许使用的事件/bar 结束边界。
- `decision_at`：默认 cutoff + 30 秒，模拟特征到齐及结果可见时间。
- `available_at`：该条信息真正可取得的最早时刻。特征同时要求 `end/event <= cutoff`、`available_at <= decision`。

例：11:30 完成的 bar 在 11:30:01 可用，可以进入 11:30:30 的结果；11:31 才到达的 bar 不能。11:35 的未来入场价只属于标签和回放执行，不能成为 11:30 模型输入。

内部必须携带时区，落盘推荐 UTC ISO-8601，同时保留 ET `session_date`。不能用上海日期分组，也不能全年固定 UTC-4 或 UTC-5。源 `time_key` 的起止语义须逐供应商验证；旧项目认为已有 Futu 60m 归档是美东结束标签，不得推广到其他粒度/提供方。

## 交易日历与数据覆盖

真实流程必须使用带来源、版本、哈希的交易所 session 表：`session_date, open_at, close_at`，覆盖特征回看、入场和未来完整标签窗口。半日按实际交易分钟累计；休市不计时。禁止从“某只股票有 bar 的日期”推断日历，停牌/缺数据不等于休市。

`timeaxis.py` 只验证传入日历的排序与算术；不能发现调用者漏给一个交易日。其生产正确性由 M1 的官方日历对账负责。`smoke.fixture()` 的工作日表专供合成验证，禁止拿去训练真实数据。

主入场/标签粒度为 5m。窗口内每个预期区间都必须存在且唯一，边界跨 bar 时必须用更细粒度，不得把窗外 high/low 计入。缺条、重复、混粒度返回不可判定。停牌、退市、拆股等另需状态记录；不能前向填价格后当作完整市场路径。

## 最小生产数据表（M1 实现；当前 Python 仅实现其核心 bar/label/sample）

| 表 | 必须字段及规则 |
|---|---|
| bars | `symbol,start_at,end_at,available_at,OHLCV,price_basis,provider,source_snapshot_id`；可选 turnover/真实 VWAP，说明单位与覆盖 |
| sessions | `session_date,open_at,close_at,calendar_version`；官方日历来源与完整日期覆盖 |
| universe_membership | `symbol,effective_from,effective_to,announced_at,available_at,source,snapshot_id,role`；半开生效区间，role=candidate/benchmark |
| events | `symbol,event_at,published_at,available_at,vintage,source`；修订不覆盖首版历史 |
| feature_rows | `sample_id,symbol,session_date,feature_cutoff_at,decision_at,feature_schema_version,source_ids,values,missing_reasons,max_source_available_at` |
| outcomes | `sample_id,entry_at,entry_price_proxy,entry_policy,price_basis,label_end_at,label_available_at,status,reason,hit,mfe,mae,hit_time_interval` |
| predictions | `sample_id,model_id,generated_at,feature_cutoff_at,reference_price,reference_price_at,p_hit_5pct_3d,data_quality,signal_age_seconds,expires_at` |

特征与 outcomes 分表。训练器显式按 feature allowlist join，不能“删除几个 label 列，剩下全喂模型”。入场价、MFE、MAE、最终 cohort 结果、parent_episode_id 都不在 allowlist。

## 股票池与价格基准

历史股票池必须同时满足成员生效和当时已知。只有今天静态名单时，模式记录为 `current_universe_retrospective`，只能探索，禁止标记独立通过；不能凭空构造历史 `available_at`。v1 真实训练要求 PIT 股票池，适配器在输入不满足时失败。

拆股/分红的价格和 volume 必须一致：保存原始价格及公司行动版本，或证明某种调整尺度在决策时可复现。禁止原价 Open 配前复权 High、也禁止用后来修订的前复权路径改变历史绝对特征。固定输入快照与归一化策略后才做追加未来不变测试。

历史归档通常没有真实到达时刻。若用 `bar_end + 固定延迟` 模拟，必须写 `availability_quality=assumed`，做延迟敏感性，并在前向采集真实时间；不得伪称已经验证实时可用。

真实 VWAP 需要正确且同尺度的成交额/成交量或逐笔依据，不能把 HLC3 当 VWAP。基准/行业代理也要有相同时间的可用数据；不能用 QQQ 全天收益代替 11:30 收益。

## 外部核对来源

- [Nasdaq 交易时间与假日安排](https://www.nasdaq.com/market-activity/stock-market-holiday-schedule)：正常时段和提前收市的官方核对入口。
- [scikit-learn 数据泄漏说明](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage)：训练内拟合预处理；库不能替代本项目的日期分组和标签隔离。

每次真实数据集保存源文件 SHA-256、来源/拉取时间、时区、复权、粒度、范围、缺失与修订策略；这些来源链接不是已经导入生产日历的证明。
