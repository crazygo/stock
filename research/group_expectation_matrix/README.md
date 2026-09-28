# 群 × 期望研究矩阵

用户的对象模型直接落地为：**分组策略 → 定义版本 → 每周成员版本 → 群；群 × 达成期望 → 历史统计**。支持一只股票多个行业/价值链标签，以及同时拥有3周、3月、6月趋势身份。

首版方向是自包含的灰阶HTML线框工作台：矩阵总览、单元格证据、群与周版本、独立模型洞察、数据口径。用户已经明确二维表结构，因此只做这一种布局；品牌、配色、交易界面留待后续。无自动交易、无自动定时任务、无行情或数据上传。

## 已完成的真实输出

最新运行目录：[outputs/20260925_v5](outputs/20260925_v5/)。本次从本地 FutuD 自定义“ETF”分组冻结了30只证券，作为 `instrument_type=etf` 的候选证券实体接入同一个 `Group × Expectation` 数据模型；ETF 分组本身也是群实体。股票101只、ETF30只，共131个候选证券实体。已有基准身份的 QQQ、IGV、SOXX、SMH 同时保留 `benchmark` 角色。

- [ETF实体列表](outputs/20260925_v5/instruments.json)、[FutuD成员快照](outputs/20260925_v5/watchlist_snapshot.json)、[群定义](outputs/20260925_v5/groups.json)。
- [交互线框页](outputs/20260925_v5/index.html)、[矩阵](outputs/20260925_v5/matrix_all.json)、[完整数据包](outputs/20260925_v5/bundle.json)。
- [模型洞察](outputs/20260925_v5/MODEL_INSIGHTS.md)、[核验结果](outputs/20260925_v5/audit.json)、[新旧回归对账](outputs/20260925_v5/etf_entity_regression.json)。

只有 QQQ、IGV、SOXX、SMH 四只已有所需5分钟行情与公司行动表；其余26只在本地与R2都没有所需5分钟归档。本轮保留全部30个ETF实体与群成员关系，缺行情的结果标为 `missing / market_data_unavailable`，不会算作目标失败。FutuD当前分组在历史周次中是**当前快照回溯**。ETF和股票共池后全池基准口径改变，v4股票研究仍作为独立冻结版本。没有为补全这次实体接入而发起大量FutuD历史K线请求。

本次运行生成58个群、6种期望、39周版本范围。20项单元测试、38项统计核验、56项新旧回归检查及30条模型洞察证据对账通过；桌面与手机浏览器交互检查通过。原101只股票的110,898条历史标签与v4逐条一致。现有8768服务可直接打开 `/research/group_expectation_matrix/outputs/20260925_v5/index.html`。

之前完整行业父子群运行保留在 [outputs/20260925_v4](outputs/20260925_v4/)：

- [交互线框页](outputs/20260925_v4/index.html)、[全期矩阵](outputs/20260925_v4/matrix_all.json)、[全部周矩阵](outputs/20260925_v4/matrix.json)。
- [完整数据包](outputs/20260925_v4/bundle.json)、[群实体](outputs/20260925_v4/groups.json)、[父子关系](outputs/20260925_v4/group_relations.json)。
- [定义配置](config/industry_tags_v3.json)、[核验结果](outputs/20260925_v4/audit.json)、[新旧统计一致性](outputs/20260925_v4/parent_entity_regression.json)。
- [模型洞察](outputs/20260925_v4/MODEL_INSIGHTS.md)：保留并对账v3的细分分析，明确30细分与11父群的统计范围。

**父级本身是群实体。** 11个父群和30个子群都有独立 `group_id`，均生成成员、周版本归属、群×期望统计与明细。`parent_group_id` 指向真实父群，`child_group_ids` 与 `group_relations.json` 可双向遍历。页面将父群放在对应子群之前，仍平级展示；明细可在父群和子群之间跳转。

父群成员取子群成员的去重并集，逐股票日重聚合；不会相加子群样本数或平均子群达成率。例如MU同时属于DRAM和闪存，但在存储父群中只计一次。股票继续支持多标签；行业基准继续去重。

本版有41个行业群实体（11父群+30子群），加上其他分组，共57群×6期望=342个全期格，39周和273策略周版本，完整矩阵13,680格。候选101证券，行业覆盖45证券；父群61条、子群70条标签边，共131条，不能将边数当成股票数。

验证：18项单元测试、34项统计与关系核验、50项新旧版本回归检查通过。父群统计与原宽群一致，子群与非行业指标保持不变；存储父群3日目标成熟样本为720条，而子群相加为900条，确认父群已去重。

已有8768服务可访问 `/research/group_expectation_matrix/outputs/20260925_v4/index.html`。历史v2保留11宽群，v3保留30细分；本版v4将它们作为关联的群实体同时保留。旧配置及产物均保留。

## 复跑

从仓库根目录执行；沿用已经安装的研究Python环境（numpy、pandas、pyarrow）。

FutuD“ETF”分组变化时，先显式运行 `python -m research.group_expectation_matrix.sync_futud_etf --snapshot <新快照路径> --universe <新证券池路径>`，然后将两个新路径写入新配置副本。命令拒绝覆盖旧快照；每次运行保留观察时刻和完整成员，不会自动抓取行情或上传R2。

```bash
research/after_open_3d5pct/.venv/bin/python -m research.group_expectation_matrix.build \
  --config research/group_expectation_matrix/config/etf_group_v1.json \
  --output research/group_expectation_matrix/outputs/<新的运行目录>

research/after_open_3d5pct/.venv/bin/python -m research.group_expectation_matrix.audit \
  --run research/group_expectation_matrix/outputs/<新的运行目录>

research/after_open_3d5pct/.venv/bin/python -m research.group_expectation_matrix.render \
  --run research/group_expectation_matrix/outputs/<新的运行目录>

research/after_open_3d5pct/.venv/bin/python -m unittest research.group_expectation_matrix.test_build -v
```

计算命令拒绝覆盖既有运行目录。输出Parquet与大型制品全部Git忽略；代码和配置独立保留，不修改旧研究。

增加期望：在配置 `expectations` 中加入唯一ID、正整数交易日和目标收益；例如`{ "expectation_id": "d7_r6", "name": "7 日 +6%", "trading_days": 7, "target_return": 0.06 }`。代码会以7×390常规分钟计算，不按7根日线或自然日替代。

每周更新：复制并保存新配置，更新数据范围和 `evaluation_as_of`，显式运行到新目录。计算只读本地已有行情；FutuD自选成员快照由 `sync_futud_etf.py` 明确采集。新行情先遵守全仓 Local→R2→OpenD 的获取约定。策略定义变化须升级定义版本；行业新版本须更新 `industry_tags.json` 的来源与观察时间，不把复制旧标签伪称重新核验。每周版本ID由规则、周边界和实际成员事实内容寻址；相同前缀数据不会因追加未来数据而改写旧成员。

`build.py` 暴露独立函数：`classify_history`、`outcome_at`、`paired_baseline`、`describe`、`build`，可在其他研究中复用。默认一次生成整个历史周序列；没有安装后台调度。

## 结构化文件

| 文件 | 主键/关系 | 内容 |
|---|---|---|
| `strategies.json` | strategy_id | 策略定义版本、更新频率、参数 |
| `groups.json` | group_id → strategy_id | 群实体、parent_group_id、child_group_ids、ETF分组与行业语义 |
| `instruments.json` | security_id / symbol | 股票和ETF同型实体、角色、行情覆盖状态 |
| `watchlist_snapshot.json` | group_id / symbol | FutuD ETF组当前成员与观察时刻 |
| `group_relations.json` | parent_group_id × child_group_id | 指向实际群实体的父子关系 |
| `industry_tags.json` | taxonomy_version / refinement[] | 11父群及30子群、成员、来源与定义映射 |
| `versions.json` | version_id → strategy_id | 每周截止、生效区间、生成时刻、归属依据 |
| `memberships.json` | week_id × strategy_id × symbol | group_ids数组、version_id、状态、计算事实 |
| `expectations.json` | expectation_id | 涨幅目标与交易分钟期限 |
| `outcomes.parquet` | sample_id × expectation_id | 每个股票日的完整标签、风险、入场和窗口时间 |
| `matrix.json` | scope × group_id × expectation_id | 全历史周归属聚合及单周矩阵 |
| `stock_metrics.json` / `monthly_metrics.json` | 群 × 期望 × 股票/月 | 成员差异与时期稳定性 |
| `coverage.json` | symbol | 行情质量、重复、完整日、公司行动 |
| `manifest.json` | protocol_version / hashes | 输入、代码、运行配置指纹与证据边界 |
| `bundle.json` | 合集 | 可供其他HTML或LLM直接读取，不含逐bar行情 |

HTML数据内嵌，支持策略/股票/周版本筛选，指标切换，点击单元格，查看来源、成员、月份和逐股统计，导出当前矩阵或完整数据。大模型只读结果提供解释，没有生成或修改统计值；新运行的模型解释需重新分析并写入 `model_insights.json` 后再render，不能复制旧洞察冒充新结果。

## 证据边界

完整冻结口径见 [PROTOCOL.md](PROTOCOL.md)。这是一张历史描述矩阵，并非已训练预测策略。行业组是当前官方业务资料快照的历史回溯，未分类证券不代表没有行业。行情趋势等周成员只使用周前数据重建，但候选池仍是今天名单的回溯，历史到达时刻为假设。

每格目标触及与研究净评价、期末收益、入场相对全窗口MAE、尾损、达成耗时分别列出。风险未成为目标达成硬门槛，不设置中途止损。净评价采用触及固定目标/未触及期末估值和双边各6bp，不是实际账户或组合收益。

不同期限的完整成熟截止日不同，比较列时必须看日期范围。提供同日期市场基准、本策略可分组基准和连续日块探索区间；多重比较尚未校正。模型洞察中的高达成率不能解释为未来保证。
