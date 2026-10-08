# AI 股票产业群落与关注缺口

打开 [index.html](index.html)，或者访问现有静态服务：

`http://127.0.0.1:8768/analysis/ai_value_chain_map_v1/index.html`

本轮该现有 HTTP 服务请求超时，未重启或另开重复服务。离线 HTML 已完成浏览器验收，优先直接打开文件。

HTML 含 CSS、JavaScript 和数据，下载单个文件也可离线使用。默认打开产业地图。
点击产业群看成员和传导解释，点击股票查看业务资料与关联依据。
可按持仓、全部自选、特别关注、QQQ、自选 ETF 成分筛选，导出 CSV / JSON。

后续跟踪需求见 [AI 产业链股票跟踪流程规格](TRACKING_REQUIREMENTS.md)：记录 V 型深弹的动因与影响时间、四种上涨微观形态的每日分析和报告时效、阶梯中继持续跟踪，以及每日领涨 AI 临时池。该文为接入设计，新增能力的交付状态以实际运行及验收为准。

任务拆分及报告精华标准见 [AI 产业链股票研究与报告 SOP](TRACKING_SOP.md)：日粒度选股、分钟形态与三日观察、AI 新发现和复盘周报，以及每项任务的输入、触发、产物和验收。[现有多股票形态页面](../ai_trend_quadrant_v2/index.html) 已登记为日度报告产物；SOP 包含其数据准备、计算、页面生成与发布的十项每日动作，统一日更接入仍待验收。

交给 coder 执行一次更新，使用 [AI 产业链股票一次更新 SOP](ONE_RUN_UPDATE_SOP.md)。该文规定精华消息、独立报告、输入证据和复现材料的交付，以及日期加 run ID 的目录命名、独立分享、跨机器合并和后续复盘。

新增「全部 AI 相关」页，合并全部 1 / 2 / 3 级股票，包含供给、部署和 AI 赋能应用。
这份名单不按与六只参考股的价格相关性筛选；导出按钮始终导出全池 AI 名单，当前搜索不缩小导出范围。

## 本版覆盖（2026-10-06）

- 当前 OpenD 池读取时间：2026-10-06 02:19:08 北京时间。
- 20 个直接持仓证券，合并 Moomoo US 和富途证券真实账户，只保存名称和成员关系；不含股数、金额、账户号码或盈亏。
- 原始全部自选 180 条；排除行业列表等非证券条目后为 165 个自选证券。
- QQQ 发行人接口本次重新读取，101 只股票，业务日仍为 2026-10-02。现金和衍生品不当作股票。
- 去重已知证券 1,847 个，含 1,779 只股票和 68 个基金。仅 20 个基金有可用的美股成分快照；其中多数复用旧文件，未冒充本次更新。非美股直接自选/持仓保留，但其他基金的全球成分尚未完整展开。
- 1,776 / 1,779 只股票有行业名称；525 只具有基于业务资料的关联评级，未评级 1,254 只仍保留 null。其中本轮 Luna 新增/复核 447 个非空评级，原版业务证据继续保留；行业初筛不冒充 AI 业务核查。
- 全池有依据的 AI 相关名单共 458 只：直接业务 297、配套部署 113、间接传导 48。已核查但未建立本版路径 67 只。
- 直接业务路径中，250 个证券未持有、未自选，作为关注缺口展示；这个数是已知范围的下界，不是收益排名。GOOG / GOOGL 合并发行人后不制造“漏看公司”。持有 ETF 仍可能产生间接暴露。
- 180 只股票有本地完整日收益。核心参照共同末日为 2026-09-29 ET；个别数据至 09-30，不能标为当前行情。

## 可重复的 AI 质量评级

使用 `$ai-stock-rating` 重新评级并更新网页。技能入口在
[.agents/skills/ai-stock-rating/SKILL.md](../../.agents/skills/ai-stock-rating/SKILL.md)，
详细门槛在 [评级协议](../../.agents/skills/ai-stock-rating/references/protocol.md)。

质量等级与业务关联 3 / 2 / 1 分开。A+ 要求 AI 重要增长、规模商业兑现、优势证据、盈利与现金兑现及风险核查；
A 表示已商业兑现、财务稳健且风险已复核；A− 用于有依据但尚在发展的路径；
B 表示已核查的外围路径或已证实重大弱点，不代表公司整体差。关键资料不足保留待评级。
数量由证据决定，不按股票池固定比例划分。

逐股核查评级页默认筛选有效 A+ / A，可以切换 A−、B、待评级、过期或全部，并组合原来的股票池、产业群与搜索。
「全部 AI 相关」的全池导出仍包含所有已建立 AI 路径的股票，质量筛选不会删去其原始名单。
详情保留经营证据、财务期间、披露及取得时间、缺项、风险与复评触发条件；估值未评估。

每次执行会保存 `quality/current.json`、不可覆盖的 `quality/runs/<run_id>/snapshot.json`、CSV 及变化。
首次评级不会制造升级或降级记录。同一天同一证据重复执行应复用快照。
默认评级有效期30日，财报、商业化、竞争及资金状况变化需要提前复评。
页面本身离线展示，打开不会联网重评；目前没有设置定时自动化。

```bash
# 先依技能核查最新官方业务和财务资料，再填逐股 reviews_*.json。
# 以下程序提取本地可追溯 SEC 缓存；重跑不等于重新下载财报。
python3 analysis/ai_value_chain_map_v1/quality/prepare_financials.py --as-of YYYY-MM-DD
python3 analysis/ai_value_chain_map_v1/quality/rate.py --as-of YYYY-MM-DD --final
python3 analysis/ai_value_chain_map_v1/build.py
```

当前评级数量及未完成范围以页面质量评级页和 `quality/current.json` 为准。
评级是研究优先级规则，未经过未来收益验证，也不构成旧量化模型的买入资格。

2026-10-07 首轮在458只已有AI路径的股票上运行：A+ 6、A 11、A− 3、B 5、待评级433。
默认核心17只证券对应16家公司；GOOG与GOOGL保留两类证券，共用已核实的Alphabet业务证据。
仅25只达到本版完整评级门槛，待评级仍缺业务、风险、可比财务或新鲜资料，不能当作低质量名单。
首次快照为 `quality/runs/aiq_5f8a863c9b20edad/`，变化表为空；原股票池、业务分类及历史行情均保留。

## 低成本批量初评与价值链分布

根据用户要求，首页新增“批量初评”，将现有官方业务标签、行业和可比财务缓存组合成研究优先级，目标覆盖AI关联池至少80%。本轮分母是458只AI相关证券，80%至少367只；全池行业覆盖另列，不与初评混算。

初评规则 `ai_research_screen_v1` 独立于 `ai_quality_v1`：未完成逐股风险核查的初评最高为A，A+仅继承有效的已核查评级。每行标出“初评”或“已核查”，详情保留原来的核查等级及缺项。初评可用于决定研究顺序，不能直接称为已核查的优质公司。

每只股票有行业、具体业务环节、已有官方业务依据、价值传导路径、经营支持和待补动作。宽泛行业字段结合已核实业务标签显示细分方向，这不是正式GICS分类。价值群统计证券数量；跨群公司会重复进入各群，不能把群占比称为公司收入、利润或AI收入占比。

```bash
# 复用现有业务和财务资料；没有逐股联网搜索。
# 若刚更新成员/业务标签，先build刷新data.json输入；旧初评会自动失效。
python3 analysis/ai_value_chain_map_v1/quality/screen.py --as-of YYYY-MM-DD
python3 analysis/ai_value_chain_map_v1/build.py
```

独立输出为 `quality/screen_current.json`、CSV和 `quality/screen_runs/<run_id>/snapshot.json`。核查原始快照及股票池保留；相同输入重复执行不产生假历史。后续付费研究优先用于持仓／自选、关键价值环节、少量补证即可完成核查以及重要财报／融资变化。

2026-10-07批量初评覆盖458/458只证券（100%）：A+6、A303、A−87、B62。其中25只沿用有效的逐股核查等级，433只为业务研究初评；财务支持已知307只、未知151只，11只业务标签置信度较低并显式标记。全部458只都有行业、业务群、具体业务依据和下一步研究动作。初评快照为 `screen_92f389e5c72d0f48`；不完整核查中的商业化占位值不会被误当成已证实的早期业务。

## 两种分群

**业务分类 / 规则标签**：16 个可重叠群，覆盖设计制造、算力、存储、互连、光通信、服务器、电力散热、能源电网、云、企业数据、安全、应用、医疗与端侧等。
`taxonomy.py` 保存原版规则和逐股证据；`reviewed_labels.json` 保存补充核查、行业来源及产业链角色校正。
本轮先尝试 agyd；三个运行反复读取文件而未形成有效标签，已停止。逐股官方业务核查实际由用户授权的 `gpt-6-luna` 完成。
发行人行业表与既有 SEC SIC 文件复用作低成本行业初筛，不能单独赋予 AI 等级。资料取得时间、持仓业务日期与本次整理时间分别保存；部分 SEC 旧文件未保留取得时间，仍显示未保留。
等级 3 / 2 / 1 / 0 表示直接路径、配套部署、间接假设、本版未建立路径。
未核实为 `null`。群代表等级取已评级成员的下中位数，同时显示评级覆盖和全部成员的等级构成。
这是分析推断，未声称 AI 收入占比、利润弹性、因果或上涨概率。

**历史价格聚类 / Agglomerative hierarchical clustering**：固定 90 日历天窗口
2026-07-02–09-29，先把各股常规盘收益对 QQQ 回归（含截距），再按残差的两两 Pearson 相关距离
`sqrt(2*(1-rho))`、average linkage，探索性地分为预设 6 群。
两两至少 40 个共同完整交易日；缺失相关不填补。6 群未通过最优群数或样本外稳定性验收。
页面改日期只重算相关系数，不偷偷改变固定价格分群。

单股 AI 参考相关：NVDA、AMD、AVGO、ARM、MRVL、MU 等权，排除股票自身；每日至少 4 只参照可用。
先对齐完整日期，再同时报告原始相关和去 QQQ 的残差相关，至少 20 个共同交易日。
两者均是历史描述，不是 AI 因果、预测置信度或交易信号。

日收益 = `log(RTH Close / 09:30 Open)`。60m 档案须具备官方日历要求的全部常规盘 bar；半日市按实际时长。
既有 `preopen_stock_cycle_v3/daily.json` 的完整 5m 日聚合优先于重叠的小时日聚合，其原生成器只输出完整常规盘路径。
日内比值避免将隔夜拆股跳变当作日收益。没有补写缺失 bar，也没有在本次下载行情或上传 R2。

## 来源与异常

- 成员来自本版 `pool_snapshot.json`、v5 发行人成分文件的成员记录。
- 部分业务证据继承 `industry_tags_v3.json`，保留 2026-09-25 观察日期；新证据链接标出本次查阅时间。
- IYM 的旧文件抬头为消费必需品基金，与 IYM 身份不符，本版排除旧成员。
- EQIX / CCI 被 OpenD 标为 ETF，本版依据官方发行人资料单列公司股票，保留原始类型和纠正说明。
- ADRO / CRGX / INH 的旧成分证券代码在本次 OpenD 基本信息查询中返回“未知股票”，保留身份异常和未评级。
- 同名 Lime 的资料曾出现错配，最终以 Neutron Holdings/Lime 的官方法律及产品资料核对；瑞典 Lime Technologies 的 CRM 产品不用于美股池中该证券的评级。
- 不自动展开杠杆 ETF 的股票代理，不把期货、现金、衍生品或非标准行业列表当成股票。
- 全部成员、分类和价格回看都基于当前已知证券，没有历史 PIT 成分承诺。没有访问目标收益标签，也没有改变旧研究的模型、阈值、协议或观察配置。

## 复跑

```bash
# 先在可终止子进程里读取 OpenD，35秒总预算；不要无边界等待。
python3 - <<'PY'
import subprocess, sys
subprocess.run([sys.executable, 'analysis/ai_value_chain_map_v1/capture.py'],
               timeout=35, check=True)
PY

# 行业初筛与逐股核查合并；没有业务证据的项目保留 null。
python3 analysis/ai_value_chain_map_v1/screen_industries.py
python3 analysis/ai_value_chain_map_v1/enrich_labels.py

# 使用本地数据和已有成员文件生成，依赖现有 numpy/pandas/scipy。
python3 analysis/ai_value_chain_map_v1/build.py

# 已授权的自选名单同步：只添加至已存在的「AI 篮子」，不删除成员、不下单。
# OpenD 不提供新建分组接口；分组不存在时保存明确的未完成状态。
python3 analysis/ai_value_chain_map_v1/sync_ai_basket.py --apply
python3 analysis/ai_value_chain_map_v1/build.py

# 浏览器验收；Playwright 可使用 Codex 捆绑运行时。
NODE_PATH=/Users/admin/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules \
  node analysis/ai_value_chain_map_v1/verify_browser.cjs
python3 analysis/ai_value_chain_map_v1/verify_data.py
```

新版采集写入本目录，不覆盖旧研究。`data.json` 是本次生成的独立完整数据；`index.html` 为单文件交付。
原始 Parquet 不进入 Git。未来如需长期保留每次成员变化，应另存带时间戳的快照目录；本版不是自动刷新或监控服务。

桌面和手机、全部筛选、业务路径、缺口去重、公式边界、日期验证、分页、下载、明暗主题和离线打开的验收记录见 `browser_verification.json`。
浏览器通过仅证明展示和计算路径工作，不证明股票预测或策略效果。
AI 等级来自资料整理和规则判断，没有统一的 AI 收入占比口径。行业标签覆盖不代表完成了每家公司的 AI 业务核查；全池中尚无具体AI路径证据的公司仍保留业务关联待核实，与AI池已有批量初评分开。
`ai_basket_members.csv` / `.json` 为与页面一致的全部 AI 相关名单，`ai_basket_action.json` 保留 futud 实际分组回读结果；不能把名单准备完成表述为分组同步完成。
