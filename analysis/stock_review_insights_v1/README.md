# 行业环境 · 价格证据展示

`industry_price_evidence_v4` 在区域最前面新增行业与股票同屏折线，默认各市场最近60个交易日。一级分组支持全部、美股宽基与行业基金、美股 AI 价值链样本及美/港小样本；可以从成员列表混合叠加行业、基金与个股，搜索名称/代码，移除或批量选择。跨市场按实际日期对齐，缺点断线，不补价格。曲线共用以0为中心的对数比值坐标，图例/指针强调曲线，键盘可切日期与对象，支持20/30/60日范围。

默认价格中心为各自本批截止 T−30 至 T 的31个收盘点的几何均值 G，y=100ln(C/G)；轴标签为 exp(y/100)−1 的百分比，0对应各自 G。中心是固定的历史描述参考，不随范围与过滤改变。另提供日涨跌幅均值口径：μ=30次日简单收益的算术均值，y=100ln[(1+r)/(1+μ)]；轴标签是增长因子相对中心的变化，读数同时列出实际日收益、均值和百分点差。完整中心窗口缺失时不画线；行业从与日 K 一致的参考篮子收盘路径计算中心，基金用自身日线。

原表涨跌幅与红涨/绿跌条保留，在「5交易日」前有日 K 列，20/60日列分别显示整体走向、拟合度 R²、方向效率 ER。名称每行只出现一次；点击名称或用键盘展开成分股，子行使用相同五列。日 K 默认近20交易日，可切到60日；最小间距5px，按官方交易日排列，缺数据不补K线。

日 K 读取同批次 `base/results/trend/prices.json`（或 staging 的 `results/trend/prices.json`），成员和原涨跌幅读取 `context.json`、`inputs/value_chain.json` 及 P08 行业结果。不联网、不更新旧行情与原分析，新增计算保存为本次 render 的描述指标。ETF 展开发行人成分快照；缺名单的基金显示未取得，不用近似名单冒充。

组日 K 为参考篮子：各股共同基期收盘归一为100，期初等资金、持股数固定，将每日 OHLC 等权合成；连续61个交易时点完整的成员固定纳入，覆盖数与排除名单保留。多股日高/日低不同时发生，合成 High/Low 仅是日内范围上/下界，以虚线影线表示；开盘/收盘是真实同一时点的合成值。单股、基金和单成分组使用真实归一化影线。它不是可交易净值，当前成员回看不是历史 PIT。

整体走向取收盘路径对交易时点的 OLS 斜率方向；R²=1−SSE/SST，恒定价格不可定义；ER=|期末−期初|/Σ|逐日变化|，恒定路径按0处理。20/60变化分别需21/61个完整收盘点。组指标在固定篮子的收盘路径上计算，不平均各股 R²/ER；原表涨跌幅是各窗口个股收益的等权均值，固定篮子的权重会随价格漂移；即使成员相同，其区间收益也不一定等于原百分比，评分样本还可能不同。拟合度、效率和方向都不表示未来概率。

全表涨跌条共用对称刻度，原表百分比、广度、未知和解释逐项核对；原表保留在折叠区域。市场及小样本通过分区行区分，各日 K 纵轴自适应。绘图采用内嵌原生 SVG，无网络依赖。

在一次 SOP 的 P08/P09 HTML 已生成、尚未写 manifest/checksums 或封存时执行：

```bash
python3 analysis/stock_review_insights_v1/render.py --staging <本次未封存目录>
```

也可在报告生成器中调用 `render_html(source_html, industry_result, prices=同批日线输入, context=同批上下文, value_chain=同批成员快照)`。输入未取得时日 K 和指标单列缺失。它返回新 HTML 和解析数据，仅替换行业价格证据区域。所有展示数值、广度、未知和解释必须与来源一致，否则拒绝生成。SOP 执行要求见 [一次更新 SOP](../ai_value_chain_map_v1/ONE_RUN_UPDATE_SOP.md)。

已封存报告改用新展示目录：

```bash
python3 analysis/stock_review_insights_v1/render.py --from-run <原封存 insight 目录>
```

新目录记录 `run_kind=render`、原批次和哈希，保留原截止及分析状态；不更新 `last_once.json`、双页面发布指针或旧 ZIP。默认保存于 `.cache/stock_review_insights_v1/renders/日期__render-UUID/`。它通过本地相对链接复用原报告详情及证据，从详情返回首页会进入新展示版本。独立分享时须物化来源依赖后重新生成包的校验文件，不能把本地链接视图当作独立分享包。

新 SOP 批次在自己的 staging 中直接渲染，记录 `results/industry_table_data.json` 和 `validation/industry_price_evidence.json`，将渲染器、HTML 模板和浏览器检查器复制到 `repro/source/`，再完成浏览器验收和封存。这项接入不把现有双页 `run.py` 扩展为自动 P08/P09 流程。

浏览器验收可运行以下命令；默认使用 Codex 本地运行时的 Playwright，其他环境可通过 `PLAYWRIGHT_MODULE` 指定实际已安装的模块路径。它在断网模式下检查六种宽度、全部绘图值、未知标签、原表展开、鼠标/键盘选择、逐股筛选、链接及详情返回，并保存截图和 `acceptance.json`。

```bash
node analysis/stock_review_insights_v1/verify_browser.cjs <报告目录>/index.html <报告目录>/validation/browser
node analysis/stock_review_insights_v1/verify_overlay.cjs <报告目录>/index.html <报告目录>/validation/overlay
python3 analysis/stock_review_insights_v1/verify_overlay_math.py <报告目录>
```
