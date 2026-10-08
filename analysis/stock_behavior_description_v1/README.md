# 过去的股票性格：历史描述审查 · 两项指标

入口：[index.html](index.html)。默认是多股票图墙：全部55个特别关注连续展示，桌面每行3张图、中等宽度2张、窄屏1张。每张图叠加日线N日历天和周线7N日历天，下面直接显示「偏上行／反复折返」两条指标，以条长、百分数、符合数/窗口数表达。按用户澄清，仅描述过去的K线；前版 operation_cadence_v1 的未来事件实验保留，不能当作当前需求的验证结果。

2026-10-08按用户决定移除第三项，模型版本为 historical_path_description_v2。原三项规则、默认结果及页面源模板保存在 [archive/three_indicators_v1](archive/three_indicators_v1/manifest.json)。两项保留指标在220个真实证券×日期案例中与归档模型完全一致。

审查流程：统一选择历史区间 → 同屏比较多股日线/周线与两条指标 → 按指标升降排序，或逗号输入多个代码集中比较。指标默认计算日线区间；可统一切换到周线背景区间，仍按该范围的日收盘计算相同描述。主视图不列单股绝对值统计面板。

日线/周线的绘图机制直接复用 ai_trend_quadrant_v2/price_chart.js 的逐字相同快照 price_overlay.js：两套独立横轴，共用右端和价格轴；周OHLC聚合、缺失、未完周、起点截断、平移与最小5.5px间距沿用参考代码。左栏是统一日期、筛选、排序、周线显隐与真实两年导航。默认日线60日历天/周线420日历天。

图墙增加前后50%日线趋势斜率：整个日线范围按官方交易日位置等分，奇数时后半多1个收盘，分别对log收盘价做最小二乘拟合。前半虚线、后半实线，图下注明斜率（对数%/交易日）与后−前的速度差（pp/日）。平移只裁切展示，拟合仍对应整个所选日线区间；周线背景和指标设置不参与拟合。缺行情的半段留空。独立辅助模块 slope_overlay.js 不改变 core.js 描述公式、两条指标、输入数据或参考绘图代码。TXG默认范围两段为8月10日—9月8日/9月9日—10月7日，斜率分别+0.545/+1.460，差值+0.915 pp/日；原偏上行仍95.5%（21/22）。

每张卡片的“详情”打开[detail.html](detail.html)，携带对应代码、指标日期范围与描述参数；全天60m、原始量、基本面与人工意见保留作补充检查。前一版单股源模板 template.html 保留，当前默认源模板为 comparison_template.html，详情源模板为 detail_template.html。

百分数分两类：实际价格变化；描述在完整已发生窗口中的占比。窗口占比只是一种历史经验频率，不是未来胜率。上行和反复折返可以同时存在，没有买卖频率输出。默认规则及算法见[PROTOCOL.md](PROTOCOL.md)。原G/V/M/J/S在区间不足窗口时留空。

特别关注快照为2026-10-08 02:38 UTC，55证券（38公司股票、17ETF）。复用已取得的真实日线、52个美国证券全天60m、官方日历及供应方完成时间戳证据，不重复获取数据、不上传R2、不修改旧实验。当前成员不是历史成员。两只港股日线截至10-06；公司资料为10-07附近的缓存，不套入历史价格描述。

两个页面均自包含、可离线打开，脚本与数据均内嵌。core.js负责描述计算，comparison_template.html负责图墙，render.py从已有捕获生成两页；输入文件SHA见[input_manifest.json](input_manifest.json)。图墙只嵌入日线，全天小时数据留在详情页。重建需要本地 operation_cadence_v1 输入；页面直接打开不需要Python、Node或原始Parquet。

本轮核对覆盖220个真实证券×日期范围案例，其中217个具有完整路径计算条件；独立复算18142个历史子窗口及465个原五参数数值，107个原始文件SHA保持一致。工程报告见[calculation_verification.json](calculation_verification.json)、[independent_verification.json](independent_verification.json)、[browser_verification.json](browser_verification.json)。不把这些核对称作语义通过或预测有效。

图墙修订单独核对55股默认描述占比保持不变、440个双范围图和28533个周OHLC聚合，见[wall_verification.json](wall_verification.json)。真实浏览器在1440×1000首屏显示6张完整价格图及对应指标条，另查1050/390px响应布局、多代码搜索、排序、显隐、日期与平移、详情跳转、缺失、探索参数及离线零HTTP请求。此次只改变展示与对比方式，没有更换描述公式。

```bash
python3 analysis/stock_behavior_description_v1/render.py
node analysis/stock_behavior_description_v1/verify.cjs
python3 analysis/stock_behavior_description_v1/verify_independent.py
node analysis/stock_behavior_description_v1/verify_wall.cjs
node analysis/stock_behavior_description_v1/verify_browser.cjs
```

Node浏览器验证器采用本机捆绑Playwright路径，其他环境需调整；既有静态服务127.0.0.1:8768继续复用。算法/工程检查通过不代表用户语义验收完成，页面保留“人工审查中”。人工审查仅保存在当前浏览器，可导出版本化JSON，不自动发往外部。

旧 slope_verification.json 和 offline_verification.json 为前次版本的历史证据；当前页面以 browser_verification.json、detail_browser_verification.json 及 calculation_verification.json 为准。
