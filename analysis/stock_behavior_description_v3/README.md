# 股票历史形态 v3 · 方向、折返频率、折返幅度

入口：[index.html](index.html)，本地地址 http://127.0.0.1:8768/analysis/stock_behavior_description_v3/index.html 。这是用户确认将折返拆成频率与幅度后的独立修订；前版目录及原结果保留。

55个特别关注证券同屏连续展示，每图日线N日历天与周线7N日历天叠加，沿用参考页逐字相同的 price_overlay.js 和既有前后半段斜率辅助显示。默认60日历天截至2026-10-07；桌面3列、中等2列、窄屏1列。灰阶、系统字体、原生表单、明确平移、真实两年OHLC导航，图内无滚轮/拖拽缩放。

“前半幅”上方增加 `偏上行 [>70][>50][<50][全部]`，默认全部；按当前指标描述范围内的原始偏上行窗口占比严格比较70%/50%，缺失仅在全部时保留。与前后半幅、类型、代码筛选取交集，清除筛选恢复全部。

左侧新增两行独立方向筛选：`前半幅 [不限][上][下]`、`后半幅 [不限][上][下]`。默认不限；上为对应日线半段拟合斜率>0，下为<0，两行与类型/代码筛选同时满足。零斜率、缺收盘或未知日历仅在对应行不限时保留。范围变更重算；平移、斜率显隐、周线显隐、指标范围与探索参数不改变当前日线两段的筛选口径。清除筛选一并恢复两行不限。

每图三个字段：

- 偏上行：符合条件的20日变化窗口占比，公式保持不变。
- 折返频率：整个所选范围内反向确认次数×20/实际交易日变化数，次/20日。初始方向不计数。
- 折返幅度：不重复完整峰谷波段的高收盘/低收盘−1中位数，显示完整波段数量；截断首段和未结束末段排除，无完整波段留空。

频率与幅度分别排序，数字使用实际单位。全55证券共用线性条形刻度，筛选与排序不改变刻度，刻度标在左侧控制栏。上行窗口设置仅影响偏上行，确认门槛同时影响折返事件与幅度样本。具体公式和边界见 [PROTOCOL.md](PROTOCOL.md)。详情页携带证券、日期与参数，列出完整峰谷段、真实端点及反向确认日，点击可核对对应日线。

默认快照中，原来同为“反复折返100%”的25证券，频率现在为2.93—6.83次/20日，幅度中位数为8.4%—37.3%；频率相同仍可能有不同幅度。例如MXL与TWST频率同为2.93次/20日，完整峰谷中位数分别15.2%和26.0%。这是历史描述，不推导未来胜率或操作收益。

计算核对见 calculation_verification.json：220个真实证券×日期范围案例的偏上行及原始量保持一致；幅度不改变频率、时间拉长不改变同等波段幅度的构造案例通过。独立Python/NumPy核对217条完整路径、3931个完整波段、18142个上行窗口，见 independent_verification.json。渲染与浏览器证据分别见 wall_verification.json、browser_verification.json。计算检查通过不代表用户语义审查通过。

复用既有真实捕获，未新增行情请求、未上传R2；输入SHA及原页面源SHA可复查。baseline保存原v2计算器和默认结果，仅供回归比较。当前成员不是历史PIT，短覆盖和未知保留，原始Parquet不入Git。

复跑：

```bash
python3 analysis/stock_behavior_description_v3/render.py
node analysis/stock_behavior_description_v3/verify.cjs
python3 analysis/stock_behavior_description_v3/verify_independent.py
node analysis/stock_behavior_description_v3/verify_wall.cjs
node analysis/stock_behavior_description_v3/verify_browser.cjs
```

沿用127.0.0.1:8768既有服务，不重启。HTML均自包含，可离线直接打开；重建需要本地已有operation_cadence_v1捕获。Node浏览器检查使用本机捆绑Playwright。审查意见以v3版本和模型SHA保存，使用独立浏览器存储键，支持JSON导出。
