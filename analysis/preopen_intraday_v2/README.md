# 盘前推断当天 +3% 调试研究

- [本地总览](http://127.0.0.1:8768/analysis/preopen_intraday_v2/index.html)
- [波动尺度与动量保留](http://127.0.0.1:8768/analysis/preopen_intraday_v2/continuation.html)
- [下跌修复](http://127.0.0.1:8768/analysis/preopen_intraday_v2/repair.html)
- [相对市场定价](http://127.0.0.1:8768/analysis/preopen_intraday_v2/relative.html)

先读 [REPORT.md](REPORT.md)，再读冻结的 [PROTOCOL.md](PROTOCOL.md)、[ADAPTIVE_PROTOCOL.md](ADAPTIVE_PROTOCOL.md)、[DIAGNOSTIC_PROTOCOL.md](DIAGNOSTIC_PROTOCOL.md)。页面可直接在现代 Chromium 打开，每页内嵌真实压缩数据，无 CDN；生成文件较大，因为包含完整可导航行情。

主标签：09:25截止输入，当天常规盘High相对09:30首根5m Open触及+3%。请求2024-10-03—2026-10-02；两年包含训练/校准，外层为2026年。当前池回溯、已经曝光的历史、假设end+1s到达、ETF全池数据缺失，因此不能称独立验证或全池覆盖。

根AGENTS.md第六节规定wireframe、起止日期和30/60/90/180日历天、关闭K线鼠标缩放、两年日K mini、默认全日时段标记。页面版本切换和日期联动只读取冻结预测；改阈值/目标是探索，不能改变通过结论。IPO前缺失和低成交不补成合格样本。

`audit.json`是逐股覆盖与源哈希；`results.json`和`adaptive_results.json`是两版指标；`diagnostics.json`是固定几何比较；`ui_verification.json`记录实际浏览器流程；`run_manifest.json`核对模型重放和源完整性。复算命令在报告末尾。原始Parquet/cache和模型不进入Git，不自动推送R2，不接交易。
