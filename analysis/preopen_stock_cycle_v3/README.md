# 单股 / 相关股 × 30 天周期盘前研究

先读 [结果与边界](REPORT.md)，再打开 [调试总览](http://127.0.0.1:8768/analysis/preopen_stock_cycle_v3/index.html)。14 个历史合格组合覆盖 5 只特别关注股票；21 个固定候选配置归为 6 个思路，各有独立 wireframe。整体择模策略仍未达到 70%，不能作为可直接交易的完成验证。

协议顺序：PROTOCOL → COMPACT_PROTOCOL → MACRO_ENROLLMENT_PROTOCOL → ALIGNMENT_PROTOCOL。STATE 记录起始问题，所有失败保留。审计见 audit / run_manifest / ui_verification / execution_diagnostics。页面单文件自包含，原始行情和模型不进 Git。
