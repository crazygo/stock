# 数值证据采用与后续优先级

2026-09-27，根采用 Astra 的 `NUMERIC_RETRY_REVIEW.md`，SHA-256 `f2215fcc1d0c02a00db1764ded7f0b2bd4f6599f359f879541eee6a482444902`。15全键、15原eval历史批次及90代表singleton配对均通过；原失败、跨批差异、历史接收证据缺失等边界保持。根亲核代表summary和终态，并审阅独立复盘。此采用不表示一般G1、G2/G3或模型达标。

根选择优先推进 **R03新训练的原生工件导出与正式预测验真合同**。该工作直接服务五路线下一次拟合及后续日期。拟合前明确旧/新数据臂身份、schema、真实训练来源与最终工件验证，避免训练完成后再临时解释来源。

`HISTORICAL_CERTIFICATE_LOADER_DRAFT.md`（SHA `d2e513773b47c5dd155eec5be12d24dc88e5670bb90aa012e12b66446e3f8dae`）保留为**未采用、暂缓**。它仅能补六历史键的正式入口证据；当前完整数值证据已能支持输入兼容结论，额外例外分支不能支持新日期或21×5性能。暂不修改loader来新增这个历史例外，不生产生效证书。

由Astra按实际baseline/prepare/inference代码新增R03原生工件STATE/BACKLOG/MATRIX/REGISTRATION_DRAFT，根审查采用后才交Sol实现。旧common臂必须保持旧输入/训练身份；新臂只有真实causal构建来源和验真通过才可声明原生causal。任何工件封装都不能代替实际拟合、正向feature-only验证或独立效果验收。

R03数据恢复及五路线匹配基线仍按原顺序。完整21候选性能等待合法的新模型/输入范围就绪；不以历史六键的有限许可放行。达标仍为0/5，目标继续。
