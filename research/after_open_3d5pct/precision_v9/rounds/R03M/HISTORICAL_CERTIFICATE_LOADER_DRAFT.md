# 六历史键最小 loader 兼容草案

2026-09-27。**仅待根采用的接口草案；没有证书、未改 loader、未执行。** 前置三数值门禁已由 NUMERIC_RETRY_REVIEW 独立通过并获根采用。此处只补真实 load_route→predict_route 的历史兼容证据；不产生 causal training provenance，不给旧模型任何新日期/新 metadata 权限，不提升 0/5 或前向可用性。

## 冻结基础与作用范围

- 代码前态：inference.py SHA `e2a0667c5d44cc62c5760a10fa6d5571f7c236e78488d4b2f5d5cf349bb9ded3`，features.py SHA `528344d95bab167fb520584e854c0223e2ccba2b592b3fcaa5a2a9818229e646`。保留所有时间/metadata/candidate/prior/禁用输入门禁；不改训练、数据、features语义或首失败。
- 只接 numeric retry1 已冻结 15 套 inventory：三带群×三 fold 共9套需显式历史例外；两无群×三 fold 共6套为不变对照。保留每套原route/fold/seed/recipe/权重/校准/真实旧身份，7566不进入。
- 只允许 AMD 的 2026-03-02、06-01、07-10、07-13、09-17 和 COHR 的 2026-06-01，各日 ET11:30 cutoff、11:30:30 deadline、原登记 decision；旧 `:v6` 对新 `:v9` 显式映射。模式只能 historical_fixture。
- 新 metadata 版本文件 SHA `1dd617f925d74024838f9cde19e58ca4296b0c5d74f64b74fec421eed9b46788`、成员文件 SHA `f1fc75227d83ab9d27caf96c47399279ca5457d6b8d547d703ef41ea8af5a740`，内容摘要 `6c5d2a16d5448b5784d74a63060143516a9aa7e8aa415503a7a62c731a56aad6`。绑定完整108 candidate角色及QQQ基准来源，不能复用证书接另一份metadata。
- 14,424 全键是数值证据人口，**不是本证书 feature-only 权限范围**。不开放全日期、不开放新证券/快照、不开放21×5性能或真实发布。

## 给 Sol 的最小接口边界

1. 保持 `load_route(frozen_artifact)` / `predict_route(loaded, feature_bundle)`；增加一个显式可选 `historical_input_equivalence_certificate` 文件+SHA记录，单独分支，不冒用 `group_training_provenance`。无该记录的旧模型仍按现有默认规则拒绝causal特征；不能改model_version前缀绕过。真实原生causal模型仍走现有训练来源分支。
2. 证书从已采用的 retry1 明确清单产生：绑定每套模型/校准/recipe及完整bundle SHA、原schema内容/SHA和来源、允许的目标causal schema内容/SHA、旧/新dataset/keys、metadata/来源、全部数值报告及最终状态/本次采用记录SHA。原schema若代码推导则诚实标记；同时保留早期训练源码快照与当前解码器身份，不把metadata修复写成重训。
3. 不接受一个自报 `pass=true`/`verified=true` 的对象即放行。load时验证证书及绑定文件实际SHA、完整15套/六键与三个通过门禁；保持现有模型、shape、recipe、校准验证。最小证书制品生成/正式接口检查保存在独立新目录（建议 `R03M_historical_loader_v1`），冻结执行源码、证书与清单字节，原run不回写。
4. predict时再验模型/证书/metadata绑定、上述精确symbol/date/cutoff/deadline/模式与真实causal lineage；保留旧模型训练语义与新feature语义的两个字段，显式记录经历史输入等价桥接。不能把输入lineage伪装legacy，不能让调用者直接拼loaded dict或修改其manifest扩大许可。
5. 六键还须绑定有效输入身份，不能仅凭日期/metadata放行任意X。由已冻结旧tensor及原变换为六键生成每路线字段的shape/dtype/NaN掩码及规范化数值摘要，并在本次正向验收对真实feature-only重验；摘要规范明确字节序/NaN表示、不降精度、不按容差量化。正式predict须核所有实际有效X字段与该键摘要一致，缺/多/改字段失败；已登记原1e-7是数值验收阈值，不是扩大输入许可。
6. 历史证书分支无论caller怎样填receipt/action_frozen都不得发布：显式 historical_interface_only，g2_eligible/publication_eligible=false；不得复用现有“目标schema causal即g1_ready”的快捷判定误报一般可用，通用g1_ready保持false，另报该历史scope接口验证结果。新日期G1另验。证书所指来源/文件改动后须失败，不能只在初次load验一次。

## 采用后的有限验收

正向只跑六固定样本×15套，共90次实际 `load_route→predict_route`（可复用该样本已构建X，五路线输入契约分别保留）；对 retry1 保存的旧 singleton 九raw/p逐一≤1e-7，记录输出/真实lineage/证书身份/质量标志。重新从冻结行情走feature-only，禁止把旧X直接当新路径。全键预测无需重跑，不以跨batch结果替代singleton参考。

定向反例至少覆盖：缺证书；改证书/模型/校准/schema/成员/来源SHA；另一fold或recipe；新增日期或同日另一股票；cutoff/deadline或`:v6→:v9`映射改动；同key改一个有效X字段；伪造receipt/live模式；载入后文件/manifest变更；证书伪装training provenance。每个必须明确拒绝；no_group/C_no_daily原禁用输入反例与无证书旧默认拒绝继续通过。保留失败而不补签、改日期或扩大范围。

成功后只声称六历史键正式接口兼容；仍无实际接收/发布/成熟cohort，不称三群未来可用或precision改善。若scope或数值失败，保存实际失败并回到旧默认拒绝，先解释而不refit。

## 成本、必要性与下一步

| 选择 | 成本与效用 | 决定 |
|---|---|---|
| 本六键小分支 | 一个局部验证入口、少量绑定artifact与定向测试；填补真实接口历史证据，不能解决新日期/性能/目标 | 可由根另采用；不建设通用证书平台 |
| 等R03原生causal模型 | 需数据/训练及拟合前原生工件导出合同；无需旧模型例外，直接服务后续日期能力 | 模型迭代主线，导出合同P0；与本历史证据分期 |
| 立即扩证书/跑21×5性能 | 当前feature-only只六键，扩权无充分身份覆盖 | 不做；PERFORMANCE_PLAN顺延至合法完整21候选范围就绪 |

本草案不授权网络/采集/训练/调度/交易或修改baseline。R03拟合前另登记原生route artifact、真实model_version、schema、group_training_provenance及load→predict验真，不留到训练后改旧版本补证明。
