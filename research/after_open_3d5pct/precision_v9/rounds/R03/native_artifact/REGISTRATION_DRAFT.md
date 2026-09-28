# R03 原生路线工件登记草案

2026-09-27，**尚未采用、未实施；不授权数据验收或拟合**。前态SHA见STATE。根采用后仅交Sol实现下面的薄接口与离线测试；实际R03来源/样本/日历完成后另补30套拟合计划并采用。旧历史证书不启用。

## 范围和不变项

五route：B_group/B_no_group/C_group/C_no_group/C_no_daily；三arm：old_common/new_common/new_expanded_fit；dev1/dev2，共30身份。seed固定3566，recipe逐route取冻结round_10 summary。旧fit_b/fit_c、support buffers、九目标顺序、entry/terminal/标签、旧训练weights、mature prior、calibration/monotone及阈值选择原样调用。无新拟合策略、无reserved/final/7566、无池缩减。

old_common选旧dataset完整共同键；new_common选新dataset完整共同键；new_expanded_fit仅增加共同键起点前且≥已登记fit_date_floor的fit行，tune/cal/eval仍共同键，eval_full另报。成熟先验继续来自所选不可变dataset的全部当时成熟本股行，不能只在allowed keys内计算。

| arm × route类型 | 拟合前schema语义 / fit后身份 |
|---|---|
| old_common × 三带群（6套） | legacy_static_only；保留旧v8 dataset/metadata/builder身份。新匹配fit有自己的run/model_version且显式old_common_legacy，不冒称原checkpoint或causal训练 |
| 新两臂 × 三带群（12套） | 仅当实际prepare/data来源及群构建审计成立，登记v9_causal_history_end_v1；fit后绑定真实group_training_provenance |
| 三臂 × 两无群（12套） | excluded；保留各臂真实dataset和prior来源，不附伪群训练证明 |

## 最小实现与拟合前冻结

1. 新增小型生产schema构建函数（如forward/artifact.py中的build_schema），生产/测试共用，禁止生产导出import test_features/test_inference。输入route、冻结recipe、明确peer语义；shape/顺序/列名/price_basis/normalizer/prior规则与当前loader严格一致，support_rule按recipe.support；两无群宽585、带群765、path84、curve8、B_group relative6，其余relative0。这只是正式声明，不改特征变换。
2. 新增薄register/export/verify入口；baseline只增加拟合前native合同校验、来源冻结与登记引用，不改fit/选择逻辑。每套fit前生成不可变native_contract.json及canonical schema.json，在registration中绑定其路径/SHA、预定model_version和明确arm语义。版本由根采用的run/route/arm/fold/seed身份决定，不能靠导出时改名字洗掉legacy；权重SHA在fit后才计算。
3. 合同绑定实际old/new/selected dataset的rows/features/manifest三文件SHA及shape/完整有序keys、config、actual_manifest、paired_audit、allowed_keys、calendar、incumbent summary、fold/fit_floor/支持量、plan/协议/采用文档。来源为真实文件，未完成的actual_manifest不能占位成accepted_for_fit。
4. 新臂额外绑定prepare/data实际源码快照、dataset.manifest的builder_sha及source_hashes逐源实核、versions/memberships/groups、完整108 candidate角色+QQQ独立基准、公司行动/归档到达假设与分类截止规则；history_end从实际参与分类的最后过去session得到。检查实际classified行类别/版本/history_end与官方session，不补日期或原时刻receipt。未通过不能声明causal；完整数据受既有独立验收管辖。
5. 冻结schema生产函数、导出/验真器、features/inference及其实际imports的字节/版本，保留baseline原SOURCE_CODE快照。schema标明“本次拟合前生成”、来源源码/输入契约，不伪称旧v8已原生保存该文件。old_common保持旧输入，不用R03M重建metadata替换它。
6. 保存每split完整有序IDs/keys、positive-weight fit IDs及权重身份，prior counts/history IDs（配方使用时）；可引用baseline既有制品。时间名称分开：训练rows.decision_at=11:30:30保持原值，feature API decision_at是11:30 anchor、information_deadline_at=11:30:30；先验成熟判断对应deadline，禁止改rows迎合接口。

## 完成fit后的只读导出

7. 新入口 `export_route(completed_run, native_contract, new_output)` 不训练/不选阈值、不改run。要求实际result完成、registration身份/来源一致，重核拟合前后源，输出目录必须新建；失败保留export_failure及stderr，模型/评估原文件不删除、不自动重训。baseline原result/selection/精度统计不回写。
8. 生成loader当前必填字段：route_id/model_version/recipe/schema/schema_file/schema_sha256、按target_0…8顺序B树或单个C model.pt及model_sha256、calibration_file/sha、source_files/source_code_sha256、score_column=`raw_3d_5pct`、该run selection.json的三群thresholds（含null）、action_frozen=false。保留全部原校准参数及原target顺序；五现有配方均calibration=true，本轮只支持shrunken_nonnegative_platt。不能复制旧cal阈值、强设.9或根据eval重选。
9. schema文件用loader _canonical相同字节（无额外换行）；模型/源码聚合SHA分别按当前loader的**有序SHA字符串列表canonical JSON**计算。C checkpoint route/recipe/state buffers严格加载；B列宽实际核验。数据身份定义为canonical三文件SHA映射的聚合SHA，同时保留展开映射，不能把无定义的一个64位串叫完整dataset身份。
10. 保存独立training_provenance.json及hash：pre-fit合同/registration、actual dataset/metadata/builder、split/监督/prior、已完成result、模型/校准/原训练代码字节。新带群group_training_provenance含当前loader所需peer_semantics/model_sha256/schema_sha256/dataset_sha256/membership_sha256，并引用上述可核文件链/versions/universe；旧带群不填该causal对象。无群保留普通训练来源。
11. 现loader对causal provenance主要查字段/哈希形状，不能把自报字段当验真。有限增补仅验证本native合同/训练来源文件及其与注册、模型/schema/dataset/metadata的实际绑定，并纳入load后变更检测；旧legacy默认拒绝causal特征不变。不加入历史证书分支、通用注册平台或新校准算法。训练membership SHA证明训练来源；未来输入metadata需按同一冻结构建/候选/时点规则另登记，不能要求所有未来日期永远等于训练membership，也不能凭任意非空hash放行。

## 正式接口验真与失败保存

12. 验真为单worker独立child，先import forward.inference，再导入训练参考/torch；不在已经import torch的baseline进程加载B，不删除_TORCH_PRELOADED门禁。冻结child命令、cwd、Python/依赖/机器、OMP/OPENBLAS/MKL各1及torch线程设置；实际stdout/stderr/退出码/阶段计时与输入/代码前后SHA保存。具体可执行命令须在Sol实现后的code review中冻结，本文不虚构未存在CLI。
13. 每套先核完整原fit/tune/cal/eval/eval_full保存预测及IDs，再独立按原eval接口/历史eval批次重放完整eval九raw/p≤1e-7。正式predict_route为singleton：对预先选定静态输入，原权重也以同singleton独立计算后配对九raw/p≤1e-7；输入字段/shape/NaN掩码另核，全部概率须有限且在[0,1]。跨batch差单列诊断，不删键、改batch取最优或提高容差。30套逐项终态，不合并幸存集称通过。
14. 代表选择在看输出差之前落单独selection manifest：共同键域固定AMD/COHR 2026-06-01数值锚点（不声称均属每折eval；任一不在实际共同键则登记明确阻塞，不静默替换）；再按实际源确定最早合法日期、126日/群分类跨界、合法空群、末评价日及新增fit历史的确定性代表。去重后的具体键及选择规则须根验收/冻结后才推理，不能按预测表现挑样本。
15. 新两臂实际完成且metadata/全部成熟prior/全108候选+QQQ依赖齐备后，代表从冻结行情真正build_features_asof，再load_route→predict_route；与该臂冻结tensor有效字段及同singleton原路径配对。只传bar_end≤11:30、archived available≤11:30:30与成熟prior，无当前/未来标签。无群同样验证；old_common带群只做legacy静态tensor对照，不能伪装causal lineage追求形式全绿。
16. 历史归档fixture即使G1输入契约通过也receipt=false、G2=false，action_frozen=false不能发布。未来日期需独立合法metadata/snapshot/模型选择与范围验真；生产receipt、发布、cohort另门禁。30个开发模型不自动成为30个前向版本。

## 当前可执行的工程测试与后续限制

根采用实现后，单元fixture覆盖五route/三arm映射、canonical/有序SHA、缺/多/篡改文件、错误arm/模型/recipe/schema/群来源、旧输入伪装causal、source漂移、null阈值、禁用group/xday及support buffer、child失败保留。人工fixture须标engineering-only，不成为真实causal训练证据；无训练测试替身不得生成可采用正式工件。

已有 `runs/precision_v9_baseline_engine_smoke_20260927` 的B_no_group/C_no_group dev1可只读用于导出及正式接口对照，在独立工程目录登记为**既有run的事后工程封装**，不回填其旧registration、不冒称它们曾预登记native合同、不重训。其余三群仅复用旧静态fixture测legacy/拒绝路径；R03M已过数值不能代替新臂真实训练。

全21×5性能须等根选择并冻结五个合法新模型、完整21候选同快照与全依赖、新臂正向契约通过后，按原PERFORMANCE_PLAN另冻结测量范围再跑；不因六历史key证书或静态对照绿而放行，不因速度减池/跳哈希。数据未齐、配对/导出/数值/来源任一失败均保留实际证据，先复盘再有限修复；不自动扩诊断折、训练轮次或部署。当前0/5不变。
