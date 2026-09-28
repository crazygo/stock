# R03M 数值首失败复盘与有限重试登记

2026-09-27，Astra xhigh 独立只读复核。**首数值阶段失败，14/15；不签证书、不进入loader。** 本补充在首失败及有界诊断后、完整retry前登记；局部reference修复候选已产生，不冒称修改前登记。执行须由根协调器采用，原四件与certificate-stage原登记/采用记录保持字节不变。

## 实际状态

- 首run `runs/precision_v9_20260927/R03M_certificate_evidence_v1` 的 `all_keys_summary.json` SHA `1d471b77edc9be0e0ab60edbb227b79c55ce5ea3e4ec1461634f4db47d094156`：15套全部完成，每套14,424键；新旧raw/p均零差，原eval参考14套过门禁，C_group selected_3566未过。六代表阶段未执行。
- 我独立读取15个全键parquet：制品SHA、完整有序keys、所有raw/p有限、新旧逐值完全相同均成立；没有删键/缩池。另独立核15套模型bundle及其原eval键序与1975/2056/1836计数均匹配冻结登记。
- 唯一越界为 `CIEN:2026-09-04:11:30:v6 / 5d_8pct`：原eval p=`0.6110438413463608`，全键重放p=`0.6110437349259211`，差 `1.0642043968278614e-7` > 原 `1e-7`。raw分别`.47456595301628113`/`.47456589341163635`，差`5.960464477539063e-8`；原校准该目标参数`[1.878070572845283, .6429348742852107]`未变。原eval只有这一目标位置超门槛。
- 首run before/after 488条来源记录完全相同，两文件SHA均 `51f77c82f14fd30b26c0f44866fc9dd3a97d2470cdfd28dd044d21ac14ec77eb`；223冻结inputs齐全。我亲算该run110份快照/清单文件均匹配。首运行源码SHA `8a45f792d4f77a3af6bd75eff95ad246e50cd2393acd19ae62b79c8e56577e2d`，结果不因后续代码改变而改判。

## 复盘：已证实的批次差异与来源边界

独立诊断为 `runs/precision_v9_20260927/R03M_certificate_batch_diagnostic_v1/diagnostic.json`，SHA `f91859f5be6ac22e7669e642f6323d91cf0856520390e26664a0c127b2719145`。Sol实际用原`predict_c(model, arrays, expected_eval_ids)`重放1,836行，raw/p最大差均0；原全14,424批次切片的越界仍单列保留。我的复核未重复推理，亲核诊断11项来源前后相同及实际文件、两源码快照、eval整数IDs/样本IDs/两种batch assignment哈希。

原训练`fit_c`逐split调用`predict_c`，其`np.array_split(ids, ceil(n/256))`依赖本次调用的n。全键为57批 `[254×3,253×54]`；原eval为8批 `[230×4,229×4]`。诊断说明**这次差异随调用批次形状改变而消失**，不是R03M群输入变化；不推断具体底层kernel，也不宣称任意batch完全不变。原eval整数IDs SHA `7c5f65b960c27b5034129a091c62fc0a1bf585dd789b4f8c539dc25b3afc1a27`，batch assignment SHA `7f9c138b3d5d5c1281090b063be2f26b949bcaf0c83ab5e37a6b94cbb9d38a5d`。

| 来源层 | 已验证字节 / 用途 |
|---|---|
| 较早6套dev训练原版本（两B及C_no_group） | Git `101d357`的run.py SHA `7f6189949f792b043c6c4a0dc4d0eedb6462717d9b9be3e4e6da5476627f8b11`、test_core SHA `974ad8aea54e868ef3a64387982fc8f751b55c71d6eeb3e10f6c6395b3ab55b1`，亲读bytes匹配各自训练登记；首run已保存source_origin_snapshots |
| R10/final及本次原解码器 | run.py SHA `abeca557643910acbb79d7a00ecbfd6572a93da6de72231df836efbfa94645cd`；对应Git `ccb96a3`。限定diff为支持mask训练/缓存等，predict_c/apply_cal未变；不能把此文件称早期6套训练当时同字节 |
| 有界诊断验证器 | numeric_equivalence.py SHA `dca10c9dc33225bcec0549e5dcda4f71a74121e335f2a81df38dcdfb640902ad`，诊断目录保存实际源码；此时已加独立eval ids reference候选，尚非完整retry采用结果 |

原checkpoint权重/校准、真实版本、原eval不修改；保留每套完整原eval数值门禁，历史源码可恢复并不替代数值检验。预运行发现的JSON索引/NaN概率/完整inventory/source-final状态问题已由Sol修在首运行前；测试绿不是本轮数值通过，更不是G1。

## 实际 backlog 与选择矩阵

| 工作 | 当前状态 / 下一步 |
|---|---|
| 全键旧/新输入与预测等价 | 首run已证实；retry仍全部重验，不能借此跳原eval |
| 原eval正确调用形状 | 单失败模型1836行诊断已证实0差；15套统一规则待retry |
| 六代表旧张量 vs feature-only | 未运行；固定原六键，每键五路线×三套共90配对，同单样本批次 |
| 有限retry | 根采用本登记后，单worker新目录；原失败保持 |
| loader/G1、30秒、G2/G3 | 均不因数值诊断前移；R03独立排队 |

| 方案 | 决定 / 原因 |
|---|---|
| 保留首失败；按原eval接口重放参考，六代表两侧同单样本 | 选用，分别隔离输入差异和跨batch数值差异；全范围/容差不变 |
| 放宽1e-7、删CIEN/目标/模型、改校准/权重 | 拒绝；会掩盖已观察失败或改变登记 |
| 首run只改汇总成pass、用张量全等代替参考复放 | 拒绝；失败与未做证据必须保留 |
| 直接refit或等R03解决本失败 | 不选；当前证据指向reference调用形状，不是群输入变化 |

## 有限retry合约

1. 新目录固定 `runs/precision_v9_20260927/R03M_certificate_evidence_v1_retry1`，须不存在。复制原stage登记/采用、本补充/根采用和诊断JSON/源码/哈希；启动前快照最终numeric源码、测试及原模型/schema/来源。现候选可继续完成以下reference边界小修，最终SHA另记在run；不得改features、inference、旧数据/模型或第一轮产物。
2. **全键配对**：原15套，每套旧/新14,424键同序、同调用形状（C同57批，B原整批推理规则不变），全部九raw/p及有效输入仍按原1e-7，finite/完整keys/source等门禁保持。不能重划批次挑配对误差最小的结果。
3. **原eval参考配对**：每套从原registration还原完整eval IDs/顺序，独立按原predict接口及该eval样本量的历史批次重放，再对原eval.parquet的九raw/p验≤1e-7。统一应用全部15套，不只修失败模型。原全键预测切eval的跨批次误差继续完整报告为`full_batch_eval_slice_*`；明确它在首run失败且可能继续越界，不能删除或声称所有批次不变。
4. **六代表配对**：原登记六键不变。先逐字段比较新feature-only与旧冻结有效输入，原1e-7/NaN掩码/shape门禁不变；再分别把两侧输入送同一冻结模型，以**同batch=1**比较九raw/p≤1e-7。旧侧必须实际单样本重放原tensor，不能用全键batch切片替代。与全键输出的跨批次数值敏感性另报；换batch不能回避tensor mismatch、改变模型或伪装causal lineage。
5. 保持15套完整inventory/hash、无群/无日线禁用输入、原校准/支持buffer、源前后冻结和失败保存。actual all-key gate通过才执行六代表；任何异常/不匹配保留各层失败。stage重试不放宽时间、receipt或metadata门禁，不注入伪造loaded/provenance。
6. 定向验收覆盖：原eval批次IDs与完整reference、full-population差异仍可见、代表两侧同singleton且故意改变有效tensor不能被批次校正掩盖、有限概率、完整15集合、禁用输入及source漂移拒绝。只检验此小修；不重写模型/loader。
7. 完成后独立复盘15套全键、15原eval、90代表配对和所有来源；三者全通过才形成待采用数值证据，**仍不是正式loader G1**。失败继续分工程/输入/数值原因，不调容差。20分钟观察预算与进度保存保持，不删样本缩时。
