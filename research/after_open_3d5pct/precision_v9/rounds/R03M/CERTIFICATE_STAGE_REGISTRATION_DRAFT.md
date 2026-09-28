# R03M 等价证书数值证据阶段登记草案

2026-09-27；**仅方案，未执行、未签发证书、不改loader**。根协调器已选定15套范围；采用此草案后由Sol另新增离线薄审计入口。新输出建议 `runs/precision_v9_20260927/R03M_certificate_evidence_v1`，必须新建，保留完整失败。此阶段不拟合、不调校准/阈值、不触R03/网络/调度/交易。

## 当前状态 / 依赖

retry1全14,424键的六数组和监督已独立全等，身份绑定见 REVIEW；新metadata versions SHA `1dd617f925d74024838f9cde19e58ca4296b0c5d74f64b74fec421eed9b46788`，memberships SHA `f1fc75227d83ab9d27caf96c47399279ca5457d6b8d547d703ef41ea8af5a740`，result SHA `d00e83c18a31ca26f65e40bc75d6c9a0f66ab401e96f97ddde49c3f50a8d88a5`。原/新NPZ与rows字节身份见REVIEW，禁止重建/抽样替代全键。

当前features SHA `528344d95bab167fb520584e854c0223e2ccba2b592b3fcaa5a2a9818229e646`，inference SHA `e2a0667c5d44cc62c5760a10fa6d5571f7c236e78488d4b2f5d5cf349bb9ded3`。当前loader明确拒绝causal→focus_v8；本阶段允许显式离线数值审计直接调用原权重计算，**不伪造legacy lineage/训练证明，不把loaded字典注入生产predict以规避load_route**。结果标 `offline_numeric_equivalence`，正式接口G1仍待后续证书入口实接。

## 冻结模型范围

根为 `runs/focus_v8_20260927`；route顺序固定 B_group/B_no_group/C_group/C_no_group/C_no_daily，fold顺序 dev1/dev2/selected_3566。共15套，7566不进入。配方来自原round_10/summary.json，SHA `ed1b7801e31c2aa5e3b616e521ae0a44e4d70445c2f2170798f8ebfa84639c61`；不重选incumbent。

每bundle SHA为有序 `[{"file":basename,"sha256":actual}, ...]` 的JSON SHA（sort_keys=True,separators=(',',':')）：B文件依次target_0.txt…target_8.txt，C为model.pt，随后均为calibration.json/registration.json/result.json/eval.parquet。先保存具体文件字节/逐文件manifest并核下列SHA，再运行预测。

| route / fold | 相对目录 | bundle SHA-256 |
|---|---|---|
| B_group dev1 | round_09/B_group/dev1 | `b6c30d280cc2330abf087b53380cd72b91d9ff53fb5e4050a51d3ab81109a714` |
| B_group dev2 | round_09/B_group/dev2 | `192e58f702cc5c9430ba03d5f1516db16fd34918aace2a3878f5d1f13cc09007` |
| B_group selected_3566 | final/B_group/selected_3566 | `cdd081b03c485b2a31fc0f7005614bc1fcf4e4718ec1f12680367ea09a92d73a` |
| B_no_group dev1 | round_09/B_no_group/dev1 | `45d7a1ed61aa60a71499380b7bbee3af540b99382a72cd2c6cdc878605644f1d` |
| B_no_group dev2 | round_09/B_no_group/dev2 | `d2e1b4ee4235783f0947a9471bc7a2f6f19bdcbcb4abef70e1933d381a1f2774` |
| B_no_group selected_3566 | final/B_no_group/selected_3566 | `95c05e8e5f727bc5224a81d08b696af9b32116d39da4adbf1534ec312b458ab9` |
| C_group dev1 | round_10/C_group/dev1 | `b1d0ca763ae09d6a9bcb4f9a0c97674b99c3738ecca29d309076dfcec47f358e` |
| C_group dev2 | round_10/C_group/dev2 | `60d9f842c9c889b71933c3b3e316c9e2c9fcf9a12df46962f6660f26e824606b` |
| C_group selected_3566 | final/C_group/selected_3566 | `021e7421c6b39a3e8ef51d0753301b2b693434ff90f05623b38892b6b3492ecd` |
| C_no_group dev1 | round_08/C_no_group/dev1 | `dcdd3b306f975e811aaa8b5c1420eea7bdd0816764f8d50f92eb38d4e172b30b` |
| C_no_group dev2 | round_08/C_no_group/dev2 | `fd525a2a200767d4be3a1258579a125eccf3310ab1da5999500fdf862a3de486` |
| C_no_group selected_3566 | final/C_no_group/selected_3566 | `a13fc799ed2ed06819523e8171f8a31b0690723fefa5a2bcdfd78315cf86dd58` |
| C_no_daily dev1 | round_10/C_no_daily/dev1 | `799e104d88022de9c1dc94cd14a11e7d9a8042a52a40e97cf81e4fef79c636ed` |
| C_no_daily dev2 | round_10/C_no_daily/dev2 | `d08d61fb9510efcae3989e18fa27c169185540a875e850a9075300b99413281a` |
| C_no_daily selected_3566 | final/C_no_daily/selected_3566 | `4e031701e8d08db900d422fb0a20ef2cc873f23a8996dc03584e05da315b4b3c` |

## 接口与全键门禁

1. 新薄入口仅接 old_dataset、retry1 dataset/metadata、上述显式model inventory、该登记、独立output。启动先冻结所有实际调用代码/配置字节、包版本与SHA；结束复核旧模型/校准/源/新metadata未变。保留原schema声明与目标causal schema的准确内容/SHA/来源；若原checkpoint无原生schema文件，明确记录为从冻结代码推导的声明，不能伪称历史训练时已保存。
2. 每套模型对**全部14,424旧键且原序**分别由旧/新输入计算九raw和九处理概率（原calibration+monotone处理）。保存逐key/target old/new/error、完整key集合SHA、形状/NaN掩码、最大误差与最差键；每项≤1e-7。不能只比已有eval交集；已有eval.parquet的每个键额外与旧路径复放核对≤1e-7，保证未同时换错参考实现。15×14,424=216,360模型-键，1,947,240个九目标位置分别核raw和p。
3. 模型有效输入同时核对：x5/x60/xday/group_seq/prior及实际使用的tabular/path/curve/relative；C_no_daily不得传日线，no-group不得传peer。训练support、成熟prior、真实model_version、recipe和校准保持原意；两无群同样跑全部三套作为不变对照。任何运行批次失败保留该失败，不能拼接幸存键称通过。

## 代表feature-only样本（在任何预测差之前固定）

固定六键，均已确认旧rows各存在一次：AMD 2026-03-02（最早旧日、trend63/126合法空群）；AMD 06-01（原107依赖失败锚点）；AMD 07-10/07-13（trend126由incomplete空群跨周到classified）；AMD 09-17（旧键最晚日）；COHR 06-01（新增七股float32分类路径）。不因耗时或差异换成容易样本。此选择不代表穷尽公司行动/全部分类组合；缺字段、动作冲突等拒绝仍由既有反例门禁约束。

每键按新metadata从全部108candidate角色确定实际六周peer依赖，另载QQQ；不能固定复制107或缩池。只给bar_end≤11:30、archived available≤11:30:30的当时前缀及成熟prior，禁止当前/未来labels或11:35 entry列进入feature builder。received_at不存在，明确历史fixture，不伪造接收时间。先比features对该旧键冻结有效输入，再用上述15套原权重做九raw/p数值对照≤1e-7，保存causal lineage、实际依赖、history IDs、逐层误差和拒绝。正规loader仍拒绝旧模型，不能把离线数值调用记成生产接口成功。

## Backlog / 复盘矩阵 / 后续证书边界

先inventory与schema来源→全键数值证据→六代表键feature-only数值证据→独立复盘。单worker、固定单线程并记录分阶段耗时；每20分钟保留checkpoint/实际进度评估，不删样本强行缩时。不能把本阶段离线耗时当30秒SLA；全21同快照测量另按PERFORMANCE_PLAN。

| 实际结果 | 后续决策 |
|---|---|
| 全键+六代表数值、来源全部通过 | 只产生待采用的exact_tensor_equivalence证书材料；另登记最小loader入口 |
| 来源/监督/数值不匹配或feature-only拒绝 | 保留失败，定位工程/依赖/时间语义；不签证书、不训练、不换样本 |
| 最终发现真实群输入差异 | 先按key/week/strategy解释，再交三模型Astra另议refit |

未来证书必须绑定旧模型真实版本及SHA、原schema及其来源、目标schema、新membership/versions、旧/新dataset与完整keys、全部输入/九输出/代表feature-only报告SHA、校准及代码、作用路线/日期/来源范围；不命名为group_training_provenance，不宣称重训。旧默认拒绝、未知模型/metadata/证书篡改及超范围拒绝保留。正式G1需后续证书接口实际端到端通过；本阶段不改loader，G2=false、G3未完成、无90%或改善声明。
