# R03M 独立复盘

2026-09-27；Astra xhigh 只读审计。采用 **metadata-only / no-refit** 分支：合法重算恢复了群时间证据，全部旧输入全等；没有模型效果改进证据。三群正式 feature-only→inference 的 G1 仍未通过，G2/G3=false，达标模型仍为 0。原四件登记不变，R03 历史扩量仍独立排队。

## 实际状态与首失败

- 首 run `runs/precision_v9_20260927/R03M_group_only_v1` 在 95.810s 停于旧成员覆盖比较；无新 dataset。运行源码快照 SHA `76904f47e8f919f9dd124552fb19a289cd803eecb6736c1410b85aa6e3d30a5c`。
- 我对照该快照、冻结 universe 及实际 21,060 行 membership_audit，独立确认旧 comparator 未按 candidate role 限域：5,850 old-only 键来自 30 非候选（IGV/QQQ/SMH/SOXX 为 benchmark，另 26 不在冻结 universe），**lost candidate=0**。这是比较器缺陷，不能解释成真实群输入变化。
- 修复另登记于 `FAILURE_REVIEW_AND_RETRY_REGISTRATION.md`（SHA `90a6b52e75ebf0c4a913faccfbd12e2c8dfd93830b8ac2dc0fb935e74c8f3b27`），保留首失败；retry1 的比较器仍拒绝原101候选缺键/重复，并要求108×39×5完整新网格。没有扩大 peer 池或改分类器。
- 唯一重试 `runs/precision_v9_20260927/R03M_group_only_v1_retry1` 于 128.668s 完成，源码快照 SHA `bfe2c08323085b5113b7647a6e4e7b11ab6dc0ae65059e0af15bede4a541fbd9`。本审计没有重复重建、训练或改数据。

## 亲核的输入、监督和因果证据

1. 原101候选经原 float64 classifier，新增七股经冻结 float32 view 路径；实际分别19,695/1,365成员。108候选+QQQ共109行情/动作依赖，QQQ只在benchmark槽；发行人去重和peer history_end门禁保留。
2. 20,647个旧candidate成员的类别/状态全部相同；另413个原来省略的新增七股成员全为 `insufficient_history` 空群（每股59），不是新classified或可用性改善。新成员共14,284 classified、3,948 insufficient、2,592 incomplete、236 corporate_action。
3. 亲算223个冻结源、9个旧identity、6个实际源码快照的文件SHA，均匹配。before/after内部共有6个section逐值全同，不能比较两份JSON自身SHA是否相同。after未含原registration字段，已补验当前原件与run副本均为 `7cd17fbcd49b8bbe648a3015ec8fd25df5dbc96155adaabf995b6095f4db7303`；不回写旧审计文件。
4. 独立读取旧/新parquet，14,424×16字段全部相同（含顺序、entry价/时点、九目标成熟时点、terminal1/3/5、own support、group_valid_count）。独立流式读取全部六个NPY成员，含头部/shape/dtype/NaN字节全等；group/group_seq全有限，14,424条key_audit全序对应且最大差均0。新旧整个features.npz SHA同为 `6bce8471011f1c2c54ce2551ba7584f18196fb5b056580e528d95532d5c0e744`，rows.parquet同为 `b6220c3645ff6477ff59763d75deaff26b6f60c7b1ad34f6d0b2b913a4fed2fe`。
5. 未调用builder，另按每股严格 decision/end/available<当前decision 重算全部prior数值、counts、末63历史IDs：三项SHA均匹配，累计历史计数658,631。另重算等群权重、2,919正权重IDs，以及dev1/dev2/reserved各fit/tune/cal/eval的完整/正权重IDs，全部匹配；协议/seed3566未变。prior SHA `e369c26d288c97930aa91471bf745c57336206e7962d4ef0470ebc663f151972`，history IDs SHA `47be572f93319a931af4d69f972c575afb603f91442bd3fcf80f1190aa24ecf8`。
6. 检查全部14,284成功成员：history_end确为记录的实际窗口末日且早于该周首日，max_available早于cutoff，bars/actions SHA绑定冻结源。源码确实从bars重算并保存窗口，不是根据旧类别补日期。历史available仍为既有假设，source_observed_at=null；不能还原真实receipt。

## W01 / W39 边界

W01首官方session为 **2025-12-29**，旧version本来也是该日，新入口保持它；2025价格缺失继续拒绝，未补假history_end。W39新effective_to为 `2026-09-28T13:30:00+00:00`，旧为null。我独立按所有旧键重算六周代表索引：旧键3/2–9/17，代表日期1/30–9/17，仅W05–W38；两端version均不被触达。这个结论限定旧键，不证明W39之后的有效性。

## 采用边界与剩余门禁

数组全等支持进入兼容性**数值证据**阶段，不需要等R03新模型才有机会解决此兼容瓶颈；也不自动放行旧checkpoint。当前loader仍拒绝causal schema+focus_v8版本，尚无证书入口。本轮未实际比较15套checkpoint全键九raw/p、未用新metadata完成代表feature-only正向配对，也未证明全21候选30秒运行或前向receipt。

下一步按 `CERTIFICATE_STAGE_REGISTRATION_DRAFT.md` 冻结15套权重/校准和代表样本，先离线数值核验；保留真实causal lineage，禁止伪装legacy、改版本字符串或伪造training provenance。证据通过后才另登记最小证书加载入口并实际连接G1；无群仍作不变对照。复盘分支见 `POST_REVIEW_MATRIX.md`，实际backlog见 `POST_REVIEW_BACKLOG.md`。

## 结果指纹（相对retry1）

| 制品 | SHA-256 |
|---|---|
| result.json | `d00e83c18a31ca26f65e40bc75d6c9a0f66ab401e96f97ddde49c3f50a8d88a5` |
| group_audit.json | `2bce59ccffe860308b8ee352a703f6bbefdf7d1aeffc4b6e7ee641a2080e74cf` |
| key_audit.jsonl | `2e7da358d9d6e7cd0f2d0e67a4683d444fead04f00fa373324047a49f0009a45` |
| supervision_audit.json | `d8f0dabb71606ca80eaa3b0b1a9f6a5c5f60b67c26311d4b16a628ab4c471b1f` |
| group_metadata/versions.json | `1dd617f925d74024838f9cde19e58ca4296b0c5d74f64b74fec421eed9b46788` |
| group_metadata/memberships.json | `f1fc75227d83ab9d27caf96c47399279ca5457d6b8d547d703ef41ea8af5a740` |
| group_metadata/classification_audit.json | `7aef4b747fbec3e743ad866e8acc932c0aaaf2d759e5ba6291f09780fa36bb9b` |
| inputs_before_sha256.json | `cdd59f938d2925fd09e0b9925cb06e22b5095876cfe48e30be0ae18dd6d7af0a` |
| inputs_after_sha256.json | `1290273b741df3f08a83b293b510bf694a600d6eb29754873a05e7312e89a8d7` |
