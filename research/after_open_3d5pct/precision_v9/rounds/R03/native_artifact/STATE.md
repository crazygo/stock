# R03 原生路线工件：实际状态

2026-09-27，Astra xhigh，只读源码后的拟合前方案；未实施、未训练、未生成新数据 actual_manifest。根决策为 [R03M 优先级采用](../../R03M/NUMERIC_EVIDENCE_ADOPTION_AND_PRIORITY.md)，SHA `1daa4ee48366153b68ff4888b9fe589c2d0b1f1149dc166e85bbfa36c357f168`。三数值门禁已采用；六历史键证书草案未采用、暂缓，不启动旧证书路径。

| 当前事实 | 工程含义 |
|---|---|
| baseline.prepare_run 已在fit前登记 plan/dataset/docs/code、seed3566、incumbent recipe、完整共同键、各split支持及prior范围 | 复用已有身份/来源链，不重写训练器 |
| baseline.run 仅给B_no_group写585+84列摘要；其余无正式route artifact/model_version/group_training_provenance | 当前训练完成不等于能正式load→predict，需拟合前冻结导出合同 |
| prepare.make_groups 从108 candidate的完整过去session分类，记录实际history_end；data.build单载QQQ作第六槽 | 新臂有causal构建路径，但尚无本轮完成数据/实际审计，不能先填来源或宣布兼容 |
| inference要求完整schema、模型/校准/源码文件及聚合SHA；causal群需训练provenance，legacy群只允许静态fixture | 不能靠改旧model_version或伪造provenance把old_common升级为causal |
| schema完整构建函数当前在forward/test_features.py；B loader拒绝torch先加载 | 提取小型生产schema函数供生产/测试共用；验真child先import inference，不改变baseline训练导入顺序 |
| 五个当前incumbent均calibration=true；只有两无群旧engine smoke已真实拟合复放零差 | 本轮只支持既有shrunken_nonnegative_platt；旧engine是工程对照，不是新臂验收 |

范围固定为5 route×3 arm×dev1/dev2＝30个未来拟合/工件身份，尚未产生这些模型：带群old_common 6套保持legacy，带群new_common/new_expanded_fit 12套需真实causal训练来源，无群12套为excluded。old_common是旧输入的新匹配拟合，须同时区分新run身份与原v8 dataset/builder身份；不冒称原旧checkpoint，也不因新run名字宣称causal。

| 本次读到的文件 | SHA-256 |
|---|---|
| precision_v9/baseline.py | `eb5c9738e6b49bc8abe475f5f2f16733e20f4c73a7a9fd7c53e89201e5206d14` |
| precision_v9/prepare.py | `83094884e6465ffa2310c930664e813107fadfe1ef55a0c58c931ee9993d3d01` |
| precision_v9/data.py | `06207586233fea7153094dbdb40e314cc7608bf5b7c96b269e21bbc083607a3e` |
| forward/inference.py | `e2a0667c5d44cc62c5760a10fa6d5571f7c236e78488d4b2f5d5cf349bb9ded3` |
| forward/features.py | `528344d95bab167fb520584e854c0223e2ccba2b592b3fcaa5a2a9818229e646` |
| focus_v8/run.py | `abeca557643910acbb79d7a00ecbfd6572a93da6de72231df836efbfa94645cd` |
| focus_v8/round_10/summary.json | `ed1b7801e31c2aa5e3b616e521ae0a44e4d70445c2f2170798f8ebfa84639c61` |

R03数据仍未完成/获拟合接受；不猜样本数、actual_manifest路径或SHA。全部2026仍exposed development，归档available为假设、历史公司行动覆盖未认证、无原时刻receipt。当前0/5，G2/G3未通过；官方日历artifact存在不等于具体cohort覆盖通过。
