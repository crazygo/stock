# v6.1.1 群类型修正后的最终研究证据（2026-09-26）

**结果：修复语义后C接近同信息B，但没有证明胜出。** C有群的3日+5%外层Brier **0.144707**，冻结的B有群为 **0.144071**；同1,140行、12个ET日的配对差C−B=+0.000636，5日日期块探索性95%区间 **[−0.002750,+0.005623]**。C的内层Brier为0.157525、B为0.159787，但内层比较也不能代替独立前向效果。2026全部日期已暴露，旧v5/v5.1质量池六时点结果与本11:30全池并不同样本。没有上线通过或正式行动推荐。

本轮只按 [拟合前结构修正登记](26_multiscale_groups_v611_typefix_registration.md) 重训受影响的 `C_patch_group`、`C_patch_no_daily`；B两项及`C_patch_no_group`直接复用 [v6.1冻结证据](25_multiscale_groups_v61_evidence.md)。源数据、9,222样本、1/3/5日×+3/+5/+8%九标签、训练/内层/外层5,606/1,521/1,140行、成本口径及全部配置不变。改动是六个固定策略槽位各有独立轻量投影，趋势15/63/126、波动、流动性、QQQ的类别位不再混成同一种语义；无周版本哈希或日期ID输入。交换不同群类型时输出必须变化的测试通过。当前股票池及周标签仍为2026-09-25事后因果重建，不是原始PIT；行业当前快照回填保持禁用。

| 同样本主目标模型 | 外层Brier ↓ | logloss ↓ | AUC ↑ | ECE10 ↓ | 内层Brier ↓ |
|---|---:|---:|---:|---:|---:|
| B有群（冻结v6.1） | **0.144071** | **0.444989** | **0.7875** | **0.0610** | 0.159787 |
| B无群（冻结v6.1） | 0.139406 | 0.439708 | 0.7965 | 0.0703 | 0.163588 |
| C有群，群类型修正 | 0.144707 | 0.448780 | 0.7815 | 0.0724 | **0.157525** |
| C无群（冻结v6.1） | 0.157473 | 0.489810 | 0.7190 | 0.0773 | 0.171569 |
| C无日级，群类型修正 | 0.168700 | 0.521982 | 0.2504 | 0.0202 | 0.177474 |

C有群相对旧无类型语义C同配置的Brier下降0.018155，但这是看过外层后修复确认的结果，不是新独立增益。C有群相对无群C下降0.012766，日期块区间[−0.018771,−0.007188]；这说明固定类型关系在这段暴露历史内有信息，并不证明将来稳定。C无日级最佳epoch仅1，AUC 0.2504，显示该单种子模型不稳定；完整126日历史在训练期只有1,316/5,606行（23.5%），到外层达1,116/1,140（97.9%），因此不能把两C之差归因为半年记忆的真实价值。B无群外层点估计虽最低，但内层弱于B有群，不能事后用外层改选型。

全部九输出仍按期限非减、涨幅非增投影；新C投影前后外层违例率都是0。原始B有群有6.58%违例，投影后0。概率未经独立校准：实际主目标率0.20965，新C均值0.21588、ECE10 0.0724；B有群均值0.19107、ECE10 0.0610。严格>60%历史质量子组仅27个外层样本，不能用单一子组分数作推荐。无止损的未命中901行3日期末gross均值−1.83%、10分位−6.39%；买卖各6bp的描述性研究净情景全体均值−0.51%、ES95−10.60%，不是可执行策略或强制三日卖出。

冻结运行目录：`runs/multiscale_groups_v61_dataset_20260926_r2/`为带209个本地源SHA的输入、标签、时序与群关系；`runs/multiscale_groups_v61_fit_20260926/`为三项未变对照；`runs/multiscale_groups_v611_typefix_fit_20260926/`为两项修正C、模型、逐行九目标预测、日期/股票诊断及合并对照分析。C修正拟合共9.9秒，序列推理重放与原外层最大绝对误差3.8e−9。未设置自动化、下单或R2上传。

复现命令（在仓库根目录，输出路径每次换新目录）：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 research/after_open_3d5pct/.venv/bin/python -m research.after_open_3d5pct.train_multiscale_v6 run --config research/after_open_3d5pct/configs/multiscale_groups_v611_typefix.json --dataset research/after_open_3d5pct/runs/multiscale_groups_v61_dataset_20260926_r2 --output research/after_open_3d5pct/runs/<new-typefix-fit> --reuse-dataset
research/after_open_3d5pct/.venv/bin/python -m research.after_open_3d5pct.evaluate_multiscale_v6 --run research/after_open_3d5pct/runs/<new-typefix-fit> --baseline-run research/after_open_3d5pct/runs/multiscale_groups_v61_fit_20260926 --output research/after_open_3d5pct/runs/<new-typefix-fit>/analysis.json
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 research/after_open_3d5pct/.venv/bin/python -m research.after_open_3d5pct.train_multiscale_v6 predict --run research/after_open_3d5pct/runs/<new-typefix-fit> --dataset research/after_open_3d5pct/runs/multiscale_groups_v61_dataset_20260926_r2 --model C_patch_group --output research/after_open_3d5pct/runs/<new-typefix-fit>/replay_C.parquet
```

最终验证：`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .../.venv/bin/python -m unittest discover -s research/after_open_3d5pct/tests -q`通过80项本工程单测（含时间边界、追加未来不变、发行人去重、群类型互换），另通过v1配置检查、合成smoke、编译、`git diff --check`及冻结模型预测重放。当前测试环境不限制数值线程时曾出现原生库退出码139，复现采用上述固定线程环境。当前仍仅支持11:30离线研究预测，未恢复六时点，没有真实`received_at`、历史原生PIT成员或未暴露前向日期。下一步应从新交易日冻结原始接收时间与成员版本，再做同样本前向配对；在此之前保留B为较简单的研究参照，并保留C代码与负/未证实结果。
