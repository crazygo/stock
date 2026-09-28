# C_no_group：给 root / Sol 的执行交接

方案/复盘：Astra；纯代码：由root调度Sol。当前仅研究交接，未授权本子任务编码或拟合。只要R03数据与共享builder审计尚未完成，就保持WAITING_DATA_AUDIT；不修改正在使用的`acquire_batch.py/acquire_pilot.py`，不覆盖旧数据/制品。

## 先做P0的数据匹配基线

1. 真源为v8 `final_selection.json.C_no_group`，仅`prior/specialize/calibration=true`；使用 `FocusC`，不能自动接入`support/capacity/within_stock_rank`。机器入口见 [core](../../../focus_v8/core.py)、[run](../../../focus_v8/run.py)、[support](../../../focus_v8/support.py)。
2. 新读取器按名称/来源allowlist提供本股`x5,x60,xday,prior`；group输入为空，QQQ/群状态不得影响本股张量。当前共享loader会预读group数组但该路线forward不接收；新路线接口应避免把群数据传给模型。fit的其他股票用于共享参数学习，与每行输入隔离分开说明。
3. 共享builder提供：不可变来源manifest/公司行动/日历哈希，146日尺度依赖边界、半日/DST/跨年对账，旧新2026配对表及字段差异。11:35 Open对应结束标记11:40的5m原始bar，不能提前用11:35结束bar。
4. P0a固定旧fit/tune/cal/eval决策，使用完整新历史构造输入与本股先验；P0b仅扩大fit至拟定2025-10-01，共同旧行的输入/prior必须逐值相同。早期prior历史可用于两臂输入，但P0a不得将新增行加入监督loss。分别保存`prior_history_ids`与`fit_supervision_ids`。
5. 三折沿用2026边界：dev1 `tune04-20/cal05-11/eval06-01/end06-27`；dev2 `06-01/06-22/07-13/08-08`；diagnostic `07-13/08-03/08-24/09-18`。样本身份如升版本，按symbol/date/decision保留映射。label/entry/terminal不一致必须停止“纯数据量收益”归因并解释，不能默默取成功交集。
6. 同日全部股票同侧；fit/tune/cal按九目标最长窗口的`label_end`及`label_available`purge。实际拟合仍只对21股正权重行执行，登记有效行数/日期；tune早停与cal群等权沿用并披露。
7. 固定32epoch/patience5/batch256/thread1/seed3566，原九头BCE、AdamW设置与校准方法不变。阈值选择算法和网格不变，数值阈值由各自cal重新选择并记录，不从eval选。
8. 阈值评价和1170分钟冷却使用同一官方日历。每折供给分母含无信号官方日；旧评价段应为19/20/18 session。三群重叠信号联合去重，冷却不看提前是否命中，零信号precision=null。
9. 每臂每折保存原始/处理九概率、prior与count、fit trace、cal参数、完整阈值曲线、TP/FP/n、signal_dates、供给、漏报/拒绝、terminal代理及代码/数据/schema/split哈希。配对Brier、同股AUC和信号交集均报告；不拿全池行数当独立样本数。

P0输入修复与新增监督日期分开复盘后，更新本路线STATE/BACKLOG；没有真实结果不得写“长历史已经改善”。旧checkpoint只作旧数据的固定参照，不拿它预测过去新增日期充当时序验证。

## P1的实现约束：own5_patch_neighbor_v1

本节是P0之后的条件候选，root登记实际数据身份及比较基线后才实现/运行。详见 [矩阵](MATRIX.md)，不直接启用现成`capacity=true`。

- 只替换5m分支的池化前变换；原12根5m一patch、16通道、位置嵌入和attention pooling不变。将8×192槽位按ET日期分别处理，每日16个patch；不能跨日期卷积。保留时钟位置，缺口不压缩；输入形状以冻结manifest实际值断言。
- 两新臂都从同一基线结构及共享初始参数创建，`z=GELU(proj(x))+position`，添加残差后沿用同一LayerNorm/池化。残差末层零初始化，使复制相同基线权重时step0输出一致，不能因创建层顺序改变其他层随机初始化而污染比较。
- neighbor：Conv1d16→16、kernel3/dilation1/padding1，GELU，再Conv1d16→16、kernel3/dilation2/padding2；新增1568参数、7patch感受野。每层后重新mask，全缺失位置不得经bias产生伪观测。
- pointwise控制：Conv1d16→47、kernel1，GELU，再Conv1d47→16、kernel1；新增1567参数。相同残差、mask、初始化和优化。若保留dropout位置，必须两臂一致，不能顺便增加正则候选。
- 60m/日线分支、先验加法、specialize、九头loss、校准及阈值算法保持P0冻结配方。没有新手工曲线、群ID、QQQ，也不带回已失败pairwise。
- 最小预算：复用同数据seed3566的incumbent；两个新臂×两个开发折共4次拟合。先写决策，再额外两次diagnostic拟合；不加种子挑胜者。按BACKLOG保留线判断，所有失败检查点也保留。

## 针对性验证与产物

工程验收不能代替效果验收，以下为有意义的机制与时间检查：

1. 固定模型时，改写其他股票/QQQ/group tensor，或改写未来/迟到本股bar、未成熟标签，不得改变该行输入/prior/raw。本股已成熟历史标签变动只允许影响之后的prior；不允许通过全池标准化混入他股。
2. 固定patch投影后，仅扰动相邻有效patch：pointwise残差的目标token不变，neighbor可变化；扰动其他日期不得跨边界影响该日残差。全mask输出零、缺失/未来槽位不产生有效观测。
3. 两新臂零残差step0与同权重incumbent输出一致；九输出有限，原单调投影保留；checkpoint复放最大误差≤1e-7。mask与两层卷积的边界行为单独核对，不能只测最终张量形状。
4. P0a/P0b旧行feature/prior逐值一致；旧新label/entry/terminal逐行对账；官方半日及跨年1170分钟、阈值冷却、重叠群联合计数回归。新source不能改旧运行哈希。

交付新运行目录内`registration/model/calibration/fit,tune,cal,eval预测/selection/precision/signals/paired_audit/decision_before_diagnostic`，并记录参数量、耗时、内存、实际early-stop。root汇总五路线信号/误报重叠；本路线不自行扩写全项目报告或宣布至少三路线达标。
