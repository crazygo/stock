# U.S. Stock Drop → Flat Screener

所有研究统一在主干 **`main`**。保留本地工作后执行 `git fetch origin`、`git switch main`、`git pull --ff-only origin main`；历史研究分支不再作为最新运行入口。

## SOXL / SOXS：14 自然日 +5% 研究

最新进展为 [v84 提前公开的政策会议日程](research/soxl_soxs_14d5pct_v84/README.md)：日程版本与 6,152 个历史时点核验通过，完成每日训练和十二份资金账本复算。新增日程主方案一至四月净资金月均 **−5.02%**，弱于匹配无日程的 **−2.94%**；概率更乐观但误差更大，未形成风险回避效果。未新跑指定月份回归，[目标继续](research/soxl_soxs_14d5pct_v84/GOAL_PROGRESS.md)，后续调查事先公告的半导体财报日程。

此前 [v83 状态条件概率修正](research/soxl_soxs_14d5pct_v83/README.md)：十二项测试、十二组拟合锚点及十二份资金账本核验通过。主方案一至四月净资金月均 **−2.94%**，二／三月亏损；相同候选上的概率评分也弱于未修正来源，未证明能力提升。未新跑指定月份回归，[目标继续](research/soxl_soxs_14d5pct_v83/GOAL_PROGRESS.md)；下一步先审查提前公开的事件日程能否提供独立风险输入。

此前 [v82 收益误差与候选选择审计](research/soxl_soxs_14d5pct_v82/README.md)：九项测试、656 个时点及 53517 项数值检查通过。v81 实际 13 笔交易的收益高估主要来自结果类别判断，普通止损概率低估比损失幅度误差更突出；可见行业走强的表现随候选筛选阶段改变，尚不能作为可靠买入门禁。本轮没有新策略收益，目标未完成，[继续检验状态条件概率](research/soxl_soxs_14d5pct_v82/GOAL_PROGRESS.md)。

此前 [v81 现在买入与次日等待的成对学习](research/soxl_soxs_14d5pct_v81/README.md)：每日成对收益模型与十三份资金账本核验通过；主方案一至四月净资金算术月均 **−1.84%**，一、二月仍亏损。相对等待的收益差接近实际，但被选交易的绝对收益继续高估；未新跑指定月份，[目标继续](research/soxl_soxs_14d5pct_v81/GOAL_PROGRESS.md)。

此前 [v80 近期无标签状态加权](research/soxl_soxs_14d5pct_v80/README.md)：用最近已观察状态重加权成熟历史校准，九项测试、六组真实锚点和十份资金账本核验通过；主方案 1–4 月净资金算术月均 **+0.86%**，三月 **−18.23%**，未加权对照反而更好。状态感知尚未带来风险回避效果，未新跑指定月份回归，[目标继续](research/soxl_soxs_14d5pct_v80/GOAL_PROGRESS.md)。

此前 [v79 历史候选的类别概率校准](research/soxl_soxs_14d5pct_v79/README.md)：每日候选成员、四偏置模型、十一份资金账本独立核验通过；主方案 1–4 月净资金算术月均 **−1.10%**，三月仍 **−24.45%**。30 笔交易中 17 次止盈、13 次止损，合格候选校准没有解决收益高估；全部第一名对照月均 +4.32% 也有负月。未新跑指定月份回归，[目标继续](research/soxl_soxs_14d5pct_v79/GOAL_PROGRESS.md)。

此前 [v78 跳空风险分类与历史预测评分混合](research/soxl_soxs_14d5pct_v78/README.md)：主方案 1–4 月月均 **−3.09%**，三月 **−24.45%**、四月 **−3.14%**。33 笔交易中 18 次止盈、15 次止损，普通止损概率被低估，细分跳空类别未形成收益增量；全部代码与结果保留。

此前 [v77 更长历史 × 价格特征每日重训](research/soxl_soxs_14d5pct_v77/README.md)：主方案 1–4 月均非负，但净资金月均仅 **+0.60%**，最大日终回撤 **−15.53%**；13 笔交易，8 次止盈、5 次止损。成本加倍后仍出现负月份，收益目标未达成，旧结果完整保留。

此前 [v75 因果历史均值完整资金对照](research/soxl_soxs_14d5pct_v75/README.md) 预测误差低于现有状态模型，但开发月均 **−0.82%**、三月 **−6.55%**，仍未达标。每日校准和 14 份账本核验通过，八份旧对照精确复现；未新跑指定月份回归。

此前 [v74 分块惩罚](research/soxl_soxs_14d5pct_v74/README.md) 月均 −1.37%；[v73 统一归一化](research/soxl_soxs_14d5pct_v73/README.md) 月均 +0.0754%；[v72 二次特征部分共享](research/soxl_soxs_14d5pct_v72/README.md) 月均 +5.49%，三者均有亏损月。原 [v71 线性模型](research/soxl_soxs_14d5pct_v71/README.md)、[v70 数据底座](research/soxl_soxs_14d5pct_v70/README.md) 和全部失败记录保留。

此前 [v59 行业状态一日预测偏离](research/soxl_soxs_14d5pct_v59/README.md)：用过去拟合的状态预测构造创新值，与简单差分、相同样本原输入对照。状态预测误差下降，但主路线开发月均 −0.78%、三月 −22.74%，仍未达标；方向误差没有改善。19 项测试、状态／收益模型及九账本核验通过，未执行新指定月份回归。[持续目标进行中](research/soxl_soxs_14d5pct_v59/GOAL_PROGRESS.md)。

此前 [v58 基础模型跨时点共享](research/soxl_soxs_14d5pct_v58/README.md)：每日联合拟合部分共享／匹配独立／完全共享，保留原十四日目标与日期权重。主路线开发月均 +0.19%，三月 −14.99%、四月 −3.77%，未达标；预测误差降低未转化为实际收益。21 项测试、全部模型／九账本核验通过，未执行新指定月份回归。[持续目标进行中](research/soxl_soxs_14d5pct_v58/GOAL_PROGRESS.md)。

此前 [v57 下一交易日短反馈](research/soxl_soxs_14d5pct_v57/README.md)：基础模型学习短路径、仍校准并执行原十四日合同，保留相同目标的延迟反馈对照；开发月均 −0.30%，二月 −10.66%、三月 −7.59%，未达标。及时反馈平均多 0.62 个训练日期，短方向误差略差，整体误差的小改善未转化为资金收益。20 项测试、全部模型／八账本核验通过，未执行新指定月份回归；下一步检查基础参数跨时点共享。[持续目标进行中](research/soxl_soxs_14d5pct_v57/GOAL_PROGRESS.md)。

此前 [v56 双边 ETF 与行业交易活跃度](research/soxl_soxs_14d5pct_v56/README.md)：当时分钟交易活动及价格交互共六字段，三组逐日基础模型与原 v55 匹配对照；开发月均 +5.04%，三月 −22.74%，仍未达标。21 项测试、全部模型／九账本核验通过，新增字段的方向误差恶化，相对旧模型资金增量区间跨零；未启动新指定月份回归。逐笔方向核查表明基础预测与校准均有错误，下一步核查短周期价格路径学习。[持续目标进行中](research/soxl_soxs_14d5pct_v56/GOAL_PROGRESS.md)。

此前 [v55 时点数量与校准收缩归一](research/soxl_soxs_14d5pct_v55/README.md)：按固定 80/k 公式保持复制时点不增惩罚；输入与原始分数不变，开发月均 +3.78%，二月 −5.20%、三月 −20.19%，仍未达标。24 项测试、全部模型／九账本核验通过，v54 及原四时点精确复现；未运行新指定月份回归。后续转向自身 ETF 交易活跃度的因果输入核查。[持续目标进行中](research/soxl_soxs_14d5pct_v55/GOAL_PROGRESS.md)。

此前 [v54 补齐日内决策时点](research/soxl_soxs_14d5pct_v54/README.md)：固定新增四个时点，逐日重新训练并保留旧四时点对照；开发月均 +1.62%、二／三月亏损，27 笔交易，未达标。20 项测试、全部模型／八账本核验和旧路线精确复现通过；未启动新指定月份回归。合成检验定位到时点数量改变相对收缩强度，后续另版验证，不改本次失败结果。[持续目标进行中](research/soxl_soxs_14d5pct_v54/GOAL_PROGRESS.md)。

此前 [v53 允许行业状态与方向可靠性呈弯曲关系](research/soxl_soxs_14d5pct_v53/README.md)：固定中点的新模型改变全部 328 个查询分数，但相对旧线性模型没有动作／资金增量；开发月均 +4.01%、三／四月亏损，Brier 恶化至 0.26101，未达标。24 项测试、完整方程／十账本核验及四组旧对照通过；未启动新指定月份回归。[持续目标进行中](research/soxl_soxs_14d5pct_v53/GOAL_PROGRESS.md)。

此前 [v52 用可见行业状态调整方向可靠性](research/soxl_soxs_14d5pct_v52/README.md)：行业状态改变分数及三个建议，但被持仓／同日限制拦截，实际成交与 v51 完全相同；开发月均 +4.01%，三／四月亏损，未达标。23 项测试及完整方程／十账本核验通过，旧对照精确复现；未启动新指定月份回归。[持续目标进行中](research/soxl_soxs_14d5pct_v52/GOAL_PROGRESS.md)。

此前 [v49 保留当时预测的方向纠偏](research/soxl_soxs_14d5pct_v49/README.md)：固定原方向估计，分别学习时点偏差与期权增量，联合重抽样。开发月均 −1.88%，二月 −11.83%、三月 −7.60%，方向误差也恶化。15 项测试、全部纠偏／校准／九份资金账本核验通过；主路线失败，未启动新指定月份回归。[持续目标进行中](research/soxl_soxs_14d5pct_v49/GOAL_PROGRESS.md)。

此前 [v48 期权与市场状态的条件方向模型](research/soxl_soxs_14d5pct_v48/README.md)：固定浅层树与线性 × 有／无期权四组比较。开发月均 +2.14%，二月 −5.69%、三月 −4.97%，仍未达标；新字段确实进入条件路径，但原始方向误差恶化。15 项测试、旧控制精确复现、全部模型／九份账本核验通过，未启动新指定月份回归。[持续目标进行中](research/soxl_soxs_14d5pct_v48/GOAL_PROGRESS.md)。

此前 [v47 期权活跃度与方向信息](research/soxl_soxs_14d5pct_v47/README.md)：复用二十份真实历史文件，只为每日方向模型加入八项延迟期权指标。开发月均 +2.03%，二月 −3.80%、三月 −10.37%，原始方向误差未改善；五天延迟也失败，未启动新指定月份回归。18 项测试、完整模型／动作／账本核验通过；原核验器类型问题与修复均保留。[持续目标进行中](research/soxl_soxs_14d5pct_v47/GOAL_PROGRESS.md)。

此前 [v46 固定强收缩与原定月份回归](research/soxl_soxs_14d5pct_v46/README.md)：开发月均 +10.68% 精确复现后延续逐日训练；五／七／九月均值仅 +1.68%、六月 −1.69%，仍未达标。三次建议的全部八种过滤组合均无法达标；420 查询均有模型，方向误差占主要部分。12 项测试、完整前缀、开发／回归及八份子集账本核验通过；下一步核查新的历史方向信息。[持续目标进行中](research/soxl_soxs_14d5pct_v46/GOAL_PROGRESS.md)。

此前 [v45 每日学习宏观校准惩罚](research/soxl_soxs_14d5pct_v45/README.md)：先定位共享宏观修正的入场高估，再用成熟历史误差逐日选择四档强度。主路线一至四月月均 +6.37%、二三月亏损，未达标；固定强收缩对照四个月均为正、月均 +10.68%，将另立候选核验原定月份。24 项测试、独立拟合／选择及九份账本通过；当前没有新指定月份回归或买入准入。[持续目标进行中](research/soxl_soxs_14d5pct_v45/GOAL_PROGRESS.md)。

此前 [v44 宏观条件残差校准](research/soxl_soxs_14d5pct_v44/README.md)：四组匹配对照隔离基础模型与校准层输入，主路线一至四月月均 −2.18%，一／三／四月亏损，未达标。20 项测试、九份账本、旧模型精确复现及原始分钟成本路径核验通过。条件校准提高了实际入场的乐观偏差；下一步先诊断复杂度和可执行机会差异，未启动新指定月份回归。[持续目标进行中](research/soxl_soxs_14d5pct_v44/GOAL_PROGRESS.md)。

此前 [v43 市场环境与半导体状态交互](research/soxl_soxs_14d5pct_v43/README.md)：固定 24 项训练期标准化交互，主路线一至四月月均 −4.68%、二月 −20.56%，三类交互均未达标。18 项测试、九份账本及 v42 无交互完整复现通过；新增诊断显示实际入场的校准高估，下一步单独检验宏观条件校准。[持续目标进行中](research/soxl_soxs_14d5pct_v43/GOAL_PROGRESS.md)。此前 [v42 跨资产风险信息](research/soxl_soxs_14d5pct_v42/README.md)：补齐六只 ETF 的原始日线与公司行动，以决策前收盘构造八项股票／债券风险输入。主路线一至四月月均 −1.99%，二月 −19.86%、三月 −7.36%，未达标；股票输入对照 +5.08% 仍弱于原方法。没有新指定月份回归。[持续目标进行中](research/soxl_soxs_14d5pct_v42/GOAL_PROGRESS.md)。此前 [v41 成熟标签重叠加权](research/soxl_soxs_14d5pct_v41/README.md)：分别检验基础训练与校准的重叠权重，主路线一至四月月均 −1.45%、二三月亏损，未达标。删掉十九笔实际交易，其中十四笔原本止盈；下一步优先核查新增行业外风险输入。17 项测试、九份账本和原模型完整复现通过，未启动新指定月份回归。[持续目标进行中](research/soxl_soxs_14d5pct_v41/GOAL_PROGRESS.md)。此前 [v40 金额加权非线性方向学习](research/soxl_soxs_14d5pct_v40/README.md)：浅层树主路线一至四月月均 −1.95%，二月 −11.81%，未达标。33.9 万项基础／树叶方程、14.5 万项校准方程和八份账本核验通过；新证据显示全部建议平均为正、实际可执行机会为负，下一步核查执行条件与学习目标。[持续目标进行中](research/soxl_soxs_14d5pct_v40/GOAL_PROGRESS.md)。此前 [v39 每日扩展与指定月份回归](research/soxl_soxs_14d5pct_v39/README.md)：开发月均 +10.45% 后另立主候选，完成每日源模型扩展与核验；原定五／七／九月均值仅 +0.047%，六月 −7.73%，未达标。三次源建议的全部八种过滤组合均不能达标，卡点回到基础机会识别。[持续目标进行中](research/soxl_soxs_14d5pct_v39/GOAL_PROGRESS.md)。此前 [v38 合格候选与输入范围保护](research/soxl_soxs_14d5pct_v38/README.md)：主路线仍有三月亏损；登记对照 all_supported 一至四月全部为正、月均 +10.45%，将另立候选验证原定月份。30 项测试、31,800 个模型方程、十份账本核验通过。[持续目标进行中](research/soxl_soxs_14d5pct_v38/GOAL_PROGRESS.md)。此前 [v37 第一名盈亏金额加权风险模型](research/soxl_soxs_14d5pct_v37/README.md)：学习第一名实际盈亏并检验行业／财务条件，一至四月主路线月均 +4.37%，二／三月亏损，未达标。过滤掉六笔止盈和一笔止损；定位到全部第一名与源合格候选的训练条件差异。16 项测试、36,000 个拟合方程和八份账本通过，未运行新指定月份回归。[持续目标进行中](research/soxl_soxs_14d5pct_v37/GOAL_PROGRESS.md)。此前 [v36 每日学习校准修正幅度](research/soxl_soxs_14d5pct_v36/README.md)：按当时成熟误差调整四时点 C／D 修正并保留原风险余量，一至四月主路线月均 +10.21%，三月仍 −3.05%，未达标。总体误差略降，只改变两次建议，三月止损未避开；17 项测试、144,000 项拟合等式和八份账本核验通过，未运行新指定月份回归。[持续目标进行中](research/soxl_soxs_14d5pct_v36/GOAL_PROGRESS.md)。此前 [v35 已披露财务状态与业绩事件](research/soxl_soxs_14d5pct_v35/README.md)：以当时假定可用的营收、利润变化及披露日期重训模型，一至四月主路线月均 +10.21%，但三月 −3.05%，仍未达标。匹配历史原输入 +5.85%；新增财务输入降低原始误差，现有校准使误差回升。17 项测试、21,512 次事实引用及八份账本核验通过，未运行新指定月份回归。[持续目标进行中](research/soxl_soxs_14d5pct_v35/GOAL_PROGRESS.md)。此前 [v34 行业同步与领涨扩散](research/soxl_soxs_14d5pct_v34/README.md)：以当时可知的相关性、参与度和落后股追赶重训模型，一至四月主路线月均 −0.35%，二／三月亏损，仍未达标。新增指标没有降低基础预测误差；原输入重训逐笔复现旧模型。14 项测试、32,098 项特征等式和八份账本核验通过，未运行新指定月份回归。[持续目标进行中](research/soxl_soxs_14d5pct_v34/GOAL_PROGRESS.md)。此前 [v33 成交成本情景学习](research/soxl_soxs_14d5pct_v33/README.md)：每日学习两种成本下较差收益，一至四月月均 +0.68%、二月 −9.01%，弱于原模型，仍未达标。建议减少却改变后续资金路径，新增四笔实际入场中三笔止损。12 项测试、8 份账本及 5,748 条原始成本路径核验通过，未运行新指定月份回归。[持续目标进行中](research/soxl_soxs_14d5pct_v33/GOAL_PROGRESS.md)。此前 [v32 四时点部分共享学习](research/soxl_soxs_14d5pct_v32/README.md)：允许时点差异并向共同关系收缩，一至四月主路线全部非负、月均 +5.54%，双倍成本月均 +5.04%，仍不足 10%；未优于匹配的独立模型。28 次建议、16 笔实际入场，三月没有合格建议。12 项测试、8 份账本和 210,000 项模型方程核验通过，未运行新指定月份回归。[持续目标进行中](research/soxl_soxs_14d5pct_v32/GOAL_PROGRESS.md)。此前 [v31 按日期共享学习与行业状态交互](research/soxl_soxs_14d5pct_v31/README.md)：四时点按日期计权，状态改变收益修正斜率；一至四月主路线月均 −2.61%，一二月亏损，仍未达标。旧分时模型有 12/82 天修正方向不同，下一步检验允许时点差异的收缩共享。13 项测试、8 份账本及 210,000 项模型方程通过，未运行新指定月份回归。[持续目标进行中](research/soxl_soxs_14d5pct_v31/GOAL_PROGRESS.md)。此前 [v30 收盘前盈利缓冲与重新训练](research/soxl_soxs_14d5pct_v30/README.md)：经验跨收盘尾部损失高于 +5% 止盈上限，471 次复评全部退出，退化为日内合约。主路线一至四月月均 −2.07%、二月 −8.27%；旧模型加退出对照月均 +7.87% 仍不足 10%，未达标。17 项测试与 8 份账本核验通过，未执行新指定月份回归。[持续目标进行中](research/soxl_soxs_14d5pct_v30/GOAL_PROGRESS.md)。此前 [v29 在线行业状态与历史记忆](research/soxl_soxs_14d5pct_v29/README.md)：因果变点滤波调整样本权重，一至四月均非负但月均仅 +3.82%；新增四月入场遭遇跳空净亏 −13.29%，双倍成本两个月亏损，仍未达标。11 项测试、7 份账本及 110 条原始退出核验通过，未启动新指定月份回归。[持续目标进行中](research/soxl_soxs_14d5pct_v29/GOAL_PROGRESS.md)。此前 [v28 未完成路径与竞争风险学习](research/soxl_soxs_14d5pct_v28/README.md)：每天使用当时已知的价格前缀，未知结果保留右删失。一至四月主路线月均 +2.20%，二月 −6.15%，4 笔；与仅用已退出样本相比没有动作增量，仍未达标。16 项测试、7 份独立账本通过，未启动新指定月份回归。[v28 进展记录](research/soxl_soxs_14d5pct_v28/GOAL_PROGRESS.md)。此前 [v27 分时机会与入场可靠性](research/soxl_soxs_14d5pct_v27/README.md)：四时点分别学习误差，完整账本一至四月 +3.29%、+11.91%、0%、+15.43%，月均 +7.66%，13 笔，仍低于 10%；双倍成本二月亏损。10 项边界／记录测试及 9 份账本核验通过，未启动新指定月份回归。[v27 进展记录](research/soxl_soxs_14d5pct_v27/GOAL_PROGRESS.md)。[v26 入场误差与日期分块](research/soxl_soxs_14d5pct_v26/README.md)未达标结果保留；[v25 买入与等待](research/soxl_soxs_14d5pct_v25/README.md)失败结果保留；[v24 联合盈亏概率](research/soxl_soxs_14d5pct_v24/README.md)失败结果保留；[v23 盘前与盘中成对收益](research/soxl_soxs_14d5pct_v23/README.md)失败结果保留；[v22 按当前行业状态训练](research/soxl_soxs_14d5pct_v22/README.md)开发失败结果保留；[v21 基础模型提前学习](research/soxl_soxs_14d5pct_v21/README.md)及 [v20 提前反馈](research/soxl_soxs_14d5pct_v20/README.md)和 [v19 按行业状态选择预测来源](research/soxl_soxs_14d5pct_v19/README.md)及 [v18 每日误差组合](research/soxl_soxs_14d5pct_v18/README.md)保留旧结果；[v17 风险筛选](research/soxl_soxs_14d5pct_v17/README.md)及 [v16 均值约束](research/soxl_soxs_14d5pct_v16/README.md)旧结果不变。[v15 条件收益分布与仓位](research/soxl_soxs_14d5pct_v15/README.md)、[v14 行业状态与成对收益](research/soxl_soxs_14d5pct_v14/README.md)、[v13 正则化与校准](research/soxl_soxs_14d5pct_v13/README.md)、[v12 收益分解](research/soxl_soxs_14d5pct_v12/README.md)、[v11 退出不确定性](research/soxl_soxs_14d5pct_v11/README.md)及 [v10 盘中复评](research/soxl_soxs_14d5pct_v10/README.md)、[v9 持仓学习](research/soxl_soxs_14d5pct_v9/README.md)、[v8 等待价值](research/soxl_soxs_14d5pct_v8/README.md)、[v7 行业广度](research/soxl_soxs_14d5pct_v7/README.md)、[v6 盘中路径](research/soxl_soxs_14d5pct_v6/README.md)及 [v5 多模型](research/soxl_soxs_14d5pct_v5/README.md)保留；[v4 每日训练及任意时点分析](research/soxl_soxs_14d5pct_v4/README.md)继续可用，当前不行动。

[v2 历史结果](research/soxl_soxs_14d5pct_v2/REPORT.md)的三个月算术均值 +11.69% 不再代表当前准入；旧数字与亏损保留。[v1 迁移记录](research/soxl_soxs_14d5pct_v1/ANALYSIS.md)保留原模型失败证据。原始行情不进 Git，真实交易仍由用户手动执行。

## 盘前 / 盘中当天 +3%：云端主动问一次

在仓库根目录运行，**三个冻结模型各返回概率最高的三个参考选项**：

```bash
python3 -m pip install -r analysis/preopen_ranked_policy_v5/requirements-cloud.txt
python3 analysis/preopen_ranked_policy_v5/recommend.py --top 3
```

命令只运行一次，主动检查 Local → R2 → 可访问 OpenD，列出每个选项的概率、置信度门槛、行情截止、延迟入场和剩余常规盘 +3% 达标条件。行情不足或闭市时明确显示旧参考，不制造当前有效信号。可移植十月模型随代码提供，原始行情不进 Git、不自动上传 R2。完整云端行情配置、输出字段与本地调试服务见 [研究 README](analysis/preopen_ranked_policy_v5/README.md)，五月至九月 23 周表见 [周度报告](analysis/preopen_ranked_policy_v5/weekly_v1/REPORT.md)。

新增两个跨日目标：**5 个交易日 +5%、10 个交易日 +10%**。每份报表同时写出路线及算法（LogisticRegression / LightGBM / ExtraTrees），目标分别训练与校准。一次输出全部目标、每条路线各前三名：

```bash
python3 analysis/preopen_ranked_policy_v5/recommend.py --target all --top 3
```

R2 配置和凭证已随仓库保存在 `config/r2_storage.json`，无需猜测或另复制配置。先执行 `python3 scripts/check_r2.py --timeout 5` 检查一次只读访问。命令默认给 R2/OpenD 补数据30秒总预算，可用 `--data-timeout 60` 调整；全部目标共用行情与特征，网络不可达时及时生成明确的历史参考或缺失报告。`--target all --top 3 --offline` 可单独验证九个模型的运行。配置可用不等于行情新鲜，详见[云端说明](analysis/preopen_ranked_policy_v5/README.md#最新行情接入和不足处理)。

完整两个23周表见 [跨日报告](analysis/preopen_ranked_policy_v5/horizon_v1/REPORT.md)。六组新回测整体55.12%–69.56%，未通过70%；尚未成熟和不可评分的有效信号保留。新目标十月仅提供研究参考概率，未获准自动发出有效信号。


## 新工程：开盘后 3 日 +5% 概率模型

入口：[`research/after_open_3d5pct/README.md`](research/after_open_3d5pct/README.md)。面向手动交易，在 11:30 ET 及后续可决策时点，研究从延迟后入场价起算的剩余上涨空间。已建立 coder 必读契约、时间/标签/切分骨架、边界测试与离线合成 smoke；真实数据适配和模型训练尚待后续里程碑完成。

## AI 股票七策略基础回测

### 开盘前、三周 +20% 单策略审计

在四只 AI/机器人/云/数据中心 ETF 的日期化成分中形成可回溯股票池；用已披露 SEC 年报筛选后，对下一交易日开盘入场、15 个交易日 +20% 目标与 -10% 风险线做训练、验证、独立测试：

```bash
python3 preopen_three_week.py
python3 render_preopen_dashboard.py
```

[交互式逐股 K 线和历史买点](analysis/preopen_three_week/dashboard.html)以及[方法与结果](analysis/preopen_three_week/report.md)。这版独立测试期平均交易净收益和组合收益为负，未通过质量门槛，当前不输出可执行的买卖建议。`results.json` 保留原始研究分数与模拟明细，便于检查失败原因。历史基金快照由 `fetch_ai_fund_holdings.py` 获取，最新富途收盘快照由 `fetch_futu_snapshot.py` 获取；提交的数据文件支持本地复算。

模型脚本需要 NumPy；只有刷新富途快照时才需要 `futu-api` 客户端和运行中的本机 OpenD。默认复算使用已保存的快照，不需要连接富途。

回测代码是 [`backtest_ai_strategies.py`](backtest_ai_strategies.py)，可直接用仓库内的行情归档与 [`data/ai_sec_annual_facts.json`](data/ai_sec_annual_facts.json) 重算：

```bash
python3 backtest_ai_strategies.py
```

中文结果位于 [`analysis/ai_strategies_baseline/report.md`](analysis/ai_strategies_baseline/report.md)，结构化指标与审计明细位于同目录。99 只 AI 相关候选股的业务组写在 [`ai_universe_candidates.json`](ai_universe_candidates.json)；每天再用当时已提交的 SEC 年报营收、经营现金流、净利润和历史成交额筛选。2026-05-01 通过门槛的名单在 `qualified_universe_2026-05-01.csv`。策略 4 需要完整且带首次发布时间的事件快照，输入契约见 [`event_feed.schema.md`](analysis/ai_strategies_baseline/event_feed.schema.md)。

如需刷新 SEC 年报原始快照，先按 SEC fair access 要求设置包含机构名称与有效联系地址的 `SEC_USER_AGENT`，再运行 `python3 build_ai_sec_facts.py`。刷新会改变可复算的数据版本；保留旧快照后再比较结果。

这是一个“候选生成器”，用于每天扫描美国普通股：先在 7 个完整交易日内显著下跌，再在随后 7 个完整交易日内进入低波动平台。它输出形态候选，不直接输出买入建议或收益概率。

## 为什么使用 15 根 K 线

原始示例用 14 根 K 线分成 7+7，但前半段 `index 0 → index 6` 只有 6 个收盘到收盘的收益区间，还漏掉 `index 6 → index 7` 的边界变化。这里使用 15 根：

- `C0 → C7`：7 个下跌期收益区间；
- `C7 → C14`：7 个平台期收益区间；
- `C7` 同时作为平台锚点，第一天再次跳空下跌时不会被误判为横盘。

## 默认筛选条件

- 最新价：$50–$200；
- 7 个交易日跌幅：18%–45%；
- 后续平台总高低区间：≤7%；
- 平台拟合趋势绝对值：≤3.5%；
- 平台日收益标准差：≤2.2%；
- 平台任一日绝对涨跌：≤3.5%；
- 平台平均成交额：≥$20M；
- 默认仅包含 Massive 类型 `CS`（美国普通股），排除 ETF、权证、优先股等。

阈值有意比“4% 总振幅”稍宽，先保证召回率，再用 `shape_score` 排序。`shape_score` 只描述形态质量，不是回涨概率。

## 运行

1. 申请 Massive Stocks API key。免费 Basic 档即可做 EOD 原型。
2. 将 `.env.example` 复制为 `.env`，把新 key 只写入本地 `.env`。`.env` 已被 Git 忽略。
3. 执行：

```bash
python3 screener.py run --config config.example.json
```

也可以不使用 `.env`，直接通过运行环境注入 `MASSIVE_API_KEY`。

第一次运行需要下载至少 15 个交易日的全市场日线和股票目录。默认将未缓存 API 请求间隔设为 12.2 秒，以兼容免费档 5 次/分钟的额度：首次初始化通常需要约 4–6 分钟。股票目录缓存复用 7 天；稳定运行后，绝大多数交易日只需 1 次全市场行情调用，目录刷新日再增加若干分页调用。

输出位于 `results/`，其中固定入口用于自动读取，日期目录用于历史追踪：

- `results/latest.md`：最近一次人类可读报告；
- `results/latest.json`：最近一次机器可读报告；
- `results/YYYY-MM-DD/screen.md`：当天归档 Markdown；
- `results/YYYY-MM-DD/screen.json`：当天完整结构化数据；
- `results/YYYY-MM-DD/candidates.csv`：候选明细。

## 全市场每日归档

筛选结果和源行情数据分开保存：

- `.cache/massive/bars/YYYY-MM-DD.json`：运行缓存，只用于减少 API 请求，不属于长期数据资产；
- `market_data/us/YYYY-MM-DD/grouped.json.gz`：Massive grouped daily 接口的完整原始 JSON 响应，经确定性 gzip 压缩后提交进 Git；
- `market_data/us/manifest.json`：归档索引，记录交易日、ticker 数量、未压缩/压缩大小与原始数据 SHA-256。

每日 workflow 在筛选完成后执行：

```bash
python3 archive_market_data.py \
  --all-cached \
  --results-json results/latest.json \
  --cache-dir .cache/massive \
  --archive-dir market_data/us \
  --min-results 1000
```

`--all-cached` 会把 Actions cache 中已有且通过完整性校验的历史交易日一并回填。当天 `results/latest.json` 对应的交易日必须成功归档，否则 workflow 失败。默认要求至少 1,000 个 ticker，防止接口异常或部分响应被误写成完整市场历史。

归档保存的是 Massive `/v2/aggs/grouped/locale/us/market/stocks/{date}` 的完整响应，而不是当前筛选器抽取后的字段。因此以后新增 VWAP、交易笔数、ETF/普通股重新分类或重新计算筛选参数时，可以直接读取历史源数据，无需重新请求行情接口。

运行测试：

```bash
python3 -m unittest -v test_screener.py test_archive_market_data.py
```

## 每日工作流

推荐分两层：

1. 本代码在美股收盘且 EOD 数据就绪后运行，生成候选列表，并把完整当日全市场源数据归档；
2. ChatGPT 只研究候选股的暴跌原因、消息是否破坏长期逻辑、估值、量化业务权重、未来 30/60 日催化与下降旗形风险。

先手动检查前 3–5 次输出，再开启无人值守任务。正式用资金前，应增加两年回测，至少比较：20 日内先涨 8% 的概率、20 日收益中位数、最大不利波动，以及相对“所有同期跌 18% 的股票”的增量优势。

## GitHub Actions

仓库 Secret 中添加 `MASSIVE_API_KEY`。任务在周二至周六 05:30 UTC 执行（美东午夜后），等待 Basic 免费版释放前一交易日 EOD 数据。若某个日期尚未释放，脚本会跳过并寻找最近可用交易日。报告和 `market_data/us/` 归档由 `github-actions[bot]` 一起提交回仓库。`results/**` 与 `market_data/**` 被 workflow 的 push trigger 忽略，避免机器人提交再次触发自身。

后续 ChatGPT 可以固定读取 `results/latest.md` / `results/latest.json` 做每日候选研究，也可以读取 `market_data/us/manifest.json` 和对应日期的 `grouped.json.gz` 做历史窗口、参数敏感性和回测分析。

`daily_research_prompt.md` 已包含第二层研究的固定提示词，重点防止把漂亮形态误当成上涨概率，并保留“业务权重必须量化”的要求。
