# v9 共用前向证据准备：实际状态

核对日期：2026-09-27。里程碑仅为可执行设计与代码边界审查；没有实现、运行采集、启动调度、预测发布或交易。下轮必须先更新本状态、BACKLOG 与 MATRIX，再登记执行，结束后另存 REVIEW。

## 不变的目标

- 五路线至少三条，各自在 chips/optics/storage 三个固定操作群，实际发出 buy 的完整成熟 `3d_5pct` precision≥90%，每群 `unique_buy/(official_sessions/5)≥1`。
- 11:30 ET 特征截止，11:35 的 5m Open 代理入场；1170 RTH 分钟内 High 触及 +5%。九目标维持 `focus_v8/protocol.json`；High touch 不等于真实成交或盈利。
- `PROTOCOL.md` 的连续至少60官方sessions、每群30成熟信号和20信号日期，是根代理预登记的工程门槛，不能说成用户要求、降低或偷偷更换。
- 全部2026为 exposed development；R03扩2025历史仍只支持开发。当前0路线通过最终验收，历史补写不能变成当时预测。

## 代码事实

| 位置 | 实际能力与缺口 |
|---|---|
| `../data.py::build`，419–423行 | 先调用 `_outcomes`，完整390根5m/九标签不存在便跳过；今天特征不能由此产生。需要独立 feature-only 路径，不能填假标签绕过。 |
| `../data.py::load_symbol/inputs_available` | 支持 archived availability，缺 received_at 仍可进入；预先编码完整源，重复bar直接报错。实时需先选截止前实际收到的版本，再编码，不能只检查最后一根。 |
| `../data.py::_group_map/_group_state` | 因果周重建与当前池用于开发；版本 effective/feature_cutoff 检查不是当时真实发布/收到证据。群分支依赖的同伴、QQQ与元数据也须逐依赖审计。 |
| `../baseline.py::run/_selection` | R03开发重放，输出模型、校准、群阈值及结果；借用 v8 `load_data`，后者读取 y/计算成熟先验。不是实时预测入口。Sol 正在修改，未作最终验收。 |
| `../evaluate.py::select_signals`，56–77行 | trigger_groups 与交叉成员 union 已定义；busy 是每调用新字典，不能跨run维持1170分钟禁重。 |
| `../../run_once_v3.py` | 有快照/特征分离、报告哈希及前向追加；属于另一套 H/HAB 模型，research_candidate 也不是 v9 buy。 |
| `../../hourly_v3_snapshot.py`，26–53行 | 可见版本筛选允许 received_at 缺失；watermark只核最后一根收到时间。不能直接证明全部v9实际输入均按时收到。 |
| `../../evaluate_forward_hourly_v3.py`，20–113行 | 评所有有 predictions 的行；按 run+symbol 去重，成熟子集算touch_rate；没有v9 route/trigger/cooldown/冻结cohort完整分母。不能直接用来验收v9。 |

## 三个独立门槛

| 门槛 | 本次状态 | 需要的实际证据 |
|---|---|---|
| G1 有模型可预测 | 待验证 | 每路线可加载制品，冻结通道/变换/校准/先验规则；无未来bar及标签仍得到九输出，和已存checkpoint重放一致。历史拟合成功不自动满足。 |
| G2 有真实接收与发布证据 | 未证明 | 所有所用版本 bar_end≤11:30、available_at和实际received_at≤11:30:30；冻结成员/同伴信息当时可知；实际发布早于11:35且满足冻结时延。 |
| G3 有可验证冻结cohort | 未建立 | 预登记连续官方session表、不可改预测/buy账本、跨run去重、全部已发buy及成熟/未解决分母、到期一次验收。 |

G1通过可称“可做研究推理”；G2通过可开始记录真实前向研究buy；只有G3完整且每群达标的路线才可计入最终至少三条。G2不是90%效果保证；未通过G2只能保留研究预测与拒绝，不能冒充有效buy。

## 本轮源码指纹

以下为审查时读取版本的SHA-256；运行登记必须重取并保存字节快照，不能直接拿本表充当后续sourcehash验收。

| 文件（相对工程根） | SHA-256 |
|---|---|
| precision_v9/PROTOCOL.md | `21dde0982b2f24edb006fc32f43d32b5e73f099e12b71e0408a2f07bea331a7a` |
| precision_v9/data.py | `06207586233fea7153094dbdb40e314cc7608bf5b7c96b269e21bbc083607a3e` |
| precision_v9/baseline.py（WIP，随后已观察到变化） | `d6689d3baebbd07ce6b78e1652d4b1d84f26b8ee2491b5ae250c876eec2fe508` |
| precision_v9/evaluate.py | `a09c38e57babca1c1a4f69dc688d47f7aaf4490b58cbb23e84f15527f8707795` |
| run_once_v3.py | `468fe954c0b6f1e1d68ee1b80f5626b1ae2dbab05cce98610115a50867676748` |
| evaluate_forward_hourly_v3.py | `d78e23cfbe006ff985284333ca77e1ed2323a9bac9c86bf5ec9f031c62a4151c` |
| hourly_v3_snapshot.py | `708a9c0aa44b052a40a4bb383dac1de6113e8c118a67b31f54ee76b2a9d4d603` |

已读工程 AGENTS/README、01–08、13–16、v9协议/README及上列代码。此设计不修改五路线研究方案，不触碰R03采集与原制品；后续代码由根代理安排Sol。
