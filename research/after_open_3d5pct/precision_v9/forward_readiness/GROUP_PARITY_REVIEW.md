# 带群 feature-only 配对失败复盘与采用决定

2026-09-27；只读审查，不训练、不采集、不调度。范围：真实 v8 冻结输入、v6/v9 群函数、Sol 的真实配对测试；该诊断不构成独立效果验证。

## 实际状态

- **三条带群路线 feature-only G1 配对 FAIL**：`B_group/C_group/C_no_daily`。五路线静态 checkpoint 复放、两无群 feature-only 配对仍成立；G2/G3 未证明，用户 90% 目标仍 0 模型通过。
- AMD / 2026-06-01，107 个实际依赖证券、58 条当时已成熟本股先验；六代表周 W18–W23。Sol 最终全链路 21.35 秒（三路线各构建约 6.8–6.9 秒），仅单证券离线 fixture，不能证明全池 30 秒时限。
- `x5/x60/xday/prior/paths/curve` 最大差均 0；`group_seq=0.511668`、`xbase=0.509106`、B_group `relative=0.011355`。
- B_group 九目标 raw/p 最大差 `0.0115003/0.0124992`；C_group `0.000236176/0.000270019`；C_no_daily `0.000546277/0.000479134`。这是输入/旧模型兼容性差异，不能推断三模型改善或退化。
- 分槽诊断：liquidity 最大差 `0.511668` 已定位；trend_15 槽 0 差 `0.062193`、trend_63 槽 1 差 `0.023631` 尚待定位；trend_126/volatility/QQQ 差 0。**不能把全部失败归因于 liquidity。** Sol 已核对 223 个冻结源 SHA 一致，107 来源六代表日的 cutoff_return/cutoff_rvol 均与旧加载器相同，初步趋势 peer 集合相同；仍须定位实际群图差异。
- 复现入口：[test_group_parity.py](../forward/test_group_parity.py)；执行证据由 Sol 的 [GROUP_PARITY_REVIEW](../forward/GROUP_PARITY_REVIEW.md) 保留。新增失败必须与此前通过的 32 项分别报告，不能写成全部通过。

## 已证实的 liquidity 根因

- [v6_data.py](../../v6_data.py) 的 `_group_state` 只验证自身 facts；peer 循环按同群/可用源/issuer 去重，未检查 peer `history_end`。
- [v9 data.py](../data.py) 额外检查 peer record 属于该群且 `facts.get('history_end','9999') < representative_date`。旧成员缺字段时被排除，不是已查明的日期比较错误。
- 冻结 v8 的 liquidity 行：六周累计 **568 个 peer×周记录、95 个 unique symbols** 缺 `history_end`。逐周缺数 `94/95/95/94/95/95`；旧去重 peer 数 `100/101/101/100/101/101`，新筛选各周仅剩 7 个。
- 剩余恰为 `AAOI/ANET/AXTI/CIEN/COHR/CRDO/FN`。[focus_v8/prepare.py](../../focus_v8/prepare.py) 只给新增 optical 成员追加 `history_end`；旧复制行未补该字段。因此新特征已不是旧模型训练时的同一个 peer 群。
- 例 W23 AAPL / liquidity:high：facts 仅 `median_daily_turnover=12469789656.941502`；COHR 同周有 `history_end=2026-05-29`。版本 `liquidity:2026-W23:80e2d328c881` 的 cutoff 为 `2026-06-01T13:30:00Z`、`source_observed_at=null`。
- [原分类器](../../../group_expectation_matrix/build.py) `classify_history` 确实以周首日之前的 `prior_dates`、完整历史及 `available < cutoff` 计算；只有 liquidity 分支返回值漏写 `history_end`。**不能据缺字段宣称原算法已发生前视。** 冻结版本的事后 generated_at 与空 source_observed_at 也不能作为历史接收证明。
- 判定：已证实的是**旧元数据证据/字段契约与 v9 不兼容**；既无依据删除 peer 因果检查，也无依据凭意图补写日期并当作原时点证据。趋势两槽剩余差异另列待查。

## 实际 backlog

| 优先级 | 缺口与明确验收 | 负责人/依赖 |
|---|---|---|
| P0 | 逐 peer/代表周定位 trend_15、trend_63 差异：成员、版本、history_end、issuer、实际 prefix/可用值与冻结张量；分清旧重建差异、真实错误及缺证据 | 当前 Sol；无需等待采集即可只读定位 |
| P0 | feature-only 输入缺必需群字段必须审计拒绝，不能悄悄删 peer 后返回 `g1_ready` | root 已指定当前 Sol 做最小门禁 |
| P0 | 固定新 schema/群语义、版本与 membership SHA；R03 新 metadata 与同契约 checkpoint 成套验证 | 完整 R03 prepare 后；不是 bars 到齐即通过 |
| P1 | 新 dataset 批量张量与 feature-only 同时点、同成熟 prior 配对，再逐路线比九目标 raw/p | 模型冻结后独立 G1 测试 |
| P1 | 全候选 prior、本地 E2E adapter、receipt/日历与全池性能仍按集成计划验收 | [INTEGRATION_PLAN](INTEGRATION_PLAN.md)；不能借本失败跳过 |

## 方案矩阵与选择

| 方案 | 能证明什么 | 代价/门禁 | 决定 |
|---|---|---|---|
| 保留 v9 因果规则，R03 重建完整 metadata 并重训 | 新数据/新群语义与模型一致的 feature-only G1 | 先解释趋势剩余差异；新版本与模型冻结、真实配对成功；仍无 G2/G3 自动通过 | **采用** |
| 显式 legacy fixture-only 复放旧算法/张量 | 旧 checkpoint 静态工程复现，可保留历史诊断基线 | 必须命名 legacy、禁止 live/publication/G2；不冒充新 schema 配对 | 仅诊断允许，不作为前向解锁 |
| 修复已证实的 v9 误判 | 完整合法证据被错误拒绝时恢复正确输入 | 必须给出真实反例和修复前后因果测试；当前 liquidity 不属于此项，趋势仍待查 | 条件保留，不能预先宣称适用 |
| 删除检查、默认补 history_end、拿新特征直接接旧群 checkpoint | 无法证明所需一致性或时点证据 | 会掩盖输入契约变更 | **拒绝** |

按当前采用路线，**三带群 feature-only G1 必须等待 R03 新 metadata 与匹配重训模型，并关闭剩余差异**。这不阻塞两无群路线独立工程工作，也不撤销五静态 checkpoint 复放。若另行证明完整合法旧证据可原样重现，须单独冻结/审查，不能默许兼容。

## 给 Sol 的最小 fail-closed 接口

- 保留 `data._group_state` 因果排除；在 `build_features_asof` 的依赖解析后、群计算/模型调用前验证实际六代表周中已分类 own/peer（`len(group_ids)==1`）必需 facts、version/strategy/group 一致性和合法 `history_end < representative_date`。
- 缺失/畸形/不合时序元数据返回 `X=None` 与稳定拒绝分类，例如 `group_metadata_incompatible`；lineage 保留 symbol、strategy、week、代表日、version_id、字段/原因及 metadata SHA。无 `g1_ready=true`、无 model call；与收不到行情/receipt 的拒绝区分。
- `insufficient_history` 等原本未分类的空群不是畸形数据，保留既定 mask；拒绝范围只针对真实依赖，不要求无关全库每行齐备。不得用当前时间或代表日推算补字段。
- schema 应显式区分 legacy 与 v9 peer 因果语义，模型 manifest 冻结对应语义及群来源；维度相同不代表可替换。历史 fixture 仍不能被授予 live 接收证据。
- 必要测试：旧 v8 缺字段审计拒绝；已分类 own/peer 缺失、畸形、等于/晚于代表日均拒绝；合法空群可 mask；合法完整历史成功；无群路线不受影响；拒绝后不推理/不发 buy。
- 原失败 parity fixture 应保留失败证据，可改为“预期拒绝不兼容”回归测试，但不能把这种测试通过报告成三群 G1 成功。完整新输入/模型仍需单独正向配对。

## 冻结证据与限制

以下 SHA 为本次只读审查快照；并行后续改动必须重新登记，不反写此失败证据。

| 文件 | SHA-256 |
|---|---|
| v8 frozen versions.json | `fcc5c218947c666a850e1984695d1c8259b795a3514a7cbc856429a74549888b` |
| v8 frozen memberships.json | `1a23e39efd8c0e3b2a6d793e8eb0ebc0e2c7c619a17abb77d7027cabe5c86a7d` |
| v8 frozen groups.json | `2d326a6ee8768e29e12ba526eeea000d04ea0f05760426da1a7bf841292c42f3` |
| v6_data.py | `c50fee6d0a0a4a3b4ff9103d415e4d1022646ba0425f6833b1cf63bb624b60c7` |
| precision_v9/data.py | `06207586233fea7153094dbdb40e314cc7608bf5b7c96b269e21bbc083607a3e` |
| forward/features.py（门禁修复前） | `65fedae4ddcc2cda359846e99f019b175c01a73ac8eaa0251d2e4b1208002de0` |
| group_expectation_matrix/build.py | `cdcc28163beb61ddef68a9eb9e940ff46d5963d0c25d956198cccdb65a7c9d93` |
| focus_v8/prepare.py | `ab99c7c8296986f2e190da76f6637a65661c049c10dab9dcc4d1f6ad966a4780` |
| forward/test_group_parity.py | `44b12471670901e16d425da2e13a6bd99fe2e1219e0340515b5d86f6c32a8398` |
| forward/GROUP_PARITY_REVIEW.md（已读） | `abade5dffaf4de07d762bb03f43e44fc547ca0e7f26acf4a3fe20eff8b5c0746` |

九目标、主目标 `3d_5pct`、官方 session 全分母、1170 RTH 分钟同股去重均不变；历史回填不是当时预测，High touch 不是成交。60 sessions/30 signals/20 signal dates 是 root 工程预登记门槛，不是用户原始数字。
