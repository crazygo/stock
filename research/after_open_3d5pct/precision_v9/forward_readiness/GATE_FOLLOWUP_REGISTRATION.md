# 群门禁补充登记（先于修复）

2026-09-27；root 已采用立即修复。范围仅已支持群 ID、真实依赖 classified 版本完整性、完整 candidate 集合绑定；由 adapter owner Sol 实现，R03M owner 只做新入口。本文件及独立复盘代理不改代码/源/模型。

修前源码 SHA：features `810c7fd9f7c7d996534e02ed7c42d549a1f18c5754a14e8a78c4f599b418a5c1`；local_adapter `403ea3246e2b346e0c83eb2d7216726b87de1916ec4e4dddf451b0988696fa4c`；inference `e2a0667c5d44cc62c5760a10fa6d5571f7c236e78488d4b2f5d5cf349bb9ded3`。已有 prior seal 修复保留；source snapshot/登记 SHA 由实现者在动手前冻结。

## 已运行的最小反例

从 `forward.test_group_gate` 导入 `_fixture, _build`，每例重新 `_fixture()`；它有 AMD/COHR candidates、独立 QQQ、合法 W23 trend15:up / history_end=05-29、06-01 prefix。无训练数据/模型或网络依赖。

| 修改 | 本次实际输出 | 应验收 |
|---|---|---|
| 所有 memberships 的 group_ids 改为 `['trend_15:unregistered_category']` | `rejections=[]`、X 非空、trend15 mask=1 | `group_metadata_incompatible`，原因明确非法已支持策略类别，不调用模型 |
| 仅 AMD classified 行 version_id 改成 `missing_version` | `rejections=[]`、X 非空、trend15 mask=0 | 真实依赖 own 未绑定合法版本必须显式拒绝，不能解释成 insufficient_history |
| 仅 COHR 同群 classified 行 version_id 改成 `missing_version` | `rejections=[]`、X 非空、trend15 mask=0 | 真实依赖 peer 未绑定合法版本必须显式拒绝，不能悄悄删 peer |
| 仅 metadata.universe_symbols 改为 `['AMD']` | `rejections=[]`、X 非空、trend15 mask=0 | features 本身无完整外部 universe 依据；adapter 必须将此列表绑定完整 prior candidate 集合后拒绝缩池 |

前三例已执行直接 features 反例；第四例直接 features 的数值行为已执行，adapter 缺绑定由 `load_local_bundle` 源码确认：当前仅检查 group 文件类型，未将 universe_symbols 与已验的 full_members 比较。当前 adapter 已挡住未知 version 的孤立引用，但这不能替代 features 自身契约。

## 最小实现与测试范围

1. 固定支持集合：trend_15/63/126 各仅 `up/range/down`；volatility/liquidity 各仅 `low/medium/high`。验证实际依赖 classified 的 group_id 与 strategy 的完整组合，不能只做 startswith。QQQ 保持 benchmark view，不作为 candidate peer。
2. 在会丢弃成员的 allowed_ids/version 过滤之前，保留并验证六代表周实际依赖 classified own 与同群 candidate peer 的版本引用及 week/strategy 一致性。未知引用或不一致要审计拒绝；合法尚未生效版本、非依赖历史行、明确未分类空 group 的既有语义不可误拒。有效期/时点仍由原因果规则限制，禁止猜版本、补日期或选后来的版本。
3. adapter 用已验证且与 PriorStore registration 绑定的 `prior_universe` 取 **role=candidate** 集合；群 universe 必须是同一完整集合，缺失/额外/重复/非法标识拒绝。真实任务为全部108 candidates，不硬编码108妨碍小型 fixture。QQQ/其他 benchmark 仅依 role 排除，QQQ 作为独立行情依赖允许；不能因此作为 peer 或偷偷增加 candidate。
4. 群 universe 文件自身包含 benchmark/QQQ 时应与声明契约一致地拒绝额外成员；features 对 QQQ 不能成为 peer 的现有防御保留。不得把 forecast subset 或 required_market_dependencies 当作完整候选群池。
5. 对 adapter 增加完整三列表 fixture：full prior candidates AMD/COHR（含一个 role=benchmark 的 QQQ 原始 universe 行），forecast 仅 AMD；群 universe 缺 COHR、多任意股票、重复成员、混入 QQQ 均拒绝，正确 AMD/COHR 可加载并保留 QQQ 独立行情依赖。文件与 manifest SHA 全部按被测输入重新合法注册，不能只触发旧文件 hash mismatch 就算集合门禁测到。
6. 补前三个直接 features 回归（own/peer 分开）；合法分类、明确 insufficient_history、无群、benchmark 槽不受影响。拒绝必须 X=None 并保留 metadata hash/成员/周/strategy/version/field/reason；adapter 不得给被拒候选虚构概率或 run_empty。

独立复盘继续等修后源码/定向结果。合成测试绿只证明门禁；三带群真实 G1、R03M 全键兼容证书、真实接收 G2 和效果 G3 仍各自未通过。本登记不改 R03M 四件、原 INTEGRATION_PLAN、旧失败诊断或日历状态。
