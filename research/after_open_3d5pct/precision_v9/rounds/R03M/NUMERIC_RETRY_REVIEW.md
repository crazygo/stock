# R03M 数值 retry1 独立复盘

2026-09-27，Astra xhigh，只读产物/源码审查，未重复预测。**三个登记数值门禁通过；根已采用数值证据，尚未签发证书、尚非正式 loader 入口 G1，更非前向可用。** 原失败、原四件登记及所有运行字节保留。达标仍 0/5，G2/G3 未通过。

## 实际状态与独立检查

唯一 retry 为 `runs/precision_v9_20260927/R03M_certificate_evidence_v1_retry1`，登记及采用分别为本目录 `NUMERIC_FAILURE_REVIEW_AND_RETRY_REGISTRATION.md` / `NUMERIC_RETRY_ADOPTION.md`。15 套范围、六代表、九目标、1e-7 容差均未改变；7566 不进入，无拟合、重选、校准变化。

| 门禁 | 亲读实际产物与独立核对 | 结果 |
|---|---|---|
| 全键旧/新同批次 | 15 个 parquet 各 14,424 原序键，逐文件 SHA 匹配，216,360 模型-键；九 raw/p 有限且在 [0,1]，两侧逐值完全相同 | 15/15，最大差 0 |
| 原 eval 独立历史批次 | 各套完整原 eval 键序/样本量 1975/2056/1836；使用原接口按该 eval 样本量分批，非全键预测切片 | 15/15，raw/p 最大差 0 |
| 六代表真实 feature-only | 固定六键、每键五路线×三套，输入字段/shape/NaN 掩码全等；两侧实际同 singleton，亲算保存的 90 对九 raw/p 全等 | 90/90，输入/raw/p 最大差 0 |
| 前后来源与失败状态 | 115 代表源前后相同，497 全局源在全键及代表结束均相同；我逐文件重哈希/大小核验零漂移；summary 与 final_status 均通过，无 fatal | 通过 |

全键 12 类有效输入比较（含 prior/counts、group_seq、两种 xbase、paths/curve/relative/y）最大差均 0；该层依赖上轮已证实的原/新 dataset 原字节、监督与来源等价。两无群路线实际仅用本股，不传 group_seq；C_no_daily 不传 xday，原 C 调用禁用槽保持空 tensor。

| 固定代表 | 完整成熟先验数 / 实用最近数 | 三群实际行情依赖数（含本股/QQQ） |
|---|---:|---:|
| AMD 2026-03-02 | 0 / 0 | 107 |
| AMD 2026-06-01 | 58 / 58 | 107 |
| AMD 2026-07-10 | 85 / 63 | 108 |
| AMD 2026-07-13 | 86 / 63 | 108 |
| AMD 2026-09-17 | 133 / 63 | 109 |
| COHR 2026-06-01 | 58 / 58 | 107 |

我从原 rows 独立筛本股 decision/label_end/label_available 全部早于各 deadline，核最近 63 个 IDs 与每路线 lineage 一致；另核六例全部 source IDs 的 5m bar_end≤cutoff、archived available≤11:30:30。群依赖从完整 108 candidate 角色求得，QQQ 单独基准；不会固定复制 107 或把 benchmark 纳入 peers。三个群路线 lineage 均为 `v9_causal_history_end_v1`，metadata 内容 SHA 均 `6c5d2a16d5448b5784d74a63060143516a9aa7e8aa415503a7a62c731a56aad6`。全部 `mode=historical_fixture`、receipt_verified/g2_eligible=false、latest_received_at=null；无真实接收证据。旧记录 `:v6` 与新 feature lineage `:v9` 为同 symbol/date/cutoff 的显式映射，不冒称同一个版本 ID。

## 批次与历史来源边界

首 run 仍是 14/15 原 eval 门禁失败。其 `C_group selected_3566 / CIEN:2026-09-04:11:30:v6 / 5d_8pct` 跨批次 p 差 `1.0642043968278614e-7` 在 retry 继续单列，未删键或放宽容差。代表 singleton 对全键批次另有 raw 最大 `1.1920928955078125e-7`（COHR 06-01，C_group dev2）与 p 最大 `1.4517552993087435e-7`（AMD 07-13，C_no_group dev1）。这些是登记保留的跨批次诊断；**不能宣称任意 batch 数值不变**，也不能用它绕过输入差异。

六套较早 dev 的训练原 run.py SHA 为 `7f6189949f792b043c6c4a0dc4d0eedb6462717d9b9be3e4e6da5476627f8b11`（Git 101d357，run 内 source_origin_snapshots 保存匹配字节）；本次解码器/R10/final 为 `abeca557643910acbb79d7a00ecbfd6572a93da6de72231df836efbfa94645cd`（ccb96a3）。已读限定 diff，predict_c/apply_cal 不变，并以完整原 eval 实际复放验真；不把当前解码器冒称早期训练原字节。模型原真实训练 lineage 不变，本轮不是 causal 重训证明。原 schema 若从冻结源码推导，继续明确该来源。

## 关键封存 SHA-256

| 相对 retry 目录的证据 | SHA |
|---|---|
| all_keys_summary.json | `4f2f520bcf0da619cc07f5667c9c5b227dd77874e96aacb631c8d09d992c7904` |
| representatives.json | `204a7bbe1949ddbd3d206b833b487a37c9f21919250fb7dcaedc6ab049ab1731` |
| representative_summary.json | `84533c1b4742a19354beb6765ce7a233e59ede4e5cdd176a4698ed6925f07c27` |
| representative_final_status.json | `767c3aa4438167cb59c5302e8fa4ef82507bdd791eae5ba0a898c8c37c65654d` |
| representative_sources_before/after.json | `a1674b4378e3d125e8a86d9346251e2819e773c472dd658c611cd08eeff8c4ac` |
| input_hashes_before/after、representative_global_sources_after.json | `b8e1d1c7a3befa4d3b8ad8e04073debf2030df8f6dfb75e00a96baf99695d3e9` |
| inventory.json | `880e338e3e35d09dbbd04f605e89a267f0a43f46343360d5d24fe71bef7a7230` |

最终 numeric 源码 SHA `1661e1a13aee23c74b4ddb7e8628e85268581840d37c03f099173c7a3d46e084`；测试 SHA `2c651633bf5d074cfd02515cf6dba0ae8dd8b0e7dc55731ae2ebb842bf3b0464`。Sol 报告定向 4/4，本文采用依据为实际数值与来源，不以测试绿替代。

后续选择见更新后的 POST_REVIEW_BACKLOG/MATRIX。仅可提出六历史键真实 load_route→predict_route 的小范围草案；不授予旧模型新日期/新 metadata 权限，不将全 14,424 张量证据泛化为全部 feature-only 许可。全 21×5 性能测量顺延；R03 原生工件导出合同另在正式拟合前登记。本轮不改变 R03 历史扩量、模型路线或 90% 目标状态。
