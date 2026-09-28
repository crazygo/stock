# R03M 数值有限重试采用

2026-09-27，根协调器采用 Astra 的 `NUMERIC_FAILURE_REVIEW_AND_RETRY_REGISTRATION.md`，SHA-256 `008709af95dc6e8ee556eb6f5b6fca21d23c4a112d9664eae4b55fdcfbac9db5`。采用在 retry 启动前完成；局部 reference 修复与有界诊断已先发生，其时序按该登记如实保留。

根已亲读诊断 JSON（SHA `f91859f5be6ac22e7669e642f6323d91cf0856520390e26664a0c127b2719145`）：原 eval 1836 行、8 批独立复放 raw/p 差为 0；全键 57 批再切 eval 的 processed 越界仍为 `1.0642043968278614e-7`。首 run 的 14/15 失败不改判，不推断任意批次数值不变。

授权 Sol 仅完成登记范围的 reference 修复、定向验证与一次完整新 run `R03M_certificate_evidence_v1_retry1`。15 套、全部 14424 键、六代表 90 配对、九 raw/p、原 `1e-7` 容差及来源门禁不变。全键同批配对、历史 eval 按原独立批次、代表两侧同 singleton；跨批误差另报，不能隐藏 tensor 不匹配。

完成后由 Astra 独立复盘。未授权修改 loader、训练或更换模型/校准/阈值、签发生效证书、启动真实 cohort；G1/G2/G3 和用户目标均未因此通过。原 R03 恢复与五路线研究登记保持。
