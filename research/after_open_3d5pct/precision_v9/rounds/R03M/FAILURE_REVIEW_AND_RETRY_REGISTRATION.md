# R03M first failure and retry registration

2026-09-27. The first immutable run is `runs/precision_v9_20260927/R03M_group_only_v1`. Its saved builder byte SHA-256 is `76904f47e8f919f9dd124552fb19a289cd803eecb6736c1410b85aa6e3d30a5c`. It verified all 223 frozen sources, loaded 109 views, and recomputed 21,060 memberships before failing at elapsed 95.81 seconds with `ValueError: old classified/empty membership key lost`. It did not build any new dataset. Keep its `progress.jsonl`, `inputs_before_sha256.json`, copied source, and failure intact.

## Evidence and defect

The frozen v8 universe SHA `cd556bc08e0315d1243baa3817440ed969cdbeb17e8debb493b84867adea5218` assigns **108 candidate** roles and seven benchmark roles. Its frozen group memberships SHA `1a23e39efd8c0e3b2a6d793e8eb0ebc0e2c7c619a17abb77d7027cabe5c86a7d` also contains old research ETF memberships. The first builder's *old comparator* selected all five strategy rows regardless of the frozen v8 candidate role, then demanded that every old key appear in the 108-candidate reconstruction. Exactly 5,850 extra rows are 30 symbols × 39 weeks × five strategies:

`ARKG BBH BOTZ COPX DRAM EWH EWY FINX GLD GRID IBB IGV ITA IWO IXC IYE IYM KSTR PAVE QQQ QUAL SLV SLVP SMH SOXS SOXX VIS VLUE VT XBI`.

In the frozen v8 universe, `IGV`, `QQQ`, `SMH`, and `SOXX` are **benchmark**, and the other 26 are **absent**; none is a candidate. The old group research universe had a different scope. This is a comparator scope bug, not evidence of a missing candidate or reason to add an ETF peer. Only QQQ is loaded as the separate benchmark market view; no other ETF is in the 109 market dependencies.

## Backlog and decision matrix

| Option | Effect | Decision |
|---|---|---|
| Add the 30 symbols to reconstructed peers | Changes the frozen 108-candidate universe, duplicates QQQ as a peer, and violates registration | Reject |
| Ignore all old-only keys or compare intersections | Could hide a lost original candidate and weaken the all-key gate | Reject |
| Scope old comparison to frozen candidate roles, enumerate extras, and require every original 101 candidate × every week/strategy | Maintains 108-candidate/full-key validation and records the 5,850 excluded metadata rows | Adopt |

The retry comparator will validate the exact 108 candidate symbols, reject missing or duplicate original-candidate old keys, and demand a full 108 × 39 × five new membership grid. Added seven candidates may have no old membership for a previously unclassified window; these are recorded as `old_missing`, not dropped. The 14,424 old dataset sample IDs and order, all protected own/label NPY payloads, row fields, prior histories, folds, positive-weight supervision, and all source/code before/after hashes remain strict. A targeted synthetic extra-benchmark test must prove that benchmark extras are counted while a missing candidate still fails. The retry output is a fresh sibling `R03M_group_only_v1_retry1`; the first run is never overwritten. This registration authorizes only this comparator repair and one full retry, without changing classification, group semantics, old source/model/schema, R03 acquisition, training, or loader compatibility.

This review was written after the failure and before any retry. A preliminary one-line candidate-role filter had been drafted immediately after diagnosis, before this requested registration; it had not been executed as a run. The full comparator guard and regression test follow this registration.
