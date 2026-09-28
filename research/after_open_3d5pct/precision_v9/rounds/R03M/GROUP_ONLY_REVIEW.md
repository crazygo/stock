# R03M group-only reconstruction review

2026-09-27. **Historical full-key tensor equivalence observed; no model compatibility certificate or training claim.** The completed run is [`runs/precision_v9_20260927/R03M_group_only_v1_retry1`](../../../runs/precision_v9_20260927/R03M_group_only_v1_retry1). Its `result.json` SHA-256 is `d00e83c18a31ca26f65e40bc75d6c9a0f66ab401e96f97ddde49c3f50a8d88a5`. The first failed run, `R03M_group_only_v1`, remains untouched; its old comparator erroneously included 5,850 noncandidate ETF metadata rows. The failure, role inventory, rejected alternatives, and retry scope are recorded in `FAILURE_REVIEW_AND_RETRY_REGISTRATION.md` (SHA `90a6b52e75ebf0c4a913faccfbd12e2c8dfd93830b8ac2dc0fb935e74c8f3b27`). The one-line comparator filter had been drafted before that retry registration, but no retry was executed until after registration and its full missing-candidate guard/test were in place.

## Frozen reconstruction and audit

The retry verified 223 frozen source hashes before and after, all nine registered v8 identity files, the original classifier definitions, `focus_v8.core` and protocol, and copied running source bytes. `inputs_before_sha256.json` SHA is `cdd59f938d2925fd09e0b9925cb06e22b5095876cfe48e30be0ae18dd6d7af0a`; `inputs_after_sha256.json` SHA is `1290273b741df3f08a83b293b510bf694a600d6eb29754873a05e7312e89a8d7`. Their source, identity, code, definition, and protocol value maps match exactly. Running builder SHA is `bfe2c08323085b5113b7647a6e4e7b11ab6dc0ae65059e0af15bede4a541fbd9`.

The run loaded 108 candidate market views plus QQQ as the sole benchmark, and rebuilt all 21,060 candidate × week × strategy memberships from actual frozen bars, action sources, and official calendar. Original 101 symbols used the float64 original classifier path; added seven used the frozen float32 focus-v8 path. All **20,647** old candidate membership records matched the new category and status. The **413** old-missing records are exactly 59 per added symbol (`AAOI ANET AXTI CIEN COHR CRDO FN`); every one is an explicit `insufficient_history` empty group. No old candidate membership was lost. The 5,850 old research ETF records are listed with their frozen roles in `classification_audit.json`: four frozen benchmarks and 26 absent from the v8 universe. They were excluded only from the candidate comparator/peer map, not used to shrink the 108-candidate rebuild.

The new ISO W01 cutoff is the actual 2025-12-29 official session; absent 2025 bars remain unclassified. W39 has a new explicit effective end 2026-09-28, while the old group metadata has `null`. All 14,424 evaluated old keys fall on 2026-03-02 through 2026-09-17; their six-week group representatives span only W05–W38. Thus neither boundary metadata difference touches the evaluated tensors. All successful classifications retain actual `window_dates`, `history_end`, maximum available timestamp, source hashes, classifier value, and action check. Each rejected window records a reason.

| Audit | Result |
|---|---:|
| Old/new sample IDs, order, and protected row fields | 14,424 exact; old-only/new-only/duplicates 0 |
| `x5`, `x60`, `xday`, `y` decompressed NPY payload bytes | Four exact SHA pairs |
| Mature own-label history IDs, prior values/counts, fold keys, weights and positive-weight IDs | Exact old/new audit |
| `group` / `group_seq` | 0 changed rows, slots, channels; max absolute delta 0 |
| `group_valid_count` | 0 changed rows; all rows equal |
| One-worker elapsed / maximum RSS | 128.668 s / 1,085,194,240 raw bytes (about 1.01 GiB) |

The full 14,424-line `key_audit.jsonl`, 21,060-line `group_metadata/membership_audit.jsonl`, channel-level `group_audit.json`, `supervision_audit.json`, old/new source maps, and frozen source copies are in the retry run. `group_audit.json` SHA is `2bce59ccffe860308b8ee352a703f6bbefdf7d1aeffc4b6e7ee641a2080e74cf`; `supervision_audit.json` SHA is `d8f0dabb71606ca80eaa3b0b1a9f6a5c5f60b67c26311d4b16a628ab4c471b1f`. An independent post-run read also found byte-for-byte equal rows and direct `np.array_equal` for old/new group, group_seq and y.

Targeted offline real-source tests were 6/6: AAPL float64 and COHR float32 W23 paths matched frozen categories, W01 used the December cutoff, late history/actions/future peer history were rejected, and extra ETF metadata could not conceal a missing original candidate. No R03 acquisition, network, training, old data/model/schema edit, loader change, checkpoint raw/probability replay, or live feature-only claim was made. An exact-tensor compatibility certificate and any loader extension require the separately registered full checkpoint/raw/p replay and source/route/schema handshake; historical equivalence here is not G2 or an effectiveness result.
