# Official 2026 calendar · code review

2026-09-27. The [execution card](EXECUTION_CARD.md) implements the existing `forward_readiness` calendar requirement. The only HTTP reads were the [Nasdaq Trader calendar](https://nasdaqtrader.com/Trader.aspx?id=Calendar) and [NYSE hours calendar](https://www.nyse.com/trade/hours-calendars). Exact response bytes, requested/final URLs, HTTP 200, UTC retrieval times, sizes, and SHA-256 digests are frozen in `sources.json`. The offline `calendar_artifact.py` reads those local bytes; it contains no downloader or market-data access.

Both captured official sources independently yield the same ten 2026 full closures (Jan 1, Jan 19, Feb 16, Apr 3, May 25, Jun 19, Jul 3, Sep 7, Nov 26, Dec 25) and two 13:00 ET closes (Nov 27, Dec 24). The normalized 2026-01-01 through 2026-12-31 boundary contains **251 sessions**, first Jan 2, last Dec 31. `sessions.json` has the `session_date`, `open_at`, `close_at`, and `duration_minutes` fields used by the ledger and feature encoder, plus a session type. `calendar_id=nasdaq_nyse_us_equities_2026_v1`; canonical session-list SHA-256 is `a2c25174768f04cb6bfb43bc527eb37c4b5ade52bcfd9178b645b76398c50a14`.

The parser rejects a changed source table, date mismatch between exchanges, invalid status, duplicate/unsorted/out-of-range dates, altered ET/UTC clocks, and raw-byte/hash changes. `America/New_York` zone rules supply DST; tests inspect both spring/fall transitions and the two 210-minute half days. `require_coverage` verifies at least 60 contiguous official sessions and the last 11:35 ET entry's full 1950 regular-trading-minute horizon; a final session too near year-end or any 2027 request fails. It only checks coverage and never registers a cohort. For comparison only, the 198 dates overlapping the existing shorter calendar have zero UTC open/close mismatches; that older file is untouched.

`manifest.json` records both source digests, the canonical session digest, generation time, and builder SHA-256 before/after generation. `builder_snapshot.py` is a byte-for-byte source snapshot; its SHA-256 and both builder checks equal `f6003ed502a20374b4f0b789fef328e0efddf3db126df9bd81a7842f24bfc05a`. Build uses exclusive creation and refuses overwrite. `verify_artifact` rechecks the official bytes, calendar content, manifest, and source snapshot. `test_calendar_artifact.py` passes **4/4** under the project Python 3.12 venv; it covers source disagreement, raw/source-code tamper, duplicate/DST mutations, and accepted/rejected 60-session/1950-minute coverage. Exact commands and outcomes are frozen in `test_evidence.json`.

Offline replay from repository root:

```bash
research/after_open_3d5pct/.venv/bin/python -m research.after_open_3d5pct.precision_v9.forward.calendar_artifact verify --run-dir research/after_open_3d5pct/precision_v9/forward/calendar_2026_official_v1
research/after_open_3d5pct/.venv/bin/python -m unittest research.after_open_3d5pct.precision_v9.forward.test_calendar_artifact -v
```

This is a calendar engineering artifact, not a live cohort. It does not prove market-data receipts, a frozen action manifest, model group parity, 30-second runtime, G2/G3, or a 2027 Nasdaq calendar. Source and sessions coverage ends on 2026-12-31.
