# v9 official 2026 calendar · execution card

Registered 2026-09-27 under `forward_readiness/{BACKLOG,MATRIX,CALENDAR_REQUIREMENTS,INTEGRATION_PLAN}.md`. This is a bounded calendar dependency repair, not a model or trading experiment.

- Fixed session boundary: 2026-01-01 through 2026-12-31, US Eastern time (`America/New_York`). The artifact will include only confirmed 2026 sessions; requests beyond that range must fail.
- Official source bytes: [Nasdaq Trader calendar](https://nasdaqtrader.com/Trader.aspx?id=Calendar) and [NYSE hours and calendars](https://www.nyse.com/trade/hours-calendars). Save exact HTTP response bytes, URL, UTC retrieval time, status, and SHA-256 before parsing. These are read-only website requests; no market-data or OpenD call.
- New outputs in this run directory: official raw responses, `sources.json`, normalized `sessions.json`, `manifest.json`, test evidence, and `CALENDAR_CODE_REVIEW.md`; pure replay/validation code lives in `forward/calendar_artifact.py`. No shared `market_data` calendar is changed.
- Acceptance: exact 2026 full-day closures and 13:00 ET half days agree across the two official sources; unique ordered dates, local/UTC consistency across DST, compatible fields for `ledger.calendar_sessions` and `features.build_features_asof`, canonical session SHA, calendar ID, and source-byte hashes. A coverage check must reject a 60-session cohort or final 1950 RTH minutes whenever its end lies beyond 2026. No 2027 session is guessed.
- Evidence boundary: completing this artifact does not establish provider receipts, G2/G3, a frozen action recipe, 30-second runtime, or a real cohort.
