"""Versioned Nasdaq sessions for the 2023 warmup and 2024–2026 data task.

Dates transcribed from Nasdaq's annual calendars, with the January 9, 2025
national day of mourning added from Nasdaq's December 30, 2024 announcement.
This does not change calendars frozen by earlier model experiments.
"""
from __future__ import annotations

import hashlib
import json
import pandas as pd

CLOSED = set("""
2023-01-02 2023-01-16 2023-02-20 2023-04-07 2023-05-29 2023-06-19
2023-07-04 2023-09-04 2023-11-23 2023-12-25
2024-01-01 2024-01-15 2024-02-19 2024-03-29 2024-05-27 2024-06-19
2024-07-04 2024-09-02 2024-11-28 2024-12-25
2025-01-01 2025-01-09 2025-01-20 2025-02-17 2025-04-18 2025-05-26
2025-06-19 2025-07-04 2025-09-01 2025-11-27 2025-12-25
2026-01-01 2026-01-19 2026-02-16 2026-04-03 2026-05-25 2026-06-19
2026-07-03 2026-09-07 2026-11-26 2026-12-25
""".split())
EARLY = set("""
2023-07-03 2023-11-24
2024-07-03 2024-11-29 2024-12-24
2025-07-03 2025-11-28 2025-12-24
2026-11-27 2026-12-24
""".split())
SOURCES = [f"https://www.nasdaqtrader.com/content/technicalsupport/{y}tradingcalendar.pdf"
           for y in (2023, 2024, 2025)] + [
    "https://ir.nasdaq.com/news-releases/news-release-details/nasdaq-announces-closure-its-us-markets-honor-national-day-0",
    "https://www.nasdaq.com/market-activity/stock-market-holiday-schedule",
]


def calendar(start="2023-04-01", end="2026-10-15"):
    if start < "2023-01-01" or end > "2026-12-31":
        raise ValueError("Official holiday transcription only covers 2023–2026")
    sessions = []
    for day in pd.bdate_range(start, end).strftime("%Y-%m-%d"):
        if day in CLOSED:
            continue
        op = pd.Timestamp(day + " 09:30", tz="America/New_York")
        cl = pd.Timestamp(day + (" 13:00" if day in EARLY else " 16:00"), tz="America/New_York")
        sessions.append(dict(session_date=day, open_at=op.tz_convert("UTC").isoformat(),
                             close_at=cl.tz_convert("UTC").isoformat(), open_at_et=op.isoformat(),
                             close_at_et=cl.isoformat(), duration_minutes=int((cl-op).total_seconds()/60),
                             is_early_close=day in EARLY))
    return dict(calendar_id="nasdaq_sessions_2023_2026_v1", sources=SOURCES,
                start_date=start, end_date=end, sessions=sessions,
                sessions_sha256=hashlib.sha256(json.dumps(sessions, sort_keys=True).encode()).hexdigest())
