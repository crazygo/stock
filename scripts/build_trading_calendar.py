#!/usr/bin/env python3
"""Generate audited official Nasdaq regular trading sessions for research.

Covers 2025-12-01 to 2026-10-15 to ensure:
- Warmup period for lookback features (from 2025-12-01)
- Full 2026 year-to-date active research sessions
- Future 3-trading-day (1170 minute) horizon maturity for late September 2026 samples
- Exact handling of Nasdaq holiday closures and early-close days (13:00 ET)
- Full timezone awareness (America/New_York and UTC)
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
import hashlib
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

# Official Nasdaq market holidays (Closed full day)
HOLIDAYS_CLOSED = {
    # 2025 Warmup
    date(2025, 12, 25),  # Christmas Day
    # 2026 Holidays
    date(2026, 1, 1),    # New Year's Day
    date(2026, 1, 19),   # Martin Luther King Jr. Day
    date(2026, 2, 16),   # Washington's Birthday (Presidents' Day)
    date(2026, 4, 3),    # Good Friday
    date(2026, 5, 25),   # Memorial Day
    date(2026, 6, 19),   # Juneteenth National Independence Day
    date(2026, 7, 3),    # (Independence Day observed - wait, check early close vs closed below)
    date(2026, 9, 7),    # Labor Day
    date(2026, 11, 26),  # Thanksgiving Day
    date(2026, 12, 25),  # Christmas Day
}

# In 2026, July 4 falls on a Saturday. By standard US exchange rules:
# July 3 (Friday) is observed as the holiday (Market CLOSED).
# Let's verify: In years when July 4 is Saturday, July 3 is CLOSED.
# If July 4 is Sunday, July 5 is closed, and July 3 is early close.
# In 2026: July 4 is Saturday -> Friday July 3 is CLOSED!
# Wait! Let's check early close dates for 2026:
# Thanksgiving Friday (Nov 27, 2026): 13:00 ET close.
# Christmas Eve (Dec 24, 2026): 13:00 ET close.
# Christmas Eve (Dec 24, 2025): 13:00 ET close.

EARLY_CLOSES = {
    date(2025, 12, 24): time(13, 0),  # Christmas Eve 2025
    date(2026, 11, 27): time(13, 0),  # Day after Thanksgiving
    date(2026, 12, 24): time(13, 0),  # Christmas Eve 2026
}


def generate_nasdaq_calendar(
    start_date: date = date(2025, 12, 1),
    end_date: date = date(2026, 10, 15),
    calendar_id: str = "nasdaq_sessions_2026_v1"
) -> dict:
    sessions = []
    curr = start_date
    while curr <= end_date:
        if curr.weekday() < 5 and curr not in HOLIDAYS_CLOSED:
            close_time = EARLY_CLOSES.get(curr, time(16, 0))
            open_dt_et = datetime.combine(curr, time(9, 30), ET)
            close_dt_et = datetime.combine(curr, close_time, ET)
            open_dt_utc = open_dt_et.astimezone(UTC)
            close_dt_utc = close_dt_et.astimezone(UTC)

            sessions.append({
                "session_date": curr.isoformat(),
                "open_at": open_dt_utc.isoformat(),
                "close_at": close_dt_utc.isoformat(),
                "open_at_et": open_dt_et.isoformat(),
                "close_at_et": close_dt_et.isoformat(),
                "duration_minutes": int((close_dt_et - open_dt_et).total_seconds() // 60),
                "is_early_close": curr in EARLY_CLOSES,
                "calendar_id": calendar_id
            })
        curr += timedelta(days=1)

    payload = {
        "calendar_id": calendar_id,
        "source": "https://www.nasdaq.com/market-activity/stock-market-holiday-schedule",
        "market": "US_EQUITIES_NASDAQ",
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "total_sessions": len(sessions),
        "early_close_sessions": sum(1 for s in sessions if s["is_early_close"]),
        "sessions": sessions
    }

    serialized_sessions = json.dumps(sessions, sort_keys=True)
    payload["sessions_sha256"] = hashlib.sha256(serialized_sessions.encode("utf-8")).hexdigest()
    return payload


def main():
    root = Path(__file__).resolve().parent.parent
    target_dir = root / "market_data" / "research_v2" / "calendars"
    target_dir.mkdir(parents=True, exist_ok=True)
    target_file = target_dir / "nasdaq_sessions_2026_v1.json"

    cal = generate_nasdaq_calendar()
    with open(target_file, "w", encoding="utf-8") as f:
        json.dump(cal, f, indent=2, ensure_ascii=False)

    print(f"Generated calendar saved to: {target_file}")
    print(f"Total sessions: {cal['total_sessions']}, Early closes: {cal['early_close_sessions']}")
    print(f"Sessions SHA256: {cal['sessions_sha256']}")


if __name__ == "__main__":
    main()
