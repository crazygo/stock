"""Versioned, backward-only intraday time resolution for hourly research runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo
import json

ET = ZoneInfo("America/New_York")
UTC = timezone.utc
DEFAULT_HOURS = ("10:30", "11:30", "12:30", "13:30", "14:30", "15:30")


def aware(value: str | datetime) -> datetime:
    dt = datetime.fromisoformat(value.replace("Z", "+00:00")) if isinstance(value, str) else value
    if dt.tzinfo is None or dt.utcoffset() is None:
        raise ValueError("timestamp requires an explicit UTC offset or timezone")
    return dt.astimezone(UTC)


@dataclass(frozen=True)
class Resolution:
    requested_at: str
    effective_cutoff: str | None
    information_available_deadline: str | None
    action_expires_at: str | None
    next_legal_cutoff: str | None
    session_date: str | None
    state: str
    reason: str
    calendar_id: str
    schedule_version: str = "hourly_once_v3"

    def json(self) -> dict:
        return asdict(self)


def resolve_time(requested_at: str | datetime, calendar: dict | Path,
                 hours: tuple[str, ...] | list[str] = DEFAULT_HOURS,
                 signal_latency_seconds: int = 30, manual_delay_seconds: int = 120,
                 entry_max_wait_minutes: int = 5) -> Resolution:
    """Floor to the latest completed, supported cutoff on this same session.

    A cutoff is available only once the configured signal latency has elapsed.
    After the delayed-entry window, its report remains inspectable but cannot
    be treated as a current action for a different entry price.
    """
    if isinstance(calendar, Path):
        calendar = json.loads(calendar.read_text())
    at = aware(requested_at)
    date = at.astimezone(ET).date().isoformat()
    matches = [s for s in calendar["sessions"] if s["session_date"] == date]
    cid = calendar.get("calendar_id", "unversioned_calendar")
    if not matches:
        return Resolution(at.isoformat(), None, None, None, None, date,
                          "unavailable", "market_closed_or_calendar_absent", cid)
    session = matches[0]
    start, close = aware(session["open_at"]), aware(session["close_at"])
    legal: list[datetime] = []
    for h in hours:
        hh, mm = map(int, h.split(":"))
        cut = datetime.fromisoformat(date).replace(hour=hh, minute=mm, tzinfo=ET).astimezone(UTC)
        if start < cut < close:
            legal.append(cut)
    legal = sorted(set(legal))
    ready = [t for t in legal if t + timedelta(seconds=signal_latency_seconds) <= at]
    future = [t for t in legal if t + timedelta(seconds=signal_latency_seconds) > at]
    if not ready:
        return Resolution(at.isoformat(), None, None, None,
                          future[0].isoformat() if future else None, date,
                          "unavailable", "before_first_supported_completed_cutoff", cid)
    cut = ready[-1]
    expires = cut + timedelta(seconds=signal_latency_seconds + manual_delay_seconds,
                              minutes=entry_max_wait_minutes)
    if at >= close:
        state, reason = "expired", "market_closed_signal_is_historical"
    elif at > expires:
        state, reason = "expired", "delayed_entry_window_elapsed_reprice_required"
    else:
        state, reason = "aligned", "latest_completed_supported_cutoff"
    return Resolution(at.isoformat(), cut.isoformat(),
                      (cut + timedelta(seconds=signal_latency_seconds)).isoformat(),
                      expires.isoformat(), future[0].isoformat() if future else None,
                      date, state, reason, cid)
