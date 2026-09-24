"""Calendar arithmetic over an explicit, externally audited session table."""

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from .contracts import Session, require_aware

ET = ZoneInfo("America/New_York")


def validate_sessions(sessions: list[Session]) -> None:
    if not sessions:
        raise ValueError("calendar is empty")
    for i, session in enumerate(sessions):
        if i and sessions[i - 1].close_at >= session.open_at:
            raise ValueError("calendar must be sorted, unique and non-overlapping")
        if session.open_at.astimezone(ET).date() != session.close_at.astimezone(ET).date():
            raise ValueError("regular session must lie within one ET date")


def session_for(t: datetime, sessions: list[Session]) -> Session:
    require_aware(t)
    validate_sessions(sessions)
    for session in sessions:
        if session.open_at <= t < session.close_at:
            return session
    raise ValueError("timestamp is outside supplied regular sessions")


def add_regular_minutes(start: datetime, minutes: int, sessions: list[Session]) -> datetime:
    session_for(start, sessions)
    if type(minutes) is not int or minutes <= 0:
        raise ValueError("horizon must be positive integer minutes")
    remaining = timedelta(minutes=minutes)
    for session in sessions:
        left = max(start, session.open_at)
        if left >= session.close_at:
            continue
        duration = session.close_at - left
        if duration >= remaining:
            return left + remaining
        remaining -= duration
    raise ValueError("calendar does not cover the future horizon")


def intervals(start: datetime, end: datetime, sessions: list[Session], bar_minutes: int):
    require_aware(start, end)
    validate_sessions(sessions)
    if start >= end or type(bar_minutes) is not int or bar_minutes <= 0:
        raise ValueError("invalid interval request")
    step = timedelta(minutes=bar_minutes)
    result = []
    for session in sessions:
        left, right = max(start, session.open_at), min(end, session.close_at)
        if left >= right:
            continue
        if (left - session.open_at) % step or (right - session.open_at) % step:
            raise ValueError("partial boundary bar needs finer data; cannot use its full high/low")
        while left < right:
            result.append((left, left + step))
            left += step
    return result


def elapsed_regular_minutes(start: datetime, end: datetime, sessions: list[Session]) -> int:
    return int(sum(max(0, (min(end, s.close_at) - max(start, s.open_at)).total_seconds())
                   for s in sessions) // 60)
