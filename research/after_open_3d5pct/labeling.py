"""Outcome-only code. Never import outcomes into feature construction."""

from datetime import datetime, timedelta

from .config import Config
from .contracts import Bar, Label, require_aware
from .timeaxis import add_regular_minutes, elapsed_regular_minutes, intervals, session_for


def select_entry(symbol: str, decision_at: datetime, bars: list[Bar], sessions, config: Config) -> Bar:
    """The first expected 5m open after manual delay; no gap skipping or overnight fallback."""
    session = session_for(decision_at, sessions)
    earliest = decision_at + timedelta(seconds=config.manual_delay_seconds)
    step = timedelta(minutes=config.entry_bar_minutes)
    offset = earliest - session.open_at
    steps = (offset + step - timedelta(microseconds=1)) // step
    expected = session.open_at + steps * step
    if expected >= session.close_at or expected - earliest > timedelta(minutes=config.entry_max_wait_minutes):
        raise ValueError("no same-session entry within allowed delay")
    matches = [b for b in bars if b.symbol == symbol and b.start_at == expected]
    if len(matches) != 1 or matches[0].end_at != expected + step:
        raise ValueError("expected entry bar missing, duplicated or not 5m")
    return matches[0]


def make_label(entry: Bar, bars: list[Bar], sessions, as_of: datetime, config: Config) -> Label:
    require_aware(as_of)
    end = add_regular_minutes(entry.start_at, config.horizon_regular_minutes, sessions)
    if as_of < end:
        return Label("pending", "horizon_not_finished", end, None)
    expected = intervals(entry.start_at, end, sessions, config.entry_bar_minutes)
    selected = [b for b in bars if b.symbol == entry.symbol and
                b.start_at >= entry.start_at and b.end_at <= end and
                any(s.open_at <= b.start_at < b.end_at <= s.close_at for s in sessions)]
    by_interval = {(b.start_at, b.end_at): b for b in selected}
    if len(by_interval) != len(selected) or set(by_interval) != set(expected):
        return Label("insufficient_data", "missing_duplicate_or_wrong_interval", end, None)
    path = [by_interval[key] for key in expected]
    if not path or path[0] != entry or any(b.price_basis != entry.price_basis for b in path):
        raise ValueError("entry or price basis differs within outcome window")
    ready_at = max(b.available_at for b in path)
    if ready_at > as_of:
        return Label("pending", "outcome_data_not_available", end, None)
    first_hit = next((b for b in path if b.high >= entry.open * (1 + config.target_return)), None)
    return Label(
        "mature", None, end, ready_at, int(first_hit is not None),
        max(0.0, max(b.high for b in path) / entry.open - 1),
        min(0.0, min(b.low for b in path) / entry.open - 1),
        elapsed_regular_minutes(entry.start_at, first_hit.start_at, sessions) if first_hit else None,
        elapsed_regular_minutes(entry.start_at, first_hit.end_at, sessions) if first_hit else None,
    )
