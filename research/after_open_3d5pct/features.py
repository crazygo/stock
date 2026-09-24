"""Small causal feature kernel; full feature families are specified in docs/03."""

from datetime import datetime, timedelta

from .config import Config
from .contracts import Bar, require_aware
from .timeaxis import ET, intervals, session_for


def opening_features(symbol: str, cutoff_at: datetime, decision_at: datetime,
                     bars: list[Bar], sessions, config: Config) -> dict[str, float]:
    require_aware(cutoff_at, decision_at)
    if cutoff_at.astimezone(ET).strftime("%H:%M") not in config.decision_hours_et:
        raise ValueError("outside v1 decision schedule")
    if cutoff_at.second or cutoff_at.microsecond or decision_at != cutoff_at + timedelta(seconds=config.signal_latency_seconds):
        raise ValueError("decision timestamp does not match feature cutoff + signal latency")
    session = session_for(decision_at, sessions)
    visible = [b for b in bars if b.symbol == symbol and session.open_at <= b.start_at and
               b.end_at <= cutoff_at and b.available_at <= decision_at]
    keyed = {(b.start_at, b.end_at): b for b in visible}
    expected = intervals(session.open_at, cutoff_at, sessions, config.entry_bar_minutes)
    if len(keyed) != len(visible) or set(keyed) != set(expected):
        raise ValueError("opening prefix missing, late, duplicated or wrong interval")
    prefix = [keyed[key] for key in expected]
    if len({b.price_basis for b in prefix}) != 1:
        raise ValueError("mixed feature price basis")
    open_price, now = prefix[0].open, prefix[-1].close
    high, low = max(b.high for b in prefix), min(b.low for b in prefix)
    return {
        "return_since_open": now / open_price - 1,
        "range_since_open": (high - low) / open_price,
        "distance_from_high": now / high - 1,
        "volume_since_open": sum(b.volume for b in prefix),
    }
