"""Normalized, timezone-aware inputs. Provider adapters must prove these fields."""

from dataclasses import dataclass
from datetime import datetime
from math import isfinite
from typing import Literal


def require_aware(*timestamps: datetime) -> None:
    if any(t.tzinfo is None or t.utcoffset() is None for t in timestamps):
        raise ValueError("timestamps must include UTC offset; never guess local timezone")


@dataclass(frozen=True)
class Session:
    open_at: datetime
    close_at: datetime

    def __post_init__(self):
        require_aware(self.open_at, self.close_at)
        if self.open_at >= self.close_at:
            raise ValueError("empty or reversed session")


@dataclass(frozen=True)
class Bar:
    symbol: str
    start_at: datetime
    end_at: datetime
    available_at: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    price_basis: str

    def __post_init__(self):
        require_aware(self.start_at, self.end_at, self.available_at)
        if not self.symbol or not self.price_basis:
            raise ValueError("symbol and price_basis are required")
        if not self.start_at < self.end_at <= self.available_at:
            raise ValueError("bar must finish before becoming available")
        if not all(isfinite(v) for v in (self.open, self.high, self.low, self.close, self.volume)):
            raise ValueError("nonfinite OHLCV")
        if not 0 < self.low <= min(self.open, self.close) <= max(self.open, self.close) <= self.high:
            raise ValueError("invalid OHLC")
        if self.volume < 0:
            raise ValueError("negative volume")


@dataclass(frozen=True)
class Label:
    status: Literal["mature", "pending", "insufficient_data"]
    reason: str | None
    end_at: datetime
    available_at: datetime | None
    hit: int | None = None
    mfe: float | None = None
    mae: float | None = None
    hit_minutes_lower: int | None = None
    hit_minutes_upper: int | None = None


@dataclass(frozen=True)
class Sample:
    sample_id: str
    symbol: str
    decision_at: datetime
    label_end_at: datetime
    label_available_at: datetime | None
    label_status: str
    target: int | None

    def __post_init__(self):
        require_aware(self.decision_at, self.label_end_at)
        if self.label_available_at is not None:
            require_aware(self.label_available_at)
        if self.label_end_at <= self.decision_at:
            raise ValueError("label horizon must follow decision")
        if self.label_status not in {"mature", "pending", "insufficient_data"}:
            raise ValueError("invalid label status")
        if self.label_status == "mature":
            if self.target not in (0, 1) or self.label_available_at is None:
                raise ValueError("mature label needs binary target and availability")
            if self.label_available_at < self.label_end_at:
                raise ValueError("early successes must wait for the full horizon")
        elif self.target is not None:
            raise ValueError("unmatured outcome cannot enter supervised training")
