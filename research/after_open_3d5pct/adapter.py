"""Real market data adapter for research/after_open_3d5pct.

Loads normalized data from market_data/research_v2:
- Calendar sessions -> contracts.Session
- Parquet 5m bars -> contracts.Bar
- Universe metadata -> list of eligible symbols
"""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo
import pandas as pd

from .contracts import Bar, Session, require_aware

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")


def load_calendar_sessions(calendar_path: Optional[Path] = None) -> list[Session]:
    """Load and parse official exchange trading sessions."""
    if calendar_path is None:
        root = Path(__file__).resolve().parent.parent.parent
        flat_p = root / "market_data" / "calendars" / "nasdaq_sessions_2026_v1.json"
        nested_p = root / "market_data" / "research_v2" / "calendars" / "nasdaq_sessions_2026_v1.json"
        calendar_path = flat_p if flat_p.exists() else nested_p

    with open(calendar_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    sessions = []
    for s in data["sessions"]:
        # open_at and close_at are stored as ISO-8601 strings
        open_dt = datetime.fromisoformat(s["open_at"])
        close_dt = datetime.fromisoformat(s["close_at"])
        sessions.append(Session(open_at=open_dt, close_at=close_dt))

    sessions.sort(key=lambda s: s.open_at)
    return sessions


def load_universe(universe_path: Optional[Path] = None, role: Optional[str] = None) -> list[str]:
    """Load list of universe symbols (e.g. role='candidate' or role='benchmark')."""
    if universe_path is None:
        root = Path(__file__).resolve().parent.parent.parent
        flat_p = root / "market_data" / "universe" / "qqq_retrospective_v1.json"
        nested_p = root / "market_data" / "research_v2" / "universe" / "qqq_retrospective_v1.json"
        universe_path = flat_p if flat_p.exists() else nested_p

    with open(universe_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    members = data.get("members", [])
    if role:
        members = [m for m in members if m["role"] == role]
    return [m["symbol"] for m in members]


def load_bars_from_parquet(
    parquet_path: Path,
    symbol: str,
    price_basis: str = "NONE_raw",
    regular_only: bool = False
) -> list[Bar]:
    """Read Parquet file and convert rows into contracts.Bar instances."""
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

    df = pd.read_parquet(parquet_path)
    if regular_only and "session_type" in df.columns:
        df = df[df["session_type"] == "regular"]

    bars = []
    for _, row in df.iterrows():
        start_at = datetime.fromisoformat(row["start_at"])
        end_at = datetime.fromisoformat(row["end_at"])
        avail_at = datetime.fromisoformat(row["available_at"]) if "available_at" in row else (end_at + pd.Timedelta(seconds=1)).to_pydatetime()

        # Handle float casting safely
        o = float(row["open"])
        h = float(row["high"])
        l = float(row["low"])
        c = float(row["close"])
        v = float(row["volume"])

        # Enforce OHLC consistency defensively
        h = max(h, o, c)
        l = min(l, o, c)

        bars.append(Bar(
            symbol=symbol,
            start_at=start_at,
            end_at=end_at,
            available_at=avail_at,
            open=o,
            high=h,
            low=l,
            close=c,
            volume=v,
            price_basis=price_basis
        ))

    bars.sort(key=lambda b: b.start_at)
    return bars


def load_research_bars(
    symbols: list[str],
    interval: str = "5m",
    price_basis: str = "NONE",
    session: str = "ALL",
    year: int = 2026,
    data_root: Optional[Path] = None,
    regular_only: bool = False
) -> dict[str, list[Bar]]:
    """Load normalized bars for multiple symbols from research_v2."""
    if data_root is None:
        root = Path(__file__).resolve().parent.parent.parent
        data_root = root / "market_data" / "research_v2"

    result = {}
    basis_tag = f"{price_basis}_raw" if price_basis == "NONE" else price_basis

    for sym in symbols:
        # Check flat path first
        if interval == "5m":
            flat_p = root / "market_data" / "us_5m" / sym / f"{year}.parquet"
        elif interval == "60m" and price_basis == "NONE":
            flat_p = root / "market_data" / "us_60m_raw" / sym / f"{year}.parquet"
        else:
            flat_p = root / "market_data" / f"us_{interval}" / sym / f"{year}.parquet"

        nested_p = data_root / "futu" / sym / interval / price_basis / session / f"{year}.parquet"
        target_p = flat_p if flat_p.exists() else nested_p

        if target_p.exists():
            result[sym] = load_bars_from_parquet(target_p, sym, price_basis=basis_tag, regular_only=regular_only)
    return result
