"""Synthetic, fully controlled fixtures (no network, no real market data)."""
from __future__ import annotations

import json
import shutil
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import sys
PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG))

from common import DEFAULT_CALENDAR, ET, Calendar  # noqa: E402

UTC = timezone.utc
CAL = Calendar(DEFAULT_CALENDAR)  # read-only official sessions (holidays e.g. 2026-09-07)


def closes_for(symbol: str, dates: list[str], seed: int = 0) -> dict[str, float]:
    rng = np.random.default_rng(int(__import__("hashlib").md5(symbol.encode()).hexdigest()[:6], 16) + seed)
    p = 50 * np.exp(np.cumsum(rng.normal(0.0005, 0.02, len(dates))))
    return {d: float(round(x, 4)) for d, x in zip(dates, p)}


def bars_5m(symbol: str, closes: dict[str, float], extra_days: dict[str, float] | None = None) -> pd.DataFrame:
    """78 regular 5m bars per session; last bar close == daily close."""
    rows = []
    allc = dict(closes)
    allc.update(extra_days or {})
    for d, c in sorted(allc.items()):
        if CAL.is_session(d):
            close_t = CAL.close_at(d).astimezone(ET)
            open_t = CAL.open_at(d).astimezone(ET)
        else:  # fixture for a non-session day
            open_t = datetime.fromisoformat(d + "T09:30:00").replace(tzinfo=ET)
            close_t = datetime.fromisoformat(d + "T16:00:00").replace(tzinfo=ET)
        n = int((close_t - open_t).total_seconds() // 300)
        path = np.linspace(c * 0.99, c, n)
        for i in range(n):
            end = open_t + timedelta(minutes=5 * (i + 1))
            px = float(round(path[i], 4))
            rows.append({"time_key": end.strftime("%Y-%m-%d %H:%M:%S"),
                         "available_at": (end.astimezone(UTC) + timedelta(seconds=1)).isoformat(),
                         "session_date": d, "session_type": "regular", "open": px, "high": px * 1.001,
                         "low": px * 0.999, "close": px, "volume": 1000.0})
    return pd.DataFrame(rows)


def write_5m(base: Path, symbol: str, df: pd.DataFrame) -> Path:
    p = base / symbol / "2026.parquet"
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False)
    return p


def universe_file(path: Path, symbols: list[str]) -> Path:
    members = [{"code": f"US.{s}", "name": f"{s} Inc", "stock_type": "STOCK"} for s in symbols]
    members.append({"code": "HK.00001", "name": "HK fixture", "stock_type": "STOCK"})
    path.write_text(json.dumps({"observed_at": "2026-10-05T13:00:00+00:00", "membership": "fixture",
                                "snapshots": [{"group": "特别关注", "status": "ok", "received_at": "2026-10-05T13:00:00+00:00",
                                               "members": members}]}, ensure_ascii=False))
    return path


class FakeR2:
    """Head/get against an in-memory map {key: (file, last_modified RFC1123)}."""

    def __init__(self, objects: dict | None = None):
        self.objects = objects or {}
        self.calls = []

    def head_object(self, key):
        self.calls.append(("head", key))
        if key not in self.objects:
            return None
        f, lm = self.objects[key]
        return {"key": key, "size": Path(f).stat().st_size, "etag": "x", "last_modified": lm}

    def get_object(self, key, dest):
        self.calls.append(("get", key))
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(self.objects[key][0], dest)
        return dest.stat().st_size


class FakeOpenD:
    def __init__(self, ok: bool = True, kline: dict | None = None):
        self.ok, self.kline, self.calls = ok, kline or {}, []

    def __call__(self, args, timeout):
        self.calls.append(list(args))
        if args[0] == "probe":
            return {"ok": self.ok, "error_type": None if self.ok else "network",
                    "error": None if self.ok else "fixture: connection refused"}
        if args[0] == "kline":
            code = args[args.index("--code") + 1]
            out = Path(args[args.index("--out") + 1])
            closes = self.kline.get(code, {})
            start, end = args[args.index("--start") + 1], args[args.index("--end") + 1]
            sel = {d: c for d, c in closes.items() if start <= d <= end}
            if not sel:
                return {"ok": True, "rows": 0, "out": None}
            df = bars_5m(code.split(".")[1], sel)
            raw = df[["time_key", "open", "high", "low", "close", "volume"]].copy()
            raw["turnover"] = raw["close"] * raw["volume"]
            raw.to_parquet(out, index=False)
            return {"ok": True, "rows": len(raw), "out": str(out)}
        return {"ok": False, "error": "fixture: unsupported action"}


class Env:
    def __init__(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="stv1-"))
        self.local = self.tmp / "local_5m"
        self.data = self.tmp / "data"
        self.cap = self.tmp / "captures"
        self.report = self.tmp / "report"
        self.universe = self.tmp / "universe.json"

    def backfill_args(self, now: str, extra: list[str] | None = None):
        import backfill
        argv = ["--data-root", str(self.data), "--capture-root", str(self.cap), "--local-5m-dir", str(self.local),
                "--universe-file", str(self.universe), "--sec-facts", str(self.tmp / "none.json"),
                "--no-corporate-actions", "--now", now, *(extra or [])]
        return backfill.parse_args(argv)

    def refresh_args(self, now: str, extra: list[str] | None = None):
        import refresh
        return refresh.parse_args(["--data-root", str(self.data), "--report-root", str(self.report), "--now", now,
                                   "--bootstrap", "16", *(extra or [])])

    def cleanup(self):
        import os
        for p in self.tmp.rglob("*"):
            try:
                os.chmod(p, 0o755 if p.is_dir() else 0o644)
            except OSError:
                pass
        shutil.rmtree(self.tmp, ignore_errors=True)
