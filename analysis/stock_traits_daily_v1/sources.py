"""Market-data layers for the daily close dataset: Local 5m -> R2 5m -> OpenD 5m.

Single price basis: Futu 5m bars, autype NONE (unadjusted USD), regular session only.
Daily close = close of the 5m bar whose end equals the official session close in the
calendar (16:00 ET, or the early-close time). Different bases (QFQ 60m, Massive daily
grouped) are *not* spliced in.
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from common import PKG, REPO, Calendar, parse_ts, sha256_file, utcnow

PRICE_BASIS = {
    "provider": "futu_opend",
    "bar": "5m",
    "autype": "NONE",
    "currency": "USD",
    "session": "regular",
    "daily_close_rule": "close of regular 5m bar ending at calendar close_at_et",
}
DAILY_COLUMNS = ["session_date", "open", "high", "low", "close", "volume", "regular_bars", "expected_bars",
                 "bar_start", "bar_end", "available_at", "layer"]
SPLIT_RATIOS = [2, 3, 4, 5, 8, 10, 15, 20, 25, 30, 50]


@dataclass
class Counters:
    requests: int = 0
    retries: int = 0
    bytes: int = 0
    cache_hits: int = 0
    errors: list = field(default_factory=list)


# ------------------------------------------------------------------ derive
def derive_daily(bars: pd.DataFrame, cal: Calendar, start: str, end: str, info_cutoff: datetime,
                 layer: str) -> tuple[dict[str, dict], dict[str, str]]:
    """Return ({session_date: daily row}, {session_date: problem}) from normalized 5m bars."""
    rows: dict[str, dict] = {}
    problems: dict[str, str] = {}
    if bars is None or len(bars) == 0:
        return rows, problems
    b = bars[(bars["session_type"] == "regular") & (bars["session_date"] >= start) & (bars["session_date"] <= end)]
    if len(b) == 0:
        return rows, problems
    b = b.assign(_avail=pd.to_datetime(b["available_at"].astype(str), utc=True, format="ISO8601"))
    for d, g in b.groupby("session_date", sort=True):
        if not cal.is_session(d):
            problems[d] = "bars_on_non_session_day"
            continue
        g = g.sort_values("time_key")
        avail = g["_avail"].max().to_pydatetime()
        if avail > info_cutoff:
            problems[d] = "not_available_before_info_cutoff"
            continue
        close_t = cal.close_time_et(d)
        last = g[g["time_key"].astype(str).str[11:19] == close_t]
        if last.empty:
            problems[d] = "closing_bar_missing"
            continue
        o, h, l, c = float(g["open"].iloc[0]), float(g["high"].max()), float(g["low"].min()), float(last["close"].iloc[0])
        vals = [o, h, l, c]
        if not all(math.isfinite(v) and v > 0 for v in vals) or l > min(o, c) + 1e-9 or h < max(o, c) - 1e-9:
            problems[d] = "invalid_ohlc"
            continue
        rows[d] = {
            "session_date": d, "open": o, "high": h, "low": l, "close": c,
            "volume": float(g["volume"].sum()), "regular_bars": int(len(g)),
            "expected_bars": cal.expected_regular_5m_bars(d),
            "bar_start": cal.open_at(d).isoformat(), "bar_end": cal.close_at(d).isoformat(),
            "available_at": avail.isoformat(), "layer": layer,
        }
    return rows, problems


def read_5m(path: Path) -> pd.DataFrame:
    cols = ["time_key", "available_at", "session_date", "session_type", "open", "high", "low", "close", "volume"]
    return pd.read_parquet(path, columns=cols)


# ------------------------------------------------------------------ local
class LocalLayer:
    name = "local"

    def __init__(self, base: Path):
        self.base = Path(base)
        self.read_hashes: dict[str, str] = {}

    def files(self, symbol: str, years: list[int]) -> list[Path]:
        return [p for y in years if (p := self.base / symbol / f"{y}.parquet").exists()]

    def load(self, symbol: str, years: list[int]) -> tuple[pd.DataFrame | None, list[dict]]:
        frames, refs = [], []
        for p in self.files(symbol, years):
            h = sha256_file(p)
            self.read_hashes[str(p)] = h
            ref = {"path": str(p.relative_to(REPO)) if p.is_relative_to(REPO) else str(p), "sha256": h,
                   "bytes": p.stat().st_size}
            try:
                frames.append(read_5m(p))
            except Exception as exc:  # unreadable/odd local file: record it and fall through to R2 / OpenD
                ref["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
            refs.append(ref)
        return (pd.concat(frames, ignore_index=True) if frames else None), refs


# ------------------------------------------------------------------ R2
class R2Layer:
    name = "r2"

    def __init__(self, client_factory: Callable[[], Any] | None, prefix: str, cache_dir: Path, enabled: bool = True):
        self.client_factory = client_factory
        self.prefix = prefix
        self.cache_dir = Path(cache_dir)
        self.enabled = enabled
        self._client = None
        self.status: dict = {"enabled": enabled}
        self.counters = Counters()

    def client(self):
        if self._client is None:
            self._client = self.client_factory()
        return self._client

    def check(self, symbol: str, year: int, need_close_at: datetime) -> dict:
        """Return {'state': miss|stale_by_last_modified|downloaded|error, ...}."""
        key = f"{self.prefix}/{symbol}/{year}.parquet"
        if not self.enabled:
            return {"state": "disabled", "key": key}
        try:
            c = self.client()
            self.counters.requests += 1
            head = c.head_object(key)
        except Exception as exc:
            self.counters.errors.append(f"{key}: {type(exc).__name__}")
            return {"state": "error", "key": key, "error": f"{type(exc).__name__}: {exc}"}
        if head is None:
            return {"state": "miss", "key": key}
        lm_raw = head.get("last_modified") or ""
        try:
            lm = parsedate_to_datetime(lm_raw) if "," in lm_raw else parse_ts(lm_raw)
        except Exception:
            lm = None
        info = {"key": key, "etag": head.get("etag"), "size": head.get("size"), "last_modified": lm.isoformat() if lm else lm_raw}
        if lm is not None and lm < need_close_at:
            # The object was last written before the first required close happened, so it
            # cannot contain it. Skip the download and keep falling back.
            return {"state": "stale_by_last_modified", **info}
        dest = self.cache_dir / self.prefix / symbol / f"{year}.parquet"
        try:
            self.counters.requests += 1
            n = c.get_object(key, dest)
            self.counters.bytes += int(n or 0)
        except Exception as exc:
            self.counters.errors.append(f"{key}: {type(exc).__name__}")
            return {"state": "error", **info, "error": f"{type(exc).__name__}: {exc}"}
        return {"state": "downloaded", **info, "path": str(dest), "sha256": sha256_file(dest)}

    def corporate_actions(self, symbol: str) -> dict:
        key = f"corporate_actions/{symbol}.parquet"
        if not self.enabled:
            return {"state": "disabled", "key": key}
        dest = self.cache_dir / "corporate_actions" / f"{symbol}.parquet"
        try:
            c = self.client()
            self.counters.requests += 1
            head = c.head_object(key)
            if head is None:
                return {"state": "miss", "key": key}
            if dest.exists() and dest.stat().st_size == int(head.get("size") or -1):
                self.counters.cache_hits += 1
            else:
                self.counters.requests += 1
                self.counters.bytes += int(c.get_object(key, dest) or 0)
            return {"state": "ok", "key": key, "path": str(dest), "sha256": sha256_file(dest),
                    "last_modified": head.get("last_modified")}
        except Exception as exc:
            return {"state": "error", "key": key, "error": f"{type(exc).__name__}: {exc}"}


def split_events(ca_path: Path, start: str, end: str) -> list[dict]:
    df = pd.read_parquet(ca_path)
    out = []
    for _, r in df.iterrows():
        d = str(r.get("ex_div_date"))[:10]
        if not (start < d <= end):
            continue
        fields = {k: r.get(k) for k in ("split_base", "split_ert", "join_base", "join_ert", "split_ratio")}
        if any(v is not None and not (isinstance(v, float) and math.isnan(v)) for v in fields.values()):
            out.append({"ex_date": d, **{k: (None if v is None or (isinstance(v, float) and math.isnan(v)) else float(v))
                                         for k, v in fields.items()}})
    return out


def suspected_splits(daily: list[dict]) -> list[dict]:
    """Unadjusted close jumps that look like corporate actions (flag; never auto-adjusted).

    rule split_ratio: close ratio within 4% of a common split ratio and |r| > ln(1.8)
    rule extreme_move: |r| > ln(2) (>= -50% / +100% in one session) - needs manual review
    """
    out = []
    for prev, cur in zip(daily, daily[1:]):
        ratio = prev["close"] / cur["close"]
        lr = abs(math.log(ratio))
        rule = None
        if lr > math.log(1.8) and any(abs(ratio / t - 1) < 0.04 for k in SPLIT_RATIOS for t in (k, 1 / k)):
            rule = "split_ratio"
        elif lr > math.log(2):
            rule = "extreme_move"
        if rule:
            out.append({"session_date": cur["session_date"], "prev_close": prev["close"], "close": cur["close"],
                        "ratio": round(ratio, 4), "rule": rule})
    return out


# ------------------------------------------------------------------ OpenD
class OpenDLayer:
    name = "opend"

    def __init__(self, host: str, port: int, timeout: float, enabled: bool = True,
                 runner: Callable[[list[str], float], dict] | None = None):
        self.host, self.port, self.timeout, self.enabled = host, port, timeout, enabled
        self.runner = runner or self._run_worker
        self.worker_cmd = [sys.executable, str(PKG / "opend_worker.py")]
        self.counters = Counters()
        self.probe_result: dict | None = None
        self.min_interval = 1.0  # seconds between worker calls: <= 30 requests / 30 s (OpenD limit 60 / 30 s)
        self.probe_detail = False  # qfq_daily mode: list securities already in the 7-day quota window
        self._last_call = 0.0

    def _run_worker(self, args: list[str], timeout: float) -> dict:
        cmd = [*self.worker_cmd, *args, "--host", self.host, "--port", str(self.port)]
        t0 = time.monotonic()
        try:
            cp = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return {"ok": False, "error_type": "timeout", "error": f"worker killed after {timeout}s",
                    "elapsed_seconds": round(time.monotonic() - t0, 3)}
        marker = "@@STDV1_RESULT@@ "
        lines = [ln[len(marker):] for ln in cp.stdout.splitlines() if ln.startswith(marker)]
        if not lines:  # legacy/plain worker output: last line that parses as a JSON object
            lines = [ln for ln in cp.stdout.splitlines() if ln.lstrip().startswith("{")]
        try:
            return json.loads(lines[-1])
        except Exception:
            return {"ok": False, "error_type": "worker", "returncode": cp.returncode,
                    "error": (cp.stderr or cp.stdout)[-400:]}

    def probe(self) -> dict:
        if not self.enabled:
            self.probe_result = {"ok": False, "error_type": "disabled", "error": "OpenD layer disabled by flag"}
        elif self.probe_result is None:
            self.counters.requests += 1
            self.probe_result = self.runner(["probe", *(["--quota-detail"] if self.probe_detail else [])], self.timeout)
        return self.probe_result

    def available(self) -> bool:
        return bool(self.probe().get("ok"))

    def run(self, args: list[str], timeout: float | None = None) -> dict:
        wait = self._last_call + self.min_interval - time.monotonic()
        if wait > 0 and self.runner == self._run_worker:
            time.sleep(wait)
        self.counters.requests += 1
        try:
            return self.runner(args, timeout or self.timeout)
        finally:
            self._last_call = time.monotonic()

    def fetch_5m(self, code: str, start: str, end: str, out: Path) -> dict:
        # Paged fetch can take many seconds for a long range; give the killable worker more time.
        return self.run(["kline", "--code", code, "--start", start, "--end", end, "--out", str(out)],
                        timeout=max(self.timeout, 120))


    def fetch_day(self, code: str, start: str, end: str, out: Path, autype: str = "QFQ") -> dict:
        """One OpenD daily K response (QFQ by default) for the whole requested range."""
        return self.run(["kline_day", "--code", code, "--start", start, "--end", end, "--out", str(out),
                         "--autype", autype], timeout=max(self.timeout, 60))

    def quota_now(self) -> dict:
        """Read-only quota re-query (used at the end of a run)."""
        if not self.enabled:
            return {"ok": False, "error": "disabled"}
        r = self.run(["probe"])
        return r.get("history_kl_quota") or {"error": r.get("error")}


def normalize_opend_5m(raw_path: Path, symbol: str) -> pd.DataFrame:
    sys.path.insert(0, str(REPO))
    from scripts.fetch_research_data import normalize_kline_dataframe  # reuse existing normalizer
    return normalize_kline_dataframe(pd.read_parquet(raw_path), symbol, "5m", "NONE")


def daily_frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=DAILY_COLUMNS)
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = df[c].astype("float64")
    for c in ("regular_bars", "expected_bars"):
        df[c] = df[c].astype("int64")
    return df.sort_values("session_date").reset_index(drop=True)


def content_hash(rows: list[dict]) -> str:
    """Hash of the economic content only (not provenance), stable across reruns."""
    import hashlib
    h = hashlib.sha256()
    for r in sorted(rows, key=lambda x: x["session_date"]):
        h.update(f"{r['session_date']}|{r['open']:.6f}|{r['high']:.6f}|{r['low']:.6f}|{r['close']:.6f}|{r['volume']:.0f}\n".encode())
    return h.hexdigest()


def write_daily_parquet(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    daily_frame(rows).to_parquet(path, compression="zstd", compression_level=7, index=False)


def np_closes(df: pd.DataFrame) -> np.ndarray:
    return df["close"].to_numpy(dtype=float)
