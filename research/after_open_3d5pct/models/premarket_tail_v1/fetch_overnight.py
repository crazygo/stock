"""Pull 2026 overnight 5-minute bars into runs/. Does not rewrite history parts or push R2."""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone

import futu as ft
import pandas as pd

from research.after_open_3d5pct.models.pugh_ge5.build_rows import ROOT, union_symbols

OUT = ROOT / "research/after_open_3d5pct/runs/premarket_tail_v1/overnight"
RANGES = (
    ("2026-01-01", "2026-01-31"),
    ("2026-02-01", "2026-02-28"),
    ("2026-03-01", "2026-03-31"),
    ("2026-04-01", "2026-04-30"),
    ("2026-05-01", "2026-05-31"),
    ("2026-06-01", "2026-06-30"),
    ("2026-07-01", "2026-07-31"),
    ("2026-08-01", "2026-08-31"),
    ("2026-09-01", "2026-09-28"),
)
JANUARY_FLOOR = 1500


def _january_ok(frame: pd.DataFrame) -> bool:
    if len(frame) < JANUARY_FLOOR:
        return False
    end = pd.to_datetime(frame["end_at_et"], utc=True).dt.tz_convert("America/New_York")
    minute = end.dt.hour * 60 + end.dt.minute
    clock_ok = (minute > 20 * 60) | (minute <= 4 * 60)
    if not bool(clock_ok.all()):
        return False
    return int((minute > 20 * 60).sum()) >= 200 and int((minute <= 4 * 60).sum()) >= 200


def _overnight_clock(time_key: pd.Series) -> pd.Series:
    end = pd.to_datetime(time_key, errors="coerce")
    minute = end.dt.hour * 60 + end.dt.minute
    return (minute > 20 * 60) | (minute <= 4 * 60)


def _localize_end(time_key: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(time_key, errors="coerce")
    if getattr(parsed.dtype, "tz", None) is not None:
        return parsed.dt.tz_convert("America/New_York")
    return parsed.dt.tz_localize("America/New_York", ambiguous="NaT", nonexistent="NaT")


def _quota(quote) -> str:
    try:
        ret, data = quote.get_history_kl_quota()
    except Exception as exc:  # noqa: BLE001 — surface the gateway text and keep going
        return f"unavailable:{exc}"
    if hasattr(data, "to_dict"):
        payload = data.to_dict(orient="records") if hasattr(data, "columns") else data.to_dict()
    else:
        payload = data
    return json.dumps({"ret": int(ret), "quota": payload}, default=str)


def _pages(quote, symbol: str, start: str, end: str) -> pd.DataFrame:
    pages = []
    cursor = None
    for page in range(40):
        time.sleep(1.0 if page == 0 else 0.35)
        got = None
        last_error = None
        for attempt in range(3):
            ret, chunk, page_key = quote.request_history_kline(
                f"US.{symbol}",
                start=start,
                end=end,
                ktype=ft.KLType.K_5M,
                autype=ft.AuType.NONE,
                max_count=1000,
                extended_time=True,
                session=ft.Session.ALL,
                page_req_key=cursor,
            )
            if ret == ft.RET_OK:
                got = (chunk, page_key)
                break
            last_error = chunk
            if attempt < 2:
                time.sleep(20)
        if got is None:
            text = str(last_error)
            if any(word in text for word in ("额度", "配额", "quota")):
                raise RuntimeError(f"quota {symbol} {start}: {text[:240]}")
            raise RuntimeError(f"{symbol} {start} page {page + 1}: {text[:240]}")
        chunk, cursor = got
        if chunk is not None and len(chunk):
            pages.append(chunk)
        if cursor is None:
            break
    else:
        raise RuntimeError(f"{symbol} {start}: page cap hit before the cursor ended")
    if not pages:
        return pd.DataFrame()
    raw = pd.concat(pages, ignore_index=True)
    raw = raw.loc[_overnight_clock(raw["time_key"])].copy()
    if raw.empty:
        return raw
    end_at = _localize_end(raw["time_key"])
    keep = end_at.notna()
    raw = raw.loc[keep].reset_index(drop=True)
    end_at = end_at.loc[keep].reset_index(drop=True)
    start_at = end_at - pd.Timedelta(minutes=5)
    return pd.DataFrame({
        "symbol": symbol,
        "start_at_et": start_at.map(lambda item: item.isoformat()),
        "end_at_et": end_at.map(lambda item: item.isoformat()),
        "session_type": "overnight",
        "session_date": end_at.dt.strftime("%Y-%m-%d"),
        "open": pd.to_numeric(raw["open"], errors="coerce"),
        "high": pd.to_numeric(raw["high"], errors="coerce"),
        "low": pd.to_numeric(raw["low"], errors="coerce"),
        "close": pd.to_numeric(raw["close"], errors="coerce"),
        "volume": pd.to_numeric(raw["volume"], errors="coerce"),
    }).drop_duplicates("end_at_et")


def main() -> None:
    ft.SysConfig.enable_proto_encrypt(False)
    OUT.mkdir(parents=True, exist_ok=True)
    quote = ft.OpenQuoteContext(host="127.0.0.1", port=11111)
    try:
        print(json.dumps({"quota": _quota(quote)}), flush=True)
        for symbol in union_symbols():
            for start, end in RANGES:
                target = OUT / f"{symbol}_{start[:7]}.parquet"
                if target.exists():
                    print(json.dumps({"symbol": symbol, "month": start[:7], "status": "already"}), flush=True)
                    continue
                try:
                    frame = _pages(quote, symbol, start, end)
                except RuntimeError as exc:
                    print(json.dumps({
                        "symbol": symbol,
                        "month": start[:7],
                        "status": "failed",
                        "error": str(exc)[:300],
                    }), flush=True)
                    if str(exc).startswith("quota"):
                        return
                    continue
                if start == "2026-01-01" and not _january_ok(frame):
                    print(json.dumps({
                        "symbol": symbol,
                        "month": "2026-01",
                        "status": "bad_filter",
                        "rows": int(len(frame)),
                    }), flush=True)
                    raise SystemExit(2)
                frame.to_parquet(target, index=False)
                print(json.dumps({
                    "symbol": symbol,
                    "month": start[:7],
                    "status": "saved",
                    "rows": int(len(frame)),
                    "at": datetime.now(timezone.utc).isoformat(),
                }), flush=True)
        print(json.dumps({"quota": _quota(quote), "status": "finished"}), flush=True)
    finally:
        quote.close()


if __name__ == "__main__":
    main()
