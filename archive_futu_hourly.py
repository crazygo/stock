#!/usr/bin/env python3
"""Archive Futu 60-minute U.S. bars, newest year first, with resumable chunks."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS = ROOT / "analysis" / "preopen_three_week" / "results.json"
DEFAULT_ARCHIVE = ROOT / "market_data" / "us_60m"
BENCHMARKS = ("QQQ", "SOXX", "IGV", "XLU")


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def as_json_value(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


def load_symbols(results_path: Path, archive_dir: Path, explicit: list[str] | None) -> list[str]:
    if explicit:
        symbols = {symbol.strip().upper().removeprefix("US.") for symbol in explicit if symbol.strip()}
    else:
        payload = json.loads(results_path.read_text(encoding="utf-8"))
        symbols = {str(row["ticker"]).upper() for row in payload.get("current", []) if row.get("ticker")}
        symbols.update(BENCHMARKS)
        pins_file = ROOT / "analysis" / "preopen_three_week" / "pinned_stocks.json"
        if pins_file.exists():
            try:
                pins = json.loads(pins_file.read_text(encoding="utf-8"))
                symbols.update(str(t).upper() for t in pins.get("tickers", []))
            except (OSError, json.JSONDecodeError, AttributeError):
                pass
    if not symbols:
        raise RuntimeError("No tickers found in the selected universe")
    return sorted(symbols)


def year_ranges(start: date, end: date):
    for year in range(end.year, start.year - 1, -1):
        chunk_start = max(start, date(year, 1, 1))
        chunk_end = min(end, date(year, 12, 31))
        if chunk_start <= chunk_end:
            yield year, chunk_start, chunk_end


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", help="Oldest date YYYY-MM-DD (default: eight years before today)")
    parser.add_argument("--end", help="Latest date YYYY-MM-DD (default: today in New York)")
    parser.add_argument("--years", type=int, default=8, help="Default lookback; Futu documents up to 8 years for minute bars")
    parser.add_argument("--symbols", nargs="*", help="Optional explicit ticker list")
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=11111)
    parser.add_argument("--first-page-delay", type=float, default=0.65, help="Seconds between first-page requests (Futu limit: 60 per 30 seconds)")
    parser.add_argument("--page-delay", type=float, default=0.15, help="Small courtesy delay between continuation pages")
    args = parser.parse_args()
    if args.years < 1 or args.years > 8:
        parser.error("--years must be between 1 and 8")
    end = date.fromisoformat(args.end) if args.end else datetime.now(ZoneInfo("America/New_York")).date()
    if args.start:
        start = date.fromisoformat(args.start)
    else:
        try:
            start = end.replace(year=end.year - args.years)
        except ValueError:
            start = end.replace(year=end.year - args.years, day=28)
    if start > end:
        parser.error("--start must be on or before --end")

    try:
        from futu import AuType, KL_FIELD, KLType, OpenQuoteContext, RET_OK, Session
    except ImportError as exc:
        raise SystemExit("Install the official Python client first: python3 -m pip install futu-api") from exc

    symbols = load_symbols(args.results, args.archive, args.symbols)
    args.archive.mkdir(parents=True, exist_ok=True)
    manifest_path = args.archive / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            manifest = {}
    else:
        manifest = {}
    manifest_symbols = sorted(set(manifest.get("symbols", [])) | set(symbols))
    manifest.update({
        "version": 1,
        "market": "us_stocks_60_minute",
        "source": "Futu OpenD request_history_kline",
        "bar_interval": "60m",
        "adjustment": "qfq",
        "extended_time": True,
        "session": "ALL",
        "requested_start": start.isoformat(),
        "requested_end": end.isoformat(),
        "symbols": manifest_symbols,
        "chunks": manifest.get("chunks", {}),
    })

    context = OpenQuoteContext(host=args.host, port=args.port)
    requests_first_page = 0
    pages_total = 0
    bars_total = 0
    failures: list[dict[str, str]] = []
    try:
        ret, quota = context.get_history_kl_quota(get_detail=False)
        if ret != RET_OK:
            raise RuntimeError(f"Could not read Futu history quota: {quota}")
        remain_quota = int(quota[1])
        print(json.dumps({"symbols": len(symbols), "remaining_history_quota": remain_quota,
                          "start": start.isoformat(), "end": end.isoformat()}, ensure_ascii=False), flush=True)
        if remain_quota < len(symbols):
            raise RuntimeError(
                f"Not enough Futu history quota for {len(symbols)} symbols ({remain_quota} remain). "
                "No history requests were sent; retry after the quota cycle resets or use --symbols."
            )

        for year, chunk_start, chunk_end in year_ranges(start, end):
            for symbol_index, ticker in enumerate(symbols, 1):
                relative = Path(ticker) / f"{year}.json.gz"
                output_path = args.archive / relative
                chunk_key = f"{ticker}:{year}"
                if output_path.exists() and chunk_key in manifest["chunks"]:
                    continue
                if requests_first_page:
                    time.sleep(args.first_page_delay)
                page_key = None
                records: list[dict[str, Any]] = []
                page_count = 0
                while True:
                    if page_count:
                        time.sleep(args.page_delay)
                    try:
                        ret, frame, next_key = context.request_history_kline(
                            code=f"US.{ticker}",
                            start=chunk_start.isoformat(),
                            end=chunk_end.isoformat(),
                            ktype=KLType.K_60M,
                            autype=AuType.QFQ,
                            fields=KL_FIELD.ALL,
                            max_count=1000,
                            page_req_key=page_key,
                            extended_time=True,
                            session=Session.ALL,
                        )
                    except Exception as exc:
                        failures.append({"ticker": ticker, "year": str(year), "error": str(exc)[:500]})
                        break
                    if page_count == 0:
                        requests_first_page += 1
                    if ret != RET_OK:
                        message = str(frame)
                        failures.append({"ticker": ticker, "year": str(year), "error": message[:500]})
                        if any(word in message.lower() for word in ("quota", "额度", "频率", "rate limit", "too frequent")):
                            print(f"Stopping on quota/rate response at {ticker} {year}: {message[:240]}", flush=True)
                            atomic_json(manifest_path, manifest)
                            return 2
                        break
                    page_count += 1
                    pages_total += 1
                    if frame is not None and not frame.empty:
                        for row in frame.to_dict("records"):
                            records.append({str(key): as_json_value(value) for key, value in row.items()})
                    if not next_key:
                        break
                    page_key = next_key
                else:
                    pass
                if failures and failures[-1].get("ticker") == ticker and failures[-1].get("year") == str(year):
                    continue
                records.sort(key=lambda row: str(row.get("time_key", row.get("time", ""))))
                payload = {
                    "ticker": ticker,
                    "year": year,
                    "interval": "60m",
                    "start": chunk_start.isoformat(),
                    "end": chunk_end.isoformat(),
                    "adjustment": "qfq",
                    "extended_time": True,
                    "session": "ALL",
                    "bars": records,
                }
                raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
                compressed = gzip.compress(raw, compresslevel=6, mtime=0)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                temporary = output_path.with_suffix(".json.gz.tmp")
                temporary.write_bytes(compressed)
                temporary.replace(output_path)
                timestamps = [str(row.get("time_key", row.get("time", ""))) for row in records]
                manifest["chunks"][chunk_key] = {
                    "path": str(relative),
                    "bars": len(records),
                    "pages": page_count,
                    "first_bar": timestamps[0] if timestamps else None,
                    "last_bar": timestamps[-1] if timestamps else None,
                    "compressed_bytes": len(compressed),
                    "sha256_uncompressed": hashlib.sha256(raw).hexdigest(),
                    "retrieved_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                }
                atomic_json(manifest_path, manifest)
                bars_total += len(records)
                print(f"[{year}] {symbol_index}/{len(symbols)} {ticker}: {len(records)} bars, {page_count} pages", flush=True)
    finally:
        context.close()

    manifest["last_run"] = {
        "finished_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "requests_first_page": requests_first_page,
        "pages": pages_total,
        "bars": bars_total,
        "failures": failures,
    }
    atomic_json(manifest_path, manifest)
    print(json.dumps(manifest["last_run"], ensure_ascii=False), flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
