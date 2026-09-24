#!/usr/bin/env python3
"""Export the Futu OpenD U.S. watchlist to a flat CSV (read-only, one row per symbol)."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path

from futu import RET_OK, OpenQuoteContext

COLUMNS = [
    "ticker", "code", "name", "lot_size", "stock_type", "stock_child_type",
    "listing_date", "last_trade_time",
    "last_price", "open_price", "high_price", "low_price", "prev_close_price",
    "volume", "turnover", "amplitude", "turnover_rate", "market_val",
    "pe_ratio", "pb_ratio", "update_time", "retrieved_at_utc",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=11111)
    parser.add_argument("--group", default="US", help="watchlist group name (default: US)")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    retrieved = datetime.now().astimezone().isoformat(timespec="seconds")
    context = OpenQuoteContext(host=args.host, port=args.port)
    rows, missing = [], []
    try:
        ret, data = context.get_user_security(args.group)
        if ret != RET_OK:
            raise SystemExit(f"cannot read watchlist group {args.group!r}: {data}")
        codes = [str(x) for x in data["code"]] if len(data) else []
        if not codes:
            raise SystemExit(f"watchlist group {args.group!r} is empty")
        snapshots = {}
        for offset in range(0, len(codes), 60):
            batch = codes[offset:offset + 60]
            sret, sdata = context.get_market_snapshot(batch)
            if sret == RET_OK:
                for _, row in sdata.iterrows():
                    snapshots[str(row["code"])] = row
        for _, row in data.iterrows():
            code = str(row["code"])
            snap = snapshots.get(code)
            if snap is None:
                missing.append(code)
            rows.append({
                "ticker": code.split(".", 1)[-1],
                "code": code,
                "name": str(row.get("name", "")),
                "lot_size": row.get("lot_size"),
                "stock_type": str(row.get("stock_type", "")),
                "stock_child_type": str(row.get("stock_child_type", "")),
                "listing_date": str(row.get("listing_date", "") or ""),
                "last_trade_time": str(row.get("last_trade_time", "") or ""),
                "last_price": None if snap is None else snap.get("last_price"),
                "open_price": None if snap is None else snap.get("open_price"),
                "high_price": None if snap is None else snap.get("high_price"),
                "low_price": None if snap is None else snap.get("low_price"),
                "prev_close_price": None if snap is None else snap.get("prev_close_price"),
                "volume": None if snap is None else snap.get("volume"),
                "turnover": None if snap is None else snap.get("turnover"),
                "amplitude": None if snap is None else snap.get("amplitude"),
                "turnover_rate": None if snap is None else snap.get("turnover_rate"),
                "market_val": None if snap is None else snap.get("market_val"),
                "pe_ratio": None if snap is None else snap.get("pe_ratio"),
                "pb_ratio": None if snap is None else snap.get("pb_ratio"),
                "update_time": None if snap is None else str(snap.get("update_time", "")),
                "retrieved_at_utc": retrieved,
            })
    finally:
        context.close()

    rows.sort(key=lambda r: r["ticker"])
    out = args.output or Path(f"data/futu_watchlist_us_{datetime.now().astimezone():%Y-%m-%d}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"group={args.group} rows={len(rows)} unique_tickers={len({r['ticker'] for r in rows})} "
          f"no_snapshot={len(missing)}")
    if missing:
        print("no snapshot for:", missing)
    print("output:", out)


if __name__ == "__main__":
    main()
