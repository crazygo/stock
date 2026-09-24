#!/usr/bin/env python3
"""Read one completed U.S. market session from local Futu OpenD, without trading."""

from __future__ import annotations

import argparse
import json
from datetime import date, datetime, time, timezone
from pathlib import Path
from zoneinfo import ZoneInfo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True, help="Completed U.S. session YYYY-MM-DD")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=11111)
    parser.add_argument("--candidates", type=Path, default=Path("ai_universe_candidates.json"))
    parser.add_argument("--holdings", type=Path, default=Path("data/ai_fund_holdings.json"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    session = date.fromisoformat(args.session)
    new_york_now = datetime.now(ZoneInfo("America/New_York"))
    if session > new_york_now.date() or (session == new_york_now.date() and new_york_now.time() < time(17)):
        raise SystemExit("The requested U.S. session has not completed; wait until at least 17:00 New York time")
    try:
        from futu import OpenQuoteContext, RET_OK
    except ImportError as exc:
        raise SystemExit("Install futu-api in the selected Python environment") from exc
    candidates = json.loads(args.candidates.read_text())
    symbols = {s for group in candidates["groups"].values() for s in group["tickers"]}
    holdings = json.loads(args.holdings.read_text())
    snapshots = [s for s in holdings["snapshots"] if s["as_of"] == args.session]
    symbols.update(h["ticker"] for s in snapshots for h in s["holdings"])
    symbols.update(("QQQ", "SOXX", "IGV", "XLU"))
    symbols = sorted(symbols)
    context = OpenQuoteContext(host=args.host, port=args.port)
    bars = {}
    errors = {}
    def fetch_group(group: list[str]) -> None:
        ret, data = context.get_market_snapshot(["US." + s for s in group])
        if ret != RET_OK:
            if len(group) > 1:
                middle = len(group) // 2
                fetch_group(group[:middle])
                fetch_group(group[middle:])
            else:
                errors[group[0]] = str(data)[:300]
            return
        returned = set()
        for _, row in data.iterrows():
            symbol = row["code"].removeprefix("US.")
            returned.add(symbol)
            update_time = str(row["update_time"])
            if not update_time.startswith(args.session):
                errors[symbol] = f"quote update {update_time} does not match {args.session}"
                continue
            if row["suspension"] or any(float(row[k]) <= 0 for k in ("last_price", "open_price", "high_price", "low_price")):
                errors[symbol] = "suspended or invalid OHLC"
                continue
            bars[symbol] = {
                "T": symbol, "o": float(row["open_price"]), "h": float(row["high_price"]),
                "l": float(row["low_price"]), "c": float(row["last_price"]),
                "v": float(row["volume"]), "update_time": update_time,
            }
        for symbol in set(group) - returned:
            errors[symbol] = "not returned by OpenD"
    try:
        for offset in range(0, len(symbols), 50):
            fetch_group(symbols[offset:offset + 50])
    finally:
        context.close()
    output = args.output or Path(f"data/futu_snapshot_{args.session}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {"session": args.session, "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
               "source": "Futu OpenD get_market_snapshot; regular-session OHLC from last completed U.S. session",
               "bars": bars, "errors": errors}
    output.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(json.dumps({"requested": len(symbols), "bars": len(bars), "errors": errors, "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
