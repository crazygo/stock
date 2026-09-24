#!/usr/bin/env python3
"""Export every Futu OpenD watchlist group to a single CSV (read-only).

Reads group membership from the local Futu OpenD gateway and joins a market
snapshot for each symbol so the CSV is usable without opening Futu.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path

from futu import RET_OK, OpenQuoteContext

COLUMNS = [
    "group_name", "group_type", "code", "ticker", "market", "name", "lot_size",
    "stock_type", "stock_child_type", "stock_owner", "option_type", "strike_time",
    "strike_price", "suspension", "delisting", "listing_date", "last_trade_time",
    "last_price", "open_price", "high_price", "low_price", "prev_close_price",
    "volume", "turnover", "amplitude", "turnover_rate", "market_val",
    "pe_ratio", "pb_ratio", "update_time", "retrieved_at_utc",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=11111)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    retrieved = datetime.now(timezone.utc).isoformat(timespec="seconds")
    context = OpenQuoteContext(host=args.host, port=args.port)
    rows, problems = [], []
    try:
        ret, groups = context.get_user_security_group()
        if ret != RET_OK:
            raise SystemExit(f"cannot list watchlist groups: {groups}")
        group_names = [str(n) for n in groups["group_name"]]
        group_types = dict(zip(groups["group_name"], groups["group_type"]))

        for name in group_names:
            ret, data = context.get_user_security(name)
            if ret != RET_OK:
                problems.append({"group_name": name, "error": str(data)[:200]})
                continue
            codes = [str(x) for x in data["code"]] if len(data) else []
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
                rows.append({
                    "group_name": name,
                    "group_type": str(group_types.get(name, "")),
                    "code": code,
                    "ticker": code.split(".", 1)[-1],
                    "market": code.split(".", 1)[0],
                    "name": str(row.get("name", "")),
                    "lot_size": row.get("lot_size"),
                    "stock_type": str(row.get("stock_type", "")),
                    "stock_child_type": str(row.get("stock_child_type", "")),
                    "stock_owner": str(row.get("stock_owner", "")),
                    "option_type": str(row.get("option_type", "")),
                    "strike_time": row.get("strike_time"),
                    "strike_price": row.get("strike_price"),
                    "suspension": row.get("suspension"),
                    "delisting": row.get("delisting"),
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

    rows.sort(key=lambda r: (r["group_name"], r["code"]))
    out = args.output or Path(f"data/futu_watchlist_{datetime.now().astimezone():%Y-%m-%d}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    by_group = {}
    for r in rows:
        by_group[r["group_name"]] = by_group.get(r["group_name"], 0) + 1
    print(f"groups: {len(by_group)}  rows: {len(rows)}  unique codes: {len({r['code'] for r in rows})}")
    for g, n in sorted(by_group.items(), key=lambda x: -x[1]):
        print(f"  {g}: {n}")
    if problems:
        print("problems:", problems)
    print("output:", out)


if __name__ == "__main__":
    main()
