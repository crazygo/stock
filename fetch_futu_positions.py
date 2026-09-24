#!/usr/bin/env python3
"""Export Futu OpenD positions to CSV (read-only, no unlock required for queries)."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path

from futu import RET_OK, OpenQuoteContext, OpenSecTradeContext, TrdEnv

COLUMNS = [
    "acc_id", "trd_env", "acc_type", "acc_status", "trdmarket_auth",
    "ticker", "code", "stock_name", "position_market", "currency",
    "qty", "can_sell_qty", "cost_price", "average_cost", "diluted_cost",
    "nominal_price", "market_val", "pl_val", "pl_ratio", "pl_ratio_avg_cost",
    "unrealized_pl", "realized_pl", "today_buy_qty", "today_buy_val",
    "today_sell_qty", "today_sell_val", "today_pl_val", "position_side",
    "last_price", "prev_close_price", "snapshot_update_time", "retrieved_at_utc",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=11111)
    parser.add_argument("--include-sim", action="store_true",
                        help="also export SIMULATE accounts (default: real accounts only)")
    parser.add_argument("--market", default="US", choices=["US", "HK", "CN", "HKCC", "HKFUT"])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    retrieved = datetime.now().astimezone().isoformat(timespec="seconds")
    rows, summary = [], []
    quote = OpenQuoteContext(host=args.host, port=args.port)
    trade = OpenSecTradeContext(host=args.host, port=args.port,
                                filter_trdmarket=getattr(TrdMarket, args.market))
    try:
        ret, accs = trade.get_acc_list()
        if ret != RET_OK:
            raise SystemExit(f"cannot list accounts: {accs}")
        for _, acc in accs.iterrows():
            if str(acc["trd_env"]) != "REAL" and not args.include_sim:
                continue
            acc_id = acc["acc_id"]
            pret, positions = trade.position_list_query(acc_id=acc_id)
            if pret != RET_OK:
                summary.append({"acc_id": acc_id, "trd_env": acc["trd_env"],
                                "error": str(positions)[:200]})
                continue
            codes = [str(x) for x in positions["code"]] if len(positions) else []
            snaps = {}
            for offset in range(0, len(codes), 60):
                sret, sdata = quote.get_market_snapshot(codes[offset:offset + 60])
                if sret == RET_OK:
                    for _, row in sdata.iterrows():
                        snaps[str(row["code"])] = row
            base = {
                "acc_id": acc_id, "trd_env": str(acc["trd_env"]),
                "acc_type": str(acc["acc_type"]), "acc_status": str(acc["acc_status"]),
                "trdmarket_auth": str(acc["trdmarket_auth"]),
                "retrieved_at_utc": retrieved,
            }
            for _, p in positions.iterrows():
                code = str(p["code"])
                snap = snaps.get(code)
                rows.append({**base,
                             "ticker": code.split(".", 1)[-1], "code": code,
                             "stock_name": str(p.get("stock_name", "")),
                             "position_market": str(p.get("position_market", "")),
                             "currency": str(p.get("currency", "")),
                             "qty": p.get("qty"), "can_sell_qty": p.get("can_sell_qty"),
                             "cost_price": p.get("cost_price"),
                             "average_cost": p.get("average_cost"),
                             "diluted_cost": p.get("diluted_cost"),
                             "nominal_price": p.get("nominal_price"),
                             "market_val": p.get("market_val"),
                             "pl_val": p.get("pl_val"), "pl_ratio": p.get("pl_ratio"),
                             "pl_ratio_avg_cost": p.get("pl_ratio_avg_cost"),
                             "unrealized_pl": p.get("unrealized_pl"),
                             "realized_pl": p.get("realized_pl"),
                             "today_buy_qty": p.get("today_buy_qty"),
                             "today_buy_val": p.get("today_buy_val"),
                             "today_sell_qty": p.get("today_sell_qty"),
                             "today_sell_val": p.get("today_sell_val"),
                             "today_pl_val": p.get("today_pl_val"),
                             "position_side": str(p.get("position_side", "")),
                             "last_price": None if snap is None else snap.get("last_price"),
                             "prev_close_price": None if snap is None else snap.get("prev_close_price"),
                             "snapshot_update_time": None if snap is None else str(snap.get("update_time", "")),
                             })
            summary.append({"acc_id": acc_id, "trd_env": str(acc["trd_env"]),
                            "positions": len(positions),
                            "market_val": float(positions["market_val"].sum()) if len(positions) else 0.0,
                            "pl_val": float(positions["pl_val"].sum()) if len(positions) else 0.0,
                            "currency": ",".join(sorted({str(x) for x in positions["currency"]})) if len(positions) else ""})
            # account cash / assets
            aret, info = trade.accinfo_query(acc_id=acc_id)
            if aret == RET_OK and len(info):
                summary[-1].update({
                    "total_assets": float(info.iloc[0].get("total_assets") or 0),
                    "cash": float(info.iloc[0].get("cash") or 0),
                    "market_val_total": float(info.iloc[0].get("market_val") or 0),
                    "unrealized_pl_total": float(info.iloc[0].get("unrealized_pl") or 0),
                })
    finally:
        quote.close()
        trade.close()

    rows.sort(key=lambda r: (r["trd_env"], r["ticker"]))
    out = args.output or Path(f"data/futu_positions_{datetime.now().astimezone():%Y-%m-%d}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    for s in summary:
        print(s)
    print(f"rows={len(rows)} unique_tickers={len({r['ticker'] for r in rows})} output={out}")


if __name__ == "__main__":
    main()
