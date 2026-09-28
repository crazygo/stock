#!/usr/bin/env python3
"""Freeze the local FutuD ETF watchlist as security entities for matrix runs.

This reads watchlist metadata only. It never requests prices, places orders, or
uploads market data. Run again to create a new immutable snapshot/version.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from .build import ROOT, write_json


def capture(group_name: str = "ETF") -> dict:
    import futu as ft

    ft.SysConfig.enable_proto_encrypt(False)
    quote = ft.OpenQuoteContext(host="127.0.0.1", port=11111)
    try:
        ret, groups = quote.get_user_security_group()
        if ret != ft.RET_OK or group_name not in set(groups.group_name):
            raise RuntimeError(f"FutuD group unavailable: {group_name}")
        ret, frame = quote.get_user_security(group_name)
        if ret != ft.RET_OK:
            raise RuntimeError(f"FutuD group read failed: {group_name}: {frame}")
    finally:
        quote.close()
    if frame.empty:
        raise ValueError("ETF group is empty; refusing to replace a snapshot")
    if not frame.code.astype(str).str.startswith("US.").all() or not frame.stock_type.astype(str).eq("ETF").all():
        raise ValueError("ETF group contains a non-US or non-ETF entry; review it before importing")
    if frame.code.duplicated().any():
        raise ValueError("ETF group contains duplicate security codes")
    observed = datetime.now(timezone.utc).isoformat()
    members = sorted(({
        "security_id": str(row.code), "symbol": str(row.code)[3:], "name": str(row.name).strip(),
        "instrument_type": "etf", "listing_date": str(row.listing_date),
    } for row in frame.itertuples(index=False)), key=lambda x: x["symbol"])
    return {"schema_version": "1.0", "group_id": "watchlist:ETF", "group_name": group_name,
            "source": "local_FutuD_OpenQuoteContext.get_user_security", "observed_at": observed,
            "membership_basis": "current_watchlist_snapshot_retrospective", "members": members}


def make_universe(original: dict, snapshot: dict) -> dict:
    if len(snapshot["members"]) != len({m["symbol"] for m in snapshot["members"]}):
        raise ValueError("ETF snapshot has duplicate symbols")
    universe = json.loads(json.dumps(original))
    members = {m["symbol"]: m for m in universe["members"]}
    for m in members.values():
        m["instrument_type"] = "etf" if m["role"] == "benchmark" else "stock"
        m["roles"] = [m["role"]]
    for item in snapshot["members"]:
        symbol = item["symbol"]
        if symbol in members:
            row = members[symbol]
            if row["role"] == "candidate" and row["instrument_type"] != "etf":
                raise ValueError(f"ETF symbol conflicts with a stock candidate: {symbol}")
            row["role"] = "candidate"
            row["roles"] = sorted(set(row["roles"] + ["candidate", "futud_etf_group"]))
        else:
            row = {"security_id": item["security_id"], "symbol": symbol,
                   "role": "candidate", "roles": ["candidate", "futud_etf_group"],
                   "universe_id": "qqq_plus_futud_etf_snapshot", "universe_mode": "current_watchlist_snapshot_retrospective",
                   "effective_from": None, "effective_to": None,
                   "announced_at": None, "available_at": snapshot["observed_at"],
                   "first_seen_at": snapshot["observed_at"], "source_url": None,
                   "industry_proxy": None}
            members[symbol] = row
        row["instrument_type"] = "etf"
        row["name"] = item["name"]
        row["listing_date"] = item["listing_date"]
        row["watchlist_group_ids"] = [snapshot["group_id"]]
        row["watchlist_observed_at"] = snapshot["observed_at"]
    universe["members"] = sorted(members.values(), key=lambda x: x["symbol"])
    universe["universe_id"] = "qqq_plus_futud_etf_snapshot"
    universe["universe_mode"] = "current_universe_and_watchlist_retrospective"
    universe["generated_at"] = snapshot["observed_at"]
    universe["source_snapshot"] = "research/group_expectation_matrix/config/futud_etf_snapshot.json"
    universe["total_members"] = len(universe["members"])
    universe["candidates_count"] = sum(m["role"] == "candidate" for m in universe["members"])
    universe["benchmarks_count"] = sum("benchmark" in m["roles"] for m in universe["members"])
    universe.pop("universe_sha256", None)
    return universe


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=ROOT/"research/group_expectation_matrix/config/futud_etf_snapshot.json")
    parser.add_argument("--universe", type=Path, default=ROOT/"market_data/universe/qqq_plus_futud_etf_snapshot.json")
    args = parser.parse_args()
    for path in (args.snapshot, args.universe):
        if path.exists():
            raise FileExistsError(f"Choose a new snapshot path instead of overwriting: {path}")
    snapshot = capture()
    source = json.loads((ROOT/"market_data/universe/qqq_retrospective_v1.json").read_text())
    universe = make_universe(source, snapshot)
    write_json(args.snapshot, snapshot)
    write_json(args.universe, universe)
    print(json.dumps({"observed_at": snapshot["observed_at"], "etf_count": len(snapshot["members"]),
                      "candidate_entities": universe["candidates_count"], "snapshot": str(args.snapshot),
                      "universe": str(args.universe)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
