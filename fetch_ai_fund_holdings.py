#!/usr/bin/env python3
"""Archive dated Global X thematic ETF holdings for point-in-time discovery."""

from __future__ import annotations

import argparse
import csv
import io
import json
import time
from datetime import date, datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

FUNDS = {
    "AIQ": "artificial_intelligence_and_technology",
    "BOTZ": "robotics_and_artificial_intelligence",
    "CLOU": "cloud_computing",
    "DTCR": "data_center_and_digital_infrastructure",
}
URL = "https://assets.globalxetfs.com/funds/holdings/{fund}_full-holdings_{day}.csv"


def parse_holdings(content: bytes) -> tuple[str, list[dict]]:
    lines = content.decode("utf-8-sig").splitlines()
    if len(lines) < 4 or "Fund Holdings Data as of" not in lines[1]:
        raise ValueError("unexpected ETF holdings format")
    reader = csv.DictReader(io.StringIO("\n".join(lines[2:])))
    rows = []
    for row in reader:
        ticker = (row.get("Ticker") or "").strip().upper()
        if not ticker or " " in ticker or not ticker.replace(".", "").isalpha():
            continue
        rows.append({"ticker": ticker, "name": row.get("Name", "").strip(),
                     "weight_pct": float(row["% of Net Assets"]),
                     "sedol": row.get("SEDOL", "").strip()})
    return f"{lines[0]} — {lines[1]}", rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-manifest", type=Path, default=Path("market_data/us/manifest.json"))
    parser.add_argument("--output", type=Path, default=Path("data/ai_fund_holdings.json"))
    parser.add_argument("--latest-as-of", default="2026-09-22")
    args = parser.parse_args()
    market_days = sorted(date.fromisoformat(x) for x in json.loads(args.archive_manifest.read_text())["archives"])
    month_ends = {}
    for day in market_days:
        month_ends[(day.year, day.month)] = day
    snapshot_days = sorted(set(month_ends.values()) | {date.fromisoformat(args.latest_as_of)})
    snapshots = []
    failures = []
    for day in snapshot_days:
        for fund, theme in FUNDS.items():
            url = URL.format(fund=fund.lower(), day=day.strftime("%Y%m%d"))
            try:
                request = Request(url, headers={"User-Agent": "Mozilla/5.0 (stock-research)"})
                with urlopen(request, timeout=20) as response:
                    raw = response.read()
                title, holdings = parse_holdings(raw)
                if len(holdings) < 10:
                    raise ValueError(f"too few stock rows: {len(holdings)}")
                snapshots.append({"fund": fund, "theme": theme, "as_of": day.isoformat(),
                                  "source_url": url, "source_title": title, "holdings": holdings})
            except Exception as exc:
                failures.append({"fund": fund, "as_of": day.isoformat(),
                                 "error": f"{type(exc).__name__}: {exc}"})
            time.sleep(0.1)
        print(f"ETF holdings: {day} ({len(snapshots)} snapshots)", flush=True)
    result = {"source": "Global X dated full-holdings CSVs", "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
              "funds": FUNDS, "snapshots": snapshots, "failures": failures}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(json.dumps({"snapshots": len(snapshots), "failures": failures, "output": str(args.output)}))


if __name__ == "__main__":
    main()
