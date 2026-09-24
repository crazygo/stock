#!/usr/bin/env python3
"""Fetch compact, filing-dated annual SEC facts for the AI research universe.

The output retains the original filing date and accession number. Backtests
must select facts with filed <= signal date; today's SEC response is never used
as a present-day fundamental snapshot for historical signals.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

SEC_TICKERS = "https://www.sec.gov/files/company_tickers.json"
SEC_FACTS = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
USER_AGENT = os.environ.get("SEC_USER_AGENT", "")
TAGS = {
    "revenue": [
        "RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues",
        "SalesRevenueNet", "SalesRevenueGoodsNet", "RevenueFromContractWithCustomerIncludingAssessedTax",
        "OperatingRevenues", "RevenueFromContractWithCustomerExcludingAssessedTaxNetOfInterestExpense",
    ],
    "operating_cashflow": ["NetCashProvidedByUsedInOperatingActivities"],
    "net_income": ["NetIncomeLoss", "ProfitLoss", "NetIncomeLossAvailableToCommonStockholdersBasic"],
}


def fetch_json(url: str) -> dict:
    request = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    for attempt in range(3):
        try:
            with urlopen(request, timeout=30) as response:
                return json.load(response)
        except (HTTPError, URLError, TimeoutError):
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


def extract_facts(payload: dict) -> dict[str, list[dict]]:
    taxonomies = payload.get("facts", {})
    out: dict[str, list[dict]] = {}
    for metric, tags in TAGS.items():
        rows: list[dict] = []
        for taxonomy in ("us-gaap", "ifrs-full"):
            facts = taxonomies.get(taxonomy, {})
            for priority, tag in enumerate(tags):
                for item in facts.get(tag, {}).get("units", {}).get("USD", []):
                    if item.get("form") not in {"10-K", "10-K/A", "20-F", "40-F"}:
                        continue
                    if not all(k in item for k in ("start", "end", "filed", "val", "accn")):
                        continue
                    duration = (datetime.fromisoformat(item["end"]) - datetime.fromisoformat(item["start"])).days
                    if not 330 <= duration <= 400:
                        continue
                    if item["end"] < "2023-01-01":
                        continue
                    rows.append({
                        "start": item["start"], "end": item["end"], "filed": item["filed"],
                        "value": item["val"], "accession": item["accn"],
                        "form": item["form"], "tag": tag, "taxonomy": taxonomy,
                        "tag_priority": priority,
                    })
        out[metric] = sorted(rows, key=lambda r: (r["end"], r["filed"], -r["tag_priority"]))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=Path("ai_universe_candidates.json"))
    parser.add_argument("--holdings", type=Path, default=Path("data/ai_fund_holdings.json"))
    parser.add_argument("--output", type=Path, default=Path("data/ai_sec_annual_facts.json"))
    args = parser.parse_args()
    if not USER_AGENT:
        raise SystemExit("Set SEC_USER_AGENT to your organization name and contact address for SEC fair access")
    candidates = json.loads(args.candidates.read_text())
    symbols = {s for group in candidates["groups"].values() for s in group["tickers"]}
    if args.holdings.exists():
        holdings = json.loads(args.holdings.read_text())
        symbols.update(h["ticker"] for snap in holdings["snapshots"] for h in snap["holdings"])
    symbols = sorted(symbols)
    ticker_map = {
        row["ticker"]: (row["cik_str"], row["title"])
        for row in fetch_json(SEC_TICKERS).values()
    }
    previous = json.loads(args.output.read_text()) if args.output.exists() else {}
    companies = previous.get("companies", {})
    errors = {}
    for index, symbol in enumerate(symbols, 1):
        if symbol in companies:
            continue
        if symbol not in ticker_map:
            errors[symbol] = "not in SEC current ticker map"
            continue
        cik, name = ticker_map[symbol]
        try:
            payload = fetch_json(SEC_FACTS.format(cik=cik))
            companies[symbol] = {"cik": cik, "name": name, "facts": extract_facts(payload)}
        except (HTTPError, URLError, TimeoutError, ValueError) as exc:
            errors[symbol] = f"{type(exc).__name__}: {exc}"
        if index % 20 == 0:
            print(f"SEC filings fetched: {index}/{len(symbols)}", flush=True)
        time.sleep(0.12)
    output = {
        "source": "SEC EDGAR companyfacts API; values are annual entity-level USD XBRL facts",
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "companies": companies,
        "errors": errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, separators=(",", ":")) + "\n")
    print(json.dumps({"candidates": len(symbols), "fetched": len(companies), "errors": errors, "output": str(args.output)}))


if __name__ == "__main__":
    main()
