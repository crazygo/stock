#!/usr/bin/env python3
"""Build a repeatable U.S. + Hong Kong optical-primary stock universe."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

USER_AGENT = "stock-optical-universe/0.1"
YAHOO_CHART = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"


def load_universe(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload.get("companies"), list):
        raise ValueError("universe JSON must contain a companies array")
    return payload


def select_companies(
    companies: Iterable[dict[str, Any]],
    *,
    market: str,
    include_broad: bool,
    include_watch: bool,
) -> list[dict[str, Any]]:
    allowed_status = {"strict"}
    if include_broad:
        allowed_status.add("broad")
    if include_watch:
        allowed_status.add("watch")
    market = market.upper()
    rows = [
        dict(company)
        for company in companies
        if (market == "BOTH" or company["market"] == market)
        and company["status"] in allowed_status
    ]
    rows.sort(key=lambda row: (row["market"], row["symbol"]))
    return rows


def fetch_yahoo_quote(symbol: str, timeout: float = 12.0) -> dict[str, Any]:
    url = YAHOO_CHART.format(symbol=quote(symbol, safe="")) + "?range=5d&interval=1d"
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    result = ((payload.get("chart") or {}).get("result") or [None])[0]
    if not result:
        error = (payload.get("chart") or {}).get("error")
        raise ValueError(f"no quote result: {error}")
    meta = result.get("meta") or {}
    price = meta.get("regularMarketPrice")
    if price is None:
        closes = (((result.get("indicators") or {}).get("quote") or [{}])[0].get("close") or [])
        price = next((value for value in reversed(closes) if value is not None), None)
    timestamp = meta.get("regularMarketTime")
    quote_time = (
        datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()
        if isinstance(timestamp, (int, float))
        else None
    )
    return {
        "price": price,
        "currency": meta.get("currency"),
        "exchange": meta.get("exchangeName") or meta.get("fullExchangeName"),
        "quote_time_utc": quote_time,
    }


def enrich_quotes(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        try:
            row.update(fetch_yahoo_quote(row["quote_symbol"]))
            row["quote_status"] = "ok"
            row["quote_error"] = None
        except (HTTPError, URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
            row.update({"price": None, "currency": None, "exchange": None, "quote_time_utc": None})
            row["quote_status"] = "error"
            row["quote_error"] = str(exc)[:240]


def render_markdown(payload: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    generated = datetime.now(timezone.utc).isoformat()
    lines = [
        "# Optical-primary stock universe",
        "",
        f"- Evidence as of: **{payload['as_of']}**",
        f"- Generated at: `{generated}`",
        f"- Scope: {payload['scope']}",
        "- Default list contains only `strict` names. `broad` and `watch` require explicit flags.",
        "",
        "| Market | Symbol | Company | Optical share | Evidence period | Basis | Price |",
        "|---|---|---|---:|---|---|---:|",
    ]
    for row in rows:
        share = "structural" if row.get("optical_revenue_pct") is None else f"{row['optical_revenue_pct']:.2f}%"
        price = "—"
        if row.get("price") is not None:
            currency = row.get("currency") or ""
            price = f"{row['price']:.2f} {currency}".strip()
        lines.append(
            f"| {row['market']} | **{row['symbol']}** | {row['name']} | {share} | "
            f"{row['evidence_period']} | {row['basis_type']} | {price} |"
        )
    lines.extend(["", "## Evidence notes", ""])
    for row in rows:
        lines.extend(
            [
                f"### {row['market']} {row['symbol']} — {row['name']}",
                "",
                f"- Status: `{row['status']}`",
                f"- Evidence: {row['basis']}",
                f"- Source: {row['source']}",
                "",
            ]
        )
    return "\n".join(lines)


def write_outputs(output_dir: Path, payload: dict[str, Any], rows: list[dict[str, Any]]) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "latest.json"
    csv_path = output_dir / "latest.csv"
    md_path = output_dir / "latest.md"

    output_payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_as_of": payload["as_of"],
        "scope": payload["scope"],
        "qualification": payload["qualification"],
        "count": len(rows),
        "companies": rows,
    }
    json_path.write_text(json.dumps(output_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    fieldnames = [
        "market", "symbol", "quote_symbol", "name", "status", "optical_revenue_pct",
        "evidence_period", "basis_type", "basis", "source", "price", "currency",
        "exchange", "quote_time_utc", "quote_status", "quote_error",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    md_path.write_text(render_markdown(payload, rows), encoding="utf-8")
    return json_path, csv_path, md_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe", type=Path, default=Path("optical_universe.json"))
    parser.add_argument("--market", choices=["US", "HK", "both"], default="both")
    parser.add_argument("--include-broad", action="store_true")
    parser.add_argument("--include-watch", action="store_true")
    parser.add_argument("--fetch-quotes", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("results/optical"))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    payload = load_universe(args.universe)
    rows = select_companies(
        payload["companies"],
        market=args.market,
        include_broad=args.include_broad,
        include_watch=args.include_watch,
    )
    if args.fetch_quotes:
        enrich_quotes(rows)
    else:
        for row in rows:
            row.update({
                "price": None, "currency": None, "exchange": None, "quote_time_utc": None,
                "quote_status": "skipped", "quote_error": None,
            })
    json_path, csv_path, md_path = write_outputs(args.output_dir, payload, rows)
    print(json.dumps({"count": len(rows), "json": str(json_path), "csv": str(csv_path), "markdown": str(md_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
