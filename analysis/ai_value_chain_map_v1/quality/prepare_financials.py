#!/usr/bin/env python3
"""Build a reproducible, read-only SEC Company Facts financial snapshot.

This script deliberately performs no network access. It consumes the already
captured companyfacts and acquisition ledgers, applies an explicit as-of cutoff,
preserves source facts and abstains when a period cannot be matched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
ROOT = PROJECT.parents[1]
CACHE = ROOT / ".cache/doubling_opportunity_v1"
INPUT = PROJECT / "data.json"
OUTPUT = HERE / "financial_snapshot.json"
AS_OF = date.today()
FILED_EXCLUSIVE = (AS_OF + timedelta(days=1)).isoformat()
FORMS = {"10-K", "10-K/A", "10-Q", "10-Q/A", "20-F", "20-F/A", "40-F", "40-F/A", "6-K", "6-K/A", "8-K", "8-K/A"}

# Tag order is a deterministic same-period preference, not a claim that every
# issuer uses the same label. Every selected item keeps its original tag.
TAGS: dict[str, dict[str, list[str]]] = {
    "revenue": {"us-gaap": ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "SalesRevenueNet", "RevenueFromContractWithCustomerIncludingAssessedTax"], "ifrs-full": ["Revenue", "RevenueFromContractsWithCustomers", "RevenueFromSaleOfGoodsOrRenderingOfServices"]},
    "gross_profit": {"us-gaap": ["GrossProfit"], "ifrs-full": ["GrossProfit"]},
    "operating_income": {"us-gaap": ["OperatingIncomeLoss"], "ifrs-full": ["ProfitLossFromOperatingActivities"]},
    "net_income": {"us-gaap": ["NetIncomeLoss", "ProfitLoss"], "ifrs-full": ["ProfitLoss"]},
    "operating_cashflow": {"us-gaap": ["NetCashProvidedByUsedInOperatingActivities"], "ifrs-full": ["CashFlowsFromUsedInOperatingActivities"]},
    "capex": {"us-gaap": ["PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets"], "ifrs-full": ["PurchaseOfPropertyPlantAndEquipmentClassifiedAsInvestingActivities"]},
    # Exclude combined restricted-cash tags from usable runway cash.
    "cash": {"us-gaap": ["CashAndCashEquivalentsAtCarryingValue"], "ifrs-full": ["CashAndCashEquivalents"]},
    "assets": {"us-gaap": ["Assets"], "ifrs-full": ["Assets"]},
    "diluted_shares": {"us-gaap": ["WeightedAverageNumberOfDilutedSharesOutstanding"], "ifrs-full": ["WeightedAverageNumberOfDilutedSharesOutstanding"]},
    "stock_compensation": {"us-gaap": ["ShareBasedCompensation", "ShareBasedCompensationExpense"], "ifrs-full": ["ShareBasedPaymentArrangementExpense"]},
}
FLOW_FIELDS = {"revenue", "gross_profit", "operating_income", "net_income", "operating_cashflow", "capex", "diluted_shares", "stock_compensation"}
DEBT_CURRENT_TAGS = ["LongTermDebtCurrent", "LongTermDebtAndFinanceLeaseObligationsCurrent", "ShortTermBorrowings", "ShortTermBorrowingsAndCurrentPortionOfLongTermDebt", "BorrowingsCurrent"]
DEBT_NONCURRENT_TAGS = ["LongTermDebtNoncurrent", "LongTermDebtAndFinanceLeaseObligationsNoncurrent", "BorrowingsNoncurrent"]
DEBT_FALLBACK_TAGS = ["LongTermDebt", "LongTermDebtAndFinanceLeaseObligations"]
DEBT_TAGS = set(DEBT_CURRENT_TAGS + DEBT_NONCURRENT_TAGS + DEBT_FALLBACK_TAGS + ["BorrowingsCurrent", "BorrowingsNoncurrent"])
CORE_NULL_CANDIDATES = {"US.AMKR", "US.FORM", "US.MDB", "US.MOD", "US.ONTO", "US.SNOW", "US.VICR", "US.VST"}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def acquisition_maps() -> tuple[dict[str, set[int]], dict[int, dict[str, str]]]:
    """Return exact registered-ticker -> CIK candidates and real SEC receipt times."""
    tickers: dict[str, set[int]] = {}
    receipts: dict[int, dict[str, str]] = {}

    def add(cik: Any, names: Any, received_at: Any = None, receipt_is_real: bool = True, cache_kind: str | None = None) -> None:
        try:
            n = int(cik)
        except (TypeError, ValueError):
            return
        if isinstance(names, str):
            names = [names]
        for ticker in names or []:
            if isinstance(ticker, str) and ticker.strip():
                tickers.setdefault(ticker.strip().upper(), set()).add(n)
        if receipt_is_real and isinstance(received_at, str) and received_at:
            bucket = receipts.setdefault(n, {})
            prior = bucket.get(cache_kind or "unknown")
            if prior is None or received_at < prior:
                bucket[cache_kind or "unknown"] = received_at

    p = CACHE / "fundamentals_acquisition.json"
    if p.exists():
        for x in read_json(p).get("results", []):
            add(x.get("cik"), x.get("tickers", []), x.get("received_at"), x.get("status") == "downloaded", "base")
    p = CACHE / "submissions_acquisition.json"
    if p.exists():
        for x in read_json(p).get("results", []):
            # This ledger's receipt is for submissions JSON, not Company Facts.
            add(x.get("cik"), x.get("registered_tickers", []), None, False)
    p = CACHE / "coverage_v4/fundamentals_acquisition.json"
    if p.exists():
        for x in read_json(p).get("results", {}).values():
            # A copied/verified time is not an acquisition time. Use only an
            # explicit receipt attached to an SEC tier, otherwise leave unknown.
            real = x.get("tier") == "SEC" and not x.get("original_received_at_unknown", False)
            add(x.get("cik"), x.get("registered_tickers", []), x.get("received_at"), real, "coverage_v4")
    return tickers, receipts


def select_payload(cik: int) -> tuple[Path | None, str | None, str | None, str | None]:
    choices = [(CACHE / "companyfacts" / f"CIK{cik:010d}.json", "base"), (CACHE / "coverage_v4/companyfacts" / f"CIK{cik:010d}.json", "coverage_v4")]
    for p, cache_kind in choices:
        if p.exists():
            return p, sha256(p), f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json", cache_kind
    return None, None, f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json", None


def valid_fact_rows(payload: dict[str, Any], cik: int, cache_path: Path, cache_sha: str, received_at: str | None) -> list[dict[str, Any]]:
    facts = payload.get("facts", {})
    source_url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
    out: list[dict[str, Any]] = []
    for taxonomy in ("us-gaap", "ifrs-full"):
        namespace = facts.get(taxonomy, {})
        groups = dict(TAGS)
        groups["__debt"] = {taxonomy: sorted(DEBT_TAGS)}
        for field, per_tax in groups.items():
            for priority, tag in enumerate(per_tax.get(taxonomy, [])):
                concept = namespace.get(tag, {})
                for unit, unit_rows in concept.get("units", {}).items():
                    for raw in unit_rows:
                        filed, end, form, accession = raw.get("filed"), raw.get("end"), raw.get("form"), raw.get("accn")
                        value = raw.get("val")
                        if not isinstance(filed, str) or filed >= FILED_EXCLUSIVE or not isinstance(end, str) or end > AS_OF.isoformat():
                            continue
                        if form not in FORMS or not accession or not isinstance(value, (int, float)) or not math.isfinite(value):
                            continue
                        start = raw.get("start")
                        if field in FLOW_FIELDS:
                            if not isinstance(start, str):
                                continue
                            try:
                                days = (date.fromisoformat(end) - date.fromisoformat(start)).days + 1
                            except ValueError:
                                continue
                            if not 1 <= days <= 400:
                                continue
                        else:
                            days = None
                        accn_compact = accession.replace("-", "")
                        filing_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accn_compact}/"
                        out.append({
                            "field": field, "taxonomy": taxonomy, "tag": tag, "tag_priority": priority,
                            "unit": unit, "value": value, "start": start, "end": end, "filed": filed,
                            "form": form, "accession": accession, "duration_days": days,
                            "source_url": source_url, "filing_url": filing_url,
                            "cache_path": str(cache_path), "cache_sha256": cache_sha, "received_at": received_at,
                        })
    return out


def dedupe_facts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep latest revision of each exact period/tag/unit; never mix tags silently."""
    best: dict[tuple[Any, ...], dict[str, Any]] = {}
    for r in rows:
        key = (r["field"], r["taxonomy"], r["tag"], r["unit"], r.get("start"), r["end"])
        rank = (r["filed"], -r["tag_priority"], r["accession"])
        if key not in best or rank > (best[key]["filed"], -best[key]["tag_priority"], best[key]["accession"]):
            best[key] = r
    return list(best.values())


def fact_rank(r: dict[str, Any]) -> tuple[Any, ...]:
    return (r["end"], r["filed"], -r["tag_priority"], r["accession"])


def annual_fact(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [r for r in rows if r.get("start") and r.get("duration_days") is not None and 330 <= r["duration_days"] <= 400]
    return max(candidates, key=fact_rank) if candidates else None


def ytd_ttm(rows: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Find current/prior matching YTD and prior annual using same tag and unit."""
    # A first fiscal quarter can be the issuer's YTD figure as well as a
    # standalone quarter. It is a valid YTD only when it begins immediately
    # after the matching annual period and has a comparable prior-year fact.
    interim = [r for r in rows if r.get("start") and r.get("duration_days") is not None and 60 <= r["duration_days"] < 320]
    if not interim:
        return None, None
    latest_end = max(r["end"] for r in interim)
    latest = [r for r in interim if r["end"] == latest_end]
    matched = []
    for current in latest:
        key = (current["taxonomy"], current["tag"], current["unit"])
        same = [r for r in rows if (r["taxonomy"], r["tag"], r["unit"]) == key]
        annuals = [r for r in same if r.get("duration_days") is not None and 330 <= r["duration_days"] <= 400
                   and 0 <= (date.fromisoformat(current["start"]) - date.fromisoformat(r["end"])).days <= 7]
        prior_ytds = [r for r in same if r.get("duration_days") is not None and 60 <= r["duration_days"] < 320
                      and 350 <= (date.fromisoformat(current["end"]) - date.fromisoformat(r["end"])).days <= 380
                      and abs(r["duration_days"] - current["duration_days"]) <= 7]
        if annuals and prior_ytds:
            matched.append((current, max(annuals, key=fact_rank), max(prior_ytds, key=fact_rank)))
    if not matched:
        return max(latest, key=fact_rank), None
    current, ann, prior = max(matched, key=lambda triple: fact_rank(triple[0]))
    # The full annual period must end before current YTD begins, and the YTD
    # periods must have same concept, currency/share unit, and comparable days.
    ttm_start = (date.fromisoformat(prior["end"]) + timedelta(days=1)).isoformat()
    ttm_days = (date.fromisoformat(current["end"]) - date.fromisoformat(ttm_start)).days + 1
    if current["field"] == "diluted_shares":
        weighted_days = ann["duration_days"] + current["duration_days"] - prior["duration_days"]
        if weighted_days <= 0 or abs(weighted_days - ttm_days) > 7:
            return current, None
        value = (ann["value"] * ann["duration_days"] + current["value"] * current["duration_days"] - prior["value"] * prior["duration_days"]) / weighted_days
        formula_name = "weighted_average_shares: (FY_avg*FY_days + current_YTD_avg*current_days - prior_YTD_avg*prior_days) / net_days"
    else:
        value = ann["value"] + current["value"] - prior["value"]
        weighted_days = ttm_days
        formula_name = "prior_fiscal_year + current_YTD - prior_comparable_YTD"
    formula = {"value": value, "unit": current["unit"], "start": ttm_start, "end": current["end"],
               "filed": max(ann["filed"], current["filed"], prior["filed"]),
               "form": sorted({ann["form"], current["form"], prior["form"]}),
               "accession": sorted({ann["accession"], current["accession"], prior["accession"]}),
               "tag": current["tag"], "taxonomy": current["taxonomy"], "period": "ttm",
               "formula": formula_name, "duration_days": weighted_days,
               "source_url": current["source_url"], "cache_path": current["cache_path"],
               "cache_sha256": current["cache_sha256"], "received_at": current["received_at"],
               "components": [ann, current, prior]}
    return current, formula


def direct_metric(r: dict[str, Any] | None, period: str) -> dict[str, Any] | None:
    if r is None:
        return None
    return {**r, "period": period}


def flow_metric(rows: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None]:
    annual = annual_fact(rows)
    current_ytd, ttm = ytd_ttm(rows)
    # If the latest reported full fiscal year ends after an interim YTD period,
    # that annual is the latest available four-quarter total.
    if annual is not None and current_ytd is not None and annual["end"] >= current_ytd["end"]:
        ttm = {**annual, "period": "ttm", "formula": "latest_reported_full_fiscal_year", "components": [annual]}
    latest_ytd = current_ytd
    latest_annual = annual
    latest = max([x for x in (latest_ytd, latest_annual) if x is not None], key=fact_rank) if latest_ytd or latest_annual else None
    if latest_ytd is None and annual is not None:
        # A full fiscal-year fact is itself a valid four-quarter total, but this
        # fallback is allowed only when no later interim YTD report exists.
        ttm = {**annual, "period": "ttm", "formula": "latest_reported_full_fiscal_year", "components": [annual]}
    return direct_metric(annual, "annual"), ttm, direct_metric(latest, "latest"), latest_ytd


def instant_fact(rows: list[dict[str, Any]], end: str | None = None) -> dict[str, Any] | None:
    candidates = [r for r in rows if r.get("start") is None and (end is None or r["end"] == end)]
    return max(candidates, key=fact_rank) if candidates else None


def debt_value(facts: list[dict[str, Any]], end: str | None = None) -> dict[str, Any] | None:
    usd_or_currency = [r for r in facts if r["field"] == "__debt" and (end is None or r["end"] == end)]
    current = [r for r in usd_or_currency if r["tag"] in DEBT_CURRENT_TAGS]
    noncurrent = [r for r in usd_or_currency if r["tag"] in DEBT_NONCURRENT_TAGS]
    # Only pair explicit total-current-debt concepts with noncurrent debt.
    # LongTermDebtCurrent alone omits short-term borrowings and is not total debt.
    current_total_tags = {"ShortTermBorrowingsAndCurrentPortionOfLongTermDebt", "BorrowingsCurrent"}
    pairs = []
    for c in current:
        for n in noncurrent:
            if c["tag"] in current_total_tags and c["end"] == n["end"] and c["unit"] == n["unit"] and c["taxonomy"] == n["taxonomy"] and c["accession"] == n["accession"]:
                pairs.append((c, n))
    if pairs:
        c, n = min(pairs, key=lambda pair: (-int(max(pair[0]["filed"], pair[1]["filed"]).replace("-", "")), -int(pair[0]["end"].replace("-", "")), pair[0]["tag_priority"] + pair[1]["tag_priority"]))
        return {"value": c["value"] + n["value"], "unit": c["unit"], "start": None, "end": c["end"],
                "filed": max(c["filed"], n["filed"]), "form": sorted({c["form"], n["form"]}),
                "accession": sorted({c["accession"], n["accession"]}), "tag": [c["tag"], n["tag"]],
                "taxonomy": c["taxonomy"], "period": "latest", "formula": "current_debt_component + noncurrent_debt_component",
                "components": [c, n], "coverage": "total_current_and_noncurrent"}
    partial_pairs = [(c, n) for c in current for n in noncurrent if c["tag"] == "LongTermDebtCurrent" and c["end"] == n["end"] and c["unit"] == n["unit"] and c["taxonomy"] == n["taxonomy"] and c["accession"] == n["accession"]]
    if partial_pairs:
        c, n = min(partial_pairs, key=lambda pair: (-int(max(pair[0]["filed"], pair[1]["filed"]).replace("-", "")), -int(pair[0]["end"].replace("-", "")), pair[0]["tag_priority"] + pair[1]["tag_priority"]))
        return {"value": c["value"] + n["value"], "unit": c["unit"], "start": None, "end": c["end"],
                "filed": max(c["filed"], n["filed"]), "form": sorted({c["form"], n["form"]}),
                "accession": sorted({c["accession"], n["accession"]}), "tag": [c["tag"], n["tag"]],
                "taxonomy": c["taxonomy"], "period": "latest", "formula": "partial_long_term_debt_current_plus_noncurrent",
                "components": [c, n], "coverage": "partial_long_term_components", "caveat": "仅长期债务流动部分与非流动部分；短期借款可能未包含，不称总债务。"}
    fallback = [r for r in usd_or_currency if r["tag"] in DEBT_FALLBACK_TAGS]
    r = max(fallback, key=fact_rank) if fallback else None
    if r:
        return {**r, "period": "latest", "components": [r], "coverage": "long_term_only", "caveat": "单一长期债务标签可未涵盖全部流动债务；按原标签保留，不估算缺项。"}
    return None


def find_instant_fields(facts: list[dict[str, Any]], taxonomy: str = "us-gaap") -> dict[str, list[dict[str, Any]]]:
    result = {k: [] for k in ("cash", "assets", "__debt")}
    debt_tags = DEBT_TAGS
    for r in facts:
        if r.get("start") is not None:
            continue
        if r["field"] in result:
            result[r["field"]].append(r)
        if r["tag"] in debt_tags:
            x = dict(r); x["field"] = "__debt"; result["__debt"].append(x)
    return result


def ensure_metric_shape(metrics: dict[str, Any], field: str) -> None:
    metrics[field] = {"annual": None, "ttm": None, "latest": None}


def derived_value(value: Any, unit: str | None, formula: str, inputs: list[str], end: str | None) -> dict[str, Any] | None:
    if value is None or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return {"value": value, "unit": unit, "formula": formula, "inputs": inputs, "period_end": end}


def prior_year_ttm(rows: list[dict[str, Any]], current_ttm: dict[str, Any]) -> tuple[float, list[dict[str, Any]]] | None:
    """Rebuild the TTM ending at the current TTM's prior-year YTD end."""
    comps = current_ttm.get("components", [])
    if current_ttm.get("formula") != "prior_fiscal_year + current_YTD - prior_comparable_YTD" or len(comps) != 3:
        return None
    prior_ytd = comps[2]
    same = [r for r in rows if (r["taxonomy"], r["tag"], r["unit"]) == (prior_ytd["taxonomy"], prior_ytd["tag"], prior_ytd["unit"])]
    annuals = [r for r in same if r.get("duration_days") is not None and 330 <= r["duration_days"] <= 400
               and 0 <= (date.fromisoformat(prior_ytd["start"]) - date.fromisoformat(r["end"])).days <= 7]
    earlier = [r for r in same if r.get("duration_days") is not None and 60 <= r["duration_days"] < 320
               and 350 <= (date.fromisoformat(prior_ytd["end"]) - date.fromisoformat(r["end"])).days <= 380
               and abs(r["duration_days"] - prior_ytd["duration_days"]) <= 7]
    if not annuals or not earlier:
        return None
    annual = max(annuals, key=fact_rank)
    prior = max(earlier, key=fact_rank)
    return annual["value"] + prior_ytd["value"] - prior["value"], [annual, prior_ytd, prior]


def process_stock(row: dict[str, Any], ticker_candidates: dict[str, set[int]], receipts: dict[int, dict[str, str]]) -> dict[str, Any]:
    code = row.get("code", row.get("symbol", ""))
    symbol = code.split(".", 1)[-1].upper() if "." in code else code.upper()
    candidates = ticker_candidates.get(symbol, set())
    financials: dict[str, Any] = {
        "metrics": {}, "derived": {k: None for k in ("revenue_yoy", "net_margin", "fcf", "fcf_margin", "cash_runway_months")},
        "latest_period_end": None, "latest_filed": None, "source_capture_at": None,
        "status": "missing", "caveats": [], "sources": [],
    }
    financials["input_missing_core_candidate"] = row.get("input_missing_core_candidate", False)
    fields = ["revenue", "gross_profit", "operating_income", "net_income", "operating_cashflow", "capex", "cash", "total_debt", "assets", "diluted_shares", "stock_compensation"]
    for field in fields:
        ensure_metric_shape(financials["metrics"], field)
    if len(candidates) != 1:
        financials["caveats"].append("未找到唯一的SEC已登记ticker→CIK映射" if not candidates else f"ticker映射到多个CIK，保留缺失：{sorted(candidates)}")
        return {"code": code, "symbol": symbol, "name": row.get("name"), "input_missing_core_candidate": row.get("input_missing_core_candidate", False), "financials": financials}
    cik = next(iter(candidates))
    cache_path, cache_digest, source_url, cache_kind = select_payload(cik)
    if cache_path is None:
        financials["caveats"].append(f"SEC Company Facts本地缓存缺失；未发起网络下载。候选CIK={cik}")
        return {"code": code, "symbol": symbol, "cik": cik, "name": row.get("name"), "input_missing_core_candidate": row.get("input_missing_core_candidate", False), "financials": financials}
    received_at = receipts.get(cik, {}).get(cache_kind or "")
    payload = read_json(cache_path)
    if int(payload.get("cik", -1)) != cik or not payload.get("facts"):
        financials["caveats"].append("缓存CIK身份或facts校验失败，所有金额保持missing。")
        return {"code": code, "symbol": symbol, "cik": cik, "name": row.get("name"), "input_missing_core_candidate": row.get("input_missing_core_candidate", False), "financials": financials}
    entity = payload.get("entityName")
    if entity and row.get("name") and entity.casefold() not in row["name"].casefold() and row["name"].casefold() not in entity.casefold():
        financials["caveats"].append(f"输入发行人名与SEC实体名需留意别名/变更：input={row['name']}; SEC={entity}")
    if not received_at:
        financials["caveats"].append("没有找到可信的原始下载received_at；不以文件修改时间或复制时间代替。")
    financials["source_capture_at"] = received_at
    financials["sources"].append({"type": "SEC Company Facts", "url": source_url, "entity_name": entity,
                                  "cik": cik, "cache_path": str(cache_path), "cache_sha256": cache_digest,
                                  "received_at": received_at})
    raw_facts = dedupe_facts(valid_fact_rows(payload, cik, cache_path, cache_digest, received_at))
    metric_rows: dict[str, list[dict[str, Any]]] = {f: [r for r in raw_facts if r["field"] == f] for f in FLOW_FIELDS}
    flow_details: dict[str, tuple[Any, Any, Any, Any]] = {}
    for f in FLOW_FIELDS:
        flow_details[f] = flow_metric(metric_rows[f])
        annual, ttm, latest, latest_ytd = flow_details[f]
        financials["metrics"][f] = {"annual": annual, "ttm": ttm, "latest": latest}

    # Keep flow facts period-aligned to revenue. This prevents old or alternate
    # contexts (for example a legacy stock-compensation tag) from appearing as
    # current-year financials when the issuer's latest filing used another tag.
    revenue_periods = {key: (financials["metrics"]["revenue"][key] or {}).get("end") for key in ("annual", "ttm", "latest")}
    for f in FLOW_FIELDS - {"revenue"}:
        for period in ("annual", "ttm", "latest"):
            value = financials["metrics"][f][period]
            if value and value.get("end") != revenue_periods[period]:
                financials["metrics"][f][period] = None
    if any(financials["metrics"]["diluted_shares"].values()):
        financials["caveats"].append("加权平均稀释股数按申报标签保留；公司可能已在原始申报中追溯调整拆股口径，本提取器不另行拆股调整。")

    # Annual instant balances are aligned to the latest available annual revenue
    # end; latest balances remain the latest individual reported instant.
    annual_rev = financials["metrics"]["revenue"]["annual"]
    annual_end = annual_rev.get("end") if annual_rev else None
    instant = find_instant_fields(raw_facts)
    for f in ("cash", "assets"):
        annual_fact_value = instant_fact(instant[f], annual_end) if annual_end else None
        latest_fact_value = instant_fact(instant[f])
        financials["metrics"][f] = {"annual": direct_metric(annual_fact_value, "annual"), "ttm": None,
                                    "latest": direct_metric(latest_fact_value, "latest")}
    debt_annual = debt_value(instant["__debt"], annual_end) if annual_end else None
    debt_latest = debt_value(instant["__debt"])
    financials["metrics"]["total_debt"] = {"annual": debt_annual, "ttm": None, "latest": debt_latest}

    # Attach selected values' core provenance in one source list and report any
    # non-USD reporting currencies without converting or blending them.
    selected_units: set[str] = set()
    selected_filed: list[str] = []
    selected_ends: list[str] = []
    for entry in financials["metrics"].values():
        for value in entry.values():
            if value:
                if value.get("unit"):
                    selected_units.add(value["unit"])
                if value.get("filed"):
                    selected_filed.append(value["filed"])
                if value.get("end"):
                    selected_ends.append(value["end"])
                # TTM is a derived amount; verify every component against the
                # exact loaded raw Company Facts payload before writing it.
                components = value.get("components") or ([value] if value.get("tag") else [])
                for component in components:
                    unit_rows = payload.get("facts", {}).get(component["taxonomy"], {}).get(component["tag"], {}).get("units", {}).get(component["unit"], [])
                    if not any(x.get("val") == component["value"] and x.get("start") == component.get("start") and x.get("end") == component["end"] and x.get("filed") == component["filed"] and x.get("accn") == component["accession"] for x in unit_rows):
                        raise AssertionError(f"Raw fact mismatch {code} {component['tag']} {component['accession']}")
    if selected_units - {"USD", "shares"}:
        financials["caveats"].append("发现非USD财务币种，金额按原单位保存且未换算/混合：" + ", ".join(sorted(selected_units - {"shares"})))
    financials["latest_period_end"] = max(selected_ends) if selected_ends else None
    financials["latest_filed"] = max(selected_filed) if selected_filed else None

    # TTM growth compares to TTM ending on the comparable day last year, never
    # to the prior complete FY when those periods differ.
    rev_ttm = financials["metrics"]["revenue"]["ttm"]
    rev_annual = financials["metrics"]["revenue"]["annual"]
    rev_rows = metric_rows["revenue"]
    if rev_ttm and rev_annual and rev_ttm.get("value") is not None:
        if rev_ttm.get("formula") == "prior_fiscal_year + current_YTD - prior_comparable_YTD":
            prior_ttm = prior_year_ttm(rev_rows, rev_ttm)
            if prior_ttm and prior_ttm[0] not in (None, 0):
                y = rev_ttm["value"] / prior_ttm[0] - 1
                financials["derived"]["revenue_yoy"] = derived_value(y, "ratio", "latest_TTM / TTM_ending_on_comparable_day_last_year - 1", ["revenue.ttm", "prior_year_comparable_ttm"], rev_ttm.get("end"))
                financials["derived"]["revenue_yoy"]["evidence"] = prior_ttm[1]
        elif rev_ttm.get("formula") == "latest_reported_full_fiscal_year":
            same_tag = [r for r in rev_rows if r.get("taxonomy") == rev_annual.get("taxonomy") and r.get("tag") == rev_annual.get("tag") and r.get("unit") == rev_annual.get("unit") and r.get("duration_days") and 330 <= r["duration_days"] <= 400 and r["end"] < rev_annual["end"] and 350 <= (date.fromisoformat(rev_annual["end"])-date.fromisoformat(r["end"])).days <= 380]
            if same_tag:
                prior = max(same_tag, key=fact_rank)
                if prior["value"]:
                    financials["derived"]["revenue_yoy"] = derived_value(rev_annual["value"] / prior["value"] - 1, "ratio", "latest_annual_revenue / prior_annual_revenue - 1", ["revenue.annual", "prior_annual_revenue"], rev_annual["end"])
                    financials["derived"]["revenue_yoy"]["evidence"] = [rev_annual, prior]
    net_ttm = financials["metrics"]["net_income"]["ttm"]
    if rev_ttm and net_ttm and rev_ttm.get("end") == net_ttm.get("end") and rev_ttm.get("unit") == net_ttm.get("unit") and rev_ttm.get("value"):
        financials["derived"]["net_margin"] = derived_value(net_ttm["value"] / rev_ttm["value"], "ratio", "net_income_TTM / revenue_TTM", ["net_income.ttm", "revenue.ttm"], rev_ttm["end"])
    ocf_ttm = financials["metrics"]["operating_cashflow"]["ttm"]
    capex_ttm = financials["metrics"]["capex"]["ttm"]
    if ocf_ttm and capex_ttm and ocf_ttm.get("end") == capex_ttm.get("end") and ocf_ttm.get("unit") == capex_ttm.get("unit"):
        fcf = ocf_ttm["value"] - abs(capex_ttm["value"])
        financials["derived"]["fcf"] = derived_value(fcf, ocf_ttm["unit"], "operating_cashflow_TTM - absolute(capex_TTM)", ["operating_cashflow.ttm", "capex.ttm"], ocf_ttm["end"])
        if rev_ttm and rev_ttm.get("end") == ocf_ttm["end"] and rev_ttm.get("unit") == ocf_ttm.get("unit") and rev_ttm.get("value"):
            financials["derived"]["fcf_margin"] = derived_value(fcf / rev_ttm["value"], "ratio", "FCF_TTM / revenue_TTM", ["fcf", "revenue.ttm"], rev_ttm["end"])
    cash_latest = financials["metrics"]["cash"]["latest"]
    if cash_latest and ocf_ttm and ocf_ttm.get("value", 0) < 0 and cash_latest.get("unit") == ocf_ttm.get("unit") and cash_latest.get("end") == ocf_ttm.get("end"):
        burn_per_month = abs(ocf_ttm["value"]) / 12
        if burn_per_month:
            financials["derived"]["cash_runway_months"] = derived_value(cash_latest["value"] / burn_per_month, "months", "latest_cash / (absolute(operating_cashflow_TTM) / 12); only when operating cash flow is negative", ["cash.latest", "operating_cashflow.ttm"], cash_latest["end"])
    if financials["derived"]["cash_runway_months"] is None:
        financials["caveats"].append("现金跑道仅在同币种最新现金与负TTM经营现金流可匹配时计算；否则保持missing。")
    values = [v for group in financials["metrics"].values() for v in group.values() if v is not None]
    financials["status"] = "available" if len(values) >= 6 else "partial" if values else "missing"
    if not values:
        financials["caveats"].append("缓存存在但未找到所支持的、截止日期内可用标准财务事实。")
    return {"code": code, "symbol": symbol, "cik": cik, "name": entity if row.get("input_missing_core_candidate") else row.get("name"), "sec_entity_name": entity, "business_tier": row.get("business", {}).get("tier"), "input_missing_core_candidate": row.get("input_missing_core_candidate", False), "financials": financials}


def validate_snapshot(records: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Independent pass: raw SEC line-item identity, formulas, and six anchors."""
    raw_fact_checks = 0
    ttm_formula_checks = 0
    derived_checks = 0
    source_payloads: dict[str, dict[str, Any]] = {}
    for code, financials in records.items():
        source = next((s for s in financials.get("sources", []) if s.get("type") == "SEC Company Facts"), None)
        if not source:
            continue
        path = Path(source["cache_path"])
        if path not in source_payloads:
            source_payloads[str(path)] = read_json(path)
        payload = source_payloads[str(path)]
        assert int(payload.get("cik", -1)) == source["cik"]
        assert sha256(path) == source["cache_sha256"]
        for metric in financials["metrics"].values():
            for selected in metric.values():
                if not selected:
                    continue
                components = selected.get("components") or ([selected] if isinstance(selected.get("tag"), str) else [])
                for fact in components:
                    tx = fact.get("taxonomy")
                    tag = fact.get("tag")
                    unit = fact.get("unit")
                    assert tx and tag and unit
                    found = payload.get("facts", {}).get(tx, {}).get(tag, {}).get("units", {}).get(unit, [])
                    assert any(x.get("val") == fact.get("value") and x.get("start") == fact.get("start") and x.get("end") == fact.get("end") and x.get("filed") == fact.get("filed") and x.get("form") == fact.get("form") and x.get("accn") == fact.get("accession") for x in found), (code, tag, fact.get("accession"), "not present in raw SEC facts")
                    assert fact["filed"] < FILED_EXCLUSIVE and fact["end"] <= AS_OF.isoformat(), (code, fact["filed"], fact["end"])
                    raw_fact_checks += 1
                formula = selected.get("formula")
                if formula == "prior_fiscal_year + current_YTD - prior_comparable_YTD":
                    assert len(selected["components"]) == 3
                    a, b, c = selected["components"]
                    assert (a["taxonomy"], a["tag"], a["unit"]) == (b["taxonomy"], b["tag"], b["unit"]) == (c["taxonomy"], c["tag"], c["unit"])
                    assert abs(selected["value"] - (a["value"] + b["value"] - c["value"])) < 1e-6
                    assert selected["start"] == (date.fromisoformat(c["end"])+timedelta(days=1)).isoformat()
                    assert abs((date.fromisoformat(b["end"])-date.fromisoformat(b["start"])).days - (date.fromisoformat(c["end"])-date.fromisoformat(c["start"])).days) <= 7
                    ttm_formula_checks += 1
                elif isinstance(formula, str) and formula.startswith("weighted_average_shares:"):
                    assert len(selected["components"]) == 3
                    a, b, c = selected["components"]
                    denom = a["duration_days"] + b["duration_days"] - c["duration_days"]
                    expected = (a["value"]*a["duration_days"] + b["value"]*b["duration_days"] - c["value"]*c["duration_days"])/denom
                    assert abs(selected["value"]-expected) < 1e-6
                    assert selected["start"] == (date.fromisoformat(c["end"])+timedelta(days=1)).isoformat()
                    ttm_formula_checks += 1
                elif formula == "latest_reported_full_fiscal_year":
                    assert len(selected["components"]) == 1 and selected["value"] == selected["components"][0]["value"]
                    ttm_formula_checks += 1
                elif formula == "current_debt_component + noncurrent_debt_component":
                    assert len(selected["components"]) == 2 and abs(selected["value"]-sum(x["value"] for x in selected["components"])) < 1e-6
                    ttm_formula_checks += 1
        revenue = financials["metrics"]["revenue"]["ttm"]
        net_income = financials["metrics"]["net_income"]["ttm"]
        margin = financials["derived"].get("net_margin")
        if margin:
            assert revenue and net_income and abs(margin["value"]-net_income["value"]/revenue["value"]) < 1e-12
            derived_checks += 1
        opcf = financials["metrics"]["operating_cashflow"]["ttm"]
        capex = financials["metrics"]["capex"]["ttm"]
        fcf = financials["derived"].get("fcf")
        if fcf:
            assert opcf and capex and abs(fcf["value"]-(opcf["value"]-abs(capex["value"]))) < 1e-6
            derived_checks += 1
        yoy = financials["derived"].get("revenue_yoy")
        if yoy:
            evidence = yoy.get("evidence", [])
            if yoy["formula"] == "latest_TTM / TTM_ending_on_comparable_day_last_year - 1":
                assert len(evidence) == 3
                prior_value = evidence[0]["value"] + evidence[1]["value"] - evidence[2]["value"]
                assert abs(yoy["value"] - (financials["metrics"]["revenue"]["ttm"]["value"] / prior_value - 1)) < 1e-12
            elif yoy["formula"] == "latest_annual_revenue / prior_annual_revenue - 1":
                assert len(evidence) == 2
                assert abs(yoy["value"] - (evidence[0]["value"] / evidence[1]["value"] - 1)) < 1e-12
            derived_checks += 1
            for fact in evidence:
                fact_components = fact.get("components") or ([fact] if isinstance(fact.get("tag"), str) else [])
                for raw_fact in fact_components:
                    source_facts = payload.get("facts", {}).get(raw_fact["taxonomy"], {}).get(raw_fact["tag"], {}).get("units", {}).get(raw_fact["unit"], [])
                    assert any(x.get("val") == raw_fact["value"] and x.get("end") == raw_fact.get("end") and x.get("filed") == raw_fact.get("filed") and x.get("accn") == raw_fact.get("accession") for x in source_facts)
        if financials["metrics"]["capex"]["ttm"] is None:
            assert financials["derived"].get("fcf") is None
    anchors = {}
    missing_anchors = []
    for code in ("US.AAPL", "US.NVDA", "US.MU", "US.CRDO", "US.VRT", "US.PLTR"):
        f = records.get(code)
        if not f:
            missing_anchors.append(code)
            continue
        rev = f["metrics"]["revenue"]
        assert rev["annual"] and rev["ttm"], f"{code} lacks FY or reproducible TTM revenue"
        assert f.get("source_capture_at"), f"{code} lacks original cache receipt time"
        anchors[code] = {"cik": next((s["cik"] for s in f["sources"] if s.get("type") == "SEC Company Facts"), None),
                         "annual_end": rev["annual"]["end"], "ttm_end": rev["ttm"]["end"],
                         "annual_value": rev["annual"]["value"], "ttm_value": rev["ttm"]["value"],
                         "ttm_formula": rev["ttm"].get("formula"), "source_capture_at": f["source_capture_at"]}
    return {"checks_passed": True, "raw_selected_fact_matches_against_sec_companyfacts": raw_fact_checks,
            "ttm_and_debt_formula_recomputations": ttm_formula_checks, "derived_ratio_and_fcf_recomputations": derived_checks,
            "as_of_filed_and_period_end_cutoff": "passed for every selected fact and component",
            "required_companyfacts_anchors": anchors, "anchor_codes_absent_from_current_input": missing_anchors,
            "core_null_candidates_flagged": sorted(code for code, f in records.items() if f.get("input_missing_core_candidate"))}


def build(as_of: date | None = None) -> dict[str, Any]:
    global AS_OF, FILED_EXCLUSIVE
    AS_OF = as_of or date.today()
    FILED_EXCLUSIVE = (AS_OF + timedelta(days=1)).isoformat()
    data = read_json(INPUT)
    rows = [r for r in data.get("rows", []) if r.get("kind") == "STOCK" and (r.get("business", {}).get("tier") in (1, 2, 3) or r.get("code") in CORE_NULL_CANDIDATES)]
    eligible_pool_count = sum(1 for r in data.get("rows", []) if r.get("kind") == "STOCK" and r.get("business", {}).get("tier") in (1, 2, 3))
    present = {r.get("code") for r in rows}
    present_all = {r.get("code") for r in data.get("rows", [])}
    missing_candidates = sorted(CORE_NULL_CANDIDATES - present_all)
    for code in missing_candidates:
        rows.append({"code": code, "symbol": code.split(".", 1)[1], "name": None, "kind": "STOCK", "business": {"tier": None}, "input_missing_core_candidate": True})
    ticker_candidates, receipts = acquisition_maps()
    records = [process_stock(r, ticker_candidates, receipts) for r in rows]
    assert len({r["code"] for r in records}) == len(records), "Duplicate input code"
    by_code = {r["code"]: r["financials"] for r in records}
    status_counts: dict[str, int] = {}
    for f in by_code.values():
        status_counts[f["status"]] = status_counts.get(f["status"], 0) + 1
    validation = validate_snapshot(by_code)
    result = {"schema_version": "financial_snapshot_v1", "as_of": AS_OF.isoformat(), "source_capture_cutoff": "cache ledgers only; no new download", "input": str(INPUT), "input_sha256": sha256(INPUT), "cache_root": str(CACHE), "eligible_stock_count": eligible_pool_count, "financial_record_count": len(records), "coverage": {"tier_1_2_3_input": eligible_pool_count, "additional_null_core_candidates": sorted(CORE_NULL_CANDIDATES), "additional_candidates_missing_from_input": missing_candidates, "total_financial_records_including_supplements": len(records), "financial_status_counts": status_counts}, "method": {"annual": "latest filed 330-400 day flow fact; point-in-time balance fact aligned to annual revenue end", "ttm": "same taxonomy/tag/unit: prior FY + current YTD - prior comparable YTD; start is the day after prior comparable YTD end; comparable YTD durations differ by at most 7 days", "diluted_shares": "duration-weighted average using annual and current/prior YTD weighted share facts; no simple arithmetic subtraction; stock splits are not independently adjusted", "cutoff": f"filed < {(AS_OF + timedelta(days=1)).isoformat()} and period end <= {AS_OF.isoformat()}", "currency": "no FX conversion or cross-currency mix", "debt": "sum only matched explicit total-current debt with noncurrent debt; otherwise use partial or long-term fallback with coverage caveat", "capex": "FCF = operating cash flow minus absolute reported PP&E capex; missing or period-mismatched capex leaves FCF unknown", "cash_runway": "only computed from unrestricted cash where same-end and same-currency with negative TTM operating cash flow"}, "validation": validation, "records": by_code}
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=OUTPUT)
    ap.add_argument("--as-of", type=date.fromisoformat, default=date.today(), help="Inclusive financial reporting cutoff YYYY-MM-DD")
    args = ap.parse_args()
    result = build(args.as_of)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix(args.output.suffix + ".tmp")
    temp.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(args.output)
    statuses: dict[str, int] = {}
    missing = 0
    for f in result["records"].values():
        statuses[f["status"]] = statuses.get(f["status"], 0) + 1
        missing += f["status"] == "missing"
    print(json.dumps({"output": str(args.output), "as_of": result["as_of"], "eligible_stock_count": result["eligible_stock_count"], "financial_record_count": result["financial_record_count"], "status_counts": statuses, "missing": missing, "input_sha256": result["input_sha256"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
