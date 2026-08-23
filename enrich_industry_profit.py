#!/usr/bin/env python3
"""Enrich the Apr-May weekly-peak stock class with industry and profitability structure.

Sources:
- SEC company_tickers.json for ticker -> CIK mapping.
- SEC submissions JSON for SIC code and SIC description.
- SEC companyfacts JSON for annual financial statement facts.

The output is intentionally designed for a later industry-neutral analysis:
for every stock it stores the median drawdown of its industry, profitability
module and industry×module peer group, plus residual drawdowns versus those
benchmarks.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

SEC_BASE = "https://data.sec.gov"
SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
DEFAULT_USER_AGENT = "crazygo-stock-research/1.0 crazygo@users.noreply.github.com"

FEATURES = (
    "revenue_growth_yoy_pct",
    "operating_margin_pct",
    "net_margin_pct",
    "fcf_margin_pct",
    "operating_margin_yoy_change_pp",
)

ANNUAL_FORMS = {"10-K", "20-F", "40-F"}

CONCEPTS: dict[str, list[tuple[str, str]]] = {
    "revenue": [
        ("us-gaap", "RevenueFromContractWithCustomerExcludingAssessedTax"),
        ("us-gaap", "Revenues"),
        ("us-gaap", "SalesRevenueNet"),
        ("us-gaap", "SalesRevenueGoodsNet"),
        ("ifrs-full", "Revenue"),
    ],
    "gross_profit": [
        ("us-gaap", "GrossProfit"),
        ("ifrs-full", "GrossProfit"),
    ],
    "operating_income": [
        ("us-gaap", "OperatingIncomeLoss"),
        ("ifrs-full", "ProfitLossFromOperatingActivities"),
    ],
    "net_income": [
        ("us-gaap", "NetIncomeLoss"),
        ("us-gaap", "NetIncomeLossAvailableToCommonStockholdersBasic"),
        ("ifrs-full", "ProfitLoss"),
    ],
    "operating_cash_flow": [
        ("us-gaap", "NetCashProvidedByUsedInOperatingActivities"),
        ("us-gaap", "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"),
        ("ifrs-full", "CashFlowsFromUsedInOperatingActivities"),
    ],
    "capex": [
        ("us-gaap", "PaymentsToAcquirePropertyPlantAndEquipment"),
        ("us-gaap", "PaymentsForAdditionsToPropertyPlantAndEquipment"),
        ("ifrs-full", "PurchaseOfPropertyPlantAndEquipmentClassifiedAsInvestingActivities"),
    ],
}


class DataError(RuntimeError):
    pass


class SecClient:
    def __init__(self, cache_dir: Path, request_delay: float = 0.20, user_agent: str | None = None) -> None:
        self.cache_dir = cache_dir
        self.request_delay = max(0.0, request_delay)
        self.user_agent = user_agent or DEFAULT_USER_AGENT
        self._last_request_at = 0.0

    def _wait(self) -> None:
        remaining = self.request_delay - (time.monotonic() - self._last_request_at)
        if remaining > 0:
            time.sleep(remaining)

    def _get_json(self, url: str, cache_path: Path, *, allow_missing: bool = False) -> Mapping[str, Any] | None:
        if cache_path.exists():
            try:
                raw = json.loads(cache_path.read_text(encoding="utf-8"))
                return raw if isinstance(raw, Mapping) else None
            except json.JSONDecodeError:
                cache_path.unlink(missing_ok=True)

        for attempt in range(5):
            self._wait()
            request = Request(
                url,
                headers={
                    "User-Agent": self.user_agent,
                    "Accept-Encoding": "identity",
                    "Accept": "application/json",
                },
            )
            try:
                with urlopen(request, timeout=45) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                self._last_request_at = time.monotonic()
                if not isinstance(payload, Mapping):
                    raise DataError(f"SEC returned non-object JSON: {url}")
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
                return payload
            except HTTPError as exc:
                self._last_request_at = time.monotonic()
                if allow_missing and exc.code == 404:
                    return None
                if exc.code not in {429, 500, 502, 503, 504} or attempt == 4:
                    raise DataError(f"SEC HTTP {exc.code}: {url}") from exc
                time.sleep(min(2 ** attempt, 12))
            except (URLError, TimeoutError, json.JSONDecodeError) as exc:
                self._last_request_at = time.monotonic()
                if attempt == 4:
                    raise DataError(f"SEC request failed: {url}: {exc}") from exc
                time.sleep(min(2 ** attempt, 12))
        raise DataError(f"SEC request failed: {url}")

    def ticker_map(self) -> dict[str, dict[str, Any]]:
        payload = self._get_json(SEC_TICKERS_URL, self.cache_dir / "company_tickers.json") or {}
        result: dict[str, dict[str, Any]] = {}
        for item in payload.values():
            if not isinstance(item, Mapping):
                continue
            ticker = str(item.get("ticker") or "").upper()
            cik = item.get("cik_str")
            if ticker and cik is not None:
                normalized = normalize_ticker(ticker)
                result[normalized] = {
                    "ticker": ticker,
                    "cik": int(cik),
                    "title": item.get("title"),
                }
        return result

    def submissions(self, cik: int) -> Mapping[str, Any] | None:
        padded = f"{cik:010d}"
        return self._get_json(
            f"{SEC_BASE}/submissions/CIK{padded}.json",
            self.cache_dir / "submissions" / f"CIK{padded}.json",
            allow_missing=True,
        )

    def companyfacts(self, cik: int) -> Mapping[str, Any] | None:
        padded = f"{cik:010d}"
        return self._get_json(
            f"{SEC_BASE}/api/xbrl/companyfacts/CIK{padded}.json",
            self.cache_dir / "companyfacts" / f"CIK{padded}.json",
            allow_missing=True,
        )


def normalize_ticker(ticker: str) -> str:
    return ticker.upper().replace(".", "-").replace("/", "-")


def _pct(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0:
        return None
    return numerator / denominator * 100.0


def _percentile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile requires values")
    pos = (len(ordered) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    w = pos - lo
    return ordered[lo] * (1.0 - w) + ordered[hi] * w


def _duration_days(item: Mapping[str, Any]) -> int | None:
    try:
        from datetime import date
        return (date.fromisoformat(str(item["end"])) - date.fromisoformat(str(item["start"]))).days
    except (KeyError, TypeError, ValueError):
        return None


def _annual_series(companyfacts: Mapping[str, Any], candidates: Sequence[tuple[str, str]]) -> dict[str, Any]:
    facts = companyfacts.get("facts")
    if not isinstance(facts, Mapping):
        return {"values": {}, "namespace": None, "tag": None, "unit": None}

    for namespace, tag in candidates:
        namespace_obj = facts.get(namespace)
        if not isinstance(namespace_obj, Mapping):
            continue
        concept = namespace_obj.get(tag)
        if not isinstance(concept, Mapping):
            continue
        units = concept.get("units")
        if not isinstance(units, Mapping):
            continue

        best_unit: str | None = None
        best_values: dict[str, dict[str, Any]] = {}
        for unit, rows in units.items():
            if not isinstance(rows, list) or "/" in str(unit).lower() or str(unit).lower() in {"shares", "pure"}:
                continue
            by_end: dict[str, dict[str, Any]] = {}
            for row in rows:
                if not isinstance(row, Mapping):
                    continue
                if row.get("form") not in ANNUAL_FORMS:
                    continue
                duration = _duration_days(row)
                if duration is None or duration < 250 or duration > 430:
                    continue
                end = row.get("end")
                val = row.get("val")
                if not isinstance(end, str) or not isinstance(val, (int, float)) or not math.isfinite(float(val)):
                    continue
                existing = by_end.get(end)
                filed = str(row.get("filed") or "")
                if existing is None or filed >= str(existing.get("filed") or ""):
                    by_end[end] = {
                        "value": float(val),
                        "filed": filed or None,
                        "form": row.get("form"),
                        "fy": row.get("fy"),
                        "fp": row.get("fp"),
                    }
            if len(by_end) > len(best_values):
                best_values = by_end
                best_unit = str(unit)
        if best_values:
            return {"values": best_values, "namespace": namespace, "tag": tag, "unit": best_unit}

    return {"values": {}, "namespace": None, "tag": None, "unit": None}


def _value_at(series: Mapping[str, Any], end: str) -> float | None:
    values = series.get("values")
    if not isinstance(values, Mapping):
        return None
    row = values.get(end)
    if not isinstance(row, Mapping):
        return None
    value = row.get("value")
    return float(value) if isinstance(value, (int, float)) else None


def extract_financial_features(companyfacts: Mapping[str, Any]) -> dict[str, Any]:
    series = {name: _annual_series(companyfacts, candidates) for name, candidates in CONCEPTS.items()}
    revenue_values = series["revenue"].get("values")
    if not isinstance(revenue_values, Mapping) or not revenue_values:
        return {
            "annual_period_end": None,
            "revenue_growth_yoy_pct": None,
            "gross_margin_pct": None,
            "operating_margin_pct": None,
            "net_margin_pct": None,
            "fcf_margin_pct": None,
            "operating_margin_yoy_change_pp": None,
            "profit_feature_count": 0,
            "provenance": {name: {k: s.get(k) for k in ("namespace", "tag", "unit")} for name, s in series.items()},
        }

    ends = sorted(revenue_values)
    latest_end = ends[-1]
    previous_end = ends[-2] if len(ends) >= 2 else None
    revenue = _value_at(series["revenue"], latest_end)
    previous_revenue = _value_at(series["revenue"], previous_end) if previous_end else None
    gross = _value_at(series["gross_profit"], latest_end)
    op = _value_at(series["operating_income"], latest_end)
    net = _value_at(series["net_income"], latest_end)
    cfo = _value_at(series["operating_cash_flow"], latest_end)
    capex = _value_at(series["capex"], latest_end)
    previous_op = _value_at(series["operating_income"], previous_end) if previous_end else None

    revenue_growth = None
    if revenue is not None and previous_revenue is not None and previous_revenue > 0:
        revenue_growth = (revenue / previous_revenue - 1.0) * 100.0
    gross_margin = _pct(gross, revenue)
    operating_margin = _pct(op, revenue)
    net_margin = _pct(net, revenue)
    fcf_margin = None
    if cfo is not None and capex is not None and revenue not in {None, 0}:
        fcf_margin = (cfo - abs(capex)) / revenue * 100.0
    op_change = None
    if operating_margin is not None and previous_op is not None and previous_revenue not in {None, 0}:
        op_change = operating_margin - previous_op / previous_revenue * 100.0

    metrics = {
        "revenue_growth_yoy_pct": revenue_growth,
        "gross_margin_pct": gross_margin,
        "operating_margin_pct": operating_margin,
        "net_margin_pct": net_margin,
        "fcf_margin_pct": fcf_margin,
        "operating_margin_yoy_change_pp": op_change,
    }
    return {
        "annual_period_end": latest_end,
        **{key: round(value, 2) if value is not None and math.isfinite(value) else None for key, value in metrics.items()},
        "profit_feature_count": sum(metrics[key] is not None and math.isfinite(float(metrics[key])) for key in FEATURES),
        "provenance": {name: {k: s.get(k) for k in ("namespace", "tag", "unit")} for name, s in series.items()},
    }


def industry_cluster(sic_code: str | None, sic_description: str | None = None) -> str:
    try:
        sic = int(str(sic_code))
    except (TypeError, ValueError):
        return "Unknown"
    if 100 <= sic <= 999:
        return "Agriculture"
    if 1000 <= sic <= 1499:
        return "Energy & Mining"
    if 1500 <= sic <= 1799:
        return "Construction"
    if 2000 <= sic <= 2099:
        return "Food & Beverage"
    if 2100 <= sic <= 2399:
        return "Consumer Goods & Apparel"
    if 2400 <= sic <= 2799:
        return "Materials, Packaging & Publishing"
    if 2800 <= sic <= 2899:
        return "Chemicals & Pharmaceuticals"
    if 2900 <= sic <= 2999:
        return "Oil Refining"
    if 3000 <= sic <= 3499:
        return "Materials & Fabricated Products"
    if 3500 <= sic <= 3599:
        return "Machinery & Computing Hardware"
    if 3600 <= sic <= 3699:
        return "Electronics & Semiconductors"
    if 3700 <= sic <= 3799:
        return "Transportation Equipment"
    if 3800 <= sic <= 3899:
        return "Instruments, Medical & Optical"
    if 3900 <= sic <= 3999:
        return "Misc Manufacturing"
    if 4000 <= sic <= 4799:
        return "Transportation & Logistics"
    if 4800 <= sic <= 4899:
        return "Telecom & Communications"
    if 4900 <= sic <= 4999:
        return "Utilities"
    if 5000 <= sic <= 5199:
        return "Wholesale"
    if 5200 <= sic <= 5999:
        return "Retail & Consumer Services"
    if 6000 <= sic <= 6199:
        return "Banking & Credit"
    if 6200 <= sic <= 6299:
        return "Securities & Brokers"
    if 6300 <= sic <= 6499:
        return "Insurance"
    if 6500 <= sic <= 6799:
        return "Real Estate & Investment"
    if 7000 <= sic <= 7199:
        return "Hospitality & Lodging"
    if 7200 <= sic <= 7299:
        return "Personal Services"
    if 7300 <= sic <= 7399:
        return "Software & Business Services"
    if 7500 <= sic <= 7799:
        return "Repair & Business Services"
    if 7800 <= sic <= 7999:
        return "Media & Entertainment"
    if 8000 <= sic <= 8099:
        return "Healthcare Services"
    if 8100 <= sic <= 8299:
        return "Legal & Education"
    if 8300 <= sic <= 8699:
        return "Social & Membership Services"
    if 8700 <= sic <= 8799:
        return "Engineering & Research"
    if 8800 <= sic <= 8999:
        return "Other Services"
    return "Other"


def is_financial_sector(cluster: str) -> bool:
    return cluster in {"Banking & Credit", "Securities & Brokers", "Insurance", "Real Estate & Investment"}


def _euclidean(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _kmeans(points: Sequence[Sequence[float]], k: int) -> tuple[list[list[float]], list[int]]:
    if len(points) < k:
        raise ValueError("not enough points")
    scalar = [sum(point) for point in points]
    order = sorted(range(len(points)), key=lambda i: scalar[i])
    centers = [list(points[order[round((len(order) - 1) * (i + 0.5) / k)]]) for i in range(k)]
    assignments = [0] * len(points)
    for _ in range(100):
        new_assignments = [min(range(k), key=lambda j: _euclidean(point, centers[j])) for point in points]
        new_centers: list[list[float]] = []
        for j in range(k):
            members = [points[i] for i, a in enumerate(new_assignments) if a == j]
            if not members:
                farthest = max(range(len(points)), key=lambda i: min(_euclidean(points[i], c) for c in centers))
                members = [points[farthest]]
            new_centers.append([statistics.fmean(member[d] for member in members) for d in range(len(points[0]))])
        if new_assignments == assignments and max(_euclidean(a, b) for a, b in zip(centers, new_centers)) < 1e-8:
            centers = new_centers
            assignments = new_assignments
            break
        centers = new_centers
        assignments = new_assignments
    return centers, assignments


def _silhouette(points: Sequence[Sequence[float]], assignments: Sequence[int], k: int) -> float:
    groups = [[i for i, a in enumerate(assignments) if a == j] for j in range(k)]
    scores: list[float] = []
    for i, point in enumerate(points):
        own = assignments[i]
        own_ids = [j for j in groups[own] if j != i]
        if not own_ids:
            scores.append(0.0)
            continue
        a = statistics.fmean(_euclidean(point, points[j]) for j in own_ids)
        b = min(
            statistics.fmean(_euclidean(point, points[j]) for j in ids)
            for group_id, ids in enumerate(groups)
            if group_id != own and ids
        )
        scores.append((b - a) / max(a, b) if max(a, b) else 0.0)
    return statistics.fmean(scores) if scores else 0.0


def _prepare_profit_points(rows: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[list[float]], dict[str, Any]]:
    eligible = [
        row for row in rows
        if not is_financial_sector(str(row.get("industry_cluster")))
        and int(row.get("profit_feature_count") or 0) >= 3
    ]
    stats: dict[str, dict[str, float]] = {}
    for feature in FEATURES:
        vals = [float(row[feature]) for row in eligible if isinstance(row.get(feature), (int, float)) and math.isfinite(float(row[feature]))]
        if not vals:
            stats[feature] = {"median": 0.0, "lo": 0.0, "hi": 0.0, "scale": 1.0}
            continue
        med = statistics.median(vals)
        q1, q3 = _percentile(vals, 0.25), _percentile(vals, 0.75)
        scale = (q3 - q1) / 1.349
        if scale <= 1e-9:
            scale = statistics.pstdev(vals) or 1.0
        stats[feature] = {
            "median": med,
            "lo": _percentile(vals, 0.05),
            "hi": _percentile(vals, 0.95),
            "scale": scale,
        }
    points: list[list[float]] = []
    for row in eligible:
        point: list[float] = []
        for feature in FEATURES:
            meta = stats[feature]
            value = row.get(feature)
            x = float(value) if isinstance(value, (int, float)) and math.isfinite(float(value)) else meta["median"]
            x = max(meta["lo"], min(meta["hi"], x))
            point.append((x - meta["median"]) / meta["scale"])
        points.append(point)
    return eligible, points, stats


def _module_label(summary: Mapping[str, Any]) -> str:
    growth = float(summary.get("revenue_growth_yoy_pct") or 0.0)
    op = float(summary.get("operating_margin_pct") or 0.0)
    fcf = float(summary.get("fcf_margin_pct") or 0.0)
    change = float(summary.get("operating_margin_yoy_change_pp") or 0.0)
    if op < 0:
        if growth >= 12 and change >= 0:
            return "高增长亏损"
        if change >= 3:
            return "亏损修复"
        return "持续亏损"
    if op >= 15 and fcf >= 8:
        if growth >= 12:
            return "高增长高盈利"
        if growth >= 4:
            return "高利润稳增长"
        return "高利润现金牛"
    if op >= 7:
        if growth >= 12:
            return "盈利增长"
        if change >= 3:
            return "利润改善"
        return "稳健盈利"
    if growth >= 12:
        return "低利润高增长"
    if change >= 3:
        return "薄利修复"
    return "低利润/周期"


def assign_profit_modules(rows: list[dict[str, Any]]) -> dict[str, Any]:
    eligible, points, feature_stats = _prepare_profit_points(rows)
    if len(eligible) < 20:
        for row in rows:
            row["profit_module"] = "P-SPARSE 数据不足"
        return {"k": 0, "silhouette": 0.0, "modules": [], "feature_stats": feature_stats}

    best: dict[str, Any] | None = None
    for k in range(4, min(7, len(points)) + 1):
        centers, assignments = _kmeans(points, k)
        counts = [assignments.count(j) for j in range(k)]
        if min(counts) < max(6, math.ceil(len(points) * 0.03)):
            continue
        silhouette = _silhouette(points, assignments, k)
        score = silhouette - 0.01 * (k - 4)
        candidate = {"k": k, "centers": centers, "assignments": assignments, "silhouette": silhouette, "score": score}
        if best is None or score > best["score"]:
            best = candidate
    if best is None:
        centers, assignments = _kmeans(points, 4)
        best = {"k": 4, "centers": centers, "assignments": assignments, "silhouette": _silhouette(points, assignments, 4), "score": 0.0}

    k = int(best["k"])
    cluster_rows: list[list[dict[str, Any]]] = [[] for _ in range(k)]
    for row, assignment in zip(eligible, best["assignments"]):
        cluster_rows[int(assignment)].append(row)

    summaries: list[dict[str, Any]] = []
    for idx, members in enumerate(cluster_rows):
        summary: dict[str, Any] = {"raw_cluster": idx, "count": len(members)}
        for feature in FEATURES:
            vals = [float(row[feature]) for row in members if isinstance(row.get(feature), (int, float))]
            summary[feature] = round(statistics.median(vals), 2) if vals else None
        summary["median_drawdown_pct"] = round(statistics.median(float(row["drawdown_from_peak_pct"]) for row in members), 2)
        summary["label"] = _module_label(summary)
        quality = (
            0.35 * float(summary.get("operating_margin_pct") or -20)
            + 0.25 * float(summary.get("fcf_margin_pct") or -20)
            + 0.20 * float(summary.get("revenue_growth_yoy_pct") or 0)
            + 0.20 * float(summary.get("operating_margin_yoy_change_pp") or 0)
        )
        summary["quality_sort"] = quality
        summaries.append(summary)

    ordered = sorted(summaries, key=lambda s: float(s["quality_sort"]), reverse=True)
    raw_to_module: dict[int, str] = {}
    final_summaries: list[dict[str, Any]] = []
    for rank, summary in enumerate(ordered, start=1):
        module = f"P{rank} {summary['label']}"
        raw_to_module[int(summary["raw_cluster"])] = module
        cleaned = {key: value for key, value in summary.items() if key not in {"raw_cluster", "quality_sort"}}
        cleaned["module"] = module
        final_summaries.append(cleaned)

    assignment_by_id = {id(row): int(a) for row, a in zip(eligible, best["assignments"])}
    for row in rows:
        if is_financial_sector(str(row.get("industry_cluster"))):
            row["profit_module"] = "P-FIN 金融口径单列"
        elif id(row) not in assignment_by_id:
            row["profit_module"] = "P-SPARSE 数据不足"
        else:
            row["profit_module"] = raw_to_module[assignment_by_id[id(row)]]

    for summary in final_summaries:
        members = [row for row in rows if row.get("profit_module") == summary["module"]]
        summary["g1"] = sum(int(row["pullback_group"]) == 1 for row in members)
        summary["g2"] = sum(int(row["pullback_group"]) == 2 for row in members)
        summary["g3"] = sum(int(row["pullback_group"]) == 3 for row in members)

    return {
        "k": k,
        "silhouette": round(float(best["silhouette"]), 4),
        "fit_count": len(eligible),
        "modules": final_summaries,
        "feature_stats": feature_stats,
    }


def _median_by(rows: Sequence[dict[str, Any]], key: str, value_key: str) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        label = str(row.get(key) or "Unknown")
        value = row.get(value_key)
        if isinstance(value, (int, float)):
            grouped[label].append(float(value))
    return {label: statistics.median(values) for label, values in grouped.items() if values}


def _cross_summary(rows: Sequence[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    overall = Counter(str(row.get(key) or "Unknown") for row in rows)
    group_totals = Counter(int(row["pullback_group"]) for row in rows)
    result: list[dict[str, Any]] = []
    for label, count in overall.items():
        item: dict[str, Any] = {"label": label, "count": count}
        members = [row for row in rows if str(row.get(key) or "Unknown") == label]
        item["median_drawdown_pct"] = round(statistics.median(float(row["drawdown_from_peak_pct"]) for row in members), 2)
        for g in (1, 2, 3):
            g_count = sum(int(row["pullback_group"]) == g for row in members)
            group_share = g_count / group_totals[g] if group_totals[g] else 0.0
            baseline_share = count / len(rows) if rows else 0.0
            item[f"g{g}_count"] = g_count
            item[f"g{g}_within_label_pct"] = round(g_count / count * 100.0, 1) if count else 0.0
            item[f"g{g}_lift"] = round(group_share / baseline_share, 2) if baseline_share else None
        result.append(item)
    result.sort(key=lambda x: (-int(x["count"]), str(x["label"])))
    return result


def _combo_summary(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row.get("industry_cluster") or "Unknown"), str(row.get("profit_module") or "Unknown"))].append(row)
    result: list[dict[str, Any]] = []
    for (industry, module), members in groups.items():
        if len(members) < 3:
            continue
        result.append({
            "industry_cluster": industry,
            "profit_module": module,
            "count": len(members),
            "g1": sum(int(row["pullback_group"]) == 1 for row in members),
            "g2": sum(int(row["pullback_group"]) == 2 for row in members),
            "g3": sum(int(row["pullback_group"]) == 3 for row in members),
            "median_drawdown_pct": round(statistics.median(float(row["drawdown_from_peak_pct"]) for row in members), 2),
        })
    result.sort(key=lambda x: (-int(x["count"]), float(x["median_drawdown_pct"])))
    return result


def enrich(input_json: Path, client: SecClient) -> dict[str, Any]:
    source = json.loads(input_json.read_text(encoding="utf-8"))
    stocks = source.get("stocks")
    if not isinstance(stocks, list):
        raise DataError("analysis JSON has no stocks list")
    ticker_map = client.ticker_map()
    rows: list[dict[str, Any]] = []
    coverage = Counter()

    for index, source_row in enumerate(stocks, start=1):
        if not isinstance(source_row, Mapping):
            continue
        row = dict(source_row)
        ticker = str(row.get("ticker") or "").upper()
        mapping = ticker_map.get(normalize_ticker(ticker))
        if mapping is None:
            row.update({
                "cik": None,
                "sic_code": None,
                "sic_description": None,
                "industry_cluster": "Unknown",
                "fundamentals_status": "missing_cik",
                "profit_feature_count": 0,
            })
            rows.append(row)
            coverage["missing_cik"] += 1
            continue

        cik = int(mapping["cik"])
        submissions = client.submissions(cik)
        facts = client.companyfacts(cik)
        sic_code = str(submissions.get("sic")) if isinstance(submissions, Mapping) and submissions.get("sic") is not None else None
        sic_description = str(submissions.get("sicDescription")) if isinstance(submissions, Mapping) and submissions.get("sicDescription") else None
        row.update({
            "cik": cik,
            "sic_code": sic_code,
            "sic_description": sic_description,
            "industry_cluster": industry_cluster(sic_code, sic_description),
        })
        if facts is None:
            row.update({
                "annual_period_end": None,
                "revenue_growth_yoy_pct": None,
                "gross_margin_pct": None,
                "operating_margin_pct": None,
                "net_margin_pct": None,
                "fcf_margin_pct": None,
                "operating_margin_yoy_change_pp": None,
                "profit_feature_count": 0,
                "fundamentals_status": "missing_companyfacts",
                "provenance": {},
            })
            coverage["missing_companyfacts"] += 1
        else:
            metrics = extract_financial_features(facts)
            row.update(metrics)
            row["fundamentals_status"] = "ok" if int(metrics["profit_feature_count"]) >= 3 else "sparse"
            coverage[row["fundamentals_status"]] += 1
        if sic_code:
            coverage["with_sic"] += 1
        rows.append(row)
        if index % 50 == 0:
            print(f"enriched {index}/{len(stocks)}")

    profit_clustering = assign_profit_modules(rows)

    industry_median = _median_by(rows, "industry_cluster", "drawdown_from_peak_pct")
    module_median = _median_by(rows, "profit_module", "drawdown_from_peak_pct")
    combo_groups: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        combo_groups[(str(row["industry_cluster"]), str(row["profit_module"]))].append(float(row["drawdown_from_peak_pct"]))
    combo_median = {key: statistics.median(values) for key, values in combo_groups.items() if len(values) >= 5}

    for row in rows:
        dd = float(row["drawdown_from_peak_pct"])
        ind = str(row["industry_cluster"])
        mod = str(row["profit_module"])
        ind_med = industry_median.get(ind)
        mod_med = module_median.get(mod)
        peer_med = combo_median.get((ind, mod), ind_med)
        row["industry_median_drawdown_pct"] = round(ind_med, 2) if ind_med is not None else None
        row["industry_adjusted_drawdown_residual_pp"] = round(dd - ind_med, 2) if ind_med is not None else None
        row["profit_module_median_drawdown_pct"] = round(mod_med, 2) if mod_med is not None else None
        row["profit_module_adjusted_drawdown_residual_pp"] = round(dd - mod_med, 2) if mod_med is not None else None
        row["peer_median_drawdown_pct"] = round(peer_med, 2) if peer_med is not None else None
        row["peer_adjusted_drawdown_residual_pp"] = round(dd - peer_med, 2) if peer_med is not None else None

    industry_summary = _cross_summary(rows, "industry_cluster")
    module_summary = _cross_summary(rows, "profit_module")
    combos = _combo_summary(rows)

    overall_group_counts = Counter(int(row["pullback_group"]) for row in rows)
    return {
        "methodology": {
            "source_wave_analysis": str(input_json),
            "industry_source": "SEC submissions SIC code and SIC description",
            "profit_source": "SEC companyfacts annual filing facts",
            "profit_features": list(FEATURES),
            "profit_cluster_scope": "non-financial stocks with >=3 available features; financial industries are a separate module",
            "industry_neutral_field": "industry_adjusted_drawdown_residual_pp = stock drawdown - industry median drawdown",
            "peer_neutral_field": "peer_adjusted_drawdown_residual_pp = stock drawdown - median of industry×profit-module peers when n>=5, otherwise industry median",
        },
        "coverage": dict(coverage),
        "stock_count": len(rows),
        "group_counts": {f"G{g}": overall_group_counts[g] for g in (1, 2, 3)},
        "profit_clustering": profit_clustering,
        "industry_summary": industry_summary,
        "profit_module_summary": module_summary,
        "industry_profit_combos": combos,
        "stocks": rows,
    }


def _fmt(value: Any, digits: int = 1) -> str:
    return "—" if value is None else f"{float(value):.{digits}f}"


def render_report(result: Mapping[str, Any]) -> str:
    rows = result["stocks"]
    industries = result["industry_summary"]
    modules = result["profit_module_summary"]
    combos = result["industry_profit_combos"]
    clustering = result["profit_clustering"]
    group_counts = result["group_counts"]
    baseline = {g: int(group_counts[f"G{g}"]) / len(rows) for g in (1, 2, 3)}

    lines = [
        "# Apr–May Peak Class — Industry & Profitability Structure",
        "",
        "## Coverage",
        "",
        f"- Matched stocks: **{len(rows)}** (G1 {group_counts['G1']}, G2 {group_counts['G2']}, G3 {group_counts['G3']}).",
        f"- SEC SIC coverage: **{result['coverage'].get('with_sic', 0)} / {len(rows)}**.",
        f"- Profit clustering fit count: **{clustering.get('fit_count', 0)}**; selected k = **{clustering.get('k', 0)}**, silhouette = **{float(clustering.get('silhouette', 0)):.3f}**.",
        "- Financial industries are separated because bank/insurance statement economics are not directly comparable with operating companies.",
        "",
        "## Industry commonality",
        "",
        "| Industry cluster | N | G1 | G2 | G3 | Median drawdown | G1 lift | G3 lift |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in industries[:25]:
        lines.append(
            f"| {item['label']} | {item['count']} | {item['g1_count']} | {item['g2_count']} | {item['g3_count']} | {item['median_drawdown_pct']:.1f}% | {item['g1_lift']:.2f}x | {item['g3_lift']:.2f}x |"
        )

    lines.extend(["", "### Industry signals (minimum N=8)", ""])
    strong_g1 = [x for x in industries if int(x["count"]) >= 8 and float(x.get("g1_lift") or 0) >= 1.25]
    strong_g3 = [x for x in industries if int(x["count"]) >= 8 and float(x.get("g3_lift") or 0) >= 1.25]
    lines.append("- **More resistant than the class:** " + (", ".join(f"{x['label']} ({x['g1_lift']:.2f}x G1 lift, N={x['count']})" for x in strong_g1[:8]) or "none"))
    lines.append("- **More likely to deep-retrace:** " + (", ".join(f"{x['label']} ({x['g3_lift']:.2f}x G3 lift, N={x['count']})" for x in strong_g3[:8]) or "none"))

    lines.extend([
        "",
        "## Profitability modules",
        "",
        "Profit modules are learned from revenue growth, operating margin, net margin, FCF margin and YoY operating-margin change. Labels describe cluster centroids; they are not hand-assigned stock-by-stock.",
        "",
        "| Module | N | Rev growth | Op margin | Net margin | FCF margin | Op margin Δ | Median drawdown | G1/G2/G3 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    module_meta = {m["module"]: m for m in clustering.get("modules", [])}
    for item in modules:
        meta = module_meta.get(item["label"], {})
        lines.append(
            f"| {item['label']} | {item['count']} | {_fmt(meta.get('revenue_growth_yoy_pct'))}% | {_fmt(meta.get('operating_margin_pct'))}% | {_fmt(meta.get('net_margin_pct'))}% | {_fmt(meta.get('fcf_margin_pct'))}% | {_fmt(meta.get('operating_margin_yoy_change_pp'))}pp | {item['median_drawdown_pct']:.1f}% | {item['g1_count']}/{item['g2_count']}/{item['g3_count']} |"
        )

    lines.extend(["", "### Profit-module signals (minimum N=8)", ""])
    mod_g1 = [x for x in modules if int(x["count"]) >= 8 and float(x.get("g1_lift") or 0) >= 1.25]
    mod_g3 = [x for x in modules if int(x["count"]) >= 8 and float(x.get("g3_lift") or 0) >= 1.25]
    lines.append("- **More represented in G1:** " + (", ".join(f"{x['label']} ({x['g1_lift']:.2f}x)" for x in mod_g1[:8]) or "none"))
    lines.append("- **More represented in G3:** " + (", ".join(f"{x['label']} ({x['g3_lift']:.2f}x)" for x in mod_g3[:8]) or "none"))

    lines.extend([
        "",
        "## Industry × profitability combinations",
        "",
        "| Industry | Profit module | N | G1 | G2 | G3 | Median drawdown |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for item in combos[:30]:
        lines.append(
            f"| {item['industry_cluster']} | {item['profit_module']} | {item['count']} | {item['g1']} | {item['g2']} | {item['g3']} | {item['median_drawdown_pct']:.1f}% |"
        )

    lines.extend([
        "",
        "## Foundation for industry-neutral company analysis",
        "",
        "Each stock now carries three benchmark residuals:",
        "",
        "- `industry_adjusted_drawdown_residual_pp`: stock drawdown minus its broad-industry median. Negative means the stock held up better than its industry; positive means it underperformed its industry.",
        "- `profit_module_adjusted_drawdown_residual_pp`: same comparison versus companies with a similar profitability structure.",
        "- `peer_adjusted_drawdown_residual_pp`: comparison versus industry × profit-module peers when that peer cell has at least five stocks; otherwise it falls back to the industry median.",
        "",
        "These are price-behaviour residuals, not intrinsic-value scores. They are intended as the next-stage control variables before judging company-specific potential.",
    ])
    return "\n".join(lines) + "\n"


def write_outputs(result: Mapping[str, Any], output_dir: Path, fundamentals_out: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "industry_profit_analysis.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "industry_profit_report.md").write_text(render_report(result), encoding="utf-8")

    stocks = result["stocks"]
    fieldnames: list[str] = []
    for row in stocks:
        for key in row:
            if key == "provenance":
                continue
            if key not in fieldnames:
                fieldnames.append(key)
    with (output_dir / "enriched_stocks.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in stocks:
            writer.writerow({key: row.get(key) for key in fieldnames})

    normalized = {
        "source": "SEC submissions + SEC companyfacts",
        "stock_count": len(stocks),
        "stocks": [
            {
                key: row.get(key)
                for key in (
                    "ticker", "cik", "sic_code", "sic_description", "industry_cluster",
                    "annual_period_end", "revenue_growth_yoy_pct", "gross_margin_pct",
                    "operating_margin_pct", "net_margin_pct", "fcf_margin_pct",
                    "operating_margin_yoy_change_pp", "profit_feature_count",
                    "fundamentals_status", "profit_module", "provenance",
                )
            }
            for row in stocks
        ],
    }
    fundamentals_out.parent.mkdir(parents=True, exist_ok=True)
    fundamentals_out.write_text(json.dumps(normalized, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("analysis/apr_may_peak/analysis.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("analysis/apr_may_peak"))
    parser.add_argument("--fundamentals-out", type=Path, default=Path("fundamentals/sec/apr_may_peak_latest.json"))
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/sec"))
    parser.add_argument("--request-delay", type=float, default=0.20)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        client = SecClient(
            args.cache_dir,
            request_delay=args.request_delay,
            user_agent=os.environ.get("SEC_USER_AGENT") or DEFAULT_USER_AGENT,
        )
        result = enrich(args.input, client)
        write_outputs(result, args.output_dir, args.fundamentals_out)
    except (DataError, OSError, ValueError) as exc:
        print(f"error: {exc}")
        return 2
    print(json.dumps({
        "stock_count": result["stock_count"],
        "coverage": result["coverage"],
        "profit_k": result["profit_clustering"].get("k"),
        "profit_silhouette": result["profit_clustering"].get("silhouette"),
        "industry_clusters": len(result["industry_summary"]),
        "profit_modules": len(result["profit_module_summary"]),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
