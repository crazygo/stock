#!/usr/bin/env python3
"""Enrich the Apr-May weekly-peak class with industry and profitability structure.

Price-wave data remains sourced from the repository's Massive EOD archive.
This enrichment layer uses Yahoo Finance snapshots through yfinance because
SEC data endpoints reject GitHub-hosted runners and the repository's Massive
plan is not entitled to the Financials API.

The outputs are designed for a later industry-neutral company analysis. Every
stock receives sector, detailed-industry and industry×profit-module benchmark
drawdowns plus residuals versus those benchmarks.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    import yfinance as yf
except ImportError:  # pragma: no cover - exercised only in an unprepared runtime
    yf = None


FEATURES = (
    "revenue_growth_yoy_pct",
    "operating_margin_pct",
    "net_margin_pct",
    "fcf_margin_pct",
    "earnings_growth_yoy_pct",
)

SPECIAL_PROFIT_SECTORS = {
    "Financial Services": "P-FIN 金融口径单列",
    "Real Estate": "P-REIT 房地产口径单列",
}

YAHOO_KEYS = (
    "sector",
    "industry",
    "country",
    "marketCap",
    "totalRevenue",
    "freeCashflow",
    "revenueGrowth",
    "operatingMargins",
    "profitMargins",
    "grossMargins",
    "earningsGrowth",
    "returnOnEquity",
)


class DataError(RuntimeError):
    pass


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    return None


def _fraction_to_pct(value: Any) -> float | None:
    number = _finite_number(value)
    return number * 100.0 if number is not None else None


def normalize_yahoo_info(info: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize only the provider fields needed for this research layer."""
    total_revenue = _finite_number(info.get("totalRevenue"))
    free_cashflow = _finite_number(info.get("freeCashflow"))
    fcf_margin = None
    if total_revenue not in {None, 0.0} and free_cashflow is not None:
        fcf_margin = free_cashflow / total_revenue * 100.0

    metrics = {
        "revenue_growth_yoy_pct": _fraction_to_pct(info.get("revenueGrowth")),
        "gross_margin_pct": _fraction_to_pct(info.get("grossMargins")),
        "operating_margin_pct": _fraction_to_pct(info.get("operatingMargins")),
        "net_margin_pct": _fraction_to_pct(info.get("profitMargins")),
        "fcf_margin_pct": fcf_margin,
        "earnings_growth_yoy_pct": _fraction_to_pct(info.get("earningsGrowth")),
        "return_on_equity_pct": _fraction_to_pct(info.get("returnOnEquity")),
    }
    normalized = {
        "sector": str(info.get("sector") or "Unknown"),
        "industry": str(info.get("industry") or "Unknown"),
        "country": str(info.get("country") or "Unknown"),
        "market_cap": _finite_number(info.get("marketCap")),
        "total_revenue": total_revenue,
        "free_cashflow": free_cashflow,
        **{
            key: round(value, 2) if value is not None and math.isfinite(value) else None
            for key, value in metrics.items()
        },
    }
    normalized["profit_feature_count"] = sum(
        normalized.get(feature) is not None for feature in FEATURES
    )
    normalized["fundamentals_status"] = (
        "ok" if normalized["profit_feature_count"] >= 3 else "sparse"
    )
    return normalized


def yahoo_ticker(ticker: str) -> str:
    return ticker.upper().replace(".", "-").replace("/", "-")


class YahooSnapshotClient:
    def __init__(self, cache_dir: Path, retries: int = 3, retry_delay: float = 1.5) -> None:
        if yf is None:
            raise DataError("yfinance is required; install it before running enrichment")
        self.cache_dir = cache_dir
        self.retries = max(1, retries)
        self.retry_delay = max(0.0, retry_delay)

    def _path(self, ticker: str) -> Path:
        safe = ticker.replace("/", "_").replace("\\", "_")
        return self.cache_dir / f"{safe}.json"

    def fetch_one(self, ticker: str) -> dict[str, Any]:
        path = self._path(ticker)
        if path.exists():
            try:
                cached = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(cached, Mapping):
                    return dict(cached)
            except json.JSONDecodeError:
                path.unlink(missing_ok=True)

        symbol = yahoo_ticker(ticker)
        last_error: str | None = None
        for attempt in range(self.retries):
            try:
                raw = yf.Ticker(symbol).get_info()
                if isinstance(raw, Mapping) and raw:
                    compact = {key: raw.get(key) for key in YAHOO_KEYS}
                    normalized = normalize_yahoo_info(compact)
                    normalized.update(
                        {
                            "ticker": ticker,
                            "provider_symbol": symbol,
                            "source": "Yahoo Finance via yfinance",
                            "fetch_status": "ok",
                        }
                    )
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(
                        json.dumps(normalized, ensure_ascii=False, separators=(",", ":")),
                        encoding="utf-8",
                    )
                    return normalized
                last_error = "empty_info"
            except Exception as exc:  # provider exceptions vary between yfinance releases
                last_error = f"{type(exc).__name__}: {exc}"[:300]
            if attempt + 1 < self.retries:
                time.sleep(self.retry_delay * (attempt + 1))

        result = {
            "ticker": ticker,
            "provider_symbol": symbol,
            "source": "Yahoo Finance via yfinance",
            "fetch_status": "error",
            "fetch_error": last_error,
            "sector": "Unknown",
            "industry": "Unknown",
            "country": "Unknown",
            "profit_feature_count": 0,
            "fundamentals_status": "missing",
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
        return result

    def fetch_many(self, tickers: Sequence[str], workers: int) -> dict[str, dict[str, Any]]:
        results: dict[str, dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            future_to_ticker = {executor.submit(self.fetch_one, ticker): ticker for ticker in tickers}
            completed = 0
            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    results[ticker] = future.result()
                except Exception as exc:
                    results[ticker] = {
                        "ticker": ticker,
                        "fetch_status": "error",
                        "fetch_error": f"{type(exc).__name__}: {exc}"[:300],
                        "sector": "Unknown",
                        "industry": "Unknown",
                        "profit_feature_count": 0,
                        "fundamentals_status": "missing",
                    }
                completed += 1
                if completed % 50 == 0:
                    print(f"fetched {completed}/{len(tickers)} Yahoo snapshots")
        return results


def _percentile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile requires values")
    pos = (len(ordered) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    weight = pos - lo
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight


def _euclidean(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _kmeans(points: Sequence[Sequence[float]], k: int) -> tuple[list[list[float]], list[int]]:
    if len(points) < k:
        raise ValueError("not enough points")
    quality_axis = [sum(point) for point in points]
    order = sorted(range(len(points)), key=lambda index: quality_axis[index])
    centers = [
        list(points[order[round((len(order) - 1) * (i + 0.5) / k)]])
        for i in range(k)
    ]
    assignments = [-1] * len(points)
    for _ in range(100):
        new_assignments = [
            min(range(k), key=lambda cluster: _euclidean(point, centers[cluster]))
            for point in points
        ]
        new_centers: list[list[float]] = []
        for cluster in range(k):
            members = [
                points[index]
                for index, assignment in enumerate(new_assignments)
                if assignment == cluster
            ]
            if not members:
                farthest = max(
                    range(len(points)),
                    key=lambda index: min(_euclidean(points[index], center) for center in centers),
                )
                members = [points[farthest]]
            new_centers.append(
                [
                    statistics.fmean(member[dimension] for member in members)
                    for dimension in range(len(points[0]))
                ]
            )
        if (
            new_assignments == assignments
            and max(_euclidean(a, b) for a, b in zip(centers, new_centers)) < 1e-8
        ):
            centers = new_centers
            assignments = new_assignments
            break
        centers = new_centers
        assignments = new_assignments
    return centers, assignments


def _silhouette(points: Sequence[Sequence[float]], assignments: Sequence[int], k: int) -> float:
    groups = [[i for i, assignment in enumerate(assignments) if assignment == group] for group in range(k)]
    scores: list[float] = []
    for index, point in enumerate(points):
        own = assignments[index]
        own_ids = [other for other in groups[own] if other != index]
        if not own_ids:
            scores.append(0.0)
            continue
        within = statistics.fmean(_euclidean(point, points[other]) for other in own_ids)
        nearest_other = min(
            statistics.fmean(_euclidean(point, points[other]) for other in ids)
            for group, ids in enumerate(groups)
            if group != own and ids
        )
        denominator = max(within, nearest_other)
        scores.append((nearest_other - within) / denominator if denominator else 0.0)
    return statistics.fmean(scores) if scores else 0.0


def _special_profit_module(sector: str) -> str | None:
    return SPECIAL_PROFIT_SECTORS.get(sector)


def _prepare_profit_points(rows: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[list[float]], dict[str, Any]]:
    eligible = [
        row
        for row in rows
        if _special_profit_module(str(row.get("sector") or "Unknown")) is None
        and int(row.get("profit_feature_count") or 0) >= 3
    ]
    stats: dict[str, dict[str, float]] = {}
    for feature in FEATURES:
        values = [
            float(row[feature])
            for row in eligible
            if _finite_number(row.get(feature)) is not None
        ]
        if not values:
            stats[feature] = {"median": 0.0, "lo": 0.0, "hi": 0.0, "scale": 1.0}
            continue
        median = statistics.median(values)
        q1, q3 = _percentile(values, 0.25), _percentile(values, 0.75)
        scale = (q3 - q1) / 1.349
        if scale <= 1e-9:
            scale = statistics.pstdev(values) or 1.0
        stats[feature] = {
            "median": median,
            "lo": _percentile(values, 0.05),
            "hi": _percentile(values, 0.95),
            "scale": scale,
        }

    points: list[list[float]] = []
    for row in eligible:
        point: list[float] = []
        for feature in FEATURES:
            meta = stats[feature]
            number = _finite_number(row.get(feature))
            value = number if number is not None else meta["median"]
            value = max(meta["lo"], min(meta["hi"], value))
            point.append((value - meta["median"]) / meta["scale"])
        points.append(point)
    return eligible, points, stats


def _module_label(summary: Mapping[str, Any]) -> str:
    growth = float(summary.get("revenue_growth_yoy_pct") or 0.0)
    operating_margin = float(summary.get("operating_margin_pct") or 0.0)
    fcf_margin = float(summary.get("fcf_margin_pct") or 0.0)
    earnings_growth = float(summary.get("earnings_growth_yoy_pct") or 0.0)
    if operating_margin < 0:
        if growth >= 15 and earnings_growth >= 0:
            return "高增长亏损"
        return "持续亏损"
    if operating_margin >= 18 and fcf_margin >= 10:
        if growth >= 12:
            return "高增长高盈利"
        if growth >= 3:
            return "高利润稳增长"
        return "高利润现金牛"
    if operating_margin >= 8:
        if growth >= 12:
            return "盈利增长"
        if earnings_growth >= 15:
            return "盈利改善"
        return "稳健盈利"
    if growth >= 15:
        return "低利润高增长"
    if earnings_growth >= 15:
        return "薄利改善"
    return "低利润/周期"


def assign_profit_modules(rows: list[dict[str, Any]]) -> dict[str, Any]:
    eligible, points, feature_stats = _prepare_profit_points(rows)
    if len(eligible) < 20:
        for row in rows:
            special = _special_profit_module(str(row.get("sector") or "Unknown"))
            row["profit_module"] = special or "P-SPARSE 数据不足"
        return {"k": 0, "silhouette": 0.0, "fit_count": len(eligible), "modules": [], "feature_stats": feature_stats}

    best: dict[str, Any] | None = None
    for k in range(4, min(7, len(points)) + 1):
        centers, assignments = _kmeans(points, k)
        counts = [assignments.count(cluster) for cluster in range(k)]
        if min(counts) < max(6, math.ceil(len(points) * 0.03)):
            continue
        silhouette = _silhouette(points, assignments, k)
        score = silhouette - 0.01 * (k - 4)
        candidate = {
            "k": k,
            "centers": centers,
            "assignments": assignments,
            "silhouette": silhouette,
            "score": score,
        }
        if best is None or score > best["score"]:
            best = candidate
    if best is None:
        centers, assignments = _kmeans(points, 4)
        best = {
            "k": 4,
            "centers": centers,
            "assignments": assignments,
            "silhouette": _silhouette(points, assignments, 4),
            "score": 0.0,
        }

    k = int(best["k"])
    cluster_rows: list[list[dict[str, Any]]] = [[] for _ in range(k)]
    for row, assignment in zip(eligible, best["assignments"]):
        cluster_rows[int(assignment)].append(row)

    summaries: list[dict[str, Any]] = []
    for raw_cluster, members in enumerate(cluster_rows):
        summary: dict[str, Any] = {"raw_cluster": raw_cluster, "count": len(members)}
        for feature in FEATURES:
            values = [
                float(row[feature])
                for row in members
                if _finite_number(row.get(feature)) is not None
            ]
            summary[feature] = round(statistics.median(values), 2) if values else None
        summary["median_drawdown_pct"] = round(
            statistics.median(float(row["drawdown_from_peak_pct"]) for row in members), 2
        )
        summary["label"] = _module_label(summary)
        quality = (
            0.28 * float(summary.get("operating_margin_pct") or -20)
            + 0.22 * float(summary.get("fcf_margin_pct") or -20)
            + 0.22 * float(summary.get("revenue_growth_yoy_pct") or 0)
            + 0.13 * float(summary.get("net_margin_pct") or -20)
            + 0.15 * float(summary.get("earnings_growth_yoy_pct") or 0)
        )
        summary["quality_sort"] = quality
        summaries.append(summary)

    ordered = sorted(summaries, key=lambda summary: float(summary["quality_sort"]), reverse=True)
    raw_to_module: dict[int, str] = {}
    final_summaries: list[dict[str, Any]] = []
    for rank, summary in enumerate(ordered, start=1):
        module = f"P{rank} {summary['label']}"
        raw_to_module[int(summary["raw_cluster"])] = module
        cleaned = {
            key: value
            for key, value in summary.items()
            if key not in {"raw_cluster", "quality_sort"}
        }
        cleaned["module"] = module
        final_summaries.append(cleaned)

    assignment_by_id = {id(row): int(assignment) for row, assignment in zip(eligible, best["assignments"])}
    for row in rows:
        sector = str(row.get("sector") or "Unknown")
        special = _special_profit_module(sector)
        if special:
            row["profit_module"] = special
        elif id(row) not in assignment_by_id:
            row["profit_module"] = "P-SPARSE 数据不足"
        else:
            row["profit_module"] = raw_to_module[assignment_by_id[id(row)]]

    for summary in final_summaries:
        members = [row for row in rows if row.get("profit_module") == summary["module"]]
        for group in (1, 2, 3):
            summary[f"g{group}"] = sum(int(row["pullback_group"]) == group for row in members)

    return {
        "k": k,
        "silhouette": round(float(best["silhouette"]), 4),
        "fit_count": len(eligible),
        "modules": final_summaries,
        "feature_stats": feature_stats,
    }


def _cross_summary(rows: Sequence[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    overall = Counter(str(row.get(key) or "Unknown") for row in rows)
    group_totals = Counter(int(row["pullback_group"]) for row in rows)
    result: list[dict[str, Any]] = []
    for label, count in overall.items():
        members = [row for row in rows if str(row.get(key) or "Unknown") == label]
        item: dict[str, Any] = {
            "label": label,
            "count": count,
            "median_drawdown_pct": round(
                statistics.median(float(row["drawdown_from_peak_pct"]) for row in members), 2
            ),
        }
        for group in (1, 2, 3):
            group_count = sum(int(row["pullback_group"]) == group for row in members)
            share_inside_group = group_count / group_totals[group] if group_totals[group] else 0.0
            baseline_share = count / len(rows) if rows else 0.0
            item[f"g{group}_count"] = group_count
            item[f"g{group}_within_label_pct"] = round(group_count / count * 100.0, 1) if count else 0.0
            item[f"g{group}_lift"] = round(share_inside_group / baseline_share, 2) if baseline_share else None
        result.append(item)
    result.sort(key=lambda item: (-int(item["count"]), str(item["label"])))
    return result


def _combo_summary(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("industry") or "Unknown"), str(row.get("profit_module") or "Unknown"))].append(row)
    result: list[dict[str, Any]] = []
    for (industry, module), members in grouped.items():
        if len(members) < 3:
            continue
        result.append(
            {
                "industry": industry,
                "profit_module": module,
                "count": len(members),
                "g1": sum(int(row["pullback_group"]) == 1 for row in members),
                "g2": sum(int(row["pullback_group"]) == 2 for row in members),
                "g3": sum(int(row["pullback_group"]) == 3 for row in members),
                "median_drawdown_pct": round(
                    statistics.median(float(row["drawdown_from_peak_pct"]) for row in members), 2
                ),
            }
        )
    result.sort(key=lambda item: (-int(item["count"]), float(item["median_drawdown_pct"])))
    return result


def _median_map(rows: Sequence[dict[str, Any]], key: str) -> tuple[dict[str, float], dict[str, int]]:
    values: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        values[str(row.get(key) or "Unknown")].append(float(row["drawdown_from_peak_pct"]))
    return (
        {label: statistics.median(group) for label, group in values.items()},
        {label: len(group) for label, group in values.items()},
    )


def add_neutral_residuals(rows: list[dict[str, Any]]) -> None:
    sector_median, _ = _median_map(rows, "sector")
    industry_median, industry_count = _median_map(rows, "industry")
    module_median, _ = _median_map(rows, "profit_module")

    combo_values: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        combo_values[(str(row["industry"]), str(row["profit_module"]))].append(
            float(row["drawdown_from_peak_pct"])
        )
    combo_median = {
        key: statistics.median(values)
        for key, values in combo_values.items()
        if len(values) >= 5
    }

    for row in rows:
        drawdown = float(row["drawdown_from_peak_pct"])
        sector = str(row.get("sector") or "Unknown")
        industry = str(row.get("industry") or "Unknown")
        module = str(row.get("profit_module") or "Unknown")
        sector_benchmark = sector_median.get(sector)
        if industry_count.get(industry, 0) >= 5:
            industry_benchmark = industry_median[industry]
            industry_basis = f"industry:{industry}"
        else:
            industry_benchmark = sector_benchmark
            industry_basis = f"sector_fallback:{sector}"
        peer_benchmark = combo_median.get((industry, module), industry_benchmark)
        peer_basis = (
            f"industry×module:{industry} × {module}"
            if (industry, module) in combo_median
            else industry_basis
        )

        row["sector_median_drawdown_pct"] = round(sector_benchmark, 2) if sector_benchmark is not None else None
        row["sector_adjusted_drawdown_residual_pp"] = round(drawdown - sector_benchmark, 2) if sector_benchmark is not None else None
        row["industry_median_drawdown_pct"] = round(industry_benchmark, 2) if industry_benchmark is not None else None
        row["industry_adjusted_drawdown_residual_pp"] = round(drawdown - industry_benchmark, 2) if industry_benchmark is not None else None
        row["industry_benchmark_basis"] = industry_basis
        row["profit_module_median_drawdown_pct"] = round(module_median[module], 2) if module in module_median else None
        row["profit_module_adjusted_drawdown_residual_pp"] = round(drawdown - module_median[module], 2) if module in module_median else None
        row["peer_median_drawdown_pct"] = round(peer_benchmark, 2) if peer_benchmark is not None else None
        row["peer_adjusted_drawdown_residual_pp"] = round(drawdown - peer_benchmark, 2) if peer_benchmark is not None else None
        row["peer_benchmark_basis"] = peer_basis


def enrich(input_json: Path, client: YahooSnapshotClient, workers: int) -> dict[str, Any]:
    source = json.loads(input_json.read_text(encoding="utf-8"))
    stocks = source.get("stocks")
    if not isinstance(stocks, list):
        raise DataError("analysis JSON has no stocks list")

    tickers = [str(row.get("ticker")) for row in stocks if isinstance(row, Mapping) and row.get("ticker")]
    snapshots = client.fetch_many(tickers, workers)
    rows: list[dict[str, Any]] = []
    coverage = Counter()
    for source_row in stocks:
        if not isinstance(source_row, Mapping):
            continue
        row = dict(source_row)
        ticker = str(row.get("ticker") or "")
        snapshot = snapshots.get(ticker) or {
            "fetch_status": "error",
            "sector": "Unknown",
            "industry": "Unknown",
            "profit_feature_count": 0,
            "fundamentals_status": "missing",
        }
        row.update(snapshot)
        coverage[str(snapshot.get("fetch_status") or "unknown")] += 1
        if str(snapshot.get("sector") or "Unknown") != "Unknown":
            coverage["with_sector"] += 1
        if str(snapshot.get("industry") or "Unknown") != "Unknown":
            coverage["with_industry"] += 1
        if int(snapshot.get("profit_feature_count") or 0) >= 3:
            coverage["profit_3plus"] += 1
        rows.append(row)

    total = len(rows)
    if total == 0:
        raise DataError("no stocks to enrich")
    if coverage["with_sector"] / total < 0.80:
        raise DataError(f"sector coverage too low: {coverage['with_sector']}/{total}")
    if coverage["profit_3plus"] / total < 0.55:
        raise DataError(f"profit feature coverage too low: {coverage['profit_3plus']}/{total}")

    profit_clustering = assign_profit_modules(rows)
    add_neutral_residuals(rows)

    sector_summary = _cross_summary(rows, "sector")
    industry_summary = _cross_summary(rows, "industry")
    profit_module_summary = _cross_summary(rows, "profit_module")
    combos = _combo_summary(rows)
    group_counts = Counter(int(row["pullback_group"]) for row in rows)

    return {
        "methodology": {
            "source_wave_analysis": str(input_json),
            "wave_source": "Massive grouped adjusted EOD archive",
            "enrichment_source": "Yahoo Finance snapshot via yfinance",
            "profit_features": list(FEATURES),
            "profit_cluster_scope": "non-Financial-Services/non-Real-Estate stocks with >=3 available features",
            "provider_semantics_note": "Yahoo snapshot fields mix trailing/current provider metrics; modules are cross-sectional operating archetypes, not audited annual-period classifications.",
            "sector_neutral_field": "sector_adjusted_drawdown_residual_pp = stock drawdown - sector median drawdown",
            "industry_neutral_field": "industry_adjusted_drawdown_residual_pp = stock drawdown - detailed-industry median when n>=5, otherwise sector median",
            "peer_neutral_field": "peer_adjusted_drawdown_residual_pp = stock drawdown - detailed-industry×profit-module median when n>=5, otherwise industry/sector benchmark",
        },
        "coverage": dict(coverage),
        "stock_count": total,
        "group_counts": {f"G{group}": group_counts[group] for group in (1, 2, 3)},
        "profit_clustering": profit_clustering,
        "sector_summary": sector_summary,
        "industry_summary": industry_summary,
        "profit_module_summary": profit_module_summary,
        "industry_profit_combos": combos,
        "stocks": rows,
    }


def _fmt(value: Any, digits: int = 1) -> str:
    number = _finite_number(value)
    return "—" if number is None else f"{number:.{digits}f}"


def _signals(summary: Sequence[Mapping[str, Any]], group: int, min_n: int, min_lift: float) -> list[Mapping[str, Any]]:
    key = f"g{group}_lift"
    rows = [
        item
        for item in summary
        if int(item.get("count") or 0) >= min_n
        and _finite_number(item.get(key)) is not None
        and float(item[key]) >= min_lift
    ]
    rows.sort(key=lambda item: (-float(item[key]), -int(item["count"])))
    return rows


def render_report(result: Mapping[str, Any]) -> str:
    rows = result["stocks"]
    sectors = result["sector_summary"]
    industries = result["industry_summary"]
    modules = result["profit_module_summary"]
    combos = result["industry_profit_combos"]
    clustering = result["profit_clustering"]
    counts = result["group_counts"]
    coverage = result["coverage"]

    lines = [
        "# Apr–May Peak Class — Industry & Profitability Structure",
        "",
        "## Coverage and source",
        "",
        f"- Wave class: **{len(rows)}** stocks (G1 {counts['G1']}, G2 {counts['G2']}, G3 {counts['G3']}).",
        f"- Sector coverage: **{coverage.get('with_sector', 0)} / {len(rows)}**; detailed-industry coverage: **{coverage.get('with_industry', 0)} / {len(rows)}**.",
        f"- Stocks with ≥3 profitability features: **{coverage.get('profit_3plus', 0)} / {len(rows)}**.",
        f"- Profit clustering fit count: **{clustering.get('fit_count', 0)}**; selected k = **{clustering.get('k', 0)}**, silhouette = **{float(clustering.get('silhouette', 0)):.3f}**.",
        "- G1/G2/G3 and drawdowns remain Massive-derived. Sector/industry and profitability fields are a Yahoo Finance snapshot via yfinance.",
        "- Financial Services and Real Estate are separated from operating-company profitability clustering because their statement economics are not directly comparable.",
        "",
        "## Sector commonality",
        "",
        "| Sector | N | G1 | G2 | G3 | Median drawdown | G1 lift | G2 lift | G3 lift |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in sectors:
        lines.append(
            f"| {item['label']} | {item['count']} | {item['g1_count']} | {item['g2_count']} | {item['g3_count']} | {item['median_drawdown_pct']:.1f}% | {item['g1_lift']:.2f}x | {item['g2_lift']:.2f}x | {item['g3_lift']:.2f}x |"
        )

    lines.extend(["", "### Sector signals", ""])
    for group in (1, 2, 3):
        signals = _signals(sectors, group, 10, 1.20)
        text = ", ".join(
            f"{item['label']} ({item[f'g{group}_lift']:.2f}x, N={item['count']})"
            for item in signals[:6]
        ) or "none"
        lines.append(f"- **G{group} over-represented:** {text}")

    lines.extend([
        "",
        "## Detailed-industry commonality",
        "",
        "Only detailed industries with at least 5 matched stocks are shown.",
        "",
        "| Industry | N | G1 | G2 | G3 | Median drawdown | G1 lift | G3 lift |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for item in [entry for entry in industries if int(entry["count"]) >= 5][:40]:
        lines.append(
            f"| {item['label']} | {item['count']} | {item['g1_count']} | {item['g2_count']} | {item['g3_count']} | {item['median_drawdown_pct']:.1f}% | {item['g1_lift']:.2f}x | {item['g3_lift']:.2f}x |"
        )

    lines.extend(["", "## Profitability modules", ""])
    lines.append(
        "Modules are learned cross-sectionally from revenue growth, operating margin, net margin, FCF margin and earnings growth. Labels describe cluster medians rather than hand-classifying individual stocks."
    )
    lines.extend([
        "",
        "| Module | N | Rev growth | Op margin | Net margin | FCF margin | Earnings growth | Median drawdown | G1/G2/G3 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    module_meta = {module["module"]: module for module in clustering.get("modules", [])}
    for item in modules:
        meta = module_meta.get(item["label"], {})
        lines.append(
            f"| {item['label']} | {item['count']} | {_fmt(meta.get('revenue_growth_yoy_pct'))}% | {_fmt(meta.get('operating_margin_pct'))}% | {_fmt(meta.get('net_margin_pct'))}% | {_fmt(meta.get('fcf_margin_pct'))}% | {_fmt(meta.get('earnings_growth_yoy_pct'))}% | {item['median_drawdown_pct']:.1f}% | {item['g1_count']}/{item['g2_count']}/{item['g3_count']} |"
        )

    lines.extend(["", "### Profit-module signals", ""])
    for group in (1, 2, 3):
        signals = _signals(modules, group, 8, 1.20)
        text = ", ".join(
            f"{item['label']} ({item[f'g{group}_lift']:.2f}x, N={item['count']})"
            for item in signals[:6]
        ) or "none"
        lines.append(f"- **G{group} over-represented:** {text}")

    lines.extend([
        "",
        "## Detailed industry × profitability module",
        "",
        "| Industry | Profit module | N | G1 | G2 | G3 | Median drawdown |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for item in combos[:40]:
        lines.append(
            f"| {item['industry']} | {item['profit_module']} | {item['count']} | {item['g1']} | {item['g2']} | {item['g3']} | {item['median_drawdown_pct']:.1f}% |"
        )

    lines.extend([
        "",
        "## Foundation for industry-neutral company analysis",
        "",
        "Each stock now carries these control variables:",
        "",
        "- `sector_adjusted_drawdown_residual_pp`: stock drawdown minus broad-sector median.",
        "- `industry_adjusted_drawdown_residual_pp`: stock drawdown minus detailed-industry median when that industry has ≥5 members; otherwise sector median.",
        "- `profit_module_adjusted_drawdown_residual_pp`: stock drawdown minus companies with a similar profitability structure.",
        "- `peer_adjusted_drawdown_residual_pp`: stock drawdown minus detailed-industry × profitability-module peers when the cell has ≥5 members; otherwise the industry/sector benchmark.",
        "",
        "A negative residual means the stock held up better than its comparison group; a positive residual means it fell more than its peers. These are price-behaviour residuals, not intrinsic-value or rebound-probability scores.",
    ])
    return "\n".join(lines) + "\n"


def write_outputs(result: Mapping[str, Any], output_dir: Path, fundamentals_out: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "industry_profit_analysis.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "industry_profit_report.md").write_text(render_report(result), encoding="utf-8")

    stocks = result["stocks"]
    fieldnames: list[str] = []
    for row in stocks:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with (output_dir / "enriched_stocks.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in stocks:
            writer.writerow({key: row.get(key) for key in fieldnames})

    normalized = {
        "source": "Yahoo Finance via yfinance",
        "stock_count": len(stocks),
        "fields": [
            "ticker", "sector", "industry", "country", "market_cap", "total_revenue",
            "free_cashflow", "revenue_growth_yoy_pct", "gross_margin_pct",
            "operating_margin_pct", "net_margin_pct", "fcf_margin_pct",
            "earnings_growth_yoy_pct", "return_on_equity_pct", "profit_feature_count",
            "fundamentals_status", "profit_module",
        ],
        "stocks": [
            {
                key: row.get(key)
                for key in (
                    "ticker", "sector", "industry", "country", "market_cap", "total_revenue",
                    "free_cashflow", "revenue_growth_yoy_pct", "gross_margin_pct",
                    "operating_margin_pct", "net_margin_pct", "fcf_margin_pct",
                    "earnings_growth_yoy_pct", "return_on_equity_pct", "profit_feature_count",
                    "fundamentals_status", "profit_module",
                )
            }
            for row in stocks
        ],
    }
    fundamentals_out.parent.mkdir(parents=True, exist_ok=True)
    fundamentals_out.write_text(
        json.dumps(normalized, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("analysis/apr_may_peak/analysis.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("analysis/apr_may_peak"))
    parser.add_argument("--fundamentals-out", type=Path, default=Path("fundamentals/yahoo/apr_may_peak_2026-08-20.json"))
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/yfinance_info"))
    parser.add_argument("--workers", type=int, default=4)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        client = YahooSnapshotClient(args.cache_dir)
        result = enrich(args.input, client, args.workers)
        write_outputs(result, args.output_dir, args.fundamentals_out)
    except (DataError, OSError, ValueError) as exc:
        print(f"error: {exc}")
        return 2
    print(
        json.dumps(
            {
                "stock_count": result["stock_count"],
                "coverage": result["coverage"],
                "profit_k": result["profit_clustering"].get("k"),
                "profit_silhouette": result["profit_clustering"].get("silhouette"),
                "sectors": len(result["sector_summary"]),
                "industries": len(result["industry_summary"]),
                "profit_modules": len(result["profit_module_summary"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
