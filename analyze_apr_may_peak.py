#!/usr/bin/env python3
"""Classify stocks whose weekly adjusted-close peak occurred in Apr-May 2026.

Definition:
- history starts 2025-08-01;
- weekly value is the adjusted close of the last available U.S. trading session
  in each ISO week;
- a stock qualifies when its maximum weekly close from the history start
  through the latest archived session occurs between 2026-04-01 and 2026-05-31;
- only currently active U.S. common stocks (Massive ticker type CS) with August
  2025 history and >=80% weekly coverage are included.

The primary pullback metric is current drawdown from that Apr-May peak. Group
thresholds are learned from the observed drawdown distribution using
one-dimensional k-means; k=3..5 is selected by silhouette score with a small
complexity penalty. A January-2026 reference is reported separately so a group
can be interpreted as "back to January" without forcing that assumption into
the clustering algorithm.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import statistics
from collections import defaultdict, deque
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from screener import MassiveClient, ScreenerError, load_env_file


def _percentile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile requires values")
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    w = pos - lo
    return ordered[lo] * (1 - w) + ordered[hi] * w


def _median(values: Iterable[float]) -> float | None:
    data = list(values)
    return statistics.median(data) if data else None


def _kmeans_1d(values: Sequence[float], k: int) -> tuple[list[float], list[int]]:
    if len(values) < k:
        raise ValueError("not enough values for k")
    centers = [_percentile(values, (i + 0.5) / k) for i in range(k)]
    assignments = [0] * len(values)
    for _ in range(100):
        assignments = [min(range(k), key=lambda j: abs(x - centers[j])) for x in values]
        new_centers: list[float] = []
        for j in range(k):
            cluster = [x for x, a in zip(values, assignments) if a == j]
            new_centers.append(
                statistics.fmean(cluster) if cluster else _percentile(values, (j + 0.5) / k)
            )
        if max(abs(a - b) for a, b in zip(centers, new_centers)) < 1e-8:
            centers = new_centers
            break
        centers = new_centers

    order = sorted(range(k), key=lambda j: centers[j])
    remap = {old: new for new, old in enumerate(order)}
    sorted_centers = [centers[j] for j in order]
    sorted_assignments = [remap[a] for a in assignments]
    return sorted_centers, sorted_assignments


def _silhouette_1d(values: Sequence[float], assignments: Sequence[int], k: int) -> float:
    clusters = [[values[i] for i, a in enumerate(assignments) if a == j] for j in range(k)]
    scores: list[float] = []
    for x, own in zip(values, assignments):
        own_values = clusters[own]
        if len(own_values) <= 1:
            scores.append(0.0)
            continue
        a = statistics.fmean(abs(x - y) for y in own_values if y != x)
        other_means = [
            statistics.fmean(abs(x - y) for y in cluster)
            for j, cluster in enumerate(clusters)
            if j != own and cluster
        ]
        b = min(other_means)
        denom = max(a, b)
        scores.append((b - a) / denom if denom else 0.0)
    return statistics.fmean(scores) if scores else 0.0


def learn_drawdown_groups(values: Sequence[float]) -> dict[str, Any]:
    """Learn stable drawdown breaks; values are positive fractions (0.25=25%)."""
    if len(values) < 9:
        center = statistics.fmean(values) if values else 0.0
        return {"k": 1, "centers": [center], "breaks": [], "silhouette": 0.0}

    # Extreme micro-cap collapses should belong to the deepest group without
    # dragging its center far away from the rest of the distribution.
    cap = _percentile(values, 0.95)
    fit_values = [min(x, cap) for x in values]
    best: dict[str, Any] | None = None
    min_group = max(3, math.ceil(len(values) * 0.025))
    for k in range(3, min(5, len(values)) + 1):
        centers, assignments = _kmeans_1d(fit_values, k)
        counts = [assignments.count(i) for i in range(k)]
        if min(counts) < min_group:
            continue
        silhouette = _silhouette_1d(fit_values, assignments, k)
        score = silhouette - 0.015 * (k - 3)
        candidate = {
            "k": k,
            "centers": centers,
            "breaks": [(centers[i] + centers[i + 1]) / 2 for i in range(k - 1)],
            "silhouette": silhouette,
            "score": score,
            "fit_cap_p95": cap,
            "fit_group_counts": counts,
        }
        if best is None or candidate["score"] > best["score"]:
            best = candidate
    if best is None:
        centers, assignments = _kmeans_1d(fit_values, 3)
        best = {
            "k": 3,
            "centers": centers,
            "breaks": [(centers[i] + centers[i + 1]) / 2 for i in range(2)],
            "silhouette": _silhouette_1d(fit_values, assignments, 3),
            "score": 0.0,
            "fit_cap_p95": cap,
            "fit_group_counts": [assignments.count(i) for i in range(3)],
        }
    return best


def group_index(value: float, breaks: Sequence[float]) -> int:
    for i, boundary in enumerate(breaks):
        if value <= boundary:
            return i
    return len(breaks)


def _load_archive(path: Path) -> Mapping[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Invalid archive payload: {path}")
    return payload


def _manifest_sessions(archive_dir: Path, start: date, end: date | None) -> list[tuple[date, Path]]:
    manifest = json.loads((archive_dir / "manifest.json").read_text(encoding="utf-8"))
    archives = manifest.get("archives")
    if not isinstance(archives, Mapping):
        raise ValueError("archive manifest has no archives map")
    rows: list[tuple[date, Path]] = []
    for raw_date, meta in archives.items():
        session = date.fromisoformat(raw_date)
        if session < start or (end is not None and session > end):
            continue
        if not isinstance(meta, Mapping) or not isinstance(meta.get("path"), str):
            continue
        rows.append((session, archive_dir / str(meta["path"])))
    rows.sort()
    return rows


def analyze(
    *,
    archive_dir: Path,
    universe: Mapping[str, str],
    start: date,
    peak_start: date,
    peak_end: date,
    end: date | None,
    min_weekly_coverage: float,
    liquid_dollar_volume: float,
) -> dict[str, Any]:
    sessions = _manifest_sessions(archive_dir, start, end)
    if not sessions:
        raise ValueError("no archive sessions in requested range")
    latest_session = sessions[-1][0]

    weekly: dict[str, dict[tuple[int, int], tuple[date, float]]] = defaultdict(dict)
    recent_dollar_volume: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=20))
    archive_weeks: set[tuple[int, int]] = set()

    for session, path in sessions:
        iso = session.isocalendar()
        week_key = (iso.year, iso.week)
        archive_weeks.add(week_key)
        payload = _load_archive(path)
        for row in payload.get("results") or []:
            if not isinstance(row, Mapping):
                continue
            ticker = row.get("T")
            if ticker not in universe:
                continue
            try:
                close = float(row["c"])
                volume = float(row.get("v", 0.0))
            except (KeyError, TypeError, ValueError):
                continue
            if close <= 0:
                continue
            weekly[ticker][week_key] = (session, close)
            recent_dollar_volume[ticker].append(close * max(volume, 0.0))

    expected_weeks = len(archive_weeks)
    records: list[dict[str, Any]] = []
    eligibility = {
        "active_common_stocks": len(universe),
        "with_any_history": 0,
        "with_august_2025_history": 0,
        "coverage_pass": 0,
        "apr_may_global_peak": 0,
    }

    for ticker, name in universe.items():
        points = sorted(weekly.get(ticker, {}).values(), key=lambda item: item[0])
        if not points:
            continue
        eligibility["with_any_history"] += 1
        if points[0][0] > date(2025, 8, 31):
            continue
        eligibility["with_august_2025_history"] += 1
        coverage = len(points) / expected_weeks if expected_weeks else 0.0
        if coverage < min_weekly_coverage:
            continue
        eligibility["coverage_pass"] += 1

        peak_date, peak_close = max(points, key=lambda item: item[1])
        if not (peak_start <= peak_date <= peak_end):
            continue
        eligibility["apr_may_global_peak"] += 1
        latest_date, latest_close = points[-1]
        january = [
            close for session, close in points if date(2026, 1, 1) <= session <= date(2026, 1, 31)
        ]
        jan_median = _median(january)
        first_close = points[0][1]
        drawdown = max(0.0, 1.0 - latest_close / peak_close)
        current_vs_jan = latest_close / jan_median - 1.0 if jan_median and jan_median > 0 else None
        peak_vs_jan = peak_close / jan_median - 1.0 if jan_median and jan_median > 0 else None
        retracement = None
        if jan_median and peak_close > jan_median:
            retracement = (peak_close - latest_close) / (peak_close - jan_median)
        avg_dv20 = statistics.fmean(recent_dollar_volume[ticker]) if recent_dollar_volume[ticker] else 0.0
        records.append(
            {
                "ticker": ticker,
                "name": name,
                "first_week": points[0][0].isoformat(),
                "latest_week": latest_date.isoformat(),
                "weekly_coverage_pct": round(coverage * 100.0, 1),
                "peak_week": peak_date.isoformat(),
                "peak_close": round(peak_close, 4),
                "latest_close": round(latest_close, 4),
                "drawdown_from_peak_pct": round(drawdown * 100.0, 2),
                "jan_median_close": round(jan_median, 4) if jan_median is not None else None,
                "current_vs_jan_pct": round(current_vs_jan * 100.0, 2) if current_vs_jan is not None else None,
                "peak_vs_jan_pct": round(peak_vs_jan * 100.0, 2) if peak_vs_jan is not None else None,
                "jan_to_peak_rally_retraced_pct": round(retracement * 100.0, 2) if retracement is not None else None,
                "gain_start_to_peak_pct": round((peak_close / first_close - 1.0) * 100.0, 2),
                "avg_dollar_volume_20d": round(avg_dv20, 0),
                "liquid": bool(avg_dv20 >= liquid_dollar_volume and latest_close >= 5.0),
            }
        )

    # The user-defined class is the full matched population. Liquidity is a
    # presentation/tradability lens only, so it must not determine the natural
    # drawdown boundaries.
    fit_records = records
    clustering = learn_drawdown_groups(
        [float(row["drawdown_from_peak_pct"]) / 100.0 for row in fit_records]
    )
    breaks = list(clustering["breaks"])
    k = int(clustering["k"])
    for row in records:
        idx = group_index(float(row["drawdown_from_peak_pct"]) / 100.0, breaks)
        row["pullback_group"] = idx + 1
        row["pullback_group_rank"] = f"G{idx + 1}/{k}"

    records.sort(key=lambda row: (row["pullback_group"], -row["avg_dollar_volume_20d"], row["ticker"]))
    group_summaries: list[dict[str, Any]] = []
    lower = 0.0
    for idx in range(k):
        upper = breaks[idx] if idx < len(breaks) else None
        members = [row for row in records if row["pullback_group"] == idx + 1]
        liquid_members = [row for row in members if row["liquid"]]
        dd = [float(row["drawdown_from_peak_pct"]) for row in members]
        jan = [float(row["current_vs_jan_pct"]) for row in members if row["current_vs_jan_pct"] is not None]
        retraced = [
            float(row["jan_to_peak_rally_retraced_pct"])
            for row in members
            if row["jan_to_peak_rally_retraced_pct"] is not None
        ]
        group_summaries.append(
            {
                "group": idx + 1,
                "lower_drawdown_pct_exclusive": round(lower * 100.0, 2) if idx else None,
                "upper_drawdown_pct_inclusive": round(upper * 100.0, 2) if upper is not None else None,
                "count": len(members),
                "liquid_count": len(liquid_members),
                "median_drawdown_pct": round(statistics.median(dd), 2) if dd else None,
                "median_current_vs_jan_pct": round(statistics.median(jan), 2) if jan else None,
                "median_jan_rally_retraced_pct": round(statistics.median(retraced), 2) if retraced else None,
                "top_liquid_tickers": [row["ticker"] for row in liquid_members[:20]],
            }
        )
        if upper is not None:
            lower = upper

    return {
        "methodology": {
            "history_start": start.isoformat(),
            "latest_session": latest_session.isoformat(),
            "weekly_value": "last available adjusted close in each ISO week",
            "peak_window": [peak_start.isoformat(), peak_end.isoformat()],
            "required_first_history": "on or before 2025-08-31",
            "min_weekly_coverage_pct": min_weekly_coverage * 100.0,
            "universe": "currently active U.S. common stocks (Massive type CS)",
            "group_metric": "current drawdown from Apr-May global weekly-close peak",
            "group_learning": "1D k-means fit on all matched stocks; k=3..5 chosen by silhouette minus small complexity penalty; p95 winsorization for fit only",
            "january_reference": "median weekly adjusted close during January 2026",
            "liquid_definition": f"latest close >= $5 and 20-session average dollar volume >= ${liquid_dollar_volume:,.0f}",
        },
        "counts": eligibility,
        "expected_weeks": expected_weeks,
        "match_count": len(records),
        "liquid_match_count": sum(bool(row["liquid"]) for row in records),
        "clustering_fit_population": "all_matches",
        "clustering": {
            **clustering,
            "centers_pct": [round(x * 100.0, 2) for x in clustering["centers"]],
            "breaks_pct": [round(x * 100.0, 2) for x in clustering["breaks"]],
        },
        "groups": group_summaries,
        "stocks": records,
    }


def render_markdown(result: Mapping[str, Any]) -> str:
    method = result["methodology"]
    lines = [
        "# Apr–May 2026 Weekly-Peak Stock Wave",
        "",
        "## Definition",
        "",
        f"- History: `{method['history_start']}` → `{method['latest_session']}`.",
        f"- Weekly point: {method['weekly_value']}.",
        f"- Peak window: `{method['peak_window'][0]}` → `{method['peak_window'][1]}`.",
        f"- Universe: {method['universe']}; first history must exist {method['required_first_history']} and weekly coverage ≥ {method['min_weekly_coverage_pct']:.0f}%.",
        f"- Pullback grouping: {method['group_learning']}.",
        f"- January reference: {method['january_reference']}.",
        "",
        "## Funnel",
        "",
    ]
    for key, value in result["counts"].items():
        lines.append(f"- `{key}`: {value:,}")
    lines.extend(
        [
            f"- **Matched class:** {result['match_count']:,}",
            f"- **Liquid matched class:** {result['liquid_match_count']:,}",
            "",
            "## Data-driven pullback breaks",
            "",
            f"Selected k = **{result['clustering']['k']}**, silhouette = **{result['clustering']['silhouette']:.3f}**.",
            f"Cluster centers: **{', '.join(f'{x:.1f}%' for x in result['clustering']['centers_pct'])}**.",
            f"Boundaries: **{', '.join(f'{x:.1f}%' for x in result['clustering']['breaks_pct']) or 'none'}**.",
            "",
            "| Group | Drawdown range | Stocks | Liquid | Median drawdown | Median vs Jan-26 | Median Jan→peak rally retraced |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for group in result["groups"]:
        lo = group["lower_drawdown_pct_exclusive"]
        hi = group["upper_drawdown_pct_inclusive"]
        if lo is None:
            span = f"≤ {hi:.1f}%" if hi is not None else "all"
        elif hi is None:
            span = f"> {lo:.1f}%"
        else:
            span = f"> {lo:.1f}% – ≤ {hi:.1f}%"
        jan = group["median_current_vs_jan_pct"]
        ret = group["median_jan_rally_retraced_pct"]
        lines.append(
            f"| G{group['group']} | {span} | {group['count']} | {group['liquid_count']} | "
            f"{group['median_drawdown_pct']:.1f}% | {jan:.1f}% | {ret:.1f}% |"
        )

    lines.extend(["", "## Group members — liquid view", ""])
    for group in result["groups"]:
        members = [
            row for row in result["stocks"] if row["pullback_group"] == group["group"] and row["liquid"]
        ]
        lines.extend(
            [
                f"### G{group['group']}",
                "",
                "| Ticker | Name | Peak week | Peak→now | Now vs Jan | Jan→peak rally retraced | 20d $ volume |",
                "|---|---|---|---:|---:|---:|---:|",
            ]
        )
        for row in members[:40]:
            safe_name = str(row["name"]).replace("|", "\\|")
            jan = row["current_vs_jan_pct"]
            retr = row["jan_to_peak_rally_retraced_pct"]
            lines.append(
                f"| **{row['ticker']}** | {safe_name} | {row['peak_week']} | "
                f"-{row['drawdown_from_peak_pct']:.1f}% | "
                f"{jan:+.1f}% | {retr:.1f}% | ${row['avg_dollar_volume_20d']/1_000_000:.1f}M |"
            )
        if len(members) > 40:
            lines.append(f"\n_+ {len(members) - 40} more liquid members in `stocks.csv`._")
        lines.append("")
    lines.extend(
        [
            "## Notes",
            "",
            "- `Peak→now` is the grouping variable. It measures how far the current weekly close sits below the Apr–May peak.",
            "- `Now vs Jan` anchors the visual observation of stocks that have fallen back to January 2026 price levels.",
            "- `Jan→peak rally retraced` = (peak − current) / (peak − Jan median). Around 100% means the whole January-to-peak rally has been given back; above 100% means current price is below the January median.",
            "- The full matched class, including less-liquid names, is in `stocks.csv` and `analysis.json`.",
            "- Current-active-CS filtering introduces survivorship bias by design: the result is intended as a current investable universe, not a historical delisting study.",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(output_dir: Path, result: Mapping[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "analysis.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    stocks = list(result["stocks"])
    if stocks:
        with (output_dir / "stocks.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(stocks[0].keys()))
            writer.writeheader()
            writer.writerows(stocks)
    (output_dir / "report.md").write_text(render_markdown(result), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-dir", type=Path, default=Path("market_data/us"))
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/massive"))
    parser.add_argument("--output-dir", type=Path, default=Path("analysis/apr_may_peak"))
    parser.add_argument("--start", default="2025-08-01")
    parser.add_argument("--end", help="Optional final archive date")
    parser.add_argument("--peak-start", default="2026-04-01")
    parser.add_argument("--peak-end", default="2026-05-31")
    parser.add_argument("--min-weekly-coverage", type=float, default=0.80)
    parser.add_argument("--liquid-dollar-volume", type=float, default=20_000_000.0)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--api-key-env", default="MASSIVE_API_KEY")
    parser.add_argument("--request-delay", type=float, default=12.2)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        start = date.fromisoformat(args.start)
        end = date.fromisoformat(args.end) if args.end else None
        peak_start = date.fromisoformat(args.peak_start)
        peak_end = date.fromisoformat(args.peak_end)
        load_env_file(args.env_file)
        api_key = os.environ.get(args.api_key_env)
        if not api_key:
            raise ScreenerError(f"Set {args.api_key_env} before analysis")
        sessions = _manifest_sessions(args.archive_dir, start, end)
        if not sessions:
            raise ValueError("no archived sessions")
        latest = sessions[-1][0]
        client = MassiveClient(
            api_key=api_key,
            cache_dir=args.cache_dir,
            request_delay=args.request_delay,
            max_retries=8,
        )
        universe = client.fetch_universe(latest, ("CS",))
        result = analyze(
            archive_dir=args.archive_dir,
            universe=universe,
            start=start,
            peak_start=peak_start,
            peak_end=peak_end,
            end=end,
            min_weekly_coverage=args.min_weekly_coverage,
            liquid_dollar_volume=args.liquid_dollar_volume,
        )
        write_outputs(args.output_dir, result)
        print(
            json.dumps(
                {
                    "match_count": result["match_count"],
                    "liquid_match_count": result["liquid_match_count"],
                    "groups": result["groups"],
                    "output_dir": str(args.output_dir),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    except (ScreenerError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
