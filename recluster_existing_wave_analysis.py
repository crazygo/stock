#!/usr/bin/env python3
"""Recluster an existing Apr-May wave analysis using every matched stock.

This avoids re-downloading market data: the committed analysis JSON already
contains every per-stock drawdown and January reference metric required to
recompute group boundaries and summaries.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence

from analyze_apr_may_peak import group_index, learn_drawdown_groups, render_markdown


def recluster(payload: dict[str, Any]) -> dict[str, Any]:
    records = list(payload.get("stocks") or [])
    if not records:
        raise ValueError("analysis JSON contains no stocks")

    clustering = learn_drawdown_groups(
        [float(row["drawdown_from_peak_pct"]) / 100.0 for row in records]
    )
    breaks = list(clustering["breaks"])
    k = int(clustering["k"])

    for row in records:
        idx = group_index(float(row["drawdown_from_peak_pct"]) / 100.0, breaks)
        row["pullback_group"] = idx + 1
        row["pullback_group_rank"] = f"G{idx + 1}/{k}"

    records.sort(
        key=lambda row: (
            row["pullback_group"],
            -float(row.get("avg_dollar_volume_20d") or 0.0),
            row["ticker"],
        )
    )

    groups: list[dict[str, Any]] = []
    lower = 0.0
    for idx in range(k):
        upper = breaks[idx] if idx < len(breaks) else None
        members = [row for row in records if row["pullback_group"] == idx + 1]
        liquid_members = [row for row in members if row.get("liquid")]
        dd = [float(row["drawdown_from_peak_pct"]) for row in members]
        jan = [
            float(row["current_vs_jan_pct"])
            for row in members
            if row.get("current_vs_jan_pct") is not None
        ]
        retraced = [
            float(row["jan_to_peak_rally_retraced_pct"])
            for row in members
            if row.get("jan_to_peak_rally_retraced_pct") is not None
        ]
        groups.append(
            {
                "group": idx + 1,
                "lower_drawdown_pct_exclusive": round(lower * 100.0, 2) if idx else None,
                "upper_drawdown_pct_inclusive": round(upper * 100.0, 2)
                if upper is not None
                else None,
                "count": len(members),
                "liquid_count": len(liquid_members),
                "median_drawdown_pct": round(statistics.median(dd), 2) if dd else None,
                "median_current_vs_jan_pct": round(statistics.median(jan), 2) if jan else None,
                "median_jan_rally_retraced_pct": round(statistics.median(retraced), 2)
                if retraced
                else None,
                "top_liquid_tickers": [row["ticker"] for row in liquid_members[:20]],
            }
        )
        if upper is not None:
            lower = upper

    method = payload.setdefault("methodology", {})
    method["group_learning"] = (
        "1D k-means fit on all matched stocks; k=3..5 chosen by silhouette minus "
        "small complexity penalty; p95 winsorization for fit only"
    )
    payload["clustering_fit_population"] = "all_matches"
    payload["clustering"] = {
        **clustering,
        "centers_pct": [round(x * 100.0, 2) for x in clustering["centers"]],
        "breaks_pct": [round(x * 100.0, 2) for x in clustering["breaks"]],
    }
    payload["groups"] = groups
    payload["stocks"] = records
    return payload


def write_outputs(analysis_dir: Path, payload: Mapping[str, Any]) -> None:
    (analysis_dir / "analysis.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    stocks = list(payload["stocks"])
    with (analysis_dir / "stocks.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(stocks[0].keys()))
        writer.writeheader()
        writer.writerows(stocks)
    (analysis_dir / "report.md").write_text(render_markdown(payload), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, default=Path("analysis/apr_may_peak"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    path = args.analysis_dir / "analysis.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit("analysis JSON must be an object")
    result = recluster(payload)
    write_outputs(args.analysis_dir, result)
    print(
        json.dumps(
            {
                "fit_population": result["clustering_fit_population"],
                "k": result["clustering"]["k"],
                "silhouette": result["clustering"]["silhouette"],
                "centers_pct": result["clustering"]["centers_pct"],
                "breaks_pct": result["clustering"]["breaks_pct"],
                "groups": result["groups"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
