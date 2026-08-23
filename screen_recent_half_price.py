#!/usr/bin/env python3
"""Find stocks whose recent 7-session prices are less than half of at least 5 prior closes.

Exact default rule:
- Use adjusted daily close from the durable grouped EOD archive.
- Let R be the most recent 7 complete trading sessions.
- Let recent_max = max(close over R).
- Search prior sessions from 2026-03-01 up to the session before R.
- Match when at least 5 prior trading days have close > 2 * recent_max.

The inequality against recent_max makes every one of those prior days more than
2x every close in the recent seven-session window.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import statistics
from dataclasses import dataclass, asdict
from datetime import date
from pathlib import Path
from typing import Any, Mapping, Sequence

from screener import MassiveClient, ScreenerError, load_env_file


@dataclass(frozen=True)
class Match:
    ticker: str
    name: str
    latest_session: str
    latest_close: float
    recent7_min: float
    recent7_max: float
    recent7_range_pct: float
    qualifying_days: int
    fifth_highest_prior_close: float
    fifth_highest_to_recent_max_ratio: float
    prior_peak_close: float
    prior_peak_session: str
    latest_vs_prior_peak_pct: float
    avg_dollar_volume_20d: float
    liquid: bool


def load_manifest(archive_dir: Path) -> dict[str, Any]:
    path = archive_dir / "manifest.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError(f"Missing manifest: {path}") from exc
    archives = payload.get("archives")
    if not isinstance(archives, dict):
        raise RuntimeError(f"Invalid manifest archives map: {path}")
    return payload


def load_day(archive_dir: Path, session: str, relative_path: str) -> dict[str, dict[str, float]]:
    path = archive_dir / relative_path
    try:
        raw = gzip.decompress(path.read_bytes())
        payload = json.loads(raw.decode("utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError(f"Missing archive: {path}") from exc
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise RuntimeError(f"Invalid archive: {path}: {exc}") from exc
    rows: dict[str, dict[str, float]] = {}
    for item in payload.get("results") or []:
        ticker = item.get("T")
        close = item.get("c")
        volume = item.get("v")
        if isinstance(ticker, str) and isinstance(close, (int, float)) and close > 0:
            rows[ticker] = {"close": float(close), "volume": float(volume or 0.0)}
    return rows


def compute_matches(
    *,
    archive_dir: Path,
    universe: Mapping[str, str],
    start: date,
    recent_sessions: int = 7,
    min_prior_days: int = 5,
    multiplier: float = 2.0,
    liquid_min_price: float = 5.0,
    liquid_min_dollar_volume: float = 20_000_000.0,
) -> tuple[list[Match], dict[str, Any]]:
    manifest = load_manifest(archive_dir)
    archives: Mapping[str, Any] = manifest["archives"]
    sessions = sorted(
        s for s in archives
        if date.fromisoformat(s) >= start
    )
    if len(sessions) < recent_sessions + 1:
        raise RuntimeError("Not enough archived sessions for the requested screen")

    recent = sessions[-recent_sessions:]
    prior = sessions[:-recent_sessions]
    latest = recent[-1]
    # Need up to 20 sessions for dollar-volume context.
    context_sessions = sessions[-20:]
    required_sessions = sorted(set(prior + recent + context_sessions))

    daily: dict[str, dict[str, dict[str, float]]] = {}
    for session in required_sessions:
        entry = archives[session]
        relative_path = entry.get("path")
        if not isinstance(relative_path, str):
            raise RuntimeError(f"Missing archive path for {session}")
        daily[session] = load_day(archive_dir, session, relative_path)

    matches: list[Match] = []
    complete_recent = 0
    for ticker, name in universe.items():
        recent_rows = [daily[s].get(ticker) for s in recent]
        if any(row is None for row in recent_rows):
            continue
        complete_recent += 1
        recent_closes = [float(row["close"]) for row in recent_rows if row is not None]
        recent_min = min(recent_closes)
        recent_max = max(recent_closes)
        threshold = multiplier * recent_max

        prior_values: list[tuple[str, float]] = []
        for session in prior:
            row = daily[session].get(ticker)
            if row is not None:
                prior_values.append((session, float(row["close"])))
        qualifying = [(s, c) for s, c in prior_values if c > threshold]
        if len(qualifying) < min_prior_days:
            continue

        ordered_prior = sorted((c for _, c in prior_values), reverse=True)
        if len(ordered_prior) < min_prior_days:
            continue
        fifth = ordered_prior[min_prior_days - 1]
        peak_session, peak_close = max(prior_values, key=lambda item: item[1])
        latest_close = recent_closes[-1]

        dv_values: list[float] = []
        for session in context_sessions:
            row = daily[session].get(ticker)
            if row is not None:
                dv_values.append(float(row["close"]) * float(row["volume"]))
        avg_dv20 = statistics.fmean(dv_values) if dv_values else 0.0
        liquid = latest_close >= liquid_min_price and avg_dv20 >= liquid_min_dollar_volume
        matches.append(
            Match(
                ticker=ticker,
                name=name,
                latest_session=latest,
                latest_close=round(latest_close, 4),
                recent7_min=round(recent_min, 4),
                recent7_max=round(recent_max, 4),
                recent7_range_pct=round((recent_max / recent_min - 1.0) * 100.0, 2),
                qualifying_days=len(qualifying),
                fifth_highest_prior_close=round(fifth, 4),
                fifth_highest_to_recent_max_ratio=round(fifth / recent_max, 3),
                prior_peak_close=round(peak_close, 4),
                prior_peak_session=peak_session,
                latest_vs_prior_peak_pct=round((latest_close / peak_close - 1.0) * 100.0, 2),
                avg_dollar_volume_20d=round(avg_dv20, 0),
                liquid=liquid,
            )
        )

    matches.sort(
        key=lambda row: (
            not row.liquid,
            -row.fifth_highest_to_recent_max_ratio,
            -row.qualifying_days,
            row.ticker,
        )
    )
    meta = {
        "start": start.isoformat(),
        "latest_session": latest,
        "recent_sessions": recent,
        "prior_session_count": len(prior),
        "active_common_stocks": len(universe),
        "complete_recent7": complete_recent,
        "match_count": len(matches),
        "liquid_match_count": sum(row.liquid for row in matches),
        "rule": {
            "min_prior_days": min_prior_days,
            "multiplier": multiplier,
            "comparison": "prior_close > multiplier * max(recent_7_closes)",
            "prior_window": f"{start.isoformat()} through session before recent-7 window",
        },
        "liquid_definition": {
            "latest_close_gte": liquid_min_price,
            "avg_dollar_volume_20d_gte": liquid_min_dollar_volume,
        },
    }
    return matches, meta


def render_markdown(matches: Sequence[Match], meta: Mapping[str, Any]) -> str:
    recent = ", ".join(meta["recent_sessions"])
    lines = [
        "# Recent 7-session Half-Price Screen",
        "",
        "## Exact rule",
        "",
        f"- Archived data through: `{meta['latest_session']}`.",
        f"- Recent 7 sessions: {recent}.",
        f"- Prior search window starts: `{meta['start']}`.",
        "- A stock matches when **at least 5 prior adjusted closes are > 2 × the highest adjusted close in the recent 7 sessions**.",
        "- Therefore each of those 5 prior closes is more than double every close in the recent-7 window.",
        f"- Liquid view: latest price ≥ ${meta['liquid_definition']['latest_close_gte']:.0f} and 20-session average dollar volume ≥ ${meta['liquid_definition']['avg_dollar_volume_20d_gte']/1_000_000:.0f}M.",
        "",
        "## Funnel",
        "",
        f"- Active U.S. common stocks: {meta['active_common_stocks']:,}",
        f"- Complete recent-7 history: {meta['complete_recent7']:,}",
        f"- All mathematical matches: **{meta['match_count']:,}**",
        f"- Liquid matches: **{meta['liquid_match_count']:,}**",
        "",
        "## Liquid matches",
        "",
        "| # | Ticker | Name | Latest | Recent-7 min–max | Days >2× recent max | 5th-highest / recent max | Prior peak | Peak date | Latest vs peak | 20d $ volume |",
        "|---:|---|---|---:|---:|---:|---:|---:|---|---:|---:|",
    ]
    liquid = [row for row in matches if row.liquid]
    for i, row in enumerate(liquid, 1):
        name = row.name.replace("|", "\\|")
        lines.append(
            f"| {i} | **{row.ticker}** | {name} | ${row.latest_close:.2f} | "
            f"${row.recent7_min:.2f}–${row.recent7_max:.2f} | {row.qualifying_days} | "
            f"{row.fifth_highest_to_recent_max_ratio:.2f}× | ${row.prior_peak_close:.2f} | "
            f"{row.prior_peak_session} | {row.latest_vs_prior_peak_pct:.1f}% | "
            f"${row.avg_dollar_volume_20d/1_000_000:.1f}M |"
        )
    if not liquid:
        lines.append("| — | — | No liquid matches | — | — | — | — | — | — | — | — |")

    lines.extend([
        "",
        "## All matches",
        "",
        "Full mathematical match set is in `matches.csv` / `analysis.json`, including illiquid and sub-$5 names.",
        "",
    ])
    return "\n".join(lines)


def write_outputs(output_dir: Path, matches: Sequence[Match], meta: Mapping[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [asdict(row) for row in matches]
    payload = {"metadata": dict(meta), "matches": rows}
    (output_dir / "analysis.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "report.md").write_text(render_markdown(matches, meta), encoding="utf-8")
    fields = list(asdict(matches[0]).keys()) if matches else [field.name for field in Match.__dataclass_fields__.values()]
    with (output_dir / "matches.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-dir", type=Path, default=Path("market_data/us"))
    parser.add_argument("--output-dir", type=Path, default=Path("analysis/recent_half_price"))
    parser.add_argument("--start", default="2026-03-01")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--api-key-env", default="MASSIVE_API_KEY")
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/massive"))
    parser.add_argument("--request-delay", type=float, default=12.2)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    load_env_file(args.env_file)
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit(f"Set {args.api_key_env} before running")
    manifest = load_manifest(args.archive_dir)
    latest = max(manifest["archives"])
    client = MassiveClient(
        api_key=api_key,
        cache_dir=args.cache_dir,
        request_delay=args.request_delay,
    )
    universe = client.fetch_universe(date.fromisoformat(latest), ("CS",))
    matches, meta = compute_matches(
        archive_dir=args.archive_dir,
        universe=universe,
        start=date.fromisoformat(args.start),
    )
    write_outputs(args.output_dir, matches, meta)
    print(json.dumps({"metadata": meta, "top_liquid": [asdict(x) for x in matches if x.liquid][:20]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
