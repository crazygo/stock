#!/usr/bin/env python3
"""Screen U.S. common stocks for a sharp selloff followed by a quiet base.

The script deliberately produces candidates, not buy recommendations. It uses
15 daily bars so that both the selloff and stabilization phases contain seven
complete close-to-close return intervals.

Data source: Massive Stocks REST API (formerly Polygon.io compatible API).
Only Python's standard library is required.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass, fields
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo


API_BASE = "https://api.massive.com"
USER_AGENT = "drop-flat-screener/0.1"


class ScreenerError(RuntimeError):
    """Raised when market data cannot be loaded or validated."""


class MarketDataNotReadyError(ScreenerError):
    """Raised when the free EOD plan has not released the requested session."""


@dataclass(frozen=True)
class ScreenConfig:
    min_price: float = 50.0
    max_price: float = 200.0
    drop_sessions: int = 7
    flat_sessions: int = 7
    min_drop_pct: float = 0.18
    max_drop_pct: float = 0.45
    max_flat_range_pct: float = 0.07
    max_flat_slope_pct: float = 0.035
    max_flat_realized_vol_pct: float = 0.022
    max_abs_flat_daily_return_pct: float = 0.035
    min_avg_dollar_volume: float = 20_000_000.0
    accepted_ticker_types: tuple[str, ...] = ("CS",)

    @property
    def required_bars(self) -> int:
        return self.drop_sessions + self.flat_sessions + 1

    @classmethod
    def from_json(cls, path: Path | None) -> "ScreenConfig":
        if path is None:
            return cls()
        with path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        allowed = {field.name for field in fields(cls)}
        unknown = sorted(set(raw) - allowed)
        if unknown:
            raise ScreenerError(f"Unknown config keys: {', '.join(unknown)}")
        if "accepted_ticker_types" in raw:
            raw["accepted_ticker_types"] = tuple(raw["accepted_ticker_types"])
        return cls(**raw)


@dataclass(frozen=True)
class Bar:
    session: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: float | None = None

    @classmethod
    def from_massive(cls, session: str, raw: Mapping[str, Any]) -> "Bar":
        return cls(
            session=session,
            open=float(raw["o"]),
            high=float(raw["h"]),
            low=float(raw["l"]),
            close=float(raw["c"]),
            volume=float(raw.get("v", 0.0)),
            vwap=float(raw["vw"]) if raw.get("vw") is not None else None,
        )


@dataclass(frozen=True)
class Candidate:
    ticker: str
    name: str
    as_of: str
    latest_price: float
    drop_pct: float
    flat_range_pct: float
    flat_slope_pct: float
    flat_realized_vol_pct: float
    max_abs_flat_daily_return_pct: float
    volatility_contraction_ratio: float | None
    volume_contraction_ratio: float | None
    avg_dollar_volume: float
    latest_close_position: float
    shape_score: float
    bear_flag_risk: str
    flags: str


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator <= 0:
        return None
    return numerator / denominator


def _simple_returns(values: Sequence[float]) -> list[float]:
    return [values[index] / values[index - 1] - 1.0 for index in range(1, len(values))]


def _realized_vol(returns: Sequence[float]) -> float:
    return statistics.pstdev(returns) if len(returns) >= 2 else 0.0


def _annualized_linear_move(values: Sequence[float]) -> float:
    """Return fitted total log-price move across the supplied observations."""
    if len(values) < 2 or any(value <= 0 for value in values):
        return 0.0
    logs = [math.log(value) for value in values]
    x_mean = (len(logs) - 1) / 2.0
    y_mean = statistics.fmean(logs)
    denominator = sum((index - x_mean) ** 2 for index in range(len(logs)))
    if denominator == 0:
        return 0.0
    slope = sum(
        (index - x_mean) * (value - y_mean) for index, value in enumerate(logs)
    ) / denominator
    return math.exp(slope * (len(logs) - 1)) - 1.0


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _shape_score(
    config: ScreenConfig,
    *,
    drop_magnitude: float,
    flat_range: float,
    flat_slope: float,
    flat_vol: float,
    max_abs_return: float,
    vol_contraction: float | None,
    volume_contraction: float | None,
) -> float:
    """A deterministic ranking score; it is not a rebound probability."""
    tightness = 25.0 * _clamp(1.0 - flat_range / config.max_flat_range_pct)
    levelness = 20.0 * _clamp(1.0 - abs(flat_slope) / config.max_flat_slope_pct)
    quietness = 15.0 * _clamp(1.0 - flat_vol / config.max_flat_realized_vol_pct)
    daily_control = 10.0 * _clamp(
        1.0 - max_abs_return / config.max_abs_flat_daily_return_pct
    )

    # Prefer a meaningful reset without rewarding catastrophic collapses.
    drop_center = 0.24
    drop_half_width = max(drop_center - config.min_drop_pct, config.max_drop_pct - drop_center)
    reset_quality = 10.0 * _clamp(1.0 - abs(drop_magnitude - drop_center) / drop_half_width)

    vol_score = 10.0 * (
        _clamp((1.0 - vol_contraction) / 0.75) if vol_contraction is not None else 0.0
    )
    volume_score = 10.0 * (
        _clamp((1.0 - volume_contraction) / 0.60)
        if volume_contraction is not None
        else 0.0
    )
    return round(
        tightness
        + levelness
        + quietness
        + daily_control
        + reset_quality
        + vol_score
        + volume_score,
        2,
    )


def evaluate_symbol(
    ticker: str,
    name: str,
    bars: Sequence[Bar],
    config: ScreenConfig,
) -> Candidate | None:
    """Evaluate one symbol using exactly the most recent required bars."""
    if len(bars) < config.required_bars:
        return None
    recent = list(bars[-config.required_bars :])
    if any(bar.close <= 0 or bar.high <= 0 or bar.low <= 0 for bar in recent):
        return None

    drop_end_index = config.drop_sessions
    drop_bars = recent[: drop_end_index + 1]
    flat_bars = recent[drop_end_index + 1 :]
    anchor = drop_bars[-1]
    latest = recent[-1]

    if not (config.min_price <= latest.close <= config.max_price):
        return None

    drop_return = anchor.close / drop_bars[0].close - 1.0
    drop_magnitude = -drop_return
    if not (config.min_drop_pct <= drop_magnitude <= config.max_drop_pct):
        return None

    # Include the selloff-end close as the plateau anchor. This captures a gap
    # or renewed break on the first purportedly quiet session.
    flat_closes = [anchor.close] + [bar.close for bar in flat_bars]
    flat_returns = _simple_returns(flat_closes)
    plateau_high = max([anchor.close] + [bar.high for bar in flat_bars])
    plateau_low = min([anchor.close] + [bar.low for bar in flat_bars])
    flat_range = plateau_high / plateau_low - 1.0
    flat_slope = _annualized_linear_move(flat_closes)
    flat_vol = _realized_vol(flat_returns)
    max_abs_return = max(abs(value) for value in flat_returns)

    if flat_range > config.max_flat_range_pct:
        return None
    if abs(flat_slope) > config.max_flat_slope_pct:
        return None
    if flat_vol > config.max_flat_realized_vol_pct:
        return None
    if max_abs_return > config.max_abs_flat_daily_return_pct:
        return None

    avg_dollar_volume = statistics.fmean(bar.volume * bar.close for bar in flat_bars)
    if avg_dollar_volume < config.min_avg_dollar_volume:
        return None

    drop_returns = _simple_returns([bar.close for bar in drop_bars])
    drop_vol = _realized_vol(drop_returns)
    vol_contraction = _safe_ratio(flat_vol, drop_vol)
    drop_median_volume = statistics.median(bar.volume for bar in drop_bars)
    flat_median_volume = statistics.median(bar.volume for bar in flat_bars)
    volume_contraction = _safe_ratio(flat_median_volume, drop_median_volume)

    close_position = (
        (latest.close - plateau_low) / (plateau_high - plateau_low)
        if plateau_high > plateau_low
        else 0.5
    )
    recent_prior_lows = [bar.low for bar in flat_bars[:-2]]
    fresh_low = bool(recent_prior_lows) and min(bar.low for bar in flat_bars[-2:]) <= min(
        recent_prior_lows
    )

    risk_points = 0
    flags: list[str] = []
    if flat_slope < -0.015:
        risk_points += 2
        flags.append("plateau_down_slope")
    elif flat_slope < -0.005:
        risk_points += 1
        flags.append("slight_down_slope")
    if close_position < 0.25:
        risk_points += 2
        flags.append("close_near_range_low")
    elif close_position < 0.40:
        risk_points += 1
        flags.append("close_below_range_mid")
    if fresh_low:
        risk_points += 1
        flags.append("fresh_flattening_low")
    if drop_magnitude > 0.32:
        risk_points += 1
        flags.append("severe_selloff")

    bear_flag_risk = "high" if risk_points >= 3 else "medium" if risk_points >= 1 else "low"
    score = _shape_score(
        config,
        drop_magnitude=drop_magnitude,
        flat_range=flat_range,
        flat_slope=flat_slope,
        flat_vol=flat_vol,
        max_abs_return=max_abs_return,
        vol_contraction=vol_contraction,
        volume_contraction=volume_contraction,
    )

    return Candidate(
        ticker=ticker,
        name=name,
        as_of=latest.session,
        latest_price=round(latest.close, 2),
        drop_pct=round(drop_return * 100.0, 2),
        flat_range_pct=round(flat_range * 100.0, 2),
        flat_slope_pct=round(flat_slope * 100.0, 2),
        flat_realized_vol_pct=round(flat_vol * 100.0, 2),
        max_abs_flat_daily_return_pct=round(max_abs_return * 100.0, 2),
        volatility_contraction_ratio=(
            round(vol_contraction, 2) if vol_contraction is not None else None
        ),
        volume_contraction_ratio=(
            round(volume_contraction, 2) if volume_contraction is not None else None
        ),
        avg_dollar_volume=round(avg_dollar_volume, 0),
        latest_close_position=round(close_position, 2),
        shape_score=score,
        bear_flag_risk=bear_flag_risk,
        flags=";".join(flags),
    )


class MassiveClient:
    def __init__(
        self,
        api_key: str,
        cache_dir: Path,
        request_delay: float = 0.0,
        max_retries: int = 4,
    ) -> None:
        self.api_key = api_key
        self.cache_dir = cache_dir
        self.request_delay = max(0.0, request_delay)
        self.max_retries = max_retries
        self._last_request_at = 0.0

    def _wait_for_rate_limit(self) -> None:
        remaining = self.request_delay - (time.monotonic() - self._last_request_at)
        if remaining > 0:
            time.sleep(remaining)

    def _with_key(self, url: str, params: Mapping[str, Any] | None = None) -> str:
        if not url.startswith("http"):
            url = f"{API_BASE}{url}"
        parsed = urlparse(url)
        query = dict(parse_qsl(parsed.query, keep_blank_values=True))
        if params:
            query.update({key: str(value) for key, value in params.items()})
        query["apiKey"] = self.api_key
        return urlunparse(parsed._replace(query=urlencode(query)))

    def _request_json(self, url: str, params: Mapping[str, Any] | None = None) -> dict[str, Any]:
        full_url = self._with_key(url, params)
        for attempt in range(self.max_retries + 1):
            self._wait_for_rate_limit()
            request = Request(full_url, headers={"User-Agent": USER_AGENT})
            try:
                with urlopen(request, timeout=45) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                self._last_request_at = time.monotonic()
                if payload.get("status") not in {None, "OK", "DELAYED"}:
                    raise ScreenerError(f"Massive returned status={payload.get('status')}")
                return payload
            except HTTPError as exc:
                self._last_request_at = time.monotonic()
                retryable = exc.code == 429 or 500 <= exc.code < 600
                if not retryable or attempt >= self.max_retries:
                    detail = ""
                    try:
                        error_payload = json.loads(exc.read().decode("utf-8", errors="replace"))
                        detail = str(
                            error_payload.get("error")
                            or error_payload.get("message")
                            or error_payload.get("status")
                            or ""
                        )
                    except (json.JSONDecodeError, AttributeError):
                        detail = ""
                    if exc.code == 403 and "before end of day" in detail.lower():
                        raise MarketDataNotReadyError(detail) from exc
                    suffix = f": {detail}" if detail else ""
                    raise ScreenerError(f"Massive HTTP error {exc.code}{suffix}") from exc
                time.sleep(min(2**attempt, 20))
            except (URLError, TimeoutError, json.JSONDecodeError) as exc:
                self._last_request_at = time.monotonic()
                if attempt >= self.max_retries:
                    raise ScreenerError(f"Market-data request failed: {exc}") from exc
                time.sleep(min(2**attempt, 20))
        raise ScreenerError("Market-data request failed after retries")

    def fetch_grouped_day(self, session: date) -> dict[str, Bar]:
        cache_path = self.cache_dir / "bars" / f"{session.isoformat()}.json"
        if cache_path.exists():
            with cache_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        else:
            try:
                payload = self._request_json(
                    f"/v2/aggs/grouped/locale/us/market/stocks/{session.isoformat()}",
                    {"adjusted": "true"},
                )
            except MarketDataNotReadyError:
                return {}
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")

        bars: dict[str, Bar] = {}
        for raw in payload.get("results") or []:
            ticker = raw.get("T")
            if ticker and all(key in raw for key in ("o", "h", "l", "c")):
                bars[ticker] = Bar.from_massive(session.isoformat(), raw)
        return bars

    def fetch_universe(
        self,
        as_of: date,
        accepted_types: Sequence[str],
        max_cache_age_days: int = 7,
    ) -> dict[str, str]:
        type_key = "-".join(sorted(accepted_types))
        cache_path = self.cache_dir / "universe" / f"active-{type_key}.json"
        if cache_path.exists():
            with cache_path.open("r", encoding="utf-8") as handle:
                cached = json.load(handle)
            cached_as_of_raw = cached.get("cached_as_of")
            cached_universe = cached.get("universe")
            if cached_as_of_raw and isinstance(cached_universe, dict):
                cached_as_of = date.fromisoformat(cached_as_of_raw)
                cache_age = (as_of - cached_as_of).days
                if 0 <= cache_age <= max_cache_age_days:
                    return dict(cached_universe)

        universe: dict[str, str] = {}
        for ticker_type in accepted_types:
            next_url: str | None = "/v3/reference/tickers"
            params: Mapping[str, Any] | None = {
                "market": "stocks",
                "locale": "us",
                "active": "true",
                "type": ticker_type,
                "date": as_of.isoformat(),
                "limit": 1000,
                "sort": "ticker",
                "order": "asc",
            }
            while next_url:
                payload = self._request_json(next_url, params)
                params = None
                for item in payload.get("results") or []:
                    ticker = item.get("ticker")
                    if ticker:
                        universe[ticker] = item.get("name") or ticker
                next_url = payload.get("next_url")

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(
                {"cached_as_of": as_of.isoformat(), "universe": universe},
                ensure_ascii=False,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return universe


def load_recent_sessions(
    client: MassiveClient,
    as_of: date,
    required_sessions: int,
    max_calendar_lookback: int = 45,
) -> list[tuple[str, dict[str, Bar]]]:
    sessions: list[tuple[str, dict[str, Bar]]] = []
    cursor = as_of
    earliest = as_of - timedelta(days=max_calendar_lookback)
    while cursor >= earliest and len(sessions) < required_sessions:
        if cursor.weekday() < 5:
            bars = client.fetch_grouped_day(cursor)
            if bars:
                sessions.append((cursor.isoformat(), bars))
        cursor -= timedelta(days=1)
    if len(sessions) < required_sessions:
        raise ScreenerError(
            f"Only found {len(sessions)} market sessions; need {required_sessions}."
        )
    sessions.reverse()
    return sessions


def screen_market(
    sessions: Sequence[tuple[str, Mapping[str, Bar]]],
    universe: Mapping[str, str],
    config: ScreenConfig,
) -> tuple[list[Candidate], dict[str, int]]:
    candidates: list[Candidate] = []
    counts = {
        "universe": len(universe),
        "complete_bar_history": 0,
        "matched": 0,
    }
    for ticker, name in universe.items():
        bars = [daily[ticker] for _, daily in sessions if ticker in daily]
        if len(bars) != config.required_bars:
            continue
        counts["complete_bar_history"] += 1
        candidate = evaluate_symbol(ticker, name, bars, config)
        if candidate is not None:
            candidates.append(candidate)
    candidates.sort(key=lambda item: (-item.shape_score, item.bear_flag_risk, item.ticker))
    counts["matched"] = len(candidates)
    return candidates, counts


def _candidate_rows(candidates: Iterable[Candidate]) -> list[dict[str, Any]]:
    return [asdict(candidate) for candidate in candidates]


def _format_ratio(value: float | None) -> str:
    return "—" if value is None else f"{value:.2f}"


def render_markdown_report(payload: Mapping[str, Any]) -> str:
    """Render a stable, human-readable contract for second-stage research."""
    config = payload["config"]
    counts = payload["counts"]
    candidates = payload["candidates"]
    lines = [
        f"# 跌后平台候选 — {payload['as_of']}",
        "",
        "> 本报告由价格与成交量规则自动生成，仅用于产生研究候选；shape_score 不是上涨概率。",
        "",
        "## 数据验收",
        "",
        f"- 市场数据日期：`{payload['as_of']}`",
        f"- 生成时间（UTC）：`{payload['generated_at_utc']}`",
        f"- 股票池：{counts['universe']:,}",
        f"- 具备完整 {config['drop_sessions']}+{config['flat_sessions']} 区间数据：{counts['complete_bar_history']:,}",
        f"- 规则匹配总数：{counts['matched']:,}",
        f"- 本报告保留：{len(candidates):,}",
        "",
        "## 筛选规则",
        "",
        "| 条件 | 数值 |",
        "|---|---:|",
        f"| 最新价 | ${config['min_price']:.0f}–${config['max_price']:.0f} |",
        f"| 前段累计跌幅 | {config['min_drop_pct'] * 100:.1f}%–{config['max_drop_pct'] * 100:.1f}% |",
        f"| 平台总振幅上限 | {config['max_flat_range_pct'] * 100:.1f}% |",
        f"| 平台趋势绝对值上限 | {config['max_flat_slope_pct'] * 100:.1f}% |",
        f"| 平台日收益波动率上限 | {config['max_flat_realized_vol_pct'] * 100:.1f}% |",
        f"| 平台单日绝对涨跌上限 | {config['max_abs_flat_daily_return_pct'] * 100:.1f}% |",
        f"| 平台平均成交额下限 | ${config['min_avg_dollar_volume'] / 1_000_000:.0f}M |",
        "",
        "## 候选排名",
        "",
    ]
    if not candidates:
        lines.extend(["**今日无形态候选。**", ""])
    else:
        lines.extend(
            [
                "| # | 股票 | 名称 | 价格 | 前段跌幅 | 平台振幅 | 平台趋势 | 波动收缩 | 量能收缩 | 成交额 | 形态分 | 下降旗形风险 |",
                "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        for rank, row in enumerate(candidates, 1):
            safe_name = str(row["name"]).replace("|", "\\|")
            lines.append(
                f"| {rank} | **{row['ticker']}** | {safe_name} | ${row['latest_price']:.2f} | "
                f"{row['drop_pct']:.2f}% | {row['flat_range_pct']:.2f}% | "
                f"{row['flat_slope_pct']:.2f}% | {_format_ratio(row['volatility_contraction_ratio'])} | "
                f"{_format_ratio(row['volume_contraction_ratio'])} | "
                f"${row['avg_dollar_volume'] / 1_000_000:.1f}M | {row['shape_score']:.2f} | "
                f"{row['bear_flag_risk']} |"
            )
        lines.append("")

    lines.extend(
        [
            "## 二次研究任务",
            "",
            "对排名靠前且下降旗形风险为 low/medium 的候选逐一补充：",
            "",
            "1. **暴跌归因**：定位下跌发生日，并核查前后 48 小时的一手信息。",
            "2. **长期逻辑损伤**：区分一次性重估、周期压力和核心逻辑破坏。",
            "3. **量化业务权重**：凡用业务占比支撑结论，必须给出收入、利润、订单或增长贡献数据；缺失时明确标注。",
            "4. **30/60 日催化**：记录确定日期、事件与市场预期；没有就写“未发现确定催化”。",
            "5. **交易触发与失效**：说明需要看到什么才买入，以及跌破何处或发生何事后失效。",
            "6. **反证优先**：财务异常、流动性风险、持续稀释、监管/FDA 二元事件和核心产品失败拥有否决权。",
            "",
            "## 自动指标释义",
            "",
            "- `shape_score`：平台紧致度、水平度、波动/量能收缩和跌幅质量的确定性排序分；不能解释为反弹概率。",
            "- `波动收缩`：平台日收益波动率 ÷ 下跌段日收益波动率，越低表示收缩越明显。",
            "- `量能收缩`：平台成交量中位数 ÷ 下跌段成交量中位数，低于 1 表示量能收缩。",
            "- 完整机器可读字段见同目录 `screen.json`。",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(
    output_dir: Path,
    as_of: str,
    candidates: Sequence[Candidate],
    config: ScreenConfig,
    counts: Mapping[str, int],
) -> tuple[Path, Path, Path, Path, Path]:
    daily_dir = output_dir / as_of
    daily_dir.mkdir(parents=True, exist_ok=True)
    rows = _candidate_rows(candidates)
    csv_path = daily_dir / "candidates.csv"
    json_path = daily_dir / "screen.json"
    markdown_path = daily_dir / "screen.md"
    latest_json_path = output_dir / "latest.json"
    latest_markdown_path = output_dir / "latest.md"

    fieldnames = [field.name for field in fields(Candidate)]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "as_of": as_of,
        "execution": {
            "github_repository": os.environ.get("GITHUB_REPOSITORY"),
            "github_run_id": os.environ.get("GITHUB_RUN_ID"),
            "github_run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
        },
        "purpose": "candidate_generation_not_investment_advice",
        "score_note": "shape_score is a deterministic ranking, not a rebound probability",
        "research_contract": {
            "required_checks": [
                "data_freshness_and_coverage",
                "selloff_primary_source_cause",
                "long_term_thesis_damage",
                "quantified_business_exposure",
                "30_and_60_day_catalysts",
                "entry_trigger_invalidation_and_counterevidence",
            ],
            "max_recommendations": 5,
            "allow_no_qualified_candidate": True,
        },
        "config": asdict(config),
        "counts": dict(counts),
        "candidates": rows,
    }
    serialized = json.dumps(payload, ensure_ascii=False, indent=2)
    json_path.write_text(serialized, encoding="utf-8")
    latest_json_path.write_text(serialized, encoding="utf-8")
    markdown = render_markdown_report(payload)
    markdown_path.write_text(markdown, encoding="utf-8")
    latest_markdown_path.write_text(markdown, encoding="utf-8")
    return csv_path, json_path, markdown_path, latest_json_path, latest_markdown_path


def _parse_date(raw: str | None) -> date:
    if raw:
        return date.fromisoformat(raw)
    return _default_as_of()


def _default_as_of(now_utc: datetime | None = None) -> date:
    """Choose a completed U.S. market date instead of the UTC calendar date."""
    current = now_utc or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    now_et = current.astimezone(ZoneInfo("America/New_York"))
    candidate = now_et.date()
    # The workflow runs after this buffer. Manual daytime runs fall back so
    # free EOD plans are never asked for an incomplete current session.
    if now_et.hour < 18:
        candidate -= timedelta(days=1)
    return candidate


def load_env_file(path: Path | None) -> None:
    """Load simple KEY=VALUE pairs without overriding existing environment values."""
    if path is None or not path.exists():
        return
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ScreenerError(f"Invalid env file line {line_number}: expected KEY=VALUE")
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise ScreenerError(f"Invalid env file line {line_number}: empty key")
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'\"', "'"}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


def run_command(args: argparse.Namespace) -> int:
    config = ScreenConfig.from_json(args.config)
    load_env_file(args.env_file)
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise ScreenerError(
            f"Set {args.api_key_env} to a Massive API key before running the screener."
        )
    as_of = _parse_date(args.as_of)
    client = MassiveClient(
        api_key=api_key,
        cache_dir=args.cache_dir,
        request_delay=args.request_delay,
    )
    sessions = load_recent_sessions(client, as_of, config.required_bars)
    effective_as_of = date.fromisoformat(sessions[-1][0])
    universe = client.fetch_universe(effective_as_of, config.accepted_ticker_types)
    candidates, counts = screen_market(sessions, universe, config)
    if args.max_results is not None:
        candidates = candidates[: args.max_results]
    csv_path, json_path, markdown_path, latest_json_path, latest_markdown_path = write_outputs(
        args.output_dir,
        effective_as_of.isoformat(),
        candidates,
        config,
        counts,
    )
    print(
        json.dumps(
            {
                "as_of": effective_as_of.isoformat(),
                "counts": counts,
                "csv": str(csv_path),
                "json": str(json_path),
                "markdown": str(markdown_path),
                "latest_json": str(latest_json_path),
                "latest_markdown": str(latest_markdown_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="Fetch EOD data and screen the market")
    run.add_argument("--as-of", help="Latest calendar date to try (YYYY-MM-DD)")
    run.add_argument("--config", type=Path, help="JSON file overriding screen thresholds")
    run.add_argument("--output-dir", type=Path, default=Path("results"))
    run.add_argument("--cache-dir", type=Path, default=Path(".cache/massive"))
    run.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
        help="Optional local KEY=VALUE file; ignored when missing",
    )
    run.add_argument("--api-key-env", default="MASSIVE_API_KEY")
    run.add_argument(
        "--request-delay",
        type=float,
        default=12.2,
        help="Minimum seconds between uncached API calls; protects free-tier quotas",
    )
    run.add_argument("--max-results", type=int)
    run.set_defaults(func=run_command)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (ScreenerError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
