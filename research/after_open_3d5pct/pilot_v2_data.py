"""Read-only, point-in-time-prefix dataset for the documented v2 exploratory run.

This dataset uses a retrospective stock pool and assumed historical availability.
It must never be described as a point-in-time universe or independent validation.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
import json
import math

import numpy as np
import pandas as pd


ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")
STEP = timedelta(minutes=5)
HORIZON_BARS = 1170 // 5
SESSION_LENGTHS = {"post_market": 48, "overnight": 96, "pre_market": 66}
SEQ_CHANNELS = ("cumulative_return_pct", "bar_return_pct", "bar_range_pct",
                "log_dollar_volume_div_20", "qqq_bar_return_pct",
                "industry_bar_return_pct", "observed_mask")


@dataclass
class Bundle:
    rows: pd.DataFrame
    h_seq: np.ndarray
    post_seq: np.ndarray
    pre_seq: np.ndarray
    b_seq: np.ndarray
    exclusions: dict[str, int]
    coverage: dict[str, object]
    source_paths: list[str]


def _safe_ratio(value: float, reference: float) -> float:
    return value / reference if math.isfinite(reference) and reference > 0 else math.nan


def _summary(frame: pd.DataFrame) -> dict[str, float]:
    valid = frame[frame["open"].notna() & frame["close"].notna() &
                  (frame["open"] > 0) & (frame["close"] > 0)]
    if valid.empty:
        return {"coverage": 0.0, "return": 0.0, "range": 0.0,
                "dollar": 0.0, "open": math.nan, "close": math.nan,
                "up_fraction": 0.0, "realized": 0.0}
    opening = float(valid.iloc[0]["open"])
    closing = float(valid.iloc[-1]["close"])
    returns = valid["close"].to_numpy(float) / valid["open"].to_numpy(float) - 1
    return {
        "coverage": len(valid) / len(frame),
        "return": closing / opening - 1,
        "range": (float(valid["high"].max()) - float(valid["low"].min())) / opening,
        "dollar": float((valid["volume"] * valid["close"]).sum()),
        "open": opening, "close": closing,
        "up_fraction": float(np.mean(returns > 0)),
        "realized": float(np.std(returns)),
    }


def _sequence(frame: pd.DataFrame, qqq: pd.DataFrame, industry: pd.DataFrame,
              output_length: int) -> np.ndarray:
    """Return fixed-length observed-only values; absence stays masked, never zero-trade."""
    out = np.zeros((output_length, len(SEQ_CHANNELS)), dtype=np.float32)
    used = min(len(frame), output_length)
    if not used:
        return out
    f = frame.iloc[:used]
    q = qqq.iloc[:used]
    ind = industry.iloc[:used]
    observed = f["open"].notna().to_numpy() & f["close"].notna().to_numpy()
    if not observed.any():
        return out
    first = float(f.iloc[int(np.flatnonzero(observed)[0])]["open"])
    open_ = f["open"].to_numpy(float)
    close = f["close"].to_numpy(float)
    high = f["high"].to_numpy(float)
    low = f["low"].to_numpy(float)
    vol = f["volume"].to_numpy(float)
    j = np.flatnonzero(observed)
    out[j, 0] = 100 * (close[j] / first - 1)
    out[j, 1] = 100 * (close[j] / open_[j] - 1)
    out[j, 2] = 100 * (high[j] - low[j]) / open_[j]
    out[j, 3] = np.log1p(np.maximum(vol[j] * close[j], 0)) / 20
    for column, ref in ((4, q), (5, ind)):
        ro = ref["open"].to_numpy(float)
        rc = ref["close"].to_numpy(float)
        good = observed & np.isfinite(ro) & np.isfinite(rc)
        ix = np.flatnonzero(good)
        out[ix, column] = 100 * (rc[ix] / ro[ix] - 1)
    out[j, 6] = 1.0
    return out


def _date_grid(day: str, segment: str) -> pd.DatetimeIndex:
    date = datetime.fromisoformat(day).date()
    if segment == "pre_market":
        a, b = datetime.combine(date, time(4), ET), datetime.combine(date, time(9, 30), ET)
    elif segment == "post_market":
        a, b = datetime.combine(date, time(16), ET), datetime.combine(date, time(20), ET)
    else:
        raise ValueError(segment)
    return pd.date_range(a, b, freq="5min", inclusive="left").tz_convert(UTC)


class SymbolView:
    def __init__(self, path: Path, grid: pd.DatetimeIndex,
                 day_slices: dict[str, slice]):
        frame = pd.read_parquet(path, columns=[
            "start_at", "available_at", "session_type", "price_basis", "open",
            "high", "low", "close", "volume"])
        if frame.empty or set(frame["price_basis"].dropna()) != {"NONE"}:
            raise ValueError("missing or mixed price basis")
        frame["start_at"] = pd.to_datetime(frame["start_at"], utc=True)
        frame["available_at"] = pd.to_datetime(frame["available_at"], utc=True)
        if frame["start_at"].duplicated().any():
            raise ValueError("duplicate bar starts")
        bad = ((frame["open"] <= 0) | (frame["close"] <= 0) | (frame["low"] <= 0) |
               (frame["high"] < frame[["open", "close", "low"]].max(axis=1)) |
               (frame["low"] > frame[["open", "close", "high"]].min(axis=1)) |
               (frame["volume"] < 0))
        self.invalid_price_bars = int(bad.sum())
        frame.loc[bad, ["open", "high", "low", "close", "volume"]] = np.nan
        self.frame = frame.set_index("start_at").sort_index()
        regular = self.frame[self.frame["session_type"] == "regular"]
        self.rth = regular.reindex(grid)
        self.daily = {}
        for day, slc in day_slices.items():
            f = self.rth.iloc[slc]
            s = _summary(f)
            s["full"] = len(f) > 0 and s["coverage"] == 1.0
            self.daily[day] = s
        self._segments: dict[tuple[str, str], pd.DataFrame] = {}

    def segment(self, day: str, kind: str) -> pd.DataFrame:
        key = (day, kind)
        if key not in self._segments:
            idx = _date_grid(day, kind)
            f = self.frame.reindex(idx)
            # A mislabeled source bar cannot silently enter a different session.
            f.loc[f["session_type"] != kind, ["open", "high", "low", "close", "volume"]] = np.nan
            self._segments[key] = f
        return self._segments[key]


def _split_days(path: Path) -> set[str]:
    if not path.exists():
        return set()
    actions = pd.read_parquet(path)
    if "ex_div_date" not in actions:
        return set()
    ratio = pd.to_numeric(actions.get("split_ratio", pd.Series(index=actions.index, dtype=float)), errors="coerce")
    base = pd.to_numeric(actions.get("split_base", pd.Series(index=actions.index, dtype=float)), errors="coerce")
    ert = pd.to_numeric(actions.get("split_ert", pd.Series(index=actions.index, dtype=float)), errors="coerce")
    active = (ratio.notna() & (ratio != 0) & (ratio != 1)) | (base.notna() & ert.notna() & (base != ert))
    return set(actions.loc[active, "ex_div_date"].astype(str))


def build_bundle(repo: Path, config: dict, symbols: set[str] | None = None) -> Bundle:
    source_root = repo / config["source_dir"]
    universe_path = repo / config["universe"]
    calendar_path = repo / config["calendar"]
    u = json.loads(universe_path.read_text())
    calendar = json.loads(calendar_path.read_text())
    members = [m for m in u["members"] if m["role"] == "candidate"]
    if symbols is not None:
        members = [m for m in members if m["symbol"] in symbols]
    dates, slices, grid_parts = [], {}, []
    cursor = 0
    for session in calendar["sessions"]:
        day = session["session_date"]
        if day < "2026-01-01" or day > "2026-09-24":
            continue
        opening = pd.Timestamp(session["open_at"])
        closing = pd.Timestamp(session["close_at"])
        ix = pd.date_range(opening, closing, freq="5min", inclusive="left")
        if len(ix) != int((closing-opening).total_seconds() / 300):
            raise ValueError(f"calendar partial 5m bar: {day}")
        dates.append(day)
        slices[day] = slice(cursor, cursor + len(ix))
        grid_parts.append(ix)
        cursor += len(ix)
    grid = pd.DatetimeIndex(np.concatenate([ix.to_numpy() for ix in grid_parts]), tz=UTC)
    date_index = {day: i for i, day in enumerate(dates)}
    sources = [str(universe_path.relative_to(repo)), str(calendar_path.relative_to(repo))]
    benchmarks = {}
    for symbol in sorted({"QQQ"} | {m["industry_proxy"] for m in members}):
        p = source_root / symbol / "2026.parquet"
        benchmarks[symbol] = SymbolView(p, grid, slices)
        sources.append(str(p.relative_to(repo)))
    excluded = Counter()
    records = []
    h_rows, post_rows, pre_rows, b_rows = [], [], [], []
    a_coverage = Counter()
    as_of = datetime.now(UTC)

    for member in members:
        symbol = member["symbol"]
        p = source_root / symbol / "2026.parquet"
        if not p.exists():
            excluded["symbol_file_missing"] += 1
            continue
        try:
            stock = SymbolView(p, grid, slices)
        except ValueError:
            excluded["symbol_bad_bar_source"] += 1
            continue
        excluded["source_invalid_price_bars"] += stock.invalid_price_bars
        sources.append(str(p.relative_to(repo)))
        sector = benchmarks[member["industry_proxy"]]
        qqq = benchmarks["QQQ"]
        split_days = _split_days(repo / config["corporate_actions_dir"] / f"{symbol}.parquet")
        action_path = repo / config["corporate_actions_dir"] / f"{symbol}.parquet"
        if action_path.exists():
            sources.append(str(action_path.relative_to(repo)))
        daily_prefix_dollar = {}
        daily_prefix_range = {}
        cutoff_counts = [24, 36, 48, 60]
        for d in dates:
            f = stock.rth.iloc[slices[d]]
            daily_prefix_dollar[d] = np.cumsum(
                (f["volume"].fillna(0) * f["close"].fillna(0)).to_numpy(float))
            daily_prefix_range[d] = {}
            for count in cutoff_counts:
                if len(f) >= count and f["open"].iloc[0] > 0:
                    daily_prefix_range[d][count] = (
                        float(f["high"].iloc[:count].max()) -
                        float(f["low"].iloc[:count].min())) / float(f["open"].iloc[0])
        for day_ix in range(config["history_sessions"], len(dates)):
            day = dates[day_ix]
            past = dates[day_ix-config["history_sessions"]:day_ix]
            if not all(stock.daily[d]["full"] and qqq.daily[d]["full"] and sector.daily[d]["full"] for d in past):
                excluded["historical_regular_gap"] += 4
                continue
            last_possible = min(len(dates)-1, day_ix+4)
            if split_days.intersection(dates[day_ix-config["history_sessions"]:last_possible+1]):
                excluded["split_in_input_or_possible_label"] += 4
                continue
            hist = [stock.daily[d] for d in past]
            hist_q = [qqq.daily[d] for d in past]
            hist_s = [sector.daily[d] for d in past]
            ranges = [v["range"] for v in hist]
            dollars = [v["dollar"] for v in hist]
            hist_features = {
                "h_return_1d": hist[-1]["return"],
                "h_return_3d": hist[-1]["close"] / hist[-3]["open"] - 1,
                "h_return_5d": hist[-1]["close"] / hist[-5]["open"] - 1,
                "h_return_10d": hist[-1]["close"] / hist[-10]["open"] - 1,
                "h_range_actual": ranges[-1],
                "h_range_relative": _safe_ratio(ranges[-1], float(np.median(ranges[:-1]))),
                "h_dollar_actual_log": math.log1p(dollars[-1]),
                "h_dollar_relative": _safe_ratio(dollars[-1], float(np.median(dollars[:-1]))),
                "h_range_mean_5d": float(np.mean(ranges[-5:])),
                "h_qqq_return_5d": hist_q[-1]["close"] / hist_q[-5]["open"] - 1,
                "h_sector_return_5d": hist_s[-1]["close"] / hist_s[-5]["open"] - 1,
            }
            hs = np.zeros((config["history_sessions"], len(SEQ_CHANNELS)), np.float32)
            anchor = hist[0]["open"]
            for j, (s, q, ind) in enumerate(zip(hist, hist_q, hist_s)):
                hs[j] = [100*(s["close"]/anchor-1), 100*s["return"], 100*s["range"],
                         math.log1p(s["dollar"])/20, 100*q["return"], 100*ind["return"], 1]

            prev = dates[day_ix-1]
            post, pre = stock.segment(prev, "post_market"), stock.segment(day, "pre_market")
            qpost, qpre = qqq.segment(prev, "post_market"), qqq.segment(day, "pre_market")
            spost, spre = sector.segment(prev, "post_market"), sector.segment(day, "pre_market")
            post_summary, pre_summary = _summary(post), _summary(pre)
            a_coverage["post_rows"] += 1
            a_coverage["post_observed_bars"] += int(post_summary["coverage"]*len(post))
            a_coverage["pre_observed_bars"] += int(pre_summary["coverage"]*len(pre))
            a_coverage["night_observed_bars"] += 0
            past_post_dollar, past_pre_dollar = [], []
            for d in past[:-1]:
                # Historic same-session denominator; no current/future bar enters it.
                past_post_dollar.append(_summary(stock.segment(d, "post_market"))["dollar"])
                past_pre_dollar.append(_summary(stock.segment(d, "pre_market"))["dollar"])
            a_features = {
                "a_post_return": post_summary["return"],
                "a_post_range": post_summary["range"],
                "a_post_dollar_actual_log": math.log1p(post_summary["dollar"]),
                "a_post_dollar_relative": _safe_ratio(post_summary["dollar"], float(np.median(past_post_dollar))),
                "a_post_coverage": post_summary["coverage"],
                "a_post_vs_qqq": post_summary["return"] - _summary(qpost)["return"],
                "a_post_vs_sector": post_summary["return"] - _summary(spost)["return"],
                "a_pre_return": pre_summary["return"],
                "a_pre_range": pre_summary["range"],
                "a_pre_dollar_actual_log": math.log1p(pre_summary["dollar"]),
                "a_pre_dollar_relative": _safe_ratio(pre_summary["dollar"], float(np.median(past_pre_dollar))),
                "a_pre_coverage": pre_summary["coverage"],
                "a_pre_vs_qqq": pre_summary["return"] - _summary(qpre)["return"],
                "a_pre_vs_sector": pre_summary["return"] - _summary(spre)["return"],
                "a_night_coverage": 0.0,
            }
            post_sequence = _sequence(post, qpost, spost, SESSION_LENGTHS["post_market"])
            pre_sequence = _sequence(pre, qpre, spre, SESSION_LENGTHS["pre_market"])
            day_slice = slices[day]
            day_frame = stock.rth.iloc[day_slice]
            q_frame = qqq.rth.iloc[day_slice]
            s_frame = sector.rth.iloc[day_slice]
            for hour in config["decision_hours_et"]:
                hh, mm = map(int, hour.split(":"))
                cutoff_bars = (hh*60+mm-(9*60+30))//5
                if not all(cutoff_bars in daily_prefix_range[d] for d in past):
                    excluded["historical_elapsed_window_missing"] += 1
                    continue
                if len(day_frame) <= cutoff_bars:
                    excluded["cutoff_after_close"] += 1
                    continue
                prefix, qp, sp = day_frame.iloc[:cutoff_bars], q_frame.iloc[:cutoff_bars], s_frame.iloc[:cutoff_bars]
                if prefix["open"].isna().any() or qp["open"].isna().any() or sp["open"].isna().any():
                    excluded["current_regular_prefix_gap"] += 1
                    continue
                cutoff_time = prefix.index[-1] + pd.Timedelta(minutes=5)
                decision_time = cutoff_time + pd.Timedelta(seconds=config["signal_latency_seconds"])
                if (prefix["available_at"] > decision_time).any() or (qp["available_at"] > decision_time).any() or (sp["available_at"] > decision_time).any():
                    excluded["prefix_available_after_decision"] += 1
                    continue
                entry_index = day_slice.start + cutoff_bars + 1
                end_index = entry_index + HORIZON_BARS
                if end_index > len(grid):
                    excluded["label_pending_beyond_source_window"] += 1
                    continue
                entry_start = grid[entry_index]
                if entry_start != cutoff_time + pd.Timedelta(minutes=5):
                    excluded["entry_time_mismatch"] += 1
                    continue
                path = stock.rth.iloc[entry_index:end_index]
                if len(path) != HORIZON_BARS or path["open"].isna().any():
                    excluded["label_missing_regular_bar"] += 1
                    continue
                ready = path["available_at"].max()
                if pd.isna(ready) or ready > as_of:
                    excluded["label_not_available"] += 1
                    continue
                target = int(path["high"].max() >= float(path.iloc[0]["open"])*(1+config["target_return"]))
                b_sum, q_sum, s_sum = _summary(prefix), _summary(qp), _summary(sp)
                recent = _summary(prefix.iloc[-12:])
                first_two = _summary(prefix.iloc[:24])
                past_prefix_dollar = [float(daily_prefix_dollar[d][cutoff_bars-1]) for d in past]
                b_features = {
                    "b_return": b_sum["return"], "b_range_actual": b_sum["range"],
                    "b_range_relative": _safe_ratio(
                        b_sum["range"], float(np.median([daily_prefix_range[d][cutoff_bars] for d in past]))),
                    "b_distance_from_high": float(prefix.iloc[-1]["close"]) / float(prefix["high"].max()) - 1,
                    "b_dollar_actual_log": math.log1p(b_sum["dollar"]),
                    "b_dollar_relative": _safe_ratio(b_sum["dollar"], float(np.median(past_prefix_dollar))),
                    "b_realized": b_sum["realized"], "b_up_fraction": b_sum["up_fraction"],
                    "b_first_2h_return": first_two["return"], "b_recent_1h_return": recent["return"],
                    "b_vs_qqq": b_sum["return"]-q_sum["return"],
                    "b_vs_sector": b_sum["return"]-s_sum["return"],
                    "b_elapsed_hours": cutoff_bars/12,
                }
                if post_summary["coverage"] and pre_summary["coverage"]:
                    b_features["ab_pre_gain_retained"] = b_sum["return"] - pre_summary["return"]
                else:
                    b_features["ab_pre_gain_retained"] = math.nan
                record = {
                    "sample_id": f"{symbol}|{decision_time.isoformat()}", "symbol": symbol,
                    "industry_proxy": member["industry_proxy"], "session_date": day,
                    "cutoff_et": hour, "cutoff_at": cutoff_time.isoformat(),
                    "decision_at": decision_time.isoformat(),
                    "entry_at": entry_start.isoformat(), "entry_price": float(path.iloc[0]["open"]),
                    "label_end_at": (path.index[-1]+pd.Timedelta(minutes=5)).isoformat(),
                    "label_available_at": ready.isoformat(), "target": target,
                    **hist_features, **a_features, **b_features,
                }
                records.append(record)
                h_rows.append(hs)
                post_rows.append(post_sequence)
                pre_rows.append(pre_sequence)
                b_rows.append(_sequence(prefix, qp, sp, 60))

    if not records:
        raise RuntimeError(f"no eligible records; exclusions={excluded}")
    result = Bundle(
        rows=pd.DataFrame(records), h_seq=np.stack(h_rows), post_seq=np.stack(post_rows),
        pre_seq=np.stack(pre_rows), b_seq=np.stack(b_rows), exclusions=dict(excluded),
        coverage={"overnight_observed_bars": 0, "post_observed_bars": a_coverage["post_observed_bars"],
                  "pre_observed_bars": a_coverage["pre_observed_bars"],
                  "post_possible_bars": a_coverage["post_rows"]*48,
                  "pre_possible_bars": a_coverage["post_rows"]*66,
                  "universe_mode": u["universe_mode"],
                  "availability_quality": config["feature_availability"]},
        source_paths=sorted(set(sources)))
    return result
