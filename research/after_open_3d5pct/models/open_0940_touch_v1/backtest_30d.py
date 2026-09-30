"""Walk-forward top-1 touch probabilities at the 09:40 bar.

The decision clock is the regular bar that ends at 09:40. Features use the
overnight and premarket path through 09:30, the prior daily history, QQQ
through 09:40, and both regular bars 09:30–09:35 and 09:35–09:40. The entry
price is that second bar's close. Its own high is not part of the label.

Labels are regular-session highs only:
  1d +5%  through the same session's close
  3d +5%  through the session two trading days later
  5d +10% through the session four trading days later
A row enters training only after its last label session is already complete
and strictly before the test morning.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.after_open_3d5pct.models.premarket_tail_v1.rows import _path_stats, _segment_return
from research.after_open_3d5pct.models.pugh_ge5.build_rows import (
    ROOT,
    load_calendar,
    union_symbols,
    valid_ohlcv,
)
from research.after_open_3d5pct.models.weekly_scale.run import MIN_OVERNIGHT, MIN_PREMARKET, _load_all

OUT_DIR = ROOT / "research/after_open_3d5pct/runs/open_0940_touch_v1"
HTML_PATH = Path(__file__).resolve().parent / "backtest_30d.html"
TEST_DAYS = 30
MIN_TRAIN = 400

FEATURES = [
    "post_return",
    "overnight_return",
    "premarket_return",
    "path_impulse",
    "path_fade",
    "path_cleanliness",
    "path_peak",
    "atr_20d_pct",
    "prior_ret_5",
    "prior_ret_20",
    "qqq_gap",
    "qqq_intra_ret",
    "qqq_vwap_dev",
    "bar1_ret",
    "bar1_range",
    "bar1_rvol",
    "bar2_ret",
    "bar2_range",
    "bar2_rvol",
    "open_to_entry",
]

TARGETS = [
    {"id": "d1_5", "name": "1 天 +5%", "days": 1, "level": 0.05},
    {"id": "d3_5", "name": "3 天 +5%", "days": 3, "level": 0.05},
    {"id": "d5_10", "name": "5 天 +10%", "days": 5, "level": 0.10},
]


def _prepare(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.loc[valid_ohlcv(frame)].dropna(subset=["start", "end"]).copy()
    out = out.loc[out["end"].gt(out["start"])].sort_values(["end", "start"])
    out = out.drop_duplicates("start").reset_index(drop=True)
    out["day"] = out["start"].dt.strftime("%Y-%m-%d")
    return out


def _clock_rows(regular: pd.DataFrame, minute: int) -> pd.DataFrame:
    clock = regular["end"].dt.hour.eq(9) & regular["end"].dt.minute.eq(minute) & regular["end"].dt.second.eq(0)
    found = regular.loc[clock, ["day", "open", "high", "low", "close", "volume"]].drop_duplicates("day")
    return found.set_index("day")


def _rvol(volume: pd.Series) -> pd.Series:
    prior = volume.shift(1).rolling(20, min_periods=5).median()
    out = volume / prior
    return out.where(prior > 0)


def _daily(regular: pd.DataFrame) -> pd.DataFrame:
    grouped = regular.groupby("day", sort=True).agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        last_end=("end", "max"),
        n=("start", "nunique"),
    )
    prev_close = grouped["close"].shift(1)
    span = grouped["high"] - grouped["low"]
    tr = pd.concat(
        [span, (grouped["high"] - prev_close).abs(), (grouped["low"] - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    grouped["atr_20d_pct"] = tr.rolling(20, min_periods=20).mean().shift(1) / prev_close
    grouped["prior_ret_5"] = prev_close / grouped["close"].shift(6) - 1
    grouped["prior_ret_20"] = prev_close / grouped["close"].shift(21) - 1
    grouped["prev_close"] = prev_close
    return grouped


def _qqq_features(frame: pd.DataFrame) -> pd.DataFrame:
    regular = frame.loc[frame["session_type"].eq("regular")].copy()
    daily = _daily(regular)
    bar1 = _clock_rows(regular, 35)
    bar2 = _clock_rows(regular, 40)
    rows = []
    for day in daily.index:
        if day not in bar2.index or day not in bar1.index:
            continue
        prev = daily.at[day, "prev_close"]
        opened = float(bar1.at[day, "open"])
        closed = float(bar2.at[day, "close"])
        if not np.isfinite(prev) or prev <= 0 or opened <= 0 or closed <= 0:
            continue
        window = regular.loc[(regular["day"].eq(day)) & (regular["end"].le(pd.Timestamp(f"{day} 09:40")))]
        volume = float(window["volume"].sum())
        typical = (window["high"] + window["low"] + window["close"]) / 3
        traded = float((typical * window["volume"]).sum())
        vwap = traded / volume if volume > 0 else np.nan
        rows.append(
            {
                "day": day,
                "qqq_gap": opened / float(prev) - 1,
                "qqq_intra_ret": closed / opened - 1,
                "qqq_vwap_dev": closed / vwap - 1 if np.isfinite(vwap) and vwap > 0 else np.nan,
            }
        )
    return pd.DataFrame(rows).set_index("day")


def _segment_on(frame: pd.DataFrame, mask: pd.Series) -> float:
    value, _count = _segment_return(
        frame["open"].to_numpy(float),
        frame["close"].to_numpy(float),
        mask.to_numpy(bool),
        2,
    )
    return value


def _symbol_rows(symbol: str, bars: pd.DataFrame, qqq: pd.DataFrame, complete_days: set[str]) -> pd.DataFrame:
    regular = bars.loc[bars["session_type"].eq("regular")].reset_index(drop=True)
    if regular.empty:
        return pd.DataFrame()
    daily = _daily(regular)
    days = list(daily.index)
    bar1 = _clock_rows(regular, 35)
    bar2 = _clock_rows(regular, 40)
    vol1 = _rvol(bar1["volume"])
    vol2 = _rvol(bar2["volume"])
    ends = bars["end"].to_numpy()
    session = bars["session_type"].to_numpy()
    high = regular["high"].to_numpy(float)
    reg_end = regular["end"].to_numpy()
    reg_day = regular["day"].to_numpy()
    rows = []
    for index, day in enumerate(days):
        if day not in bar1.index or day not in bar2.index or day not in qqq.index:
            continue
        prev_close = daily.at[day, "prev_close"]
        if not np.isfinite(prev_close) or prev_close <= 0 or index == 0:
            continue
        previous = days[index - 1]
        prev_end = np.datetime64(pd.Timestamp(daily.at[previous, "last_end"]))
        decision = np.datetime64(pd.Timestamp(f"{day} 09:30"))
        left = int(np.searchsorted(ends, prev_end, side="right"))
        right = int(np.searchsorted(ends, decision, side="right"))
        path = bars.iloc[left:right]
        kind = session[left:right]
        chosen = (kind == "overnight") | (kind == "pre_market")
        overnight, _n_night = _segment_return(
            path["open"].to_numpy(float), path["close"].to_numpy(float), kind == "overnight", MIN_OVERNIGHT
        )
        premarket, _n_pre = _segment_return(
            path["open"].to_numpy(float), path["close"].to_numpy(float), kind == "pre_market", MIN_PREMARKET
        )
        impulse, cleanliness, fade, peak = _path_stats(float(prev_close), path["close"].to_numpy(float)[chosen])
        post_mask = (bars["day"].eq(previous)) & (bars["session_type"].eq("post_market"))
        first = bar1.loc[day]
        second = bar2.loc[day]
        opened = float(first["open"])
        entry = float(second["close"])
        if opened <= 0 or entry <= 0:
            continue
        entry_end = np.datetime64(pd.Timestamp(f"{day} 09:40"))
        record = {
            "symbol": symbol,
            "session_date": day,
            "entry": entry,
            "post_return": _segment_on(bars, post_mask),
            "overnight_return": overnight,
            "premarket_return": premarket,
            "path_impulse": impulse,
            "path_fade": fade,
            "path_cleanliness": cleanliness,
            "path_peak": peak,
            "atr_20d_pct": float(daily.at[day, "atr_20d_pct"]),
            "prior_ret_5": float(daily.at[day, "prior_ret_5"]),
            "prior_ret_20": float(daily.at[day, "prior_ret_20"]),
            "qqq_gap": float(qqq.at[day, "qqq_gap"]),
            "qqq_intra_ret": float(qqq.at[day, "qqq_intra_ret"]),
            "qqq_vwap_dev": float(qqq.at[day, "qqq_vwap_dev"]),
            "bar1_ret": float(first["close"]) / float(first["open"]) - 1,
            "bar1_range": (float(first["high"]) - float(first["low"])) / float(first["open"]),
            "bar1_rvol": float(vol1.get(day, np.nan)),
            "bar2_ret": entry / float(second["open"]) - 1,
            "bar2_range": (float(second["high"]) - float(second["low"])) / float(second["open"]),
            "bar2_rvol": float(vol2.get(day, np.nan)),
            "open_to_entry": entry / opened - 1,
        }
        for target in TARGETS:
            horizon = target["days"]
            key = target["id"]
            if index + horizon > len(days):
                record[f"y_{key}"] = np.nan
                record[f"final_{key}"] = None
                record[f"ready_{key}"] = False
                continue
            final_day = days[index + horizon - 1]
            final_end = np.datetime64(pd.Timestamp(daily.at[final_day, "last_end"]))
            left_h = int(np.searchsorted(reg_end, entry_end, side="right"))
            right_h = int(np.searchsorted(reg_end, final_end, side="right"))
            window = high[left_h:right_h]
            same_span = right_h > left_h and reg_day[left_h] >= day and reg_day[right_h - 1] <= final_day
            ready = bool(final_day in complete_days and same_span and len(window))
            record[f"final_{key}"] = final_day
            record[f"ready_{key}"] = ready
            record[f"y_{key}"] = float(np.max(window) >= entry * (1 + target["level"])) if ready else np.nan
        rows.append(record)
    return pd.DataFrame(rows)


def _session_complete(regular_counts: dict[str, int], calendar: pd.DataFrame) -> set[str]:
    expected = {
        str(row.session_date): int(row.duration_minutes) // 5
        for row in calendar.itertuples(index=False)
    }
    return {day for day, count in regular_counts.items() if count >= expected.get(day, 78) - 1}


def load_market(symbol: str) -> pd.DataFrame:
    frame = _load_all(symbol)
    return _prepare(frame)


def build_frame() -> pd.DataFrame:
    qqq = _qqq_features(load_market("QQQ"))
    calendar = load_calendar()
    frames = []
    for symbol in union_symbols():
        bars = load_market(symbol)
        regular = bars.loc[bars["session_type"].eq("regular")]
        counts = regular.groupby("day")["start"].nunique().to_dict()
        done = _session_complete(counts, calendar)
        built = _symbol_rows(symbol, bars, qqq, done)
        print(json.dumps({"symbol": symbol, "rows": int(len(built))}), flush=True)
        if len(built):
            frames.append(built)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _fit_and_pick(train: pd.DataFrame, today: pd.DataFrame, target_id: str) -> dict | None:
    import lightgbm as lgb

    label = f"y_{target_id}"
    usable = train.loc[train[label].notna()].copy()
    if len(usable) < MIN_TRAIN or usable[label].nunique() < 2 or today.empty:
        return None
    model = lgb.LGBMClassifier(
        n_estimators=40,
        num_leaves=7,
        max_depth=3,
        min_child_samples=10,
        learning_rate=0.05,
        reg_lambda=10.0,
        n_jobs=1,
        verbosity=-1,
        random_state=3566,
    )
    model.fit(usable[FEATURES], usable[label].to_numpy(int))
    scored = today.copy()
    scored["score"] = model.predict_proba(scored[FEATURES])[:, 1]
    scored = scored.sort_values(["score", "symbol"], ascending=[False, True])
    pick = scored.iloc[0]
    ready = bool(pick[f"ready_{target_id}"])
    hit = None if not ready else int(pick[label] == 1)
    names = today.loc[today[f"ready_{target_id}"] & today[label].notna()]
    return {
        "symbol": str(pick["symbol"]),
        "probability": float(pick["score"]),
        "hit": hit,
        "ready": ready,
        "candidates": int(len(today)),
        "base_rate": None if names.empty else float(names[label].mean()),
        "base_n": int(len(names)),
    }


def walk(frame: pd.DataFrame) -> list[dict]:
    days = sorted(frame["session_date"].unique())
    chosen = days[-TEST_DAYS:]
    reports = []
    for offset, day in enumerate(chosen, start=1):
        today = frame.loc[frame["session_date"].eq(day)].copy()
        row = {"session_date": day, "targets": {}}
        for target in TARGETS:
            key = target["id"]
            train = frame.loc[frame[f"ready_{key}"].eq(True) & frame[f"final_{key}"].astype(str).lt(day)]
            row["targets"][key] = _fit_and_pick(train, today, key)
        print(json.dumps({"day": day, "step": offset, "of": len(chosen)}), flush=True)
        reports.append(row)
    return reports


def _week_start(day: str) -> str:
    stamp = pd.Timestamp(day)
    return (stamp - pd.Timedelta(days=int(stamp.weekday()))).strftime("%Y-%m-%d")


def _rate(picks: list[dict]) -> dict:
    mature = [item for item in picks if item["hit"] is not None]
    hits = sum(item["hit"] for item in mature)
    return {
        "forecasts": len(picks),
        "mature": len(mature),
        "hits": hits,
        "pending": len(picks) - len(mature),
        "rate": None if not mature else hits / len(mature),
    }


def summarise(reports: list[dict]) -> dict:
    weeks: dict[str, dict] = {}
    for report in reports:
        start = _week_start(report["session_date"])
        bucket = weeks.setdefault(start, {item["id"]: [] for item in TARGETS})
        for target in TARGETS:
            pick = report["targets"][target["id"]]
            if pick is None:
                continue
            bucket[target["id"]].append({"session_date": report["session_date"], **pick})
    weekly = []
    for start in sorted(weeks):
        weekly.append({"week_start": start, "targets": {key: _rate(picks) for key, picks in weeks[start].items()}, "picks": weeks[start]})
    overall = {item["id"]: _rate([
        {"hit": report["targets"][item["id"]]["hit"]}
        for report in reports
        if report["targets"][item["id"]] is not None
    ]) for item in TARGETS}
    return {"weekly": weekly, "overall": overall}


def _pct(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    return f"{value * 100:.1f}%"


def _hit_word(hit: int | None) -> str:
    if hit is None:
        return "未到期"
    return "达成" if hit == 1 else "未达成"


def render_html(reports: list[dict], summary: dict, span: dict) -> str:
    target_heads = "".join(f"<th>{item['name']}</th>" for item in TARGETS)
    daily_rows = []
    for report in reports:
        cells = []
        for target in TARGETS:
            pick = report["targets"][target["id"]]
            if pick is None:
                cells.append("<td>无预报</td>")
                continue
            cells.append(
                "<td><b>{}</b><br>预报概率 {}<br>{}</td>".format(
                    pick["symbol"], _pct(pick["probability"]), _hit_word(pick["hit"])
                )
            )
        daily_rows.append(f"<tr><th>{report['session_date']}</th>{''.join(cells)}</tr>")
    week_blocks = []
    for week in summary["weekly"]:
        lines = []
        for target in TARGETS:
            stat = week["targets"][target["id"]]
            picks = week["picks"][target["id"]]
            detail = "；".join(
                f"{item['session_date'][5:]} {item['symbol']} {_pct(item['probability'])} {_hit_word(item['hit'])}"
                for item in picks
            )
            lines.append(
                f"<p><b>{target['name']}</b>　预报 {stat['forecasts']}　到期 {stat['mature']}　达成 {stat['hits']}　达成率 {_pct(stat['rate'])}　未到期 {stat['pending']}<br>{detail}</p>"
            )
        week_blocks.append(f"<section><h2>{week['week_start']} 当周</h2>{''.join(lines)}</section>")
    overall = "　".join(
        f"{item['name']} {_pct(summary['overall'][item['id']]['rate'])}（{summary['overall'][item['id']]['hits']}/{summary['overall'][item['id']]['mature']}）"
        for item in TARGETS
    )
    return f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>09:40 最近 30 个交易日预报</title>
<style>
body {{ margin: 0; padding: 24px; font-family: sans-serif; color: #111827; background: #fff; }}
main {{ max-width: 980px; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border-bottom: 1px solid #d1d5db; text-align: left; padding: 8px; vertical-align: top; }}
.scroll {{ overflow-x: auto; }}
</style>
</head>
<body>
<main>
<h1>09:40 预报，最近 30 个交易日</h1>
<p>{span['first']} 至 {span['last']}。每个目标每天只留预报概率最高的一只。概率是 LGBMClassifier 的 predict_proba，标签是入场价之后常规盘最高价触及目标。未到期的日子不进达成率分母。</p>
<p>合计：{overall}</p>
<h2>按周达成率</h2>
{''.join(week_blocks)}
<h2>每日最高概率</h2>
<div class="scroll">
<table>
<thead><tr><th>交易日</th>{target_heads}</tr></thead>
<tbody>
{''.join(daily_rows)}
</tbody>
</table>
</div>
</main>
</body>
</html>
"""


def main() -> None:
    frame = build_frame()
    if frame.empty:
        raise RuntimeError("no rows")
    reports = walk(frame)
    summary = summarise(reports)
    span = {"first": reports[0]["session_date"], "last": reports[-1]["session_date"], "rows": int(len(frame))}
    payload = {"span": span, "features": FEATURES, "reports": reports, "summary": summary}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "backtest_30d.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    HTML_PATH.write_text(render_html(reports, summary, span))
    print(json.dumps({"html": str(HTML_PATH), "overall": summary["overall"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
