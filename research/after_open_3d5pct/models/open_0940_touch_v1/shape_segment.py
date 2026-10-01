"""3-day +5% model trained only on completed 8-shape episodes.

An episode is a contiguous run of regular bars whose 24-bar close return
stays beyond +5% or -5%, classified with the same cuts as the 8-shape board.
The signal is that episode plus the 24 bars before it. The label matches the
09:40 full-timeline model: regular highs after the entry bar, through the
session two trading days later.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.after_open_3d5pct.models.open_0940_touch_v1.backtest_30d import (
    MIN_TRAIN,
    OUT_DIR,
    _daily,
    _session_complete,
    load_market,
)
from research.after_open_3d5pct.models.pugh_ge5.build_rows import load_calendar, union_symbols

HTML_PATH = Path(__file__).resolve().parent / "shape_segment.html"
PRIOR_PATH = OUT_DIR / "backtest_30d_bars.json"
N = 24
PREFIX = 24
M = 0.05
D = 0.05
SHAPES = [
    "单边拉升",
    "V型深弹",
    "阶梯中继",
    "冲高回落",
    "单边下杀",
    "倒V冲顶",
    "探底回抽",
    "破位阴跌",
]
SHAPE_COLUMNS = [f"shape_{index}" for index in range(len(SHAPES))]
FEATURES = SHAPE_COLUMNS + [
    "window_ret",
    "window_min_ret",
    "window_max_ret",
    "window_min_pos",
    "window_max_pos",
    "window_range",
    "window_bars",
    "window_vol_share",
    "prefix_ret",
    "prefix_min_ret",
    "prefix_max_ret",
    "prefix_range",
    "prefix_vol_share",
    "atr_20d_pct",
    "minutes_from_open",
]


def _classify(kind: str, min_ret: float, max_ret: float, min_pos: float, max_pos: float, end_ret: float) -> str:
    if kind == "surge":
        if min_ret >= -0.005 and max_pos >= 0.75:
            return "单边拉升"
        if min_pos <= 0.45 and min_ret <= -0.015:
            return "V型深弹"
        if max_pos <= 0.55 and end_ret < max_ret - 0.015:
            return "冲高回落"
        return "阶梯中继"
    if max_ret <= 0.005 and min_pos >= 0.75:
        return "单边下杀"
    if max_pos <= 0.45 and max_ret >= 0.015:
        return "倒V冲顶"
    if min_pos <= 0.55 and end_ret > min_ret + 0.015:
        return "探底回抽"
    return "破位阴跌"


def _episodes(close: np.ndarray, level: float, kind: str) -> list[tuple[int, int]]:
    found = []
    active = False
    start = 0
    for index in range(N, len(close)):
        previous = close[index - N]
        change = close[index] / previous - 1 if previous > 0 else 0.0
        beyond = change >= level if kind == "surge" else change <= -level
        if beyond and not active:
            active = True
            start = index
        elif active and not beyond:
            found.append((start - N, index - 1))
            active = False
    if active:
        found.append((start - N, len(close) - 1))
    return found


def _path_stats(close: np.ndarray, start: int, end: int) -> tuple[float, float, float, float, float]:
    anchor = float(close[start])
    min_ret = 0.0
    max_ret = 0.0
    min_at = 0
    max_at = 0
    count = max(1, end - start + 1)
    for offset, price in enumerate(close[start : end + 1]):
        change = price / anchor - 1 if anchor > 0 else 0.0
        if change < min_ret:
            min_ret = change
            min_at = offset
        if change > max_ret:
            max_ret = change
            max_at = offset
    end_ret = float(close[end]) / anchor - 1 if anchor > 0 else 0.0
    return min_ret, max_ret, min_at / count, max_at / count, end_ret


def _symbol_events(symbol: str, bars: pd.DataFrame, complete_days: set[str]) -> list[dict]:
    regular = bars.loc[bars["session_type"].eq("regular")].sort_values("end").reset_index(drop=True)
    if len(regular) < N + PREFIX + 10:
        return []
    daily = _daily(regular)
    days = list(daily.index)
    day_index = {day: index for index, day in enumerate(days)}
    day_volume = regular.groupby("day")["volume"].sum()
    close = regular["close"].to_numpy(float)
    high = regular["high"].to_numpy(float)
    low = regular["low"].to_numpy(float)
    volume = regular["volume"].to_numpy(float)
    end_at = regular["end"].to_numpy()
    day = regular["day"].to_numpy()
    rows = []
    spans = [(item, "surge") for item in _episodes(close, M, "surge")]
    spans += [(item, "drop") for item in _episodes(close, D, "drop")]
    for (start, end), kind in spans:
        if start < PREFIX or end >= len(close) - 1:
            continue
        anchor = float(close[start])
        if anchor <= 0:
            continue
        min_ret, max_ret, min_pos, max_pos, end_ret = _path_stats(close, start, end)
        shape = _classify(kind, min_ret, max_ret, min_pos, max_pos, end_ret)
        prefix_min, prefix_max, _prefix_min_pos, _prefix_max_pos, prefix_ret = _path_stats(close, start - PREFIX, start - 1)
        entry_day = str(day[end])
        if entry_day not in day_index:
            continue
        index = day_index[entry_day]
        if index + 2 >= len(days):
            final_day = None
            ready = False
            hit = np.nan
        else:
            final_day = days[index + 2]
            final_end = np.datetime64(pd.Timestamp(daily.at[final_day, "last_end"]))
            entry_end = np.datetime64(pd.Timestamp(end_at[end]))
            left = int(np.searchsorted(end_at, entry_end, side="right"))
            right = int(np.searchsorted(end_at, final_end, side="right"))
            window = high[left:right]
            ready = bool(final_day in complete_days and len(window) and day[left] >= entry_day and day[right - 1] <= final_day)
            hit = float(np.max(window) >= float(close[end]) * 1.05) if ready else np.nan
        previous = days[index - 1] if index else None
        base_volume = float(day_volume.get(previous, np.nan)) if previous else np.nan
        window_volume = float(volume[start : end + 1].sum())
        prefix_volume = float(volume[start - PREFIX : start].sum())
        window_high = float(np.max(high[start : end + 1]))
        window_low = float(np.min(low[start : end + 1]))
        prefix_high = float(np.max(high[start - PREFIX : start]))
        prefix_low = float(np.min(low[start - PREFIX : start]))
        stamp = pd.Timestamp(end_at[end])
        record = {
            "symbol": symbol,
            "session_date": entry_day,
            "shape": shape,
            "final_day": final_day,
            "ready": ready,
            "y": hit,
            "window_ret": end_ret,
            "window_min_ret": min_ret,
            "window_max_ret": max_ret,
            "window_min_pos": min_pos,
            "window_max_pos": max_pos,
            "window_range": (window_high - window_low) / anchor,
            "window_bars": float(end - start + 1),
            "window_vol_share": window_volume / base_volume if base_volume > 0 else np.nan,
            "prefix_ret": prefix_ret,
            "prefix_min_ret": prefix_min,
            "prefix_max_ret": prefix_max,
            "prefix_range": (prefix_high - prefix_low) / float(close[start - PREFIX]) if close[start - PREFIX] > 0 else np.nan,
            "prefix_vol_share": prefix_volume / base_volume if base_volume > 0 else np.nan,
            "atr_20d_pct": float(daily.at[entry_day, "atr_20d_pct"]),
            "minutes_from_open": float((stamp - stamp.normalize()).total_seconds() / 60 - (9 * 60 + 30)),
        }
        for shape_index, name in enumerate(SHAPES):
            record[f"shape_{shape_index}"] = 1.0 if shape == name else 0.0
        rows.append(record)
    return rows


def build_frame() -> pd.DataFrame:
    calendar = load_calendar()
    frames = []
    for symbol in union_symbols():
        bars = load_market(symbol)
        regular = bars.loc[bars["session_type"].eq("regular")]
        counts = regular.groupby(regular["start"].dt.strftime("%Y-%m-%d"))["start"].nunique().to_dict()
        rows = _symbol_events(symbol, bars, _session_complete(counts, calendar))
        print(json.dumps({"symbol": symbol, "events": len(rows)}), flush=True)
        if rows:
            frames.append(pd.DataFrame(rows))
    return pd.DataFrame() if not frames else pd.concat(frames, ignore_index=True)


def _rate(hits: list[int | None]) -> dict:
    mature = [item for item in hits if item is not None]
    got = sum(int(item) for item in mature)
    return {"forecasts": len(hits), "mature": len(mature), "hits": got, "rate": None if not mature else got / len(mature)}


def walk(frame: pd.DataFrame, days: list[str]) -> dict:
    import lightgbm as lgb

    day_rows = []
    selected_hits = []
    top_hits = []
    for offset, day in enumerate(days, start=1):
        train = frame.loc[frame["ready"].eq(True) & frame["final_day"].astype(str).lt(day) & frame["y"].notna()]
        today = frame.loc[frame["session_date"].eq(day)].copy()
        if len(train) < MIN_TRAIN or train["y"].nunique() < 2 or today.empty:
            day_rows.append({"session_date": day, "events": int(len(today)), "pick": None, "selected": 0})
            print(json.dumps({"day": day, "step": offset, "events": int(len(today)), "skipped": True}), flush=True)
            continue
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
        y = train["y"].to_numpy(int)
        model.fit(train[FEATURES], y)
        train_score = model.predict_proba(train[FEATURES])[:, 1]
        threshold = float(np.quantile(train_score, 0.80))
        today["score"] = model.predict_proba(today[FEATURES])[:, 1]
        today = today.sort_values(["score", "symbol"], ascending=[False, True])
        pick = today.iloc[0]
        pick_hit = None if not bool(pick["ready"]) else int(pick["y"] == 1)
        top_hits.append(pick_hit)
        chosen = today.loc[today["score"].ge(threshold)]
        for item in chosen.itertuples(index=False):
            selected_hits.append(None if not bool(item.ready) else int(item.y == 1))
        day_rows.append(
            {
                "session_date": day,
                "events": int(len(today)),
                "selected": int(len(chosen)),
                "pick": {
                    "symbol": str(pick["symbol"]),
                    "shape": str(pick["shape"]),
                    "probability": float(pick["score"]),
                    "hit": pick_hit,
                },
            }
        )
        print(json.dumps({"day": day, "step": offset, "events": int(len(today))}), flush=True)
    test = frame.loc[frame["session_date"].isin(days)]
    by_shape = {}
    for shape, group in test.groupby("shape"):
        mature = group.loc[group["ready"].eq(True) & group["y"].notna()]
        by_shape[str(shape)] = {
            "events": int(len(group)),
            "mature": int(len(mature)),
            "hits": int(mature["y"].sum()) if len(mature) else 0,
            "rate": None if mature.empty else float(mature["y"].mean()),
        }
    gain = np.zeros(len(FEATURES))
    return {
        "days": day_rows,
        "top1": _rate(top_hits),
        "selected": _rate(selected_hits),
        "by_shape": by_shape,
        "test_events": int(len(test)),
        "gain_placeholder": gain.tolist(),
    }


def _pct(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    return f"{value * 100:.1f}%"


def render(result: dict, prior_rate: dict) -> str:
    shape_rows = []
    for shape in SHAPES:
        stat = result["by_shape"].get(shape, {"events": 0, "mature": 0, "hits": 0, "rate": None})
        shape_rows.append(
            f"<tr><th>{shape}</th><td>{stat['events']}</td><td>{stat['hits']}/{stat['mature']}</td><td>{_pct(stat['rate'])}</td></tr>"
        )
    day_rows = []
    for row in result["days"]:
        pick = row["pick"]
        if pick is None:
            day_rows.append(f"<tr><th>{row['session_date']}</th><td>{row['events']}</td><td>无预报</td></tr>")
            continue
        word = "未到期" if pick["hit"] is None else ("达成" if pick["hit"] else "未达成")
        day_rows.append(
            f"<tr><th>{row['session_date']}</th><td>{row['events']}</td><td>{pick['symbol']} · {pick['shape']} · {_pct(pick['probability'])} · {word}</td></tr>"
        )
    top = result["top1"]
    selected = result["selected"]
    return f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>8 种图形区间上的 3 天 +5%</title>
<style>
body {{ margin: 0; padding: 24px; font-family: sans-serif; color: #111827; background: #fff; }}
main {{ max-width: 880px; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border-bottom: 1px solid #d1d5db; text-align: left; padding: 8px; vertical-align: top; }}
</style>
</head>
<body>
<main>
<h1>只在 8 种图形走完的区间上学习 3 天 +5%</h1>
<p>图形规则与看板相同：常规盘连续 24 根收盘涨跌达到 ±5% 的一段，再分成 8 类。信号是这一段加上它前面的 24 根。入场价是这段最后一根的收盘价。标签与完整时间轴相同：其后常规盘最高价，收到决策日之后第 2 个交易日。树仍是 40 棵、深度 3、7 叶。考试日子是 2026-08-18 至 2026-09-29。</p>
<p>完整时间轴每天 09:40 留一只：3 天 +5% {prior_rate['hits']}/{prior_rate['mature']} = {_pct(prior_rate['rate'])}。图形模型每天在当天走完的图形里留概率最高的一只：{top['hits']}/{top['mature']} = {_pct(top['rate'])}。训练分 80 分位以上的图形：{selected['hits']}/{selected['mature']} = {_pct(selected['rate'])}，预报 {selected['forecasts']} 次。</p>
<h2>这 30 天里每种图形自己的达成率</h2>
<table>
<thead><tr><th>图形</th><th>段数</th><th>达成</th><th>达成率</th></tr></thead>
<tbody>{''.join(shape_rows)}</tbody>
</table>
<h2>每天概率最高的一段</h2>
<table>
<thead><tr><th>交易日</th><th>当天段数</th><th>最高概率</th></tr></thead>
<tbody>{''.join(day_rows)}</tbody>
</table>
</main>
</body>
</html>
"""


def main() -> None:
    prior = json.loads(PRIOR_PATH.read_text())
    days = [report["session_date"] for report in prior["reports"]]
    prior_rate = prior["summary"]["overall"]["d3_5"]
    frame = build_frame()
    if frame.empty:
        raise RuntimeError("no events")
    print(json.dumps({"events": int(len(frame)), "shapes": frame["shape"].value_counts().to_dict()}), flush=True)
    result = walk(frame, days)
    result["prior_top1"] = prior_rate
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "shape_segment.json").write_text(json.dumps(result, ensure_ascii=False))
    HTML_PATH.write_text(render(result, prior_rate))
    print(json.dumps({"top1": result["top1"], "selected": result["selected"], "by_shape": result["by_shape"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
