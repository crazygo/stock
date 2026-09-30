"""Per-bar walk-forward at the same 09:40 clock as backtest_30d.py.

The collapsed overnight, premarket, and previous-session returns are removed.
Each 5-minute bar keeps its own return, range, volume share, and gap from the
previous bar. Clock slots stay fixed so a missing bar is missing, not shifted
onto the next column.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.after_open_3d5pct.models.open_0940_touch_v1.backtest_30d import (
    MIN_TRAIN,
    OUT_DIR,
    TARGETS,
    _clock_rows,
    _daily,
    _qqq_features,
    _session_complete,
    load_market,
)
from research.after_open_3d5pct.models.pugh_ge5.build_rows import load_calendar, union_symbols

HTML_PATH = Path(__file__).resolve().parent / "backtest_30d_bars.html"
PRIOR_PATH = OUT_DIR / "backtest_30d.json"
FIELDS = ("ret", "range", "vol", "gap")


def _clocks(start: str, stop: str, day: str = "2000-01-01") -> list[str]:
    stamps = pd.date_range(f"{day} {start}", f"{day} {stop}", freq="5min")
    return [stamp.strftime("%H:%M") for stamp in stamps]


SEGMENTS = {
    "rth": _clocks("09:35", "16:00"),
    "post": _clocks("16:05", "20:00"),
    "on": _clocks("20:05", "23:55") + _clocks("00:00", "04:00", "2000-01-02"),
    "pre": _clocks("04:05", "09:30"),
    "d0": ["09:35", "09:40"],
}
SLOTS = [(segment, clock) for segment, clocks in SEGMENTS.items() for clock in clocks]
SLOT_INDEX = {slot: index for index, slot in enumerate(SLOTS)}
BAR_FEATURES = [f"{segment}_{clock.replace(':', '')}_{field}" for segment, clock in SLOTS for field in FIELDS]
CONTEXT = ["atr_20d_pct", "prior_ret_5", "prior_ret_20", "qqq_gap", "qqq_intra_ret", "qqq_vwap_dev"]
FEATURES = CONTEXT + BAR_FEATURES


def _symbol_rows(symbol: str, bars: pd.DataFrame, qqq: pd.DataFrame, complete_days: set[str]) -> pd.DataFrame:
    regular = bars.loc[bars["session_type"].eq("regular")].reset_index(drop=True)
    if regular.empty:
        return pd.DataFrame()
    daily = _daily(regular)
    days = list(daily.index)
    day_volume = regular.groupby("day")["volume"].sum()
    bar1 = _clock_rows(regular, 35)
    bar2 = _clock_rows(regular, 40)
    ends = bars["end"].to_numpy()
    session = bars["session_type"].to_numpy()
    hhmm = bars["end"].dt.strftime("%H:%M").to_numpy()
    day_of = bars["day"].to_numpy()
    open_ = bars["open"].to_numpy(float)
    high_ = bars["high"].to_numpy(float)
    low_ = bars["low"].to_numpy(float)
    close_ = bars["close"].to_numpy(float)
    volume_ = bars["volume"].to_numpy(float)
    reg_high = regular["high"].to_numpy(float)
    reg_end = regular["end"].to_numpy()
    reg_day = regular["day"].to_numpy()
    meta_rows = []
    matrices = []
    for index, day in enumerate(days):
        if index == 0 or day not in bar1.index or day not in bar2.index or day not in qqq.index:
            continue
        previous = days[index - 1]
        prev_close = float(daily.at[day, "prev_close"])
        anchor = float(daily.at[previous, "prev_close"]) if index > 1 else np.nan
        base_volume = float(day_volume.get(previous, np.nan))
        opened = float(bar1.at[day, "open"])
        entry = float(bar2.at[day, "close"])
        if prev_close <= 0 or opened <= 0 or entry <= 0 or not np.isfinite(base_volume) or base_volume <= 0:
            continue
        values = np.full(len(SLOTS) * len(FIELDS), np.nan)
        prev_end = np.datetime64(pd.Timestamp(daily.at[previous, "last_end"]))
        decision = np.datetime64(pd.Timestamp(f"{day} 09:30"))
        entry_end = np.datetime64(pd.Timestamp(f"{day} 09:40"))
        chosen = np.flatnonzero(
            ((day_of == previous) & (session == "regular"))
            | ((ends > prev_end) & (ends <= entry_end) & (session != "regular"))
            | ((day_of == day) & (session == "regular") & (ends <= entry_end))
        )
        last_close = anchor if np.isfinite(anchor) and anchor > 0 else np.nan
        for pos in chosen:
            kind = session[pos]
            if kind == "regular" and day_of[pos] == previous:
                segment = "rth"
            elif kind == "post_market":
                segment = "post"
            elif kind == "overnight":
                segment = "on"
            elif kind == "pre_market":
                segment = "pre"
            elif kind == "regular" and day_of[pos] == day:
                segment = "d0"
            else:
                continue
            slot = SLOT_INDEX.get((segment, hhmm[pos]))
            if slot is None or open_[pos] <= 0:
                continue
            base = slot * len(FIELDS)
            values[base] = close_[pos] / open_[pos] - 1
            values[base + 1] = (high_[pos] - low_[pos]) / open_[pos]
            values[base + 2] = volume_[pos] / base_volume
            if np.isfinite(last_close) and last_close > 0:
                values[base + 3] = open_[pos] / last_close - 1
            last_close = close_[pos]
        record = {
            "symbol": symbol,
            "session_date": day,
            "entry": entry,
            "atr_20d_pct": float(daily.at[day, "atr_20d_pct"]),
            "prior_ret_5": float(daily.at[day, "prior_ret_5"]),
            "prior_ret_20": float(daily.at[day, "prior_ret_20"]),
            "qqq_gap": float(qqq.at[day, "qqq_gap"]),
            "qqq_intra_ret": float(qqq.at[day, "qqq_intra_ret"]),
            "qqq_vwap_dev": float(qqq.at[day, "qqq_vwap_dev"]),
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
            window = reg_high[left_h:right_h]
            same_span = bool(right_h > left_h and reg_day[left_h] >= day and reg_day[right_h - 1] <= final_day)
            ready = bool(final_day in complete_days and same_span and len(window))
            record[f"final_{key}"] = final_day
            record[f"ready_{key}"] = ready
            record[f"y_{key}"] = float(np.max(window) >= entry * (1 + target["level"])) if ready else np.nan
        meta_rows.append(record)
        matrices.append(values)
    if not meta_rows:
        return pd.DataFrame()
    meta = pd.DataFrame(meta_rows)
    bars_out = pd.DataFrame(np.vstack(matrices), columns=BAR_FEATURES)
    return pd.concat([meta.reset_index(drop=True), bars_out], axis=1)


def build_frame() -> pd.DataFrame:
    qqq = _qqq_features(load_market("QQQ"))
    calendar = load_calendar()
    frames = []
    for symbol in union_symbols():
        bars = load_market(symbol)
        regular = bars.loc[bars["session_type"].eq("regular")]
        counts = regular.groupby("day")["start"].nunique().to_dict()
        built = _symbol_rows(symbol, bars, qqq, _session_complete(counts, calendar))
        filled = int(built[BAR_FEATURES].notna().sum().sum()) if len(built) else 0
        print(json.dumps({"symbol": symbol, "rows": int(len(built)), "filled_cells": filled}), flush=True)
        if len(built):
            frames.append(built)
    return pd.DataFrame() if not frames else pd.concat(frames, ignore_index=True)


def _fit_and_pick(train: pd.DataFrame, today: pd.DataFrame, target_id: str) -> tuple[dict | None, np.ndarray | None]:
    import lightgbm as lgb

    label = f"y_{target_id}"
    usable = train.loc[train[label].notna()]
    if len(usable) < MIN_TRAIN or usable[label].nunique() < 2 or today.empty:
        return None, None
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
    names = today.loc[today[f"ready_{target_id}"] & today[label].notna()]
    payload = {
        "symbol": str(pick["symbol"]),
        "probability": float(pick["score"]),
        "hit": None if not ready else int(pick[label] == 1),
        "ready": ready,
        "candidates": int(len(today)),
        "base_rate": None if names.empty else float(names[label].mean()),
        "base_n": int(len(names)),
    }
    return payload, model.booster_.feature_importance(importance_type="gain")


def walk(frame: pd.DataFrame, days: list[str]) -> tuple[list[dict], dict[str, list[str]]]:
    reports = []
    gains = {item["id"]: [] for item in TARGETS}
    for offset, day in enumerate(days, start=1):
        today = frame.loc[frame["session_date"].eq(day)].copy()
        row = {"session_date": day, "targets": {}}
        for target in TARGETS:
            key = target["id"]
            train = frame.loc[frame[f"ready_{key}"].eq(True) & frame[f"final_{key}"].astype(str).lt(day)]
            pick, gain = _fit_and_pick(train, today, key)
            row["targets"][key] = pick
            if gain is not None:
                gains[key].append(gain)
        print(json.dumps({"day": day, "step": offset, "of": len(days)}), flush=True)
        reports.append(row)
    ranked = {}
    for key, rows in gains.items():
        if not rows:
            ranked[key] = []
            continue
        total = np.vstack(rows).sum(axis=0)
        order = np.argsort(total)[::-1]
        share = total / total.sum() if total.sum() else total
        ranked[key] = [
            {"feature": FEATURES[int(pos)], "gain_share": float(share[int(pos)])}
            for pos in order[:15]
            if share[int(pos)] > 0
        ]
    return reports, ranked


def _week_start(day: str) -> str:
    stamp = pd.Timestamp(day)
    return (stamp - pd.Timedelta(days=int(stamp.weekday()))).strftime("%Y-%m-%d")


def _rate(picks: list[dict]) -> dict:
    mature = [item for item in picks if item.get("hit") is not None]
    hits = sum(int(item["hit"]) for item in mature)
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
        bucket = weeks.setdefault(_week_start(report["session_date"]), {item["id"]: [] for item in TARGETS})
        for target in TARGETS:
            pick = report["targets"][target["id"]]
            if pick is not None:
                bucket[target["id"]].append({"session_date": report["session_date"], **pick})
    weekly = [
        {
            "week_start": start,
            "targets": {key: _rate(picks) for key, picks in weeks[start].items()},
            "picks": weeks[start],
        }
        for start in sorted(weeks)
    ]
    overall = {}
    for item in TARGETS:
        overall[item["id"]] = _rate(
            [
                {"hit": report["targets"][item["id"]]["hit"]}
                for report in reports
                if report["targets"][item["id"]] is not None
            ]
        )
    return {"weekly": weekly, "overall": overall}


def _pct(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    return f"{value * 100:.1f}%"


def _hit_word(hit: int | None) -> str:
    if hit is None:
        return "未到期"
    return "达成" if int(hit) == 1 else "未达成"


def _prior_index(prior: dict) -> dict[str, dict]:
    found = {}
    for report in prior["reports"]:
        found[report["session_date"]] = report["targets"]
    return found


def render_html(reports: list[dict], summary: dict, ranked: dict, prior: dict) -> str:
    old = _prior_index(prior)
    compare_rows = []
    for report in reports:
        cells = [f"<th>{report['session_date']}</th>"]
        for target in TARGETS:
            newer = report["targets"][target["id"]]
            older = old.get(report["session_date"], {}).get(target["id"])
            cells.append(
                "<td>{} {} {}<br>上一轮 {} {} {}</td>".format(
                    newer["symbol"] if newer else "无",
                    _pct(newer["probability"]) if newer else "",
                    _hit_word(newer["hit"]) if newer else "",
                    older["symbol"] if older else "无",
                    _pct(older["probability"]) if older else "",
                    _hit_word(older["hit"]) if older else "",
                )
            )
        compare_rows.append(f"<tr>{''.join(cells)}</tr>")
    week_blocks = []
    old_weeks = {item["week_start"]: item for item in prior["summary"]["weekly"]}
    for week in summary["weekly"]:
        lines = []
        previous = old_weeks.get(week["week_start"], {}).get("targets", {})
        for target in TARGETS:
            stat = week["targets"][target["id"]]
            before = previous.get(target["id"], {})
            detail = "；".join(
                f"{item['session_date'][5:]} {item['symbol']} {_pct(item['probability'])} {_hit_word(item['hit'])}"
                for item in week["picks"][target["id"]]
            )
            lines.append(
                "<p><b>{}</b>　本轮 {}/{} = {}　上一轮 {}/{} = {}<br>{}</p>".format(
                    target["name"],
                    stat["hits"],
                    stat["mature"],
                    _pct(stat["rate"]),
                    before.get("hits", "—"),
                    before.get("mature", "—"),
                    _pct(before.get("rate")),
                    detail,
                )
            )
        week_blocks.append(f"<section><h2>{week['week_start']} 当周</h2>{''.join(lines)}</section>")
    importance = []
    for target in TARGETS:
        items = "，".join(f"{item['feature']} {_pct(item['gain_share'])}" for item in ranked[target["id"]][:8])
        importance.append(f"<p><b>{target['name']}</b>　{items}</p>")
    heads = "".join(f"<th>{item['name']}<br>本轮 / 上一轮</th>" for item in TARGETS)
    overall = "　".join(
        f"{item['name']} 本轮 {_pct(summary['overall'][item['id']]['rate'])}（{summary['overall'][item['id']]['hits']}/{summary['overall'][item['id']]['mature']}），上一轮 {_pct(prior['summary']['overall'][item['id']]['rate'])}（{prior['summary']['overall'][item['id']]['hits']}/{prior['summary']['overall'][item['id']]['mature']}）"
        for item in TARGETS
    )
    return f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>逐根 K 线的 09:40 预报</title>
<style>
body {{ margin: 0; padding: 24px; font-family: sans-serif; color: #111827; background: #fff; }}
main {{ max-width: 1080px; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border-bottom: 1px solid #d1d5db; text-align: left; padding: 8px; vertical-align: top; }}
.scroll {{ overflow-x: auto; }}
</style>
</head>
<body>
<main>
<h1>逐根 5 分钟 K 线，同一套 09:40 预报</h1>
<p>上一轮把夜盘、盘前收成一个收益和四个路径数。这一轮每一根固定时刻的 K 线单独有四个数：这根的涨跌 (close−open)/open、振幅 (high−low)/open、成交量占上一常规日总成交量的比例、开盘价相对链条上一根收盘价的跳空。没有这根 K 线时，这四个数空着，不把后面的 K 线前移。上一常规盘、盘后、夜盘、当天盘前，以及当天 09:35 和 09:40 两根，都按这个口径进入。20 日波动、5 日和 20 日收益、QQQ 的三列保留。树、买入规则、三个标签和这 30 个交易日与上一轮相同。</p>
<p>{overall}</p>
<h2>树实际分裂最多的列</h2>
{''.join(importance)}
<h2>按周</h2>
{''.join(week_blocks)}
<h2>每日对照</h2>
<div class="scroll">
<table>
<thead><tr><th>交易日</th>{heads}</tr></thead>
<tbody>
{''.join(compare_rows)}
</tbody>
</table>
</div>
</main>
</body>
</html>
"""


def main() -> None:
    prior = json.loads(PRIOR_PATH.read_text())
    days = [report["session_date"] for report in prior["reports"]]
    frame = build_frame()
    if frame.empty:
        raise RuntimeError("no rows")
    print(json.dumps({"rows": int(len(frame)), "features": len(FEATURES), "slots": len(SLOTS)}), flush=True)
    reports, ranked = walk(frame, days)
    summary = summarise(reports)
    payload = {
        "features": len(FEATURES),
        "slots": len(SLOTS),
        "fields": list(FIELDS),
        "reports": reports,
        "summary": summary,
        "importance": ranked,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "backtest_30d_bars.json").write_text(json.dumps(payload, ensure_ascii=False))
    HTML_PATH.write_text(render_html(reports, summary, ranked, prior))
    print(json.dumps({"html": str(HTML_PATH), "overall": summary["overall"], "importance": ranked}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
