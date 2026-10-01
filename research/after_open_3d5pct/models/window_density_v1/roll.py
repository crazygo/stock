"""Roll a 2-hour, 5-minute window across the regular session.

The decision is every bar end on a fixed stride: 5, 15, 30, or 60 minutes,
from the first legal bar through 16:00. Each stride is its own model. A
forecast is the highest probability among the names that have a bar at that
clock. Clocks are not collapsed into one daily pick.

Slots before 09:35 stay empty. The first regular bar still gaps against the
previous regular close. The label is the 3-session regular-high +5% from the
decision bar's close.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from research.after_open_3d5pct.models.open_0940_touch_v1.backtest_30d import MIN_TRAIN, load_market
from research.after_open_3d5pct.models.pugh_ge5.build_rows import ROOT, load_calendar, union_symbols
from research.after_open_3d5pct.models.window_density_v1.run import (
    FIELDS,
    SLOT0,
    Book,
    _auc,
    _feature_names,
    _features,
    _label,
)

OUT_DIR = ROOT / "research/after_open_3d5pct/runs/window_density_v1"
CACHE_META = OUT_DIR / "roll_meta.parquet"
CACHE_X = OUT_DIR / "roll_x.npy"
HTML_PATH = Path(__file__).resolve().parent / "roll.html"
PRIOR_PATH = ROOT / "research/after_open_3d5pct/runs/open_0940_touch_v1/backtest_30d_bars.json"
WINDOW = 120
BAR = 5
STEPS = [5, 15, 30, 60]
OPEN_MINUTE = 9 * 60 + 30
CLOSE_MINUTE = 16 * 60


def _clocks(step: int) -> list[int]:
    return list(range(OPEN_MINUTE + step, CLOSE_MINUTE + 1, step))


def _hhmm(minute: int) -> str:
    return f"{minute // 60:02d}:{minute % 60:02d}"


def _audit(books: dict[str, Book]) -> None:
    book = books.get("AMD")
    if book is None or "2026-08-18" not in book.day_index:
        raise RuntimeError("AMD 2026-08-18 is missing")
    day_i = book.day_index["2026-08-18"]
    previous = float(book.close[book.last_pos[day_i - 1]])
    noon = _features(book, day_i, {"window": WINDOW, "bar": BAR, "clock": 11 * 60 + 30})
    if noon is None or not np.isclose(noon[3], float(book.open[book.slots[day_i, 0]]) / previous - 1):
        raise RuntimeError("11:30 first bar does not gap against the previous close")
    early = _features(book, day_i, {"window": WINDOW, "bar": BAR, "clock": 9 * 60 + 40})
    if early is None or np.isfinite(early[0]) or not np.isfinite(early[-4]):
        raise RuntimeError("09:40 window does not leave the pre-open slots empty")
    open_0935 = float(book.open[book.slots[day_i, 0]])
    if not np.isclose(early[22 * len(FIELDS) + 3], open_0935 / previous - 1):
        raise RuntimeError("09:40 first regular bar lost the previous-close gap")
    print(json.dumps({"audit": "AMD 2026-08-18", "clocks_5m": len(_clocks(5))}), flush=True)


def _matrix(books: dict[str, Book]) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    names = _feature_names({"window": WINDOW, "bar": BAR})
    clocks = _clocks(5)
    metas = []
    blocks = []
    for book in books.values():
        rows = []
        features = []
        for day_i, day in enumerate(book.days):
            for clock in clocks:
                cell = {"window": WINDOW, "bar": BAR, "clock": clock}
                values = _features(book, day_i, cell)
                if values is None:
                    continue
                final_day, ready, hit = _label(book, day_i, cell)
                rows.append((book.symbol, day, clock, final_day, ready, hit))
                features.append(values)
        if not features:
            continue
        metas.append(pd.DataFrame(rows, columns=["symbol", "session_date", "clock", "final_day", "ready", "y"]))
        blocks.append(np.vstack(features).astype(np.float32))
        print(json.dumps({"symbol": book.symbol, "rows": len(rows)}), flush=True)
    meta = pd.concat(metas, ignore_index=True)
    return meta, np.vstack(blocks), names


def _rate(hits: list[int | None]) -> dict:
    mature = [item for item in hits if item is not None]
    got = sum(int(item) for item in mature)
    return {"forecasts": len(hits), "mature": len(mature), "hits": got, "rate": None if not mature else got / len(mature)}


def _run_step(meta: pd.DataFrame, values: np.ndarray, names: list[str], days: list[str], step: int) -> dict:
    import lightgbm as lgb

    chosen_clocks = set(_clocks(step))
    shared_clocks = set(_clocks(60))
    keep = meta["clock"].isin(chosen_clocks).to_numpy()
    frame = meta.loc[keep].reset_index(drop=True)
    data = values[keep]
    final = frame["final_day"].fillna("").astype(str).to_numpy(dtype=str)
    ready = frame["ready"].to_numpy(bool)
    label = frame["y"].to_numpy(float)
    session = frame["session_date"].astype(str).to_numpy(dtype=str)
    clock = frame["clock"].to_numpy(int)
    symbol = frame["symbol"].to_numpy()
    gain = np.zeros(len(names))
    train_aucs: list[float] = []
    top_hits: list[int | None] = []
    selected_hits: list[int | None] = []
    shared_hits: list[int | None] = []
    day_rates: list[float] = []
    base_hits = 0
    base_n = 0
    shared_base_hits = 0
    shared_base_n = 0
    hour_hits: dict[int, list[int | None]] = {}
    hour_base: dict[int, list[int]] = {}
    clock_hits: dict[int, list[int | None]] = {item: [] for item in sorted(shared_clocks)}
    clock_base: dict[int, list[int]] = {item: [0, 0] for item in sorted(shared_clocks)}
    symbols = Counter()
    for offset, day in enumerate(days, start=1):
        train_mask = ready & (final < day) & np.isfinite(label)
        today_mask = session == day
        if int(train_mask.sum()) < MIN_TRAIN or len(np.unique(label[train_mask])) < 2 or not today_mask.any():
            print(json.dumps({"step": step, "day": day, "offset": offset, "skipped": True}), flush=True)
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
        y = label[train_mask].astype(int)
        model.fit(data[train_mask], y)
        trained = model.predict_proba(data[train_mask])[:, 1]
        score = _auc(y, trained)
        if score is not None:
            train_aucs.append(float(score))
        gain += model.booster_.feature_importance(importance_type="gain")
        today = pd.DataFrame(
            {
                "symbol": symbol[today_mask],
                "clock": clock[today_mask],
                "ready": ready[today_mask],
                "y": label[today_mask],
                "score": model.predict_proba(data[today_mask])[:, 1],
            }
        )
        threshold = float(np.quantile(trained, 0.80))
        chosen = today.loc[today["score"].ge(threshold)]
        for item in chosen.itertuples(index=False):
            selected_hits.append(None if not bool(item.ready) or not np.isfinite(item.y) else int(item.y == 1))
        day_hits: list[int | None] = []
        for minute, group in today.groupby("clock", sort=True):
            mature = group.loc[group["ready"].eq(True) & np.isfinite(group["y"])]
            got = int(mature["y"].sum()) if len(mature) else 0
            base_hits += got
            base_n += int(len(mature))
            ordered = group.sort_values(["score", "symbol"], ascending=[False, True])
            pick = ordered.iloc[0]
            pick_hit = None if not bool(pick["ready"]) or not np.isfinite(pick["y"]) else int(pick["y"] == 1)
            top_hits.append(pick_hit)
            day_hits.append(pick_hit)
            hour = int(minute) // 60
            hour_hits.setdefault(hour, []).append(pick_hit)
            hour_base.setdefault(hour, [0, 0])
            hour_base[hour][0] += got
            hour_base[hour][1] += int(len(mature))
            if int(minute) in shared_clocks:
                shared_hits.append(pick_hit)
                shared_base_hits += got
                shared_base_n += int(len(mature))
                clock_hits[int(minute)].append(pick_hit)
                clock_base[int(minute)][0] += got
                clock_base[int(minute)][1] += int(len(mature))
            if pick_hit is not None:
                symbols[str(pick["symbol"])] += 1
        mature_day = [item for item in day_hits if item is not None]
        if mature_day:
            day_rates.append(sum(mature_day) / len(mature_day))
        print(json.dumps({"step": step, "day": day, "offset": offset, "clocks": int(today["clock"].nunique())}), flush=True)
    total = float(gain.sum()) or 1.0
    field_share = {field: 0.0 for field in FIELDS}
    for index, name in enumerate(names):
        field_share[name.rsplit("_", 1)[1]] += float(gain[index] / total)
    hours = []
    for hour in sorted(hour_hits):
        stat = _rate(hour_hits[hour])
        hits, count = hour_base[hour]
        hours.append(
            {
                "hour": hour,
                "top1": stat,
                "base": {"hits": hits, "n": count, "rate": None if not count else hits / count},
            }
        )
    shared_rows = []
    for minute in sorted(shared_clocks):
        stat = _rate(clock_hits[minute])
        hits, count = clock_base[minute]
        shared_rows.append(
            {
                "clock": _hhmm(minute),
                "top1": stat,
                "base": {"hits": hits, "n": count, "rate": None if not count else hits / count},
            }
        )
    leader = symbols.most_common(1)
    return {
        "step": step,
        "clocks": len(chosen_clocks),
        "top1": _rate(top_hits),
        "day_mean_rate": None if not day_rates else float(np.mean(day_rates)),
        "base": {"hits": base_hits, "n": base_n, "rate": None if not base_n else base_hits / base_n},
        "selected": _rate(selected_hits),
        "mean_train_auc": None if not train_aucs else float(np.mean(train_aucs)),
        "gain_fields": field_share,
        "shared": _rate(shared_hits),
        "shared_base": {
            "hits": shared_base_hits,
            "n": shared_base_n,
            "rate": None if not shared_base_n else shared_base_hits / shared_base_n,
        },
        "hours": hours,
        "shared_clocks": shared_rows,
        "leader": None if not leader else {"symbol": leader[0][0], "picks": leader[0][1]},
    }


def _pct(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    return f"{value * 100:.1f}%"


def _count(stat: dict) -> str:
    return f"{stat['hits']}/{stat['mature']} = {_pct(stat['rate'])}"


def render(steps: list[dict]) -> str:
    body = []
    for step in steps:
        body.append(
            "<tr>"
            f"<th>每 {step['step']} 分钟</th>"
            f"<td>{step['clocks']}</td>"
            f"<td>{_count(step['top1'])}</td>"
            f"<td>{step['base']['hits']}/{step['base']['n']} = {_pct(step['base']['rate'])}</td>"
            f"<td>{_pct(step['day_mean_rate'])}</td>"
            f"<td>{_count(step['shared'])}</td>"
            f"<td>{_count(step['selected'])}</td>"
            f"<td>{_pct(step['mean_train_auc'])}</td>"
            "</tr>"
        )
    fine = next(step for step in steps if step["step"] == 5)
    hour_rows = []
    for row in fine["hours"]:
        hour = row["hour"]
        label = "16:00" if hour == 16 else f"{hour:02d}:35–{hour:02d}:55" if hour == 9 else f"{hour:02d}:00–{hour:02d}:55"
        hour_rows.append(
            "<tr>"
            f"<th>{label}</th>"
            f"<td>{_count(row['top1'])}</td>"
            f"<td>{row['base']['hits']}/{row['base']['n']} = {_pct(row['base']['rate'])}</td>"
            "</tr>"
        )
    clock_head = "".join(f"<th>每 {step['step']} 分钟</th>" for step in steps)
    clock_rows = []
    clocks = fine["shared_clocks"]
    for index, clock in enumerate(clocks):
        cells = []
        for step in steps:
            stat = step["shared_clocks"][index]["top1"]
            cells.append(f"<td>{_count(stat)}</td>")
        base = clock["base"]
        clock_rows.append(
            f"<tr><th>{clock['clock']}</th><td>{base['hits']}/{base['n']} = {_pct(base['rate'])}</td>{''.join(cells)}</tr>"
        )
    return f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>全天滚动的 2 小时窗口</title>
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
<h1>2 小时窗口沿常规盘滚动</h1>
<p>输入仍是截止决策时刻的 2 小时、5 分钟 K 线，每根保留涨跌、振幅、成交量占前一常规盘的比例、跳空。09:35 之前的位置留空，第一根常规 K 线相对前一常规盘收盘计算跳空。入场价是决策那根的收盘价。标签是其后常规盘最高价，收到决策日之后第 2 个交易日，有没有碰到 +5%。每个时刻在当时有 K 线的股票里留概率最高的一只，不把全天收成一次预报。</p>
<p>偏移量是两次决策的间隔。5 分钟就是每根常规 K 线收盘都跑一次，从 09:35 到 16:00。15、30、60 分钟是同一套对齐到 09:30 的较稀时刻，各自单独训练。相邻时刻共用后面的走势，所以 5 分钟那一列的预报次数很多，但不是相互独立的样本。考试日是 2026-08-18 至 2026-09-29。树仍是 40 棵、深度 3、7 叶。</p>
<h2>四种偏移</h2>
<div class="scroll">
<table>
<thead><tr><th>偏移</th><th>每天时刻数</th><th>每次跑都留一只</th><th>该时刻全部股票</th><th>按天平均</th><th>只看 6 个共同时刻</th><th>训练分前 20%</th><th>训练 AUC</th></tr></thead>
<tbody>{''.join(body)}</tbody>
</table>
</div>
<h2>每 5 分钟跑一次，按小时</h2>
<table>
<thead><tr><th>时段</th><th>每次留一只</th><th>该时段全部股票</th></tr></thead>
<tbody>{''.join(hour_rows)}</tbody>
</table>
<h2>10:30、11:30、12:30、13:30、14:30、15:30</h2>
<p>这 6 个时刻四种偏移都有。同一时刻的标签相同，底只列一次。</p>
<div class="scroll">
<table>
<thead><tr><th>时刻</th><th>全部股票</th>{clock_head}</tr></thead>
<tbody>{''.join(clock_rows)}</tbody>
</table>
</div>
</main>
</body>
</html>
"""


def main() -> None:
    prior = json.loads(PRIOR_PATH.read_text())
    days = [report["session_date"] for report in prior["reports"]]
    names = _feature_names({"window": WINDOW, "bar": BAR})
    if CACHE_META.exists() and CACHE_X.exists():
        meta = pd.read_parquet(CACHE_META)
        values = np.load(CACHE_X)
        print(json.dumps({"cached_rows": int(len(meta))}), flush=True)
    else:
        calendar = load_calendar()
        books = {symbol: Book(symbol, load_market(symbol), calendar) for symbol in union_symbols()}
        _audit(books)
        meta, values, names = _matrix(books)
        del books
        meta["final_day"] = meta["final_day"].fillna("").astype(str)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        meta.to_parquet(CACHE_META, index=False)
        np.save(CACHE_X, values)
        print(json.dumps({"rows": int(len(meta)), "features": len(names)}), flush=True)
    steps = []
    out = OUT_DIR / "roll.json"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for step in STEPS:
        result = _run_step(meta, values, names, days, step)
        steps.append(result)
        out.write_text(json.dumps({"steps": steps}, ensure_ascii=False))
        print(
            json.dumps(
                {
                    "done": step,
                    "top1": result["top1"],
                    "base": result["base"],
                    "shared": result["shared"],
                    "day_mean": result["day_mean_rate"],
                }
            ),
            flush=True,
        )
    HTML_PATH.write_text(render(steps))
    print(json.dumps({"html": str(HTML_PATH)}), flush=True)


if __name__ == "__main__":
    main()
