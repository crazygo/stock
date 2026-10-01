"""Window length and bar size for a 3-day +5% touch.

Each row is one stock at one decision clock. Features are the bars inside a
fixed lookback: return, range, volume share, and gap, one slot per bar.
Missing bars stay missing. The entry is the close of the bar that ends at the
clock, and the label is the same 3-session regular-high +5% used by the 09:40
model. The 2-hour, 5-minute cell at 11:30 is the proposal. The other cells
only change the window or the bar size. Cells that share a clock share the
entry price and the label.

The archive has 5-minute bars. Coarser bars are sums of those bars. A bar is
kept only when every 5-minute piece is present.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.after_open_3d5pct.models.open_0940_touch_v1.backtest_30d import (
    MIN_TRAIN,
    _session_complete,
    load_market,
)
from research.after_open_3d5pct.models.pugh_ge5.build_rows import ROOT, load_calendar, union_symbols

OUT_DIR = ROOT / "research/after_open_3d5pct/runs/window_density_v1"
HTML_PATH = Path(__file__).resolve().parent / "report.html"
PRIOR_PATH = ROOT / "research/after_open_3d5pct/runs/open_0940_touch_v1/backtest_30d_bars.json"
SLOT0 = 9 * 60 + 35
N_SLOT = (16 * 60 - SLOT0) // 5 + 1
FIELDS = ("ret", "range", "vol", "gap")
CELLS = [
    {"name": "w120_m5", "label": "2 小时，5 分钟", "window": 120, "bar": 5, "clock": 11 * 60 + 30},
    {"name": "w30_m5", "label": "30 分钟，5 分钟", "window": 30, "bar": 5, "clock": 11 * 60 + 30},
    {"name": "w60_m5", "label": "60 分钟，5 分钟", "window": 60, "bar": 5, "clock": 11 * 60 + 30},
    {"name": "w120_m15", "label": "2 小时，15 分钟", "window": 120, "bar": 15, "clock": 11 * 60 + 30},
    {"name": "w120_m30", "label": "2 小时，30 分钟", "window": 120, "bar": 30, "clock": 11 * 60 + 30},
    {"name": "w120_m60", "label": "2 小时，60 分钟", "window": 120, "bar": 60, "clock": 11 * 60 + 30},
    {"name": "w240_m5", "label": "4 小时，5 分钟", "window": 240, "bar": 5, "clock": 13 * 60 + 30},
    {"name": "w120_m5_1330", "label": "2 小时，5 分钟，13:30", "window": 120, "bar": 5, "clock": 13 * 60 + 30},
]


class Book:
    def __init__(self, symbol: str, bars: pd.DataFrame, calendar: pd.DataFrame) -> None:
        regular = bars.loc[bars["session_type"].eq("regular")].sort_values("end")
        regular = regular.drop_duplicates("end", keep="last")
        self.symbol = symbol
        self.day = regular["day"].to_numpy()
        self.end = regular["end"].to_numpy(dtype="datetime64[ns]")
        self.minute = (regular["end"].dt.hour * 60 + regular["end"].dt.minute).to_numpy(np.int16)
        self.open = regular["open"].to_numpy(float)
        self.high = regular["high"].to_numpy(float)
        self.low = regular["low"].to_numpy(float)
        self.close = regular["close"].to_numpy(float)
        self.volume = regular["volume"].to_numpy(float)
        ordered: list[str] = []
        seen: set[str] = set()
        last_pos: dict[str, int] = {}
        for index, day in enumerate(self.day.tolist()):
            text = str(day)
            if text not in seen:
                seen.add(text)
                ordered.append(text)
            last_pos[text] = index
        self.days = ordered
        self.day_index = {day: index for index, day in enumerate(ordered)}
        self.last_pos = np.array([last_pos[day] for day in ordered], dtype=np.int32)
        self.slots = np.full((len(ordered), N_SLOT), -1, dtype=np.int32)
        for index, (day, minute) in enumerate(zip(self.day.tolist(), self.minute.tolist())):
            minute = int(minute)
            if minute < SLOT0 or minute > 16 * 60 or (minute - SLOT0) % 5:
                continue
            self.slots[self.day_index[str(day)], (minute - SLOT0) // 5] = index
        self.day_volume = np.zeros(len(ordered), dtype=float)
        for index in range(len(ordered)):
            present = self.slots[index]
            present = present[present >= 0]
            if len(present):
                self.day_volume[index] = float(self.volume[present].sum())
        counts = {day: int(np.count_nonzero(self.slots[index] >= 0)) for index, day in enumerate(ordered)}
        self.complete = _session_complete(counts, calendar)


def _features(book: Book, day_i: int, cell: dict) -> np.ndarray | None:
    n_bars = cell["window"] // cell["bar"]
    n5 = cell["bar"] // 5
    clock_slot = (cell["clock"] - SLOT0) // 5
    if book.slots[day_i, clock_slot] < 0:
        return None
    values = np.full(n_bars * len(FIELDS), np.nan, dtype=float)
    anchor_minute = cell["clock"] - cell["window"]
    if anchor_minute < SLOT0:
        anchor = float(book.close[book.last_pos[day_i - 1]]) if day_i else np.nan
    else:
        anchor_slot = (anchor_minute - SLOT0) // 5
        anchor_index = int(book.slots[day_i, anchor_slot]) if 0 <= anchor_slot < N_SLOT else -1
        anchor = float(book.close[anchor_index]) if anchor_index >= 0 else np.nan
    previous_volume = float(book.day_volume[day_i - 1]) if day_i else np.nan
    for slot in range(n_bars):
        end_minute = cell["clock"] - (n_bars - 1 - slot) * cell["bar"]
        if end_minute < SLOT0:
            continue
        end_slot = (end_minute - SLOT0) // 5
        start_slot = end_slot - n5 + 1
        if start_slot < 0 or end_slot >= N_SLOT:
            anchor = np.nan
            continue
        indexes = book.slots[day_i, start_slot : end_slot + 1]
        if np.any(indexes < 0):
            anchor = np.nan
            continue
        opened = float(book.open[indexes[0]])
        closed = float(book.close[indexes[-1]])
        base = slot * len(FIELDS)
        if opened > 0 and closed > 0:
            values[base] = closed / opened - 1
            values[base + 1] = (float(book.high[indexes].max()) - float(book.low[indexes].min())) / opened
        if previous_volume > 0:
            values[base + 2] = float(book.volume[indexes].sum()) / previous_volume
        if np.isfinite(anchor) and anchor > 0 and opened > 0:
            values[base + 3] = opened / anchor - 1
        anchor = closed if closed > 0 else np.nan
    return values


def _label(book: Book, day_i: int, cell: dict) -> tuple[str | None, bool, float]:
    if day_i + 2 >= len(book.days):
        return None, False, np.nan
    final_day = book.days[day_i + 2]
    entry_pos = int(book.slots[day_i, (cell["clock"] - SLOT0) // 5])
    left = entry_pos + 1
    right = int(book.last_pos[day_i + 2]) + 1
    window = book.high[left:right]
    ready = bool(
        final_day in book.complete
        and len(window)
        and str(book.day[left]) >= book.days[day_i]
        and str(book.day[right - 1]) <= final_day
    )
    if not ready:
        return final_day, False, np.nan
    entry = float(book.close[entry_pos])
    if entry <= 0:
        return final_day, False, np.nan
    return final_day, True, float(np.max(window) >= entry * 1.05)


def _feature_names(cell: dict) -> list[str]:
    names = []
    for slot in range(cell["window"] // cell["bar"]):
        for field in FIELDS:
            names.append(f"s{slot:02d}_{field}")
    return names


def build_frame(books: dict[str, Book], cell: dict) -> pd.DataFrame:
    names = _feature_names(cell)
    meta_rows = []
    matrix = []
    for book in books.values():
        for day_i, day in enumerate(book.days):
            features = _features(book, day_i, cell)
            if features is None:
                continue
            final_day, ready, hit = _label(book, day_i, cell)
            meta_rows.append(
                {
                    "symbol": book.symbol,
                    "session_date": day,
                    "final_day": final_day,
                    "ready": ready,
                    "y": hit,
                }
            )
            matrix.append(features)
    frame = pd.DataFrame(meta_rows)
    values = np.vstack(matrix) if matrix else np.zeros((0, len(names)))
    for index, name in enumerate(names):
        frame[name] = values[:, index]
    return frame


def _audit(books: dict[str, Book]) -> None:
    book = books.get("AMD")
    if book is None or "2026-08-18" not in book.day_index:
        print(json.dumps({"audit": "skipped"}), flush=True)
        return
    day_i = book.day_index["2026-08-18"]
    cell = CELLS[0]
    features = _features(book, day_i, cell)
    if features is None:
        raise RuntimeError("AMD 2026-08-18 has no 11:30 bar")
    entry_pos = int(book.slots[day_i, (cell["clock"] - SLOT0) // 5])
    opened = float(book.open[entry_pos])
    closed = float(book.close[entry_pos])
    last_ret = features[-len(FIELDS)]
    if opened <= 0 or not np.isclose(last_ret, closed / opened - 1):
        raise RuntimeError("last 5-minute slot does not match the 11:30 bar")
    first_pos = int(book.slots[day_i, 0])
    first_open = float(book.open[first_pos])
    first_close = float(book.close[first_pos])
    if first_open <= 0 or not np.isclose(features[0], first_close / first_open - 1):
        raise RuntimeError("first slot does not match the 09:35 bar")
    coarse = next(item for item in CELLS if item["name"] == "w120_m60")
    coarse_features = _features(book, day_i, coarse)
    if coarse_features is None:
        raise RuntimeError("AMD 2026-08-18 has no 60-minute window")
    hour_start = int(book.slots[day_i, (10 * 60 + 35 - SLOT0) // 5])
    hour_open = float(book.open[hour_start])
    if hour_open <= 0 or not np.isclose(coarse_features[len(FIELDS)], closed / hour_open - 1):
        raise RuntimeError("60-minute bar does not span 10:30 to 11:30")
    print(json.dumps({"audit": "AMD 2026-08-18", "bars": 24, "last_ret": float(last_ret)}), flush=True)


def _same_labels(frames: dict[str, pd.DataFrame]) -> None:
    groups = {}
    for cell in CELLS:
        groups.setdefault(cell["clock"], []).append(cell["name"])
    for names in groups.values():
        base = frames[names[0]].set_index(["symbol", "session_date"])["y"]
        for name in names[1:]:
            other = frames[name].set_index(["symbol", "session_date"])["y"]
            if not base.index.equals(other.index):
                raise RuntimeError(f"{name} does not have the same rows as {names[0]}")
            left = base.to_numpy(float)
            right = other.to_numpy(float)
            if not np.array_equal(np.isnan(left), np.isnan(right)) or not np.allclose(left, right, equal_nan=True):
                raise RuntimeError(f"{name} label drifted from {names[0]}")


def _rate(hits: list[int | None]) -> dict:
    mature = [item for item in hits if item is not None]
    got = sum(int(item) for item in mature)
    return {"forecasts": len(hits), "mature": len(mature), "hits": got, "rate": None if not mature else got / len(mature)}


def _train_before(frame: pd.DataFrame, day: str) -> pd.DataFrame:
    return frame.loc[frame["ready"].eq(True) & frame["final_day"].astype(str).lt(day) & frame["y"].notna()]


def _run_cell(frame: pd.DataFrame, days: list[str], cell: dict) -> dict:
    import lightgbm as lgb

    names = _feature_names(cell)
    n_bars = cell["window"] // cell["bar"]
    gain = np.zeros(len(names))
    train_aucs: list[float] = []
    day_rows = []
    top_hits: list[int | None] = []
    selected_hits: list[int | None] = []
    base_hits = 0
    base_n = 0
    for offset, day in enumerate(days, start=1):
        train = _train_before(frame, day)
        today = frame.loc[frame["session_date"].eq(day)].copy()
        mature = today.loc[today["ready"].eq(True) & today["y"].notna()]
        base_hits += int(mature["y"].sum()) if len(mature) else 0
        base_n += int(len(mature))
        if len(train) < MIN_TRAIN or train["y"].nunique() < 2 or today.empty:
            day_rows.append({"session_date": day, "pick": None, "candidates": int(len(today))})
            print(json.dumps({"cell": cell["name"], "day": day, "step": offset, "skipped": True}), flush=True)
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
        model.fit(train[names], y)
        trained = model.predict_proba(train[names])[:, 1]
        score = _auc(y, trained)
        if score is not None:
            train_aucs.append(float(score))
        gain += model.booster_.feature_importance(importance_type="gain")
        today["score"] = model.predict_proba(today[names])[:, 1]
        today = today.sort_values(["score", "symbol"], ascending=[False, True])
        pick = today.iloc[0]
        pick_hit = None if not bool(pick["ready"]) else int(pick["y"] == 1)
        top_hits.append(pick_hit)
        threshold = float(np.quantile(trained, 0.80))
        chosen = today.loc[today["score"].ge(threshold)]
        for item in chosen.itertuples(index=False):
            selected_hits.append(None if not bool(item.ready) else int(item.y == 1))
        day_rows.append(
            {
                "session_date": day,
                "candidates": int(len(today)),
                "base_hits": int(mature["y"].sum()) if len(mature) else 0,
                "base_n": int(len(mature)),
                "pick": {
                    "symbol": str(pick["symbol"]),
                    "probability": float(pick["score"]),
                    "hit": pick_hit,
                },
            }
        )
        print(json.dumps({"cell": cell["name"], "day": day, "step": offset, "hit": pick_hit}), flush=True)
    total = float(gain.sum()) or 1.0
    field_share = {field: 0.0 for field in FIELDS}
    half_share = {"early": 0.0, "late": 0.0}
    for index, name in enumerate(names):
        share = float(gain[index] / total)
        field_share[name.rsplit("_", 1)[1]] += share
        slot = int(name[1:3])
        half_share["early" if slot < n_bars / 2 else "late"] += share
    order = np.argsort(gain)[::-1][:5]
    top = [
        {"feature": names[int(pos)], "gain_share": float(gain[int(pos)] / total)}
        for pos in order
        if gain[int(pos)] > 0
    ]
    return {
        "name": cell["name"],
        "label": cell["label"],
        "window": cell["window"],
        "bar": cell["bar"],
        "clock": f"{cell['clock'] // 60:02d}:{cell['clock'] % 60:02d}",
        "bars": n_bars,
        "features": len(names),
        "mean_train_auc": None if not train_aucs else float(np.mean(train_aucs)),
        "top1": _rate(top_hits),
        "selected": _rate(selected_hits),
        "base": {"hits": base_hits, "n": base_n, "rate": None if not base_n else base_hits / base_n},
        "gain_fields": field_share,
        "gain_half": half_share,
        "top_features": top,
        "days": day_rows,
    }


def _auc(label: np.ndarray, score: np.ndarray) -> float | None:
    order = np.argsort(score, kind="mergesort")
    y = label[order]
    pos = int(y.sum())
    neg = int(len(y) - pos)
    if pos == 0 or neg == 0:
        return None
    ranks = np.empty(len(y), dtype=float)
    ranks[order] = np.arange(1, len(y) + 1)
    sorted_score = score[order]
    start = 0
    while start < len(y):
        stop = start + 1
        while stop < len(y) and sorted_score[stop] == sorted_score[start]:
            stop += 1
        if stop - start > 1:
            ranks[order[start:stop]] = 0.5 * (start + 1 + stop)
        start = stop
    sum_pos = float(ranks[label == 1].sum())
    return (sum_pos - pos * (pos + 1) / 2) / (pos * neg)


def _pct(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    return f"{value * 100:.1f}%"


def render(cells: list[dict]) -> str:
    body = []
    for cell in cells:
        top = cell["top1"]
        selected = cell["selected"]
        base = cell["base"]
        body.append(
            "<tr>"
            f"<th>{cell['label']}</th>"
            f"<td>{cell['clock']}</td>"
            f"<td>{cell['bars']}</td>"
            f"<td>{top['hits']}/{top['mature']} = {_pct(top['rate'])}</td>"
            f"<td>{base['hits']}/{base['n']} = {_pct(base['rate'])}</td>"
            f"<td>{selected['hits']}/{selected['mature']} = {_pct(selected['rate'])}</td>"
            f"<td>{_pct(cell['mean_train_auc'])}</td>"
            f"<td>振幅 {_pct(cell['gain_fields']['range'])}，涨跌 {_pct(cell['gain_fields']['ret'])}，"
            f"量 {_pct(cell['gain_fields']['vol'])}，跳空 {_pct(cell['gain_fields']['gap'])}</td>"
            "</tr>"
        )
    primary = cells[0]
    day_rows = []
    for row in primary["days"]:
        pick = row["pick"]
        if pick is None:
            day_rows.append(f"<tr><th>{row['session_date']}</th><td>{row['candidates']}</td><td>无预报</td></tr>")
            continue
        word = "未到期" if pick["hit"] is None else ("达成" if pick["hit"] else "未达成")
        base = "—" if not row.get("base_n") else f"{row['base_hits']}/{row['base_n']}"
        day_rows.append(
            f"<tr><th>{row['session_date']}</th><td>{row['candidates']}</td>"
            f"<td>{pick['symbol']} · {_pct(pick['probability'])} · {word} · 当天 {base}</td></tr>"
        )
    return f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>窗口长度和 K 线粗细</title>
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
<h1>用一段 K 线窗口预测 3 天 +5%</h1>
<p>每一行是一只股票在一个时刻的窗口。特征是窗口里每一根 K 线的涨跌、振幅、成交量占前一常规盘的比例、和相对前一根的跳空。缺的 K 线留空，后面的不往前补。入场价是决策时刻那根 K 线的收盘价。标签是这根收盘之后的常规盘最高价，收到决策日之后第 2 个交易日，有没有碰到 +5%。当天把 21 只里概率最高的一只留下。树是 40 棵、深度 3、7 叶。考试日是 2026-08-18 至 2026-09-29。</p>
<p>提议的输入是 2 小时、5 分钟，决策在 11:30，也就是开盘后第一段完整的 2 小时。同一决策时刻的标签相同，所以命中差只来自窗口。4 小时要到 13:30 才覆盖得满，因此和另一组 2 小时、5 分钟一起放在 13:30。</p>
<h2>八组输入</h2>
<div class="scroll">
<table>
<thead><tr><th>输入</th><th>决策</th><th>根数</th><th>每天留一只</th><th>当天全部股票</th><th>训练分前 20%</th><th>训练 AUC</th><th>分裂增益</th></tr></thead>
<tbody>{''.join(body)}</tbody>
</table>
</div>
<h2>2 小时、5 分钟，11:30，每天留一只</h2>
<table>
<thead><tr><th>交易日</th><th>只数</th><th>最高概率</th></tr></thead>
<tbody>{''.join(day_rows)}</tbody>
</table>
</main>
</body>
</html>
"""


def main() -> None:
    for cell in CELLS:
        if cell["window"] % cell["bar"] or cell["bar"] % 5:
            raise RuntimeError(f"{cell['name']} does not divide into 5-minute bars")
    prior = json.loads(PRIOR_PATH.read_text())
    days = [report["session_date"] for report in prior["reports"]]
    calendar = load_calendar()
    books = {}
    for symbol in union_symbols():
        books[symbol] = Book(symbol, load_market(symbol), calendar)
        print(json.dumps({"symbol": symbol, "days": len(books[symbol].days)}), flush=True)
    _audit(books)
    frames = {}
    for cell in CELLS:
        frames[cell["name"]] = build_frame(books, cell)
        print(json.dumps({"frame": cell["name"], "rows": int(len(frames[cell["name"]]))}), flush=True)
    _same_labels(frames)
    cells = []
    out = OUT_DIR / "window_density.json"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for cell in CELLS:
        result = _run_cell(frames[cell["name"]], days, cell)
        cells.append(result)
        out.write_text(json.dumps({"cells": cells}, ensure_ascii=False))
        print(
            json.dumps(
                {
                    "done": cell["name"],
                    "top1": result["top1"],
                    "base": result["base"],
                    "selected": result["selected"],
                    "train_auc": result["mean_train_auc"],
                }
            ),
            flush=True,
        )
    HTML_PATH.write_text(render(cells))
    print(json.dumps({"html": str(HTML_PATH)}), flush=True)


if __name__ == "__main__":
    main()
