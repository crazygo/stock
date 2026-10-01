"""Raise the one-name pick above the name's own hit rate.

The 2-hour window's tree spends its splits on absolute range, so the highest
score sticks to the most volatile name. On 2026-08-18 to 2026-09-29 that name's
own 15-minute hit rate is about 55%, and 88.6% of clocks have some other name
that does touch +5%.

Two walk-forward procedures use only information available at the clock:
the stock's own hit rate over the previous 60 sessions at that same clock, and
how today's window compares with the other names. The linear score's two
weights are chosen on the previous 20 matured sessions. The tree is the same
40-tree, depth-3 model, fit only on rows whose label is already complete.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.after_open_3d5pct.models.open_0940_touch_v1.backtest_30d import MIN_TRAIN
from research.after_open_3d5pct.models.pugh_ge5.build_rows import ROOT

OUT_DIR = ROOT / "research/after_open_3d5pct/runs/window_density_v1"
HTML_PATH = Path(__file__).resolve().parent / "lift.html"
PRIOR_PATH = ROOT / "research/after_open_3d5pct/runs/open_0940_touch_v1/backtest_30d_bars.json"
CACHE_META = OUT_DIR / "roll_meta.parquet"
CACHE_X = OUT_DIR / "roll_x.npy"
RECENT = 60
MIN_RATE = 15
VAL_SESSIONS = 20
WEIGHTS = (-1.0, 0.0, 1.0)
FIELDS = ("rate", "z_ret", "z_recent", "z_range", "range_ratio", "window_ret")


def _clocks(step: int = 15) -> list[int]:
    return list(range(9 * 60 + 30 + step, 16 * 60 + 1, step))


def _hhmm(minute: int) -> str:
    return f"{minute // 60:02d}:{minute % 60:02d}"


def _load() -> pd.DataFrame:
    meta = pd.read_parquet(CACHE_META)
    values = np.load(CACHE_X)
    if len(meta) != len(values):
        raise RuntimeError("roll cache row count does not match the feature matrix")
    keep = meta["clock"].isin(_clocks()).to_numpy()
    frame = meta.loc[keep].reset_index(drop=True)
    data = values[keep]
    rets = data[:, 0::4].astype(float)
    ranges = data[:, 1::4].astype(float)
    finite_ret = np.isfinite(rets)
    finite_range = np.isfinite(ranges)
    window_ret = np.where(finite_ret.any(axis=1), np.nansum(rets, axis=1), np.nan)
    recent = rets[:, -6:]
    finite_recent = np.isfinite(recent)
    window_recent = np.where(finite_recent.any(axis=1), np.nansum(recent, axis=1), np.nan)
    window_range = np.where(finite_range.any(axis=1), np.nanmean(ranges, axis=1), np.nan)
    frame["window_ret"] = window_ret
    frame["window_recent"] = window_recent
    frame["window_range"] = window_range
    frame["final_day"] = frame["final_day"].fillna("").astype(str)
    frame["session_date"] = frame["session_date"].astype(str)
    frame["symbol"] = frame["symbol"].astype(str)
    frame = frame.sort_values(["symbol", "clock", "session_date"]).reset_index(drop=True)
    grouped = frame.groupby(["symbol", "clock"], sort=False)["window_range"]
    frame["own_median"] = grouped.transform(lambda series: series.shift(1).rolling(20, min_periods=10).median())
    frame["range_ratio"] = frame["window_range"] / frame["own_median"]
    frame.loc[~np.isfinite(frame["range_ratio"]), "range_ratio"] = np.nan
    for source, target in (
        ("window_ret", "z_ret"),
        ("window_recent", "z_recent"),
        ("range_ratio", "z_range"),
    ):
        group = frame.groupby(["session_date", "clock"], sort=False)[source]
        center = group.transform("mean")
        spread = group.transform("std")
        score = (frame[source] - center) / spread
        score = score.where(spread > 1e-8, 0.0)
        frame[target] = score.fillna(0.0)
    return frame


def _rate_book(frame: pd.DataFrame) -> dict[tuple[str, int], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    book = {}
    ready = frame.loc[frame["ready"].eq(True) & frame["y"].notna() & frame["final_day"].ne("")]
    for (symbol, clock), part in ready.groupby(["symbol", "clock"], sort=False):
        ordered = part.sort_values("final_day")
        book[(str(symbol), int(clock))] = (
            ordered["final_day"].to_numpy(dtype=str),
            ordered["session_date"].to_numpy(dtype=str),
            ordered["y"].to_numpy(float),
        )
    return book


def _rate(book: dict, symbol: str, clock: int, before_day: str) -> float:
    found = book.get((symbol, clock))
    if found is None:
        return np.nan
    finals, _sessions, labels = found
    stop = int(np.searchsorted(finals, before_day, side="left"))
    start = max(0, stop - RECENT)
    if stop - start < MIN_RATE:
        return np.nan
    return float(labels[start:stop].mean())


def _z(frame: pd.DataFrame, column: str, weight: float) -> pd.Series:
    if weight == 0:
        return pd.Series(0.0, index=frame.index)
    return frame[column] * weight


def _pick(today: pd.DataFrame, score: pd.Series) -> pd.DataFrame:
    work = today.copy()
    work["score"] = score.to_numpy()
    work = work.loc[np.isfinite(work["score"])]
    if work.empty:
        return work
    return work.sort_values(["score", "symbol"], ascending=[False, True]).groupby("clock", sort=True).head(1)


def _hits(picks: pd.DataFrame) -> list[int | None]:
    out = []
    for item in picks.itertuples(index=False):
        if not bool(item.ready) or not np.isfinite(item.y):
            out.append(None)
        else:
            out.append(int(item.y == 1))
    return out


def _rate_dict(hits: list[int | None]) -> dict:
    mature = [item for item in hits if item is not None]
    got = sum(int(item) for item in mature)
    return {"forecasts": len(hits), "mature": len(mature), "hits": got, "rate": None if not mature else got / len(mature)}


def _attach_rates(frame: pd.DataFrame, book: dict) -> pd.DataFrame:
    """Hit rate at this row's own session, using only labels whose final day is already over."""
    work = frame.copy()
    work["rate"] = [
        _rate(book, symbol, int(clock), day)
        for symbol, clock, day in zip(work["symbol"], work["clock"], work["session_date"])
    ]
    return work


def _apply_linear(today: pd.DataFrame, weight_ret: float, weight_range: float) -> pd.DataFrame:
    score = today["rate"] + _z(today, "z_ret", weight_ret) + _z(today, "z_range", weight_range)
    return _pick(today, score)


def _nested(frame: pd.DataFrame, days: list[str]) -> dict:
    labeled = frame.loc[frame["ready"].eq(True) & frame["y"].notna() & frame["final_day"].ne("")]
    matured = labeled.groupby("session_date", sort=True)["final_day"].max()
    sessions = list(matured.index)
    counts: dict[tuple[float, float], int] = {}
    hits: list[int | None] = []
    clock_hits: dict[int, list[int | None]] = {clock: [] for clock in _clocks()}
    symbols: dict[str, int] = {}
    parts = {str(key): value for key, value in frame.groupby("session_date", sort=False)}
    for offset, day in enumerate(days, start=1):
        window = [item for item in sessions if item < day and matured[item] < day][-VAL_SESSIONS:]
        best = (0.0, 0.0)
        best_key = None
        if len(window) >= 8:
            for weight_ret in WEIGHTS:
                for weight_range in WEIGHTS:
                    val_hits: list[int | None] = []
                    for val_day in window:
                        picks = _apply_linear(parts[val_day], weight_ret, weight_range)
                        val_hits.extend(_hits(picks))
                    stat = _rate_dict(val_hits)
                    rank = (
                        -1.0 if stat["rate"] is None else stat["rate"],
                        -abs(weight_ret) - abs(weight_range),
                        -abs(weight_ret),
                        -abs(weight_range),
                    )
                    if best_key is None or rank > best_key:
                        best_key = rank
                        best = (weight_ret, weight_range)
        counts[best] = counts.get(best, 0) + 1
        picks = _apply_linear(parts[day], best[0], best[1])
        for item in picks.itertuples(index=False):
            hit = None if not bool(item.ready) or not np.isfinite(item.y) else int(item.y == 1)
            hits.append(hit)
            clock_hits[int(item.clock)].append(hit)
            if hit is not None:
                symbols[str(item.symbol)] = symbols.get(str(item.symbol), 0) + 1
        print(json.dumps({"nested": day, "step": offset, "weights": best}), flush=True)
    return {
        "name": "nested",
        "label": "最近达成率 + 当天相对强弱，权重看前 20 天",
        "top1": _rate_dict(hits),
        "clocks": _clock_rows(clock_hits, frame, days),
        "weights": {f"{pair[0]:g},{pair[1]:g}": count for pair, count in sorted(counts.items())},
        "leader": _leader(symbols),
    }


def _fixed(frame: pd.DataFrame, days: list[str], name: str, label: str, weight_ret: float, weight_range: float) -> dict:
    hits: list[int | None] = []
    clock_hits: dict[int, list[int | None]] = {clock: [] for clock in _clocks()}
    symbols: dict[str, int] = {}
    parts = {str(key): value for key, value in frame.groupby("session_date", sort=False)}
    for day in days:
        picks = _apply_linear(parts[day], weight_ret, weight_range)
        for item in picks.itertuples(index=False):
            hit = None if not bool(item.ready) or not np.isfinite(item.y) else int(item.y == 1)
            hits.append(hit)
            clock_hits[int(item.clock)].append(hit)
            if hit is not None:
                symbols[str(item.symbol)] = symbols.get(str(item.symbol), 0) + 1
    return {
        "name": name,
        "label": label,
        "top1": _rate_dict(hits),
        "clocks": _clock_rows(clock_hits, frame, days),
        "leader": _leader(symbols),
    }


def _tree(frame: pd.DataFrame, days: list[str]) -> dict:
    import lightgbm as lgb

    work = frame
    matrix = work.loc[:, list(FIELDS)].to_numpy(float)
    label = work["y"].to_numpy(float)
    final = work["final_day"].to_numpy(dtype=str)
    ready = work["ready"].to_numpy(bool) & np.isfinite(label) & np.array([item != "" for item in final])
    hits: list[int | None] = []
    clock_hits: dict[int, list[int | None]] = {clock: [] for clock in _clocks()}
    symbols: dict[str, int] = {}
    for offset, day in enumerate(days, start=1):
        train = ready & (final < day)
        today_mask = work["session_date"].eq(day).to_numpy()
        if int(train.sum()) < MIN_TRAIN or len(np.unique(label[train])) < 2 or not today_mask.any():
            print(json.dumps({"tree": day, "step": offset, "skipped": True}), flush=True)
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
        model.fit(matrix[train], label[train].astype(int))
        today = work.loc[today_mask].copy()
        today["score"] = model.predict_proba(matrix[today_mask])[:, 1]
        picks = today.sort_values(["score", "symbol"], ascending=[False, True]).groupby("clock", sort=True).head(1)
        for item in picks.itertuples(index=False):
            hit = None if not bool(item.ready) or not np.isfinite(item.y) else int(item.y == 1)
            hits.append(hit)
            clock_hits[int(item.clock)].append(hit)
            if hit is not None:
                symbols[str(item.symbol)] = symbols.get(str(item.symbol), 0) + 1
        print(json.dumps({"tree": day, "step": offset}), flush=True)
    return {
        "name": "tree",
        "label": "相对特征小树",
        "top1": _rate_dict(hits),
        "clocks": _clock_rows(clock_hits, work, days),
        "leader": _leader(symbols),
    }


def _leader(symbols: dict[str, int]) -> dict | None:
    if not symbols:
        return None
    symbol = max(symbols, key=lambda key: (symbols[key], key))
    return {"symbol": symbol, "picks": symbols[symbol]}


def _clock_rows(clock_hits: dict[int, list[int | None]], frame: pd.DataFrame, days: list[str]) -> list[dict]:
    rows = []
    test = frame.loc[frame["session_date"].isin(days) & frame["ready"].eq(True) & frame["y"].notna()]
    for clock in _clocks():
        part = test.loc[test["clock"].eq(clock)]
        stat = _rate_dict(clock_hits[clock])
        rows.append(
            {
                "clock": _hhmm(clock),
                "top1": stat,
                "base": {
                    "hits": int(part["y"].sum()) if len(part) else 0,
                    "n": int(len(part)),
                    "rate": None if part.empty else float(part["y"].mean()),
                },
            }
        )
    return rows


def _pct(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    return f"{value * 100:.1f}%"


def _count(stat: dict) -> str:
    return f"{stat['hits']}/{stat['mature']} = {_pct(stat['rate'])}"


def render(result: dict, procedures: list[dict]) -> str:
    body = []
    for item in procedures:
        leader = item["leader"]
        who = "—" if leader is None else f"{leader['symbol']} {leader['picks']} 次"
        body.append(
            f"<tr><th>{item['label']}</th><td>{_count(item['top1'])}</td><td>{who}</td></tr>"
        )
    head = "".join(f"<th>{item['label']}</th>" for item in procedures)
    clock_rows = []
    for index, clock in enumerate(procedures[0]["clocks"]):
        cells = "".join(f"<td>{_count(item['clocks'][index]['top1'])}</td>" for item in procedures)
        base = clock["base"]
        clock_rows.append(
            f"<tr><th>{clock['clock']}</th><td>{base['hits']}/{base['n']} = {_pct(base['rate'])}</td>{cells}</tr>"
        )
    nested = next(item for item in procedures if item["name"] == "nested")
    weights = "，".join(f"{key} 用了 {count} 天" for key, count in nested["weights"].items())
    oracle = result["oracle"]
    return f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>每次留一只的优化</title>
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
<h1>每次留一只，改成跟自己的历史比</h1>
<p>间隔 15 分钟，从 09:45 到 16:00。每个时刻仍只留一只。原先的小树主要按振幅挑股票，留下的达成率是 317/702 = 45.2%。这 30 天里，同一个时刻 21 只中至少有一只碰到 +5% 的次数是 {oracle['hits']}/{oracle['clocks']} = {_pct(oracle['rate'])}。单只股票里 AXTI 自己的达成率是 54.8%。</p>
<p>新的分数用该股票在这个时刻最近 60 个已到期交易日的达成率。后两列仍保留这个达成率，分别再加上今天这段 2 小时相对其他股票的涨跌、或相对振幅。嵌套在 9 组权重里选，选择用的是考试日之前最近 20 个已经全部到期的交易日，当天的标签不参与选权重。树用同样的相对特征，参数仍是 40 棵、深度 3、7 叶。权重选择：{weights}。</p>
<h2>全部时刻合在一起</h2>
<table>
<thead><tr><th>方法</th><th>每次留一只</th><th>留得最多的股票</th></tr></thead>
<tbody>{''.join(body)}</tbody>
</table>
<h2>每个时刻</h2>
<div class="scroll">
<table>
<thead><tr><th>时刻</th><th>全部股票</th>{head}</tr></thead>
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
    loaded = _load()
    frame = _attach_rates(loaded, _rate_book(loaded))
    test = frame.loc[frame["session_date"].isin(days) & frame["ready"].eq(True) & frame["y"].notna()]
    oracle_hits = int(test.groupby(["session_date", "clock"])["y"].max().sum())
    oracle_n = int(test.groupby(["session_date", "clock"]).ngroups)
    print(json.dumps({"rows": int(len(frame)), "oracle": [oracle_hits, oracle_n]}), flush=True)
    procedures = [
        _fixed(frame, days, "rate", "只看最近 60 天达成率", 0.0, 0.0),
        _fixed(frame, days, "momentum", "达成率 + 当天相对涨跌", 1.0, 0.0),
        _fixed(frame, days, "range", "达成率 + 当天相对振幅", 0.0, 1.0),
        _nested(frame, days),
        _tree(frame, days),
    ]
    for item in procedures:
        print(json.dumps({"done": item["name"], "top1": item["top1"], "leader": item["leader"]}, ensure_ascii=False), flush=True)
    payload = {
        "oracle": {"hits": oracle_hits, "clocks": oracle_n, "rate": oracle_hits / oracle_n},
        "procedures": procedures,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "lift.json").write_text(json.dumps(payload, ensure_ascii=False))
    HTML_PATH.write_text(render(payload, procedures))
    print(json.dumps({"html": str(HTML_PATH)}), flush=True)


if __name__ == "__main__":
    main()
