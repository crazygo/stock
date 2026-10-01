"""Choose how far back the same-clock hit rate looks.

Before 2026-08-18, the highest trailing hit rate beat the biggest 2-hour
move. Both tails of that move land in the high 30%s, so a single weight on
the move cannot use them. The 60-session rule on the already-scored 30 days
was 328/702 = 46.7%.

This file does not retune those facts. Each test day chooses one rule from
four, using the previous 20 sessions whose labels are all already complete.
The four rules are the trailing 20, 60, and 120 session hit rates, and the
60-session rate restricted to names in the outer half of today's 2-hour move.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.after_open_3d5pct.models.window_density_v1.lift import (
    MIN_RATE,
    OUT_DIR,
    PRIOR_PATH,
    VAL_SESSIONS,
    _clock_rows,
    _clocks,
    _hits,
    _leader,
    _load,
    _pct,
    _pick,
    _rate,
    _rate_book,
    _rate_dict,
)

HTML_PATH = Path(__file__).resolve().parent / "adapt.html"
WINDOWS = (20, 60, 120)
RULES = ("rate20", "rate60", "rate120", "tails")
# Tie-break prefers the already measured 60-session rate, then the smoother one.
PRIORITY = {"rate60": 0, "rate120": 1, "rate20": 2, "tails": 3}
LABELS = {
    "rate20": "最近 20 个已到期交易日",
    "rate60": "最近 60 个已到期交易日",
    "rate120": "最近 120 个已到期交易日",
    "tails": "60 日达成率，只留今天涨跌两端",
    "nested": "前 20 天在这四条里选",
}


def _attach_windows(frame: pd.DataFrame, book: dict) -> pd.DataFrame:
    work = frame.reset_index(drop=True)
    columns = {window: np.full(len(work), np.nan) for window in WINDOWS}
    for (symbol, clock), index in work.groupby(["symbol", "clock"], sort=False).groups.items():
        found = book.get((str(symbol), int(clock)))
        if found is None:
            continue
        finals, _sessions, labels = found
        totals = np.cumsum(labels)
        positions = np.asarray(index, dtype=int)
        stops = np.searchsorted(finals, work.loc[positions, "session_date"].to_numpy(dtype=str), side="left")
        for window in WINDOWS:
            starts = np.maximum(stops - window, 0)
            counts = stops - starts
            summed = np.zeros(len(stops), dtype=float)
            has_stop = stops > 0
            summed[has_stop] = totals[stops[has_stop] - 1]
            has_start = starts > 0
            summed[has_start] -= totals[starts[has_start] - 1]
            rate = np.full(len(stops), np.nan)
            ok = counts >= MIN_RATE
            rate[ok] = summed[ok] / counts[ok]
            columns[window][positions] = rate
    for window, values in columns.items():
        work[f"rate{window}"] = values
    return work


def _apply(today: pd.DataFrame, rule: str) -> pd.DataFrame:
    if rule == "tails":
        work = today.loc[np.isfinite(today["rate60"]) & np.isfinite(today["z_ret"])].copy()
        if work.empty:
            return work
        cutoff = work.groupby("clock")["z_ret"].transform(lambda series: series.abs().median())
        work = work.loc[work["z_ret"].abs() >= cutoff]
        return _pick(work, work["rate60"])
    column = {"rate20": "rate20", "rate60": "rate60", "rate120": "rate120"}[rule]
    return _pick(today, today[column])


def _score_days(parts: dict[str, pd.DataFrame], days: list[str], rule: str) -> tuple[list[int | None], dict[int, list[int | None]], dict[str, int]]:
    hits: list[int | None] = []
    clock_hits: dict[int, list[int | None]] = {clock: [] for clock in _clocks()}
    symbols: dict[str, int] = {}
    for day in days:
        picks = _apply(parts[day], rule)
        for item in picks.itertuples(index=False):
            hit = None if not bool(item.ready) or not np.isfinite(item.y) else int(item.y == 1)
            hits.append(hit)
            clock_hits[int(item.clock)].append(hit)
            if hit is not None:
                symbols[str(item.symbol)] = symbols.get(str(item.symbol), 0) + 1
    return hits, clock_hits, symbols


def _pack(name: str, frame: pd.DataFrame, days: list[str], hits: list[int | None], clock_hits: dict, symbols: dict) -> dict:
    return {
        "name": name,
        "label": LABELS[name],
        "top1": _rate_dict(hits),
        "clocks": _clock_rows(clock_hits, frame, days),
        "leader": _leader(symbols),
    }


def _nested(frame: pd.DataFrame, parts: dict[str, pd.DataFrame], days: list[str]) -> dict:
    labeled = frame.loc[frame["ready"].eq(True) & frame["y"].notna() & frame["final_day"].ne("")]
    matured = labeled.groupby("session_date", sort=True)["final_day"].max()
    sessions = list(matured.index)
    counts: dict[str, int] = {}
    hits: list[int | None] = []
    clock_hits: dict[int, list[int | None]] = {clock: [] for clock in _clocks()}
    symbols: dict[str, int] = {}
    for offset, day in enumerate(days, start=1):
        window = [item for item in sessions if item < day and matured[item] < day][-VAL_SESSIONS:]
        best = "rate60"
        best_key = None
        if len(window) >= 8:
            for rule in RULES:
                val_hits: list[int | None] = []
                for val_day in window:
                    val_hits.extend(_hits(_apply(parts[val_day], rule)))
                stat = _rate_dict(val_hits)
                rank = (-1.0 if stat["rate"] is None else stat["rate"], -PRIORITY[rule])
                if best_key is None or rank > best_key:
                    best_key = rank
                    best = rule
        counts[best] = counts.get(best, 0) + 1
        for item in _apply(parts[day], best).itertuples(index=False):
            hit = None if not bool(item.ready) or not np.isfinite(item.y) else int(item.y == 1)
            hits.append(hit)
            clock_hits[int(item.clock)].append(hit)
            if hit is not None:
                symbols[str(item.symbol)] = symbols.get(str(item.symbol), 0) + 1
        print(json.dumps({"nested": day, "step": offset, "rule": best}), flush=True)
    packed = _pack("nested", frame, days, hits, clock_hits, symbols)
    packed["rules"] = {rule: counts.get(rule, 0) for rule in RULES}
    return packed


def _train_top1(frame: pd.DataFrame) -> dict[str, dict]:
    work = frame.copy()
    late = work["final_day"].eq("") | ~work["final_day"].lt("2026-08-18")
    work.loc[late, "y"] = np.nan
    work.loc[late, "ready"] = False
    train_days = sorted(set(work.loc[work["ready"].eq(True), "session_date"]))
    parts = {str(key): value for key, value in work.groupby("session_date", sort=False)}
    out = {}
    for rule in RULES:
        hits, _clocks_ignored, _symbols = _score_days(parts, train_days, rule)
        out[rule] = _rate_dict(hits)
        print(json.dumps({"train": rule, "top1": out[rule]}), flush=True)
    return out


def render(procedures: list[dict], train: dict[str, dict]) -> str:
    body = []
    for item in procedures:
        leader = item["leader"]
        who = "—" if leader is None else f"{leader['symbol']} {leader['picks']} 次"
        stat = item["top1"]
        body.append(
            f"<tr><th>{item['label']}</th><td>{stat['hits']}/{stat['mature']} = {_pct(stat['rate'])}</td><td>{stat['forecasts']}</td><td>{who}</td></tr>"
        )
    nested = next(item for item in procedures if item["name"] == "nested")
    used = "，".join(f"{LABELS[rule]} {count} 天" for rule, count in nested["rules"].items())
    head = "".join(f"<th>{item['label']}</th>" for item in procedures)
    clock_rows = []
    for index, clock in enumerate(procedures[0]["clocks"]):
        cells = "".join(
            f"<td>{item['clocks'][index]['top1']['hits']}/{item['clocks'][index]['top1']['mature']}</td>" for item in procedures
        )
        base = clock["base"]
        clock_rows.append(f"<tr><th>{clock['clock']}</th><td>{base['hits']}/{base['n']}</td>{cells}</tr>")
    train_line = "，".join(
        f"{LABELS[rule]} {train[rule]['hits']}/{train[rule]['mature']} = {_pct(train[rule]['rate'])}" for rule in RULES
    )
    return f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>达成率回溯长度</title>
<style>
body {{ margin: 0; padding: 24px; font-family: sans-serif; color: #111827; background: #fff; }}
main {{ max-width: 1100px; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border-bottom: 1px solid #d1d5db; text-align: left; padding: 8px; vertical-align: top; }}
.scroll {{ overflow-x: auto; }}
</style>
</head>
<body>
<main>
<h1>每次留一只，改达成率要往回看多久</h1>
<p>间隔 15 分钟，09:45 到 16:00，每个时刻仍只留一只。原先按振幅挑的小树是 317/702 = 45.2%。只看该股票这个时刻最近 60 个已到期交易日的达成率，是 328/702 = 46.7%。这 30 天里 AXTI 自己的达成率是 54.8%，21 只里至少有一只碰到 +5% 是 622/702 = 88.6%。</p>
<p>考试日之前，同样的 60 日规则是 {train['rate60']['hits']}/{train['rate60']['mature']} = {_pct(train['rate60']['rate'])}。当天 2 小时涨跌最大或最小的那一只，在考试日前大约只有 47%。所以这一轮不改树，只改往回看的长度。每天用的规则，由那天之前最近 20 个已经全部到期的交易日选出。四条固定规则的考试数字写在下面，供对照；当天的标签不参与选择。选择次数：{used}。</p>
<p>考试日之前四条规则：{train_line}。</p>
<h2>2026-08-18 至 2026-09-29</h2>
<table>
<thead><tr><th>方法</th><th>每次留一只</th><th>发出预报的时刻</th><th>留得最多的股票</th></tr></thead>
<tbody>{''.join(body)}</tbody>
</table>
<h2>每个时刻的命中数</h2>
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
    book = _rate_book(loaded)
    frame = _attach_windows(loaded, book)
    check = frame.sample(30, random_state=3566)
    for item in check.itertuples(index=False):
        expect = _rate(book, str(item.symbol), int(item.clock), str(item.session_date))
        got = float(item.rate60)
        if np.isfinite(expect) or np.isfinite(got):
            if not np.isfinite(expect) or not np.isfinite(got) or abs(expect - got) > 1e-9:
                raise RuntimeError("60-session rate does not match the earlier definition")
    parts = {str(key): value for key, value in frame.groupby("session_date", sort=False)}
    train = _train_top1(frame)
    procedures = []
    for rule in RULES:
        hits, clock_hits, symbols = _score_days(parts, days, rule)
        procedures.append(_pack(rule, frame, days, hits, clock_hits, symbols))
        print(json.dumps({"done": rule, "top1": procedures[-1]["top1"], "leader": procedures[-1]["leader"]}, ensure_ascii=False), flush=True)
    procedures.append(_nested(frame, parts, days))
    print(json.dumps({"done": "nested", "top1": procedures[-1]["top1"], "leader": procedures[-1]["leader"], "rules": procedures[-1]["rules"]}, ensure_ascii=False), flush=True)
    payload = {"train": train, "procedures": procedures}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "adapt.json").write_text(json.dumps(payload, ensure_ascii=False))
    HTML_PATH.write_text(render(procedures, train))
    print(json.dumps({"html": str(HTML_PATH)}), flush=True)


if __name__ == "__main__":
    main()
