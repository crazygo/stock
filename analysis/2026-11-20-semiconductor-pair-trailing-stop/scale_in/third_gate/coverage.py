"""Partition paired, unadjusted SOXL/SOXS five-minute tapes by input coverage."""
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
ET = "America/New_York"
PHASES = {"development": ("2026-08-31", "2026-10-09"),
          "evaluation": ("2026-10-12", "2026-10-30"),
          "november": ("2026-11-02", "2026-11-19")}
SYMBOLS = ("SOXL", "SOXS")
FIVE_MINUTES = pd.Timedelta(minutes=5)


def fingerprint(frame):
    return hashlib.sha256(frame.to_csv(index=True, lineterminator="\n").encode()).hexdigest()


def inspect_symbol(symbol, calendar):
    path = ROOT/f"market_data/us_5m/{symbol}/2026.parquet"
    tape = pd.read_parquet(path, columns=["start_at", "end_at", "available_at",
                                          "open", "high", "low", "close", "volume", "price_basis"])
    tape.index = pd.to_datetime(tape.pop("start_at"), utc=True)
    actions_path = ROOT/f"market_data/corporate_actions/{symbol}.parquet"
    actions = pd.read_parquet(actions_path)
    ex = actions["ex_div_date"].astype(str)
    quality = {}
    for session in calendar:
        day = session["session_date"]
        start = pd.Timestamp(day, tz=ET) - pd.Timedelta(hours=4)
        grid = pd.date_range(start, periods=288, freq=FIVE_MINUTES).tz_convert("UTC")
        within = tape.loc[(tape.index >= grid[0]) & (tape.index < grid[-1] + FIVE_MINUTES)]
        duplicates = within.index[within.index.duplicated()].unique()
        offgrid = within.index.difference(grid)
        rows = within.loc[~within.index.duplicated()].reindex(grid)
        ohlc = rows[["open", "high", "low", "close"]]
        finite = np.isfinite(ohlc).all(axis=1)
        checks = {
            "missing_bar": ~grid.isin(within.index),
            "bad_ohlc": ~(finite & (ohlc > 0).all(axis=1)
                          & (ohlc.low <= ohlc[["open","close"]].min(axis=1))
                          & (ohlc.high >= ohlc[["open","close"]].max(axis=1))),
            "bad_volume": ~(rows.volume.gt(0) & np.isfinite(rows.volume)),
            "wrong_basis": ~rows.price_basis.eq("NONE").fillna(False),
            "bad_end": pd.to_datetime(rows.end_at,utc=True) != grid+FIVE_MINUTES,
            "bad_available_at": pd.to_datetime(rows.available_at,utc=True) != grid+FIVE_MINUTES+pd.Timedelta(seconds=1),
        }
        issues = {why:[t.tz_convert(ET).isoformat() for t in grid[mask]]
                  for why,mask in checks.items() if np.any(mask)}
        if len(duplicates):
            issues["duplicate_start"] = [t.tz_convert(ET).isoformat() for t in duplicates]
        if len(offgrid):
            issues["unexpected_start"] = [t.tz_convert(ET).isoformat() for t in offgrid]
        corporate = [record.dropna().to_dict() for _,record in actions.loc[ex.eq(day)].iterrows()]
        quality[day] = dict(bars=len(within), issues=issues, corporate_actions=corporate,
                            sha256=fingerprint(within))
    return quality, dict(bars=str(path.relative_to(ROOT)),actions=str(actions_path.relative_to(ROOT)))


def partition(phase, days, issues):
    result, skipped, streak = [], [], []

    def flush():
        if len(streak) >= 2:
            name = "november" if phase == "november" else f"{phase}_{streak[0]}"
            result.append(dict(name=name, start=streak[0], end=streak[-1],
                               dates=streak.copy(),days=len(streak)))
        elif streak:
            skipped.extend(streak)
        streak.clear()

    prior_week = None
    for day in days:
        week = datetime.fromisoformat(day).isocalendar()[:2]
        if phase != "november" and week != prior_week:
            flush()
        prior_week = week
        if day in issues:
            flush()
        else:
            streak.append(day)
    flush()
    return result, skipped


def build():
    path = ROOT/"market_data/calendars/nasdaq_sessions_2026_v2.json"
    raw = json.loads(path.read_text())
    valid = [r for r in raw["sessions"] if any(a <= r["session_date"] <= b for a,b in PHASES.values())]
    daily, source = {}, {}
    for sym in SYMBOLS:
        daily[sym], source[sym] = inspect_symbol(sym,valid)
    blocks, excluded, totals = {}, {}, {}
    for phase,(begin,end) in PHASES.items():
        sessions = [r for r in valid if begin<=r["session_date"]<=end]
        dates = [r["session_date"] for r in sessions]
        issues = {}
        for session in sessions:
            day = session["session_date"]
            reasons = {s:{"issues":daily[s][day]["issues"], "corporate_actions":daily[s][day]["corporate_actions"]}
                       for s in SYMBOLS if daily[s][day]["issues"] or daily[s][day]["corporate_actions"]}
            if session["duration_minutes"]!=390:
                reasons["calendar"] = {"regular_session_minutes":session["duration_minutes"]}
            if reasons:
                issues[day]=reasons
        blocks[phase], singletons = partition(phase,dates,issues)
        if phase=="november":
            assert len(blocks[phase])==1 and blocks[phase][0]["dates"]==dates
        excluded[phase]=dict(data_or_action=issues,valid_singletons=singletons)
        kept = sum(b["days"] for b in blocks[phase])
        assert kept+len(issues)+len(singletons)==len(dates)
        totals[phase] = dict(calendar_days=len(dates),used_days=kept,
                             rejected_data_or_action=len(issues),dropped_singletons=len(singletons))
    return dict(ranges=PHASES,calendar=str(path.relative_to(ROOT)),sources=source,blocks=blocks,
                excluded=excluded,totals=totals,daily=daily,
                rule="Each eligible official full session needs both full 288-slot tapes (preceding calendar day 20:00 ET to current 20:00 ET), finite positive traded OHLC and volume, NONE basis, ordered in-slot prices, end=start+5m, available=end+1s and zero corporate actions. Invalid dates split runs. Development and evaluation also split at ISO week boundaries; runs under two official sessions are excluded. November replay is a single 14-session block.")


if __name__ == "__main__":
    report = build()
    (HERE/"coverage.json").write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False,default=str)+"\n")
    print(json.dumps(report["totals"],indent=2))
