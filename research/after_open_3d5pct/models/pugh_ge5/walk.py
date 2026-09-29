"""Fit each week on matured rows and score only the next week."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.after_open_3d5pct.models.pugh_ge5.build_rows import ROOT, build_panel

FEATURES = ["ret_5", "ret_20", "ret_60", "drawdown_20", "volume_ratio_20", "morning_return", "morning_vol", "last6_return"]
THRESHOLDS = [0.0, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20]
FALLBACK = 0.05
MIN_BUYS = 8
MIN_FIT = 80
LOCKED = ["2026-09-21", "2026-09-22", "2026-09-23"]
OUT = ROOT / "research/after_open_3d5pct/runs/pugh_ge5"


def choose_threshold(frame: pd.DataFrame) -> tuple[float, str]:
    best: tuple[float, float] | None = None
    for threshold in THRESHOLDS:
        buys = frame.loc[frame["excess_score"] >= threshold]
        if len(buys) < MIN_BUYS:
            continue
        excess = float((buys["y_3d_5pct"] - buys["own_base"]).mean())
        if best is None or excess > best[0] + 1e-12 or (abs(excess - best[0]) <= 1e-12 and threshold > best[1]):
            best = (excess, threshold)
    if best is None:
        return FALLBACK, "fallback"
    return best[1], "selected"


def _fit_probability(train: pd.DataFrame, target: pd.DataFrame) -> np.ndarray | None:
    import lightgbm as lgb
    y = train["y_3d_5pct"].to_numpy(int)
    if len(train) < MIN_FIT or len(np.unique(y)) < 2:
        return None
    model = lgb.LGBMClassifier(
        n_estimators=80, num_leaves=7, max_depth=3, min_child_samples=20,
        learning_rate=0.05, reg_lambda=10, n_jobs=1, verbosity=-1, random_state=3566,
    )
    model.fit(train[FEATURES], y)
    return model.predict_proba(target[FEATURES])[:, 1]


def _dates_before(dates: list[str], start: str, panel: pd.DataFrame) -> list[str]:
    cutoff = pd.Timestamp(f"{start} 11:30")
    ready = []
    for day in dates:
        if day >= start:
            break
        ends = panel.loc[panel["session_date"].eq(day), "label_end"]
        if len(ends) and pd.Timestamp(ends.max()) < cutoff:
            ready.append(day)
    return ready[-5:]


def _summarize(rows: pd.DataFrame, threshold: float) -> dict:
    buys = rows.loc[rows["excess_score"] >= threshold]
    if buys.empty:
        return {"rows": int(len(rows)), "buys": 0, "precision": None, "mean_excess": None, "judgement": "undetermined"}
    hit = buys["y_3d_5pct"].to_numpy(float)
    excess = float((hit - buys["own_base"].to_numpy(float)).mean())
    judgement = "undetermined"
    if len(buys) >= MIN_BUYS:
        judgement = "positive" if excess > 0 else "not_positive"
    return {
        "rows": int(len(rows)),
        "buys": int(len(buys)),
        "precision": float(hit.mean()),
        "mean_excess": excess,
        "judgement": judgement,
    }


def score_week(panel: pd.DataFrame, dates: list[str], week: list[str], role: str) -> dict:
    start = week[0]
    cutoff = pd.Timestamp(f"{start} 11:30")
    selection_days = _dates_before(dates, start, panel)
    fit = panel.loc[panel["label_end"].lt(cutoff) & ~panel["session_date"].isin(selection_days)].copy()
    record = {"role": role, "dates": week, "selection_days": selection_days, "fit_rows": int(len(fit))}
    target = panel.loc[panel["session_date"].isin(week)].copy()
    if target.empty:
        record.update({"threshold": None, "threshold_rule": "no_rows", "rows": 0, "buys": 0,
                       "precision": None, "mean_excess": None, "judgement": "undetermined"})
        return record
    selection = panel.loc[panel["session_date"].isin(selection_days)].copy()
    hold = pd.concat([selection, target], ignore_index=True)
    probability = _fit_probability(fit, hold)
    if probability is None:
        record.update({"threshold": FALLBACK, "threshold_rule": "unfit", "rows": int(len(target)), "buys": 0,
                       "precision": None, "mean_excess": None, "judgement": "undetermined"})
        return record
    hold = hold.copy()
    hold["p"] = probability
    hold["excess_score"] = hold["p"] - hold["own_base"]
    if len(selection_days) < 5:
        threshold, rule = FALLBACK, "selection_days_short"
    else:
        threshold, rule = choose_threshold(hold.loc[hold["session_date"].isin(selection_days)])
    held = hold.loc[hold["session_date"].isin(week)].copy()
    record.update({"threshold": threshold, "threshold_rule": rule, **_summarize(held, threshold)})
    record["symbols"] = sorted(held["symbol"].unique())
    return record


def run() -> dict:
    panel, coverage = build_panel()
    dates = sorted(panel["session_date"].unique())
    dev = [day for day in dates if day <= "2026-09-18"]
    weeks = [dev[i:i + 5] for i in range(0, len(dev), 5)]
    locked = [day for day in LOCKED if day in set(dates)]
    missing = [day for day in LOCKED if day not in set(dates)]
    results = [score_week(panel, dates, week, "development") for week in weeks if week]
    if locked:
        results.append(score_week(panel, dates, locked, "locked"))
    judged = [item for item in results if item["role"] == "development" and item["judgement"] != "undetermined"]
    payload = {
        "coverage": coverage,
        "locked_dates_requested": LOCKED,
        "locked_dates_scored": locked,
        "locked_dates_immature": missing,
        "weeks": results,
        "development_judged_weeks": len(judged),
        "development_positive_weeks": sum(item["judgement"] == "positive" for item in judged),
        "locked": next((item for item in results if item["role"] == "locked"), None),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "result.json").write_text(json.dumps(payload, indent=2, default=str))
    return payload


if __name__ == "__main__":
    done = run()
    locked = done["locked"]
    print(json.dumps({
        "full_126_fraction": done["coverage"]["full_126_fraction"],
        "rows": done["coverage"]["rows"],
        "development_positive_weeks": done["development_positive_weeks"],
        "development_judged_weeks": done["development_judged_weeks"],
        "locked": None if locked is None else {k: locked[k] for k in ("dates", "buys", "precision", "mean_excess", "judgement", "threshold")},
        "immature": done["locked_dates_immature"],
    }, default=str))
