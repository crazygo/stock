"""Choose one buy rule per stock on 2025 H2, then score 2026 once."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from research.after_open_3d5pct.models.pugh_ge5.build_rows import ROOT, build_panel
from research.after_open_3d5pct.models.pugh_ge5.walk import FEATURES

FIT_BEFORE = pd.Timestamp("2025-07-01")
TUNE_START, TUNE_END = "2025-07-01", "2025-12-31"
CONFIRM_START, CONFIRM_END = "2026-01-02", "2026-09-18"
LOCKED = ["2026-09-21", "2026-09-22", "2026-09-23"]
PROBABILITIES = [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90]
VOL_QUANTILES = [0.75, 0.85, 0.90, 0.95]
VOL_AND_BASE = [(0.80, 0.50), (0.80, 0.60), (0.90, 0.50), (0.90, 0.60)]
MIN_TUNE_BUYS = 10
MIN_CONFIRM_BUYS = 15
MIN_FIT_MODEL = 80
MIN_FIT_VOL = 40
TARGET = 0.90
OUT = ROOT / "research/after_open_3d5pct/runs/per_stock_90"


def wilson_lower(hits: int, count: int) -> float | None:
    if count <= 0:
        return None
    z = 1.96
    proportion = hits / count
    denominator = 1 + z * z / count
    centre = proportion + z * z / (2 * count)
    margin = z * (proportion * (1 - proportion) / count + z * z / (4 * count * count)) ** 0.5
    return (centre - margin) / denominator


def choose_candidate(candidates: list[dict]) -> dict | None:
    eligible = [item for item in candidates if item["buys"] >= MIN_TUNE_BUYS and item["precision"] >= TARGET]
    if not eligible:
        return None
    eligible.sort(key=lambda item: (item["precision"], item["buys"], item["selectivity"]), reverse=True)
    return eligible[0]


def _score(frame: pd.DataFrame, mask: pd.Series) -> dict:
    buys = frame.loc[mask]
    count = int(len(buys))
    if count == 0:
        return {"buys": 0, "hits": 0, "precision": None, "wilson_lower": None}
    hits = int(buys["y_3d_5pct"].sum())
    return {
        "buys": count,
        "hits": hits,
        "precision": hits / count,
        "wilson_lower": wilson_lower(hits, count),
    }


def _between(frame: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    return frame.loc[frame["session_date"].ge(start) & frame["session_date"].le(end)].copy()


def _model_probability(train: pd.DataFrame, later: pd.DataFrame) -> np.ndarray | None:
    import lightgbm as lgb
    target = train["y_3d_5pct"].to_numpy(int)
    if len(train) < MIN_FIT_MODEL or len(np.unique(target)) < 2:
        return None
    model = lgb.LGBMClassifier(
        n_estimators=40, num_leaves=4, max_depth=2, min_child_samples=25,
        learning_rate=0.05, reg_lambda=10, n_jobs=1, verbosity=-1, random_state=3566,
    )
    model.fit(train[FEATURES], target)
    return model.predict_proba(later[FEATURES])[:, 1]


def _candidates(fit: pd.DataFrame, tune: pd.DataFrame) -> list[dict]:
    found = []
    if len(fit) >= MIN_FIT_VOL:
        for quantile in VOL_QUANTILES:
            level = float(fit["morning_vol"].quantile(quantile))
            score = _score(tune, tune["morning_vol"].ge(level))
            found.append({"family": "morning_vol", "quantile": quantile, "level": level, "selectivity": quantile, **score})
        for quantile, base in VOL_AND_BASE:
            level = float(fit["morning_vol"].quantile(quantile))
            mask = tune["morning_vol"].ge(level) & tune["own_base"].ge(base)
            score = _score(tune, mask)
            found.append({
                "family": "morning_vol_and_base", "quantile": quantile, "base": base,
                "level": level, "selectivity": quantile + base, **score,
            })
    probability = _model_probability(fit, tune)
    if probability is not None:
        scored = tune.copy()
        scored["p"] = probability
        for threshold in PROBABILITIES:
            score = _score(scored, scored["p"].ge(threshold))
            found.append({"family": "lightgbm", "threshold": threshold, "selectivity": threshold, **score})
    for item in found:
        item["precision"] = None if item["precision"] is None else float(item["precision"])
    return found


def _apply(rule: dict, fit: pd.DataFrame, frame: pd.DataFrame, probability: np.ndarray | None) -> dict:
    if rule["family"] == "lightgbm":
        scored = frame.copy()
        scored["p"] = probability
        return _score(scored, scored["p"].ge(rule["threshold"]))
    mask = frame["morning_vol"].ge(rule["level"])
    if rule["family"] == "morning_vol_and_base":
        mask = mask & frame["own_base"].ge(rule["base"])
    return _score(frame, mask)


def _windows(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    fit = frame.loc[frame["label_end"].lt(FIT_BEFORE)].copy()
    tune = _between(frame, TUNE_START, TUNE_END)
    if len(fit) >= MIN_FIT_VOL:
        return fit, tune, "standard"
    pre = frame.loc[frame["label_end"].lt(pd.Timestamp(CONFIRM_START))].sort_values("session_date")
    if len(pre) < MIN_FIT_VOL + MIN_TUNE_BUYS:
        return pre.iloc[0:0], pre.iloc[0:0], "too_short"
    return pre.iloc[:MIN_FIT_VOL].copy(), pre.iloc[MIN_FIT_VOL:].copy(), "late_start"


def evaluate_symbol(frame: pd.DataFrame) -> dict:
    fit, tune, split = _windows(frame)
    confirm = _between(frame, CONFIRM_START, CONFIRM_END)
    locked = frame.loc[frame["session_date"].isin(LOCKED)].copy()
    record = {
        "split": split,
        "fit_rows": int(len(fit)),
        "tune_rows": int(len(tune)),
        "confirm_rows": int(len(confirm)),
        "confirm_base_rate": None if confirm.empty else float(confirm["y_3d_5pct"].mean()),
    }
    candidates = _candidates(fit, tune)
    record["tune_candidates"] = candidates
    chosen = choose_candidate(candidates)
    record["chosen"] = chosen
    if chosen is None:
        record["confirm"] = None
        record["locked"] = None
        record["passed"] = False
        return record
    probability = None
    if chosen["family"] == "lightgbm":
        probability = _model_probability(fit, pd.concat([confirm, locked], ignore_index=True))
        if probability is None:
            record["confirm"] = None
            record["locked"] = None
            record["passed"] = False
            return record
        confirm_probability = probability[:len(confirm)]
        locked_probability = probability[len(confirm):]
    else:
        confirm_probability = None
        locked_probability = None
    confirm_score = _apply(chosen, fit, confirm, confirm_probability)
    locked_score = _apply(chosen, fit, locked, locked_probability)
    record["confirm"] = confirm_score
    record["locked"] = locked_score
    record["passed"] = bool(
        confirm_score["buys"] >= MIN_CONFIRM_BUYS and confirm_score["precision"] is not None and confirm_score["precision"] >= TARGET
    )
    return record


def run() -> dict:
    panel, coverage = build_panel("2024")
    symbols = []
    for symbol, frame in panel.groupby("symbol", sort=True):
        result = evaluate_symbol(frame.sort_values("session_date"))
        result["symbol"] = symbol
        symbols.append(result)
        print(json.dumps({
            "symbol": symbol,
            "passed": result["passed"],
            "rule": None if result["chosen"] is None else result["chosen"]["family"],
            "confirm": result["confirm"],
        }, default=str), flush=True)
    payload = {
        "coverage": {key: coverage[key] for key in ("history", "rows", "first", "last")},
        "fit_before": str(FIT_BEFORE.date()),
        "tune": [TUNE_START, TUNE_END],
        "confirm": [CONFIRM_START, CONFIRM_END],
        "symbols": symbols,
        "passed_symbols": [item["symbol"] for item in symbols if item["passed"]],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "result.json").write_text(json.dumps(payload, indent=2, default=str))
    return payload


if __name__ == "__main__":
    done = run()
    print(json.dumps({"passed": done["passed_symbols"]}, default=str))
