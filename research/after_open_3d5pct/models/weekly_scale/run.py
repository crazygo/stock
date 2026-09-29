"""Slide one-week train, next-week test, and the week after that as validation.

Micro is the 5-minute surge already printed by 11:30, including overnight and
premarket. Macro is the hourly curve ending at 11:30. Relation features are
their products. No published codebook is loaded.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from research.after_open_3d5pct.models.curve_split.run import curve_geometry, macro_window, zscore
from research.after_open_3d5pct.models.pugh_ge5.build_rows import (
    HISTORY,
    ROOT,
    _listing_floors,
    _naive_et,
    _read_5m,
    build_panel,
    history_floor,
    valid_ohlcv,
)
from research.after_open_3d5pct.models.pugh_ge5.walk import FEATURES

WEEK = 5
MIN_TRAIN = 30
MIN_OVERNIGHT = 6
MIN_PREMARKET = 6
BUY_QUANTILE = 0.80
HOUR_ENDS = {
    pd.Timestamp("10:30").time(),
    pd.Timestamp("11:30").time(),
    pd.Timestamp("12:30").time(),
    pd.Timestamp("13:30").time(),
    pd.Timestamp("14:30").time(),
    pd.Timestamp("15:30").time(),
    pd.Timestamp("16:00").time(),
}
MICRO = [
    "gap",
    "overnight_return",
    "premarket_return",
    "burst_5m",
    "burst_pos",
    "open_30m",
    "micro_fade",
    "micro_peak",
]
MACRO = ["macro_beta", "macro_kappa", "macro_fade", "macro_peak", "macro_trough", "macro_ret", "macro_vol"]
RELATION = [
    "gap_x_beta",
    "gap_x_kappa",
    "burst_x_beta",
    "burst_x_fade",
    "overnight_x_kappa",
    "premarket_x_beta",
    "gap_over_vol",
]
VARIANTS = {
    "baseline": list(FEATURES),
    "macro": list(FEATURES) + MACRO,
    "micro": list(FEATURES) + MICRO,
    "relation": list(FEATURES) + MICRO + MACRO + RELATION,
}
OUT = ROOT / "research/after_open_3d5pct/runs/weekly_scale"


def week_blocks(dates: list[str], size: int = WEEK) -> list[list[str]]:
    usable = len(dates) - (len(dates) % size)
    return [dates[i:i + size] for i in range(0, usable, size)]


def matured_before(frame: pd.DataFrame, days: list[str], before: pd.Timestamp) -> pd.DataFrame:
    chosen = frame.loc[frame["session_date"].isin(days) & frame["label_end"].notna()]
    return chosen.loc[pd.to_datetime(chosen["label_end"]).lt(before)].copy()


def burst_from_closes(closes: np.ndarray) -> tuple[float, float] | None:
    if len(closes) < 2 or np.any(closes <= 0) or not np.isfinite(closes).all():
        return None
    returns = np.diff(closes) / closes[:-1]
    where = int(np.argmax(returns))
    return float(returns[where]), where / (len(closes) - 1)


def add_relations(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["gap_x_beta"] = out["gap"] * out["macro_beta"]
    out["gap_x_kappa"] = out["gap"] * out["macro_kappa"]
    out["burst_x_beta"] = out["burst_5m"] * out["macro_beta"]
    out["burst_x_fade"] = out["burst_5m"] * out["macro_fade"]
    out["overnight_x_kappa"] = out["overnight_return"] * out["macro_kappa"]
    out["premarket_x_beta"] = out["premarket_return"] * out["macro_beta"]
    vol = out["macro_vol"].to_numpy(float)
    gap = out["gap"].to_numpy(float)
    scaled = np.full(len(out), np.nan)
    usable = np.isfinite(vol) & np.isfinite(gap) & (vol > 1e-8)
    scaled[usable] = gap[usable] / vol[usable]
    out["gap_over_vol"] = scaled
    return out


def _load_all(symbol: str) -> pd.DataFrame:
    floor = history_floor(_listing_floors().get(symbol))
    folder = HISTORY / "parts" / symbol
    paths = []
    if folder.exists():
        paths.extend(path for path in sorted(folder.glob("*/bars.parquet")) if path.parent.name >= floor[:7])
    extra = ROOT / "market_data/us_5m" / symbol / "2026.parquet"
    if extra.exists():
        paths.append(extra)
    if not paths:
        return pd.DataFrame()
    frame = pd.concat([_read_5m(path) for path in paths], ignore_index=True)
    frame = frame.loc[frame["session_date"].astype(str).ge(floor) & valid_ohlcv(frame)].copy()
    frame["start"] = _naive_et(frame["start_at_et"])
    frame["end"] = _naive_et(frame["end_at_et"])
    frame = frame.loc[frame["end"].gt(frame["start"])].sort_values("start").drop_duplicates("start")
    return frame.reset_index(drop=True)


def _session_return(bars: pd.DataFrame, minimum: int) -> float:
    if len(bars) < minimum:
        return np.nan
    first = float(bars["open"].iloc[0])
    last = float(bars["close"].iloc[-1])
    if first <= 0 or last <= 0:
        return np.nan
    return last / first - 1


def _macro_row(closes: np.ndarray) -> dict | None:
    z = zscore(closes)
    if z is None:
        return None
    beta, kappa, peak, trough, fade = curve_geometry(z)
    returns = np.diff(closes) / closes[:-1]
    return {
        "macro_beta": beta,
        "macro_kappa": kappa,
        "macro_fade": fade,
        "macro_peak": peak,
        "macro_trough": trough,
        "macro_ret": float(closes[-1] / closes[0] - 1),
        "macro_vol": float(np.std(returns)) if len(returns) else np.nan,
    }


def rows_for_symbol(symbol: str, days: set[str]) -> pd.DataFrame:
    bars = _load_all(symbol)
    if bars.empty or not days:
        return pd.DataFrame()
    regular = bars.loc[bars["session_type"].eq("regular")].copy()
    regular["day"] = regular["start"].dt.strftime("%Y-%m-%d")
    daily = regular.groupby("day", sort=True).agg(close=("close", "last"), end=("end", "max")).reset_index()
    slots = regular.loc[regular["end"].dt.time.isin(HOUR_ENDS), ["end", "close"]].drop_duplicates("end").sort_values("end")
    slot_ends = slots["end"].to_numpy()
    slot_closes = slots["close"].to_numpy(float)
    prior_close = dict(zip(daily["day"], daily["close"]))
    prior_end = dict(zip(daily["day"], daily["end"]))
    ordered_days = list(daily["day"])
    rows = []
    for day in ordered_days:
        if day not in days:
            continue
        earlier = [item for item in ordered_days if item < day]
        if not earlier:
            continue
        previous = earlier[-1]
        anchor = float(prior_close[previous])
        if not np.isfinite(anchor) or anchor <= 0:
            continue
        decision = pd.Timestamp(f"{day} 11:30")
        morning = regular.loc[regular["day"].eq(day) & regular["end"].le(decision)].sort_values("end")
        if len(morning) != 24 or pd.Timestamp(morning["end"].iloc[-1]) != decision:
            continue
        macro = _macro_row(macro_window(slot_ends, slot_closes, decision))
        if macro is None:
            continue
        opened = float(morning["open"].iloc[0])
        if opened <= 0:
            continue
        path = bars.loc[bars["end"].gt(prior_end[previous]) & bars["end"].le(decision)].sort_values("end")
        closes = np.concatenate([[anchor], path["close"].to_numpy(float)])
        burst = burst_from_closes(closes)
        if burst is None:
            continue
        night = path.loc[path["session_type"].eq("overnight") & path["end"].le(pd.Timestamp(f"{day} 04:00"))]
        pre = path.loc[path["session_type"].eq("pre_market") & path["end"].le(pd.Timestamp(f"{day} 09:30"))]
        until_10 = morning.loc[morning["end"].dt.time.le(pd.Timestamp("10:00").time())]
        z = zscore(morning["close"].to_numpy(float))
        if z is None or until_10.empty:
            continue
        _beta, _kappa, peak, _trough, fade = curve_geometry(z)
        row = {
            "symbol": symbol,
            "session_date": day,
            "gap": opened / anchor - 1,
            "overnight_return": _session_return(night, MIN_OVERNIGHT),
            "premarket_return": _session_return(pre, MIN_PREMARKET),
            "burst_5m": burst[0],
            "burst_pos": burst[1],
            "open_30m": float(until_10["close"].iloc[-1] / opened - 1),
            "micro_fade": fade,
            "micro_peak": peak,
            **macro,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def attach(panel: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for symbol, group in panel.groupby("symbol", sort=True):
        built = rows_for_symbol(symbol, set(group["session_date"]))
        if len(built):
            frames.append(built)
            print(json.dumps({"symbol": symbol, "rows": int(len(built))}), flush=True)
    if not frames:
        return panel.iloc[0:0].copy()
    extra = pd.concat(frames, ignore_index=True)
    merged = panel.merge(extra, on=["symbol", "session_date"], how="inner")
    return add_relations(merged)


def _auc(y: np.ndarray, score: np.ndarray) -> float | None:
    from sklearn.metrics import roc_auc_score
    if len(y) < 2 or len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, score))


def _within_auc(frame: pd.DataFrame) -> tuple[float | None, int]:
    scores = []
    weights = []
    for _symbol, group in frame.groupby("symbol"):
        if len(group) < 20 or group["y_3d_5pct"].nunique() < 2:
            continue
        value = _auc(group["y_3d_5pct"].to_numpy(int), group["score"].to_numpy(float))
        if value is None:
            continue
        scores.append(value)
        weights.append(len(group))
    if not scores:
        return None, 0
    return float(np.average(scores, weights=weights)), len(scores)


def _fit(train: pd.DataFrame, features: list[str]):
    import lightgbm as lgb
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
    model.fit(train[features], train["y_3d_5pct"].to_numpy(int))
    return model


def _score_block(model, features: list[str], block: pd.DataFrame, level: float) -> pd.DataFrame:
    scored = block.copy()
    scored["score"] = model.predict_proba(scored[features])[:, 1]
    scored["level"] = level
    return scored


def run() -> dict:
    panel, coverage = build_panel("2024")
    frame = attach(panel)
    dates = sorted(frame["session_date"].unique())
    weeks = week_blocks(dates)
    pooled = {name: {"test": [], "validation": [], "train_auc": []} for name in VARIANTS}
    skipped = 0
    used = 0
    last = None
    last_blocks = {name: {} for name in VARIANTS}
    for index in range(len(weeks) - 2):
        train_days, test_days, val_days = weeks[index], weeks[index + 1], weeks[index + 2]
        before = pd.Timestamp(f"{test_days[0]} 11:30")
        train = matured_before(frame, train_days, before)
        test = frame.loc[frame["session_date"].isin(test_days)].copy()
        validation = frame.loc[frame["session_date"].isin(val_days)].copy()
        if train["session_date"].max() >= test_days[0] or test_days[-1] >= val_days[0]:
            raise RuntimeError("week order collapsed")
        if len(train) < MIN_TRAIN or train["y_3d_5pct"].nunique() < 2:
            skipped += 1
            continue
        used += 1
        snapshot = {"train": [train_days[0], train_days[-1]], "test": [test_days[0], test_days[-1]], "validation": [val_days[0], val_days[-1]]}
        for name, features in VARIANTS.items():
            model = _fit(train, features)
            level = float(np.quantile(model.predict_proba(train[features])[:, 1], BUY_QUANTILE))
            train_score = model.predict_proba(train[features])[:, 1]
            pooled[name]["train_auc"].append(_auc(train["y_3d_5pct"].to_numpy(int), train_score))
            scored_test = _score_block(model, features, test, level)
            scored_validation = _score_block(model, features, validation, level)
            pooled[name]["test"].append(scored_test)
            pooled[name]["validation"].append(scored_validation)
            last_blocks[name] = {"test": scored_test, "validation": scored_validation}
        last = snapshot
        if used % 20 == 0:
            print(json.dumps({"fitted_triplets": used, "through": test_days[-1]}), flush=True)
    variants = {}
    for name in VARIANTS:
        test = pd.concat(pooled[name]["test"], ignore_index=True) if pooled[name]["test"] else pd.DataFrame()
        validation = pd.concat(pooled[name]["validation"], ignore_index=True) if pooled[name]["validation"] else pd.DataFrame()
        train_aucs = [item for item in pooled[name]["train_auc"] if item is not None]
        variants[name] = {
            "mean_train_auc": None if not train_aucs else float(np.mean(train_aucs)),
            "test": _metrics_by_block(test),
            "validation": _metrics_by_block(validation),
        }
        print(json.dumps({"variant": name, "test": variants[name]["test"], "validation": variants[name]["validation"]}, default=str), flush=True)
    payload = {
        "weeks": len(weeks),
        "fitted_triplets": used,
        "skipped_triplets": skipped,
        "last_triplet": last,
        "last_triplet_metrics": {
            name: {
                "test": _metrics_by_block(last_blocks[name]["test"]),
                "validation": _metrics_by_block(last_blocks[name]["validation"]),
            }
            for name in VARIANTS
            if last_blocks[name]
        },
        "coverage": {key: coverage[key] for key in ("rows", "first", "last")},
        "rows": int(len(frame)),
        "variants": variants,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "result.json").write_text(json.dumps(payload, indent=2, default=str))
    return payload


def _metrics_by_block(scored: pd.DataFrame) -> dict:
    if scored.empty:
        return {"buys": 0, "precision": None, "lift": None}
    bought = scored.loc[scored["score"] >= scored["level"]]
    base = float(scored["y_3d_5pct"].mean())
    precision = None if bought.empty else float(bought["y_3d_5pct"].mean())
    within, names = _within_auc(scored)
    return {
        "rows": int(len(scored)),
        "base_rate": base,
        "auc": _auc(scored["y_3d_5pct"].to_numpy(int), scored["score"].to_numpy(float)),
        "within_stock_auc": within,
        "within_stock_names": names,
        "buys": int(len(bought)),
        "precision": precision,
        "lift": None if precision is None else precision - base,
    }


if __name__ == "__main__":
    done = run()
    print(json.dumps({"fitted_triplets": done["fitted_triplets"], "skipped_triplets": done["skipped_triplets"], "last": done["last_triplet"]}))
