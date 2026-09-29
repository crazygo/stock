"""Fit four models on dates through 2025-06-30. 2026 is test only.

Published micro and macro codebooks were clustered on samples that include 2026.
This run does not load them. Curves are recomputed from bars already closed at 11:30.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from research.after_open_3d5pct.models.pugh_ge5.build_rows import ROOT, _regular_from_2024, build_panel
from research.after_open_3d5pct.models.pugh_ge5.walk import FEATURES

TRAIN_END = "2025-06-30"
VAL_START, VAL_END = "2025-07-01", "2025-12-31"
TEST_START, TEST_END = "2026-01-02", "2026-09-18"
HOUR_ENDS = {
    pd.Timestamp("10:30").time(),
    pd.Timestamp("11:30").time(),
    pd.Timestamp("12:30").time(),
    pd.Timestamp("13:30").time(),
    pd.Timestamp("14:30").time(),
    pd.Timestamp("15:30").time(),
    pd.Timestamp("16:00").time(),
}
MACRO_BARS = 35
MICRO_BARS = 24
MIN_TRAIN_DAYS = 40
BUY_QUANTILE = 0.80
OUT = ROOT / "research/after_open_3d5pct/runs/curve_split"

MICRO_GEOM = ["micro_beta", "micro_kappa", "micro_peak", "micro_trough", "micro_fade"]
MACRO_GEOM = ["macro_beta", "macro_kappa", "macro_peak", "macro_trough", "macro_fade"]
MICRO = [f"micro_z{i:02d}" for i in range(MICRO_BARS)] + MICRO_GEOM
MACRO = [f"macro_z{i:02d}" for i in range(MACRO_BARS)] + MACRO_GEOM
VARIANTS = {
    "baseline": list(FEATURES),
    "micro": list(FEATURES) + MICRO,
    "macro": list(FEATURES) + MACRO,
    "both": list(FEATURES) + MICRO + MACRO,
}


def split_role(day: str) -> str | None:
    if day <= TRAIN_END:
        return "train"
    if VAL_START <= day <= VAL_END:
        return "validation"
    if TEST_START <= day <= TEST_END:
        return "test"
    return None


def _t_norm(length: int) -> np.ndarray:
    return (np.arange(length) - (length - 1) / 2) / ((length - 1) / 3.4)


def zscore(close: np.ndarray | None) -> np.ndarray | None:
    if close is None or len(close) == 0 or np.any(close <= 0) or not np.isfinite(close).all():
        return None
    std = float(np.std(close))
    if std < 1e-8:
        return None
    return (close - float(np.mean(close))) / std


def curve_geometry(z: np.ndarray) -> tuple[float, float, float, float, float]:
    beta = float(np.polyfit(_t_norm(len(z)), z, 1)[0])
    kappa = float(np.polyfit(_t_norm(len(z)), z, 2)[0])
    peak = int(np.argmax(z))
    trough = int(np.argmin(z))
    span = float(z[peak] - z[trough])
    fade = 0.0 if span <= 1e-8 else float((z[peak] - z[-1]) / span)
    return beta, kappa, peak / (len(z) - 1), trough / (len(z) - 1), fade


def macro_window(ends: np.ndarray, closes: np.ndarray, decision: pd.Timestamp) -> np.ndarray | None:
    hits = np.flatnonzero(ends == np.datetime64(decision))
    if len(hits) != 1:
        return None
    index = int(hits[0])
    if index < MACRO_BARS - 1:
        return None
    chosen = ends[index - MACRO_BARS + 1:index + 1]
    if chosen[-1] != np.datetime64(decision) or np.any(chosen > np.datetime64(decision)):
        return None
    return closes[index - MACRO_BARS + 1:index + 1]


def micro_closes(day_bars: pd.DataFrame) -> np.ndarray | None:
    done = day_bars.loc[day_bars["end"].dt.time.le(pd.Timestamp("11:30").time())].sort_values("end")
    if len(done) != MICRO_BARS or done["end"].iloc[-1].time() != pd.Timestamp("11:30").time():
        return None
    delta = done["end"].diff().dt.total_seconds().iloc[1:]
    if not bool(np.all(delta.eq(300))):
        return None
    close = done["close"].to_numpy(float)
    if np.any(close <= 0) or not np.isfinite(close).all():
        return None
    return close


def _curve_row(micro: np.ndarray, macro: np.ndarray) -> dict:
    micro_g = curve_geometry(micro)
    macro_g = curve_geometry(macro)
    row = {f"micro_z{i:02d}": float(micro[i]) for i in range(MICRO_BARS)}
    row.update(dict(zip(MICRO_GEOM, micro_g)))
    row.update({f"macro_z{i:02d}": float(macro[i]) for i in range(MACRO_BARS)})
    row.update(dict(zip(MACRO_GEOM, macro_g)))
    return row


def curves_for_symbol(symbol: str, days: set[str]) -> pd.DataFrame:
    bars, _floor = _regular_from_2024(symbol)
    if bars.empty or not days:
        return pd.DataFrame()
    bars = bars.copy()
    bars["session_date"] = bars["start"].dt.strftime("%Y-%m-%d")
    slots = bars.loc[bars["end"].dt.time.isin(HOUR_ENDS), ["end", "close"]].drop_duplicates("end").sort_values("end")
    ends = slots["end"].to_numpy()
    closes = slots["close"].to_numpy(float)
    rows = []
    for day, frame in bars.groupby("session_date", sort=True):
        if day not in days:
            continue
        micro = zscore(micro_closes(frame))
        macro = zscore(macro_window(ends, closes, pd.Timestamp(f"{day} 11:30")))
        if micro is None or macro is None:
            continue
        row = _curve_row(micro, macro)
        row["symbol"] = symbol
        row["session_date"] = day
        rows.append(row)
    return pd.DataFrame(rows)


def attach_curves(panel: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for symbol, frame in panel.groupby("symbol", sort=True):
        curves = curves_for_symbol(symbol, set(frame["session_date"]))
        if len(curves):
            frames.append(curves)
            print(json.dumps({"symbol": symbol, "curve_rows": int(len(curves))}), flush=True)
    if not frames:
        return panel.iloc[0:0].copy()
    curves = pd.concat(frames, ignore_index=True)
    merged = panel.merge(curves, on=["symbol", "session_date"], how="inner")
    merged["role"] = merged["session_date"].map(split_role)
    return merged.loc[merged["role"].notna()].copy()


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


def _buy_stats(split: pd.DataFrame, thresholds: dict[str, float]) -> dict:
    eligible = split.loc[split["symbol"].isin(thresholds)]
    bought = eligible.loc[eligible.apply(lambda row: row["score"] >= thresholds[row["symbol"]], axis=1)]
    base = None if eligible.empty else float(eligible["y_3d_5pct"].mean())
    precision = None if bought.empty else float(bought["y_3d_5pct"].mean())
    return {
        "eligible_rows": int(len(eligible)),
        "eligible_base_rate": base,
        "buys": int(len(bought)),
        "precision": precision,
        "lift": None if precision is None or base is None else precision - base,
    }


def _metrics(split: pd.DataFrame, thresholds: dict[str, float]) -> dict:
    y = split["y_3d_5pct"].to_numpy(int)
    within, names = _within_auc(split)
    return {
        "rows": int(len(split)),
        "base_rate": None if split.empty else float(y.mean()),
        "auc": _auc(y, split["score"].to_numpy(float)),
        "within_stock_auc": within,
        "within_stock_names": names,
        **_buy_stats(split, thresholds),
    }


def _fit(train: pd.DataFrame, features: list[str]):
    import lightgbm as lgb
    model = lgb.LGBMClassifier(
        n_estimators=200,
        num_leaves=15,
        max_depth=4,
        min_child_samples=40,
        learning_rate=0.05,
        reg_lambda=5.0,
        n_jobs=1,
        verbosity=-1,
        random_state=3566,
    )
    model.fit(train[features], train["y_3d_5pct"].to_numpy(int))
    return model


def _importance_share(model, features: list[str]) -> dict:
    gain = model.booster_.feature_importance(importance_type="gain")
    names = list(model.booster_.feature_name())
    total = float(np.sum(gain))
    buckets = {"baseline": 0.0, "micro": 0.0, "macro": 0.0}
    for name, value in zip(names, gain):
        if name.startswith("micro_"):
            buckets["micro"] += float(value)
        elif name.startswith("macro_"):
            buckets["macro"] += float(value)
        else:
            buckets["baseline"] += float(value)
    if total <= 0:
        return {key: None for key in buckets}
    return {key: value / total for key, value in buckets.items()}


def _thresholds(train: pd.DataFrame) -> dict[str, float]:
    levels = {}
    for symbol, group in train.groupby("symbol"):
        if len(group) < MIN_TRAIN_DAYS:
            continue
        levels[symbol] = float(np.quantile(group["score"].to_numpy(float), BUY_QUANTILE))
    return levels


def _assert_split(frame: pd.DataFrame) -> None:
    train = set(frame.loc[frame["role"].eq("train"), "session_date"])
    validation = set(frame.loc[frame["role"].eq("validation"), "session_date"])
    test = set(frame.loc[frame["role"].eq("test"), "session_date"])
    if train & validation or train & test or validation & test:
        raise RuntimeError("train, validation, and test dates overlap")
    if train and max(train) > TRAIN_END:
        raise RuntimeError("training dates pass 2025-06-30")
    if test and min(test) < TEST_START:
        raise RuntimeError("test dates start before 2026-01-02")
    if frame.loc[frame["role"].eq("train"), "session_date"].astype(str).str.startswith("2026").any():
        raise RuntimeError("2026 rows entered training")


def run() -> dict:
    panel, coverage = build_panel("2024")
    frame = attach_curves(panel)
    _assert_split(frame)
    parts = {role: frame.loc[frame["role"].eq(role)].copy() for role in ("train", "validation", "test")}
    if parts["train"]["y_3d_5pct"].nunique() < 2:
        raise RuntimeError("training labels have a single class")
    variants = {}
    for name, features in VARIANTS.items():
        model = _fit(parts["train"], features)
        scored = {}
        thresholds = {}
        for role, split in parts.items():
            split = split.copy()
            split["score"] = model.predict_proba(split[features])[:, 1]
            if role == "train":
                thresholds = _thresholds(split)
            scored[role] = _metrics(split, thresholds)
        variants[name] = {
            "features": len(features),
            "importance_share": _importance_share(model, features),
            **scored,
        }
        print(json.dumps({"variant": name, "validation": variants[name]["validation"], "test": variants[name]["test"]}, default=str), flush=True)
    payload = {
        "split": {"train_end": TRAIN_END, "validation": [VAL_START, VAL_END], "test": [TEST_START, TEST_END]},
        "rows": {role: int(len(parts[role])) for role in parts},
        "coverage": {key: coverage[key] for key in ("rows", "first", "last")},
        "variants": variants,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "result.json").write_text(json.dumps(payload, indent=2, default=str))
    return payload


if __name__ == "__main__":
    done = run()
    print(json.dumps({"rows": done["rows"]}, default=str))
