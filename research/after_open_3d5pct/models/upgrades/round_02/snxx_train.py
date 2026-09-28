"""Round 02: SNXX hourly proxy retraining with extension_20 feature."""
from __future__ import annotations

import json
from pathlib import Path
import sys
from datetime import time

import lightgbm as lgb
import numpy as np
import pandas as pd

cur = Path(__file__).resolve()
while cur.parent != cur and not (cur / "research" / "after_open_3d5pct").is_dir():
    cur = cur.parent
REPO_ROOT = cur
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.after_open_3d5pct.models.SNXX_3d3pct.labels import regular_bars, horizon_slice, MIN_HISTORY, RTH_OPEN, DECISION

BARS = REPO_ROOT / "market_data" / "us_60m" / "SNXX" / "2026.parquet"
FEATURES = [
    "r_1", "r_5", "r_10", "r_20", "morning_return", "overnight_gap",
    "range_position_20", "volume_ratio_20", "range_mean_10", "extension_20"
]
FOLDS = {
    "fit": ("1900-01-01", "2026-07-13"),
    "tune": ("2026-07-13", "2026-08-03"),
    "cal": ("2026-08-03", "2026-08-24"),
    "eval": ("2026-08-24", "2026-09-18"),
}
SEED = 3566


def _features_ext(completed: pd.DataFrame, day: pd.Timestamp) -> dict | None:
    if len(completed) < MIN_HISTORY:
        return None
    close = completed["close"].to_numpy(float)
    if np.any(close[-MIN_HISTORY:] <= 0) or np.any(~np.isfinite(close[-MIN_HISTORY:])):
        return None
    window = completed.iloc[-20:]
    span = float(window["high"].max() - window["low"].min())
    if span <= 0:
        return None
    prior_vol = completed["volume"].iloc[-21:-1].to_numpy(float)
    med = float(np.median(prior_vol))
    if med <= 0:
        return None
    today = completed[completed["start"].dt.normalize().eq(day)]
    open_bar = today[today["start"].dt.time.eq(RTH_OPEN)]
    ten_thirty = today[today["start"].dt.time.eq(time(10, 30))]
    if open_bar.empty or ten_thirty.empty:
        return None
    previous = completed[completed["start"].dt.normalize().lt(day)]
    if previous.empty or float(open_bar["open"].iloc[0]) <= 0:
        return None
    last_ranges = ((window["high"] - window["low"]) / window["close"]).iloc[-10:]
    max_high_20 = float(window["high"].max())
    out = {
        "r_1": float(close[-1] / close[-2] - 1),
        "r_5": float(close[-1] / close[-6] - 1),
        "r_10": float(close[-1] / close[-11] - 1),
        "r_20": float(close[-1] / close[-21] - 1),
        "morning_return": float(ten_thirty["close"].iloc[0] / open_bar["open"].iloc[0] - 1),
        "overnight_gap": float(open_bar["open"].iloc[0] / previous["close"].iloc[-1] - 1),
        "range_position_20": float((close[-1] - window["low"].min()) / span),
        "volume_ratio_20": float(completed["volume"].iloc[-1] / med),
        "range_mean_10": float(last_ranges.mean()),
        "extension_20": float(close[-1] / max_high_20 - 1),
    }
    if not np.isfinite(list(out.values())).all():
        return None
    return out


def build_rows_ext(frame: pd.DataFrame, barrier: float = 0.03) -> tuple[pd.DataFrame, dict]:
    regular = regular_bars(frame)
    dates = sorted(regular.loc[regular["start"].dt.time.eq(DECISION), "start"].dt.normalize().unique())
    rows = []
    excluded = {"no_entry": 0, "immature": 0, "short_history": 0, "bad_features": 0}
    for day in dates:
        day = pd.Timestamp(day)
        decision = day + pd.Timedelta(hours=11, minutes=30)
        entry_pos = regular.index[regular["start"].eq(decision)]
        if len(entry_pos) != 1:
            excluded["no_entry"] += 1
            continue
        pos = int(entry_pos[0])
        window = horizon_slice(regular, pos)
        if window is None:
            excluded["immature"] += 1
            continue
        completed = regular[regular["end"].le(decision)]
        features = _features_ext(completed, day)
        if features is None:
            excluded["short_history" if len(completed) < MIN_HISTORY else "bad_features"] += 1
            continue
        entry = float(regular.iloc[pos]["open"])
        if entry <= 0:
            excluded["bad_features"] += 1
            continue
        label_end = pd.Timestamp(window["end"].iloc[-1])
        rows.append({
            "session_date": day.strftime("%Y-%m-%d"),
            "entry": entry,
            "y": int(float(window["high"].max()) >= entry * (1 + barrier)),
            "label_end": label_end.strftime("%Y-%m-%d %H:%M:%S"),
            **features,
        })
    return pd.DataFrame(rows), excluded


def _split(rows: pd.DataFrame, name: str) -> pd.DataFrame:
    start, end = FOLDS[name]
    m = rows["session_date"].ge(start) & rows["session_date"].lt(end) & rows["label_end"].lt(end + " 00:00:00")
    return rows.loc[m].reset_index(drop=True)


def _scores(y: np.ndarray, p: np.ndarray) -> dict:
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    pred = p >= 0.5
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    n = int(len(y))
    base = float(y.mean()) if n else None
    return {
        "rows": n,
        "base_rate": base,
        "accuracy_at_0.5": (tp + tn) / n if n else None,
        "majority_class_accuracy": max(base, 1 - base) if n else None,
        "precision_at_0.5": tp / (tp + fp) if tp + fp else None,
        "recall_at_0.5": tp / (tp + fn) if tp + fn else None,
        "brier": float(np.mean((p - y) ** 2)) if n else None,
        "predicted_positive": int(tp + fp),
    }


def train_round2(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    rows, excluded = build_rows_ext(pd.read_parquet(BARS), barrier=0.03)
    parts = {name: _split(rows, name) for name in FOLDS}
    counts = {name: int(len(part)) for name, part in parts.items()}

    best_spw = 1
    best_tune_acc = -1.0
    best_model = None

    candidates = [1, 2, 4]
    candidate_results = {}

    for spw in candidates:
        model = lgb.LGBMClassifier(
            n_estimators=40 if counts["tune"] < 10 else 80,
            num_leaves=7, max_depth=3, min_child_samples=8, learning_rate=0.05,
            reg_lambda=10, n_jobs=1, verbosity=-1, random_state=SEED,
            scale_pos_weight=spw
        )
        kwargs = {}
        if counts["tune"] >= 10:
            kwargs = {
                "eval_set": [(parts["tune"][FEATURES], parts["tune"]["y"].to_numpy())],
                "callbacks": [lgb.early_stopping(10, verbose=False)],
            }
        model.fit(parts["fit"][FEATURES], parts["fit"]["y"].to_numpy(), **kwargs)

        p_tune = model.predict_proba(parts["tune"][FEATURES])[:, 1]
        t_acc = float(((p_tune >= 0.5) == (parts["tune"]["y"].to_numpy() == 1)).mean())
        candidate_results[spw] = {"tune_acc": t_acc}

        if t_acc > best_tune_acc:
            best_tune_acc = t_acc
            best_spw = spw
            best_model = model

    best_model.booster_.save_model(str(output / "model.txt"))

    scored = {}
    for name, part in parts.items():
        prob = best_model.predict_proba(part[FEATURES])[:, 1]
        part = part.copy()
        part["p"] = prob
        part.to_parquet(output / f"{name}.parquet", index=False)
        scored[name] = _scores(part["y"].to_numpy(), prob)

    payload = {
        "id": "SNXX_3d3pct",
        "round": 2,
        "seed": SEED,
        "selected_scale_pos_weight": best_spw,
        "candidate_results": candidate_results,
        "counts": counts,
        "splits": scored,
        "accuracy_at_0.5": scored["eval"]["accuracy_at_0.5"],
    }
    (output / "result.json").write_text(json.dumps(payload, indent=2))
    return payload
