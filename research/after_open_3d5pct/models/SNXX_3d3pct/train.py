"""Train the single-name SNXX hourly proxy. Rules are frozen in EXPERIMENT.md."""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from research.after_open_3d5pct.models.SNXX_3d3pct.labels import build_rows

ROOT = Path(__file__).resolve().parents[4]
BARS = ROOT / "market_data" / "us_60m" / "SNXX" / "2026.parquet"
FEATURES = ["r_1", "r_5", "r_10", "r_20", "morning_return", "overnight_gap",
            "range_position_20", "volume_ratio_20", "range_mean_10"]
FOLDS = {
    "fit": ("1900-01-01", "2026-07-13"),
    "tune": ("2026-07-13", "2026-08-03"),
    "cal": ("2026-08-03", "2026-08-24"),
    "eval": ("2026-08-24", "2026-09-18"),
}
SEED = 3566


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


def _choose_threshold(y: np.ndarray, p: np.ndarray) -> float | None:
    best = None
    for step in range(6, 20):
        level = step / 20
        picked = p >= level
        n = int(picked.sum())
        if n < 3:
            continue
        precision = float(y[picked].mean())
        rank = (precision, level)
        if best is None or rank > best[0]:
            best = (rank, level)
    return None if best is None else best[1]


def run(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    if not BARS.exists():
        payload = {"id": "SNXX_3d3pct", "status": "missing_bars", "bars": str(BARS)}
        (output / "result.json").write_text(json.dumps(payload, indent=2))
        return payload
    rows, excluded = build_rows(pd.read_parquet(BARS), barrier=0.03)
    parts = {name: _split(rows, name) for name in FOLDS}
    counts = {name: int(len(part)) for name, part in parts.items()}
    if counts["fit"] < 40 or counts["eval"] < 15:
        payload = {"id": "SNXX_3d3pct", "status": "insufficient_sample", "counts": counts,
                   "excluded": excluded, "accuracy_at_0.5": None}
        (output / "result.json").write_text(json.dumps(payload, indent=2))
        rows.to_parquet(output / "rows.parquet", index=False)
        return payload
    model = lgb.LGBMClassifier(
        n_estimators=40 if counts["tune"] < 10 else 80,
        num_leaves=7, max_depth=3, min_child_samples=8, learning_rate=0.05,
        reg_lambda=10, n_jobs=1, verbosity=-1, random_state=SEED)
    fit_x = parts["fit"][FEATURES]
    fit_y = parts["fit"]["y"].to_numpy()
    kwargs = {}
    if counts["tune"] >= 10:
        kwargs = {"eval_set": [(parts["tune"][FEATURES], parts["tune"]["y"].to_numpy())],
                  "callbacks": [lgb.early_stopping(10, verbose=False)]}
    model.fit(fit_x, fit_y, **kwargs)
    model.booster_.save_model(str(output / "model.txt"))
    scored = {}
    for name, part in parts.items():
        prob = model.predict_proba(part[FEATURES])[:, 1]
        part = part.copy()
        part["p"] = prob
        part.to_parquet(output / f"{name}.parquet", index=False)
        scored[name] = _scores(part["y"].to_numpy(), prob)
    threshold = _choose_threshold(parts["cal"]["y"].to_numpy(),
                                  pd.read_parquet(output / "cal.parquet")["p"].to_numpy())
    eval_part = pd.read_parquet(output / "eval.parquet")
    buy = None
    if threshold is not None:
        picked = eval_part["p"].to_numpy() >= threshold
        buy = {"threshold": threshold, "signals": int(picked.sum()),
               "precision": float(eval_part.loc[picked, "y"].mean()) if picked.any() else None}
    payload = {
        "id": "SNXX_3d3pct", "status": "trained", "seed": SEED, "excluded": excluded,
        "counts": counts, "splits": scored, "buy": buy,
        "accuracy_at_0.5": scored["eval"]["accuracy_at_0.5"],
        "trees": int(model.booster_.num_trees()),
        "contract": "hourly_proxy_v1",
    }
    (output / "result.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps({"id": "SNXX_3d3pct", "eval_accuracy": payload["accuracy_at_0.5"],
                      "eval_rows": counts["eval"], "base_rate": scored["eval"]["base_rate"]}), flush=True)
    return payload


if __name__ == "__main__":
    run(ROOT / "research" / "after_open_3d5pct" / "runs" / "model_registry_v1" / "SNXX_3d3pct")
