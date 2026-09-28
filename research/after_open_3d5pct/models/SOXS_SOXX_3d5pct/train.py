"""Train the SOXS/SOXX three-way action model. Rules are frozen in EXPERIMENT.md."""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from research.after_open_3d5pct.models.SOXS_SOXX_3d5pct.labels import build_rows

ROOT = Path(__file__).resolve().parents[4]
FEATURES = [
    "SOXS_r_1", "SOXS_r_6", "SOXS_r_12", "SOXS_r_39", "SOXS_morning", "SOXS_range_pos", "SOXS_volume_ratio",
    "SOXX_r_1", "SOXX_r_6", "SOXX_r_12", "SOXX_r_39", "SOXX_morning", "SOXX_range_pos", "SOXX_volume_ratio",
    "morning_spread", "r12_spread",
]
FOLDS = {
    "fit": ("1900-01-01", "2026-07-13"),
    "tune": ("2026-07-13", "2026-08-03"),
    "cal": ("2026-08-03", "2026-08-24"),
    "eval": ("2026-08-24", "2026-09-18"),
}
SEED = 3566


def _split(rows: pd.DataFrame, name: str) -> pd.DataFrame:
    start, end = FOLDS[name]
    mask = rows["session_date"].ge(start) & rows["session_date"].lt(end) & rows["label_end"].lt(end + " 00:00:00")
    return rows.loc[mask].reset_index(drop=True)


def _action(p_soxs: np.ndarray, p_soxx: np.ndarray) -> np.ndarray:
    out = np.array(["flat"] * len(p_soxs), dtype=object)
    buy = (p_soxs >= 0.5) | (p_soxx >= 0.5)
    out[buy & (p_soxs >= p_soxx)] = "SOXS"
    out[buy & (p_soxx > p_soxs)] = "SOXX"
    return out


def _metrics(part: pd.DataFrame) -> dict:
    chosen = _action(part["p_SOXS"].to_numpy(), part["p_SOXX"].to_numpy())
    y = part["action"].to_numpy()
    n = len(part)
    buy = chosen != "flat"
    correct_buy = [
        (name == "SOXS" and hit == 1) or (name == "SOXX" and hit == 1)
        for name, hit in zip(chosen[buy], np.where(chosen[buy] == "SOXS", part.loc[buy, "y_SOXS"], part.loc[buy, "y_SOXX"]))
    ] if buy.any() else []
    return {
        "rows": int(n),
        "accuracy": float((chosen == y).mean()) if n else None,
        "base_action_rate": {key: float((y == key).mean()) for key in ("flat", "SOXS", "SOXX")} if n else {},
        "buy_signals": int(buy.sum()),
        "buy_precision": float(np.mean(correct_buy)) if correct_buy else None,
    }


def run(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    soxs = pd.read_parquet(ROOT / "market_data" / "us_5m" / "SOXS" / "2026.parquet")
    soxx = pd.read_parquet(ROOT / "market_data" / "us_5m" / "SOXX" / "2026.parquet")
    rows, excluded = build_rows(soxs, soxx, barrier=0.05)
    parts = {name: _split(rows, name) for name in FOLDS}
    counts = {name: int(len(part)) for name, part in parts.items()}
    if counts["fit"] < 40 or counts["eval"] < 15:
        payload = {"id": "SOXS_SOXX_3d5pct", "status": "insufficient_sample", "counts": counts,
                   "excluded": excluded, "accuracy": None}
        (output / "result.json").write_text(json.dumps(payload, indent=2))
        return payload
    probs = {}
    for name in ("SOXS", "SOXX"):
        model = lgb.LGBMClassifier(
            n_estimators=40 if counts["tune"] < 10 else 80, num_leaves=7, max_depth=3,
            min_child_samples=8, learning_rate=0.05, reg_lambda=10, n_jobs=1,
            verbosity=-1, random_state=SEED)
        kwargs = {}
        if counts["tune"] >= 10:
            kwargs = {"eval_set": [(parts["tune"][FEATURES], parts["tune"][f"y_{name}"].to_numpy())],
                      "callbacks": [lgb.early_stopping(10, verbose=False)]}
        model.fit(parts["fit"][FEATURES], parts["fit"][f"y_{name}"].to_numpy(), **kwargs)
        model.booster_.save_model(str(output / f"{name}.txt"))
        for split, part in parts.items():
            probs.setdefault(split, part.copy())
            probs[split][f"p_{name}"] = model.predict_proba(part[FEATURES])[:, 1]
    scored = {}
    for split, part in probs.items():
        part.to_parquet(output / f"{split}.parquet", index=False)
        scored[split] = _metrics(part)
    payload = {"id": "SOXS_SOXX_3d5pct", "status": "trained", "seed": SEED, "excluded": excluded,
               "counts": counts, "splits": scored, "accuracy": scored["eval"]["accuracy"]}
    (output / "result.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps({"id": payload["id"], "eval_accuracy": payload["accuracy"], "eval_rows": counts["eval"],
                      "buy_precision": scored["eval"]["buy_precision"]}), flush=True)
    return payload


if __name__ == "__main__":
    run(ROOT / "research" / "after_open_3d5pct" / "runs" / "model_registry_v1" / "SOXS_SOXX_3d5pct")
