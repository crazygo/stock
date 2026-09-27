"""Offline, no-group v7 LightGBM iterations on the frozen v6.1 data."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from research.after_open_3d5pct.train_multiscale_v6 import _folds, _stats, project_monotone, tabular

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
BASE = ROOT / "research/after_open_3d5pct"
DATASET = BASE / "runs/multiscale_groups_v61_dataset_20260926_r2"
R0 = BASE / "runs/multiscale_groups_v61_fit_20260926"
CONFIG = json.loads((R0 / "config.json").read_text())
EXPECTED_MANIFEST = "56b3d04def58b72e388f98e41c0d72df1b996aef3ecf7c8ee6325349082a143b"
SEED = 3566


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load():
    assert sha(DATASET / "manifest.json") == EXPECTED_MANIFEST
    table = pd.read_parquet(DATASET / "rows.parquet")
    with np.load(DATASET / "features.npz") as z:
        data = {k: z[k] for k in ("x5", "x60", "xday", "y")}
    folds = _folds(table, CONFIG)
    assert {k: len(v) for k, v in folds.items()} == {"train": 5606, "inner": 1521, "outer": 1140}
    assert len(table) == len(data["y"]) == 9222
    return table, data, folds


def _path_block(x: np.ndarray) -> np.ndarray:
    """Past-only ordered close path within one fixed window; NaN if absent."""
    mask = x[..., 10] > 0
    n = mask.sum(axis=-1)
    log_open = x[..., 12].astype(np.float64) * 5
    log_close = log_open + x[..., 0]
    start_ix = mask.argmax(axis=-1)
    end_ix = np.maximum(mask.shape[-1] - 1 - mask[..., ::-1].argmax(axis=-1), 0)
    first = np.take_along_axis(log_open, start_ix[..., None], axis=-1)[..., 0]
    last = np.take_along_axis(log_close, end_ix[..., None], axis=-1)[..., 0]
    count = np.cumsum(mask, axis=-1)
    one_ix = (count >= np.maximum(1, np.ceil(n / 3))[..., None]).argmax(axis=-1)
    two_ix = (count >= np.maximum(1, np.ceil(n * 2 / 3))[..., None]).argmax(axis=-1)
    one = np.take_along_axis(log_close, one_ix[..., None], axis=-1)[..., 0]
    two = np.take_along_axis(log_close, two_ix[..., None], axis=-1)[..., 0]
    level = np.where(mask, log_close, -np.inf)
    running_high = np.maximum(first[..., None], np.maximum.accumulate(level, axis=-1))
    drawdown = np.min(np.where(mask, log_close - running_high, np.inf), axis=-1)
    low = np.minimum(first, np.min(np.where(mask, log_close, np.inf), axis=-1))
    slot = np.arange(mask.shape[-1])
    preceding = np.maximum.accumulate(np.where(mask, slot, 0), axis=-1)
    preceding = np.concatenate((np.zeros_like(preceding[..., :1]), preceding[..., :-1]), axis=-1)
    previous_close = np.take_along_axis(log_close, preceding, axis=-1)
    step = np.where(mask, log_close - np.where(slot == start_ix[..., None],
                                                first[..., None], previous_close), 0)
    path_length = np.sum(np.abs(step), axis=-1)
    squared = np.sum(step**2, axis=-1)
    net = last - first
    # Do not assign path efficiency when an unseen interval separates observations.
    gap = (end_ix - start_ix + 1) > n
    efficiency = np.where(gap, np.nan, net / np.maximum(path_length, 1e-6))
    block = np.stack((one - first, two - one, last - two, drawdown,
                      last - low, efficiency,
                      np.sqrt(squared / np.maximum(n, 1))), axis=-1)
    block[n == 0] = np.nan
    return block.astype(np.float32)


def ordered_features(data: dict) -> pd.DataFrame:
    parts = []
    for key in ("x5", "x60"):
        x = data[key]
        pieces = [_path_block(x[:, i]) for i in range(x.shape[1])]
        parts.append(np.concatenate(pieces, axis=1))
    x = data["xday"]
    parts.append(np.concatenate([_path_block(x[:, a:b]) for a, b in
                                 ((0, 21), (21, 42), (42, 63), (63, 84), (84, 105), (105, 126))], axis=1))
    matrix = np.concatenate(parts, axis=1)
    return pd.DataFrame(matrix, columns=[f"path_{i:03d}" for i in range(matrix.shape[1])])


def path_boundary_check():
    falling = np.zeros((1, 3, 14), np.float32)
    falling[0, :2, 10] = 1
    falling[0, :2, 12] = np.log([100., 90.]) / 5
    falling[0, 0, 0] = np.log(90. / 100.)
    a = _path_block(falling)[0]
    assert a[3] < -0.10 and abs(a[4]) < 1e-6, a
    gap = np.zeros((1, 3, 14), np.float32)
    gap[0, [0, 2], 10] = 1
    gap[0, [0, 2], 12] = np.log([100., 120.]) / 5
    gap[0, 0, 0] = np.log(110. / 100.)
    b = _path_block(gap)[0]
    assert np.isnan(b[5]) and np.isfinite(b[6]), b
    shape_a = np.zeros((1, 4, 14), np.float32)
    shape_a[0, :, 10] = 1
    shape_a[0, :, 12] = np.log([100., 110., 120., 130.]) / 5
    shape_a[0, :, 0] = np.log(1.05)
    shape_b = shape_a[:, [1, 0, 2, 3], :]
    assert np.allclose(_stats(shape_a), _stats(shape_b), equal_nan=True)
    assert not np.allclose(_path_block(shape_a), _path_block(shape_b), equal_nan=True)


def matrix(data: dict, round_name: str) -> pd.DataFrame:
    base = tabular(data, group=False)
    if round_name == "r0":
        return base
    result = pd.concat((base, ordered_features(data)), axis=1)
    assert result.shape[1] == 900
    return result


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return {"n": len(y), "rate": float(np.mean(y)), "mean_p": float(np.mean(p)),
            "brier": float(np.mean((y-p)**2)),
            "logloss": float(log_loss(y, p, labels=[0, 1])),
            "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else None}


def summarize(y: np.ndarray, p: np.ndarray) -> dict:
    return {f"{h}_{t}": metrics(y[:, h, t], p[:, h, t]) for h in range(3) for t in range(3)}


def frame(table: pd.DataFrame, ids: np.ndarray, p: np.ndarray, y: np.ndarray | None):
    out = table.iloc[ids][["sample_id", "session_date", "symbol"]].reset_index(drop=True).copy()
    for h in range(3):
        for t in range(3):
            out[f"p_{h}_{t}"] = p[:, h, t]
            if y is not None:
                out[f"y_{h}_{t}"] = y[:, h, t]
    return out


def r0_diagnostic(table, data, folds, out):
    x = matrix(data, "r0")
    inner_raw = np.empty((len(folds["inner"]), 9))
    train_raw = np.empty((len(folds["train"]), 9))
    curves = {}
    for j in range(9):
        booster = lgb.Booster(model_file=str(R0 / f"models/B_lgbm_no_group_{j}.txt"))
        inner_raw[:, j] = booster.predict(x.iloc[folds["inner"]], num_threads=1)
        train_raw[:, j] = booster.predict(x.iloc[folds["train"]], num_threads=1)
        if j == 4:
            curves = {str(k): {
                "train": metrics(data["y"][folds["train"], 1, 1], booster.predict(x.iloc[folds["train"]], num_iteration=k, num_threads=1)),
                "inner": metrics(data["y"][folds["inner"], 1, 1], booster.predict(x.iloc[folds["inner"]], num_iteration=k, num_threads=1))}
                for k in (20, 40, 80, 120)}
    pi = project_monotone(inner_raw).reshape(-1, 3, 3)
    pt = project_monotone(train_raw).reshape(-1, 3, 3)
    frame(table, folds["inner"], pi, data["y"][folds["inner"]]).to_parquet(out / "inner.parquet", index=False)
    diagnostic = {"round": "r0", "train": summarize(data["y"][folds["train"]], pt),
                  "inner": summarize(data["y"][folds["inner"]], pi), "learning_curve_primary": curves,
                  "missing_fraction": {"base": float(x.isna().mean().mean()),
                                       "daily_126_full_train": float(table.iloc[folds["train"]].full_126_prior_days.mean()),
                                       "daily_126_full_inner": float(table.iloc[folds["inner"]].full_126_prior_days.mean())}}
    (out / "diagnostic.json").write_text(json.dumps(diagnostic, indent=2) + "\n")
    return diagnostic


def fit(round_name, table, data, folds, out, *, leaves, min_child, reg_lambda,
        max_trees, early_stop):
    x = matrix(data, round_name)
    y = data["y"].reshape(-1, 9)
    models = out / "models"
    models.mkdir(exist_ok=True)
    (out / "columns.json").write_text(json.dumps(list(x.columns)) + "\n")
    pi = np.empty((len(folds["inner"]), 9))
    po = np.empty((len(folds["outer"]), 9))
    pt = np.empty((len(folds["train"]), 9))
    details = []
    started = time.monotonic()
    for j in range(9):
        model = lgb.LGBMClassifier(n_estimators=max_trees, num_leaves=leaves,
                                   max_depth=4 if leaves == 7 else -1,
                                   min_child_samples=min_child, learning_rate=0.035,
                                   reg_lambda=reg_lambda, verbosity=-1, n_jobs=1,
                                   random_state=SEED, deterministic=True, force_col_wise=True)
        callback = [lgb.early_stopping(35, verbose=False)] if early_stop else []
        model.fit(x.iloc[folds["train"]], y[folds["train"], j],
                  eval_set=[(x.iloc[folds["inner"]], y[folds["inner"], j])] if early_stop else None,
                  eval_metric="binary_logloss" if early_stop else None, callbacks=callback)
        model.booster_.save_model(str(models / f"target_{j}.txt"))
        iteration = model.best_iteration_ if early_stop else max_trees
        pi[:, j] = model.predict_proba(x.iloc[folds["inner"]], num_iteration=iteration)[:, 1]
        po[:, j] = model.predict_proba(x.iloc[folds["outer"]], num_iteration=iteration)[:, 1]
        pt[:, j] = model.predict_proba(x.iloc[folds["train"]], num_iteration=iteration)[:, 1]
        detail = {"target": j, "trees": int(iteration), "model_sha256": sha(models / f"target_{j}.txt")}
        if j == 4:
            detail["learning_curve_primary"] = {str(k): {
                "train": metrics(y[folds["train"], j], model.predict_proba(x.iloc[folds["train"]], num_iteration=k)[:, 1]),
                "inner": metrics(y[folds["inner"], j], model.predict_proba(x.iloc[folds["inner"]], num_iteration=k)[:, 1])}
                for k in (20, 40, 80, 120, 200, 300, 450, 600) if k <= model.booster_.current_iteration()}
        details.append(detail)
    pi = project_monotone(pi).reshape(-1, 3, 3)
    po = project_monotone(po).reshape(-1, 3, 3)
    pt = project_monotone(pt).reshape(-1, 3, 3)
    frame(table, folds["inner"], pi, data["y"][folds["inner"]]).to_parquet(out / "inner.parquet", index=False)
    # Outer y is deliberately not read or scored until all three rounds are frozen.
    frame(table, folds["outer"], po, None).to_parquet(out / "outer_blind.parquet", index=False)
    diagnostic = {"round": round_name, "train": summarize(data["y"][folds["train"]], pt),
                  "inner": summarize(data["y"][folds["inner"]], pi),
                  "fit_seconds": time.monotonic()-started, "target_models": details,
                  "feature_count": x.shape[1], "feature_missing_fraction": float(x.isna().mean().mean())}
    (out / "diagnostic.json").write_text(json.dumps(diagnostic, indent=2) + "\n")
    return diagnostic


def replay(round_name, data, folds, out):
    x = matrix(data, round_name)
    cols = json.loads((out / "columns.json").read_text())
    assert list(x.columns) == cols
    saved = pd.read_parquet(out / "outer_blind.parquet")
    raw = np.column_stack([lgb.Booster(model_file=str(out / f"models/target_{j}.txt")).predict(
        x.iloc[folds["outer"]], num_threads=1) for j in range(9)])
    p = project_monotone(raw).reshape(-1, 3, 3)
    error = max(float(np.max(np.abs(p[:, h, t] - saved[f"p_{h}_{t}"].to_numpy())))
                for h in range(3) for t in range(3))
    assert error < 1e-7, error
    return error


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("round", choices=("r0", "r1", "r2"))
    args = ap.parse_args()
    path_boundary_check()
    table, data, folds = load()
    out = BASE / f"runs/v7_b_no_group_{args.round}"
    out.mkdir(exist_ok=True)
    if args.round == "r0":
        result = r0_diagnostic(table, data, folds, out)
    elif args.round == "r1":
        result = fit("r1", table, data, folds, out, leaves=7, min_child=100,
                     reg_lambda=10, max_trees=120, early_stop=False)
    else:
        result = fit("r2", table, data, folds, out, leaves=7, min_child=100,
                     reg_lambda=10, max_trees=600, early_stop=True)
    if args.round != "r0":
        result["replay_max_abs_error"] = replay(args.round, data, folds, out)
    result["lineage"] = {"dataset_manifest_sha256": EXPECTED_MANIFEST,
                         "runner_sha256": sha(Path(__file__)),
                         "r0_config_sha256": sha(R0 / "config.json"),
                         "python": os.sys.version.split()[0], "lightgbm": lgb.__version__}
    (out / "diagnostic.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"round": args.round, "primary_train": result["train"]["1_1"],
                      "primary_inner": result["inner"]["1_1"],
                      "fit_seconds": result.get("fit_seconds"),
                      "replay_error": result.get("replay_max_abs_error")}, indent=2))


if __name__ == "__main__":
    main()
