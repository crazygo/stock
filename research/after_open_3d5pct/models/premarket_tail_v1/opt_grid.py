"""Seven pre-registered training combinations. The exam weeks do not choose the next cell."""
from __future__ import annotations

import json

import lightgbm as lgb
import numpy as np
import pandas as pd

from research.after_open_3d5pct.models.break_v1.run import MIN_TRAIN, TRAIN_WEEKS, window_days
from research.after_open_3d5pct.models.premarket_tail_v1.rows import FEATURES, FOLLOW_UP, build_frame
from research.after_open_3d5pct.models.premarket_tail_v1.run import (
    OUT,
    _plain,
    decision_cutoff,
)
from research.after_open_3d5pct.models.pugh_ge5.build_rows import ROOT
from research.after_open_3d5pct.models.weekly_scale.run import (
    _auc,
    _metrics_by_block,
    matured_before,
    week_blocks,
)

GRID_FEATURES = FEATURES + FOLLOW_UP
MIN_STOP = 20
PATIENCE = 20
CELLS = (
    {"id": "R", "reg": True, "early": False, "rank": False},
    {"id": "E", "reg": False, "early": True, "rank": False},
    {"id": "K", "reg": False, "early": False, "rank": True},
    {"id": "RE", "reg": True, "early": True, "rank": False},
    {"id": "RK", "reg": True, "early": False, "rank": True},
    {"id": "EK", "reg": False, "early": True, "rank": True},
    {"id": "REK", "reg": True, "early": True, "rank": True},
)
GRID_OUT = ROOT / "research/after_open_3d5pct/runs/premarket_tail_v1/opt_grid"


def tree_kwargs(reg: bool) -> dict:
    return {
        "n_estimators": 40,
        "num_leaves": 7,
        "max_depth": 3,
        "min_child_samples": 150 if reg else 10,
        "learning_rate": 0.05,
        "reg_lambda": 30.0 if reg else 10.0,
        "n_jobs": 1,
        "verbosity": -1,
        "random_state": 3566,
    }


def grouped(frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    ordered = frame.sort_values(["symbol", "session_date"], kind="mergesort").reset_index(drop=True)
    counts = ordered.groupby("symbol", sort=False).size().to_numpy(int)
    if int(counts.sum()) != len(ordered):
        raise RuntimeError("group sizes do not cover the fit rows")
    return ordered, counts


def fit_and_stop(frame: pd.DataFrame, weeks: list[list[str]], index: int, before: pd.Timestamp, early: bool):
    blocks = weeks[index - TRAIN_WEEKS:index]
    if early:
        fit_days = [day for block in blocks[:-1] for day in block]
        stop_days = list(blocks[-1])
    else:
        fit_days = [day for block in blocks for day in block]
        stop_days = []
    fit = matured_before(frame, fit_days, before)
    stop = matured_before(frame, stop_days, before) if stop_days else fit.iloc[0:0].copy()
    return fit, stop


def usable(fit: pd.DataFrame, stop: pd.DataFrame, early: bool) -> bool:
    if len(fit) < MIN_TRAIN or fit["y_3d_5pct"].nunique() < 2:
        return False
    if early and (len(stop) < MIN_STOP or stop["y_3d_5pct"].nunique() < 2):
        return False
    return True


def _fit_model(fit: pd.DataFrame, stop: pd.DataFrame, features: list[str], reg: bool, early: bool, rank: bool):
    kwargs = tree_kwargs(reg)
    if rank:
        ordered, groups = grouped(fit)
        model = lgb.LGBMRanker(objective="lambdarank", label_gain=[0, 1], **kwargs)
        extra = {"group": groups}
        if early:
            held, held_groups = grouped(stop)
            extra.update(
                eval_X=held[features],
                eval_y=held["y_3d_5pct"].to_numpy(int),
                eval_group=[held_groups],
                eval_at=[3],
                callbacks=[lgb.early_stopping(PATIENCE, verbose=False)],
            )
        model.fit(ordered[features], ordered["y_3d_5pct"].to_numpy(int), **extra)
        return model, model.predict(ordered[features])
    model = lgb.LGBMClassifier(**kwargs)
    extra = {}
    if early:
        extra = {
            "eval_X": stop[features],
            "eval_y": stop["y_3d_5pct"].to_numpy(int),
            "eval_metric": "auc",
            "callbacks": [lgb.early_stopping(PATIENCE, verbose=False)],
        }
    model.fit(fit[features], fit["y_3d_5pct"].to_numpy(int), **extra)
    return model, model.predict_proba(fit[features])[:, 1]


def _scores(model, block: pd.DataFrame, features: list[str], rank: bool) -> np.ndarray:
    if rank:
        return np.asarray(model.predict(block[features]), float)
    return np.asarray(model.predict_proba(block[features])[:, 1], float)


def _trees(model) -> int:
    best = getattr(model, "best_iteration_", None)
    if not best:
        return 40
    return int(best)


def beats_datum(block: dict, datum: dict) -> bool:
    precision = block.get("precision")
    reference = datum.get("precision")
    if precision is None or reference is None:
        return False
    return int(block.get("buys") or 0) >= 15 and float(precision) > float(reference)


def choose_bound(cells: list[dict], datum_test: dict, datum_validation: dict) -> str:
    beaters = []
    for cell in cells:
        if not cell.get("comparable"):
            continue
        if beats_datum(cell["test"], datum_test) and beats_datum(cell["validation"], datum_validation):
            beaters.append(cell)
    if not beaters:
        return "datum"
    def key(cell: dict):
        lower = min(float(cell["test"]["precision"]), float(cell["validation"]["precision"]))
        buys = int(cell["test"]["buys"]) + int(cell["validation"]["buys"])
        return (lower, buys, cell["id"])
    return max(beaters, key=key)["id"]


def _cell_walk(frame: pd.DataFrame, weeks: list[list[str]], cell: dict) -> dict:
    pooled = {"test": [], "validation": [], "train_auc": [], "trees": []}
    used = 0
    skipped = 0
    last = None
    for index in range(TRAIN_WEEKS, len(weeks) - 1):
        _train_days, test_days, val_days = window_days(weeks, index)
        if set(_train_days) & set(test_days) or set(test_days) & set(val_days):
            raise RuntimeError("train, test, and validation weeks overlap")
        before = decision_cutoff(test_days)
        fit, stop = fit_and_stop(frame, weeks, index, before, cell["early"])
        test = frame.loc[frame["session_date"].isin(test_days)].copy()
        validation = frame.loc[frame["session_date"].isin(val_days)].copy()
        if fit.empty or fit["session_date"].max() >= test_days[0] or test_days[-1] >= val_days[0]:
            raise RuntimeError("week order collapsed")
        if cell["early"] and (stop.empty or stop["session_date"].max() >= test_days[0]):
            raise RuntimeError("early-stopping week is not inside the train window")
        if not usable(fit, stop, cell["early"]) or test.empty or validation.empty:
            skipped += 1
            continue
        model, train_score = _fit_model(fit, stop, GRID_FEATURES, cell["reg"], cell["early"], cell["rank"])
        level = float(np.quantile(train_score, 0.80))
        train_y = fit.sort_values(["symbol", "session_date"], kind="mergesort")["y_3d_5pct"] if cell["rank"] else fit["y_3d_5pct"]
        pooled["train_auc"].append(_auc(train_y.to_numpy(int), train_score))
        pooled["trees"].append(_trees(model))
        for key, block in (("test", test), ("validation", validation)):
            scored = block.copy()
            scored["score"] = _scores(model, block, GRID_FEATURES, cell["rank"])
            scored["level"] = level
            pooled[key].append(scored)
        last = {"test": [test_days[0], test_days[-1]], "validation": [val_days[0], val_days[-1]]}
        used += 1
    test = pd.concat(pooled["test"], ignore_index=True) if pooled["test"] else pd.DataFrame()
    validation = pd.concat(pooled["validation"], ignore_index=True) if pooled["validation"] else pd.DataFrame()
    aucs = [item for item in pooled["train_auc"] if item is not None]
    return {
        **cell,
        "features": list(GRID_FEATURES),
        "fitted_triplets": used,
        "skipped_triplets": skipped,
        "mean_train_auc": None if not aucs else float(np.mean(aucs)),
        "mean_trees": None if not pooled["trees"] else float(np.mean(pooled["trees"])),
        "last_triplet": last,
        "test": _metrics_by_block(test),
        "validation": _metrics_by_block(validation),
    }


def run() -> dict:
    datum_path = OUT / "result.json"
    datum = json.loads(datum_path.read_text())["round2"]
    frame = build_frame()
    dates = sorted(frame["session_date"].unique())
    weeks = week_blocks(dates)
    cells = []
    for cell in CELLS:
        done = _cell_walk(frame, weeks, cell)
        done["comparable"] = (
            done["fitted_triplets"] == datum["fitted_triplets"]
            and done["test"]["rows"] == datum["test"]["rows"]
            and done["validation"]["rows"] == datum["validation"]["rows"]
        )
        done["beats_datum"] = bool(
            done["comparable"] and beats_datum(done["test"], datum["test"]) and beats_datum(done["validation"], datum["validation"])
        )
        cells.append(done)
        print(json.dumps(_plain({
            "id": done["id"],
            "comparable": done["comparable"],
            "beats_datum": done["beats_datum"],
            "fitted": done["fitted_triplets"],
            "skipped": done["skipped_triplets"],
            "trees": done["mean_trees"],
            "test_precision": done["test"]["precision"],
            "test_buys": done["test"]["buys"],
            "validation_precision": done["validation"]["precision"],
            "validation_buys": done["validation"]["buys"],
        })), flush=True)
    payload = {
        "datum": "round2",
        "datum_test": datum["test"],
        "datum_validation": datum["validation"],
        "bound": choose_bound(cells, datum["test"], datum["validation"]),
        "cells": cells,
    }
    GRID_OUT.mkdir(parents=True, exist_ok=True)
    (GRID_OUT / "result.json").write_text(json.dumps(_plain(payload), indent=2))
    print(json.dumps({"bound": payload["bound"]}), flush=True)
    return payload


if __name__ == "__main__":
    run()
