"""Round 6 content changes. Probability shifts are not a candidate."""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from research.after_open_3d5pct.focus_v8.core import GROUPS, PROTOCOL, TARGETS, focus_weights, splits
from research.after_open_3d5pct.focus_v8.run import load_data, training_weights
from research.after_open_3d5pct.models.SNXX_3d3pct.labels import build_rows as snxx_rows
from research.after_open_3d5pct.models.SNXX_3d3pct.train import FEATURES as SNXX_FEATURES
from research.after_open_3d5pct.models.SNXX_3d3pct.train import FOLDS as SNXX_FOLDS
from research.after_open_3d5pct.models.SOXS_SOXX_3d5pct.labels import build_rows as soxs_rows
from research.after_open_3d5pct.models.SOXS_SOXX_3d5pct.train import FEATURES as SOXS_FEATURES
from research.after_open_3d5pct.models.SOXS_SOXX_3d5pct.train import FOLDS as SOXS_FOLDS
from research.after_open_3d5pct.models.SOXS_SOXX_3d5pct.train import _action

ROOT = Path(__file__).resolve().parents[5]
AFTER = ROOT / "research" / "after_open_3d5pct"
RUN = AFTER / "runs" / "model_registry_v1"
OUT = AFTER / "runs" / "model_registry_v1_upgrades" / "round_06"
PRIMARY = TARGETS.index("3d_5pct")
# Completed 5-minute bars: index 77 ends 10:30, index 89 ends 11:30. Channel 12 is log price.
SLOT_1030, SLOT_1130, LOG_PRICE = 77, 89, 12


def accuracy(y, p) -> float:
    return float(np.mean((np.asarray(p) >= 0.5) == np.asarray(y).astype(bool)))


def load_split(model_id: str, split: str) -> pd.DataFrame:
    path = RUN / model_id / "reserved" / f"{split}.parquet"
    if not path.exists():
        path = RUN / model_id / f"{split}.parquet"
    return pd.read_parquet(path)


def slice_ids(frame: pd.DataFrame, symbols: list[str] | None) -> np.ndarray:
    if symbols is None:
        return np.arange(len(frame))
    return np.flatnonzero(frame.symbol.isin(symbols).to_numpy())


def hour_features(data: dict, rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    x5 = data["x5"]
    own = x5[:, 7, SLOT_1130, LOG_PRICE] - x5[:, 7, SLOT_1030, LOG_PRICE]
    valid = (x5[:, 7, SLOT_1130, 10] > 0) & (x5[:, 7, SLOT_1030, 10] > 0) & np.isfinite(own)
    own = np.where(valid, own, np.nan).astype(np.float32)
    peers = sorted({symbol for symbols in GROUPS.values() for symbol in symbols})
    peer = np.full(len(rows), np.nan, np.float32)
    dates = rows.session_date.to_numpy()
    symbols = rows.symbol.to_numpy()
    for date in pd.unique(dates):
        member = np.flatnonzero((dates == date) & np.isin(symbols, peers) & valid)
        if len(member) < 2:
            continue
        total = float(own[member].sum())
        count = len(member)
        for index in member:
            peer[index] = (total - float(own[index])) / (count - 1)
    if not np.isfinite(own[valid]).size or np.median(np.abs(own[valid])) > 0.5:
        raise RuntimeError("10:30 to 11:30 log-price gap is not a return; channel layout changed")
    return own, peer


def b_matrix(route: str, xbase, paths, curve, relative, own, peer) -> np.ndarray:
    columns = [xbase if route == "B_group" else xbase[:, :585], paths, own[:, None]]
    if route == "B_group":
        columns.extend((curve, relative, peer[:, None]))
    return np.column_stack(columns).astype(np.float32)


def c_matrix(route: str, own, peer, parent_logit) -> np.ndarray:
    columns = [parent_logit.astype(np.float32), own[:, None]]
    if route == "C_group":
        columns.append(peer[:, None])
    return np.column_stack(columns)


def fit_targets(x, y, rows, fold, parent_logit, seed: int) -> np.ndarray:
    """Nine trees. parent_logit is None, or shape (n, 9) frozen scores."""
    if parent_logit is not None:
        present = np.isfinite(parent_logit).all(axis=1)
        fold = {name: ids[present[ids]] for name, ids in fold.items()}
    weight = focus_weights(rows.iloc[fold["fit"]])
    tune_weight = focus_weights(rows.iloc[fold["tune"]])
    tune_weight = tune_weight / max(tune_weight.mean(), 1e-8)
    raw = {split: np.zeros((len(ids), 9)) for split, ids in fold.items()}
    for j in range(9):
        kwargs = {}
        if parent_logit is not None:
            kwargs = {"init_score": parent_logit[fold["fit"], j],
                      "eval_init_score": [parent_logit[fold["tune"], j]]}
        model = lgb.LGBMClassifier(
            n_estimators=40 if parent_logit is not None else 120,
            num_leaves=7, max_depth=3 if parent_logit is not None else 4,
            min_child_samples=20 if parent_logit is not None else 150,
            learning_rate=0.05 if parent_logit is not None else 0.035,
            reg_lambda=10, n_jobs=1, verbosity=-1, random_state=seed)
        model.fit(x[fold["fit"]], y[fold["fit"], j], sample_weight=weight,
                  eval_set=[(x[fold["tune"]], y[fold["tune"], j])],
                  eval_sample_weight=[tune_weight],
                  callbacks=[lgb.early_stopping(15, verbose=False)], **kwargs)
        booster = model.booster_
        replay = lgb.Booster(model_str=booster.model_to_string())
        for split, ids in fold.items():
            score = booster.predict(x[ids], raw_score=True, num_threads=1)
            again = replay.predict(x[ids], raw_score=True, num_threads=1)
            if float(np.max(np.abs(score - again))) > 1e-6:
                raise AssertionError("residual tree replay mismatch")
            if parent_logit is not None:
                score = score + parent_logit[ids, j]
            raw[split][:, j] = 1 / (1 + np.exp(-np.clip(score, -30, 30)))
    return raw["eval"][:, PRIMARY], raw["tune"][:, PRIMARY], raw["cal"][:, PRIMARY]


def decide(tune_new, cal_new, tune_base, cal_base) -> bool:
    return tune_new > tune_base + 1e-12 and cal_new + 1e-12 >= cal_base


def score_model(model_id, symbols, eval_p, tune_p, cal_p, baseline_eval) -> dict:
    frames = {split: load_split(model_id, split) for split in ("tune", "cal", "eval")}
    column = "y_3d_5pct" if "y_3d_5pct" in frames["eval"].columns else "y"
    p_column = "p_3d_5pct" if "p_3d_5pct" in frames["eval"].columns else "p"
    def take(split, values):
        frame = frames[split]
        mask = np.ones(len(frame), bool) if symbols is None else frame.symbol.isin(symbols).to_numpy()
        return frame.loc[mask, column].to_numpy(), np.asarray(values)[mask], frame.loc[mask, p_column].to_numpy()
    y_tune, new_tune, base_tune = take("tune", tune_p)
    y_cal, new_cal, base_cal = take("cal", cal_p)
    y_eval, new_eval, _ = take("eval", eval_p)
    adopted = decide(accuracy(y_tune, new_tune), accuracy(y_cal, new_cal),
                     accuracy(y_tune, base_tune), accuracy(y_cal, base_cal))
    official = accuracy(y_eval, new_eval) if adopted else baseline_eval
    return {"id": model_id, "adopted": adopted, "baseline_eval": baseline_eval,
            "eval_accuracy": official, "candidate_eval": accuracy(y_eval, new_eval),
            "tune_base": accuracy(y_tune, base_tune), "tune_new": accuracy(y_tune, new_tune),
            "rows": int(len(y_eval))}


def parent_logits(model_id: str, rows: pd.DataFrame) -> np.ndarray:
    pieces = []
    for split in ("fit", "tune", "cal", "eval"):
        frame = load_split(model_id, split)
        pieces.append(frame)
    parent = pd.concat(pieces, ignore_index=True).drop_duplicates("sample_id").set_index("sample_id")
    columns = [f"raw_{name}" for name in TARGETS]
    aligned = parent.reindex(rows.sample_id.to_numpy())
    raw = aligned[columns].to_numpy(np.float64)
    logits = np.full_like(raw, np.nan)
    ok = np.isfinite(raw).all(axis=1)
    clipped = np.clip(raw[ok], 1e-4, 1 - 1e-4)
    logits[ok] = np.log(clipped / (1 - clipped))
    return logits


def train_probability_models(loaded) -> list[dict]:
    rows, data, y, prior, counts, paths, xbase, curve, relative, arrays, identity = loaded
    own, peer = hour_features(data, rows)
    fold_config = next(item for item in PROTOCOL["folds"] if item["id"] == "reserved")
    previous = {item["id"]: item["eval_accuracy"] for item in json.loads(
        (AFTER / "runs/model_registry_v1_upgrades/round_05/result.json").read_text())["models"]}
    union = sorted({symbol for symbols in GROUPS.values() for symbol in symbols})
    jobs = [(name, None, union, False) for name in ("B_group", "B_no_group")]
    jobs += [(name, None, union, True) for name in ("C_group", "C_no_group", "C_no_daily")]
    for group, symbols in GROUPS.items():
        for prefix, residual in (("B_group", False), ("B_no_group", False),
                                 ("C_group", True), ("C_no_group", True), ("C_no_daily", True)):
            jobs.append((f"{prefix}_{group}", symbols, symbols, residual))
    reports = []
    for model_id, train_symbols, score_symbols, residual in jobs:
        mask = np.ones(len(rows), bool) if train_symbols is None else rows.symbol.isin(train_symbols).to_numpy()
        sub_rows = rows.loc[mask].reset_index(drop=True)
        fold = splits(sub_rows, fold_config)
        if residual:
            route = model_id.split("_chips")[0].split("_optics")[0].split("_storage")[0]
            parent = parent_logits(model_id, sub_rows)
            x = c_matrix(route, own[mask], peer[mask], parent)
        else:
            route = "B_group" if model_id.startswith("B_group") else "B_no_group"
            x = b_matrix(route, xbase[mask], paths[mask], curve[mask], relative[mask], own[mask], peer[mask])
            parent = None
        eval_p, tune_p, cal_p = fit_targets(x, y[mask], sub_rows, fold, parent, PROTOCOL["seed"])
        full = {}
        pred = {"tune": tune_p, "cal": cal_p, "eval": eval_p}
        for split, values in pred.items():
            saved = load_split(model_id, split)
            position = pd.Series(values, index=sub_rows.iloc[fold[split]].sample_id.to_numpy())
            full[split] = position.reindex(saved.sample_id.to_numpy()).to_numpy()
            if not np.isfinite(full[split]).all():
                raise RuntimeError(f"{model_id} {split} predictions do not cover the saved rows")
        report = score_model(model_id, score_symbols, full["eval"], full["tune"], full["cal"], previous[model_id])
        report["change"] = "residual_head" if residual else "hour_return"
        reports.append(report)
        print(json.dumps({"id": model_id, "adopted": report["adopted"], "eval": report["eval_accuracy"]}), flush=True)
    return reports


def train_snxx(previous: float) -> dict:
    bars = pd.read_parquet(ROOT / "market_data/us_60m/SNXX/2026.parquet")
    rows, _ = snxx_rows(bars, barrier=0.03)
    # Rebuild the same completed-bar extension from the label module's row order by a second pass.
    from research.after_open_3d5pct.models.SNXX_3d3pct.labels import regular_bars
    regular = regular_bars(bars)
    extension, drawdown = [], []
    for day in rows.session_date:
        stamp = pd.Timestamp(day) + pd.Timedelta(hours=11, minutes=30)
        done = regular.loc[regular.end.le(stamp)].tail(20)
        last = float(done.close.iloc[-1])
        extension.append(last / float(done.high.max()) - 1)
        drawdown.append(last / float(done.close.max()) - 1)
    rows = rows.copy()
    rows["extension_20"] = extension
    rows["drawdown_20"] = drawdown
    features = SNXX_FEATURES + ["extension_20", "drawdown_20"]
    parts = {}
    for name, (start, end) in SNXX_FOLDS.items():
        mask = rows.session_date.ge(start) & rows.session_date.lt(end) & rows.label_end.lt(end + " 00:00:00")
        parts[name] = rows.loc[mask]
    model = lgb.LGBMClassifier(n_estimators=40, num_leaves=7, max_depth=3, min_child_samples=8,
                               learning_rate=0.05, reg_lambda=10, n_jobs=1, verbosity=-1, random_state=3566)
    model.fit(parts["fit"][features], parts["fit"].y.to_numpy(),
              eval_set=[(parts["tune"][features], parts["tune"].y.to_numpy())],
              callbacks=[lgb.early_stopping(10, verbose=False)])
    pred = {name: model.predict_proba(part[features])[:, 1] for name, part in parts.items()}
    base = {name: load_split("SNXX_3d3pct", name) for name in ("tune", "cal")}
    adopted = decide(accuracy(parts["tune"].y, pred["tune"]), accuracy(parts["cal"].y, pred["cal"]),
                     accuracy(base["tune"].y, base["tune"].p), accuracy(base["cal"].y, base["cal"].p))
    official = accuracy(parts["eval"].y, pred["eval"]) if adopted else previous
    return {"id": "SNXX_3d3pct", "adopted": adopted, "baseline_eval": previous, "eval_accuracy": official,
            "candidate_eval": accuracy(parts["eval"].y, pred["eval"]), "change": "extension_drawdown",
            "rows": int(len(parts["eval"]))}


def train_soxs(previous: float) -> dict:
    soxs = pd.read_parquet(ROOT / "market_data/us_5m/SOXS/2026.parquet")
    soxx = pd.read_parquet(ROOT / "market_data/us_5m/SOXX/2026.parquet")
    rows, _ = soxs_rows(soxs, soxx, barrier=0.05)
    rows = rows.copy()
    rows["r39_spread"] = rows.SOXS_r_39 - rows.SOXX_r_39
    rows["range_spread"] = rows.SOXS_range_pos - rows.SOXX_range_pos
    rows["volume_spread"] = rows.SOXS_volume_ratio - rows.SOXX_volume_ratio
    features = SOXS_FEATURES + ["r39_spread", "range_spread", "volume_spread"]
    parts = {}
    for name, (start, end) in SOXS_FOLDS.items():
        mask = rows.session_date.ge(start) & rows.session_date.lt(end) & rows.label_end.lt(end + " 00:00:00")
        parts[name] = rows.loc[mask].copy()
    for symbol in ("SOXS", "SOXX"):
        model = lgb.LGBMClassifier(n_estimators=40, num_leaves=7, max_depth=3, min_child_samples=8,
                                   learning_rate=0.05, reg_lambda=10, n_jobs=1, verbosity=-1, random_state=3566)
        model.fit(parts["fit"][features], parts["fit"][f"y_{symbol}"].to_numpy(),
                  eval_set=[(parts["tune"][features], parts["tune"][f"y_{symbol}"].to_numpy())],
                  callbacks=[lgb.early_stopping(10, verbose=False)])
        for name, part in parts.items():
            part[f"p_{symbol}"] = model.predict_proba(part[features])[:, 1]
    def action_accuracy(part):
        return float(np.mean(_action(part.p_SOXS.to_numpy(), part.p_SOXX.to_numpy()) == part.action.to_numpy()))
    base = {name: load_split("SOXS_SOXX_3d5pct", name) for name in ("tune", "cal")}
    adopted = decide(action_accuracy(parts["tune"]), action_accuracy(parts["cal"]),
                     action_accuracy(base["tune"]), action_accuracy(base["cal"]))
    official = action_accuracy(parts["eval"]) if adopted else previous
    return {"id": "SOXS_SOXX_3d5pct", "adopted": adopted, "baseline_eval": previous,
            "eval_accuracy": official, "candidate_eval": action_accuracy(parts["eval"]),
            "change": "relative_path", "rows": int(len(parts["eval"]))}


def main() -> None:
    loaded = load_data(AFTER / "runs/focus_v8_20260927")
    reports = train_probability_models(loaded)
    previous = {item["id"]: item["eval_accuracy"] for item in json.loads(
        (AFTER / "runs/model_registry_v1_upgrades/round_05/result.json").read_text())["models"]}
    reports.append(train_snxx(previous["SNXX_3d3pct"]))
    reports.append(train_soxs(previous["SOXS_SOXX_3d5pct"]))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "result.json").write_text(json.dumps({"round": 6, "window": "development_2026-08-24_2026-09-18",
                                                 "models": reports}, indent=2))
    print(json.dumps({"adopted": sum(item["adopted"] for item in reports), "models": len(reports)}), flush=True)


if __name__ == "__main__":
    main()
