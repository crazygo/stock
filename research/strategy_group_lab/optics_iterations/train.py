"""Three finite iterations; outer outcomes never choose model budgets."""
from __future__ import annotations
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pickle
import time
import numpy as np
import pandas as pd
import torch
from lightgbm import LGBMClassifier, early_stopping
from research.strategy_group_lab.models import Estimator, Preprocessor, TCN, BRANCHES, calibrate, apply_calibration, train_status
from research.group_expectation_matrix.build import write_json, file_hash
from .prepare import OLD
from .acquire import FOCUS, ROOT

def stable_key(*parts):
    return hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:16]

def temporal_split(rows, labels, outer_start, scope_mask=None):
    start = pd.Timestamp(outer_start, tz="UTC")
    ready = pd.to_datetime(labels.label_available_at, utc=True)
    known = labels.status.eq("mature").to_numpy() & (ready < start).to_numpy() & (rows.date < outer_start).to_numpy()
    dates = sorted(rows.loc[known, "date"].unique())
    if len(dates) < 21:
        return np.zeros(len(rows), bool), np.zeros(len(rows), bool), None
    cal_start = dates[-20]
    cal = known & (rows.date >= cal_start).to_numpy()
    fit = known & (rows.date < cal_start).to_numpy() & (ready < pd.Timestamp(cal_start, tz="UTC")).to_numpy()
    if scope_mask is not None:
        fit &= scope_mask
        cal &= scope_mask
    return fit, cal, cal_start

def inner_split(rows, labels, fit):
    dates = sorted(rows.loc[fit, "date"].unique())
    if len(dates) < 30:
        return None
    cut = dates[-20]
    ready = pd.to_datetime(labels.label_available_at, utc=True)
    train = fit & (rows.date < cut).to_numpy() & (ready < pd.Timestamp(cut, tz="UTC")).to_numpy()
    valid = fit & (rows.date >= cut).to_numpy()
    return np.flatnonzero(train), np.flatnonzero(valid), cut

def loss(y, p):
    p = np.clip(p, 1e-8, 1-1e-8)
    return float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p)))

def choose_budget(data, rows, labels, fit, algorithm, cfg, seed):
    split = inner_split(rows, labels, fit)
    default = cfg["lgbm"]["n_estimators"] if algorithm == "lgbm" else cfg["tcn"]["epochs"]
    if split is None:
        return default, {"status": "insufficient_inner_dates", "budget": default, "curve": []}
    train, valid, cut = split
    y = labels.hit.to_numpy(float)
    if len(train) < 120 or len(valid) < 40 or len(np.unique(y[train])) < 2 or len(np.unique(y[valid])) < 2:
        return default, {"status": "insufficient_inner_rows_or_classes", "budget": default, "curve": [], "inner_cutoff": cut}
    prep = Preprocessor().fit(data, train)
    static = prep.transform(data)
    curve = []
    if algorithm == "lgbm":
        params = {**cfg["lgbm"], "n_estimators": 500, "random_state": seed}
        model = LGBMClassifier(**params)
        model.fit(static[train], y[train], eval_set=[(static[train], y[train]), (static[valid], y[valid])],
                  eval_names=["train", "inner"], eval_metric="binary_logloss", callbacks=[early_stopping(30, verbose=False)])
        history = model.evals_result_
        for i, (tr, va) in enumerate(zip(history["train"]["binary_logloss"], history["inner"]["binary_logloss"])):
            curve.append({"step": i+1, "train_logloss": float(tr), "inner_logloss": float(va)})
        budget = int(model.best_iteration_)
    else:
        torch.manual_seed(seed)
        model = TCN(static.shape[1], cfg["tcn"]["channels"])
        opt = torch.optim.AdamW(model.parameters(), lr=cfg["tcn"]["learning_rate"], weight_decay=cfg["tcn"]["weight_decay"])
        rng = np.random.default_rng(seed)
        batch = cfg["tcn"]["batch_size"]
        arrays = [torch.from_numpy(data[k][train]) for k in BRANCHES]
        validation = [torch.from_numpy(data[k][valid]) for k in BRANCHES]
        st, sv = torch.from_numpy(static[train]), torch.from_numpy(static[valid])
        target, target_v = torch.from_numpy(y[train].astype(np.float32)), torch.from_numpy(y[valid].astype(np.float32))
        best, budget, stale = float("inf"), 1, 0
        for epoch in range(1, 31):
            model.train()
            total = 0.
            order = rng.permutation(len(train))
            for at in range(0, len(train), batch):
                ix = order[at:at+batch]
                opt.zero_grad(set_to_none=True)
                l = torch.nn.functional.binary_cross_entropy_with_logits(model([x[ix] for x in arrays], st[ix]), target[ix])
                l.backward()
                opt.step()
                total += float(l.detach())*len(ix)
            model.eval()
            with torch.no_grad():
                vl = float(torch.nn.functional.binary_cross_entropy_with_logits(model(validation, sv), target_v))
            curve.append({"step": epoch, "train_logloss": total/len(train), "inner_logloss": vl})
            if vl < best-1e-5:
                best, budget, stale = vl, epoch, 0
            else:
                stale += 1
            if stale >= 5:
                break
    return budget, {"status": "inner_temporal_selection", "budget": budget, "curve": curve,
                    "inner_cutoff": cut, "inner_train_n": len(train), "inner_validation_n": len(valid),
                    "original_budget": default, "outer_labels_used": False}

def run_round(output, round_id):
    cfg = json.loads((output/"config.json").read_text())
    rows = pd.read_parquet(output/"data/rows.parquet")
    all_labels = pd.read_parquet(output/"data/labels.parquet")
    old_bindings = json.loads((OLD/"bindings.json").read_text())
    destination = output/round_id
    destination.mkdir(exist_ok=True)
    signature_files = [Path(__file__), Path(__file__).with_name("features.py"),
                       Path(__file__).parents[1]/"models.py", output/"config.json", output/"source_manifest.json",
                       output/"data/rows.parquet", output/"data/labels.parquet"]
    signature_files += [output/"data"/(("enhanced" if round_id in ["R2", "R3"] else "features")+f"_{g}.npz") for g in [5, 60]]
    signature = {str(p.relative_to(ROOT)): file_hash(p) for p in signature_files}
    registration = destination/"registration.json"
    if registration.exists() and json.loads(registration.read_text())["files"] != signature:
        raise ValueError("Refusing to reuse trials after code, inputs or configuration changed")
    if not registration.exists():
        write_json(registration, {"registered_at": datetime.now(timezone.utc).isoformat(), "round": round_id, "files": signature})
    chain_mask = rows.symbol.isin(FOCUS).to_numpy()
    records, start_time = [], time.monotonic()
    for gran in [5, 60]:
        file = ("enhanced" if round_id in ["R2", "R3"] else "features")+f"_{gran}.npz"
        data = dict(np.load(output/"data"/file))
        for exp in cfg["expectations"]:
            labels = all_labels[all_labels.expectation_id == exp["id"]].sort_values("row_id").reset_index(drop=True)
            assert np.array_equal(rows.row_id, labels.row_id)
            for algorithm in cfg["algorithms"]:
                for fold in cfg["folds"]:
                    scopes = ["pooled"] if round_id == "R0" else ["pooled", "chain"]
                    for scope in scopes:
                        model_id = stable_key(round_id, gran, exp["id"], algorithm, fold["id"], scope)
                        stem = destination/model_id
                        if stem.with_suffix(".json").exists() and stem.with_suffix(".parquet").exists():
                            records.append(json.loads(stem.with_suffix(".json").read_text()))
                            continue
                        fit, cal, cal_start = temporal_split(rows, labels, fold["outer_start"], chain_mask if scope == "chain" else None)
                        outer = rows.date.between(fold["outer_start"], fold["outer_end"]).to_numpy().copy()
                        if scope == "chain":
                            outer &= chain_mask
                        ti, ci, fi = np.flatnonzero(outer), np.flatnonzero(cal), np.flatnonzero(fit)
                        seed = cfg["seed"]+int(stable_key(gran, exp["id"], algorithm, fold["id"], scope)[:6], 16)
                        record = {"model_id": model_id, "round": round_id, "granularity": gran, "algorithm": algorithm,
                                  "expectation": exp["id"], "fold": fold["id"], "scope": scope,
                                  "outer_start": fold["outer_start"], "outer_end": fold["outer_end"],
                                  "calibration_start": cal_start, "calibration_n": len(ci), "prediction_n": len(ti),
                                  "source_features": file, "seed": seed}
                        try:
                            if round_id == "R0":
                                old = next(b for b in old_bindings if b["expectation"] == exp["id"] and b["algorithm"] == algorithm
                                           and b["granularity"] == gran and b["scope"] == "pooled" and b["group_id"] == "all:all")
                                old_fold = "july" if fold["id"] == "july" else "august"
                                old_stem = OLD/"trials"/f"{old['id']}_{old_fold}"
                                with old_stem.with_suffix(".pkl").open("rb") as f:
                                    model = pickle.load(f)
                                calibration = json.loads(old_stem.with_suffix(".pool.json").read_text())
                                record.update(status="frozen_replay", old_model=str(old_stem.relative_to(ROOT)), old_fold=old_fold,
                                              transfer_to_expanded_pool=True)
                            else:
                                counts, reason = train_status(rows, fi, labels.hit.to_numpy(float)[fi], cfg)
                                record.update(counts)
                                if reason:
                                    raise ValueError("Insufficient training: "+reason)
                                train_cfg = copy.deepcopy(cfg)
                                if round_id == "R3":
                                    budget, trace = choose_budget(data, rows, labels, fit, algorithm, cfg, seed)
                                    if algorithm == "lgbm": train_cfg["lgbm"]["n_estimators"] = budget
                                    else: train_cfg["tcn"]["epochs"] = budget
                                    record["learning_curve"] = trace
                                model = Estimator(algorithm, train_cfg, seed).fit(data, fi, labels.hit.to_numpy(float)[fi])
                                cp = model.predict(data, ci)
                                calibration = calibrate(cp, labels.hit.to_numpy(float)[ci], rows.date.to_numpy()[ci])
                                record.update(status="trained", training_config={"lgbm": train_cfg["lgbm"], "tcn": train_cfg["tcn"]})
                                with stem.with_suffix(".pkl").open("wb") as f:
                                    pickle.dump(model, f)
                            raw = model.predict(data, ti)
                            pred = pd.DataFrame({"row_id": ti, "raw_p": raw, "p": apply_calibration(raw, calibration)})
                            record["calibration"] = calibration
                            pred.to_parquet(stem.with_suffix(".parquet"), index=False)
                        except Exception as exc:
                            record.update(status="failed", reason=f"{type(exc).__name__}: {exc}")
                            pd.DataFrame({"row_id": pd.Series(dtype=int), "raw_p": pd.Series(dtype=float), "p": pd.Series(dtype=float)}).to_parquet(stem.with_suffix(".parquet"), index=False)
                        write_json(stem.with_suffix(".json"), record)
                        records.append(record)
                        print(f"{round_id} {len(records)} {exp['id']} {algorithm} {gran}m {fold['id']} {scope} {record['status']} {time.monotonic()-start_time:.0f}s", flush=True)
    write_json(destination/"trials.json", records)
    write_json(destination/"complete.json", {"completed_at": datetime.now(timezone.utc).isoformat(), "trial_count": len(records),
               "status_counts": pd.Series([r["status"] for r in records]).value_counts().to_dict(),
               "source_hashes": {str(p.relative_to(ROOT)): file_hash(p) for p in [Path(__file__), output/"config.json", output/"source_manifest.json"]}})

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("output", type=Path)
    p.add_argument("--round", required=True, choices=["R0", "R1", "R2", "R3"])
    args = p.parse_args()
    run_round(args.output.resolve(), args.round)
