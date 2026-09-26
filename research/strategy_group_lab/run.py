"""python -m research.strategy_group_lab.run --output <new directory> [--resume]."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import pickle
import time
import traceback
import numpy as np
import pandas as pd
from .data import ROOT, build, provenance, split_masks
from research.group_expectation_matrix.build import file_hash, write_json

HERE = Path(__file__).resolve().parent

def key(*parts):
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]

def baseline(labels, cal, fit, group_mask):
    for mask, status in [(cal & group_mask, "group_calibration"), (fit & group_mask, "group_training"),
                         (cal, "pool_calibration"), (fit, "pool_training")]:
        y = labels.loc[mask, "hit"].to_numpy(float)
        if len(y):
            return float((y.sum()+1)/(len(y)+2)), status, len(y)
    raise ValueError("No baseline data")

def train(cfg, output):
    from .models import Estimator, calibrate, apply_calibration, train_status
    rows = pd.read_parquet(output/"rows.parquet")
    all_labels = pd.read_parquet(output/"labels.parquet")
    masks = np.load(output/"memberships.npz")["mask"]
    groups = json.loads((output/"groups.json").read_text())
    trial_dir = output/"trials"
    trial_dir.mkdir(exist_ok=True)
    trials, bindings = [], {}
    completed = 0
    start = time.monotonic()
    def save(binding, fold, predictions, meta, model=None):
        nonlocal completed
        bid = binding["id"]
        bindings[bid] = binding
        stem = trial_dir/f"{bid}_{fold['id']}"
        if model is not None:
            with stem.with_suffix(".pkl").open("wb") as f:
                pickle.dump(model, f)
        predictions.to_parquet(stem.with_suffix(".parquet"), index=False)
        record = {**meta, "binding_id": bid, "fold": fold["id"], "prediction_rows": len(predictions)}
        write_json(stem.with_suffix(".json"), record)
        trials.append(record)
        completed += 1
        if completed % 20 == 0:
            print(f"trials {completed}: {binding['algorithm']} {binding['expectation']} {binding['scope']} {binding['group_name']} ({time.monotonic()-start:.0f}s)", flush=True)
    def bind(exp, algo, gran, scope, group):
        return {"id": key(exp["id"], algo, str(gran), scope, group["group_id"]), "expectation": exp["id"],
                "expectation_name": exp["name"], "days": exp["days"], "target": exp["target"],
                "algorithm": algo, "granularity": gran, "scope": scope,
                "group_id": group["group_id"], "group_name": group["name"]}
    def cached(binding, fold):
        stem = trial_dir/f"{binding['id']}_{fold['id']}"
        if stem.with_suffix(".json").exists() and stem.with_suffix(".parquet").exists():
            bindings[binding["id"]] = binding
            trials.append(json.loads(stem.with_suffix(".json").read_text()))
            return True
        return False
    empty = pd.DataFrame({"row_id": pd.Series(dtype=int), "raw_p": pd.Series(dtype=float),
                          "p": pd.Series(dtype=float), "b0": pd.Series(dtype=float), "fold": pd.Series(dtype=str)})
    for exp in cfg["expectations"]:
        labels = all_labels[all_labels.expectation_id == exp["id"]].sort_values("row_id").reset_index(drop=True)
        assert np.array_equal(labels.row_id, rows.row_id)
        for fold in cfg["folds"]:
            fit, cal, outer = split_masks(rows, labels, fold)
            for gi, group in enumerate(groups):
                binding = bind(exp, "B0", 0, "baseline", group)
                if cached(binding, fold):
                    continue
                rate, basis, n = baseline(labels, cal, fit, masks[:, gi])
                ids = np.flatnonzero(outer & masks[:, gi])
                pred = pd.DataFrame({"row_id": ids, "raw_p": rate, "p": rate, "b0": rate, "fold": fold["id"]})
                save(binding, fold, pred, {"status": "baseline", "baseline_basis": basis, "baseline_n": n})
            for gran in cfg["granularities"]:
                data = dict(np.load(output/f"features_{gran}.npz"))
                for algorithm in cfg["algorithms"]:
                    seed = cfg["seed"] + int(key(exp["id"], str(gran), algorithm, fold["id"])[:6], 16)
                    global_binding = bind(exp, algorithm, gran, "pooled", groups[0])
                    global_stem = trial_dir/f"{global_binding['id']}_{fold['id']}"
                    global_cache = global_stem.with_suffix(".pool.npz")
                    if global_cache.exists() and global_stem.with_suffix(".pkl").exists():
                        cache = np.load(global_cache)
                        cal_p, test_p = cache["cal_p"], cache["test_p"]
                        global_cal = json.loads(global_stem.with_suffix(".pool.json").read_text())
                        global_model = None
                    else:
                        ids = np.flatnonzero(fit)
                        global_model = Estimator(algorithm, cfg, seed).fit(data, ids, labels.hit.to_numpy(float)[ids])
                        cal_p = np.full(len(rows), np.nan)
                        test_p = np.full(len(rows), np.nan)
                        cal_p[cal] = global_model.predict(data, np.flatnonzero(cal))
                        test_p[outer] = global_model.predict(data, np.flatnonzero(outer))
                        global_cal = calibrate(cal_p[cal], labels.hit.to_numpy(float)[cal], rows.date.to_numpy()[cal])
                        with global_stem.with_suffix(".pkl").open("wb") as f:
                            pickle.dump(global_model, f)
                        np.savez_compressed(global_cache, cal_p=cal_p, test_p=test_p)
                        write_json(global_stem.with_suffix(".pool.json"), global_cal)
                    for gi, group in enumerate(groups):
                        gm = masks[:, gi]
                        rate, basis, bn = baseline(labels, cal, fit, gm)
                        for scope in ["pooled", "group"]:
                            if scope == "group" and group["group_id"] == "all:all":
                                continue
                            binding = bind(exp, algorithm, gran, scope, group)
                            if cached(binding, fold):
                                continue
                            ids = np.flatnonzero(fit & (gm if scope == "group" else True))
                            counts, reason = train_status(rows, ids, labels.hit.to_numpy(float)[ids], cfg)
                            meta = {**counts, "status": "trained", "baseline_basis": basis, "baseline_n": bn,
                                    "fit_before": fold["fit_before"], "calibration_before": fold["outer_start"]}
                            if reason:
                                save(binding, fold, empty, {**meta, "status": "skipped", "reason": reason})
                                continue
                            ci, ti = np.flatnonzero(cal & gm), np.flatnonzero(outer & gm)
                            try:
                                if scope == "pooled":
                                    cp, tp, model = cal_p[ci], test_p[ti], None
                                else:
                                    model = Estimator(algorithm, cfg, seed+gi).fit(data, ids, labels.hit.to_numpy(float)[ids])
                                    cp, tp = model.predict(data, ci), model.predict(data, ti)
                                calibration = calibrate(cp, labels.hit.to_numpy(float)[ci], rows.date.to_numpy()[ci],
                                                        global_cal if scope == "pooled" else None)
                                pred = pd.DataFrame({"row_id": ti, "raw_p": tp, "p": apply_calibration(tp, calibration),
                                                     "b0": rate, "fold": fold["id"]})
                                save(binding, fold, pred, {**meta, "calibration": calibration, "calibration_n": len(ci),
                                                         "model_reference": global_binding["id"] if scope == "pooled" else binding["id"]}, model)
                            except Exception as exc:
                                save(binding, fold, empty, {**meta, "status": "failed", "reason": str(exc), "traceback": traceback.format_exc()})
    write_json(output/"bindings.json", list(bindings.values()))
    write_json(output/"trials.json", trials)
    print(f"TRAINING COMPLETE: {len(bindings)} bindings, {len(trials)} trial records", flush=True)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=HERE/"configs/v1.json")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--stage", choices=["all", "data", "train", "evaluate"], default="all")
    args = p.parse_args()
    cfg = json.loads(args.config.read_text())
    source_hashes = provenance(cfg)
    config_hash = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
    if args.output.exists():
        if not args.resume:
            raise FileExistsError("Choose fresh output or explicitly --resume")
        manifest = json.loads((args.output/"manifest.json").read_text())
        if manifest["config_hash"] != config_hash or manifest["source_files"] != source_hashes:
            raise ValueError("Cannot resume with changed configuration or data")
        # Evaluation/presentation may be repaired independently. Reusing fitted
        # models with changed feature or estimator implementations is forbidden.
        for entry in manifest["code_files"]:
            if Path(entry["path"]).name in ["data.py", "models.py"] and file_hash(ROOT/entry["path"]) != entry["sha256"]:
                raise ValueError(f"Cannot resume changed training implementation: {entry['path']}")
    else:
        args.output.mkdir(parents=True)
        write_json(args.output/"manifest.json", {"protocol": cfg["protocol"], "config": cfg, "config_hash": config_hash,
                   "source_files": source_hashes, "created_at": datetime.now(timezone.utc).isoformat(),
                   "evidence_status": "exposed_history_development_forward_replay", "code_files":
                   [{"path": str(x.relative_to(ROOT)), "sha256": file_hash(x)} for x in sorted(HERE.glob("*.py"))]})
    if args.stage in ["all", "data"] and not (args.output/"data_complete.json").exists():
        build(cfg, args.output)
        write_json(args.output/"data_complete.json", {"complete": True})
    if args.stage in ["all", "train"]:
        train(cfg, args.output)
    if args.stage in ["all", "evaluate"]:
        from .evaluate import evaluate
        evaluate(cfg, args.output)
    write_json(args.output/f"execution_{args.stage}.json", {
        "completed_at": datetime.now(timezone.utc).isoformat(), "stage": args.stage,
        "code_files": [{"path": str(x.relative_to(ROOT)), "sha256": file_hash(x)}
                       for x in sorted(HERE.glob("*.py")) + [ROOT/"research/group_expectation_matrix/build.py", HERE/"report.html"]]})

if __name__ == "__main__":
    main()
