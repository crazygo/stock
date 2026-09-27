"""Pre-registered hourly H/HA/HB/HAB LightGBM and small TCN run.

Run from the repository root. This is a development experiment on a retrospective
stock pool, not an independently validated model or a live trading signal.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import time
import traceback
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.linear_model import LogisticRegression
import torch
from torch import nn

from .hourly_v3_data import Bundle, build_bundle, SEQ_CHANNELS


UTC = ZoneInfo("UTC")
REPO = Path(__file__).resolve().parents[2]
PROJECT = Path(__file__).resolve().parent
SEED = 3505
HOURS_V3 = ("10:30", "11:30", "12:30", "13:30", "14:30", "15:30")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _scores(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    p = np.clip(np.asarray(p, dtype=float), 1e-7, 1-1e-7)
    return {"brier": float(brier_score_loss(y, p)),
            "log_loss": float(log_loss(y, p, labels=[0, 1])),
            "observed_rate": float(np.mean(y)), "predicted_mean": float(np.mean(p))}


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-5, 1-1e-5)
    return np.log(p/(1-p))


def _fit_calibrator(y: np.ndarray, p: np.ndarray) -> dict:
    if len(y) < 100 or len(np.unique(y)) < 2:
        raise ValueError("calibration requires >=100 mature observations and both classes")
    model = LogisticRegression(C=1.0, random_state=3505)
    model.fit(_logit(p).reshape(-1, 1), y)
    return {"kind": "sigmoid_logit_v1", "coefficient": float(model.coef_[0, 0]),
            "intercept": float(model.intercept_[0]), "n": int(len(y)),
            "positives": int(y.sum())}


def apply_calibrator(p: np.ndarray, calibrator: dict) -> np.ndarray:
    z = calibrator["coefficient"]*_logit(p) + calibrator["intercept"]
    return 1/(1+np.exp(-np.clip(z, -30, 30)))


def _select_threshold(config: dict, y: np.ndarray, p: np.ndarray) -> dict:
    baseline = float(np.mean(y))
    for threshold in config["threshold_grid"]:
        selected = p >= threshold
        if selected.sum() >= config["threshold_min_recommendations"]:
            precision = float(np.mean(y[selected]))
            if precision >= baseline + config["threshold_min_precision_lift_absolute"]:
                return {"status": "development_selected", "threshold": threshold,
                        "selected": int(selected.sum()), "hits": int(y[selected].sum()),
                        "precision": precision, "pool_rate": baseline}
    return {"status": "disabled_no_inner_threshold_met_rule", "threshold": None,
            "pool_rate": baseline, "grid": config["threshold_grid"]}


class CausalBlock(nn.Module):
    def __init__(self, channels: int, kernel: int, dilation: int, dropout: float):
        super().__init__()
        pad = (kernel-1)*dilation
        self.pad = pad
        self.a = nn.Conv1d(channels, channels, kernel, padding=pad, dilation=dilation)
        self.b = nn.Conv1d(channels, channels, kernel, padding=pad, dilation=dilation)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        z = self.drop(torch.relu(self.a(x)[..., :-self.pad])) * mask
        z = self.drop(self.b(z)[..., :-self.pad]) * mask
        return torch.relu(x+z) * mask


class Encoder(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        channels = config["channels"]
        self.project = nn.Conv1d(7, channels, 1)
        self.blocks = nn.ModuleList([
            CausalBlock(channels, config["kernel_size"], d, config["dropout"])
            for d in config["dilations"]
        ])

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        mask = seq[..., 6:7].transpose(1, 2)
        x = self.project(seq.transpose(1, 2)) * mask
        for block in self.blocks:
            x = block(x, mask)
        mean = (x*mask).sum(dim=2) / mask.sum(dim=2).clamp(min=1)
        maximum = x.max(dim=2).values
        has_data = (mask.sum(dim=2) > 0).to(x.dtype)
        return torch.cat((mean, maximum), dim=1)*has_data


class MultiBranchTCN(nn.Module):
    def __init__(self, config: dict, input_set: str, static_dim: int):
        super().__init__()
        self.input_set = input_set
        self.h = Encoder(config)
        if "A" in input_set:
            self.post = Encoder(config)
            self.pre = Encoder(config)
        if "B" in input_set:
            self.b = Encoder(config)
        branches = 1 + (2 if "A" in input_set else 0) + (1 if "B" in input_set else 0)
        self.output = nn.Sequential(
            nn.Linear(branches*config["channels"]*2+static_dim, config["fusion_width"]),
            nn.ReLU(), nn.Dropout(config["dropout"]),
            nn.Linear(config["fusion_width"], 1))

    def forward(self, h: torch.Tensor, post: torch.Tensor, pre: torch.Tensor,
                b: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        parts = [self.h(h)]
        if "A" in self.input_set:
            parts.extend((self.post(post), self.pre(pre)))
        if "B" in self.input_set:
            parts.append(self.b(b))
        parts.append(static)
        return self.output(torch.cat(parts, dim=1)).squeeze(1)


def _static(bundle: Bundle, input_set: str) -> np.ndarray:
    r = bundle.rows
    cols = ["h_range_actual", "h_range_relative", "h_dollar_actual_log", "h_dollar_relative"]
    if "A" in input_set:
        cols += ["a_post_range", "a_post_dollar_actual_log", "a_post_dollar_relative",
                 "a_post_coverage", "a_pre_range", "a_pre_dollar_actual_log",
                 "a_pre_dollar_relative", "a_pre_coverage"]
    if "B" in input_set:
        cols += ["b_range_actual", "b_range_relative", "b_dollar_actual_log", "b_dollar_relative"]
    data = r[cols].to_numpy(np.float32, copy=True)
    for j, c in enumerate(cols):
        if c.endswith("range_actual"):
            data[:, j] *= 100
        elif "dollar_actual_log" in c:
            data[:, j] /= 20
    hour = r["cutoff_et"].map({h: i for i, h in enumerate(HOURS_V3)}).to_numpy(np.float32)
    return np.clip(np.nan_to_num(np.column_stack([data, hour]), nan=0.0, posinf=10, neginf=-10), -10, 10).astype(np.float32)


def _tabular(bundle: Bundle, input_set: str) -> pd.DataFrame:
    r = bundle.rows
    columns = [c for c in r if c.startswith("h_")]
    if "A" in input_set:
        columns += [c for c in r if c.startswith("a_")]
    if "B" in input_set:
        columns += [c for c in r if c.startswith("b_")]
    if "A" in input_set and "B" in input_set:
        columns += [c for c in r if c.startswith("ab_")]
    out = r[columns].copy()
    out["cutoff_index"] = r["cutoff_et"].map({h: i for i, h in enumerate(HOURS_V3)}).astype(float)
    return out.replace([np.inf, -np.inf], np.nan)


def _fit_tcn(bundle: Bundle, input_set: str, train: np.ndarray, inner: np.ndarray,
             calibration: np.ndarray, outer: np.ndarray, config: dict,
             model_path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    model_cfg = config["tcn"]
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    torch.set_num_threads(4)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    static = _static(bundle, input_set)
    model = MultiBranchTCN(model_cfg, input_set, static.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=model_cfg["learning_rate"],
                                  weight_decay=model_cfg["weight_decay"])
    loss_fn = nn.BCEWithLogitsLoss()
    targets = bundle.rows["target"].to_numpy(np.float32)
    arrays = [bundle.h_seq, bundle.post_seq, bundle.pre_seq, bundle.b_seq]

    def tensor_batch(indices: np.ndarray):
        seqs = [torch.from_numpy(np.clip(a[indices], -50, 50)).to(device) for a in arrays]
        return (*seqs, torch.from_numpy(static[indices]).to(device))

    def predict(indices: np.ndarray) -> np.ndarray:
        model.eval()
        result = []
        with torch.no_grad():
            for chunk in np.array_split(indices, max(1, math.ceil(len(indices)/512))):
                if not len(chunk):
                    continue
                result.append(torch.sigmoid(model(*tensor_batch(chunk))).cpu().numpy())
        return np.concatenate(result)

    import math
    best, best_epoch, best_state, bad = float("inf"), -1, None, 0
    epoch_log = []
    rng = np.random.default_rng(config["seed"])
    training_started = time.monotonic()
    for epoch in range(model_cfg["max_epochs"]):
        epoch_started = time.monotonic()
        model.train()
        shuffled = rng.permutation(train)
        losses = []
        for ix in np.array_split(shuffled, max(1, math.ceil(len(shuffled)/model_cfg["batch_size"]))):
            optimizer.zero_grad(set_to_none=True)
            x = tensor_batch(ix)
            pred = model(*x)
            loss = loss_fn(pred, torch.from_numpy(targets[ix]).to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.item()))
        inner_score = _scores(targets[inner], predict(inner))["brier"]
        epoch_log.append({"epoch": epoch+1, "train_loss": float(np.mean(losses)),
                          "inner_brier": inner_score,
                          "epoch_seconds": time.monotonic() - epoch_started,
                          "cumulative_seconds": time.monotonic() - training_started})
        if inner_score < best - 1e-5:
            best, best_epoch, best_state, bad = inner_score, epoch+1, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= model_cfg["patience"]:
                break
    if best_state is None:
        raise RuntimeError("TCN training produced no usable epoch")
    model.load_state_dict(best_state)
    torch.save({"state_dict": best_state, "input_set": input_set,
                "static_columns": static.shape[1], "config": model_cfg}, model_path)
    return predict(outer), predict(calibration), {"inner_best_brier": best, "best_epoch": best_epoch,
                            "epochs": epoch_log, "parameters": sum(p.numel() for p in model.parameters()),
                            "device": str(device)}


def _fold_indices(bundle: Bundle, fold: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    r = bundle.rows
    inner_start = pd.Timestamp(fold["early_start"], tz="America/New_York").tz_convert("UTC")
    cal_start = pd.Timestamp(fold["calibration_start"], tz="America/New_York").tz_convert("UTC")
    outer_start = pd.Timestamp(fold["outer_start"], tz="America/New_York").tz_convert("UTC")
    outer_end = pd.Timestamp(fold["outer_end_exclusive"], tz="America/New_York").tz_convert("UTC")
    decision = pd.to_datetime(r["decision_at"], utc=True)
    available = pd.to_datetime(r["label_available_at"], utc=True)
    end = pd.to_datetime(r["label_end_at"], utc=True)
    train_mask = (decision < inner_start) & (end < inner_start) & (available < inner_start)
    inner_mask = (decision >= inner_start) & (decision < cal_start) & (end < cal_start) & (available < cal_start)
    cal_mask = (decision >= cal_start) & (decision < outer_start) & (end < outer_start) & (available < outer_start)
    outer_mask = (decision >= outer_start) & (decision < outer_end)
    ids = (np.flatnonzero(train_mask.to_numpy()), np.flatnonzero(inner_mask.to_numpy()),
           np.flatnonzero(cal_mask.to_numpy()), np.flatnonzero(outer_mask.to_numpy()))
    return (*ids, {"train": len(ids[0]), "inner": len(ids[1]), "calibration": len(ids[2]), "outer": len(ids[3]),
                    "purged_before_inner": int(((decision < inner_start) & ~train_mask).sum()),
                    "purged_inner": int(((decision >= inner_start) & (decision < cal_start) & ~inner_mask).sum()),
                    "purged_calibration": int(((decision >= cal_start) & (decision < outer_start) & ~cal_mask).sum())})


def run(config_path: Path, output: Path, smoke_symbols: set[str] | None = None) -> None:
    if output.exists():
        raise FileExistsError(f"experiment output already exists: {output}")
    config = json.loads(config_path.read_text())
    if config["universe_mode"] != "current_universe_retrospective":
        raise ValueError("pilot script requires explicit retrospective universe declaration")
    if tuple(config["decision_hours_et"]) != HOURS_V3:
        raise ValueError("hourly_v3 feature schema requires its declared six-hour schedule")
    started = time.monotonic()
    print("building source-prefix dataset", flush=True)
    bundle = build_bundle(REPO, config, smoke_symbols)
    output.mkdir(parents=True)
    (output/"models").mkdir()
    (output/"config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False)+"\n")
    rows = bundle.rows
    feature_columns = [c for c in rows if c.startswith(("h_", "a_", "b_", "ab_"))]
    rows[["sample_id", "symbol", "session_date", "cutoff_et", "cutoff_at", "decision_at", *feature_columns]].to_parquet(output/"features.parquet")
    rows[["sample_id", "entry_at", "entry_price", "label_end_at", "label_available_at", "target"]].to_parquet(output/"outcomes.parquet")
    np.savez(output/"sequences.npz", h=bundle.h_seq, post=bundle.post_seq,
             pre=bundle.pre_seq, b=bundle.b_seq)
    git = subprocess.run(["git", "status", "--short"], cwd=REPO, capture_output=True, text=True).stdout
    files = bundle.source_paths + [str(config_path.relative_to(REPO)),
                                   str((PROJECT/"hourly_v3_data.py").relative_to(REPO)),
                                   str((PROJECT/"train_hourly_v3.py").relative_to(REPO)),
                                   str((PROJECT/"docs/14_hourly_v3_training_registration.md").relative_to(REPO))]
    manifest = {"created_at": datetime.now(UTC).isoformat(), "config_id": config["experiment_id"],
                "scope": "development_exploratory_retrospective_universe",
                "synthetic_availability": config["feature_availability"],
                "schedule_version": config["schedule_version"],
                "feature_schema": config["feature_schema"],
                "supported_cutoffs_et": config["decision_hours_et"],
                "sequence_channels": SEQ_CHANNELS,
                "sequence_lengths": {"h": config["history_sessions"], "post": 48,
                                     "pre": 66, "b": 72},
                "feature_columns": feature_columns,
                "preprocessing": "lgbm_nan_native; tcn_static_nan_to_zero_clip_-10_10; sequences_clip_-50_50_mask_channel_6",
                "coverage": bundle.coverage, "exclusions": bundle.exclusions,
                "total_eligible": len(rows), "dates": int(rows.session_date.nunique()),
                "symbols": int(rows.symbol.nunique()), "observed_rate": float(rows.target.mean()),
                "environment": {"python": os.sys.version.split()[0], "lightgbm": lgb.__version__,
                                "torch": torch.__version__, "sklearn": sklearn.__version__,
                                "pandas": pd.__version__, "numpy": np.__version__},
                "sources_sha256": {p: _sha256(REPO/p) for p in files},
                "git_status_at_start": git.splitlines(), "smoke_symbols": sorted(smoke_symbols) if smoke_symbols else None}
    (output/"manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False)+"\n")
    print(f"eligible={len(rows)} dates={rows.session_date.nunique()} symbols={rows.symbol.nunique()} build_seconds={time.monotonic()-started:.1f}", flush=True)

    predictions, metrics, trials = [], [], []
    y = rows["target"].to_numpy(int)
    for fold in config["folds"]:
        train, inner, calibration, outer, audit = _fold_indices(bundle, fold)
        print(f"{fold['name']} fold={audit}", flush=True)
        if min(len(train), len(inner), len(calibration), len(outer)) == 0 or len(np.unique(y[train])) < 2:
            raise RuntimeError(f"unusable fold {fold['name']}: {audit}")
        b0 = float(np.mean(y[train]))
        cutoff_train = rows.iloc[train]["cutoff_et"]
        b1_by_hour = {hour: (int(y[train][cutoff_train.to_numpy() == hour].sum())+100*b0) /
                     (int((cutoff_train.to_numpy() == hour).sum())+100)
                     for hour in config["decision_hours_et"]}
        common = {"fold": fold["name"], "train_count": len(train), "inner_count": len(inner),
                  "calibration_count": len(calibration),
                  "outer_count": len(outer), "outer_dates": int(rows.iloc[outer].session_date.nunique())}
        for name, proba in (("B0", np.full(len(outer), b0)),
                            ("B1_hour", rows.iloc[outer]["cutoff_et"].map(b1_by_hour).to_numpy(float))):
            metric = {**common, "model": name, "input_set": "common", "score_kind": "baseline", **_scores(y[outer], proba)}
            metric["brier_1130"] = _scores(y[outer][rows.iloc[outer].cutoff_et.to_numpy()=="11:30"],
                                            proba[rows.iloc[outer].cutoff_et.to_numpy()=="11:30"])["brier"]
            metrics.append(metric)
            predictions.extend({"sample_id": rows.iloc[ix].sample_id, "symbol": rows.iloc[ix].symbol,
                                "session_date": rows.iloc[ix].session_date,
                                "cutoff_et": rows.iloc[ix].cutoff_et, "fold": fold["name"],
                                "model": name, "input_set": "common", "target": int(y[ix]),
                                "probability": float(p), "raw_probability": float(p),
                                "score_kind": "baseline"} for ix, p in zip(outer, proba))
        for group in config["input_sets"]:
            X = _tabular(bundle, group)
            for family in config["models"]:
                key = f"{family}_{group}"
                checkpoint = output/"models"/f"{fold['name']}_{key}.{'txt' if family=='lgbm' else 'pt'}"
                start = time.monotonic()
                try:
                    if family == "lgbm":
                        clf = lgb.LGBMClassifier(**config["lightgbm"], random_state=config["seed"])
                        inner_curve = {}
                        iteration_seconds = []
                        fit_started = time.monotonic()
                        def record_iteration_time(env):
                            iteration_seconds.append(time.monotonic() - fit_started)
                        record_iteration_time.order = 25
                        record_iteration_time.before_iteration = False
                        clf.fit(X.iloc[train], y[train], eval_X=X.iloc[inner], eval_y=y[inner],
                                eval_metric="binary_logloss",
                                callbacks=[lgb.early_stopping(20, verbose=False),
                                           lgb.record_evaluation(inner_curve), record_iteration_time])
                        proba = clf.predict_proba(X.iloc[outer])[:, 1]
                        cal_raw = clf.predict_proba(X.iloc[calibration])[:, 1]
                        clf.booster_.save_model(str(checkpoint))
                        details = {"best_iteration": clf.best_iteration_,
                                   "inner_brier": _scores(y[inner], clf.predict_proba(X.iloc[inner])[:, 1])["brier"],
                                   "features": X.columns.tolist(),
                                   "inner_validation_curve": inner_curve,
                                   "iteration_cumulative_seconds": iteration_seconds}
                    else:
                        proba, cal_raw, details = _fit_tcn(bundle, group, train, inner, calibration, outer,
                                                   config, checkpoint)
                    calibrator = _fit_calibrator(y[calibration], cal_raw)
                    cal_probability = apply_calibrator(proba, calibrator)
                    cal_inner = apply_calibrator(cal_raw, calibrator)
                    threshold = _select_threshold(config, y[calibration], cal_inner)
                    cal_path = output/"models"/f"{fold['name']}_{key}.calibration.json"
                    cal_path.write_text(json.dumps({"calibrator": calibrator, "threshold": threshold,
                                                   "calibration_dates": [fold["calibration_start"], fold["outer_start"]]},
                                                  ensure_ascii=False, indent=2)+"\n")
                    details["calibration"] = calibrator
                    details["threshold"] = threshold
                    details["calibration_file"] = cal_path.name
                    elapsed = time.monotonic()-start
                    mask_1130 = rows.iloc[outer].cutoff_et.to_numpy()=="11:30"
                    for kind, score in (("raw", proba), ("calibrated", cal_probability)):
                        metric = {**common, "model": family, "input_set": group,
                                  "score_kind": kind, "seconds": elapsed,
                                  **_scores(y[outer], score),
                                  "brier_1130": _scores(y[outer][mask_1130], score[mask_1130])["brier"]}
                        metrics.append(metric)
                        predictions.extend({"sample_id": rows.iloc[ix].sample_id,
                                            "symbol": rows.iloc[ix].symbol,
                                            "session_date": rows.iloc[ix].session_date,
                                            "cutoff_et": rows.iloc[ix].cutoff_et,
                                            "fold": fold["name"], "model": family,
                                            "input_set": group, "target": int(y[ix]),
                                            "probability": float(p), "raw_probability": float(raw),
                                            "score_kind": kind}
                                           for ix, p, raw in zip(outer, score, proba))
                    trials.append({"fold": fold["name"], "model": family,
                                   "input_set": group, "status": "completed",
                                   "seconds": elapsed, "details": details})
                    print(f"{fold['name']} {key} raw_brier={_scores(y[outer], proba)['brier']:.5f} calibrated_brier={_scores(y[outer], cal_probability)['brier']:.5f} seconds={elapsed:.1f}", flush=True)
                except Exception as exc:
                    trials.append({"fold": fold["name"], "model": family, "input_set": group,
                                   "status": "failed", "error": str(exc), "traceback": traceback.format_exc()})
                    print(f"{fold['name']} {key} FAILED: {exc}", flush=True)
                finally:
                    (output/"trials.jsonl").write_text("".join(json.dumps(t, ensure_ascii=False)+"\n" for t in trials))
                    pd.DataFrame(metrics).to_csv(output/"metrics.csv", index=False)
                    pd.DataFrame(predictions).to_parquet(output/"predictions.parquet", index=False)
    manifest["completed_at"] = datetime.now(UTC).isoformat()
    manifest["total_seconds"] = time.monotonic()-started
    manifest["completed_trials"] = sum(t["status"]=="completed" for t in trials)
    manifest["failed_trials"] = sum(t["status"]=="failed" for t in trials)
    manifest["model_artifacts_sha256"] = {p.name: _sha256(p) for p in sorted((output/"models").iterdir())}
    manifest["result_artifacts_sha256"] = {p.name: _sha256(p) for p in [output/"features.parquet", output/"outcomes.parquet", output/"sequences.npz", output/"predictions.parquet", output/"metrics.csv", output/"trials.jsonl"]}
    (output/"manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False)+"\n")
    print(f"finished completed={manifest['completed_trials']} failed={manifest['failed_trials']}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT/"configs/hourly_v3.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke-symbols", nargs="*", help="engineering check only")
    args = parser.parse_args()
    run(args.config.resolve(), args.output.resolve(),
        set(args.smoke_symbols) if args.smoke_symbols else None)
