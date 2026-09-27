"""Finite, offline v6 B/C development comparison and prediction replay."""
from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from torch import nn

from .v6_data import ROOT, build, digest


def project_monotone(p: np.ndarray) -> np.ndarray:
    """Nested horizon/target partial order; preserves finite [0,1] estimates."""
    q = np.clip(np.asarray(p, float).reshape(-1, 3, 3), 1e-6, 1-1e-6)
    for _ in range(3):
        q = np.maximum.accumulate(q, axis=1)
        q = np.minimum.accumulate(q, axis=2)
    return q


def violations(p: np.ndarray) -> float:
    q = np.asarray(p).reshape(-1, 3, 3)
    return float(np.mean((np.diff(q, axis=1) < -1e-8).any(axis=(1, 2)) |
                         (np.diff(q, axis=2) > 1e-8).any(axis=(1, 2))))


def _stats(x: np.ndarray) -> np.ndarray:
    """Per-window summary over precisely the same masked source path as C."""
    mask = x[..., 10] > 0
    y = np.where(mask[..., None], x, np.nan)
    n = mask.sum(axis=-1)
    with np.errstate(all="ignore"):
        level_columns = [0, 3, 4, 5, 9, 11] + ([12, 13] if x.shape[-1] >= 14 else [])
        mean = np.nanmean(y[..., level_columns], axis=-2)
        high = np.nanmax(y[..., 1], axis=-1)
        low = np.nanmin(y[..., 2], axis=-1)
        first = np.nanmean(y[..., :max(1, x.shape[-2]//2), 0], axis=-1)
        last = np.nanmean(y[..., x.shape[-2]//2:, 0], axis=-1)
    return np.concatenate((np.expand_dims(n / x.shape[-2], -1), mean,
                           high[..., None], low[..., None], first[..., None], last[..., None]), -1)


def tabular(data: dict[str, np.ndarray], *, group: bool) -> pd.DataFrame:
    n = len(data["x5"])
    parts = []
    # The same 7 + current 5m and 30 + current 60m windows; no future filled slots.
    parts.append(_stats(data["x5"]).reshape(n, -1))
    parts.append(_stats(data["x60"]).reshape(n, -1))
    day = data["xday"]
    for a, b in ((0, 21), (21, 42), (42, 63), (63, 84), (84, 105), (105, 126)):
        parts.append(_stats(day[:, a:b, :]).reshape(n, -1))
    if group:
        seq = data["group_seq"]
        parts.append(seq[:, -1].reshape(n, -1))
        parts.append(np.nanmean(np.where(seq[..., 9:10] > 0, seq, np.nan), axis=1).reshape(n, -1))
        parts.append((seq[:, -1] - seq[:, 0]).reshape(n, -1))
    matrix = np.column_stack(parts)
    return pd.DataFrame(matrix, columns=[f"f_{i:03d}" for i in range(matrix.shape[1])]).replace([np.inf, -np.inf], np.nan)


class PatchBranch(nn.Module):
    def __init__(self, patch: int, max_tokens: int, input_channels: int, width: int = 16):
        super().__init__()
        self.patch = patch
        self.proj = nn.Conv1d(input_channels, width, kernel_size=patch, stride=patch)
        self.pos = nn.Embedding(max_tokens, width)
        self.query = nn.Parameter(torch.zeros(width))
        self.norm = nn.LayerNorm(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = torch.nn.functional.avg_pool1d(x[..., 10].unsqueeze(1), self.patch,
                                               stride=self.patch).squeeze(1) > 0
        z = self.proj(x.transpose(1, 2)).transpose(1, 2)
        z = self.norm(torch.nn.functional.gelu(z) + self.pos(torch.arange(z.shape[1], device=z.device)))
        z = z.masked_fill(~mask[..., None], 0)
        score = (z * self.query).sum(-1) / math.sqrt(z.shape[-1])
        score = score.masked_fill(~mask, -1e4)
        weight = torch.softmax(score, dim=1) * mask
        return (z * weight[..., None]).sum(1) / weight.sum(1, keepdim=True).clamp_min(1e-6)


class CModel(nn.Module):
    def __init__(self, *, group: bool = True, daily: bool = True, input_channels: int = 14,
                 type_identity: bool = True):
        super().__init__()
        self.group = group
        self.daily = daily
        self.type_identity = type_identity
        self.five = PatchBranch(12, 128, input_channels)
        self.hour = PatchBranch(3, 176, input_channels)
        if daily:
            self.day = PatchBranch(6, 21, input_channels)
        if group:
            if type_identity:
                # Fixed strategy slots: trend15/63/126, volatility, liquidity,
                # QQQ. No weekly hash enters the learned identity.
                self.member_weight = nn.Parameter(torch.randn(6, 10, 12) * 0.02)
                self.member_bias = nn.Parameter(torch.zeros(6, 12))
            else:
                self.member = nn.Linear(10, 12)
            self.week = nn.GRU(12, 12, batch_first=True)
            self.gate = nn.Linear(12, 16)
        dim = 32 + (16 if daily else 0) + (28 if group else 0)
        self.head = nn.Sequential(nn.Linear(dim, 32), nn.GELU(), nn.Dropout(0.1), nn.Linear(32, 9))

    def forward(self, x5: torch.Tensor, x60: torch.Tensor, xday: torch.Tensor,
                group_seq: torch.Tensor) -> torch.Tensor:
        five = self.five(x5.flatten(1, 2))
        pieces = [five, self.hour(x60.flatten(1, 2))]
        if self.daily:
            pieces.append(self.day(xday))
        if self.group:
            # Set aggregation avoids counting a multi-label stock as extra rows.
            m = group_seq[..., 9:10]
            if self.type_identity:
                typed = torch.einsum("bwgc,gcd->bwgd", group_seq, self.member_weight) + self.member_bias
            else:
                typed = self.member(group_seq)
            z = torch.nn.functional.gelu(typed) * m
            weekly = z.sum(2) / m.sum(2).clamp_min(1)
            _, h = self.week(weekly)
            relation = h[-1]
            pieces.extend((relation, five * torch.sigmoid(self.gate(relation))))
        return self.head(torch.cat(pieces, 1))


def _metrics(y: np.ndarray, p: np.ndarray) -> dict:
    if not len(y):
        return {"rows": 0}
    z = np.clip(p, 1e-6, 1-1e-6)
    result = {"rows": int(len(y)), "observed": float(np.mean(y)),
              "predicted": float(np.mean(z)), "brier": float(brier_score_loss(y, z)),
              "logloss": float(log_loss(y, z, labels=[0, 1]))}
    result["auc"] = float(roc_auc_score(y, z)) if len(np.unique(y)) == 2 else None
    return result


def _folds(table: pd.DataFrame, config: dict) -> dict[str, np.ndarray]:
    date = table.session_date
    end = pd.to_datetime(table.label_end_at, utc=True)
    available = pd.to_datetime(table.label_available_at, utc=True)
    inner = pd.Timestamp(config["inner_start"], tz="America/New_York").tz_convert("UTC")
    outer = pd.Timestamp(config["outer_start"], tz="America/New_York").tz_convert("UTC")
    decision = pd.to_datetime(table.decision_at, utc=True)
    return {"train": np.flatnonzero(((date < config["train_end_exclusive"]) &
                                     (end < inner) & (available < inner)).to_numpy()),
            "inner": np.flatnonzero(((date >= config["inner_start"]) &
                                     (date < config["outer_start"]) &
                                     (end < outer) & (available < outer)).to_numpy()),
            "outer": np.flatnonzero(((date >= config["outer_start"]) &
                                     (date < config["outer_end_exclusive"])).to_numpy())}


def _torch_arrays(data: dict[str, np.ndarray]) -> tuple[torch.Tensor, ...]:
    return tuple(torch.from_numpy(np.nan_to_num(data[k], nan=0, posinf=0, neginf=0).astype(np.float32))
                 for k in ("x5", "x60", "xday", "group_seq"))


def _predict_c(model: CModel, arrays: tuple[torch.Tensor, ...], indices: np.ndarray) -> np.ndarray:
    model.eval()
    chunks = []
    with torch.no_grad():
        for ix in np.array_split(indices, max(1, math.ceil(len(indices) / 256))):
            if len(ix):
                index = torch.as_tensor(ix, dtype=torch.long)
                chunks.append(torch.sigmoid(model(*(torch.index_select(a, 0, index) for a in arrays))).cpu().numpy().reshape(-1, 3, 3))
    return np.concatenate(chunks)


def _fit_c(name: str, data: dict[str, np.ndarray], folds: dict, config: dict,
           models: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    torch.manual_seed(config["seeds"][0]); np.random.seed(config["seeds"][0])
    torch.set_num_threads(config.get("torch_threads", 1))
    arrays = _torch_arrays(data)
    target = torch.from_numpy(data["y"].astype(np.float32).reshape(-1, 9))
    model = CModel(group=name != "C_patch_no_group", daily=name != "C_patch_no_daily",
                   input_channels=data["x5"].shape[-1],
                   type_identity=config.get("group_type_identity", False))
    opt = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    rng = np.random.default_rng(config["seeds"][0])
    best, state, best_epoch, bad, log = math.inf, None, 0, 0, []
    started = time.monotonic()
    for epoch in range(config["max_epochs"]):
        model.train(); losses = []
        order = rng.permutation(folds["train"])
        for ix in np.array_split(order, max(1, math.ceil(len(order)/config["batch_size"]))):
            index = torch.as_tensor(ix, dtype=torch.long)
            opt.zero_grad(set_to_none=True)
            logits = model(*(torch.index_select(a, 0, index) for a in arrays))
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.index_select(target, 0, index))
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1)
            opt.step(); losses.append(float(loss.item()))
        inner_p = project_monotone(_predict_c(model, arrays, folds["inner"]))
        inner_y = data["y"][folds["inner"]]
        score = _metrics(inner_y[:, 1, 1], inner_p[:, 1, 1])["brier"]
        log.append({"epoch": epoch+1, "train_loss": float(np.mean(losses)),
                    "inner_primary_brier": score, "elapsed_seconds": time.monotonic()-started})
        if score < best - 1e-5:
            best, state, best_epoch, bad = score, copy.deepcopy(model.state_dict()), epoch+1, 0
        else:
            bad += 1
        if bad >= config["patience"] or time.monotonic()-started > config["max_fit_seconds_each"]:
            break
    if state is None:
        raise RuntimeError("C had no usable epoch")
    model.load_state_dict(state)
    torch.save({"state_dict": state, "name": name, "protocol": config["protocol"]}, models/f"{name}.pt")
    return (_predict_c(model, arrays, folds["inner"]), _predict_c(model, arrays, folds["outer"]),
            {"best_epoch": best_epoch, "parameters": sum(p.numel() for p in model.parameters()),
             "seconds": time.monotonic()-started, "epochs": log})


def _fit_b(name: str, data: dict[str, np.ndarray], folds: dict, config: dict,
           models: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    x = tabular(data, group=name == "B_lgbm_group")
    y = data["y"].reshape(-1, 9)
    pi = np.zeros((len(folds["inner"]), 9), float)
    po = np.zeros((len(folds["outer"]), 9), float)
    started = time.monotonic()
    for j in range(9):
        model = lgb.LGBMClassifier(n_estimators=120, num_leaves=7, max_depth=4,
                                   min_child_samples=100, learning_rate=0.035,
                                   reg_lambda=10, verbosity=-1, n_jobs=config.get("lightgbm_jobs", 1),
                                   random_state=config["seeds"][0])
        model.fit(x.iloc[folds["train"]], y[folds["train"], j])
        model.booster_.save_model(str(models/f"{name}_{j}.txt"))
        pi[:, j] = model.predict_proba(x.iloc[folds["inner"]])[:, 1]
        po[:, j] = model.predict_proba(x.iloc[folds["outer"]])[:, 1]
    (models/f"{name}_columns.json").write_text(json.dumps(x.columns.tolist()))
    return pi.reshape(-1, 3, 3), po.reshape(-1, 3, 3), {"seconds": time.monotonic()-started,
                                                           "trees_each": 120, "features": x.shape[1]}


def _date_block_delta(table: pd.DataFrame, ids: np.ndarray, y: np.ndarray,
                      p: np.ndarray, baseline: np.ndarray, seed: int) -> list[float]:
    dates = sorted(table.iloc[ids].session_date.unique())
    if len(dates) < 10:
        return [float("nan"), float("nan")]
    index = {d: np.flatnonzero(table.iloc[ids].session_date.to_numpy() == d) for d in dates}
    rng = np.random.default_rng(seed)
    delta = []
    for _ in range(200):
        chosen = []
        while len(chosen) < len(dates):
            start = rng.integers(0, len(dates))
            chosen.extend(dates[start:min(len(dates), start+5)])
        ix = np.concatenate([index[d] for d in chosen[:len(dates)]])
        delta.append(float(np.mean((y[ix]-p[ix])**2 - (y[ix]-baseline[ix])**2)))
    return [float(v) for v in np.quantile(delta, [0.025, 0.975])]


def run(config_path: Path, dataset: Path, output: Path, *, build_dataset: bool = True) -> dict:
    if output.exists():
        raise FileExistsError(output)
    config = json.loads(config_path.read_text())
    started = time.monotonic()
    if build_dataset:
        build(config, dataset)
    output.mkdir(parents=True)
    models = output/"models"; models.mkdir()
    (output/"config.json").write_text(json.dumps(config, indent=2) + "\n")
    table = pd.read_parquet(dataset/"rows.parquet")
    with np.load(dataset/"features.npz") as z:
        data = {k: z[k] for k in z.files}
    if len(table) != len(data["y"]):
        raise ValueError("row/sequence misalignment")
    folds = _folds(table, config)
    if min(map(len, folds.values())) < 100:
        raise ValueError({k: len(v) for k, v in folds.items()})
    reference = float(np.mean(data["y"][folds["train"], 1, 1]))
    predictions = {}
    report = {"protocol": config["protocol"], "dataset_manifest_sha256": digest(dataset/"manifest.json"),
              "config_sha256": digest(config_path), "fold_rows": {k:len(v) for k,v in folds.items()},
              "daily_126_coverage": {k: {"full": int(table.iloc[v].full_126_prior_days.sum()),
                                         "total":len(v), "median_available": float(table.iloc[v].available_daily_days.median())}
                                     for k,v in folds.items()},
              "train_primary_rate": reference, "trials": {}, "research_status": "development_exposed_not_release"}
    quality_path = ROOT / config["quality_eval_source"]
    quality = pd.read_parquet(quality_path, columns=["sample_id", "quality_eligible"])
    quality_map = quality.set_index("sample_id").quality_eligible.to_dict()
    quality_mask = np.array([quality_map.get(f"{r.symbol}|{r.decision_at}", False)
                             for r in table.itertuples(index=False)], dtype=bool)
    report["quality_eval_source_sha256"] = digest(quality_path)
    y_outer = data["y"][folds["outer"]]
    base_outer = np.full(len(folds["outer"]), reference)
    report["train_rate_outer_primary"] = _metrics(y_outer[:,1,1], base_outer)
    # Run PyTorch before LightGBM to avoid mixed OpenMP thread pools on macOS.
    for name in sorted(config["families"], key=lambda x: 0 if x.startswith("C_") else 1):
        if time.monotonic() - started > config["max_total_seconds"]:
            report["trials"][name] = {"status": "skipped_global_time_budget"}
            continue
        try:
            if name.startswith("B_"):
                inner_raw, outer_raw, detail = _fit_b(name, data, folds, config, models)
            else:
                inner_raw, outer_raw, detail = _fit_c(name, data, folds, config, models)
            pi = project_monotone(inner_raw); po = project_monotone(outer_raw)
            prediction = table.iloc[folds["outer"]][["sample_id", "symbol", "session_date", "decision_at",
                                                      "label_end_at", "label_available_at", "terminal_3d",
                                                      "available_daily_days", "full_126_prior_days",
                                                      "group_valid_count"]].copy()
            for h in range(3):
                for t in range(3):
                    prediction[f"y_{h}_{t}"] = y_outer[:,h,t]
                    prediction[f"p_{h}_{t}"] = po[:,h,t]
            prediction.to_parquet(output/f"predictions_{name}.parquet", index=False)
            metrics = {f"{h}_{t}": _metrics(y_outer[:,h,t], po[:,h,t]) for h in range(3) for t in range(3)}
            outer_table = table.iloc[folds["outer"]].reset_index(drop=True)
            primary_y, primary_p = y_outer[:,1,1], po[:,1,1]
            quality_outer = quality_mask[folds["outer"]]
            breakdown = {
                "quality_gt60_v5_prior_mature": _metrics(primary_y[quality_outer], primary_p[quality_outer]),
                "other": _metrics(primary_y[~quality_outer], primary_p[~quality_outer]),
                "full_126_prior_days": _metrics(primary_y[outer_table.full_126_prior_days.to_numpy(bool)],
                                                  primary_p[outer_table.full_126_prior_days.to_numpy(bool)]),
                "partial_126_prior_days": _metrics(primary_y[~outer_table.full_126_prior_days.to_numpy(bool)],
                                                     primary_p[~outer_table.full_126_prior_days.to_numpy(bool)]),
                "by_date": {}, "by_symbol": {}}
            for key, col in (("by_date", "session_date"), ("by_symbol", "symbol")):
                for value, indices in outer_table.groupby(col).indices.items():
                    breakdown[key][str(value)] = _metrics(primary_y[indices], primary_p[indices])
            (output/f"diagnostics_{name}.json").write_text(json.dumps(breakdown, indent=2, allow_nan=False) + "\n")
            report["trials"][name] = {"status": "complete", "fit": detail,
                "inner_primary": _metrics(data["y"][folds["inner"],1,1], pi[:,1,1]),
                "outer_primary": metrics["1_1"], "outer_targets": metrics,
                "raw_violation_fraction": violations(outer_raw), "projected_violation_fraction": violations(po),
                "quality_subgroup": breakdown["quality_gt60_v5_prior_mature"],
                "full_126_subgroup": breakdown["full_126_prior_days"],
                "outer_brier_delta_vs_train_rate_ci_5day": _date_block_delta(table, folds["outer"],
                    y_outer[:,1,1], po[:,1,1], base_outer, config["seeds"][0])}
            predictions[name] = po
        except Exception as exc:
            report["trials"][name] = {"status":"failed", "error":str(exc)}
    if "B_lgbm_group" in predictions:
        b = predictions["B_lgbm_group"][:,1,1]
        for name,p in predictions.items():
            report["trials"][name]["outer_brier_delta_vs_B_group_ci_5day"] = _date_block_delta(
                table, folds["outer"], y_outer[:,1,1], p[:,1,1], b, config["seeds"][0]+1)
    report["elapsed_seconds"] = time.monotonic()-started
    (output/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def replay(run_dir: Path, dataset: Path, name: str, output: Path) -> None:
    """Offline inference of a saved model on frozen feature rows; no order actions."""
    if output.exists():
        raise FileExistsError(output)
    config = json.loads((run_dir/"config.json").read_text())
    table = pd.read_parquet(dataset/"rows.parquet")
    with np.load(dataset/"features.npz") as z:
        data = {k:z[k] for k in z.files}
    if name.startswith("B_"):
        x = tabular(data, group=name == "B_lgbm_group")
        expected = json.loads((run_dir/"models"/f"{name}_columns.json").read_text())
        if expected != x.columns.tolist():
            raise ValueError("B feature schema mismatch")
        p = np.stack([lgb.Booster(model_file=str(run_dir/"models"/f"{name}_{j}.txt")).predict(x)
                      for j in range(9)], axis=-1).reshape(-1,3,3)
    else:
        checkpoint = torch.load(run_dir/"models"/f"{name}.pt", map_location="cpu", weights_only=False)
        if checkpoint["protocol"] != config["protocol"]:
            raise ValueError("C checkpoint protocol mismatch")
        model = CModel(group=name != "C_patch_no_group", daily=name != "C_patch_no_daily",
                       input_channels=checkpoint["state_dict"]["five.proj.weight"].shape[1],
                       type_identity="member_weight" in checkpoint["state_dict"])
        model.load_state_dict(checkpoint["state_dict"])
        p = _predict_c(model, _torch_arrays(data), np.arange(len(table)))
    p = project_monotone(p)
    rows = table[["sample_id", "symbol", "session_date", "decision_at"]].copy()
    for h in range(3):
        for t in range(3):
            rows[f"p_{h}_{t}"] = p[:,h,t]
    rows.to_parquet(output, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    run_cmd = sub.add_parser("run")
    run_cmd.add_argument("--config", type=Path, required=True)
    run_cmd.add_argument("--dataset", type=Path, required=True)
    run_cmd.add_argument("--output", type=Path, required=True)
    run_cmd.add_argument("--reuse-dataset", action="store_true")
    predict_cmd = sub.add_parser("predict")
    predict_cmd.add_argument("--run", type=Path, required=True)
    predict_cmd.add_argument("--dataset", type=Path, required=True)
    predict_cmd.add_argument("--model", required=True)
    predict_cmd.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "run":
        print(json.dumps(run(args.config, args.dataset, args.output,
                             build_dataset=not args.reuse_dataset), indent=2))
    else:
        replay(args.run, args.dataset, args.model, args.output)


if __name__ == "__main__":
    main()
