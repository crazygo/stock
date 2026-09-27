"""Offline C/no-daily v7 iterations. `fit` never scores outer; `final` does."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch import nn

from research.after_open_3d5pct.train_multiscale_v6 import CModel, _folds, project_monotone

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "research/after_open_3d5pct"
DATASET = BASE / "runs/multiscale_groups_v61_dataset_20260926_r2"
CONFIG = BASE / "configs/multiscale_groups_v611_typefix.json"
R0 = BASE / "runs/multiscale_groups_v611_typefix_fit_20260926/models/C_patch_no_daily.pt"
MANIFEST_HASH = "56b3d04def58b72e388f98e41c0d72df1b996aef3ecf7c8ee6325349082a143b"
TARGETS = [f"{d}d_{t}pct" for d in (1, 3, 5) for t in (3, 5, 8)]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_data():
    if sha(DATASET / "manifest.json") != MANIFEST_HASH:
        raise ValueError("dataset manifest hash mismatch")
    table = pd.read_parquet(DATASET / "rows.parquet")
    with np.load(DATASET / "features.npz") as z:
        # Deliberately never access xday (or the redundant group snapshot).
        data = {k: z[k] for k in ("x5", "x60", "group_seq", "y")}
    folds = _folds(table, json.loads(CONFIG.read_text()))
    if len(table) != 9222 or {k: len(v) for k, v in folds.items()} != {
        "train": 5606, "inner": 1521, "outer": 1140,
    }:
        raise ValueError("frozen sample or fold count changed")
    arrays = tuple(torch.from_numpy(np.nan_to_num(data[k], nan=0, posinf=0, neginf=0).astype(np.float32))
                   for k in ("x5", "x60", "group_seq"))
    return table, data, folds, arrays


def relative_coordinates(x: torch.Tensor) -> torch.Tensor:
    """Use first observed Open as origin; retain exact valid masks and time/volume fields."""
    flat = x.flatten(1, -2)
    valid = flat[..., 10] > 0
    first = torch.argmax(valid.int(), dim=1)
    origin = torch.gather(flat[..., 12], 1, first[:, None]) * 5
    out = flat.clone()
    out[..., :4] = flat[..., :4] * 50 * valid[..., None]
    out[..., 12] = (5 * flat[..., 12] - origin) * 10 * valid
    return out.reshape_as(x)


def shape_features(x: torch.Tensor) -> torch.Tensor:
    """Observed Close path including gaps between valid bars, anchored at first valid Open."""
    mask = x[..., 10] > 0
    log_open = x[..., 12] * 5
    log_close = log_open + x[..., 0]
    positions = torch.arange(x.shape[1], device=x.device)[None, :].expand_as(mask)
    prev = torch.cummax(torch.where(mask, positions, -1), dim=1).values
    prev = torch.cat((torch.full_like(prev[:, :1], -1), prev[:, :-1]), dim=1)
    prev_close = torch.gather(log_close, 1, prev.clamp_min(0))
    r = torch.where(prev >= 0, log_close - prev_close, log_close - log_open) * mask
    rank = mask.long().cumsum(1) - 1
    count = mask.sum(1).clamp_min(1)
    thirds = (rank * 3 // count[:, None]).clamp(0, 2)
    segments = torch.stack([(r * (thirds == j)).sum(1) for j in range(3)], 1)
    path = r.cumsum(1)
    peak = torch.cummax(torch.cat((torch.zeros_like(path[:, :1]), path), 1), 1).values[:, 1:]
    drawdown = (peak - path).amax(1, keepdim=True)
    gross = r.abs().sum(1, keepdim=True)
    efficiency = segments.sum(1, keepdim=True) / gross.clamp_min(1e-6)
    early = (r.abs() * (thirds == 0)).sum(1) / (mask & (thirds == 0)).sum(1).clamp_min(1)
    late = (r.abs() * (thirds == 2)).sum(1) / (mask & (thirds == 2)).sum(1).clamp_min(1)
    return torch.cat((segments * 100, drawdown * 100, efficiency, (late - early)[:, None] * 100), 1).clamp(-30, 30)


class NoDailyC(CModel):
    def __init__(self, shape: bool):
        super().__init__(group=True, daily=False, input_channels=14, type_identity=True)
        self.shape = shape
        if shape:
            self.head = nn.Sequential(nn.Linear(72, 32), nn.GELU(), nn.Dropout(.1), nn.Linear(32, 9))

    def forward(self, x5: torch.Tensor, x60: torch.Tensor, group: torch.Tensor) -> torch.Tensor:
        a = relative_coordinates(x5)
        b = relative_coordinates(x60)
        five = self.five(a.flatten(1, 2))
        hour = self.hour(b.flatten(1, 2))
        m = group[..., 9:10]
        typed = torch.einsum("bwgc,gcd->bwgd", group, self.member_weight) + self.member_bias
        weekly = (torch.nn.functional.gelu(typed) * m).sum(2) / m.sum(2).clamp_min(1)
        _, h = self.week(weekly)
        relation = h[-1]
        pieces = [five, hour, relation, five * torch.sigmoid(self.gate(relation))]
        if self.shape:
            pieces += [shape_features(x5[:, -1]), shape_features(x60.flatten(1, 2))]
        return self.head(torch.cat(pieces, 1))


def predict(model, arrays, indices, *, baseline=False):
    model.eval()
    out = []
    with torch.no_grad():
        for ix in np.array_split(indices, max(1, math.ceil(len(indices) / 256))):
            if not len(ix):
                continue
            a, b, g = (torch.index_select(t, 0, torch.as_tensor(ix, dtype=torch.long)) for t in arrays)
            logits = model(a, b, torch.empty(0), g) if baseline else model(a, b, g)
            out.append(torch.sigmoid(logits).numpy().reshape(-1, 3, 3))
    return np.concatenate(out)


def metric(y, p):
    y = np.asarray(y).reshape(-1)
    p = np.clip(np.asarray(p).reshape(-1), 1e-6, 1 - 1e-6)
    return {"n": int(len(y)), "rate": float(y.mean()), "mean_p": float(p.mean()),
            "brier": float(np.mean((y-p)**2)),
            "logloss": float(np.mean(-y*np.log(p) - (1-y)*np.log(1-p))),
            "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else None,
            "p05": float(np.quantile(p, .05)), "p50": float(np.quantile(p, .5)),
            "p95": float(np.quantile(p, .95))}


def buckets(y, p):
    y = np.asarray(y).reshape(-1)
    p = np.asarray(p).reshape(-1)
    edges = np.linspace(0, 1, 11)
    out = []
    for j in range(10):
        mask = (p >= edges[j]) & (p < edges[j+1] if j < 9 else p <= 1)
        out.append({"lo": float(edges[j]), "hi": float(edges[j+1]), "n": int(mask.sum()),
                    "mean_p": float(p[mask].mean()) if mask.any() else None,
                    "rate": float(y[mask].mean()) if mask.any() else None})
    return out


def save_predictions(path, table, ix, y, raw):
    projected = project_monotone(raw).reshape(-1, 9)
    frame = table.iloc[ix][["sample_id", "symbol", "session_date", "decision_at"]].reset_index(drop=True).copy()
    for j, name in enumerate(TARGETS):
        frame[f"y_{name}"] = y.reshape(-1, 9)[:, j].astype(np.int8)
        frame[f"raw_{name}"] = raw.reshape(-1, 9)[:, j]
        frame[f"p_{name}"] = projected[:, j]
    frame.to_parquet(path, index=False)
    return frame


def summarize(frame):
    return {name: {"raw": metric(frame[f"y_{name}"], frame[f"raw_{name}"]),
                   "projected": metric(frame[f"y_{name}"], frame[f"p_{name}"]),
                   "buckets10": buckets(frame[f"y_{name}"], frame[f"p_{name}"])}
            for name in TARGETS}


def fit(round_name, output: Path, seed: int):
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    table, data, folds, arrays = read_data()
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    np.random.seed(seed)
    baseline = round_name == "R1control"
    model = (CModel(group=True, daily=False, input_channels=14, type_identity=True)
             if baseline else NoDailyC(shape=round_name == "R2"))
    target = torch.from_numpy(data["y"].astype(np.float32).reshape(-1, 9))
    opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    weights = torch.ones(9)
    if round_name == "R3":
        weights[4] = 3
    rng = np.random.default_rng(seed)
    best, best_epoch, best_state, bad = math.inf, 0, None, 0
    curve = []
    started = time.monotonic()
    try:
        for epoch in range(1, 41):
            model.train()
            losses = []
            order = rng.permutation(folds["train"])
            for ix in np.array_split(order, max(1, math.ceil(len(order)/128))):
                index = torch.as_tensor(ix, dtype=torch.long)
                a, b, g = (torch.index_select(t, 0, index) for t in arrays)
                logits = model(a, b, torch.empty(0), g) if baseline else model(a, b, g)
                loss_matrix = nn.functional.binary_cross_entropy_with_logits(
                    logits, torch.index_select(target, 0, index), reduction="none")
                loss = (loss_matrix * weights).sum(1).mean() / weights.sum()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1)
                opt.step()
                losses.append(float(loss.item()))
            inner_raw = predict(model, arrays, folds["inner"], baseline=baseline)
            train_raw = predict(model, arrays, folds["train"], baseline=baseline)
            inner_p = project_monotone(inner_raw)[:, 1, 1]
            train_p = project_monotone(train_raw)[:, 1, 1]
            inner = metric(data["y"][folds["inner"], 1, 1], inner_p)
            train = metric(data["y"][folds["train"], 1, 1], train_p)
            curve.append({"epoch": epoch, "train_loss9": float(np.mean(losses)),
                          "train_primary": train, "inner_primary": inner,
                          "seconds": time.monotonic()-started})
            if inner["brier"] < best - 1e-5:
                best, best_epoch, best_state, bad = inner["brier"], epoch, copy.deepcopy(model.state_dict()), 0
            else:
                bad += 1
            if bad >= 6:
                break
        model.load_state_dict(best_state)
        inner_raw = predict(model, arrays, folds["inner"], baseline=baseline)
        frame = save_predictions(output/"inner.parquet", table, folds["inner"], data["y"][folds["inner"]], inner_raw)
        torch.save({"state_dict": best_state, "round": round_name, "seed": seed,
                    "shape": getattr(model, "shape", False), "baseline": baseline,
                    "weights": weights.tolist()}, output/"model.pt")
        report = {"status": "complete", "round": round_name, "seed": seed,
                  "best_epoch": best_epoch, "parameters": sum(p.numel() for p in model.parameters()),
                  "seconds": time.monotonic()-started, "curve": curve,
                  "inner_targets": summarize(frame), "outer_scored": False,
                  "hashes": {"manifest": sha(DATASET/"manifest.json"), "features": sha(DATASET/"features.npz"),
                             "rows": sha(DATASET/"rows.parquet"), "script": sha(Path(__file__)),
                             "config": sha(CONFIG)}}
        (output/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
        print(json.dumps({"round": round_name, "seed": seed, "best_epoch": best_epoch,
                          "inner_primary": report["inner_targets"]["3d_5pct"]["projected"],
                          "seconds": report["seconds"]}, indent=2))
    except Exception as exc:
        (output/"failure.json").write_text(json.dumps({"error": repr(exc), "seconds": time.monotonic()-started})+"\n")
        raise


def final(output: Path, runs: dict[str, Path]):
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    table, data, folds, arrays = read_data()
    torch.set_num_threads(1)
    results = {}
    for name, path in runs.items():
        if name == "R0":
            ckpt = torch.load(R0, map_location="cpu", weights_only=False)
            model = CModel(group=True, daily=False, input_channels=14, type_identity=True)
            model.load_state_dict(ckpt["state_dict"])
            baseline = True
        else:
            ckpt = torch.load(path/"model.pt", map_location="cpu", weights_only=False)
            model = NoDailyC(shape=ckpt["shape"])
            model.load_state_dict(ckpt["state_dict"])
            baseline = False
        inner_raw = predict(model, arrays, folds["inner"], baseline=baseline)
        outer_raw = predict(model, arrays, folds["outer"], baseline=baseline)
        inner_frame = save_predictions(output/f"{name}_inner.parquet", table, folds["inner"], data["y"][folds["inner"]], inner_raw)
        outer_frame = save_predictions(output/f"{name}_outer.parquet", table, folds["outer"], data["y"][folds["outer"]], outer_raw)
        results[name] = {"inner": summarize(inner_frame), "outer": summarize(outer_frame),
                         "source": str(path), "model_sha256": sha(R0 if name == "R0" else path/"model.pt"),
                         "max_replay_delta_inner": float(np.max(np.abs(
                             inner_frame.filter(regex="^p_").to_numpy() -
                             pd.read_parquet(path/"inner.parquet").filter(regex="^p_").to_numpy())))
                         if name != "R0" else None}
    (output/"results.json").write_text(json.dumps(results, indent=2, allow_nan=False)+"\n")
    print(json.dumps({k: {"inner": v["inner"]["3d_5pct"]["projected"],
                          "outer": v["outer"]["3d_5pct"]["projected"]}
                      for k, v in results.items()}, indent=2))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--round", choices=["R1", "R1control", "R2", "R3"], required=True)
    f.add_argument("--seed", type=int, required=True)
    f.add_argument("--output", type=Path, required=True)
    q = sub.add_parser("final")
    q.add_argument("--output", type=Path, required=True)
    q.add_argument("--r1", type=Path, required=True)
    q.add_argument("--r2", type=Path, required=True)
    q.add_argument("--r3a", type=Path, required=True)
    q.add_argument("--r3b", type=Path, required=True)
    args = p.parse_args()
    if args.cmd == "fit":
        fit(args.round, args.output, args.seed)
    else:
        final(args.output, {"R0": R0.parent, "R1": args.r1, "R2": args.r2,
                            "R3_seed3566": args.r3a, "R3_seed4577": args.r3b})


if __name__ == "__main__":
    main()
