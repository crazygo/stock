"""Launch every registered model into its own run directory."""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from research.after_open_3d5pct.focus_v8.core import GROUPS, PROTOCOL
from research.after_open_3d5pct.focus_v8.run import fit_one, load_data

HERE = Path(__file__).resolve().parent
AFTER = HERE.parent
RUN = AFTER / "runs" / "model_registry_v1"


def _block(y, p) -> dict:
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
        "majority_class_accuracy": (max(base, 1 - base) if n else None),
        "precision_at_0.5": (tp / (tp + fp) if tp + fp else None),
        "recall_at_0.5": (tp / (tp + fn) if tp + fn else None),
        "brier": (float(np.mean((p - y) ** 2)) if n else None),
        "predicted_positive": int(tp + fp),
    }


def _incumbent(spec: dict, cache: dict) -> dict:
    dataset = AFTER / spec["code"]["dataset"]
    if cache.get("root") != dataset:
        cache["root"] = dataset
        cache["loaded"] = load_data(dataset)
    fold = next(item for item in PROTOCOL["folds"] if item["id"] == spec["code"]["fold"])
    output = RUN / spec["id"] / fold["id"]
    result = fit_one(dataset, spec["code"]["route"], spec["code"]["recipe"], fold,
                     cache["loaded"], spec["code"]["seed"], output)
    frame = pd.read_parquet(output / "eval.parquet")
    y = frame["y_" + spec["target"]["primary"]].to_numpy()
    p = frame["p_" + spec["target"]["primary"]].to_numpy()
    groups = {name: _block(y[frame.symbol.isin(symbols)], p[frame.symbol.isin(symbols)])
              for name, symbols in GROUPS.items()}
    union = sorted({s for symbols in GROUPS.values() for s in symbols})
    archived = json.loads((dataset / "final" / spec["id"] / "selected_3566" / "result.json").read_text())
    score = result["results"]["eval"]["score"]
    archived_score = archived["results"]["eval"]["score"]
    return {
        "id": spec["id"],
        "status": "trained",
        "primary": spec["target"]["primary"],
        "universe": spec["universe"]["predict"],
        "eval": _block(y, p),
        "groups": groups,
        "union_21": _block(y[frame.symbol.isin(union)], p[frame.symbol.isin(union)]),
        "eval_score": score,
        "archived_eval_score": archived_score,
        "score_delta": score - archived_score,
        "replay_ok": abs(score - archived_score) <= 1e-4,
        "seconds": result["seconds"],
    }


def _subset(loaded, symbols: list[str]):
    rows, data, y, prior, counts, paths, xbase, curve, relative, arrays, identity = loaded
    mask = rows.symbol.isin(symbols).to_numpy()
    if int(mask.sum()) == 0:
        raise ValueError("group has no rows in the frozen dataset")
    rows = rows.loc[mask].reset_index(drop=True)
    data = {key: value[mask] for key, value in data.items()}
    arrays = tuple(value[torch.as_tensor(mask)] for value in arrays)
    return (rows, data, y[mask], prior[mask], counts[mask], paths[mask], xbase[mask],
            curve[mask], relative[mask], arrays, identity)


def _group(spec: dict, cache: dict) -> dict:
    dataset = AFTER / spec["code"]["dataset"]
    if cache.get("root") != dataset:
        cache["root"] = dataset
        cache["loaded"] = load_data(dataset)
    symbols = GROUPS[spec["code"]["group"]]
    loaded = _subset(cache["loaded"], symbols)
    fold = next(item for item in PROTOCOL["folds"] if item["id"] == spec["code"]["fold"])
    output = RUN / spec["id"] / fold["id"]
    result = fit_one(dataset, spec["code"]["route"], spec["code"]["recipe"], fold,
                     loaded, spec["code"]["seed"], output)
    frame = pd.read_parquet(output / "eval.parquet")
    y = frame["y_" + spec["target"]["primary"]].to_numpy()
    p = frame["p_" + spec["target"]["primary"]].to_numpy()
    return {
        "id": spec["id"],
        "status": "trained",
        "primary": spec["target"]["primary"],
        "universe": spec["universe"]["predict"],
        "eval": _block(y, p),
        "eval_score": result["results"]["eval"]["score"],
        "seconds": result["seconds"],
        "rows_fit_universe": int(loaded[0].shape[0]),
    }


def _extension(spec: dict) -> dict:
    module = importlib.import_module(spec["code"]["module"])
    payload = module.run(RUN / spec["id"])
    payload["primary"] = spec["target"]["primary"]
    payload["universe"] = spec["universe"]["predict"]
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", action="append", default=[])
    args = parser.parse_args()
    registry = json.loads((HERE / "registry.json").read_text())
    chosen = args.only or registry["models"]
    cache: dict = {}
    reports = []
    for model_id in chosen:
        spec = json.loads((HERE / model_id / "spec.json").read_text())
        kind = spec["code"]["kind"]
        if kind == "focus_v8_route":
            reports.append(_incumbent(spec, cache))
        elif kind == "focus_v8_group":
            reports.append(_group(spec, cache))
        else:
            reports.append(_extension(spec))
        print(json.dumps({"id": reports[-1]["id"], "status": reports[-1]["status"]}), flush=True)
    summary = {"experiment": "model_registry_v1", "models": reports}
    RUN.mkdir(parents=True, exist_ok=True)
    name = "ACCURACY.json" if not args.only else "ACCURACY_partial.json"
    (RUN / name).write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
