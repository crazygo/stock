"""Round 5: one global bias, chosen on tune and confirmed on cal."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[5]
AFTER = ROOT / "research" / "after_open_3d5pct"
RUN = AFTER / "runs" / "model_registry_v1"
ROUND4 = AFTER / "runs" / "model_registry_v1_upgrades" / "round_04" / "result.json"
OUT = AFTER / "runs" / "model_registry_v1_upgrades" / "round_05"
PRIMARY = "3d_5pct"
B_GRID = np.round(np.arange(-2, 2.001, 0.05), 2)


def logit(p: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return np.log(clipped / (1 - clipped))


def expit(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


def accuracy(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((np.asarray(p) >= 0.5) == np.asarray(y).astype(bool)))


def load_split(model_id: str, split: str) -> pd.DataFrame:
    path = RUN / model_id / "reserved" / f"{split}.parquet"
    if not path.exists():
        path = RUN / model_id / f"{split}.parquet"
    return pd.read_parquet(path)


def choose_bias(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    base = logit(p)
    best = (0.0, accuracy(y, p), float(np.mean(p)))
    for shift in B_GRID:
        mapped = expit(base + shift)
        score = accuracy(y, mapped)
        mean_p = float(np.mean(mapped))
        better = score > best[1] + 1e-12 or (abs(score - best[1]) <= 1e-12 and mean_p < best[2] - 1e-12)
        if better:
            best = (float(shift), score, mean_p)
    return best[0], best[1]


def optimize(model_id: str, baseline_eval: float, symbols: list[str] | None) -> dict:
    frames = {}
    for split in ("tune", "cal", "eval"):
        frame = load_split(model_id, split)
        if symbols is not None:
            frame = frame.loc[frame.symbol.isin(symbols)].reset_index(drop=True)
        frames[split] = frame
    y = {split: frames[split][f"y_{PRIMARY}"].to_numpy() for split in frames}
    p = {split: frames[split][f"p_{PRIMARY}"].to_numpy() for split in frames}
    shift, tune_score = choose_bias(y["tune"], p["tune"])
    cal_base = accuracy(y["cal"], p["cal"])
    cal_new = accuracy(y["cal"], expit(logit(p["cal"]) + shift))
    adopted = shift != 0 and cal_new + 1e-12 >= cal_base
    eval_acc = accuracy(y["eval"], expit(logit(p["eval"]) + shift)) if adopted else baseline_eval
    lift = eval_acc / baseline_eval - 1
    return {"id": model_id, "baseline_eval": baseline_eval, "b": shift if adopted else 0.0,
            "tune_accuracy": tune_score, "cal_base": cal_base, "cal_new": cal_new,
            "eval_accuracy": eval_acc, "relative_lift": lift, "passed": bool(lift >= 0.10 - 1e-12),
            "adopted": adopted}


def soxs_action(p_soxs: np.ndarray, p_soxx: np.ndarray) -> np.ndarray:
    out = np.full(len(p_soxs), "flat", dtype=object)
    buy = (p_soxs >= 0.5) | (p_soxx >= 0.5)
    out[buy & (p_soxs >= p_soxx)] = "SOXS"
    out[buy & (p_soxx > p_soxs)] = "SOXX"
    return out


def optimize_single(baseline_eval: float) -> dict:
    frames = {split: load_split("SNXX_3d3pct", split) for split in ("tune", "cal", "eval")}
    y = {split: frames[split]["y"].to_numpy() for split in frames}
    p = {split: frames[split]["p"].to_numpy() for split in frames}
    shift, tune_score = choose_bias(y["tune"], p["tune"])
    cal_base = accuracy(y["cal"], p["cal"])
    cal_new = accuracy(y["cal"], expit(logit(p["cal"]) + shift))
    adopted = shift != 0 and cal_new + 1e-12 >= cal_base
    eval_acc = accuracy(y["eval"], expit(logit(p["eval"]) + shift)) if adopted else baseline_eval
    lift = eval_acc / baseline_eval - 1
    return {"id": "SNXX_3d3pct", "baseline_eval": baseline_eval, "b": shift if adopted else 0.0,
            "tune_accuracy": tune_score, "cal_base": cal_base, "cal_new": cal_new,
            "eval_accuracy": eval_acc, "relative_lift": lift, "passed": bool(lift >= 0.10 - 1e-12),
            "adopted": adopted}


def optimize_soxs(baseline_eval: float) -> dict:
    frames = {split: load_split("SOXS_SOXX_3d5pct", split) for split in ("tune", "cal", "eval")}

    def score(split: str, b_soxs: float, b_soxx: float) -> tuple[float, float]:
        p_soxs = expit(logit(frames[split]["p_SOXS"]) + b_soxs)
        p_soxx = expit(logit(frames[split]["p_SOXX"]) + b_soxx)
        action = soxs_action(p_soxs, p_soxx)
        return float(np.mean(action == frames[split]["action"].to_numpy())), float(np.mean(np.maximum(p_soxs, p_soxx)))

    best = (0.0, 0.0, *score("tune", 0.0, 0.0))
    for b_soxs in np.round(np.arange(-2, 2.001, 0.25), 2):
        for b_soxx in np.round(np.arange(-2, 2.001, 0.25), 2):
            tune_score, mean_p = score("tune", b_soxs, b_soxx)
            better = tune_score > best[2] + 1e-12 or (abs(tune_score - best[2]) <= 1e-12 and mean_p < best[3] - 1e-12)
            if better:
                best = (float(b_soxs), float(b_soxx), tune_score, mean_p)
    cal_base, _ = score("cal", 0.0, 0.0)
    cal_new, _ = score("cal", best[0], best[1])
    adopted = (best[0] != 0 or best[1] != 0) and cal_new + 1e-12 >= cal_base
    eval_acc = score("eval", best[0], best[1])[0] if adopted else baseline_eval
    lift = eval_acc / baseline_eval - 1
    return {"id": "SOXS_SOXX_3d5pct", "baseline_eval": baseline_eval, "b_soxs": best[0] if adopted else 0.0,
            "b_soxx": best[1] if adopted else 0.0, "tune_accuracy": best[2], "cal_base": cal_base,
            "cal_new": cal_new, "eval_accuracy": eval_acc, "relative_lift": lift,
            "passed": bool(lift >= 0.10 - 1e-12), "adopted": adopted}


def main() -> None:
    protocol = json.loads((AFTER / "focus_v8" / "protocol.json").read_text())
    names = protocol["groups"]
    union = sorted({symbol for symbols in names.values() for symbol in symbols})
    previous = {item["id"]: item for item in json.loads(ROUND4.read_text())["models"]}
    rows = []
    for model_id, item in previous.items():
        if item["passed"]:
            rows.append({**item, "round": 4, "frozen": True})
            continue
        if model_id == "SOXS_SOXX_3d5pct":
            rows.append({**optimize_soxs(item["baseline_eval"]), "round": 5})
        elif model_id == "SNXX_3d3pct":
            rows.append({**optimize_single(item["baseline_eval"]), "round": 5})
        elif model_id in {"B_group", "B_no_group", "C_group", "C_no_group"}:
            rows.append({**optimize(model_id, item["baseline_eval"], union), "round": 5})
        else:
            group = model_id.rsplit("_", 1)[-1]
            rows.append({**optimize(model_id, item["baseline_eval"], names[group]), "round": 5})
        print(json.dumps({"id": rows[-1]["id"], "lift": rows[-1]["relative_lift"], "passed": rows[-1]["passed"]}), flush=True)
    # C_no_daily was frozen before round 4.
    rows.append({"id": "C_no_daily", "baseline_eval": 0.4576719576719577, "eval_accuracy": 0.531746,
                 "relative_lift": 0.531746 / 0.4576719576719577 - 1, "passed": True, "round": 1, "frozen": True,
                 "b": -0.45})
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "result.json").write_text(json.dumps({"round": 5, "models": rows}, indent=2))


if __name__ == "__main__":
    main()
