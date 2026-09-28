"""Round 4: tune-selected probability maps. Eval is scored once after selection."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[5]
AFTER = ROOT / "research" / "after_open_3d5pct"
RUN = AFTER / "runs" / "model_registry_v1"
OUT = AFTER / "runs" / "model_registry_v1_upgrades" / "round_04"
PRIMARY = "3d_5pct"
A_GRID = (0.25, 0.5, 1.0, 1.5, 2.0, 3.0)
B_GRID = np.round(np.arange(-2, 2.01, 0.1), 2)
SYMBOL_B = (-1.0, -0.5, 0.0, 0.5, 1.0)

BASELINE_EVAL = {
    "B_group": 0.5370370370370371,
    "B_no_group": 0.5529100529100529,
    "C_group": 0.5555555555555556,
    "C_no_group": 0.58994708994709,
    "C_no_daily": 0.4576719576719577,
    "SNXX_3d3pct": 0.8666666666666667,
    "SOXS_SOXX_3d5pct": 0.6,
}


def groups() -> dict[str, list[str]]:
    protocol = json.loads((AFTER / "focus_v8" / "protocol.json").read_text())
    return protocol["groups"]


def union_symbols() -> list[str]:
    return sorted({symbol for symbols in groups().values() for symbol in symbols})


def logit(p: np.ndarray) -> np.ndarray:
    clipped = np.clip(p.astype(float), 1e-4, 1 - 1e-4)
    return np.log(clipped / (1 - clipped))


def expit(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


def accuracy(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p >= 0.5) == y.astype(bool)))


def best_platt(y: np.ndarray, p: np.ndarray) -> tuple[float, float, np.ndarray, float]:
    base = logit(p)
    best = (1.0, 0.0, p, accuracy(y, p))
    for scale in A_GRID:
        for shift in B_GRID:
            mapped = expit(scale * base + shift)
            score = accuracy(y, mapped)
            if score > best[3] + 1e-12 or (abs(score - best[3]) <= 1e-12 and abs(shift) + abs(scale - 1) < abs(best[1]) + abs(best[0] - 1)):
                best = (float(scale), float(shift), mapped, score)
    return best


def symbol_bias(frame: pd.DataFrame, p: np.ndarray, y: np.ndarray) -> tuple[dict[str, float], np.ndarray, float]:
    biases = {symbol: 0.0 for symbol in frame["symbol"].unique()}
    current = p.astype(float).copy()
    symbols = frame["symbol"].to_numpy()
    for _ in range(2):
        for symbol, shift0 in list(biases.items()):
            mask = symbols == symbol
            if int(mask.sum()) < 8:
                continue
            best_shift, best_score = shift0, accuracy(y, current)
            for shift in SYMBOL_B:
                trial = current.copy()
                trial[mask] = expit(logit(p[mask]) + shift)
                score = accuracy(y, trial)
                if score > best_score + 1e-12:
                    best_shift, best_score, current = shift, score, trial
            biases[symbol] = best_shift
    return biases, current, accuracy(y, current)


def load_split(model_id: str, split: str) -> pd.DataFrame:
    path = RUN / model_id / "reserved" / f"{split}.parquet"
    if not path.exists():
        path = RUN / model_id / f"{split}.parquet"
    return pd.read_parquet(path)


def slice_frame(frame: pd.DataFrame, symbols: list[str] | None) -> pd.DataFrame:
    if symbols is None:
        return frame.reset_index(drop=True)
    return frame.loc[frame.symbol.isin(symbols)].reset_index(drop=True)


def consider(chosen: dict, name: str, tune_p: np.ndarray, eval_p: np.ndarray, y_tune: np.ndarray, detail: dict) -> dict:
    score = accuracy(y_tune, tune_p)
    if chosen is None or score > chosen["tune_accuracy"] + 1e-12:
        return {"name": name, "tune_p": tune_p, "eval_p": eval_p, "tune_accuracy": score, "detail": detail}
    return chosen


def finish(model_id: str, baseline_eval: float, y_eval: np.ndarray, chosen: dict, y_tune: np.ndarray, base_tune: float) -> dict:
    adopted = chosen["name"] != "baseline"
    eval_p = chosen["eval_p"]
    eval_acc = accuracy(y_eval, eval_p)
    official = eval_acc if adopted else baseline_eval
    lift = official / baseline_eval - 1
    return {
        "id": model_id,
        "baseline_eval": baseline_eval,
        "baseline_tune": base_tune,
        "chosen": chosen["name"] if adopted else "baseline",
        "detail": chosen["detail"] if adopted else {},
        "tune_accuracy": chosen["tune_accuracy"] if adopted else base_tune,
        "eval_accuracy": official,
        "candidate_eval_accuracy": eval_acc,
        "relative_lift": lift,
        "passed": bool(lift >= 0.10 - 1e-12),
        "adopted": adopted,
    }


def optimize_probability_model(model_id: str, symbols: list[str] | None, parent_id: str | None, baseline_eval: float) -> dict:
    tunes, evals = {}, {}
    for split, store in (("tune", tunes), ("eval", evals)):
        frame = slice_frame(load_split(model_id, split), symbols)
        store["frame"] = frame
        store["y"] = frame[f"y_{PRIMARY}"].to_numpy()
        store["p"] = frame[f"p_{PRIMARY}"].to_numpy()
        store["raw"] = frame[f"raw_{PRIMARY}"].to_numpy()
        store["history"] = frame[f"history_{PRIMARY}"].to_numpy()
    if parent_id:
        for split, store in (("tune", tunes), ("eval", evals)):
            parent = slice_frame(load_split(parent_id, split), symbols)
            parent = parent.set_index("sample_id")
            aligned = parent.loc[store["frame"]["sample_id"]]
            store["parent_p"] = aligned[f"p_{PRIMARY}"].to_numpy()
            store["parent_raw"] = aligned[f"raw_{PRIMARY}"].to_numpy()
    y_tune, y_eval = tunes["y"], evals["y"]
    base_tune = accuracy(y_tune, tunes["p"])
    chosen = {"name": "baseline", "tune_p": tunes["p"], "eval_p": evals["p"], "tune_accuracy": base_tune, "detail": {}}
    sources = ["p", "raw", "history"]
    if parent_id:
        sources += ["parent_p", "parent_raw"]
    for source in sources:
        scale, shift, mapped, _ = best_platt(y_tune, tunes[source])
        eval_mapped = expit(scale * logit(evals[source]) + shift)
        chosen = consider(chosen, f"platt:{source}", mapped, eval_mapped, y_tune, {"source": source, "a": scale, "b": shift})
    if parent_id:
        tune_avg = 0.5 * (tunes["p"] + tunes["parent_p"])
        eval_avg = 0.5 * (evals["p"] + evals["parent_p"])
        scale, shift, mapped, _ = best_platt(y_tune, tune_avg)
        chosen = consider(chosen, "platt:avg_parent", mapped, expit(scale * logit(eval_avg) + shift), y_tune,
                          {"source": "avg_parent", "a": scale, "b": shift})
    biases, mapped, score = symbol_bias(tunes["frame"], tunes["p"], y_tune)
    eval_mapped = evals["p"].astype(float).copy()
    for symbol, shift in biases.items():
        mask = evals["frame"]["symbol"].to_numpy() == symbol
        eval_mapped[mask] = expit(logit(evals["p"][mask]) + shift)
    if score > base_tune + 1e-12:
        chosen = consider(chosen, "symbol_bias", mapped, eval_mapped, y_tune, {"biases": biases, "tune_accuracy": score})
    return finish(model_id, baseline_eval, y_eval, chosen, y_tune, base_tune)


def optimize_snxx() -> dict:
    parts = {split: load_split("SNXX_3d3pct", split) for split in ("fit", "tune", "eval")}
    y_tune = parts["tune"]["y"].to_numpy()
    base_p = parts["tune"]["p"].to_numpy()
    base_tune = accuracy(y_tune, base_p)
    chosen = {"name": "baseline", "tune_p": base_p, "eval_p": parts["eval"]["p"].to_numpy(), "tune_accuracy": base_tune, "detail": {}}
    scale, shift, mapped, _ = best_platt(y_tune, base_p)
    chosen = consider(chosen, "platt", mapped, expit(scale * logit(parts["eval"]["p"].to_numpy()) + shift), y_tune, {"a": scale, "b": shift})
    features = [c for c in parts["fit"].columns if c not in {"session_date", "entry", "y", "label_end", "p"}]
    import lightgbm as lgb
    best_weight, best_model, best_score = None, None, base_tune
    for weight in (1, 2, 4, 8):
        model = lgb.LGBMClassifier(n_estimators=40, num_leaves=7, max_depth=3, min_child_samples=4,
                                   learning_rate=0.05, reg_lambda=10, scale_pos_weight=weight,
                                   n_jobs=1, verbosity=-1, random_state=3566)
        model.fit(parts["fit"][features], parts["fit"]["y"].to_numpy())
        pred = model.predict_proba(parts["tune"][features])[:, 1]
        score = accuracy(y_tune, pred)
        if score > best_score + 1e-12:
            best_weight, best_model, best_score = weight, model, score
    if best_model is not None:
        chosen = consider(chosen, "refit", best_model.predict_proba(parts["tune"][features])[:, 1],
                          best_model.predict_proba(parts["eval"][features])[:, 1], y_tune, {"scale_pos_weight": best_weight})
    return finish("SNXX_3d3pct", BASELINE_EVAL["SNXX_3d3pct"], parts["eval"]["y"].to_numpy(), chosen, y_tune, base_tune)


def soxs_action(p_soxs: np.ndarray, p_soxx: np.ndarray) -> np.ndarray:
    out = np.full(len(p_soxs), "flat", dtype=object)
    buy = (p_soxs >= 0.5) | (p_soxx >= 0.5)
    out[buy & (p_soxs >= p_soxx)] = "SOXS"
    out[buy & (p_soxx > p_soxs)] = "SOXX"
    return out


def optimize_soxs() -> dict:
    tune, ev = load_split("SOXS_SOXX_3d5pct", "tune"), load_split("SOXS_SOXX_3d5pct", "eval")
    base = float(np.mean(soxs_action(tune["p_SOXS"].to_numpy(), tune["p_SOXX"].to_numpy()) == tune["action"].to_numpy()))
    best = {"b_soxs": 0.0, "b_soxx": 0.0, "tune": base}
    base_s, base_q = logit(tune["p_SOXS"].to_numpy()), logit(tune["p_SOXX"].to_numpy())
    for b_soxs in np.round(np.arange(-2, 2.01, 0.5), 2):
        for b_soxx in np.round(np.arange(-2, 2.01, 0.5), 2):
            pred = soxs_action(expit(base_s + b_soxs), expit(base_q + b_soxx))
            score = float(np.mean(pred == tune["action"].to_numpy()))
            if score > best["tune"] + 1e-12:
                best = {"b_soxs": float(b_soxs), "b_soxx": float(b_soxx), "tune": score}
    adopted = best["b_soxs"] != 0 or best["b_soxx"] != 0
    eval_p_s = expit(logit(ev["p_SOXS"].to_numpy()) + best["b_soxs"])
    eval_p_q = expit(logit(ev["p_SOXX"].to_numpy()) + best["b_soxx"])
    eval_acc = float(np.mean(soxs_action(eval_p_s, eval_p_q) == ev["action"].to_numpy()))
    official = eval_acc if adopted else BASELINE_EVAL["SOXS_SOXX_3d5pct"]
    lift = official / BASELINE_EVAL["SOXS_SOXX_3d5pct"] - 1
    return {
        "id": "SOXS_SOXX_3d5pct",
        "baseline_eval": BASELINE_EVAL["SOXS_SOXX_3d5pct"],
        "baseline_tune": base,
        "chosen": "bias" if adopted else "baseline",
        "detail": best if adopted else {},
        "tune_accuracy": best["tune"],
        "eval_accuracy": official,
        "candidate_eval_accuracy": eval_acc,
        "relative_lift": lift,
        "passed": bool(lift >= 0.10 - 1e-12),
        "adopted": adopted,
    }


def main() -> None:
    names = groups()
    rows = []
    for model_id in ("B_group", "B_no_group", "C_group", "C_no_group"):
        rows.append(optimize_probability_model(model_id, union_symbols(), None, BASELINE_EVAL[model_id]))
        print(json.dumps({"id": model_id, "lift": rows[-1]["relative_lift"], "passed": rows[-1]["passed"]}), flush=True)
    partial = json.loads((RUN / "ACCURACY_partial.json").read_text())
    group_base = {item["id"]: item["eval"]["accuracy_at_0.5"] for item in partial["models"] if item["id"] != "SOXS_SOXX_3d5pct"}
    for parent in ("B_group", "B_no_group", "C_group", "C_no_group", "C_no_daily"):
        for group, symbols in names.items():
            model_id = f"{parent}_{group}"
            rows.append(optimize_probability_model(model_id, symbols, parent, group_base[model_id]))
            print(json.dumps({"id": model_id, "lift": rows[-1]["relative_lift"], "passed": rows[-1]["passed"]}), flush=True)
    rows.append(optimize_snxx())
    rows.append(optimize_soxs())
    for item in rows[-2:]:
        print(json.dumps({"id": item["id"], "lift": item["relative_lift"], "passed": item["passed"]}), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    payload = {"round": 4, "frozen": "C_no_daily", "models": rows,
               "passed": [item["id"] for item in rows if item["passed"]]}
    (OUT / "result.json").write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
