"""Round 03: Group bias coordinate descent / SNXX tree shrinkage."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.special import expit, logit

cur = Path(__file__).resolve()
while cur.parent != cur and not (cur / "research" / "after_open_3d5pct").is_dir():
    cur = cur.parent
REPO_ROOT = cur
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.after_open_3d5pct.focus_v8.core import GROUPS
from research.after_open_3d5pct.models.upgrades.round_03.snxx_train import train_round3

AFTER = REPO_ROOT / "research" / "after_open_3d5pct"
RUNS_BASE = AFTER / "runs" / "model_registry_v1"
R1_DIR = AFTER / "runs" / "model_registry_v1_upgrades" / "round_01"
R2_DIR = AFTER / "runs" / "model_registry_v1_upgrades" / "round_02"
OUTPUT_DIR = AFTER / "runs" / "model_registry_v1_upgrades" / "round_03"

MODELS_5 = ["B_group", "B_no_group", "C_group", "C_no_group", "C_no_daily"]
ALL_MODELS = MODELS_5 + ["SNXX_3d3pct"]

GRID = np.round(np.arange(-2.0, 2.05, 0.1), 1)


def get_union_symbols() -> list[str]:
    return sorted({s for symbols in GROUPS.values() for s in symbols})


def get_intersecting_symbols() -> set[str]:
    chips = set(GROUPS["chips"])
    optics = set(GROUPS["optics"])
    storage = set(GROUPS["storage"])
    union = get_union_symbols()
    return {s for s in union if ((s in chips) + (s in optics) + (s in storage)) > 1}


def calc_acc(p: np.ndarray, y: np.ndarray) -> float:
    return float(((p >= 0.5) == (y == 1)).mean())


def calc_symbol_bias(biases: dict[str, float], sym_arr: np.ndarray, inter_stocks: set[str]) -> np.ndarray:
    b_c = biases["chips"]
    b_o = biases["optics"]
    b_s = biases["storage"]
    b_inter = (b_c + b_o + b_s) / 3.0

    chips = set(GROUPS["chips"])
    optics = set(GROUPS["optics"])
    storage = set(GROUPS["storage"])

    arr = np.zeros(len(sym_arr), dtype=float)
    for i, s in enumerate(sym_arr):
        if s in inter_stocks:
            arr[i] = b_inter
        elif s in chips:
            arr[i] = b_c
        elif s in optics:
            arr[i] = b_o
        elif s in storage:
            arr[i] = b_s
    return arr


def run() -> dict:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    union_symbols = get_union_symbols()
    inter_stocks = get_intersecting_symbols()

    # Load round 1 and round 2 results
    r1_results = json.loads((R1_DIR / "result.json").read_text())
    r2_results = json.loads((R2_DIR / "result.json").read_text())

    results = {}

    for model_id in ALL_MODELS:
        r2_mod = r2_results[model_id]
        if r2_mod.get("passed", False):
            # Already passed in earlier rounds, keep frozen
            stopped_round = r2_mod.get("stopped_round", 2)
            results[model_id] = {
                "model_id": model_id,
                "status": f"already_passed_round_0{stopped_round}",
                "stopped_round": stopped_round,
                "passed": True,
                "retained": True,
                "tune_baseline_accuracy": r2_mod["tune_baseline_accuracy"],
                "tune_accuracy": r2_mod["tune_accuracy"],
                "eval_baseline_accuracy": r2_mod["eval_baseline_accuracy"],
                "eval_accuracy": r2_mod["eval_accuracy"],
                "relative_improvement": r2_mod["relative_improvement"],
            }
            continue

        model_out = OUTPUT_DIR / model_id
        model_out.mkdir(parents=True, exist_ok=True)

        if model_id in MODELS_5:
            # Load round 2 retained parquets
            r2_tune = pd.read_parquet(R2_DIR / model_id / "tune.parquet")
            r2_cal = pd.read_parquet(R2_DIR / model_id / "cal.parquet")
            r2_eval = pd.read_parquet(R2_DIR / model_id / "eval.parquet")

            u_tune = r2_tune[r2_tune.symbol.isin(union_symbols)]
            u_cal = r2_cal[r2_cal.symbol.isin(union_symbols)]
            u_eval = r2_eval[r2_eval.symbol.isin(union_symbols)]

            y_t = u_tune["y_3d_5pct"].to_numpy().astype(int)
            p2_t = u_tune["p_3d_5pct"].to_numpy().astype(float)
            sym_t = u_tune["symbol"].to_numpy()

            y_c = u_cal["y_3d_5pct"].to_numpy().astype(int)
            p2_c = u_cal["p_3d_5pct"].to_numpy().astype(float)
            sym_c = u_cal["symbol"].to_numpy()

            y_e = u_eval["y_3d_5pct"].to_numpy().astype(int)
            p2_e = u_eval["p_3d_5pct"].to_numpy().astype(float)
            sym_e = u_eval["symbol"].to_numpy()

            base_t_acc = r1_results[model_id]["tune_baseline_accuracy"]
            base_e_acc = r1_results[model_id]["eval_baseline_accuracy"]

            l_t = logit(np.clip(p2_t, 1e-4, 1.0 - 1e-4))
            l_c = logit(np.clip(p2_c, 1e-4, 1.0 - 1e-4))
            l_e = logit(np.clip(p2_e, 1e-4, 1.0 - 1e-4))

            biases = {"chips": 0.0, "optics": 0.0, "storage": 0.0}
            best_acc = calc_acc(p2_t, y_t)

            order = ["chips", "optics", "storage"]
            for cycle in range(2):
                for grp in order:
                    best_val = biases[grp]
                    best_score = best_acc
                    for val in sorted(GRID, key=lambda x: (abs(x), x)):
                        test_biases = dict(biases)
                        test_biases[grp] = float(val)
                        b_arr = calc_symbol_bias(test_biases, sym_t, inter_stocks)
                        p3_t = expit(l_t + b_arr)
                        acc = calc_acc(p3_t, y_t)
                        if acc > best_score:
                            best_score = acc
                            best_val = float(val)
                        elif acc == best_score:
                            if abs(val) < abs(best_val):
                                best_val = float(val)
                            elif abs(val) == abs(best_val) and best_val != 0 and val == 0:
                                best_val = 0.0
                    biases[grp] = best_val
                    best_acc = best_score

            # Retention condition: tune accuracy must be strictly greater than baseline tune accuracy
            retained = bool(best_acc > base_t_acc)

            # Evaluate on cal and eval
            b_arr_c = calc_symbol_bias(biases, sym_c, inter_stocks)
            p3_c = expit(l_c + b_arr_c)
            cal_acc = calc_acc(p3_c, y_c)

            b_arr_e = calc_symbol_bias(biases, sym_e, inter_stocks)
            p3_e = expit(l_e + b_arr_e)
            eval_acc = calc_acc(p3_e, y_e)

            rel_improvement = float(eval_acc / base_e_acc - 1.0)
            passed = bool(rel_improvement >= 0.10)

            # Save parquets: if retained update with p3, else rollback to round 2
            for sname, sdf in [("tune", r2_tune), ("cal", r2_cal), ("eval", r2_eval)]:
                out_df = sdf.copy()
                if retained:
                    sym_all = out_df["symbol"].to_numpy()
                    b_all = calc_symbol_bias(biases, sym_all, inter_stocks)
                    p_all = out_df["p_3d_5pct"].to_numpy().astype(float)
                    l_all = logit(np.clip(p_all, 1e-4, 1.0 - 1e-4))
                    out_df["p_3d_5pct"] = expit(l_all + b_all)
                out_df.to_parquet(model_out / f"{sname}.parquet", index=False)

            results[model_id] = {
                "model_id": model_id,
                "biases": biases,
                "tune_baseline_accuracy": base_t_acc,
                "tune_accuracy": best_acc,
                "cal_accuracy": cal_acc,
                "eval_baseline_accuracy": base_e_acc,
                "eval_accuracy": eval_acc,
                "relative_improvement": rel_improvement,
                "retained": retained,
                "passed": passed,
            }
        else:
            # SNXX_3d3pct retraining with tree shrinkage
            snxx_payload = train_round3(model_out)
            base_t_acc = r1_results[model_id]["tune_baseline_accuracy"]
            base_e_acc = r1_results[model_id]["eval_baseline_accuracy"]

            new_t_acc = snxx_payload["splits"]["tune"]["accuracy_at_0.5"]
            new_e_acc = snxx_payload["splits"]["eval"]["accuracy_at_0.5"]
            new_c_acc = snxx_payload["splits"]["cal"]["accuracy_at_0.5"]

            retained = bool(new_t_acc > base_t_acc)
            rel_improvement = float(new_e_acc / base_e_acc - 1.0)
            passed = bool(rel_improvement >= 0.10)

            # If not retained, rollback parquets to round 2 (or baseline)
            if not retained:
                for sname in ["tune", "cal", "eval"]:
                    prev_df = pd.read_parquet(R2_DIR / model_id / f"{sname}.parquet")
                    prev_df.to_parquet(model_out / f"{sname}.parquet", index=False)

            results[model_id] = {
                "model_id": model_id,
                "scale_pos_weight": snxx_payload["selected_scale_pos_weight"],
                "tune_baseline_accuracy": base_t_acc,
                "tune_accuracy": new_t_acc,
                "cal_accuracy": new_c_acc,
                "eval_baseline_accuracy": base_e_acc,
                "eval_accuracy": new_e_acc,
                "relative_improvement": rel_improvement,
                "retained": retained,
                "passed": passed,
            }

    out_file = OUTPUT_DIR / "result.json"
    out_file.write_text(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    res = run()
    print(json.dumps(res, indent=2))
