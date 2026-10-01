"""Morning 5-minute path. Thresholds are frozen on tune and checked on cal."""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from research.after_open_3d5pct.focus_v8.core import GROUPS, PROTOCOL, splits, focus_weights
from research.after_open_3d5pct.focus_v8.run import load_data

ROOT = Path(__file__).resolve().parents[5]
AFTER = ROOT / "research" / "after_open_3d5pct"
RUN = AFTER / "runs" / "model_registry_v1"
OUT = AFTER / "runs" / "model_registry_v1_upgrades" / "overnight"
PRIMARY = 4
LEVELS = np.round(np.arange(0.30, 0.81, 0.05), 2)


def morning_shape(x5: np.ndarray) -> np.ndarray:
    seq = x5[:, 7, 66:90, :]
    valid = seq[..., 10] > 0
    ret = np.where(valid, seq[..., 0], 0.0).astype(np.float64)
    count = np.maximum(valid.sum(1), 1)
    total = ret.sum(1)
    mean = total / count
    std = np.sqrt((((ret - mean[:, None]) ** 2) * valid).sum(1) / count)
    thirds = [ret[:, a:b].sum(1) for a, b in ((0, 8), (8, 16), (16, 24))]
    cumulative = np.cumsum(ret, axis=1)
    drawdown = (np.maximum.accumulate(cumulative, axis=1) - cumulative).max(1)
    up = ((ret > 0) & valid).sum(1) / count
    return np.column_stack([total, std, *thirds, drawdown, up, ret[:, :6].sum(1), ret[:, -6:].sum(1), valid.mean(1)]).astype(np.float32)


def peer_mean(rows: pd.DataFrame, value: np.ndarray) -> np.ndarray:
    peers = sorted({symbol for symbols in GROUPS.values() for symbol in symbols})
    out = np.full(len(rows), np.nan, np.float32)
    dates = rows.session_date.to_numpy()
    symbols = rows.symbol.to_numpy()
    for date in pd.unique(dates):
        member = np.flatnonzero((dates == date) & np.isin(symbols, peers) & np.isfinite(value))
        if len(member) < 2:
            continue
        total = float(value[member].sum())
        for index in member:
            out[index] = (total - float(value[index])) / (len(member) - 1)
    return out


def choose_threshold(y: np.ndarray, p: np.ndarray, minimum: int) -> float | None:
    best = None
    for level in LEVELS:
        picked = p >= level
        count = int(picked.sum())
        if count < minimum:
            continue
        precision = float(y[picked].mean()) if count else 0.0
        rank = (precision, float(level))
        if best is None or rank > best[0]:
            best = (rank, float(level))
    return None if best is None else best[1]


def precision_of(y, p, level) -> tuple[int, float | None]:
    picked = np.asarray(p) >= level
    count = int(picked.sum())
    if count == 0:
        return 0, None
    return count, float(np.asarray(y)[picked].mean())


def fit_predict(kind: str, x, y, fold, rows):
    weight = focus_weights(rows.iloc[fold["fit"]])
    if weight.sum() <= 0:
        weight = np.ones(len(fold["fit"]))
    if kind == "tree":
        model = lgb.LGBMClassifier(n_estimators=80, num_leaves=7, max_depth=3, min_child_samples=8,
                                   learning_rate=0.05, reg_lambda=10, n_jobs=1, verbosity=-1, random_state=3566)
        model.fit(x[fold["fit"]], y[fold["fit"]], sample_weight=weight,
                  eval_set=[(x[fold["tune"]], y[fold["tune"]])],
                  callbacks=[lgb.early_stopping(10, verbose=False)])
        return {name: model.predict_proba(x[ids])[:, 1] for name, ids in fold.items()}
    model = LogisticRegression(max_iter=400, C=0.5)
    model.fit(np.nan_to_num(x[fold["fit"]]), y[fold["fit"]], sample_weight=weight)
    return {name: model.predict_proba(np.nan_to_num(x[ids]))[:, 1] for name, ids in fold.items()}


def evaluate_candidate(x, y, rows, fold, kind: str, minimum: int, book: np.ndarray) -> dict | None:
    pred = fit_predict(kind, x, y, fold, rows)
    tune_ids = fold["tune"][book[fold["tune"]]]
    cal_ids = fold["cal"][book[fold["cal"]]]
    eval_ids = fold["eval"][book[fold["eval"]]]
    level = choose_threshold(y[tune_ids], pred["tune"][book[fold["tune"]]], minimum)
    if level is None:
        return None
    tune_n, tune_p = precision_of(y[tune_ids], pred["tune"][book[fold["tune"]]], level)
    cal_n, cal_p = precision_of(y[cal_ids], pred["cal"][book[fold["cal"]]], level)
    cal_floor = max(5, minimum // 2)
    if cal_n < cal_floor or cal_p is None or cal_p + 1e-12 < tune_p - 0.10:
        return None
    eval_n, eval_p = precision_of(y[eval_ids], pred["eval"][book[fold["eval"]]], level)
    acc = float(np.mean((pred["eval"][book[fold["eval"]]] >= 0.5) == y[eval_ids].astype(bool)))
    base = float(y[eval_ids].mean())
    majority = max(base, 1 - base)
    return {"learner": kind, "threshold": level, "tune_precision": tune_p, "tune_signals": tune_n,
            "cal_precision": cal_p, "cal_signals": cal_n, "eval_precision": eval_p, "eval_signals": eval_n,
            "eval_accuracy_at_0.5": acc, "eval_majority": majority,
            "tradable": bool(eval_n >= cal_floor and eval_p is not None and eval_p >= 0.90 and acc >= majority + 0.02)}


def run_slice(model_id: str, x_all, y_all, rows, train_symbols, score_symbols, with_peer: bool, peer: np.ndarray) -> dict:
    mask = np.ones(len(rows), bool) if train_symbols is None else rows.symbol.isin(train_symbols).to_numpy()
    sub = rows.loc[mask].reset_index(drop=True)
    fold = splits(sub, next(item for item in PROTOCOL["folds"] if item["id"] == "reserved"))
    base = x_all[mask]
    columns = [base]
    if with_peer:
        columns.append(peer[mask, None])
    x = np.column_stack(columns).astype(np.float32)
    y = y_all[mask, PRIMARY]
    minimum = 30 if train_symbols is None else 12
    book = np.ones(len(sub), bool) if score_symbols is None else sub.symbol.isin(score_symbols).to_numpy()
    confirmed = []
    for kind in ("tree", "logistic"):
        result = evaluate_candidate(x, y, sub, fold, kind, minimum, book)
        if result is not None:
            confirmed.append(result)
    if not confirmed:
        return {"id": model_id, "confirmed": False, "eval_precision": None, "eval_accuracy_at_0.5": None}
    best = max(confirmed, key=lambda item: (item["cal_precision"], item["threshold"]))
    best["id"] = model_id
    best["confirmed"] = True
    return best


def main() -> None:
    rows, data, y, prior, counts, paths, xbase, curve, relative, arrays, identity = load_data(AFTER / "runs/focus_v8_20260927")
    shape = morning_shape(data["x5"])
    peer = peer_mean(rows, shape[:, 0])
    union = sorted({symbol for symbols in GROUPS.values() for symbol in symbols})
    jobs = []
    for name, peer_ok in (("B_group", True), ("B_no_group", False), ("C_group", True), ("C_no_group", False), ("C_no_daily", True)):
        jobs.append((name, None, union, peer_ok))
    for group, symbols in GROUPS.items():
        for prefix, peer_ok in (("B_group", True), ("B_no_group", False), ("C_group", True), ("C_no_group", False), ("C_no_daily", True)):
            jobs.append((f"{prefix}_{group}", symbols, symbols, peer_ok))
    reports = []
    for model_id, train_symbols, score_symbols, peer_ok in jobs:
        report = run_slice(model_id, shape, y, rows, train_symbols, score_symbols, peer_ok, peer)
        reports.append(report)
        print(json.dumps({"id": model_id, "confirmed": report["confirmed"], "eval_precision": report.get("eval_precision")}), flush=True)
    confirmed = [item for item in reports if item["confirmed"] and item.get("eval_precision") is not None]
    best = max(confirmed, key=lambda item: (item["cal_precision"], item["eval_signals"])) if confirmed else None
    OUT.mkdir(parents=True, exist_ok=True)
    payload = {"round": "morning_5m_path", "selection": "highest cal precision among cal-confirmed models",
               "best": best, "models": reports}
    (OUT / "morning_path.json").write_text(json.dumps(payload, indent=2))
    lines = ["# 早上 5 分钟路径", "", "模型按 cal 精准率选择，开发段只在选定后打分。", ""]
    if best is None:
        lines.append("没有模型通过 cal 确认。")
    else:
        lines.append(f"当前最可交易的候选是 `{best['id']}`，学习器 {best['learner']}，门槛 {best['threshold']}。")
        lines.append(f"cal 精准率 {best['cal_precision']:.1%}（{best['cal_signals']} 笔），开发段精准率 {best['eval_precision']:.1%}（{best['eval_signals']} 笔），0.5 准确度 {best['eval_accuracy_at_0.5']:.1%}，多数类 {best['eval_majority']:.1%}。")
        lines.append("开发段已经暴露，不能当作实盘许可。")
    lines.append("")
    lines.append("| 模型 | 确认 | 门槛 | cal 精准率 | 开发段精准率 | 开发段笔数 | 0.5 准确度 |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for item in reports:
        if not item["confirmed"]:
            lines.append(f"| `{item['id']}` | 否 |  |  |  |  |  |")
            continue
        prec = "" if item["eval_precision"] is None else f"{item['eval_precision']:.1%}"
        lines.append(f"| `{item['id']}` | 是 | {item['threshold']:.2f} | {item['cal_precision']:.1%} | {prec} | {item['eval_signals']} | {item['eval_accuracy_at_0.5']:.1%} |")
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
