"""Read-only paired comparison of independently implemented v7 research lanes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

TARGETS = [(h, t) for h in (1, 3, 5) for t in (3, 5, 8)]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize(path: Path, reference: pd.DataFrame | None = None) -> pd.DataFrame:
    source = pd.read_parquet(path)
    if source.sample_id.duplicated().any():
        raise ValueError(f"Duplicate predictions: {path}")
    out = source[["sample_id", "symbol", "session_date"]].copy()
    for index, (h, t) in enumerate(TARGETS):
        i, j = divmod(index, 3)
        for kind in ("p", "y"):
            possibilities = [f"{kind}_{i}_{j}", f"{kind}_{h}d_{t}pct"]
            key = next((c for c in possibilities if c in source), None)
            if key is None:
                if kind != "y" or reference is None:
                    raise ValueError(f"Missing {kind} {h}d {t}pct: {path}")
                labels = reference.set_index("sample_id")[f"y_{i}_{j}"]
                out[f"y_{i}_{j}"] = source.sample_id.map(labels)
                if out[f"y_{i}_{j}"].isna().any():
                    raise ValueError(f"Blind predictions contain unknown sample IDs: {path}")
            else:
                out[f"{kind}_{i}_{j}"] = source[key].to_numpy()
    return out.sort_values("sample_id").reset_index(drop=True)


def summarize(y: np.ndarray, p: np.ndarray) -> dict:
    p = np.asarray(p, float)
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Nonfinite or out-of-range prediction")
    bins = []
    for index in range(10):
        lo, hi = index / 10, (index + 1) / 10
        mask = (p >= lo) & ((p < hi) if index < 9 else (p <= hi))
        bins.append({"lower": lo, "upper": hi, "n": int(mask.sum()),
                     "hits": int(y[mask].sum()),
                     "predicted": float(p[mask].mean()) if mask.any() else None,
                     "observed": float(y[mask].mean()) if mask.any() else None})
    return {"n": len(y), "hits": int(y.sum()), "observed": float(y.mean()),
            "predicted": float(p.mean()), "brier": float(np.mean((p-y)**2)),
            "logloss": float(log_loss(y, np.clip(p, 1e-6, 1-1e-6), labels=[0, 1])),
            "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else None,
            "ece10": float(sum(b["n"]*abs(b["predicted"]-b["observed"])
                               for b in bins if b["n"]) / len(y)), "reliability": bins}


def paired_interval(frame: pd.DataFrame, baseline: pd.DataFrame,
                    block: int, repeats: int = 1000) -> dict:
    """Synchronous contiguous date blocks; all stocks of each date stay together."""
    loss = ((frame.p_1_1-frame.y_1_1)**2 - (baseline.p_1_1-baseline.y_1_1)**2)
    dates = sorted(frame.session_date.unique())
    date_sums = np.array([loss[frame.session_date == d].sum() for d in dates])
    date_n = np.array([(frame.session_date == d).sum() for d in dates])
    rng = np.random.default_rng(7391 + block)
    scores = []
    for _ in range(repeats):
        chosen = []
        while len(chosen) < len(dates):
            # Circular blocks give every date equal marginal sampling weight.
            start = int(rng.integers(len(dates)))
            chosen.extend((start+np.arange(block)) % len(dates))
        idx = np.asarray(chosen[:len(dates)])
        scores.append(float(date_sums[idx].sum()/date_n[idx].sum()))
    return {"delta_brier": float(loss.mean()), "block_dates": block,
            "repeats": repeats, "exploratory_95pct": np.quantile(scores, [.025, .975]).tolist(),
            "scope": "selected_exposed_development_not_independent_significance"}


def collect(mapping_path: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    mapping = json.loads(mapping_path.read_text())
    records = mapping["records"]
    reference = normalize(Path(records[0]["predictions"]))
    frames = {(r["lane"], r["round"]): normalize(Path(r["predictions"]), reference) for r in records}
    if len(frames) != len(records):
        raise ValueError("Duplicate lane/round keys")
    reference = next(iter(frames.values()))
    ycols = [f"y_{i}_{j}" for i in range(3) for j in range(3)]
    for key, frame in frames.items():
        if not frame[["sample_id", "symbol", "session_date", *ycols]].equals(
                reference[["sample_id", "symbol", "session_date", *ycols]]):
            # Parquet serializers may use float labels vs integers; compare values.
            if (not frame[["sample_id", "symbol", "session_date"]].equals(
                    reference[["sample_id", "symbol", "session_date"]]) or
                    not np.array_equal(frame[ycols].to_numpy(), reference[ycols].to_numpy())):
                raise ValueError(f"Sample or label mismatch: {key}")
    report = {"scope": "v7_three_iterations_exposed_development",
              "mapping_sha256": digest(mapping_path), "collector_sha256": digest(Path(__file__)),
              "rows": len(reference), "dates": int(reference.session_date.nunique()),
              "symbols": int(reference.symbol.nunique()), "records": []}
    for item in records:
        lane, round_name = item["lane"], item["round"]
        frame = frames[(lane, round_name)]
        baseline = frames[(lane, "R0")]
        result = dict(item)
        result["predictions_sha256"] = digest(Path(item["predictions"]))
        result["targets"] = {}
        for index, (h, t) in enumerate(TARGETS):
            i, j = divmod(index, 3)
            result["targets"][f"{h}d_{t}pct"] = summarize(
                frame[f"y_{i}_{j}"].to_numpy(float), frame[f"p_{i}_{j}"].to_numpy(float))
        probability = frame[[f"p_{i}_{j}" for i in range(3) for j in range(3)]].to_numpy().reshape(-1, 3, 3)
        result["monotonic_violation_fraction"] = float(np.mean(
            (np.diff(probability, axis=1) < -1e-7).any((1, 2)) |
            (np.diff(probability, axis=2) > 1e-7).any((1, 2))))
        result["paired_primary_vs_R0"] = [paired_interval(frame, baseline, b) for b in (5, 10)]
        report["records"].append(result)
    output.mkdir(parents=True)
    (output/"comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    lines = ["# 五条路线三轮开发期对照", "", f"相同 {report['rows']} 行、{report['dates']} 个日期、{report['symbols']} 只股票。所有日期已暴露。", "",
             "| 路线 | R0 Brier | R1 Brier | R2 Brier | R3 Brier | R3−R0 |", "|---|---:|---:|---:|---:|---:|"]
    lanes = list(dict.fromkeys(r["lane"] for r in records))
    by_key = {(r["lane"], r["round"]): r for r in report["records"]}
    for lane in lanes:
        values = [by_key[(lane, r)]["targets"]["3d_5pct"]["brier"] for r in ("R0", "R1", "R2", "R3")]
        lines.append("| " + lane + " | " + " | ".join(f"{v:.6f}" for v in values) + f" | {values[-1]-values[0]:+.6f} |")
    lines += ["", "模型列为逐样本条件概率估计的平均值，不能视为筛选后实际胜率。", "",
              "| 目标 | 同期实际率 | " + " | ".join(lanes) + " |",
              "|---|---:|" + "---:|"*len(lanes)]
    for h, t in TARGETS:
        key = f"{h}d_{t}pct"
        actual = by_key[(lanes[0], "R3")]["targets"][key]["observed"]
        values = [by_key[(lane, "R3")]["targets"][key]["predicted"] for lane in lanes]
        lines.append(f"| {h}日+{t}% | {actual:.1%} | " + " | ".join(f"{v:.1%}" for v in values) + " |")
    lines += ["", "所有R0/R1/R2/R3均列出，不按外层选最好轮次；种子、校准、变换及轮次定义以各路线登记为准。主差额5/10日同步块区间、九目标可靠性分桶及输入哈希见comparison.json。"]
    (output/"TABLES.md").write_text("\n".join(lines)+"\n")
    return {"records": len(records), "rows": report["rows"], "output": str(output)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(collect(args.mapping, args.output)))
