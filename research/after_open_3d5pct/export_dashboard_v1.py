"""Export a static research dashboard from an immutable exploratory run.

No training, market download, broker access, or overwrite of source artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from .policy_replay_v1 import POLICY_VERSION, PolicyConfig, instant, replay, summarize


REPO = Path(__file__).resolve().parents[2]
PROJECT = Path(__file__).resolve().parent
UTC = ZoneInfo("UTC")
BAR = timedelta(minutes=5)
SCHEMA = "research_dashboard_v1"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def object_digest(payload: object) -> str:
    body = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def write_json(path: Path, payload: object) -> str:
    def clean(value):
        if isinstance(value, dict):
            return {str(k): clean(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(v) for v in value]
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(payload), ensure_ascii=False, separators=(",", ":"),
                               allow_nan=False) + "\n", encoding="utf-8")
    return digest(path)


def regular_calendar(path: Path) -> list[datetime]:
    data = json.loads(path.read_text(encoding="utf-8"))
    starts = []
    for session in data["sessions"]:
        at, end = instant(session["open_at"]), instant(session["close_at"])
        if (end - at).total_seconds() % 300:
            raise ValueError(f"calendar has a partial 5m session: {session['session_date']}")
        while at < end:
            starts.append(at)
            at += BAR
    if starts != sorted(set(starts)):
        raise ValueError("invalid calendar order or duplicates")
    return starts


def price_bars(path: Path) -> tuple[dict[datetime, dict], list[dict]]:
    columns = ["start_at", "end_at", "session_type", "price_basis", "open", "high", "low", "close"]
    frame = pd.read_parquet(path, columns=columns)
    frame = frame[frame["session_type"] == "regular"]
    bars: dict[datetime, dict] = {}
    compact = []
    for row in frame.itertuples(index=False):
        at = instant(row.start_at)
        if at in bars:
            raise ValueError(f"duplicate bar in {path}: {at}")
        bar = {"session_type": row.session_type, "price_basis": row.price_basis,
               "open": row.open, "high": row.high, "low": row.low, "close": row.close}
        bars[at] = bar
        if at >= datetime(2026, 8, 1, tzinfo=UTC):
            compact.append({"t": at.isoformat(), "o": _native(row.open),
                            "h": _native(row.high), "l": _native(row.low),
                            "c": _native(row.close)})
    return bars, compact


def _native(value):
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def model_spec(model: str, group: str, config: dict) -> dict:
    modules = {"H": "已完成的十个常规交易日，含个股与市场/行业背景",
               "A": "上一交易日盘后与当日盘前；本次夜盘没有观测",
               "B": "当日开盘至决策时点的 5 分钟路径"}
    if model == "lgbm":
        structure = "LightGBM 决策树集成；读取各时间段的收益、波幅、成交与相对尺度摘要。"
        hypothesis = "检验多尺度摘要和 A/B 组合是否增加前向时间折的概率质量。"
    elif model == "tcn":
        structure = "小型因果时间卷积：H、盘后、盘前、B 分支编码后拼接，以小型 MLP 输出。"
        hypothesis = "检验有序价格与成交路径是否提供摘要以外的信息。"
    else:
        structure = ("训练期成熟标签的常数率基准。" if model == "B0" else
                     "训练期按决策时点平滑的基准率。")
        hypothesis = "为模型增量提供同折、同分母比较。"
    if model in ("B0", "B1_hour"):
        signals = (["仅训练期成熟标签的整体发生率"] if model == "B0" else
                   ["训练期成熟标签的决策时点分组发生率，向整体发生率平滑"])
    else:
        signals = ["H：1/3/5/10 日收益、实际/自身历史相对波幅与成交、QQQ/行业背景"]
        if "A" in group:
            signals.append("A：盘后与盘前的收益、振幅、成交及相对市场变化；夜盘无覆盖")
        if "B" in group:
            signals.append("B：开盘后收益/振幅/最近一小时、成交及相对 QQQ/行业变化")
        if model == "tcn":
            signals.append("序列通道：累计收益、单根收益/振幅、成交规模、QQQ/行业收益、有效性 mask")
    return {"id": f"{model}_{group}", "family": model, "input_set": group,
            "modules": [modules[k] for k in "HAB" if k in group],
            "structure": structure, "design_hypothesis": hypothesis,
            "key_signals": signals,
            "evidence_scope": "2026 年 8/9 月已暴露开发期外层时间折；无独立验证",
            "probability_kind": "raw_model_output" if model not in ("B0", "B1_hour") else "training_rate_baseline",
            "feature_schema": config["feature_schema"], "label_contract": config["label_contract"]}


def export(source: Path, output: Path, threshold: float,
           fee_bps_per_side: float = 1.0,
           slippage_bps_per_side: float = 5.0) -> dict:
    if output.exists():
        raise FileExistsError(f"output must not exist: {output}")
    source = source.resolve()
    output = output.resolve()
    config = json.loads((source / "config.json").read_text())
    manifest = json.loads((source / "manifest.json").read_text())
    if config["label_contract"] != "after_open_3d5pct_v1":
        raise ValueError("unsupported label contract")
    if manifest["scope"] != "development_exploratory_retrospective_universe":
        raise ValueError("unrecognized evidence scope")
    policy = PolicyConfig(threshold=threshold,
                          fee_bps_per_side=fee_bps_per_side,
                          slippage_bps_per_side=slippage_bps_per_side)
    source_files = {name: digest(source / name) for name in (
        "config.json", "manifest.json", "features.parquet", "outcomes.parquet",
        "predictions.parquet", "metrics.csv", "trials.jsonl")}
    features = pd.read_parquet(source / "features.parquet", columns=[
        "sample_id", "symbol", "session_date", "cutoff_et", "cutoff_at", "decision_at"])
    outcomes = pd.read_parquet(source / "outcomes.parquet")
    preds = pd.read_parquet(source / "predictions.parquet")
    if features.sample_id.duplicated().any() or outcomes.sample_id.duplicated().any():
        raise ValueError("duplicate sample IDs in source")
    if preds.duplicated(["sample_id", "model", "input_set"]).any():
        raise ValueError("duplicate model prediction")
    joined = preds.merge(features, on="sample_id", how="left", validate="many_to_one")
    joined = joined.merge(outcomes.rename(columns={"target": "outcome_target"}),
                          on="sample_id", how="left", validate="many_to_one")
    if joined[["symbol", "cutoff_at", "entry_at", "label_end_at"]].isna().any().any():
        raise ValueError("prediction without complete feature/outcome lineage")
    if (joined["target"] != joined["outcome_target"]).any():
        raise ValueError("prediction target disagrees with frozen outcome")
    for fold in config["folds"]:
        subset = joined[joined.fold == fold["name"]]
        if not subset.session_date.between(fold["outer_start"],
                                            fold["outer_end_exclusive"], inclusive="left").all():
            raise ValueError(f"training/inner row found in outer predictions: {fold['name']}")
    if not joined.fold.isin([f["name"] for f in config["folds"]]).all():
        raise ValueError("prediction has unknown fold")
    as_of = datetime.now(UTC)
    if not all(instant(v) <= as_of for v in joined.label_available_at):
        raise ValueError("source contains predictions whose labels are not yet mature")
    calendar_path = REPO / config["calendar"]
    if digest(calendar_path) != manifest["sources_sha256"][config["calendar"]]:
        raise ValueError("calendar differs from frozen source snapshot")
    starts = regular_calendar(calendar_path)
    symbols = sorted(joined.symbol.unique().tolist())
    bars_by_symbol: dict[str, dict[datetime, dict]] = {}
    output.mkdir(parents=True)
    files: dict[str, str] = {}
    for symbol in symbols:
        rel = f"{config['source_dir']}/{symbol}/2026.parquet"
        source_path = REPO / rel
        if digest(source_path) != manifest["sources_sha256"].get(rel):
            raise ValueError(f"price source differs from frozen snapshot: {symbol}")
        bars, compact = price_bars(source_path)
        bars_by_symbol[symbol] = bars
        files[f"paths/{symbol}.json"] = write_json(output / "paths" / f"{symbol}.json",
                                                    {"symbol": symbol, "bars": compact,
                                                     "source_sha256": digest(source_path)})
    trials = [json.loads(line) for line in (source / "trials.jsonl").read_text().splitlines()]
    metrics = pd.read_csv(source / "metrics.csv").to_dict(orient="records")
    models = []
    totals = {}
    prediction_sets = []
    policy_runs = []
    for (model, group), frame in joined.groupby(["model", "input_set"], sort=True):
        model_id = f"{model}_{group}"
        records = []
        for row in frame.itertuples(index=False):
            records.append({k: _native(getattr(row, k)) for k in (
                "sample_id", "fold", "symbol", "session_date", "cutoff_et",
                "cutoff_at", "decision_at", "entry_at", "entry_price",
                "label_end_at", "label_available_at", "target", "probability")})
        ledger, trades = replay(records, bars_by_symbol, starts, policy, as_of)
        by_sample = {row["sample_id"]: row for row in records}
        for trade in trades:
            frozen = by_sample[trade["sample_id"]]
            if trade.get("entry_price") is not None:
                if (trade["entry_at"] != instant(frozen["entry_at"]).isoformat() or
                        abs(trade["entry_price"] - frozen["entry_price"]) > 1e-9):
                    raise ValueError(f"replay entry differs from frozen label: {trade['sample_id']}")
            if trade["status"] in ("target_touch_proxy", "window_end_proxy"):
                if (trade["target_hit"] != bool(frozen["target"]) or
                        trade["expected_end_at"] != instant(frozen["label_end_at"]).isoformat()):
                    raise ValueError(f"replay outcome differs from frozen label: {trade['sample_id']}")
        actions = {row["sample_id"]: row for row in ledger}
        for record in records:
            action = actions[record["sample_id"]]
            record["action"] = action["action"]
            record["trade_id"] = action.get("trade_id")
            record["held_trade_id"] = action.get("held_trade_id")
        payload = {"schema": SCHEMA, "model_id": model_id,
                   "prediction_set_id": f"{config['experiment_id']}|outer|{model_id}",
                   "policy_version": policy.record(), "rows": records, "trades": trades}
        filename = f"models/{model_id}.json"
        files[filename] = write_json(output / filename, payload)
        totals[model_id] = summarize(records, ledger, trades)
        spec = model_spec(model, group, config)
        spec["spec_version"] = "model_spec_v1"
        spec["spec_sha256"] = object_digest(spec)
        models.append(spec)
        prediction_sets.append({"type": "PredictionSet", "id": payload["prediction_set_id"],
                                "model_spec_id": model_id, "source_run_id": config["experiment_id"],
                                "scope": "outer_time_fold_development", "rows": len(records),
                                "source_file": "predictions.parquet",
                                "source_sha256": source_files["predictions.parquet"]})
        policy_runs.append({"type": "PolicyRun", "id": f"{POLICY_VERSION}|{payload['prediction_set_id']}",
                            "prediction_set_id": payload["prediction_set_id"],
                            "policy_version": policy.record()["version"],
                            "policy_config_sha256": object_digest(policy.record()),
                            "ledger_file": filename, "ledger_sha256": files[filename]})
    training_runs = []
    for trial in trials:
        if trial["model"] not in ("lgbm", "tcn"):
            continue
        model_id = f"{trial['model']}_{trial['input_set']}"
        suffix = "txt" if trial["model"] == "lgbm" else "pt"
        rel = f"models/{trial['fold']}_{model_id}.{suffix}"
        model_path = source / rel
        training_runs.append({"type": "TrainingRun",
                              "id": f"{config['experiment_id']}|{trial['fold']}|{model_id}",
                              "model_spec_id": model_id, "source_run_id": config["experiment_id"],
                              "config_sha256": source_files["config.json"],
                              "fold": trial["fold"], "status": trial["status"],
                              "seconds": trial.get("seconds"),
                              "model_file": rel if model_path.exists() else None,
                              "model_sha256": digest(model_path) if model_path.exists() else None,
                              "per_round_seconds_status": "not_recorded_in_original_run"})
    overview = {"schema": SCHEMA, "generated_at": as_of.isoformat(),
                "research_status": "development_exploratory_not_independently_validated",
                "source_run": config["experiment_id"], "source_scope": manifest["scope"],
                "source_files_sha256": source_files,
                "source_manifest_sha256": source_files["manifest.json"],
                "prediction_set_sha256": source_files["predictions.parquet"],
                "policy": policy.record(), "model_specs": models, "metrics": metrics,
                "trials": trials, "totals": totals, "files_sha256": files,
                "artifact_registry": {"version": "research_artifacts_v1",
                                      "model_specs": [{"type": "ModelSpec", **spec} for spec in models],
                                      "training_runs": training_runs,
                                      "prediction_sets": prediction_sets,
                                      "policy_runs": policy_runs,
                                      "reports": [{"type": "Report",
                                                   "id": f"{config['experiment_id']}|dashboard|{POLICY_VERSION}",
                                                   "overview_file": "overview.json",
                                                   "integrity_manifest": "EXPORT_MANIFEST.json"}]},
                "source_exclusions_unallocated": manifest.get("exclusions", {}),
                "source_coverage": manifest.get("coverage", {}),
                "notes": ["仅外层时间折预测进入评分；旧运行未保存训练内预测。",
                          "旧 LightGBM 无逐轮曲线；旧 TCN 有逐 epoch 指标但无每轮耗时。",
                          "股票池为当前名单回溯，可用时间为 bar_end+1 秒假设，夜盘未覆盖。",
                          "阈值为演示配置，原始模型输出未校准；+5%触及率不是盈利概率。"]}
    files["overview.json"] = write_json(output / "overview.json", overview)
    (output / "EXPORT_MANIFEST.json").write_text(json.dumps({"schema": SCHEMA,
        "overview_sha256": files["overview.json"], "files_sha256": files,
        "source_files_sha256": source_files}, indent=2) + "\n")
    return {"output": str(output), "models": len(models), "symbols": len(symbols),
            "outer_rows_per_model": int(len(joined) / len(models)), "files": len(files)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
                        default=PROJECT / "runs" / "market_dual_track_pilot_20260925")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--fee-bps-per-side", type=float, default=1.0)
    parser.add_argument("--slippage-bps-per-side", type=float, default=5.0)
    args = parser.parse_args()
    print(json.dumps(export(args.source, args.output, args.threshold,
                            args.fee_bps_per_side, args.slippage_bps_per_side), indent=2))
