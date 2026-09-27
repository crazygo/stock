"""One-shot, read-only hourly research decision report; never submits orders."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import json
import math
import os

import pandas as pd

from .hourly_v3_time import aware, resolve_time
from .hourly_v3_snapshot import make_snapshot, sha256, visible_versions
from .inference_hourly_v3 import load_artifact, build_visible_features, predict

REPO = Path(__file__).resolve().parents[2]
PROJECT = Path(__file__).resolve().parent
UTC = timezone.utc


def _jsonable(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if hasattr(value, "item"):
        return _jsonable(value.item())
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _cohort_evidence(run: Path, fold: str, family: str, input_set: str,
                     hour: str) -> dict:
    predictions = pd.read_parquet(run/"predictions.parquet")
    p = predictions[(predictions.fold == fold) & (predictions.model == family) &
                    (predictions.input_set == input_set) &
                    (predictions.score_kind == "calibrated") &
                    (predictions.cutoff_et == hour)]
    n = len(p)
    return {"scope": "exposed_development_outer_same_hour", "fold": fold,
            "hour_et": hour, "mature_n": n, "touches": int(p.target.sum()) if n else 0,
            "touch_rate": float(p.target.mean()) if n else None,
            "mean_calibrated_output": float(p.probability.mean()) if n else None,
            "brier": float(((p.probability-p.target)**2).mean()) if n else None,
            "independent": False}


def _reference(snapshot: dict, repo: Path, symbol: str,
               cutoff: datetime, deadline: datetime) -> dict:
    info = snapshot["watermarks"].get(symbol, {})
    for rel in info.get("snapshot_files", []):
        f = pd.read_parquet(repo/rel)
        f = visible_versions(f, cutoff, deadline)
        bar = f[(f.session_type == "regular") & (f.end_at == pd.Timestamp(cutoff))]
        if len(bar):
            row = bar.iloc[-1]
            return {"price": float(row.close), "price_kind": "last_completed_5m_close",
                    "price_at": cutoff.isoformat(), "source_file": rel,
                    "source_sha256": info.get("snapshot_sha256", {}).get(rel)}
    return {"price": None, "price_kind": "unavailable", "price_at": None}


def _append_forward(records: list[dict], run_dir: Path, report_hash: str) -> None:
    path = PROJECT/"runs"/"forward_hourly_v3.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        for row in records:
            line = json.dumps({"version": "forward_research_record_v3",
                               "run": str(run_dir), "report_sha256": report_hash,
                               "sample_id": row.get("sample_id"), "symbol": row["symbol"],
                               "status": row["status"], "requested_at": row["requested_at"],
                               "effective_cutoff": row["effective_cutoff"],
                               "generated_at": row["generated_at"]}, ensure_ascii=False)+"\n"
            os.write(fd, line.encode())
        os.fsync(fd)
    finally:
        os.close(fd)


def classify_action(*, complete: bool, score: float | None,
                    resolution_state: str, historical: bool,
                    generated_at: datetime, expires_at: datetime,
                    received: bool, universe_known: bool,
                    manually_excluded: bool, threshold: float | None) -> tuple[str, str]:
    if not complete:
        return "unavailable", "stock_or_market_or_industry_latest_bar_missing"
    if score is None:
        return "unavailable", "required_H_A_B_feature_incomplete_or_unavailable"
    if resolution_state == "expired":
        return "expired", "delayed_entry_window_elapsed_before_report_generated"
    if historical:
        return "historical_research_only", "requested historical time; not a current action"
    if generated_at > expires_at:
        return "expired", "delayed_entry_window_elapsed_before_report_generated"
    if not received or not universe_known:
        return "data_unverified", "receipt or PIT membership not verified at information deadline"
    if manually_excluded:
        return "observe_manual_exclusion", "symbol manually marked as already held or excluded"
    if threshold is None:
        return "no_action", "development calibration did not select an eligible threshold"
    if score >= threshold:
        return "research_candidate", "development threshold passed; independent release gate pending"
    return "no_action", "development threshold not passed"


def run_once(config_path: Path, model_run: Path, output: Path,
             requested_at: datetime, *, historical_parameter: bool,
             acquire: str = "auto", symbols: set[str] | None = None,
             manual_exclusions: set[str] | None = None) -> dict:
    if output.exists():
        raise FileExistsError(f"new run output already exists: {output}")
    config = json.loads(config_path.read_text())
    artifact = load_artifact(model_run, fold=config["enable_fold"],
                             family=config["enable_model"].split("_")[0],
                             input_set=config["enable_model"].split("_")[1])
    calendar = json.loads((REPO/config["calendar"]).read_text())
    supported = [h for h in config["decision_hours_et"] if h in artifact.manifest["supported_cutoffs_et"]]
    metric_frame = pd.read_csv(model_run/"metrics.csv")
    metric_rows = metric_frame[(metric_frame.fold == artifact.fold) &
                               (metric_frame.model == artifact.family) &
                               (metric_frame.input_set == artifact.input_set)]
    raw_metric = metric_rows[metric_rows.score_kind == "raw"].iloc[0]
    cal_metric = metric_rows[metric_rows.score_kind == "calibrated"].iloc[0]
    resolution = resolve_time(requested_at, calendar, supported,
                              config["signal_latency_seconds"], config["manual_delay_seconds"])
    generated = datetime.now(UTC)
    output.mkdir(parents=True)
    universe = json.loads((REPO/config["universe"]).read_text())
    candidates = [m for m in universe["members"] if m["role"] == "candidate" and
                  (symbols is None or m["symbol"] in symbols)]
    base = {"protocol": "hourly_once_v3", "generated_at": generated.isoformat(),
            "requested_at": requested_at.isoformat(), "historical_parameter": historical_parameter,
            "resolution": resolution.json(), "target_contract": config["label_contract"],
            "target_definition": "touch +5% from delayed 5m Open within 1170 regular trading minutes; not net profit",
            "model_artifact": {"id": artifact.id, "model_run": str(model_run),
                               "checkpoint": str(artifact.checkpoint),
                               "checkpoint_sha256": sha256(artifact.checkpoint),
                               "calibration": artifact.calibration,
                               "calibration_sha256": sha256(model_run/"models"/f"{artifact.id}.calibration.json"),
                               "raw_output_preserved": True,
                               "supported_cutoffs_et": artifact.manifest["supported_cutoffs_et"]},
            "release_gate": {"status": "pending_forward_independent_validation",
                             "high_confidence_advice_enabled": False,
                             "reason": "minimum success, coverage, risk and sample criteria not frozen or independently passed",
                             "development_outer_raw_brier": float(raw_metric.brier),
                             "development_outer_calibrated_brier": float(cal_metric.brier),
                             "development_calibration_worsened": bool(cal_metric.brier > raw_metric.brier)},
            "rows": []}
    if resolution.effective_cutoff is None:
        for m in candidates:
            base["rows"].append({"symbol": m["symbol"], "status": "unavailable",
                                 "reason": resolution.reason, "requested_at": requested_at.isoformat(),
                                 "effective_cutoff": None, "generated_at": generated.isoformat(),
                                 "time_state": resolution.state,
                                 "evidence": {"calendar": resolution.json()}})
    else:
        cutoff = aware(resolution.effective_cutoff)
        deadline = aware(resolution.information_available_deadline)
        refresh = acquire == "auto" and not historical_parameter and resolution.state == "aligned"
        snapshot = make_snapshot(REPO, output, config, cutoff, deadline,
                                 refresh=refresh, symbols=symbols)
        base["snapshot_manifest"] = str(output/"snapshot_manifest.json")
        base["snapshot_sha256"] = sha256(output/"snapshot_manifest.json")
        bundle = None
        try:
            bundle = build_visible_features(REPO, snapshot, artifact,
                                            resolution.session_date, pd.Timestamp(cutoff), symbols)
            scored = predict(artifact, bundle)
            scored_by_symbol = {r.symbol: r for r in scored.itertuples()}
            features_by_symbol = {r.symbol: r for r in bundle.rows.itertuples()}
            base["feature_exclusions"] = bundle.exclusions
            bundle.rows.to_parquet(output/"visible_features.parquet", index=False)
            scored.to_parquet(output/"predictions.parquet", index=False)
            base["visible_features_sha256"] = sha256(output/"visible_features.parquet")
            base["predictions_sha256"] = sha256(output/"predictions.parquet")
        except Exception as exc:
            base["feature_build_error"] = str(exc)
            scored_by_symbol, features_by_symbol = {}, {}
        # Acquisition can take longer than the delayed-entry window. Action
        # status must use the actual post-acquisition generation time.
        generated = datetime.now(UTC)
        base["generated_at"] = generated.isoformat()
        hour = pd.Timestamp(cutoff).tz_convert("America/New_York").strftime("%H:%M")
        cohort = _cohort_evidence(model_run, artifact.fold, artifact.family, artifact.input_set, hour)
        threshold = artifact.calibration["threshold"]
        universe_available = aware(universe.get("generated_at")) <= deadline
        for member in candidates:
            symbol = member["symbol"]
            peer = member["industry_proxy"]
            member_known = universe_available and (
                member.get("available_at") is not None and aware(member["available_at"]) <= deadline) and (
                member.get("first_seen_at") is not None and aware(member["first_seen_at"]) <= deadline) and (
                member.get("effective_from") is None or aware(member["effective_from"]) <= cutoff) and (
                member.get("effective_to") is None or cutoff < aware(member["effective_to"]))
            wm = snapshot["watermarks"].get(symbol, {"status": "source_missing"})
            qm = snapshot["watermarks"].get("QQQ", {"status": "source_missing"})
            sm = snapshot["watermarks"].get(peer, {"status": "source_missing"})
            reference = _reference(snapshot, REPO, symbol, cutoff, deadline)
            entry_cap = reference["price"]
            evidence = {"calendar": resolution.json(), "stock_watermark": wm,
                        "qqq_watermark": qm, "industry_proxy": peer,
                        "industry_watermark": sm, "reference": reference,
                        "universe_mode": universe["universe_mode"],
                        "universe_generated_at": universe.get("generated_at"),
                        "universe_known_by_deadline": member_known,
                        "member_available_at": member.get("available_at"),
                        "member_first_seen_at": member.get("first_seen_at"),
                        "corporate_action_point_in_time": "not_verified; not used as inference input",
                        "source_availability_quality": snapshot["historical_availability"],
                        "cohort_validation": cohort,
                        "threshold_rule": threshold,
                        "release_gate": base["release_gate"]}
            score = scored_by_symbol.get(symbol)
            feature = features_by_symbol.get(symbol)
            score_values = {}
            if score is not None:
                score_values = {"raw_model_output": float(score.raw_model_output),
                                "calibrated_development_probability": float(score.calibrated_development_probability)}
                evidence["feature_values"] = {k: _jsonable(getattr(feature, k)) for k in bundle.rows.columns
                                              if k.startswith(("h_", "a_", "b_", "ab_"))}
                evidence["market_relative"] = {"b_vs_qqq": evidence["feature_values"].get("b_vs_qqq"),
                                               "b_vs_sector": evidence["feature_values"].get("b_vs_sector")}
            all_complete = all(w.get("status", "").startswith("complete") for w in (wm, qm, sm))
            all_received = all(w.get("status") == "complete_observed_receipt" for w in (wm, qm, sm))
            status, reason = classify_action(
                complete=all_complete,
                score=float(score.calibrated_development_probability) if score is not None else None,
                resolution_state=resolution.state, historical=historical_parameter,
                generated_at=generated, expires_at=aware(resolution.action_expires_at),
                received=all_received, universe_known=member_known,
                manually_excluded=symbol in (manual_exclusions or set()),
                threshold=threshold["threshold"])
            base["rows"].append({"sample_id": getattr(score, "sample_id", None),
                                 "symbol": symbol, "industry_proxy": peer,
                                 "status": status, "reason": reason,
                                 "time_state": "expired" if resolution.state == "expired" or
                                               (not historical_parameter and generated > aware(resolution.action_expires_at))
                                               else resolution.state,
                                 "requested_at": requested_at.isoformat(),
                                 "effective_cutoff": cutoff.isoformat(),
                                 "generated_at": generated.isoformat(),
                                 "information_available_deadline": deadline.isoformat(),
                                 "valid_until": resolution.action_expires_at,
                                 "reference_price": reference["price"],
                                 "reference_price_at": reference["price_at"],
                                 "maximum_acceptable_entry_price": entry_cap,
                                 "maximum_price_rule": "development guard: no price above completed cutoff close; not validated",
                                 "indicative_target_if_entry_at_cap": entry_cap*1.05 if entry_cap else None,
                                 "target_rule": "actual target is actual entry price * 1.05, never cutoff close * 1.05",
                                 "expiry_rule": "after delayed-entry window or price above cap, signal invalid; rerun at next cutoff",
                                 "horizon_regular_minutes": 1170,
                                 "predictions": score_values, "evidence": evidence})
    base["artifact_registry"] = {
        "version": "research_artifacts_hourly_v3",
        "ModelSpec": {"id": f"market_hab_hourly_v3_{artifact.family}_{artifact.input_set}",
                      "feature_schema": artifact.manifest["feature_schema"],
                      "label_contract": config["label_contract"],
                      "supported_cutoffs_et": artifact.manifest["supported_cutoffs_et"],
                      "sequence_channels": artifact.manifest["sequence_channels"],
                      "preprocessing": artifact.manifest["preprocessing"],
                      "model_file_sha256": sha256(artifact.checkpoint)},
        "TrainingRun": {"id": f"{artifact.manifest['config_id']}:{artifact.fold}:{artifact.family}_{artifact.input_set}",
                        "manifest_sha256": sha256(model_run/"manifest.json"),
                        "config_sha256": sha256(model_run/"config.json"),
                        "scope": artifact.manifest["scope"]},
        "PredictionSet": {"id": output.name+":predictions",
                          "sha256": base.get("predictions_sha256"),
                          "snapshot_sha256": base.get("snapshot_sha256"),
                          "label_status": "pending_no_future_outcomes_in_inference"},
        "PolicyRun": {"id": output.name+":manual_hourly_research_v3",
                      "threshold_source_sha256": sha256(model_run/"models"/f"{artifact.id}.calibration.json"),
                      "manual_exclusions": sorted(manual_exclusions or set()),
                      "real_account_positions_used": False,
                      "high_confidence_advice_enabled": False},
        "Report": {"id": output.name+":report", "path": str(output/"report.json"),
                   "scope": "research_only"}}
    report = _jsonable(base)
    report_path = output/"report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    if not historical_parameter:
        _append_forward(report["rows"], output, sha256(report_path))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of", help="ISO timestamp with UTC offset; omitted means now")
    parser.add_argument("--config", type=Path, default=PROJECT/"configs/hourly_v3.json")
    parser.add_argument("--model-run", type=Path, default=PROJECT/"runs/hourly_once_v3_20260925")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--acquire", choices=["auto", "local"], default="auto")
    parser.add_argument("--symbols", nargs="*", help="optional research subset")
    parser.add_argument("--exclude-held", nargs="*", default=[], help="manual symbol exclusions, no account sync")
    args = parser.parse_args()
    report = run_once(args.config.resolve(), args.model_run.resolve(), args.output.resolve(),
                      aware(args.as_of) if args.as_of else datetime.now(UTC),
                      historical_parameter=bool(args.as_of), acquire=args.acquire,
                      symbols=set(args.symbols) if args.symbols else None,
                      manual_exclusions=set(args.exclude_held))
    counts = pd.Series([r["status"] for r in report["rows"]]).value_counts().to_dict()
    print(json.dumps({"output": str(args.output.resolve()), "resolution": report["resolution"],
                      "status_counts": counts, "release_gate": report["release_gate"]["status"]},
                     ensure_ascii=False, indent=2))
