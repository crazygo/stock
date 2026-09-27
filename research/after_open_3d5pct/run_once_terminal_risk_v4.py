"""Read-only one-shot terminal-risk research prediction from an isolated snapshot."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch

from .hourly_v3_data import build_bundle
from .hourly_v3_snapshot import make_snapshot
from .hourly_v3_time import aware, resolve_time
from .hourly_v4_visibility import (materialize_visible_view, source_watermark_v4,
                                   visible_versions_v4)
from .terminal_risk_v4 import REPO, mixed_expected_net, net_return, sha256
from .train_hourly_v3 import MultiBranchTCN, _static, _tabular, apply_calibrator

PROJECT = Path(__file__).resolve().parent
UTC = timezone.utc


def _load_model_bundle(model_run: Path) -> tuple[dict, dict, dict]:
    manifest = json.loads((model_run / "manifest.json").read_text())
    config = json.loads((model_run / "config.json").read_text())
    registered = PROJECT / "configs/terminal_risk_v4.json"
    if sha256(registered) != manifest["config_sha256"] or config != json.loads(registered.read_text()):
        raise ValueError("v4 model config differs from frozen registration")
    for name, expected in manifest["model_artifacts_sha256"].items():
        if sha256(model_run / "models" / name) != expected:
            raise ValueError(f"v4 model artifact hash mismatch: {name}")
    if sha256(model_run / "provenance_at_launch.json") != manifest["provenance_at_launch_sha256"]:
        raise ValueError("v4 provenance hash mismatch")
    v3_run = REPO / config["source_run"]
    v3_config = json.loads((v3_run / "config.json").read_text())
    provenance = json.loads((model_run / "provenance_at_launch.json").read_text())
    if sha256(v3_run / "config.json") != provenance["v3_actual_config_sha256"]:
        raise ValueError("v3 source training config hash mismatch")
    if v3_config["feature_schema"] != config["feature_schema"]:
        raise ValueError("v4/v3 feature schema mismatch")
    return config, manifest, v3_config


def _risk_scores(model_run: Path, fold: str, tabular: pd.DataFrame,
                 config: dict) -> dict[str, np.ndarray]:
    models = {name: lgb.Booster(model_file=str(model_run / "models" / f"{fold}_risk_{name}.txt"))
              for name in ("mean", "q10", "q50", "q90", "loss")}
    adjustment = json.loads((model_run / "models" / f"{fold}_risk.calibration.json").read_text())
    if tabular.columns.tolist() != adjustment["feature_columns"]:
        raise ValueError("risk head feature order mismatch")
    mean = np.asarray(models["mean"].predict(tabular), dtype=float) + adjustment["mean_offset"]
    qs = np.column_stack([models[f"q{int(q*100)}"].predict(tabular) for q in (.1, .5, .9)])
    qs = np.sort(qs + np.asarray(adjustment["quantile_offsets"]), axis=1)
    loss_raw = np.asarray(models["loss"].predict(tabular), dtype=float)
    loss = apply_calibrator(loss_raw, adjustment["loss_calibrator"])
    return {"failure_mean_gross": mean,
            "failure_q10_gross": qs[:, 0], "failure_q50_gross": qs[:, 1],
            "failure_q90_gross": qs[:, 2], "failure_loss_probability": loss}


def _touch_scores(model_run: Path, fold: str, family: str, input_set: str,
                  bundle, v3_config: dict) -> tuple[np.ndarray, np.ndarray]:
    key = f"{fold}_{family}_{input_set}"
    if family == "lgbm":
        model = lgb.Booster(model_file=str(model_run / "models" / f"{key}.txt"))
        x = _tabular(bundle, input_set)
        if x.columns.tolist() != model.feature_name():
            raise ValueError(f"touch feature order mismatch: {key}")
        raw = np.asarray(model.predict(x), dtype=float)
    else:
        torch.set_num_threads(4)
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        saved = torch.load(model_run / "models" / f"{key}.pt", map_location="cpu", weights_only=False)
        if saved["input_set"] != input_set or saved["config"] != v3_config["tcn"]:
            raise ValueError(f"TCN schema/config mismatch: {key}")
        model = MultiBranchTCN(v3_config["tcn"], input_set, saved["static_columns"])
        model.load_state_dict(saved["state_dict"])
        model.to(device).eval()
        static = _static(bundle, input_set)
        arrays = [bundle.h_seq, bundle.post_seq, bundle.pre_seq, bundle.b_seq]
        tensors = [torch.from_numpy(np.clip(a, -50, 50)).to(device) for a in arrays]
        with torch.no_grad():
            raw = torch.sigmoid(model(*tensors, torch.from_numpy(static).to(device))).cpu().numpy()
    calibration = json.loads((model_run / "models" / f"{key}.calibration.json").read_text())
    return raw, apply_calibrator(raw, calibration["calibrator"])


def _reference(snapshot: dict, symbol: str, cutoff: datetime,
               knowledge_at: datetime) -> dict:
    info = snapshot["watermarks"].get(symbol, {})
    for rel in info.get("snapshot_files", []):
        f = pd.read_parquet(REPO / rel)
        v = visible_versions_v4(f, pd.Timestamp(cutoff), pd.Timestamp(knowledge_at))
        v = v[(v.end_at == pd.Timestamp(cutoff)) & (v.session_type == "regular")]
        if not v.empty:
            row = v.iloc[-1]
            return {"price": float(row.close), "price_kind": "last_completed_5m_close_not_entry_price",
                    "price_at": cutoff.isoformat(), "source_file": rel,
                    "source_sha256": info.get("snapshot_sha256", {}).get(rel)}
    return {"price": None, "price_kind": "unavailable", "price_at": None}


def _reuse_verified_raw_snapshot(previous: Path, output: Path, requested_day: str) -> dict:
    previous_manifest = previous / "snapshot_manifest.json"
    original = json.loads(previous_manifest.read_text())
    if original["session_date"] != requested_day:
        raise ValueError("reused raw snapshot is from a different ET session")
    target_root = output / "snapshot_sources"
    target_root.mkdir()
    reused = json.loads(json.dumps(original))
    for symbol, info in reused["watermarks"].items():
        replacement_files = []
        replacement_hashes = {}
        for rel in info.get("snapshot_files", []):
            source = REPO / rel
            expected = info["snapshot_sha256"].get(rel)
            if not expected or sha256(source) != expected:
                raise ValueError(f"reused raw snapshot file hash mismatch: {rel}")
            target = target_root / symbol / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            new_rel = str(target.relative_to(REPO))
            if sha256(target) != expected:
                raise ValueError(f"reused copy hash mismatch: {new_rel}")
            replacement_files.append(new_rel)
            replacement_hashes[new_rel] = expected
        info["snapshot_files"] = replacement_files
        info["snapshot_sha256"] = replacement_hashes
    reused["version"] = "terminal_risk_v4_reused_raw_snapshot"
    reused["snapshot_dir"] = str(target_root)
    reused["raw_snapshot_source_manifest"] = str(previous_manifest)
    reused["raw_snapshot_source_manifest_sha256"] = sha256(previous_manifest)
    reused["raw_snapshot_source_run"] = str(previous)
    reused["reuse_verified_at"] = datetime.now(UTC).isoformat()
    return reused


def _jsonable(value):
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    return value


def run_once(model_run: Path, output: Path, requested_at: datetime,
             *, historical_parameter: bool = False,
             acquire: str = "auto", symbols: set[str] | None = None,
             snapshot_from: Path | None = None) -> dict:
    if requested_at > datetime.now(UTC):
        raise ValueError("future_requested_at: cannot predict from a future knowledge time")
    if output.exists():
        raise FileExistsError(output)
    config, manifest, v3_config = _load_model_bundle(model_run)
    calendar = json.loads((REPO / v3_config["calendar"]).read_text())
    resolution = resolve_time(requested_at, calendar, v3_config["decision_hours_et"],
                              v3_config["signal_latency_seconds"], v3_config["manual_delay_seconds"])
    resolved_at = datetime.now(UTC)
    universe = json.loads((REPO / v3_config["universe"]).read_text())
    candidates = [m for m in universe["members"] if m["role"] == "candidate" and
                  (symbols is None or m["symbol"] in symbols)]
    output.mkdir(parents=True)
    fold = "september_2026"
    winner = manifest["folds"][fold]["selected_candidate"]
    selected = None if winner == "no_candidate_passed" else winner
    base = {"protocol": "terminal_risk_once_v4", "research_scope": "no_formal_action",
            "requested_at": requested_at.isoformat(), "requested_at_et": requested_at.astimezone(__import__("zoneinfo").ZoneInfo("America/New_York")).isoformat(),
            "historical_parameter": historical_parameter, "resolution": resolution.json(),
            "resolved_at": resolved_at.isoformat(),
            "effective_cutoff_at": resolution.effective_cutoff,
            "model_available_at": manifest["created_at"],
            "model_trained_after_requested_at": requested_at < aware(manifest["created_at"]),
            "registered_model_selection": selected,
            "selection_status": "no_candidate_passed_inner_development_scenario" if selected is None else
                                "selected_only_in_exposed_development",
            "high_confidence_or_formal_action_enabled": False,
            "reason_formal_action_disabled": "no_user_approved_risk_budget_or_independent_forward_validation",
            "model_run": str(model_run), "model_manifest_sha256": sha256(model_run / "manifest.json"),
            "model_artifacts_sha256": manifest["model_artifacts_sha256"],
            "target": "5pct high touch within 1170 regular minutes from delayed 5m Open; no interim stop",
            "costed_valuation": "touch fixed +5pct proxy; otherwise final regular 5m Close mark-to-liquidate proxy; not forced sale",
            "predicted_es95_loss": None, "predicted_es95_reason": "three failure quantiles do not identify mixture ES95",
            "rows": []}
    now = datetime.now(UTC)
    same_day_session = any(s["session_date"] == requested_at.astimezone(__import__("zoneinfo").ZoneInfo("America/New_York")).date().isoformat()
                           for s in calendar["sessions"])
    snapshot = None
    if same_day_session:
        capture_cutoff = aware(resolution.effective_cutoff) if resolution.effective_cutoff else requested_at
        # Current-day refresh is read-only and isolated. An invalid pre-first
        # state still captures the real market watermark before reporting it.
        if snapshot_from is None:
            snapshot = make_snapshot(REPO, output, v3_config, capture_cutoff, now,
                                     refresh=acquire == "auto" and not historical_parameter,
                                     symbols=symbols)
        else:
            snapshot = _reuse_verified_raw_snapshot(snapshot_from, output,
                                                     requested_at.astimezone(__import__("zoneinfo").ZoneInfo("America/New_York")).date().isoformat())
        knowledge_at = requested_at if historical_parameter else datetime.now(UTC)
        snapshot["version"] = "terminal_risk_snapshot_v4_knowledge_time"
        snapshot["requested_at"] = requested_at.isoformat()
        snapshot["cutoff_at"] = capture_cutoff.isoformat()
        snapshot["knowledge_at"] = knowledge_at.isoformat()
        snapshot["information_deadline"] = knowledge_at.isoformat()
        for symbol, info in snapshot["watermarks"].items():
            paths = info.get("snapshot_files", [])
            if paths:
                frames = [pd.read_parquet(REPO / path) for path in paths]
                wm = source_watermark_v4(pd.concat(frames, ignore_index=True),
                                         snapshot["session_date"], pd.Timestamp(capture_cutoff),
                                         pd.Timestamp(knowledge_at))
                info.update(wm)
        view = materialize_visible_view(snapshot, output, pd.Timestamp(capture_cutoff),
                                        pd.Timestamp(knowledge_at))
        (output / "visible_view_manifest.json").write_text(json.dumps(view, ensure_ascii=False, indent=2) + "\n")
        base["visible_view_manifest"] = str(output / "visible_view_manifest.json")
        base["visible_view_manifest_sha256"] = sha256(output / "visible_view_manifest.json")
        (output / "snapshot_manifest.json").write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n")
        base["snapshot_manifest"] = str(output / "snapshot_manifest.json")
        base["snapshot_sha256"] = sha256(output / "snapshot_manifest.json")
        base["raw_snapshot_source_manifest_sha256"] = snapshot.get("raw_snapshot_source_manifest_sha256")
        base["knowledge_at"] = knowledge_at.isoformat()
    else:
        knowledge_at = requested_at
        base["knowledge_at"] = knowledge_at.isoformat()
    score_by_symbol = {}
    feature_by_symbol = {}
    if resolution.effective_cutoff and snapshot:
        cutoff = aware(resolution.effective_cutoff)
        try:
            infer_cfg = dict(v3_config)
            infer_cfg["source_dir"] = str(Path(view["view_dir"]).relative_to(REPO))
            infer_cfg["source_start_date"] = snapshot["source_start_date"]
            infer_cfg["source_end_date"] = snapshot["session_date"]
            # v3 builder ties visibility time to cutoff+latency. In this new
            # inference protocol the actual knowledge time is later, while
            # the *bar/event cutoff stays fixed*. Never claim early receipt.
            latency = int((knowledge_at - cutoff).total_seconds())
            if latency < v3_config["signal_latency_seconds"]:
                raise ValueError("knowledge time precedes legal decision")
            infer_cfg["signal_latency_seconds"] = latency
            view_at = pd.Timestamp(cutoff + __import__("datetime").timedelta(seconds=latency))
            bundle = build_bundle(REPO, infer_cfg, symbols, include_labels=False,
                                  only_date=snapshot["session_date"], as_of=view_at)
            bundle.rows.to_parquet(output / "visible_features.parquet", index=False)
            base["visible_features_sha256"] = sha256(output / "visible_features.parquet")
            risk = _risk_scores(model_run, fold, _tabular(bundle, "HAB"), config)
            for key in [f"{family}_{input_set}" for input_set in config["candidate_input_sets"]
                        for family in config["candidate_families"]]:
                path = model_run / "models" / f"{fold}_{key}.{'txt' if key.startswith('lgbm') else 'pt'}"
                if not path.exists():
                    continue
                family, input_set = key.split("_")
                raw, cal = _touch_scores(model_run, fold, family, input_set, bundle, v3_config)
                for i, member in enumerate(bundle.rows.itertuples()):
                    score_by_symbol.setdefault(member.symbol, {})[key] = {
                        "raw_touch_output": float(raw[i]), "calibrated_development_p_touch": float(cal[i]),
                        "failure_mean_gross": float(risk["failure_mean_gross"][i]),
                        "failure_q10_gross": float(risk["failure_q10_gross"][i]),
                        "failure_q50_gross": float(risk["failure_q50_gross"][i]),
                        "failure_q90_gross": float(risk["failure_q90_gross"][i]),
                        "failure_conditional_loss_probability": float(risk["failure_loss_probability"][i]),
                        "mixed_expected_net_valuation": float(mixed_expected_net(
                            cal[i], risk["failure_mean_gross"][i],
                            touch_gross=config["touch_gross_return"],
                            buy_cost=config["buy_cost_rate"], sell_cost=config["sell_cost_rate"])),
                        "mixed_terminal_loss_probability": float((1 - cal[i]) * risk["failure_loss_probability"][i]),
                        "failure_q10_net_valuation": float(net_return(risk["failure_q10_gross"][i],
                                                                     config["buy_cost_rate"], config["sell_cost_rate"])),
                        "model_checkpoint_sha256": manifest["model_artifacts_sha256"][path.name]}
            feature_by_symbol = {r.symbol: r for r in bundle.rows.itertuples()}
            base["feature_exclusions"] = bundle.exclusions
        except Exception as exc:
            base["feature_or_prediction_error"] = str(exc)
    generated = datetime.now(UTC)
    base["generated_at"] = generated.isoformat()
    for member in candidates:
        symbol = member["symbol"]
        peer = member["industry_proxy"]
        wm = snapshot["watermarks"].get(symbol, {}) if snapshot else {}
        qqq = snapshot["watermarks"].get("QQQ", {}) if snapshot else {}
        industry = snapshot["watermarks"].get(peer, {}) if snapshot else {}
        if resolution.effective_cutoff is None:
            status, reason = "unavailable", resolution.reason
            reference = {"price": None, "price_kind": "unavailable", "price_at": None}
        else:
            cutoff = aware(resolution.effective_cutoff)
            reference = _reference(snapshot, symbol, cutoff, knowledge_at) if snapshot else {"price": None}
            if symbol not in score_by_symbol:
                status, reason = "unavailable", "required_stock_QQQ_industry_or_history_feature_missing_at_knowledge_time"
            elif any(w.get("status") not in ("complete_observed_receipt", "complete_assumed_availability")
                     for w in (wm, qqq, industry)):
                status, reason = "unavailable", "latest_completed_market_bar_watermark_missing_or_stale"
            elif historical_parameter and base["model_trained_after_requested_at"]:
                status, reason = "retrospective_model_application", "model_artifact_was_trained_after_requested_historical_time"
            elif historical_parameter:
                status, reason = "historical_research_score_only", "historical_parameter_not_current_action"
            elif generated > aware(resolution.action_expires_at):
                status, reason = "expired_research_score_only", "delayed_entry_window_expired_or_acquisition_too_late"
            else:
                status, reason = "research_score_only", "no_formal_action_without_risk_budget_and_forward_validation"
        feature = feature_by_symbol.get(symbol)
        scores = score_by_symbol.get(symbol, {})
        primary = scores.get(selected) if selected else None
        base["rows"].append({"symbol": symbol, "industry_proxy": peer, "status": status,
                             "reason": reason, "requested_at": requested_at.isoformat(),
                             "effective_cutoff": resolution.effective_cutoff,
                             "knowledge_at": knowledge_at.isoformat(), "generated_at": generated.isoformat(),
                             "action_expires_at": resolution.action_expires_at,
                             "market_data_asof": {"stock": wm.get("latest_closed_end_at"),
                                                   "QQQ": qqq.get("latest_closed_end_at"),
                                                   "industry": industry.get("latest_closed_end_at")},
                             "latest_visible_source_end_at": wm.get("latest_visible_source_end_at"),
                             "receive_at": {"stock": wm.get("latest_received_at"),
                                            "QQQ": qqq.get("latest_received_at"),
                                            "industry": industry.get("latest_received_at")},
                             "reference": reference,
                             "selected_model": selected, "selected_model_scores": primary,
                             "all_research_model_scores": scores,
                             "evidence": {"snapshot_sha256": base.get("snapshot_sha256"),
                                          "visible_view_manifest_sha256": base.get("visible_view_manifest_sha256"),
                                          "visible_features_sha256": base.get("visible_features_sha256"),
                                          "stock_watermark": wm, "qqq_watermark": qqq,
                                          "industry_watermark": industry,
                                          "feature_values": {c: getattr(feature, c) for c in bundle.rows.columns
                                                             if c.startswith(("h_", "a_", "b_", "ab_"))} if feature is not None else None,
                                          "universe_mode": universe["universe_mode"],
                                          "historical_availability": v3_config["historical_source_quality"],
                                          "model_selection": base["selection_status"],
                                          "model_available_at": base["model_available_at"],
                                          "model_trained_after_requested_at": base["model_trained_after_requested_at"],
                                          "predicted_es95_loss": None}})
    report = _jsonable(base)
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    rows_csv = []
    for row in report["rows"]:
        reference_scores = row["selected_model_scores"] or row["all_research_model_scores"].get("lgbm_HAB", {})
        rows_csv.append({"symbol": row["symbol"], "status": row["status"], "reason": row["reason"],
                         "requested_at": row["requested_at"], "effective_cutoff": row["effective_cutoff"],
                         "knowledge_at": row["knowledge_at"], "market_data_asof": row["market_data_asof"]["stock"],
                         "latest_visible_source_end_at": row["latest_visible_source_end_at"],
                         "receive_at": row["receive_at"]["stock"],
                         "selected_model": row["selected_model"], "research_display_model": row["selected_model"] or "lgbm_HAB_unselected_comparator",
                         "p_touch": reference_scores.get("calibrated_development_p_touch"),
                         "expected_net_valuation": reference_scores.get("mixed_expected_net_valuation"),
                         "terminal_loss_probability": reference_scores.get("mixed_terminal_loss_probability"),
                         "failure_q10_net_valuation": reference_scores.get("failure_q10_net_valuation")})
    pd.DataFrame(rows_csv).to_csv(output / "per_symbol.csv", index=False)
    report["report_sha256"] = sha256(report_path)
    report["per_symbol_csv_sha256"] = sha256(output / "per_symbol.csv")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-run", type=Path, default=PROJECT / "runs/terminal_risk_v4_20260925_r3")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--as-of", help="ISO timestamp with offset; omitted means actual now")
    parser.add_argument("--acquire", choices=("auto", "local"), default="auto")
    parser.add_argument("--snapshot-from", type=Path, help="reuse and hash-verify an earlier raw snapshot from this ET session")
    parser.add_argument("--symbols", nargs="*")
    args = parser.parse_args()
    requested = aware(args.as_of) if args.as_of else datetime.now(UTC)
    report = run_once(args.model_run.resolve(), args.output.resolve(), requested,
                      historical_parameter=bool(args.as_of), acquire=args.acquire,
                      symbols=set(args.symbols) if args.symbols else None,
                      snapshot_from=args.snapshot_from.resolve() if args.snapshot_from else None)
    print(json.dumps({"output": str(args.output.resolve()),
                      "requested_at": report["requested_at"],
                      "effective_cutoff": report["resolution"]["effective_cutoff"],
                      "status_counts": pd.Series([r["status"] for r in report["rows"]]).value_counts().to_dict(),
                      "selected_model": report["registered_model_selection"]}, ensure_ascii=False, indent=2))
