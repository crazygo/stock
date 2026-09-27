"""Artifact loading, visible feature construction, and label-free prediction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch

from .hourly_v3_data import Bundle, build_bundle, SEQ_CHANNELS
from .hourly_v3_snapshot import sha256
from .train_hourly_v3 import MultiBranchTCN, _static, _tabular, apply_calibrator


@dataclass
class LoadedArtifact:
    run: Path
    fold: str
    family: str
    input_set: str
    config: dict
    manifest: dict
    checkpoint: Path
    calibration: dict
    feature_columns: list[str]
    model: object

    @property
    def id(self) -> str:
        return f"{self.fold}_{self.family}_{self.input_set}"


def load_artifact(run: Path, *, fold: str = "september_2026",
                  family: str = "lgbm", input_set: str = "HAB") -> LoadedArtifact:
    run = run.resolve()
    config = json.loads((run/"config.json").read_text())
    manifest = json.loads((run/"manifest.json").read_text())
    if manifest.get("feature_schema") != "market_hab_hourly_v3" or manifest.get("supported_cutoffs_et") != config["decision_hours_et"]:
        raise ValueError("model artifact does not declare hourly_v3 feature/time support")
    if tuple(manifest["sequence_channels"]) != SEQ_CHANNELS or manifest["sequence_lengths"]["b"] != 72:
        raise ValueError("model sequence schema mismatch")
    ext = "txt" if family == "lgbm" else "pt"
    checkpoint = run/"models"/f"{fold}_{family}_{input_set}.{ext}"
    cal_path = run/"models"/f"{fold}_{family}_{input_set}.calibration.json"
    for path in (checkpoint, cal_path):
        expected = manifest["model_artifacts_sha256"].get(path.name)
        if not expected or sha256(path) != expected:
            raise ValueError(f"model artifact hash mismatch: {path.name}")
    calibration = json.loads(cal_path.read_text())
    if family == "lgbm":
        model = lgb.Booster(model_file=str(checkpoint))
        feature_columns = model.feature_name()
    elif family == "tcn":
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if saved["input_set"] != input_set or saved["config"] != config["tcn"]:
            raise ValueError("TCN checkpoint config mismatch")
        model = MultiBranchTCN(config["tcn"], input_set, saved["static_columns"])
        model.load_state_dict(saved["state_dict"])
        model.eval()
        feature_columns = []
    else:
        raise ValueError(f"unsupported model family: {family}")
    return LoadedArtifact(run, fold, family, input_set, config, manifest,
                          checkpoint, calibration, feature_columns, model)


def build_visible_features(repo: Path, snapshot_manifest: dict, artifact: LoadedArtifact,
                           day: str, cutoff: pd.Timestamp,
                           symbols: set[str] | None = None) -> Bundle:
    from zoneinfo import ZoneInfo
    hour_et = cutoff.tz_convert(ZoneInfo("America/New_York")).strftime("%H:%M")
    if hour_et not in artifact.config["decision_hours_et"]:
        raise ValueError("unsupported ET cutoff in artifact")
    config = dict(artifact.config)
    config["source_dir"] = str((Path(snapshot_manifest["snapshot_dir"])).relative_to(repo))
    config["source_start_date"] = snapshot_manifest["source_start_date"]
    config["source_end_date"] = day
    return build_bundle(repo, config, symbols, include_labels=False, only_date=day,
                        as_of=cutoff + pd.Timedelta(seconds=config["signal_latency_seconds"]))


def predict(artifact: LoadedArtifact, bundle: Bundle) -> pd.DataFrame:
    """Return raw and separately calibrated scores; no outcomes are loaded."""
    if artifact.family == "lgbm":
        x = _tabular(bundle, artifact.input_set)
        if x.columns.tolist() != artifact.feature_columns:
            raise ValueError("inference feature order differs from training artifact")
        raw = np.asarray(artifact.model.predict(x), dtype=float)
    else:
        static = _static(bundle, artifact.input_set)
        arrays = [bundle.h_seq, bundle.post_seq, bundle.pre_seq, bundle.b_seq]
        tensors = [torch.from_numpy(np.clip(a, -50, 50)) for a in arrays]
        with torch.no_grad():
            raw = torch.sigmoid(artifact.model(*tensors, torch.from_numpy(static))).numpy()
    calibrated = apply_calibrator(raw, artifact.calibration["calibrator"])
    result = bundle.rows[["sample_id", "symbol", "session_date", "cutoff_et", "cutoff_at", "decision_at"]].copy()
    result["raw_model_output"] = raw
    result["calibrated_development_probability"] = calibrated
    result["label_status"] = "pending"
    return result
