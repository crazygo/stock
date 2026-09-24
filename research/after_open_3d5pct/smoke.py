"""Deterministic synthetic integration exercise. Never reads market/account data."""

from collections import Counter
from dataclasses import asdict
from datetime import date, datetime, time, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess

from .baseline import fit_base_rate, score_constant
from .config import Config
from .contracts import Bar, Sample, Session
from .features import opening_features
from .labeling import make_label, select_entry
from .splits import purged_fold
from .timeaxis import ET


def encode(value):
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    raise TypeError(type(value).__name__)


def serialized(value) -> str:
    return json.dumps(value, default=encode, ensure_ascii=False, sort_keys=True, allow_nan=False)


def fixture():
    """Weekdays are ONLY a mock calendar here, never a production exchange calendar."""
    sessions = []
    day = date(2026, 1, 5)
    while len(sessions) < 35:
        if day.weekday() < 5:
            sessions.append(Session(datetime.combine(day, time(9, 30), ET),
                                    datetime.combine(day, time(16), ET)))
        day += timedelta(days=1)
    bars = []
    for symbol, phase in (("SYN_A", 0), ("SYN_B", 1.7)):
        index = 0
        for session in sessions[:30]:
            start = session.open_at
            while start < session.close_at:
                end = start + timedelta(minutes=5)
                o = 100 + 6 * math.sin(index / 220 + phase) + 0.002 * index
                c = 100 + 6 * math.sin((index + 1) / 220 + phase) + 0.002 * (index + 1)
                bars.append(Bar(symbol, start, end, end + timedelta(seconds=1),
                                o, max(o, c) + 0.08, min(o, c) - 0.08, c,
                                1000 + index % 137, "synthetic_v1"))
                start = end
                index += 1
    return sessions, bars


def run_smoke(output: Path, config: Config) -> dict:
    if output.exists():
        raise ValueError("output already exists; use a new run directory to preserve evidence")
    sessions, bars = fixture()
    as_of = sessions[29].close_at + timedelta(seconds=1)
    records, samples = [], []
    for session in sessions[:30]:
        for hour in config.decision_hours_et:
            hh, mm = map(int, hour.split(":"))
            cutoff = session.open_at.replace(hour=hh, minute=mm)
            decision = cutoff + timedelta(seconds=config.signal_latency_seconds)
            for symbol in ("SYN_A", "SYN_B"):
                features = opening_features(symbol, cutoff, decision, bars, sessions, config)
                entry = select_entry(symbol, decision, bars, sessions, config)
                label = make_label(entry, bars, sessions, as_of, config)
                sample = Sample(f"{symbol}|{decision.isoformat()}", symbol, decision,
                                label.end_at, label.available_at, label.status, label.hit)
                samples.append(sample)
                records.append({**asdict(sample), "feature_cutoff_at": cutoff, "features": features,
                                "entry_at": entry.start_at, "entry_price_proxy": entry.open,
                                "outcome": asdict(label)})
    folds, predictions = [], []
    for number, (start, end) in enumerate(((12, 17), (22, 27)), 1):
        start_at = sessions[start].open_at.replace(hour=0, minute=0)
        end_at = sessions[end].open_at.replace(hour=0, minute=0)
        fold = purged_fold(samples, start_at, end_at, as_of)
        probability = fit_base_rate(fold.train)
        folds.append({"fold": number, "train_ids": [s.sample_id for s in fold.train],
                      "validation_ids": [s.sample_id for s in fold.validation],
                      "excluded": fold.excluded, "train_base_rate": probability,
                      "validation_start": start_at, "validation_end": end_at,
                      "metrics": score_constant(probability, fold.validation)})
        predictions.extend({"sample_id": s.sample_id, "fold": number, "model": "B0_train_rate",
                            "p_hit_5pct_3d": probability, "target": s.target,
                            "data_kind": "synthetic"} for s in fold.validation)
    source_root = Path(__file__).parent
    source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in sorted(source_root.glob("*.py"))}
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source_root,
                                         text=True, stderr=subprocess.DEVNULL).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = "unavailable"
    manifest = {
        "scope": "scaffold_smoke", "data_kind": "synthetic", "claim": "no_predictive_evidence",
        "created_at": datetime.now(timezone.utc), "strategy_id": config.strategy_id,
        "config_sha256": hashlib.sha256(serialized(config.to_dict()).encode()).hexdigest(),
        "dataset_sha256": hashlib.sha256(serialized([asdict(b) for b in bars]).encode()).hexdigest(),
        "calendar_sha256": hashlib.sha256(serialized([asdict(s) for s in sessions]).encode()).hexdigest(),
        "calendar_source": "synthetic_weekday_fixture_not_exchange_calendar",
        "source_sha256": source_hashes, "git_commit": commit, "evaluation_as_of": as_of,
        "samples": len(samples), "label_status_counts": dict(Counter(s.label_status for s in samples)),
        "validation_predictions": len(predictions),
    }
    output.mkdir(parents=True, exist_ok=False)
    for name, value in (("manifest.json", manifest), ("config.json", config.to_dict()), ("folds.json", folds)):
        (output / name).write_text(json.dumps(value, default=encode, ensure_ascii=False, sort_keys=True,
                                            indent=2, allow_nan=False) + "\n")
    for name, values in (("samples.jsonl", records), ("predictions.jsonl", predictions)):
        (output / name).write_text("".join(serialized(row) + "\n" for row in values))
    (output / "report.md").write_text(
        "# Synthetic scaffold smoke\n\n合成数据：只验证工程连通性，不构成预测或收益证据。\n\n"
        f"样本 {len(samples)}；验证预测 {len(predictions)}；两次按时间向前验证。\n\n"
        "模型为 B0 训练集常数基准率；未训练特征模型。详见 manifest.json 与 folds.json。\n")
    return manifest
