"""Real dataset builder for research/after_open_3d5pct.

Constructs:
- Eligible samples across (symbol x trading_session x decision_hour)
- Causal opening features (return, range, distance, volume)
- Delayed simulated entry (11:35, 12:35, 13:35, 14:35)
- Mature outcome labels across 1170 regular market minutes (+5%, MFE, MAE)
- Full exclusion tracking with explicit reasons (insufficient_data, pending, outside_hours)
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timedelta
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo
import pandas as pd

from .config import Config, load_config
from .contracts import Bar, Label, Sample, Session, require_aware
from .features import opening_features
from .labeling import make_label, select_entry
from .timeaxis import ET, session_for

UTC = ZoneInfo("UTC")


def build_dataset(
    symbols: list[str],
    bars_by_symbol: dict[str, list[Bar]],
    sessions: list[Session],
    config: Config,
    as_of: Optional[datetime] = None,
    output_dir: Optional[Path] = None,
    dataset_id: str = "dataset_v1"
) -> dict[str, Any]:
    """Build structured supervised training dataset from real bars."""
    if as_of is None:
        as_of = sessions[-1].close_at + timedelta(seconds=1)

    require_aware(as_of)

    samples: list[Sample] = []
    features_rows: list[dict[str, Any]] = []
    outcomes_rows: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []

    # Map sessions by ET date string for lookups
    session_by_date = {
        s.open_at.astimezone(ET).strftime("%Y-%m-%d"): s for s in sessions
    }

    total_decisions = 0
    mature_count = 0
    hit_count = 0

    for s_date, session in sorted(session_by_date.items()):
        session_open_et = session.open_at.astimezone(ET)

        for hour_str in config.decision_hours_et:
            hh, mm = map(int, hour_str.split(":"))
            cutoff_et = session_open_et.replace(hour=hh, minute=mm)
            cutoff_utc = cutoff_et.astimezone(UTC)
            decision_utc = cutoff_utc + timedelta(seconds=config.signal_latency_seconds)

            if decision_utc >= as_of:
                continue

            for symbol in symbols:
                total_decisions += 1
                sample_id = f"{symbol}|{decision_utc.isoformat()}"
                bars = bars_by_symbol.get(symbol, [])

                if not bars:
                    exclusions.append({
                        "sample_id": sample_id,
                        "symbol": symbol,
                        "decision_at": decision_utc.isoformat(),
                        "reason": "no_bars_for_symbol"
                    })
                    continue

                # 1. Feature Extraction
                try:
                    feat = opening_features(symbol, cutoff_utc, decision_utc, bars, sessions, config)
                except Exception as exc:
                    exclusions.append({
                        "sample_id": sample_id,
                        "symbol": symbol,
                        "decision_at": decision_utc.isoformat(),
                        "reason": f"feature_error: {exc}"
                    })
                    continue

                # 2. Entry Selection
                try:
                    entry = select_entry(symbol, decision_utc, bars, sessions, config)
                except Exception as exc:
                    exclusions.append({
                        "sample_id": sample_id,
                        "symbol": symbol,
                        "decision_at": decision_utc.isoformat(),
                        "reason": f"entry_error: {exc}"
                    })
                    continue

                # 3. Labeling (1170 regular minutes)
                try:
                    label = make_label(entry, bars, sessions, as_of, config)
                except Exception as exc:
                    exclusions.append({
                        "sample_id": sample_id,
                        "symbol": symbol,
                        "decision_at": decision_utc.isoformat(),
                        "reason": f"label_error: {exc}"
                    })
                    continue

                sample = Sample(
                    sample_id=sample_id,
                    symbol=symbol,
                    decision_at=decision_utc,
                    label_end_at=label.end_at,
                    label_available_at=label.available_at,
                    label_status=label.status,
                    target=label.hit
                )
                samples.append(sample)

                features_rows.append({
                    "sample_id": sample_id,
                    "symbol": symbol,
                    "session_date": s_date,
                    "cutoff_at": cutoff_utc.isoformat(),
                    "decision_at": decision_utc.isoformat(),
                    **feat
                })

                outcomes_rows.append({
                    "sample_id": sample_id,
                    "symbol": symbol,
                    "entry_at": entry.start_at.isoformat(),
                    "entry_price": entry.open,
                    "label_status": label.status,
                    "label_end_at": label.end_at.isoformat(),
                    "label_available_at": label.available_at.isoformat() if label.available_at else None,
                    "hit": label.hit,
                    "mfe": label.mfe,
                    "mae": label.mae,
                    "time_to_hit_min": label.hit_minutes_lower,
                    "time_to_hit_max": label.hit_minutes_upper,
                    "reason": label.reason
                })

                if label.status == "mature":
                    mature_count += 1
                    if label.hit == 1:
                        hit_count += 1

    summary = {
        "dataset_id": dataset_id,
        "as_of": as_of.isoformat(),
        "total_evaluated_decisions": total_decisions,
        "total_samples": len(samples),
        "mature_samples": mature_count,
        "base_hit_rate": round(hit_count / mature_count, 4) if mature_count > 0 else 0.0,
        "exclusions_count": len(exclusions),
        "features_count": len(features_rows),
        "outcomes_count": len(outcomes_rows)
    }

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        # Save Parquet tables
        if features_rows:
            pd.DataFrame(features_rows).to_parquet(output_dir / "features.parquet", engine="pyarrow", compression="zstd")
        if outcomes_rows:
            pd.DataFrame(outcomes_rows).to_parquet(output_dir / "outcomes.parquet", engine="pyarrow", compression="zstd")
        if exclusions:
            pd.DataFrame(exclusions).to_parquet(output_dir / "exclusions.parquet", engine="pyarrow", compression="zstd")
        with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    return {
        "summary": summary,
        "samples": samples,
        "features": features_rows,
        "outcomes": outcomes_rows,
        "exclusions": exclusions
    }
