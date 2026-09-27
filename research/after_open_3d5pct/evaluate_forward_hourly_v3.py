"""Mature append-only paper forecasts after 1170 regular trading minutes."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
import json

import pandas as pd

from .hourly_v3_snapshot import sha256
from .hourly_v3_time import aware

REPO = Path(__file__).resolve().parents[2]
PROJECT = Path(__file__).resolve().parent
UTC = timezone.utc


def evaluate_row(row: dict, calendar: dict, source_dir: Path, now: datetime) -> dict:
    result = {"sample_id": row.get("sample_id"), "symbol": row["symbol"],
              "forecast_status": row["status"], "label_status": "pending",
              "target": None, "entry_price": None, "label_end_at": None,
              "source_sha256": {}}
    if not row.get("predictions"):
        result["label_status"] = "not_scored"
        return result
    cutoff = aware(row["effective_cutoff"])
    entry_start = cutoff+timedelta(minutes=5)
    grid = []
    for s in calendar["sessions"]:
        op, close = aware(s["open_at"]), aware(s["close_at"])
        if close <= entry_start:
            continue
        t = max(op, entry_start)
        while t+timedelta(minutes=5) <= close and len(grid) < 234:
            grid.append(pd.Timestamp(t))
            t += timedelta(minutes=5)
        if len(grid) >= 234:
            break
    if len(grid) < 234:
        result["label_status"] = "pending_calendar_coverage"
        return result
    grid = pd.DatetimeIndex(grid)
    label_end = grid[-1]+pd.Timedelta(minutes=5)
    result["label_end_at"] = label_end.isoformat()
    if now < label_end.to_pydatetime():
        return result
    frames = []
    for year in sorted({str(t.year) for t in grid}):
        path = source_dir/row["symbol"]/f"{year}.parquet"
        if path.exists():
            frames.append(pd.read_parquet(path))
            result["source_sha256"][str(path.relative_to(REPO))] = sha256(path)
    if not frames:
        result["label_status"] = "missing_source"
        return result
    bars = pd.concat(frames, ignore_index=True)
    bars["start_at"] = pd.to_datetime(bars.start_at, utc=True)
    bars["available_at"] = pd.to_datetime(bars.available_at, utc=True)
    bars = bars[(bars.session_type == "regular") & (bars.available_at <= pd.Timestamp(now))]
    bars = bars.sort_values(["start_at", "available_at"]).drop_duplicates("start_at", keep="last")
    path = bars.set_index("start_at").reindex(grid)
    if path["open"].isna().any() or path["high"].isna().any() or path["low"].isna().any():
        result["label_status"] = "missing_regular_bar_or_not_yet_available"
        result["observed_bars"] = int(path.open.notna().sum())
        result["required_bars"] = 234
        return result
    entry = float(path.iloc[0].open)
    if entry <= 0:
        result["label_status"] = "invalid_entry_price"
        return result
    hits = path.high.to_numpy(float) >= entry*1.05
    result.update(label_status="matured_assumed_historical_availability",
                  entry_at=entry_start.isoformat(), entry_price=entry,
                  target=int(hits.any()), first_touch_interval_regular_minutes=[int(hits.argmax()*5), int((hits.argmax()+1)*5)] if hits.any() else None,
                  mfe_pct=float(100*(path.high.max()/entry-1)),
                  mae_pct=float(100*(path.low.min()/entry-1)),
                  net_profit_claim=False)
    return result


def evaluate(log_path: Path, output: Path, now: datetime) -> dict:
    if output.exists():
        raise FileExistsError(output)
    calendar_path = REPO/"market_data/calendars/nasdaq_sessions_2026_v1.json"
    calendar = json.loads(calendar_path.read_text())
    entries = [json.loads(line) for line in log_path.read_text().splitlines() if line]
    seen = set()
    rows = []
    for entry in entries:
        key = (entry["run"], entry["symbol"])
        if key in seen:
            continue
        seen.add(key)
        report_path = Path(entry["run"])/"report.json"
        if sha256(report_path) != entry["report_sha256"]:
            rows.append({"run": entry["run"], "symbol": entry["symbol"],
                         "label_status": "report_hash_mismatch"})
            continue
        report = json.loads(report_path.read_text())
        item = next(r for r in report["rows"] if r["symbol"] == entry["symbol"])
        rows.append({"run": entry["run"], **evaluate_row(item, calendar,
                     REPO/"market_data/us_5m", now)})
    matured = [r for r in rows if r["label_status"] == "matured_assumed_historical_availability"]
    payload = {"version": "forward_hourly_evaluation_v3", "evaluated_at": now.isoformat(),
               "forward_log_sha256": sha256(log_path), "calendar_sha256": sha256(calendar_path),
               "rows": rows, "matured_n": len(matured),
               "touches": sum(r["target"] for r in matured),
               "touch_rate": sum(r["target"] for r in matured)/len(matured) if matured else None,
               "pending_or_missing_n": len(rows)-len(matured),
               "independent_release_gate": "pending_pre_frozen_criteria_and_sufficient_forward_samples"}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2)+"\n")
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forward-log", type=Path, default=PROJECT/"runs/forward_hourly_v3.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--as-of", help="ISO timestamp with timezone, defaults to now")
    args = parser.parse_args()
    result = evaluate(args.forward_log.resolve(), args.output.resolve(),
                      aware(args.as_of) if args.as_of else datetime.now(UTC))
    print(json.dumps({"output": str(args.output), "matured_n": result["matured_n"],
                      "pending_or_missing_n": result["pending_or_missing_n"],
                      "release_gate": result["independent_release_gate"]}, ensure_ascii=False))
