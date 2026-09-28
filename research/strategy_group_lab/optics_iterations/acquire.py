"""Isolated snapshot acquisition: local -> R2 -> paced OpenD; never upload."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
from datetime import datetime, timezone
import pandas as pd
from scripts.fetch_research_data import ResearchDataFetcher
from research.group_expectation_matrix.build import file_hash, write_json

ROOT = Path(__file__).resolve().parents[3]
FOCUS = ["LITE", "COHR", "CIEN", "AAOI", "FN", "AXTI", "CRDO", "MRVL", "AVGO", "ALAB", "ANET", "CSCO"]

def acquire(output, end="2026-09-25"):
    target = output/"market_data"
    target.mkdir(parents=True, exist_ok=True)
    log = []
    fetcher = ResearchDataFetcher(data_root=target)
    try:
        for sym in FOCUS+["QQQ"]:
            dst = target/"us_5m"/sym/"2026.parquet"
            src = ROOT/"market_data/us_5m"/sym/"2026.parquet"
            if not dst.exists() and src.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
            start = "2026-01-02"
            if dst.exists():
                current = pd.read_parquet(dst)
                if current.session_date.min() <= start:
                    start = min(str(current.session_date.max()), end)
            # The shared fetcher checks local, then R2, before any OpenD request.
            frame = fetcher.fetch_kline(sym, start, end, interval="5m", price_basis="NONE", session="ALL")
            action = target/"corporate_actions"/f"{sym}.parquet"
            original_action = ROOT/"market_data/corporate_actions"/f"{sym}.parquet"
            if not action.exists() and original_action.exists():
                action.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(original_action, action)
            if not action.exists() and fetcher.r2_client:
                remote = f"corporate_actions/{sym}.parquet"
                try:
                    if fetcher.r2_client.head_object(remote):
                        action.parent.mkdir(parents=True, exist_ok=True)
                        fetcher.r2_client.get_object(remote, action)
                except Exception:
                    print(f"{sym}: remote action metadata unavailable; checking OpenD", flush=True)
            if not action.exists():
                fetcher.fetch_corporate_actions(sym)
            if dst.exists():
                frame = pd.read_parquet(dst)
                frame.to_parquet(dst, compression="zstd", compression_level=7, index=False)
            if action.exists():
                pd.read_parquet(action).to_parquet(action, compression="zstd", compression_level=7, index=False)
            record = {"symbol": sym, "status": "available" if dst.exists() and action.exists() else "missing",
                      "bars": len(frame) if frame is not None else 0,
                      "last_end_at": str(frame.end_at.max()) if frame is not None and len(frame) else None,
                      "price_basis": "NONE", "bar_hash": file_hash(dst) if dst.exists() else None,
                      "actions_hash": file_hash(action) if action.exists() else None}
            log.append(record)
            write_json(output/"acquisition.json", {"observed_at": datetime.now(timezone.utc).isoformat(), "requested_end": end, "securities": log})
            print(json.dumps(record, ensure_ascii=False), flush=True)
    finally:
        fetcher.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--end", default="2026-09-25")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    acquire(args.output, args.end)
