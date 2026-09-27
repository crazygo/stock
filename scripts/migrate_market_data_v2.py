#!/usr/bin/env python3
"""Migrate research_v2 directory layout to clean first-class market_data directories.

Target layout:
  market_data/us_5m/<SYMBOL>/2026.parquet
  market_data/us_60m_raw/<SYMBOL>/2026.parquet
  market_data/corporate_actions/<SYMBOL>.parquet
  market_data/calendars/
  market_data/universe/
  market_data/manifests/
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parent.parent


def migrate(dry_run: bool = False):
    src_base = ROOT / "market_data" / "research_v2"
    if not src_base.exists():
        print(f"Source directory {src_base} does not exist. Migration not needed.")
        return

    dest_base = ROOT / "market_data"

    # 1. Migrate 5m and 60m futu data
    src_futu = src_base / "futu"
    if src_futu.exists():
        for sym_dir in sorted(src_futu.iterdir()):
            if not sym_dir.is_dir():
                continue
            sym = sym_dir.name

            # 5m
            src_5m = sym_dir / "5m" / "NONE" / "ALL" / "2026.parquet"
            if src_5m.exists():
                dest_5m = dest_base / "us_5m" / sym / "2026.parquet"
                if not dry_run:
                    dest_5m.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(src_5m), str(dest_5m))
                print(f"[Move] 5m: {sym} -> market_data/us_5m/{sym}/2026.parquet")

            # 60m
            src_60m = sym_dir / "60m" / "NONE" / "ALL" / "2026.parquet"
            if src_60m.exists():
                dest_60m = dest_base / "us_60m_raw" / sym / "2026.parquet"
                if not dry_run:
                    dest_60m.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(src_60m), str(dest_60m))
                print(f"[Move] 60m: {sym} -> market_data/us_60m_raw/{sym}/2026.parquet")

    # 2. Migrate corporate_actions
    src_ca = src_base / "corporate_actions"
    if src_ca.exists():
        dest_ca = dest_base / "corporate_actions"
        dest_ca.mkdir(parents=True, exist_ok=True)
        for ca_file in sorted(src_ca.glob("*.parquet")):
            dest_file = dest_ca / ca_file.name
            if not dry_run:
                shutil.move(str(ca_file), str(dest_file))
            print(f"[Move] Rehab: {ca_file.name} -> market_data/corporate_actions/{ca_file.name}")

    # 3. Migrate calendars, universe, manifests
    for folder in ["calendars", "universe", "manifests"]:
        src_folder = src_base / folder
        if src_folder.exists():
            dest_folder = dest_base / folder
            dest_folder.mkdir(parents=True, exist_ok=True)
            for f in src_folder.iterdir():
                if f.is_file():
                    dest_f = dest_folder / f.name
                    if not dry_run:
                        shutil.move(str(f), str(dest_f))
                    print(f"[Move] {folder}: {f.name} -> market_data/{folder}/{f.name}")

    # Clean up empty source directory
    if not dry_run:
        try:
            shutil.rmtree(str(src_base))
            print(f"\nSuccessfully cleaned up {src_base}")
        except Exception as e:
            print(f"Warning: could not remove {src_base}: {e}")

    print("\nMigration completed successfully.")


if __name__ == "__main__":
    dry = "--dry-run" in sys.argv
    migrate(dry_run=dry)
