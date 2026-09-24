#!/usr/bin/env python3
"""Incrementally update market_data/us_60m from Futu OpenD for 2026-09-18 through today."""

import gzip
import json
import time
from pathlib import Path
import futu

ROOT = Path("/Users/admin/Code/stock")
HOURLY_DIR = ROOT / "market_data" / "us_60m"
BENCHMARKS = {"QQQ", "SOXX", "IGV", "XLU"}

futu.SysConfig.enable_proto_encrypt(False)
ctx = futu.OpenQuoteContext(host="127.0.0.1", port=11111)

tickers = sorted([p.parent.name for p in HOURLY_DIR.glob("*/2026.json.gz")])

print(f"Starting incremental sync for {len(tickers)} stocks from Futu OpenD...")
success_count = 0
added_bars_total = 0

for i, t in enumerate(tickers, 1):
    p = HOURLY_DIR / t / "2026.json.gz"
    try:
        with gzip.open(p, "rt") as f:
            d = json.load(f)
        existing_bars = d.get("bars", [])
        existing_keys = {b["time_key"] for b in existing_bars}
        
        ret, df, _ = ctx.request_history_kline(
            code=f"US.{t}",
            start="2026-09-18",
            end="2026-09-24",
            ktype=futu.KLType.K_60M,
            autype=futu.AuType.QFQ,
            fields=futu.KL_FIELD.ALL,
            max_count=1000,
            extended_time=True,
            session=futu.Session.ALL
        )
        
        if ret == futu.RET_OK:
            records = df.to_dict(orient="records")
            added = 0
            for r in records:
                clean = {k: (None if str(v) == "nan" else v) for k, v in r.items()}
                if clean["time_key"] not in existing_keys:
                    existing_bars.append(clean)
                    existing_keys.add(clean["time_key"])
                    added += 1
            existing_bars.sort(key=lambda x: x["time_key"])
            d["end"] = "2026-09-24"
            with gzip.open(p, "wt") as f_out:
                json.dump(d, f_out)
            added_bars_total += added
            success_count += 1
            if added > 0:
                print(f"[{i}/{len(tickers)}] {t}: +{added} bars (latest: {existing_bars[-1]['time_key']})")
        else:
            print(f"[{i}/{len(tickers)}] {t}: Futu error: {df}")
    except Exception as exc:
        print(f"[{i}/{len(tickers)}] {t}: Exception: {exc}")
    
    time.sleep(0.4)

ctx.close()
print(f"Sync complete! {success_count}/{len(tickers)} updated. Total new bars: {added_bars_total}")
