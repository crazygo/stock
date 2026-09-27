#!/usr/bin/env python3
"""Build universe membership and benchmark mapping for research v2.

Formalizes:
- Universe role: 'candidate' vs 'benchmark'
- Effective interval: 2026-01-01 to 2026-09-24 (and forward)
- Mode: 'current_universe_retrospective' (as required by docs/02_data_time.md)
- Industry/Sector mapping proxies (SOXX, IGV, XLU, SMH, etc.)
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

# Industry / Theme proxy ETF mappings
INDUSTRY_PROXIES = {
    # Semiconductor / Hardware
    "NVDA": "SOXX", "AVGO": "SOXX", "AMD": "SOXX", "QCOM": "SOXX", "TXN": "SOXX",
    "INTC": "SOXX", "MU": "SOXX", "AMAT": "SOXX", "LRCX": "SOXX", "KLAC": "SOXX",
    "MRVL": "SOXX", "ADI": "SOXX", "MCHP": "SOXX", "NXPI": "SOXX", "MPWR": "SOXX",
    "ARM": "SOXX", "ASML": "SOXX", "TER": "SOXX", "ALAB": "SOXX", "CRDO": "SOXX",
    # Software / Cloud / Cybersecurity
    "MSFT": "IGV", "ADBE": "IGV", "CRM": "IGV", "INTU": "IGV", "PANW": "IGV",
    "CRWD": "IGV", "FTNT": "IGV", "DDOG": "IGV", "WDAY": "IGV", "ADSK": "IGV",
    "SNPS": "IGV", "CDNS": "IGV", "PLTR": "IGV", "SHOP": "IGV", "APP": "IGV",
    # Utilities / Clean Energy
    "CEG": "XLU", "AEP": "XLU", "EXC": "XLU", "XEL": "XLU",
    # Broad tech / internet defaults to QQQ
}


def build_universe():
    root = Path(__file__).resolve().parent.parent
    src_file = root / "analysis" / "qqq_constituents.json"
    with open(src_file, "r", encoding="utf-8") as f:
        src_data = json.load(f)

    raw_tickers = src_data.get("tickers", [])

    BENCHMARKS = {"QQQ", "SPY", "DIA", "SOXX", "SMH", "IGV", "XLU"}

    members = []
    for ticker in raw_tickers:
        role = "benchmark" if ticker in BENCHMARKS else "candidate"
        ind_proxy = INDUSTRY_PROXIES.get(ticker, "QQQ" if role == "candidate" else None)

        members.append({
            "security_id": f"US.{ticker}",
            "symbol": ticker,
            "role": role,
            "universe_id": "qqq_nasdaq100_retrospective_v1",
            "universe_mode": "current_universe_retrospective",
            "effective_from": "2026-01-01T00:00:00Z",
            "effective_to": "2026-12-31T23:59:59Z",
            "announced_at": None,
            "available_at": "2026-09-24T00:00:00Z",
            "first_seen_at": "2026-09-24T00:00:00Z",
            "source_url": "https://www.slickcharts.com/nasdaq100",
            "industry_proxy": ind_proxy
        })

    # Ensure auxiliary benchmark ETFs are included if not present
    existing_symbols = {m["symbol"] for m in members}
    for bmk in ["SOXX", "SMH", "IGV", "XLU"]:
        if bmk not in existing_symbols:
            members.append({
                "security_id": f"US.{bmk}",
                "symbol": bmk,
                "role": "benchmark",
                "universe_id": "qqq_nasdaq100_retrospective_v1",
                "universe_mode": "benchmark_auxiliary",
                "effective_from": "2026-01-01T00:00:00Z",
                "effective_to": "2026-12-31T23:59:59Z",
                "announced_at": None,
                "available_at": "2026-09-24T00:00:00Z",
                "first_seen_at": "2026-09-24T00:00:00Z",
                "source_url": "etf_benchmarks",
                "industry_proxy": None
            })

    # Sort deterministically
    members.sort(key=lambda m: (0 if m["role"] == "benchmark" else 1, m["symbol"]))

    payload = {
        "universe_id": "qqq_nasdaq100_retrospective_v1",
        "universe_mode": "current_universe_retrospective",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_members": len(members),
        "candidates_count": sum(1 for m in members if m["role"] == "candidate"),
        "benchmarks_count": sum(1 for m in members if m["role"] == "benchmark"),
        "members": members
    }

    serialized = json.dumps(members, sort_keys=True)
    payload["universe_sha256"] = hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    target_dir = root / "market_data" / "research_v2" / "universe"
    target_dir.mkdir(parents=True, exist_ok=True)
    target_file = target_dir / "qqq_retrospective_v1.json"

    with open(target_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"Universe saved to {target_file}")
    print(f"Candidates: {payload['candidates_count']}, Benchmarks: {payload['benchmarks_count']}")
    print(f"Universe SHA256: {payload['universe_sha256']}")


if __name__ == "__main__":
    build_universe()
