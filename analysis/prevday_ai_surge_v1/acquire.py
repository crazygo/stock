"""Refresh only the tail in an isolated cache; never push or alter shared bars."""
import json
from pathlib import Path
import shutil
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from scripts.fetch_research_data import ResearchDataFetcher


def universe():
    source = json.loads((ROOT / 'ai_universe_candidates.json').read_text())
    wanted = {}
    for group, spec in source['groups'].items():
        for symbol in spec['tickers']:
            wanted.setdefault(symbol, group)
    wanted.update(dict.fromkeys(['ALAB', 'CRDO', 'SNDK'], 'compute_and_network_hardware'))
    wanted.update(dict.fromkeys(['ARM', 'ASML'], 'semiconductors_and_equipment'))
    wanted.update(dict.fromkeys(['CRWV', 'NBIS'], 'cloud_and_platform'))
    present = {s: g for s, g in wanted.items() if (ROOT / f'market_data/us_5m/{s}/2026.parquet').exists()}
    return present, sorted(set(wanted) - set(present))


def main():
    present, missing = universe()
    (HERE / 'universe.json').write_text(json.dumps({'members': present, 'missing_5m': missing, 'mode': 'current_theme_snapshot_replay'}, ensure_ascii=False, indent=2))
    cache = HERE / 'data_cache'
    fetcher = ResearchDataFetcher(cache)
    try:
        for i, symbol in enumerate([*sorted(present), 'QQQ']):
            target = cache / f'us_5m/{symbol}/2026.parquet'
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT / f'market_data/us_5m/{symbol}/2026.parquet', target)
            print(f'[{i+1}/{len(present)+1}] {symbol}', flush=True)
            fetcher.fetch_kline(symbol, '2026-09-24', '2026-09-25')
    finally:
        fetcher.close()


if __name__ == '__main__':
    main()
