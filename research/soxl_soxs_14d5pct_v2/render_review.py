"""Generate an offline grayscale review with full minute data kept in .cache/."""
from __future__ import annotations

import argparse
import base64
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

import core


def epoch(value):
    return int(pd.Timestamp(value).value // 1_000_000)


def render(run, data_root):
    run = Path(run); market = core.load_market(data_root)
    results = json.loads((run / 'results.json').read_text())
    if not results['trend20_premarket']['primary_pass']:
        raise ValueError('This template requires the registered primary trend result')
    data = {'bars': {}, 'quotes': {}, 'daily': {}, 'calendar': {}, 'results': results}
    sessions = {'overnight': 0, 'regular': 1, 'pre_market': 2, 'post_market': 3}
    for symbol in core.SYMBOLS:
        frame = market.bars[symbol]
        data['quotes'][symbol] = np.column_stack([
            frame.end_at.astype('int64').to_numpy() // 1_000_000,
            frame.available_at.astype('int64').to_numpy() // 1_000_000]).tolist()
        if symbol in core.ETFS:
            data['bars'][symbol] = [
                [epoch(r.end_at), epoch(r.start_at), str(r.date.date()), sessions[r.session_type],
                 float(r.econ_open), float(r.econ_high), float(r.econ_low), float(r.econ_close),
                 float(r.volume), float(r.open), float(r.close), float(r.share_factor)]
                for r in frame.itertuples()]
            d = market.daily[symbol]
            data['daily'][symbol] = [[str(at.date()), float(r.open), float(r.high), float(r.low), float(r.close)]
                                     for at, r in d.iterrows()]
    d = market.daily['SOXX']
    data['sectorDaily'] = [[str(at.date()), float(r.open), float(r.high), float(r.low), float(r.close), epoch(r.available_at)]
                            for at, r in d.iterrows()]
    data['calendar'] = {str(day.date()): {'open': epoch(v['open_at']), 'close': epoch(v['close_at'])}
                        for day, v in market.calendar.items()}
    trades = pd.read_csv(run / 'trend20_premarket_trades.csv')
    features = pd.read_parquet(run / 'features.parquet')
    records = []
    for _, row in trades.iterrows():
        r = {k: (None if pd.isna(v) else v) for k, v in row.to_dict().items()}
        r['entry_ms'] = epoch(row.entry_at); r['expiry_ms'] = epoch(row.expiry_at)
        r['exit_ms'] = epoch(row.exit_at) if pd.notna(row.exit_at) else None
        r['pending_at_end'] = pd.isna(row.exit_at) or pd.Timestamp(row.exit_at) > pd.Timestamp('2026-09-30T20:00:00Z')
        r['SOXX_ret20'] = float(features.loc[pd.Timestamp(row.decision_at), 'SOXX_ret20'])
        records.append(r)
    data['trades'] = records
    data['trade_csv'] = (run / 'trend20_premarket_trades.csv').read_text()
    equity = pd.read_csv(run / 'trend20_premarket_equity.csv')
    data['equity'] = [[str(r.date), float(r.equity)] for r in equity.itertuples()]
    payload = base64.b64encode(gzip.compress(json.dumps(data, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode(), mtime=0)).decode()
    template = Path(__file__).with_name('review_template.html').read_text()
    output = run / 'review.html'; output.write_text(template.replace('__PAYLOAD__', payload))
    print(json.dumps({'output': str(output.resolve()), 'bytes': output.stat().st_size,
                      'minute_bars_embedded': sum(len(x) for x in data['bars'].values()), 'trades': len(records)}))
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, default=core.DATA_ROOT)
    args = parser.parse_args(); render(args.run, args.data_root)
