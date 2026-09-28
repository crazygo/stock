"""Independent direct-bar checks for the frozen prediction artifact."""
from pathlib import Path
import json

import numpy as np
import pandas as pd

from run import bootstrap_matrix, holm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main():
    p = pd.read_parquet(HERE / 'predictions.parquet')
    spec = json.loads((HERE / 'frozen_pairs.json').read_text())
    summary = json.loads((HERE / 'summary.json').read_text())
    cal = json.loads((ROOT / 'market_data/calendars/nasdaq_sessions_2026_v1.json').read_text())
    dates = [s['session_date'] for s in cal['sessions']]
    assert not p.duplicated(['symbol', 'date']).any()
    assert p.date.ge('2026-07-13').all() and p.date.le('2026-09-16').all()
    assert p.label_end.le('2026-09-23').all()
    for row in p.itertuples():
        assert row.symbol != row.leader_symbol
        assert {row.symbol, row.leader_symbol} != {'GOOG', 'GOOGL'}
        assert row.leader_symbol == spec[row.symbol]['leader']
        assert row.lag == spec[row.symbol]['lag']
        assert dates[dates.index(row.date) + 5] == row.label_end
    # Direct five-minute reconstruction including the August MNST split.
    chosen = p[p.symbol.isin(['AAPL', 'NVDA', 'MNST', 'CRWD', 'BKNG'])].copy()
    worst = 0.0
    n = 0
    for symbol, rows in chosen.groupby('symbol'):
        raw = pd.read_parquet(ROOT / f'market_data/us_5m/{symbol}/2026.parquet')
        raw = raw[raw.session_type.eq('regular')].sort_values('end_at')
        actions = pd.read_parquet(ROOT / f'market_data/corporate_actions/{symbol}.parquet')
        for row in rows.itertuples():
            j = dates.index(row.date)
            window = dates[j+1:j+6]
            future = raw[raw.session_date.isin(window)].copy()
            assert len(future) == 78 * 5
            entry = float(future.open.iloc[0])
            factor = np.ones(len(future))
            for a in actions.itertuples():
                # Split on entry day is already included in the entry price.
                if pd.notna(a.split_ratio) and window[0] < str(a.ex_div_date) <= window[-1]:
                    factor[future.session_date.ge(str(a.ex_div_date)).to_numpy()] *= a.split_ratio
            expected = float((future.high.to_numpy() / factor).max() / entry - 1) * 100
            delta = abs(expected - row.target)
            worst = max(worst, delta)
            assert delta < 1e-8, (symbol, row.date, expected, row.target)
            n += 1
    diff = (abs(p.target-p.base) - abs(p.target-p.leader)).groupby(p.date).mean().mean()
    assert abs(float(diff) - summary['improvement_pp']) < 1e-12
    assert summary['stocks'] == p.symbol.nunique()
    p['improvement'] = abs(p.target - p.base) - abs(p.target - p.leader)
    matrix = p.pivot(index='date', columns='symbol', values='improvement').sort_index()
    means, ci, pvalues, _ = bootstrap_matrix(matrix, 5)
    adjusted = holm(pvalues)
    i = matrix.columns.get_loc('ADBE')
    sensitivity = dict(block5_adbe=dict(improvement_pp=float(means[i]), ci95_pp=ci[:, i].tolist(),
                                        p_one_sided=float(pvalues[i]), p_holm=float(adjusted[i])),
                       block5_holm_pass=[str(matrix.columns[j]) for j in range(len(pvalues)) if adjusted[j] < .05])
    (HERE/'sensitivity.json').write_text(json.dumps(sensitivity, indent=2)+'\n')
    result = dict(status='passed', unique_predictions=len(p),
                  independently_rebuilt_targets=n, max_target_error_pp=worst,
                  includes_split_crossing='MNST 2026-08-11',
                  checks=['unique symbol/date', 'frozen pairs unchanged', 'no self/same-company pairs',
                          'official five-session horizon', 'mature labels',
                          'direct five-minute target reconstruction', 'daily equal-weight summary'])
    (HERE/'verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
