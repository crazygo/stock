"""Meaningful financial-data audits, including future truncation invariance."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_panel import summarize, DATES
from evaluate import apply_rule

HERE=Path(__file__).resolve().parent


def main():
    checks=[]
    for symbol in ['AMD','ALAB','MU']:
        raw=pd.read_parquet(HERE/f'data_cache/us_5m/{symbol}/2026.parquet')
        full,_=summarize(symbol)
        cut,_=summarize(symbol,raw[raw.session_date.le('2026-09-10')])
        cols=[c for c in full if c!='max_available_at']
        pd.testing.assert_frame_equal(full.loc[full.date.le('2026-09-10'),cols].reset_index(drop=True),
                                      cut.loc[cut.date.le('2026-09-10'),cols].reset_index(drop=True),check_dtype=False)
        checks.append(f'{symbol}: removing all future bars leaves all prefix features unchanged')
    panel=pd.read_parquet(HERE/'panel.parquet')
    lookup=dict(zip(DATES[1:],DATES[:-1]))
    assert all(r.prev_date==lookup[r.date] for r in panel.itertuples() if r.date in lookup)
    cutoff=pd.to_datetime(panel.prev_feature_cutoff,utc=True)
    available=pd.to_datetime(panel.prev_max_available_at,utc=True)
    keep=available.notna()
    assert (available[keep] <= cutoff[keep]).all()
    checks.append('Every T-1 is the official previous trading session, and feature availability precedes cutoff')
    valid=panel[panel.valid_label]
    assert valid[['close_return','high_return','gap','open_close','open_high','open_low']].notna().all().all()
    np.testing.assert_allclose((1+valid.gap)*(1+valid.open_close)-1,valid.close_return,atol=1e-12)
    assert (valid.close8.eq(valid.close_return.ge(.08))).all()
    assert (valid.high10.eq(valid.high_return.ge(.10))).all()
    checks.append('All labels reconcile to raw price returns; gap and open-close compound to close-close')
    rules=json.loads((HERE/'frozen_rules.json').read_text())['rules']
    val=json.loads((HERE/'validation.json').read_text())
    w=val['windows']
    names=['discovery_5','prior_5','prior_15','prior_40']
    for i,a in enumerate(names):
        for b in names[i+1:]:assert not(set(w[a])&set(w[b]))
    for r in val['results']:
        d=valid[valid.date.isin(w[r['window']])]
        rule=next(x for x in rules if x['id']==r['rule'])
        available,hit=apply_rule(d,rule)
        assert hit.sum()==r['n'] and int(d.loc[hit,r['target']].sum())==r['k']
        assert available.sum()==r['available']
    checks.append('All reported rule counts reconcile and the four displayed time windows are disjoint')
    audit={'passed':True,'checks':checks,'eligible_stock_days':len(valid),
           'discovery_eligible':int(valid.date.isin(w['discovery_5']).sum()),
           'prior60_eligible':int(valid.date.isin(w['prior_60']).sum())}
    (HERE/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2))
    print(json.dumps(audit,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
