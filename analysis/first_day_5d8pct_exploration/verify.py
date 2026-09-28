"""Temporal and raw-formula checks for the expanded feature research."""
from pathlib import Path
import json

import numpy as np
import pandas as pd

from build_expanded import build_symbol, EXTRA
from build_dataset import corporate_dates

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
RUN=HERE/'expanded_v2'
SOURCE=ROOT/'research/group_expectation_matrix/outputs/20260925_v5'


def main():
    data=pd.read_parquet(RUN/'features.parquet')
    defs=json.loads((RUN/'feature_definitions.json').read_text())
    assert len(defs)==51
    assert not data.duplicated(['sample_id','checkpoint']).any()
    assert data.groupby('date').split.nunique().max()==1
    for split,boundary in [('discovery','2026-06-01 09:30'),('selection','2026-07-20 09:30')]:
        assert pd.to_datetime(data.loc[data.split.eq(split),'label_available_at'],utc=True).max()<pd.Timestamp(boundary,tz='America/New_York')
    assert (data.loc[data.already_hit,'target']==1).all()
    assert data.loc[data.checkpoint.eq('11:30'),'progress'].isna().all()
    assert (~data.loc[data.checkpoint.eq('11:30'),'already_hit']).all()
    assert (data.loc[~data.already_hit,'max_progress'].dropna()<.08+1e-12).all()
    assert data.groupby(['sample_id']).target.nunique().max()==1
    raw_checks=0
    sampled=data.groupby('checkpoint',group_keys=False).sample(12,random_state=20260926)
    for symbol,part in sampled.groupby('symbol'):
        raw=pd.read_parquet(ROOT/'market_data/us_5m'/symbol/'2026.parquet')
        raw['end']=pd.to_datetime(raw.end_at,utc=True)
        raw['start']=pd.to_datetime(raw.start_at,utc=True)
        raw['available']=pd.to_datetime(raw.available_at,utc=True)
        for row in part.itertuples():
            limit=pd.Timestamp(row.feature_available_at)
            p=raw[raw.session_date.eq(row.date)&raw.session_type.eq('regular')&raw.available.le(limit)].sort_values('start')
            assert (p.end<=limit-pd.Timedelta(seconds=1)).all()
            assert np.isclose(row.session_return,p.iloc[-1].close/p.iloc[0].open-1)
            assert np.isclose(row.open60_return,p.iloc[11].close/p.iloc[0].open-1)
            assert np.isclose(row.up_bar_fraction,float((p.close>p.open).mean()))
            if row.checkpoint!='11:30':
                post=p[p.start.ge(pd.Timestamp(row.entry_at))]
                assert np.isclose(row.progress,p.iloc[-1].close/row.entry_price-1)
                assert np.isclose(row.max_progress,max(0,post.high.max()/row.entry_price-1))
                assert bool(post.high.ge(row.entry_price*1.08).any())==row.already_hit
            pre=raw[raw.session_date.eq(row.date)&raw.session_type.eq('pre_market')].sort_values('start')
            assert len(pre)==66
            assert np.isclose(row.premarket_last60,pre.iloc[-1].close/pre.iloc[-12].open-1)
            assert np.isclose(row.premarket_return,pre.iloc[-1].close/pre.iloc[0].open-1)
            raw_checks+=1
    sessions=json.loads((ROOT/'market_data/calendars/nasdaq_sessions_2026_v1.json').read_text())['sessions']
    future_checks=0
    for symbol in ('NVDA','AAPL','QQQ'):
        raw=pd.read_parquet(ROOT/'market_data/us_5m'/symbol/'2026.parquet')
        actions=corporate_dates(pd.read_parquet(ROOT/'market_data/corporate_actions'/f'{symbol}.parquet'))
        original=pd.DataFrame(build_symbol(raw,sessions,actions)).set_index('date')
        day='2026-07-20'
        future=raw.copy()
        mask=pd.to_datetime(future.start_at,utc=True).ge(pd.Timestamp(day+' 09:30',tz='America/New_York'))
        future.loc[mask,['open','high','low','close']]*=7
        future.loc[mask,['volume','turnover']]*=11
        changed=pd.DataFrame(build_symbol(future,sessions,actions)).set_index('date')
        fields=list(EXTRA)
        pd.testing.assert_frame_equal(original.loc[:day,fields],changed.loc[:day,fields])
        # Removing every later bar must likewise leave current/past context intact.
        truncated=pd.DataFrame(build_symbol(raw.loc[~mask],sessions,actions)).set_index('date')
        pd.testing.assert_frame_equal(original.loc[:day,fields],truncated.loc[:day,fields])
        future_checks+=2
    result={'passed':True,'raw_prefix_formula_checks':raw_checks,'expanded_future_mutation_or_deletion_checks':future_checks,
            'split_and_target_invariants':'passed','features':len(defs),
            'scope':'Current and past features only; no claim of external predictive validation'}
    (RUN/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
