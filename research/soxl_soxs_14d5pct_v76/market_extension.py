"""Extend the existing market with audited earlier raw paths; never replace overlap."""
from pathlib import Path
import hashlib, json, sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
CACHE=ROOT/'.cache/soxl_soxs_14d5pct_v76'
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'research/soxl_soxs_14d5pct_v71'))
import transfer_learning as old
from scripts.model_history_calendar import calendar
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())


def load(for_audit=False):
    audit=read(CACHE/'data_audit.json')
    assert audit['structure_passed'] and audit['actions_passed']
    if not for_audit:
        acceptance=read(CACHE/'acceptance.json')
        assert acceptance['accepted_for_causal_research']
        assert acceptance['data_audit_sha256']==sha(CACHE/'data_audit.json')
    base=old.load_market()
    cal={pd.Timestamp(r['session_date']):r for r in calendar('2023-04-03','2026-12-31')['sessions']}
    for d,row in base.calendar.items():assert cal[d]==row
    sessions=pd.DatetimeIndex(sorted(cal)); bars={};daily={};actions={};complete={};manifest={}
    for receipt in audit['symbols']:
        s=receipt['symbol'];path=ROOT/receipt['path'];assert sha(path)==receipt['sha256']
        parent_root='soxl_soxs_14d5pct_v2' if s in ['SOXL','SOXS','SOXX','QQQ'] else 'soxl_soxs_14d5pct_v70'
        assert sha(ROOT/'.cache'/parent_root/'data/bars'/f'{s}.parquet')==receipt['overlap']['old_sha256']
        actionpath=CACHE/'ibkr'/f'{s}.json';assert sha(actionpath)==receipt['actions']['ibkr_sha256']
        earlier=pd.read_parquet(path)
        earlier=earlier[earlier.session_date<='2024-10-11'].copy()
        frame=pd.concat([earlier,base.bars[s].reset_index(drop=True)],ignore_index=True)
        for c in ['start_at','end_at','available_at']:frame[c]=pd.to_datetime(frame[c],utc=True).dt.as_unit('ns')
        frame=frame.sort_values('end_at');assert frame.end_at.is_unique
        frame['date']=pd.to_datetime(frame.session_date)
        old_actions=base.actions[s]
        # Existing action metadata remains authoritative on the old interval.
        early_actions=[a for a in read(actionpath)['corp_actions'] if pd.Timestamp('2023-04-03')<=pd.Timestamp(a['date'])<=pd.Timestamp('2024-10-11')]
        actions[s]=early_actions+[a for a in old_actions if pd.Timestamp(a['date'])>pd.Timestamp('2024-10-11')]
        factor=np.ones(len(frame))
        for a in actions[s]:
            if a['type']=='Splits':factor[frame.date>=pd.Timestamp(a['date'])]*=float(a['value'])
        frame['share_factor']=factor
        for c in ['open','high','low','close']:frame['econ_'+c]=frame[c]*factor
        frame['econ_volume']=frame.volume/factor
        opening=pd.to_datetime(frame.date.map({d:r['open_at'] for d,r in cal.items()}),utc=True)
        closing=pd.to_datetime(frame.date.map({d:r['close_at'] for d,r in cal.items()}),utc=True)
        frame.loc[frame.start_at.ge(opening)&frame.end_at.le(closing),'session_type']='regular'
        frame.loc[frame.start_at.ge(closing)&frame.session_type.ne('overnight'),'session_type']='post_market'
        frame=frame.set_index('end_at',drop=False);bars[s]=frame
        reg=frame[frame.session_type=='regular'];checks={}
        for day,g in reg.groupby('date'):
            expected=pd.date_range(cal[day]['open_at'],cal[day]['close_at'],freq='5min',inclusive='right')
            checks[day]=g.index.equals(expected)
        complete[s]=checks
        d=reg.groupby('date').agg(open=('econ_open','first'),high=('econ_high','max'),low=('econ_low','min'),
            close=('econ_close','last'),volume=('econ_volume','sum'),raw_close=('close','last'),
            share_factor=('share_factor','last'),available_at=('available_at','max'))
        d['complete']=[checks[x] for x in d.index];daily[s]=d
        manifest[s]=dict(earlier_bars_sha256=receipt['sha256'],earlier_actions_sha256=sha(actionpath),
            existing=base.manifest[s],merge='old interval unchanged raw; earlier only through2024-10-11')
    return old.core.Market(bars,daily,cal,sessions,actions,complete,manifest)
