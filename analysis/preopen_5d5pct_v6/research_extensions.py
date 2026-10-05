"""Causal R02 input additions; exact clock differences and original SEC events."""
import json
import numpy as np
import pandas as pd
from common import OUT,OLD,DATES,SESSIONS
from prepare import extend_peers

def peers_exact(panel):
    panel=extend_peers(panel.drop(columns=[c for c in panel if c.startswith('b_')],errors='ignore'))
    from feature_primitives import GROUP_OF
    panel['_group']=panel.symbol.map(GROUP_OF).fillna('unmapped')
    keys=['day','minute','_group']
    aggregates=panel.groupby(keys,observed=True).agg(_positive=('b_breadth_i30_ret','first'),_lead_return=('i30_ret','max')).reset_index()
    leaders=panel.sort_values(keys+['i30_ret','symbol'],ascending=[True,True,True,False,True],na_position='last').drop_duplicates(keys)[keys+['symbol']].rename(columns={'symbol':'_leader'})
    leaders.loc[panel.sort_values(keys+['i30_ret','symbol'],ascending=[True,True,True,False,True],na_position='last').drop_duplicates(keys).i30_ret.isna().to_numpy(),'_leader']=None
    aggregates=aggregates.merge(leaders,on=keys,validate='one_to_one')
    previous=aggregates.rename(columns={'_positive':'_positive_prior','_lead_return':'_lead_prior','_leader':'_leader_prior'}).copy();previous['minute']+=30
    aggregates=aggregates.merge(previous,on=keys,how='left',validate='one_to_one')
    aggregates['b_positive_change_30']=aggregates._positive-aggregates._positive_prior
    aggregates['b_leader_change_30']=aggregates._lead_return-aggregates._lead_prior
    aggregates['b_leader_persist_30']=(aggregates._leader==aggregates._leader_prior).astype(float).where(aggregates._leader.notna()&aggregates._leader_prior.notna())
    cols=['b_positive_change_30','b_leader_change_30','b_leader_persist_30']
    panel=panel.drop(columns=cols).merge(aggregates[keys+cols],on=keys,how='left',validate='many_to_one',sort=False)
    panel.loc[panel._group=='unmapped',cols]=np.nan
    return panel.drop(columns='_group')

def event_features(panel,latency_seconds=60,events_path=None,audit_path=None):
    events=json.loads((events_path or OUT/'cache/sec/events.json').read_text())['events'];audit=json.loads((audit_path or OUT/'sec_audit.json').read_text())
    ev=pd.DataFrame(events);valid_ts=ev.accepted_at.str.endswith('Z')
    ev=ev[valid_ts].copy();ev['available']=pd.to_datetime(ev.accepted_at,utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None)+pd.Timedelta(seconds=latency_seconds)
    ev['earnings']=(ev.form.eq('8-K')&ev['items'].astype(str).str.contains('2.02',regex=False)).astype(float)
    ev['quarterly']=ev.form.isin(['10-Q','10-K','20-F','40-F']).astype(float)
    ev=ev[(ev.earnings>0)|(ev.quarterly>0)].drop_duplicates(['symbol','accession']).sort_values('available')
    covered={s['symbol'] for s in audit['symbols'] if s['status']=='ok'}
    cols=['e_filing_age_hours','e_earnings_filing','e_quarterly_filing','e_post_return','e_post_volume_ratio','e_anchor_missing','e_missing']
    panel=panel.drop(columns=[c for c in panel if c.startswith('e_')],errors='ignore')
    for c in cols:panel[c]=np.nan
    panel['e_missing']=1.;panel['sec_source_covered']=panel.symbol.isin(covered).astype(float)
    for symbol,indices in panel.groupby('symbol',sort=False,observed=True).groups.items():
        es=ev[ev.symbol==symbol]
        if es.empty:continue
        frame=panel.loc[indices];cut=pd.to_datetime(frame.day)+pd.to_timedelta(frame.minute,unit='m')+pd.Timedelta(seconds=30)
        pos=np.searchsorted(es.available.to_numpy(),cut.to_numpy(),side='right')-1
        have=pos>=0;ages=np.full(len(frame),np.nan);ages[have]=(cut.to_numpy()[have]-es.iloc[pos[have]].available.to_numpy())/np.timedelta64(1,'h')
        active=have&(ages<=168);ids=np.asarray(indices)[active]
        if not len(ids):continue
        selected=es.iloc[pos[active]]
        panel.loc[ids,'e_filing_age_hours']=ages[active];panel.loc[ids,'e_earnings_filing']=selected.earnings.to_numpy();panel.loc[ids,'e_quarterly_filing']=selected.quarterly.to_numpy();panel.loc[ids,'e_missing']=0.;panel.loc[ids,'e_anchor_missing']=1.
        raw=pd.read_parquet(OLD/'raw'/f'{symbol}.parquet').sort_values('available')
        good=np.isfinite(raw[['open','high','low','close','volume']]).all(axis=1)&(raw[['open','high','low','close']]>0).all(axis=1)&(raw.volume>=0)&raw.day.isin(SESSIONS)
        raw=raw[good].copy()
        # Expected volume is shifted by one official trading day, with missing
        # days left missing. Today never participates in its own denominator.
        byday=raw.pivot(index='day',columns='minute',values='volume').reindex(DATES)
        expected=byday.shift(1).rolling(20,min_periods=10).mean().stack().rename('_expected').reset_index()
        raw=raw.merge(expected,on=['day','minute'],how='left',validate='one_to_one').sort_values('available')
        actual_cum=np.r_[0.,raw.volume.cumsum().to_numpy()];expect_cum=np.r_[0.,raw._expected.fillna(0).cumsum().to_numpy()];missing_cum=np.r_[0,np.cumsum(raw._expected.isna().to_numpy())]
        times=raw.available.to_numpy();anchor=np.searchsorted(times,selected.available.to_numpy(),side='left')-1
        current=np.searchsorted(times,cut.to_numpy()[active],side='right')
        ok=anchor>=0;anchor=np.maximum(anchor,0);anchor_times=times[anchor]
        # Anchors older than 24h are uncertain, including weekend filings.
        ok &= (selected.available.to_numpy()-anchor_times)/np.timedelta64(1,'h')<=24
        anchor_prices=raw.close.to_numpy()[anchor];goodids=ids[ok]
        panel.loc[goodids,'e_post_return']=panel.loc[goodids,'reference'].to_numpy()/anchor_prices[ok]-1
        panel.loc[goodids,'e_anchor_missing']=0.
        begin=anchor+1;sumv=actual_cum[current]-actual_cum[begin];sume=expect_cum[current]-expect_cum[begin]
        vok=ok&(sume>0)&((missing_cum[current]-missing_cum[begin])==0)
        panel.loc[ids[vok],'e_post_volume_ratio']=sumv[vok]/sume[vok]
    return panel,dict(covered_symbols=len(covered),events=len(ev),eligible_rows=int((panel.e_missing==0).sum()),latency_seconds=latency_seconds,
        timestamp='original acceptance UTC + explicit delay assumption',scope='filing confirmation; no consensus or earliest announcement')

def repair_action_labels(panel):
    """New label version: known actions invalidate raw, unadjusted windows."""
    changed=0
    for symbol,indices in panel.groupby('symbol',sort=False,observed=True).groups.items():
        path=OUT/'cache/corporate_actions'/f'{symbol}.parquet'
        if not path.exists():continue
        actions=pd.read_parquet(path).ex_div_date.astype(str).str[:10]
        rows=panel.loc[indices];bad=np.zeros(len(rows),bool)
        for day in actions:bad|=((rows.day<=day)&(rows.label_end.str[:10]>=day)&(rows.label_end.str[:10]<='2026-09-30')&rows.entry.notna()&(rows.entry>0)).to_numpy()
        ids=np.asarray(indices)[bad];changed+=int(panel.loc[ids,'y'].notna().sum())
        panel.loc[ids,'y']=np.nan;panel.loc[ids,'future_status']='corporate_action_window'
    return panel,dict(known_rows_newly_unknown=changed,rule='fresh get_rehab actions invalidate unadjusted mature-window labels; all rows and signals retained')

def quarter_features(panel,source_path=None,latency_seconds=300):
    records=json.loads((source_path or OUT/'cache/sec_facts/events.json').read_text())['events']
    rows=[]
    for r in records:
        if not r['accepted_at'].endswith('Z'):continue
        accepted=pd.Timestamp(r['accepted_at']).tz_convert('America/New_York').tz_localize(None)
        if any(pd.Timestamp(v['end'])>accepted.normalize() for v in r['metrics'].values()):continue
        row=dict(symbol=r['symbol'],available=accepted+pd.Timedelta(seconds=latency_seconds))
        for metric,v in r['metrics'].items():
            row['f_'+metric+'_value']=v['value'];row['f_'+metric+'_yoy']=v.get('yoy_change',np.nan)
        income=r['metrics'].get('income',{});revenue=r['metrics'].get('revenue',{})
        row['f_profit_margin']=income.get('value',np.nan)/revenue['value'] if revenue.get('value',0)>0 else np.nan
        row['f_turn_to_profit']=float(income['value']>0 and income['prior_value']<=0) if 'prior_value'in income else np.nan
        row['f_dual_yoy_growth']=float(revenue['yoy_change']>0 and income['yoy_change']>0) if revenue.get('yoy_change')is not None and income.get('yoy_change')is not None else np.nan
        rows.append(row)
    facts=pd.DataFrame(rows).sort_values('available');cols=[c for c in facts if c.startswith('f_')]
    for c in cols+['f_age_days']:panel[c]=np.nan
    panel['f_missing']=1.
    for symbol,indices in panel.groupby('symbol',observed=True,sort=False).groups.items():
        es=facts[facts.symbol==symbol]
        if es.empty:continue
        frame=panel.loc[indices];cut=pd.to_datetime(frame.day)+pd.to_timedelta(frame.minute,unit='m')+pd.Timedelta(seconds=30)
        pos=np.searchsorted(es.available.to_numpy(),cut.to_numpy(),side='right')-1;have=pos>=0;age=np.full(len(frame),np.nan)
        age[have]=(cut.to_numpy()[have]-es.iloc[pos[have]].available.to_numpy())/np.timedelta64(1,'D')
        active=have&(age<=120);ids=np.asarray(indices)[active]
        if not len(ids):continue
        panel.loc[ids,cols]=es.iloc[pos[active]][cols].to_numpy();panel.loc[ids,'f_age_days']=age[active];panel.loc[ids,'f_missing']=0.
    return panel,dict(events=len(rows),stock_sources=facts.symbol.nunique(),eligible_rows=int((panel.f_missing==0).sum()),latency_seconds=latency_seconds,fields=cols,known_not_consensus=True)
