"""Reconstruct template arrays and release times directly from immutable market bars."""
import argparse
from path_learning import *


def audit(run):
    run=Path(run);m=read(run/'input_manifest.json')
    for f,h in m['derived_files_sha256'].items():assert sha(run/f)==h
    for f,h in m['original_inputs_sha256'].items():assert sha(run/f)==h
    market=core.load_market();assert market.manifest==m['full_market'];market=restrict_market(market,pd.Timestamp(m['observation_cutoff_exclusive']))
    panel=pd.read_parquet(run/'activity_panel.parquet');bank=dict(np.load(run/'path_bank.npz'));records=read(run/'bank_records.json');sessions=list(market.sessions);good=[];checks=0;corp=0
    days=[d for d in sessions if panel.date.min()<=d<=panel.date.max()];assert len(days)==len(records)
    for i,(d,r) in enumerate(zip(days,records)):
        assert pd.Timestamp(r['date'])==d and bank['date_ns'][i]==d.value and r['index']==i
        prev=sessions[sessions.index(d)-1];assert pd.Timestamp(r['previous_date'])==prev
        expected=pd.date_range(market.calendar[d]['open_at'],market.calendar[d]['close_at'],freq='5min',inclusive='right');valid=len(expected)==78;times=[]
        for s in core.ETFS:
            b=market.bars[s];g=b[(b.date==d)&(b.session_type=='regular')];daily=market.daily[s]
            valid=valid and g.index.equals(expected) and g.available_at.notna().all() and g.available_at.ge(g.end_at).all() and g.start_at.eq(g.end_at-pd.Timedelta(minutes=5)).all() and prev in daily.index and bool(daily.loc[prev,'complete'])
            if len(g):
                a=g[['econ_open','econ_high','econ_low','econ_close']].to_numpy();valid=valid and np.isfinite(a).all() and (a>0).all() and (a[:,1]>=np.maximum(a[:,0],a[:,3])).all() and (a[:,2]<=np.minimum(a[:,0],a[:,3])).all()
            for ev in market.actions[s]:
                if ev['type']=='CashDividends' and pd.Timestamp(ev['date'])==d:
                    valid=valid and pd.notna(pd.to_datetime(ev.get('payment_date'),errors='coerce')) and np.isfinite(float(ev['value'])) and float(ev['value'])>=0
        assert r['complete']==bool(valid)
        if not valid:
            assert np.isnan(bank['ohlc'][i]).all() and bank['release_ns'][i]==pd.NaT.value
            continue
        good.append(i)
        for sidx,s in enumerate(core.ETFS):
            b=market.bars[s].loc[expected];pc=market.daily[s].loc[prev];factor=b.share_factor.iloc[0];assert b.share_factor.eq(factor).all()
            np.testing.assert_allclose(bank['ohlc'][i,:,sidx],b[['open','high','low','close']].to_numpy()*factor,rtol=1e-12)
            assert bank['previous_close'][i,sidx]==pc.raw_close*pc.share_factor
            assert bank['split_ratio'][i,sidx]==factor/pc.share_factor
            cash=0.;times.extend([b.available_at.max(),pc.available_at])
            for ev in market.actions[s]:
                if ev['type']=='CashDividends' and pd.Timestamp(ev['date'])==d:
                    cash+=float(ev['value'])*factor;times.append(pd.Timestamp(pd.Timestamp(ev['payment_date']).date(),tz=core.ET).tz_convert('UTC')+pd.Timedelta(days=1));corp+=1
            np.testing.assert_allclose(bank['dividend_ratio'][i,sidx],cash/pc.close,rtol=1e-12)
            checks+=4
        for c,clock in enumerate(CLOCKS):
            at=pd.Timestamp(str(d.date())+' '+clock,tz=core.ET).tz_convert('UTC');end=at-pd.Timedelta(seconds=30);expectedquote=[]
            for s in core.ETFS:
                b=market.bars[s]
                if end not in b.index:break
                z=b.loc[end]
                if not (pd.notna(z.available_at) and end<=z.available_at<=at and z.start_at==end-pd.Timedelta(minutes=5) and np.isfinite(z.close) and z.close>0):break
                expectedquote.append(z)
            if len(expectedquote)==2:
                np.testing.assert_allclose(bank['quotes'][i,c],[z.close*z.share_factor for z in expectedquote],rtol=1e-12);times.extend(z.available_at for z in expectedquote)
                assert len(r['quotes'][c])==2
                for j,z in enumerate(expectedquote):
                    rec=r['quotes'][c][j];assert pd.Timestamp(rec['at'])==end and pd.Timestamp(rec['available_at'])==z.available_at and rec['raw_close']==z.close and rec['economic_close']==z.econ_close
            else:assert np.isnan(bank['quotes'][i,c]).all() and r['quotes'][c] is None
            want=panel.loc[at,FIELDS].to_numpy(float) if at in panel.index else np.full(4,np.nan);np.testing.assert_allclose(bank['states'][i,c],want,equal_nan=True);checks+=2
        assert bank['release_ns'][i]==max(times).value==pd.Timestamp(r['release']).value
    pairs=[[i,i+1] for i in good if i+1 in good and sessions[sessions.index(days[i])+1]==days[i+1]]
    np.testing.assert_array_equal(bank['blocks'],pairs)
    features=pd.read_parquet(run/'features.parquet');successors=0
    for i,(_,last) in enumerate(pairs):
        day=days[last];position=sessions.index(day)+1
        if position>=len(sessions):
            assert bank['successor_at_ns'][i]==pd.NaT.value and np.isnan(bank['successor_states'][i]).all();continue
        at=pd.Timestamp(str(sessions[position].date())+' '+CLOCKS[0],tz=core.ET).tz_convert('UTC')
        assert bank['successor_at_ns'][i]==at.value
        want=panel.loc[at,PRIOR].to_numpy(float) if at in panel.index else np.full(3,np.nan)
        np.testing.assert_array_equal(bank['successor_states'][i],want)
        if np.isfinite(want).all():
            f=features.loc[at];sv=max(f.SOXX_vol20,.0001)
            raw=[np.clip(f.sector_relative_5/(sv*np.sqrt(5)),-5,5),2*f.breadth_ret5_positive_fraction-1,np.log(sv)]
            np.testing.assert_allclose(want,raw,rtol=1e-11,atol=1e-11);successors+=1
    assert bank['successor_states'].shape==(len(pairs),3) and bank['successor_at_ns'].shape==(len(pairs),)
    out={'passed':True,'days':len(days),'complete_full_days':len(good),'two_session_blocks':len(pairs),'array_checks':checks,'complete_successors':successors,'successor_records':len(pairs),'cash_events':corp,'input_manifest_sha256':sha(run/'input_manifest.json'),'new_capital_outcomes_seen':False}
    dump(run/'bank_audit.json',out);print(out,flush=True);return out

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);audit(p.parse_args().inputs)
