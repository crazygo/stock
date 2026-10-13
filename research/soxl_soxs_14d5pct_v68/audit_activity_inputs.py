"""Reconstruct then-known relative sector activity from raw cached prefixes."""
import argparse
from path_learning import *
from audit_paths import reference_kernel


def audit(inputs):
    inputs=Path(inputs);manifest=read(inputs/'input_manifest.json');base=Path(manifest['base_inputs']);old=read(base/'input_manifest.json')
    assert sha(base/'input_manifest.json')==manifest['base_input_manifest_sha256']
    for f,h in manifest['derived_files_sha256'].items():assert sha(inputs/f)==h,f
    for f,h in old['derived_files_sha256'].items():assert sha(base/f)==h,f
    market=core.load_market();assert market.manifest==manifest['full_market'];cutoff=pd.Timestamp(manifest['observation_cutoff_exclusive']);market=restrict_market(market,cutoff)
    prefixes=pd.read_parquet(base/'activity_prefixes.parquet');panel=pd.read_parquet(inputs/'activity_panel.parquet');original=pd.read_parquet(base/'activity_panel.parquet');pd.testing.assert_frame_equal(panel,original)
    receipts=read(base/'activity_receipts.json');assert len(receipts)==len(panel);known={};prefix_checks=0
    for symbol in ['SOXL','SOXS','SOXX','QQQ']:
        bars=market.bars[symbol];times=bars.index.asi8
        for day in market.sessions:
            for clock in CLOCKS:
                at=pd.Timestamp(str(day.date())+' '+clock,tz=core.ET).tz_convert('UTC')
                if at>=cutoff or at>=pd.Timestamp(market.calendar[day]['close_at']):continue
                start=day.tz_localize(core.ET).tz_convert('UTC')+pd.Timedelta(hours=4);end=at-pd.Timedelta(seconds=30)
                ids=np.arange(times.searchsorted(start.value,side='right'),times.searchsorted(end.value,side='right'));g=bars.iloc[ids];grid=np.arange(start.value+300_000_000_000,end.value+1,300_000_000_000,dtype=np.int64)
                good=len(g)>0 and np.array_equal(times[ids],grid) and g.available_at.notna().all() and g.available_at.between(g.end_at,at).all() and g.start_at.eq(g.end_at-pd.Timedelta(minutes=5)).all() and g.volume.notna().all() and g.volume.ge(0).all()
                amount=float(np.dot(g.volume.to_numpy(float),g[['high','low','close']].to_numpy(float).mean(axis=1))) if good else np.nan
                if good:np.testing.assert_allclose(amount,np.dot(g.econ_volume.to_numpy(float),g[['econ_high','econ_low','econ_close']].to_numpy(float).mean(axis=1)),rtol=1e-12,atol=1e-7)
                good=bool(good and np.isfinite(amount) and amount>0);amount=amount if good else np.nan;r=prefixes.loc[(symbol,at)]
                assert r.complete==good and r.rows==len(g) and r.date==day and r.clock==clock
                np.testing.assert_allclose(r.activity,amount,rtol=1e-12,atol=1e-7,equal_nan=True)
                assert (pd.isna(r.last_available) and pd.isna(g.available_at.max())) or r.last_available==g.available_at.max()
                known[symbol,at]=(good,amount);prefix_checks+=1
    assert prefix_checks==len(prefixes)
    for (at,row),receipt in zip(panel.iterrows(),receipts):
        assert pd.Timestamp(receipt['at'])==at;eligible=[d for d in market.sessions if d<row.date and pd.Timestamp(str(d.date())+' '+row.clock,tz=core.ET)<pd.Timestamp(market.calendar[d]['close_at'])][-20:]
        past=[pd.Timestamp(str(d.date())+' '+row.clock,tz=core.ET).tz_convert('UTC') for d in eligible];values=[];ready=True
        for symbol in ['SOXL','SOXS','SOXX','QQQ']:
            okay=known[symbol,at][0] and len(past)==20 and all(known[symbol,t][0] for t in past);r=receipt['members'][symbol]
            assert list(pd.to_datetime(r['prior'],utc=True))==past and pd.Timestamp(r['current'])==at and r['complete']==okay
            if okay:
                average=sum(known[symbol,t][1] for t in past)/20;value=min(3.,max(-3.,float(np.log(known[symbol,at][1]/average))))
                np.testing.assert_allclose(r['prior_mean'],average,rtol=1e-12);np.testing.assert_allclose(r['log_relative'],value,rtol=1e-11,atol=1e-11)
            else:value=np.nan;assert r['prior_mean'] is None and r['log_relative'] is None
            values.append(value);ready=ready and okay
        expected=(values[2]-values[3])/2 if ready else np.nan
        assert row.activity_ready==ready;np.testing.assert_allclose(row[ACTIVITY],expected,equal_nan=True,rtol=1e-11,atol=1e-11)
    bank=dict(np.load(inputs/'path_bank.npz'));ess=[];removed=0;ready_missing=0
    for at,row in panel.iterrows():
        c=CLOCKS.index(row.clock);b=bank['blocks'];a=b[:,0];z=b[:,1];cut=day_cutoff(at).value
        good=(bank['release_ns'][a]<cut)&(bank['release_ns'][z]<cut)&(bank['date_ns'][z]<row.date.value)
        good&=np.isfinite(bank['quotes'][a,c]).all(axis=1)&np.isfinite(bank['states'][a,c]).all(axis=1)&np.isfinite(bank['states'][a,0,:3]).all(axis=1)
        good&=np.isfinite(bank['successor_states']).all(axis=1)&(bank['successor_at_ns']!=pd.NaT.value)&(bank['successor_at_ns']<cut)&np.isfinite(bank['absolute_states'][a,c])
        old_members=np.flatnonzero(good);members=source_pool(bank,at,row.clock);removed+=len(old_members)-len(members)
        if row.feature_ready and not np.isfinite(row[ACTIVITY]):ready_missing+=1
        ids=bank['blocks'][members,0]
        if len(members)>=120 and np.isfinite(row[FIELDS+[ABSOLUTE,ACTIVITY]].to_numpy(float)).all():
            x=np.column_stack([bank['states'][ids,c],bank['absolute_states'][ids,c],bank['activity_states'][ids,c]])
            k=reference_kernel(x,row[FIELDS+[ABSOLUTE,ACTIVITY]].to_numpy(float),True);ess.append(k['effective_dates'])
    if manifest['phase']=='development':assert removed==ready_missing==0 and min(ess)>=20
    out={'passed':True,'phase':manifest['phase'],'raw_prefix_checks':prefix_checks,'relative_activity_formula_checks':len(panel),'query_ready':int(panel.feature_ready.sum()),'finite_activity':int(panel[ACTIVITY].notna().sum()),'removed_source_members_summed_across_queries':removed,'ready_queries_missing_activity':ready_missing,'six_field_ess_range':[min(ess),max(ess)],'original_field_already_used_by_v56':True,'raw_economic_activity_equivalent':True,'input_manifest_sha256':sha(inputs/'input_manifest.json'),'source_activity_manifest_sha256':sha(base/'input_manifest.json'),'new_capital_outcomes_seen':False}
    dump(inputs/'activity_audit.json',out);print(out,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);audit(p.parse_args().inputs)
