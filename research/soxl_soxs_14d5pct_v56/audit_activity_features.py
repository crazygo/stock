"""Independent raw-array prefix sums, past-date memberships and six formulas."""
import argparse
from activity_learning import *


def audit(inputs):
    inputs=Path(inputs);manifest=read(inputs/'input_manifest.json');source,source_sha=verified_source();assert manifest['source_v55_manifest_sha256']==source_sha
    base=Path(manifest['base_inputs']);assert sha(base/'input_manifest.json')==manifest['base_input_manifest_sha256']
    for f,h in manifest['derived_files_sha256'].items():assert sha(inputs/f)==h
    old=read(inputs/'base_input_manifest.json');assert old==read(base/'input_manifest.json')
    market=core.load_market();assert market.manifest==manifest['full_market'];cutoff=pd.Timestamp(manifest['observation_cutoff_exclusive']);market=restrict_market(market,cutoff)
    leaders=load_panel(market.calendar,PANEL_ROOT);assert leaders[2]==old['panel'];_,financial=load_sources();assert financial==old['financial_panel_source']
    for item in financial.values():
        for kind in ['companyfacts','submissions']:assert sha(core.ROOT/item[kind]['path'])==item[kind]['sha256']
    for f in ['features.parquet','financial_panel.parquet','baseline_labels.json','stress_labels.json']:assert sha(inputs/f)==sha(base/f)
    prefixes=pd.read_parquet(inputs/'activity_prefixes.parquet');want={};prefix_checks=0
    for symbol in SYMBOLS:
        bars=market.bars[symbol];times=bars.index.asi8
        for day in market.sessions:
            for clock in CLOCKS:
                at=pd.Timestamp(str(day.date())+' '+clock,tz=core.ET).tz_convert('UTC')
                if at>=cutoff or at>=pd.Timestamp(market.calendar[day]['close_at']):continue
                start=(day.tz_localize(core.ET)+pd.Timedelta(hours=4)).tz_convert('UTC');end=at-pd.Timedelta(seconds=30)
                ix=np.arange(times.searchsorted(start.value,side='right'),times.searchsorted(end.value,side='right'));g=bars.iloc[ix]
                expected=np.arange(start.value+300_000_000_000,end.value+1,300_000_000_000,dtype=np.int64)
                good=np.array_equal(times[ix],expected) and len(g)>0 and g.available_at.notna().all() and g.available_at.le(at).all() and g.available_at.ge(g.end_at).all() and g.start_at.eq(g.end_at-pd.Timedelta(seconds=300)).all() and g.volume.notna().all() and g.volume.ge(0).all()
                amount=float(np.dot(g.volume.to_numpy(float),(g.high.to_numpy(float)+g.low.to_numpy(float)+g.close.to_numpy(float))/3)) if good else np.nan
                if good:
                    equivalent=float(np.dot(g.econ_volume.to_numpy(float),(g.econ_high.to_numpy(float)+g.econ_low.to_numpy(float)+g.econ_close.to_numpy(float))/3));np.testing.assert_allclose(equivalent,amount,rtol=1e-12,atol=1e-7)
                good=bool(good and np.isfinite(amount) and amount>0);amount=amount if good else np.nan
                row=prefixes.loc[(symbol,at)];assert row.complete==good and row.rows==len(g) and row.date==day and row.clock==clock
                np.testing.assert_allclose(row.activity,amount,rtol=1e-12,atol=1e-7,equal_nan=True)
                assert (pd.isna(row.last_available) and pd.isna(g.available_at.max())) or row.last_available==g.available_at.max()
                want[(symbol,at)]=(good,amount);prefix_checks+=1
    assert len(want)==len(prefixes)
    original=pd.read_parquet(inputs/'financial_panel.parquet');new=pd.read_parquet(inputs/'activity_features.parquet');panel=pd.read_parquet(inputs/'activity_panel.parquet');receipts=read(inputs/'activity_receipts.json');assert list(new.index)==list(original.index)==list(panel.index);assert len(receipts)==len(original)
    feature_checks=0
    for (at,row),receipt in zip(original.iterrows(),receipts):
        assert pd.Timestamp(receipt['at'])==at;values=[];ready=True
        eligible=[d for d in market.sessions if d<row.date and pd.Timestamp(str(d.date())+' '+row.clock,tz=core.ET)<pd.Timestamp(market.calendar[d]['close_at'])][-20:]
        past=[pd.Timestamp(str(d.date())+' '+row.clock,tz=core.ET).tz_convert('UTC') for d in eligible]
        for symbol in SYMBOLS:
            current=want[(symbol,at)];okay=current[0] and len(past)==20 and all(want[(symbol,t)][0] for t in past);r=receipt['members'][symbol]
            assert list(pd.to_datetime(r['prior'],utc=True))==past and pd.Timestamp(r['current'])==at and r['complete']==okay
            mean=sum(want[(symbol,t)][1] for t in past)/20 if okay else None
            value=max(-3.,min(3.,float(np.log(current[1]/mean)))) if okay else np.nan
            if okay:np.testing.assert_allclose(r['prior_mean'],mean,rtol=1e-12);np.testing.assert_allclose(r['log_relative'],value,rtol=1e-11,atol=1e-11)
            else:assert r['prior_mean'] is None and r['log_relative'] is None
            values.append(value);ready=ready and okay
        l,s,x,q=values;move=float(row.sector_premarket)
        expected=[(l+s)/2,(l-s)/2,(l+s)*move/2,(l-s)*move/2,(x-q)/2,(x-q)*move/2] if ready else [np.nan]*6
        np.testing.assert_allclose(new.loc[at,FEATURES].to_numpy(float),expected,rtol=1e-11,atol=1e-11,equal_nan=True);assert new.loc[at,'activity_ready']==ready;feature_checks+=6
    pd.testing.assert_frame_equal(panel[original.columns].drop(columns='feature_ready'),original.drop(columns='feature_ready'))
    pd.testing.assert_frame_equal(panel[new.columns],new);assert panel.feature_ready.equals(original.feature_ready&new.activity_ready)
    receipt={'passed':True,'prefix_checks':prefix_checks,'raw_economic_activity_equal':True,'feature_equalities':feature_checks,'query_count':len(panel),'activity_ready':int(new.activity_ready.sum()),'base_columns_and_labels_exact':True,'unknown_activity_retained':int((~new.activity_ready).sum()),'input_manifest_sha256':sha(inputs/'input_manifest.json'),'new_capital_outcomes_seen':False}
    dump(inputs/'activity_audit.json',receipt);print(receipt,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);audit(p.parse_args().inputs)
