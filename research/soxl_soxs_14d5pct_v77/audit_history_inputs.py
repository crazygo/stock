"""Independent prefix/features and all six raw-contract checks for added history."""
import argparse
import math
from history_learning import *
from audit_inputs import reference_features, raw_contract


def audit(inputs):
    inputs=Path(inputs);manifest=read(inputs/'input_manifest.json')
    registration=read(inputs/'input_registration.json')
    assert registration['protocol_sha256']==manifest['protocol_sha256']
    for f,h in manifest['derived_files_sha256'].items():assert sha(inputs/f)==h,f
    for f,h in registration['code_sha256'].items():assert sha(ROOT/f)==h,f
    assert manifest['parent_manifests']==verify_parents()
    market=extended_market();assert market.manifest==manifest['expanded_market_manifest']
    assert load_market().manifest==manifest['ledger_market_manifest']
    prefixes={s:pd.read_parquet(inputs/(s+'_prefix.parquet')) for s in sorted({s for p in PAIRS.values() for s in p})}
    prefix_count=0;unready={}
    for s,table in prefixes.items():
        f=market.bars[s];unready[s]=[]
        for at,row in table.iterrows():
            start=pd.Timestamp(str(row.date.date())+' 04:00',tz=core.ET).tz_convert('UTC');end=at-pd.Timedelta(seconds=30)
            ix=pd.date_range(start+pd.Timedelta(minutes=5),end,freq='5min');g=f.reindex(ix)
            good=(len(g)>=4 and g.available_at.notna().all() and g.available_at.le(at).all() and
                  g.available_at.ge(g.end_at).all() and g.start_at.eq(g.end_at-pd.Timedelta(minutes=5)).all())
            v=g[['open','high','low','close','volume']].to_numpy(float)
            good=bool(good and np.isfinite(v).all() and (v[:,:4]>0).all() and (v[:,4]>=0).all())
            amount=np.sum((v[:,1]+v[:,2]+v[:,3])*v[:,4]/3) if good else np.nan;good=bool(good and amount>0)
            assert row.prefix_ready==good,(s,at)
            if good:
                np.testing.assert_allclose([row.amount,row.quote,row.quote15],[amount,g.econ_close.iloc[-1],g.econ_close.iloc[-4]],rtol=1e-12,atol=1e-9)
                assert row.available_at==g.available_at.max()
            else:unready[s].append(at)
            prefix_count+=1
        print('Verified prefixes',s,len(table),flush=True)
    lab=labels(inputs/'earlier_labels.json');feature_checks=contract_checks=release_checks=overlap=0;outcomes={};notready={}
    for pair,(long,short,_) in PAIRS.items():
        panel=pd.read_parquet(inputs/(pair+'_panel.parquet'));frozen=pd.read_parquet(PARENT/(pair+'_panel.parquet'))
        assert panel.loc[frozen.index].equals(frozen)
        assert sha(inputs/(pair+'_short_panel.parquet'))==sha(PARENT/(pair+'_panel.parquet'))
        overlap+=len(frozen);notready[pair]=[]
        for i,(at,row) in enumerate(panel.iterrows()):
            want=reference_features(market,prefixes,pair,at)
            assert row.feature_ready==(want is not None),(pair,at)
            if want is not None:
                np.testing.assert_allclose(row[FIELDS].to_numpy(float),want,rtol=1e-10,atol=1e-10);feature_checks+=len(FIELDS)
            else:notready[pair].append(at)
            if at in frozen.index:continue
            pairlabels=[lab[at.isoformat()+'|'+s] for s in [long,short]]
            for l in pairlabels:
                ref=raw_contract(market,l['symbol'],at)
                assert l['reason']==ref['reason'] and l['complete_window']==ref['complete_window'],(l['symbol'],at)
                assert l['label_available_at']==ref['release']
                for k in ['entry_fill_raw','exit_fill_economic','net_return','dividend_per_economic_unit']:
                    if k in ref:np.testing.assert_allclose(l[k],ref[k],rtol=1e-11,atol=1e-11)
                for k in ['entry_at','exit_at']:
                    if k in ref:assert l[k]==ref[k]
                contract_checks+=1;outcomes[ref['reason']]=outcomes.get(ref['reason'],0)+1
            known=all(l['complete_window'] and l.get('label_available_at') is not None and l.get('net_return') is not None and np.isfinite(l['net_return']) and l['net_return']>-1 for l in pairlabels)
            if known:
                y=np.array([math.log1p(l['net_return']) for l in pairlabels])
                np.testing.assert_allclose([row.C,row.D],[sum(y)/2,(y[0]-y[1])/2],rtol=1e-12,atol=1e-12)
                assert row.terminal_release==max(l['label_available_at'] for l in pairlabels)
                assert row.terminal_release>=max(l['expiry_at'] for l in pairlabels)+pd.Timedelta(seconds=1)
            else:assert pd.isna(row.terminal_release) and pd.isna(row.C) and pd.isna(row.D)
            release_checks+=1
            if (i+1)%800==0:print('Verified features/contracts',pair,i+1,flush=True)
        print('Verified all features',pair,len(panel),flush=True)
    for f in ['features.parquet','baseline_labels.json','stress_labels.json']:assert sha(inputs/f)==sha(PARENT/f)
    result={'passed':True,'prefixes':prefix_count,'feature_values':feature_checks,'six_etf_earlier_contracts':contract_checks,
            'paired_release_checks':release_checks,'earlier_outcomes':outcomes,'old_panel_rows_exact':overlap,
            'unready_prefixes':unready,'unready_features':notready,'input_manifest_sha256':sha(inputs/'input_manifest.json'),
            'source_sha256':sha(Path(__file__)),'strategy_outcomes_evaluated':False,'old_ledger_inputs_exact':True}
    dump(inputs/'input_audit.json',result);print({k:v for k,v in result.items() if not k.startswith('unready')},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);audit(p.parse_args().inputs)
