"""Extend causal panels; keep old panel values and ledger inputs exactly frozen."""
import argparse, shutil
from datetime import datetime,timezone
from history_learning import *
sys.path.insert(0,str(ROOT/'research/soxl_soxs_14d5pct_v3'))
from risk import risk_label


def prepare(output):
    output=Path(output);assert not output.exists();parents=verify_parents();output.mkdir(parents=True)
    dump(output/'input_registration.json',{'created_at':datetime.now(timezone.utc).isoformat(),
        'protocol_sha256':sha(SOURCE/'protocol.json'),'parents':parents,
        'code_sha256':{str(p.relative_to(ROOT)):sha(p) for p in SOURCE.glob('*.py')},'strategy_capital_outcomes_seen':False})
    market=extended_market();ledger=load_market();frozen={k:pd.read_parquet(PARENT/(k+'_panel.parquet')) for k in PAIRS}
    start=frozen['semiconductor'].index.min();end=frozen['semiconductor'].date.max()
    index=pd.DatetimeIndex([pd.Timestamp(str(d.date())+' '+c,tz=core.ET).tz_convert('UTC')
        for d in market.sessions[(market.sessions>='2023-04-03')&(market.sessions<=end)]
        for c in CLOCKS if pd.Timestamp(str(d.date())+' '+c,tz=core.ET)<pd.Timestamp(market.calendar[d]['close_at'])])
    prefixes=old.prefix_table(market,index)
    for s,p in prefixes.items():p.to_parquet(output/(s+'_prefix.parquet'))
    for name in ['features.parquet','baseline_labels.json','stress_labels.json']:
        shutil.copyfile(PARENT/name,output/name)
    outcomes=[];receipts=[];compatibility=0;coverage={}
    for pair,(long,short,_) in PAIRS.items():
        rows=[]
        for i,at in enumerate(index):
            day=at.tz_convert(core.ET).tz_localize(None).normalize();clock=at.tz_convert(core.ET).strftime('%H:%M:%S')
            x,receipt=old.feature_row(market,prefixes,pair,at)
            if at>=start:
                prior=frozen[pair].loc[at];assert bool(prior.feature_ready)==(x is not None),(pair,at)
                if x is not None:
                    np.testing.assert_allclose(x,prior[FIELDS].to_numpy(float),atol=1e-10,rtol=1e-10);compatibility+=len(FIELDS)
                rows.append({'at':at,**prior.to_dict()})
            else:
                labs=[]
                for s in [long,short]:
                    label=risk_label(market,s,at,{'id':'fixed6','stop_fraction':.06,'breakeven_trigger':None})
                    expiry=label.get('expiry_at');entry=label.get('entry_at');f=market.bars[s]
                    path=f[(f.start_at>=entry)&(f.end_at<=expiry)&f.session_type.eq('regular')] if entry is not None and expiry is not None else f.iloc[:0]
                    label['label_available_at']=path.available_at.max() if label.get('complete_window') and len(path) else None
                    labs.append(label);outcomes.append(label)
                known=all(l.get('complete_window') and l.get('label_available_at') is not None and l.get('net_return') is not None and np.isfinite(l['net_return']) and l['net_return']>-1 for l in labs)
                y=np.log1p([l['net_return'] for l in labs]) if known else [np.nan,np.nan]
                rows.append({'at':at,'date':day,'clock':clock,'feature_ready':x is not None,
                    **dict(zip(FIELDS,x if x is not None else np.full(len(FIELDS),np.nan))),
                    'C':float((y[0]+y[1])/2),'D':float((y[0]-y[1])/2),
                    'terminal_release':max(l['label_available_at'] for l in labs) if known else pd.NaT})
                receipts.append({'at':at,'pair':pair,**receipt})
            if (i+1)%800==0:print('Prepared',pair,i+1,'/',len(index),flush=True)
        panel=pd.DataFrame(rows).set_index('at');panel['terminal_release']=pd.to_datetime(panel.terminal_release,utc=True).astype('datetime64[ns, UTC]')
        panel.to_parquet(output/(pair+'_panel.parquet'))
        shutil.copyfile(PARENT/(pair+'_panel.parquet'),output/(pair+'_short_panel.parquet'))
        assert panel.loc[frozen[pair].index].equals(frozen[pair])
        coverage[pair]={'rows':len(panel),'feature_ready':int(panel.feature_ready.sum()),'mature':int(panel.terminal_release.notna().sum()),'earlier_rows':int((panel.index<start).sum()),'first_ready':str(panel[panel.feature_ready].index.min())}
    dump(output/'earlier_labels.json',outcomes);dump(output/'earlier_feature_receipts.json',receipts)
    dump(output/'input_manifest.json',{'created_at':datetime.now(timezone.utc).isoformat(),'protocol_sha256':sha(SOURCE/'protocol.json'),
         'parent_manifests':parents,'expanded_market_manifest':market.manifest,'ledger_market_manifest':ledger.manifest,
         'old_feature_compatibility_values':compatibility,'coverage':coverage,
         'derived_files_sha256':{p.name:sha(p) for p in output.iterdir() if p.is_file()}})
    print(coverage,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);prepare(p.parse_args().output)
