"""Derive the fixed eight-clock panel using only existing immutable local inputs."""
import argparse
from datetime import datetime,timezone
from dense_learning import *


def prepare_inputs(output,phase='development'):
    output=Path(output);assert not output.exists();output.mkdir(parents=True)
    p=read(SOURCE/'protocol.json');source,hashes=verified_source(phase,p);oldmanifest=read(source/'input_manifest.json')
    cutoff=pd.Timestamp(p[phase]['evaluation_end'],tz=core.ET)+pd.Timedelta(days=1)
    market=core.load_market();assert market.manifest==oldmanifest['full_market'];full_manifest=market.manifest;market=restrict_market(market,cutoff)
    leaders=load_panel(market.calendar,PANEL_ROOT)
    leader_manifest=read(core.ROOT/'.cache/soxl_soxs_14d5pct_v23/runs/clocks_20261012/input_manifest.json')['panel'];assert leaders[2]==leader_manifest
    dump(output/'input_registration.json',{'started_at':datetime.now(timezone.utc).isoformat(),'phase':phase,'protocol_sha256':sha(SOURCE/'protocol.json'),
        'source_sha256':{str(f.relative_to(core.ROOT)):sha(f) for f in [SOURCE/'prepare_dense.py',SOURCE/'dense_learning.py']},'new_capital_outcomes_seen':False})
    features,audits,_=build_features(market,leaders,{'analysis_clocks_et':CLOCKS})
    old=pd.read_parquet(source/'features.parquet');subset=features[features.clock.isin(OLD_CLOCKS)]
    pd.testing.assert_frame_equal(subset[old.columns],old)
    features.to_parquet(output/'features.parquet',compression='zstd',compression_level=7);dump(output/'feature_audits.json',audits)
    print('causal features prepared',len(features),'old clocks exact',len(old),flush=True)
    base=baseline_labels(market,features);stress=baseline_labels(market,features,2);dump(output/'baseline_labels.json',list(base.values()));dump(output/'stress_labels.json',list(stress.values()))
    parts=[];receipts=[];volume_audits=[]
    for clock in CLOCKS:
        f=features[features.clock==clock];vol,va=clock_volumes(leaders[0],market.sessions,market.calendar,cutoff,clock)
        vol.to_parquet(output/('volumes_'+clock[:5].replace(':','')+'.parquet'),compression='zstd',compression_level=7)
        part=prepare(f,base,joint_releases(f,base),factors(f,vol));part,rec=terminal_panel(market,part,base)
        part['clock']=f.clock;parts.append(part);receipts.extend(rec);volume_audits.extend(va)
        print('factor/terminal panel prepared',clock,len(part),flush=True)
    panel=pd.concat(parts).sort_index();oldpanel=pd.read_parquet(source/'panel.parquet')
    pd.testing.assert_frame_equal(panel.loc[oldpanel.index,oldpanel.columns],oldpanel)
    panel.to_parquet(output/'panel.parquet',compression='zstd',compression_level=7);dump(output/'terminal_receipts.json',receipts);dump(output/'volume_audit.json',volume_audits)
    financial,financial_manifest=load_sources();new,fr=build(features,financial,market.sessions)
    new.to_parquet(output/'financial_features.parquet',compression='zstd',compression_level=7);dump(output/'feature_receipts.json',fr)
    panel=panel.join(new);panel['feature_ready']=panel.feature_ready & panel.financial_ready
    oldfinancial=pd.read_parquet(source/'financial_panel.parquet');pd.testing.assert_frame_equal(panel.loc[oldfinancial.index,oldfinancial.columns],oldfinancial)
    panel.to_parquet(output/'financial_panel.parquet',compression='zstd',compression_level=7)
    manifest={'phase':phase,'source_run':str(source),'source_files_sha256':hashes,'full_market':full_manifest,'panel':leaders[2],
        'observation_cutoff_exclusive':cutoff,'financial_panel_source':financial_manifest,'clock_schedule':CLOCKS,'old_clock_inputs_exact':True,
        'derived_files_sha256':{f.name:sha(f) for f in sorted(output.iterdir()) if f.is_file()}}
    dump(output/'input_manifest.json',manifest)
    summary={'clock_rows':features.groupby('clock').size().to_dict(),'feature_ready':panel.groupby('clock').feature_ready.sum().to_dict(),
        'old_clock_rows_exact':len(old),'source_files_unchanged':True,'new_data_downloaded':False,'capital_outcomes_generated':False}
    dump(output/'coverage.json',summary);print(summary,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--phase',choices=['development','cases'],default='development');a=p.parse_args();prepare_inputs(a.output,a.phase)
