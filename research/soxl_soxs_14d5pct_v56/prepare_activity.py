"""Derive fixed activity inputs from immutable local data, before new outcomes."""
import argparse,subprocess,shutil
from datetime import datetime,timezone
from activity_learning import *


def prepare_inputs(output,phase='development'):
    output=Path(output);assert not output.exists();source,source_sha=verified_source();p=read(SOURCE/'protocol.json')
    if phase=='development':base=source
    else:
        base=core.ROOT/'.cache/soxl_soxs_14d5pct_v56/inputs/base_cases_20261013'
        if not base.exists():subprocess.run([sys.executable,str(core.ROOT/'research/soxl_soxs_14d5pct_v54/prepare_dense.py'),'--output',str(base),'--phase','cases'],check=True)
    old=read(base/'input_manifest.json');assert old['phase']==phase
    for f,h in old['derived_files_sha256'].items():assert sha(base/f)==h
    market=core.load_market();assert market.manifest==old['full_market'];cutoff=pd.Timestamp(old['observation_cutoff_exclusive']);market=restrict_market(market,cutoff)
    output.mkdir(parents=True);dump(output/'input_registration.json',{'started_at':datetime.now(timezone.utc).isoformat(),'phase':phase,'protocol_sha256':sha(SOURCE/'protocol.json'),'source_sha256':{str(f.relative_to(core.ROOT)):sha(f) for f in [SOURCE/'prepare_activity.py',SOURCE/'activity_learning.py']},'new_capital_outcomes_seen':False})
    for name in ['features.parquet','financial_panel.parquet','baseline_labels.json','stress_labels.json']:
        shutil.copyfile(base/name,output/name)
    dump(output/'base_input_manifest.json',old)
    if (base/'input_audit.json').exists():shutil.copyfile(base/'input_audit.json',output/'base_input_audit.json')
    panel=pd.read_parquet(output/'financial_panel.parquet');prefixes=build_prefixes(market,cutoff);features,receipts=build_features(panel,prefixes)
    prefixes.to_parquet(output/'activity_prefixes.parquet',compression='zstd',compression_level=7);features.to_parquet(output/'activity_features.parquet',compression='zstd',compression_level=7)
    joined=panel.join(features);joined['feature_ready']=panel.feature_ready&joined.activity_ready
    joined.to_parquet(output/'activity_panel.parquet',compression='zstd',compression_level=7);dump(output/'activity_receipts.json',receipts)
    manifest={'phase':phase,'source_v55_run':str(source),'source_v55_manifest_sha256':source_sha,'base_inputs':str(base),'base_input_manifest_sha256':sha(base/'input_manifest.json'),'full_market':old['full_market'],'observation_cutoff_exclusive':old['observation_cutoff_exclusive'],'derived_files_sha256':{f.name:sha(f) for f in sorted(output.iterdir()) if f.is_file()}}
    dump(output/'input_manifest.json',manifest)
    receipt={'queries':len(panel),'activity_ready':int(features.activity_ready.sum()),'financial_and_activity_ready':int(joined.feature_ready.sum()),'base_feature_ready':int(panel.feature_ready.sum()),'unknown_activity_retained':int((~features.activity_ready).sum()),'prefixes':len(prefixes),'new_data_downloaded':False,'new_capital_outcomes_seen':False}
    dump(output/'coverage.json',receipt);print(receipt,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--phase',choices=['development','cases'],default='development');a=p.parse_args();prepare_inputs(a.output,a.phase)
