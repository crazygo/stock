"""Copy verified source inputs and freeze the paired historical price bank."""
import argparse,shutil
from datetime import datetime,timezone
from path_learning import *


def prepare_inputs(output,phase='development',base=None):
    output=Path(output);assert not output.exists();source,source_sha,prior_sha=verified_source()
    if base is None:
        assert phase=='development','Case inputs require an audited extension'
        base=source
    base=Path(base);old=read(base/'input_manifest.json');assert old['phase']==phase
    for f,h in old['derived_files_sha256'].items():assert sha(base/f)==h,f
    market=core.load_market();assert market.manifest==old['full_market'];market=restrict_market(market,pd.Timestamp(old['observation_cutoff_exclusive']))
    output.mkdir(parents=True);dump(output/'input_registration.json',{'started_at':datetime.now(timezone.utc).isoformat(),'phase':phase,'protocol_sha256':sha(SOURCE/'protocol.json'),'source_sha256':{str(f.relative_to(core.ROOT)):sha(f) for f in [Path(__file__),SOURCE/'path_learning.py']},'source_v56_manifest_sha256':source_sha,'prior_v59_manifest_sha256':prior_sha,'new_capital_outcomes_seen':False})
    copied=['features.parquet','financial_panel.parquet','activity_panel.parquet','baseline_labels.json','stress_labels.json']
    for f in copied:shutil.copyfile(base/f,output/f)
    panel=pd.read_parquet(output/'activity_panel.parquet');bank,receipts=build_bank(market,panel)
    from state_paths import prepare_context
    prepare_context(output,market,bank,panel,pd.Timestamp(old['observation_cutoff_exclusive']))
    np.savez_compressed(output/'path_bank.npz',**bank);dump(output/'bank_records.json',receipts);dump(output/'base_input_manifest.json',old)
    manifest={'phase':phase,'base_inputs':str(base),'source_v56_run':str(source),'source_v56_manifest_sha256':source_sha,'prior_v59_manifest_sha256':prior_sha,'base_input_manifest_sha256':sha(base/'input_manifest.json'),'full_market':old['full_market'],'observation_cutoff_exclusive':old['observation_cutoff_exclusive'],'original_inputs_sha256':{f:sha(base/f) for f in copied},'derived_files_sha256':{p.name:sha(p) for p in output.iterdir() if p.is_file()}}
    dump(output/'input_manifest.json',manifest)
    out={'queries':len(panel),'feature_ready':int(panel.feature_ready.sum()),'source_days':len(receipts),'complete_full_days':sum(x['complete'] for x in receipts),'paired_blocks':len(bank['blocks']),'new_capital_outcomes_seen':False};dump(output/'coverage.json',out);print(out,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--phase',choices=['development','cases'],default='development');p.add_argument('--base',type=Path);a=p.parse_args();prepare_inputs(a.output,a.phase,a.base)
