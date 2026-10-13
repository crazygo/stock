"""Reuse audited original-horizon inputs byte for byte; no new proxy targets."""
import argparse,shutil
from datetime import datetime,timezone
from shared_learning import *


def prepare_inputs(output,phase='development',base=None):
    output=Path(output);assert not output.exists();source,source_sha,prior_sha=verified_source()
    if base is None:
        assert phase=='development','An audited case feature extension is required'
        base=source
    base=Path(base);old=read(base/'input_manifest.json');assert old['phase']==phase
    for f,h in old['derived_files_sha256'].items():assert sha(base/f)==h,f
    output.mkdir(parents=True)
    dump(output/'input_registration.json',{'started_at':datetime.now(timezone.utc).isoformat(),'phase':phase,'protocol_sha256':sha(SOURCE/'protocol.json'),'prepare_source_sha256':sha(Path(__file__)),'source_v56_manifest_sha256':source_sha,'prior_v57_manifest_sha256':prior_sha,'new_capital_outcomes_seen':False})
    copied=['features.parquet','financial_panel.parquet','activity_panel.parquet','baseline_labels.json','stress_labels.json']
    for f in copied:shutil.copyfile(base/f,output/f)
    dump(output/'base_input_manifest.json',old)
    manifest={'phase':phase,'base_inputs':str(base),'source_v56_run':str(source),'source_v56_manifest_sha256':source_sha,'prior_v57_manifest_sha256':prior_sha,'base_input_manifest_sha256':sha(base/'input_manifest.json'),'full_market':old['full_market'],'observation_cutoff_exclusive':old['observation_cutoff_exclusive'],'original_inputs_sha256':{f:sha(base/f) for f in copied},'derived_files_sha256':{p.name:sha(p) for p in output.iterdir() if p.is_file()}}
    dump(output/'input_manifest.json',manifest)
    panel=pd.read_parquet(output/'activity_panel.parquet')
    out={'queries':len(panel),'feature_ready_queries':int(panel.feature_ready.sum()),'dates':panel.date.nunique(),'feature_ready_dates':panel[panel.feature_ready].date.nunique(),'all_inputs_exact':True,'new_capital_outcomes_seen':False}
    dump(output/'coverage.json',out);print(out,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--phase',choices=['development','cases'],default='development');p.add_argument('--base',type=Path);a=p.parse_args();prepare_inputs(a.output,a.phase,a.base)
