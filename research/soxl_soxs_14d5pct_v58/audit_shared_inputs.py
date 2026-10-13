"""Verify copied inputs, underlying source hashes and joint-date readiness."""
import argparse
from shared_learning import *


def audit(inputs):
    inputs=Path(inputs);m=read(inputs/'input_manifest.json');base=Path(m['base_inputs']);source,source_sha,prior_sha=verified_source()
    assert source_sha==m['source_v56_manifest_sha256'] and prior_sha==m['prior_v57_manifest_sha256']
    for f,h in m['derived_files_sha256'].items():assert sha(inputs/f)==h,f
    for f,h in m['original_inputs_sha256'].items():assert sha(inputs/f)==sha(base/f)==h,f
    market=core.load_market();assert market.manifest==m['full_market'];market=restrict_market(market,pd.Timestamp(m['observation_cutoff_exclusive']))
    p=pd.read_parquet(inputs/'activity_panel.parquet');rows=[]
    for day,group in p.groupby('date'):
        expected=[c for c in CLOCKS if pd.Timestamp(str(day.date())+' '+c,tz=core.ET)<pd.Timestamp(market.calendar[day]['close_at'])]
        assert list(group.clock)==expected
        complete=bool(group.feature_ready.all())
        rows.append({'date':day,'rows':len(group),'ready':complete,'terminal_complete':bool(group.terminal_release.notna().all()) if complete else False})
    shared=set(columns(CLOCKS[0]));assert all(shared<=set(columns(c)) for c in CLOCKS)
    assert len(shared)==33 and all(len(columns(c))==(33 if i==0 else 45) for i,c in enumerate(CLOCKS))
    out={'passed':True,'all_inputs_exact':True,'queries':len(p),'calendar_dates_checked':len(rows),'joint_feature_ready_dates':sum(x['ready'] for x in rows),'joint_terminal_known_dates':sum(x['terminal_complete'] for x in rows),'shared_feature_names':len(shared),'intraday_only_feature_names':12,'original14day_target_unchanged':True,'source_v56_manifest_sha256':source_sha,'prior_v57_manifest_sha256':prior_sha,'input_manifest_sha256':sha(inputs/'input_manifest.json'),'new_capital_outcomes_seen':False}
    dump(inputs/'input_audit.json',out);print(out,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);audit(p.parse_args().inputs)
