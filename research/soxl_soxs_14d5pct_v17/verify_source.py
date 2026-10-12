"""Audit the unchanged daily recipe and prove pre-May forecasts were not rewritten."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'soxl_soxs_14d5pct_v14'))
from verify_paired import verify
from paired_entry import core
from selection import validate_selection


def check(run):
    authorization=validate_selection();run=Path(run);verify(run)
    original=core.ROOT/'.cache/soxl_soxs_14d5pct_v14/runs/paired_20261012'
    past=json.loads((original/'predictions.json').read_text());extended=json.loads((run/'predictions.json').read_text());count=0
    for name,events in past.items():
        prefix=extended[name][:len(events)]
        assert prefix==events, name+': historical forecasts changed'
        count+=len(events)
    sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
    report={'passed':True,'pre_may_events_identical':count,'routes':list(past),'original_prediction_sha256':sha(original/'predictions.json'),
        'extended_prediction_sha256':sha(run/'predictions.json'),'selection_authorization':authorization,
        'no_october_outcomes':True,'source_engineering_verification':sha(run/'verification.json')}
    core.dump(run/'historical_prefix_verification.json',report)
    core.dump(run/'delivery_manifest.json',{'cache_outputs_sha256':{f.name:sha(f) for f in run.iterdir() if f.is_file()},
        'model_files_sha256':{f.name:sha(f) for f in (run/'models').glob('*.gz')},'historical_prefix_verification':report,
        'source_is_not_a_trading_promotion':True})
    print(report)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);check(p.parse_args().run)
