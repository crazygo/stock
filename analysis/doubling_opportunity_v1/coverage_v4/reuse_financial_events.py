"""Copy validated, identical-source/period financial events into the isolated cache."""
from __future__ import annotations
import hashlib,json,shutil
from pathlib import Path
from context import CACHE,PARENT

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def reuse(start,end):
    pointer=PARENT/'raw_daily_v3/latest_backtest.json'
    if not pointer.exists():return {'copied':0,'reason':'no_original_frozen_manifest'}
    back=Path(json.loads(pointer.read_text())['path']);lineage=json.loads((back/'lineage.json').read_text())
    original=lineage['SEC_financial_event_manifest'];parser=PARENT/'joint_v2/financials.py'
    if original['start']!=start or original['end']!=end or original['parser_sha256']!=sha(parser):
        return {'copied':0,'reason':'period_or_parser_differs_parse_normally'}
    raw_sources=CACHE/'companyfacts';target=CACHE/'financial_events_v2';target.mkdir(exist_ok=True)
    records=[];copied=0
    for item in original['issuers']:
        if not item.get('path'):continue
        raw=raw_sources/f"CIK{item['cik']:010d}.json"
        if not raw.exists() or sha(raw)!=item['raw_sha256']:continue
        source=Path(item['path']);expected=lineage['financial_input_hashes'].get(str(source.resolve()))
        if not expected or sha(source)!=expected:raise ValueError('Old frozen financial event source changed')
        destination=target/source.name
        if destination.exists():
            if sha(destination)!=expected:raise ValueError('Existing identical-source financial event cache changed')
        else:
            tmp=destination.with_suffix('.copy.tmp');shutil.copyfile(source,tmp)
            if sha(tmp)!=expected:raise ValueError('Financial event changed during copy')
            tmp.replace(destination);copied+=1
        records.append({'cik':item['cik'],'source':str(source.resolve()),'destination':str(destination.resolve()),
            'raw_sha256':item['raw_sha256'],'event_sha256':expected})
    journal={'start':start,'end':end,'parser_sha256':sha(parser),'original_training_key':lineage['training_key'],
        'original_lineage_sha256':sha(back/'lineage.json'),'validated_identical_source_events':records,
        'model_or_features_changed':False,'original_files_modified':False}
    p=CACHE/'financial_derived_local_reuse.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(journal,indent=2));tmp.replace(p)
    print('same-source financial events reused',len(records),'new copies',copied,flush=True)
    return {'copied':copied,'validated':len(records),'reason':'exact_source_parser_period_and_event_SHA_match'}
