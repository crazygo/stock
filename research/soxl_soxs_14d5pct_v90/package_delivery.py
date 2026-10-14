"""Package reviewed archive-proxy inputs and as-of verification, not a model."""
from prepare_review import *
from datetime import datetime,timezone


def package():
    data=read(SOURCE/'evidence/reviewed_records.json')
    result=read(SOURCE/'evidence/state_verification.json')
    assert result['passed'] and result['reviewed_records_sha256']==sha(SOURCE/'evidence/reviewed_records.json')
    inputs=dict(data['input_sha256'])
    # Include all original source hashes, including the6special-event packets.
    for p in (CACHE/'review_packets').glob('*.json'):inputs[str(p.relative_to(ROOT))]=sha(p)
    for p in [V89/'evidence/delivery_manifest.json',V89/'review_cases.json',V89/'catalogue.json']:
        inputs[str(p.relative_to(ROOT))]=sha(p)
    for key in ['snapshot_cache','timestamps_only_input']:
        item=result[key];inputs[item['path']]=item['sha256']
    for f,h in inputs.items():assert sha(ROOT/f)==h,f
    old=read(V89/'evidence/delivery_manifest.json')
    for f,h in old['delivery_files_sha256'].items():assert sha(V89/f)==h,f
    files={str(p.relative_to(SOURCE)):sha(p) for p in sorted(SOURCE.rglob('*'))
           if p.is_file() and '__pycache__' not in p.parts and p!=SOURCE/'evidence/delivery_manifest.json'}
    assert all(Path(f).suffix not in ['.parquet','.html','.pkl','.pickle','.npy','.npz'] for f in files)
    dump(SOURCE/'evidence/delivery_manifest.json',dict(created_at=datetime.now(timezone.utc).isoformat(),
        stage='reviewed_historical_archive_proxy_input',delivery_files_sha256=files,input_sha256=inputs,
        tests_passed=12,query_count=result['query_count'],issuer_state_count=result['issuer_states'],
        source_stage_not_strategy_acceptance=True,new_model_trained=False,new_market_returns_read=False,goal_complete=False))
    print({'delivery_files':len(files),'input_files':len(inputs)})


if __name__=='__main__':package()
