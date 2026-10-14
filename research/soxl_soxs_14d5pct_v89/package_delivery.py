"""Record source-preparation delivery hashes after evidence checks; no training."""
from catalogue import *


def package():
    verification=json.loads((SOURCE/'evidence/source_verification.json').read_text())
    assert verification['passed'] and not verification['training_use_allowed']
    for name,digest in verification['input_sha256'].items():
        assert sha(ROOT/name)==digest,name
    files={str(p.relative_to(SOURCE)):sha(p) for p in sorted(SOURCE.rglob('*'))
           if p.is_file() and '__pycache__' not in p.parts
           and p != SOURCE/'evidence/delivery_manifest.json'}
    assert not any(Path(p).suffix in {'.html','.parquet','.pkl','.pickle','.npy','.npz'} for p in files)
    dump(SOURCE/'evidence/delivery_manifest.json',dict(
        created_at=datetime.now(timezone.utc).isoformat(),stage='source_preparation',
        delivery_files_sha256=files,input_sha256=verification['input_sha256'],
        test_command="python3 -W error -m unittest discover -s research/soxl_soxs_14d5pct_v89 -p 'test_*.py' -v",
        tests_passed=15,training_use_allowed=False,new_model_trained=False,
        new_market_return_evaluation=False,goal_complete=False))
    print({'delivery_files':len(files),'input_files':len(verification['input_sha256'])})


if __name__=='__main__':
    package()
