"""Verify native inference and no-data abstention without deleting real caches."""
import contextlib,io,json,pickle,sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from common import OUT,write,now
import recommend

def invoke():
    output=io.StringIO()
    with patch.object(sys,'argv',['recommend.py','--top','3','--format','json']),contextlib.redirect_stdout(output):recommend.main()
    return json.loads(output.getvalue())

def main():
    path=OUT/'reference_latest.json';original=path.read_bytes() if path.exists() else None
    checks=[]
    try:
        with patch.object(pickle,'loads',side_effect=RuntimeError('pickle forbidden at inference')):
            report=invoke();options=[r for route in report['routes'] for r in route['options']]
            assert options and all(0<=r['probability']<=1 and r['effective_threshold'] is None and not r['issued_signal'] for r in options)
            assert not report['issued_signal'] and not report['orders_sent'] and not report['current_probability']
            assert all(route.get('artifact_format')=='data_only_native' for route in report['routes'] if route['options'])
            checks.append(dict(scenario='native_models_without_pickle',options=len(options),routes=len(report['routes']),passed=True))
        original_exists=Path.exists;missing=OUT/'cache/latest_features.parquet'
        def exists(p):return False if p==missing else original_exists(p)
        with patch.object(Path,'exists',exists),patch.object(recommend.subprocess,'run',return_value=SimpleNamespace(returncode=1)):
            no_data=invoke()
            assert no_data['routes'] and all(not r['options'] for r in no_data['routes'])
            assert not no_data['issued_signal'] and not no_data['orders_sent']
            checks.append(dict(scenario='missing_causal_prices_and_provider_failure',routes=len(no_data['routes']),passed=True))
        write(OUT/'inference_verification.json',dict(at=now(),status='passed',checks=checks,note='Capability and abstention checks, not independent model effect validation.'))
    finally:
        if original is not None:path.write_bytes(original)
        elif path.exists():path.unlink()
    print(json.dumps(dict(status='passed',checks=len(checks))),flush=True)
if __name__=='__main__':main()
