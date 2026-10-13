"""Read-only authorization gate before any new target-month data/model run."""
import argparse
import json
from normalized_learning import *


def authorize(run):
    run=Path(run);reg=read(run/'registration.json');r=read(run/'results.json');v=read(run/'verification.json');p=read(SOURCE/'protocol.json')
    assert reg['phase']=='development' and reg['protocol_sha256']==sha(SOURCE/'protocol.json')
    assert v['passed'] and v['results_sha256']==sha(run/'results.json')
    assert r['primary_name']==p['primary'] and r['primary']==r['routes'][p['primary']]
    gate=development_gate(r['primary'],r['coverage']['complete'],p)
    if not gate['passed']:raise ValueError('Verified primary development failed; new case evaluation is not authorized')
    return {'run':str(run.resolve()),'results_sha256':sha(run/'results.json'),'verification_sha256':sha(run/'verification.json'),'gate':gate}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--development',required=True,type=Path);args=parser.parse_args()
    try:print(json.dumps(authorize(args.development),default=str))
    except ValueError as error:print(json.dumps({'authorized':False,'reason':str(error)}));raise SystemExit(2)
