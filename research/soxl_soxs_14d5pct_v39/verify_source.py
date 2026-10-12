"""Verify unchanged financial recipe and retain all original preMay records."""
import argparse
from supported import *
from verify_financial import verify

def main(run):
    run=Path(run);p,prereg=load_frozen();reg=read(run/'registration.json')
    assert reg['consumer_authorization']==authorize(reg['consumer_authorization']['run'])
    assert reg['authorization']['upstream_policy_qualified'] is False
    assert Path(reg['authorization']['run'])==(core.ROOT/p['forecast_source']).resolve()
    verify(run)
    names=[f for f in run.iterdir() if f.is_file() and f.name!='extension_receipt.json']
    for folder in ['training','daily','source_daily','models']:names.extend((run/folder).glob('*'))
    dump(run/'extension_receipt.json',{'consumer_only':True,'upstream_policy_qualified':False,'verification_sha256':sha(run/'verification.json'),
        'files_sha256':{str(f.relative_to(run)):sha(f) for f in sorted(names) if f.is_file()}})
    print('Financial extension verified for v39 consumer only',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);main(p.parse_args().run)
