"""Prove the serialization repair did not change any model, action or return."""
import argparse,gzip
from pathlib import Path
import numpy as np
from joint_loss import core
from run_joint import sha
from verify_continuation import read


def verify(run):
    run=Path(run);reg=read(run/'registration.json');original=core.ROOT/reg['preregistration']['repair']['original_run']
    same=['predictions.json','results.json',*sorted(p.name for p in original.glob('*.csv'))]
    for n in same:assert sha(original/n)==sha(run/n),n
    a=read(original/'decisions_frozen.json');b=read(run/'decisions_frozen.json')
    for k in a:
        if k!='frozen_at':assert a[k]==b[k],k
    model_checks=record_checks=0
    for f in original.glob('models/*.pkl.gz'):
        with gzip.open(f,'rb') as stream:first=stream.read()
        with gzip.open(run/'models'/f.name,'rb') as stream:second=stream.read()
        assert first==second,f.name;model_checks+=1
    for f in original.glob('training/*.json'):
        a=read(f);b=read(run/'training'/f.name)
        for kind in ['hgb','logit']:
            for field in ['raw','adjusted']:
                old=a['learners'][kind][field];new=b['learners'][kind][field]
                if new is None:assert old is None
                else:
                    assert isinstance(old,str) and isinstance(new,list)
                    assert old==str(np.asarray(new)),(f.name,kind,field)
                    a['learners'][kind][field]=new
        assert a==b,f.name;record_checks+=1
    receipt={'passed':True,'original_run':str(original),'repaired_run':str(run.resolve()),'identical_output_files':same,
        'bit_identical_uncompressed_models':model_checks,'training_records_only_arrays_to_numeric_lists':record_checks,
        'no_new_strategy_trial':True,'results_sha256':sha(run/'results.json'),'source_sha256':sha(Path(__file__))}
    core.dump(run/'record_repair_verification.json',receipt);print({'passed':True,'identical_models':model_checks,'identical_decisions_and_results':True})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);verify(p.parse_args().run)
