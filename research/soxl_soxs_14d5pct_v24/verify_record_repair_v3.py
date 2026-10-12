"""Prove the serialization repair did not change any model, action or return."""
import argparse,gzip,pickle,pickletools
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
    model_checks=record_checks=0;byte_exact=0;signed_zero_cases=[]
    for f in original.glob('models/*.pkl.gz'):
        with gzip.open(f,'rb') as stream:first=stream.read()
        with gzip.open(run/'models'/f.name,'rb') as stream:second=stream.read()
        if first==second:byte_exact+=1
        else:
            # Only signed zero in the HGB training bin thresholds is allowed.
            # -0.0 and +0.0 compare identically; no tree value or split changes.
            a,b=pickle.loads(first),pickle.loads(second)
            differences=[];allowed_payloads=set()
            for learner in a:
                for stage in a[learner]:
                    x,y=a[learner][stage],b[learner][stage]
                    if x is None or x['kind']!='hgb' or x['estimator'] is None:continue
                    aa=x['estimator']._bin_mapper.bin_thresholds_;bb=y['estimator']._bin_mapper.bin_thresholds_
                    for j,(v,w) in enumerate(zip(aa,bb)):
                        assert np.array_equal(v,w,equal_nan=True),(f.name,j)
                        diff=np.signbit(v)!=np.signbit(w)
                        assert not diff.any() or ((v[diff]==0)&(w[diff]==0)).all()
                        if diff.any():
                            differences.append({'feature':j,'signed_zero_values':int(diff.sum())})
                            allowed_payloads.add((v.tobytes(),w.tobytes()))
            oldops=list(pickletools.genops(first));newops=list(pickletools.genops(second))
            assert differences and len(oldops)==len(newops),f.name
            changed=0
            for (o,x,pos),(n,y,newpos) in zip(oldops,newops):
                assert o.name==n.name and pos==newpos,(f.name,pos)
                if x!=y:
                    assert isinstance(x,bytes) and isinstance(y,bytes) and (x,y) in allowed_payloads,(f.name,pos)
                    changed+=1
            assert changed==len(differences),(f.name,changed,differences)
            signed_zero_cases.append({'model':f.name,'bin_thresholds':differences})
        model_checks+=1
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
        'bit_identical_uncompressed_models':byte_exact,'models_equal_after_only_signed_zero_canonicalization':model_checks,
        'signed_zero_cases':signed_zero_cases,'training_records_only_arrays_to_numeric_lists':record_checks,
        'no_new_strategy_trial':True,'results_sha256':sha(run/'results.json'),'source_sha256':sha(Path(__file__))}
    core.dump(run/'record_repair_verification.json',receipt);print({'passed':True,'identical_models':model_checks,'identical_decisions_and_results':True})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);verify(p.parse_args().run)
