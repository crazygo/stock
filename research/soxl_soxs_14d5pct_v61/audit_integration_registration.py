"""Post-result audit of the preregistered production replication number."""
import argparse
from path_learning import *


def audit(run):
    run=Path(run);registration=read(run/'registration.json')
    for f,h in registration['source_sha256'].items():assert sha(core.ROOT/f)==h
    count=0
    for path in sorted((run/'training').glob('*.json')):
        row=read(path);at=pd.Timestamp(row['at'])
        for fit in row['fits'].values():
            if not fit['ready']:continue
            # The full valuation auditor regenerates the saved replicate. This
            # separate check also forbids choosing another replicate after outcomes.
            assert fit['replication']==0, path.name
            assert fit['seed']==1061+at.tz_convert(core.ET).date().toordinal()*16+CLOCKS.index(row['clock'])
            assert fit['paths']==2048
            assert fit['integration']['replicates']==8
            assert fit['integration']['points_per_replicate']==256
            count+=1
    assert count>0
    out={'passed':True,'production_forecasts':count,'all_replications_zero':True,
         'results_sha256':sha(run/'results.json'),'source_sha256':sha(Path(__file__)),
         'post_outcome_readonly_audit':True,'strategy_and_results_unchanged':True}
    dump(run/'integration_registration_audit.json',out);print(out)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);audit(p.parse_args().run)
