"""Post-result audit: changed model forecasts versus executable capital changes."""
import argparse
from state_reliability import *


def audit(run):
    run=Path(run);r=read(run/'results.json');v=read(run/'verification.json')
    assert v['passed'] and v['results_sha256']==sha(run/'results.json')
    events=read(run/'decisions_frozen.json')['routes'];p=events['state_band'];base=events['global_band']
    assert len(p)==len(base)
    a=pd.read_csv(run/'state_band_decisions.csv').set_index('decision_at')
    b=pd.read_csv(run/'global_band_decisions.csv').set_index('decision_at')
    changed=[];raw_changed=score_changed=radius_changed=0;maximum_q_shift=0.
    for left,right in zip(p,base):
        assert left['at']==right['at'];at=pd.Timestamp(left['at']);rec=read(run/'training'/(tag(at)+'.json'))
        x=rec['fits']['state']['raw'];y=rec['fits']['global']['raw']
        if x and y:
            delta=abs(x['q']-y['q']);raw_changed+=int(delta>1e-12);maximum_q_shift=max(maximum_q_shift,delta)
        x=rec['estimates']['state'];y=rec['estimates']['global']
        if x['ready'] and y['ready']:
            score_changed+=int(max(abs(np.array(x['point'])-y['point']))>1e-12)
            radius_changed+=int(abs(x['radius']-y['radius'])>1e-12)
        if (left['action'],left['allocation'])!=(right['action'],right['allocation']):
            changed.append({'at':left['at'],'primary_action':left['action'],'global_action':right['action'],
                'primary_capital_decision':a.loc[left['at'],'decision'],
                'global_capital_decision':b.loc[right['at'],'decision']})
    books={}
    for suffix in ['equity','trades','orders','decisions']:
        x=pd.read_csv(run/('state_band_'+suffix+'.csv'));y=pd.read_csv(run/('global_band_'+suffix+'.csv'))
        books[suffix]=x.equals(y)
    result={'passed':True,'post_result_only':True,'rules_changed':False,'queries':len(p),'raw_q_changed':raw_changed,
        'calibrated_pair_changed':score_changed,'joint_radius_changed':radius_changed,'max_q_difference':maximum_q_shift,
        'changed_action_count':len(changed),'changed_actions':changed,'exact_csv_equality':books,
        'primary_minus_global_monthly_mean':r['routes']['state_band']['primary_average']-r['routes']['global_band']['primary_average'],
        'results_sha256':sha(run/'results.json')}
    dump(run/'increment_audit.json',result);print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);audit(p.parse_args().run)
