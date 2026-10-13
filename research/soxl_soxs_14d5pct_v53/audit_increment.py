"""Frozen post-result comparison of curve versus linear/global capital paths."""
import argparse
from curve_reliability import *


def csv(path):
    try:return pd.read_csv(path)
    except pd.errors.EmptyDataError:return pd.DataFrame()


def audit(run):
    run=Path(run);r=read(run/'results.json');v=read(run/'verification.json')
    assert v['passed'] and v['results_sha256']==sha(run/'results.json')
    events=read(run/'decisions_frozen.json')['routes'];p=events['curve_band'];comparisons={}
    a=csv(run/'curve_band_decisions.csv').set_index('decision_at')
    for control in ['linear','global']:
        name=control+'_band';base=events[name];assert len(p)==len(base)
        b=csv(run/(name+'_decisions.csv')).set_index('decision_at')
        changed=[];raw_changed=score_changed=radius_changed=0;maximum_q_shift=0.
        for left,right in zip(p,base):
            assert left['at']==right['at'];at=pd.Timestamp(left['at']);rec=read(run/'training'/(tag(at)+'.json'))
            x=rec['fits']['curve']['raw'];y=rec['fits'][control]['raw']
            if x and y:
                delta=abs(x['q']-y['q']);raw_changed+=int(delta>1e-12);maximum_q_shift=max(maximum_q_shift,delta)
            x=rec['estimates']['curve'];y=rec['estimates'][control]
            if x['ready'] and y['ready']:
                score_changed+=int(max(abs(np.array(x['point'])-y['point']))>1e-12)
                radius_changed+=int(abs(x['radius']-y['radius'])>1e-12)
            if (left['action'],left['allocation'])!=(right['action'],right['allocation']):
                changed.append({'at':left['at'],'primary_action':left['action'],'control_action':right['action'],
                    'primary_capital_decision':a.loc[left['at'],'decision'],'control_capital_decision':b.loc[right['at'],'decision']})
        equal={suffix:csv(run/('curve_band_'+suffix+'.csv')).equals(csv(run/(name+'_'+suffix+'.csv'))) for suffix in ['equity','trades','orders','decisions']}
        primary_trades=csv(run/'curve_band_trades.csv');control_trades=csv(run/(name+'_trades.csv'))
        entry_set=lambda t:{(row.decision_at,row.symbol) for row in t.itertuples()}
        x,y=entry_set(primary_trades),entry_set(control_trades)
        comparisons[control]={'queries':len(p),'raw_q_changed':raw_changed,'calibrated_pair_changed':score_changed,
            'joint_radius_changed':radius_changed,'max_q_difference':maximum_q_shift,'changed_action_count':len(changed),
            'changed_actions':changed,'exact_csv_equality':equal,'added_actual_buys':sorted(x-y),'removed_actual_buys':sorted(y-x),
            'primary_minus_control_monthly_mean':r['routes']['curve_band']['primary_average']-r['routes'][name]['primary_average']}
    result={'passed':True,'post_result_only':True,'rules_changed':False,'comparisons':comparisons,'results_sha256':sha(run/'results.json')}
    dump(run/'increment_audit.json',result);print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);audit(p.parse_args().run)
