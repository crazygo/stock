"""Separate signal reliability, filter rejection and actual capital execution."""
import argparse
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
from run_reliability import read,labels,sha,dump,core
from verify_continuation import key


def group(rows):
    known=[x for x in rows if x['net_return'] is not None and np.isfinite(x['net_return'])]
    return {'count':len(rows),'known':len(known),'unknown':len(rows)-len(known),
        'profitable':sum(x['net_return']>0 for x in known),
        'mean_known_return':float(np.mean([x['net_return'] for x in known])) if known else None,
        'not_a_capital_return_or_independent_trade_count':True}


def audit(run):
    run=Path(run);v=read(run/'verification.json');assert v['passed'] and v['results_sha256']==sha(run/'results.json')
    m=read(run/'input_manifest.json');labs=labels(Path(m['label_source'])/'baseline_labels.json');stress=labels(Path(m['label_source'])/'stress_labels.json')
    f=read(run/'decisions_frozen.json')['routes'];r=read(run/'results.json');routes={};choices=[]
    for name,events in f.items():
        ds=pd.read_csv(run/(name+'_decisions.csv'));ds.index=pd.to_datetime(ds.decision_at,utc=True);rows=[]
        for e in events:
            if e['action'] not in core.ETFS:continue
            at=pd.Timestamp(e['at']);label=(stress if name=='double_cost' else labs)[key(at,e['action'])]
            net=label.get('net_return');status=ds.loc[at,'decision']
            rows.append({'at':e['at'],'symbol':e['action'],'net_return':net if net is not None and np.isfinite(net) else None,
                'status':status,'executed':status.startswith('buy_'),'reason':label['reason'],'gap':label.get('gap_below_stop',False)})
        routes[name]={'proposed':group(rows),'executed':group([x for x in rows if x['executed']]),
            'skipped':group([x for x in rows if not x['executed']]),'account_statuses':dict(Counter(ds.decision)),
            'executed_symbols':dict(Counter(x['symbol'] for x in rows if x['executed'])),
            'executed_gap_losses':sum(x['executed'] and x['gap'] and x['net_return'] is not None and x['net_return']<0 for x in rows),
            'net_capital_return':r['routes'][name]['continuous_return']}
    for e,p in zip(f['context_band'],f['context_point']):
        assert e['at']==p['at'];rec=read(run/'training'/(e['date'][:10]+'.json'));info=rec['estimates']['context']
        if not info['ready']:continue
        j=int(np.argmax(info['point']));symbol=core.ETFS[j];lab=labs[key(pd.Timestamp(e['at']),symbol)];net=lab.get('net_return')
        assert e['action'] in ['CASH',p['action']]
        choices.append({'at':e['at'],'symbol':symbol,'point':info['point'][j],'radius':info['radius'],'lower':info['lower'][j],
            'point_proposed':p['action']!='CASH','band_proposed':e['action']!='CASH',
            'net_return':net if net is not None and np.isfinite(net) else None,
            'block_radius':info['radius'],'iid_radius':rec['estimates']['context_iid']['radius']})
    changed={}
    for name in ['context_point','score_band','score_point','intercept_band','context_iid_band','source_early']:
        changed[name]=sum(a['action']!=b['action'] for a,b in zip(f['context_band'],f[name]))
    radius={k:float(np.mean([x[k] for x in choices])) for k in ['block_radius','iid_radius','point','lower']}
    out={'results_sha256':sha(run/'results.json'),'routes':routes,'primary_action_differences':changed,
        'point_candidates_accepted':group([x for x in choices if x['band_proposed']]),
        'point_candidates_rejected':group([x for x in choices if x['point_proposed'] and not x['band_proposed']]),
        'average_query_values':radius,'block_radius_exceeds_iid_dates':sum(x['block_radius']>x['iid_radius'] for x in choices),
        'choices':choices,'band_is_estimator_sensitivity_not_individual_return_coverage':True,'post_result_diagnostic_not_new_selection':True}
    dump(run/'diagnostics.json',out);print({k:v for k,v in out.items() if k not in ['routes','choices']},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);audit(p.parse_args().run)
