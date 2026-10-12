"""Post-run diagnostic only: paired opportunities do not create tradable hindsight."""
import argparse
from pathlib import Path
import numpy as np
from linear_entry import core
from verify_continuation import read, labels, key


def diagnose(run):
    run=Path(run); registration=read(run/'registration.json'); protocol=registration['preregistration']['protocol']
    settings=protocol[registration['phase']]; events=read(run/'predictions.json')[protocol['primary']]
    lab=labels(run/'baseline_labels.json'); rows=[]; unknown=[]
    for e in events:
        if not settings['evaluation_start']<=e['date'][:10]<=settings['evaluation_end']:continue
        pair={s:lab[key(e['at'],s)] for s in core.ETFS}
        if not all(x['complete_window'] and x.get('label_available_at') is not None for x in pair.values()):
            unknown.append(e['date']);continue
        if not e['raw']:continue
        values={s:float(np.log1p(x['net_return'])) for s,x in pair.items()};top=e['top_symbol']
        other='SOXS' if top=='SOXL' else 'SOXL'
        rows.append({'date':e['date'],'raw_top':top,'values':values,'hits':{s:x['target_hit'] for s,x in pair.items()},
            'top_minus_other':values[top]-values[other]})
    n=len(rows)
    out={'diagnostic_only_after_run':True,'not_a_tradable_foresight_policy':True,
        'opportunity_count_is_not_a_capital_return_bound':True,'mature_dates':n,'joint_immature_dates':unknown,
        'joint_hits':{str(k):sum(sum(x['hits'].values())==k for x in rows) for k in [0,1,2]},
        'raw_top_choices':{s:sum(x['raw_top']==s for x in rows) for s in core.ETFS},
        'top_hit_count':sum(x['hits'][x['raw_top']] for x in rows),
        'top_minus_other_mean_log':float(np.mean([x['top_minus_other'] for x in rows])) if n else None,
        'top_actual_mean_log':float(np.mean([x['values'][x['raw_top']] for x in rows])) if n else None,
        'always_soxl_mean_log':float(np.mean([x['values']['SOXL'] for x in rows])) if n else None,
        'always_soxs_mean_log':float(np.mean([x['values']['SOXS'] for x in rows])) if n else None}
    core.dump(run/'pair_diagnosis.json',out);print(out)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);diagnose(p.parse_args().run)
