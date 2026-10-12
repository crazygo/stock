"""Empirical coverage and actual-path differences; no parameter search or case scoring."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from uncertainty import core


def diagnose(run):
    run=Path(run);registration=json.loads((run/'registration.json').read_text());p=registration['preregistration']['protocol']
    phase=registration['phase'];start=pd.Timestamp(p[phase]['evaluation_start'],tz=core.ET)
    cutoff=pd.Timestamp(p[phase]['evaluation_end'],tz=core.ET)+pd.Timedelta(days=1)
    records={r['id']:r for r in json.loads((run/'review_forecasts.json').read_text())}
    bank=pd.read_parquet(run/'calibration_bank.parquet');events=json.loads((run/'manager_predictions.json').read_text())
    coverage={}
    for method in [p['primary'],p['diagnostic']]:
        selected=[e for e in events if e['policy']==method and e['ready'] and start<=pd.Timestamp(e['at'])<cutoff]
        measured=[];unknown=0
        for e in selected:
            r=records[e['id']];release=bank.loc[e['id'],'release']
            if pd.isna(release) or release>=cutoff or not np.isfinite(r['target']):unknown+=1;continue
            measured.append({'parent':r['entry_date'],'covered':r['target']<=e['bound'],'exit':e['action']=='EXIT'})
        frame=pd.DataFrame(measured)
        coverage[method]={'ready_review_rows':len(selected),'unknown_or_immature_rows_retained':unknown,
            'mature_parent_dates':frame.parent.nunique() if len(frame) else 0,
            'parent_weighted_fraction_below_bound':float(frame.groupby('parent').covered.mean().mean()) if len(frame) else None,
            'fraction_parents_with_all_issued_bounds_covering':float(frame.groupby('parent').covered.all().mean()) if len(frame) else None,
            'warning':'Empirical descriptive coverage; overlapping parent dates are not independent. Local mean bound is not an individual-return interval.'}
    a=pd.read_csv(run/(p['primary']+'_trades.csv'));b=pd.read_csv(run/'double_cost_trades.csv')
    keys=['entry_date','symbol'];cols=keys+['exit_at','reason','net_return']
    paired=a[cols].merge(b[cols],on=keys,how='outer',suffixes=('_base','_stress'))
    changes=paired[paired.exit_at_base.ne(paired.exit_at_stress)|paired.reason_base.ne(paired.reason_stress)]
    out={'phase':phase,'post_run_only':True,'models_and_thresholds_unchanged':True,'empirical_coverage':coverage,
        'stress_path_changes':changes.to_dict('records'),
        'stress_interpretation':'Higher entry cost changes cent-rounded attached stop/target prices. Earlier stop execution can avoid a later gap; cost sensitivity need not be monotone when the path changes. This does not make higher costs beneficial in general.',
        'case_months_computed':phase=='cases',
        'next_research':'Test entry-model information and calibration against simple no-model baselines on pre-case data; check history length and target alignment before adding more exit rules.'}
    core.dump(run/'diagnosis.json',out);print(json.dumps(out,indent=2));return out


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('run',type=Path)
    diagnose(parser.parse_args().run)
