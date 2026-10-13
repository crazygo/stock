"""Post-verification numerical decomposition, not a strategy selection rule."""
import argparse
from path_learning import *


def report(run):
    run=Path(run);results=read(run/'results.json');v=read(run/'verification.json');assert v['passed'] and v['results_sha256']==sha(run/'results.json')
    p=read(SOURCE/'protocol.json')[results['phase']];panel=pd.read_parquet(run/'activity_panel.parquet');rows=[]
    for at,row in panel[panel.date.between(p['evaluation_start'],p['evaluation_end'])].iterrows():
        r=read(run/'training'/(tag(at)+'.json'));a,b=r['fits']['conditional'],r['fits']['ordinary']
        if not a['ready']:continue
        assert b['ready'] and a['members']==b['members'] and a['kernels']==b['kernels']
        change=np.array(r['raws']['conditional'])-np.array(r['raws']['ordinary'])
        rows.append({'at':at,'date':row.date,'resolved_probability':1-a['partition']['unresolved_probability'],
            'new_trace':float(np.trace(a['integration']['mean_covariance'])),'ordinary_trace':float(np.trace(b['integration']['mean_covariance'])),
            'raw_difference':change,'changed_action':r['events']['conditional_band']['action']!=r['events']['ordinary_band']['action'],
            'enumerated_contracts':2*len(a['members']),'tail_contracts':2*a['paths']})
    frame=pd.DataFrame(rows);daily=frame.groupby('date')[['resolved_probability','new_trace','ordinary_trace','changed_action']].mean()
    out={'queries':len(rows),'real_dates':len(daily),'date_equal_resolved_probability':float(daily.resolved_probability.mean()),
         'date_equal_reported_covariance_trace_ratio':float(daily.new_trace.mean()/daily.ordinary_trace.mean()),
         'changed_proposed_actions':int(frame.changed_action.sum()),'enumerated_contracts':int(frame.enumerated_contracts.sum()),'tail_contracts':int(frame.tail_contracts.sum()),
         'raw_difference_mean':np.mean(frame.raw_difference.to_list(),axis=0),'maximum_absolute_raw_difference':float(np.max(np.abs(frame.raw_difference.to_list()))),
         'primary_month_mean':results['routes']['conditional_band']['primary_average'],'ordinary_month_mean':results['routes']['ordinary_band']['primary_average'],
         'numerical_precision_does_not_establish_prediction_or_returns':True,'results_sha256':sha(run/'results.json'),'rows':rows}
    dump(run/'integration_diagnostics.json',out);print({k:v for k,v in out.items() if k!='rows'},flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);report(p.parse_args().run)
