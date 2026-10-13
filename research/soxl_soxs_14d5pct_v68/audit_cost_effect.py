"""Post-result attribution of an execution-cost stress book, not policy selection."""
import argparse
from path_learning import *


def audit(run):
    run=Path(run);v=read(run/'verification.json');assert v['passed'] and v['results_sha256']==sha(run/'results.json')
    books={}
    for name in ['scaled_point','double_cost']:
        frame=pd.read_csv(run/(name+'_trades.csv')).set_index(['decision_at','symbol']);assert frame.index.is_unique;books[name]=frame
    a,b=books['scaled_point'],books['double_cost'];shared=a.index.intersection(b.index);x,y=a.loc[shared],b.loc[shared]
    known=x.net_return.notna()&y.net_return.notna();difference=y.loc[known,'net_return']-x.loc[known,'net_return'];fields=['entry_fill_economic','stop_economic','exit_at','exit_fill_economic','net_return','gap_below_stop','reason'];improved=[]
    for at,symbol in difference[difference>1e-12].index:
        improved.append({'at':at,'symbol':symbol,'base':a.loc[(at,symbol),fields].to_dict(),'double_cost':b.loc[(at,symbol),fields].to_dict()})
    result={'passed':True,'post_result_diagnostic':True,'registered_before_capital_results':False,'strategy_changed':False,'base_entries':len(a),'double_cost_entries':len(b),'shared_entry_events':len(shared),'shared_known_outcomes':int(known.sum()),'unknown_shared_retained':int((~known).sum()),'same_exit_reason':int((x.reason==y.reason).sum()),'only_base_events':[{'at':at,'symbol':s} for at,s in a.index.difference(b.index)],'only_double_cost_events':[{'at':at,'symbol':s} for at,s in b.index.difference(a.index)],'mean_net_return_difference_on_shared_known_events':float(difference.mean()),'shared_events_more_profitable_under_stress':improved,'scope':'Both complete books independently verified. Extra costs change entry/stop/target prices and exit time; account sequencing may then differ. This is not a frozen-fill fee subtraction, and higher stress-book return is not evidence costs are beneficial. No future label becomes a decision rule.','results_sha256':sha(run/'results.json'),'verification_sha256':sha(run/'verification.json'),'source_sha256':sha(Path(__file__))}
    dump(run/'cost_effect_audit.json',result);print({k:v for k,v in result.items() if k!='shared_events_more_profitable_under_stress'},flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);audit(p.parse_args().run)
