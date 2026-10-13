"""Post-result diagnosis of executed direction changes; never alters decisions."""
import argparse
from path_learning import *
from verify_continuation import raw_exit,key


def check_primary_entry(symbol,point):
    j=core.ETFS.index(symbol)
    assert symbol==core.ETFS[int(np.argmax(point))] and point[j]>1e-12


def audit(run):
    run=Path(run);r=read(run/'results.json');v=read(run/'verification.json');assert v['passed'] and v['results_sha256']==sha(run/'results.json')
    m=read(run/'input_manifest.json');market=core.load_market();assert market.manifest==m['full_market'];market=restrict_market(market,pd.Timestamp(m['observation_cutoff_exclusive']))
    lab=labels(run/'baseline_labels.json');rows=[];path_checks=0
    try:trades=pd.read_csv(run/'dynamic_point_trades.csv')
    except pd.errors.EmptyDataError:trades=pd.DataFrame()
    for t in trades.itertuples():
        at=pd.Timestamp(t.decision_at);rec=read(run/'training'/(tag(at)+'.json'));raw=rec['fits']['dynamic']['raw'];point=rec['estimates']['dynamic']['point'];lower=rec['estimates']['dynamic']['lower'];other=next(s for s in core.ETFS if s!=t.symbol)
        actual=lab[key(at,t.symbol)];counter=lab[key(at,other)]
        for outcome in [actual,counter]:
            if outcome.get('exit_at') is not None and outcome.get('net_return') is not None:raw_exit(market,outcome,None,1);path_checks+=1
        truth={s:lab[key(at,s)].get('net_return') for s in core.ETFS};known=all(z is not None and np.isfinite(z) for z in truth.values());raw_top=core.ETFS[0] if raw[core.ETFS[0]]>=raw[core.ETFS[1]] else core.ETFS[1];j=core.ETFS.index(t.symbol)
        check_primary_entry(t.symbol,point)
        if actual.get('net_return') is None:assert pd.isna(t.net_return)
        else:np.testing.assert_allclose(t.net_return,actual['net_return'],rtol=1e-10,atol=1e-10)
        rows.append({'at':at,'symbol':t.symbol,'reason':t.reason,'net_return':t.net_return,'gap_below_stop':bool(t.gap_below_stop) if pd.notna(t.gap_below_stop) else None,'counterfactual_other':other,'counterfactual_other_return':truth[other],
                     'raw_top':raw_top,'matched_proposed_action':rec['events']['fresh_point']['action'],'matched_raw_pair_log_returns':rec['raws']['fresh'],'calibration_changed_direction':raw_top!=t.symbol,'raw_pair_log_returns':[raw[s] for s in core.ETFS],'point_pair_log_returns':point,'chosen_lower_log_return':lower[j],
                     'both_outcomes_known':known,'calibration_direction_helped':truth[t.symbol]>truth[raw_top]+1e-12 if known and raw_top!=t.symbol else None,
                     'calibration_direction_hurt':truth[t.symbol]<truth[raw_top]-1e-12 if known and raw_top!=t.symbol else None,
                     'other_profitable_on_loss':truth[other]>0 if known and t.net_return<0 else None})
    losses=[x for x in rows if x['net_return']<0];flips=[x for x in rows if x['calibration_changed_direction']]
    out={'passed':True,'diagnostic_registered_before_results':True,'rules_changed':False,'results_sha256':sha(run/'results.json'),'actual_trades':len(rows),'losing_trades':len(losses),'losses_raw_preferred_same_side':sum(x['raw_top']==x['symbol'] for x in losses),
         'losses_other_side_profitable':sum(bool(x['other_profitable_on_loss']) for x in losses if x['other_profitable_on_loss'] is not None),'calibration_direction_changes':len(flips),'calibration_helped':sum(bool(x['calibration_direction_helped']) for x in flips if x['calibration_direction_helped'] is not None),'calibration_hurt':sum(bool(x['calibration_direction_hurt']) for x in flips if x['calibration_direction_hurt'] is not None),
         'unknown_pairs_retained':sum(not x['both_outcomes_known'] for x in rows),'raw_path_checks':path_checks,'rows':rows,'scope':'Counterfactual labels at actual entry times are diagnostics, not a replayed alternative account or a then-known gate. No predictive effect claim.','source_sha256':sha(Path(__file__))}
    dump(run/'entry_direction_audit.json',out);print({k:v for k,v in out.items() if k!='rows'},flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);audit(p.parse_args().run)
