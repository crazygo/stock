"""Post-result counterfactual path attribution; no policy selection."""
import argparse
from overnight import *
from verify_continuation import near


def diagnose(run):
    run=Path(run);assert read(run/'verification.json')['passed']
    base=labels(run/'baseline_labels.json');managed=read_managed(run/'buffer_labels.json')
    preds=read(run/'predictions.json');months=['2026-01','2026-02','2026-03','2026-04']
    reviews=read(run/'buffer_reviews.json');gaps=read(run/'gap_forecasts.json');development=lambda t:'2026-01-01'<=str(t)[:10]<='2026-04-30'
    distribution={}
    for symbol in core.ETFS:
        rows=[r for r in gaps if r['symbol']==symbol and development(r['date'])];vals=[r['adverse_tail_mean'] for r in rows if r['ready']]
        distribution[symbol]={'dates':len(rows),'ready_dates':len(vals),'tail_min':min(vals,default=None),
            'tail_median':float(np.median(vals)) if vals else None,'tail_max':max(vals,default=None)}
    choices={};paths={};daily=[]
    flat=read_managed(run/'flat_labels.json');identical=0
    for k,a in managed.items():
        if not development(a['decision_at']):continue
        b=flat[k]
        for field in ['exit_at','reason']:assert a.get(field)==b.get(field)
        for field in ['exit_fill_economic','net_return']:
            if a.get(field) is None:assert b.get(field) is None
            else:near(a[field],b[field])
        identical+=1
    rdev=[r for r in reviews if development(r['at'])]
    review_actions={a:sum(r['action']==a for r in rdev) for a in ['HOLD','EXIT']}

    for name in preds:
        chosen=[e for e in preds[name] if development(e['date'])];choices[name]={s:sum(e['action']==s for e in chosen) for s in [*core.ETFS,'CASH']}
    for name in ['buffer_band','buffer_point','source_v29_buffer']:
        try:trades=pd.read_csv(run/(name+'_trades.csv'))
        except pd.errors.EmptyDataError:trades=pd.DataFrame()
        records=[]
        for row in trades.itertuples():
            at=pd.Timestamp(row.decision_at);a=managed[key(at,row.symbol)];b=base[key(at,row.symbol)]
            rs=[r for r in reviews if pd.Timestamp(r['decision_at'])==at and r['symbol']==row.symbol]
            near(row.net_return,a['net_return'])
            records.append({'decision_at':at,'symbol':row.symbol,'entry_at':a['entry_at'],'managed_exit':a['exit_at'],
                'managed_reason':a['reason'],'managed_net':a['net_return'],'fixed_exit':b['exit_at'],'fixed_reason':b['reason'],
                'fixed_net':b['net_return'],'difference':a['net_return']-b['net_return'] if b.get('net_return') is not None else None,
                'hold_reviews':sum(r['action']=='HOLD' for r in rs),'exit_reviews':sum(r['action']=='EXIT' for r in rs),
                'later_target_cut':bool(a['reason']=='preclose_exit' and b['reason']=='target'),
                'fixed_loss_reduced':bool(a['reason']=='preclose_exit' and b.get('net_return') is not None and b['net_return']<0 and a['net_return']>b['net_return']),
                'reviews':rs})
        paths[name]={'trades':len(records),'later_targets_cut':sum(r['later_target_cut'] for r in records),
            'fixed_losses_reduced':sum(r['fixed_loss_reduced'] for r in records),'records':records}
    for f in sorted((run/'training').glob('*.json')):
        rec=read(f)
        if not development(rec['date']):continue
        fb=rec['feedback'];daily.append({'at':rec['at'],'clock':rec['clock'],'parents':len(rec['fit']['members']),
            'mature_prefix':fb['selected_early_dates'],'expired':fb['expired_unresolved_prefix'],
            'ready':rec['estimate']['ready'],'action':rec['band']['action']})
    out={'passed':True,'post_result_only':True,'risk_tail_distribution':distribution,'candidate_actions':choices,
        'development_reviews':review_actions,'development_label_outcomes_identical_to_flat':identical,'actual_entry_counterfactuals':paths,'training_readiness':daily,
        'same_candidate_actions_do_not_mean_same_later_actual_entries':True,
        'counterfactual_return_differences_are_not_monthly_capital_returns':True,
        'results_sha256':sha(run/'results.json'),'verification_sha256':sha(run/'verification.json'),'source_sha256':sha(Path(__file__))}
    dump(run/'diagnostics.json',out)
    print({'risk':distribution,'paths':{n:{k:v for k,v in x.items() if k!='records'} for n,x in paths.items()}},flush=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);diagnose(p.parse_args().run)
