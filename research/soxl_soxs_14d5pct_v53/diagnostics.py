"""Post-result date-equal errors and executable selection; no rule changes."""
import argparse
from collections import Counter,defaultdict
from curve_reliability import *


def diagnose(run):
    run=Path(run);r=read(run/'results.json');v=read(run/'verification.json');assert v['passed'] and v['results_sha256']==sha(run/'results.json')
    p=read(run/'registration.json')['preregistration']['protocol'];settings=p[r['phase']];root=Path(read(run/'input_manifest.json')['source'])
    panel=pd.read_parquet(root/'financial_panel.parquet');part=panel[panel.date.between(settings['evaluation_start'],settings['evaluation_end'])]
    cutoff=pd.Timestamp(read(root/'input_manifest.json')['observation_cutoff_exclusive']);rows={k:[] for k in VARIANTS};unknown=[]
    for at,row in part.iterrows():
        rec=read(run/'training'/(tag(at)+'.json'));known=pd.notna(row.terminal_release) and row.terminal_release<cutoff and np.isfinite(row.C) and np.isfinite(row.D)
        if not known:unknown.append(at)
        for k in VARIANTS:
            raw=rec['raws'][k];est=rec['estimates'][k]
            if raw is None or not known or not est['ready']:continue
            truth=np.array([row.C+row.D,row.C-row.D]);out={'date':row.date,'magnitude':abs(row.D),'weighted_brier_num':abs(row.D)*(rec['fits'][k]['raw']['q']-int(row.D>0))**2}
            for field,y in [('raw',raw),('calibrated',est['point'])]:
                y=np.array(y);j=int(y[1]>y[0]);out.update({field+'_mse':float(np.mean((y-truth)**2)),field+'_chosen_log':float(truth[j]),field+'_positive':int(truth[j]>0),field+'_C_mse':float((y.mean()-row.C)**2),field+'_D_mse':float(((y[0]-y[1])/2-row.D)**2)})
            rows[k].append(out)
    metrics={}
    for k,rs in rows.items():
        frame=pd.DataFrame(rs);daily=frame.groupby('date').mean(numeric_only=True);stats=daily.mean().to_dict();stats['weighted_brier']=stats['weighted_brier_num']/stats['magnitude']
        metrics[k]={'queries':len(rs),'dates':len(daily),'date_equal':stats}
    events=read(run/'decisions_frozen.json')['routes'];selection={};trade_sets={};proposal_sets={}
    for name in ['curve_band','curve_point','linear_band','global_band','neutral_band','direction_only_band','baseline_band']:
        proposals=[e for e in events[name] if e['action']!='CASH'];proposal_sets[name]={(e['at'],e['action']) for e in proposals}
        decisions=pd.read_csv(run/(name+'_decisions.csv'));decisions.index=pd.to_datetime(decisions.decision_at,utc=True)
        try:trades=pd.read_csv(run/(name+'_trades.csv'))
        except pd.errors.EmptyDataError:trades=pd.DataFrame()
        trade_sets[name]={(t.decision_at,t.symbol):{'at':t.decision_at,'symbol':t.symbol,'reason':t.reason,'net_return':t.net_return} for t in trades.itertuples()}
        grouped=defaultdict(list)
        for e in proposals:grouped[e['date']].append(e)
        groups={'all':proposals,'first_of_date':[sorted(es,key=lambda x:x['at'])[0] for es in grouped.values()],
                'actual_buy':[e for e in proposals if decisions.loc[pd.Timestamp(e['at']),'decision'].startswith('buy_')],
                'unavailable':[e for e in proposals if not decisions.loc[pd.Timestamp(e['at']),'decision'].startswith('buy_')]}
        output={}
        for kind,es in groups.items():
            ds=defaultdict(list);pending=[];records=[]
            for e in es:
                at=pd.Timestamp(e['at']);row=panel.loc[at]
                if pd.isna(row.terminal_release) or row.terminal_release>=cutoff or not np.isfinite(row.C) or not np.isfinite(row.D):pending.append(e['at']);continue
                value=float(row.C+(row.D if e['action']=='SOXL' else -row.D));ds[e['date']].append(value)
                records.append({'at':e['at'],'date':e['date'],'symbol':e['action'],'net_log':value,'status':decisions.loc[at,'decision']})
                if kind=='actual_buy':np.testing.assert_allclose(np.log1p(trade_sets[name][(e['at'],e['action'])]['net_return']),value,atol=1e-12)
            output[kind]={'queries':len(es),'known_dates':len(ds),'unknown':pending,'date_equal_log':float(np.mean([np.mean(x) for x in ds.values()])) if ds else None,
                          'date_equal_positive':float(np.mean([np.mean(np.array(x)>0) for x in ds.values()])) if ds else None,'statuses':dict(Counter(z['status'] for z in records)),'rows':records}
        assert len(groups['actual_buy'])==len(trades);selection[name]=output
    changes={}
    for name in ['curve_band','curve_point','linear_band','global_band','neutral_band','direction_only_band']:
        current=trade_sets[name];old=trade_sets['baseline_band'];changes[name]={'common_proposals':len(proposal_sets[name]&proposal_sets['baseline_band']),
            'added_proposals':len(proposal_sets[name]-proposal_sets['baseline_band']),'removed_proposals':len(proposal_sets['baseline_band']-proposal_sets[name]),
            'added_actual':[current[k] for k in sorted(current.keys()-old.keys())],'removed_actual':[old[k] for k in sorted(old.keys()-current.keys())]}
    receipt={'post_result_only':True,'rules_changed':False,'unknown_queries_retained':unknown,'maturity_cutoff':cutoff,'metrics':metrics,
        'selection':selection,'changes':changes,'option_counts_do_not_identify_trade_buyer_seller_or_net_position':True,
        'results_sha256':sha(run/'results.json')}
    dump(run/'diagnostics.json',receipt)
    print({'metrics':metrics,'selection':{n:{k:{f:v for f,v in x.items() if f!='rows'} for k,x in gs.items()} for n,gs in selection.items()}},flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);diagnose(p.parse_args().run)
