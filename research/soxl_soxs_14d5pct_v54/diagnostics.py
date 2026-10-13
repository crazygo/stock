"""Preregistered clock coverage, prediction errors and executable increments."""
import argparse
from dense_learning import *


def table(path):
    try:return pd.read_csv(path)
    except pd.errors.EmptyDataError:return pd.DataFrame()


def diagnose(run):
    run=Path(run);r=read(run/'results.json');v=read(run/'verification.json');assert v['passed'] and v['results_sha256']==sha(run/'results.json')
    p=read(run/'registration.json')['preregistration']['protocol'];settings=p[r['phase']];panel=pd.read_parquet(run/'financial_panel.parquet');cutoff=pd.Timestamp(read(run/'input_manifest.json')['observation_cutoff_exclusive'])
    rows=[];unknown=[]
    for f in sorted((run/'training').glob('*.json')):
        rec=read(f)
        if not settings['evaluation_start']<=rec['date'][:10]<=settings['evaluation_end']:continue
        at=pd.Timestamp(rec['at']);row=panel.loc[at];raw=rec['fit']['raw'];est=rec['estimate']
        if not raw or not est['ready']:continue
        if pd.isna(row.terminal_release) or row.terminal_release>=cutoff or not np.isfinite(row.C) or not np.isfinite(row.D):unknown.append(at);continue
        pair=np.array([row.C+row.D,row.C-row.D]);pred=np.array(est['point']);original=np.array([raw[s] for s in core.ETFS]);m=abs(row.D)
        rows.append({'at':at,'date':row.date,'clock':row.clock,'group':'old' if row.clock in OLD_CLOCKS else 'added','raw_pair_mse':np.mean((original-pair)**2),'calibrated_pair_mse':np.mean((pred-pair)**2),
            'raw_C_mse':(raw['C']-row.C)**2,'raw_D_mse':(raw['D']-row.D)**2,'magnitude':m,'brier_numerator':m*(raw['q']-int(row.D>0))**2})
    frame=pd.DataFrame(rows);metrics={}
    for name,selected in [('all',frame),('old',frame[frame.group=='old']),('added',frame[frame.group=='added']),*[(c,frame[frame.clock==c]) for c in CLOCKS]]:
        d=selected.groupby('date').mean(numeric_only=True);vals=d.mean().to_dict();vals['magnitude_weighted_brier']=float(d.brier_numerator.mean()/d.magnitude.mean()) if len(d) and d.magnitude.sum()>0 else None
        metrics[name]={'mature_queries':len(selected),'dates':len(d),'date_equal':vals}
    routes={};events=read(run/'decisions_frozen.json')['routes']
    for name,es in events.items():
        decisions=table(run/(name+'_decisions.csv'));trades=table(run/(name+'_trades.csv'));proposals=[e for e in es if e['action']!='CASH'];byclock={}
        for c in CLOCKS:
            ds=decisions[decisions.clock==c];ts=trades[pd.to_datetime(trades.decision_at,utc=True).dt.tz_convert(core.ET).dt.strftime('%H:%M:%S')==c] if len(trades) else trades
            byclock[c]={'proposals':int(ds.requested_action.ne('CASH').sum()),'actual_buys':len(ts),'target_exits':int(ts.reason.eq('target').sum()) if len(ts) else 0,'stop_exits':int(ts.reason.eq('risk_stop').sum()) if len(ts) else 0,
                'blocked_proposals':ds.loc[ds.requested_action.ne('CASH')&~ds.decision.str.startswith('buy_'),'decision'].value_counts().to_dict()}
        routes[name]={'proposal_count':len(proposals),'proposal_dates':len(set(e['date'] for e in proposals)),'actual_trades':len(trades),'clocks':byclock}
    increments={};primary=table(run/'dense_band_trades.csv');entryset=lambda t:{(x.decision_at,x.symbol) for x in t.itertuples()};a=entryset(primary)
    for name in ['old_four_band','dense_four_only','added_only']:
        other=table(run/(name+'_trades.csv'));b=entryset(other)
        details=lambda t,keys:[{'at':x.decision_at,'symbol':x.symbol,'reason':x.reason,'net_return':x.net_return} for x in t.itertuples() if (x.decision_at,x.symbol) in keys]
        increments[name]={'added_actual_buys':details(primary,a-b),'removed_actual_buys':details(other,b-a),'common_actual_buys':len(a&b),
            'primary_minus_control_monthly_mean':r['primary']['primary_average']-r['routes'][name]['primary_average'],'paired_capital_interval':r['paired_increments'][name]}
    out={'post_result_only':True,'rules_changed':False,'metrics':metrics,'unknown_queries_retained':unknown,'routes':routes,'increments':increments,'same_day_clocks_not_independent':True,'results_sha256':sha(run/'results.json')}
    dump(run/'diagnostics.json',out);print({'metrics':metrics,'route_counts':{k:{x:z[x] for x in ['proposal_count','proposal_dates','actual_trades']} for k,z in routes.items()}},flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);diagnose(p.parse_args().run)
