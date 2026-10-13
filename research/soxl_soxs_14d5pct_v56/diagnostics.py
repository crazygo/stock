"""Preregistered errors, actual increments, and own-feature participation."""
import argparse
from activity_learning import *


def table(path):
    try:return pd.read_csv(path)
    except pd.errors.EmptyDataError:return pd.DataFrame()


def diagnose(run):
    run=Path(run);r=read(run/'results.json');v=read(run/'verification.json');assert v['passed'] and v['results_sha256']==sha(run/'results.json')
    p=read(run/'registration.json')['preregistration']['protocol'];settings=p[r['phase']];panel=pd.read_parquet(run/'activity_panel.parquet');cutoff=pd.Timestamp(read(run/'input_manifest.json')['observation_cutoff_exclusive']);source,_=verified_source()
    rows=[];unknown=[];changes={k:{'raw_queries_changed':0,'calibrated_queries_changed':0,'q_max_abs_change':0.,'new_q_coefficient_max_abs':0.} for k in VARIANTS}
    for file in sorted((run/'training').glob('*.json')):
        rec=read(file)
        if not settings['evaluation_start']<=rec['date'][:10]<=settings['evaluation_end']:continue
        at=pd.Timestamp(rec['at']);truth=panel.loc[at];baseline=rec['baseline']['fit']['raw'];oldfile=source/'training'/file.name;old=read(oldfile) if oldfile.exists() else None
        mature=pd.notna(truth.terminal_release) and truth.terminal_release<cutoff and np.isfinite(truth.C) and np.isfinite(truth.D)
        if not mature:unknown.append(at)
        for k in VARIANTS+['baseline']:
            raw=baseline if k=='baseline' else rec['fits'][k]['raw'];estimate=rec['baseline']['estimate'] if k=='baseline' else rec['estimates'][k]
            if k!='baseline' and raw and baseline:
                delta=max(abs(raw[s]-baseline[s]) for s in core.ETFS);changes[k]['raw_queries_changed']+=int(delta>1e-12);changes[k]['q_max_abs_change']=max(changes[k]['q_max_abs_change'],abs(raw['q']-baseline['q']))
                head=rec['fits'][k]['fit']['heads'].get('q');extra=len(FEATURES if k=='full' else PAIR if k=='pair' else SECTOR)
                if head:changes[k]['new_q_coefficient_max_abs']=max(changes[k]['new_q_coefficient_max_abs'],float(np.max(abs(np.array(head['coefficients'])[0,-extra:]))))
                if old and estimate['ready'] and old['estimate']['ready']:changes[k]['calibrated_queries_changed']+=int(max(abs(a-b) for a,b in zip(estimate['point'],old['estimate']['point']))>1e-12)
            if not raw or not mature:continue
            pair=np.array([truth.C+truth.D,truth.C-truth.D]);pred=np.array([raw[s] for s in core.ETFS]);m=abs(truth.D)
            rows.append({'kind':k,'date':truth.date,'clock':truth.clock,'raw_pair_mse':float(np.mean((pred-pair)**2)),'calibrated_pair_mse':float(np.mean((np.array(estimate['point'])-pair)**2)) if estimate and estimate['ready'] else np.nan,'C_mse':(raw['C']-truth.C)**2,'D_mse':(raw['D']-truth.D)**2,'magnitude':m,'brier_numerator':m*(raw['q']-int(truth.D>0))**2})
    frame=pd.DataFrame(rows);metrics={}
    for k,g in frame.groupby('kind'):
        daily=g.groupby('date').mean(numeric_only=True);stats=daily.mean().to_dict();stats['magnitude_weighted_brier']=float(daily.brier_numerator.mean()/daily.magnitude.mean());metrics[k]={'queries':len(g),'dates':len(daily),'date_equal':stats}
    events=read(run/'decisions_frozen.json')['routes'];routes={}
    for name,es in events.items():
        proposals=[e for e in es if e['action']!='CASH'];trades=table(run/(name+'_trades.csv'));decisions=table(run/(name+'_decisions.csv'));clocks={}
        for clock in CLOCKS:
            ds=decisions[decisions.clock==clock];ts=trades[pd.to_datetime(trades.decision_at,utc=True).dt.tz_convert(core.ET).dt.strftime('%H:%M:%S')==clock] if len(trades) else trades
            clocks[clock]={'proposals':int(ds.requested_action.ne('CASH').sum()),'actual_trades':len(ts),'target_exits':int(ts.reason.eq('target').sum()) if len(ts) else 0,'stop_exits':int(ts.reason.eq('risk_stop').sum()) if len(ts) else 0,'blocked':ds.loc[ds.requested_action.ne('CASH')&~ds.decision.str.startswith('buy_'),'decision'].value_counts().to_dict()}
        routes[name]={'proposals':len(proposals),'proposal_dates':len(set(e['date'] for e in proposals)),'actual_trades':len(trades),'clocks':clocks}
    primary=table(run/'activity_band_trades.csv');entryset=lambda t:{(x.decision_at,x.symbol) for x in t.itertuples()};a=entryset(primary);increments={}
    for name in ['source_v55','pair_band','sector_band','four_only']:
        control=table(run/(name+'_trades.csv'));b=entryset(control);details=lambda t,keys:[{'at':x.decision_at,'symbol':x.symbol,'reason':x.reason,'net_return':x.net_return} for x in t.itertuples() if (x.decision_at,x.symbol) in keys]
        increments[name]={'added_actual_buys':details(primary,a-b),'removed_actual_buys':details(control,b-a),'common_actual_buys':len(a&b),'primary_minus_control_monthly_mean':r['primary']['primary_average']-r['routes'][name]['primary_average'],'paired_capital_interval':r['paired_increments'][name]}
    receipt={'post_result_only':True,'rules_changed':False,'metrics':metrics,'unknown_queries_retained':unknown,'feature_participation':changes,'routes':routes,'increments':increments,'results_sha256':sha(run/'results.json')}
    dump(run/'diagnostics.json',receipt);print({'metrics':metrics,'feature_participation':changes,'unknown_queries':len(unknown),'actual_trades':{k:v['actual_trades'] for k,v in routes.items()}},flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);diagnose(p.parse_args().run)
