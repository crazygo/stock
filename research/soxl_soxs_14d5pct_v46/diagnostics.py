"""Post-result opportunity/entry diagnostics; no hindsight changes to the policy."""
import argparse
from collections import Counter
from fixed_candidate import *


def diagnose(run):
    run=Path(run);r=read(run/'results.json');v=read(run/'verification.json');assert v['passed'] and v['results_sha256']==sha(run/'results.json')
    p=read(run/'registration.json')['preregistration']['protocol'];settings=p[r['phase']];panel=pd.read_parquet(run/'macro_panel.parquet');ev=panel[panel.date.between(settings['evaluation_start'],settings['evaluation_end'])];cutoff=pd.Timestamp(read(run/'input_manifest.json')['market']['observation_cutoff_exclusive'])
    names=['strong_band','strong_point','plain_band','premarket_only'];decisions={n:pd.read_csv(run/(n+'_decisions.csv')) for n in names}
    for d in decisions.values():d.index=pd.to_datetime(d.decision_at,utc=True)
    trades={}
    for n in names:
        try:trades[n]={(t.decision_at,t.symbol):t for t in pd.read_csv(run/(n+'_trades.csv')).itertuples()}
        except pd.errors.EmptyDataError:trades[n]={}
    rows=[];forecast_rows=[]
    for at,row in ev.iterrows():
        rec=read(run/'training'/(tag(at)+'.json'));known=pd.notna(row.terminal_release) and row.terminal_release<cutoff and np.isfinite(row.C) and np.isfinite(row.D)
        truth=np.array([row.C+row.D,row.C-row.D]) if known else None
        for kind in CANDIDATES:
            e=rec['estimates'][kind];assert e['ready'];raw=np.array(rec['raw']);point=np.array(e['point']);j=int(point[1]>point[0])
            forecast_rows.append({'date':row.date,'at':at,'kind':kind,'known':known,'raw_mse':float(np.mean((raw-truth)**2)) if known else None,'mse':float(np.mean((point-truth)**2)) if known else None,
                'chosen_truth':float(truth[j]) if known else None,'chosen_point':float(point[j]),'chosen_raw':float(raw[j]),'radius':e['radius'],'lower':float(e['lower'][j])})
        for name in names:
            event=rec['events'][name];est=rec['estimates']['plain' if name=='plain_band' else 'strong'];action=event['action'];status=decisions[name].loc[at,'decision'];actual=status.startswith('buy_');proposed=action!='CASH'
            assert not actual or proposed
            if not proposed:continue
            j=core.ETFS.index(action);value=float(truth[j]) if known else None
            if actual and known:np.testing.assert_allclose(np.log1p(trades[name][(str(at),action)].net_return),value,atol=1e-12)
            rows.append({'name':name,'at':at,'date':row.date,'symbol':action,'status':status,'actual':actual,'known':known,'point':est['point'][j],'raw':rec['raw'][j],'radius':est['radius'],'net_log':value,'overprediction':est['point'][j]-value if known else None})
    f=pd.DataFrame(forecast_rows);proposals=pd.DataFrame(rows);stats={};selection={}
    for k in CANDIDATES:
        q=f[f.kind==k];m=q[q.known].groupby('date')[['raw_mse','mse','chosen_truth','chosen_point','chosen_raw','radius','lower']].mean()
        stats[k]={'queries':len(q),'known_queries':int(q.known.sum()),'known_dates':len(m),'unknown':q.loc[~q.known,'at'].tolist(),'date_equal':m.mean().to_dict()}
    months={}
    for n in names:
        q=proposals[proposals.name==n] if len(proposals) else proposals;selection[n]={}
        for group,g in [('proposed',q),('actual',q[q.actual] if len(q) else q),('unavailable',q[~q.actual] if len(q) else q)]:
            known=g[g.known] if len(g) else g;means=known.groupby('date')[['point','raw','radius','net_log','overprediction']].mean().mean().to_dict() if len(known) else None
            selection[n][group]={'queries':len(g),'known_dates':known.date.nunique() if len(known) else 0,'unknown':g.loc[~g.known,'at'].tolist() if len(g) else [],'date_equal':means,'statuses':dict(Counter(g.status)) if len(g) else {}}
        months[n]={}
        for m,g in ev.groupby(ev.date.dt.to_period('M')):
            rows_for_month=q[q.date.dt.to_period('M')==m] if len(q) else q
            months[n][str(m)]={'query_ticks':len(g),'proposals':len(rows_for_month),'actual_entries':int(rows_for_month.actual.sum()) if len(rows_for_month) else 0}
    new=trades['strong_band'];old=trades['plain_band'];pack=lambda t:{'at':t.decision_at,'symbol':t.symbol,'reason':t.reason,'net_return':t.net_return}
    changes={'common_actual':len(new.keys()&old.keys()),'added_actual':[pack(new[k]) for k in sorted(new.keys()-old.keys())],'removed_actual':[pack(old[k]) for k in sorted(old.keys()-new.keys())]}
    f.to_csv(run/'diagnostic_forecasts.csv',index=False);proposals.to_csv(run/'diagnostic_proposals.csv',index=False)
    receipt={'post_result_only':True,'rules_changed':False,'phase':r['phase'],'forecast_metrics':stats,'selection':selection,'monthly_opportunities':months,'changes_vs_plain':changes,'results_sha256':sha(run/'results.json')}
    dump(run/'diagnostics.json',receipt);print(receipt,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);diagnose(p.parse_args().run)
