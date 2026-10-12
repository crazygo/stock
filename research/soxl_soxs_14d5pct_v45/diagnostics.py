"""Post-result forecast and execution comparisons; no candidate promotion here."""
import argparse
from collections import Counter
from adaptive_penalty import *


def diagnose(run):
    run=Path(run);r=read(run/'results.json');v=read(run/'verification.json');assert v['passed'] and v['results_sha256']==sha(run/'results.json')
    p=read(run/'registration.json')['preregistration']['protocol'];settings=p[r['phase']];panel=pd.read_parquet(run/'macro_panel.parquet');ev=panel[panel.date.between(settings['evaluation_start'],settings['evaluation_end'])];cutoff=pd.Timestamp(read(run/'input_manifest.json')['market']['observation_cutoff_exclusive'])
    ds={k:pd.read_csv(run/(k+'_band_decisions.csv')) for k in CANDIDATES+['adaptive']}
    for d in ds.values():d.index=pd.to_datetime(d.decision_at,utc=True)
    rows=[];days=[]
    for day,part in ev.groupby('date'):
        daily=read(run/'daily'/(str(day.date())+'.json'));days.append({'date':day,'selector_ready':daily['selector_ready'],'selector_dates':len(daily['selector_parents']),'selected':daily['selected'],'losses':daily['losses'],'bootstrap_choices':dict(Counter(daily['bootstrap_selected']))})
        for at,row in part.iterrows():
            rec=read(run/'training'/(tag(at)+'.json'));known=pd.notna(row.terminal_release) and row.terminal_release<cutoff and np.isfinite(row.C) and np.isfinite(row.D)
            for k in CANDIDATES+['adaptive']:
                est=rec['estimates'][k];assert est['ready'];point=np.array(est['point']);j=int(point[1]>point[0]);raw=np.array(rec['raw']);truth=np.array([row.C+row.D,row.C-row.D]) if known else None
                action=rec['events'][k+'_band']['action'];status=ds[k].loc[at,'decision'];actual=status.startswith('buy_');proposed=action!='CASH';assert not actual or proposed
                if proposed:assert action==core.ETFS[j]
                rows.append({'date':day,'at':at,'kind':k,'proposed':proposed,'actual':actual,'status':status,'symbol':core.ETFS[j],'known':known,
                    'raw':raw[j],'point':point[j],'radius':est['radius'],'truth':truth[j] if known else None,'overprediction':point[j]-truth[j] if known else None,
                    'mse':float(np.mean((point-truth)**2)) if known else None,'raw_mse':float(np.mean((raw-truth)**2)) if known else None})
    f=pd.DataFrame(rows);summary={};changed={}
    for k in CANDIDATES+['adaptive']:
        q=f[f.kind==k];summary[k]={}
        for name,g in [('all',q),('proposed',q[q.proposed]),('actual',q[q.actual]),('unavailable',q[q.proposed&~q.actual])]:
            known=g[g.known];summary[k][name]={'queries':len(g),'known_dates':known.date.nunique(),'unknown':g.loc[~g.known,'at'].tolist(),'date_equal':known.groupby('date')[['raw','point','radius','truth','overprediction','mse','raw_mse']].mean().mean().to_dict(),'statuses':dict(Counter(g.status))}
    trades={k:{(t.decision_at,t.symbol):{'at':t.decision_at,'symbol':t.symbol,'reason':t.reason,'net_return':t.net_return} for t in pd.read_csv(run/(k+'_band_trades.csv')).itertuples()} for k in ['adaptive','strong','plain']}
    for k in ['adaptive','strong']:
        new=trades[k];old=trades['plain'];changed[k]={'common_actual':len(new.keys()&old.keys()),'added_actual':[new[t] for t in sorted(new.keys()-old.keys())],'removed_actual':[old[t] for t in sorted(old.keys()-new.keys())]}
    f.to_csv(run/'diagnostic_queries.csv',index=False)
    out={'post_result_only':True,'new_primary_selected':False,'rules_changed':False,'selection_days':days,'selection_counts':dict(Counter(d['selected'] for d in days)),'learned_selection_counts':dict(Counter(d['selected'] for d in days if d['selector_ready'])),'warmup_dates':sum(not d['selector_ready'] for d in days),'summary':summary,'changes_vs_plain':changed,'results_sha256':sha(run/'results.json')}
    dump(run/'diagnostics.json',out);print({'selection_counts':out['selection_counts'],'learned_selection_counts':out['learned_selection_counts'],'summary':summary},flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);diagnose(p.parse_args().run)
