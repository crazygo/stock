"""Recompute selection-subset evidence independently of report aggregation."""
import argparse
from selected_risk import Path,np,pd,read,dump,sha,tag
from verify_continuation import near


def audit(run):
    run=Path(run);d=read(run/'diagnostics.json');panel=pd.read_parquet(run/'selected_panel.parquet')
    source=Path(read(run/'input_manifest.json')['verified_source']);checks=0
    for row in d['daily_training_subsets']:
        if not row['ready']:continue
        day=read(run/'daily'/(row['date']+'.json'));ids=list(pd.to_datetime(day['members'],utc=True));w=day['row_weights']
        samples=[]
        for t,weight in zip(ids,w):
            r=read(source/'training'/(tag(t)+'.json'));samples.append((t,weight,float(panel.loc[t,'selected_y']),r['events']['full_band']['action']!='CASH'))
        for name,stats in row['training_subsets'].items():
            selected=[r for r in samples if name=='all' or r[3]==(name=='source_eligible')]
            assert stats['rows']==len(selected);assert stats['dates']==len(set(panel.loc[r[0],'date'] for r in selected))
            near(stats['weighted_mean_netlog'],sum(w*y for _,w,y,_ in selected)/sum(w for _,w,_,_ in selected))
            near(stats['positive_gain_share'],sum(w*max(y,0) for _,w,y,_ in selected)/sum(w*abs(y) for _,w,y,_ in selected));checks+=4
    old=pd.read_csv(run/'source_v35_trades.csv');new=pd.read_csv(run/'context_gate_trades.csv')
    oldkeys=set(zip(old.decision_at,old.symbol));newkeys=set(zip(new.decision_at,new.symbol));assert newkeys-oldkeys==set() and d['actual_added']==[]
    assert set((r['decision_at'],r['symbol']) for r in d['actual_removed'])==oldkeys-newkeys
    active=[]
    for t,row in panel[panel.date.between('2026-01-01','2026-04-30')].iterrows():
        issued=read(run/'training'/(tag(t)+'.json'));v=issued['estimates']['context']
        if not v['ready'] or v['source_fallback'] or not np.isfinite(row.selected_y):continue
        origin=read(source/'training'/(tag(t)+'.json'));active.append({'day':row.date,'y':row.selected_y,'eligible':origin['events']['full_band']['action']!='CASH',
            **{k:issued['estimates'][k]['quality'] for k in ['context','scores','frequency']}})
    for name,metric in d['errors'].items():
        rows=[r for r in active if name=='all_active_top' or r['eligible']==(name=='source_eligible')];days=sorted(set(r['day'] for r in rows))
        for kind,stats in metric.items():
            sq=ll=mass=win=ret=qmean=0.
            for day in days:
                g=[r for r in rows if r['day']==day];n=len(g)
                for r in g:
                    y=r['y'];q=r[kind];positive=int(y>0);m=abs(y)
                    sq+=m*(q-positive)**2/n;ll+=m*(-positive*np.log(max(q,1e-15))-(1-positive)*np.log(max(1-q,1e-15)))/n
                    mass+=m/n;win+=positive/n;ret+=y/n;qmean+=q/n
            assert stats['queries']==len(rows) and stats['dates']==len(days)
            for field,value in [('magnitude_weighted_brier',sq/mass),('magnitude_weighted_logloss',ll/mass),('actual_win_fraction_date_equal',win/len(days)),('actual_mean_netlog_date_equal',ret/len(days)),('mean_quality_date_equal',qmean/len(days))]:near(stats[field],value)
            checks+=7
    receipt={'passed':True,'independent_diagnostic_assertions':checks,'removed_trade_keys':len(oldkeys-newkeys),'added_trade_keys':len(newkeys-oldkeys),
        'diagnostics_sha256':sha(run/'diagnostics.json'),'results_sha256':sha(run/'results.json'),'audit_source_sha256':sha(Path(__file__))}
    dump(run/'diagnostics_audit.json',receipt);print(receipt)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);audit(p.parse_args().run)
