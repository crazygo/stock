"""Post-result decomposition; these outputs cannot alter registered actions."""
import argparse
from collections import Counter
from adaptive_correction import core,np,pd,Path,read,dump,sha,components,tag,CLOCKS


def diagnose(run):
    run=Path(run);manifest=read(run/'input_manifest.json');root=Path(manifest['verified_source'])
    panel=pd.read_parquet(root/'financial_panel.parquet');rows=[];dates=[]
    for f in sorted((run/'daily').glob('*.json')):
        d=read(f)
        if '2026-01-01'<=d['date'][:10]<='2026-04-30':
            dates.append({'date':d['date'][:10],'ready':d['ready'],'meta_dates':len(d['parents']),
                'clock_alpha':d['models'].get('clock',{}).get('alpha'),
                'shared_alpha':d['models'].get('shared',{}).get('alpha')})
    for at,row in panel[panel.date.between('2026-01-01','2026-04-30')].iterrows():
        source=read(root/'training'/(tag(at)+'.json'));record=read(run/'training'/(tag(at)+'.json'));info=record['estimates']['clock']
        r={'at':at,'date':row.date,'clock':row.clock,'known':bool(np.isfinite([row.C,row.D]).all()),
            'source_fallback':info.get('source_fallback'),'alpha':info.get('alpha'),
            'source_radius':info.get('source_radius'),'joint_radius':info.get('joint_radius'),
            'floor_binding':info.get('source_radius',0)>=info.get('joint_radius',0),
            'source_action':source['events']['full_band']['action'],'new_action':record['events']['adaptive_band']['action']}
        if r['known']:
            truth=np.array([row.C,row.D]);experts={'original_raw':source['raws']['full'],'source_calibrated':source['estimates']['full']['point'],
                **{k:v['point'] for k,v in record['estimates'].items()}}
            for k,x in experts.items():
                c=components(x);r[k+'_C_mse']=float((c[0]-truth[0])**2);r[k+'_D_mse']=float((c[1]-truth[1])**2)
                r[k+'_mse']=float(np.sum((c-truth)**2))
        rows.append(r)
    frame=pd.DataFrame(rows);known=frame[frame.known];metrics=[x for x in frame if x.endswith('_mse')]
    errors={scope:g.groupby('date')[metrics].mean().mean().to_dict() for scope,g in [('all',known),('adaptive_active',known[~known.source_fallback])]+[(c,known[known.clock==c]) for c in CLOCKS]}
    active=frame[~frame.source_fallback];alphas={}
    for c,g in active.groupby('clock'):
        a=np.array(list(g.alpha));alphas[c]={'dates':g.date.nunique(),'mean':a.mean(axis=0),'minimum':a.min(axis=0),'maximum':a.max(axis=0),
            'zero_days':np.sum(a<=1e-12,axis=0),'one_days':np.sum(a>=1-1e-12,axis=0)}
    trade=lambda path:pd.read_csv(path).assign(decision_at=lambda x:pd.to_datetime(x.decision_at,utc=True))
    current=trade(run/'adaptive_band_trades.csv');old=trade(root/'full_band_trades.csv')
    curkeys={(r.decision_at,r.symbol) for r in current.itertuples()};oldkeys={(r.decision_at,r.symbol) for r in old.itertuples()}
    details=lambda t,keys:t[[ (r.decision_at,r.symbol) in keys for r in t.itertuples()]][['decision_at','symbol','entry_at','exit_at','reason','net_return','capital']].to_dict('records')
    candidates={};pred=read(run/'predictions.json')
    for name,es in pred.items():candidates[name]=dict(Counter(e['action'] for e in es if '2026-01-01'<=e['date'][:10]<='2026-04-30'))
    march=[]
    for t in current[current.decision_at.dt.tz_convert(core.ET).dt.month==3].itertuples():
        record=read(run/'training'/(tag(t.decision_at)+'.json'));d=read(run/'daily'/(str(t.decision_at.tz_convert(core.ET).date())+'.json'))
        march.append({'at':t.decision_at,'reason':t.reason,'net_return':t.net_return,'meta_dates':len(d['parents']),
            'point':record['estimates']['clock']['point'],'lower':record['estimates']['clock']['lower'],
            'alpha':record['estimates']['clock']['alpha'],'source_radius':record['estimates']['clock']['source_radius'],
            'joint_radius':record['estimates']['clock']['joint_radius']})
    result={'post_result_only':True,'rule_changed_after_results':False,'known_queries':len(known),'unknown_retained':int((~frame.known).sum()),
        'first_meta_ready_date':next((d['date'] for d in dates if d['ready']),None),'active_meta_dates':len([d for d in dates if d['ready']]),
        'fallback_ticks':int(frame.source_fallback.sum()),'active_ticks':len(active),'active_floor_binding_ticks':int(active.floor_binding.sum()),
        'errors_date_equal':errors,'alphas':alphas,'daily_history':dates,'candidate_counts':candidates,
        'changed_candidate_actions':frame[frame.source_action!=frame.new_action].to_dict('records'),
        'actual_added':details(current,curkeys-oldkeys),'actual_removed':details(old,oldkeys-curkeys),'march_actual':march,
        'results_sha256':sha(run/'results.json')}
    dump(run/'diagnostics.json',result);print({k:v for k,v in result.items() if k not in ['daily_history','changed_candidate_actions','march_actual']})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);diagnose(p.parse_args().run)
