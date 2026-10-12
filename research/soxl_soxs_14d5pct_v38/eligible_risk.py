"""Condition risk learning on source eligibility; bound overrides to known ranges."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'soxl_soxs_14d5pct_v37'))
from selected_risk import (core,np,pd,read,dump,sha,CLOCKS,tag,day_cutoff,registry,pool,
    block_counts,restrict_market,labels,development_gate,load_source,selected_panel,
    SCORES,CONTEXT,fit_logit,quality)

SOURCE=Path(__file__).resolve().parent
MODELS={'eligible_context':('eligible',SCORES+CONTEXT),'eligible_scores':('eligible',SCORES),
        'all_context':('all',SCORES+CONTEXT)}
ROUTES=['eligible_supported','eligible_gate','all_supported','all_gate','eligible_scores',
        'eligible_band','source_v37','source_v35','cash','double_cost']


def subsets(parents,records):
    out={'all':[],'eligible':[]}
    for i,p in enumerate(parents):
        out['all'].append({'parent_index':i,'date':p['date'],'members':p['members']})
        ids=[t for t in p['members'] if records[t]['events']['full_band']['action'] in core.ETFS]
        if ids:out['eligible'].append({'parent_index':i,'date':p['date'],'members':ids})
    return out


def subset_weights(parents,n_parent_dates):
    # Age is in the common forecast-date sequence, not number of eligible events.
    d=np.array([2.**((p['parent_index']-n_parent_dates+1)/63.) for p in parents]);d*=len(parents)/d.sum()
    w=np.concatenate([np.repeat(v/len(p['members']),len(p['members'])) for p,v in zip(parents,d)])
    groups=np.repeat([p['parent_index'] for p in parents],[len(p['members']) for p in parents])
    return w,groups


def fit_day(panel,records,reg,at,source_day=None):
    history={t:r['raws']['full'] if r['estimates']['full']['ready'] else None for t,r in records.items() if t<day_cutoff(at)}
    parents,fb=pool(reg,history,at);parts=subsets(parents,records);n=len(parents)
    ready=n>=60 and len(parts['eligible'])>=30
    out={'date':panel.loc[at,'date'],'cutoff':day_cutoff(at),'parents':parents,'feedback':fb,
        'subsets':parts,'ready':ready,'models':{},'bootstrap':[],'inputs':{}}
    if not ready:return out
    counts,starts,seed=block_counts(n,at,10);out.update(block_starts=starts,seed=seed)
    matrices={}
    for kind,(group,cols) in MODELS.items():
        pp=parts[group];ids=[t for p in pp for t in p['members']];w,g=subset_weights(pp,n)
        X=panel.loc[ids,cols].to_numpy(float);r=panel.loc[ids,'selected_y'].to_numpy(float)
        out['inputs'][kind]={'members':ids,'weights':w,'row_parent_groups':g,'distinct_dates':len(pp),
            'columns':cols,'minimum':X.min(axis=0),'maximum':X.max(axis=0)}
        out['models'][kind]=fit_logit(X,r,w,len(pp));matrices[kind]=(X,r,w,g,len(pp))
    for count in counts:
        models={}
        for kind,(X,r,w,g,nn) in matrices.items():
            v=w*count[g]
            models[kind]=fit_logit(X,r,v,nn) if np.sum(v)>0 else None
        out['bootstrap'].append(models)
    return out


def outside_columns(x,support):
    return [c for c,v,low,high in zip(support['columns'],x,support['minimum'],support['maximum'])
            if v<low-1e-12 or v>high+1e-12]


def issue(panel,at,record,daily):
    ready=bool(record['estimates']['full']['ready'] and panel.loc[at,'feature_ready'])
    out={k:{'ready':ready,'quality':None,'lower_quality':None,'source_fallback':bool(ready and not daily['ready'])} for k in MODELS}
    if not ready or not daily['ready']:return out
    for kind,(_,cols) in MODELS.items():
        x=panel.loc[at,cols].to_numpy(float);point=quality(daily['models'][kind],x)
        draws=[quality(b[kind],x) if b[kind] is not None else None for b in daily['bootstrap']]
        valid=all(v is not None for v in draws)
        radius=max(0.,float(np.quantile(np.asarray(draws)-point,.9,method='higher'))) if valid else None
        out[kind].update(quality=point,lower_quality=point-radius if valid else None,
            bootstrap_complete=valid,upward_radius=radius,bootstrap_qualities=draws,
            outside_columns=outside_columns(x,daily['inputs'][kind]))
    return out


def events(at,date,estimates,daily,record,old_v37):
    old=record['events']['full_band'];expired=daily['feedback']['expired_unresolved_prefix']
    def gate(kind,supported=False,band=False):
        v=estimates[kind];ready=v['ready'] and not expired;q=None;take=False;reason='unready'
        if ready:
            if old['action'] not in core.ETFS:reason='source_rejected'
            elif v['source_fallback']:take=True;reason='source_warmup'
            elif supported and v['outside_columns']:take=True;reason='source_outside_support'
            elif band and not v['bootstrap_complete']:reason='bootstrap_incomplete'
            else:
                q=v['lower_quality'] if band else v['quality'];take=q>.5+1e-12
                reason='positive_gain_loss_balance' if take else 'risk_rejected'
        return {'ready':ready,'action':old['action'] if take else 'CASH','allocation':float(take),
                'reason':reason,'quality_used':q}
    primary=gate('eligible_context',True)
    actions={'eligible_supported':primary,'eligible_gate':gate('eligible_context'),
        'all_supported':gate('all_context',True),'all_gate':gate('all_context'),
        'eligible_scores':gate('eligible_scores',True),'eligible_band':gate('eligible_context',True,True),
        'source_v37':old_v37,'source_v35':old,'cash':{'ready':True,'action':'CASH','allocation':0.},'double_cost':primary}
    return {k:{**v,'at':at,'date':date} for k,v in actions.items()}


def load_control():
    root=core.ROOT/'.cache/soxl_soxs_14d5pct_v37/runs/selected_20261013'
    delivery=read(core.ROOT/'research/soxl_soxs_14d5pct_v37/evidence/delivery_manifest.json');hashes={}
    for f in ['predictions.json','results.json','verification.json']+['context_gate_'+s+'.csv' for s in ['trades','equity','decisions','orders']]:
        assert sha(root/f)==delivery['cache_outputs_sha256'][f];hashes[f]=sha(root/f)
    return root,{pd.Timestamp(e['at']):e for e in read(root/'predictions.json')['context_gate']},hashes
