"""Promote the previously registered strong candidate without changing its model."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'soxl_soxs_14d5pct_v45'))
from adaptive_penalty import *
SOURCE=Path(__file__).resolve().parent
CANDIDATES=['strong','plain']
ROUTES=['strong_band','strong_point','plain_band','premarket_only','cash','double_cost']


def verified_v45():
    root=core.ROOT/'.cache/soxl_soxs_14d5pct_v45/runs/penalty_20261013';dpath=core.ROOT/'research/soxl_soxs_14d5pct_v45/evidence/delivery_manifest.json';manifest=read(dpath)
    assert str(root)==manifest['run']
    for n,h in manifest['cache_outputs_sha256'].items():assert sha(root/n)==h,n
    for folder in ['daily','training']:
        for n,h in manifest[folder+'_sha256'].items():assert sha(root/folder/n)==h,(folder,n)
    handpath=core.ROOT/'research/soxl_soxs_14d5pct_v45/evidence/candidate_handoff.json';hand=read(handpath);r=read(root/'results.json');v=read(root/'verification.json')
    assert sha(handpath)==manifest['delivery_files_sha256']['evidence/candidate_handoff.json']
    assert hand['candidate']=='strong_band' and hand['fixed_raw_penalty']==20 and hand['fixed_macro_penalty']==2000 and hand['number_of_registered_penalty_candidates']==4
    assert hand['source_results_sha256']==sha(root/'results.json')==v['results_sha256'] and v['passed'] and hand['source_verification_sha256']==sha(root/'verification.json')
    assert hand['source_primary_failed'] and not r['development_gate']['passed']
    assert hand['candidate_development_statistics']==r['routes']['strong_band'] and r['routes']['strong_band']['primary_pass']
    return root,{'delivery_manifest_sha256':sha(dpath),'candidate_handoff_sha256':sha(handpath),'results_sha256':sha(root/'results.json'),'verification_sha256':sha(root/'verification.json')}


def fit_day(panel,raw_history,reg,at):
    parents,fb=pool(reg,raw_history,at);ids=[t for r in parents for t in r['members']];n=len(parents)
    d={'date':panel.loc[at,'date'],'cutoff':day_cutoff(at),'parents':parents,'members':ids,'feedback':fb,'ready':n>=60,'models':{}}
    if n<60:return d
    dw,w,groups=date_weights(parents);X=np.array([[*raw_history[t],*[panel.loc[t,c] for c in MACRO],CLOCKS.index(panel.loc[t,'clock'])] for t in ids],float)
    truth=panel.loc[ids,['C','D']].to_numpy(float);Y=np.column_stack((truth[:,0]+truth[:,1],truth[:,0]-truth[:,1]))-X[:,:2]
    counts,starts,seed=block_counts(n,at,10);d.update(day_weights=dw,row_weights=w,row_groups=groups,block_starts=starts,seed=seed)
    for k in CANDIDATES:d['models'][k]={'point_model':fit_candidate(X,Y,w,n,k),'bootstrap_models':[fit_candidate(X,Y,w*c[groups],n,k) for c in counts]}
    return d


def issue(panel,at,raw,daily):
    ready=raw is not None and panel.loc[at,'feature_ready'] and daily['ready'];out={}
    if ready:x=np.r_[raw,[panel.loc[at,c] for c in MACRO],CLOCKS.index(panel.loc[at,'clock'])]
    for k in CANDIDATES:
        info={'ready':bool(ready),'point':None,'lower':None}
        if ready:
            models=daily['models'][k];point=np.array(raw)+candidate_prediction(models['point_model'],x)[0];draws=np.array([raw+candidate_prediction(m,x)[0] for m in models['bootstrap_models']]);radius=max(0.,float(np.quantile(np.max(draws-point,axis=1),.9,method='higher')))
            info.update(point=point.tolist(),lower=(point-radius).tolist(),radius=radius,bootstrap_predictions=draws.tolist())
        out[k]=info
    return out


def events(at,date,clock,estimates,feedback):
    primary=decide(estimates['strong'],feedback,True);cash={'ready':True,'action':'CASH','allocation':0.}
    out={'strong_band':primary,'strong_point':decide(estimates['strong'],feedback,False),'plain_band':decide(estimates['plain'],feedback,True),'premarket_only':primary if clock==CLOCKS[0] else cash,'cash':cash,'double_cost':primary}
    return {k:{**e,'at':at,'date':date} for k,e in out.items()}
