"""Independent lag membership and state equations using explicit intercept design."""
import argparse,json
from pathlib import Path
from innovation_learning import core,np,pd,read,dump,sha,restrict_market,verified_source,SOURCE,STATES,INNOV,DIFF,CLOCKS
from verify_continuation import near
from verify_reliability import close
from run_planning import json_ready


def check_record(panel,sessions,at,rec):
    day=panel.loc[at,'date'];clock=panel.loc[at,'clock'];cutoff=pd.Timestamp(day,tz=core.ET).tz_convert('UTC');position=list(sessions).index(day)
    lag=pd.Timestamp(str(sessions[position-1].date())+' '+clock,tz=core.ET).tz_convert('UTC') if position else None
    ready=lag in panel.index and all(np.isfinite(float(panel.loc[t,c])) for t in [at,lag] for c in STATES)
    expected={'at':at,'date':day,'clock':clock,'lag_at':lag,'ready':bool(ready)};canon=lambda x:json.loads(json.dumps(json_ready(x),default=str))
    assert rec['at']==str(at) and pd.Timestamp(rec['cutoff'])==cutoff and rec['pair']==canon(expected)
    members=[];lags=[]
    for i in range(1,position):
        t=pd.Timestamp(str(sessions[i].date())+' '+clock,tz=core.ET).tz_convert('UTC');u=pd.Timestamp(str(sessions[i-1].date())+' '+clock,tz=core.ET).tz_convert('UTC')
        if t in panel.index and u in panel.index and all(np.isfinite(float(panel.loc[a,c])) for a in [t,u] for c in STATES):members.append(t);lags.append(u)
    fit=rec['fit'];assert fit['members']==[str(x) for x in members] and fit['lag_members']==[str(x) for x in lags] and fit['n']==len(members) and fit['ready']==(len(members)>=60)
    query_ready=ready and len(members)>=60;assert rec['query']['ready']==query_ready
    if len(members)<60:
        assert not rec['query']['ready'];return None,0
    n=len(members);w=np.array([2**((i-n+1)/126.) for i in range(n)]);w*=n/sum(w)
    x=np.array([[float(panel.loc[t,c]) for c in STATES] for t in lags]);y=np.array([[float(panel.loc[t,c]) for c in STATES] for t in members])
    mean=(w[:,None]*x).sum(axis=0)/sum(w);scale=np.sqrt((w[:,None]*(x-mean)**2).sum(axis=0)/sum(w));scale[scale<1e-12]=1
    z=(x-mean)/scale;design=np.column_stack([np.ones(n),z]);pen=np.diag([0.,20.,20.,20.]);beta=np.linalg.solve(design.T@(w[:,None]*design)+pen,design.T@(w[:,None]*y))
    ym=(w[:,None]*y).sum(axis=0)/sum(w);residual=y-design@beta;rs=np.sqrt((w[:,None]*residual**2).sum(axis=0)/sum(w));ds=np.sqrt((w[:,None]*(y-x)**2).sum(axis=0)/sum(w));rs[rs<1e-12]=1;ds[ds<1e-12]=1
    for key,value in [('weights',w),('input_center',mean),('input_scale',scale),('output_center',ym),('coefficients',beta[1:]),('residual_scale',rs),('difference_scale',ds)]:close(fit[key],value)
    close(beta[0],ym)
    if not query_ready:
        assert all(rec['query'][k] is None for k in ['prediction','innovation','difference']);return None,3
    current=np.array([float(panel.loc[at,c]) for c in STATES]);prior=np.array([float(panel.loc[lag,c]) for c in STATES]);prediction=np.r_[1,(prior-mean)/scale]@beta
    innovation=(current-prediction)/rs;difference=(current-prior)/ds
    for key,value in [('prediction',prediction),('innovation',innovation),('difference',difference)]:close(rec['query'][key],value)
    # Same information span for a fixed fitted transform, not new external data.
    from_current_difference=(current-(ym+((current-difference*ds-mean)/scale)@beta[1:]))/rs;close(innovation,from_current_difference)
    return np.r_[innovation,difference],3


def audit(inputs):
    inputs=Path(inputs);m=read(inputs/'input_manifest.json');source,source_sha,prior_sha=verified_source();assert source_sha==m['source_v56_manifest_sha256'] and prior_sha==m['prior_v58_manifest_sha256']
    for f,h in m['derived_files_sha256'].items():assert sha(inputs/f)==h,f
    for f,h in m['original_inputs_sha256'].items():assert sha(inputs/f)==sha(Path(m['base_inputs'])/f)==h,f
    market=core.load_market();assert market.manifest==m['full_market'];market=restrict_market(market,pd.Timestamp(m['observation_cutoff_exclusive']))
    original=pd.read_parquet(inputs/'activity_panel.parquet');panel=pd.read_parquet(inputs/'innovation_panel.parquet');records=read(inputs/'state_records.json');assert len(records)==len(panel)
    assert list(panel.index)==list(original.index);pd.testing.assert_frame_equal(panel[original.columns.drop('feature_ready')],original.drop(columns=['feature_ready']))
    equations=ready=0
    for i,(at,row) in enumerate(original.iterrows()):
        values,n=check_record(original,market.sessions,at,records[i]);equations+=n;ok=values is not None
        assert panel.loc[at,'state_ready']==ok and panel.loc[at,'feature_ready']==bool(ok and row.feature_ready)
        if ok:close(panel.loc[at,INNOV+DIFF].to_numpy(float),values);ready+=1
        else:assert panel.loc[at,INNOV+DIFF].isna().all()
        if (i+1)%800==0:print('state features independently checked through',at,flush=True)
    out={'passed':True,'queries':len(panel),'state_equations':equations,'state_ready_queries':ready,'feature_ready_queries':int(panel.feature_ready.sum()),'input_manifest_sha256':sha(inputs/'input_manifest.json'),'source_v56_manifest_sha256':source_sha,'prior_v58_manifest_sha256':prior_sha,'targets_unchanged':True,'fixed_transform_span_identity_checked':True,'new_capital_outcomes_seen':False};dump(inputs/'state_audit.json',out);print(out,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);audit(p.parse_args().inputs)
