"""Revalue every pilot contract and independently recalculate variance comparison."""
import gzip,pickle,json
from threadpoolctl import threadpool_limits
from path_learning import *
from audit_paths import check_forecast
from run_planning import json_ready
from verify_reliability import close


def audit():
    reg=read(SOURCE/'evidence/pilot_registration.json');result=read(SOURCE/'evidence/pilot.json');assert result['registration_sha256']==sha(SOURCE/'evidence/pilot_registration.json')
    for f,h in reg['source_sha256'].items():assert sha(core.ROOT/f)==h,f
    inputs=Path(reg['inputs']);manifest=read(inputs/'input_manifest.json');assert sha(inputs/'input_manifest.json')==reg['input_manifest_sha256']
    for f,h in manifest['derived_files_sha256'].items():assert sha(inputs/f)==h
    panel=pd.read_parquet(inputs/'activity_panel.parquet');bank=dict(np.load(inputs/'path_bank.npz'));market=core.load_market();assert market.manifest==manifest['full_market'];market=restrict_market(market,pd.Timestamp(manifest['observation_cutoff_exclusive']))
    p=read(SOURCE/'protocol.json')['numerical_pilot'];expected={pd.Timestamp(d+' '+c,tz=core.ET).tz_convert('UTC') for d in p['dates'] for c in CLOCKS}
    assert len(result['records'])==len(expected)==24 and {pd.Timestamp(r['at']) for r in result['records']}==expected
    payloads=Path(reg['local_payloads']);counts={'conditional':0,'ordinary':0};empirical={k:[] for k in counts};ratios=[];seen=set();canonical=lambda v:json.loads(json.dumps(json_ready(v),default=str))
    with threadpool_limits(limits=1):
        for row in result['records']:
            at=pd.Timestamp(row['at']);estimates={k:[] for k in counts}
            for j,replica in enumerate(range(p['replication_start'],p['replication_start']+p['independent_estimators'])):
                for kind in counts:
                    name=tag(at)+'_'+kind+'_'+str(replica)+'.pkl.gz';assert sha(payloads/name)==result['local_payloads_sha256'][name];seen.add(name)
                    with gzip.open(payloads/name,'rb') as f:v=pickle.load(f)
                    assert v['at']==at and v['kind']==kind and v['replication']==replica
                    raw,n=check_forecast(bank,market,panel,at,kind,canonical(v['info']),v['model'],count=p['paths'],replication=replica)
                    assert n==row['contracts_evaluated'][kind][j];counts[kind]+=n;close(v['raw'],raw);close(row['estimates'][kind][j],raw);estimates[kind].append(raw)
                    close(row['reported_covariances'][kind][j],v['info']['integration']['mean_covariance'])
                    if kind=='conditional':close(row['unresolved_probability'][j],v['info']['partition']['unresolved_probability'])
            cov={}
            for k,vs in estimates.items():
                values=np.array(vs);delta=values-values.mean(axis=0);cov[k]=sum(np.outer(x,x) for x in delta)/(len(values)-1);empirical[k].append(np.trace(cov[k]));close(row['empirical_covariance'][k],cov[k])
            ratio=np.trace(cov['conditional'])/np.trace(cov['ordinary']);ratios.append(ratio);close(row['empirical_trace_ratio'],ratio)
            diff=np.array(estimates['conditional'])-np.array(estimates['ordinary']);mean=diff.mean(axis=0);se=np.sqrt(np.sum((diff-mean)**2,axis=0)/(len(diff)*(len(diff)-1)))
            close(row['paired_mean_difference'],mean);close(row['paired_standard_error'],se);close(row['paired_standardized_difference'],np.divide(mean,se,out=np.zeros(2),where=se>0))
            print('pilot independent valuation',at,flush=True)
    assert set(result['local_payloads_sha256'])==seen
    s=result['summary'];assert s['queries']==24 and s['contracts_evaluated']==counts
    traces={k:float(np.mean(v)) for k,v in empirical.items()};ratio=traces['conditional']/traces['ordinary']
    for k,v in traces.items():close(s['mean_empirical_trace'][k],v);close(s['runtime_seconds'][k],sum(sum(r['seconds'][k]) for r in result['records']))
    close(s['pooled_variance_ratio'],ratio);close(s['median_query_variance_ratio'],np.median(ratios));assert result['passed']==(ratio<1)
    out={'passed':True,'contracts_independently_valued':counts,'pilot_gate_passed':result['passed'],'pooled_variance_ratio':ratio,'estimators':len(seen),'pilot_sha256':sha(SOURCE/'evidence/pilot.json'),'new_capital_outcomes_seen':False}
    dump(SOURCE/'evidence/pilot_audit.json',out);print(out,flush=True)

if __name__=='__main__':audit()
