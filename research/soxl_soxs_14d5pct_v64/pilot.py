"""Registered integration-only pilot, retaining every independent estimator."""
import gzip,pickle,time
from datetime import datetime,timezone
from threadpoolctl import threadpool_limits
from path_learning import *


def pilot():
    registration=read(SOURCE/'evidence/pilot_registration.json')
    for f,h in registration['source_sha256'].items():assert sha(core.ROOT/f)==h,f
    inputs=Path(registration['inputs']);manifest=read(inputs/'input_manifest.json');assert sha(inputs/'input_manifest.json')==registration['input_manifest_sha256']
    for f,h in manifest['derived_files_sha256'].items():assert sha(inputs/f)==h,f
    panel=pd.read_parquet(inputs/'activity_panel.parquet');bank=dict(np.load(inputs/'path_bank.npz'));market=core.load_market();assert market.manifest==manifest['full_market']
    market=restrict_market(market,pd.Timestamp(manifest['observation_cutoff_exclusive']));p=read(SOURCE/'protocol.json')['numerical_pilot']
    output=Path(registration['local_payloads']);assert not output.exists();output.mkdir(parents=True)
    records=[];files={};variants=['conditional','ordinary']
    with threadpool_limits(limits=1):
        for at,row in panel[panel.date.astype(str).isin(p['dates'])].iterrows():
            estimates={k:[] for k in variants};reported={k:[] for k in variants};seconds={k:[] for k in variants};counts={k:[] for k in variants};parts=[]
            for replica in range(p['replication_start'],p['replication_start']+p['independent_estimators']):
                for kind in variants:
                    begin=time.perf_counter();model,info,raw=forecast(bank,market,panel,at,kind,count=p['paths'],replication=replica);elapsed=time.perf_counter()-begin
                    assert raw is not None and info['ready'];name=tag(at)+'_'+kind+'_'+str(replica)+'.pkl.gz'
                    with gzip.open(output/name,'wb') as f:pickle.dump({'at':at,'kind':kind,'replication':replica,'model':model,'info':info,'raw':raw},f)
                    files[name]=sha(output/name);estimates[kind].append(raw);reported[kind].append(info['integration']['mean_covariance']);seconds[kind].append(elapsed)
                    counts[kind].append(2*info['paths']+(2*len(info['members']) if kind=='conditional' else 0))
                    if kind=='conditional':parts.append(info['partition']['unresolved_probability'])
            cov={k:np.cov(v,rowvar=False,ddof=1) for k,v in estimates.items()};difference=np.array(estimates['conditional'])-np.array(estimates['ordinary'])
            se=difference.std(axis=0,ddof=1)/np.sqrt(p['independent_estimators']);mean=difference.mean(axis=0)
            records.append({'at':at,'estimates':estimates,'reported_covariances':reported,'empirical_covariance':cov,'empirical_trace_ratio':np.trace(cov['conditional'])/np.trace(cov['ordinary']),
                'paired_mean_difference':mean,'paired_standard_error':se,'paired_standardized_difference':np.divide(mean,se,out=np.zeros(2),where=se>0),'seconds':seconds,'contracts_evaluated':counts,'unresolved_probability':parts})
            print('exact-absorption numerical pilot',at,flush=True)
    traces={k:float(np.mean([np.trace(r['empirical_covariance'][k]) for r in records])) for k in variants}
    summary={'queries':len(records),'mean_empirical_trace':traces,'pooled_variance_ratio':traces['conditional']/traces['ordinary'],
        'median_query_variance_ratio':float(np.median([r['empirical_trace_ratio'] for r in records])),
        'runtime_seconds':{k:float(sum(sum(r['seconds'][k]) for r in records)) for k in variants},
        'contracts_evaluated':{k:int(sum(sum(r['contracts_evaluated'][k]) for r in records)) for k in variants}}
    passed=len(records)==24 and summary['pooled_variance_ratio']<1
    result={'finished_at':datetime.now(timezone.utc).isoformat(),'passed':passed,'summary':summary,'records':records,'local_payloads_sha256':files,
        'new_capital_outcomes_seen':False,'future_labels_used':False,'return_acceptance_established':False,'registration_sha256':sha(SOURCE/'evidence/pilot_registration.json')}
    dump(SOURCE/'evidence/pilot.json',result);print({'passed':passed,'summary':summary},flush=True)

if __name__=='__main__':pilot()
