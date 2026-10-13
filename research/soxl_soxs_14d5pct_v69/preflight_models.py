"""25 fixed real queries, independent full prices, and exact legacy predictions."""
import argparse,json,time
from path_learning import *
from audit_paths import check_forecast
from audit_matched_control import matched_source,compare_record
from run_planning import json_ready
from threadpoolctl import threadpool_limits


def preflight(inputs):
    inputs=Path(inputs);manifest=read(inputs/'input_manifest.json');assert read(inputs/'freshness_audit.json')['passed'] and read(inputs/'state_input_audit.json')['passed']
    from state_paths import load_context
    context=load_context(inputs);bank=dict(np.load(inputs/'path_bank.npz'));panel=pd.read_parquet(inputs/'activity_panel.parquet');market=core.load_market();assert market.manifest==manifest['full_market'];market=restrict_market(market,pd.Timestamp(manifest['observation_cutoff_exclusive']))
    old=matched_source();selected=panel[panel.date.astype(str).isin(['2025-11-12','2026-01-05','2026-04-01'])];assert len(selected)==24;half=pd.Timestamp('2025-11-28 09:25:30',tz=core.ET).tz_convert('UTC');selected=pd.concat([selected,panel.loc[[half]]]);rows=[];checks=0
    canonical=lambda x:json.loads(json.dumps(json_ready(x),default=str))
    with threadpool_limits(limits=1):
        for at,row in selected.iterrows():
            values={};runtimes={}
            for k in VARIANTS:
                started=time.perf_counter();values[k]=forecast(bank,market,panel,at,k,context=context);runtimes[k]=time.perf_counter()-started
            fits={k:v[1] for k,v in values.items()};raws={k:v[2] for k,v in values.items()};compare_record(canonical({'fits':fits,'raws':raws}),read(old/'training'/(tag(at)+'.json')))
            for k,(model,info,raw) in values.items():
                expected,n=check_forecast(bank,market,panel,at,k,canonical(info),model,context=context);checks+=n
                assert raw is not None;np.testing.assert_allclose(raw,expected,atol=1e-10)
                started=time.perf_counter();large,large_info,large_raw=forecast(bank,market,panel,at,k,count=8192,context=context);large_elapsed=time.perf_counter()-started;_,n=check_forecast(bank,market,panel,at,k,canonical(large_info),large,count=8192,context=context);checks+=n
                rows.append({'at':at,'variant':k,'members':len(info['members']),'difference_2048_minus_8192':raw-large_raw,'standard_error_2048':info['monte_carlo_se'],'standard_error_8192':large_info['monte_carlo_se'],'state_transition':info.get('state_transition'),'halfday_in_horizon':42 in info['plan']['bar_counts'],'forecast_seconds_2048':runtimes[k],'forecast_seconds_8192':large_elapsed})
            print('preflight verified',at,flush=True)
    result={'passed':True,'queries':len(selected),'exact_v68_fresh_queries':len(selected),'simulated_contracts_independently_valued':checks,'queries_with_both_numerical_budgets':rows,'daily_calibration_code_unchanged':'full own-history equations verified after replay and before case authorization','new_capital_outcomes_seen':False,'input_manifest_sha256':sha(inputs/'input_manifest.json'),'freshness_audit_sha256':sha(inputs/'freshness_audit.json'),'state_input_audit_sha256':sha(inputs/'state_input_audit.json')}
    dump(SOURCE/'evidence/preflight_models.json',result);print({k:v for k,v in result.items() if k!='queries_with_both_numerical_budgets'},flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);preflight(p.parse_args().inputs)
