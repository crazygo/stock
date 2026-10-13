"""Fixed-query input/numerical feasibility only; never replays a capital policy."""
import argparse,time,json
from datetime import datetime,timezone
from pathlib import Path
from prototype_state_paths import *
from path_learning import dump,day_cutoff,draw_blocks,simulate
from sampling import uniforms
from prototype_state_tests import scalar_walk
from audit_paths import reference_values


def run(inputs,output):
    inputs=Path(inputs);output=Path(output);assert not output.exists();source=Path(__file__).parent;m=read(inputs/'input_manifest.json');assert m['phase']=='development'
    files=[Path(__file__),source/'prototype_state_paths.py',source/'prototype_state_tests.py']
    registration={'at':datetime.now(timezone.utc).isoformat(),'source_sha256':{str(f.resolve()):sha(f) for f in files},'input_manifest_sha256':sha(inputs/'input_manifest.json'),'fixed_dates':['2025-11-12','2026-01-05','2026-04-01'],'halfday_query':'2025-11-28 09:25:30 America/New_York','regular_budget':256,'larger_budget':2048,'larger_budget_clocks':[CLOCKS[0]],'v68_case_results_existed':(core.ROOT/'.cache/soxl_soxs_14d5pct_v68/runs/cases_20261013/results.json').exists(),'capital_outcomes_read':False,'v68_frozen_strategy_changed':False,'scope':'Prototype rolling state equations, dynamic kernel selection, original-contract price valuation and runtime only. No residual calibration, exact absorption estimator or account policy is implemented here.'}
    output.mkdir(parents=True);dump(output/'registration.json',registration)
    market=core.load_market();assert market.manifest==m['full_market'];cutoff=pd.Timestamp(m['observation_cutoff_exclusive']);market=restrict_market(market,cutoff)
    bank=dict(np.load(inputs/'path_bank.npz'));panel=pd.read_parquet(inputs/'activity_panel.parquet');reference,bars,daily,symbols=reference_inputs(market,bank,panel,cutoff)
    queries=list(panel[panel.date.astype(str).isin(registration['fixed_dates'])].index);queries.append(pd.Timestamp('2025-11-28 09:25:30',tz=core.ET).tz_convert('UTC'));assert len(queries)==25
    rows=[];statechecks=contractchecks=0
    for at in queries:
        row=panel.loc[at];clock=row.clock;members=price_known_pool(bank,at,clock);ids=bank['blocks'][members].ravel();assert len(members)>=120
        assert np.all(reference['release_ns'][ids]!=pd.NaT.value) and np.all(reference['release_ns'][ids]<day_cutoff(at).value)
        assert np.isfinite(reference['regular_close'][ids]).all() and (reference['regular_close'][ids]>0).all() and np.isfinite(reference['previous_close'][ids]).all()
        assert np.isfinite(reference['quotes'][bank['blocks'][members,0],CLOCKS.index(clock)]).all()
        initial,quotes=current_history(market,bars,daily,symbols,at);np.testing.assert_allclose(roll_state(initial[None,:])[0],row[PRIOR].to_numpy(float),rtol=1e-10,atol=1e-12)
        a,b,first=first_kernels(bank,bank['blocks'][members,0],clock,row);later=kernel(bank['states'][bank['blocks'][members,0],0,:3],row[PRIOR].to_numpy(float),'conditional');assert all(k['ready'] for k in [a,b,first,later]);plan=horizon(market.calendar,at);raw,_=quote(market,at);seed=1062+at.tz_convert(core.ET).date().toordinal()*16+CLOCKS.index(clock)
        for count in ([256,2048] if clock==CLOCKS[0] and str(row.date.date()) in registration['fixed_dates'] else [256]):
            points=uniforms(seed,(len(plan['dates'])+1)//2,count);start=time.perf_counter();actual=dynamic_blocks(members,first['probability'],points,initial,quotes,reference,bank,clock,plan);elapsed=time.perf_counter()-start
            check_indices=np.arange(count) if count==256 else np.arange(0,count,32)
            for i in check_indices:
                chosen,states,closes,final,ess=scalar_walk(members,first['probability'],points[i],initial,quotes,reference,bank,clock,plan)
                np.testing.assert_array_equal(actual['selected_blocks'][i],chosen)
                for key,want in [('states_before_blocks',states),('generated_daily_closes',closes),('final_history',final),('effective_dates',ess)]:np.testing.assert_allclose(actual[key][i],want,rtol=1e-10,atol=1e-11)
                statechecks+=len(states)
            result=simulate(bank,actual['selected_blocks'],clock,raw,plan);check=reference_values(bank,actual['selected_blocks'],clock,raw,plan)
            for key,value in check.items():np.testing.assert_allclose(result[key],value,rtol=1e-10,atol=1e-10)
            contractchecks+=2*count;fixed,_=draw_blocks(members,first['probability'],later['probability'],at,clock,len(plan['dates']),count)
            np.testing.assert_array_equal(actual['selected_blocks'][:,0],fixed[:,0]);extended=np.concatenate((np.broadcast_to(initial,(count,21,10)),actual['generated_daily_closes']),axis=1)
            for step in range(len(plan['dates'])//2+len(plan['dates'])%2):np.testing.assert_allclose(actual['states_before_blocks'][:,step],roll_state(extended[:,step*2:step*2+21]),rtol=1e-12,atol=1e-12)
            rows.append({'at':at,'count':count,'source_members':len(members),'first_draw_exact_fixed':True,'later_draw_change_fraction':float(np.mean(actual['selected_blocks'][:,1:]!=fixed[:,1:])),'minimum_future_effective_dates':float(actual['effective_dates'][:,1:].min()),'elapsed_dynamic_generation_seconds':elapsed,'scalar_path_checks':len(check_indices),'halfday_truncated':42 in plan['bar_counts'],'absolute_dynamic_state_change_mean':np.abs(actual['states_before_blocks'][:,1:]-actual['states_before_blocks'][:,:1]).mean(axis=(0,1)).tolist()})
        print('prototype verified',at,flush=True)
    assert all(sha(Path(f))==h for f,h in registration['source_sha256'].items())
    out={'passed':True,'scope':registration['scope'],'query_count':len(queries),'budget_runs':len(rows),'scalar_path_state_checks':statechecks,'original_contracts_independently_valued':contractchecks,'rows':rows,'capital_outcomes_read':False,'no_new_strategy_qualification':True,'registration_sha256':sha(output/'registration.json')};dump(output/'results.json',out);dump(source/'evidence/state_prototype_registration.json',registration);dump(source/'evidence/state_prototype_results.json',out);print({k:v for k,v in out.items() if k!='rows'},flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.inputs,a.output)
