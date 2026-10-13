"""Post-result current-query integration stability at the original 24 pilot queries."""
import argparse,json
from collections import Counter
from path_learning import *
from audit_dense_calibration import predict
from run_planning import json_ready


def inspect(run):
    run=Path(run);v=read(run/'verification.json');assert v['passed'] and v['results_sha256']==sha(run/'results.json')
    pilot=read(SOURCE/'evidence/pilot.json');audit=read(SOURCE/'evidence/pilot_audit.json');assert audit['passed'] and audit['pilot_sha256']==sha(SOURCE/'evidence/pilot.json')
    p=read(SOURCE/'protocol.json')['numerical_pilot'];panel=pd.read_parquet(run/'activity_panel.parquet');rows=[];canonical=lambda x:json.loads(json.dumps(json_ready(x),default=str));equations=0
    expected={pd.Timestamp(day+' '+clock,tz=core.ET).tz_convert('UTC') for day in p['dates'] for clock in CLOCKS}
    assert {pd.Timestamp(x['at']) for x in pilot['records']}==expected
    for q in pilot['records']:
        at=pd.Timestamp(q['at']);daily=read(run/'daily'/(str(at.tz_convert(core.ET).date())+'.json'));rec=read(run/'training'/(tag(at)+'.json'));c=CLOCKS.index(panel.loc[at,'clock'])
        for kind in ['conditional','ordinary']:
            saved=daily[kind];assert saved['ready'];actions=[];lower=[]
            for values in q['estimates'][kind]:
                raw=np.array(values);info=issue(panel,at,raw,saved);action=decide(info,saved['feedback'],True)['action']
                models=saved['model'];point=raw+predict(models['point_model'],np.r_[raw,c])[0]
                draws=np.array([raw+predict(b,np.r_[raw,c])[0] for b in models['bootstrap_models']]);radius=max(0.,sorted(np.max(draws-point,axis=1))[179]);lo=point-radius
                np.testing.assert_allclose(info['point'],point,atol=1e-11);np.testing.assert_allclose(info['lower'],lo,atol=1e-11)
                expected_action=core.ETFS[int(np.argmax(lo))] if max(lo)>1e-12 and not saved['feedback']['expired_unresolved_prefix'] else 'CASH';assert action==expected_action
                actions.append(action);lower.append(lo);equations+=200
            official=rec['events'][kind+'_band']['action'];counts=dict(Counter(actions));different=sum(a!=official for a in actions)
            rows.append({'at':at,'kind':kind,'official_replication0_action':official,'independent_current_query_actions':counts,'different_from_official':different,'estimators':len(actions),'minimum_lower':np.min(lower,axis=0),'maximum_lower':np.max(lower,axis=0),'replications':list(range(100,116))})
    summary={}
    for k in ['conditional','ordinary']:
        rs=[r for r in rows if r['kind']==k]
        summary[k]={'queries':len(rs),'queries_with_multiple_actions':sum(len(r['independent_current_query_actions'])>1 for r in rs),'queries_with_any_difference_from_official':sum(r['different_from_official']>0 for r in rs),'different_from_official_fraction':sum(r['different_from_official'] for r in rs)/sum(r['estimators'] for r in rs)}
    out={'passed':True,'post_result_diagnostic':True,'rules_changed':False,'source_sha256':sha(Path(__file__)),'results_sha256':sha(run/'results.json'),'pilot_sha256':sha(SOURCE/'evidence/pilot.json'),'calibration_equations':equations,'summary':summary,'rows':rows,
         'scope':'Original 24 predeclared numerical queries and all 16 independent estimators; each query holds its official as-issued daily calibrator fixed. No future return labels, history refit, alternative capital replay or seed selection. Action variation is numerical stability, not market prediction quality.'}
    dump(run/'decision_stability_audit.json',out);print({k:v for k,v in out.items() if k!='rows'},flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);inspect(p.parse_args().run)
