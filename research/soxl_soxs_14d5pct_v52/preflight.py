"""Real prior records: scalar/KKT bootstrap checks before any v51 capital replay."""
import json
from threadpoolctl import threadpool_limits
from state_reliability import *
from audit_joint import daily_check
from audit_state import audit_state
from run_planning import json_ready


def run():
    p=read(SOURCE/'protocol.json');old=verified_v51(p);root,_=verified_source('development',p)
    panel=add_state(pd.read_parquet(old/'option_panel.parquet'));market=core.load_market()
    preview=core.ROOT/'.cache/soxl_soxs_14d5pct_v52/preflight_state';preview.mkdir(parents=True,exist_ok=True)
    panel.to_parquet(preview/'option_panel.parquet');dump(preview/'input_manifest.json',read(old/'input_manifest.json'))
    state_receipt=audit_state(preview,pd.read_parquet(root/'features.parquet'));print(state_receipt,flush=True)
    records={pd.Timestamp(r['at']):r for f in sorted((root/'training').glob('*.json')) if (r:=read(f))}
    earlier={pd.Timestamp(r['at']):r for f in sorted((old/'training').glob('*.json')) if (r:=read(f))}
    reg=registry(panel,market);receipts=[]
    with threadpool_limits(limits=1):
        for date in ['2025-11-05','2026-01-05','2026-04-01']:
            at=panel[panel.date==date].index[0];histories={k:{} for k in VARIANTS}
            for t,r in earlier.items():
                if t>=at:continue
                for current,previous in [('global','signed'),('neutral','neutral'),('baseline','baseline'),('state','baseline'),('additive','baseline')]:
                    histories[current][t]=r['raws'][previous]
            daily=fit_day(panel,records,histories,reg,at)
            frozen=json.loads(json.dumps(json_ready(daily),default=str))
            _,_,equations,certificates=daily_check(panel,market,records,histories,at,frozen)
            previous=read(old/'daily'/(date+'.json'))
            assert frozen['offsets']['global']==previous['offsets']['signed']
            for current,before in [('global','signed'),('neutral','neutral'),('baseline','baseline')]:
                assert frozen['variants'][current]==previous['variants'][before]
            raws,details,estimates,_=issue(panel,records,at,daily)
            old_record=earlier[at]
            for current,before in [('global','signed'),('neutral','neutral'),('baseline','baseline')]:
                assert json_ready(raws[current])==old_record['raws'][before]
                assert details[current]==old_record['fits'][before]
                assert estimates[current]==old_record['estimates'][before]
            receipt={'passed':True,'date':date,'mature_dates':len(daily['parents']),
                'KKT_certificates':certificates,'payoff_equations':equations,'three_v51_controls_exact':True,
                'state_objective_real_feedback':True,
                'new_payoff_history_placeholder':'state/additive baseline histories only for preflight; full run builds own immutable history',
                'new_capital_outcomes_seen':False}
            receipts.append(receipt);print(receipt,flush=True)
    dump(SOURCE/'evidence/preflight.json',{'passed':True,'dates':receipts,'state_input_audit':state_receipt})


if __name__=='__main__':run()
