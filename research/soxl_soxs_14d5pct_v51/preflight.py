"""Real prior records: scalar/KKT bootstrap checks before any v51 capital replay."""
import json
from threadpoolctl import threadpool_limits
from signed_reliability import *
from audit_joint import daily_check
from run_planning import json_ready


def run():
    p=read(SOURCE/'protocol.json');old=verified_v50(p);root,_=verified_source('development',p)
    panel=pd.read_parquet(old/'option_panel.parquet');market=core.load_market()
    records={pd.Timestamp(r['at']):r for f in sorted((root/'training').glob('*.json')) if (r:=read(f))}
    earlier={pd.Timestamp(r['at']):r for f in sorted((old/'training').glob('*.json')) if (r:=read(f))}
    reg=registry(panel,market);receipts=[]
    with threadpool_limits(limits=1):
        for date in ['2025-11-05','2026-01-05','2026-04-01']:
            at=panel[panel.date==date].index[0];histories={k:{} for k in VARIANTS}
            for t,r in earlier.items():
                if t>=at:continue
                for current,previous in [('nonnegative','temperature'),('neutral','neutral'),('baseline','baseline'),('signed','baseline')]:
                    histories[current][t]=r['raws'][previous]
                original=records[t]['fits']['full']['raw']
                inverted=payoff(original,1-float(original['q'])) if original and panel.loc[t,'feature_ready'] else None
                histories['inverse'][t]=[inverted[s] for s in core.ETFS] if inverted else None
            daily=fit_day(panel,records,histories,reg,at)
            frozen=json.loads(json.dumps(json_ready(daily),default=str))
            _,_,equations,certificates=daily_check(panel,market,records,histories,at,frozen)
            previous=read(old/'daily'/(date+'.json'))
            assert frozen['offsets']['nonnegative']==previous['offsets']['temperature']
            for current,before in [('nonnegative','temperature'),('neutral','neutral'),('baseline','baseline')]:
                assert frozen['variants'][current]==previous['variants'][before]
            raws,details,estimates,_=issue(panel,records,at,daily)
            old_record=earlier[at]
            for current,before in [('nonnegative','temperature'),('neutral','neutral'),('baseline','baseline')]:
                assert json_ready(raws[current])==old_record['raws'][before]
                assert details[current]==old_record['fits'][before]
                assert estimates[current]==old_record['estimates'][before]
            receipt={'passed':True,'date':date,'mature_dates':len(daily['parents']),
                'KKT_certificates':certificates,'payoff_equations':equations,'three_v50_controls_exact':True,
                'signed_scalar_objective_real_feedback':True,
                'signed_payoff_history_placeholder':'baseline history only for preflight; full run must build its own immutable signed history',
                'new_capital_outcomes_seen':False}
            receipts.append(receipt);print(receipt,flush=True)
    dump(SOURCE/'evidence/preflight.json',{'passed':True,'dates':receipts})


if __name__=='__main__':run()
