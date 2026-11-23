"""Post-freeze diagnostics for absolute allocations and minimum inter-fill time."""
import json
from pathlib import Path

from comparison import attribution
from study_scale import Config, Context, PATHS, compact

HERE = Path(__file__).resolve().parent
FEE = 2.50
CAPITAL = 1000
PROFILES = [(100,150,350), (100,200,300)]
WAITS = (0,1,3,6,12)
TOTALS = (300,600,900)


def replay(ctx, config, path):
    run = ctx.execute(config,path,FEE)
    result = compact(run)
    total = sum(config.amounts)
    matched = attribution(run['lots'],FEE,total,CAPITAL)
    result.update(config=config.record(), path=path, cycle_totals=run['summary']['cycles'],
                  gross_at_same_cycle_starts_full_size=matched['fixed_600_on_same_cycle_starts'],
                  same_cycle_starts_cash_limited=matched['same_starts_600_with_cash_limit'],
                  same_start_one_shot_cash_blocked=matched['capital_blocked_600'],
                  frozen_trajectory_breakeven_nominal=(total*result['fees']/result['gross']
                                                      if result['gross']>0 else None))
    return result


def main():
    blocks = {}
    for name in ('november','october_check'):
        ctx = Context(name)
        profiles, singles, example = [], [], []
        for total in TOTALS:
            single = Config((total,),(),'q90','hold')
            singles.extend(replay(ctx,single,path) for path in PATHS)
            for profile in PROFILES:
                amounts = tuple(int(n*total/600) for n in profile)
                assert sum(amounts)==total
                for w in WAITS:
                    config = Config(amounts,(w,w),'q90','hold')
                    for path in PATHS:
                        z = replay(ctx,config,path)
                        z.update(profile=list(profile),total_cap=total,min_fill_gap_minutes=5*(w+1))
                        profiles.append(z)
        ex = Config((100,200,300),(0,0),'q50','hold')
        for path in PATHS:
            example.append(replay(ctx,ex,path))
        blocks[name]=dict(q90_profiles=profiles,q90_single=singles,q50_example_five_min=example)
    output=dict(study='Post-freeze sensitivity; no parameter reselection', initial=CAPITAL,
                fee_each_side=FEE, scale_breakeven='Holding all realized entry/exit times and fill counts fixed. Estimated notional can exceed available cash. Cash-limited order counts can change when the size changes.',
                blocks=blocks)
    (HERE/'sensitivity.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    main()
