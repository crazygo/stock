"""Independent raw-bar barrier, capital and regression-boundary verification."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
from risk import core


def verify(run):
    run=Path(run)
    frozen=json.loads((run/'frozen_policy.json').read_text())
    trials=json.loads((run/'selection_trials.json').read_text())
    results=json.loads((run/'results.json').read_text())
    assert len(trials)==72
    assert all(t['selection_end']=='2026-04-30' and pd.Timestamp(t['last_contractual_expiry'])<pd.Timestamp('2026-05-01') for t in trials)
    assert all(m['month']<'2026-05' for t in trials for m in t['score']['monthly'])
    chosen=sorted(trials,key=lambda t:(-t['score']['minimum_month_return'],-t['score']['all_months_geometric_return'],-t['score']['close_max_drawdown'],t['id']))[0]
    assert chosen['id']==frozen['chosen']['id']==results['selected_policy']
    assert frozen['regression_replays_before_freeze']==0
    market=core.load_market()
    source=json.loads((run/'input_manifest.json').read_text())
    assert all(market.manifest[s]['sha256']==source['market'][s]['sha256'] for s in core.SYMBOLS)
    total=0; gaps=0; ambiguous=0
    for name,multiple in [('frozen_candidate',1),('double_cost',2)]:
        trades=pd.read_csv(run/f'{name}_trades.csv')
        equity=pd.read_csv(run/f'{name}_equity.csv',parse_dates=['date']).set_index('date')
        assert (equity.cash_settled>=-1e-12).all()
        assert not trades.entry_date.duplicated().any()
        fee=.0005*multiple; slip=.0005*multiple
        closed_profit=0.; open_profit=0.
        last_exit=None
        for t in trades.itertuples():
            at=pd.Timestamp(t.decision_at); start=pd.Timestamp(t.entry_at); finish=pd.Timestamp(t.expiry_at)
            assert start>at+pd.Timedelta(seconds=30)
            assert last_exit is None or start>last_exit
            f=market.bars[t.symbol]
            first=f[f.start_at==start].iloc[0]
            entry=float(first.open)*(1+slip)*float(first.share_factor)
            assert np.isclose(entry,t.entry_fill_economic,rtol=0,atol=1e-9)
            assert np.isclose(t.units_economic*entry*(1+fee),t.capital,rtol=0,atol=1e-10)
            target=math.ceil(first.open*(1+slip)*1.05*100-1e-10)/100*first.share_factor
            stop=entry*(1-chosen['risk']['stop_fraction']); raised=False; pending=None
            path=f[(f.start_at>=start)&(f.end_at<=finish)&(f.session_type=='regular')]
            event=None
            for bar in path.itertuples():
                if pending is not None and bar.start_at > pending[1]:
                    stop=max(stop,pending[0]);raised=True;pending=None
                raw_stop=math.floor(stop/bar.share_factor*100+1e-10)/100
                raw_limit=math.ceil(target/bar.share_factor*100-1e-10)/100
                if bar.low<=raw_stop:
                    event=(bar.end_at,min(bar.open,raw_stop)*(1-slip)*bar.share_factor,'profit_protection' if raised else 'risk_stop')
                    gaps+=int(bar.open<raw_stop); ambiguous+=int(bar.high>=raw_limit)
                    break
                if bar.high>=raw_limit:
                    event=(bar.end_at,raw_limit*bar.share_factor,'target'); break
                trigger=chosen['risk']['breakeven_trigger']
                if trigger is not None and not raised and pending is None and bar.high*bar.share_factor>=entry*(1+trigger):
                    needed=entry*(1+fee)/((1-fee)*(1-slip))
                    pending=(math.ceil(needed/bar.share_factor*100-1e-10)/100*bar.share_factor,bar.available_at+pd.Timedelta(seconds=30))
            if event is None and len(path) and path.end_at.iloc[-1]==finish:
                event=(finish,path.econ_close.iloc[-1]*(1-slip),'expiry')
            if event is not None:
                assert event[0]==pd.Timestamp(t.exit_at),(name,t.symbol,t.entry_at)
                assert event[2]==t.reason
                assert np.isclose(event[1],t.exit_fill_economic,rtol=0,atol=1e-9)
                last_exit=event[0]
            else:
                assert pd.isna(t.exit_at)
            if pd.notna(t.exit_at) and pd.Timestamp(t.exit_at)<=pd.Timestamp('2026-09-30 16:00',tz=core.ET):
                independent_net=(t.exit_fill_economic*(1-fee)+t.dividend_per_economic_unit)/(entry*(1+fee))-1
                assert np.isclose(independent_net,t.net_return,rtol=0,atol=1e-10)
                closed_profit+=t.capital*independent_net
            else:
                mark=float(market.daily[t.symbol].loc[equity.index[-1],'close'])
                open_profit+=t.units_economic*mark+t.units_economic*t.accrued_dividends-t.capital
            total+=1
        assert np.isclose(float(equity.equity.iloc[-1]),1+closed_profit+open_profit,rtol=0,atol=1e-9)
        ends=equity.equity.groupby(equity.index.to_period('M')).last()
        monthly=ends/pd.Series([1.,*ends.iloc[:-1].tolist()],index=ends.index)-1
        reported={m['month']:m['return'] for m in results[name]['monthly']}
        assert all(np.isclose(v,reported[str(m)],rtol=0,atol=1e-10) for m,v in monthly.items())
        mean=float(np.mean([monthly[pd.Period(m)] for m in ['2026-05','2026-07','2026-09']]))
        assert np.isclose(mean,results[name]['primary_average'],rtol=0,atol=1e-10)
    if not results['selection_admission']:
        admitted=pd.read_csv(run/'admitted_policy_equity.csv')
        assert admitted.equity.eq(1.).all()
        assert not results['admitted_policy']['primary_pass']
    out={'pass':True,'selection_trials':len(trials),'no_regression_month_in_selection':True,
         'frozen_policy':chosen['id'],'only_one_frozen_candidate_replayed':True,
         'raw_bar_trade_checks':total,'adverse_gap_exits':gaps,'ambiguous_stop_first_exits':ambiguous,
         'monthly_returns_and_terminal_equity_recomputed':True,
         'cash_not_counted_as_return_success':True,
         'frozen_policy_sha256':hashlib.sha256((run/'frozen_policy.json').read_bytes()).hexdigest()}
    core.dump(run/'verification.json',out)
    print(json.dumps(out,ensure_ascii=False,indent=2))
    return out


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    verify(p.parse_args().run)
