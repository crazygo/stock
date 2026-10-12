"""Independent multiclock cash, T+1, dividend and one-entry reconstruction."""
import numpy as np
import pandas as pd
from planning import core
from verify_continuation import near,key


def book_check(run,name,market,features,lab,events,multiple,stats,months,require_full_cash=True):
    eq=pd.read_csv(run/(name+'_equity.csv'),parse_dates=['date']).set_index('date')
    ds=pd.read_csv(run/(name+'_decisions.csv')); ds.index=pd.to_datetime(ds.decision_at,utc=True)
    try: trades=pd.read_csv(run/(name+'_trades.csv'))
    except pd.errors.EmptyDataError:trades=pd.DataFrame()
    actual={pd.Timestamp(r.decision_at):r for r in trades.itertuples()}; actions={pd.Timestamp(e['at']):e for e in events}
    cash=1.; unsettled=[];receivables=[];held=None;units=0.;count=0; ticks=0
    for day,mark in eq.iterrows():
        cash+=sum(v for d,v in unsettled if d<=day);unsettled=[(d,v) for d,v in unsettled if d>day]
        cash+=sum(v for d,v in receivables if d<=day);receivables=[(d,v) for d,v in receivables if d>day]
        blocked=False
        if held is not None:
            for d in held['dividends']:
                if d['ex']==day:
                    value=units*d['economic_amount']
                    if d['pay']<=day:cash+=value
                    else:receivables.append((d['pay'],value))
        for at,r in features[features.date==day].iterrows():
            if held is not None and held['exit_at'] is not None and held['exit_at']<=at:
                unsettled.append((market.sessions[market.sessions>day][0],units*held['exit_fill_economic']*(1-.0005*multiple)))
                held=None;blocked=True
            e=actions[at];assert 0<=e['allocation']<=1 and (e['action'] in core.ETFS)==(e['allocation']>0)
            status='no_action'
            if held is not None:status='position_held'
            elif blocked:status='daily_entry_limit'
            elif cash<=1e-15:status='cash_unsettled'
            elif require_full_cash and (unsettled or receivables):status='waiting_for_full_cash'
            elif e['action'] in core.ETFS:
                chosen=lab[key(at,e['action'])]
                if 'entry_fill_economic' not in chosen:status=chosen['reason']
                else:
                    assert at in actual and actual[at].symbol==e['action']
                    capital=cash*e['allocation'];units=capital/(chosen['entry_fill_economic']*(1+.0005*multiple))
                    near(actual[at].capital,capital);near(actual[at].units_economic,units);near(actual[at].allocation,e['allocation'])
                    assert actual[at].reason==chosen['reason']
                    for c in ['entry_fill_economic','exit_fill_economic','net_return','dividend_per_economic_unit']:
                        v=chosen.get(c,np.nan)
                        if v is not None and np.isfinite(v):near(getattr(actual[at],c),v)
                    if chosen['exit_at'] is not None:assert pd.Timestamp(actual[at].exit_at)==chosen['exit_at']
                    held=chosen;cash-=capital;count+=1;blocked=True;status='buy_'+e['action']
            if not status.startswith('buy_'):assert at not in actual
            saved=ds.loc[at]; assert saved.decision==status and saved.requested_action==e['action']
            near(saved.requested_allocation,e['allocation']); ticks+=1
        if held is not None and held['exit_at'] is not None and held['exit_at'].tz_convert(core.ET).date()==day.date():
            unsettled.append((market.sessions[market.sessions>day][0],units*held['exit_fill_economic']*(1-.0005*multiple)));held=None
        asset=units*market.daily[held['symbol']].loc[day,'close'] if held else 0.
        pending=sum(v for _,v in unsettled);owed=sum(v for _,v in receivables)
        for field,value in [('cash_settled',cash),('cash_unsettled',pending),('receivables',owed),('position_value',asset),('equity',cash+pending+owed+asset)]:near(mark[field],value)
        assert mark.symbol==(held['symbol'] if held else 'CASH') and mark.mark_complete
    assert count==len(trades) and ticks==len(ds)
    prior=1.;rates=[]
    for month,g in eq.groupby(eq.index.to_period('M')):
        rate=g.equity.iloc[-1]/prior-1;prior=g.equity.iloc[-1];rates.append((str(month),rate))
        near(rate,next(m['return'] for m in stats['monthly'] if m['month']==str(month)))
    avg=np.mean([v for m,v in rates if m in months]);near(stats['primary_average'],avg);near(stats['continuous_return'],eq.equity.iloc[-1]-1)
    assert stats['primary_pass']==(avg>=.1 and all(v>=-1e-12 for _,v in rates))
    return {'name':name,'actual_trades':count,'daily_marks':len(eq),'decision_ticks':ticks,'capital_reconciled':True}
