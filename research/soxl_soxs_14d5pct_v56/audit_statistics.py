"""Recompute each month and distinguish the selected mean from all-month risk."""
from activity_learning import core,pd,np,read
from verify_continuation import near

def audit_statistics(run):
    r=read(run/'results.json');p=read(run/'registration.json')['preregistration']['protocol'];settings=p[r['phase']]
    months=settings.get('months',settings.get('regression_months'));checks=0
    for name,s in r['routes'].items():
        eq=pd.read_csv(run/(name+'_equity.csv'),parse_dates=['date']).set_index('date')
        try:t=pd.read_csv(run/(name+'_trades.csv'))
        except pd.errors.EmptyDataError:t=pd.DataFrame()
        end=pd.Timestamp(str(eq.index[-1].date())+' 23:59:59',tz=core.ET)
        done=t[t.exit_at.notna()&(pd.to_datetime(t.exit_at,utc=True)<=end)] if len(t) else t
        assert s['entries']==len(t) and s['completed_trades']==len(done) and s['pending_trades']==len(t)-len(done)
        for field,reason in [('target_exits','target'),('risk_stop_exits','risk_stop'),('expiry_exits','expiry')]:assert s[field]==(int(done.reason.eq(reason).sum()) if len(done) else 0)
        rates={};prior=1.
        for month,g in eq.groupby(eq.index.to_period('M')):rates[str(month)]=g.equity.iloc[-1]/prior-1;prior=g.equity.iloc[-1]
        for row in s['monthly']:near(row['return'],rates[row['month']])
        selected=[rates[m] for m in months];near(s['primary_average'],np.mean(selected));near(s['geometric_average'],np.exp(np.mean(np.log1p(selected)))-1)
        nav=eq.equity.to_numpy();near(s['continuous_return'],nav[-1]-1);near(s['close_max_drawdown'],min(nav/np.maximum.accumulate(np.r_[1.,nav])[1:]-1))
        all_positive=all(v>=-1e-12 for v in rates.values());assert s['all_months_nonnegative']==all_positive
        assert s['primary_pass']==bool(np.mean(selected)>=.1 and all_positive);checks+=13+len(rates)
    return checks
