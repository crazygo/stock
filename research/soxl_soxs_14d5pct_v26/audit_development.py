"""Independent result totals and actual rejection of case replay after failed development."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd
from reliability import core
from run_reliability import sha
from verify_continuation import read,near


def audit(run):
    run=Path(run);result=read(run/'results.json');v=read(run/'verification.json');reg=read(run/'registration.json')
    assert reg['phase']=='development' and v['passed'] and v['results_sha256']==sha(run/'results.json')
    checks=0
    for n,s in result['routes'].items():
        eq=pd.read_csv(run/(n+'_equity.csv'),parse_dates=['date']).set_index('date')
        try:t=pd.read_csv(run/(n+'_trades.csv'))
        except pd.errors.EmptyDataError:t=pd.DataFrame()
        end=pd.Timestamp(str(eq.index[-1].date())+' 23:59:59',tz=core.ET)
        done=t[t.exit_at.notna()&(pd.to_datetime(t.exit_at,utc=True)<=end)] if len(t) else t
        assert s['entries']==len(t) and s['completed_trades']==len(done) and s['pending_trades']==len(t)-len(done)
        for field,reason in [('target_exits','target'),('risk_stop_exits','risk_stop'),('expiry_exits','expiry')]:
            assert s[field]==(int(done.reason.eq(reason).sum()) if len(done) else 0)
        nav=eq.equity.to_numpy();prior=1.;rates=[]
        for _,g in eq.groupby(eq.index.to_period('M')):
            rates.append(g.equity.iloc[-1]/prior-1);prior=g.equity.iloc[-1]
        near(s['primary_average'],np.mean(rates));near(s['geometric_average'],np.exp(np.mean(np.log1p(rates)))-1)
        near(s['continuous_return'],nav[-1]-1);near(s['close_max_drawdown'],min(nav/np.maximum.accumulate(np.r_[1.,nav])[1:]-1))
        assert s['all_months_nonnegative']==all(x>=-1e-12 for x in rates)
        assert s['primary_pass']==(np.mean(rates)>=.1 and all(x>=-1e-12 for x in rates))
        checks+=12
    assert not result['development_gate']['passed'] and not result['case_regression']['executed']
    target=run.parent/'blocked_cases_guard_check'
    assert not target.exists()
    command=[sys.executable,str(Path(__file__).parent/'run_reliability.py'),'--output',str(target),'--case-after',str(run)]
    proc=subprocess.run(command,capture_output=True,text=True)
    assert proc.returncode!=0 and 'Development primary failed; new case replay is forbidden' in proc.stderr
    assert not target.exists()
    receipt={'passed':True,'independent_statistic_assertions':checks,'new_case_replay_blocked':True,
        'guard_exit_code':proc.returncode,'guard_message':'Development primary failed; new case replay is forbidden',
        'case_output_created':False,'results_sha256':sha(run/'results.json'),'verification_sha256':sha(run/'verification.json'),
        'audit_source_sha256':sha(Path(__file__)),'post_result_audit_only':True}
    core.dump(run/'development_audit.json',receipt);print(json.dumps(receipt))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);audit(p.parse_args().run)
