"""Independent formula, coverage, causal-boundary and counterexample checks."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,math
import numpy as np
import pandas as pd
import build
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
def independent(a):
    def four(z):
        r=z[-60:];left=r[:-1]-r[:-1].mean();right=r[1:]-r[1:].mean();energy=(r-r.mean())**2
        return np.array([sum(z[-120:])/120*252,np.sqrt(sum((r-r.mean())**2)/59)*np.sqrt(252),sum(left*right)/np.sqrt(sum(left**2)*sum(right**2)),sum(sorted(energy)[-6:])/sum(energy)])
    v=four(a);series=[]
    for i in range(20,-1,-1):
        q=four(a if i==0 else a[:-i]);series.append([math.tanh(q[0]/.5),math.tanh((q[1]-.35)/.25),q[2],math.tanh((q[3]-.55)/.2)])
    s=math.exp(-10*sum(abs(series[i][k]-series[i-1][k]) for i in range(1,21) for k in range(4))/80/2)
    return np.r_[v,s]
def main():
    p=json.loads((HERE/'REPORT_POINTER.json').read_text());snapshot_path=ROOT/p['snapshot'];assert build.sha(snapshot_path)==p['snapshot_sha256'];j=json.loads(snapshot_path.read_text());dp=json.loads((HERE/'DATA_POINTER.json').read_text());mp=ROOT/dp['manifest'];assert build.sha(mp)==dp['sha256'];m=json.loads(mp.read_text());cal=[r['session_date'] for r in build.calendar(build.START,build.END)['sessions']];count=0;future_count=0
    records={r['ticker']:r for r in j['stocks']}
    for r in j['stocks']:
        d=m['datasets']['US.'+r['ticker']];path=ROOT/d['path'];assert build.sha(path)==d['sha256'];f=pd.read_parquet(path);f['day']=f.time_key.astype(str).str[:10];assert f.day.max()<=build.END
        c=f.close.to_numpy(float);a=np.diff(np.log(c))[-140:];expected=independent(a);assert np.allclose(expected,r['v'],rtol=1e-9,atol=1e-10),r['ticker'];count+=5
        for n in [20,60,120]:assert np.isclose(r['price']['ret'+str(n)],c[-1]/c[-n-1]-1)
        assert set(d['missing_sessions'])==set(cal)-set(f.day)
        if d['status']=='complete':assert set(x for x in cal if x>='2026-01-01')<=set(f.day)
        if r['ticker'] in ['ON','POWL','ENS','AVGO','AEIS','TTD']:
            v,st,_=build.descriptors(f,r['price_cutoff'],cal);future=f.iloc[-1:].copy();future['day']='2026-10-07';future['close']*=10
            v2,st2,_=build.descriptors(pd.concat([f,future]),r['price_cutoff'],cal);assert np.allclose(v,v2) and st==st2;future_count+=1
            gap=f.drop(f.index[-10]);vg,sg,_=build.descriptors(gap,r['price_cutoff'],cal);assert all(s=='missing_sessions' for s in sg);assert all(x is None for x in vg)
    assert j['metadata']['priority']==['ON','POWL','ENS']
    assert records['TTD']['research_status']=='业务待修复';assert records['TTD']['ret120']<-.45;assert records['TTD']['v'][4]>.9;assert records['TTD']['days_since_60d_close_low']==0
    assert records['AVGO']['research_status']=='反转待确认';assert records['AEIS']['research_status']=='修复中';assert not any(r['coverage']['status']!='complete' for r in j['stocks'] if r['research_status']=='优先跟踪')
    synthetic=-.005+.0002*np.sin(np.arange(140)*.8);v=independent(synthetic);assert v[0]<-.1 and v[4]>.75
    kernel_hash=build.sha(HERE/'five_traits.py');assert kernel_hash==build.sha(ROOT/'analysis/ai_trend_quadrant_v2/model.py')
    for key,value in j['metadata']['inputs'].items():assert build.sha(ROOT/key)==value
    result={'passed':True,'verified_at':datetime.now(timezone.utc).isoformat(),'independent_trait_values':count,'causal_future_exclusion_cases':future_count,'internal_gap_cases':future_count,'all_original_members_preserved':len(j['stocks'])==150,'priority':j['metadata']['priority'],'TTD_stable_downtrend_counterexample':True,'kernel_unchanged':True,'raw_inputs_verified':len(m['datasets']),'current_primary_two_year_bars':{t:m['datasets']['US.'+t]['bars'] for t in ['ON','POWL','ENS','AVGO','AEIS','TTD']},'strategy_effect_validated':False}
    (HERE/'DATA_VERIFICATION.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
