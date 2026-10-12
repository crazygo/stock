"""Causal run-length filtering changes memory weights, not historical labels."""
from pathlib import Path
import sys
from scipy.special import gammaln,logsumexp
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'soxl_soxs_14d5pct_v27'))
from clock_reliability import (core,np,pd,read,dump,sha,labels,prepare_inputs,base_forecast,
    day_cutoff,restrict_market,development_gate,CLOCKS,tag,rows,decide)
from reliability import fit_error,predict_error,block_counts,weights

HAZARD=1/63
ROUTES=['regime_band','regime_point','hazard_only_band','premarket_only','source_v27','cash','double_cost']


class RunFilter:
    def __init__(self):
        self.logp=np.array([0.]);self.mu=np.zeros((1,3));self.k=np.ones((1,3))
        self.a=np.full((1,3),2.);self.b=np.full((1,3),2.)

    def update(self,x):
        x=np.asarray(x,float);assert x.shape==(3,) and np.isfinite(x).all()
        df=2*self.a;scale2=self.b*(self.k+1)/(self.a*self.k)
        lp=(gammaln((df+1)/2)-gammaln(df/2)-.5*np.log(df*np.pi*scale2)
            -(df+1)/2*np.log1p((x-self.mu)**2/(df*scale2))).sum(axis=1)
        joint=self.logp+lp
        q=np.r_[logsumexp(joint)+np.log(HAZARD),joint+np.log1p(-HAZARD)]
        self.logp=q-logsumexp(q)
        k=self.k+1;mu=(self.k*self.mu+x)/k;b=self.b+self.k*(x-self.mu)**2/(2*k)
        self.mu=np.vstack([np.zeros(3),mu]);self.k=np.vstack([np.ones(3),k])
        self.a=np.vstack([np.full(3,2.),self.a+.5]);self.b=np.vstack([np.full(3,2.),b])
        return np.exp(self.logp)


def coordinates(row):
    sv=max(float(row.SOXX_vol20),1e-4);qv=max(float(row.QQQ_vol20),1e-4)
    return np.clip([row.SOXX_ret1/sv,(row.SOXX_ret1-row.QQQ_ret1)/np.sqrt(sv*sv+qv*qv),
        2*(2*row.breadth_ret1_positive_fraction-1)],-5,5)


def filter_states(features,sessions):
    """One observation per date, released at the premarket decision, never four."""
    f=RunFilter();records={};seen=0
    daily=features[features.clock==CLOCKS[0]].set_index('date',drop=False)
    dates=sessions[(sessions>=features.date.min())&(sessions<=features.date.max())]
    for index,date in enumerate(dates):
        at=pd.Timestamp(str(date.date())+' '+CLOCKS[0],tz=core.ET).tz_convert('UTC')
        x=coordinates(daily.loc[date]) if date in daily.index else np.full(3,np.nan);valid=np.isfinite(x).all()
        if valid:posterior=f.update(x);seen+=1
        else:f=RunFilter();posterior=None;seen=0
        records[date]={'at':at,'date':date,'index':index,'coordinates':x,'consecutive_observations':seen,
            'ready':seen>=20,'posterior':posterior,'last_five_change_probability':float(posterior[:5].sum()) if valid else None,
            'expected_run_length':float(posterior@np.arange(len(posterior))) if valid else None}
    return records


def memory_weights(ids,panel,states,date,mode):
    state=states[date];ages=np.array([state['index']-states[panel.loc[t,'date']]['index'] for t in ids]);assert (ages>0).all()
    posterior=np.asarray(state['posterior']);tail=np.r_[np.cumsum(posterior[::-1])[::-1],0.]
    membership=np.array([tail[min(int(age)+1,len(posterior))] for age in ages]) if mode=='regime' else np.power(1-HAZARD,ages+1)
    w=weights(len(ids),63)*(.25+.75*membership);w*=len(ids)/w.sum()
    return w,membership,ages


def estimate(panel,history,states,at,raw,ids,mode):
    state=states[panel.loc[at,'date']]
    info={'ready':False,'members':ids,'variant':mode,'point':None,'lower':None}
    if raw is None or len(ids)<60 or not panel.loc[at,'feature_ready'] or not state['ready']:return info
    X=np.array([history[t] for t in ids]);x=np.asarray(raw);truth=panel.loc[ids,['C','D']].to_numpy(float)
    Y=np.c_[truth[:,0]+truth[:,1],truth[:,0]-truth[:,1]]-X
    w,membership,ages=memory_weights(ids,panel,states,panel.loc[at,'date'],mode)
    effective=w.sum()**2/(w@w)
    info.update(weights=w.tolist(),membership=membership.tolist(),ages=ages.tolist(),effective_dates=effective,
        current_regime_mature_mass=float(membership.sum()))
    # A common floor across all obsolete samples would cancel on normalization.
    # Do not let that manufacture confidence just after a detected regime change.
    if effective<30 or membership.sum()<10:return info
    model=fit_error(X,Y,w);point=x+predict_error(model,x);counts,starts,seed=block_counts(len(ids),at,10)
    samples=np.array([x+predict_error(fit_error(X,Y,w*c),x) for c in counts]);radius=max(0.,float(np.quantile(np.max(samples-point,axis=1),.9,method='higher')))
    info.update(ready=True,point_model=model,point=point.tolist(),bootstrap_predictions=samples.tolist(),
        block_starts=starts.tolist(),seed=seed,radius=radius,lower=(point-radius).tolist(),length=10)
    return info


def issue(panel,history,states,at,raw):
    same=panel[panel.clock==panel.loc[at,'clock']];ids,feedback=rows(same,history,at)
    return {'feedback':feedback,'state_date':panel.loc[at,'date'],
        'estimates':{mode:estimate(same,history,states,at,raw,ids,mode) for mode in ['regime','hazard']}}


def route_events(at,date,clock,decision,old):
    fb=decision['feedback'];e=decision['estimates'];band=decide(e['regime'],fb,True);point=decide(e['regime'],fb,False)
    hazard=decide(e['hazard'],fb,True);cash={'ready':True,'action':'CASH','allocation':0.}
    routes={'regime_band':band,'regime_point':point,'hazard_only_band':hazard,'premarket_only':band if clock==CLOCKS[0] else cash,
        'source_v27':old,'cash':cash,'double_cost':band}
    return {n:{**ev,'at':at,'date':date} for n,ev in routes.items()}
