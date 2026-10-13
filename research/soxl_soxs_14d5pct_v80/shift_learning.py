"""Unlabeled recent-state density ratios for causal class-bias calibration."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'research/soxl_soxs_14d5pct_v79'))
from tilt_learning import *
import tilt_learning as prior
SOURCE=Path(__file__).resolve().parent;CACHE=ROOT/'.cache/soxl_soxs_14d5pct_v80'
PARENT=ROOT/'.cache/soxl_soxs_14d5pct_v79/runs/development';BASE=prior.PARENT
VARIANTS=['recent','lagged']
CONTROL_MAP={'source_top':'top_tilt','source_eligible':'eligible_tilt','source_both':'both_tilt','source_blend':'source_blend'}
ROUTES=['recent_ratio','lagged_ratio','uniform_top']+list(CONTROL_MAP)+['premarket_only','cash','double_cost']


def verify_parent():
    prior.verify_parent();folder=ROOT/'research/soxl_soxs_14d5pct_v79';m=read(folder/'evidence/delivery_manifest.json')
    for f,h in m['delivery_files_sha256'].items():assert sha(folder/f)==h,f
    for f,h in m['cache_files_sha256'].items():assert sha(PARENT/f)==h,f
    for f,h in m['input_files_sha256'].items():assert sha(ROOT/m['input_root']/f)==h,f
    assert read(PARENT/'verification.json')['passed'];return sha(folder/'evidence/delivery_manifest.json')


def context_windows(reg,sessions,history,at):
    today=at.tz_convert(core.ET).tz_localize(None).normalize();past=list(sessions[sessions<today]);lookup={r['date']:r for r in reg};out={}
    for mode,days in [('recent',past[-20:]),('lagged',past[-40:-20])]:
        ready=len(days)==20 and all(d in lookup and lookup[d]['feature_ready'] and all(t in history and history[t]['events']['gap_blend']['ready'] for t in lookup[d]['members']) for d in days)
        out[mode]={'ready':ready,'dates':days,'parents':[{'date':d,'members':lookup[d]['members']} for d in days] if ready else []}
    return out


def context_rows(panel,parents,history):
    X=[];keys=[];weights=[]
    for p in parents:
        for at in p['members']:
            info=history[at]['estimates']['gap_blend'];assert info['ready'];j=int(info['values'][1]>info['values'][0])
            X.append(np.r_[signed_price(panel.loc[at,FIELDS].to_numpy(float),1 if j==0 else -1),float(j)])
            keys.append([at,j]);weights.append(1/len(parents)/len(p['members']))
    return np.asarray(X).reshape(-1,14),keys,np.array(weights)


def domain_derivatives(beta,A,y,w):
    z=A@beta;p=expit(z);loss=float(w@(np.logaddexp(0,z)-y*z)+5*np.dot(beta,beta))
    gradient=A.T@(w*(p-y))+10*beta;H=A.T@((w*p*(1-p))[:,None]*A)+10*np.eye(A.shape[1])
    return loss,gradient,H


def domain_fit(old,new,old_weights,new_weights):
    old=np.asarray(old,float);new=np.asarray(new,float);u=np.asarray(old_weights,float);v=np.asarray(new_weights,float)
    if old.ndim!=2 or new.shape[1:]!=old.shape[1:] or old.shape[1]!=14 or min(len(old),len(new))==0:raise ValueError('Two nonempty fourteen-field domains required')
    if not all(np.isfinite(a).all() for a in [old,new,u,v]) or u.shape!=(len(old),) or v.shape!=(len(new),) or min(u.min(),v.min())<=0:raise ValueError('Finite contexts and positive weights required')
    u=u/u.sum();v=v/v.sum();center=u@old;scale=np.sqrt(u@((old-center)**2));scale[scale<1e-12]=1.
    A=np.column_stack([np.ones(len(old)+len(new)),(np.concatenate([old,new])-center)/scale]);y=np.r_[np.zeros(len(old)),np.ones(len(new))];w=np.r_[20*u,20*v]
    beta=np.zeros(A.shape[1])
    for iteration in range(80):
        loss,g,H=domain_derivatives(beta,A,y,w)
        if max(abs(g))<=1e-9:break
        step=np.linalg.solve(H,g);rate=1.
        for _ in range(50):
            trial=beta-rate*step;value,grad,_=domain_derivatives(trial,A,y,w)
            if value<=loss-1e-4*rate*np.dot(g,step) or (value<=loss+64*np.finfo(float).eps*max(1.,abs(loss)) and max(abs(grad))<max(abs(g))/2):
                beta=trial;break
            rate*=.5
        else:raise ValueError('Domain Newton line search failed')
    loss,g,_=domain_derivatives(beta,A,y,w)
    if max(abs(g))>1e-7:raise ValueError('Domain gradient failed')
    raw_log_ratio=A[:len(old)]@beta;ratios=np.exp(np.clip(raw_log_ratio,-np.log(5),np.log(5)))
    return {'center':center,'scale':scale,'beta':beta,'objective':loss,'gradient_inf':float(max(abs(g))),'iterations':iteration,
        'old_domain_weights':20*u,'new_domain_weights':20*v,'raw_log_ratios':raw_log_ratio,'ratios':ratios,
        'clipped_low':int((raw_log_ratio< -np.log(5)).sum()),'clipped_high':int((raw_log_ratio>np.log(5)).sum())}


def reweight(weights,ratios,members):
    weights=np.asarray(weights,float);ratios=np.asarray(ratios,float)
    if weights.shape!=ratios.shape or len(members)!=len(weights) or not np.isfinite(ratios).all() or min(ratios)<=0:raise ValueError('Positive aligned ratios required')
    w=weights*ratios;w*=weights.sum()/w.sum();mass={}
    for (at,_),value in zip(members,w):
        day=pd.Timestamp(at).tz_convert(core.ET).date();mass[day]=mass.get(day,0.)+float(value)
    b=np.array(list(mass.values()));ess=float(b.sum()**2/(b@b))
    return w,ess


def daily_models(panel,reg,market,history,old_daily,at):
    windows=context_windows(reg,market.sessions,history,at);out={'at':at,'cutoff':day_cutoff(at),'source_ready':old_daily['ready'],'variants':{}}
    parents=[]
    for p in old_daily['parents']:
        parents.append({**p,'date':pd.Timestamp(p['date']),'members':[pd.Timestamp(t) for t in p['members']]})
    if old_daily['ready']:
        P,y,w,meta=prior.samples(panel,parents,history,'top');X,keys,_=context_rows(panel,parents,history)
        assert keys==meta['members'];np.testing.assert_allclose(w,old_daily['models']['top']['weights'],atol=1e-11,rtol=1e-11)
    for mode,window in windows.items():
        record={'ready':False,'window':window,'reason':'source_or_context_unready'}
        if old_daily['ready'] and window['ready']:
            Z,targetkeys,v=context_rows(panel,window['parents'],history);domain=domain_fit(X,Z,w,v);adjusted,ess=reweight(w,domain['ratios'],keys)
            model=prior.fit(P,y,adjusted) if ess>=20 else {'ready':False,'reason':'weighted_date_ess_below20'}
            record={'ready':model['ready'],'window':window,'domain':domain,'model':model,'members':keys,'target_members':targetkeys,
                'weights':adjusted,'original_weight_sum':float(w.sum()),'weighted_date_ess':ess,
                'overlap_dates':len(set(meta['dates']) & set(window['dates']))}
        out['variants'][mode]=record
    return out


def issue(at,row,daily,old,previous):
    cash={'at':at,'date':row.date,'clock':row.clock,'ready':True,'action':'CASH','allocation':0.,'values':None}
    inactive={'at':at,'date':row.date,'clock':row.clock,'ready':False,'action':'CASH','allocation':0.,'probabilities':None,'values':None};out={}
    for mode,v in daily['variants'].items():
        estimate=prior.estimate(old,{'ready':v['ready'],'models':{mode:v.get('model')}},mode)
        out[mode+'_ratio']={'at':at,'date':row.date,'clock':row.clock,**estimate}
    out['uniform_top']=previous['events']['top_tilt'] if daily['variants']['recent']['ready'] else inactive
    for name,source in CONTROL_MAP.items():out[name]=previous['events'][source]
    out['cash']=cash;out['double_cost']=dict(out['recent_ratio'])
    out['premarket_only']=dict(out['recent_ratio']) if row.clock==CLOCKS[0] else cash
    return out
