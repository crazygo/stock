"""Suffix sufficient statistics and normal equations, independently evaluated."""
from scipy.stats import t as student_t
from regime_learning import *
from verify_reliability import solve,close
from verify_continuation import near


def reference_states(features,sessions):
    pre=features[features.clock==CLOCKS[0]].set_index('date');sequence=[];prob=np.ones(1);out={}
    dates=sessions[(sessions>=features.date.min())&(sessions<=features.date.max())]
    for index,day in enumerate(dates):
        if day in pre.index:
            r=pre.loc[day];sv=max(r.SOXX_vol20,.0001);qv=max(r.QQQ_vol20,.0001)
            x=np.clip([r.SOXX_ret1/sv,(r.SOXX_ret1-r.QQQ_ret1)/np.hypot(sv,qv),4*r.breadth_ret1_positive_fraction-2],-5,5)
            valid=np.isfinite([r.SOXX_ret1,r.QQQ_ret1,r.SOXX_vol20,r.QQQ_vol20,r.breadth_ret1_positive_fraction]).all()
        else:valid=False;x=np.full(3,np.nan)
        if valid:
            n=np.arange(len(sequence)+1);sums=np.vstack([np.zeros(3),np.cumsum(sequence[::-1],axis=0)]) if sequence else np.zeros((1,3))
            squares=np.vstack([np.zeros(3),np.cumsum(np.square(sequence[::-1]),axis=0)]) if sequence else np.zeros((1,3))
            k=1+n[:,None];mean=sums/k;alpha=2+n[:,None]/2;beta=2+(squares-sums**2/k)/2
            scale=np.sqrt(beta*(k+1)/(alpha*k));ll=student_t.logpdf(x,df=2*alpha,loc=mean,scale=scale).sum(axis=1)
            # Preserve small but representable probabilities without log(0).
            positive=prob>0;z=np.full(len(prob),-np.inf);z[positive]=np.log(prob[positive])+ll[positive]
            likelihood=np.exp(z-np.max(z));likelihood/=likelihood.sum()
            prob=np.r_[1/63,(62/63)*likelihood];sequence.append(x)
            posterior=prob.copy()
        else:sequence=[];prob=np.ones(1);posterior=None
        out[day]={'at':pd.Timestamp(str(day.date())+' 09:25:30',tz=core.ET).tz_convert('UTC'),'date':day,'index':index,
            'coordinates':x,'consecutive_observations':len(sequence),'ready':len(sequence)>=20,'posterior':posterior,
            'last_five_change_probability':sum(prob[:5]) if valid else None,
            'expected_run_length':sum(i*v for i,v in enumerate(prob)) if valid else None}
    return out


def check_estimate(panel,history,states,ids,at,raw,mode,info):
    assert list(pd.to_datetime(info['members'],utc=True))==ids and info['variant']==mode
    state=states[panel.loc[at,'date']];eligible=raw is not None and len(ids)>=60 and panel.loc[at,'feature_ready'] and state['ready']
    if not eligible:assert not info['ready'] and info['point'] is None and info['lower'] is None;return 0
    n=len(ids);ages=np.array([state['index']-states[panel.loc[t,'date']]['index'] for t in ids]);assert (ages>0).all()
    membership=np.array([sum(state['posterior'][age+1:]) if mode=='regime' else (62/63)**(age+1) for age in ages])
    w=np.array([2.**((j-n+1)/63) for j in range(n)])*(.25+.75*membership);w*=n/w.sum()
    close(info['weights'],w);close(info['membership'],membership);assert info['ages']==ages.tolist()
    effective=w.sum()**2/(w@w);near(info['effective_dates'],effective)
    near(info['current_regime_mature_mass'],membership.sum())
    if effective<30 or membership.sum()<10:assert not info['ready'];return 0
    assert info['ready'];X=np.array([history[t] for t in ids]);truth=panel.loc[ids,['C','D']].to_numpy()
    Y=np.column_stack([truth[:,0]+truth[:,1],truth[:,0]-truth[:,1]])-X;x=np.array(raw)
    value,model=solve(X,Y,w,x);point=x+value;close(info['point'],point)
    for key in model:close(info['point_model'][key],model[key])
    rng=np.random.default_rng(826+at.tz_convert(core.ET).date().toordinal());starts=rng.integers(0,n,size=(199,(n+9)//10))
    assert starts.tolist()==info['block_starts'];pred=[]
    for draw in starts:
        indexes=[(int(start)+j)%n for start in draw for j in range(10)][:n]
        v,_=solve(X[indexes],Y[indexes],w[indexes],x);pred.append(x+v)
    close(info['bootstrap_predictions'],pred);margin=max(0.,sorted(max(pair-point) for pair in pred)[179])
    near(info['radius'],margin);close(info['lower'],point-margin);return 400
