"""Independent dynamic-state reconstruction, conditional draws and full contract integral."""
import numpy as np
import pandas as pd
from scipy.stats import qmc
from path_learning import core,CLOCKS
from audit_paths import reference_values
from verify_reliability import close


def state_equations(history):
    changes=history[:,1:,0]/history[:,:-1,0]-1;center=changes.sum(axis=1)/20
    sigma=np.maximum(np.sqrt(np.sum((changes-center[:,None])**2,axis=1)/19),1e-4)
    r=history[:,-1]/history[:,-6]-1
    return np.stack((np.minimum(5,np.maximum(-5,(r[:,0]-r[:,1])/(sigma*np.sqrt(5)))),(np.sum(r[:,2:]>0,axis=1)/4)-1,np.log(sigma)),axis=1)


def independent_walk(bank,members,first,points,history,quotes,reference,clock,plan):
    n=len(points);x=bank['states'][bank['blocks'][members,0],0,:3];w=np.exp2(-np.arange(len(members))[::-1]/126);w/=sum(w)
    mu=np.sum(w[:,None]*x,axis=0);sd=np.sqrt(np.sum(w[:,None]*(x-mu)**2,axis=0));sd=np.where(sd<1e-12,1.,sd)
    past=np.repeat(history[None,:,:],n,axis=0);selected=[];states=[];effective=[];c=CLOCKS.index(clock)
    for j in range(points.shape[1]):
        z=state_equations(past);states.append(z)
        if j==0:p=np.tile(first,(n,1))
        else:
            distance=np.linalg.norm((x[None,:,:]-z[:,None,:])/sd,axis=2)
            bandwidth=np.quantile(distance,.4,axis=1) if np.min(distance)>1e-12 else np.array([np.quantile(a[a>1e-12],.4) if (a>1e-12).any() else 1. for a in distance])
            p=w[None,:]*np.exp(-distance**2/(2*bandwidth[:,None]**2));total=p.sum(axis=1)
            if not np.isfinite(total).all() or not (total>0).all():return {'ready':False,'reason':'Unsupported path state'}
            p/=total[:,None]
        ess=1/np.sum(p*p,axis=1)
        if j and (not np.isfinite(p).all() or not (ess>=20).all()):return {'ready':False,'reason':'A simulated state lacks required effective support'}
        effective.append(ess);cdf=np.cumsum(p,axis=1);cdf[:,-1]=1.;index=np.count_nonzero(points[:,j,None]>=cdf,axis=1);chosen=np.asarray(members)[index];selected.append(chosen)
        for day in range(2*j,min(2*j+2,len(plan['dates']))):
            ids=bank['blocks'][chosen,day%2];ref=reference['quotes'][ids,c] if day==0 else reference['previous_close'][ids];known=np.tile(quotes,(n,1)) if day==0 else past[:,-1]
            prices=reference['regular_close'][ids,plan['bar_counts'][day]-1]*(known/ref)
            if not np.isfinite(prices).all() or not (prices>0).all():return {'ready':False,'reason':'Invalid generated reference price'}
            past=np.concatenate([past[:,1:],prices[:,None,:]],axis=1)
    return {'ready':True,'selected_blocks':np.array(selected).transpose(1,0),'states_before_blocks':np.array(states).transpose(1,0,2),'effective_dates':np.array(effective).T}


def moments(logs):
    d=(logs[:,0]-logs[:,1])/2;return np.column_stack((logs[:,0],logs[:,1],np.abs(d),np.maximum(d,0)))


def audit_dynamic_integral(bank,members,clock,at,quote,plan,kernels,info,model,context,count,replication):
    weights=np.array(kernels[0]['probability']);seed=1062+at.tz_convert(core.ET).date().toordinal()*16+CLOCKS.index(clock)
    short=reference_values(bank,members[:,None],clock,quote,{**plan,'dates':plan['dates'][:2],'bar_counts':plan['bar_counts'][:2],'expiry_at':None});evaluations=2*len(members)
    resolved=(short['reason'][:,0]!=3)&(short['reason'][:,1]!=3);exact=np.sum(weights[resolved,None]*moments(short['log_returns'][resolved]),axis=0);mass=float(sum(weights[~resolved])) if resolved.any() else 1.
    if mass>0:
        conditioned=weights.copy();conditioned[resolved]=0.;conditioned/=mass
        children=np.random.SeedSequence([seed,replication]).spawn(8);points_per=count//8;points=np.concatenate([qmc.Sobol((len(plan['dates'])+1)//2,scramble=True,bits=30,rng=np.random.default_rng(child)).random_base2(points_per.bit_length()-1) for child in children])
        i=context['positions'][at.value];q=context['queries'];trajectory=independent_walk(bank,members,conditioned,points,q['history'][i],q['quotes'][i],context['reference'],clock,plan)
        if not trajectory['ready']:
            assert not info['ready'] and model is None and info['state_transition']=={'ready':False,'reason':trajectory['reason']};return None,evaluations
        assert info['ready'];np.testing.assert_array_equal(model['selected_blocks'],trajectory['selected_blocks'])
        for key in ['states_before_blocks','effective_dates']:close(model['state_path'][key],trajectory[key])
        assert not np.isin(model['selected_blocks'][:,0],members[resolved]).any()
        values=reference_values(bank,model['selected_blocks'],clock,quote,plan);evaluations+=2*count
        for key,value in values.items():close(model['result'][key],value)
        total=exact+mass*moments(values['log_returns']).mean(axis=0);means=np.array([exact[:2]+mass*values['log_returns'][j*points_per:(j+1)*points_per].mean(axis=0) for j in range(8)])
        dynamic=info['state_transition'];assert dynamic['ready'] and dynamic['paths']==count;close(dynamic['minimum_effective_dates'],trajectory['effective_dates'][:,1:].min());close(dynamic['mean_absolute_state_change'],np.abs(trajectory['states_before_blocks'][:,1:]-trajectory['states_before_blocks'][:,:1]).mean(axis=(0,1)))
    else:
        assert info['ready'] and model['state_path'] is None and model['result'] is None and model['selected_blocks'].shape==(0,(len(plan['dates'])+1)//2)
        assert info['state_transition']=={'ready':True,'paths':0,'minimum_effective_dates':None,'mean_absolute_state_change':[0.,0.,0.]};values=None;total=exact;means=np.tile(exact[:2],(8,1))
    for key,value in short.items():close(model['partition_result'][key],value)
    np.testing.assert_array_equal(model['resolved'],resolved);part=info['partition'];assert info['seed']==part['seed']==seed and info['replication']==replication and part['enumerated_source_blocks']==len(members)
    assert part['resolved_members']==members[resolved].tolist() and part['unresolved_members']==members[~resolved].tolist();close(part['unresolved_probability'],mass);close(part['exact_moments'],exact);close(part['estimated_moments'],total)
    l,r,m,pos=total;wanted={'SOXL':l,'SOXS':r,'C':(l+r)/2,'D':(l-r)/2,'M':m,'q':pos/m if m>0 else .5}
    for key,value in wanted.items():close(info['raw'][key],value)
    close(m*(2*wanted['q']-1),wanted['D']);deviation=means-means.mean(axis=0);cov=sum(np.outer(z,z) for z in deviation)/(8*7) if mass>0 else np.zeros((2,2))
    integ=info['integration'];assert integ['method']=='exact_absorption_plus_dynamic_eight_scrambles' and integ['replicates']==8 and integ['points_per_replicate']==(count//8 if mass>0 else 0)
    close(integ['replicate_means'],means);close(integ['mean_covariance'],cov);close(integ['standard_error'],np.sqrt(np.diag(cov)));close(info['monte_carlo_se'],np.sqrt(np.diag(cov)))
    assert info['paths']==(count if mass>0 else 0) and info['requested_paths']==count
    frequencies=[]
    for reason in [1,2,3]:
        freq=np.sum(weights[resolved,None]*(short['reason'][resolved]==reason),axis=0)+(mass*np.mean(values['reason']==reason,axis=0) if values is not None else 0);close(info['reason_frequencies'][str(reason)],freq);frequencies.append(freq)
    close(sum(frequencies),np.ones(2));return np.array([l,r]),evaluations
