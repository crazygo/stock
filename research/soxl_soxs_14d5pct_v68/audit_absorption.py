"""Independent enumeration and law-of-total-expectation integration audit."""
import numpy as np
import pandas as pd
from scipy.stats import qmc
from path_learning import CLOCKS,core
from verify_reliability import close


def moments(logs):
    difference=(logs[:,0]-logs[:,1])*0.5
    return np.stack([logs[:,0],logs[:,1],abs(difference),np.where(difference>0,difference,0.)],axis=1)


def audit_integration(bank,members,clock,at,quote,plan,kernels,kind,info,model,count,replication=0):
    from audit_paths import reference_values
    assert info['ready'] and info['replication']==replication
    seed=1062+pd.Timestamp(at.tz_convert(core.ET).date()).toordinal()*16+CLOCKS.index(clock)
    assert info['seed']==seed
    logvol=info['volatility']['current_log_volatility'] if kind=='scaled' else None
    weights=np.array(kernels[0]['probability']);later=np.array(kernels[1]['probability'])
    exact=np.zeros(4);mass=1.;evaluations=0;resolved=None;short=None
    if kind in ['conditional','fixed','fresh','scaled']:
        shortplan={**plan,'dates':plan['dates'][:2],'bar_counts':plan['bar_counts'][:2],'expiry_at':None}
        short=reference_values(bank,members[:,None],clock,quote,shortplan,logvol);evaluations+=2*len(members)
        for key,value in short.items():close(model['partition_result'][key],value)
        resolved=(short['reason'][:,0]!=3)&(short['reason'][:,1]!=3)
        np.testing.assert_array_equal(model['resolved'],resolved)
        # Use a sum of explicit weighted rows, independently of matrix multiplication.
        exact=np.sum(weights[resolved,None]*moments(short['log_returns'][resolved]),axis=0)
        mass=float(sum(weights[~resolved])) if resolved.any() else 1.
        part=info['partition'];assert part['seed']==seed and part['enumerated_source_blocks']==len(members)
        assert part['resolved_members']==members[resolved].tolist() and part['unresolved_members']==members[~resolved].tolist()
        close(part['unresolved_probability'],mass);close(part['exact_moments'],exact)
        assert info['requested_paths']==count
    physical=count if mass>0 else 0;assert info['paths']==physical
    if mass>0:
        first=weights.copy()
        if resolved is not None:first[resolved]=0.;first/=mass
        points=count//8;children=np.random.SeedSequence([seed,replication]).spawn(8)
        uniforms=np.concatenate([qmc.Sobol((len(plan['dates'])+1)//2,scramble=True,bits=30,rng=np.random.default_rng(s)).random_base2(points.bit_length()-1) for s in children])
        firstcdf=np.cumsum(first);firstcdf[-1]=1.;latercdf=np.cumsum(later);latercdf[-1]=1.
        indices=np.column_stack([np.searchsorted(firstcdf,uniforms[:,0],side='right')]+[np.searchsorted(latercdf,uniforms[:,j],side='right') for j in range(1,uniforms.shape[1])])
        selected=members[indices];np.testing.assert_array_equal(model['selected_blocks'],selected)
        if resolved is not None:assert not np.any(resolved[indices[:,0]])
        values=reference_values(bank,selected,clock,quote,plan,logvol);evaluations+=2*count
        for key,value in values.items():close(model['result'][key],value)
        means=np.array([exact[:2]+mass*values['log_returns'][j*points:(j+1)*points].mean(axis=0) for j in range(8)])
        total=exact+mass*moments(values['log_returns']).mean(axis=0)
    else:
        assert model['result'] is None and model['selected_blocks'].shape==(0,(len(plan['dates'])+1)//2)
        values=None;means=np.tile(exact[:2],(8,1));total=exact
    left,right,m,positive=total;q=positive/m if m>0 else .5
    expected={'SOXL':left,'SOXS':right,'C':(left+right)/2,'D':(left-right)/2,'M':m,'q':q}
    for key,value in expected.items():close(info['raw'][key],value)
    assert 0<=q<=1;close(m*(2*q-1),(left-right)/2)
    if kind in ['conditional','fixed','fresh','scaled']:close(info['partition']['estimated_moments'],total)
    deviations=means-means.mean(axis=0);covariance=sum(np.outer(x,x) for x in deviations)/(8*7)
    if mass==0:covariance=np.zeros((2,2))
    integration=info['integration'];assert integration['method']==('exact_absorption_plus_eight_scrambles' if kind in ['conditional','fixed','fresh','scaled'] else 'eight_independent_scrambles')
    assert integration['replicates']==8 and integration['points_per_replicate']==physical//8
    close(integration['replicate_means'],means);close(integration['mean_covariance'],covariance)
    close(integration['standard_error'],np.sqrt(np.diag(covariance)));close(info['monte_carlo_se'],np.sqrt(np.diag(covariance)))
    frequencies=[]
    for reason in [1,2,3]:
        freq=np.zeros(2)
        if resolved is not None:freq=np.sum(weights[resolved,None]*(short['reason'][resolved]==reason),axis=0)
        if values is not None:freq+=mass*np.mean(values['reason']==reason,axis=0)
        close(info['reason_frequencies'][str(reason)],freq);frequencies.append(freq)
    close(sum(frequencies),np.ones(2))
    return np.array([left,right]),evaluations
