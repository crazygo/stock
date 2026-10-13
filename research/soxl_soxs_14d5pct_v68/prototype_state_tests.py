"""Tests of an independent future-method prototype, outside frozen v68 logic."""
import unittest,statistics,copy
from prototype_state_paths import *


def scalar_state(history):
    returns=[history[-1,j]/history[-6,j]-1 for j in range(10)];changes=[history[i,0]/history[i-1,0]-1 for i in range(1,21)];sigma=max(statistics.stdev(changes),1e-4)
    return np.array([max(-5.,min(5.,(returns[0]-returns[1])/(sigma*np.sqrt(5)))),2*sum(x>0 for x in returns[2:])/8-1,np.log(sigma)])


def scalar_walk(members,first,uniform,initial,quotes,reference,bank,clock,plan):
    history=initial.copy();chosen=[];states=[];closes=[];ess=[];source=bank['states'][bank['blocks'][members,0],0,:3];c=CLOCKS.index(clock)
    for step,u in enumerate(uniform):
        state=scalar_state(history);states.append(state)
        if step==0:weights=np.array(first);neff=1/(weights@weights)
        else:
            info=kernel(source,state,'conditional');assert info['ready'];weights=np.array(info['probability']);neff=info['effective_dates']
        cumulative=np.cumsum(weights);cumulative[-1]=1.;index=int(np.searchsorted(cumulative,u,side='right'));b=int(members[index]);chosen.append(b);ess.append(neff)
        for d in range(step*2,min(step*2+2,len(plan['dates']))):
            source_day=bank['blocks'][b,d%2];closing=[]
            for j in range(10):
                base=quotes[j] if d==0 else history[-1,j];denominator=reference['quotes'][source_day,c,j] if d==0 else reference['previous_close'][source_day,j]
                closing.append(reference['regular_close'][source_day,plan['bar_counts'][d]-1,j]*(base/denominator))
            closes.append(closing);history=np.vstack([history[1:],closing])
    return np.array(chosen),np.array(states),np.array(closes),history,np.array(ess)


def fixture():
    rng=np.random.default_rng(6901);n=260;bank={'blocks':np.array([[i,i+1] for i in range(n-1)]),'states':rng.normal(size=(n,8,4))};ref={'regular_close':100*np.exp(rng.normal(0,.02,(n,78,10))),'previous_close':np.full((n,10),100.),'quotes':np.full((n,8,10),100.)};initial=100*np.exp(np.cumsum(rng.normal(0,.01,(21,10)),axis=0));q=initial[-1]*1.01;members=np.arange(220);first=np.arange(1,221,dtype=float);first/=sum(first);u=rng.random((8,3));plan={'dates':list(range(5)),'bar_counts':[78,42,78,78,42],'first_offset':10};return bank,ref,initial,q,members,first,u,plan


class StatePrototypeTests(unittest.TestCase):
    def test_state_equations_match_scalar(self):
        _,_,h,_,_,_,_,_=fixture();np.testing.assert_allclose(roll_state(h[None,:])[0],scalar_state(h),atol=1e-12)

    def test_vectorized_kernel_matches_original_every_row(self):
        rng=np.random.default_rng(69);source=rng.normal(size=(220,3));states=rng.normal(size=(12,3));p,e=query_kernels(source,states)
        for i,state in enumerate(states):
            old=kernel(source,state,'conditional');np.testing.assert_allclose(p[i],old['probability'],rtol=1e-12,atol=1e-12);self.assertAlmostEqual(e[i],old['effective_dates'])

    def test_all_draws_prices_states_match_scalar_walk(self):
        bank,ref,h,q,m,f,u,plan=fixture();actual=dynamic_blocks(m,f,u,h,q,ref,bank,CLOCKS[2],plan)
        for i in range(len(u)):
            chosen,states,closes,final,ess=scalar_walk(m,f,u[i],h,q,ref,bank,CLOCKS[2],plan)
            np.testing.assert_array_equal(actual['selected_blocks'][i],chosen)
            for key,want in [('states_before_blocks',states),('generated_daily_closes',closes),('final_history',final),('effective_dates',ess)]:np.testing.assert_allclose(actual[key][i],want,rtol=1e-11,atol=1e-11)

    def test_identical_uniforms_reproduce(self):
        bank,ref,h,q,m,f,u,plan=fixture();a=dynamic_blocks(m,f,u,h,q,ref,bank,CLOCKS[0],plan);b=dynamic_blocks(m,f,u,h,q,ref,bank,CLOCKS[0],plan)
        for k in a:np.testing.assert_array_equal(a[k],b[k])

    def test_unused_future_days_and_successor_states_do_not_change(self):
        bank,ref,h,q,m,f,u,plan=fixture();a=dynamic_blocks(m,f,u,h,q,ref,bank,CLOCKS[0],plan)
        bank['states'][240:]=np.nan;bank['successor_states']=np.full((259,3),999.);ref['regular_close'][240:]=np.nan
        b=dynamic_blocks(m,f,u,h,q,ref,bank,CLOCKS[0],plan)
        for k in a:np.testing.assert_array_equal(a[k],b[k])

    def test_history_keeps_current_past_instead_of_historical_successor(self):
        bank,ref,h,q,m,f,u,plan=fixture();a=dynamic_blocks(m,f,u,h,q,ref,bank,CLOCKS[0],plan);np.testing.assert_array_equal(a['final_history'][0,:16],h[5:])
        extended=np.concatenate((np.broadcast_to(h,(len(u),21,10)),a['generated_daily_closes']),axis=1)
        for step in range(3):np.testing.assert_allclose(a['states_before_blocks'][:,step],roll_state(extended[:,step*2:step*2+21]),rtol=1e-11,atol=1e-11)

    def test_halfday_uses_truncated_source_close(self):
        bank,ref,h,q,m,f,u,plan=fixture();plan['dates']=[0];plan['bar_counts']=[42];u=u[:,:1];ref['regular_close'][:,41]=101.;ref['regular_close'][:,77]=999.
        a=dynamic_blocks(m,f,u,h,q,ref,bank,CLOCKS[0],plan);np.testing.assert_allclose(a['generated_daily_closes'][:,0],np.broadcast_to(q*1.01,(len(u),10)))

    def test_invalid_history_is_not_filled(self):
        _,_,h,_,_,_,_,_=fixture()
        for value in [0.,np.nan,-1.]:
            z=h.copy();z[0,0]=value
            with self.assertRaises(ValueError):roll_state(z[None,:])

if __name__=='__main__':unittest.main()
